//! Per-file mutation serialization for the file tools.
//!
//! Ports the safety property of upstream pi's `file-mutation-queue.ts`: the whole
//! read-modify-write of `edit` and the overwrite of `write` must be serialized *per file* so
//! two concurrent tool calls against the same path cannot interleave and lose an update.
//! Distinct files still run in parallel.
//!
//! The lock key is the canonical (realpath) path when the target exists, so two aliases /
//! symlinks to the same inode share one lock. When the target does not exist yet (the common
//! case for `write` creating a new file) we fall back to the canonical parent joined with the
//! file name, and finally to an absolutized-but-not-canonical path — mirroring upstream's
//! "falls back to the resolved path on ENOENT" behavior. The fallbacks are best-effort: they
//! guarantee correctness for the realistic cases (same existing file, same new file in an
//! existing dir) without pretending to resolve unresolvable paths.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};

use once_cell::sync::Lazy;
use tokio::sync::Mutex as AsyncMutex;

/// Registry of per-file locks, keyed by resolved path. The `std::sync::Mutex` only guards the
/// brief get-or-insert of the per-key `Arc<AsyncMutex>`; the actual mutation runs while holding
/// the async lock, not this one.
static REGISTRY: Lazy<StdMutex<HashMap<PathBuf, Arc<AsyncMutex<()>>>>> =
    Lazy::new(|| StdMutex::new(HashMap::new()));

/// Resolve the lock key for `path`. Prefer the realpath so aliases to one inode serialize;
/// degrade gracefully when the file (or its parent) does not exist yet.
async fn lock_key(path: &Path) -> PathBuf {
    if let Ok(canon) = tokio::fs::canonicalize(path).await {
        return canon;
    }
    if let (Some(parent), Some(name)) = (path.parent(), path.file_name()) {
        // `parent()` of a bare relative filename is `""`; canonicalizing that resolves to the
        // current dir, which is the correct anchor for a new file created there.
        let anchor = if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        };
        if let Ok(canon_parent) = tokio::fs::canonicalize(anchor).await {
            return canon_parent.join(name);
        }
    }
    absolutize(path)
}

/// Best-effort absolutization without touching the filesystem (used only when canonicalization
/// is impossible). Joins the process cwd for a relative path; leaves an absolute path as-is.
fn absolutize(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    match std::env::current_dir() {
        Ok(cwd) => cwd.join(path),
        Err(_) => path.to_path_buf(),
    }
}

/// Run `f` while holding the exclusive per-file lock for `path`. Concurrent calls for the same
/// resolved file are serialized; calls for different files proceed in parallel.
pub async fn with_file_lock<F, Fut, T>(path: &Path, f: F) -> T
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = T>,
{
    let key = lock_key(path).await;
    let lock = {
        let mut reg = registry();
        Arc::clone(
            reg.entry(key.clone())
                .or_insert_with(|| Arc::new(AsyncMutex::new(()))),
        )
    };
    let out = {
        let _guard = lock.lock().await;
        f().await
    };
    // Drop the registry entry once nobody else holds or waits on it, so a long-running
    // session that touches many files does not grow the map without bound. Waiters hold
    // their own `Arc` clone, so a count of 2 (registry + ours) means we are the last user.
    // Clones are only taken under the registry mutex, which we hold here, so the count cannot
    // race upward between the check and the removal.
    let mut reg = registry();
    if Arc::strong_count(&lock) == 2 && reg.get(&key).is_some_and(|l| Arc::ptr_eq(l, &lock)) {
        reg.remove(&key);
    }
    out
}

fn registry() -> std::sync::MutexGuard<'static, HashMap<PathBuf, Arc<AsyncMutex<()>>>> {
    // The map holds no invariants a panic could break; recover instead of cascading panics.
    REGISTRY.lock().unwrap_or_else(|p| p.into_inner())
}

/// Replace `path`'s contents atomically: write a sibling temp file, `fsync` it, then `rename`
/// it over the target. A crash, full disk or kill mid-write leaves either the old or the new
/// contents, never a truncated file. Symlinks are followed (the link target is replaced, the
/// link itself is kept) and an existing file's permission bits are preserved.
///
/// Callers should hold [`with_file_lock`] for `path`.
pub async fn write_atomic(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    // Resolve a symlinked target so we replace the file it points at, not the link.
    let target = match tokio::fs::canonicalize(path).await {
        Ok(real) => real,
        Err(_) => path.to_path_buf(),
    };
    let existing_perms = tokio::fs::metadata(&target)
        .await
        .ok()
        .map(|m| m.permissions());

    let dir = match target.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let name = target
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "file".into());
    let tmp = dir.join(format!(".{name}.pie-tmp-{}", uuid::Uuid::new_v4().simple()));

    let result = async {
        let mut f = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .await?;
        tokio::io::AsyncWriteExt::write_all(&mut f, contents).await?;
        f.sync_all().await?;
        drop(f);
        if let Some(perms) = existing_perms {
            tokio::fs::set_permissions(&tmp, perms).await?;
        }
        tokio::fs::rename(&tmp, &target).await?;
        // Persist the rename itself (best effort; not supported on every platform/filesystem).
        #[cfg(unix)]
        if let Ok(d) = tokio::fs::File::open(&dir).await {
            let _ = d.sync_all().await;
        }
        Ok(())
    }
    .await;
    if result.is_err() {
        let _ = tokio::fs::remove_file(&tmp).await;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tempfile::tempdir;

    /// The lock is exclusive per file: overlapping critical sections never run concurrently.
    #[tokio::test]
    async fn serializes_same_file() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("f.txt");
        std::fs::write(&p, "x").unwrap();

        let inside = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..16 {
            let p = p.clone();
            let inside = Arc::clone(&inside);
            let max_seen = Arc::clone(&max_seen);
            handles.push(tokio::spawn(async move {
                with_file_lock(&p, || async {
                    let n = inside.fetch_add(1, Ordering::SeqCst) + 1;
                    max_seen.fetch_max(n, Ordering::SeqCst);
                    // Yield so a broken (non-exclusive) lock would let peers enter here.
                    tokio::task::yield_now().await;
                    inside.fetch_sub(1, Ordering::SeqCst);
                })
                .await;
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(
            max_seen.load(Ordering::SeqCst),
            1,
            "at most one critical section may run at a time for the same file"
        );
    }

    /// The registry must not keep an entry per file forever.
    #[tokio::test]
    async fn registry_entries_are_released() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("released.txt");
        std::fs::write(&p, "x").unwrap();
        let key = lock_key(&p).await;
        with_file_lock(&p, || async {
            assert!(registry().contains_key(&key));
        })
        .await;
        assert!(
            !registry().contains_key(&key),
            "entry should be dropped after use"
        );
    }

    #[tokio::test]
    async fn write_atomic_replaces_contents_and_keeps_mode() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("script.sh");
        std::fs::write(&p, "old").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        write_atomic(&p, b"new contents").await.unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "new contents");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o755);
        }
        // No temp files left behind.
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".pie-tmp-"))
            .collect();
        assert!(leftovers.is_empty());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn write_atomic_follows_symlink() {
        let dir = tempdir().unwrap();
        let real = dir.path().join("real.txt");
        let link = dir.path().join("link.txt");
        std::fs::write(&real, "old").unwrap();
        std::os::unix::fs::symlink(&real, &link).unwrap();
        write_atomic(&link, b"new").await.unwrap();
        assert!(
            std::fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(std::fs::read_to_string(&real).unwrap(), "new");
    }

    #[tokio::test]
    async fn write_atomic_failure_leaves_original_untouched() {
        let dir = tempdir().unwrap();
        // Target is a directory: rename over it fails, the temp file must be cleaned up.
        let target = dir.path().join("adir");
        std::fs::create_dir(&target).unwrap();
        std::fs::write(target.join("keep.txt"), "keep").unwrap();
        assert!(write_atomic(&target, b"x").await.is_err());
        assert_eq!(
            std::fs::read_to_string(target.join("keep.txt")).unwrap(),
            "keep"
        );
        let leftovers = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".pie-tmp-"))
            .count();
        assert_eq!(leftovers, 0);
    }

    /// Different files are not blocked by each other.
    #[tokio::test]
    async fn distinct_files_run_in_parallel() {
        let dir = tempdir().unwrap();
        let a = dir.path().join("a.txt");
        let b = dir.path().join("b.txt");
        std::fs::write(&a, "a").unwrap();
        std::fs::write(&b, "b").unwrap();

        // Hold a's lock, then confirm b's lock is still acquirable without waiting for a.
        let held = with_file_lock(&a, || async {
            // While inside a's critical section, b must be independently lockable.
            with_file_lock(&b, || async { 42 }).await
        })
        .await;
        assert_eq!(held, 42);
    }

    /// Upstream-parity: a symlink alias shares the lock of its canonical target (the key is the
    /// realpath). Holding the target's lock must block a lock request made via the symlink.
    #[cfg(unix)]
    #[tokio::test]
    async fn symlink_alias_shares_target_lock() {
        use std::sync::atomic::AtomicBool;
        use std::time::Duration;

        let dir = tempdir().unwrap();
        let real = dir.path().join("real.txt");
        std::fs::write(&real, "x").unwrap();
        let link = dir.path().join("link.txt");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        // Hold the target's lock via a holder task until we release it.
        let (tx_acquired, rx_acquired) = tokio::sync::oneshot::channel::<()>();
        let (tx_release, rx_release) = tokio::sync::oneshot::channel::<()>();
        let real_holder = real.clone();
        let holder = tokio::spawn(async move {
            with_file_lock(&real_holder, || async move {
                tx_acquired.send(()).unwrap();
                rx_release.await.unwrap();
            })
            .await;
        });
        rx_acquired.await.unwrap();

        // A request via the symlink must block, because the key is the canonical target path.
        let completed = Arc::new(AtomicBool::new(false));
        let waiter = tokio::spawn({
            let completed = Arc::clone(&completed);
            async move {
                with_file_lock(&link, || async {}).await;
                completed.store(true, Ordering::SeqCst);
            }
        });

        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            !completed.load(Ordering::SeqCst),
            "a symlink alias must share the lock of its canonical target"
        );

        tx_release.send(()).unwrap();
        holder.await.unwrap();
        waiter.await.unwrap();
        assert!(completed.load(Ordering::SeqCst));
    }
}
