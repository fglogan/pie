//! Append-only jsonl `SessionStorage`. 1:1 port of
//! `packages/agent/src/harness/session/jsonl-storage.ts` (~293 lines).
//!
//! Layout: line 1 is the JSON-encoded `JsonlSessionMetadata` header. Subsequent lines are
//! `SessionTreeEntry` rows in append order. No in-place edits; the leaf pointer is derived from
//! the latest `leaf` entry (or, if absent, the last appended row).
//!
//! Durability / crash tolerance (pie divergence from the TS original):
//! - Each append is `fsync`ed (`sync_data`) before returning.
//! - A crash mid-append can leave a torn final line with no trailing newline. Loading skips
//!   that fragment (with a warning) instead of failing the whole session, and the next append
//!   truncates it away so it can never end up sandwiched between valid rows. A malformed line
//!   anywhere *else* is still reported as `Corrupted`.
//!
//! Performance: parsed entries are cached together with an id index and the current leaf, and
//! appends update that cache in place. The cache is revalidated against the file length (one
//! `stat`) on every access, so writes made through another handle or process force a re-read
//! instead of being silently missed.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use async_trait::async_trait;
use serde_json::Value;
use tokio::fs::{self, OpenOptions};
use tokio::io::AsyncWriteExt;

use super::super::types::{SessionError, SessionErrorCode};
use super::session::{JsonlSessionMetadata, SessionMetadata, SessionStorage, SessionTreeEntry};
use super::uuid::uuidv7;

pub struct JsonlSessionStorage {
    path: PathBuf,
    metadata: JsonlSessionMetadata,
    // In-process cache. Reads parse the file lazily on first use; appends update it in place.
    cache: Mutex<Option<EntryCache>>,
    // Serializes appends (write + cache update) within this handle so the cached order always
    // matches the on-disk order.
    append_lock: tokio::sync::Mutex<()>,
}

/// Parsed view of the file body plus the bookkeeping needed to keep it in sync with disk.
#[derive(Default)]
struct EntryCache {
    entries: Vec<SessionTreeEntry>,
    /// `entry id -> index into entries`. The first row with a given id wins, matching the
    /// original linear `find` lookup (ids are uuidv7, so duplicates only arise from hand edits).
    by_id: HashMap<String, usize>,
    leaf: Option<String>,
    /// File length (bytes) this cache reflects. Used to detect writes from other handles.
    file_len: u64,
    /// Length of the well-formed prefix (ends in `\n`). Anything past it is a torn tail from
    /// an interrupted append, which the next append truncates away.
    valid_len: u64,
    /// The file ends in a complete, parseable row that lacks a trailing newline (e.g. hand
    /// edited). The next append must emit a separator first.
    needs_newline: bool,
}

impl EntryCache {
    fn push(&mut self, entry: SessionTreeEntry) {
        match &entry {
            SessionTreeEntry::Leaf { target_id, .. } => self.leaf = target_id.clone(),
            other => self.leaf = Some(other.id().to_string()),
        }
        self.by_id
            .entry(entry.id().to_string())
            .or_insert(self.entries.len());
        self.entries.push(entry);
    }

    fn get(&self, id: &str) -> Option<&SessionTreeEntry> {
        self.by_id.get(id).map(|&i| &self.entries[i])
    }

    /// Parse a whole session file (header included). Tolerates a torn final line.
    fn parse(raw: &[u8], path: &std::path::Path) -> Result<Self, SessionError> {
        let mut cache = EntryCache {
            file_len: raw.len() as u64,
            ..Default::default()
        };
        let complete_len = raw.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
        let (complete, tail) = raw.split_at(complete_len);
        cache.valid_len = complete_len as u64;

        let text = std::str::from_utf8(complete).map_err(|e| SessionError {
            code: SessionErrorCode::Corrupted,
            message: format!("{} is not valid UTF-8: {e}", path.display()),
        })?;
        // Line 1 is the header; `open` already validated it.
        for (lineno, line) in text.split('\n').enumerate().skip(1) {
            if line.trim().is_empty() {
                continue;
            }
            let entry: SessionTreeEntry = serde_json::from_str(line).map_err(|e| SessionError {
                code: SessionErrorCode::Corrupted,
                message: format!("invalid entry at {}:{}: {e}", path.display(), lineno + 1),
            })?;
            cache.push(entry);
        }

        if complete_len == 0 {
            // Header only, without a trailing newline: treat the header as complete.
            cache.valid_len = cache.file_len;
            cache.needs_newline = !raw.is_empty();
        } else if !tail.iter().all(u8::is_ascii_whitespace) {
            match std::str::from_utf8(tail)
                .ok()
                .and_then(|t| serde_json::from_str::<SessionTreeEntry>(t.trim()).ok())
            {
                Some(entry) => {
                    // Complete row that is merely missing its newline: keep it.
                    cache.push(entry);
                    cache.valid_len = cache.file_len;
                    cache.needs_newline = true;
                }
                None => tracing::warn!(
                    path = %path.display(),
                    bytes = tail.len(),
                    "ignoring torn final line in session file (interrupted write); it will be \
                     truncated on the next append"
                ),
            }
        }
        Ok(cache)
    }
}

impl JsonlSessionStorage {
    /// Create a fresh session file at `path`, writing the header. Errors if the file exists.
    pub async fn create(
        path: impl Into<PathBuf>,
        cwd: impl Into<String>,
    ) -> Result<Self, SessionError> {
        let path = path.into();
        if path.exists() {
            return Err(SessionError {
                code: SessionErrorCode::AlreadyExists,
                message: format!("{} already exists", path.display()),
            });
        }
        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or_else(uuidv7);
        let metadata = JsonlSessionMetadata {
            base: SessionMetadata {
                id,
                created_at: chrono::Utc::now().to_rfc3339(),
            },
            cwd: cwd.into(),
            path: path.to_string_lossy().to_string(),
            parent_session_path: None,
            imported_from: None,
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await.map_err(io_err)?;
        }
        let header = serde_json::to_string(&metadata).map_err(json_err)? + "\n";
        fs::write(&path, header).await.map_err(io_err)?;
        Ok(Self {
            path,
            metadata,
            cache: Mutex::new(None),
            append_lock: tokio::sync::Mutex::new(()),
        })
    }

    /// Open an existing session file. Parses the header to recover metadata.
    pub async fn open(path: impl Into<PathBuf>) -> Result<Self, SessionError> {
        let path = path.into();
        let raw = fs::read_to_string(&path).await.map_err(io_err)?;
        let mut lines = raw.split('\n');
        let header_line = lines.next().ok_or_else(|| SessionError {
            code: SessionErrorCode::Corrupted,
            message: format!("{} is empty", path.display()),
        })?;
        let metadata: JsonlSessionMetadata =
            serde_json::from_str(header_line).map_err(|e| SessionError {
                code: SessionErrorCode::Corrupted,
                message: format!("invalid header in {}: {e}", path.display()),
            })?;
        Ok(Self {
            path,
            metadata,
            cache: Mutex::new(None),
            append_lock: tokio::sync::Mutex::new(()),
        })
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    pub fn metadata(&self) -> &JsonlSessionMetadata {
        &self.metadata
    }

    /// Make sure the cache reflects the file on disk, re-reading only when the file length
    /// changed since the cache was built (i.e. someone else wrote to it).
    async fn ensure_loaded(&self) -> Result<(), SessionError> {
        let len = fs::metadata(&self.path).await.map_err(io_err)?.len();
        if let Some(c) = self.cache_guard().as_ref() {
            if c.file_len == len {
                return Ok(());
            }
        }
        let raw = fs::read(&self.path).await.map_err(io_err)?;
        let parsed = EntryCache::parse(&raw, &self.path)?;
        *self.cache_guard() = Some(parsed);
        Ok(())
    }

    /// Run `f` against the (freshly validated) cache.
    async fn with_cache<R>(&self, f: impl FnOnce(&EntryCache) -> R) -> Result<R, SessionError> {
        self.ensure_loaded().await?;
        let guard = self.cache_guard();
        // `ensure_loaded` just populated it; `None` would mean a concurrent reset, which never
        // happens (the cache is only ever replaced, not cleared).
        let cache = guard.as_ref().ok_or_else(|| SessionError {
            code: SessionErrorCode::Unknown,
            message: "session cache unexpectedly empty".into(),
        })?;
        Ok(f(cache))
    }

    fn cache_guard(&self) -> std::sync::MutexGuard<'_, Option<EntryCache>> {
        // A panic while holding this lock cannot leave the cache half-updated in a way that
        // matters (worst case it is stale and gets revalidated by length), so recover from
        // poisoning instead of propagating the panic.
        self.cache.lock().unwrap_or_else(|p| p.into_inner())
    }

    async fn current_leaf(&self) -> Result<Option<String>, SessionError> {
        self.with_cache(|c| c.leaf.clone()).await
    }
}

fn io_err(e: std::io::Error) -> SessionError {
    SessionError {
        code: SessionErrorCode::StorageFailure,
        message: e.to_string(),
    }
}

fn json_err(e: serde_json::Error) -> SessionError {
    SessionError {
        code: SessionErrorCode::Corrupted,
        message: e.to_string(),
    }
}

#[async_trait]
impl SessionStorage for JsonlSessionStorage {
    async fn get_metadata_json(&self) -> Result<Value, SessionError> {
        serde_json::to_value(&self.metadata).map_err(json_err)
    }

    async fn get_leaf_id(&self) -> Result<Option<String>, SessionError> {
        self.current_leaf().await
    }

    async fn set_leaf_id(&self, id: Option<String>) -> Result<(), SessionError> {
        // Record as an explicit `leaf` entry — append-only by design.
        let entry = SessionTreeEntry::Leaf {
            id: uuidv7(),
            parent_id: self.current_leaf().await?,
            timestamp: chrono::Utc::now().to_rfc3339(),
            target_id: id,
        };
        self.append_entry(entry).await
    }

    async fn create_entry_id(&self) -> Result<String, SessionError> {
        Ok(uuidv7())
    }

    async fn append_entry(&self, entry: SessionTreeEntry) -> Result<(), SessionError> {
        let _serial = self.append_lock.lock().await;
        self.ensure_loaded().await?;
        let (file_len, valid_len, needs_newline) = self
            .with_cache(|c| (c.file_len, c.valid_len, c.needs_newline))
            .await?;

        let mut line = serde_json::to_string(&entry).map_err(json_err)?;
        line.push('\n');
        if needs_newline {
            line.insert(0, '\n');
        }

        let mut f = OpenOptions::new()
            .append(true)
            .open(&self.path)
            .await
            .map_err(io_err)?;
        if file_len > valid_len {
            // Drop the torn fragment left by an interrupted append so it cannot end up in the
            // middle of the file (where it would be unrecoverable corruption).
            tracing::warn!(
                path = %self.path.display(),
                bytes = file_len - valid_len,
                "truncating torn final line in session file before append"
            );
            f.set_len(valid_len).await.map_err(io_err)?;
        }
        f.write_all(line.as_bytes()).await.map_err(io_err)?;
        f.flush().await.map_err(io_err)?;
        f.sync_data().await.map_err(io_err)?;

        let mut guard = self.cache_guard();
        if let Some(c) = guard.as_mut() {
            c.push(entry);
            c.file_len = valid_len + line.len() as u64;
            c.valid_len = c.file_len;
            c.needs_newline = false;
        }
        Ok(())
    }

    async fn get_entry(&self, id: &str) -> Result<Option<SessionTreeEntry>, SessionError> {
        self.with_cache(|c| c.get(id).cloned()).await
    }

    async fn get_entries(&self) -> Result<Vec<SessionTreeEntry>, SessionError> {
        self.with_cache(|c| c.entries.clone()).await
    }

    async fn get_path_to_root(
        &self,
        leaf_id: Option<&str>,
    ) -> Result<Vec<SessionTreeEntry>, SessionError> {
        let Some(start) = leaf_id else {
            return Ok(Vec::new());
        };
        self.with_cache(|c| {
            let mut chain: Vec<SessionTreeEntry> = Vec::new();
            let mut current = Some(start);
            let mut seen = std::collections::HashSet::new();
            while let Some(id) = current {
                if !seen.insert(id) {
                    return Err(SessionError {
                        code: SessionErrorCode::Corrupted,
                        message: format!("cycle in parent chain at {id}"),
                    });
                }
                let Some(entry) = c.get(id) else {
                    return Err(SessionError {
                        code: SessionErrorCode::Corrupted,
                        message: format!("parent {id} not found"),
                    });
                };
                current = entry.parent_id();
                chain.push(entry.clone());
            }
            chain.reverse();
            Ok(chain)
        })
        .await?
    }

    async fn find_entries(&self, entry_type: &str) -> Result<Vec<SessionTreeEntry>, SessionError> {
        self.with_cache(|c| {
            c.entries
                .iter()
                .filter(|e| e.type_str() == entry_type)
                .cloned()
                .collect()
        })
        .await
    }

    async fn get_label(&self, id: &str) -> Result<Option<String>, SessionError> {
        self.with_cache(|c| {
            let mut latest: Option<String> = None;
            for entry in &c.entries {
                if let SessionTreeEntry::Label {
                    target_id, label, ..
                } = entry
                {
                    if target_id == id {
                        latest = label.clone();
                    }
                }
            }
            latest
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    fn leaf(id: &str, parent: Option<&str>, target: Option<&str>) -> SessionTreeEntry {
        SessionTreeEntry::Leaf {
            id: id.into(),
            parent_id: parent.map(Into::into),
            timestamp: "2026-01-01T00:00:00Z".into(),
            target_id: target.map(Into::into),
        }
    }

    async fn fresh() -> (tempfile::TempDir, PathBuf, JsonlSessionStorage) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.jsonl");
        let st = JsonlSessionStorage::create(&path, "/cwd").await.unwrap();
        (dir, path, st)
    }

    #[tokio::test]
    async fn append_then_read_back_through_new_handle() {
        let (_d, path, st) = fresh().await;
        st.append_entry(leaf("a", None, Some("x"))).await.unwrap();
        st.append_entry(leaf("b", Some("a"), Some("y")))
            .await
            .unwrap();
        assert_eq!(st.get_leaf_id().await.unwrap().as_deref(), Some("y"));

        let reopened = JsonlSessionStorage::open(&path).await.unwrap();
        assert_eq!(reopened.get_entries().await.unwrap().len(), 2);
        assert_eq!(reopened.get_leaf_id().await.unwrap().as_deref(), Some("y"));
        assert!(reopened.get_entry("b").await.unwrap().is_some());
        let chain = reopened.get_path_to_root(Some("b")).await.unwrap();
        let ids: Vec<_> = chain.iter().map(|e| e.id().to_string()).collect();
        assert_eq!(ids, ["a", "b"]);
    }

    #[tokio::test]
    async fn torn_final_line_is_skipped_and_truncated_on_next_append() {
        let (_d, path, st) = fresh().await;
        st.append_entry(leaf("a", None, None)).await.unwrap();
        drop(st);
        // Simulate a crash half-way through writing the next row.
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        f.write_all(br#"{"type":"leaf","id":"b","parentId":"#)
            .unwrap();
        drop(f);

        let st = JsonlSessionStorage::open(&path).await.unwrap();
        let entries = st.get_entries().await.unwrap();
        assert_eq!(
            entries.len(),
            1,
            "torn row must be ignored, not fail the load"
        );

        st.append_entry(leaf("c", Some("a"), None)).await.unwrap();
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(
            !raw.contains(r#""id":"b""#),
            "torn fragment should be truncated"
        );
        assert!(raw.ends_with('\n'));

        let reopened = JsonlSessionStorage::open(&path).await.unwrap();
        let ids: Vec<_> = reopened
            .get_entries()
            .await
            .unwrap()
            .iter()
            .map(|e| e.id().to_string())
            .collect();
        assert_eq!(ids, ["a", "c"]);
    }

    #[tokio::test]
    async fn complete_row_missing_newline_is_kept_and_separated() {
        let (_d, path, st) = fresh().await;
        drop(st);
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        f.write_all(
            serde_json::to_string(&leaf("a", None, None))
                .unwrap()
                .as_bytes(),
        )
        .unwrap();
        drop(f);

        let st = JsonlSessionStorage::open(&path).await.unwrap();
        assert_eq!(st.get_entries().await.unwrap().len(), 1);
        st.append_entry(leaf("b", Some("a"), None)).await.unwrap();

        let reopened = JsonlSessionStorage::open(&path).await.unwrap();
        assert_eq!(reopened.get_entries().await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn malformed_middle_line_is_still_corruption() {
        let (_d, path, st) = fresh().await;
        drop(st);
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        f.write_all(b"not json\n").unwrap();
        f.write_all((serde_json::to_string(&leaf("a", None, None)).unwrap() + "\n").as_bytes())
            .unwrap();
        drop(f);

        let st = JsonlSessionStorage::open(&path).await.unwrap();
        let err = st.get_entries().await.unwrap_err();
        assert_eq!(err.code, SessionErrorCode::Corrupted);
    }

    #[tokio::test]
    async fn writes_from_another_handle_are_observed() {
        let (_d, path, a) = fresh().await;
        a.append_entry(leaf("a", None, None)).await.unwrap();
        assert_eq!(a.get_entries().await.unwrap().len(), 1);

        let b = JsonlSessionStorage::open(&path).await.unwrap();
        b.append_entry(leaf("b", Some("a"), Some("t")))
            .await
            .unwrap();

        // `a`'s cache is stale; the length check must force a re-read.
        assert_eq!(a.get_entries().await.unwrap().len(), 2);
        assert_eq!(a.get_leaf_id().await.unwrap().as_deref(), Some("t"));
    }

    #[tokio::test]
    async fn concurrent_appends_keep_cache_and_disk_in_same_order() {
        let (_d, path, st) = fresh().await;
        let st = std::sync::Arc::new(st);
        let mut tasks = Vec::new();
        for i in 0..32 {
            let st = st.clone();
            tasks.push(tokio::spawn(async move {
                st.append_entry(leaf(&format!("e{i}"), None, None))
                    .await
                    .unwrap();
            }));
        }
        for t in tasks {
            t.await.unwrap();
        }
        let cached: Vec<_> = st
            .get_entries()
            .await
            .unwrap()
            .iter()
            .map(|e| e.id().to_string())
            .collect();
        let reopened = JsonlSessionStorage::open(&path).await.unwrap();
        let on_disk: Vec<_> = reopened
            .get_entries()
            .await
            .unwrap()
            .iter()
            .map(|e| e.id().to_string())
            .collect();
        assert_eq!(cached.len(), 32);
        assert_eq!(cached, on_disk);
    }
}
