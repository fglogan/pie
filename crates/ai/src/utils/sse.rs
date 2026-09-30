//! Server-Sent-Events parser. No 1:1 TS counterpart — the TS package depends on the third-party
//! `eventsource-parser`. We inline a minimal parser here so providers don't have to.
//!
//! Spec: <https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation>
//! We deliberately implement only what providers use (`event:` and `data:` fields; comments
//! starting with `:` are dropped).

use bytes::{Bytes, BytesMut};
use futures::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};

#[derive(Clone, Debug, Default)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

/// Stream adapter: turns an HTTP body byte stream into a stream of parsed `SseEvent`s.
pub struct SseStream<S> {
    inner: S,
    buf: BytesMut,
    current: SseEvent,
    pending: Vec<SseEvent>,
    upstream_done: bool,
}

impl<S> SseStream<S> {
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            buf: BytesMut::new(),
            current: SseEvent::default(),
            pending: Vec::new(),
            upstream_done: false,
        }
    }

    fn flush_current(&mut self) {
        if !self.current.data.is_empty() || self.current.event.is_some() {
            self.pending.push(std::mem::take(&mut self.current));
        }
    }

    fn drain_lines(&mut self) {
        // Find each complete line in `self.buf`.
        loop {
            let nl = match self.buf.iter().position(|&b| b == b'\n') {
                Some(i) => i,
                None => return,
            };
            let line = self.buf.split_to(nl + 1);
            let line = std::str::from_utf8(&line[..line.len().saturating_sub(1)])
                .unwrap_or("")
                .trim_end_matches('\r');
            if line.is_empty() {
                self.flush_current();
                continue;
            }
            if line.starts_with(':') {
                // Comment — ignore.
                continue;
            }
            let (field, value) = match line.find(':') {
                Some(i) => (&line[..i], line[i + 1..].trim_start_matches(' ')),
                None => (line, ""),
            };
            match field {
                "event" => self.current.event = Some(value.to_string()),
                "data" => {
                    if !self.current.data.is_empty() {
                        self.current.data.push('\n');
                    }
                    self.current.data.push_str(value);
                }
                _ => {} // ignore `id`, `retry`, etc.
            }
        }
    }
}

impl<S> Stream for SseStream<S>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Unpin,
{
    type Item = Result<SseEvent, reqwest::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(ev) = self.pending.pop() {
                return Poll::Ready(Some(Ok(ev)));
            }
            if self.upstream_done {
                // A stream may end without the terminating newline(s). Treat EOF as ending the
                // last line and the last event instead of silently dropping them.
                if !self.buf.is_empty() {
                    self.buf.extend_from_slice(b"\n");
                    self.drain_lines();
                }
                self.flush_current();
                self.pending.reverse();
                if let Some(ev) = self.pending.pop() {
                    return Poll::Ready(Some(Ok(ev)));
                }
                return Poll::Ready(None);
            }
            match Pin::new(&mut self.inner).poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(None) => {
                    self.upstream_done = true;
                }
                Poll::Ready(Some(Ok(chunk))) => {
                    self.buf.extend_from_slice(&chunk);
                    self.drain_lines();
                    // Order: FIFO. drain_lines pushes to back; we pop from back. Reverse so the
                    // first event we parsed comes out first.
                    self.pending.reverse();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    async fn collect(chunks: &[&'static str]) -> Vec<SseEvent> {
        let body = futures::stream::iter(
            chunks
                .iter()
                .map(|c| Ok::<_, reqwest::Error>(Bytes::from_static(c.as_bytes())))
                .collect::<Vec<_>>(),
        );
        SseStream::new(body)
            .map(|r| r.expect("no transport errors in fixture"))
            .collect()
            .await
    }

    #[tokio::test]
    async fn events_split_across_chunks_arrive_in_order() {
        let evs = collect(&["event: a\ndata: 1\n\nevent: b\nda", "ta: 2\n\ndata: 3\n\n"]).await;
        let got: Vec<_> = evs
            .iter()
            .map(|e| (e.event.as_deref(), e.data.as_str()))
            .collect();
        assert_eq!(got, [(Some("a"), "1"), (Some("b"), "2"), (None, "3")]);
    }

    #[tokio::test]
    async fn final_event_without_trailing_newline_is_not_dropped() {
        let evs = collect(&["data: first\n\n", "event: done\ndata: last"]).await;
        let got: Vec<_> = evs
            .iter()
            .map(|e| (e.event.as_deref(), e.data.as_str()))
            .collect();
        assert_eq!(got, [(None, "first"), (Some("done"), "last")]);
    }

    #[tokio::test]
    async fn multiline_data_and_comments() {
        let evs = collect(&[": keepalive\ndata: a\r\ndata: b\r\n\r\n"]).await;
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "a\nb");
    }
}
