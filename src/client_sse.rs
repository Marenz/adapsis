//! Byte-safe Server-Sent Events framing for API clients.

/// Incremental SSE parser that returns complete `data` payloads.
#[derive(Default)]
pub(crate) struct SseParser {
    pending: Vec<u8>,
}

impl SseParser {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn feed(&mut self, chunk: &[u8]) -> Result<Vec<String>, String> {
        self.pending.extend_from_slice(chunk);
        let mut payloads = Vec::new();
        while let Some((end, delimiter_len)) = event_boundary(&self.pending) {
            let event = self.pending.drain(..end).collect::<Vec<_>>();
            self.pending.drain(..delimiter_len);
            let event = std::str::from_utf8(&event)
                .map_err(|error| format!("SSE stream contained invalid UTF-8: {error}"))?;
            if let Some(payload) = event_data(event) {
                payloads.push(payload);
            }
        }
        Ok(payloads)
    }

    pub(crate) fn has_pending_data(&self) -> bool {
        !self.pending.is_empty()
    }
}

fn event_boundary(bytes: &[u8]) -> Option<(usize, usize)> {
    for index in 0..bytes.len() {
        if bytes[index..].starts_with(b"\r\n\r\n") {
            return Some((index, 4));
        }
        if bytes[index..].starts_with(b"\n\n") {
            return Some((index, 2));
        }
    }
    None
}

fn event_data(event: &str) -> Option<String> {
    let lines = event
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(|line| line.strip_prefix(' ').unwrap_or(line))
        .collect::<Vec<_>>();
    (!lines.is_empty()).then(|| lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::SseParser;

    #[test]
    fn frames_multiple_events_and_crlf() {
        let mut parser = SseParser::new();
        let payloads = parser
            .feed(b": keepalive\r\n\r\ndata: one\r\n\r\ndata: two\ndata: lines\n\n")
            .unwrap();
        assert_eq!(payloads, ["one", "two\nlines"]);
        assert!(!parser.has_pending_data());
    }

    #[test]
    fn preserves_utf8_split_across_chunks() {
        let mut parser = SseParser::new();
        let bytes = "data: {\"text\":\"Grüße\"}\n\n".as_bytes();
        let split = bytes.iter().position(|byte| *byte == 0xc3).unwrap() + 1;
        assert!(parser.feed(&bytes[..split]).unwrap().is_empty());
        assert_eq!(
            parser.feed(&bytes[split..]).unwrap(),
            ["{\"text\":\"Grüße\"}"]
        );
    }

    #[test]
    fn waits_for_complete_event() {
        let mut parser = SseParser::new();
        assert!(parser.feed(b"data: partial").unwrap().is_empty());
        assert!(parser.has_pending_data());
        assert_eq!(parser.feed(b"\n\n").unwrap(), ["partial"]);
    }
}
