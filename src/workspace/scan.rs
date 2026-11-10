//! Streaming over file content with bounded memory: binary sniffing, a line
//! scanner that keeps only a prefix of each line, hashing while reading, and
//! decoding of a (possibly cut) line for display.

use std::io::{self, BufRead, BufReader, Read};

use sha2::{Digest, Sha256};

/// A byte-level ceiling, independent of line boundaries or producer EOF.
pub struct Ceiling<R> {
    reader: R,
    remaining: u64,
    pub hit: std::sync::Arc<std::sync::atomic::AtomicBool>,
}
impl<R> Ceiling<R> {
    pub fn new(reader: R, limit: u64) -> Self {
        Self { reader, remaining: limit, hit: Default::default() }
    }
}
impl<R: Read> Read for Ceiling<R> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        if bytes.is_empty() { return Ok(0); }
        if self.remaining == 0 {
            self.hit.store(true, std::sync::atomic::Ordering::Relaxed);
            return Ok(0);
        }
        let len = bytes.len().min(self.remaining as usize);
        let count = self.reader.read(&mut bytes[..len])?;
        self.remaining -= count as u64;
        Ok(count)
    }
}

/// Read up to `limit` bytes from the start of `reader`.
pub fn read_head(reader: &mut impl Read, limit: usize) -> io::Result<Vec<u8>> {
    let mut head = vec![0u8; limit];
    let mut filled = 0;
    while filled < limit {
        match reader.read(&mut head[filled..]) {
            Ok(0) => break,
            Ok(read) => filled += read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    head.truncate(filled);
    Ok(head)
}

/// The rule every tool uses: a NUL byte in the sniffed prefix means binary.
pub fn looks_binary(head: &[u8]) -> bool {
    head.contains(&0)
}

/// What the scanner learned about one line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineInfo {
    /// Length of the whole line in bytes, without its newline.
    pub len: u64,
    /// Whether the line ended with a newline (false only for a last line
    /// without one).
    pub terminated: bool,
}

/// Splits a byte stream at `\n` without ever holding more than a caller-chosen
/// prefix of a line, so a single 10 MB line costs one buffer, not 10 MB.
pub struct LineScanner<R: Read> {
    reader: BufReader<R>,
}

impl<R: Read> LineScanner<R> {
    pub fn new(reader: R) -> Self {
        Self { reader: BufReader::with_capacity(64 * 1024, reader) }
    }

    /// Advance to the next line. `kept` receives its first `keep` bytes.
    pub fn next_line(&mut self, kept: &mut Vec<u8>, keep: usize) -> io::Result<Option<LineInfo>> {
        kept.clear();
        let mut len = 0u64;
        let mut started = false;
        loop {
            let buffer = match self.reader.fill_buf() {
                Ok(buffer) => buffer,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            };
            if buffer.is_empty() {
                return Ok(started.then_some(LineInfo { len, terminated: false }));
            }
            started = true;
            let newline = buffer.iter().position(|byte| *byte == b'\n');
            let chunk = &buffer[..newline.unwrap_or(buffer.len())];
            let room = keep.saturating_sub(kept.len());
            kept.extend_from_slice(&chunk[..chunk.len().min(room)]);
            len += chunk.len() as u64;
            let consumed = chunk.len() + usize::from(newline.is_some());
            self.reader.consume(consumed);
            if newline.is_some() {
                return Ok(Some(LineInfo { len, terminated: true }));
            }
        }
    }

    pub fn into_inner(self) -> R {
        self.reader.into_inner()
    }
}

/// Remove the `\r` of a CRLF line ending, if the whole line was kept.
/// Returns whether one was removed.
pub fn strip_carriage_return(kept: &mut Vec<u8>, info: LineInfo) -> bool {
    if info.terminated && kept.len() as u64 == info.len && kept.last() == Some(&b'\r') {
        kept.pop();
        true
    } else {
        false
    }
}

/// Hashes and counts everything read through it.
pub struct Tee<R> {
    inner: R,
    hasher: Sha256,
    bytes: u64,
}

impl<R: Read> Tee<R> {
    pub fn new(inner: R) -> Self {
        Self { inner, hasher: Sha256::new(), bytes: 0 }
    }

    /// Bytes read so far and the hex SHA-256 of them.
    pub fn finish(self) -> (u64, String) {
        let digest = self.hasher.finalize();
        (self.bytes, digest.iter().map(|byte| format!("{byte:02x}")).collect())
    }
}

impl<R: Read> Read for Tee<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let read = self.inner.read(buffer)?.min(buffer.len());
        self.hasher.update(&buffer[..read]);
        self.bytes += read as u64;
        Ok(read)
    }
}

/// A line prefix as text.
pub struct Decoded {
    pub text: String,
    /// The line was longer than `max_chars` and has been cut.
    pub cut: bool,
    /// Invalid UTF-8 was replaced by U+FFFD.
    pub lossy: bool,
}

/// Decode the kept prefix of a line and cut it to `max_chars` characters.
/// `whole` says whether `kept` is the entire line. A multi-byte character
/// split by the prefix boundary is not reported as invalid UTF-8.
pub fn decode_line(kept: &[u8], whole: bool, max_chars: usize) -> Decoded {
    let (valid, lossy) = match std::str::from_utf8(kept) {
        Ok(text) => (std::borrow::Cow::Borrowed(text), false),
        Err(error) if !whole && error.error_len().is_none() => {
            // Only the character split by the prefix boundary is incomplete.
            (String::from_utf8_lossy(&kept[..error.valid_up_to()]), false)
        }
        Err(_) => (String::from_utf8_lossy(kept), true),
    };
    match valid.char_indices().nth(max_chars) {
        Some((index, _)) => Decoded { text: valid[..index].to_string(), cut: true, lossy },
        None => Decoded { text: valid.into_owned(), cut: !whole, lossy },
    }
}
