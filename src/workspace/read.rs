//! `read`: a range of lines of one text file, plus the facts a later edit
//! needs to prove the file is still what was read.

use std::fs::Metadata;
use std::io::{Cursor, Read};
use std::os::unix::fs::MetadataExt;

use serde::{Deserialize, Serialize};

use super::error::{Error, Result};
use super::lifecycle::Workspace;
use super::limits::{BINARY_SNIFF_BYTES, LINE_MAX_CHARS, READ_MAX_BYTES, READ_MAX_LINES, SCAN_MAX_BYTES};
use super::resolve::{Access, Area, Resolved, resolve};
use super::scan::{LineScanner, Tee, decode_line, looks_binary, read_head, strip_carriage_return};

/// Bytes of one line that are buffered: enough for `LINE_MAX_CHARS` characters
/// of any width, plus one to see that the line goes on.
const LINE_KEEP_BYTES: usize = LINE_MAX_CHARS * 4 + 4;

/// Identity of a file's content at one moment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileFacts {
    pub size: u64,
    /// Modification time in nanoseconds since the epoch.
    pub mtime_ns: i64,
    /// Hex SHA-256 of the whole content. `None` when the file is larger than
    /// the scan limit or changed while it was being read; such a file cannot
    /// be proven unchanged.
    pub sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadLine {
    pub number: u64,
    pub text: String,
    /// Byte length of the whole line when it was cut.
    pub cut_from_bytes: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadResult {
    /// The path as shown to the model (see `Resolved::display`).
    pub path: String,
    /// Whether the file is in the workspace itself (and so can be edited).
    pub writable: bool,
    /// The 1-based line the caller asked to start at.
    pub offset: u64,
    pub lines: Vec<ReadLine>,
    /// `None` when the file is larger than the scan limit.
    pub total_lines: Option<u64>,
    /// Lines remain after the last returned one.
    pub more: bool,
    /// The offset that continues this read.
    pub next_offset: Option<u64>,
    pub facts: FileFacts,
    /// Invalid UTF-8 was replaced in the shown lines.
    pub lossy: bool,
    /// Shown lines end in CRLF (the `\r` is not shown).
    pub crlf: bool,
    /// The file ends with a newline. Meaningless for an empty file.
    pub final_newline: bool,
    /// The file changed while it was being read.
    pub changed_during_read: bool,
}

/// Read lines `offset ..` (1-based, default 1) of a text file, at most
/// `limit` lines (default and maximum `READ_MAX_LINES`).
pub fn read(workspace: &Workspace, path: &str, offset: Option<u64>, limit: Option<usize>) -> Result<ReadResult> {
    let resolved = resolve(workspace, path, Access::Read)?;
    let (file, before) = resolved.open_file()?;
    let offset = offset.unwrap_or(1).max(1);
    let limit = limit.unwrap_or(READ_MAX_LINES).clamp(1, READ_MAX_LINES);
    let io_error = |error: std::io::Error| Error::io(&resolved.supplied, "read", &error);

    let mut source = Tee::new((&file).take(SCAN_MAX_BYTES));
    let head = read_head(&mut source, BINARY_SNIFF_BYTES).map_err(io_error)?;
    if looks_binary(&head) {
        return Err(Error::Binary { path: resolved.supplied.clone(), size: before.len() });
    }
    let mut scanner = LineScanner::new(Cursor::new(head).chain(source));
    let mut window = Window::new(offset, limit);
    let mut kept = Vec::new();
    let mut count = 0u64;
    let mut final_newline = true;
    loop {
        let keep = if window.wants(count + 1) { LINE_KEEP_BYTES } else { 0 };
        let Some(info) = scanner.next_line(&mut kept, keep).map_err(io_error)? else { break };
        count += 1;
        final_newline = info.terminated;
        if keep > 0 {
            window.take(count, &mut kept, info);
        }
    }
    let (_, source) = scanner.into_inner().into_inner();
    let (scanned, digest) = source.finish();
    let after = file.metadata().map_err(io_error)?;
    let changed = stamp(&before) != stamp(&after) || (after.len() <= SCAN_MAX_BYTES && scanned != after.len());
    let complete = after.len() <= SCAN_MAX_BYTES;
    let facts = FileFacts { size: after.len(), mtime_ns: stamp(&after).1, sha256: (complete && !changed).then_some(digest) };
    Ok(window.into_result(&resolved, facts, complete.then_some(count), final_newline, changed))
}

/// Hash a resolved file without returning any of it: the facts an edit
/// compares against the read ledger, or records after its own write.
pub fn file_facts(resolved: &Resolved) -> Result<FileFacts> {
    let (file, before) = resolved.open_file()?;
    let io_error = |error: std::io::Error| Error::io(&resolved.supplied, "read", &error);
    let mut source = Tee::new((&file).take(SCAN_MAX_BYTES));
    std::io::copy(&mut source, &mut std::io::sink()).map_err(io_error)?;
    let (scanned, digest) = source.finish();
    let after = file.metadata().map_err(io_error)?;
    let stable = stamp(&before) == stamp(&after) && after.len() <= SCAN_MAX_BYTES && scanned == after.len();
    Ok(FileFacts { size: after.len(), mtime_ns: stamp(&after).1, sha256: stable.then_some(digest) })
}

/// Size and modification time, for "did it change underneath us".
fn stamp(metadata: &Metadata) -> (u64, i64) {
    (metadata.len(), metadata.mtime().saturating_mul(1_000_000_000).saturating_add(metadata.mtime_nsec()))
}

/// Collects the requested lines under the line and byte caps.
struct Window {
    offset: u64,
    limit: usize,
    lines: Vec<ReadLine>,
    bytes: usize,
    full: bool,
    lossy: bool,
    crlf: bool,
}

impl Window {
    fn new(offset: u64, limit: usize) -> Self {
        Self { offset, limit, lines: Vec::new(), bytes: 0, full: false, lossy: false, crlf: false }
    }

    fn wants(&self, number: u64) -> bool {
        !self.full && number >= self.offset
    }

    fn take(&mut self, number: u64, kept: &mut Vec<u8>, info: super::scan::LineInfo) {
        let crlf = strip_carriage_return(kept, info);
        let whole = kept.len() as u64 + u64::from(crlf) == info.len;
        let decoded = decode_line(kept, whole, LINE_MAX_CHARS);
        if !self.lines.is_empty() && self.bytes + decoded.text.len() > READ_MAX_BYTES {
            self.full = true;
            return;
        }
        self.bytes += decoded.text.len();
        self.lossy |= decoded.lossy;
        self.crlf |= crlf;
        self.lines.push(ReadLine { number, text: decoded.text, cut_from_bytes: decoded.cut.then_some(info.len) });
        self.full = self.lines.len() >= self.limit;
    }

    fn into_result(self, resolved: &Resolved, facts: FileFacts, total: Option<u64>, final_newline: bool, changed: bool) -> ReadResult {
        let last = self.lines.last().map(|line| line.number);
        // With an unknown total (file beyond the scan limit) more remains
        // only if a cap, not the scan limit, ended the collection.
        let more = match (total, last) {
            (Some(total), Some(last)) => last < total,
            (Some(_), None) => false,
            (None, _) => self.full,
        };
        ReadResult {
            path: resolved.display.clone(),
            writable: resolved.area == Area::Root,
            offset: self.offset,
            next_offset: last.filter(|_| more).map(|last| last + 1),
            lines: self.lines,
            total_lines: total,
            more,
            facts,
            lossy: self.lossy,
            crlf: self.crlf,
            final_newline,
            changed_during_read: changed,
        }
    }
}

impl ReadResult {
    /// The text the model sees: a header, `N: text` lines, then notes.
    pub fn render(&self) -> String {
        let mut out = self.header();
        for line in &self.lines {
            out.push_str(&format!("\n{}: {}", line.number, line.text));
            if let Some(bytes) = line.cut_from_bytes {
                out.push_str(&format!(" …[line cut after {LINE_MAX_CHARS} characters; it has {bytes} bytes]"));
            }
        }
        if let (true, Some(next)) = (self.more, self.next_offset) {
            match (self.total_lines, self.lines.last()) {
                (Some(total), Some(last)) => out.push_str(&format!("\n[{} more lines; continue with offset={next}]", total - last.number)),
                _ => out.push_str(&format!("\n[more lines follow; continue with offset={next}]")),
            }
        }
        for note in self.notes() {
            out.push_str(&format!("\n[note: {note}]"));
        }
        out
    }

    fn header(&self) -> String {
        let size = self.facts.size;
        match (self.lines.first(), self.lines.last(), self.total_lines) {
            (Some(first), Some(last), Some(total)) => {
                format!("{}: lines {}-{} of {total} ({size} bytes)", self.path, first.number, last.number)
            }
            (Some(first), Some(last), None) => {
                format!("{}: lines {}-{} ({size} bytes; too large to count its lines)", self.path, first.number, last.number)
            }
            (_, _, Some(0)) => format!("{}: empty file (0 bytes)", self.path),
            (_, _, Some(total)) => {
                format!("{}: no lines at offset {}; the file has {total} lines ({size} bytes)", self.path, self.offset)
            }
            (_, _, None) => format!("{}: no lines at offset {} ({size} bytes)", self.path, self.offset),
        }
    }

    fn notes(&self) -> Vec<String> {
        let mut notes = Vec::new();
        if self.lossy {
            notes.push("the file is not valid UTF-8; invalid bytes are shown as \u{FFFD}, so the text is not exact".to_string());
        }
        if self.crlf {
            notes.push("lines end in CRLF; the carriage returns are not shown".to_string());
        }
        if !self.final_newline && self.total_lines.is_some_and(|total| total > 0) && !self.more {
            notes.push("no newline at the end of the file".to_string());
        }
        if self.changed_during_read {
            notes.push("the file changed while it was being read; read it again before editing".to_string());
        } else if self.facts.sha256.is_none() {
            notes.push(format!("the file is larger than {SCAN_MAX_BYTES} bytes; only that much of it can be read, and it cannot be edited"));
        }
        if !self.writable {
            notes.push("this file is in a read-only root".to_string());
        }
        notes
    }
}
