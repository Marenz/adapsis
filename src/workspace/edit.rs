//! `edit`: replace an exact piece of text in a file the model has read.
//!
//! The rule that makes this safe to hand to a model: the file must be, byte
//! for byte, what the model read (`ReadLedger`). The decision is taken on the
//! very bytes that are edited: one open, one read, hash, compare with the
//! ledger, replace in memory, write atomically (`secure_write`), tied to the
//! identity of the file that was read.
//!
//! Two things `read` hides are handled here so the model does not have to
//! know about them:
//!
//! * CRLF. `read` does not show `\r`. In a file whose every line ends in
//!   CRLF, each bare `\n` of `old` and `new` is turned into `\r\n` (an
//!   existing `\r\n` is left alone). A file with mixed endings is matched
//!   exactly as given.
//! * The final newline. Whether the file ends with a newline is kept as it
//!   was, whatever the replacement says; the result notes when that mattered.
//!   `write` is the way to change it on purpose.

use std::io::Read;

use sha2::{Digest, Sha256};

use super::error::{Error, Result};
use super::ledger::{Freshness, ReadLedger};
use super::lifecycle::Workspace;
use super::limits::{AMBIGUOUS_LINES_SHOWN, EDIT_MAX_FILE_BYTES};
use super::read::FileFacts;
use super::resolve::{Access, Kind, Resolved, resolve};
use super::secure_write::{Replace, Stamp, Written};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditResult {
    /// Workspace-relative path of the edited file.
    pub path: String,
    pub replacements: usize,
    /// Lines of the new file that the replacements span, 1-based, inclusive.
    pub first_line: u64,
    pub last_line: u64,
    /// Set when the replacement would have changed whether the file ends with
    /// a newline and that was prevented: the state that was kept.
    pub kept_final_newline: Option<bool>,
}

impl EditResult {
    pub fn render(&self) -> String {
        let count = if self.replacements == 1 { "1 replacement".to_string() } else { format!("{} replacements", self.replacements) };
        let lines = if self.first_line == self.last_line {
            format!("line {}", self.first_line)
        } else {
            format!("lines {}-{}", self.first_line, self.last_line)
        };
        let mut out = format!("edited {}: {count}, {lines}", self.path);
        if let Some(kept) = self.kept_final_newline {
            let state = if kept { "it still ends with a newline" } else { "it still ends without a newline" };
            out.push_str(&format!(
                "\n[note: edit keeps the file's final-newline state ({state}); to change that on purpose, rewrite the file with write]"
            ));
        }
        out
    }
}

/// Replace `old` by `new` in the file at `path`. Without `replace_all`, `old`
/// must occur exactly once.
pub fn edit(workspace: &Workspace, ledger: &mut ReadLedger, path: &str, old: &str, new: &str, replace_all: bool) -> Result<EditResult> {
    if old.is_empty() {
        return Err(Error::InvalidArgument("the text to replace is empty; to create a file or replace all of it, use write".to_string()));
    }
    if old == new {
        return Err(Error::InvalidArgument("the old and the new text are identical; there is nothing to change".to_string()));
    }
    let resolved = resolve(workspace, path, Access::Write)?;
    let loaded = load_for_change(&resolved, ledger, EDIT_MAX_FILE_BYTES)?;
    let text = String::from_utf8(loaded.bytes).map_err(|_| Error::WrongKind {
        path: resolved.supplied.clone(),
        expected: "valid UTF-8 text; a file with other bytes cannot be edited exactly",
    })?;
    let replaced = replace_exact(&text, old, new, replace_all).map_err(|failure| failure.into_error(&resolved.supplied))?;
    let written = resolved.write_content(replaced.text.as_bytes(), Replace::Overwrite { expect: Some(loaded.stamp) })?;
    ledger.record(&resolved.display, facts_of(replaced.text.as_bytes(), &written));
    Ok(EditResult {
        path: resolved.display,
        replacements: replaced.replacements,
        first_line: replaced.first_line,
        last_line: replaced.last_line,
        kept_final_newline: replaced.kept_final_newline,
    })
}

/// The facts of content this process just wrote.
pub(super) fn facts_of(content: &[u8], written: &Written) -> FileFacts {
    FileFacts { size: written.size, mtime_ns: written.mtime_ns, sha256: Some(sha256_hex(content)) }
}

fn sha256_hex(content: &[u8]) -> String {
    Sha256::digest(content).iter().map(|byte| format!("{byte:02x}")).collect()
}

/// An existing file, loaded for a change that the ledger allows.
pub(super) struct Loaded {
    pub bytes: Vec<u8>,
    pub stamp: Stamp,
}

/// Load the file behind a write resolution, but only if the ledger says it is
/// what the model read. The freshness verdict is taken on the loaded bytes.
pub(super) fn load_for_change(resolved: &Resolved, ledger: &ReadLedger, max_bytes: u64) -> Result<Loaded> {
    let not_fresh = |state| Error::NotFresh { path: resolved.supplied.clone(), state };
    if resolved.kind == Kind::Missing {
        return Err(match ledger.recorded(&resolved.display) {
            Some(_) => not_fresh(Freshness::Missing),
            None => Error::NotFound { path: resolved.supplied.clone() },
        });
    }
    let (file, before) = resolved.open_file()?;
    // Refuse before reading anything when the model never read the file,
    // or read it in a way that left nothing to compare with.
    match ledger.recorded(&resolved.display) {
        None => return Err(not_fresh(Freshness::Unread)),
        Some(recorded) if recorded.sha256.is_none() => return Err(not_fresh(Freshness::Unverifiable)),
        Some(_) => {}
    }
    if before.len() > max_bytes {
        return Err(Error::InvalidArgument(format!(
            "`{}` is {} bytes; files larger than {max_bytes} bytes cannot be changed with this tool",
            resolved.supplied,
            before.len()
        )));
    }
    let mut bytes = Vec::with_capacity(before.len() as usize);
    (&file)
        .take(max_bytes + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| Error::io(&resolved.supplied, "read", &error))?;
    let after = file.metadata().map_err(|error| Error::io(&resolved.supplied, "read", &error))?;
    let stamp = Stamp::of(&before);
    let stable = stamp == Stamp::of(&after) && bytes.len() as u64 == after.len();
    let facts = FileFacts { size: after.len(), mtime_ns: stamp.mtime_ns, sha256: stable.then(|| sha256_hex(&bytes)) };
    match ledger.judge(&resolved.display, &facts) {
        Freshness::Unchanged => Ok(Loaded { bytes, stamp }),
        state => Err(not_fresh(state)),
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Replaced {
    pub text: String,
    pub replacements: usize,
    pub first_line: u64,
    pub last_line: u64,
    pub kept_final_newline: Option<bool>,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Failure {
    NoMatch { mixed_endings: bool },
    Ambiguous { count: usize, lines: Vec<u64> },
    Unchanged,
}

impl Failure {
    fn into_error(self, path: &str) -> Error {
        let path = path.to_string();
        match self {
            Failure::NoMatch { mixed_endings: true } => Error::NoMatch {
                path,
                hint: "The file mixes CRLF and LF line endings, which read does not show; replace text within a single line, or rewrite the file with write.",
            },
            Failure::NoMatch { mixed_endings: false } => Error::NoMatch {
                path,
                hint: "It must match the file exactly, including indentation and blank lines; read the lines again and copy them.",
            },
            Failure::Ambiguous { count, lines } => Error::Ambiguous { path, count, lines },
            Failure::Unchanged => Error::InvalidArgument(format!("`{path}` was not changed: the replacement would leave the file as it is")),
        }
    }
}

/// The replacement itself, on text: line-ending adaptation, the uniqueness
/// rule, the final-newline rule, and the line range of the result.
pub(super) fn replace_exact(text: &str, old: &str, new: &str, replace_all: bool) -> std::result::Result<Replaced, Failure> {
    let crlf = is_crlf(text);
    let (old, new) = if crlf { (to_crlf(old), to_crlf(new)) } else { (old.to_string(), new.to_string()) };
    let starts = occurrences(text, &old);
    if starts.is_empty() {
        return Err(Failure::NoMatch { mixed_endings: !crlf && text.contains('\r') });
    }
    if starts.len() > 1 && !replace_all {
        let lines = starts.iter().take(AMBIGUOUS_LINES_SHOWN).map(|start| line_of(text, *start)).collect();
        return Err(Failure::Ambiguous { count: starts.len(), lines });
    }
    // Build the result left to right over non-overlapping matches.
    let mut out = String::with_capacity(text.len() + new.len());
    let (mut cursor, mut replacements, mut first_start, mut last_end) = (0usize, 0usize, 0usize, 0usize);
    for (start, _) in text.match_indices(&old) {
        out.push_str(&text[cursor..start]);
        if replacements == 0 {
            first_start = out.len();
        }
        out.push_str(&new);
        last_end = out.len();
        cursor = start + old.len();
        replacements += 1;
        if !replace_all {
            break;
        }
    }
    out.push_str(&text[cursor..]);
    let kept_final_newline = keep_final_newline(text, &mut out, crlf);
    if out == text {
        return Err(Failure::Unchanged);
    }
    let first_start = first_start.min(out.len());
    let last_byte = last_end.saturating_sub(1).clamp(first_start, out.len());
    Ok(Replaced {
        first_line: line_of(&out, first_start),
        last_line: line_of(&out, last_byte),
        text: out,
        replacements,
        kept_final_newline,
    })
}

/// Every line break is CRLF (and there is at least one).
fn is_crlf(text: &str) -> bool {
    let bytes = text.as_bytes();
    let mut any = false;
    for (index, byte) in bytes.iter().enumerate() {
        if *byte == b'\n' {
            if index == 0 || bytes[index - 1] != b'\r' {
                return false;
            }
            any = true;
        }
    }
    any
}

/// Turn each bare `\n` into `\r\n`; a `\r\n` that is already there stays.
fn to_crlf(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + text.len() / 16);
    let mut previous = '\0';
    for character in text.chars() {
        if character == '\n' && previous != '\r' {
            out.push('\r');
        }
        out.push(character);
        previous = character;
    }
    out
}

/// Start offsets of every occurrence of `needle`, overlapping ones included:
/// in `aaa` the text `aa` is not unique.
fn occurrences(text: &str, needle: &str) -> Vec<usize> {
    let step = needle.chars().next().map_or(1, char::len_utf8);
    let mut starts = Vec::new();
    let mut from = 0;
    while let Some(found) = text.get(from..).and_then(|rest| rest.find(needle)) {
        starts.push(from + found);
        from += found + step;
    }
    starts
}

/// 1-based line of the byte at `offset`.
fn line_of(text: &str, offset: usize) -> u64 {
    let end = offset.min(text.len());
    1 + text.as_bytes()[..end].iter().filter(|byte| **byte == b'\n').count() as u64
}

/// Make `out` end like `original` ended. Returns the kept state if `out` had
/// to be adjusted for it.
fn keep_final_newline(original: &str, out: &mut String, crlf: bool) -> Option<bool> {
    let had = original.ends_with('\n');
    if had == out.ends_with('\n') || out.is_empty() {
        return None;
    }
    if had {
        out.push_str(if crlf { "\r\n" } else { "\n" });
    } else {
        while out.ends_with('\n') {
            out.pop();
            if crlf && out.ends_with('\r') {
                out.pop();
            }
        }
    }
    Some(had)
}
