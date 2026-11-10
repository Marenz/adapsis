//! The read ledger: which files were read, and what they looked like.
//!
//! An edit tool must refuse to change a file the model has not read, or one
//! that changed since. The ledger is the memory for that rule. It is a plain
//! owned value with no global state and no workspace reference inside, and it
//! serializes with serde, so the caller decides its lifetime: one ledger per
//! (conversation, workspace) is the intended use, kept wherever conversation
//! state lives.
//!
//! Keys are workspace-relative real paths (`Resolved::display`), so two
//! spellings of one file share a record. Only files in the workspace itself
//! are recorded; a read-only root can never be edited.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::error::Result;
use super::lifecycle::Workspace;
use super::limits::LEDGER_MAX_ENTRIES;
use super::read::{FileFacts, ReadResult, file_facts};
use super::resolve::{Access, Kind, resolve};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadLedger {
    entries: HashMap<String, Entry>,
    clock: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Entry {
    facts: FileFacts,
    /// Logical time of the record; the oldest is dropped when the ledger is full.
    seen: u64,
}

/// Whether a file may be edited on the strength of an earlier read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Freshness {
    /// Never read (or the record was dropped): read it first.
    Unread,
    /// The content is byte-for-byte what was read.
    Unchanged,
    /// The content differs from what was read: read it again.
    Changed,
    /// Read earlier, but the file no longer exists.
    Missing,
    /// The file is too large to hash, or it changed while being read, so
    /// nothing can be proven either way.
    Unverifiable,
}

impl Freshness {
    /// Why an edit must be refused, or `None` when it may proceed.
    pub fn refusal(self) -> Option<&'static str> {
        match self {
            Freshness::Unchanged => None,
            Freshness::Unread => Some("the file has not been read in this conversation; read it first"),
            Freshness::Changed => Some("the file changed since it was read; read it again"),
            Freshness::Missing => Some("the file was read earlier but no longer exists; read the directory again"),
            Freshness::Unverifiable => {
                Some("the file is too large, or changed while it was being read, so it cannot be verified as unchanged; read it again")
            }
        }
    }
}

impl ReadLedger {
    /// Record a successful read. Reads of read-only roots are not recorded.
    pub fn note(&mut self, result: &ReadResult) {
        if result.writable {
            self.record(&result.path, result.facts.clone());
        }
    }

    /// Record the state of a file under its workspace-relative path; an edit
    /// tool calls this after its own write so the next edit needs no re-read.
    pub fn record(&mut self, path: &str, facts: FileFacts) {
        self.clock += 1;
        self.entries.insert(path.to_string(), Entry { facts, seen: self.clock });
        while self.entries.len() > LEDGER_MAX_ENTRIES {
            let Some(oldest) = self.entries.iter().min_by_key(|(_, entry)| entry.seen).map(|(path, _)| path.clone()) else { break };
            self.entries.remove(&oldest);
        }
    }

    pub fn recorded(&self, path: &str) -> Option<&FileFacts> {
        self.entries.get(path).map(|entry| &entry.facts)
    }

    pub fn forget(&mut self, path: &str) {
        self.entries.remove(path);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Compare a file with its record. The path is resolved for writing, so
    /// every write refusal (symlink, `.git`, outside) surfaces here already.
    /// The content hash decides; a changed timestamp alone is not a change.
    pub fn check(&self, workspace: &Workspace, path: &str) -> Result<Freshness> {
        let resolved = resolve(workspace, path, Access::Write)?;
        let Some(recorded) = self.recorded(&resolved.display) else { return Ok(Freshness::Unread) };
        if resolved.kind == Kind::Missing {
            return Ok(Freshness::Missing);
        }
        let _ = recorded;
        match file_facts(&resolved) {
            Ok(current) => Ok(self.judge(&resolved.display, &current)),
            Err(super::error::Error::NotFound { .. }) => Ok(Freshness::Missing),
            Err(error) => Err(error),
        }
    }

    /// Compare facts the caller just took from a file with its record. An
    /// edit uses this with the facts of the very bytes it is about to change,
    /// so "unchanged" is decided on what is edited, not on an earlier look.
    pub fn judge(&self, path: &str, current: &FileFacts) -> Freshness {
        let Some(recorded) = self.recorded(path) else { return Freshness::Unread };
        match (&recorded.sha256, &current.sha256) {
            (Some(before), Some(now)) if before == now && recorded.size == current.size => Freshness::Unchanged,
            (Some(_), Some(_)) => Freshness::Changed,
            _ => Freshness::Unverifiable,
        }
    }
}
