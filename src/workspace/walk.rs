//! Walking a tree for `search` and `glob`: one walker configuration, one glob
//! matcher, one budget.
//!
//! The walk honours `.gitignore` and `.git/info/exclude` of the repository the
//! tree belongs to. It deliberately ignores the owner's global excludes and
//! `.ignore` files, so the same tree gives the same answer on every machine.
//! Hidden files are included (CI and tool configuration live there); git's own
//! directory never is. Symlinks are not followed: a link is listed by name by
//! `glob` and skipped by `search`, so a walk cannot leave the tree it started in.

use std::path::Path;
use std::time::Instant;

use ignore::overrides::{Override, OverrideBuilder};
use ignore::{Walk, WalkBuilder};

use super::error::{Error, Result, shown};
use super::limits::{PATTERN_MAX_BYTES, WALK_MAX_ENTRIES, WALK_TIME_BUDGET};

pub fn walker(start: &Path) -> Walk {
    WalkBuilder::new(start)
        .hidden(false)
        .follow_links(false)
        .ignore(false)
        .git_global(false)
        .git_ignore(true)
        .git_exclude(true)
        .parents(true)
        .sort_by_file_name(|left, right| left.cmp(right))
        .filter_entry(|entry| entry.file_name() != ".git")
        .build()
}

/// A gitignore-style glob, as `rg -g` uses it: `*.rs` matches at any depth, a
/// pattern containing `/` is anchored at the directory the walk started in,
/// `**` crosses directories.
pub struct GlobFilter {
    matcher: Override,
}

impl GlobFilter {
    pub fn new(anchor: &Path, pattern: &str) -> Result<Self> {
        let invalid = |reason: &str| Error::InvalidPattern(format!("glob `{}` {reason}", shown(pattern)));
        if pattern.trim().is_empty() {
            return Err(invalid("is empty; give a pattern such as `**/*.rs`"));
        }
        if pattern.len() > PATTERN_MAX_BYTES {
            return Err(invalid(&format!("is longer than {PATTERN_MAX_BYTES} bytes")));
        }
        if pattern.contains(['\0', '\n', '\r']) {
            return Err(invalid("contains a NUL or line break"));
        }
        if pattern.starts_with('!') {
            return Err(invalid("starts with `!`; negated globs are not supported, describe what to include"));
        }
        let mut builder = OverrideBuilder::new(anchor);
        builder.add(pattern).map_err(|error| invalid(&format!("is not valid: {}", glob_reason(&error))))?;
        let matcher = builder.build().map_err(|error| invalid(&format!("is not valid: {}", glob_reason(&error))))?;
        Ok(Self { matcher })
    }

    /// Whether a file at `path` (inside the anchor) matches.
    pub fn matches(&self, path: &Path) -> bool {
        self.matcher.matched(path, false).is_whitelist()
    }
}

/// The glob crate's explanation without the pattern repeated in it.
fn glob_reason(error: &ignore::Error) -> String {
    let text = error.to_string();
    let reason = text.rsplit(": ").next().unwrap_or(&text);
    shown(reason)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// The wall-clock budget ran out.
    Time,
    /// The entry budget ran out.
    Entries,
}

impl Stop {
    pub fn describe(self) -> &'static str {
        match self {
            Stop::Time => "stopped after the time budget; the result is incomplete, narrow the path or pattern",
            Stop::Entries => "stopped after the entry budget; the result is incomplete, narrow the path",
        }
    }
}

/// Bounds one walk in time and in entries.
pub struct Budget {
    started: Instant,
    entries: u64,
    pub stopped: Option<Stop>,
}

impl Budget {
    pub fn new() -> Self {
        Self { started: Instant::now(), entries: 0, stopped: None }
    }

    /// Count one directory entry; false when the walk must stop.
    pub fn entry(&mut self) -> bool {
        self.entries += 1;
        if self.entries > WALK_MAX_ENTRIES {
            self.stopped = Some(Stop::Entries);
        }
        self.running()
    }

    /// Check the clock; false when the walk must stop.
    pub fn running(&mut self) -> bool {
        if self.stopped.is_none() && self.started.elapsed() > WALK_TIME_BUDGET {
            self.stopped = Some(Stop::Time);
        }
        self.stopped.is_none()
    }
}
