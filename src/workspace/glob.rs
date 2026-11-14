//! `glob`: the files of a tree whose path matches a pattern, newest first.

use std::os::unix::fs::MetadataExt;

use super::error::{Error, Result, shown_name};
use super::lifecycle::Workspace;
use super::limits::{GLOB_MAX_OUTPUT_BYTES, GLOB_MAX_RESULTS};
use super::resolve::{Access, Kind, display_of, resolve};
use super::secure_open::{Strategy, stat_beneath};
use super::walk::{Budget, GlobFilter, Stop, walker};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobEntry {
    /// Relative to the workspace root (full path in a read-only root).
    pub path: String,
    pub mtime_ns: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobResult {
    /// Newest first; ties in path order. At most `GLOB_MAX_RESULTS` paths and
    /// `GLOB_MAX_OUTPUT_BYTES` of them.
    pub entries: Vec<GlobEntry>,
    /// All matching paths, including those not in `entries`.
    pub total: u64,
    /// Set when a budget ended the walk; `total` is then a lower bound.
    pub stopped: Option<Stop>,
    /// The output bound this listing was cut to.
    pub max_bytes: usize,
}

/// List files (and symlinks, by name) under `path` that match `pattern`.
/// Directories are not listed. `.gitignore` applies; symlinks are not followed.
pub fn glob(workspace: &Workspace, pattern: &str, path: &str) -> Result<GlobResult> {
    glob_within(workspace, pattern, path, GLOB_MAX_OUTPUT_BYTES)
}

/// [`glob`] with a smaller bound on the rendered output (clamped to
/// `RESULT_MIN_BYTES ..= GLOB_MAX_OUTPUT_BYTES`).
pub fn glob_within(workspace: &Workspace, pattern: &str, path: &str, max_bytes: usize) -> Result<GlobResult> {
    let max_bytes = max_bytes.clamp(super::limits::RESULT_MIN_BYTES, GLOB_MAX_OUTPUT_BYTES);
    let start = resolve(workspace, if path.is_empty() { "." } else { path }, Access::Read)?;
    if start.kind != Kind::Directory {
        return Err(Error::WrongKind { path: start.supplied, expected: "a directory to list" });
    }
    let filter = GlobFilter::new(&start.real, pattern)?;
    let mut budget = Budget::new();
    let mut found: Vec<GlobEntry> = Vec::new();
    let mut total = 0u64;
    for entry in walker(&start.real) {
        if !budget.entry() {
            break;
        }
        let Ok(entry) = entry else { continue };
        if entry.file_type().is_none_or(|kind| kind.is_dir()) || !filter.matches(entry.path()) {
            continue;
        }
        // Stat beneath the area root without following symlinks: an entry
        // reached through a directory that was swapped for a link is dropped.
        let Ok(relative) = entry.path().strip_prefix(&start.base) else { continue };
        let Ok(metadata) = stat_beneath(&start.base, relative, Strategy::Auto) else { continue };
        total += 1;
        let mtime_ns = metadata.mtime().saturating_mul(1_000_000_000).saturating_add(metadata.mtime_nsec());
        found.push(GlobEntry { path: display_of(workspace, entry.path()), mtime_ns });
        if found.len() >= 4 * GLOB_MAX_RESULTS {
            newest_first(&mut found);
        }
    }
    newest_first(&mut found);
    let mut bytes = 0;
    let fitting = found.iter().take_while(|entry| {
        bytes += shown_name(&entry.path).len() + 1;
        bytes <= max_bytes - 512
    });
    let keep = fitting.count();
    found.truncate(keep);
    Ok(GlobResult { entries: found, total, stopped: budget.stopped, max_bytes })
}

/// Sort newest first and keep the cap; called repeatedly to bound memory.
fn newest_first(found: &mut Vec<GlobEntry>) {
    found.sort_by(|left, right| right.mtime_ns.cmp(&left.mtime_ns).then_with(|| left.path.cmp(&right.path)));
    found.truncate(GLOB_MAX_RESULTS);
}

impl GlobResult {
    /// The text the model sees: one path per line, then what was left out.
    pub fn render(&self) -> String {
        let mut out: Vec<String> = self.entries.iter().map(|entry| shown_name(&entry.path)).collect();
        let quoted = out.iter().any(|name| super::error::is_quoted_name(name));
        if self.entries.is_empty() {
            out.push("no paths match".to_string());
        }
        let hidden = self.total.saturating_sub(self.entries.len() as u64);
        if hidden > 0 {
            let bound = if self.stopped.is_some() { "at least " } else { "" };
            out.push(format!("[{bound}{hidden} more paths not shown (newest first, limit {GLOB_MAX_RESULTS} paths or {} bytes); narrow the pattern or the path]", self.max_bytes));
        }
        if let Some(stop) = self.stopped {
            out.push(format!("[note: {}]", stop.describe()));
        }
        if quoted {
            out.push(format!("[note: {}]", super::error::QUOTED_NAME_NOTE));
        }
        out.join("\n")
    }
}
