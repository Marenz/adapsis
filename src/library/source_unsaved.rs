//! Which live modules are NOT saved as source, and why (#146).
//!
//! One record, written only by the managed writer at the end of every flush
//! (under its Program read guard), read by the mutation result, the finish
//! gate, `?library` and the runtime tail. It cannot go stale: a later flush
//! that saves a module removes its entry, a module that is no longer live is
//! removed, and a new process starts with an empty record (a start loads the
//! source or adopts or drops the cached module).
//!
//! `dirty` names modules an entry point changed and whose save has not been
//! confirmed yet. The writer clears it when a flush completes.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::LibraryState;

/// Longest an entry point waits for its save (#146). The writer itself is not
/// cancelled; it finishes in the background and updates the record.
pub const SAVE_WAIT_MS: u64 = 10_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnsavedKind {
    /// A member or stored test cannot be written as Python.
    Unprintable,
    /// The printed form of an expression exceeds the stored-source nesting limit.
    Nesting,
    /// Its text needs these modules, whose source is not saved.
    WaitsFor(Vec<String>),
    /// On a cross-module cycle with these modules.
    Cycle(Vec<String>),
    /// The save was not confirmed within the wait bound (still running).
    NotConfirmed,
    /// The source file or the manifest changed on disk; nothing was overwritten.
    ChangedOnDisk,
    /// Disk or permission error while writing.
    WriteFailed,
    Other,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unsaved {
    pub kind: UnsavedKind,
    /// Categorical reason, never source text.
    pub detail: String,
}

impl Unsaved {
    pub(super) fn new(kind: UnsavedKind, detail: impl std::fmt::Display) -> Self {
        let detail: String = detail.to_string().chars().take(400).collect();
        Self { kind, detail }
    }

    pub fn not_confirmed() -> Self {
        Self::new(UnsavedKind::NotConfirmed, format!("the save was not confirmed within {} s and continues in the background", SAVE_WAIT_MS / 1000))
    }

    /// Short reason for one-line lists.
    pub fn reason(&self) -> String {
        match &self.kind {
            UnsavedKind::Unprintable => "a member cannot be written as Python".into(),
            UnsavedKind::Nesting => "the printed form of an expression is nested too deeply".into(),
            UnsavedKind::WaitsFor(names) => format!("waits for {}", list(names)),
            UnsavedKind::Cycle(names) => format!("cycle with {}", list(names)),
            UnsavedKind::NotConfirmed => "save not confirmed yet".into(),
            UnsavedKind::ChangedOnDisk => "the file or the manifest changed on disk".into(),
            UnsavedKind::WriteFailed => "disk or permission error".into(),
            UnsavedKind::Other => "the writer refused it".into(),
        }
    }

    /// The complete result line of a mutation whose module was not saved. The
    /// remedy depends on the reason; an environment cause never suggests
    /// repeating or changing the mutation.
    pub fn result_line(&self, module: &str) -> String {
        let lost = "The edit is live in memory and will be lost at the next restart";
        let head = format!("module `{module}` NOT saved: {}.", self.detail.trim_end_matches('.'));
        let no_retry = "Do not repeat the mutation";
        match &self.kind {
            UnsavedKind::Nesting => format!("{head} {lost} unless you split the expression into typed local bindings or helper functions."),
            UnsavedKind::Unprintable => format!("{head} {lost} unless you change that member so it can be written as Python."),
            UnsavedKind::WaitsFor(names) => format!("{head} {lost} unless you save or fix module {} first.", list(names)),
            UnsavedKind::Cycle(names) => format!("{head} {lost} unless you break the cycle between `{module}` and {}: move what both need into a third module that neither calls back from.", list(names)),
            UnsavedKind::ChangedOnDisk => format!("{head} {lost}. The file or the manifest changed on disk and nothing was overwritten: inspect it and tell the owner. {no_retry}."),
            UnsavedKind::WriteFailed => format!("{head} {lost}. This is a disk or permission error: tell the owner. {no_retry}."),
            UnsavedKind::NotConfirmed => format!("{head} Treat the edit as not saved until `inspect library` shows no unsaved line. {no_retry}."),
            UnsavedKind::Other => format!("{head} {lost}. Tell the owner. {no_retry}."),
        }
    }
}

fn list(names: &[String]) -> String {
    if names.is_empty() {
        return "another module".into();
    }
    names.iter().map(|name| format!("`{name}`")).collect::<Vec<_>>().join(", ")
}

/// Can the model that made the edit fix the cause itself? A module that waits
/// for others is fixable only if every one of them is.
pub fn fixable(record: &BTreeMap<String, Unsaved>, module: &str) -> bool {
    fn walk(record: &BTreeMap<String, Unsaved>, module: &str, seen: &mut BTreeSet<String>) -> bool {
        if !seen.insert(module.to_string()) {
            return true;
        }
        match record.get(module).map(|entry| &entry.kind) {
            Some(UnsavedKind::Unprintable | UnsavedKind::Nesting | UnsavedKind::Cycle(_)) => true,
            Some(UnsavedKind::WaitsFor(names)) => !names.is_empty() && names.iter().all(|name| walk(record, name, seen)),
            _ => false,
        }
    }
    walk(record, module, &mut BTreeSet::new())
}

/// One line for `?library`, `inspect library` and the runtime tail; `None` when
/// nothing is unsaved (callers then add nothing at all).
pub fn unsaved_line(record: &BTreeMap<String, Unsaved>, visible: impl Fn(&str) -> bool) -> Option<String> {
    let items: Vec<String> =
        record.iter().filter(|(name, _)| visible(name)).map(|(name, entry)| format!("{name} ({})", entry.reason())).collect();
    (!items.is_empty()).then(|| format!("Source NOT saved (live in memory only, lost at the next restart): {}", items.join("; ")))
}

/// A writer refusal with its kind, so the cause survives `anyhow` contexts.
#[derive(Debug)]
pub(super) struct Refusal {
    pub(super) kind: UnsavedKind,
    text: String,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.text)
    }
}

impl std::error::Error for Refusal {}

pub(super) fn refusal(kind: UnsavedKind, text: impl std::fmt::Display) -> anyhow::Error {
    anyhow::Error::new(Refusal { kind, text: text.to_string() })
}

/// The kind an error carries, else `default`.
pub(super) fn classify(error: &anyhow::Error, default: UnsavedKind) -> Unsaved {
    let kind = error.chain().find_map(|cause| cause.downcast_ref::<Refusal>()).map(|refusal| refusal.kind.clone());
    Unsaved::new(kind.unwrap_or(default), error)
}

#[derive(Clone, Debug)]
pub struct SaveStatus {
    record: Arc<Mutex<BTreeMap<String, Unsaved>>>,
    dirty: Arc<Mutex<BTreeSet<String>>>,
    /// Turns (by the address of their completion baseline) whose `finish` was
    /// refused once for an unsaved edit. Bounded; an entry leaves on acceptance.
    refused_turns: Arc<Mutex<Vec<usize>>>,
    wait_ms: Arc<AtomicU64>,
    /// The directory this library was loaded from (the writer saves there).
    dir: Arc<Mutex<Option<std::path::PathBuf>>>,
    /// Test hook: every flush waits here BEFORE it takes any lock.
    #[cfg(test)]
    gate: Arc<Mutex<Option<std::sync::mpsc::Receiver<()>>>>,
}

impl Default for SaveStatus {
    fn default() -> Self {
        Self {
            record: Arc::default(),
            dirty: Arc::default(),
            refused_turns: Arc::default(),
            wait_ms: Arc::new(AtomicU64::new(SAVE_WAIT_MS)),
            dir: Arc::default(),
            #[cfg(test)]
            gate: Arc::default(),
        }
    }
}

fn guard<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl LibraryState {
    /// Modules an entry point changed; cleared by the writer when a flush ends.
    pub fn mark_dirty(&self, names: &BTreeSet<String>) {
        guard(&self.saves.dirty).extend(names.iter().cloned());
    }

    /// The writer's record only (after a completed flush this is exact).
    pub fn unsaved_record(&self) -> BTreeMap<String, Unsaved> {
        guard(&self.saves.record).clone()
    }

    /// The record plus modules whose save is not confirmed yet.
    pub fn unsaved(&self) -> BTreeMap<String, Unsaved> {
        let mut all = self.unsaved_record();
        for name in guard(&self.saves.dirty).iter() {
            all.entry(name.clone()).or_insert_with(Unsaved::not_confirmed);
        }
        all
    }

    pub fn has_unconfirmed_saves(&self) -> bool {
        !guard(&self.saves.dirty).is_empty()
    }

    pub fn save_wait(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.saves.wait_ms.load(Ordering::Relaxed))
    }

    pub(super) fn set_source_dir(&self, dir: &std::path::Path) {
        *guard(&self.saves.dir) = Some(dir.to_path_buf());
    }

    pub(super) fn source_dir(&self) -> Option<std::path::PathBuf> {
        guard(&self.saves.dir).clone()
    }

    /// `X.py` (or `X.ax` for a transitional module) of a listed module.
    pub fn source_file_name(&self, module: &str) -> Option<String> {
        let managed = self.managed_sources.lock().ok()?;
        let source = managed.as_ref()?.manifest.modules.get(module)?;
        Some(format!("{module}.{}", source.format.extension()))
    }

    /// End of a flush, called while the writer still holds its Program read
    /// guard: no write-back can have happened since the Program was read, so
    /// every dirty mark is answered by this flush.
    pub(super) fn settle_saves(&self, live: &BTreeSet<String>, saved: &[String], failed: BTreeMap<String, Unsaved>) {
        let mut record = guard(&self.saves.record);
        record.retain(|name, _| live.contains(name));
        for name in saved {
            record.remove(name);
        }
        record.extend(failed.into_iter().filter(|(name, _)| live.contains(name)));
        guard(&self.saves.dirty).clear();
    }

    /// The flush could not even check the policy: every module whose save was
    /// requested stays unsaved for that reason.
    pub(super) fn settle_policy_failure(&self, live: &BTreeSet<String>, cause: &Unsaved) {
        let mut record = guard(&self.saves.record);
        record.retain(|name, _| live.contains(name));
        let mut dirty = guard(&self.saves.dirty);
        for name in dirty.iter().filter(|name| live.contains(*name)) {
            record.insert(name.clone(), cause.clone());
        }
        dirty.clear();
    }

    /// True the first time it is asked for this turn; false afterwards.
    pub fn first_finish_refusal(&self, turn: usize) -> bool {
        let mut turns = guard(&self.saves.refused_turns);
        if turns.contains(&turn) {
            return false;
        }
        if turns.len() >= 16 {
            turns.remove(0);
        }
        turns.push(turn);
        true
    }

    pub fn finish_accepted(&self, turn: usize) {
        guard(&self.saves.refused_turns).retain(|known| *known != turn);
    }

    #[cfg(test)]
    pub(crate) fn set_save_wait_for_tests(&self, wait: std::time::Duration) {
        self.saves.wait_ms.store(wait.as_millis() as u64, Ordering::Relaxed);
    }

    /// Test hook: every flush waits before taking any lock until the returned
    /// sender is dropped (or sends once per flush).
    #[cfg(test)]
    pub(crate) fn hold_flushes_for_tests(&self) -> std::sync::mpsc::Sender<()> {
        let (release, wait) = std::sync::mpsc::channel();
        *guard(&self.saves.gate) = Some(wait);
        release
    }

    pub(super) fn wait_at_flush_gate(&self) {
        #[cfg(test)]
        {
            // Held across the wait on purpose: later flushes queue behind it.
            let mut gate = guard(&self.saves.gate);
            if gate.as_ref().is_some_and(|wait| wait.recv().is_err()) {
                *gate = None;
            }
        }
    }
}
