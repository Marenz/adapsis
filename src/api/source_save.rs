//! The real outcome of the source save belongs in the result of the call that
//! changed a module (#146).
//!
//! A mutation runs on a clone and is committed to the live Program by the
//! write-back; the managed writer reads the live Program. So only the entry
//! point can know whether the source was saved: after its write-back it runs
//! the writer and reports "saved to X.py" or "NOT saved: <reason>".
//!
//! Nothing here holds a Program, runtime or session-meta lock across the wait:
//! the library state handle is cloned under a brief meta lock, and the writer
//! takes its own locks on a blocking thread (Program read, source transaction,
//! metadata). The conversation's turn lock is the caller's and stays held, as
//! it is across model calls.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use super::{AppConfig, WorkingSet};
use crate::ast::Program;
use crate::library::{fixable, LibraryState, Unsaved};
use crate::permissions::PermissionLevel;

/// (result line, is it good news).
pub(super) type SaveLines = Vec<(String, bool)>;

/// What ONE conversational turn did to module sources. A fresh value belongs
/// to every turn (`AppConfig::for_new_turn`, and the takeover's own config); the
/// clones a turn makes of its config share it.
#[derive(Clone, Default)]
pub struct TurnSaves(Arc<TurnSavesInner>);

#[derive(Default)]
struct TurnSavesInner {
    /// Modules this turn's own calls changed. Never derived from the live
    /// Program: other conversations change it while this turn runs.
    changed: Mutex<BTreeSet<String>>,
    /// `finish` was refused once for an unsaved edit. Never reset in a turn.
    refused: AtomicBool,
}

impl TurnSaves {
    fn record(&self, names: &BTreeSet<String>) {
        self.0.changed.lock().unwrap_or_else(std::sync::PoisonError::into_inner).extend(names.iter().cloned());
    }

    fn changed(&self) -> BTreeSet<String> {
        self.0.changed.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
    }

    /// True exactly once per turn.
    fn first_refusal(&self) -> bool {
        !self.0.refused.swap(true, Ordering::SeqCst)
    }
}

/// End of one executor call: its save lines, and for a requested finish the
/// gate's decision.
#[derive(Default)]
pub(super) struct CallSave {
    pub(super) lines: SaveLines,
    /// Refuse this finish (at most once per turn).
    pub(super) refusal: Option<String>,
    /// Runtime line for the visible reply when finish is accepted although a
    /// module this turn changed is not saved.
    pub(super) notice: Option<String>,
}

fn governed(state: Option<&LibraryState>) -> Option<LibraryState> {
    state.filter(|state| state.has_source_manifest().unwrap_or(true)).cloned()
}

impl AppConfig {
    /// Every conversational turn starts here (HTTP ask, streaming ask; the
    /// takeover builds its own config). A turn must not share this state.
    pub(super) fn for_new_turn(mut self) -> Self {
        self.turn_saves = TurnSaves::default();
        self
    }

    /// The library state when a source manifest governs the library.
    pub(super) fn governed_library(&self) -> Option<LibraryState> {
        governed(self.meta.lock().unwrap().library_state.as_ref())
    }

    fn may_read(&self, module: &str) -> bool {
        let model = self.llm_model.read().unwrap().clone();
        self.permission_config.resolve(self.access_level, &model, module) >= PermissionLevel::Read
    }

    fn speaker_may_write(&self) -> bool {
        let model = self.llm_model.read().unwrap().clone();
        self.permission_config.authority(self.access_level, &model).may_write
            && self.turn.as_ref().is_none_or(|turn| turn.may_write && turn.guest_tools.is_none())
    }

    /// The single place where an executor call waits for a source save. The
    /// working set is committed immediately BEFORE the wait and marked
    /// (`commit_before_wait`, one function for both), so the
    /// caller's write-back after the call is skipped: no snapshot taken before
    /// the wait is written back after it. Nothing touches the working set
    /// during the wait.
    ///
    /// Waits when this call changed a module, or when a finish is requested
    /// while a save of one of THIS turn's modules is still unconfirmed. Another
    /// conversation's unsaved or unconfirmed modules never make this turn wait.
    pub(super) async fn save_and_gate(&self, ws: &mut WorkingSet, finish_requested: bool, conversational: bool) -> CallSave {
        let mut out = CallSave::default();
        let Some(state) = governed(ws.meta.library_state.as_ref()) else {
            return out;
        };
        let changed = match ws.source_checkpoint.as_ref().or(ws.base_program.as_ref()) {
            Some(reference) => Program::changed_module_names(reference, &ws.program),
            None => BTreeSet::new(),
        };
        self.turn_saves.record(&changed);
        let mine = self.turn_saves.changed();
        if !changed.is_empty() || (finish_requested && state.has_unconfirmed(&mine)) {
            self.commit_before_wait(ws).await;
            ws.source_checkpoint = Some(ws.program.clone());
            out.lines = self.save_modules(&state, changed).await;
        }
        if finish_requested && conversational {
            self.finish_gate(&state, &mine, &mut out);
        }
        out
    }

    /// Refused at most once per turn, only for a speaker who may write, and
    /// only when a module this turn's own calls changed is unsaved for a reason
    /// the model can fix. A cause that is not the model's (disk, external edit,
    /// unconfirmed save) is never refused. Neither text names a module the
    /// speaker cannot read.
    fn finish_gate(&self, state: &LibraryState, mine: &BTreeSet<String>, out: &mut CallSave) {
        let record = state.unsaved();
        let unsaved: BTreeMap<&String, &Unsaved> = record.iter().filter(|(name, _)| mine.contains(*name)).collect();
        if unsaved.is_empty() {
            return;
        }
        let readable: BTreeMap<&String, &Unsaved> = unsaved.iter().filter(|(name, _)| self.may_read(name)).map(|(name, entry)| (*name, *entry)).collect();
        if self.speaker_may_write() && readable.keys().any(|name| fixable(&record, name)) && self.turn_saves.first_refusal() {
            let reasons: Vec<String> = readable.iter().map(|(name, unsaved)| unsaved.result_line(name)).collect();
            out.refusal = Some(format!(
                "Cannot accept finish yet: {} module(s) changed in this turn are NOT saved as source.\n{}\nFix or revert them, then finish. A second finish is accepted as it is, and the person is told what is not saved.",
                readable.len(),
                reasons.join("\n")
            ));
            return;
        }
        let items = if readable.is_empty() {
            format!("{} module(s) changed in this turn", unsaved.len())
        } else {
            readable.iter().map(|(name, unsaved)| format!("{name} ({})", unsaved.reason())).collect::<Vec<_>>().join("; ")
        };
        out.notice = Some(format!("[Runtime: not saved to source: {items}. The change is live only until the next restart.]"));
    }

    /// Run the writer for modules that are already committed to the live
    /// Program, wait for it (bounded), and describe the outcome.
    pub(super) async fn save_modules(&self, state: &LibraryState, changed: BTreeSet<String>) -> SaveLines {
        save_committed(self.program.clone(), state, changed, &|name| self.may_read(name)).await
    }

    /// `save_modules` for a delta that was just committed to the live Program.
    /// An unchanged Program (an empty mutation) runs no writer.
    pub(super) async fn save_committed_delta(&self, base: &Program, new: &Program) -> SaveLines {
        let changed = Program::changed_module_names(base, new);
        match self.governed_library() {
            Some(state) if !changed.is_empty() => self.save_modules(&state, changed).await,
            _ => Vec::new(),
        }
    }

    /// For callers nobody waits for (route handlers, `/api/eval`): start the
    /// writer now instead of waiting for the debounce. Failures reach the
    /// journal and the unsaved record.
    pub(crate) fn start_source_save(&self, base: &Program, new: &Program) {
        start_save(self.program.clone(), self.governed_library(), base, new);
    }

    /// See `unsaved_tail`.
    pub(crate) fn unsaved_tail(&self, model: &str) -> String {
        let state = self.meta.lock().unwrap().library_state.clone();
        unsaved_tail(state.as_ref(), &self.permission_config, self.access_level, model)
    }
}

/// One line for the volatile runtime tail (it follows the program summary) of
/// a write-capable speaker, naming only modules that speaker may read. Empty
/// when nothing is unsaved, so the request is byte-identical to one without
/// this feature.
pub(crate) fn unsaved_tail(
    state: Option<&LibraryState>,
    permissions: &crate::permissions::PermissionConfig,
    access: crate::permissions::AccessLevel,
    model: &str,
) -> String {
    if !permissions.authority(access, model).may_write {
        return String::new();
    }
    let Some(state) = governed(state) else {
        return String::new();
    };
    let visible = |name: &str| permissions.resolve(access, model, name) >= PermissionLevel::Read;
    crate::library::unsaved_line(&state.unsaved(), visible).map(|line| format!("{line}\n")).unwrap_or_default()
}

/// See `AppConfig::start_source_save`; for callers without an `AppConfig`.
pub(crate) fn start_save(
    program: std::sync::Arc<tokio::sync::RwLock<Program>>,
    state: Option<LibraryState>,
    base: &Program,
    new: &Program,
) {
    let Some(state) = governed(state.as_ref()) else {
        return;
    };
    let changed = Program::changed_module_names(base, new);
    if changed.is_empty() {
        return;
    }
    state.mark_dirty(&changed);
    tokio::spawn(async move {
        let _ = crate::library::flush_live_sources(program, Some(state)).await;
    });
}

/// The library state of `meta` when a source manifest governs the library.
pub(crate) fn governed_state(meta: &crate::session::SessionMeta) -> Option<LibraryState> {
    governed(meta.library_state.as_ref())
}

const HIDDEN_NOT_SAVED: &str = "a module this call changed was NOT saved as source (not named: no read permission). Tell the owner.";

/// Run the writer for modules that are already committed to the live Program,
/// wait for it (bounded by `LibraryState::save_wait`), and describe the outcome:
/// one line per changed module, then what else was saved or is still unsaved
/// (only modules `may_read` admits).
pub(crate) async fn save_committed(
    program: std::sync::Arc<tokio::sync::RwLock<Program>>,
    state: &LibraryState,
    changed: BTreeSet<String>,
    may_read: &(dyn Fn(&str) -> bool + Sync),
) -> SaveLines {
    let before: BTreeSet<String> = state.unsaved_record().into_keys().collect();
    state.mark_dirty(&changed);
    let flush = crate::library::flush_live_sources(program, Some(state.clone()));
    // Neither failure path below takes a lock: the writer may still hold its
    // Program read guard, and a queued Program writer would block a reader.
    let named = |unsaved: Unsaved| -> SaveLines {
        changed
            .iter()
            .filter(|name| !name.starts_with("__") && *name != "RunToolScratch")
            .map(|name| (if may_read(name) { unsaved.result_line(name) } else { HIDDEN_NOT_SAVED.to_string() }, false))
            .collect()
    };
    let receipt = match tokio::time::timeout(state.save_wait(), flush).await {
        Ok(Ok(receipt)) => receipt,
        // The writer returned an error: it is not running any more.
        Ok(Err(_)) => return named(Unsaved::writer_failed()),
        // Not confirmed in time: the writer goes on in the background.
        Err(_) => return named(Unsaved::not_confirmed()),
    };
    // The same log lines the autosave writes (names and counts only).
    for module in &receipt.enrolled {
        eprintln!("[save] module `{module}` added to the source manifest with its first source file");
    }
    for note in &receipt.notes {
        eprintln!("[save] {note}");
    }
    let mut lines = Vec::new();
    for name in &changed {
        if !may_read(name) {
            // A call can change a module its speaker cannot read (a function
            // it may execute mutates through a builtin): no name, no reason.
            if receipt.unsaved.contains_key(name) {
                lines.push((HIDDEN_NOT_SAVED.to_string(), false));
            }
        } else if let Some(unsaved) = receipt.unsaved.get(name) {
            lines.push((unsaved.result_line(name), false));
        } else if receipt.saved_names.contains(name) {
            let file = state.source_file_name(name).unwrap_or_else(|| format!("{name}.py"));
            let new = if receipt.enrolled.contains(name) { " (new module, added to the library)" } else { "" };
            lines.push((format!("saved to {file}{new}"), true));
        }
    }
    let also: Vec<String> = before
        .iter()
        .filter(|name| !changed.contains(*name) && !receipt.unsaved.contains_key(*name) && receipt.saved_names.contains(*name) && may_read(name))
        .filter_map(|name| state.source_file_name(name))
        .collect();
    if !also.is_empty() {
        lines.push((format!("Also saved now: {}", also.join(", ")), true));
    }
    let others: Vec<String> = receipt
        .unsaved
        .iter()
        .filter(|(name, _)| !changed.contains(*name) && may_read(name))
        .map(|(name, unsaved)| format!("{name} ({})", unsaved.reason()))
        .collect();
    if !others.is_empty() {
        lines.push((format!("Other modules still not saved: {}", others.join("; ")), true));
    }
    lines
}
