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

use super::{AppConfig, WorkingSet};
use crate::ast::Program;
use crate::library::{fixable, LibraryState, Unsaved};
use crate::permissions::PermissionLevel;

/// (result line, is it good news).
pub(super) type SaveLines = Vec<(String, bool)>;

fn governed(state: Option<&LibraryState>) -> Option<LibraryState> {
    state.filter(|state| state.has_source_manifest().unwrap_or(true)).cloned()
}

impl AppConfig {
    /// The library state when a source manifest governs the library.
    pub(super) fn governed_library(&self) -> Option<LibraryState> {
        governed(self.meta.lock().unwrap().library_state.as_ref())
    }

    fn may_read(&self, module: &str) -> bool {
        let model = self.llm_model.read().unwrap().clone();
        self.permission_config.resolve(self.access_level, &model, module) >= PermissionLevel::Read
    }

    /// Commit the working set and save the modules it changed since the last
    /// report. Returns nothing when no module changed (no flush happens then).
    pub(super) async fn save_changed_sources(&self, ws: &mut WorkingSet) -> SaveLines {
        let Some(state) = governed(ws.meta.library_state.as_ref()) else {
            return Vec::new();
        };
        let Some(reference) = ws.source_checkpoint.as_ref().or(ws.base_program.as_ref()) else {
            return Vec::new();
        };
        let changed = Program::changed_module_names(reference, &ws.program);
        if changed.is_empty() {
            return Vec::new();
        }
        self.write_back_working_set(ws).await;
        ws.source_checkpoint = Some(ws.program.clone());
        self.save_modules(&state, changed).await
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

    /// The finish gate for unsaved edits. `Some(text)`: refuse this finish.
    /// `None`: accept; `notice` then holds the runtime line for the visible
    /// reply when something this turn changed is not saved.
    ///
    /// Refused at most once per turn, and only when a module this turn changed
    /// is unsaved for a reason the model can fix. A cause that is not the
    /// model's (disk, external edit, unconfirmed save) is never refused.
    pub(super) async fn unsaved_finish_gate(
        &self,
        ws: &mut WorkingSet,
        baseline: Option<&Program>,
        notice: &mut Option<String>,
    ) -> Option<String> {
        let state = governed(ws.meta.library_state.as_ref())?;
        // This call's own changes first (a legacy block can mutate and finish).
        let _ = self.save_changed_sources(ws).await;
        if state.has_unconfirmed_saves() {
            let _ = self.save_modules(&state, BTreeSet::new()).await;
        }
        let baseline = baseline?;
        let turn = baseline as *const Program as usize;
        let record = state.unsaved();
        let changed = Program::changed_module_names(baseline, &ws.program);
        let mine: BTreeMap<&String, &Unsaved> = record.iter().filter(|(name, _)| changed.contains(*name)).collect();
        if mine.is_empty() {
            state.finish_accepted(turn);
            return None;
        }
        if mine.keys().any(|name| fixable(&record, name)) && state.first_finish_refusal(turn) {
            let reasons: Vec<String> = mine.iter().map(|(name, unsaved)| unsaved.result_line(name)).collect();
            return Some(format!(
                "Cannot accept finish yet: {} module(s) changed in this turn are NOT saved as source.\n{}\nFix or revert them, then finish. A second finish is accepted as it is, and the person is told what is not saved.",
                mine.len(),
                reasons.join("\n")
            ));
        }
        state.finish_accepted(turn);
        let items: Vec<String> = mine.iter().map(|(name, unsaved)| format!("{name} ({})", unsaved.reason())).collect();
        *notice = Some(format!(
            "[Runtime: not saved to source: {}. The change is live only until the next restart.]",
            items.join("; ")
        ));
        None
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
    let flush = crate::library::flush_live_sources(program.clone(), Some(state.clone()));
    let receipt = match tokio::time::timeout(state.save_wait(), flush).await {
        Ok(Ok(receipt)) => receipt,
        // Not confirmed in time (the writer goes on in the background), or the
        // writer could not run at all.
        Ok(Err(_)) | Err(_) => {
            let live = program.read().await;
            return changed
                .iter()
                .filter(|name| live.modules.iter().any(|module| &module.name == *name) && !name.starts_with("__") && *name != "RunToolScratch")
                .map(|name| (Unsaved::not_confirmed().result_line(name), false))
                .collect();
        }
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
        if let Some(unsaved) = receipt.unsaved.get(name) {
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
