//! Short-lived execution snapshots and per-function CoW write-back.
use super::AppConfig;
use crate::session::{RuntimeState, SandboxState, SessionMeta};

/// Temporary mutable working state used by complex handlers.
/// Live truth is still the three tiers in AppConfig; this is just a local bag.
pub(crate) struct WorkingSet {
    pub(super) program: crate::ast::Program,
    pub(super) runtime: RuntimeState,
    pub(super) meta: SessionMeta,
    pub(super) sandbox: Option<SandboxState>,
    /// Snapshot of the live program at the moment this working set was created.
    /// Used by `write_back_working_set` to compute the per-function CoW delta
    /// (issue #9), so a concurrent writer touching a different function is not
    /// clobbered. `None` when the working set was built by hand (e.g. tests or
    /// llm_takeover), in which case write-back falls back to a full overwrite.
    pub(super) base_program: Option<crate::ast::Program>,
    /// The program as of the last source save this working set reported
    /// (`save_changed_sources`); `None`: nothing reported yet, compare with the base.
    pub(super) source_checkpoint: Option<crate::ast::Program>,
}

impl AppConfig {
    /// Build a temporary working snapshot from the live tiers.
    pub(super) async fn snapshot_working_set(&self) -> WorkingSet {
        let program = self.program.read().await.clone();
        WorkingSet {
            base_program: Some(program.clone()),
            program,
            runtime: self.runtime.read().unwrap().clone(),
            meta: self.meta.lock().unwrap().clone(),
            sandbox: None,
            source_checkpoint: None,
        }
    }

    /// Merge only this operation's base→new delta, preserving concurrent edits
    /// to other functions (issue #9). No write lock is held during execution.
    pub(super) async fn commit_program_delta(
        &self,
        base: &crate::ast::Program,
        new: &crate::ast::Program,
    ) {
        self.program.write().await.merge_changed_from(base, new);
    }

    /// Native calls always come from snapshot_working_set and use its CoW base.
    /// The legacy manually constructed WorkingSet retains its existing fallback.
    pub(super) async fn write_back_working_set(&self, ws: &WorkingSet) {
        match &ws.base_program {
            Some(base) => self.commit_program_delta(base, &ws.program).await,
            None => *self.program.write().await = ws.program.clone(),
        }
        *self.meta.lock().unwrap() = ws.meta.clone();
        if let Ok(mut rt) = self.runtime.write() {
            rt.shared_vars
                .replace_from(ws.runtime.shared_vars.snapshot());
            rt.http_routes = ws.runtime.http_routes.clone();
            rt.agent_mailbox = ws.meta.agent_mailbox.clone();
            if let Some(ref lib_state) = ws.meta.library_state {
                if let Ok(errs) = lib_state.errors.lock() {
                    rt.library_errors = errs.clone();
                }
                if let Ok(load_errs) = lib_state.load_errors.lock() {
                    rt.library_load_errors = load_errs
                        .iter()
                        .map(|le| (le.module_name.clone(), le.error.clone()))
                        .collect();
                }
            }
        }
        self.notify_save();
    }

    /// Pull shared async side effects into the snapshot before write-back.
    pub(super) fn sync_async_side_effects_into(&self, ws: &mut WorkingSet) {
        if let Ok(meta) = self.meta.lock() {
            ws.meta.roadmap = meta.roadmap.clone();
            ws.meta.plan = meta.plan.clone();
            ws.meta.agent_mailbox = meta.agent_mailbox.clone();
            ws.meta.io_mocks = meta.io_mocks.clone();
        }
        if let Ok(runtime) = self.runtime.read() {
            ws.runtime = runtime.clone();
        }
    }
}
