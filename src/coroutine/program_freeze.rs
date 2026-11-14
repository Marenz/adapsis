//! Program freeze for native `run` blocks (#129).
//!
//! A run block never keeps Program changes. Refusing the program-changing builtins at
//! dispatch, before they touch anything, is what keeps that true without half-applied
//! effects: no private Program edit, no initialized `+shared` slot, no library hash update.
//! The flag is a field of the handle, so it travels with every clone (nested calls, user
//! awaits, `run_module_startups`) and, through `IoRequest::Spawn`, into spawned tasks.
//!
//! Not frozen, because the runtime starts them with a handle of their own: the nested
//! conversation of `llm_takeover`, reply callbacks of `conversation_notify`, and timer or
//! event-source handlers registered by a called function.
use super::{CoroutineHandle, IoRequest, TaskId, TaskRegistry, TaskSnapshotRegistry};
use anyhow::{bail, Result};
use tokio::sync::mpsc;

impl CoroutineHandle {
    /// Freeze the Program for everything this handle (and every task it spawns) executes.
    pub fn with_program_frozen(mut self, frozen: bool) -> Self {
        self.program_frozen = frozen;
        self
    }

    pub fn program_frozen(&self) -> bool {
        self.program_frozen
    }

    /// The handle of a task started by an `IoRequest::Spawn`: it inherits the spawning
    /// handle's Program freeze. Every IO loop builds spawned-task handles through this.
    pub fn for_spawned_task(
        io_tx: mpsc::Sender<IoRequest>,
        task_id: TaskId,
        registry: TaskRegistry,
        snapshot_registry: TaskSnapshotRegistry,
        program_frozen: bool,
    ) -> Self {
        Self::new_with_task(io_tx, task_id, registry, snapshot_registry)
            .with_program_frozen(program_frozen)
    }

    /// The refusal a frozen handle gives a program-changing builtin. Checked first in
    /// `execute_await`: no Program, runtime, library or file state is reached.
    pub(super) fn check_program_frozen(&self, op: &str) -> Result<()> {
        if self.program_frozen && crate::builtins::program_effects::changes_program(op) {
            bail!(
                "{op}: refused. This code runs inside a native `run` block, and a run block \
                 cannot change the program. Use the mutate tool to declare, replace or remove code."
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
