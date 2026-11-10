//! One evaluation on the blocking pool with the coroutine IO runtime.
//!
//! Shared by named-function evaluation (`!eval` / native `evaluate`) and the native `run`
//! tool, so both get the same thread-local state, the same coroutine handle bound to the
//! turn identity, and the same private Program wrapper for program-editing builtins.
//! `evaluate` merges that wrapper back. `run` passes [`ProgramPolicy::Frozen`]: its handle
//! refuses every program-changing builtin at dispatch, so there is nothing to merge.
use crate::api::AppConfig;
use crate::ast::Program;
use crate::coroutine::{CoroutineHandle, IoRequest};
use crate::eval::Value;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::api) enum ProgramPolicy {
    /// Program-editing builtins work on the private wrapper (named-function evaluation).
    Editable,
    /// Program-changing builtins are refused at dispatch, also in spawned tasks (`run`).
    Frozen,
}

pub(in crate::api) struct EvalFailure {
    /// The blocking task itself failed (panic or cancellation), not the evaluated code.
    pub task: bool,
    pub detail: String,
}

impl EvalFailure {
    /// `"eval error: …"` / `"eval task error: …"` for the given noun.
    pub(in crate::api) fn describe(&self, noun: &str) -> String {
        let task = if self.task { " task" } else { "" };
        format!("{noun}{task} error: {}", self.detail)
    }
}

pub(in crate::api) struct IoEvaluation {
    pub outcome: Result<Value, EvalFailure>,
    /// The Program the evaluation started from.
    pub base: Program,
    /// The private wrapper after evaluation (`None` if its lock was poisoned). It differs
    /// from `base` only when a program-editing builtin ran, which a frozen handle refuses.
    pub after: Option<Program>,
}

pub(in crate::api) async fn evaluate_with_io<F>(
    config: &AppConfig,
    program: &Program,
    sender: tokio::sync::mpsc::Sender<IoRequest>,
    policy: ProgramPolicy,
    evaluate: F,
) -> IoEvaluation
where
    F: FnOnce(&Program, CoroutineHandle) -> anyhow::Result<Value> + Send + 'static,
{
    let program = program.clone();
    // Base for the per-function CoW merge (issue #9).
    let base = program.clone();
    let program_mut = crate::eval::make_shared_program_mut(&program);
    let turn = config.turn.clone();
    let ctx = crate::eval::EvalContext::new(
        config.runtime.clone(),
        config.meta.clone(),
        config.event_broadcast.clone(),
        &program,
        program_mut.clone(),
    );
    let joined = tokio::task::spawn_blocking(move || {
        ctx.install();
        let handle = CoroutineHandle::new(sender)
            .with_turn(turn)
            .with_program_frozen(policy == ProgramPolicy::Frozen);
        evaluate(&program, handle)
    })
    .await;
    let outcome = match joined {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(error)) => Err(EvalFailure {
            task: false,
            detail: error.to_string(),
        }),
        Err(error) => Err(EvalFailure {
            task: true,
            detail: error.to_string(),
        }),
    };
    IoEvaluation {
        outcome,
        base,
        after: crate::eval::read_back_program_mutations(&program_mut),
    }
}
