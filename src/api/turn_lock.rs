//! One turn at a time per conversation, and one piece of blocking workspace
//! work at a time per conversation.
//!
//! * The TURN lock: every conversational turn (`llm_takeover`, `/api/ask`,
//!   `/api/ask-stream`) holds it from start to end.
//! * The WORK lock: a workspace tool call holds it from before its blocking
//!   task starts until that task has really finished. The guard is owned by
//!   the blocking task, not by the turn's future, because a turn's future can
//!   be dropped while its blocking task runs on: `/api/ask` is dropped when
//!   its client disconnects (the handler future is the turn), any caller can
//!   wrap a turn in a timeout, and a task can be aborted. `/api/ask-stream`
//!   and `llm_takeover` turns run in spawned tasks nobody aborts today.
//!
//! A conversation is IDLE when both are free. Another conversation may take a
//! workspace from it only while holding both ([`idle`]).
//!
//! The turn lock is keyed by the conversation key. The work lock is keyed by the
//! workspace state root and the conversation key: bindings of two state roots
//! are two worlds (a production process has one; tests have one each). Both maps
//! are process-wide and exclude nothing in another process: one active daemon
//! per state root.

type Lock = std::sync::Arc<tokio::sync::Mutex<()>>;
type Guard = tokio::sync::OwnedMutexGuard<()>;

fn lock(kind: char, scope: &std::path::Path, context: &str) -> Lock {
    static LOCKS: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<(char, std::path::PathBuf, String), Lock>>> = std::sync::OnceLock::new();
    let locks = LOCKS.get_or_init(Default::default);
    let mut locks = locks.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    locks.entry((kind, scope.to_path_buf(), context.to_string())).or_default().clone()
}

/// The turn lock of `context`.
pub(super) fn of(context: &str) -> Lock {
    lock('t', std::path::Path::new(""), context)
}

/// The work lock of `context` in the state root `scope`, waiting at most `wait`
/// for an earlier piece of work of the same conversation. `None`: still running.
pub(super) async fn work(scope: &std::path::Path, context: &str, wait: std::time::Duration) -> Option<Guard> {
    tokio::time::timeout(wait, lock('w', scope, context).lock_owned()).await.ok()
}

/// The work lock of `context` if no work is running there. Never waits.
pub(super) fn work_now(scope: &std::path::Path, context: &str) -> Option<Guard> {
    lock('w', scope, context).try_lock_owned().ok()
}

/// Both locks of `context` if it is idle: no turn is running and no workspace
/// work is in flight. While the guards are held neither can start. Never waits.
pub(super) fn idle(scope: &std::path::Path, context: &str) -> Option<(Guard, Guard)> {
    let turn = of(context).try_lock_owned().ok()?;
    let work = work_now(scope, context)?;
    Some((turn, work))
}
