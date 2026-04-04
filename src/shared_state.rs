//! Thread-local shared state that must be accessible from both `eval` and
//! `coroutine` without creating a circular dependency.
//!
//! Previously these thread-locals and their accessors lived in `eval/mod.rs`,
//! which forced `coroutine.rs` to import from `eval`. By moving them here,
//! both crates can depend on `shared_state` independently.
//!
//! `eval/mod.rs` re-exports everything via `pub use crate::shared_state::*;`
//! so all existing `eval::set_shared_*` / `eval::get_shared_*` call sites
//! continue to compile without changes.

std::thread_local! {
    /// Thread-local SharedRuntime so newly-created Env instances automatically
    /// have access to +shared variables without explicit plumbing.
    static SHARED_RUNTIME: std::cell::RefCell<Option<crate::session::SharedRuntime>> = std::cell::RefCell::new(None);
    /// Thread-local SharedMeta so roadmap/plan/mock builtins in coroutine.rs
    /// access the same data as handle_roadmap() in session.rs. No more syncing.
    static SHARED_META: std::cell::RefCell<Option<crate::session::SharedMeta>> = std::cell::RefCell::new(None);
    /// Thread-local SSE broadcast sender so IO builtins can emit API events.
    static SHARED_EVENT_BROADCAST: std::cell::RefCell<Option<tokio::sync::broadcast::Sender<String>>> = std::cell::RefCell::new(None);
    /// Thread-local Program snapshot for query builtins (query_symbols, query_source, etc.)
    /// that need access to the AST from within coroutine IO dispatch.
    static SHARED_PROGRAM: std::cell::RefCell<Option<std::sync::Arc<crate::ast::Program>>> = std::cell::RefCell::new(None);
    /// Thread-local mutable Program reference for mutation builtins (mutate, fn_remove, etc.)
    /// that need write access to the AST from within coroutine IO dispatch.
    static SHARED_PROGRAM_MUT: std::cell::RefCell<Option<std::sync::Arc<std::sync::RwLock<crate::ast::Program>>>> = std::cell::RefCell::new(None);
}

/// Set the thread-local SharedRuntime for +shared variable access.
/// Call this before any eval functions that need shared variable support.
pub fn set_shared_runtime(rt: Option<crate::session::SharedRuntime>) {
    SHARED_RUNTIME.with(|s| *s.borrow_mut() = rt);
}

/// Get the thread-local SharedRuntime (if set). Used by coroutine.rs for
/// roadmap builtins that need access to runtime state.
pub fn get_shared_runtime() -> Option<crate::session::SharedRuntime> {
    SHARED_RUNTIME.with(|s| s.borrow().clone())
}

/// Set the thread-local SharedMeta for roadmap/plan/mock builtin access.
/// Call this alongside set_shared_runtime before spawn_blocking eval tasks.
pub fn set_shared_meta(meta: Option<crate::session::SharedMeta>) {
    SHARED_META.with(|s| *s.borrow_mut() = meta);
}

/// Get the thread-local SharedMeta (if set). Used by coroutine.rs for
/// roadmap_add/roadmap_done/plan_set/plan_done/plan_fail builtins.
pub fn get_shared_meta() -> Option<crate::session::SharedMeta> {
    SHARED_META.with(|s| s.borrow().clone())
}

pub fn set_shared_event_broadcast(sender: Option<tokio::sync::broadcast::Sender<String>>) {
    SHARED_EVENT_BROADCAST.with(|s| *s.borrow_mut() = sender);
}

pub fn get_shared_event_broadcast() -> Option<tokio::sync::broadcast::Sender<String>> {
    SHARED_EVENT_BROADCAST.with(|s| s.borrow().clone())
}

/// Set the thread-local Program snapshot for query builtins.
/// Call this alongside set_shared_runtime before spawn_blocking eval tasks.
pub fn set_shared_program(program: Option<std::sync::Arc<crate::ast::Program>>) {
    SHARED_PROGRAM.with(|s| *s.borrow_mut() = program);
}

/// Get the thread-local Program snapshot (if set). Used by coroutine.rs for
/// query builtins (query_symbols, query_source, etc.) that need AST access.
pub fn get_shared_program() -> Option<std::sync::Arc<crate::ast::Program>> {
    SHARED_PROGRAM.with(|s| s.borrow().clone())
}

/// Set the thread-local mutable Program reference for mutation builtins.
/// Call this alongside set_shared_runtime before spawn_blocking eval tasks.
pub fn set_shared_program_mut(
    program: Option<std::sync::Arc<std::sync::RwLock<crate::ast::Program>>>,
) {
    SHARED_PROGRAM_MUT.with(|s| *s.borrow_mut() = program);
}

/// Get the thread-local mutable Program reference (if set). Used by coroutine.rs for
/// mutation builtins (mutate, fn_remove, type_remove, module_remove) that need write access.
pub fn get_shared_program_mut() -> Option<std::sync::Arc<std::sync::RwLock<crate::ast::Program>>> {
    SHARED_PROGRAM_MUT.with(|s| s.borrow().clone())
}

/// Create a shared mutable program wrapper for use in `spawn_blocking` contexts.
/// Returns the `Arc<RwLock<Program>>` so the caller can read back mutations after
/// the blocking task completes. Pass `arc.clone()` to `set_shared_program_mut()`
/// inside the `spawn_blocking` closure, then call `read_back_program_mutations()`
/// after the task returns.
pub fn make_shared_program_mut(
    program: &crate::ast::Program,
) -> std::sync::Arc<std::sync::RwLock<crate::ast::Program>> {
    std::sync::Arc::new(std::sync::RwLock::new(program.clone()))
}

/// Read back a potentially-mutated program from the shared mutable wrapper.
/// Returns `Some(program)` if the lock can be acquired, `None` on lock error.
/// The caller should compare with the original program to detect mutations.
pub fn read_back_program_mutations(
    program_mut: &std::sync::Arc<std::sync::RwLock<crate::ast::Program>>,
) -> Option<crate::ast::Program> {
    program_mut.read().ok().map(|p| p.clone())
}

/// Bundles all shared state that must be installed on each eval worker thread.
///
/// Instead of calling 5 separate `set_shared_*()` functions at every
/// `spawn_blocking` boundary, construct an `EvalContext` once and call
/// `ctx.install()` inside the closure. This is the first step toward
/// replacing thread-local globals with explicit parameter passing.
///
/// # Usage
/// ```ignore
/// let ctx = EvalContext::new(runtime, meta, broadcast, program_snapshot, program_mut);
/// tokio::task::spawn_blocking(move || {
///     ctx.install();
///     // ... eval code ...
/// });
/// ```
#[derive(Clone)]
pub struct EvalContext {
    pub runtime: Option<crate::session::SharedRuntime>,
    pub meta: Option<crate::session::SharedMeta>,
    pub event_broadcast: Option<tokio::sync::broadcast::Sender<String>>,
    pub program_snapshot: Option<std::sync::Arc<crate::ast::Program>>,
    pub program_mut: Option<std::sync::Arc<std::sync::RwLock<crate::ast::Program>>>,
}

impl EvalContext {
    /// Create a new context with all fields populated.
    pub fn new(
        runtime: crate::session::SharedRuntime,
        meta: crate::session::SharedMeta,
        event_broadcast: tokio::sync::broadcast::Sender<String>,
        program: &crate::ast::Program,
        program_mut: std::sync::Arc<std::sync::RwLock<crate::ast::Program>>,
    ) -> Self {
        Self {
            runtime: Some(runtime),
            meta: Some(meta),
            event_broadcast: Some(event_broadcast),
            program_snapshot: Some(std::sync::Arc::new(program.clone())),
            program_mut: Some(program_mut),
        }
    }

    /// Create a minimal context (no event broadcast — used by main.rs spawn paths).
    pub fn new_minimal(
        runtime: crate::session::SharedRuntime,
        meta: crate::session::SharedMeta,
        program: &crate::ast::Program,
        program_mut: std::sync::Arc<std::sync::RwLock<crate::ast::Program>>,
    ) -> Self {
        Self {
            runtime: Some(runtime),
            meta: Some(meta),
            event_broadcast: None,
            program_snapshot: Some(std::sync::Arc::new(program.clone())),
            program_mut: Some(program_mut),
        }
    }

    /// Create an empty context (for tests or paths that don't need shared state).
    pub fn empty() -> Self {
        Self {
            runtime: None,
            meta: None,
            event_broadcast: None,
            program_snapshot: None,
            program_mut: None,
        }
    }

    /// Install all fields into the current thread's thread-local globals.
    /// Call this at the top of every `spawn_blocking` closure.
    pub fn install(&self) {
        set_shared_runtime(self.runtime.clone());
        set_shared_meta(self.meta.clone());
        set_shared_event_broadcast(self.event_broadcast.clone());
        set_shared_program(self.program_snapshot.clone());
        set_shared_program_mut(self.program_mut.clone());
    }

    /// Install and also set up the display interner from a program.
    /// Use this when the worker thread needs to format `Value` types.
    pub fn install_with_interner(&self, program: &crate::ast::Program) {
        self.install();
        crate::intern::set_display_interner(&program.shared_interner);
    }
}
