//! The interpreter's stack (#129, review finding F4).
//!
//! The tree-walker recurses on the native stack: one level of user-function recursion
//! costs 10 to 20 KiB, so the call-depth guard (256) never held on a 2 MiB worker stack;
//! the stack overflowed first, and a stack overflow aborts the whole process.
//!
//! Two parts:
//! * [`check`] runs at every call, on any thread: it refuses the call with an ordinary
//!   error when less than [`STACK_RESERVE_BYTES`] of stack is left. The bounds of the
//!   current thread's stack are asked from the system once per thread.
//! * [`on_deep_stack`] runs an evaluation on a thread with [`EVAL_STACK_BYTES`] of stack,
//!   so the call-depth guard is reached long before the stack check. `run` blocks and
//!   async `evaluate` use it.
use crate::run_limits::{EVAL_STACK_BYTES, STACK_RESERVE_BYTES};
use anyhow::{bail, Result};
use std::cell::Cell;

const UNKNOWN: usize = usize::MAX;

thread_local! {
    /// Lowest stack address a call may start at on this thread. 0 = not asked yet.
    static FLOOR: Cell<usize> = const { Cell::new(0) };
    #[cfg(test)]
    static LOWEST_SEEN: Cell<usize> = const { Cell::new(usize::MAX) };
}

/// An address inside the current stack frame.
#[inline(always)]
fn here() -> usize {
    let marker = 0u8;
    std::hint::black_box(&marker) as *const u8 as usize
}

/// Lowest usable address of this thread's stack plus the reserve, or [`UNKNOWN`].
#[cfg(target_os = "linux")]
fn floor_of_this_thread() -> usize {
    // SAFETY: plain libc calls on the calling thread with an attribute object that is
    // initialized by `pthread_getattr_np` and destroyed before returning.
    unsafe {
        let mut attributes: libc::pthread_attr_t = std::mem::zeroed();
        if libc::pthread_getattr_np(libc::pthread_self(), &mut attributes) != 0 {
            return UNKNOWN;
        }
        let mut lowest: *mut libc::c_void = std::ptr::null_mut();
        let mut size: libc::size_t = 0;
        let found = libc::pthread_attr_getstack(&attributes, &mut lowest, &mut size);
        libc::pthread_attr_destroy(&mut attributes);
        if found != 0 || lowest.is_null() || size == 0 {
            return UNKNOWN;
        }
        // A stack smaller than twice the reserve keeps a quarter of itself as reserve.
        (lowest as usize).saturating_add(STACK_RESERVE_BYTES.min(size / 4))
    }
}

#[cfg(not(target_os = "linux"))]
fn floor_of_this_thread() -> usize {
    UNKNOWN
}

/// Refuse a call that would start with less than the reserve of stack left.
#[inline]
pub(super) fn check(callee: &str) -> Result<()> {
    let floor = FLOOR.with(|floor| {
        let known = floor.get();
        if known != 0 {
            return known;
        }
        let asked = floor_of_this_thread();
        floor.set(asked);
        asked
    });
    let here = here();
    #[cfg(test)]
    LOWEST_SEEN.with(|lowest| lowest.set(lowest.get().min(here)));
    if floor != UNKNOWN && here < floor {
        bail!(
            "call stack exhausted in `{callee}`: the calls are nested too deeply for this thread's stack \
             (possible infinite recursion)"
        );
    }
    Ok(())
}

/// Run `work` on a thread with the evaluation stack, inside the caller's tokio runtime
/// context if it has one. A panic in `work` propagates to the caller unchanged.
pub fn on_deep_stack<T: Send>(work: impl FnOnce() -> T + Send) -> std::io::Result<T> {
    let runtime = tokio::runtime::Handle::try_current().ok();
    std::thread::scope(|scope| {
        let thread = std::thread::Builder::new()
            .name("adapsis-eval".into())
            .stack_size(EVAL_STACK_BYTES)
            .spawn_scoped(scope, move || {
                let _context = runtime.as_ref().map(|handle| handle.enter());
                work()
            })?;
        match thread.join() {
            Ok(value) => Ok(value),
            Err(panic) => std::panic::resume_unwind(panic),
        }
    })
}

/// Test-only: stack bytes used below `top` at the deepest call checked since `reset`.
#[cfg(test)]
pub(crate) fn deepest_below(top: usize) -> usize {
    top.saturating_sub(LOWEST_SEEN.with(Cell::get))
}

#[cfg(test)]
pub(crate) fn reset_and_mark() -> usize {
    LOWEST_SEEN.with(|lowest| lowest.set(usize::MAX));
    here()
}
