//! JIT admission, caching (#97/#119) and containment of compiler panics
//! (#96). This does NOT alter VM fallback semantics: the VM
//! execution/admission distinction belongs to the next safety slice.
//!
//! # What a cache hit means
//!
//! A cached module is used only for the program it was compiled from. That is
//! proven by [`CompileInput::matches`]: the live program must still have the
//! very functions and type declarations the compiler was given. The session
//! revision is compared too, as a cheap first test, but nothing relies on it:
//! writers do not keep it in step with the program (see `compiler::input`).
//! A remembered refusal is tied to the same input, so it is neither applied
//! to a different program that happens to carry the same revision number nor
//! retried for the same one.
//!
//! # A panicking compile (#96)
//!
//! Cranelift panics instead of returning an error for some conditions it does
//! not expect (the known one is a relocation between blocks more than 2 GiB
//! apart; `compiler::relocation` removes that cause, this module removes the
//! consequences of any other). A panic inside `compiler::compile` happens
//! before any code of the new module is executable, so it is safe to stop it
//! here:
//!
//! * the call continues on the VM/tree path, like any other JIT refusal;
//! * the cache remembers the refusal for that revision, so the same program is
//!   not compiled (and does not panic) again on every call;
//! * one journal entry names the event and where in the compiler it happened,
//!   without the panic's text (the process panic hook asks
//!   [`contained_panic`] and leaves the journal to this module);
//! * the cache lock is never left poisoned, and a lock poisoned by something
//!   else is recovered with an empty slot instead of silently disabling the
//!   JIT for the rest of the process.
//!
//! Only `compile` is contained. A panic while compiled code runs is not
//! caught: at that point effects of the call may already have happened.
use super::input_to_i64_args;
use crate::compiler::input::CompileInput;
use crate::{ast, compiler, parser};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex, MutexGuard};

/// What the cache holds: the input that was compiled, and what came of it.
pub struct JitSlot {
    input: CompileInput,
    outcome: Outcome,
}

enum Outcome {
    Compiled(compiler::CompiledProgram),
    /// Compiling this input panicked. It runs without the JIT.
    Refused,
}

impl JitSlot {
    /// Was this slot made from exactly what `live` would be compiled from?
    fn is_for(&self, live: &ast::Program) -> bool {
        self.input.matches(live)
    }

    #[cfg(test)]
    pub(crate) fn is_compiled(&self) -> bool {
        matches!(self.outcome, Outcome::Compiled(_))
    }

    #[cfg(test)]
    pub(crate) fn is_refused(&self) -> bool {
        matches!(self.outcome, Outcome::Refused)
    }
}

/// Cache for JIT-compiled programs, keyed by session revision.
/// When the revision matches, the compiled module is reused instead of recompiling.
pub type JitCache = Arc<Mutex<Option<(usize, JitSlot)>>>;

/// Create a new empty JIT cache.
pub fn new_jit_cache() -> JitCache {
    Arc::new(Mutex::new(None))
}

/// Test seam, per thread so parallel tests cannot affect each other.
#[cfg(test)]
pub(crate) mod seam {
    use std::cell::{Cell, RefCell};
    thread_local! {
        /// Runs inside every compile on this thread, where Cranelift would
        /// run: it may panic like Cranelift does, or observe which locks the
        /// caller holds while it evaluates.
        pub(crate) static COMPILE_HOOK: RefCell<Option<Box<dyn Fn()>>> = const { RefCell::new(None) };
        /// Compile attempts and journal entries made on this thread.
        pub(crate) static COMPILE_ATTEMPTS: Cell<u32> = const { Cell::new(0) };
        pub(crate) static PANICS_JOURNALED: Cell<u32> = const { Cell::new(0) };
    }

    /// Install `hook` for the current thread until the returned guard drops.
    pub(crate) fn hook(hook: impl Fn() + 'static) -> Reset {
        COMPILE_HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
        COMPILE_ATTEMPTS.with(|count| count.set(0));
        PANICS_JOURNALED.with(|count| count.set(0));
        Reset
    }

    pub(crate) struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            COMPILE_HOOK.with(|slot| *slot.borrow_mut() = None);
        }
    }

    pub(crate) fn attempts() -> u32 {
        COMPILE_ATTEMPTS.with(Cell::get)
    }

    pub(crate) fn journaled() -> u32 {
        PANICS_JOURNALED.with(Cell::get)
    }
}

thread_local! {
    /// True while this thread is inside the contained compile.
    static CONTAINING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// `file:line` of the panic the hook saw during the contained compile.
    static PANIC_LOCATION: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// For the process panic hook: is the panic it is looking at one this module
/// contains? If so its location is kept (file name and line only, never the
/// message) and the hook must not journal the panic itself: this module
/// writes the one journal entry.
pub fn contained_panic(location: Option<&std::panic::Location<'_>>) -> bool {
    if !CONTAINING.with(std::cell::Cell::get) {
        return false;
    }
    let place = location.map(|location| {
        let file = location.file().rsplit(['/', '\\']).next().unwrap_or_default();
        format!("{file}:{}", location.line())
    });
    PANIC_LOCATION.with(|slot| *slot.borrow_mut() = place);
    true
}

enum Compiled {
    Ready(compiler::CompiledProgram),
    /// An ordinary refusal (`Err`): cheap and not remembered, as before.
    Declined,
    Panicked,
}

/// Compile with panics contained. The partly built module is dropped during
/// unwinding; none of its code was ever made executable or called.
fn compile(input: &CompileInput) -> Compiled {
    let program = input.program();
    #[cfg(test)]
    seam::COMPILE_ATTEMPTS.with(|count| count.set(count.get() + 1));
    PANIC_LOCATION.with(|slot| *slot.borrow_mut() = None);
    CONTAINING.with(|flag| flag.set(true));
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        #[cfg(test)]
        seam::COMPILE_HOOK.with(|hook| {
            if let Some(hook) = hook.borrow().as_ref() {
                hook();
            }
        });
        compiler::compile(program)
    }));
    CONTAINING.with(|flag| flag.set(false));
    match outcome {
        Ok(Ok(compiled)) => Compiled::Ready(compiled),
        Ok(Err(_)) => Compiled::Declined,
        Err(payload) => {
            // The payload may quote program text; it is not journaled.
            drop(payload);
            #[cfg(test)]
            seam::PANICS_JOURNALED.with(|count| count.set(count.get() + 1));
            let place = PANIC_LOCATION.with(|slot| slot.borrow_mut().take()).unwrap_or_else(|| "an unrecorded location".to_string());
            crate::error_journal::record(
                crate::error_journal::Category::Panic,
                "jit.compile",
                format!("the JIT compiler panicked at {place}; this program revision runs without the JIT"),
            );
            Compiled::Panicked
        }
    }
}

/// Lock the cache. A poisoned lock means a panic happened while the slot was
/// in use; what it holds is not trusted, so the slot is emptied and the JIT
/// starts over, instead of every later call silently skipping the JIT.
fn lock(cache: &JitCache) -> MutexGuard<'_, Option<(usize, JitSlot)>> {
    cache.lock().unwrap_or_else(|poisoned| {
        let mut guard = poisoned.into_inner();
        *guard = None;
        cache.clear_poison();
        guard
    })
}

pub(super) fn try_execute(
    program: &ast::Program,
    function: &ast::FunctionDecl,
    name: &str,
    input: &parser::Expr,
    cache: Option<&JitCache>,
    revision: usize,
) -> Option<String> {
    // The current graph must be safe even on a cache hit. Never infer safety from the
    // old compiled module, or from the local root's eligibility alone.
    if !compiler::is_compilable_function(function)
        || compiler::failure_admission::check_root(function, program).is_err()
    {
        return None;
    }
    let args = input_to_i64_args(input, function).ok()?;
    let Some(cache) = cache else {
        return match compile(&CompileInput::of(program)) {
            Compiled::Ready(mut compiled) => invoke(&mut compiled, function, name, &args),
            Compiled::Declined | Compiled::Panicked => None,
        };
    };
    let mut slot = lock(cache);
    let hit = matches!(&*slot, Some((cached_revision, cached)) if *cached_revision == revision && cached.is_for(program));
    if !hit {
        // Drop the stale module BEFORE attempting a replacement. A failed compile
        // must never leave old code available to the call below (#119).
        *slot = None;
        let input = CompileInput::of(program);
        let outcome = match compile(&input) {
            Compiled::Ready(compiled) => Outcome::Compiled(compiled),
            Compiled::Declined => return None,
            Compiled::Panicked => Outcome::Refused,
        };
        *slot = Some((revision, JitSlot { input, outcome }));
    }
    match slot.as_mut() {
        Some((_, JitSlot { outcome: Outcome::Compiled(compiled), .. })) => invoke(compiled, function, name, &args),
        _ => None,
    }
}

fn invoke(
    compiled: &mut compiler::CompiledProgram,
    function: &ast::FunctionDecl,
    name: &str,
    args: &[i64],
) -> Option<String> {
    if matches!(function.return_type, ast::Type::String) {
        compiled
            .call_string(name, args)
            .ok()
            .map(|text| format!("\"{text}\""))
    } else {
        compiled
            .call_i64(name, args)
            .ok()
            .map(|value| value.to_string())
    }
}
