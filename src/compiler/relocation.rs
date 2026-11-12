//! Relocations the JIT may emit (#96).
//!
//! `cranelift-jit` 0.119 allocates every function and every data object as a
//! separate heap block, wherever the allocator puts it. A PC-relative
//! relocation between two such blocks (`call rel32` to another function,
//! `lea rip+rel32` to a string literal) only reaches 2 GiB, and Cranelift
//! panics when the blocks are further apart, which happens in a process with
//! a large address-space reservation. So nothing emitted here may depend on
//! the distance between blocks:
//!
//! * calls to other compiled functions are marked non-colocated ([`far_call`]):
//!   the callee's absolute address is loaded and called through a register;
//! * string literals are referenced as non-colocated symbols ([`far_data`]):
//!   the absolute address is loaded. The bytes stay a Cranelift data object,
//!   so they have exactly the lifetime of the code that points at them (both
//!   are kept for the life of the process, see `cranelift_jit`'s `Memory`);
//! * [`check`] refuses a function whose machine code contains any
//!   relocation other than an absolute 64-bit address, before the module is
//!   finalized. A future change that reintroduces a distance-limited
//!   relocation therefore becomes an ordinary compile error (the caller falls
//!   back to the VM), never a panic.

use anyhow::{Result, bail};
use cranelift::codegen::binemit::Reloc;
use cranelift::codegen::ir::{FuncRef, Function, GlobalValue, GlobalValueData};
use cranelift::prelude::codegen;
use cranelift_jit::JITModule;
use cranelift_module::{DataId, FuncId, Module};

// Tests only: emit the colocated declarations used before #96 on this thread,
// unchecked. `relocation_tests::jit_call_cost_measurement` times both variants
// in one process; `far_apart_tests` uses it in a child process to reproduce
// Cranelift's relocation panic and show that it is contained.
#[cfg(test)]
thread_local! {
    pub(super) static COLOCATED_FOR_TESTS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

// Tests only: the machine code of every function checked on this thread
// (before relocation, so it does not depend on addresses), for comparing two
// compilations. Taken and cleared with `take_emitted`.
#[cfg(test)]
thread_local! {
    static EMITTED: std::cell::RefCell<Vec<(String, Vec<u8>)>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
pub(crate) fn take_emitted() -> Vec<(String, Vec<u8>)> {
    EMITTED.with(|emitted| std::mem::take(&mut *emitted.borrow_mut()))
}

fn absolute() -> bool {
    #[cfg(test)]
    return !COLOCATED_FOR_TESTS.with(std::cell::Cell::get);
    #[cfg(not(test))]
    true
}

/// The relocation kinds in the machine code just defined from `ctx`, refusing
/// every kind whose correctness depends on where the allocator placed another
/// block. Only an absolute 64-bit address (`Abs8`) is distance-independent on
/// every architecture Cranelift's JIT supports.
pub(super) fn check(ctx: &codegen::Context, function: &str) -> Result<Vec<Reloc>> {
    let Some(code) = ctx.compiled_code() else {
        bail!("JIT: `{function}` has no machine code to check");
    };
    let kinds: Vec<Reloc> = code.buffer.relocs().iter().map(|reloc| reloc.kind).collect();
    #[cfg(test)]
    EMITTED.with(|emitted| {
        let mut emitted = emitted.borrow_mut();
        // Bounded: a thread that never takes the record keeps only a few.
        if emitted.len() < 4096 {
            emitted.push((function.to_string(), code.code_buffer().to_vec()));
        }
    });
    if let Some(kind) = kinds.iter().find(|kind| absolute() && **kind != Reloc::Abs8) {
        bail!(
            "JIT: `{function}` would need a `{kind}` relocation, which depends on the distance between \
             separately allocated blocks; not compiled (#96)"
        );
    }
    Ok(kinds)
}

/// Declare a call target in `func` that is reached through its absolute
/// address. `declare_func_in_func` marks module-local functions as colocated,
/// which on x86-64 becomes `call rel32` (#96, see the module docs).
pub(super) fn far_call(module: &mut JITModule, id: FuncId, func: &mut Function) -> FuncRef {
    let callee = module.declare_func_in_func(id, func);
    if absolute() {
        func.dfg.ext_funcs[callee].colocated = false;
    }
    callee
}

/// Declare a data object in `func` that is addressed absolutely. The default
/// for module-local data is colocated, i.e. `lea rip+rel32` on x86-64 (#96).
pub(super) fn far_data(module: &mut JITModule, id: DataId, func: &mut Function) -> GlobalValue {
    let value = module.declare_data_in_func(id, func);
    if let GlobalValueData::Symbol { colocated, .. } = &mut func.global_values[value] {
        *colocated = !absolute();
    }
    value
}
