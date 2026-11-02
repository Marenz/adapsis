//! JIT admission and revision-safe caching (#97/#119). This does NOT alter VM fallback
//! semantics: the VM execution/admission distinction belongs to the next safety slice.
use super::{input_to_i64_args, JitCache};
use crate::{ast, compiler, parser};

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
    if let Some(cache) = cache {
        let mut slot = cache.lock().ok()?;
        if !matches!(&*slot, Some((cached_revision, _)) if *cached_revision == revision) {
            // Drop the stale module BEFORE attempting a replacement. A failed compile
            // must never leave old code available to the call below (#119).
            *slot = None;
            *slot = Some((revision, compiler::compile(program).ok()?));
        }
        match slot.as_mut() {
            Some((cached_revision, compiled)) if *cached_revision == revision => {
                invoke(compiled, function, name, &args)
            }
            _ => None,
        }
    } else {
        invoke(&mut compiler::compile(program).ok()?, function, name, &args)
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
