//! JIT safety checkpoint for #97/#119: the current native ABI erases Result wrappers and
//! Check lowers to an uncatchable trap. Refuse reachable Check, [fail] and Result graphs
//! before emission OR a cache hit. This is admission, not a new failure implementation.
//! VM/tree Result inconsistencies remain for the later coordinated migration.
use crate::{
    ast,
    vm::await_admission::{expression_work, statement_work, Work},
};
use anyhow::{bail, Result};
use std::collections::HashSet;

const REFUSAL: &str =
    "jit: reachable Check, fallible or Result contract requires a value-capable execution tier";

/// The compiler emits all top-level functions; do not hide an unsafe sibling just because
/// the requested root is safe. Nested modules are reached by call resolution as needed.
pub(crate) fn check_program(program: &ast::Program) -> Result<()> {
    check_roots(program.functions.iter().map(AsRef::as_ref), program)
}

/// Re-evaluate the current immutable program, including before invoking cached code.
/// No cached admission result or caller-module-first name rewriting is permitted.
pub(crate) fn check_root(root: &ast::FunctionDecl, program: &ast::Program) -> Result<()> {
    check_roots(std::iter::once(root), program)
}

fn check_roots<'a>(
    roots: impl Iterator<Item = &'a ast::FunctionDecl>,
    program: &'a ast::Program,
) -> Result<()> {
    let mut visited = HashSet::new();
    let mut work: Vec<_> = roots.map(Work::Function).collect();
    while let Some(item) = work.pop() {
        match item {
            Work::Function(function) => {
                if !visited.insert(function as *const ast::FunctionDecl) {
                    continue;
                }
                if function.effects.contains(&ast::Effect::Fail)
                    || contains_result(&function.return_type, program)
                    || function
                        .params
                        .iter()
                        .any(|p| contains_result(&p.ty, program))
                {
                    bail!("{REFUSAL}");
                }
                // Separate safety rule, same walk: see `string_length_admission`.
                if super::string_length_admission::consumes_unknown_string_length(function, program) {
                    bail!("{}", super::string_length_admission::REFUSAL);
                }
                work.extend(function.body.iter().map(Work::Statement));
            }
            Work::Statement(statement) => {
                use ast::StatementKind as S;
                let ty = match &statement.kind {
                    S::Check { .. } => bail!("{REFUSAL}"),
                    S::Let { ty, .. } | S::Await { ty, .. } => Some(ty),
                    S::Call { binding, .. } | S::Spawn { binding, .. } => {
                        binding.as_ref().map(|b| &b.ty)
                    }
                    S::Each { binding, .. } => Some(&binding.ty),
                    _ => None,
                };
                if ty.is_some_and(|ty| contains_result(ty, program)) {
                    bail!("{REFUSAL}");
                }
                statement_work(&statement.kind, &mut work);
            }
            Work::Expression(expression) => expression_work(expression, &mut work),
            Work::Call(call, _) => {
                if matches!(call.callee.as_str(), "Ok" | "Err" | "Some") {
                    // Conservatively refuse constructor-name collisions too. The JIT
                    // must not erase a wrapper because a declaration shadows its name.
                    bail!("{REFUSAL}");
                }
                if let Some(function) = program.get_function(&call.callee) {
                    work.push(Work::Function(function));
                }
                work.extend(call.args.iter().map(Work::Expression));
            }
            Work::Source(source) => {
                if let ast::SourceType::Timer(expr) = source {
                    work.push(Work::Expression(expr));
                }
            }
        }
    }
    Ok(())
}

/// Includes nested containers and named struct/union fields. Ambiguous names inspect all
/// candidates conservatively; this does not change runtime type lookup. Both AST and type
/// recursion are iterative and visited only within this snapshot.
pub(crate) fn contains_result<'a>(ty: &'a ast::Type, program: &'a ast::Program) -> bool {
    use ast::Type as T;
    let mut work = vec![ty];
    let mut visited = HashSet::new();
    while let Some(ty) = work.pop() {
        match ty {
            T::Result(_) => return true,
            T::List(inner) | T::Set(inner) | T::Option(inner) => work.push(inner),
            T::Map(key, value) => work.extend([key.as_ref(), value.as_ref()]),
            T::Struct(name) | T::TaggedUnion(name) => {
                if !visited.insert(name) {
                    continue;
                }
                let mut modules: Vec<_> = program.modules.iter().collect();
                let mut types: Vec<_> = program.types.iter().collect();
                while let Some(module) = modules.pop() {
                    modules.extend(&module.modules);
                    types.extend(&module.types);
                }
                for declaration in types.into_iter().filter(|t| t.name() == name) {
                    match declaration {
                        ast::TypeDecl::Struct(s) => work.extend(s.fields.iter().map(|f| &f.ty)),
                        ast::TypeDecl::TaggedUnion(u) => {
                            work.extend(u.variants.iter().flat_map(|v| &v.payload))
                        }
                    }
                }
            }
            _ => {}
        }
    }
    false
}

#[cfg(test)]
mod tests;
