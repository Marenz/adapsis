//! Temporary B-safety quarantine. Full Result normalization/catching is not activated:
//! graphs with Result signatures, bindings or constructors must run on the existing tree
//! path until the coordinated B/C migration. Direct builtin IO needs a coroutine handle
//! unavailable to CallBuiltin, so it also selects tree BEFORE any earlier await executes.
//! This checks authored contracts/captures, not the shape of every dynamic builtin value:
//! builtin IO returning an Err value keeps that existing value behavior. No Result
//! normalization, implicit unwrapping or catch mode is introduced in VM execution here.
use super::await_admission::{expression_work, statement_work, Work};
use crate::{ast, compiler::failure_admission::contains_result};
use std::collections::HashSet;

pub(super) const RESULT_TREE: &str =
    "vm: Result contract or capture requires tree before execution until invocation migration";
pub(super) const DIRECT_IO_TREE: &str = "vm: non-await builtin IO requires tree before execution";

pub(super) fn check<'a>(
    root: &'a ast::FunctionDecl,
    program: &'a ast::Program,
) -> anyhow::Result<()> {
    let mut visited = HashSet::new();
    let mut work = vec![Work::Function(root)];
    while let Some(item) = work.pop() {
        match item {
            Work::Function(function) => {
                if !visited.insert(function as *const ast::FunctionDecl) {
                    continue;
                }
                if contains_result(&function.return_type, program)
                    || function
                        .params
                        .iter()
                        .any(|p| contains_result(&p.ty, program))
                {
                    anyhow::bail!(RESULT_TREE);
                }
                work.extend(function.body.iter().map(Work::Statement));
            }
            Work::Statement(statement) => {
                use ast::StatementKind as S;
                let ty = match &statement.kind {
                    S::Let { ty, .. } | S::Await { ty, .. } => Some(ty),
                    S::Call { binding, .. } | S::Spawn { binding, .. } => {
                        binding.as_ref().map(|b| &b.ty)
                    }
                    S::Each { binding, .. } => Some(&binding.ty),
                    _ => None,
                };
                if ty.is_some_and(|ty| contains_result(ty, program)) {
                    anyhow::bail!(RESULT_TREE);
                }
                statement_work(&statement.kind, &mut work);
            }
            Work::Expression(expression) => expression_work(expression, &mut work),
            Work::Call(call, awaiting) => {
                if matches!(call.callee.as_str(), "Ok" | "Err" | "Some") {
                    anyhow::bail!(RESULT_TREE);
                }
                if let Some(callee) = program.get_function(&call.callee) {
                    work.push(Work::Function(callee));
                } else if !awaiting && crate::builtins::is_io_builtin(&call.callee) {
                    anyhow::bail!(DIRECT_IO_TREE);
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
