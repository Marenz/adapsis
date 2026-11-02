//! C plumbing checkpoint: one actual-declaration boundary for interpreter invocations.
//!
//! No Result policy is activated here yet. Bodies and stubs return their existing raw
//! outcomes; named eval/test/+call adapters retain their old wrapping/catching behavior.
//! C activation will normalize once HERE, after every genuine caller uses this boundary.
//! Raw lifecycle/block execution intentionally remains separate and cannot acquire a
//! Result contract by accidentally matching a synthetic label to a declared function.
use super::{Env, Value};
use crate::ast;
use anyhow::Result;

pub(super) enum Body<'a> {
    Declared(&'a mut Env),
    /// CLI run-async historically entered without an FN_NAME_STACK frame. Preserve that
    /// setup explicitly in plumbing; do not silently bundle a shared-scope behavior fix.
    LegacyUnframedDeclared(&'a mut Env),
    Stub(&'a str),
}

/// Trusted internal observer only. It sees the actual declaration and structured outcome,
/// not a lookup by label or a parsed Display string. No observer is installed by the
/// production callers in this plumbing slice; legacy trace remains explicitly partial.
pub(super) trait OutcomeObserver {
    fn observe(&mut self, function: &ast::FunctionDecl, outcome: &Result<Value>);
}

pub fn eval_declared_function(
    program: &ast::Program,
    name: &str,
    function: &ast::FunctionDecl,
    env: &mut Env,
) -> Result<Value> {
    invoke(program, name, function, Body::Declared(env), None)
}

pub fn eval_declared_function_unframed(
    program: &ast::Program,
    name: &str,
    function: &ast::FunctionDecl,
    env: &mut Env,
) -> Result<Value> {
    invoke(
        program,
        name,
        function,
        Body::LegacyUnframedDeclared(env),
        None,
    )
}

pub(super) fn invoke(
    program: &ast::Program,
    name: &str,
    function: &ast::FunctionDecl,
    body: Body<'_>,
    observer: Option<&mut dyn OutcomeObserver>,
) -> Result<Value> {
    let outcome = match body {
        Body::Declared(env) => super::eval_function_body_named(program, name, &function.body, env),
        Body::LegacyUnframedDeclared(env) => {
            super::eval_function_body(program, &function.body, env)
        }
        Body::Stub(source) => {
            // Preserve existing stub evaluation: selection and permission/identity lookup
            // happen at the caller, and the expression uses the old program-only context
            // (no newly inherited env/handle or invented invocation frame).
            let expression = crate::parser::parse_single_expr(source)
                .map_err(|e| anyhow::anyhow!("stub expression parse error: {e}"));
            expression
                .and_then(|expression| super::eval_parser_expr_with_program(&expression, program))
        }
    };
    // Intentionally raw until the coordinated C activation, including for Result<T>.
    if let Some(observer) = observer {
        observer.observe(function, &outcome);
    }
    outcome
}

#[cfg(test)]
mod tests;
