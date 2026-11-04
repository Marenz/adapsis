//! One actual-declaration boundary for interpreter invocations (#82/#97).
//!
//! Declared bodies and selected stubs normalize once here, never in block execution or
//! display adapters. A Result declaration catches typed language failure, not operational
//! errors. Result graphs remain quarantined from VM/JIT while this interpreter contract
//! is established; user awaits preserve their declared Result values.
//! Raw lifecycle/block execution intentionally remains separate and cannot acquire a
//! Result contract by accidentally matching a synthetic label to a declared function.
use super::{Env, Value};
use crate::ast;
use anyhow::Result;

pub(super) enum Body<'a> {
    Declared(&'a mut Env),
    ObservedDeclared(
        &'a mut Env,
        &'a mut dyn super::statement_execution::StatementObserver,
    ),
    /// CLI run-async historically entered without an FN_NAME_STACK frame. Preserve that
    /// setup explicitly in plumbing; do not silently bundle a shared-scope behavior fix.
    LegacyUnframedDeclared(&'a mut Env),
    Stub(&'a str),
}

/// Trusted internal observer only. It sees the actual declaration and structured outcome,
/// not a lookup by label or a parsed Display string. Observers never inherit into nested
/// declarations, and receive no environment or additional coroutine authority.
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
        Body::ObservedDeclared(env, observer) => with_named_frame(program, name, || {
            super::statement_execution::run(program, &function.body, env, Some(observer))
        }),
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
    let outcome = super::result_contract::complete_invocation(name, &function.return_type, outcome);
    if let Some(observer) = observer {
        observer.observe(function, &outcome);
    }
    outcome
}

pub(super) fn with_named_frame(
    program: &ast::Program,
    name: &str,
    execute: impl FnOnce() -> Result<Value>,
) -> Result<Value> {
    super::FN_NAME_STACK.with(|stack| stack.borrow_mut().push(program.qualify_function_name(name)));
    let result = execute();
    super::FN_NAME_STACK.with(|stack| stack.borrow_mut().pop());
    result
}

pub(super) fn with_call_depth(
    callee: &str,
    execute: impl FnOnce() -> Result<Value>,
) -> Result<Value> {
    let depth = super::CALL_DEPTH.with(|d| {
        let value = d.get();
        d.set(value + 1);
        value
    });
    if depth >= super::MAX_CALL_DEPTH {
        super::CALL_DEPTH.with(|d| d.set(0));
        anyhow::bail!(
            "maximum call depth ({}) exceeded — possible infinite recursion in `{callee}`",
            super::MAX_CALL_DEPTH
        );
    }
    let result = execute();
    super::CALL_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    result
}

/// Evaluate arguments before entering the target's explicit capture/propagate scope.
/// Thus a failing argument cannot be caught as if the target function had failed.
pub(super) fn eval_bound_call(
    program: &ast::Program,
    call: &ast::CallExpr,
    binding: Option<&ast::Binding>,
    env: &mut Env,
) -> Result<Value> {
    with_call_depth(&call.callee, || {
        let args = call
            .args
            .iter()
            .map(|arg| super::eval_ast_expr(program, arg, env))
            .collect::<Result<Vec<_>>>()?;
        let outcome = super::eval_builtin_or_user(program, &call.callee, args, env);
        match binding.map(|binding| &binding.ty) {
            Some(ast::Type::Result(payload)) => {
                super::result_contract::capture_call(&call.callee, payload, outcome)
            }
            _ => super::result_contract::propagate_call(outcome),
        }
    })
}

#[cfg(test)]
mod activation_tests;
#[cfg(test)]
mod tests;
