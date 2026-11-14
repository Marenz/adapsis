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

/// The tree-walker environment of an async entry point: shared slots materialized from
/// the Program and the coroutine handle that services every `+await`. One construction
/// for named-function evaluation and ad-hoc blocks, so both reach IO the same way.
pub(super) fn async_tree_env(
    program: &ast::Program,
    handle: crate::coroutine::CoroutineHandle,
) -> Env {
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    env.populate_shared_from_program(program);
    env.set("__coroutine_handle", Value::CoroutineHandle(handle));
    env
}

/// Execute the statements of an ad-hoc block (#129) that is NOT declared in `program`.
///
/// This is the raw named-block entry lifecycle blocks use: a module frame for `+shared`
/// resolution and no declared-return contract, so the value of `return` comes back as
/// written and a language failure (failed check, propagated Err) stays an error. It always
/// runs on the tree-walker: the only tier that executes statements without a Program
/// entry, and the one whose callees keep `Env`-based shared access.
///
/// The block runs under `context`: its resource limits and, for restricted speakers, the
/// call policy of the block's own frame (`eval/meter.rs`). The meter exists exactly for
/// the duration of this call.
pub fn eval_detached_block(
    program: &ast::Program,
    frame: &str,
    body: &[ast::Statement],
    handle: crate::coroutine::CoroutineHandle,
    context: super::meter::RunContext,
) -> Result<Value> {
    let _meter = super::meter::install(context);
    let mut env = async_tree_env(program, handle);
    let value = super::eval_function_body_named(program, frame, body, &mut env)?;
    // The result is rendered as text by the caller; bound it like every other value.
    super::meter::admit_value(&value)?;
    Ok(value)
}

pub(super) fn invoke(
    program: &ast::Program,
    name: &str,
    function: &ast::FunctionDecl,
    body: Body<'_>,
    observer: Option<&mut dyn OutcomeObserver>,
) -> Result<Value> {
    // Inside a native `run` block, mark this as a callee frame: the block's call policy
    // applies to the block's own statements only. Popped on every exit path.
    let _frame = if super::meter::active() { super::meter::enter_callee() } else { None };
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
    // A stack overflow aborts the process; refuse the call while there is still room.
    super::stack::check(callee)?;
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
