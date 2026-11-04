//! One statement executor for ordinary and traced interpretation. Observation is scoped
//! to this body's root statements: nested blocks and declarations execute normally, but
//! do not inherit the observer. Expressions and IO execute once, never to obtain a log.
use super::{Env, Value};
use crate::ast;
use anyhow::Result;

pub(super) trait StatementObserver {
    fn completed(
        &mut self,
        statement: &ast::Statement,
        outcome: &Result<Value>,
        binding: Option<&Value>,
    );
}

pub(crate) fn eval_function_body(
    program: &ast::Program,
    body: &[ast::Statement],
    env: &mut Env,
) -> Result<Value> {
    run(program, body, env, None)
}

pub(super) fn run(
    program: &ast::Program,
    body: &[ast::Statement],
    env: &mut Env,
    mut observer: Option<&mut dyn StatementObserver>,
) -> Result<Value> {
    for statement in body {
        let outcome = super::execute_statement(program, statement, env);
        if let Some(observer) = observer.as_deref_mut() {
            use ast::StatementKind as S;
            let name = match &statement.kind {
                S::Let { name, .. } | S::Set { name, .. } | S::Await { name, .. } => Some(name),
                S::Call { binding, .. } | S::Spawn { binding, .. } => {
                    binding.as_ref().map(|b| &b.name)
                }
                _ => None,
            };
            let binding = if outcome.is_ok() {
                name.and_then(|name| env.get(name).ok())
            } else {
                None
            };
            observer.completed(statement, &outcome, binding.as_ref());
        }
        let value = outcome?;
        if matches!(statement.kind, ast::StatementKind::Return { .. })
            || !matches!(value, Value::None)
        {
            return Ok(value);
        }
    }
    Ok(Value::None)
}
