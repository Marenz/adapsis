//! Trace is an observer of the real root statement executor, not a second interpreter.
//! It executes branches/loops/matches, captures only the root's statement outcomes, and
//! appends the normalized invocation outcome. Nested declared callees do not inherit the
//! observer. The public trace entry does not gain a coroutine handle or IO authority.
use super::{
    invocation, statement_execution::StatementObserver, Env, TraceStatus, TraceStep, Value,
};
use crate::{ast, parser};
use anyhow::Result;

pub fn trace_function(
    program: &ast::Program,
    name: &str,
    input: &parser::Expr,
) -> Result<Vec<TraceStep>> {
    let function = program.get_function(name).ok_or_else(|| {
        anyhow::anyhow!(
            "function `{name}` not found{}",
            super::suggest_similar(program, name)
        )
    })?;
    crate::intern::set_display_interner(&program.shared_interner);
    super::set_shared_program(Some(std::sync::Arc::new(program.clone())));
    let input = super::eval_parser_expr_standalone(input)?;
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    env.populate_shared_from_program(program);
    super::bind_input_to_params(program, function, &input, &mut env);
    let (_, steps) = observe_invocation(program, name, function, &mut env);
    Ok(steps)
}

/// Structured outcome remains available internally for tests/adapters. The public trace
/// API renders observation rows, not a recoverable in-language Err on operational failure.
pub(super) fn observe_invocation(
    program: &ast::Program,
    name: &str,
    function: &ast::FunctionDecl,
    env: &mut Env,
) -> (Result<Value>, Vec<TraceStep>) {
    let mut recorder = Recorder { steps: Vec::new() };
    let outcome = invocation::invoke(
        program,
        name,
        function,
        invocation::Body::ObservedDeclared(env, &mut recorder),
        None,
    );
    let (result, status) = match &outcome {
        Ok(value) => (format!("{value}"), TraceStatus::Return),
        Err(error) => (format!("ERROR: {error}"), TraceStatus::Fail),
    };
    recorder.steps.push(TraceStep {
        stmt_id: function.id.clone(),
        description: format!("invocation {name}"),
        result,
        status,
    });
    (outcome, recorder.steps)
}

struct Recorder {
    steps: Vec<TraceStep>,
}

impl StatementObserver for Recorder {
    fn completed(
        &mut self,
        statement: &ast::Statement,
        outcome: &Result<Value>,
        binding: Option<&Value>,
    ) {
        let (result, status) = match outcome {
            Err(error) => (format!("ERROR: {error}"), TraceStatus::Fail),
            Ok(value)
                if !matches!(value, Value::None)
                    || matches!(statement.kind, ast::StatementKind::Return { .. }) =>
            {
                (format!("{value}"), TraceStatus::Return)
            }
            Ok(_) => (
                binding.map_or_else(|| "completed".into(), |v| format!("{v}")),
                TraceStatus::Pass,
            ),
        };
        self.steps.push(TraceStep {
            stmt_id: statement.id.clone(),
            description: description(&statement.kind),
            result,
            status,
        });
    }
}

fn description(statement: &ast::StatementKind) -> String {
    use ast::StatementKind as S;
    match statement {
        S::Let { name, ty, .. } => format!("let {name}:{ty:?}"),
        S::Call { call, binding } => match binding {
            Some(binding) => format!("call {}:{:?} = {}()", binding.name, binding.ty, call.callee),
            None => format!("call {}()", call.callee),
        },
        S::Check { label, .. } => format!("check {label}"),
        S::Return { .. } => "return".into(),
        S::Set { name, .. } => format!("set {name}"),
        S::Await { name, .. } => format!("await {name}"),
        S::Branch { .. } => "branch".into(),
        S::Each { .. } => "each".into(),
        S::While { .. } => "while".into(),
        S::Match { .. } => "match".into(),
        S::Spawn { .. } => "spawn".into(),
        S::Yield { .. } => "yield".into(),
        S::Source(_) => "source".into(),
        S::Event(_) => "event".into(),
    }
}
