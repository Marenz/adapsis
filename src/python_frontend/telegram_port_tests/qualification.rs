//! Test-only proof of the explicitly approved local-call qualification transform.
use crate::ast;
use std::collections::BTreeSet;

pub(super) fn verify(
    original: &ast::FunctionDecl,
    corrected: &ast::FunctionDecl,
    owner: &str,
    locals: &BTreeSet<String>,
) -> usize {
    let mut expected = original.clone();
    let mut count = 0;
    body(&mut expected.body, owner, locals, &mut count);
    assert!(
        super::super::ast_equality::function(&expected, corrected),
        "{}.{}: {}",
        owner,
        original.name,
        super::super::ast_equality::function_difference(&expected, corrected)
    );
    count
}

fn call(call: &mut ast::CallExpr, owner: &str, locals: &BTreeSet<String>, count: &mut usize) {
    if locals.contains(&call.callee) {
        call.callee = format!("{owner}.{}", call.callee);
        *count += 1;
    }
    for arg in &mut call.args {
        expression(arg, owner, locals, count);
    }
}

fn expression(expr: &mut ast::Expr, owner: &str, locals: &BTreeSet<String>, count: &mut usize) {
    use ast::Expr as E;
    match expr {
        E::Call(value) => call(value, owner, locals, count),
        E::Binary { left, right, .. } => {
            expression(left, owner, locals, count);
            expression(right, owner, locals, count);
        }
        E::Unary { expr, .. } => expression(expr, owner, locals, count),
        E::FieldAccess { base, .. } => expression(base, owner, locals, count),
        E::StructInit { fields, .. } => {
            for field in fields {
                expression(&mut field.value, owner, locals, count);
            }
        }
        E::Literal(_) | E::Identifier(_) => {}
    }
}

fn body(
    statements: &mut [ast::Statement],
    owner: &str,
    locals: &BTreeSet<String>,
    count: &mut usize,
) {
    use ast::StatementKind as S;
    for statement in statements {
        match &mut statement.kind {
            S::Let { value, .. }
            | S::Set { value, .. }
            | S::Return { value }
            | S::Yield { value } => expression(value, owner, locals, count),
            S::Call { call: value, .. }
            | S::Await { call: value, .. }
            | S::Spawn { call: value, .. } => call(value, owner, locals, count),
            S::Check { condition, .. } => expression(condition, owner, locals, count),
            S::Branch {
                condition,
                then_body,
                else_body,
            } => {
                expression(condition, owner, locals, count);
                body(then_body, owner, locals, count);
                body(else_body, owner, locals, count);
            }
            S::While {
                condition,
                body: nested,
            } => {
                expression(condition, owner, locals, count);
                body(nested, owner, locals, count);
            }
            S::Each {
                iterator,
                body: nested,
                ..
            } => {
                expression(iterator, owner, locals, count);
                body(nested, owner, locals, count);
            }
            S::Match { expr, arms } => {
                expression(expr, owner, locals, count);
                for arm in arms {
                    body(&mut arm.body, owner, locals, count);
                }
            }
            S::Event(event) => match event {
                ast::EventOp::Emit { value, .. } => expression(value, owner, locals, count),
                ast::EventOp::Register { .. } => {}
            },
            S::Source(source) => match source {
                ast::SourceOp::Add { source_type, .. }
                | ast::SourceOp::Replace { source_type, .. } => match source_type {
                    ast::SourceType::Timer(value) => expression(value, owner, locals, count),
                    ast::SourceType::Channel | ast::SourceType::Event(_, _) => {}
                },
                ast::SourceOp::Remove { .. } | ast::SourceOp::List => {}
            },
        }
    }
}
