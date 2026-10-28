//! User Await needs the tree's function resolution, stubs and unwrap semantics.
//! Reject its entire reachable graph before root bytecode can perform any IO.
//! No continuation is restarted and no runtime replay flag is reset.
use crate::ast;
use std::collections::HashSet;

const REQUIRES_TREE: &str =
    "vm: reachable user-function await requires tree evaluation before execution";

pub(super) fn check(function: &ast::FunctionDecl, program: &ast::Program) -> anyhow::Result<()> {
    if analyze(function, program).requires_tree {
        anyhow::bail!(REQUIRES_TREE);
    }
    Ok(())
}

#[derive(Default)]
struct Analysis {
    requires_tree: bool,
    #[cfg(test)]
    functions: usize,
    #[cfg(test)]
    nodes: usize,
}

enum Work<'a> {
    Function(&'a ast::FunctionDecl),
    Statement(&'a ast::Statement),
    Expression(&'a ast::Expr),
    Call(&'a ast::CallExpr, bool),
    Source(&'a ast::SourceType),
}

fn analyze<'a>(root: &'a ast::FunctionDecl, program: &'a ast::Program) -> Analysis {
    let mut result = Analysis::default();
    // Pointer identity is used only during this immutable snapshot traversal.
    // Never dereferenced, persisted, or reused across Program revisions.
    let mut visited: HashSet<*const ast::FunctionDecl> = HashSet::new();
    let mut work = vec![Work::Function(root)];
    while let Some(item) = work.pop() {
        #[cfg(test)]
        {
            result.nodes += 1;
        }
        match item {
            Work::Function(function) => {
                if visited.insert(function as *const _) {
                    #[cfg(test)]
                    {
                        result.functions += 1;
                    }
                    work.extend(function.body.iter().map(Work::Statement));
                }
            }
            Work::Statement(statement) => statement_work(&statement.kind, &mut work),
            Work::Expression(expression) => expression_work(expression, &mut work),
            Work::Call(call, awaiting) => {
                // Exactly the original spelling and tree's get_function lookup;
                // no caller-module-first normalization. Conservative graph edges
                // may over-refuse a builtin/variant collision, never redirect it.
                if let Some(function) = program.get_function(&call.callee) {
                    if awaiting {
                        result.requires_tree = true;
                        return result;
                    }
                    work.push(Work::Function(function));
                }
                work.extend(call.args.iter().map(Work::Expression));
            }
            Work::Source(source) => match source {
                ast::SourceType::Timer(expression) => work.push(Work::Expression(expression)),
                ast::SourceType::Channel | ast::SourceType::Event(_, _) => {}
            },
        }
    }
    result
}

fn statement_work<'a>(statement: &'a ast::StatementKind, work: &mut Vec<Work<'a>>) {
    use ast::StatementKind as S;
    match statement {
        S::Let { value, .. } | S::Set { value, .. } | S::Return { value } | S::Yield { value } => {
            work.push(Work::Expression(value))
        }
        S::Call { call, .. } | S::Spawn { call, .. } => work.push(Work::Call(call, false)),
        S::Await { call, .. } => work.push(Work::Call(call, true)),
        S::Check { condition, .. } => work.push(Work::Expression(condition)),
        S::Branch {
            condition,
            then_body,
            else_body,
        } => {
            work.push(Work::Expression(condition));
            work.extend(then_body.iter().chain(else_body).map(Work::Statement));
        }
        S::Each { iterator, body, .. } => {
            work.push(Work::Expression(iterator));
            work.extend(body.iter().map(Work::Statement));
        }
        S::While { condition, body } => {
            work.push(Work::Expression(condition));
            work.extend(body.iter().map(Work::Statement));
        }
        S::Match { expr, arms } => {
            work.push(Work::Expression(expr));
            work.extend(arms.iter().flat_map(|arm| &arm.body).map(Work::Statement));
        }
        S::Event(event) => match event {
            ast::EventOp::Emit { value, .. } => work.push(Work::Expression(value)),
            ast::EventOp::Register { .. } => {}
        },
        S::Source(source) => match source {
            ast::SourceOp::Add { source_type, .. } | ast::SourceOp::Replace { source_type, .. } => {
                work.push(Work::Source(source_type))
            }
            // Handler names are registration metadata, not expressions executed
            // by this statement. Their future invocation has its own admission.
            ast::SourceOp::Remove { .. } | ast::SourceOp::List => {}
        },
    }
}

fn expression_work<'a>(expression: &'a ast::Expr, work: &mut Vec<Work<'a>>) {
    use ast::Expr as E;
    match expression {
        E::Call(call) => work.push(Work::Call(call, false)),
        E::Binary { left, right, .. } => {
            work.push(Work::Expression(left));
            work.push(Work::Expression(right));
        }
        E::Unary { expr, .. } => work.push(Work::Expression(expr)),
        E::FieldAccess { base, .. } => work.push(Work::Expression(base)),
        E::StructInit { fields, .. } => {
            work.extend(fields.iter().map(|field| Work::Expression(&field.value)))
        }
        E::Literal(_) | E::Identifier(_) => {}
    }
}

#[cfg(test)]
mod tests;
