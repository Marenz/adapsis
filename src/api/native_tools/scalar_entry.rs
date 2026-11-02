//! Narrow #83 proof: no callable/helper semantics in this entry or submitted data.
//! Recomputed from the current snapshot; never a blanket builtin-collision waiver.
use crate::{ast, parser};
use std::collections::HashSet;

pub(super) enum EntryUse<'a> {
    Inspect,
    Execute(&'a [parser::Operation]),
}

impl EntryUse<'_> {
    pub(super) fn scalar_data(&self) -> bool {
        match self {
            Self::Inspect => true,
            Self::Execute(ops) => submission(ops),
        }
    }

    pub(super) fn safe_initialization(&self, program: &ast::Program) -> bool {
        if matches!(self, Self::Inspect) {
            return true; // Printing executes no shared defaults.
        }
        let bound = HashSet::new();
        let mut modules: Vec<_> = program.modules.iter().collect();
        while let Some(module) = modules.pop() {
            if module
                .shared_vars
                .iter()
                .any(|var| !scalar_type(&var.ty) || !expression(&var.default, &bound))
            {
                return false;
            }
            modules.extend(&module.modules);
        }
        true
    }
}

pub(super) fn proven(program: &ast::Program, target: &str) -> bool {
    let Some(function) = program.get_function(target) else {
        return false;
    };
    if !function.effects.is_empty() || !scalar_type(&function.return_type) {
        return false;
    }
    let mut bound = HashSet::new();
    for parameter in &function.params {
        if !scalar_type(&parameter.ty) || !bound.insert(parameter.name.as_str()) {
            return false;
        }
    }
    let Some((last, prefix)) = function.body.split_last() else {
        return false;
    };
    for statement in prefix {
        let ast::StatementKind::Let { name, ty, value } = &statement.kind else {
            return false;
        };
        if !scalar_type(ty) || !expression(value, &bound) || !bound.insert(name.as_str()) {
            return false;
        }
    }
    matches!(&last.kind, ast::StatementKind::Return { value } if expression(value, &bound))
}

fn scalar_type(ty: &ast::Type) -> bool {
    matches!(
        ty,
        ast::Type::Int | ast::Type::Float | ast::Type::Bool | ast::Type::String | ast::Type::Byte
    )
}

fn expression(root: &ast::Expr, bound: &HashSet<&str>) -> bool {
    let mut pending = vec![root];
    while let Some(expr) = pending.pop() {
        match expr {
            ast::Expr::Literal(_) => {}
            ast::Expr::Identifier(name) if bound.contains(name.as_str()) => {}
            ast::Expr::Unary { expr, .. } => pending.push(expr),
            ast::Expr::Binary { left, right, .. } => {
                pending.extend([left.as_ref(), right.as_ref()])
            }
            _ => return false,
        }
    }
    true
}

fn literal(expr: &parser::Expr) -> bool {
    matches!(
        expr,
        parser::Expr::Int(_)
            | parser::Expr::Float(_)
            | parser::Expr::Bool(_)
            | parser::Expr::String(_)
    ) || matches!(expr, parser::Expr::Unary { op: parser::UnaryOp::Neg, expr }
            if matches!(expr.as_ref(), parser::Expr::Int(_) | parser::Expr::Float(_)))
}

fn input(expr: &parser::Expr) -> bool {
    // A top-level record is the parser's positional/keyword argument carrier,
    // not permission for nested record values or constructors.
    literal(expr)
        || matches!(expr, parser::Expr::StructLiteral(fields)
        if fields.iter().all(|field| literal(&field.value)))
}

pub(super) fn submission(ops: &[parser::Operation]) -> bool {
    !ops.is_empty()
        && ops.iter().all(|op| match op {
            parser::Operation::Eval(eval) => input(&eval.input),
            parser::Operation::Test(test) => {
                !test.cases.is_empty()
                    && test.cases.iter().all(|case| {
                        input(&case.input)
                            && literal(&case.expected)
                            && case.matcher.is_none()
                            && case.after_checks.is_empty()
                    })
            }
            _ => false,
        })
}
