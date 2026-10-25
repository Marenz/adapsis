//! Whole-source structural admission without runtime type/signature lookup.
//! Type-dependent semantics are checked by strict final lowering at dispatch.
use super::{Context, Diagnostic, expressions};
use rustpython_parser::ast as py;
use std::collections::HashMap;

pub(super) fn statements(
    cx: &Context<'_>,
    body: &[py::Stmt],
    doc_allowed: bool,
    refs: &mut Vec<String>,
) -> Result<(), Diagnostic> {
    for (index, stmt) in body.iter().enumerate() {
        if doc_allowed && index == 0 && super::docstring(stmt).is_some() {
            continue;
        }
        statement(cx, stmt, refs)?;
    }
    Ok(())
}

fn statement(cx: &Context<'_>, stmt: &py::Stmt, refs: &mut Vec<String>) -> Result<(), Diagnostic> {
    match stmt {
        py::Stmt::AnnAssign(value) => {
            if !matches!(value.target.as_ref(), py::Expr::Name(_)) {
                return Err(cx.at(stmt, "annotated bindings require a simple name"));
            }
            expressions::annotation(cx, &value.annotation)?;
            let expr = value
                .value
                .as_ref()
                .ok_or_else(|| cx.at(stmt, "annotated binding requires a value"))?;
            if let py::Expr::Await(awaited) = expr.as_ref() {
                if !matches!(awaited.value.as_ref(), py::Expr::Call(_)) {
                    return Err(cx.at(expr.as_ref(), "await requires a direct call"));
                }
                expression(cx, &awaited.value, refs)
            } else {
                expression(cx, expr, refs)
            }
        }
        py::Stmt::Assign(value) => {
            if value.targets.len() != 1
                || !matches!(&value.targets[0], py::Expr::Name(_))
                || value.type_comment.is_some()
            {
                return Err(cx.at(stmt, "assignment requires one simple name; unpacking and type comments are unsupported"));
            }
            expression(cx, &value.value, refs)
        }
        py::Stmt::Return(value) => expression(
            cx,
            value
                .value
                .as_deref()
                .ok_or_else(|| cx.at(stmt, "bare return/None is unsupported"))?,
            refs,
        ),
        py::Stmt::If(branch) => {
            expression(cx, &branch.test, refs)?;
            statements(cx, &branch.body, false, refs)?;
            statements(cx, &branch.orelse, false, refs)
        }
        py::Stmt::While(loop_) => {
            if !loop_.orelse.is_empty() {
                return Err(cx.at(stmt, "loop else is unsupported"));
            }
            expression(cx, &loop_.test, refs)?;
            statements(cx, &loop_.body, false, refs)
        }
        py::Stmt::For(loop_) => {
            if !loop_.orelse.is_empty()
                || loop_.type_comment.is_some()
                || !matches!(loop_.target.as_ref(), py::Expr::Name(_))
            {
                return Err(cx.at(
                    stmt,
                    "for requires one simple target and no else/type comment",
                ));
            }
            expression(cx, &loop_.iter, refs)?;
            statements(cx, &loop_.body, false, refs)
        }
        _ => Err(cx.at(stmt, "unsupported Python statement in the typed frontend")),
    }
}

fn expression(cx: &Context<'_>, expr: &py::Expr, refs: &mut Vec<String>) -> Result<(), Diagnostic> {
    match expr {
        py::Expr::Constant(_) => {
            expressions::literal(cx, expr)?;
            Ok(())
        }
        py::Expr::Name(name)
            if super::identifier(name.id.as_str()) && !super::forbidden(name.id.as_str()) =>
        {
            Ok(())
        }
        py::Expr::List(list) => list
            .elts
            .iter()
            .try_for_each(|item| expression(cx, item, refs)),
        py::Expr::Call(call) => {
            if !call.keywords.is_empty() {
                return Err(cx.at(expr, "keyword arguments in function bodies are unsupported"));
            }
            let name = expressions::direct_name(cx, &call.func, &HashMap::new())?;
            expressions::reject_unsafe_builtin(cx, call, &name)?;
            if name.contains('.') {
                refs.push(name);
            } else if !crate::builtins::is_builtin(&name) {
                refs.push(format!("{}.{}", cx.module, name));
            }
            call.args
                .iter()
                .try_for_each(|arg| expression(cx, arg, refs))
        }
        py::Expr::BinOp(binary) => {
            if !matches!(
                binary.op,
                py::Operator::Add | py::Operator::Sub | py::Operator::Mult
            ) {
                return Err(cx.at(
                    expr,
                    "unsupported arithmetic operator; only +, -, * are supported",
                ));
            }
            expression(cx, &binary.left, refs)?;
            expression(cx, &binary.right, refs)
        }
        py::Expr::UnaryOp(unary) if matches!(unary.op, py::UnaryOp::Not | py::UnaryOp::USub) => {
            expression(cx, &unary.operand, refs)
        }
        py::Expr::BoolOp(boolean) => boolean
            .values
            .iter()
            .try_for_each(|value| expression(cx, value, refs)),
        py::Expr::Compare(compare) => {
            if compare.ops.len() != 1
                || !matches!(
                    compare.ops[0],
                    py::CmpOp::Eq
                        | py::CmpOp::NotEq
                        | py::CmpOp::Lt
                        | py::CmpOp::LtE
                        | py::CmpOp::Gt
                        | py::CmpOp::GtE
                )
            {
                return Err(cx.at(expr, "only single value comparisons are supported"));
            }
            expression(cx, &compare.left, refs)?;
            compare
                .comparators
                .iter()
                .try_for_each(|value| expression(cx, value, refs))
        }
        _ => Err(cx.at(expr, "unsupported Python expression in the typed frontend")),
    }
}
