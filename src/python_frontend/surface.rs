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
    type_refs: &mut Vec<String>,
) -> Result<(), Diagnostic> {
    for (index, stmt) in body.iter().enumerate() {
        if doc_allowed && index == 0 && super::docstring(stmt).is_some() {
            continue;
        }
        statement(cx, stmt, refs, type_refs)?;
    }
    Ok(())
}

fn statement(
    cx: &Context<'_>,
    stmt: &py::Stmt,
    refs: &mut Vec<String>,
    type_refs: &mut Vec<String>,
) -> Result<(), Diagnostic> {
    match stmt {
        py::Stmt::AnnAssign(value) => {
            if !matches!(value.target.as_ref(), py::Expr::Name(_)) {
                return Err(cx.at(stmt, "annotated bindings require a simple name"));
            }
            let ty = expressions::annotation(cx, &value.annotation, &Default::default())?;
            expressions::collect_ty_names(&ty, type_refs);
            let expr = value
                .value
                .as_ref()
                .ok_or_else(|| cx.at(stmt, "annotated binding requires a value"))?;
            if let py::Expr::Await(awaited) = expr.as_ref() {
                if !matches!(awaited.value.as_ref(), py::Expr::Call(_)) {
                    return Err(cx.at(expr.as_ref(), "await requires a direct call"));
                }
                expression(cx, &awaited.value, refs, type_refs)
            } else {
                expression(cx, expr, refs, type_refs)
            }
        }
        py::Stmt::Assign(value) => {
            if value.targets.len() != 1
                || !matches!(&value.targets[0], py::Expr::Name(_))
                || value.type_comment.is_some()
            {
                return Err(cx.at(stmt, "assignment requires one simple name; unpacking and type comments are unsupported"));
            }
            expression(cx, &value.value, refs, type_refs)
        }
        py::Stmt::Return(value) => expression(
            cx,
            value
                .value
                .as_deref()
                .ok_or_else(|| cx.at(stmt, "bare return/None is unsupported"))?,
            refs,
            type_refs,
        ),
        py::Stmt::If(branch) => {
            expression(cx, &branch.test, refs, type_refs)?;
            statements(cx, &branch.body, false, refs, type_refs)?;
            statements(cx, &branch.orelse, false, refs, type_refs)
        }
        py::Stmt::While(loop_) => {
            if !loop_.orelse.is_empty() {
                return Err(cx.at(stmt, "loop else is unsupported"));
            }
            expression(cx, &loop_.test, refs, type_refs)?;
            statements(cx, &loop_.body, false, refs, type_refs)
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
            expression(cx, &loop_.iter, refs, type_refs)?;
            statements(cx, &loop_.body, false, refs, type_refs)
        }
        py::Stmt::Match(match_) => {
            expression(cx, &match_.subject, refs, type_refs)?;
            for case in &match_.cases {
                if case.guard.is_some() {
                    return Err(cx.at(stmt, "match guards (case ... if ...) are unsupported"));
                }
                arm_pattern(cx, &case.pattern, type_refs)?;
                statements(cx, &case.body, false, refs, type_refs)?;
            }
            Ok(())
        }
        py::Stmt::Expr(value) => source_event_call(cx, value, refs, type_refs),
        py::Stmt::Pass(_) => Ok(()),
        _ => Err(cx.at(stmt, "unsupported Python statement in the typed frontend")),
    }
}

/// A bare expression statement must call a reserved source/event helper with
/// literal string arguments (and a value expression for `event_emit`).
fn source_event_call(
    cx: &Context<'_>,
    value: &py::StmtExpr,
    refs: &mut Vec<String>,
    type_refs: &mut Vec<String>,
) -> Result<(), Diagnostic> {
    let py::Expr::Call(call) = value.value.as_ref() else {
        return Err(cx.at(value, "expression statements must call a source/event helper"));
    };
    let name = expressions::direct_name(cx, &call.func, &HashMap::new())?;
    if !matches!(
        name.as_str(),
        "source_add" | "source_remove" | "source_replace" | "source_list"
            | "event_register" | "event_emit" | "check" | "spawn"
    ) {
        return Err(cx.at(value, format!("unsupported expression statement {name}")));
    }
    if !call.keywords.is_empty() {
        return Err(cx.at(value, "source/event helpers take positional arguments"));
    }
    for arg in &call.args {
        expression(cx, arg, refs, type_refs)?;
    }
    Ok(())
}

/// Structurally validate a case arm pattern and collect its nested call
/// references. Class patterns must name a declared variant or Ok/Err/Some/None;
/// literal/`_` sub-patterns are allowed; guards, `|`, as-, sequence, mapping and
/// star patterns are rejected.
fn arm_pattern(
    cx: &Context<'_>,
    pattern: &py::Pattern,
    type_refs: &mut Vec<String>,
) -> Result<(), Diagnostic> {
    match pattern {
        py::Pattern::MatchClass(class) => {
            if !matches!(class.cls.as_ref(), py::Expr::Name(_)) {
                return Err(cx.at(pattern, "case class patterns require a bare variant name"));
            }
            if let py::Expr::Name(n) = class.cls.as_ref() {
                type_refs.push(n.id.to_string());
            }
            if !class.kwd_attrs.is_empty() || !class.kwd_patterns.is_empty() {
                return Err(cx.at(pattern, "keyword patterns are unsupported in case arms"));
            }
            class
                .patterns
                .iter()
                .try_for_each(|sub| sub_pattern(cx, sub, type_refs))
        }
        py::Pattern::MatchAs(aspat) if aspat.pattern.is_none() && aspat.name.is_none() => Ok(()),
        py::Pattern::MatchSingleton(single) if matches!(single.value, py::Constant::None) => Ok(()),
        py::Pattern::MatchAs(aspat) if aspat.pattern.is_none() => {
            Err(cx.at(pattern, "capture patterns are unsupported; use _ to ignore"))
        }
        py::Pattern::MatchAs(_) => Err(cx.at(pattern, "as-patterns are unsupported")),
        py::Pattern::MatchValue(_) | py::Pattern::MatchSingleton(_) => {
            Err(cx.at(pattern, "top-level literal case patterns are unsupported"))
        }
        py::Pattern::MatchOr(_) => Err(cx.at(pattern, "'|' alternative patterns are unsupported")),
        py::Pattern::MatchSequence(_) => Err(cx.at(pattern, "sequence patterns are unsupported")),
        py::Pattern::MatchMapping(_) => Err(cx.at(pattern, "mapping patterns are unsupported")),
        py::Pattern::MatchStar(_) => Err(cx.at(pattern, "star patterns are unsupported")),
    }
}

fn sub_pattern(
    cx: &Context<'_>,
    pattern: &py::Pattern,
    type_refs: &mut Vec<String>,
) -> Result<(), Diagnostic> {
    match pattern {
        py::Pattern::MatchValue(_) | py::Pattern::MatchSingleton(_) => Ok(()),
        py::Pattern::MatchAs(aspat) if aspat.pattern.is_none() => {
            if let Some(name) = &aspat.name {
                let name = name.as_str();
                if !super::identifier(name) || super::forbidden(name)
                    || super::reserved_positional_name(name) || super::reserved_helper_name(name)
                    || crate::builtins::is_builtin(name) || name == cx.module
                    || matches!(name, "int" | "float" | "bool" | "str" | "Result" | "effects") {
                    return Err(cx.at(pattern, "match capture shadows a reserved binding name"));
                }
            }
            Ok(())
        },
        py::Pattern::MatchAs(aspat) if aspat.pattern.is_some() && aspat.name.is_some() => {
            Err(cx.at(pattern, "as-patterns are unsupported"))
        }
        py::Pattern::MatchClass(class) => {
            if !matches!(class.cls.as_ref(), py::Expr::Name(_)) {
                return Err(cx.at(pattern, "nested variant patterns require a bare variant name"));
            }
            if let py::Expr::Name(n) = class.cls.as_ref() {
                type_refs.push(n.id.to_string());
            }
            if !class.kwd_attrs.is_empty() || !class.kwd_patterns.is_empty() {
                return Err(cx.at(pattern, "keyword patterns are unsupported in case arms"));
            }
            class
                .patterns
                .iter()
                .try_for_each(|sub| sub_pattern(cx, sub, type_refs))
        }
        _ => Err(cx.at(pattern, "unsupported case pattern")),
    }
}

fn expression(
    cx: &Context<'_>,
    expr: &py::Expr,
    refs: &mut Vec<String>,
    type_refs: &mut Vec<String>,
) -> Result<(), Diagnostic> {
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
            .try_for_each(|item| expression(cx, item, refs, type_refs)),
        py::Expr::Call(call) => {
            if !call.args.is_empty() && !call.keywords.is_empty() {
                return Err(cx.at(
                    expr,
                    "mixed positional/keyword calls are unsupported",
                ));
            }
            if !call.keywords.is_empty() {
                // Struct construction: a bare-name callee with named keyword fields.
                let name = expressions::direct_name(cx, &call.func, &HashMap::new())?;
                if name.contains('.') {
                    return Err(cx.at(
                        expr,
                        "struct construction requires a bare type name; dotted calls are functions",
                    ));
                }
                expressions::reject_unsafe_builtin(cx, call, &name)?;
                type_refs.push(name);
                call.keywords.iter().try_for_each(|keyword| {
                    if keyword.arg.is_none() {
                        return Err(cx.at(expr, "** expansion is unsupported in struct construction"));
                    }
                    expression(cx, &keyword.value, refs, type_refs)
                })?;
                return Ok(());
            }
            let name = expressions::direct_name(cx, &call.func, &HashMap::new())?;
            expressions::reject_unsafe_builtin(cx, call, &name)?;
            if !super::reserved_helper_name(&name) {
                if name.contains('.') {
                    refs.push(name);
                } else if !crate::builtins::is_builtin(&name) {
                    refs.push(format!("@local:{name}"));
                }
            }
            call.args
                .iter()
                .try_for_each(|arg| expression(cx, arg, refs, type_refs))
        }
        py::Expr::Attribute(attribute) => attribute_access(cx, attribute, refs, type_refs),
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
            expression(cx, &binary.left, refs, type_refs)?;
            expression(cx, &binary.right, refs, type_refs)
        }
        py::Expr::UnaryOp(unary) if matches!(unary.op, py::UnaryOp::Not | py::UnaryOp::USub) => {
            expression(cx, &unary.operand, refs, type_refs)
        }
        py::Expr::BoolOp(boolean) => boolean
            .values
            .iter()
            .try_for_each(|value| expression(cx, value, refs, type_refs)),
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
            expression(cx, &compare.left, refs, type_refs)?;
            compare
                .comparators
                .iter()
                .try_for_each(|value| expression(cx, value, refs, type_refs))
        }
        _ => Err(cx.at(expr, "unsupported Python expression in the typed frontend")),
    }
}

/// Field access (`p.x`, `p.x.y`) walks a chain of safe identifiers. It never
/// collects a callable reference: dotted *calls* are handled in the `Call` arm.
fn attribute_access(
    cx: &Context<'_>,
    attribute: &py::ExprAttribute,
    refs: &mut Vec<String>,
    type_refs: &mut Vec<String>,
) -> Result<(), Diagnostic> {
    let name = attribute.attr.as_str();
    if !super::identifier(name) || super::forbidden(name) {
        return Err(cx.at(attribute, "unsupported field access identifier"));
    }
    match attribute.value.as_ref() {
        py::Expr::Name(n) => {
            if !super::identifier(n.id.as_str()) || super::forbidden(n.id.as_str()) {
                return Err(cx.at(attribute, "unsupported field access base"));
            }
            Ok(())
        }
        py::Expr::Attribute(inner) => attribute_access(cx, inner, refs, type_refs),
        _ => Err(cx.at(attribute, "field access requires a struct-typed base name")),
    }
}
