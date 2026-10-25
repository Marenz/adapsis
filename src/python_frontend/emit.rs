//! Serialization of the explicitly supported parser nodes. Never copies Python text.
use crate::parser as p;

pub(super) fn quote(text: &str) -> Result<String, String> {
    let mut out = String::from("\"");
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => return Err("unsupported control character in string".into()),
            c => out.push(c),
        }
    }
    out.push('"');
    Ok(out)
}

pub(super) fn name(value: &str) -> Result<String, String> {
    if value.split('.').all(super::identifier) && value.split('.').count() <= 2 {
        Ok(value.into())
    } else {
        Err(format!("unsupported identifier {value:?}"))
    }
}

pub(super) fn ty(value: &p::TypeExpr) -> Result<String, String> {
    match value {
        p::TypeExpr::Named(n) if matches!(n.as_str(), "Int" | "Float" | "Bool" | "String") => {
            Ok(n.clone())
        }
        p::TypeExpr::Generic { name, args }
            if matches!(name.as_str(), "List" | "Result") && args.len() == 1 =>
        {
            Ok(format!("{name}<{}>", ty(&args[0])?))
        }
        _ => Err("unsupported type in canonical emitter".into()),
    }
}

pub(super) fn expr(value: &p::Expr) -> Result<String, String> {
    use p::Expr::*;
    match value {
        Int(n) if *n >= 0 => Ok(n.to_string()),
        Int(_) => Err("negative integer must be a checked unary literal".into()),
        Float(n) if n.is_finite() && !n.is_sign_negative() => {
            // Display uses decimal notation (unlike Debug's exponent form, which
            // the legacy Adapsis lexer does not accept).
            let s = n.to_string();
            Ok(if s.contains('.') { s } else { format!("{s}.0") })
        }
        Float(_) => Err("nonfinite/negative float must be a checked unary literal".into()),
        Bool(v) => Ok(v.to_string()),
        String(s) => quote(s),
        Ident(s) => name(s),
        FieldAccess { base, field } => Ok(format!("{}.{}", expr(base)?, name(field)?)),
        Call { callee, args } => Ok(format!(
            "{}({})",
            expr(callee)?,
            args.iter()
                .map(expr)
                .collect::<Result<Vec<_>, _>>()?
                .join(", ")
        )),
        Binary { op, left, right } => Ok(format!(
            "({} {} {})",
            expr(left)?,
            binary(op)?,
            expr(right)?
        )),
        Unary { op, expr: value } => Ok(format!(
            "({}{})",
            match op {
                p::UnaryOp::Not => "NOT ",
                p::UnaryOp::Neg => "-",
            },
            expr(value)?
        )),
        StructLiteral(fields) => Ok(format!(
            "{{{}}}",
            fields
                .iter()
                .map(|f| Ok(format!("{}: {}", name(&f.name)?, expr(&f.value)?)))
                .collect::<Result<Vec<_>, std::string::String>>()?
                .join(", ")
        )),
        _ => Err("unsupported expression in canonical emitter".into()),
    }
}

fn binary(op: &p::BinaryOp) -> Result<&'static str, String> {
    Ok(match op {
        p::BinaryOp::Add => "+",
        p::BinaryOp::Sub => "-",
        p::BinaryOp::Mul => "*",
        p::BinaryOp::Eq => "==",
        p::BinaryOp::Neq => "!=",
        p::BinaryOp::Gt => ">",
        p::BinaryOp::Lt => "<",
        p::BinaryOp::Gte => ">=",
        p::BinaryOp::Lte => "<=",
        p::BinaryOp::And => "AND",
        p::BinaryOp::Or => "OR",
        _ => return Err("unsupported binary operator in canonical emitter".into()),
    })
}

pub(super) fn operations(ops: &[p::Operation]) -> Result<String, String> {
    let mut out = String::new();
    block(ops, 0, &mut out)?;
    Ok(out)
}

fn line(out: &mut String, indent: usize, text: impl AsRef<str>) {
    out.push_str(&"  ".repeat(indent));
    out.push_str(text.as_ref());
    out.push('\n');
}

fn block(ops: &[p::Operation], indent: usize, out: &mut String) -> Result<(), String> {
    for op in ops {
        operation(op, indent, out)?;
    }
    Ok(())
}

fn operation(op: &p::Operation, i: usize, out: &mut String) -> Result<(), String> {
    use p::Operation::*;
    match op {
        Module(m) => {
            line(out, i, format!("+module {}", name(&m.name)?));
            if let Some(doc) = &m.doc {
                line(out, i, format!("+doc {}", quote(doc)?));
            }
            block(&m.body, i, out)?;
        }
        Function(f) => function(f, i, out)?,
        Let(v) => line(
            out,
            i,
            format!(
                "+let {}:{} = {}",
                name(&v.name)?,
                ty(&v.ty)?,
                expr(&v.expr)?
            ),
        ),
        Set(v) => line(
            out,
            i,
            format!("+set {} = {}", name(&v.name)?, expr(&v.expr)?),
        ),
        Await(v) => line(
            out,
            i,
            format!(
                "+await {}:{} = {}",
                name(&v.name)?,
                ty(&v.ty)?,
                expr(&v.call)?
            ),
        ),
        Return(v) => line(out, i, format!("+return {}", expr(&v.expr)?)),
        If(v) => {
            line(out, i, format!("+if {}", expr(&v.condition)?));
            block(&v.then_body, i + 1, out)?;
            for (cond, body) in &v.elif_branches {
                line(out, i, format!("+elif {}", expr(cond)?));
                block(body, i + 1, out)?;
            }
            if !v.else_body.is_empty() {
                line(out, i, "+else");
                block(&v.else_body, i + 1, out)?;
            }
            line(out, i, "+end");
        }
        While(v) => {
            line(out, i, format!("+while {}", expr(&v.condition)?));
            block(&v.body, i + 1, out)?;
            line(out, i, "+end");
        }
        Each(v) => {
            line(
                out,
                i,
                format!(
                    "+each {} {}:{}",
                    expr(&v.collection)?,
                    name(&v.item)?,
                    ty(&v.item_type)?
                ),
            );
            block(&v.body, i + 1, out)?;
            line(out, i, "+end");
        }
        Test(t) => {
            line(out, i, format!("+test {}", name(&t.function_name)?));
            for case in &t.cases {
                if case.matcher.is_some() || !case.after_checks.is_empty() {
                    return Err("unsupported test matcher/after check".into());
                }
                line(
                    out,
                    i + 1,
                    format!(
                        "+with {} -> expect {}",
                        input(&case.input)?,
                        expr(&case.expected)?
                    ),
                );
            }
        }
        _ => return Err("unsupported operation in canonical emitter".into()),
    }
    Ok(())
}

fn function(f: &p::FunctionDecl, i: usize, out: &mut String) -> Result<(), String> {
    let params = f
        .params
        .iter()
        .map(|p| Ok(format!("{}:{}", name(&p.name)?, ty(&p.ty)?)))
        .collect::<Result<Vec<_>, String>>()?
        .join(", ");
    if f.effects
        .iter()
        .any(|e| !matches!(e.as_str(), "io" | "async" | "fail" | "mut"))
    {
        return Err("unsupported effect in canonical emitter".into());
    }
    let effects = if f.effects.is_empty() {
        String::new()
    } else {
        format!(" [{}]", f.effects.join(", "))
    };
    line(
        out,
        i,
        format!(
            "+fn {}({params}) -> {}{effects}",
            name(&f.name)?,
            ty(&f.return_type)?
        ),
    );
    block(&f.body, i + 1, out)?;
    line(out, i, "+end");
    if let Some(doc) = &f.doc {
        line(out, i, format!("+doc {}", quote(doc)?));
    }
    Ok(())
}

pub(super) fn input(value: &p::Expr) -> Result<String, String> {
    if matches!(value, p::Expr::StructLiteral(f) if f.is_empty()) {
        return Ok(String::new());
    }
    expr(value)
}
