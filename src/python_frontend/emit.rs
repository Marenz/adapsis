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

/// A bare identifier that may be a Python keyword: `None` names the Adapsis
/// `None` variant (which is not a Python identifier but is a valid Adapsis one).
pub(super) fn bare_name(value: &str) -> Result<String, String> {
    if value == "None" {
        return Ok("None".into());
    }
    name(value)
}

pub(super) fn ty(value: &p::TypeExpr) -> Result<String, String> {
    match value {
        p::TypeExpr::Named(n) => {
            if n.contains('.') {
                return Err(format!(
                    "unsupported qualified type name {n:?} in canonical emitter"
                ));
            }
            name(n)
        }
        p::TypeExpr::Generic { name, args } => match name.as_str() {
            "List" | "Set" | "Option" | "Result" if args.len() == 1 => {
                Ok(format!("{name}<{}>", ty(&args[0])?))
            }
            "Map" if args.len() == 2 => {
                Ok(format!("Map<{}, {}>", ty(&args[0])?, ty(&args[1])?))
            }
            _ => Err(format!(
                "unsupported generic type {name} in canonical emitter"
            )),
        },
        p::TypeExpr::Struct(_) => {
            Err("unsupported anonymous struct type in canonical emitter".into())
        }
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
        Ident(s) => bare_name(s),
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
        p::BinaryOp::Div => "/",
        p::BinaryOp::Mod => "%",
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
        Type(t) => type_decl(t, i, out)?,
        SharedVar(_) | Route { .. } | ModuleSource(_) | Startup(_) | Shutdown(_)
        | SourceAdd(_) | SourceRemove(_) | SourceReplace(_) | SourceList
        | EventRegister(_) | EventEmit(_) => declaration(op, i, out)?,
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
        Call(_) | Check(_) | Spawn(_) => call_check_spawn(op, i, out)?,
        If(v) => if_op(v, i, out)?,
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
        Test(t) => test_block(t, i, out)?,
        Match(v) => {
            line(out, i, format!("+match {}", expr(&v.expr)?));
            for arm in &v.arms {
                line(out, i, format!("+case {}", case_pattern(arm)?));
                block(&arm.body, i + 1, out)?;
            }
            line(out, i, "+end");
        }
        _ => return Err("unsupported operation in canonical emitter".into()),
    }
    Ok(())
}

/// Module-level and in-body declaration operations (`+shared`, `+route`,
/// `+source`, `+startup`/`+shutdown`, `+event`).
fn declaration(op: &p::Operation, i: usize, out: &mut String) -> Result<(), String> {
    use p::Operation::*;
    match op {
        SharedVar(v) => line(
            out,
            i,
            format!("+shared {}:{} = {}", name(&v.name)?, ty(&v.ty)?, expr(&v.default)?),
        ),
        Route { method, path, handler_fn } => line(
            out,
            i,
            format!("+route {} {} -> {}", method.to_uppercase(), quote(path)?, name(handler_fn)?),
        ),
        ModuleSource(src) => {
            let mut config = String::new();
            for (key, value) in &src.config {
                config.push_str(&format!(" {key}={value}"));
            }
            line(
                out,
                i,
                format!(
                    "+source {} {}{} -> {}",
                    name(&src.name)?,
                    name(&src.source_type)?,
                    config,
                    name(&src.handler)?
                ),
            );
        }
        Startup(f) => lifecycle(out, i, "+startup", f)?,
        Shutdown(f) => lifecycle(out, i, "+shutdown", f)?,
        SourceAdd(v) => line(
            out,
            i,
            format!("+source add {} as {} -> {}", v.kind_text, name(&v.alias)?, name(&v.handler)?),
        ),
        SourceRemove(alias) => line(out, i, format!("+source remove {}", name(alias)?)),
        SourceReplace(v) => line(
            out,
            i,
            format!("+source replace {} {} -> {}", name(&v.alias)?, v.kind_text, name(&v.handler)?),
        ),
        SourceList => line(out, i, "+source list"),
        EventRegister(v) => line(
            out,
            i,
            format!("+event register {}({})", name(&v.name)?, v.payload_type),
        ),
        EventEmit(v) => line(
            out,
            i,
            format!("+event emit {} {}", name(&v.name)?, expr(&v.value)?),
        ),
        _ => unreachable!("declaration() only handles declaration operations"),
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
    if f.effects.iter().any(|e| {
        !matches!(
            e.as_str(),
            "io" | "async" | "fail" | "mut" | "rand" | "yield" | "parallel" | "unsafe"
        )
    }) {
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

/// `+if`/`+elif`/`+else` serialization.
fn if_op(v: &p::IfDecl, i: usize, out: &mut String) -> Result<(), String> {
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
    Ok(())
}

/// `+call`/`+check`/`+spawn` serialization.
fn call_check_spawn(op: &p::Operation, i: usize, out: &mut String) -> Result<(), String> {
    use p::Operation::*;
    match op {
        Call(v) => line(
            out,
            i,
            format!("+call {}:{} = {}", name(&v.name)?, ty(&v.ty)?, expr(&v.expr)?),
        ),
        Check(v) => line(
            out,
            i,
            format!(
                "+check {} {} ~{}",
                name(&v.name)?,
                expr(&v.expr)?,
                name(&v.err_label)?
            ),
        ),
        Spawn(v) => match &v.binding {
            Some((binding_name, binding_ty)) => line(
                out,
                i,
                format!(
                    "+spawn {}:{} = {}",
                    name(binding_name)?,
                    ty(binding_ty)?,
                    expr(&v.call)?
                ),
            ),
            None => line(out, i, format!("+spawn {}", expr(&v.call)?)),
        },
        _ => unreachable!("call_check_spawn only handles call/check/spawn"),
    }
    Ok(())
}

/// `+test`/`+with`/`+after` serialization (including matchers).
fn test_block(t: &p::TestMutation, i: usize, out: &mut String) -> Result<(), String> {
    line(out, i, format!("+test {}", name(&t.function_name)?));
    for case in &t.cases {
        let expected = match &case.matcher {
            None => format!("expect {}", expr(&case.expected)?),
            Some(p::TestMatcher::Contains(s)) => format!("expect contains({})", quote(s)?),
            Some(p::TestMatcher::StartsWith(s)) => format!("expect starts_with({})", quote(s)?),
            Some(p::TestMatcher::AnyOk) => "expect Ok".to_string(),
            Some(p::TestMatcher::AnyErr) => "expect Err".to_string(),
            Some(p::TestMatcher::ErrContaining(s)) => format!("expect Err({})", quote(s)?),
        };
        line(out, i + 1, format!("+with {} -> {expected}", input(&case.input)?));
        for after in &case.after_checks {
            line(
                out,
                i + 1,
                format!(
                    "+after {} {} {}",
                    name(&after.target)?,
                    name(&after.matcher)?,
                    quote(&after.value)?
                ),
            );
        }
    }
    Ok(())
}

/// `+startup`/`+shutdown` lifecycle blocks: effects in brackets, then body.
fn lifecycle(
    out: &mut String,
    i: usize,
    keyword: &str,
    f: &p::FunctionDecl,
) -> Result<(), String> {
    let effects = if f.effects.is_empty() {
        String::new()
    } else {
        format!(" [{}]", f.effects.join(", "))
    };
    line(out, i, format!("{keyword}{effects}"));
    block(&f.body, i + 1, out)?;
    line(out, i, "+end");
    Ok(())
}

fn type_decl(t: &p::TypeDecl, i: usize, out: &mut String) -> Result<(), String> {
    match &t.body {
        p::TypeBody::Struct(fields) => {
            let fields = fields
                .iter()
                .map(|f| Ok(format!("{}:{}", name(&f.name)?, ty(&f.ty)?)))
                .collect::<Result<Vec<_>, String>>()?
                .join(", ");
            line(out, i, format!("+type {} = {{{fields}}}", name(&t.name)?));
        }
        p::TypeBody::Union(variants) => {
            let variants = variants
                .iter()
                .map(|v| {
                    if v.payload.is_empty() {
                        bare_name(&v.name)
                    } else {
                        let payload = v
                            .payload
                            .iter()
                            .map(ty)
                            .collect::<Result<Vec<_>, String>>()?
                            .join(", ");
                        Ok(format!("{}({payload})", bare_name(&v.name)?))
                    }
                })
                .collect::<Result<Vec<_>, String>>()?
                .join(" | ");
            line(out, i, format!("+type {} = {variants}", name(&t.name)?));
        }
        p::TypeBody::Alias(_) => {
            return Err("unsupported type alias in canonical emitter".into());
        }
    }
    Ok(())
}

/// Render a `+case` arm head: the variant name plus its bindings or nested
/// patterns. `_` is the wildcard; zero-payload variants render as a bare name.
fn case_pattern(arm: &p::MatchArmDecl) -> Result<String, String> {
    if arm.variant == "_" {
        return Ok("_".into());
    }
    match &arm.patterns {
        None => {
            if arm.bindings.is_empty() {
                bare_name(&arm.variant)
            } else {
                Ok(format!(
                    "{}({})",
                    bare_name(&arm.variant)?,
                    arm.bindings
                        .iter()
                        .map(|b| bare_name(b))
                        .collect::<Result<Vec<_>, String>>()?
                        .join(", ")
                ))
            }
        }
        Some(patterns) => Ok(format!(
            "{}({})",
            bare_name(&arm.variant)?,
            patterns
                .iter()
                .map(match_pattern)
                .collect::<Result<Vec<_>, String>>()?
                .join(", ")
        )),
    }
}

fn match_pattern(pattern: &p::MatchPatternDecl) -> Result<String, String> {
    match pattern {
        p::MatchPatternDecl::Binding(name) => bare_name(name),
        p::MatchPatternDecl::LiteralInt(n) => Ok(n.to_string()),
        p::MatchPatternDecl::LiteralBool(b) => Ok(b.to_string()),
        p::MatchPatternDecl::LiteralString(s) => quote(s),
        p::MatchPatternDecl::Variant {
            variant,
            sub_patterns,
        } => {
            if sub_patterns.is_empty() {
                bare_name(variant)
            } else {
                Ok(format!(
                    "{}({})",
                    bare_name(variant)?,
                    sub_patterns
                        .iter()
                        .map(match_pattern)
                        .collect::<Result<Vec<_>, String>>()?
                        .join(", ")
                ))
            }
        }
    }
}

pub(super) fn input(value: &p::Expr) -> Result<String, String> {
    if matches!(value, p::Expr::StructLiteral(f) if f.is_empty()) {
        return Ok(String::new());
    }
    expr(value)
}
