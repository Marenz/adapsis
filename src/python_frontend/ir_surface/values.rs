use super::{Checked, p};

pub(super) fn bare(value: &str) -> Checked {
    if super::super::identifier(value) {
        Ok(())
    } else {
        Err("unsupported identifier in Python IR".into())
    }
}

pub(super) fn name(value: &str) -> Checked {
    if value.split('.').count() > 2 {
        return Err("call/name has too many owner components".into());
    }
    value.split('.').try_for_each(bare)
}

pub(super) fn variant(value: &str) -> Checked {
    if value == "None" { Ok(()) } else { bare(value) }
}

pub(super) fn text(value: &str) -> Checked {
    if value
        .chars()
        .any(|c| c.is_control() && !matches!(c, '\n' | '\r' | '\t'))
    {
        Err("unsupported control character in string".into())
    } else {
        Ok(())
    }
}

pub(super) fn ty(value: &p::TypeExpr) -> Checked {
    match value {
        p::TypeExpr::Named(n) => bare(n),
        p::TypeExpr::Generic { name, args } => {
            let count = match name.as_str() {
                "List" | "Set" | "Option" | "Result" => 1,
                "Map" => 2,
                _ => return Err("unsupported generic type".into()),
            };
            if args.len() != count {
                return Err("invalid generic arity".into());
            }
            args.iter().try_for_each(ty)
        }
        p::TypeExpr::Struct(_) => {
            Err("anonymous type annotation is outside supported Python IR".into())
        }
    }
}

pub(super) fn call(value: &p::Expr) -> Checked {
    let p::Expr::Call { callee, args } = value else {
        return Err("expected an explicit call".into());
    };
    match callee.as_ref() {
        p::Expr::Ident(n) => {
            if n == "None" {
                Ok(())
            } else {
                name(n)
            }
        }
        p::Expr::FieldAccess { base, field } if matches!(base.as_ref(), p::Expr::Ident(_)) => {
            let p::Expr::Ident(owner) = base.as_ref() else {
                unreachable!()
            };
            bare(owner)?;
            bare(field)
        }
        _ => Err("dynamic call target is outside supported Python IR".into()),
    }?;
    args.iter().try_for_each(expr)
}

pub(super) fn expr(value: &p::Expr) -> Checked {
    match value {
        p::Expr::Int(n) if *n >= 0 => Ok(()),
        p::Expr::Float(n) if n.is_finite() && !n.is_sign_negative() => Ok(()),
        p::Expr::Bool(_) => Ok(()),
        p::Expr::String(s) => text(s),
        p::Expr::Ident(n) => {
            if n == "None" {
                Ok(())
            } else {
                name(n)
            }
        }
        p::Expr::FieldAccess { base, field } => {
            expr(base)?;
            bare(field)
        }
        p::Expr::Call { .. } => call(value),
        p::Expr::Binary { left, right, .. } => {
            expr(left)?;
            expr(right)
        }
        p::Expr::Unary { expr: inner, .. } => expr(inner),
        p::Expr::StructLiteral(fields) => {
            let mut names = std::collections::HashSet::new();
            for field in fields {
                bare(&field.name)?;
                expr(&field.value)?;
                if !names.insert(&field.name) {
                    return Err("duplicate record field".into());
                }
            }
            Ok(())
        }
        p::Expr::Cast { .. } => {
            Err("casts are unsupported; canonical conversion would erase the cast".into())
        }
        _ => Err(
            "nonfinite/negative numeric constant requires supported unary representation".into(),
        ),
    }
}

pub(super) fn arm(arm: &p::MatchArmDecl) -> Checked {
    if arm.variant == "_" {
        return if arm.bindings.is_empty() && arm.patterns.is_none() {
            Ok(())
        } else {
            Err("wildcard arm cannot carry bindings/patterns".into())
        };
    }
    variant(&arm.variant)?;
    let mut captures = std::collections::HashSet::new();
    if let Some(patterns) = &arm.patterns {
        if patterns.len() != arm.bindings.len()
            || patterns
                .iter()
                .all(|p| matches!(p, p::MatchPatternDecl::Binding(_)))
        {
            return Err("nested pattern/binding shape mismatch".into());
        }
        for (index, (p, binding)) in patterns.iter().zip(&arm.bindings).enumerate() {
            let expected = match p {
                p::MatchPatternDecl::Binding(n) => n.clone(),
                _ => format!("__pat_{index}"),
            };
            if *binding != expected {
                return Err("nested pattern placeholder mismatch".into());
            }
            pattern(p, &mut captures)?;
        }
    } else {
        for binding in &arm.bindings {
            capture(binding, &mut captures)?;
        }
    }
    Ok(())
}

fn capture(value: &str, captures: &mut std::collections::HashSet<String>) -> Checked {
    bare(value)?;
    if value != "_" && !captures.insert(value.to_owned()) {
        return Err("duplicate pattern capture".into());
    }
    Ok(())
}

fn pattern(pat: &p::MatchPatternDecl, captures: &mut std::collections::HashSet<String>) -> Checked {
    match pat {
        p::MatchPatternDecl::Binding(n) => capture(n, captures),
        p::MatchPatternDecl::Variant {
            variant: n,
            sub_patterns,
        } => {
            variant(n)?;
            if n == "_" || sub_patterns.is_empty() {
                return Err("unsupported nested variant pattern shape".into());
            }
            for sub in sub_patterns {
                pattern(sub, captures)?;
            }
            Ok(())
        }
        p::MatchPatternDecl::LiteralInt(_) | p::MatchPatternDecl::LiteralBool(_) => Ok(()),
        p::MatchPatternDecl::LiteralString(s) => text(s),
    }
}

pub(super) fn payload_type(value: &str) -> Checked {
    // Reuse the existing low-level type grammar, never parse a synthesized
    // declaration. This mini-language remains textual in the canonical IR.
    if value.contains(['\n', '\r']) {
        return Err("invalid event payload type token".into());
    }
    let parsed = p::parse_type(1, value).map_err(|_| "invalid event payload type token")?;
    ty(&parsed)
}

pub(super) fn test(test: &p::TestMutation) -> Checked {
    name(&test.function_name)?;
    if test.cases.is_empty() {
        return Err("empty test set".into());
    }
    for case in &test.cases {
        expr(&case.input)?;
        expr(&case.expected)?;
        if let p::Expr::StructLiteral(fields) = &case.input {
            for field in fields {
                data(&field.value)?;
            }
            let positional = fields
                .iter()
                .any(|field| super::super::reserved_positional_name(&field.name));
            if positional
                && (fields.len() < 2
                    || fields
                        .iter()
                        .enumerate()
                        .any(|(i, field)| field.name != format!("_{i}")))
            {
                return Err("invalid positional argument carrier".into());
            }
        } else {
            data(&case.input)?;
        }
        expectation(case)?;
        for after in &case.after_checks {
            bare(&after.target)?;
            bare(&after.matcher)?;
            text(&after.value)?;
        }
    }
    Ok(())
}

fn data(value: &p::Expr) -> Checked {
    match value {
        p::Expr::Int(_) | p::Expr::Float(_) | p::Expr::Bool(_) | p::Expr::String(_) => expr(value),
        p::Expr::Unary {
            op: p::UnaryOp::Neg,
            expr: inner,
        } if matches!(inner.as_ref(), p::Expr::Int(_) | p::Expr::Float(_)) => expr(value),
        p::Expr::Call { callee, args } if matches!(callee.as_ref(), p::Expr::Ident(n) if n == "list") => {
            args.iter().try_for_each(data)
        }
        _ => Err("test data must be literals or list constructors".into()),
    }
}

fn expectation(case: &p::TestCase) -> Checked {
    use p::{Expr, TestMatcher};
    let matches = match &case.matcher {
        Some(TestMatcher::Contains(s) | TestMatcher::StartsWith(s)) => {
            matches!(&case.expected, Expr::String(v) if v == s)
        }
        Some(TestMatcher::AnyOk) => matches!(&case.expected, Expr::Ident(v) if v == "Ok"),
        Some(TestMatcher::AnyErr) => matches!(&case.expected, Expr::Ident(v) if v == "Err"),
        Some(TestMatcher::ErrContaining(s)) => matches!(&case.expected, Expr::Call { callee, args }
            if matches!(callee.as_ref(), Expr::Ident(v) if v == "Err") && matches!(args.as_slice(), [Expr::String(v)] if v == s)),
        None => {
            if let Expr::Call { callee, args } = &case.expected {
                if matches!(callee.as_ref(), Expr::Ident(v) if v == "Err")
                    && matches!(args.as_slice(), [Expr::Ident(_)])
                {
                    let Expr::Ident(label) = &args[0] else {
                        unreachable!()
                    };
                    return bare(label); // Existing exact Err(label) representation.
                }
            }
            return data(&case.expected);
        }
    };
    if matches {
        Ok(())
    } else {
        Err("matcher/expected-value mismatch".into())
    }
}
