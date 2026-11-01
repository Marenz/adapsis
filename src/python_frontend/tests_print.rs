//! Harness-only inverse of `tests_lower`: prints parser test data back into the
//! restricted Python test syntax. It accepts exactly what the lowerer accepts (bool,
//! str, int, finite float, signed numerics and lists as data; keyword or positional
//! arguments; equality, `contains`/`starts_with`/`is_ok`/`is_err`/`Err(..)` and
//! `assert_after`) and refuses everything else with a precise reason. The Python
//! test-data literal policy is NOT broadened: records, variant constructors, calls,
//! `Ok(..)`/`Err(..)` inputs and non-finite floats are unsupported here.
use super::emit;
use crate::parser as p;

/// One `def test_case_<n>() -> None:` per case, joined by blank lines. The def names
/// are positional (never derived from the target) so that dunder-bearing function
/// names cannot make the declaration itself invalid.
pub(crate) fn print_block(test: &p::TestMutation) -> Result<String, String> {
    if test.cases.is_empty() {
        return Err("test block has no cases".into());
    }
    let mut defs = Vec::new();
    for (index, case) in test.cases.iter().enumerate() {
        let body = print_case(&test.function_name, case)
            .map_err(|reason| format!("case {}: {reason}", index + 1))?;
        defs.push(format!("def test_case_{}() -> None:\n{body}", index + 1));
    }
    Ok(defs.join("\n"))
}

/// The indented assertion line plus any `assert_after` lines for one case.
pub(crate) fn print_case(function: &str, case: &p::TestCase) -> Result<String, String> {
    let call = format!("{}({})", emit::name(function)?, arguments(&case.input)?);
    let mut lines = vec![format!("    assert {}", assertion(&call, case)?)];
    for check in &case.after_checks {
        lines.push(format!(
            "    assert_after({}, {}, {})",
            emit::quote(&check.target)?,
            emit::quote(&check.matcher)?,
            emit::quote(&check.value)?
        ));
    }
    Ok(lines.join("\n") + "\n")
}

fn assertion(call: &str, case: &p::TestCase) -> Result<String, String> {
    match &case.matcher {
        None => match &case.expected {
            // `Err(label)` with an identifier label is the exact-label form.
            p::Expr::Call { callee, args } if is_ident(callee, "Err") => match args.as_slice() {
                [p::Expr::Ident(label)] if super::identifier(label) => {
                    Ok(format!("{call} == Err({label})"))
                }
                _ => Err(
                    "an Err expectation needs an identifier label or an ErrContaining matcher"
                        .into(),
                ),
            },
            expected => Ok(format!("{call} == {}", literal(expected)?)),
        },
        Some(p::TestMatcher::Contains(text)) => {
            require_string(&case.expected, text)?;
            Ok(format!("contains({call}, {})", emit::quote(text)?))
        }
        Some(p::TestMatcher::StartsWith(text)) => {
            require_string(&case.expected, text)?;
            Ok(format!("starts_with({call}, {})", emit::quote(text)?))
        }
        Some(p::TestMatcher::AnyOk) => {
            require_ident(&case.expected, "Ok")?;
            Ok(format!("is_ok({call})"))
        }
        Some(p::TestMatcher::AnyErr) => {
            require_ident(&case.expected, "Err")?;
            Ok(format!("is_err({call})"))
        }
        Some(p::TestMatcher::ErrContaining(text)) => match &case.expected {
            p::Expr::Call { callee, args }
                if is_ident(callee, "Err")
                    && matches!(args.as_slice(), [p::Expr::String(s)] if s == text) =>
            {
                Ok(format!("{call} == Err({})", emit::quote(text)?))
            }
            _ => Err("ErrContaining matcher without the matching Err(\"..\") expectation".into()),
        },
    }
}

fn is_ident(expr: &p::Expr, name: &str) -> bool {
    matches!(expr, p::Expr::Ident(n) if n == name)
}

fn require_string(expected: &p::Expr, text: &str) -> Result<(), String> {
    match expected {
        p::Expr::String(s) if s == text => Ok(()),
        _ => Err("matcher text differs from the stored expectation".into()),
    }
}

fn require_ident(expected: &p::Expr, name: &str) -> Result<(), String> {
    if is_ident(expected, name) {
        Ok(())
    } else {
        Err(format!("matcher requires the bare `{name}` expectation"))
    }
}

/// Keyword arguments for named fields, positional for `_0.._n` (n >= 1), one
/// positional literal otherwise. Exactly the shapes `tests_lower::arguments` builds.
fn arguments(input: &p::Expr) -> Result<String, String> {
    let p::Expr::StructLiteral(fields) = input else {
        return literal(input);
    };
    if fields.is_empty() {
        return Ok(String::new());
    }
    let positional = fields.len() >= 2
        && fields
            .iter()
            .enumerate()
            .all(|(index, field)| field.name == format!("_{index}"));
    if positional {
        let values = fields
            .iter()
            .map(|field| literal(&field.value))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(values.join(", "));
    }
    let mut seen = std::collections::HashSet::new();
    let mut parts = Vec::new();
    for field in fields {
        if super::reserved_positional_name(&field.name) {
            return Err(format!(
                "field name {} is reserved for positional arguments and cannot be a keyword",
                field.name
            ));
        }
        if !super::identifier(&field.name) || !seen.insert(field.name.as_str()) {
            return Err(format!(
                "field name {:?} is not a unique safe keyword",
                field.name
            ));
        }
        parts.push(format!("{}={}", field.name, literal(&field.value)?));
    }
    Ok(parts.join(", "))
}

/// #111: the parser yields a single negative CONSTANT (`Int(-1)`, `Float(-1.5)`) only for
/// keyword shorthand (`+with a=-1`); every other spelling (`{a:-1}`, positional, list
/// items, expectations) is `Unary Neg` over a positive constant, which is also what the
/// Python lowerer builds. A negative constant node therefore has no Python spelling that
/// lowers back to it, and printing `-1` would silently change the AST.
const NEGATIVE_CONSTANT: &str = "a negative numeric constant (keyword shorthand `name=-1`) has no exact Python form: Python lowers `-1` to a unary minus, a different AST (#111)";

/// What `expressions::literal` accepts, printed back.
fn literal(expr: &p::Expr) -> Result<String, String> {
    match expr {
        p::Expr::Bool(value) => Ok(if *value { "True" } else { "False" }.into()),
        p::Expr::String(text) => emit::quote(text),
        p::Expr::Int(n) if *n == i64::MIN => {
            Err("-9223372036854775808 cannot roundtrip through the Adapsis lexer".into())
        }
        p::Expr::Int(n) if *n < 0 => Err(NEGATIVE_CONSTANT.into()),
        p::Expr::Int(n) => Ok(n.to_string()),
        p::Expr::Float(f) if f.is_finite() && f.is_sign_negative() => Err(NEGATIVE_CONSTANT.into()),
        p::Expr::Float(f) if f.is_finite() => Ok(format!("{f:?}")),
        p::Expr::Float(_) => Err("non-finite float is not a test literal".into()),
        p::Expr::Unary {
            op: p::UnaryOp::Neg,
            expr,
        } => match expr.as_ref() {
            // A negative inner constant (double sign) is refused by `literal` itself.
            inner @ (p::Expr::Int(_) | p::Expr::Float(_)) => Ok(format!("-{}", literal(inner)?)),
            _ => Err("a sign is only supported on one numeric constant".into()),
        },
        p::Expr::Call { callee, args } if is_ident(callee, "list") => {
            let items = args.iter().map(literal).collect::<Result<Vec<_>, _>>()?;
            Ok(format!("[{}]", items.join(", ")))
        }
        other => Err(format!("{} is not a Python test-data literal", kind(other))),
    }
}

fn kind(expr: &p::Expr) -> String {
    match expr {
        p::Expr::Ident(name) => format!("identifier {name}"),
        p::Expr::Call { callee, .. } => match callee.as_ref() {
            p::Expr::Ident(name) => format!("call to {name}"),
            _ => "call".into(),
        },
        p::Expr::StructLiteral(_) => "record literal".into(),
        p::Expr::FieldAccess { .. } => "field access".into(),
        p::Expr::Binary { .. } => "operator expression".into(),
        p::Expr::Unary { .. } => "unary expression".into(),
        p::Expr::Cast { .. } => "cast".into(),
        _ => "expression".into(),
    }
}
