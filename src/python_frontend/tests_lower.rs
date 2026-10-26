use super::expressions::{direct_name, literal};
use super::{Context, Diagnostic, LoweredSource};
use crate::parser as p;
use rustpython_parser::ast as py;
use std::collections::{HashMap, HashSet};

pub fn lower_input(source: &str) -> Result<String, Diagnostic> {
    // This is parsed only; no Python code is executed. Validate the entire suite
    // so text closing the synthetic call cannot introduce another statement.
    let wrapped = format!("__input({source})");
    let cx = Context {
        source: &wrapped,
        module: "<input>",
    };
    let result = (|| {
        let suite = cx.parse()?;
        let [py::Stmt::Expr(e)] = suite.as_slice() else {
            return Err(cx.error(0, "expected only a literal argument list"));
        };
        let py::Expr::Call(c) = e.value.as_ref() else {
            return Err(cx.at(e, "expected literal arguments"));
        };
        if !matches!(c.func.as_ref(), py::Expr::Name(n) if n.id.as_str() == "__input") {
            return Err(cx.at(c, "expected literal argument list"));
        }
        let value = arguments(&cx, c)?;
        let emitted = super::emit::input(&value).map_err(|e| cx.error(0, e))?;
        let reparsed = p::parse_test_input(1, &emitted).map_err(|e| {
            cx.error(
                0,
                format!("canonical input cannot represent these literals: {e}"),
            )
        })?;
        if format!("{value:?}") != format!("{reparsed:?}") {
            return Err(cx.error(
                0,
                "canonical input structural roundtrip mismatch; input rejected",
            ));
        }
        Ok(emitted)
    })();
    result.map_err(|mut e: Diagnostic| {
        if e.line == 1 {
            e.column = e.column.saturating_sub("__input(".len()).max(1);
        }
        e
    })
}

pub fn lower_tests(source: &str, module: &str) -> Result<LoweredSource, Diagnostic> {
    let cx = Context { source, module };
    if !super::identifier(module) {
        return Err(cx.error(0, "test module must be a simple identifier"));
    }
    let suite = cx.parse()?;
    let mut operations = Vec::new();
    let mut source_map = Vec::new();
    let mut names = HashSet::new();
    for stmt in &suite {
        let py::Stmt::FunctionDef(f) = stmt else {
            return Err(cx.at(
                stmt,
                "tests must be def test_name(): declarations containing one literal assertion",
            ));
        };
        if !f.name.starts_with("test_")
            || !super::identifier(f.name.as_str())
            || !names.insert(f.name.to_string())
        {
            return Err(cx.at(stmt, "test requires a unique simple test_name identifier"));
        }
        super::lower::ordinary_args(&cx, stmt, &f.args)?;
        if !f.args.args.is_empty()
            || !f.decorator_list.is_empty()
            || !f.type_params.is_empty()
            || f.type_comment.is_some()
        {
            return Err(cx.at(
                stmt,
                "tests require zero parameters and no decorators, generics or type comments",
            ));
        }
        if f.returns
            .as_deref()
            .is_some_and(|r| !matches!(r, py::Expr::Constant(c) if c.value == py::Constant::None))
        {
            return Err(cx.at(
                f.returns.as_deref().unwrap(),
                "test return annotation must be None or omitted",
            ));
        }
        let body = if f.body.first().and_then(super::docstring).is_some() {
            &f.body[1..]
        } else {
            &f.body
        };
        let mut iter = body.iter();
        let first = iter
            .next()
            .ok_or_else(|| cx.at(stmt, "test body must contain an assertion"))?;
        let op = test_assertion(&cx, stmt, first, iter)?;
        source_map.push(cx.span(stmt, operations.len()));
        operations.push(op);
    }
    if operations.is_empty() {
        return Err(cx.error(0, "provide at least one test function"));
    }
    cx.finish(operations, source_map)
}

/// Lower a test body: one leading assertion plus zero or more `assert_after(...)`
/// side-effect checks.
fn test_assertion<'a>(
    cx: &Context<'_>,
    fn_stmt: &py::Stmt,
    assert_stmt: &py::Stmt,
    after: impl Iterator<Item = &'a py::Stmt>,
) -> Result<p::Operation, Diagnostic> {
    let py::Stmt::Assert(a) = assert_stmt else {
        return Err(cx.at(
            assert_stmt,
            "test body must begin with assert target(...) == expected or a matcher helper",
        ));
    };
    let (function_name, input, expected, matcher) = assertion(cx, fn_stmt, a)?;
    let mut after_checks = Vec::new();
    for stmt in after {
        after_checks.push(assert_after(cx, stmt)?);
    }
    Ok(p::Operation::Test(p::TestMutation {
        function_name,
        cases: vec![p::TestCase {
            input,
            expected,
            matcher,
            after_checks,
        }],
    }))
}

fn assertion(
    cx: &Context<'_>,
    fn_stmt: &py::Stmt,
    a: &py::StmtAssert,
) -> Result<(String, p::Expr, p::Expr, Option<p::TestMatcher>), Diagnostic> {
    if a.msg.is_some() {
        return Err(cx.at(a, "assert messages are unsupported in test declarations"));
    }
    if let py::Expr::Call(call) = a.test.as_ref() {
        return matcher_assertion(cx, call);
    }
    let py::Expr::Compare(c) = a.test.as_ref() else {
        return Err(cx.at(a, "test requires assert target(literals) == literal or a matcher helper"));
    };
    if c.ops != [py::CmpOp::Eq] || c.comparators.len() != 1 {
        return Err(cx.at(c, "test requires one == comparison"));
    }
    let py::Expr::Call(call) = c.left.as_ref() else {
        return Err(cx.at(
            c.left.as_ref(),
            "test left side must call the named target function",
        ));
    };
    let function_name = qualify(cx, call)?;
    let input = arguments(cx, call)?;
    let (expected, matcher) = parse_expected(cx, &c.comparators[0])?;
    let _ = fn_stmt;
    Ok((function_name, input, expected, matcher))
}

/// `assert contains(f(x), "s")`, `assert starts_with(f(x), "p")`,
/// `assert is_ok(f(x))` and `assert is_err(f(x))` matcher helpers.
fn matcher_assertion(
    cx: &Context<'_>,
    call: &py::ExprCall,
) -> Result<(String, p::Expr, p::Expr, Option<p::TestMatcher>), Diagnostic> {
    let name = direct_name(cx, &call.func, &HashMap::new())?;
    if name.contains('.') {
        return Err(cx.at(call, "matcher helpers must be bare names"));
    }
    match name.as_str() {
        "contains" | "starts_with" => {
            if !call.keywords.is_empty() || call.args.len() != 2 {
                return Err(cx.at(call, format!("{name}(f(x), \"substring\") expects 2 arguments")));
            }
            let py::Expr::Call(inner) = &call.args[0] else {
                return Err(cx.at(&call.args[0], "matcher first argument must be a function call"));
            };
            let function_name = qualify(cx, inner)?;
            let input = arguments(cx, inner)?;
            let s = string_literal(cx, &call.args[1])?
                .ok_or_else(|| cx.at(&call.args[1], "matcher substring must be a string literal"))?;
            let matcher = if name == "contains" {
                p::TestMatcher::Contains(s.clone())
            } else {
                p::TestMatcher::StartsWith(s.clone())
            };
            Ok((function_name, input, p::Expr::String(s), Some(matcher)))
        }
        "is_ok" | "is_err" => {
            if !call.keywords.is_empty() || call.args.len() != 1 {
                return Err(cx.at(call, format!("{name}(f(x)) expects 1 argument")));
            }
            let py::Expr::Call(inner) = &call.args[0] else {
                return Err(cx.at(&call.args[0], "matcher first argument must be a function call"));
            };
            let function_name = qualify(cx, inner)?;
            let input = arguments(cx, inner)?;
            let (variant, matcher) = if name == "is_ok" {
                ("Ok", p::TestMatcher::AnyOk)
            } else {
                ("Err", p::TestMatcher::AnyErr)
            };
            Ok((
                function_name,
                input,
                p::Expr::Ident(variant.to_string()),
                Some(matcher),
            ))
        }
        other => Err(cx.at(call, format!("unsupported matcher helper {other}"))),
    }
}

fn qualify(cx: &Context<'_>, call: &py::ExprCall) -> Result<String, Diagnostic> {
    let name = direct_name(cx, &call.func, &HashMap::new())?;
    Ok(if name.contains('.') {
        name
    } else {
        format!("{}.{}", cx.module, name)
    })
}

/// `assert f(x) == Err("msg")` → `ErrContaining`; `assert f(x) == Err(label)` →
/// an exact `Err(label)` match; otherwise a literal equality.
fn parse_expected(cx: &Context<'_>, e: &py::Expr) -> Result<(p::Expr, Option<p::TestMatcher>), Diagnostic> {
    if let Some(found) = parse_err_expected(cx, e)? {
        return Ok(found);
    }
    Ok((literal(cx, e)?.0, None))
}

fn parse_err_expected(
    cx: &Context<'_>,
    e: &py::Expr,
) -> Result<Option<(p::Expr, Option<p::TestMatcher>)>, Diagnostic> {
    let py::Expr::Call(c) = e else {
        return Ok(None);
    };
    let py::Expr::Name(n) = c.func.as_ref() else {
        return Ok(None);
    };
    if n.id.as_str() != "Err" || !c.keywords.is_empty() || c.args.len() != 1 {
        return Ok(None);
    }
    if let Some(s) = string_literal(cx, &c.args[0])? {
        let expected = super::expressions::call("Err", vec![p::Expr::String(s.clone())]);
        return Ok(Some((expected, Some(p::TestMatcher::ErrContaining(s)))));
    }
    let py::Expr::Name(arg) = &c.args[0] else {
        return Err(cx.at(&c.args[0], "Err(...) expected a string or identifier"));
    };
    let expected = super::expressions::call("Err", vec![p::Expr::Ident(arg.id.to_string())]);
    Ok(Some((expected, None)))
}

/// `assert_after("target", "matcher", "value")` → `AfterCheck`.
fn assert_after(cx: &Context<'_>, stmt: &py::Stmt) -> Result<p::AfterCheck, Diagnostic> {
    let py::Stmt::Expr(e) = stmt else {
        return Err(cx.at(stmt, "only the leading assert and assert_after(...) calls are allowed in test bodies"));
    };
    let py::Expr::Call(c) = e.value.as_ref() else {
        return Err(cx.at(e, "assert_after requires a call"));
    };
    let name = direct_name(cx, &c.func, &HashMap::new())?;
    if name != "assert_after" {
        return Err(cx.at(e, "only assert_after(...) calls may follow the leading assert"));
    }
    if !c.keywords.is_empty() || c.args.len() != 3 {
        return Err(cx.at(c, "assert_after(target, matcher, value) expects 3 string arguments"));
    }
    let target = string_literal(cx, &c.args[0])?
        .ok_or_else(|| cx.at(&c.args[0], "assert_after target must be a string literal"))?;
    let matcher = string_literal(cx, &c.args[1])?
        .ok_or_else(|| cx.at(&c.args[1], "assert_after matcher must be a string literal"))?;
    let value = string_literal(cx, &c.args[2])?
        .ok_or_else(|| cx.at(&c.args[2], "assert_after value must be a string literal"))?;
    Ok(p::AfterCheck { target, matcher, value })
}

fn string_literal(cx: &Context<'_>, e: &py::Expr) -> Result<Option<String>, Diagnostic> {
    let _ = cx;
    match e {
        py::Expr::Constant(c) => match &c.value {
            py::Constant::Str(s) => Ok(Some(s.clone())),
            _ => Ok(None),
        },
        _ => Ok(None),
    }
}

fn arguments(cx: &Context<'_>, c: &py::ExprCall) -> Result<p::Expr, Diagnostic> {
    if !c.args.is_empty() && !c.keywords.is_empty() {
        return Err(cx.at(c, "mixed positional/keyword arguments are unsupported; use only positional or only named literals"));
    }
    if !c.keywords.is_empty() {
        let mut names = HashSet::new();
        let mut fields = Vec::new();
        for keyword in &c.keywords {
            let Some(name) = &keyword.arg else {
                return Err(cx.at(keyword, "** keyword expansion is unsupported"));
            };
            if super::reserved_positional_name(name.as_str()) {
                return Err(cx.at(
                    keyword,
                    "keyword names _0, _1, ... are reserved; use the declared parameter names",
                ));
            }
            if !super::identifier(name.as_str()) || !names.insert(name.to_string()) {
                return Err(cx.at(keyword, "keyword names must be unique safe identifiers"));
            }
            fields.push(p::FieldValue {
                name: name.to_string(),
                value: literal(cx, &keyword.value)?.0,
            });
        }
        return Ok(p::Expr::StructLiteral(fields));
    }
    let values = c
        .args
        .iter()
        .map(|e| literal(cx, e).map(|(e, _)| e))
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() == 1 {
        return Ok(values.into_iter().next().unwrap());
    }
    Ok(p::Expr::StructLiteral(
        values
            .into_iter()
            .enumerate()
            .map(|(i, value)| p::FieldValue {
                name: format!("_{i}"),
                value,
            })
            .collect(),
    ))
}
