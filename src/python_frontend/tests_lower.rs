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
        let [py::Stmt::Assert(assert)] = body else {
            return Err(cx.at(stmt, "test body must contain exactly one assert target(literals) == literal (optional initial docstring)"));
        };
        let op = assertion(&cx, assert)?;
        source_map.push(cx.span(stmt, operations.len()));
        operations.push(op);
    }
    if operations.is_empty() {
        return Err(cx.error(0, "provide at least one test function"));
    }
    cx.finish(operations, source_map)
}

fn assertion(cx: &Context<'_>, a: &py::StmtAssert) -> Result<p::Operation, Diagnostic> {
    if a.msg.is_some() {
        return Err(cx.at(a, "assert messages are unsupported in test declarations"));
    }
    let py::Expr::Compare(c) = a.test.as_ref() else {
        return Err(cx.at(a, "test requires assert target(literals) == literal"));
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
    let name = direct_name(cx, &call.func, &HashMap::new())?;
    let function_name = if name.contains('.') {
        name
    } else {
        format!("{}.{}", cx.module, name)
    };
    let input = arguments(cx, call)?;
    let expected = literal(cx, &c.comparators[0])?.0;
    Ok(p::Operation::Test(p::TestMutation {
        function_name,
        cases: vec![p::TestCase {
            input,
            expected,
            matcher: None,
            after_checks: vec![],
        }],
    }))
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
