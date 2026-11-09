use super::*;
use crate::{ast, eval, parser, validator};

fn result(ty: Ty) -> Ty {
    Ty::Result(Box::new(ty))
}

fn build(source: &str) -> ast::Program {
    let lowered =
        super::super::lower_module(source, "ResultGate").unwrap_or_else(|e| panic!("{e}"));
    let mut p = ast::Program::default();
    for op in &lowered.operations {
        validator::apply_and_validate(&mut p, op).unwrap();
    }
    p.rebuild_function_index();
    p
}

#[test]
fn implicit_return_requires_exact_known_inner_type_and_is_never_nested_lifting() {
    for ty in [
        Ty::Int,
        Ty::Float,
        Ty::Str,
        Ty::Bool,
        Ty::List(Box::new(Ty::Int)),
        Ty::Named("M.Point".into()),
    ] {
        assert!(implicit_inner_matches(&result(ty.clone()), &ty));
    }
    for (expected, actual) in [
        (result(Ty::Float), Ty::Int),
        (result(Ty::Int), Ty::Float),
        (result(Ty::Int), Ty::Unknown),
        (result(Ty::Unknown), Ty::Unknown),
        (
            result(Ty::List(Box::new(Ty::Int))),
            Ty::List(Box::new(Ty::Unknown)),
        ),
        (result(result(Ty::Int)), result(Ty::Int)),
        (
            result(Ty::Named("A.Point".into())),
            Ty::Named("B.Point".into()),
        ),
        (Ty::Int, Ty::Int),
    ] {
        assert!(
            !implicit_inner_matches(&expected, &actual),
            "{expected:?} / {actual:?}"
        );
    }
}

#[test]
fn return_lowers_to_the_original_plain_ast_and_inspects_back_exactly() {
    let source = "@effects('fail')\ndef checked(n: int) -> Result[int]:\n    check('positive', n > 0, 'not_positive')\n    return n\n";
    let p = build(source);
    let f = p.get_function("ResultGate.checked").unwrap();
    assert!(
        matches!(&f.body.last().unwrap().kind, ast::StatementKind::Return { value: ast::Expr::Identifier(name) } if name == "n")
    );
    let printed = super::super::print_function(&p, "ResultGate.checked").unwrap();
    assert!(printed.contains("return n") && !printed.contains("return Ok(n)"));
    let again = build(&printed);
    assert!(super::super::ast_equality::function(
        f,
        again.get_function("ResultGate.checked").unwrap()
    ));
    for (input, expected) in [("1", "Ok(1)"), ("0", "Err(not_positive)")] {
        assert_eq!(
            eval::eval_call_with_input(
                &p,
                "ResultGate.checked",
                &parser::parse_single_expr(input).unwrap()
            )
            .unwrap(),
            expected
        );
    }
}

#[test]
fn named_struct_and_list_returns_keep_their_declared_owner_and_type() {
    let p = build("@dataclass\nclass Point:\n    x: int\n\ndef point(n: int) -> Result[Point]:\n    p: Point = Point(x=n)\n    return p\n\ndef numbers(xs: list[int]) -> Result[list[int]]:\n    return xs\n");
    for name in ["ResultGate.point", "ResultGate.numbers"] {
        let printed = super::super::print_function(&p, name).unwrap();
        assert!(printed.contains("return p") || printed.contains("return xs"));
    }
}

#[test]
fn implicit_rule_does_not_relax_assignments_arguments_numeric_types_or_nested_results() {
    for source in [
        "def f() -> Result[float]:\n    return 1\n",
        "def f() -> Result[int]:\n    x: Result[int] = 1\n    return x\n",
        "def callee(x: Result[int]) -> int:\n    return 1\n\ndef f() -> int:\n    return callee(1)\n",
        "def f(inner: Result[int]) -> Result[Result[int]]:\n    return inner\n",
        "def f() -> Result[list[int]]:\n    return []\n",
    ] { assert!(super::super::lower_module(source, "ResultGate").is_err(), "{source}"); }
    let p = build("def nested(inner: Result[int]) -> Result[Result[int]]:\n    return Ok(inner)\n\ndef fail() -> Result[int]:\n    return Err('domain')\n");
    assert!(super::super::print_function(&p, "ResultGate.nested")
        .unwrap()
        .contains("return Ok(inner)"));
}

#[test]
fn nested_result_layers_are_explicit_and_never_filled_in_by_auto_wrap() {
    let p = build("def inner() -> Result[int]:\n    return 3\n\ndef outer() -> Result[Result[int]]:\n    return Ok(inner())\n");
    let f = p.get_function("ResultGate.outer").unwrap();
    let mut env = eval::Env::new_with_shared_interner(&p.shared_interner);
    let value = eval::eval_declared_function(&p, "ResultGate.outer", f, &mut env).unwrap();
    let eval::Value::Ok(outer) = value else {
        panic!("missing outer Result")
    };
    let eval::Value::Ok(inner) = *outer else {
        panic!("missing inner Result")
    };
    assert!(matches!(*inner, eval::Value::Int(3)));
    // Explicit constructors historically have an Unknown payload catalog. Do not broaden
    // payload validation in this slice, but the runtime MUST still reject a missing inner
    // Result layer rather than silently inserting it.
    let malformed = build("def bad() -> Result[Result[int]]:\n    return Ok(1)\n");
    let f = malformed.get_function("ResultGate.bad").unwrap();
    let mut env = eval::Env::new_with_shared_interner(&malformed.shared_interner);
    let error =
        eval::eval_declared_function(&malformed, "ResultGate.bad", f, &mut env).unwrap_err();
    assert!(error.to_string().contains("explicit outer Ok/Err"));
    assert!(error
        .downcast_ref::<eval::result_contract::LanguageFailure>()
        .is_none());
}

#[test]
fn same_source_and_existing_user_result_await_preserve_result_binding() {
    let leaf = "async def leaf(n: int) -> Result[int]:\n    return n\n";
    let consumer = "async def consume(n: int) -> int:\n    value: Result[int] = await leaf(n)\n    match value:\n        case Ok(inner):\n            return inner\n        case Err(error):\n            return 0\n";
    let p = build(&format!("{leaf}\n{consumer}"));
    let f = p.get_function("ResultGate.consume").unwrap();
    let input = parser::parse_single_expr("7").unwrap();
    let value = eval::eval_async_function(
        &p,
        "ResultGate.consume",
        &input,
        crate::coroutine::CoroutineHandle::new_mock(vec![]),
    )
    .unwrap();
    assert!(matches!(value, eval::Value::Int(7)));
    assert!(
        crate::vm::compile_function(f, &p).is_err(),
        "user-await/Result quarantine remains"
    );
    let printed = super::super::print_function(&p, "ResultGate.consume").unwrap();
    assert!(
        printed.contains("value: Result[int] = await ResultGate.leaf(n)"),
        "{printed}"
    );
    let again = super::super::lower_module_with_program(&printed, "ResultGate", &p).unwrap();
    assert_eq!(again.operations.len(), 1);
    let bad = "async def bad(n: int) -> int:\n    value: int = await ResultGate.leaf(n)\n    return value\n";
    let error = super::super::lower_module_with_program(bad, "ResultGate", &p).unwrap_err();
    assert!(error.to_string().contains("type mismatch"), "{error}");
}

#[test]
fn async_and_io_effect_guards_still_apply_to_result_returning_users() {
    for source in [
        "def leaf() -> Result[int]:\n    return 1\n\nasync def f() -> int:\n    value: Result[int] = await leaf()\n    return 0\n",
        "async def leaf() -> Result[int]:\n    return 1\n\ndef f() -> Result[int]:\n    return leaf()\n",
        "@effects('io')\nasync def leaf() -> Result[str]:\n    value: str = await read_file('fixture')\n    return value\n\nasync def f() -> int:\n    value: Result[str] = await leaf()\n    return 0\n",
    ] { assert!(super::super::lower_module(source, "ResultGate").is_err(), "{source}"); }
}

#[test]
fn actual_signature_change_rejects_plain_await_and_keeps_nested_result_binding() {
    let mut p = build("async def leaf() -> int:\n    return 1\n\nasync def caller() -> int:\n    value: int = await leaf()\n    return value\n");
    let plain = parser::parse("+module Caller\n+fn f ()->Int [async]\n  +await value:Int = ResultGate.leaf()\n  +return value\n+end\n").unwrap();
    assert!(super::super::validate_runtime_arguments(&plain, &p).is_ok());
    p.get_function_mut("ResultGate.leaf").unwrap().return_type =
        ast::Type::Result(Box::new(ast::Type::Int));
    let error = super::super::validate_runtime_arguments(&plain, &p).unwrap_err();
    assert!(error.contains("preserves Result"), "{error}");
    assert!(
        super::super::print_function(&p, "ResultGate.caller").is_err(),
        "inspection rechecks the actual callee signature"
    );
    let fixed = "async def caller() -> int:\n    value: Result[int] = await ResultGate.leaf()\n    match value:\n        case Ok(inner):\n            return inner\n        case Err(error):\n            return 0\n";
    let lowered = super::super::lower_module_with_program(fixed, "Caller", &p).unwrap();
    assert!(super::super::validate_runtime_arguments(&lowered.operations, &p).is_ok());
    p.get_function_mut("ResultGate.leaf").unwrap().return_type =
        ast::Type::Result(Box::new(ast::Type::Result(Box::new(ast::Type::Int))));
    let nested = "async def f() -> Result[Result[int]]:\n    value: Result[Result[int]] = await ResultGate.leaf()\n    return value\n";
    let lowered = super::super::lower_module_with_program(nested, "Caller", &p).unwrap();
    assert!(super::super::validate_runtime_arguments(&lowered.operations, &p).is_ok());
    assert!(
        super::super::lower_module_with_program(fixed, "Caller", &p).is_err(),
        "strict lowering also checks the exact nested payload type"
    );
}

#[test]
fn nested_user_result_await_keeps_both_explicit_layers_at_runtime() {
    let p = build("async def child() -> Result[Result[int]]:\n    return Ok(Ok(3))\n\nasync def consume() -> int:\n    result: Result[Result[int]] = await child()\n    match result:\n        case Ok(inner):\n            match inner:\n                case Ok(value):\n                    return value\n                case Err(error):\n                    return 0\n        case Err(error):\n            return 0\n");
    let value = eval::eval_async_function(
        &p,
        "ResultGate.consume",
        &parser::parse_single_expr("{}").unwrap(),
        crate::coroutine::CoroutineHandle::new_mock(vec![]),
    )
    .unwrap();
    assert!(matches!(value, eval::Value::Int(3)));
    assert!(super::super::print_function(&p, "ResultGate.consume").is_ok());
}

#[test]
fn outer_option_requires_explicit_result_wrapper_but_list_option_does_not() {
    let source = "def f(x: Option[int]) -> Result[Option[int]]:\n    return x\n";
    assert!(super::super::lower_module(source, "ResultGate").is_err());
    let p = build("def f(x: Option[int]) -> Result[Option[int]]:\n    return Ok(x)\n\ndef items(x: list[Option[int]]) -> Result[list[Option[int]]]:\n    return x\n");
    let f = p.get_function("ResultGate.f").unwrap();
    for input in [
        eval::Value::Ok(Box::new(eval::Value::Int(7))),
        eval::Value::None,
    ] {
        let mut env = eval::Env::new_with_shared_interner(&p.shared_interner);
        env.set("x", input.clone());
        let actual = eval::eval_declared_function(&p, "ResultGate.f", f, &mut env).unwrap();
        let eval::Value::Ok(inner) = actual else {
            panic!("missing explicit Result")
        };
        match (input, *inner) {
            (eval::Value::Ok(_), eval::Value::Ok(n)) => assert!(matches!(*n, eval::Value::Int(7))),
            (eval::Value::None, eval::Value::None) => {}
            _ => panic!("Option layer changed"),
        }
    }
}

#[test]
fn actual_signature_result_to_scalar_drift_rejects_unchanged_await_operations() {
    let mut p = build("async def leaf() -> Result[int]:\n    return 1\n");
    let caller = super::super::lower_module_with_program("async def caller() -> Result[int]:\n    value: Result[int] = await ResultGate.leaf()\n    return value\n", "Caller", &p).unwrap();
    assert!(super::super::validate_runtime_arguments(&caller.operations, &p).is_ok());
    for ty in [ast::Type::Int, ast::Type::Bool] {
        p.get_function_mut("ResultGate.leaf").unwrap().return_type = ty;
        let before = serde_json::to_value(&p).unwrap();
        let error = super::super::validate_runtime_arguments(&caller.operations, &p).unwrap_err();
        assert!(error.contains("Result binding"), "{error}");
        assert_eq!(before, serde_json::to_value(&p).unwrap());
    }
}

#[test]
fn stored_entry_outer_contract_walk_is_reachable_identity_safe_and_payload_agnostic() {
    let mut p = ast::Program::default();
    let source = "+fn leaf ()->Int [async]\n +return 1\n+end\n+module A\n+fn leaf ()->Result<Int> [async]\n +return Ok(1)\n+end\n+fn a ()->Int [async]\n +await x:Int = leaf()\n +return A.b()\n+end\n+fn b ()->Int [async]\n +return A.a()\n+end\n+fn unrelated ()->Int [async]\n +await x:Result<Int> = leaf()\n +return 0\n+end\n";
    for op in parser::parse(source).unwrap() {
        validator::apply_and_validate(&mut p, &op).unwrap();
    }
    p.rebuild_function_index();
    let check = super::super::validate_entry_await_contract;
    assert!(
        check(&p, "A.a").is_ok(),
        "bare leaf must resolve top-level; unrelated mismatch ignored; cycle terminates"
    );
    assert!(check(&p, "A.unrelated").is_err());
    // The same-name qualified function is distinct and resolves to Result.
    let ast::StatementKind::Await { call, .. } =
        &mut p.get_function_mut("A.a").unwrap().body[0].kind
    else {
        panic!()
    };
    call.callee = "A.leaf".into();
    let error = check(&p, "A.b").unwrap_err();
    assert!(!error.contains("A.leaf") && !error.contains("A.a"));
    let ast::StatementKind::Await { ty, .. } = &mut p.get_function_mut("A.a").unwrap().body[0].kind
    else {
        panic!()
    };
    *ty = ast::Type::Result(Box::new(ast::Type::Result(Box::new(ast::Type::Bool))));
    assert!(
        check(&p, "A.b").is_ok(),
        "same outer Result; payload validation is not this preflight"
    );
    p.get_function_mut("A.leaf").unwrap().return_type = ast::Type::Bool;
    assert!(
        check(&p, "A.b").is_err(),
        "transitive reverse drift checked fresh"
    );
}
