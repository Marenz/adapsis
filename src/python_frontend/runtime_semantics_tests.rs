//! #90 tier-equivalence evidence. Tree short-circuit is normative; these tests
//! assert that the VM and JIT now AGREE with it (they previously characterized a
//! disagreement, which is why several names still say "audit").
use crate::{ast, compiler, coroutine, eval, parser, validator, vm};
use eval::{Env, Value};
use std::cell::RefCell;

fn program(source: &str) -> ast::Program {
    let mut program = ast::Program::default();
    for op in parser::parse(source).unwrap() {
        validator::apply_and_validate(&mut program, &op).unwrap();
    }
    program.rebuild_function_index();
    program
}

fn tree(program: &ast::Program, target: &str, args: Vec<Value>) -> anyhow::Result<Value> {
    let function = program.get_function(target).unwrap();
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    for (param, arg) in function.params.iter().zip(args) {
        env.set(&param.name, arg);
    }
    eval::eval_function_body_named(program, target, &function.body, &mut env)
}

fn bytecode(program: &ast::Program, target: &str, args: Vec<Value>) -> anyhow::Result<Value> {
    let compiled = vm::compile_function(program.get_function(target).unwrap(), program)?;
    vm::execute_with_io(&compiled, args, program, &|_, _| {
        anyhow::bail!("unexpected IO")
    })
}

fn division_program() -> ast::Program {
    program(
        "+fn rhs (n:Int)->Bool\n  +return 1/n>0\n+end\n+fn conjunction (flag:Bool,n:Int)->Int\n  +if flag AND rhs(n)\n    +return 1\n  +else\n    +return 0\n  +end\n+end\n+fn disjunction (flag:Bool,n:Int)->Int\n  +if flag OR rhs(n)\n    +return 1\n  +else\n    +return 0\n  +end\n+end\n",
    )
}

#[test]
fn python_boolean_rhs_error_is_skipped_on_all_tiers() {
    let program = division_program();
    for (target, flag, expected) in [("conjunction", false, 0), ("disjunction", true, 1)] {
        let args = vec![Value::Bool(flag), Value::Int(0)];
        assert!(
            matches!(tree(&program, target, args.clone()).unwrap(), Value::Int(n) if n == expected)
        );
        // #90: the VM now short-circuits too, so no division-by-zero error.
        assert!(
            matches!(bytecode(&program, target, args).unwrap(), Value::Int(n) if n == expected)
        );
    }
    let mut jit = compiler::compile(&program).unwrap();
    for (target, flag, expected) in [
        ("conjunction", false, 0),
        ("conjunction", true, 1),
        ("disjunction", false, 1),
        ("disjunction", true, 1),
    ] {
        let args = vec![Value::Bool(flag), Value::Int(1)];
        assert!(
            matches!(tree(&program, target, args.clone()).unwrap(), Value::Int(n) if n == expected)
        );
        assert!(
            matches!(bytecode(&program, target, args).unwrap(), Value::Int(n) if n == expected)
        );
        assert_eq!(jit.call_i64(target, &[flag as i64, 1]).unwrap(), expected);
    }
}

/// #90: the JIT now short-circuits, so the skipped rhs division is never
/// executed and must NOT trap. Previously this ran in a subprocess because the
/// eager JIT trapped; the assertion is now in-process and positive.
#[test]
fn python_jit_skips_short_circuited_rhs_error() {
    let program = division_program();
    let mut jit = compiler::compile(&program).unwrap();
    for (target, flag, expected) in [("conjunction", false, 0), ("disjunction", true, 1)] {
        assert_eq!(
            jit.call_i64(target, &[flag as i64, 0]).unwrap(),
            expected,
            "{target} must skip the dividing rhs"
        );
    }
}

#[test]
fn python_boolean_io_count_order_matches_short_circuit_across_tiers() {
    for (operator, left, expected, tree_count) in [
        ("AND", false, false, 1),
        ("OR", true, true, 1),
        ("AND", true, true, 2),
        ("OR", false, true, 2),
    ] {
        let program = program(&format!(
            "+fn record (path:String)->Bool [io,async]\n  +await value:Bool = file_exists(path)\n  +return value\n+end\n+fn probe ()->Bool [io,async]\n  +return record(\"left\") {operator} record(\"right\")\n+end\n"
        ));
        let (sender, mut receiver) = tokio::sync::mpsc::channel(2);
        let responder = std::thread::spawn(move || {
            let mut calls = Vec::new();
            while let Some(request) = receiver.blocking_recv() {
                let coroutine::IoRequest::FileExists { path, reply } = request else {
                    panic!("only fixture file_exists is allowed");
                };
                let value = if path == "left" { left } else { true };
                calls.push(path);
                reply.send(Ok(value)).unwrap();
            }
            calls
        });
        let mut env = Env::new_with_shared_interner(&program.shared_interner);
        env.set(
            "__coroutine_handle",
            Value::CoroutineHandle(coroutine::CoroutineHandle::new(sender)),
        );
        let result = eval::eval_function_body_named(
            &program,
            "probe",
            &program.get_function("probe").unwrap().body,
            &mut env,
        )
        .unwrap();
        drop(env);
        let calls = responder.join().unwrap();
        assert!(matches!(result, Value::Bool(b) if b == expected));
        assert_eq!(calls, ["left", "right"][..tree_count]);
        let compiled =
            vm::compile_function(program.get_function("probe").unwrap(), &program).unwrap();
        let calls = RefCell::new(Vec::new());
        let result = vm::execute_with_io(&compiled, vec![], &program, &|name, args| {
            assert_eq!(name, "file_exists");
            let Value::String(path) = &args[0] else {
                panic!("expected fixture path")
            };
            calls.borrow_mut().push(path.to_string());
            Ok(Value::Bool(if path.as_str() == "left" {
                left
            } else {
                true
            }))
        })
        .unwrap();
        assert!(matches!(result, Value::Bool(b) if b == expected));
        // #90: the VM must consume exactly the same IoRequests, in the same
        // order, as the tree walker — a short-circuited rhs is never requested.
        assert_eq!(*calls.borrow(), ["left", "right"][..tree_count]);
    }
}

#[test]
fn python_empty_collection_and_result_truthiness_is_shared_across_tiers() {
    let program = program("+fn probe (value:Bool)->Bool\n  +return value AND true\n+end\n");
    // The raw execution APIs are deliberately supplied each Value category to
    // inspect their truthiness independently of frontend admission.
    for (value, truth) in [
        (Value::list(vec![]), true),
        (Value::map(vec![]), true),
        (Value::set(vec![]), true),
        (Value::Ok(Box::new(Value::Int(0))), true),
        (Value::Err("problem".into()), false),
        (Value::None, false),
        (Value::string(""), false),
        (Value::string("x"), true),
    ] {
        assert!(
            matches!(tree(&program, "probe", vec![value.clone()]).unwrap(), Value::Bool(b) if b == truth)
        );
        // #90: the VM now consumes the same truthiness definition.
        assert!(
            matches!(bytecode(&program, "probe", vec![value]).unwrap(), Value::Bool(b) if b == truth)
        );
    }
}

#[test]
fn python_audit_diag_fixture_retains_invalid_shared_reference_diagnostic() {
    let program = program(include_str!(
        "../../tests/fixtures/python-parity/library/Diag.ax"
    ));
    assert!(program.modules[0].shared_vars.is_empty());
    let error = tree(&program, "Diag.run", vec![]).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("undefined variable `telegram_token`"),
        "{error}"
    );
    let error = super::print_function(&program, "Diag.run").unwrap_err();
    assert!(
        error
            .message
            .contains("unbound or out-of-block name telegram_token"),
        "{error}"
    );
}

#[test]
fn python_audit_transcribe_voice_fixture_has_out_of_arm_error_reference() {
    let program = program(include_str!(
        "../../tests/fixtures/python-parity/library/TelegramBot.ax"
    ));
    let function = program
        .get_function("TelegramBot.transcribe_voice")
        .unwrap();
    let ast::StatementKind::Match { arms, .. } = &function.body[1].kind else {
        panic!("expected outer HTTP match");
    };
    // The source's second Err is indented under the inner match, not the outer
    // HTTP match. Its following return is outside that inner arm's scope.
    assert_eq!(
        arms.iter().map(|a| a.variant.as_str()).collect::<Vec<_>>(),
        ["Ok"]
    );
    let outer_body = &arms[0].body;
    let nested = outer_body
        .iter()
        .find_map(|stmt| match &stmt.kind {
            ast::StatementKind::Match { arms, .. } => Some(arms),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        nested
            .iter()
            .map(|a| a.variant.as_str())
            .collect::<Vec<_>>(),
        ["Ok", "Err", "Err"]
    );
    assert!(nested[2].body.is_empty());
    let ast::StatementKind::Return {
        value: ast::Expr::Call(call),
    } = &outer_body.last().unwrap().kind
    else {
        panic!("expected unbound tail concat");
    };
    assert_eq!(call.callee, "concat");
    assert!(
        call.args
            .iter()
            .any(|a| matches!(a, ast::Expr::Identifier(name) if name == "e"))
    );
    let error = super::print_function(&program, "TelegramBot.transcribe_voice").unwrap_err();
    assert!(
        error.message.contains("unbound or out-of-block name e")
            || error.message.contains(
                "unqualified user call transcription_upload_response has no exact Python roundtrip"
            ),
        "{error}"
    );
}

#[test]
fn python_result_match_adapsis_and_python_agree_across_tree_and_vm() {
    // #93 regression: both source origins lower to the same persisted AST,
    // and the normal routing path must agree with direct tree/VM evaluation.
    for (ax_result, py_result, expected) in [
        ("Ok(5)", "Ok(5)", 5),
        ("Err(\"fixture\")", "Err('fixture')", 0),
    ] {
        let ax = program(&format!(
            "+module ResultMatchProbe\n+fn wrap ()->Result<Int>\n +return {ax_result}\n+end\n+fn read_value ()->Int\n +match ResultMatchProbe.wrap()\n +case Ok(value)\n  +return value\n +case Err(error)\n  +return 0\n +end\n+end\n"
        ));
        let source = format!(
            "def wrap() -> Result[int]:\n    return {py_result}\ndef read_value() -> int:\n    match wrap():\n        case Ok(value):\n            return value\n        case Err(error):\n            return 0\n"
        );
        let lowered = super::lower_module(&source, "ResultMatchProbe").unwrap();
        let mut python = ast::Program::default();
        for op in lowered.operations {
            validator::apply_and_validate(&mut python, &op).unwrap();
        }
        python.rebuild_function_index();
        assert!(super::ast_equality::module(
            &ax.modules[0],
            &python.modules[0]
        ));
        for (origin, program) in [("adapsis", &ax), ("python", &python)] {
            let subject = bytecode(program, "ResultMatchProbe.wrap", vec![]).unwrap();
            match (&subject, expected) {
                (Value::Ok(inner), 5) => assert!(matches!(inner.as_ref(), Value::Int(5))),
                (Value::Err(message), 0) => assert_eq!(message, "fixture"),
                _ => panic!("unexpected Result subject {subject}"),
            }
            assert!(
                matches!(tree(program, "ResultMatchProbe.read_value", vec![]).unwrap(), Value::Int(n) if n == expected)
            );
            assert!(
                matches!(bytecode(program, "ResultMatchProbe.read_value", vec![]).unwrap(), Value::Int(n) if n == expected)
            );
            let input = parser::parse_test_input(1, "").unwrap();
            let (actual, jitted) = eval::eval_compiled_or_interpreted_cached(
                program,
                "ResultMatchProbe.read_value",
                &input,
                None,
                0,
            )
            .unwrap();
            assert_eq!(actual, expected.to_string());
            println!(
                "RESULT_MATCH_REGRESSION origin={origin} subject={subject} tree={expected} vm={expected} routed={actual} jitted={jitted}"
            );
        }
    }
}
