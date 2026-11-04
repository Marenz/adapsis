//! C-core activation matrix. The prior plumbing characterizations are deliberately
//! replaced by structural normalization, typed catching and real trace execution tests.
use super::*;
use crate::{coroutine::CoroutineHandle, eval, parser, session, validator};

fn program(source: &str) -> ast::Program {
    let mut program = ast::Program::default();
    for operation in parser::parse(source).unwrap() {
        validator::apply_and_validate(&mut program, &operation).unwrap();
    }
    program.rebuild_function_index();
    crate::intern::set_display_interner(&program.shared_interner);
    program
}

#[derive(Default)]
struct Capture {
    functions: Vec<(String, ast::Type)>,
    values: Vec<Value>,
    errors: Vec<Option<eval::result_contract::LanguageFailure>>,
}

impl OutcomeObserver for Capture {
    fn observe(&mut self, function: &ast::FunctionDecl, outcome: &Result<Value>) {
        self.functions
            .push((function.name.clone(), function.return_type.clone()));
        match outcome {
            Ok(value) => self.values.push(value.clone()),
            Err(error) => self.errors.push(
                error
                    .downcast_ref::<eval::result_contract::LanguageFailure>()
                    .cloned(),
            ),
        }
    }
}

fn declared(program: &ast::Program, name: &str) -> Result<Value> {
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    eval_declared_function(program, name, program.get_function(name).unwrap(), &mut env)
}

fn assert_ok_int(value: Value, expected: i64) {
    assert!(matches!(value, Value::Ok(inner) if matches!(*inner, Value::Int(n) if n == expected)));
}

#[test]
fn actual_declaration_not_the_label_selects_the_body_and_observed_contract() {
    let program = program("+module InvocationOwner\n+fn label ()->Result<Int>\n  +return 9\n+end\n+fn actual ()->String\n  +return \"actual\"\n+end\n");
    let function = program.get_function("InvocationOwner.actual").unwrap();
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    let mut capture = Capture::default();
    let value = invoke(
        &program,
        "InvocationOwner.label",
        function,
        Body::Declared(&mut env),
        Some(&mut capture),
    )
    .unwrap();
    assert!(matches!(value, Value::String(text) if text.as_str() == "actual"));
    assert_eq!(capture.functions, [("actual".into(), ast::Type::String)]);
    assert_eq!(capture.values.len(), 1);
}

#[test]
fn raw_lifecycle_body_never_acquires_a_contract_from_its_label() {
    let program = program("+module InvocationLifecycle\n+fn startup ()->Result<Int>\n  +return 99\n+end\n+fn body ()->Int\n  +return 3\n+end\n");
    let body = &program
        .get_function("InvocationLifecycle.body")
        .unwrap()
        .body;
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    let value =
        eval::eval_function_body_named(&program, "InvocationLifecycle.startup", body, &mut env)
            .unwrap();
    assert!(
        matches!(value, Value::Int(3)),
        "raw body is not the similarly named declaration"
    );
}

#[test]
fn cli_unframed_declared_entry_keeps_its_existing_frame_behavior() {
    let program = program("+module InvocationCliScope\n+shared count:Int = 7\n+fn read ()->Int\n  +return count\n+end\n");
    let function = program.get_function("InvocationCliScope.read").unwrap();
    assert!(eval::FN_NAME_STACK.with(|s| s.borrow().is_empty()));
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    env.populate_shared_from_program(&program);
    let raw = eval::eval_function_body_pub(&program, &function.body, &mut env).unwrap_err();
    let unframed =
        eval_declared_function_unframed(&program, "InvocationCliScope.read", function, &mut env)
            .unwrap_err();
    assert_eq!(
        unframed.to_string(),
        raw.to_string(),
        "do not implicitly fix CLI scope during plumbing"
    );
    let named =
        eval_declared_function(&program, "InvocationCliScope.read", function, &mut env).unwrap();
    assert!(matches!(named, Value::Int(7)));
    assert!(eval::FN_NAME_STACK.with(|s| s.borrow().is_empty()));
}

#[test]
fn observer_sees_one_normalized_outcome_and_no_nested_callee_body_or_events() {
    let program = program("+module PrivateInvocation\n+fn hidden ()->Int\n  +return 14\n+end\n+module PublicInvocation\n+fn outer ()->Result<Int>\n  +return PrivateInvocation.hidden()\n+end\n");
    let function = program.get_function("PublicInvocation.outer").unwrap();
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    let mut capture = Capture::default();
    let before = eval::FN_NAME_STACK.with(|stack| stack.borrow().clone());
    let value = invoke(
        &program,
        "PublicInvocation.outer",
        function,
        Body::Declared(&mut env),
        Some(&mut capture),
    )
    .unwrap();
    assert_ok_int(value, 14);
    assert_eq!(
        capture.functions,
        [("outer".into(), ast::Type::Result(Box::new(ast::Type::Int)))]
    );
    assert_eq!(capture.values.len(), 1);
    assert_ok_int(capture.values.remove(0), 14);
    assert_eq!(
        eval::FN_NAME_STACK.with(|stack| stack.borrow().clone()),
        before
    );
}

#[test]
fn declared_result_catches_typed_failure_but_never_operational_failure() {
    let program = program("+fn checked ()->Result<Int> [fail]\n  +check ready false ~not_ready\n  +return 1\n+end\n+fn broken ()->Result<Int>\n  +return 1 / 0\n+end\n");
    for (name, language) in [("checked", true), ("broken", false)] {
        let function = program.get_function(name).unwrap();
        let mut env = Env::new_with_shared_interner(&program.shared_interner);
        let mut capture = Capture::default();
        let before = eval::FN_NAME_STACK.with(|stack| stack.borrow().clone());
        let outcome = invoke(
            &program,
            name,
            function,
            Body::Declared(&mut env),
            Some(&mut capture),
        );
        if language {
            assert!(matches!(outcome.unwrap(), Value::Err(label) if label == "not_ready"));
            assert!(capture.errors.is_empty());
            assert!(
                matches!(capture.values.as_slice(), [Value::Err(label)] if label == "not_ready")
            );
        } else {
            let error = outcome.unwrap_err();
            assert!(error
                .downcast_ref::<eval::result_contract::LanguageFailure>()
                .is_none());
            assert_eq!(capture.errors, [None]);
            assert!(capture.values.is_empty());
        }
        assert_eq!(
            eval::FN_NAME_STACK.with(|stack| stack.borrow().clone()),
            before
        );
    }
}

#[test]
fn named_parser_nested_and_async_invocations_agree_and_user_await_preserves_result() {
    let program = program("+fn plain ()->Result<Int>\n  +return 7\n+end\n+fn wrapped ()->Result<Int>\n  +return Ok(7)\n+end\n+fn ordinary ()->Result<Int>\n  +return plain()\n+end\n+fn awaited ()->Int [async]\n  +await value:Result<Int> = wrapped()\n  +match value\n  +case Ok(inner)\n    +return inner\n  +case Err(error)\n    +return 0\n  +end\n+end\n+module InvocationParser\n+fn plain ()->Result<Int>\n  +return 7\n+end\n");
    assert_ok_int(declared(&program, "plain").unwrap(), 7);
    assert_ok_int(declared(&program, "ordinary").unwrap(), 7);
    for expression in [
        "plain",
        "plain()",
        "InvocationParser.plain",
        "InvocationParser.plain()",
    ] {
        let input = parser::parse_single_expr(expression).unwrap();
        assert_ok_int(
            eval::eval_parser_expr_with_program(&input, &program).unwrap(),
            7,
        );
    }
    let input = parser::parse_single_expr("{}").unwrap();
    assert_eq!(
        eval::eval_call_with_input(&program, "plain", &input).unwrap(),
        "Ok(7)"
    );
    assert_ok_int(
        eval::eval_async_function(&program, "plain", &input, CoroutineHandle::new_mock(vec![]))
            .unwrap(),
        7,
    );
    assert!(
        matches!(
            eval::eval_async_function(
                &program,
                "awaited",
                &input,
                CoroutineHandle::new_mock(vec![])
            )
            .unwrap(),
            Value::Int(7)
        ),
        "the awaited value must reach the Ok arm, not arrive as an unwrapped Int"
    );
}

#[test]
fn qualified_stub_lookup_wins_and_stubs_normalize_without_extra_io_authority() {
    let program =
        program("+module InvocationStub\n+fn target ()->Result<Int>\n  +return 7\n+end\n");
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    let stubs = vec![
        session::FunctionStub {
            function_name: "InvocationStub.target".into(),
            patterns: vec![],
            response_expr: "11".into(),
        },
        session::FunctionStub {
            function_name: "target".into(),
            patterns: vec![],
            response_expr: "22".into(),
        },
    ];
    env.set(
        "__coroutine_handle",
        Value::CoroutineHandle(CoroutineHandle::new_mock_with_stubs(vec![], stubs)),
    );
    assert_ok_int(
        eval::eval_builtin_or_user(&program, "target", vec![], &mut env).unwrap(),
        11,
    );
    let function = program.get_function("InvocationStub.target").unwrap();
    let mut capture = Capture::default();
    let value = invoke(
        &program,
        "target",
        function,
        Body::Stub("Ok(7)"),
        Some(&mut capture),
    )
    .unwrap();
    assert!(matches!(value, Value::Ok(inner) if matches!(*inner, Value::Int(7))));
    assert_eq!(
        capture.functions[0].1,
        ast::Type::Result(Box::new(ast::Type::Int))
    );
    let error = invoke(&program, "target", function, Body::Stub("("), None).unwrap_err();
    assert!(error
        .to_string()
        .starts_with("stub expression parse error:"));
    assert!(
        invoke(
            &program,
            "target",
            function,
            Body::Stub("read_file(\"must-not-run\")"),
            None
        )
        .is_err(),
        "stub program-only expression context must not gain a coroutine handle"
    );
}

#[test]
fn observer_does_not_execute_body_twice_or_gain_a_nested_io_handle() {
    let program = program("+fn once ()->Bool [io,async]\n  +await value:Bool = file_exists(\"once\")\n  +return value\n+end\n");
    let (sender, mut receiver) = tokio::sync::mpsc::channel(4);
    let responder = std::thread::spawn(move || {
        let mut calls = 0;
        while let Some(request) = receiver.blocking_recv() {
            let crate::coroutine::IoRequest::FileExists { path, reply } = request else {
                panic!("only synthetic IO")
            };
            assert_eq!(path, "once");
            calls += 1;
            reply.send(Ok(true)).unwrap();
        }
        calls
    });
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    env.set(
        "__coroutine_handle",
        Value::CoroutineHandle(CoroutineHandle::new(sender)),
    );
    let mut capture = Capture::default();
    let result = invoke(
        &program,
        "once",
        program.get_function("once").unwrap(),
        Body::Declared(&mut env),
        Some(&mut capture),
    );
    assert!(matches!(result.unwrap(), Value::Bool(true)));
    assert!(matches!(capture.values.as_slice(), [Value::Bool(true)]));
    drop(env);
    assert_eq!(responder.join().unwrap(), 1);
}

#[test]
fn existing_call_depth_guard_is_not_bypassed_by_the_wrapper() {
    let program = program("+fn recurse ()->Int\n  +return recurse()\n+end\n");
    let before = eval::FN_NAME_STACK.with(|stack| stack.borrow().clone());
    // Exercise the boundary deterministically without depending on the test thread's
    // native stack size or executing hundreds of recursive frames.
    let old_depth = eval::CALL_DEPTH.with(|depth| depth.replace(eval::MAX_CALL_DEPTH));
    let outcome = declared(&program, "recurse");
    eval::CALL_DEPTH.with(|depth| depth.set(old_depth));
    let error = outcome.unwrap_err();
    assert!(error.to_string().contains("maximum call depth"));
    assert_eq!(
        eval::FN_NAME_STACK.with(|stack| stack.borrow().clone()),
        before
    );
}

#[test]
fn trace_executes_branches_and_checks_instead_of_skipping_them() {
    let program = program("+fn branch ()->Int\n  +if true\n    +return 7\n  +end\n  +return 9\n+end\n+fn checked_branch ()->Int [fail]\n  +if true\n    +check ready false ~not_ready\n  +end\n  +return 9\n+end\n");
    assert!(matches!(
        declared(&program, "branch").unwrap(),
        Value::Int(7)
    ));
    assert!(declared(&program, "checked_branch").is_err());
    for name in ["branch", "checked_branch"] {
        let steps = eval::trace_function(&program, name, &parser::parse_single_expr("{}").unwrap())
            .unwrap();
        assert!(!steps.iter().any(|s| s.result == "..." || s.result == "9"));
        let terminal = steps.last().unwrap();
        if name == "branch" {
            assert!(matches!(terminal.status, eval::TraceStatus::Return));
            assert_eq!(terminal.result, "7");
        } else {
            assert!(matches!(terminal.status, eval::TraceStatus::Fail));
            assert!(terminal.result.contains("not_ready"));
        }
    }
    // Deliberate correction of the plumbing checkpoint's partial-trace characterization.
}
