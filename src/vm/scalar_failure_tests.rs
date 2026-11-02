//! #97: real normal-dispatch IO counts, with a graph that is VM-admitted (no user
//! await, field access, Result signatures or capture bindings to hide the failure).
use crate::{
    ast,
    coroutine::{CoroutineHandle, IoRequest},
    eval, parser, validator, vm,
};

pub(super) fn program(source: &str) -> ast::Program {
    let mut program = ast::Program::default();
    for operation in parser::parse(source).unwrap() {
        validator::apply_and_validate(&mut program, &operation).unwrap();
    }
    program.rebuild_function_index();
    program
}

fn assert_check(error: &anyhow::Error, label: &str) {
    assert_eq!(
        error.downcast_ref::<eval::result_contract::LanguageFailure>(),
        Some(&eval::result_contract::LanguageFailure::Check {
            label: label.into()
        })
    );
}

#[test]
fn scalar_failure_stops_later_arguments_operands_loops_and_recursive_calls() {
    let prefix = "+fn fail ()->Int [fail]\n  +check required false ~stop\n  +return 3\n+end\n+fn recurse (n:Int)->Int [fail]\n  +if n == 0\n    +return fail()\n  +end\n  +return recurse(n - 1)\n+end\n+fn pair (a:Int,b:Int)->Int\n  +return a + b\n+end\n";
    for body in [
        "+return fail()",
        "+return 17 + fail()",
        "+return fail() + 17",
        "+call ignored:Int = fail()\n  +return 0",
        "+return pair(fail(), 1 / 0)",
        "+return pair(17, fail())",
        "+if fail() > 0\n    +return 1\n  +end\n  +return 0",
        "+while fail() > 0\n    +return 1\n  +end\n  +return 0",
        "+check outer fail() > 0 ~outer_failed\n  +return 0",
        "+call value:Int = recurse(5)\n  +return value",
    ] {
        let program = program(&format!(
            "{prefix}+fn root ()->Int [fail]\n  {body}\n+end\n"
        ));
        let root = program.get_function("root").unwrap();
        let direct = vm::attempt(root, vec![], &program, false, &|_, _| {
            anyhow::bail!("no IO")
        });
        let vm::VmAttempt::Executed(Err(error)) = direct else {
            panic!("must execute VM: {body}")
        };
        assert_check(&error, "stop");
        let mut env = eval::Env::new_with_shared_interner(&program.shared_interner);
        let error =
            eval::eval_declared_function(&program, "root", root, &mut env).unwrap_err();
        assert_check(&error, "stop");
        let (text, jit) = eval::eval_compiled_or_interpreted(
            &program,
            "root",
            &parser::parse_single_expr("{}").unwrap(),
        )
        .unwrap();
        assert!(!jit);
        assert_eq!(text, "Err(stop)", "legacy sync formatting without rerun");
    }
}

#[test]
fn failure_before_io_suppresses_all_requests_and_success_runs_once() {
    for (condition, expected_requests) in [("false", vec![]), ("true", vec!["after"])] {
        let program = program(&format!("+fn check_it ()->Int [fail]\n  +check ready {condition} ~not_ready\n  +return 7\n+end\n+fn root ()->Int [io,async,fail]\n  +let result:Int = check_it()\n  +await after:Bool = file_exists(\"after\")\n  +return result\n+end\n"));
        assert!(vm::compile_function(program.get_function("root").unwrap(), &program).is_ok());
        let (result, requests) = run_async(&program, "root");
        assert_eq!(requests, expected_requests);
        if condition == "false" {
            assert_check(&result.unwrap_err(), "not_ready");
        } else {
            assert!(matches!(result.unwrap(), eval::Value::Int(7)));
        }
    }
}

#[test]
fn operational_execution_failure_is_not_language_failure_and_never_replays_io() {
    let program = program("+fn divide ()->Int\n  +return 1 / 0\n+end\n+fn root ()->Int [io,async]\n  +await before:Bool = file_exists(\"before\")\n  +let n:Int = divide()\n  +await after:Bool = file_exists(\"after\")\n  +return n\n+end\n");
    assert!(vm::compile_function(program.get_function("root").unwrap(), &program).is_ok());
    let (result, requests) = run_async(&program, "root");
    assert_eq!(requests, ["before"]);
    let error = result.unwrap_err();
    assert!(error
        .downcast_ref::<eval::result_contract::LanguageFailure>()
        .is_none());
    assert!(error.to_string().contains("division by zero"));
}

#[test]
fn operational_io_failure_is_executed_not_an_admission_refusal() {
    let program = program("+fn root ()->Bool [io,async]\n  +await result:Bool = file_exists(\"once\")\n  +await after:Bool = file_exists(\"after\")\n  +return result\n+end\n");
    let calls = std::cell::Cell::new(0);
    let result = vm::attempt(
        program.get_function("root").unwrap(),
        vec![],
        &program,
        true,
        &|_, _| {
            calls.set(calls.get() + 1);
            Err(std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied").into())
        },
    );
    let vm::VmAttempt::Executed(Err(error)) = result else {
        panic!("must be executed failure")
    };
    assert_eq!(calls.get(), 1);
    assert_eq!(
        error.downcast_ref::<std::io::Error>().unwrap().kind(),
        std::io::ErrorKind::PermissionDenied
    );
    assert!(error
        .downcast_ref::<eval::result_contract::LanguageFailure>()
        .is_none());
}

#[test]
fn result_capture_graphs_are_refused_and_normal_tree_catching_remains_compatible() {
    let program = program("+fn fail ()->Int [fail]\n  +check ready false ~caught\n  +return 1\n+end\n+fn capture ()->Int [io,async]\n  +call result:Result<Int> = fail()\n  +match result\n  +case Ok(value)\n    +return value\n  +case Err(error)\n    +await recovered:Bool = file_exists(\"recovery\")\n    +return 8\n  +end\n+end\n");
    let root = program.get_function("capture").unwrap();
    assert!(vm::compile_function(root, &program).is_err());
    assert!(matches!(
        vm::attempt(root, vec![], &program, true, &|_, _| panic!(
            "preparation cannot do IO"
        )),
        vm::VmAttempt::NotAdmitted(_)
    ));
    let (result, requests) = run_async(&program, "capture");
    assert!(matches!(result.unwrap(), eval::Value::Int(8)));
    assert_eq!(requests, ["recovery"]);
    // No half-activation of C: legacy tree Result success still wraps at its existing
    // entry boundary, and explicit Result-returning source remains quarantined.
    let program = self::program("+fn result ()->Result<Int>\n  +return 4\n+end\n");
    assert!(vm::compile_function(program.get_function("result").unwrap(), &program).is_err());
    let (value, jit) = eval::eval_compiled_or_interpreted(
        &program,
        "result",
        &parser::parse_single_expr("{}").unwrap(),
    )
    .unwrap();
    assert_eq!(value, "Ok(4)");
    assert!(!jit);
}

#[test]
fn quarantined_legacy_capture_keeps_operational_catching_until_combined_migration() {
    let program = program("+fn broken ()->Int\n  +return 1 / 0\n+end\n+fn root ()->Int\n  +call result:Result<Int> = broken()\n  +match result\n  +case Ok(value)\n    +return value\n  +case Err(error)\n    +return 8\n  +end\n+end\n");
    assert!(vm::compile_function(program.get_function("root").unwrap(), &program).is_err());
    let (value, jit) = eval::eval_compiled_or_interpreted(
        &program,
        "root",
        &parser::parse_single_expr("{}").unwrap(),
    )
    .unwrap();
    assert_eq!(
        value, "8",
        "do not half-activate the new typed-only catch policy in legacy tree paths"
    );
    assert!(!jit);
}

#[test]
fn production_result_quarantine_covers_signatures_bindings_and_reachable_constructors() {
    for source in [
        "+fn root ()->Result<Int>\n  +return 1\n+end\n",
        "+fn root (r:Result<Int>)->Int\n  +return 1\n+end\n",
        "+fn root ()->Int\n  +let r:Result<Int> = Ok(1)\n  +return 1\n+end\n",
        "+fn leaf ()->Result<Int>\n  +return 1\n+end\n+fn root ()->Int\n  +return leaf()\n+end\n",
        "+fn leaf ()->Int\n  +return 1\n+end\n+fn root ()->Int\n  +call r:Result<Int> = leaf()\n  +return 1\n+end\n",
        "+fn root ()->Int\n  +return Err(\"explicit result\")\n+end\n",
        "+type Envelope = value:Result<Int>\n+fn root (e:Envelope)->Int\n  +return 1\n+end\n",
    ] {
        let program = program(source);
        let root = program.get_function("root").unwrap();
        let error = vm::compile_function(root, &program).unwrap_err();
        assert!(error.to_string().contains(super::scalar_admission::RESULT_TREE), "{error}");
        assert!(matches!(vm::attempt(root, vec![], &program, true, &|_, _| panic!("no IO")), vm::VmAttempt::NotAdmitted(_)));
    }
}

#[test]
fn unsupported_nested_callee_selects_tree_before_first_io() {
    let program = program("+module VmScalarShared\n+shared count:Int = 9\n+fn leaf ()->Int\n  +return count\n+end\n+fn middle ()->Int\n  +return VmScalarShared.leaf()\n+end\n+fn root ()->Int [io,async]\n  +await before:Bool = file_exists(\"before\")\n  +let n:Int = VmScalarShared.middle()\n  +await after:Bool = file_exists(\"after\")\n  +return n\n+end\n");
    let root = program.get_function("VmScalarShared.root").unwrap();
    assert!(
        vm::compile_function(root, &program).is_ok(),
        "root emission itself supports all instructions"
    );
    assert!(matches!(
        vm::attempt(root, vec![], &program, true, &|_, _| panic!(
            "no IO during preparation"
        )),
        vm::VmAttempt::NotAdmitted(_)
    ));
    let (result, requests) = run_async(&program, "VmScalarShared.root");
    assert!(matches!(result.unwrap(), eval::Value::Int(9)));
    assert_eq!(requests, ["before", "after"]);
}

#[test]
fn builtin_io_call_after_an_await_selects_tree_before_first_io() {
    let program = program("+fn root ()->Bool [io,async]\n  +await before:Bool = file_exists(\"before\")\n  +return file_exists(\"direct\")\n+end\n");
    let root = program.get_function("root").unwrap();
    assert!(vm::compile_function(root, &program).is_err());
    assert!(matches!(
        vm::attempt(root, vec![], &program, true, &|_, _| panic!(
            "no IO during preparation"
        )),
        vm::VmAttempt::NotAdmitted(_)
    ));
    let (result, requests) = run_async(&program, "root");
    assert!(matches!(result.unwrap(), eval::Value::Bool(true)));
    assert_eq!(requests, ["before", "direct"]);
}

#[test]
fn synchronous_entry_refuses_reachable_await_before_execution() {
    let program = program("+fn leaf ()->Bool [io,async]\n  +await value:Bool = file_exists(\"never\")\n  +return value\n+end\n+fn root ()->Bool [io,async]\n  +return leaf()\n+end\n");
    let root = program.get_function("root").unwrap();
    assert!(matches!(
        vm::attempt(root, vec![], &program, false, &|_, _| panic!(
            "sync preparation must not do IO"
        )),
        vm::VmAttempt::NotAdmitted(_)
    ));
}

#[test]
fn an_executed_failure_before_io_does_not_retry_via_a_tree_stub() {
    let program = program("+fn leaf ()->Int\n  +return 1 / 0\n+end\n+fn root ()->Int [async]\n  +return leaf()\n+end\n");
    let handle = CoroutineHandle::new_mock_with_stubs(
        vec![],
        vec![crate::session::FunctionStub {
            function_name: "leaf".into(),
            patterns: vec![],
            response_expr: "99".into(),
        }],
    );
    // Discriminator for the old execution-error fallback: the VM errors, whereas a
    // restarted tree invocation would consume the stub and incorrectly return 99.
    // This does not claim the VM supports interpreter-only stubs on successful calls.
    let error = eval::eval_async_function(
        &program,
        "root",
        &parser::parse_single_expr("{}").unwrap(),
        handle,
    )
    .unwrap_err();
    assert!(error.to_string().contains("division by zero"));
    assert!(error
        .downcast_ref::<eval::result_contract::LanguageFailure>()
        .is_none());
}

#[test]
fn same_named_root_and_callee_keep_runtime_resolution_and_interner_ids() {
    let program = program("+module A\n+fn child ()->Int\n  +return 1\n+end\n+module B\n+fn child ()->Int\n  +return child()\n+end\n");
    let root = program.get_function("B.child").unwrap();
    assert!(matches!(
        vm::attempt(root, vec![], &program, false, &|_, _| anyhow::bail!(
            "no IO"
        )),
        vm::VmAttempt::Executed(Ok(eval::Value::Int(1)))
    ));
    let program = self::program("+fn total (xs:List<Int>)->Int\n  +let n:Int = 0\n  +each xs x:Int\n    +set n = n + x\n  +end\n  +return n\n+end\n+fn root ()->Int\n  +return total(list(1,2,3))\n+end\n");
    // The callee introduces len/get bytecodes for +each. Preparation shares one growing
    // interner, so those ids cannot collide with names introduced by other callees.
    assert!(matches!(
        vm::attempt(
            program.get_function("root").unwrap(),
            vec![],
            &program,
            false,
            &|_, _| anyhow::bail!("no IO")
        ),
        vm::VmAttempt::Executed(Ok(eval::Value::Int(6)))
    ));
}

fn run_async(program: &ast::Program, target: &str) -> (anyhow::Result<eval::Value>, Vec<String>) {
    let (sender, mut receiver) = tokio::sync::mpsc::channel(8);
    let responder = std::thread::spawn(move || {
        let mut requests = Vec::new();
        while let Some(request) = receiver.blocking_recv() {
            let IoRequest::FileExists { path, reply } = request else {
                panic!("only synthetic FileExists")
            };
            requests.push(path);
            assert!(requests.len() < 8, "replay or unbounded requests");
            reply.send(Ok(true)).unwrap();
        }
        requests
    });
    let result = eval::eval_async_function(
        program,
        target,
        &parser::parse_single_expr("{}").unwrap(),
        CoroutineHandle::new(sender),
    );
    (result, responder.join().unwrap())
}

#[test]
fn vm_scalar_check_unwinds_before_fail_after_without_extra_io() {
    let program = program("+fn reject ()->Bool [fail]\n  +check ready false ~not_ready\n  +return true\n+end\n+fn root ()->Int [io,async,fail]\n  +await before:Bool = file_exists(\"before\")\n  +let answer:Bool = reject()\n  +await after:Bool = file_exists(\"after\")\n  +return 0\n+end\n");
    assert!(
        vm::compile_function(program.get_function("root").unwrap(), &program).is_ok(),
        "this must exercise VM execution, not pre-execution fallback"
    );
    let (result, requests) = run_async(&program, "root");
    assert_eq!(requests, ["before"], "failed callee must suppress later IO");
    let error = result.unwrap_err();
    assert_eq!(
        error.downcast_ref::<eval::result_contract::LanguageFailure>(),
        Some(&eval::result_contract::LanguageFailure::Check {
            label: "not_ready".into()
        })
    );
}
