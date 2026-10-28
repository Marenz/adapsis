use super::program;
use crate::{
    ast,
    coroutine::{CoroutineHandle, IoRequest, TurnIdentity},
    eval::{self, Value},
    parser, session, vm,
};

const FLOW: &str = "+module Flow\n+fn child ()->String [io,async]\n  +await value:String = read_file(\"child\")\n  +return value\n+end\n+fn helper ()->String [io,async]\n  +await value:String = child()\n  +return value\n+end\n+fn helper2 ()->String [io,async]\n  +return helper()\n+end\n+fn identity (value:String)->String\n  +return value\n+end\n";

fn read_requests(
    program: &ast::Program,
    target: &str,
    fail_child: bool,
) -> (anyhow::Result<Value>, Vec<String>) {
    let (sender, mut receiver) = tokio::sync::mpsc::channel(8);
    let responder = std::thread::spawn(move || {
        let mut seen = Vec::new();
        while let Some(request) = receiver.blocking_recv() {
            let IoRequest::FileRead { path, reply } = request else {
                panic!("only synthetic FileRead allowed")
            };
            seen.push(path.clone());
            assert!(seen.len() <= 4, "request replay/unbounded execution");
            reply
                .send(if fail_child && path == "child" {
                    Err(anyhow::anyhow!("child fixture failure"))
                } else {
                    Ok(path)
                })
                .unwrap();
        }
        seen
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
fn vm_user_await_normal_dispatch_before_after_and_transitive_arguments_once() {
    for (body, expected, order) in [
        ("  +await value:String = child()\n  +await after:String = read_file(\"after\")\n  +return concat(value, after)", "childafter", vec!["child", "after"]),
        ("  +await first:String = read_file(\"first\")\n  +await value:String = Flow.child()\n  +return concat(first, value)", "firstchild", vec!["first", "child"]),
        ("  +await first:String = read_file(\"first\")\n  +return concat(first, helper2())", "firstchild", vec!["first", "child"]),
        ("  +await first:String = read_file(\"first\")\n  +await last:String = read_file(identity(helper2()))\n  +return last", "child", vec!["first", "child", "child"]),
    ] {
        let program = program(&format!("{FLOW}+fn root ()->String [io,async]\n{body}\n+end\n"));
        let error = vm::compile_function(program.get_function("Flow.root").unwrap(), &program).unwrap_err();
        assert!(error.to_string().contains(super::REQUIRES_TREE));
        let (result, requests) = read_requests(&program, "Flow.root", false);
        assert!(matches!(result.unwrap(), Value::String(ref value) if value.as_str() == expected));
        assert_eq!(requests, order);
    }
}

#[test]
fn vm_user_await_child_failure_after_first_io_does_not_replay_or_continue() {
    let program = program(&format!("{FLOW}+fn root ()->String [io,async]\n  +await first:String = read_file(\"first\")\n  +let child:String = helper2()\n  +await never:String = read_file(\"must-not-run\")\n  +return never\n+end\n"));
    let (result, requests) = read_requests(&program, "Flow.root", true);
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("child fixture failure"));
    assert_eq!(requests, ["first", "child"]);
}

#[test]
fn vm_user_await_bare_and_qualified_resolution_is_not_caller_module_first() {
    for (callee, expected) in [("child", "A"), ("B.child", "B")] {
        let program = program(&format!("+module A\n+fn child ()->String [io,async]\n  +await value:String = read_file(\"A\")\n  +return value\n+end\n+module B\n+fn child ()->String [io,async]\n  +await value:String = read_file(\"B\")\n  +return value\n+end\n+fn root ()->String [io,async]\n  +await value:String = {callee}()\n  +return value\n+end\n"));
        let (result, requests) = read_requests(&program, "B.root", false);
        assert!(matches!(result.unwrap(), Value::String(ref text) if text.as_str() == expected));
        assert_eq!(requests, [expected]);
    }
}

#[test]
fn vm_user_await_same_named_functions_have_distinct_snapshot_identity() {
    let program = program("+module A\n+fn helper ()->String\n  +return \"A\"\n+end\n+module B\n+fn leaf ()->String [io,async]\n  +await value:String = read_file(\"B\")\n  +return value\n+end\n+fn helper ()->String [io,async]\n  +await value:String = B.leaf()\n  +return value\n+end\n+module C\n+fn root ()->String [io,async]\n  +return concat(B.helper(), A.helper())\n+end\n+fn control ()->String [async]\n  +return helper()\n+end\n");
    // LIFO traversal sees A.helper first. Tracking only the bare name would
    // incorrectly skip B.helper and miss its reachable user await.
    assert!(
        vm::compile_function(program.get_function("C.root").unwrap(), &program)
            .unwrap_err()
            .to_string()
            .contains(super::REQUIRES_TREE)
    );
    let (result, requests) = read_requests(&program, "C.root", false);
    assert!(matches!(result.unwrap(), Value::String(ref s) if s.as_str() == "BA"));
    assert_eq!(requests, ["B"]);
    // The bare helper in C resolves to A, not the unrelated B implementation.
    assert!(vm::compile_function(program.get_function("C.control").unwrap(), &program).is_ok());
    let (result, requests) = read_requests(&program, "C.control", false);
    assert!(matches!(result.unwrap(), Value::String(ref s) if s.as_str() == "A"));
    assert!(requests.is_empty());
}

fn mock_read() -> session::IoMock {
    session::IoMock {
        operation: "read_file".into(),
        patterns: vec!["child".into()],
        response: "mocked".into(),
    }
}

#[test]
fn vm_user_await_original_handle_mocks_and_stubs_are_preserved() {
    let program = program(&format!("{FLOW}+fn root ()->String [io,async]\n  +await value:String = child()\n  +return value\n+end\n"));
    let input = parser::parse_single_expr("{}").unwrap();
    let mocked = eval::eval_async_function(
        &program,
        "Flow.root",
        &input,
        CoroutineHandle::new_mock(vec![mock_read()]),
    )
    .unwrap();
    assert!(matches!(mocked, Value::String(ref s) if s.as_str() == "mocked"));
    let stub = session::FunctionStub {
        function_name: "Flow.child".into(),
        patterns: vec![],
        response_expr: "\"stubbed\"".into(),
    };
    let stubbed = eval::eval_async_function(
        &program,
        "Flow.root",
        &input,
        CoroutineHandle::new_mock_with_stubs(vec![], vec![stub]),
    )
    .unwrap();
    assert!(matches!(stubbed, Value::String(ref s) if s.as_str() == "stubbed"));
}

#[test]
fn vm_user_await_pure_async_dispatch_preserves_frontend_result_restriction() {
    let program = program("+fn child ()->Int [async]\n  +return 7\n+end\n+fn root ()->Int [async]\n  +await value:Int = child()\n  +return value\n+end\n");
    let result = eval::eval_async_function(
        &program,
        "root",
        &parser::parse_single_expr("{}").unwrap(),
        CoroutineHandle::new_mock(vec![]),
    )
    .unwrap();
    assert!(matches!(result, Value::Int(7)));
    let error = crate::python_frontend::lower_module("async def child() -> Result[str]:\n    return Ok('x')\nasync def root() -> str:\n    value: Result[str] = await child()\n    return 'x'\n", "Guard").unwrap_err();
    assert!(
        error
            .to_string()
            .contains("awaiting Result-returning user functions is unsupported"),
        "{error}"
    );
}

#[test]
fn vm_user_await_guest_denial_precedes_mock_and_request() {
    let program = program(&format!("{FLOW}+fn root ()->String [io,async]\n  +await value:String = child()\n  +return value\n+end\n"));
    let (sender, mut receiver) = tokio::sync::mpsc::channel(2);
    let handle = CoroutineHandle::new_mock_with_sender(vec![mock_read()], sender).with_turn(Some(
        TurnIdentity {
            context: "fixture:context".into(),
            principal: "fixture:guest".into(),
            may_write: false,
            guest_tools: Some(vec![]),
        },
    ));
    let result = eval::eval_async_function(
        &program,
        "Flow.root",
        &parser::parse_single_expr("{}").unwrap(),
        handle,
    );
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("permission denied"));
    assert!(receiver.try_recv().is_err());
}

#[test]
fn vm_user_await_turn_identity_reaches_scoped_builtin() {
    let program = program("+module Flow\n+fn child ()->String [io,async]\n  +await value:String = memory_inspect(\"fixture-query\")\n  +return value\n+end\n+fn root ()->String [io,async]\n  +await value:String = Flow.child()\n  +return value\n+end\n");
    let (sender, mut receiver) = tokio::sync::mpsc::channel(2);
    let responder = std::thread::spawn(move || {
        let IoRequest::MemoryInspect {
            context_id,
            principal_id,
            query,
            reply,
        } = receiver.blocking_recv().unwrap()
        else {
            panic!("only scoped fixture request")
        };
        assert_eq!(context_id, "fixture:context");
        assert_eq!(principal_id, "fixture:guest");
        assert_eq!(query, "fixture-query");
        reply.send(Ok("bound".into())).unwrap();
        assert!(receiver.blocking_recv().is_none(), "extra request");
    });
    let handle = CoroutineHandle::new(sender).with_turn(Some(TurnIdentity {
        context: "fixture:context".into(),
        principal: "fixture:guest".into(),
        may_write: false,
        guest_tools: Some(vec!["memory_inspect".into()]),
    }));
    let result = eval::eval_async_function(
        &program,
        "Flow.root",
        &parser::parse_single_expr("{}").unwrap(),
        handle,
    )
    .unwrap();
    assert!(matches!(result, Value::String(ref value) if value.as_str() == "bound"));
    responder.join().unwrap();
}

#[test]
fn vm_user_await_unknown_operation_does_not_reset_after_io_guard() {
    let program = program("+fn root ()->String [io,async]\n  +await first:String = read_file(\"first\")\n  +await value:String = unknown_target()\n  +return value\n+end\n");
    assert!(vm::compile_function(program.get_function("root").unwrap(), &program).is_ok());
    let (result, requests) = read_requests(&program, "root", false);
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("unknown await operation: unknown_target"));
    assert_eq!(requests, ["first"]);
}
