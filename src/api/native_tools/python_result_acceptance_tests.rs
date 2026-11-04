use super::{
    python_catalog_tests::{inspect, mutate, permissions},
    *,
};
use serde_json::json;

const SOURCE: &str = "@effects('fail')\nasync def checked(n: int) -> Result[int]:\n    check('positive', n >= 0, 'negative')\n    return n\n\nasync def explicit(n: int) -> Result[int]:\n    return Ok(n)\n\nasync def rejected() -> Result[int]:\n    return Err('domain')\n\nasync def divided(n: int) -> Result[int]:\n    return div(8, n)\n\n@effects('fail')\nasync def consume(n: int) -> int:\n    result: Result[int] = await checked(n)\n    match result:\n        case Ok(value):\n            return value\n        case Err(error):\n            return -1\n\nasync def consume_explicit(n: int) -> int:\n    result: Result[int] = await explicit(n)\n    match result:\n        case Ok(value):\n            return value\n        case Err(error):\n            return -1\n\nasync def consume_rejected() -> int:\n    result: Result[int] = await rejected()\n    match result:\n        case Ok(value):\n            return value\n        case Err(error):\n            return -1\n\nasync def consume_divided(n: int) -> int:\n    result: Result[int] = await divided(n)\n    match result:\n        case Ok(value):\n            return value\n        case Err(error):\n            return -1\n";

async fn evaluate(
    cfg: &AppConfig,
    ws: &mut WorkingSet,
    baseline: &Program,
    function: &str,
    input: &str,
) -> super::super::NativeToolResult {
    execute_native_tool(
        "evaluate",
        &json!({"language":"python","function":function,"input":input}).to_string(),
        cfg,
        ws,
        baseline,
    )
    .await
}

#[tokio::test]
async fn native_python_result_return_await_test_inspect_resubmit_and_operational_failure() {
    let baseline = Program::default();
    let mut cfg = config_for(
        "writer",
        AccessLevel::Full,
        permissions(None),
        baseline.clone(),
    );
    // Even pure async functions use the existing coroutine dispatch adapter; no requests
    // are expected here. Keep a receiver to prove the pure graph doesn't gain IO.
    let (sender, mut receiver) = tokio::sync::mpsc::channel(4);
    cfg.io_sender = Some(sender);
    let mut ws = cfg.snapshot_working_set().await;
    let applied = mutate(&cfg, &mut ws, &baseline, "NativeResultGate", SOURCE).await;
    assert!(!applied.execution.has_errors, "{}", applied.content);
    // Scalar consumers provide literal expectations. No native Result test literal policy
    // is relaxed, and seeing a double/unwrapped value would make the match fail.
    let tests = "def test_success():\n    assert consume(7) == 7\n\ndef test_check():\n    assert consume(-2) == -1\n\ndef test_explicit():\n    assert consume_explicit(4) == 4\n\ndef test_rejected():\n    assert consume_rejected() == -1\n\ndef test_divided():\n    assert consume_divided(2) == 4\n";
    let tested = execute_native_tool(
        "test",
        &json!({"language":"python","module":"NativeResultGate","source":tests}).to_string(),
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(!tested.execution.has_errors, "{}", tested.content);
    assert!(!tested.execution.test_results.is_empty());
    cfg.write_back_working_set(&ws).await;
    for (function, input, expected) in [
        ("consume", "7", "7"),
        ("consume", "n=-2", "-1"),
        ("consume_explicit", "4", "4"),
        ("consume_rejected", "", "-1"),
    ] {
        let result = evaluate(
            &cfg,
            &mut ws,
            &baseline,
            &format!("NativeResultGate.{function}"),
            input,
        )
        .await;
        assert!(!result.execution.has_errors, "{}", result.content);
        assert!(
            result.content.contains(&format!("= {expected}")),
            "{}",
            result.content
        );
    }
    let before = serde_json::to_value(&ws.program).unwrap();
    let failed = evaluate(
        &cfg,
        &mut ws,
        &baseline,
        "NativeResultGate.consume_divided",
        "0",
    )
    .await;
    assert!(
        failed.execution.has_errors && failed.content.contains("division by zero"),
        "{}",
        failed.content
    );
    assert_eq!(before, serde_json::to_value(&ws.program).unwrap());
    assert!(receiver.try_recv().is_err());
    let printed = inspect(&cfg, &mut ws, &baseline, "NativeResultGate.consume").await;
    assert!(
        printed.contains("Result[int] = await NativeResultGate.checked(n)"),
        "{printed}"
    );
    let resubmitted = mutate(&cfg, &mut ws, &baseline, "NativeResultGate", &printed).await;
    assert!(!resubmitted.execution.has_errors, "{}", resubmitted.content);
    let plain = inspect(&cfg, &mut ws, &baseline, "NativeResultGate.checked").await;
    assert!(
        plain.contains("return n") && !plain.contains("return Ok(n)"),
        "{plain}"
    );
    // A plain T await binding stays refused; no partial mutation is applied.
    let before = serde_json::to_value(&ws.program).unwrap();
    let bad = mutate(&cfg,&mut ws,&baseline,"NativeResultBad","async def f() -> int:\n    value: int = await NativeResultGate.checked(1)\n    return value\n").await;
    assert!(bad.execution.has_errors, "{}", bad.content);
    assert_eq!(before, serde_json::to_value(&ws.program).unwrap());
}

#[tokio::test]
async fn native_result_await_inherits_turn_identity_and_performs_one_scoped_request() {
    let baseline = Program::default();
    let mut cfg = config_for(
        "writer",
        AccessLevel::Full,
        permissions(None),
        baseline.clone(),
    );
    cfg.turn = Some(crate::coroutine::TurnIdentity {
        context: "fixture-result-context".into(),
        principal: "fixture-result-principal".into(),
        may_write: true,
        guest_tools: None,
    });
    let (sender, mut receiver) = tokio::sync::mpsc::channel(4);
    cfg.io_sender = Some(sender);
    let mut ws = cfg.snapshot_working_set().await;
    let source="@effects('io')\nasync def leaf() -> Result[str]:\n    value: str = await memory_inspect()\n    return value\n\n@effects('io')\nasync def consume() -> str:\n    result: Result[str] = await leaf()\n    match result:\n        case Ok(value):\n            return value\n        case Err(error):\n            return error\n";
    let applied = mutate(&cfg, &mut ws, &baseline, "NativeResultIdentity", source).await;
    assert!(!applied.execution.has_errors, "{}", applied.content);
    cfg.write_back_working_set(&ws).await;
    let responder = tokio::spawn(async move {
        match receiver.recv().await.unwrap() {
            crate::coroutine::IoRequest::MemoryInspect {
                context_id,
                principal_id,
                query,
                reply,
            } => {
                assert_eq!(context_id, "fixture-result-context");
                assert_eq!(principal_id, "fixture-result-principal");
                assert!(query.is_empty());
                reply.send(Ok("scoped-result".into())).unwrap();
            }
            _ => panic!("only a synthetic scoped request is permitted"),
        }
        receiver
    });
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        evaluate(&cfg, &mut ws, &baseline, "NativeResultIdentity.consume", ""),
    )
    .await
    .unwrap();
    assert!(
        !result.execution.has_errors && result.content.contains("scoped-result"),
        "{}",
        result.content
    );
    let mut receiver = responder.await.unwrap();
    assert!(
        receiver.try_recv().is_err(),
        "user Result await must not replay the scoped call"
    );
    // Narrow the inherited turn to a guest with no allowed tools, then use normal runtime
    // dispatch on exactly the installed Python function. No request may cross the boundary.
    let handle = crate::coroutine::CoroutineHandle::new(cfg.io_sender.as_ref().unwrap().clone())
        .with_turn(Some(crate::coroutine::TurnIdentity {
            context: "fixture-result-context".into(),
            principal: "fixture-result-principal".into(),
            may_write: false,
            guest_tools: Some(vec![]),
        }));
    let p = ws.program.clone();
    let denied = tokio::task::spawn_blocking(move || {
        crate::eval::eval_async_function(
            &p,
            "NativeResultIdentity.consume",
            &crate::parser::parse_single_expr("{}").unwrap(),
            handle,
        )
    })
    .await
    .unwrap();
    assert!(denied.is_err());
    assert!(
        receiver.try_recv().is_err(),
        "nested user await cannot widen guest authority"
    );
}

#[tokio::test]
async fn native_user_result_await_cannot_consult_a_denied_signature() {
    let baseline=build_program("+module NativeResultHidden\n+fn leaf ()->Result<String> [async]\n  +return Ok(\"PRIVATE_RESULT_SENTINEL\")\n+end\n").await;
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        permissions(Some("NativeResultHidden")),
        baseline.clone(),
    );
    let mut ws = cfg.snapshot_working_set().await;
    let before = serde_json::to_value(&ws.program).unwrap();
    let denied=mutate(&cfg,&mut ws,&baseline,"NativeResultDenied","async def f() -> Result[str]:\n    value: Result[str] = await NativeResultHidden.leaf()\n    return value\n").await;
    assert!(denied.execution.has_errors);
    assert!(!denied.content.contains("PRIVATE_RESULT_SENTINEL"));
    assert_eq!(before, serde_json::to_value(&ws.program).unwrap());
}

#[tokio::test]
async fn native_python_result_binding_after_scalar_drift_refuses_before_io() {
    for (initial, binding, ty, value) in [
        ("Result[int]", "Result[int]", "Int", "7"),
        ("Result[int]", "Result[int]", "Bool", "true"),
        ("int", "int", "Result<Int>", "Ok(7)"),
    ] {
        let baseline = Program::default();
        let mut cfg = config_for(
            "writer",
            AccessLevel::Full,
            permissions(None),
            baseline.clone(),
        );
        let (sender, mut receiver) = tokio::sync::mpsc::channel(4);
        cfg.io_sender = Some(sender);
        let mut ws = cfg.snapshot_working_set().await;
        let source = format!("@effects('io')\nasync def leaf() -> {initial}:\n    return 7\n\n@effects('io')\nasync def caller() -> int:\n    value: {binding} = await leaf()\n    return 1\n\n@effects('io')\nasync def root() -> int:\n    value: int = await caller()\n    return value\n");
        let applied = mutate(&cfg, &mut ws, &baseline, "NativeResultDrift", &source).await;
        assert!(!applied.execution.has_errors, "{}", applied.content);
        let caller = ws
            .program
            .get_function("NativeResultDrift.caller")
            .unwrap()
            .clone();
        let edited = execute_native_tool("mutate", &json!({"source":format!("+module NativeResultDrift\n+fn leaf ()->{ty} [io,async]\n  +await ignored:String = read_file(\"inert-result-drift\")\n  +return {value}\n+end\n")}).to_string(), &cfg, &mut ws, &baseline).await;
        assert!(!edited.execution.has_errors, "{}", edited.content);
        assert_eq!(
            serde_json::to_value(&caller).unwrap(),
            serde_json::to_value(ws.program.get_function("NativeResultDrift.caller").unwrap())
                .unwrap()
        );
        cfg.write_back_working_set(&ws).await;
        let before = serde_json::to_value(&ws.program).unwrap();
        let requests = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = requests.clone();
        let responder = tokio::spawn(async move {
            while let Some(request) = receiver.recv().await {
                counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                drop(request); // Close reply immediately: an unexpected IO never hangs.
            }
        });
        for entry in ["NativeResultDrift.caller", "NativeResultDrift.root"] {
            let denied = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                evaluate(&cfg, &mut ws, &baseline, entry, ""),
            )
            .await
            .unwrap();
            assert!(denied.execution.has_errors, "{}", denied.content);
            assert!(
                denied
                    .content
                    .contains("incompatible user-await Result binding"),
                "{}",
                denied.content
            );
        }
        let tested = execute_native_tool("test", &json!({"language":"python","module":"NativeResultDrift","source":"def test_caller():\n    assert caller() == 1\n\ndef test_root():\n    assert root() == 1\n"}).to_string(), &cfg, &mut ws, &baseline).await;
        assert!(
            tested.execution.has_errors
                && tested
                    .content
                    .contains("incompatible user-await Result binding"),
            "{}",
            tested.content
        );
        assert_eq!(requests.load(std::sync::atomic::Ordering::SeqCst), 0);
        responder.abort();
        assert_eq!(before, serde_json::to_value(&ws.program).unwrap());
    }
}
