//! Real native-path admission regressions complement direct visitor tests.
use super::python_catalog_tests::{mutate, permissions};
use super::*;
async fn fixture(tag: &str) -> (AppConfig, WorkingSet, Program) {
    let source = "+module NtCallSites\n+fn task (x:Int)->Int [async]\n +return x\n+end\n+fn result (x:Int)->Result<Int> [async]\n +return Ok(x)\n+end\n+fn sync_result (x:Int)->Result<Int>\n +return Ok(x)\n+end\n+fn sync (x:Int)->Int\n +return x\n+end\n+fn io (x:Int)->Int [io,async]\n +return x\n+end\n";
    let program = build_program(&source.replace("NtCallSites", &format!("NtCallSites{tag}"))).await;
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        permissions(None),
        program.clone(),
    );
    let ws = cfg.snapshot_working_set().await;
    (cfg, ws, program)
}

#[tokio::test]
async fn python_native_call_modes_reject_user_async_match_and_nested_unawaited_calls() {
    let (cfg, mut ws, baseline) = fixture("Reject").await;
    let cases = [
        (
            "def f() -> int:\n    match NtCallSites.result(1):\n        case Ok(value):\n            return value\n        case Err(error):\n            return 0\n",
            "annotated await",
        ),
        (
            "@effects('io')\nasync def f() -> int:\n    match NtCallSites.result(1):\n        case Ok(value):\n            return value\n        case Err(error):\n            return 0\n",
            "annotated await",
        ),
        (
            "async def f() -> int:\n    value: Result[int] = await NtCallSites.result(1)\n    return 0\n",
            "#82",
        ),
        (
            "def f() -> int:\n    match NtCallSites.sync_result(1):\n        case Ok(value):\n            return NtCallSites.task(value)\n        case Err(error):\n            return 0\n",
            "annotated await",
        ),
        (
            "async def f() -> int:\n    check('ready', NtCallSites.task(1) == 1, 'err')\n    return 0\n",
            "annotated await",
        ),
        (
            "async def f() -> int:\n    spawn(NtCallSites.sync(NtCallSites.task(1)))\n    return 0\n",
            "annotated await",
        ),
        (
            "async def f() -> int:\n    spawn(NtCallSites.sync())\n    return 0\n",
            "arguments do not match",
        ),
        (
            "def f() -> int:\n    spawn(NtCallSites.task(1))\n    return 0\n",
            "spawn requires async def",
        ),
        (
            "async def f() -> int:\n    spawn(NtCallSites.io(1))\n    return 0\n",
            "@effects('io')",
        ),
        (
            "@effects('io')\nasync def f() -> int:\n    spawn(http_get('fixture://no-spawn'))\n    return 0\n",
            "qualified user-function",
        ),
        (
            "@effects('io')\nasync def startup() -> None:\n    spawn(NtCallSites.task(1))\n",
            "forbids",
        ),
    ];
    for (index, (source, message)) in cases.into_iter().enumerate() {
        let before = serde_json::to_value(&ws.program).unwrap();
        let source = source.replace("NtCallSites", &baseline.modules[0].name);
        let result = mutate(
            &cfg,
            &mut ws,
            &baseline,
            &format!("NtCallModeReject{index}"),
            &source,
        )
        .await;
        assert!(
            result.execution.has_errors && result.content.contains(message),
            "{index}: {}",
            result.content
        );
        assert_eq!(serde_json::to_value(&ws.program).unwrap(), before);
    }
}

#[tokio::test]
async fn python_native_builtin_io_match_executes_once_with_mocked_reply() {
    let (mut cfg, mut ws, baseline) = fixture("Match").await;
    let (sender, mut receiver) = tokio::sync::mpsc::channel(4);
    cfg.io_sender = Some(sender);
    let responder = tokio::spawn(async move {
        match receiver.recv().await.unwrap() {
            crate::coroutine::IoRequest::HttpGet { url, reply } => {
                assert_eq!(url, "fixture://native-match");
                reply.send(Ok("matched-once".into())).unwrap();
            }
            _ => panic!("unexpected IO; this fixture never uses real network"),
        }
        receiver
    });
    let source = "@effects('io')\nasync def fetch() -> str:\n    match http_get('fixture://native-match'):\n        case Ok(value):\n            return value\n        case Err(error):\n            return error\n";
    let created = mutate(&cfg, &mut ws, &baseline, "NtCallModeMatch", source).await;
    assert!(!created.execution.has_errors, "{}", created.content);
    cfg.write_back_working_set(&ws).await;
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        execute_native_tool(
            "evaluate",
            r#"{"language":"python","function":"NtCallModeMatch.fetch","input":""}"#,
            &cfg,
            &mut ws,
            &baseline,
        ),
    )
    .await
    .unwrap();
    assert!(
        !result.execution.has_errors && result.content.contains("matched-once"),
        "{}",
        result.content
    );
    let mut receiver = responder.await.unwrap();
    assert!(receiver.try_recv().is_err(), "IO was repeated");
}

#[tokio::test]
async fn python_native_spawn_uses_qualified_task_with_unchanged_arguments_once() {
    let (mut cfg, mut ws, baseline) = fixture("Spawn").await;
    let (sender, mut receiver) = tokio::sync::mpsc::channel(4);
    cfg.io_sender = Some(sender);
    let source = "async def launch() -> int:\n    spawn(NtCallSites.task(7))\n    return 1\n"
        .replace("NtCallSites", &baseline.modules[0].name);
    let created = mutate(&cfg, &mut ws, &baseline, "NtCallModeSpawn", &source).await;
    assert!(!created.execution.has_errors, "{}", created.content);
    cfg.write_back_working_set(&ws).await;
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        execute_native_tool(
            "evaluate",
            r#"{"language":"python","function":"NtCallModeSpawn.launch","input":""}"#,
            &cfg,
            &mut ws,
            &baseline,
        ),
    )
    .await
    .unwrap();
    assert!(!result.execution.has_errors, "{}", result.content);
    match receiver.try_recv().unwrap() {
        crate::coroutine::IoRequest::Spawn {
            function_name,
            args,
            ..
        } => {
            assert_eq!(function_name, format!("{}.task", baseline.modules[0].name));
            assert!(matches!(args.as_slice(), [crate::eval::Value::Int(7)]));
        }
        _ => panic!("expected one inert spawn request"),
    }
    assert!(receiver.try_recv().is_err(), "spawn was repeated");
}

#[tokio::test]
async fn python_native_call_visitor_uses_latest_accepted_signature_without_rolling_it_back() {
    let (cfg, mut ws, baseline) = fixture("Fresh").await;
    let target = &baseline.modules[0].name;
    let edited = mutate(
        &cfg,
        &mut ws,
        &baseline,
        target,
        "def sync(x: int, y: int) -> int:\n    return x + y\n",
    )
    .await;
    assert!(!edited.execution.has_errors, "{}", edited.content);
    cfg.write_back_working_set(&ws).await;
    let accepted = serde_json::to_value(&ws.program).unwrap();
    let rejected = mutate(
        &cfg,
        &mut ws,
        &baseline,
        "NtCallModeFresh",
        &format!(
            "def f() -> int:\n    check('valid', {target}.sync(1) == 1, 'err')\n    return 0\n"
        ),
    )
    .await;
    assert!(
        rejected.execution.has_errors && rejected.content.contains("arguments do not match"),
        "{}",
        rejected.content
    );
    assert_eq!(serde_json::to_value(&ws.program).unwrap(), accepted);
    assert_eq!(
        ws.program
            .get_function(&format!("{target}.sync"))
            .unwrap()
            .params
            .len(),
        2
    );
    let control = mutate(
        &cfg,
        &mut ws,
        &baseline,
        "NtCallModeFresh",
        &format!(
            "def f() -> int:\n    check('valid', {target}.sync(1, 2) == 3, 'err')\n    return 0\n"
        ),
    )
    .await;
    assert!(!control.execution.has_errors, "{}", control.content);
}

#[tokio::test]
async fn python_native_same_submission_io_variant_cannot_take_match_exception() {
    let (cfg, mut ws, baseline) = fixture("Shadow").await;
    let before = serde_json::to_value(&ws.program).unwrap();
    let source = "@union\nclass Collision:\n    http_get: tuple[str]\n    Other: tuple[()]\n@effects('io')\nasync def f() -> int:\n    match http_get('fixture://no-request'):\n        case _:\n            return 0\n";
    let result = mutate(&cfg, &mut ws, &baseline, "NtCallModeShadow", source).await;
    assert!(
        result.execution.has_errors && result.content.contains("shadowed"),
        "{}",
        result.content
    );
    assert_eq!(serde_json::to_value(&ws.program).unwrap(), before);
}

#[tokio::test]
async fn python_native_ordinary_synchronous_result_match_remains_supported() {
    let (cfg, mut ws, baseline) = fixture("SyncResult").await;
    let source = format!(
        "def f() -> int:\n    match {}.sync_result(5):\n        case Ok(value):\n            return value\n        case Err(error):\n            return 0\n",
        baseline.modules[0].name
    );
    let created = mutate(&cfg, &mut ws, &baseline, "NtCallModeSyncResult", &source).await;
    assert!(!created.execution.has_errors, "{}", created.content);
    cfg.write_back_working_set(&ws).await;
    let result = execute_native_tool(
        "evaluate",
        r#"{"language":"python","function":"NtCallModeSyncResult.f","input":""}"#,
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(!result.execution.has_errors, "{}", result.content);
    assert!(
        result
            .execution
            .mutation_results
            .iter()
            .any(|r| r.success && r.message.ends_with('5')),
        "{}",
        result.content
    );
}

#[tokio::test]
async fn python_native_synchronous_err_match_returns_zero() {
    let (cfg, mut ws, baseline) = fixture("ErrorResult").await;
    let source = "def wrap() -> Result[int]:\n    return Err('fixture')\ndef f() -> int:\n    match wrap():\n        case Ok(value):\n            return value\n        case Err(error):\n            return 0\n";
    let created = mutate(&cfg, &mut ws, &baseline, "NtCallModeErrorResult", source).await;
    assert!(!created.execution.has_errors, "{}", created.content);
    cfg.write_back_working_set(&ws).await;
    let result = execute_native_tool(
        "evaluate",
        r#"{"language":"python","function":"NtCallModeErrorResult.f","input":""}"#,
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(!result.execution.has_errors, "{}", result.content);
    assert!(
        result.execution.mutation_results.iter().any(|r| r.success
            && r.message
                .rsplit_once(" = ")
                .is_some_and(|(_, actual)| actual == "0")),
        "{}",
        result.content
    );
}

#[tokio::test]
async fn python_native_result_match_after_vm_io_does_not_replay_request() {
    for fail in [false, true] {
        let tag = if fail { "AwaitErr" } else { "AwaitOk" };
        let (mut cfg, mut ws, baseline) = fixture(tag).await;
        let (sender, mut receiver) = tokio::sync::mpsc::channel(4);
        cfg.io_sender = Some(sender);
        let responder = tokio::spawn(async move {
            let crate::coroutine::IoRequest::HttpGet { url, reply } =
                receiver.recv().await.unwrap()
            else {
                panic!("only inert HTTP fixture allowed")
            };
            assert_eq!(url, "fixture://after-vm-io");
            reply
                .send(if fail {
                    Err(anyhow::anyhow!("fixture-error"))
                } else {
                    Ok("fixture-ok".into())
                })
                .unwrap();
            receiver
        });
        let module = format!("NtCallMode{tag}");
        let source = "@effects('io')\nasync def fetch() -> str:\n    response: Result[str] = await http_get('fixture://after-vm-io')\n    match response:\n        case Ok(value):\n            return value\n        case Err(error):\n            return error\n";
        let created = mutate(&cfg, &mut ws, &baseline, &module, source).await;
        assert!(!created.execution.has_errors, "{}", created.content);
        cfg.write_back_working_set(&ws).await;
        let args = serde_json::json!({"language":"python","function":format!("{module}.fetch"),"input":""}).to_string();
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            execute_native_tool("evaluate", &args, &cfg, &mut ws, &baseline),
        )
        .await
        .unwrap();
        let expected = if fail { "fixture-error" } else { "fixture-ok" };
        assert!(
            !result.execution.has_errors && result.content.contains(expected),
            "{}",
            result.content
        );
        let mut receiver = responder.await.unwrap();
        assert!(receiver.try_recv().is_err(), "completed IO was replayed");
    }
}
