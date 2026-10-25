use super::*;
use serde_json::json;

async fn setup() -> (AppConfig, WorkingSet, Program) {
    let program = build_program("+module PyCallShape\n+fn constant ()->Int\n +return 1\n+end\n+fn ordered (x:Int,y:Int)->Int\n +return x*10+y\n+end").await;
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        PermissionConfig::default(),
        program.clone(),
    );
    let ws = cfg.snapshot_working_set().await;
    (cfg, ws, program)
}

#[tokio::test]
async fn python_argument_binding_rejects_ignored_missing_extra_and_reserved_arguments() {
    let (cfg, mut ws, baseline) = setup().await;
    for (target, input) in [
        ("constant", "999"),
        ("constant", "unexpected=999"),
        ("ordered", "1"),
        ("ordered", "1, 2, 3"),
        ("ordered", "x=1, y=2, extra=3"),
        ("ordered", "wrong=1, y=2"),
        ("ordered", "_0=1, _1=2"),
    ] {
        let result = execute_native_tool(
            "evaluate",
            &json!({"language":"python","function":format!("PyCallShape.{target}"),"input":input})
                .to_string(),
            &cfg,
            &mut ws,
            &baseline,
        )
        .await;
        assert!(
            result.execution.has_errors,
            "{target}({input}): {}",
            result.content
        );
        let source = format!("def test_shape() -> None:\n    assert {target}({input}) == 1\n");
        let result = execute_native_tool(
            "test",
            &json!({"language":"python","module":"PyCallShape","source":source}).to_string(),
            &cfg,
            &mut ws,
            &baseline,
        )
        .await;
        assert!(result.execution.has_errors, "{input}: {}", result.content);
        assert!(
            result.execution.test_results.is_empty(),
            "invalid call shape must not execute tests"
        );
    }
    for input in ["1, 2", "y=2, x=1"] {
        let result = execute_native_tool(
            "evaluate",
            &json!({"language":"python","function":"PyCallShape.ordered","input":input})
                .to_string(),
            &cfg,
            &mut ws,
            &baseline,
        )
        .await;
        assert!(
            !result.execution.has_errors && result.content.contains("12"),
            "{}",
            result.content
        );
    }
    let legacy = execute_native_tool(
        "evaluate",
        r#"{"function":"PyCallShape.constant","input":"999"}"#,
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(
        !legacy.execution.has_errors,
        "Adapsis argument policy remains unchanged"
    );
}

#[tokio::test]
async fn python_direct_external_call_shape_is_checked_before_mutation() {
    let (cfg, mut ws, baseline) = setup().await;
    for expression in [
        "PyCallShape.constant(999)",
        "PyCallShape.ordered(1)",
        "PyCallShape.ordered(1, 2, 3)",
    ] {
        let source = format!("def probe() -> int:\n    return {expression}\n");
        let result = execute_native_tool(
            "mutate",
            &json!({"language":"python","module":"PyBadCaller","source":source}).to_string(),
            &cfg,
            &mut ws,
            &baseline,
        )
        .await;
        assert!(result.execution.has_errors, "{}", result.content);
        assert!(ws.program.get_function("PyBadCaller.probe").is_none());
    }
}

#[test]
fn python_reserved_carrier_identifiers_cannot_masquerade_as_keywords_or_parameters() {
    for name in ["_0", "_1", "_123"] {
        assert!(
            crate::python_frontend::lower_module(
                &format!("def bad({name}: int) -> int:\n    return {name}\n"),
                "PyReserved"
            )
            .is_err()
        );
        assert!(crate::python_frontend::lower_input(&format!("{name}=1")).is_err());
    }
}

#[tokio::test]
async fn python_async_calls_require_exact_await_semantics() {
    let (cfg, mut ws, baseline) = setup().await;
    let good = "async def delayed(x: int) -> int:\n    return x\n\nasync def use(x: int) -> int:\n    value: int = await delayed(x)\n    return value\n";
    let result = execute_native_tool(
        "mutate",
        &json!({"language":"python","module":"PyAsyncExact","source":good}).to_string(),
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(!result.execution.has_errors, "{}", result.content);
    let result = execute_native_tool("test", &json!({"language":"python","module":"PyAsyncExact","source":"def test_async() -> None:\n    assert use(3) == 3\n"}).to_string(), &cfg, &mut ws, &baseline).await;
    assert!(!result.execution.has_errors, "{}", result.content);
    for source in [
        "async def bad() -> int:\n    value: int = await PyCallShape.constant()\n    return value\n",
        "def bad() -> int:\n    return PyAsyncExact.delayed(3)\n",
        "async def bad() -> int:\n    value: int = await len([1])\n    return value\n",
    ] {
        let result = execute_native_tool(
            "mutate",
            &json!({"language":"python","module":"PyAsyncBad","source":source}).to_string(),
            &cfg,
            &mut ws,
            &baseline,
        )
        .await;
        assert!(result.execution.has_errors, "{}", result.content);
        assert!(ws.program.get_function("PyAsyncBad.bad").is_none());
    }
}

#[tokio::test]
async fn python_execution_rechecks_helpers_after_later_adapsis_mutation_and_reload() {
    let (cfg, mut ws, baseline) = setup().await;
    let result = execute_native_tool("mutate", &json!({"language":"python","module":"PySavedList","source":"def values() -> list[int]:\n    return [1, 2]\n"}).to_string(), &cfg, &mut ws, &baseline).await;
    assert!(!result.execution.has_errors, "{}", result.content);
    let result = execute_native_tool(
        "evaluate",
        r#"{"language":"python","function":"PySavedList.values"}"#,
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(!result.execution.has_errors && result.content.contains("[1, 2]"));
    let result = execute_native_tool(
        "mutate",
        &json!({"source":"+module PyLaterShadow\n+fn list (x:Int,y:Int)->Int\n +return 999\n+end"})
            .to_string(),
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(!result.execution.has_errors);
    ws.program = serde_json::from_str(&serde_json::to_string(&ws.program).unwrap()).unwrap();
    ws.program.rebuild_function_index();
    let result = execute_native_tool(
        "evaluate",
        r#"{"language":"python","function":"PySavedList.values"}"#,
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(result.execution.has_errors && result.content.contains("shadows"));
    let result = execute_native_tool("test", &json!({"language":"python","module":"PySavedList","source":"def test_values() -> None:\n    assert values() == [1, 2]\n"}).to_string(), &cfg, &mut ws, &baseline).await;
    assert!(result.execution.has_errors && result.execution.test_results.is_empty());
}

#[tokio::test]
async fn python_single_function_inspection_is_resubmittable_after_reload() {
    let (cfg, mut ws, baseline) = setup().await;
    let source = "def inc(x: int) -> int:\n    return x + 1\n\ndef twice(x: int) -> int:\n    return inc(inc(x)) * 2\n\ndef below(x: int) -> bool:\n    return x < 3\n\ndef count() -> int:\n    index: int = 0\n    while below(index):\n        index = index + 1\n    return index\n";
    let result = execute_native_tool(
        "mutate",
        &json!({"language":"python","module":"PyInspectRepair","source":source}).to_string(),
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(!result.execution.has_errors, "{}", result.content);
    for reload in [false, true] {
        if reload {
            ws.program =
                serde_json::from_str(&serde_json::to_string(&ws.program).unwrap()).unwrap();
            ws.program.rebuild_function_index();
        }
        for target in ["PyInspectRepair.twice", "PyInspectRepair.count"] {
            let inspected = execute_native_tool(
                "inspect",
                &json!({"kind":"source","target":target,"format":"python"}).to_string(),
                &cfg,
                &mut ws,
                &baseline,
            )
            .await;
            assert!(!inspected.execution.has_errors, "{}", inspected.content);
            let envelope: serde_json::Value = serde_json::from_str(&inspected.content).unwrap();
            let printed = envelope["query_results"].as_str().unwrap();
            let reapplied = execute_native_tool(
                "mutate",
                &json!({"language":"python","module":"PyInspectRepair","source":printed})
                    .to_string(),
                &cfg,
                &mut ws,
                &baseline,
            )
            .await;
            assert!(!reapplied.execution.has_errors, "{}", reapplied.content);
            assert_eq!(
                printed,
                crate::python_frontend::print_function(&ws.program, target).unwrap()
            );
        }
    }
    let tests = "def test_twice() -> None:\n    assert twice(1) == 6\n\ndef test_count() -> None:\n    assert count() == 3\n";
    let tested = execute_native_tool(
        "test",
        &json!({"language":"python","module":"PyInspectRepair","source":tests}).to_string(),
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(!tested.execution.has_errors, "{}", tested.content);
}

#[tokio::test]
async fn python_hidden_signature_never_precedes_reference_authorization() {
    let program = build_program(
        "+module PyHiddenSignature\n+fn secret ()->String\n +return \"hidden\"\n+end",
    )
    .await;
    let permissions = toml::from_str(
        r#"
        [groups]
        hidden = ["PyHiddenSignature"]
        [model.writer]
        hidden = "none"
        user = "write"
    "#,
    )
    .unwrap();
    let cfg = config_for("writer", AccessLevel::Full, permissions, program.clone());
    let mut ws = cfg.snapshot_working_set().await;
    let args = json!({"language":"python","module":"PyPublicSignature","source":"def probe() -> int:\n    return PyHiddenSignature.secret() + 1\n"});
    let first = execute_native_tool("mutate", &args.to_string(), &cfg, &mut ws, &program).await;
    assert!(first.execution.has_errors && first.content.contains("permission denied"));
    let function = &mut ws.program.modules[0].functions[0];
    Arc::make_mut(function).return_type = crate::ast::Type::Int;
    let second = execute_native_tool("mutate", &args.to_string(), &cfg, &mut ws, &program).await;
    assert_eq!(first.content, second.content);
    assert!(ws.program.get_function("PyPublicSignature.probe").is_none());
}

#[tokio::test]
async fn python_source_inspection_is_independent_of_hidden_return_signatures() {
    let program = build_program("+module PyPrivatePrint\n+fn secret ()->Int\n +return 7\n+end\n+module PyPublicPrint\n+fn wrapper ()->Int\n +return PyPrivatePrint.secret()\n+end").await;
    let permissions = toml::from_str(
        r#"
        [groups]
        hidden = ["PyPrivatePrint"]
        [model.writer]
        hidden = "none"
        user = "write"
    "#,
    )
    .unwrap();
    let cfg = config_for("writer", AccessLevel::Full, permissions, program.clone());
    let mut ws = cfg.snapshot_working_set().await;
    let args = r#"{"kind":"source","target":"PyPublicPrint.wrapper","format":"python"}"#;
    let first = execute_native_tool("inspect", args, &cfg, &mut ws, &program).await;
    Arc::make_mut(&mut ws.program.modules[0].functions[0]).return_type = crate::ast::Type::String;
    let second = execute_native_tool("inspect", args, &cfg, &mut ws, &program).await;
    assert!(first.execution.has_errors);
    assert_eq!(
        first.content, second.content,
        "hidden signatures must not affect Python source inspection"
    );
}

#[tokio::test]
async fn python_execution_checks_io_registry_names_and_aliases_after_edits() {
    for (index, helper) in ["read_file", "file_read", "http_get"].iter().enumerate() {
        let target = format!("PyIOShadowTarget{index}");
        let private = format!("PyIOShadowOwner{index}");
        let program = Program::default();
        let mut cfg = config_for(
            "writer",
            AccessLevel::Full,
            PermissionConfig::default(),
            program.clone(),
        );
        let mut ws = cfg.snapshot_working_set().await;
        let source = "@effects(\"io\")\nasync def read_text(path: str) -> str:\n    text: str = await read_file(path)\n    return text\n";
        let created = execute_native_tool(
            "mutate",
            &json!({"language":"python","module":target,"source":source}).to_string(),
            &cfg,
            &mut ws,
            &program,
        )
        .await;
        assert!(!created.execution.has_errors, "{}", created.content);
        ws.meta.io_mocks.push(crate::session::IoMock {
            operation: "read_file".into(),
            patterns: vec!["fixture".into()],
            response: "expected".into(),
        });
        let tests = json!({"language":"python","module":target,"source":"def test_read() -> None:\n    assert read_text(\"fixture\") == \"expected\"\n"});
        let tested = execute_native_tool("test", &tests.to_string(), &cfg, &mut ws, &program).await;
        assert!(!tested.execution.has_errors, "{}", tested.content);
        let shadow = format!(
            "+module {private}\n+fn {helper} (path:String)->String\n +return \"INTERCEPTED\"\n+end"
        );
        let edited = execute_native_tool(
            "mutate",
            &json!({"source":shadow}).to_string(),
            &cfg,
            &mut ws,
            &program,
        )
        .await;
        assert!(!edited.execution.has_errors);
        for hidden in [false, true] {
            if hidden {
                cfg.permission_config = Arc::new(toml::from_str(&format!("[groups]\nhidden=[\"{private}\"]\n[model.writer]\nhidden=\"none\"\nuser=\"write\"\n")).unwrap());
            }
            let denied =
                execute_native_tool("test", &tests.to_string(), &cfg, &mut ws, &program).await;
            assert!(denied.execution.has_errors && denied.content.contains("shadows"));
            assert!(denied.execution.test_results.is_empty());
            assert!(!denied.content.contains("INTERCEPTED"));
        }
    }
}

#[tokio::test]
async fn python_user_result_await_is_refused_for_local_existing_and_printed_calls() {
    let (cfg, mut ws, baseline) = setup().await;
    let source = "@effects(\"io\")\nasync def wrapped(url: str) -> Result[str]:\n    value: Result[str] = await http_get(url)\n    return value\n\n@effects(\"io\")\nasync def probe(url: str) -> Result[str]:\n    value: Result[str] = await wrapped(url)\n    return value\n";
    let local = execute_native_tool(
        "mutate",
        &json!({"language":"python","module":"PyResultLocal","source":source}).to_string(),
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(
        local.execution.has_errors && local.content.contains("#82"),
        "{}",
        local.content
    );
    assert!(ws.program.get_function("PyResultLocal.wrapped").is_none());
    let legacy = "+module PyResultTrusted\n+fn wrapped ()->Result<String> [async]\n +return Ok(\"ok\")\n+end\n+fn probe ()->Result<String> [async]\n +await value:Result<String> = PyResultTrusted.wrapped()\n +return value\n+end";
    let existing = execute_native_tool(
        "mutate",
        &json!({"source":legacy}).to_string(),
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(!existing.execution.has_errors);
    let source = "async def probe() -> Result[str]:\n    value: Result[str] = await PyResultTrusted.wrapped()\n    return value\n";
    let external = execute_native_tool(
        "mutate",
        &json!({"language":"python","module":"PyResultExternal","source":source}).to_string(),
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(external.execution.has_errors && external.content.contains("#82"));
    let inspected = execute_native_tool(
        "inspect",
        r#"{"kind":"source","target":"PyResultTrusted.probe","format":"python"}"#,
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(inspected.execution.has_errors && inspected.content.contains("#82"));
}

#[tokio::test]
async fn python_direct_io_result_await_preserves_the_actual_runtime_wrapper() {
    let (mut cfg, mut ws, baseline) = setup().await;
    let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
    cfg.io_sender = Some(sender);
    let responder = tokio::spawn(async move {
        match receiver.recv().await.unwrap() {
            crate::coroutine::IoRequest::HttpGet { url, reply } => {
                assert_eq!(url, "fixture://result");
                let _ = reply.send(Ok("builtin-result".into()));
            }
            _ => panic!("only the in-memory HTTP result fixture is allowed"),
        }
    });
    let source = "@effects(\"io\")\nasync def fetch(url: str) -> Result[str]:\n    value: Result[str] = await http_get(url)\n    return value\n";
    let created = execute_native_tool(
        "mutate",
        &json!({"language":"python","module":"PyDirectIOResult","source":source}).to_string(),
        &cfg,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(!created.execution.has_errors, "{}", created.content);
    cfg.write_back_working_set(&ws).await;
    let evaluated = tokio::time::timeout(std::time::Duration::from_secs(10), execute_native_tool("evaluate",
        r#"{"language":"python","function":"PyDirectIOResult.fetch","input":"'fixture://result'"}"#,
        &cfg, &mut ws, &baseline)).await.unwrap();
    assert!(!evaluated.execution.has_errors, "{}", evaluated.content);
    assert!(
        evaluated
            .execution
            .mutation_results
            .iter()
            .any(|result| result.message.contains("Ok(")
                && result.message.contains("builtin-result")),
        "{}",
        evaluated.content
    );
    responder.await.unwrap();
}
