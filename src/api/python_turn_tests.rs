use super::*;

fn python_tool(id: &str, name: &str, module: &str, source: &str) -> Value {
    tool(
        id,
        name,
        json!({"language":"python", "module":module, "source":source}),
    )
}

async fn lifecycle(streaming: bool) {
    let module = if streaming {
        "PyNativeStream"
    } else {
        "PyNativeHttp"
    };
    let fixture = Fixture::new(
        vec![
            python_tool(
                "mutate",
                "mutate",
                module,
                include_str!("../../examples/python/basic.py"),
            ),
            tool("early", "finish", json!({})),
            tool(
                "source",
                "inspect",
                json!({"kind":"source","target":format!("{module}.add_two"),"format":"python"}),
            ),
            python_tool(
                "test",
                "test",
                module,
                include_str!("../../examples/python/test_basic.py"),
            ),
            tool(
                "evaluate",
                "evaluate",
                json!({"language":"python","function":format!("{module}.add_two"),"input":"x=3"}),
            ),
            tool("finish", "finish", json!({})),
        ],
        "",
    )
    .await;
    let result = fixture.run(streaming).await;
    assert_eq!(result.completed, Some(true), "{}", result.reply);
    assert!(result.has_errors, "premature finish must still be recorded");
    assert!(
        result
            .results
            .iter()
            .any(|r| !r.success && r.message.contains(module))
    );
    assert!(
        result
            .results
            .iter()
            .any(|r| r.success && r.message.contains("def add_two"))
    );
    assert!(
        result
            .results
            .iter()
            .any(|r| r.success && r.message.contains('5'))
    );
    assert_eq!(fixture.requests.lock().unwrap().len(), 6);
}

#[tokio::test]
async fn python_http_mutate_test_evaluate_finish_uses_the_existing_gate() {
    lifecycle(false).await;
}

#[tokio::test]
async fn python_stream_mutate_test_evaluate_finish_uses_the_existing_gate() {
    lifecycle(true).await;
}

#[tokio::test]
async fn python_lists_booleans_and_quoted_literal_data_reach_native_evaluation() {
    let fixture = Fixture::new(vec![
        python_tool("mutate", "mutate", "PyLiteralNative", include_str!("../../examples/python/flow.py")),
        python_tool("test", "test", "PyLiteralNative", include_str!("../../examples/python/test_flow.py")),
        tool("list", "evaluate", json!({"language":"python","function":"PyLiteralNative.sum_values","input":"[1, 2, 3]"})),
        tool("bool", "evaluate", json!({"language":"python","function":"PyLiteralNative.both","input":"True, False"})),
        tool("bool-true", "evaluate", json!({"language":"python","function":"PyLiteralNative.both","input":"True, True"})),
        tool("finish", "finish", json!({})),
    ], "").await;
    let response = fixture.run(false).await;
    assert_eq!(response.completed, Some(true));
    assert!(!response.has_errors, "{:?}", response.results);
    assert!(
        response
            .results
            .iter()
            .any(|r| r.success && r.message.contains('6'))
    );
    assert!(
        response
            .results
            .iter()
            .any(|r| r.success && r.message.contains("false"))
    );
    assert!(
        response
            .results
            .iter()
            .any(|r| r.success && r.message.contains("true"))
    );
}

#[tokio::test]
async fn python_local_calls_do_not_resolve_hidden_same_named_functions() {
    let mut fixture = Fixture::new(
        vec![
            python_tool(
                "mutate",
                "mutate",
                "PyLocalPublic",
                "def secret() -> int:\n    return 1\n\ndef probe() -> int:\n    return secret()\n",
            ),
            tool(
                "eval",
                "evaluate",
                json!({"function":"PyLocalPublic.probe","language":"python"}),
            ),
            tool("finish", "finish", json!({})),
        ],
        "+module PyLocalPrivate\n+fn secret ()->Int\n +return 987654\n+end",
    )
    .await;
    fixture.config.permission_config = restricted("PyLocalPrivate");
    let response = fixture.run(false).await;
    assert!(!response.has_errors, "{:?}", response.results);
    assert!(!serde_json::to_string(&response).unwrap().contains("987654"));
    assert!(
        response
            .results
            .iter()
            .any(|r| r.success && r.message.ends_with('1'))
    );
}

fn restricted(private: &str) -> Arc<crate::permissions::PermissionConfig> {
    Arc::new(
        toml::from_str(&format!(
            r#"
        [groups]
        hidden = ["{private}"]
        [model.native-fixture]
        hidden = "none"
        user = "write"
    "#
        ))
        .unwrap(),
    )
}

#[tokio::test]
async fn python_authored_hidden_reference_is_denied_before_mutation() {
    let mut fixture = Fixture::new(
        vec![python_tool(
            "mutate",
            "mutate",
            "PyForbiddenPublic",
            "def probe() -> str:\n    return PyForbiddenPrivate.secret()\n",
        )],
        "+module PyForbiddenPrivate\n+fn secret ()->String\n +return \"PRIVATE_SENTINEL\"\n+end",
    )
    .await;
    fixture.config.permission_config = restricted("PyForbiddenPrivate");
    let response = fixture.run(false).await;
    assert!(response.has_errors);
    assert!(
        fixture
            .config
            .program
            .read()
            .await
            .get_function("PyForbiddenPublic.probe")
            .is_none()
    );
    assert!(
        !serde_json::to_string(&response)
            .unwrap()
            .contains("PRIVATE_SENTINEL")
    );
}

#[tokio::test]
async fn python_effectful_function_tests_use_only_trusted_runtime_mocks() {
    let fixture = Fixture::new(vec![
        python_tool("mutate", "mutate", "PyMockedIO", "@effects(\"io\")\nasync def read_text(path: str) -> str:\n    text: str = await read_file(path)\n    return text\n"),
        python_tool("test", "test", "PyMockedIO", "def test_read() -> None:\n    assert read_text(\"fixture\") == \"hello\"\n"),
        tool("finish", "finish", json!({})),
    ], "").await;
    fixture
        .config
        .meta
        .lock()
        .unwrap()
        .io_mocks
        .push(crate::session::IoMock {
            operation: "read_file".into(),
            patterns: vec!["fixture".into()],
            response: "hello".into(),
        });
    let response = fixture.run(false).await;
    assert_eq!(response.completed, Some(true));
    assert!(!response.has_errors, "{:?}", response.results);
    assert_eq!(
        response
            .test_results
            .iter()
            .filter(|test| test.pass)
            .count(),
        1
    );
}

#[tokio::test]
async fn python_source_roundtrips_after_session_and_library_reload() {
    let fixture = Fixture::new(
        vec![
            python_tool(
                "mutate",
                "mutate",
                "PyPersisted",
                include_str!("../../examples/python/basic.py"),
            ),
            python_tool(
                "test",
                "test",
                "PyPersisted",
                include_str!("../../examples/python/test_basic.py"),
            ),
            tool("finish", "finish", json!({})),
        ],
        "",
    )
    .await;
    let library = crate::library::LibraryState::new();
    fixture.config.meta.lock().unwrap().library_state = Some(library.clone());
    assert!(!fixture.run(false).await.has_errors);
    let program = fixture.config.program.read().await.clone();
    let printed = crate::python_frontend::print_function(&program, "PyPersisted.add_two").unwrap();
    let session = crate::session::Session {
        program: program.clone(),
        runtime: fixture.config.runtime.read().unwrap().clone(),
        meta: fixture.config.meta.lock().unwrap().clone(),
        sandbox: None,
    };
    let mut restored: crate::session::Session =
        serde_json::from_str(&serde_json::to_string(&session).unwrap()).unwrap();
    restored.program.rebuild_function_index();
    assert_eq!(
        printed,
        crate::python_frontend::print_function(&restored.program, "PyPersisted.add_two").unwrap()
    );

    let module = program
        .modules
        .iter()
        .find(|module| module.name == "PyPersisted")
        .unwrap();
    crate::library::persist_module(module, Some(&library)).unwrap();
    let mut reloaded = crate::ast::Program::default();
    crate::library::reload_module(&mut reloaded, "PyPersisted", Some(&library)).unwrap();
    assert_eq!(
        printed,
        crate::python_frontend::print_function(&reloaded, "PyPersisted.add_two").unwrap()
    );

    let lowered = crate::python_frontend::lower_module(&printed, "PyPersisted").unwrap();
    let mut reparsed = crate::ast::Program::default();
    for operation in lowered.operations {
        crate::validator::apply_and_validate(&mut reparsed, &operation).unwrap();
    }
    assert_eq!(
        printed,
        crate::python_frontend::print_function(&reparsed, "PyPersisted.add_two").unwrap()
    );
    let baseline = reloaded.clone();
    let mut ws = fixture.config.snapshot_working_set().await;
    ws.program = reloaded;
    let tested = crate::api::native_tools::execute_native_tool("test", &json!({
        "language":"python","module":"PyPersisted","source":include_str!("../../examples/python/test_basic.py")
    }).to_string(), &fixture.config, &mut ws, &baseline).await;
    assert!(!tested.execution.has_errors, "{}", tested.content);
    let evaluated = crate::api::native_tools::execute_native_tool(
        "evaluate",
        r#"{"language":"python","function":"PyPersisted.add_two","input":"3"}"#,
        &fixture.config,
        &mut ws,
        &baseline,
    )
    .await;
    assert!(!evaluated.execution.has_errors && evaluated.content.contains('5'));
}

#[tokio::test]
async fn python_partial_module_mutation_preserves_unsubmitted_function_identity() {
    let fixture = Fixture::new(vec![
        python_tool("mutate", "mutate", "PyCow", "def changed() -> int:\n    return 2\n"),
        tool("finish", "finish", json!({})),
    ], "+module PyCow\n+fn changed ()->Int\n +return 1\n+end\n+fn untouched (x:Int)->Int\n +let a:Int = x+1\n +let b:Int = a+1\n +return b\n+end").await;
    let before = fixture
        .config
        .program
        .read()
        .await
        .modules
        .iter()
        .find(|module| module.name == "PyCow")
        .unwrap()
        .functions
        .iter()
        .find(|function| function.name == "untouched")
        .unwrap()
        .clone();
    let response = fixture.run(false).await;
    assert_eq!(response.completed, Some(true));
    assert!(!response.has_errors);
    let after = fixture
        .config
        .program
        .read()
        .await
        .modules
        .iter()
        .find(|module| module.name == "PyCow")
        .unwrap()
        .functions
        .iter()
        .find(|function| function.name == "untouched")
        .unwrap()
        .clone();
    assert!(Arc::ptr_eq(&before, &after));
}

#[tokio::test]
async fn python_explicit_cross_module_calls_and_positional_arguments_execute_correctly() {
    let fixture = Fixture::new(vec![
        python_tool("library", "mutate", "PyCrossLibrary", "def weighted(left: int, right: int) -> int:\n    return left * 10 + right\n"),
        python_tool("caller", "mutate", "PyCrossCaller", "def calculate(left: int, right: int) -> int:\n    value: int = PyCrossLibrary.weighted(left, right)\n    return value + 1\n"),
        python_tool("test", "test", "PyCrossCaller", "def test_order() -> None:\n    assert calculate(2, 3) == 24\n"),
        tool("evaluate", "evaluate", json!({"function":"PyCrossCaller.calculate","language":"python","input":"2, 3"})),
        tool("finish", "finish", json!({})),
    ], "").await;
    let response = fixture.run(false).await;
    assert_eq!(response.completed, Some(true));
    assert!(!response.has_errors, "{:?}", response.results);
    assert!(
        response
            .results
            .iter()
            .any(|r| r.success && r.message.contains("24"))
    );
}

#[tokio::test]
async fn python_batch_final_lowering_uses_earlier_created_signatures() {
    let fixture = Fixture::new(vec![response("", vec![
        call("lib", "mutate", json!({"language":"python","module":"PyBatchSignature","source":"def inc(x: int) -> int:\n    return x + 1\n"})),
        call("caller", "mutate", json!({"language":"python","module":"PyBatchCaller","source":"def calculate(x: int) -> int:\n    return PyBatchSignature.inc(x) * 2\n"})),
    ]), tool("evaluate", "evaluate", json!({"language":"python","function":"PyBatchCaller.calculate","input":"2"})),
        tool("finish", "finish", json!({}))], "").await;
    let result = fixture.run(false).await;
    assert!(!result.has_errors, "{:?}", result.results);
    assert_eq!(result.completed, Some(true));
    assert!(
        result
            .results
            .iter()
            .any(|result| result.success && result.message.contains('6'))
    );
}

#[tokio::test]
async fn python_batch_call_mode_checks_use_updated_signature_without_undoing_accepted_edit() {
    let fixture = Fixture::new(vec![response("", vec![
        call("make-async", "mutate", json!({"language":"python","module":"PyBatchCallModeTarget","source":"async def value(x: int) -> Result[int]:\n    return Ok(x)\n"})),
        call("bad-caller", "mutate", json!({"language":"python","module":"PyBatchCallModeCaller","source":"def probe() -> int:\n    match PyBatchCallModeTarget.value(1):\n        case Ok(v):\n            return v\n        case Err(e):\n            return 0\n"})),
    ]), tool("finish", "finish", json!({}))],
        "+module PyBatchCallModeTarget\n+fn value (x:Int)->Result<Int>\n +return Ok(x)\n+end\n").await;
    let result = fixture.run(false).await;
    assert!(result.has_errors && result.results.iter().any(|r| !r.success && r.message.contains("annotated await")), "{:?}", result.results);
    assert_eq!(result.completed, Some(true), "a corrected/refused call does not roll back an earlier accepted edit");
    let program = fixture.config.program.read().await;
    assert!(program.get_function("PyBatchCallModeTarget.value").unwrap().effects.contains(&crate::ast::Effect::Async));
    assert!(program.get_function("PyBatchCallModeCaller.probe").is_none());
}

#[tokio::test]
async fn python_batch_preflight_checks_the_whole_source_before_any_effect() {
    let fixture = Fixture::new(vec![response("", vec![
        call("good", "mutate", json!({"language":"python","module":"PyBatchMustNotExist","source":"def value() -> int:\n    return 1\n"})),
        call("bad", "mutate", json!({"language":"python","module":"PyBatchBadSyntax","source":"def invalid() -> int:\n    value: int = Future.function(1)\n    return (lambda: 1)()\n"})),
    ])], "").await;
    assert!(fixture.run(false).await.has_errors);
    assert!(
        fixture
            .config
            .program
            .read()
            .await
            .get_function("PyBatchMustNotExist.value")
            .is_none()
    );
}

#[tokio::test]
async fn python_reference_batch_resolves_target_constructor_from_earlier_call() {
    let mut fixture = Fixture::new(vec![response("", vec![
        call("type", "mutate", json!({"language":"python","module":"PyRoleBatch","source":"@union\nclass Expr:\n    Leaf: tuple[int]\n    Empty: tuple[()]\n"})),
        call("function", "mutate", json!({"language":"python","module":"PyRoleBatch","source":"def score() -> int:\n    value: Expr = Leaf(7)\n    match value:\n        case Leaf(n):\n            return n\n        case Empty():\n            return 0\n"})),
    ]), tool("eval", "evaluate", json!({"language":"python","function":"PyRoleBatch.score","input":""})), tool("finish", "finish", json!({}))], "").await;
    fixture.config.permission_config = Arc::new(toml::from_str("[groups]\nown=['PyRoleBatch']\n[model.native-fixture]\nown='write'\nuser='none'\n").unwrap());
    let result = fixture.run(false).await;
    assert!(!result.has_errors && result.completed == Some(true), "{:?}", result.results);
    assert!(result.results.iter().any(|r| r.success && r.message.contains("= 7")));
}

#[tokio::test]
async fn python_reference_dynamic_call_shape_rejects_batch_before_first_effect() {
    let fixture = Fixture::new(vec![response("", vec![
        call("sentinel", "mutate", json!({"language":"python","module":"PyRoleMustNotExist","source":"def value() -> int:\n    return 1\n"})),
        call("invalid", "mutate", json!({"language":"python","module":"PyRoleInvalid","source":"def f(p: Point) -> int:\n    return p.hidden()\n"})),
    ])], "").await;
    assert!(fixture.run(false).await.has_errors);
    assert!(fixture.config.program.read().await.get_function("PyRoleMustNotExist.value").is_none());
}

/// A config whose live Program and library state come from a private
/// manifest-managed library with one module.
async fn managed_fixture(outputs: Vec<Value>) -> (Fixture, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("library-sources.json"),
        r#"{"version":1,"modules":{"PyListed":{"format":"python","dependencies":[]}}}"#,
    )
    .unwrap();
    std::fs::write(dir.path().join("PyListed.py"), "def one() -> int:\n    return 1\n").unwrap();
    let fixture = Fixture::new(outputs, "").await;
    let mut program = crate::ast::Program::default();
    let state = crate::library::load_library_from_dir_for_tests(&mut program, dir.path());
    state.require_authoritative_sources().unwrap();
    program.require_modules = true;
    *fixture.config.program.write().await = program;
    fixture.config.meta.lock().unwrap().library_state = Some(state);
    (fixture, dir)
}

#[tokio::test]
async fn managed_native_python_mutate_creates_a_module_the_writer_enrols() {
    // #128: the way a conversation creates a module under a manifest.
    let (fixture, dir) = managed_fixture(vec![
        python_tool("mutate", "mutate", "PyEnrolled", include_str!("../../examples/python/basic.py")),
        python_tool("test", "test", "PyEnrolled", include_str!("../../examples/python/test_basic.py")),
        tool("finish", "finish", json!({})),
    ])
    .await;
    let response = fixture.run(false).await;
    assert_eq!(response.completed, Some(true), "{}", response.reply);
    assert!(!response.has_errors, "{:?}", response.results);
    assert!(
        response.results.iter().any(|r| r.success
            && r.message.contains("module `PyEnrolled` accepted in memory")
            && r.message.contains("added to the source manifest with its first source save")),
        "{:?}",
        response.results
    );
    assert!(!dir.path().join("PyEnrolled.py").exists(), "the turn itself writes no source");
    let state = fixture.config.meta.lock().unwrap().library_state.clone().unwrap();
    assert!(state.pending_source_names().contains("PyEnrolled"));
    let receipt =
        crate::library::flush_live_sources_to_dir_for_tests(fixture.config.program.clone(), state.clone(), dir.path()).await;
    assert!(receipt.failures.is_empty(), "{:?}", receipt.failures);
    assert_eq!(receipt.enrolled, vec!["PyEnrolled".to_string()]);
    assert!(state.pending_source_names().is_empty());
    // A new process: manifest + source restore the module and its earned cases.
    let mut restarted = crate::ast::Program::default();
    let restarted_state = crate::library::load_library_from_dir_for_tests(&mut restarted, dir.path());
    restarted_state.require_authoritative_sources().unwrap();
    let function = restarted.get_function("PyEnrolled.add_two").unwrap();
    assert!(!function.tests.is_empty() && function.tests.iter().all(|case| case.passed));
}

#[tokio::test]
async fn managed_native_adapsis_mutate_cannot_create_a_module() {
    let (fixture, dir) = managed_fixture(vec![
        tool("mutate", "mutate", json!({"source":"+module PyAxFresh\n+fn one ()->Int\n  +return 1\n+end\n"})),
        tool("finish", "finish", json!({})),
    ])
    .await;
    let before = serde_json::to_string(&*fixture.config.program.read().await).unwrap();
    let revision = fixture.config.meta.lock().unwrap().revision;
    let response = fixture.run(false).await;
    assert!(response.has_errors);
    assert!(
        response.results.iter().any(|r| !r.success
            && r.message.contains("`PyAxFresh` was not created")
            && r.message.contains("this library stores Python source; create the module with a Python mutation")),
        "{:?}",
        response.results
    );
    assert_eq!(serde_json::to_string(&*fixture.config.program.read().await).unwrap(), before);
    assert_eq!(fixture.config.meta.lock().unwrap().revision, revision);
    let state = fixture.config.meta.lock().unwrap().library_state.clone().unwrap();
    assert!(state.pending_source_names().is_empty());
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 2);
}
