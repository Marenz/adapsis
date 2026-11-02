//! Scripted offline acceptance: completed-work recovery, not in-flight replay.
use super::*;

const BUGGY: &str = "def total(price: int, quantity: int, fee: int) -> int:\n    product: int = price * quantity\n    subtotal: int = product + fee\n    return subtotal + 1\n";
const TESTS: &str = "def test_total() -> None:\n    assert total(7, 3, 2) == 23\n\ndef test_zero() -> None:\n    assert total(0, 3, 2) == 2\n";

fn python_args(module: &str, source: &str) -> Value {
    json!({"language":"python", "module":module, "source":source})
}

fn evaluate(id: &str, module: &str) -> Value {
    tool(
        id,
        "evaluate",
        json!({"language":"python", "function":format!("{module}.total"), "input":"7, 3, 2"}),
    )
}

fn history(config: &AppConfig) -> Vec<Value> {
    config
        .meta
        .lock()
        .unwrap()
        .conversations
        .get("main")
        .unwrap()
        .to_llm_messages()
        .iter()
        .map(|m| serde_json::to_value(m).unwrap())
        .collect()
}

fn envelope(messages: &[Value], id: &str) -> Value {
    let matches: Vec<_> = messages
        .iter()
        .filter(|m| m["role"] == "tool" && m["tool_call_id"] == id)
        .collect();
    assert_eq!(
        matches.len(),
        1,
        "missing/duplicate result for {id}: {messages:?}"
    );
    serde_json::from_str(matches[0]["content"].as_str().unwrap()).unwrap()
}

fn assert_call(messages: &[Value], id: &str, name: &str, arguments: Value) {
    let calls: Vec<_> = messages
        .iter()
        .filter_map(|m| m["tool_calls"].as_array())
        .flatten()
        .filter(|c| c["id"] == id)
        .collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["function"]["name"], name);
    let actual: Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(actual, arguments);
}

fn assert_tests(messages: &[Value], id: &str, pass: bool) {
    let result = envelope(messages, id);
    assert_eq!(result["tool"], "test");
    assert_eq!(result["success"], pass, "{result}");
    assert_eq!(result["has_errors"], !pass);
    let tests = result["test_results"].as_array().unwrap();
    assert_eq!(tests.len(), 2, "{result}");
    assert!(tests.iter().all(|t| t["pass"] == pass), "{result}");
    if !pass {
        // These must be assertion failures after execution, not parse/type refusals.
        for (test, expected) in tests.iter().zip(["24, expected 23", "3, expected 2"]) {
            assert_eq!(
                test["message"]
                    .as_str()
                    .unwrap()
                    .split_once(" => ")
                    .unwrap()
                    .1,
                expected
            );
        }
    }
}

fn assert_evaluation(messages: &[Value], id: &str, module: &str) {
    let result = envelope(messages, id);
    assert_eq!(result["tool"], "evaluate");
    assert_eq!(result["success"], true, "{result}");
    assert_eq!(
        result["eval_results"],
        format!("eval {module}.total() = 23")
    );
}

async fn untouched(config: &AppConfig, module: &str) -> Arc<crate::ast::FunctionDecl> {
    config
        .program
        .read()
        .await
        .modules
        .iter()
        .find(|m| m.name == module)
        .unwrap()
        .functions
        .iter()
        .find(|f| f.name == "untouched")
        .unwrap()
        .clone()
}

async fn repair(streaming: bool, module: &str, collision: bool) {
    let fixed = BUGGY.replace("return subtotal + 1", "return subtotal");
    let mut prelude = format!("+module {module}\n+fn untouched ()->Int\n +return 91\n+end");
    if collision {
        prelude.push_str(&format!("\n+module {module}Desk\n+shared seed:Int = 7\n+fn restart ()->String [io,async]\n +return \"SCALAR_RESTART_SENTINEL\"\n+end"));
    }
    let fixture = Fixture::new(
        vec![
            tool("bug", "mutate", python_args(module, BUGGY)),
            tool("red", "test", python_args(module, TESTS)),
            tool("early", "finish", json!({})),
            tool("repair", "mutate", python_args(module, &fixed)),
            tool("green", "test", python_args(module, TESTS)),
            evaluate("value", module),
            response(
                "Repaired and verified: total is 23.",
                vec![call("done", "finish", json!({}))],
            ),
        ],
        &prelude,
    )
    .await;
    let before = untouched(&fixture.config, module).await;
    let result = fixture.run(streaming).await;
    assert_eq!(result.completed, Some(true), "{:?}", result.results);
    assert!(result.has_errors, "historical failures remain visible");
    assert!(result.reply.contains("Repaired and verified: total is 23."));
    assert!(Arc::ptr_eq(
        &before,
        &untouched(&fixture.config, module).await
    ));
    let messages = history(&fixture.config);
    assert_call(&messages, "bug", "mutate", python_args(module, BUGGY));
    assert_call(&messages, "repair", "mutate", python_args(module, &fixed));
    for id in ["red", "green"] {
        assert_call(&messages, id, "test", python_args(module, TESTS));
    }
    assert_tests(&messages, "red", false);
    assert_tests(&messages, "green", true);
    for id in ["bug", "repair"] {
        assert_eq!(envelope(&messages, id)["success"], true);
    }
    assert_eq!(envelope(&messages, "early")["done_accepted"], false);
    assert_eq!(envelope(&messages, "early")["success"], false);
    assert_eq!(envelope(&messages, "done")["done_accepted"], true);
    assert_evaluation(&messages, "value", module);
    let requests = fixture.requests.lock().unwrap().clone();
    assert_eq!(requests.len(), 7);
    for (index, id) in ["bug", "red", "early", "repair", "green", "value"]
        .iter()
        .enumerate()
    {
        assert_eq!(
            envelope(requests[index + 1]["messages"].as_array().unwrap(), id),
            envelope(&messages, id)
        );
    }
    recover(&fixture, streaming, module, messages).await;
}

async fn recover(fixture: &Fixture, streaming: bool, module: &str, messages: Vec<Value>) {
    let session = crate::session::Session {
        program: fixture.config.program.read().await.clone(),
        runtime: fixture.config.runtime.read().unwrap().clone(),
        meta: fixture.config.meta.lock().unwrap().clone(),
        sandbox: None,
    };
    let source =
        crate::python_frontend::print_function(&session.program, &format!("{module}.total"))
            .unwrap();
    let tests = serde_json::to_value(
        &session
            .program
            .get_function(&format!("{module}.total"))
            .unwrap()
            .tests,
    )
    .unwrap();
    assert_eq!(tests.as_array().unwrap().len(), 2);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("completed.json");
    session.save(&path).unwrap();
    let loaded = crate::session::Session::load(&path).unwrap();
    assert_eq!(
        source,
        crate::python_frontend::print_function(&loaded.program, &format!("{module}.total"))
            .unwrap()
    );
    assert_eq!(
        tests,
        serde_json::to_value(
            &loaded
                .program
                .get_function(&format!("{module}.total"))
                .unwrap()
                .tests
        )
        .unwrap()
    );
    let fresh = Fixture::new(
        vec![
            tool("retest", "test", python_args(module, TESTS)),
            evaluate("revalue", module),
            tool("redone", "finish", json!({})),
        ],
        "",
    )
    .await;
    *fresh.config.program.write().await = loaded.program;
    *fresh.config.runtime.write().unwrap() = loaded.runtime;
    *fresh.config.meta.lock().unwrap() = loaded.meta;
    assert_eq!(history(&fresh.config), messages);
    // Exercise saved tests before any resubmission can reinstall them.
    assert_autoretest(&fresh.config, streaming, module).await;
    let before = fresh
        .config
        .program
        .read()
        .await
        .get_function(&format!("{module}.total"))
        .unwrap()
        .body
        .clone();
    let result = fresh.run(streaming).await;
    assert_eq!(result.completed, Some(true));
    assert!(!result.has_errors, "{:?}", result.results);
    let recovered = history(&fresh.config);
    assert_tests(&recovered, "retest", true);
    assert_evaluation(&recovered, "revalue", module);
    assert_eq!(envelope(&recovered, "redone")["done_accepted"], true);
    assert_eq!(
        format!("{before:?}"),
        format!(
            "{:?}",
            fresh
                .config
                .program
                .read()
                .await
                .get_function(&format!("{module}.total"))
                .unwrap()
                .body
        )
    );
    let requests = fresh.requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    let replay = requests[0]["messages"].as_array().unwrap();
    for id in ["bug", "red", "early", "repair", "green", "value", "done"] {
        assert_eq!(envelope(replay, id), envelope(&messages, id));
    }
}

async fn assert_autoretest(config: &AppConfig, streaming: bool, module: &str) {
    // Preserves the LAST case (zero -> 2) but breaks the FIRST (7*3+2 -> 23).
    let bad = BUGGY.replace("return subtotal + 1", "return fee");
    let fixture = Fixture::new(
        vec![
            tool("regress", "mutate", python_args(module, &bad)),
            evaluate("retained", module),
            tool("retained-done", "finish", json!({})),
        ],
        "",
    )
    .await;
    *fixture.config.program.write().await = config.program.read().await.clone();
    *fixture.config.runtime.write().unwrap() = config.runtime.read().unwrap().clone();
    *fixture.config.meta.lock().unwrap() = config.meta.lock().unwrap().clone();
    let result = fixture.run(streaming).await;
    assert!(result.has_errors);
    let messages = history(&fixture.config);
    assert_eq!(envelope(&messages, "regress")["success"], false);
    assert_evaluation(&messages, "retained", module);
    assert_eq!(result.completed, Some(true));
}

#[tokio::test]
async fn native_python_repair_all_cases_required_and_later_submission_replaces() {
    let module = "RepairTestSets";
    let fixture = Fixture::new(
        vec![
            tool(
                "create",
                "mutate",
                python_args(module, &BUGGY.replace("return subtotal + 1", "return fee")),
            ),
            tool("mixed", "test", python_args(module, TESTS)),
            tool("refused", "finish", json!({})),
            tool(
                "establish",
                "test",
                python_args(module, &TESTS.replace("== 23", "== 2")),
            ),
            tool(
                "replace",
                "test",
                python_args(
                    module,
                    "def test_replacement():\n    assert total(5, 4, 6) == 6\n",
                ),
            ),
            tool("accepted", "finish", json!({})),
        ],
        "",
    )
    .await;
    let result = fixture.run(false).await;
    assert!(result.has_errors);
    assert_eq!(result.completed, Some(true));
    let messages = history(&fixture.config);
    let mixed = envelope(&messages, "mixed");
    assert_eq!(
        mixed["test_results"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["pass"].clone())
            .collect::<Vec<_>>(),
        [json!(false), json!(true)]
    );
    assert_eq!(envelope(&messages, "refused")["done_accepted"], false);
    assert_tests(&messages, "establish", true);
    let program = fixture.config.program.read().await;
    let tests = &program
        .get_function(&format!("{module}.total"))
        .unwrap()
        .tests;
    assert_eq!(tests.len(), 1);
    assert_eq!(tests[0].expected, "6");
}

#[tokio::test]
async fn native_python_repair_http_and_completed_session_recovery() {
    repair(false, "RepairHttp", false).await;
}

#[tokio::test]
async fn native_python_repair_sse_and_completed_session_recovery() {
    repair(true, "RepairSse", false).await;
}

#[tokio::test]
async fn native_python_scalar_repair_http_with_unrelated_restart() {
    repair(false, "ScalarRepairHttp", true).await;
}

#[tokio::test]
async fn native_python_scalar_repair_sse_with_unrelated_restart() {
    repair(true, "ScalarRepairSse", true).await;
}

async fn denied_batch(streaming: bool, authored: bool, module: &str) {
    let bad = if authored {
        call(
            "denied",
            "mutate",
            python_args(
                module,
                "def probe() -> str:\n    return RepairVault.peek()\n",
            ),
        )
    } else {
        call(
            "denied",
            "evaluate",
            json!({"language":"python", "function":"RepairVault.peek"}),
        )
    };
    let mut fixture = Fixture::new(
        vec![
            response(
                "",
                vec![
                    call(
                        "permitted",
                        "mutate",
                        python_args(module, "def permitted() -> int:\n    return 1\n"),
                    ),
                    bad,
                ],
            ),
            tool("after-denial", "finish", json!({})),
        ],
        "+module RepairVault\n+fn peek ()->String\n +return \"REPAIR_VAULT_SENTINEL\"\n+end",
    )
    .await;
    fixture.config.permission_config = Arc::new(toml::from_str(
        "[groups]\nsecret=['RepairVault']\n[model.native-fixture]\nsecret='none'\nuser='write'\n"
    ).unwrap());
    let result = fixture.run(streaming).await;
    assert!(result.has_errors);
    assert!(
        fixture
            .config
            .program
            .read()
            .await
            .get_function(&format!("{module}.permitted"))
            .is_none()
    );
    assert!(
        fixture
            .config
            .program
            .read()
            .await
            .get_function(&format!("{module}.probe"))
            .is_none()
    );
    let output = serde_json::to_string(&result).unwrap();
    let reason = if authored {
        "permission denied for authored reference `RepairVault.peek` (need Execute)"
    } else {
        "permission denied: cannot evaluate function 'RepairVault.peek' (module 'RepairVault' needs Execute)"
    };
    let messages = history(&fixture.config);
    for id in ["permitted", "denied"] {
        assert_eq!(
            envelope(&messages, id),
            json!({"success":false, "error":format!("Batch skipped before execution: {reason}")})
        );
    }
    assert!(!output.contains("REPAIR_VAULT_SENTINEL"));
    let requests = fixture.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(
        !serde_json::to_string(&*requests)
            .unwrap()
            .contains("REPAIR_VAULT_SENTINEL")
    );
}

#[tokio::test]
async fn native_python_repair_authority_preflight_http_and_sse() {
    denied_batch(false, false, "RepairDeniedEvalHttp").await;
    denied_batch(false, true, "RepairDeniedSourceHttp").await;
    denied_batch(true, false, "RepairDeniedEvalSse").await;
    denied_batch(true, true, "RepairDeniedSourceSse").await;
}
