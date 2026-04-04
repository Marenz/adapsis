use super::*;
use axum::response::IntoResponse;

/// Helper: build a minimal AppConfig for testing multi-session endpoints.
fn test_config() -> AppConfig {
    let (trigger_tx, _trigger_rx) = tokio::sync::mpsc::channel::<String>(1);
    AppConfig {
        program: std::sync::Arc::new(tokio::sync::RwLock::new(crate::ast::Program::default())),
        meta: std::sync::Arc::new(std::sync::Mutex::new(crate::session::SessionMeta::new())),
        llm_url: String::new(),
        llm_model: String::new(),
        llm_api_key: None,
        project_dir: ".".to_string(),
        io_sender: None,
        self_trigger: trigger_tx,
        task_registry: None,
        snapshot_registry: None,
        log_file: None,
        training_log: None,
        jit_cache: crate::eval::new_jit_cache(),
        event_broadcast: tokio::sync::broadcast::channel(16).0,
        opencode_git_dir: ".".to_string(),
        opencode_lock: std::sync::Arc::new(tokio::sync::Mutex::new(())),
        message_queue: std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new())),
        max_iterations: 1,
        runtime: std::sync::Arc::new(std::sync::RwLock::new(
            crate::session::RuntimeState::default(),
        )),
        sessions: std::sync::Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new())),
    }
}

async fn recv_event(rx: &mut tokio::sync::broadcast::Receiver<String>) -> serde_json::Value {
    let raw = rx.recv().await.unwrap();
    let value: serde_json::Value =
        serde_json::from_str(&raw).unwrap_or_else(|_| serde_json::json!({"raw": raw.clone()}));
    if let Some(encoded) = value.as_str() {
        serde_json::from_str(encoded).unwrap_or(value)
    } else {
        value
    }
}

#[test]
fn collect_opencode_tasks_keeps_all_in_order() {
    let ops = crate::parser::parse(
        "!opencode First task\n!opencode Second task\n+fn hi ()->Int\n  +return 1\n",
    )
    .unwrap();
    let tasks = collect_opencode_tasks(&ops);
    assert_eq!(
        tasks,
        vec!["First task".to_string(), "Second task".to_string()]
    );
}

#[test]
fn collect_opencode_tasks_ignores_non_opencode_ops() {
    let ops = crate::parser::parse("+fn hi ()->Int\n  +return 1\n!eval hi\n").unwrap();
    let tasks = collect_opencode_tasks(&ops);
    assert!(tasks.is_empty());
}

#[tokio::test]
async fn events_stream_sets_sse_headers() {
    let config = test_config();
    let response = events_stream(State(config)).await.into_response();
    assert_eq!(
        response.headers()[axum::http::header::CONTENT_TYPE],
        "text/event-stream"
    );
    assert_eq!(
        response.headers()[axum::http::header::CACHE_CONTROL],
        "no-cache"
    );
    assert_eq!(
        response.headers()[axum::http::header::CONNECTION],
        "keep-alive"
    );
}

// ═════════════════════════════════════════════════════════════════════
// GET /api/sessions — list sessions
// ═════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn list_sessions_includes_main() {
    let config = test_config();
    let Json(result) = list_sessions(State(config)).await;
    let ids = result.as_array().unwrap();
    assert!(
        ids.iter().any(|v| v.as_str() == Some("main")),
        "list should always include 'main'"
    );
}

#[tokio::test]
async fn list_sessions_includes_created_sessions() {
    let config = test_config();
    // Create a session directly in the map
    config.sessions.lock().await.insert(
        "test-session".to_string(),
        std::sync::Arc::new(tokio::sync::Mutex::new(crate::ast::Program::default())),
    );

    let Json(result) = list_sessions(State(config)).await;
    let ids: Vec<&str> = result
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert!(ids.contains(&"main"));
    assert!(ids.contains(&"test-session"));
}

// ═════════════════════════════════════════════════════════════════════
// POST /api/sessions — create session
// ═════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn create_session_happy_path() {
    let config = test_config();
    let (status, Json(body)) = create_session(
        State(config.clone()),
        Json(CreateSessionRequest {
            session_id: "my-session".to_string(),
        }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::CREATED);
    assert_eq!(body["session_id"], "my-session");
    assert_eq!(body["status"], "created");

    // Verify it's in the map
    let sessions = config.sessions.lock().await;
    assert!(sessions.contains_key("my-session"));
}

#[tokio::test]
async fn create_session_empty_id_rejected() {
    let config = test_config();
    let (status, Json(body)) = create_session(
        State(config),
        Json(CreateSessionRequest {
            session_id: "  ".to_string(),
        }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .contains("must not be empty")
    );
}

#[tokio::test]
async fn create_session_main_reserved() {
    let config = test_config();
    let (status, Json(body)) = create_session(
        State(config),
        Json(CreateSessionRequest {
            session_id: "main".to_string(),
        }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::CONFLICT);
    assert!(body["error"].as_str().unwrap().contains("reserved"));
}

#[tokio::test]
async fn create_session_duplicate_rejected() {
    let config = test_config();
    // Create first
    create_session(
        State(config.clone()),
        Json(CreateSessionRequest {
            session_id: "dup".to_string(),
        }),
    )
    .await;
    // Create duplicate
    let (status, Json(body)) = create_session(
        State(config),
        Json(CreateSessionRequest {
            session_id: "dup".to_string(),
        }),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::CONFLICT);
    assert!(body["error"].as_str().unwrap().contains("already exists"));
}

// ═════════════════════════════════════════════════════════════════════
// DELETE /api/sessions/:id — delete session
// ═════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn delete_session_happy_path() {
    let config = test_config();
    // Create then delete
    create_session(
        State(config.clone()),
        Json(CreateSessionRequest {
            session_id: "to-delete".to_string(),
        }),
    )
    .await;
    let (status, Json(body)) = delete_session(
        State(config.clone()),
        axum::extract::Path("to-delete".to_string()),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::OK);
    assert_eq!(body["status"], "deleted");

    // Verify removed
    let sessions = config.sessions.lock().await;
    assert!(!sessions.contains_key("to-delete"));
}

#[tokio::test]
async fn delete_session_main_rejected() {
    let config = test_config();
    let (status, Json(body)) =
        delete_session(State(config), axum::extract::Path("main".to_string())).await;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST);
    assert!(body["error"].as_str().unwrap().contains("cannot delete"));
}

#[tokio::test]
async fn delete_session_not_found() {
    let config = test_config();
    let (status, Json(body)) = delete_session(
        State(config),
        axum::extract::Path("nonexistent".to_string()),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::NOT_FOUND);
    assert!(body["error"].as_str().unwrap().contains("not found"));
}

// ═════════════════════════════════════════════════════════════════════
// POST /api/sessions/:id/eval — eval in session
// ═════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn session_eval_not_found() {
    let config = test_config();
    let Json(response) = session_eval(
        State(config),
        axum::extract::Path("nonexistent".to_string()),
        Json(EvalRequest {
            function: "foo".to_string(),
            input: String::new(),
            expression: None,
        }),
    )
    .await;
    assert!(!response.success);
    assert!(response.result.contains("not found"));
}

#[tokio::test]
async fn session_eval_inline_expression() {
    let config = test_config();
    // Create a session
    create_session(
        State(config.clone()),
        Json(CreateSessionRequest {
            session_id: "eval-test".to_string(),
        }),
    )
    .await;

    // Eval an inline expression (no functions needed)
    let Json(response) = session_eval(
        State(config),
        axum::extract::Path("eval-test".to_string()),
        Json(EvalRequest {
            function: String::new(),
            input: String::new(),
            expression: Some("1 + 2".to_string()),
        }),
    )
    .await;
    assert!(response.success, "eval should succeed: {}", response.result);
    assert_eq!(response.result, "3");
}

#[tokio::test]
async fn session_eval_function_in_session() {
    let config = test_config();
    // Create session
    create_session(
        State(config.clone()),
        Json(CreateSessionRequest {
            session_id: "fn-test".to_string(),
        }),
    )
    .await;

    // Add a function via session_mutate
    session_mutate(
        State(config.clone()),
        axum::extract::Path("fn-test".to_string()),
        Json(MutateRequest {
            source: "+fn double (x:Int)->Int\n  +return x * 2\n".to_string(),
        }),
    )
    .await;

    // Eval the function
    let Json(response) = session_eval(
        State(config),
        axum::extract::Path("fn-test".to_string()),
        Json(EvalRequest {
            function: "double".to_string(),
            input: "5".to_string(),
            expression: None,
        }),
    )
    .await;
    assert!(response.success, "eval should succeed: {}", response.result);
    assert_eq!(response.result, "10");
}

// ═════════════════════════════════════════════════════════════════════
// POST /api/sessions/:id/mutate — mutate in session
// ═════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn session_mutate_happy_path() {
    let config = test_config();
    create_session(
        State(config.clone()),
        Json(CreateSessionRequest {
            session_id: "mut-test".to_string(),
        }),
    )
    .await;

    let Json(response) = session_mutate(
        State(config.clone()),
        axum::extract::Path("mut-test".to_string()),
        Json(MutateRequest {
            source: "+fn greet ()->String\n  +return \"hello\"\n".to_string(),
        }),
    )
    .await;
    assert!(
        response.results.iter().all(|r| r.success),
        "mutate should succeed: {:?}",
        response.results
    );

    // Verify the function exists in the session
    let sessions = config.sessions.lock().await;
    let program = sessions.get("mut-test").unwrap().lock().await;
    assert!(program.get_function("greet").is_some());
}

#[tokio::test]
async fn session_mutate_not_found() {
    let config = test_config();
    let Json(response) = session_mutate(
        State(config),
        axum::extract::Path("nonexistent".to_string()),
        Json(MutateRequest {
            source: "+fn greet ()->String\n  +return \"hello\"\n".to_string(),
        }),
    )
    .await;
    assert!(response.results.iter().any(|r| !r.success));
    assert!(response.results[0].message.contains("not found"));
}

// ═════════════════════════════════════════════════════════════════════
// Isolation: sessions have independent Programs
// ═════════════════════════════════════════════════════════════════════

#[tokio::test]
async fn sessions_are_isolated() {
    let config = test_config();

    // Create two sessions
    create_session(
        State(config.clone()),
        Json(CreateSessionRequest {
            session_id: "session-a".to_string(),
        }),
    )
    .await;
    create_session(
        State(config.clone()),
        Json(CreateSessionRequest {
            session_id: "session-b".to_string(),
        }),
    )
    .await;

    // Add a function to session-a only
    session_mutate(
        State(config.clone()),
        axum::extract::Path("session-a".to_string()),
        Json(MutateRequest {
            source: "+fn only_in_a ()->Int\n  +return 42\n".to_string(),
        }),
    )
    .await;

    // session-a should have the function
    let Json(resp_a) = session_eval(
        State(config.clone()),
        axum::extract::Path("session-a".to_string()),
        Json(EvalRequest {
            function: "only_in_a".to_string(),
            input: String::new(),
            expression: None,
        }),
    )
    .await;
    assert!(resp_a.success, "session-a should have the function");

    // session-b should NOT have it
    let Json(resp_b) = session_eval(
        State(config),
        axum::extract::Path("session-b".to_string()),
        Json(EvalRequest {
            function: "only_in_a".to_string(),
            input: String::new(),
            expression: None,
        }),
    )
    .await;
    assert!(
        !resp_b.success,
        "session-b should not have session-a's function"
    );
}

#[tokio::test]
async fn sync_async_side_effects_preserves_live_roadmap_and_runtime_state() {
    let config = test_config();
    let mut session = config.snapshot_working_set().await;

    session.meta.plan.push(crate::session::PlanStep {
        description: "keep local plan".to_string(),
        status: crate::session::PlanStatus::Pending,
    });
    config.write_back_working_set(&session).await;

    {
        let mut meta = config.meta.lock().unwrap();
        crate::session::roadmap_add(&mut meta.roadmap, "#30 Sync GitHub issues");
        meta.io_mocks.push(crate::session::IoMock {
            operation: "http_get".to_string(),
            patterns: vec!["api.github.com".to_string()],
            response: "[]".to_string(),
        });
        crate::session::send_agent_message(&mut meta, "main", "worker", "hello");
    }
    {
        let mut runtime = config.runtime.write().unwrap();
        runtime.http_routes.push(crate::ast::HttpRoute {
            method: "GET".to_string(),
            path: "/health".to_string(),
            handler_fn: "health_check".to_string(),
        });
    }

    config.sync_async_side_effects_into(&mut session);
    config.write_back_working_set(&session).await;

    let meta = config.meta.lock().unwrap();
    assert_eq!(meta.plan.len(), 1, "local plan should survive sync");
    assert_eq!(
        meta.roadmap.len(),
        1,
        "async roadmap additions should survive write-back"
    );
    assert_eq!(
        meta.io_mocks.len(),
        1,
        "async mock mutations should survive write-back"
    );
    assert_eq!(meta.agent_mailbox.get("worker").map(Vec::len), Some(1));
    drop(meta);

    let runtime = config.runtime.read().unwrap();
    assert_eq!(
        runtime.http_routes.len(),
        1,
        "async runtime mutations should survive write-back"
    );
}

#[tokio::test]
async fn mutate_broadcasts_sse_event() {
    let config = test_config();
    let mut rx = config.event_broadcast.subscribe();

    let Json(response) = mutate(
        State(config),
        Json(MutateRequest {
            source: "+fn ping ()->String\n  +return \"pong\"\n+end".to_string(),
        }),
    )
    .await;

    assert!(response.results.iter().all(|r| r.success));
    let event = recv_event(&mut rx).await;
    assert_eq!(event["type"], "mutation");
    assert_eq!(event["revision"], 1);
    assert_eq!(event["summary"], response.results[0].message);
}

#[tokio::test]
async fn mutate_repeated_errors_add_hint() {
    let config = test_config();
    let bad_source = "+fn bad ()->String\n  +let user:String = {first: \"a\" second: \"b\"}\n  +return user\n+end";

    let Json(first) = mutate(
        State(config.clone()),
        Json(MutateRequest {
            source: bad_source.to_string(),
        }),
    )
    .await;
    assert!(!first.results[0].success);
    assert!(!first.results[0].message.contains("Suggestion:"));

    let Json(response) = mutate(
        State(config),
        Json(MutateRequest {
            source: bad_source.to_string(),
        }),
    )
    .await;

    assert!(!response.results[0].success);
    assert!(
        response.results[0].message.contains("Suggestion:"),
        "got: {}",
        response.results[0].message
    );
    assert!(
        response.results[0]
            .message
            .contains("missing commas between fields"),
        "got: {}",
        response.results[0].message
    );
}

#[tokio::test]
async fn eval_broadcasts_sse_event() {
    let config = test_config();
    let mut rx = config.event_broadcast.subscribe();
    let _ = mutate(
        State(config.clone()),
        Json(MutateRequest {
            source: "+fn forty_two ()->Int\n  +return 42\n+end".to_string(),
        }),
    )
    .await;
    let _ = recv_event(&mut rx).await;

    let Json(response) = eval_fn(
        State(config),
        Json(EvalRequest {
            function: "forty_two".to_string(),
            input: "".to_string(),
            expression: None,
        }),
    )
    .await;

    assert!(response.success);
    let event = recv_event(&mut rx).await;
    assert_eq!(event["type"], "eval");
    assert_eq!(event["expression"], "forty_two");
    assert_eq!(event["result"], "42");
}

#[tokio::test]
async fn eval_error_broadcasts_sse_event() {
    let config = test_config();
    let mut rx = config.event_broadcast.subscribe();

    let Json(response) = eval_fn(
        State(config),
        Json(EvalRequest {
            function: String::new(),
            input: String::new(),
            expression: Some(String::new()),
        }),
    )
    .await;

    assert!(!response.success);
    let event = recv_event(&mut rx).await;
    assert_eq!(event["type"], "eval");
    assert_eq!(event["expression"], "");
    assert_eq!(event["result"], "empty expression");
}

#[tokio::test]
async fn test_endpoint_broadcasts_sse_event() {
    let config = test_config();
    let mut rx = config.event_broadcast.subscribe();
    let _ = mutate(
        State(config.clone()),
        Json(MutateRequest {
            source: "+fn one ()->Int\n  +return 1\n+end".to_string(),
        }),
    )
    .await;
    let _ = recv_event(&mut rx).await;

    let Json(response) = test_fn(
        State(config),
        Json(TestRequest {
            source: "!test one\n  +with -> expect 1".to_string(),
        }),
    )
    .await;

    assert_eq!(response.passed, 1);
    let event = recv_event(&mut rx).await;
    assert_eq!(event["type"], "test");
    assert_eq!(event["function"], "one");
    assert_eq!(event["passed"], 1);
    assert_eq!(event["failed"], 0);
}

#[tokio::test]
async fn ui_page_contains_event_source_client() {
    let Html(page) = ui_page().await;
    assert!(page.contains("AdapsisOS Dashboard"));
    assert!(page.contains("textarea"));
    assert!(page.contains("Eval"));
    assert!(page.contains("Apply"));
    assert!(page.contains("/api/events"));
    assert!(page.contains("EventSource"));
    assert!(page.contains("const formatPayload = (payload) =>"));
    assert!(page.contains("(empty event)"));
}

/// Helper: build a test AppConfig with IO sender for async eval tests.
fn test_config_with_io() -> (
    AppConfig,
    tokio::sync::mpsc::Receiver<crate::coroutine::IoRequest>,
) {
    let (io_tx, io_rx) = tokio::sync::mpsc::channel::<crate::coroutine::IoRequest>(32);
    let (trigger_tx, _trigger_rx) = tokio::sync::mpsc::channel::<String>(1);
    let config = AppConfig {
        program: std::sync::Arc::new(tokio::sync::RwLock::new(crate::ast::Program::default())),
        meta: std::sync::Arc::new(std::sync::Mutex::new(crate::session::SessionMeta::new())),
        llm_url: String::new(),
        llm_model: String::new(),
        llm_api_key: None,
        project_dir: ".".to_string(),
        io_sender: Some(io_tx),
        self_trigger: trigger_tx,
        task_registry: None,
        snapshot_registry: None,
        log_file: None,
        training_log: None,
        jit_cache: crate::eval::new_jit_cache(),
        event_broadcast: tokio::sync::broadcast::channel(16).0,
        opencode_git_dir: ".".to_string(),
        opencode_lock: std::sync::Arc::new(tokio::sync::Mutex::new(())),
        message_queue: std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new())),
        max_iterations: 1,
        runtime: std::sync::Arc::new(std::sync::RwLock::new(
            crate::session::RuntimeState::default(),
        )),
        sessions: std::sync::Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new())),
    };
    (config, io_rx)
}

#[tokio::test]
async fn execute_code_eval_io_function_does_not_reject() {
    // Define an [io,async] function and verify !eval doesn't reject it
    let (config, _io_rx) = test_config_with_io();
    let module_code = "!module TestIO\n+fn greet(name:String) -> String [io,async]\n  +return concat(\"hello \", name)\n+end";
    let mut session = config.snapshot_working_set().await;
    // First apply the module
    let res = execute_code(module_code, &config, &mut session, None).await;
    assert!(
        !res.has_errors,
        "module definition should succeed: {:?}",
        res.mutation_results
    );
    config.write_back_working_set(&session).await;

    // Add a test so it's not blocked by "untested" check
    let test_code = "!test TestIO.greet\n  +with name=\"world\" -> expect \"hello world\"";
    let mut session = config.snapshot_working_set().await;
    let res = execute_code(test_code, &config, &mut session, None).await;
    config.write_back_working_set(&session).await;

    // Now eval it — should NOT get "cannot call in test expression"
    let eval_code = "!eval TestIO.greet(name=\"test\")";
    let mut session = config.snapshot_working_set().await;
    let res = execute_code(eval_code, &config, &mut session, None).await;
    // Should not contain "side effects" error
    let has_side_effects_error = res
        .mutation_results
        .iter()
        .any(|r| r.message.contains("side effects"));
    assert!(
        !has_side_effects_error,
        "!eval of [io,async] function should not be rejected. Results: {:?}",
        res.mutation_results
    );
}

#[tokio::test]
async fn execute_code_eval_io_function_positional_args() {
    // Test that !eval Module.func("arg1", 42, "arg3") also works for [io,async] functions
    let (config, _io_rx) = test_config_with_io();
    let module_code = "!module TestIO2\n+fn process(a:String, b:Int, c:String) -> String [io,async]\n  +return concat(a, to_string(b), c)\n+end";
    let mut session = config.snapshot_working_set().await;
    let res = execute_code(module_code, &config, &mut session, None).await;
    assert!(
        !res.has_errors,
        "module definition should succeed: {:?}",
        res.mutation_results
    );
    config.write_back_working_set(&session).await;

    // Add a test
    let test_code = "!test TestIO2.process\n  +with a=\"x\" b=1 c=\"y\" -> expect \"x1y\"";
    let mut session = config.snapshot_working_set().await;
    let res = execute_code(test_code, &config, &mut session, None).await;
    config.write_back_working_set(&session).await;

    // Eval with positional args — should NOT reject
    let eval_code = "!eval TestIO2.process(\"hello\", 42, \"/tmp/out\")";
    let mut session = config.snapshot_working_set().await;
    let res = execute_code(eval_code, &config, &mut session, None).await;
    let has_side_effects_error = res
        .mutation_results
        .iter()
        .any(|r| r.message.contains("side effects"));
    assert!(
        !has_side_effects_error,
        "!eval with positional args should not be rejected for [io,async]. Results: {:?}",
        res.mutation_results
    );
}

#[tokio::test]
async fn execute_code_agent_spawned_flag() {
    let (config, _io_rx) = test_config_with_io();
    let code = "!agent test-agent --scope full Just a test task";
    let mut session = config.snapshot_working_set().await;
    let res = execute_code(code, &config, &mut session, None).await;
    assert!(res.agent_spawned, "agent_spawned should be true");
    assert_eq!(res.spawned_agent_names, vec!["test-agent"]);
}
