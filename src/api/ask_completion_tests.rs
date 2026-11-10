use super::*;
use std::sync::{Arc, Mutex};

async fn scripted_ask(outputs: Vec<&str>, budget: usize) -> (AskResponse, Vec<serde_json::Value>) {
    scripted_ask_with_existing_program(outputs, budget, "").await
}

async fn scripted_ask_with_existing_program(outputs: Vec<&str>, budget: usize, prelude: &str) -> (AskResponse, Vec<serde_json::Value>) {
    scripted_turn(outputs, budget, prelude, false).await
}

async fn scripted_turn(outputs: Vec<&str>, budget: usize, prelude: &str, streaming: bool) -> (AskResponse, Vec<serde_json::Value>) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let recorded = requests.clone();
    let outputs: Vec<String> = outputs.into_iter().map(str::to_owned).collect();
    let app = axum::Router::new().route("/v1/chat/completions", axum::routing::post(
        move |Json(body): Json<serde_json::Value>| {
            let recorded = recorded.clone();
            let outputs = outputs.clone();
            async move {
                let mut requests = recorded.lock().unwrap();
                let index = requests.len();
                requests.push(body);
                let content = outputs.get(index).expect("unexpected extra LLM request");
                let chunk = serde_json::json!({"choices":[{"delta":{"content":content}}]});
                ([(axum::http::header::CONTENT_TYPE, "text/event-stream")],
                 format!("data: {chunk}\n\ndata: [DONE]\n\n"))
            }
        }
    ));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let mut config = test_config();
    config.llm_url = format!("http://{address}");
    config.max_iterations = budget;
    *config.llm_model.write().unwrap() = "fixture".into();
    if !prelude.is_empty() {
        let mut session = config.snapshot_working_set().await;
        let result = execute_code(prelude, &config, &mut session, None).await;
        assert!(!result.has_errors, "{:?}", result.mutation_results);
        config.write_back_working_set(&session).await;
    }
    config.program.write().await.require_modules = true;
    let response = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        let request = Json(AskRequest {
            message: "Inspect and report".into(), caller: None,
        });
        if !streaming {
            return super::super::llm_handlers::ask(State(config), request).await.0;
        }
        let response = super::super::llm_handlers::ask_stream(State(config), request).await.into_response();
        let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let body = String::from_utf8(bytes.to_vec()).unwrap();
        let events: Vec<serde_json::Value> = body.lines().filter_map(|line| {
            line.strip_prefix("data:").and_then(|data| serde_json::from_str(data.trim()).ok())
        }).collect();
        let results: Vec<MutationResult> = events.iter().filter(|e| e["type"] == "result")
            .map(|e| MutationResult {
                message: e["message"].as_str().unwrap_or_default().into(),
                success: e["success"].as_bool().unwrap_or(false),
            }).collect();
        AskResponse {
            completed: None,
            has_errors: results.iter().any(|r| !r.success) || events.iter().any(|e| e["type"] == "error"),
            reply: body, code: String::new(), results, test_results: vec![],
        }
    }).await.expect("scripted ask timed out");
    server.abort();
    let requests = requests.lock().unwrap().clone();
    (response, requests)
}

#[tokio::test]
async fn prose_then_empty_resumes_without_repeating_an_operation() {
    let (response, requests) = scripted_ask(vec![
        "<code>!eval 41 + 1</code>", "I will inspect next.", "",
        "The answer is 42.<code>!done</code>",
    ], 4).await;
    assert!(!response.has_errors);
    assert_eq!(requests.len(), 4);
    assert_eq!(response.results.len(), 1, "execute the operation exactly once");
    assert!(response.results[0].message.contains("42"));
    for request in &requests[2..] {
        let messages = request["messages"].as_array().unwrap();
        assert!(messages.last().unwrap()["content"].as_str().unwrap()
            .contains("without repeating completed operations"));
        assert!(messages.iter().any(|m| m["content"].as_str()
            .is_some_and(|s| s.contains("Results:") && s.contains("42"))));
    }
}

#[tokio::test]
async fn clarification_with_explicit_done_finishes_immediately() {
    let (response, requests) = scripted_ask(vec!["Which room?<code>!done</code>"], 3).await;
    assert_eq!(requests.len(), 1);
    assert!(!response.has_errors);
    assert!(response.reply.contains("Which room?"));
}

#[tokio::test]
async fn repeated_prose_exhausts_budget_truthfully() {
    let (response, requests) = scripted_ask(vec!["Working.", "Still working."], 2).await;
    assert_eq!(requests.len(), 2);
    assert!(response.has_errors);
    assert!(response.reply.contains("completion is unconfirmed"));
}

#[tokio::test]
async fn last_operation_is_not_a_completion_claim() {
    let (response, requests) = scripted_ask(vec!["<code>!eval 41 + 1</code>"], 1).await;
    assert_eq!(requests.len(), 1);
    assert!(response.results[0].success);
    assert!(response.has_errors);
    assert!(response.reply.contains("completion is unconfirmed"));
}

const OLD_DEBT: &str = "+module CompletionDebt\n+fn pending (x:Int)->Int\n  +let a:Int = x+1\n  +let b:Int = a+1\n  +return b\n+end\n";

#[test]
fn takeover_completion_uses_executor_acceptance_not_raw_done_text() {
    // Full takeover requires the memory/identity service; the behavioral tests
    // below exercise the shared gate, while this guard protects its call site.
    let source = include_str!("llm_handlers.rs");
    let takeover = source.split("let tmp_config = AppConfig {").nth(1).unwrap();
    assert!(!takeover.contains("if code == \"!done\""));
    assert!(takeover.contains("if exec_result.done_accepted && !exec_result.has_errors"));
    assert!(takeover.contains("Some(&completion_baseline)"));
}

#[tokio::test]
async fn mixed_inspection_and_done_ignore_unrelated_existing_test_debt() {
    let (response, requests) = scripted_ask_with_existing_program(
        vec!["<code>?symbols\n!done</code>"], 2, &OLD_DEBT.replace("CompletionDebt", "CompletionReadDebt"),
    ).await;
    assert_eq!(requests.len(), 1);
    assert!(!response.has_errors, "{:?}", response.results);
}

#[tokio::test]
async fn changed_functions_still_owe_tests_across_rounds() {
    let replacement = "<code>+module CompletionDebt\n+fn pending (x:Int)->Int\n  +let a:Int = x+2\n  +let b:Int = a+2\n  +return b\n+end\n</code>";
    let (response, requests) = scripted_ask_with_existing_program(
        vec![replacement, "<code>!done</code>"], 2, OLD_DEBT,
    ).await;
    assert_eq!(requests.len(), 2);
    assert!(response.has_errors);
    assert!(response.results.iter().any(|r| !r.success && r.message.contains("CompletionDebt.pending")));
}

#[tokio::test]
async fn streaming_inspection_completion_ignores_existing_debt() {
    let (response, requests) = scripted_turn(
        vec!["I will inspect.", "<code>?symbols\n!done</code>"], 2,
        &OLD_DEBT.replace("CompletionDebt", "CompletionStreamReadDebt"), true,
    ).await;
    assert_eq!(requests.len(), 2);
    assert!(!response.has_errors, "{}", response.reply);
}

#[tokio::test]
async fn streaming_new_function_still_requires_tests() {
    let code = format!("<code>{}</code>", OLD_DEBT.replace("CompletionDebt", "CompletionStreamNewDebt"));
    let (response, requests) = scripted_turn(vec![&code, "<code>!done</code>"], 2, "", true).await;
    assert_eq!(requests.len(), 2);
    assert!(response.has_errors);
    assert!(response.results.iter().any(|r| !r.success && r.message.contains("CompletionStreamNewDebt.pending")));
}

/// `/api/ask` and `/api/ask-stream` across failed turns: every request's cache
/// boundaries must be acceptable to the gateway. The `main` epoch tracker is
/// process-global and shared with other tests, so whether a given request also
/// carries the turn-boundary marker is not asserted here; the rule holds for
/// every boundary index (context_log tests) and deterministically through the
/// takeover loop (takeover_native tests).
#[tokio::test]
async fn ask_and_ask_stream_requests_after_failed_turns_have_gateway_valid_cache_boundaries() {
    for streaming in [false, true] {
        let script: Vec<Option<&str>> = vec![
            Some("first answer<code>!done</code>"), None, None,
            Some("recovered<code>!done</code>"), Some("still fine<code>!done</code>"),
        ];
        let requests = Arc::new(Mutex::new(Vec::<serde_json::Value>::new()));
        let recorded = requests.clone();
        let app = axum::Router::new().route("/v1/chat/completions", axum::routing::post(
            move |Json(body): Json<serde_json::Value>| {
                let recorded = recorded.clone();
                let script = script.clone();
                async move {
                    let mut requests = recorded.lock().unwrap();
                    let index = requests.len();
                    requests.push(body);
                    match script.get(index).copied().flatten() {
                        Some(content) => {
                            let chunk = serde_json::json!({"choices":[{"delta":{"content":content}}]});
                            ([(axum::http::header::CONTENT_TYPE, "text/event-stream")],
                             format!("data: {chunk}\n\ndata: [DONE]\n\n")).into_response()
                        }
                        // A provider failure that is not retried.
                        None => (axum::http::StatusCode::BAD_REQUEST, r#"{"error":{"message":"scripted failure"}}"#).into_response(),
                    }
                }
            }
        ));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let mut config = test_config();
        config.llm_url = format!("http://{address}");
        config.max_iterations = 2;
        *config.llm_model.write().unwrap() = "fixture".into();
        for turn in 0..5 {
            let request = Json(AskRequest { message: format!("question {turn}"), caller: None });
            tokio::time::timeout(std::time::Duration::from_secs(10), async {
                if streaming {
                    let response = super::super::llm_handlers::ask_stream(State(config.clone()), request).await.into_response();
                    let _ = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
                } else {
                    let _ = super::super::llm_handlers::ask(State(config.clone()), request).await;
                }
            }).await.expect("ask turn timed out");
        }
        server.abort();
        let requests = requests.lock().unwrap().clone();
        assert_eq!(requests.len(), 5, "streaming={streaming}: one request per turn");
        for (index, request) in requests.iter().enumerate() {
            let messages = request["messages"].as_array().unwrap();
            assert_eq!(crate::context_log::gateway_boundary_errors(messages), Vec::<String>::new(), "streaming={streaming} request {index}");
        }
        // `ask` stores an assistant entry for a failed turn too, so its history
        // keeps alternating; every boundary is the stable prefix or an assistant
        // message, never a user message next to the runtime envelope.
        let roles: Vec<&str> = requests[3]["messages"].as_array().unwrap().iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(roles, ["system", "user", "assistant", "user", "assistant", "user", "assistant", "user"], "streaming={streaming}");
        for request in &requests[1..] {
            let marked: Vec<&str> = request["messages"].as_array().unwrap().iter()
                .filter(|m| m.get("cache_control").is_some()).map(|m| m["role"].as_str().unwrap()).collect();
            assert!(marked.len() >= 2 && marked[0] == "system" && marked[1..].iter().all(|role| *role == "assistant"), "streaming={streaming}: {marked:?}");
        }
    }
}
