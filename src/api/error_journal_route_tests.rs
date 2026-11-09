//! Route handler failures feed the runtime error journal (watchdog), redacted.
use super::*;

fn journal_entries_since(seq: u64) -> Vec<serde_json::Value> {
    let value: serde_json::Value = serde_json::from_str(&crate::error_journal::query(seq)).unwrap();
    value["entries"].as_array().unwrap().clone()
}

fn journal_last_seq() -> u64 {
    let value: serde_json::Value = serde_json::from_str(&crate::error_journal::query(u64::MAX - 1)).unwrap();
    value["last_seq"].as_u64().unwrap()
}

async fn post(router: axum::Router, path: &str, body: &str) -> axum::http::StatusCode {
    router
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri(path)
                .body(axum::body::Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
        .status()
}

#[tokio::test]
async fn route_handler_error_and_missing_handler_are_journaled() {
    let config = test_config();
    {
        let mut program = config.program.write().await;
        // The handler interpolates the request body (and a token URL) into its error,
        // so the guarantee must come from dispatch, not from handler hygiene.
        let source = "+module JournalRoute\n+fn boom (body:String)->String\n  +return to_string(to_int(concat(body, \" https://api.example.org/bot1234567:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/x\")))\n+end\n";
        for op in crate::parser::parse(source).unwrap() {
            crate::validator::apply_and_validate(&mut program, &op).unwrap();
        }
        program.rebuild_function_index();
    }
    {
        let mut runtime = config.runtime.write().unwrap();
        for (path, handler) in [("/journal/boom", "JournalRoute.boom"), ("/journal/gone", "JournalRoute.gone")] {
            runtime.http_routes.push(crate::ast::HttpRoute {
                method: "POST".into(),
                path: path.into(),
                handler_fn: handler.into(),
            });
        }
    }
    let before = journal_last_seq();
    let router = router_with_llm(config.clone());
    let chat_text = "ROUTE_BODY_CANARY_77 private chat text must not be journaled";
    assert_eq!(post(router.clone(), "/journal/boom", chat_text).await, axum::http::StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(post(router, "/journal/gone", chat_text).await, axum::http::StatusCode::INTERNAL_SERVER_ERROR);
    let entries = journal_entries_since(before);
    let boom = entries
        .iter()
        .find(|e| e["category"] == "route" && e["source"] == "JournalRoute.boom")
        .unwrap_or_else(|| panic!("no handler error entry: {entries:#?}"));
    let message = boom["message"].as_str().unwrap();
    assert!(message.starts_with("POST /journal/boom: "), "{message}");
    assert!(message.contains("<request body>"), "body was not interpolated or not replaced: {message}");
    assert!(!message.contains("ROUTE_BODY_CANARY_77"), "{message}");
    assert!(!message.contains("AAAAAAAAAAAAAAAA") && !message.contains("1234567"), "{message}");
    assert!(!message.contains('\n'), "first line only: {message}");
    assert!(entries.iter().any(|e| e["category"] == "route" && e["source"] == "JournalRoute.gone"));
    assert!(entries.iter().all(|e| !e.to_string().contains("ROUTE_BODY_CANARY_77")), "request body journaled");
}
