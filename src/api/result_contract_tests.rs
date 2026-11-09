use super::*;

const ERROR_SOURCE: &str = "+module CCoreApi\n+fn application ()->Result<String>\n  +return Err(\"division by zero\")\n+end\n+fn text ()->String\n  +return \"division by zero\"\n+end\n+fn broken_result ()->Result<Int>\n  +return 1 / 0\n+end\n+fn broken_scalar ()->Int [fail]\n  +return 1 / 0\n+end\n+fn effect ()->Int [io,async]\n  +await before:Bool = file_exists(\"once\")\n  +return 1 / 0\n+end\n";

#[tokio::test]
async fn named_inline_and_session_eval_distinguish_operational_error_from_same_text_data() {
    let config = test_config();
    {
        let mut program = config.program.write().await;
        for op in crate::parser::parse(ERROR_SOURCE).unwrap() {
            crate::validator::apply_and_validate(&mut program, &op).unwrap();
        }
    }
    let created = create_session(
        State(config.clone()),
        Json(CreateSessionRequest {
            session_id: "c-contract".into(),
        }),
    )
    .await;
    assert_eq!(created.0, axum::http::StatusCode::CREATED);
    let router = router_with_llm(config);
    http_json(
        router.clone(),
        "POST",
        "/api/sessions/c-contract/mutate",
        Some(serde_json::json!({"source":ERROR_SOURCE})),
    )
    .await;
    for endpoint in ["/api/eval", "/api/sessions/c-contract/eval"] {
        for (name, success) in [
            ("application", true),
            ("text", true),
            ("broken_result", false),
            ("broken_scalar", false),
        ] {
            for body in [
                serde_json::json!({"function":format!("CCoreApi.{name}")}),
                serde_json::json!({"function":"","expression":format!("CCoreApi.{name}()")}),
            ] {
                let result = http_json(router.clone(), "POST", endpoint, Some(body)).await;
                assert_eq!(result["success"], success, "{endpoint}/{name}: {result}");
                assert!(
                    result["result"]
                        .as_str()
                        .unwrap()
                        .contains("division by zero"),
                    "{result}"
                );
            }
        }
    }
}

#[tokio::test]
async fn executed_failure_after_io_reports_failure_once_without_program_writeback() {
    let mut config = test_config();
    {
        let mut program = config.program.write().await;
        for op in crate::parser::parse(ERROR_SOURCE).unwrap() {
            crate::validator::apply_and_validate(&mut program, &op).unwrap();
        }
    }
    let before = serde_json::to_value(&*config.program.read().await).unwrap();
    let (sender, mut receiver) = tokio::sync::mpsc::channel(4);
    config.io_sender = Some(sender);
    let responder = tokio::spawn(async move {
        let crate::coroutine::IoRequest::FileExists { path, reply } =
            receiver.recv().await.unwrap()
        else {
            panic!("only FileExists")
        };
        assert_eq!(path, "once");
        reply.send(Ok(true)).unwrap();
        receiver
    });
    let result = http_json(
        router_with_llm(config.clone()),
        "POST",
        "/api/eval",
        Some(serde_json::json!({"function":"CCoreApi.effect"})),
    )
    .await;
    assert_eq!(result["success"], false, "{result}");
    assert!(result["result"]
        .as_str()
        .unwrap()
        .contains("division by zero"));
    assert_eq!(
        before,
        serde_json::to_value(&*config.program.read().await).unwrap()
    );
    let mut receiver = responder.await.unwrap();
    assert!(
        receiver.try_recv().is_err(),
        "executed failure must not replay earlier IO"
    );
}

#[tokio::test]
async fn route_declared_result_is_normalized_and_operational_error_stays_http_failure() {
    let config = test_config();
    let source = "+module CCoreRoute\n+fn respond (payload:String)->Result<String> [fail]\n  +check valid payload != \"bad\" ~bad_input\n  +return payload\n+end\n+fn broken (payload:String)->Result<String>\n  +return to_string(1 / 0)\n+end\n";
    {
        let mut program = config.program.write().await;
        for operation in crate::parser::parse(source).unwrap() {
            crate::validator::apply_and_validate(&mut program, &operation).unwrap();
        }
    }
    for (path, handler) in [
        ("/c-result", "CCoreRoute.respond"),
        ("/c-broken", "CCoreRoute.broken"),
    ] {
        config
            .runtime
            .write()
            .unwrap()
            .http_routes
            .push(crate::ast::HttpRoute {
                method: "POST".into(),
                path: path.into(),
                handler_fn: handler.into(),
            });
    }
    let router = router_with_llm(config);
    for (path, body, status, expected) in [
        ("/c-result", "hi", 200, "Ok(\"hi\")"),
        ("/c-result", "bad", 200, "Err(bad_input)"),
        ("/c-broken", "hi", 500, "division by zero"),
    ] {
        let request = axum::http::Request::builder()
            .method("POST")
            .uri(path)
            .body(axum::body::Body::from(body))
            .unwrap();
        let response = router.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status().as_u16(), status);
        let bytes = axum::body::to_bytes(response.into_body(), 65536)
            .await
            .unwrap();
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        assert!(text.contains(expected), "{text}");
    }
    // Application Err HTTP policy is deliberately unchanged: this is not an ACK/status
    // reinterpretation. Operational Rust Err still follows the existing HTTP 500 path.
}
