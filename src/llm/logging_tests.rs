use super::*;

#[tokio::test]
async fn malformed_provider_shape_logging_uses_actual_deserialization_error() {
    if !crate::logging::tests::child() {
        for enabled in [false, true] {
            let captured = crate::logging::tests::capture(
                "llm::logging_tests::malformed_provider_shape_logging_uses_actual_deserialization_error", enabled);
            assert_eq!(captured.contains("CANARY_STREAM_SHAPE"), enabled);
            assert_eq!(captured.contains("CANARY_HTTP_BODY"), enabled);
        }
        return;
    }
    use axum::response::IntoResponse;
    for api_error in [false, true] {
        let app = axum::Router::new().route("/v1/chat/completions",
            axum::routing::post(move || async move {
                if api_error {
                    (axum::http::StatusCode::BAD_REQUEST, "CANARY_HTTP_BODY").into_response()
                } else {
                    ([(axum::http::header::CONTENT_TYPE, "text/event-stream")],
                        "data: {\"choices\":\"CANARY_STREAM_SHAPE\"}\n\n").into_response()
                }
            }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let backend = OpenAiBackend::new(&format!("http://{}", listener.local_addr().unwrap()));
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let request = LlmRequest { messages: vec![ChatMessage::user("hello")],
            temperature: -1.0, max_tokens: 10 };
        let error = backend.generate_streaming_into(&Client::new(), &request, Vec::new())
            .await.err().expect("wrong shape/non-2xx must fail");
        let original = error.downcast_ref::<LlmError>().unwrap().with_body();
        let canary = if api_error { "CANARY_HTTP_BODY" } else { "CANARY_STREAM_SHAPE" };
        assert!(original.contains(canary), "premise: original error must contain provider data");
        assert!(!error.to_string().contains(canary), "safe user-facing diagnostic");
        let logged = render_error(&error);
        assert_eq!(logged.contains(canary), crate::logging::content());
        eprintln!("{logged}");
        server.abort();
    }
}
