//! A refused turn (`finish_reason` content_filter / refusal) is reported on the
//! output so request accounting can skip it. MEASURED 2026-11-10: a refused
//! request reports a cache write that the next identical request cannot read.
use super::*;

async fn provider(streaming_reason: &'static str) -> (String, tokio::task::JoinHandle<()>) {
    let app = axum::Router::new().route(
        "/v1/chat/completions",
        axum::routing::post(move |axum::Json(body): axum::Json<serde_json::Value>| async move {
            use axum::response::IntoResponse;
            let usage = serde_json::json!({"prompt_tokens":2,"completion_tokens":1,"total_tokens":3,
                "cache_read_input_tokens":0,"cache_creation_input_tokens":5000});
            if body["stream"] == true {
                let first = serde_json::json!({"choices":[{"delta":{"content":"no"},"finish_reason":null}]});
                let last = serde_json::json!({"choices":[{"delta":{},"finish_reason":streaming_reason}],"usage":usage});
                ([(axum::http::header::CONTENT_TYPE, "text/event-stream")],
                 format!("data: {first}\n\ndata: {last}\n\ndata: [DONE]\n\n")).into_response()
            } else {
                axum::Json(serde_json::json!({"choices":[{"message":{"content":"no"},"finish_reason":streaming_reason}],"usage":usage})).into_response()
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    (url, tokio::spawn(async move { axum::serve(listener, app).await.unwrap() }))
}

#[tokio::test]
async fn refused_turns_are_flagged_on_streaming_and_non_streaming_outputs() {
    for (reason, refused) in [("content_filter", true), ("refusal", true), ("stop", false), ("length", false)] {
        let (url, server) = provider(reason).await;
        let client = LlmClient::new_with_model_and_key(&url, "dev-bot", None);
        let messages = vec![ChatMessage::system("s"), ChatMessage::user("q")];
        let streamed = client.generate(messages.clone()).await.unwrap();
        assert_eq!(streamed.refused, refused, "streaming {reason}");
        assert_eq!(streamed.usage.cache_creation_input_tokens, Some(5_000), "usage is still reported");
        // Channel streaming (ask_stream).
        let mut rx = client.generate_streaming(messages.clone()).await.unwrap();
        let mut done = None;
        while let Some(chunk) = rx.recv().await {
            if let StreamChunk::Done(output) = chunk {
                done = Some(output);
            }
        }
        assert_eq!(done.expect("stream finished").refused, refused, "channel {reason}");
        // Non-streaming request.
        let http = reqwest::Client::new();
        let request = LlmRequest { messages, temperature: -1.0, max_tokens: 16 };
        let plain = client.backend.generate_non_streaming(&http, &request).await.unwrap();
        assert_eq!(plain.refused, refused, "non-streaming {reason}");
        server.abort();
    }
    assert!(!is_refusal(None));
}

/// `prompt_cache_key`: sent on every request shape of a client that has one
/// (streaming, channel streaming, non-streaming, native), and absent otherwise.
#[tokio::test]
async fn prompt_cache_key_is_sent_on_every_request_shape_when_set() {
    let bodies = std::sync::Arc::new(std::sync::Mutex::new(Vec::<serde_json::Value>::new()));
    let recorded = bodies.clone();
    let app = axum::Router::new().route(
        "/v1/chat/completions",
        axum::routing::post(move |axum::Json(body): axum::Json<serde_json::Value>| {
            let recorded = recorded.clone();
            async move {
                use axum::response::IntoResponse;
                let streaming = body["stream"] == true;
                recorded.lock().unwrap().push(body);
                if streaming {
                    let chunk = serde_json::json!({"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]});
                    let end = serde_json::json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]});
                    ([(axum::http::header::CONTENT_TYPE, "text/event-stream")],
                     format!("data: {chunk}\n\ndata: {end}\n\ndata: [DONE]\n\n")).into_response()
                } else {
                    axum::Json(serde_json::json!({"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]})).into_response()
                }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let messages = vec![ChatMessage::system("s"), ChatMessage::user("q")];
    let http = reqwest::Client::new();
    for key in [Some("adapsis-conv-0123456789abcdef0123456789abcdef"), None] {
        bodies.lock().unwrap().clear();
        let mut client = LlmClient::new_with_model_and_key(&url, "chatgpt/gpt-6-astra", None);
        if let Some(key) = key {
            client = client.with_prompt_cache_key(key.to_string());
        }
        client.generate(messages.clone()).await.unwrap();
        let mut rx = client.generate_streaming(messages.clone()).await.unwrap();
        while rx.recv().await.is_some() {}
        let request = LlmRequest { messages: messages.clone(), temperature: -1.0, max_tokens: 16 };
        client.backend.generate_non_streaming(&http, &request).await.unwrap();
        let turn = || TurnRequest { messages: messages.clone(), temperature: -1.0, max_tokens: 16, tools: vec![] };
        client.complete_turn(turn()).await.unwrap();
        let mut events = client.stream_turn(turn()).await.unwrap();
        while events.recv().await.is_some() {}
        let bodies = bodies.lock().unwrap();
        assert_eq!(bodies.len(), 5, "five request shapes");
        for (index, body) in bodies.iter().enumerate() {
            assert_eq!(body.get("prompt_cache_key").and_then(serde_json::Value::as_str), key, "shape {index}");
        }
    }
    server.abort();
}
