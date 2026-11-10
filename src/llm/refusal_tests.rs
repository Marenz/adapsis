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
