//! Too-large request recovery and the shared retry policy (#68): one shrink-and-
//! retry for a request refused as too large, a timeout retried at most once, and
//! the flag that tells callers to persist a forced view checkpoint.
use super::*;

/// Retry policy shared by every request path: retryable errors up to
/// MAX_LLM_RETRIES, but a timeout at most once, so a hung gateway cannot hold a
/// live reply for retries × timeout (#68).
pub(crate) struct RetryPolicy {
    timeouts: usize,
}

impl RetryPolicy {
    pub(crate) fn new() -> Self {
        Self { timeouts: 0 }
    }

    /// Delay before the next attempt, or `None` to stop with the error.
    pub(crate) fn next_delay(&mut self, error: &anyhow::Error, attempt: usize) -> Option<Duration> {
        let llm_error = error.downcast_ref::<LlmError>()?;
        if !llm_error.is_retryable() || attempt + 1 >= MAX_LLM_RETRIES {
            return None;
        }
        if llm_error.is_timeout() {
            self.timeouts += 1;
            if self.timeouts > 1 {
                return None;
            }
        }
        Some(llm_error.retry_after().unwrap_or_else(|| default_retry_delay(attempt, llm_error)))
    }
}

/// Whether `error` is a too-large refusal (pre-send, 413 or context length).
pub fn is_too_large(error: &anyhow::Error) -> bool {
    error.downcast_ref::<LlmError>().is_some_and(LlmError::is_too_large)
}

/// Self-heal for a too-large request (#68): keep the stable system prefix and the
/// current turn (from the last runtime envelope, or the last user message, to the
/// end, so in-turn tool batches are kept), and the newest older history units
/// that fit `target_bytes`; never split an assistant/tool-result batch. Returns
/// `None` when nothing can be dropped or the kept part alone does not fit.
pub fn shrink_request(messages: &[ChatMessage], target_bytes: usize) -> Option<Vec<ChatMessage>> {
    let size = |m: &ChatMessage| serde_json::to_vec(m).map_or(0, |bytes| bytes.len());
    let head = usize::from(messages.first().is_some_and(|m| m.role == ChatRole::System));
    let boundary = messages
        .iter()
        .rposition(crate::context_log::is_runtime_envelope)
        .or_else(|| messages.iter().rposition(|m| m.role == ChatRole::User))?
        .max(head);
    let notice = ChatMessage::user(
        "[System: older history omitted from this request to fit the size limit; the raw history is retained.]",
    );
    let fixed: usize = messages[..head].iter().chain(&messages[boundary..]).map(size).sum::<usize>() + size(&notice);
    if fixed > target_bytes {
        return None;
    }
    // Units of the history between the prefix and the boundary, newest first.
    // A tool result belongs to the assistant message with the tool calls before it.
    let mut units: Vec<std::ops::Range<usize>> = Vec::new();
    let mut end = boundary;
    while end > head {
        let mut start = end - 1;
        if messages[start].role == ChatRole::Tool {
            while start > head && messages[start].role == ChatRole::Tool {
                start -= 1;
            }
        }
        units.push(start..end);
        end = start;
    }
    let mut budget = target_bytes - fixed;
    let mut keep_from = boundary;
    for unit in &units {
        let bytes: usize = messages[unit.clone()].iter().map(size).sum();
        if bytes > budget {
            break;
        }
        budget -= bytes;
        keep_from = unit.start;
    }
    if keep_from == head {
        return None; // nothing to drop
    }
    let mut shrunk = messages[..head].to_vec();
    shrunk.push(notice);
    shrunk.extend_from_slice(&messages[keep_from..]);
    Some(shrunk)
}

impl<B> LlmClient<B> {
    /// Flag a too-large failure so the caller forces a view checkpoint.
    pub fn mark_recovered(&self) {
        self.recovered.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Whether the last call self-healed a too-large request (and reset the flag).
    pub fn take_recovered(&self) -> bool {
        self.recovered.swap(false, std::sync::atomic::Ordering::SeqCst)
    }

    /// One shrink-and-retry for a too-large request. Returns the reduced message
    /// list, or `None` when the error is something else or nothing can be dropped.
    pub(crate) fn shrink_after(&self, error: &anyhow::Error, messages: &[ChatMessage], limit: usize) -> Option<Vec<ChatMessage>> {
        if !is_too_large(error) {
            return None;
        }
        let current: usize = messages.iter().map(|m| serde_json::to_vec(m).map_or(0, |b| b.len())).sum();
        let shrunk = shrink_request(messages, current.min(limit) / 2)?;
        self.recovered.store(true, std::sync::atomic::Ordering::SeqCst);
        eprintln!("[llm:recover] request too large ({current} bytes): retrying once with {} of {} messages", shrunk.len(), messages.len());
        Some(shrunk)
    }
}

#[cfg(test)]
mod request_size_tests {
    use super::super::*;
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    async fn gateway_413() -> (String, std::sync::Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
        let hits = std::sync::Arc::new(AtomicUsize::new(0));
        let counter = hits.clone();
        let app = axum::Router::new().route(
            "/v1/chat/completions",
            axum::routing::post(move || {
                let counter = counter.clone();
                async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    (StatusCode::PAYLOAD_TOO_LARGE, "Failed to buffer the request body: length limit exceeded")
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (url, hits, server)
    }

    #[tokio::test]
    async fn oversized_request_fails_before_sending_and_is_not_retried() {
        let (url, hits, server) = gateway_413().await;
        let mut client = LlmClient::new_with_model_and_key(&url, "dev-bot", None);
        assert_eq!(client.backend.max_request_bytes, 60 * 1024 * 1024, "default route limit");
        client.backend.max_request_bytes = 2_000_000; // a 2 MB route (old gateway)
        let huge = "x".repeat(2_100_000);
        let error = client.generate(vec![ChatMessage::user(huge.clone())]).await.unwrap_err();
        let text = error.to_string();
        assert!(text.contains("LLM request too large") && text.contains("max_request_bytes"), "{text}");
        assert_eq!(render_error(&error), text, "journal sees the same clear class");
        let native = client.complete_turn(TurnRequest {
            messages: vec![ChatMessage::user(huge)], temperature: -1.0, max_tokens: 16, tools: vec![],
        }).await.unwrap_err();
        assert!(native.to_string().contains("LLM request too large"));
        assert_eq!(hits.load(Ordering::SeqCst), 0, "nothing was sent");
        server.abort();
    }

    #[tokio::test]
    async fn gateway_413_surfaces_clearly_once_without_retries() {
        let (url, hits, server) = gateway_413().await;
        let client = LlmClient::new_with_model_and_key(&url, "dev-bot", None);
        let error = client.generate(vec![ChatMessage::user("small")]).await.unwrap_err();
        let text = error.to_string();
        assert!(text.contains("413") && text.contains("exceeds the gateway/provider limit") && text.contains("not retried"), "{text}");
        assert_eq!(hits.load(Ordering::SeqCst), 1, "413 is not retried");
        let native = client.complete_turn(TurnRequest {
            messages: vec![ChatMessage::user("small")], temperature: -1.0, max_tokens: 16, tools: vec![],
        }).await.unwrap_err();
        assert!(native.to_string().contains("413"));
        assert_eq!(hits.load(Ordering::SeqCst), 2, "native path: one more attempt, no retry loop");
        server.abort();
    }
}

#[cfg(test)]
mod recovery_tests {
    use super::super::*;
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Mock provider: records request sizes, answers one SSE completion.
    async fn provider() -> (String, std::sync::Arc<std::sync::Mutex<Vec<usize>>>, tokio::task::JoinHandle<()>) {
        let sizes = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let recorded = sizes.clone();
        let app = axum::Router::new()
            .route("/v1/chat/completions", axum::routing::post(move |body: axum::body::Bytes| {
                let recorded = recorded.clone();
                async move {
                    recorded.lock().unwrap().push(body.len());
                    let chunk = serde_json::json!({"choices":[{"delta":{"content":"ok"}}]});
                    ([(axum::http::header::CONTENT_TYPE, "text/event-stream")], format!("data: {chunk}\n\ndata: [DONE]\n\n"))
                }
            }))
            .layer(axum::extract::DefaultBodyLimit::max(64 * 1024 * 1024));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        (url, sizes, tokio::spawn(async move { axum::serve(listener, app).await.unwrap() }))
    }

    fn envelope(tail_bytes: usize, text: &str) -> ChatMessage {
        ChatMessage::user_parts(vec![
            ChatContentPart::Text { text: "[Adapsis runtime envelope: trusted runtime blocks follow before the final speaker-input text block.]".into() },
            ChatContentPart::Text { text: "t".repeat(tail_bytes) },
            ChatContentPart::Text { text: text.into() },
        ])
    }

    #[tokio::test]
    async fn too_large_request_with_a_big_runtime_tail_self_heals_once() {
        let (url, sizes, server) = provider().await;
        let mut client = LlmClient::new_with_model_and_key(&url, "dev-bot", None);
        client.backend.max_request_bytes = 2 * 1024 * 1024; // a 2 MiB route
        let mut messages = vec![ChatMessage::system("stable")];
        for i in 0..40 {
            messages.push(ChatMessage::user(format!("u{i} {}", "x".repeat(25_000))));
            messages.push(ChatMessage::assistant_with_tool_calls("", vec![ToolCall { id: format!("c{i}"), name: "inspect".into(), arguments: "{}".into() }]));
            messages.push(ChatMessage::tool_result(format!("c{i}"), "r".repeat(25_000)));
        }
        // > 192 KiB runtime tail in the current turn's envelope.
        messages.push(envelope(300 * 1024, "the current question"));
        assert!(serde_json::to_vec(&messages).unwrap().len() > 2 * 1024 * 1024);
        let output = client.generate(messages).await.unwrap();
        assert_eq!(output.text, "ok");
        assert!(client.take_recovered(), "caller is told to force a checkpoint");
        assert!(!client.take_recovered(), "flag resets");
        let sizes = sizes.lock().unwrap().clone();
        assert_eq!(sizes.len(), 1, "the oversized request was never sent; one reduced request was");
        assert!(sizes[0] <= 2 * 1024 * 1024, "{}", sizes[0]);
        server.abort();
    }

    #[test]
    fn shrink_keeps_prefix_current_turn_and_whole_batches() {
        let mut messages = vec![ChatMessage::system("stable")];
        for i in 0..10 {
            messages.push(ChatMessage::user(format!("u{i} {}", "x".repeat(1_000))));
            messages.push(ChatMessage::assistant_with_tool_calls("", vec![ToolCall { id: format!("c{i}"), name: "n".into(), arguments: "{}".into() }]));
            messages.push(ChatMessage::tool_result(format!("c{i}"), "r".repeat(1_000)));
        }
        messages.push(envelope(100, "now"));
        // In-turn work after the envelope is part of the current turn and kept.
        messages.push(ChatMessage::assistant_with_tool_calls("", vec![ToolCall { id: "t".into(), name: "n".into(), arguments: "{}".into() }]));
        messages.push(ChatMessage::tool_result("t", "in-turn"));
        let shrunk = shrink_request(&messages, 6_000).unwrap();
        assert_eq!(shrunk[0].role, ChatRole::System);
        assert!(serde_json::to_string(&shrunk[1]).unwrap().contains("older history omitted"));
        let tail = &shrunk[shrunk.len() - 3..];
        assert!(crate::context_log::is_runtime_envelope(&tail[0]) && tail[2].tool_call_id.as_deref() == Some("t"));
        for (i, m) in shrunk.iter().enumerate() {
            if m.role == ChatRole::Tool {
                assert!(!shrunk[i - 1].tool_calls.is_empty() || shrunk[i - 1].role == ChatRole::Tool, "no orphan tool result");
            }
        }
        assert!(serde_json::to_vec(&shrunk).unwrap().len() <= 6_000 + 1_000);
        assert!(shrink_request(&messages, 100).is_none(), "the current turn alone does not fit");
    }

    #[test]
    fn timeouts_retry_at_most_once_other_transport_errors_keep_the_budget() {
        let error = |timeout: bool| anyhow!(LlmError::Transport { message: "x".into(), retryable: true, timeout });
        let mut policy = RetryPolicy::new();
        assert!(policy.next_delay(&error(true), 0).is_some());
        assert!(policy.next_delay(&error(true), 1).is_none(), "second timeout stops");
        let mut policy = RetryPolicy::new();
        for attempt in 0..MAX_LLM_RETRIES - 1 {
            assert!(policy.next_delay(&error(false), attempt).is_some());
        }
        let too_large = anyhow!(LlmError::RequestTooLarge { bytes: 3, limit: 2 });
        assert!(RetryPolicy::new().next_delay(&too_large, 0).is_none());
        let context = anyhow!(LlmError::Api { status: StatusCode::BAD_REQUEST, body: "prompt is too long: 300000 tokens".into(), retry_after: None });
        assert!(is_too_large(&context));
        let _ = AtomicUsize::new(0).load(Ordering::SeqCst);
    }

    #[test]
    fn non_streaming_requests_have_a_capped_total_timeout() {
        let backend = OpenAiBackend::new("http://127.0.0.1:1").with_model("anthropic/claude-opus-5");
        assert!(crate::model_limits::for_model("anthropic/claude-opus-5").request_timeout > NON_STREAMING_TIMEOUT_CAP);
        assert_eq!(backend.non_streaming_timeout(), NON_STREAMING_TIMEOUT_CAP);
        assert_eq!(NON_STREAMING_TIMEOUT_CAP, Duration::from_secs(1200));
    }

    /// The self-heal retry keeps a contiguous suffix of the request, so its cache
    /// boundaries stay valid for the gateway (markers stay on assistant run ends).
    #[test]
    fn shrunk_requests_keep_gateway_valid_cache_boundaries() {
        let mut request = vec![ChatMessage::system("stable")];
        for i in 0..30 {
            request.push(ChatMessage::user(format!("q{i} {}", "x".repeat(2_000))));
            request.push(ChatMessage::assistant(format!("a{i} {}", "y".repeat(2_000))));
        }
        request.push(ChatMessage::user("failed"));
        request.push(ChatMessage::user("current"));
        let mut log = crate::context_log::ContextLog::from_conversation(request);
        log.set_turn_boundary(40);
        let messages = log.into_messages();
        let wire = |messages: &[ChatMessage]| messages.iter().map(|m| serde_json::to_value(m).unwrap()).collect::<Vec<_>>();
        let marks = |messages: &[ChatMessage]| messages.iter().filter(|m| m.cache_control.is_some()).count();
        assert_eq!(marks(&messages), 3);
        let mut exercised = 0;
        for target in [200_000, 60_000, 20_000, 8_000] {
            if let Some(shrunk) = shrink_request(&messages, target) {
                exercised += 1;
                assert_eq!(crate::context_log::gateway_boundary_errors(&wire(&shrunk)), Vec::<String>::new(), "target {target}");
                assert!(marks(&shrunk) >= 1);
            }
        }
        assert!(exercised >= 2, "the shrink path was exercised: {exercised}");
    }
}

#[cfg(test)]
pub(crate) fn test_too_large_error() -> anyhow::Error {
    anyhow!(LlmError::RequestTooLarge { bytes: 3, limit: 2 })
}

#[cfg(test)]
mod first_headers_tests {
    use super::super::*;

    /// Accepts connections and reads, but never writes a response.
    async fn silent_gateway() -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let mut held = Vec::new();
            loop {
                if let Ok((socket, _)) = listener.accept().await {
                    held.push(socket); // keep it open, never answer
                }
            }
        });
        (url, server)
    }

    fn client(url: &str) -> LlmClient<OpenAiBackend> {
        let mut client = LlmClient::new_with_model_and_key(url, "anthropic/claude-opus-5", None);
        client.backend.chunk_timeout = Duration::from_secs(1);
        client
    }

    #[tokio::test]
    async fn streaming_paths_fail_fast_when_no_response_headers_arrive() {
        let (url, server) = silent_gateway().await;
        // The total is long (allows long streams); only the headers wait is short.
        assert!(crate::model_limits::for_model("anthropic/claude-opus-5").request_timeout > Duration::from_secs(3600));
        let bound = Duration::from_secs(10); // 1 s headers bound, retried once with a 1 s backoff
        let messages = || vec![ChatMessage::user("hello")];

        let started = std::time::Instant::now();
        let error = client(&url).generate_streaming(messages()).await.unwrap_err();
        assert!(started.elapsed() < bound, "generate_streaming took {:?}", started.elapsed());
        assert!(error.downcast_ref::<LlmError>().is_some_and(LlmError::is_timeout), "{error:#}");

        let started = std::time::Instant::now();
        let request = TurnRequest { messages: messages(), temperature: -1.0, max_tokens: 16, tools: vec![] };
        assert!(client(&url).stream_turn(request).await.is_err());
        assert!(started.elapsed() < bound, "stream_turn took {:?}", started.elapsed());

        // The legacy generate path streams first and fails the same way (no fallback
        // to non-streaming for a timeout).
        let started = std::time::Instant::now();
        assert!(client(&url).generate(messages()).await.is_err());
        assert!(started.elapsed() < bound, "generate took {:?}", started.elapsed());
        server.abort();
    }
}
