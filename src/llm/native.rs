//! Native tool-capable OpenAI-compatible transport (issue #78).
//!
//! A parallel, stricter request/response path next to the legacy `generate`
//! family. `complete_turn` and `stream_turn` carry native tool definitions and
//! assemble streamed tool-call fragments into whole [`ToolCall`]s. The legacy
//! `generate`/`generate_streaming`/`LlmRequest` paths are untouched.
//!
//! Strictness is deliberate: the transport can reject a malformed stream but
//! cannot certify upstream truth, so it never guesses missing arguments or
//! finish reasons and never falls back to the legacy path once a stream has
//! begun. Any transport-level failure surfaces as `NativeStreamEvent::Error`
//! (or an `Err` for `complete_turn`), never as a fabricated result.

use anyhow::{Result, anyhow};
use reqwest::Response;
use serde::Deserialize;
use std::collections::HashSet;
use std::time::Duration;
use tracing::{info, warn};

use super::tool_call::{NativeFinishReason, NativeToolDefinition, ToolCall};
use super::{ChatMessage, LlmClient, LlmError, LlmUsage, OpenAiBackend, TokenUsage};

/// Maximum number of distinct tool calls assembled from a single stream.
/// A call-count bound, deliberately not a maximum-index bound: provider tool
/// indices are content-block indices and may be sparse or start nonzero.
const MAX_TOOL_CALLS: usize = 64;
/// Maximum total UTF-8 bytes accumulated across a single call's argument deltas.
const MAX_TOOL_ARG_BYTES: usize = 1_000_000;

/// A native turn request carrying optional tool definitions.
///
/// If `tools` is non-empty the transport advertises them and requests
/// `tool_choice: "auto"`; otherwise it requests `tool_choice: "none"`.
#[derive(Debug, Clone)]
pub struct TurnRequest {
    pub messages: Vec<ChatMessage>,
    pub temperature: f32,
    pub max_tokens: u32,
    pub tools: Vec<NativeToolDefinition>,
}

/// A completed native turn. `content` never runs through the legacy
/// `<code>`-extraction path (`build_output`); it is the raw model text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeResponse {
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: NativeFinishReason,
    pub usage: LlmUsage,
}

/// Incremental events from a native streaming turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeStreamEvent {
    Content(String),
    Thinking(String),
    Done(NativeResponse),
    Error(String),
}

impl LlmClient<OpenAiBackend> {
    /// Run a non-streaming native turn. Returns the assembled response, or an
    /// error for any unsupported/missing finish reason or malformed response.
    pub async fn complete_turn(&self, request: TurnRequest) -> Result<NativeResponse> {
        info!(
            "native LLM complete_turn: {} messages, {} tools, max_tokens={}",
            request.messages.len(),
            request.tools.len(),
            request.max_tokens
        );
        let body = native_request_body(&self.backend, &request, false);

        with_retries(move || {
            let body = body.clone();
            async move {
                let req = self.http.post(self.backend.endpoint()).json(&body);
                let response = self
                    .backend
                    .send_request(req, "failed to send native tool chat completion request")
                    .await?;
                let parsed: NativeCompletionResponse = response.json().await.map_err(|error| {
                    anyhow!(LlmError::Decode(format!(
                        "failed to deserialize native chat completion response: {error}"
                    )))
                })?;
                assemble_native_response(parsed)
            }
        })
        .await
    }

    /// Start a native streaming turn. The HTTP connection is established with
    /// retries; once the stream begins, chunks are emitted with no replay. The
    /// final event is always `Done` or `Error` (never both).
    pub async fn stream_turn(
        &self,
        request: TurnRequest,
    ) -> Result<tokio::sync::mpsc::Receiver<NativeStreamEvent>> {
        info!(
            "native LLM stream_turn: {} messages, {} tools",
            request.messages.len(),
            request.tools.len()
        );
        let body = native_request_body(&self.backend, &request, true);

        with_retries(move || {
            let body = body.clone();
            async move {
                let req = self.http.post(self.backend.endpoint()).json(&body);
                let response = self
                    .backend
                    .send_request(
                        req,
                        "failed to send native streaming chat completion request",
                    )
                    .await?;
                let (tx, rx) = tokio::sync::mpsc::channel::<NativeStreamEvent>(64);
                tokio::spawn(native_stream_task(response, tx));
                Ok::<_, anyhow::Error>(rx)
            }
        })
        .await
    }
}

/// Build the OpenAI-shaped native request body. Reuses the legacy temperature
/// omission convention (negative = unset).
fn native_request_body(
    backend: &OpenAiBackend,
    request: &TurnRequest,
    stream: bool,
) -> serde_json::Value {
    let tools: Vec<serde_json::Value> = request
        .tools
        .iter()
        .map(|tool| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        })
        .collect();

    let mut body = serde_json::json!({
        "model": backend.model,
        "messages": request.messages,
        "max_tokens": request.max_tokens,
        "stream": stream,
    });

    if request.temperature >= 0.0 {
        body["temperature"] = serde_json::json!(request.temperature);
    }

    if tools.is_empty() {
        body["tool_choice"] = serde_json::json!("none");
    } else {
        body["tools"] = serde_json::json!(tools);
        body["tool_choice"] = serde_json::json!("auto");
    }

    body
}

fn parse_native_finish_reason(raw: &str) -> Result<NativeFinishReason> {
    match raw {
        "stop" => Ok(NativeFinishReason::Stop),
        "tool_calls" => Ok(NativeFinishReason::ToolCalls),
        other => Err(anyhow!(
            "unsupported finish_reason '{other}'; expected 'stop' or 'tool_calls'"
        )),
    }
}

fn assemble_native_response(parsed: NativeCompletionResponse) -> Result<NativeResponse> {
    if let Some(error) = parsed.error {
        return Err(anyhow!(provider_error_message(error)));
    }
    if parsed.choices.len() > 1 {
        return Err(anyhow!(
            "native chat completion response contained {} choices (n=1 expected)",
            parsed.choices.len()
        ));
    }
    let choice = parsed
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("native chat completion response contained no choices"))?;

    let finish_reason = choice
        .finish_reason
        .ok_or_else(|| anyhow!("native chat completion response missing finish_reason"))?;
    let finish = parse_native_finish_reason(&finish_reason)?;

    let message = choice.message;
    let tool_calls = message.tool_calls;
    validate_tool_calls(&tool_calls).map_err(anyhow::Error::msg)?;

    match finish {
        NativeFinishReason::Stop if !tool_calls.is_empty() => {
            return Err(anyhow!(
                "finish_reason=stop but response contains tool_calls"
            ));
        }
        NativeFinishReason::ToolCalls if tool_calls.is_empty() => {
            return Err(anyhow!(
                "finish_reason=tool_calls but response contains no tool calls"
            ));
        }
        _ => {}
    }

    Ok(NativeResponse {
        content: message.content.unwrap_or_default(),
        reasoning: message.reasoning_content.unwrap_or_default(),
        tool_calls,
        finish_reason: finish,
        usage: parsed.usage.map(LlmUsage::from).unwrap_or_default(),
    })
}

/// Validate a fully-assembled tool-call list before it can become a
/// `NativeResponse`. Shared by the non-streaming and streaming paths so neither
/// can hand back an oversized, nameless, id-less or duplicate call.
fn validate_tool_calls(calls: &[ToolCall]) -> Result<(), String> {
    if calls.len() > MAX_TOOL_CALLS {
        return Err(format!(
            "tool call count exceeds the limit of {MAX_TOOL_CALLS}"
        ));
    }
    let mut seen_ids = HashSet::new();
    for call in calls {
        if call.id.trim().is_empty() {
            return Err("tool call is missing an id".to_string());
        }
        if !seen_ids.insert(call.id.as_str()) {
            return Err(format!("duplicate tool call id '{}'", call.id));
        }
        if call.name.trim().is_empty() {
            return Err(format!("tool call '{}' is missing a name", call.id));
        }
        if call.arguments.len() > MAX_TOOL_ARG_BYTES {
            return Err(format!(
                "tool call arguments exceed the limit of {MAX_TOOL_ARG_BYTES} bytes"
            ));
        }
    }
    Ok(())
}

/// Render an OpenAI-shaped provider error body into an actionable message.
fn provider_error_message(error: serde_json::Value) -> String {
    if let Some(message) = error.get("message").and_then(serde_json::Value::as_str) {
        return format!("provider error: {message}");
    }
    format!("provider error: {error}")
}

/// Retry a fallible request up to `MAX_LLM_RETRIES` times, reusing the legacy
/// error classification and backoff helpers.
async fn with_retries<T, F, Fut>(mut attempt: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let mut last_err = None;
    for attempt_index in 0..super::MAX_LLM_RETRIES {
        match attempt().await {
            Ok(value) => return Ok(value),
            Err(error) => {
                let Some(llm_error) = error.downcast_ref::<LlmError>() else {
                    return Err(error);
                };
                if llm_error.is_retryable() && attempt_index + 1 < super::MAX_LLM_RETRIES {
                    let delay = llm_error
                        .retry_after()
                        .unwrap_or_else(|| super::default_retry_delay(attempt_index, llm_error));
                    warn!(
                        "[llm:retry] attempt {}/{} in {}ms: {}",
                        attempt_index + 2,
                        super::MAX_LLM_RETRIES,
                        delay.as_millis(),
                        llm_error
                    );
                    tokio::time::sleep(delay).await;
                    last_err = Some(error);
                    continue;
                }
                return Err(error);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| {
        anyhow!(
            "LLM request failed after {} retries",
            super::MAX_LLM_RETRIES
        )
    }))
}

async fn native_stream_task(
    mut response: Response,
    tx: tokio::sync::mpsc::Sender<NativeStreamEvent>,
) {
    let mut parser = NativeSseParser::new();
    let mut state = StreamState::new();
    let chunk_timeout = Duration::from_secs(120);

    loop {
        let chunk = match tokio::time::timeout(chunk_timeout, response.chunk()).await {
            Ok(Ok(Some(chunk))) => chunk,
            Ok(Ok(None)) => break,
            Ok(Err(error)) => {
                let _ = tx
                    .send(NativeStreamEvent::Error(format!(
                        "failed to read streaming response chunk: {error}"
                    )))
                    .await;
                return;
            }
            Err(_) => {
                let _ = tx
                    .send(NativeStreamEvent::Error(
                        "streaming response stalled (no data for 120s)".to_string(),
                    ))
                    .await;
                return;
            }
        };

        for data in parser.feed(&chunk) {
            if data == "[DONE]" {
                match state.finalize() {
                    Ok(response) => {
                        let _ = tx.send(NativeStreamEvent::Done(response)).await;
                    }
                    Err(error) => {
                        let _ = tx.send(NativeStreamEvent::Error(error)).await;
                    }
                }
                return;
            }

            let parsed: NativeChunk = match serde_json::from_str(&data) {
                Ok(parsed) => parsed,
                Err(error) => {
                    let _ = tx
                        .send(NativeStreamEvent::Error(format!(
                            "failed to parse SSE chunk: {error}; chunk={data}"
                        )))
                        .await;
                    return;
                }
            };

            match state.apply_chunk(parsed) {
                Ok(events) => {
                    for event in events {
                        if tx.send(event).await.is_err() {
                            return;
                        }
                    }
                }
                Err(error) => {
                    let _ = tx.send(NativeStreamEvent::Error(error)).await;
                    return;
                }
            }
        }
    }

    let _ = tx
        .send(NativeStreamEvent::Error(
            "stream ended without terminal finish_reason or [DONE]".to_string(),
        ))
        .await;
}

/// Mutable state for one native stream, extracted out of `native_stream_task`
/// so chunk handling is testable and the task stays small.
struct StreamState {
    content: String,
    reasoning: String,
    usage: Option<TokenUsage>,
    terminal: Option<String>,
    accumulator: ToolCallAccumulator,
}

impl StreamState {
    fn new() -> Self {
        Self {
            content: String::new(),
            reasoning: String::new(),
            usage: None,
            terminal: None,
            accumulator: ToolCallAccumulator::new(),
        }
    }

    /// Apply one parsed chunk, returning the incremental `Thinking`/`Content`
    /// events to emit. Rejects provider error frames, multiple choice indices,
    /// action data after a terminal finish reason, and repeated or conflicting
    /// finish reasons.
    fn apply_chunk(&mut self, chunk: NativeChunk) -> Result<Vec<NativeStreamEvent>, String> {
        if let Some(error) = chunk.error {
            return Err(provider_error_message(error));
        }
        if chunk.choices.len() > 1 {
            return Err(format!(
                "unexpected {} completion choices (n=1 expected)",
                chunk.choices.len()
            ));
        }
        if let Some(usage) = chunk.usage {
            self.usage = Some(usage);
        }

        let mut events = Vec::new();
        if let Some(choice) = chunk.choices.into_iter().next() {
            if let Some(index) = choice.index {
                if index != 0 {
                    return Err(format!("unsupported completion choice index {index}"));
                }
            }

            let has_action = choice.delta.reasoning_content.is_some()
                || choice.delta.content.is_some()
                || !choice.delta.tool_calls.is_empty();
            if self.terminal.is_some() && has_action {
                return Err("action data received after terminal finish_reason".to_string());
            }

            if let Some(reasoning) = choice.delta.reasoning_content {
                self.reasoning.push_str(&reasoning);
                events.push(NativeStreamEvent::Thinking(reasoning));
            }
            if let Some(content) = choice.delta.content {
                self.content.push_str(&content);
                events.push(NativeStreamEvent::Content(content));
            }
            for delta in choice.delta.tool_calls {
                self.accumulator.apply(delta)?;
            }

            if let Some(finish_reason) = choice.finish_reason {
                match &self.terminal {
                    None => self.terminal = Some(finish_reason),
                    Some(previous) if previous == &finish_reason => {
                        return Err("repeated terminal finish_reason".to_string());
                    }
                    Some(previous) => {
                        return Err(format!(
                            "conflicting finish_reason '{previous}' then '{finish_reason}'"
                        ));
                    }
                }
            }
        }

        Ok(events)
    }

    /// Finalize the assembled stream into a `NativeResponse`, enforcing the
    /// same strict finish-reason/tool-call invariants as `complete_turn`.
    fn finalize(self) -> Result<NativeResponse, String> {
        let finish_reason = self
            .terminal
            .ok_or_else(|| "stream completed without a terminal finish_reason".to_string())?;
        let finish =
            parse_native_finish_reason(&finish_reason).map_err(|error| error.to_string())?;

        let tool_calls = self.accumulator.finish()?;

        match finish {
            NativeFinishReason::Stop if !tool_calls.is_empty() => {
                return Err("finish_reason=stop but tool_calls were streamed".to_string());
            }
            NativeFinishReason::ToolCalls if tool_calls.is_empty() => {
                return Err("finish_reason=tool_calls but no tool calls were streamed".to_string());
            }
            _ => {}
        }

        Ok(NativeResponse {
            content: self.content,
            reasoning: self.reasoning,
            tool_calls,
            finish_reason: finish,
            usage: self.usage.map(LlmUsage::from).unwrap_or_default(),
        })
    }
}

struct ToolCallAccumulator {
    calls: Vec<AccumulatedCall>,
}

struct AccumulatedCall {
    index: u64,
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl ToolCallAccumulator {
    fn new() -> Self {
        Self { calls: Vec::new() }
    }

    fn apply(&mut self, delta: NativeStreamToolCall) -> Result<(), String> {
        let index = delta
            .index
            .ok_or_else(|| "tool call delta missing index".to_string())?;

        let call = match self.calls.iter_mut().find(|call| call.index == index) {
            Some(call) => call,
            None => {
                if self.calls.len() >= MAX_TOOL_CALLS {
                    return Err(format!(
                        "tool call count exceeds the limit of {MAX_TOOL_CALLS}"
                    ));
                }
                self.calls.push(AccumulatedCall {
                    index,
                    id: None,
                    name: None,
                    arguments: String::new(),
                });
                self.calls.last_mut().expect("just pushed a call")
            }
        };

        if let Some(id) = delta.id {
            match &call.id {
                Some(existing) if existing != &id => {
                    return Err(format!(
                        "tool call index {index} received conflicting ids '{existing}' and '{id}'"
                    ));
                }
                _ => call.id = Some(id),
            }
        }

        if let Some(function) = delta.function {
            if let Some(name) = function.name {
                match &call.name {
                    Some(existing) if existing != &name => {
                        return Err(format!(
                            "tool call index {index} received conflicting names '{existing}' and '{name}'"
                        ));
                    }
                    _ => call.name = Some(name),
                }
            }
            if let Some(arguments) = function.arguments {
                if call.arguments.len().saturating_add(arguments.len()) > MAX_TOOL_ARG_BYTES {
                    return Err(format!(
                        "tool call arguments exceed the limit of {MAX_TOOL_ARG_BYTES} bytes"
                    ));
                }
                call.arguments.push_str(&arguments);
            }
        }

        Ok(())
    }

    fn finish(self) -> Result<Vec<ToolCall>, String> {
        let mut calls = self.calls;
        calls.sort_by_key(|call| call.index);

        let mut seen_ids = HashSet::new();
        let mut assembled = Vec::with_capacity(calls.len());
        for call in calls {
            let id = call
                .id
                .ok_or_else(|| "streamed tool call is missing an id".to_string())?;
            if !seen_ids.insert(id.clone()) {
                return Err(format!("duplicate tool call id '{id}'"));
            }
            let name = call
                .name
                .ok_or_else(|| format!("tool call '{id}' is missing a name"))?;
            assembled.push(ToolCall {
                id,
                name,
                arguments: call.arguments,
            });
        }
        validate_tool_calls(&assembled)?;
        Ok(assembled)
    }
}

#[derive(Deserialize)]
struct NativeCompletionResponse {
    #[serde(default)]
    choices: Vec<NativeChoice>,
    #[serde(default)]
    usage: Option<TokenUsage>,
    #[serde(default)]
    error: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct NativeChoice {
    #[serde(default)]
    finish_reason: Option<String>,
    message: NativeMessage,
}

#[derive(Deserialize)]
struct NativeMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<ToolCall>,
}

#[derive(Deserialize)]
struct NativeChunk {
    #[serde(default)]
    choices: Vec<NativeChunkChoice>,
    #[serde(default)]
    usage: Option<TokenUsage>,
    #[serde(default)]
    error: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct NativeChunkChoice {
    #[serde(default)]
    index: Option<u64>,
    #[serde(default)]
    finish_reason: Option<String>,
    delta: NativeChunkDelta,
}

#[derive(Deserialize)]
struct NativeChunkDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<NativeStreamToolCall>,
}

#[derive(Deserialize)]
struct NativeStreamToolCall {
    #[serde(default)]
    index: Option<u64>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<NativeStreamFunction>,
}

#[derive(Deserialize)]
struct NativeStreamFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

/// SSE parser for the native stream. Stricter than the legacy parser: it
/// handles CRLF/LF line endings, joins multi-line `data:` payloads per the SSE
/// spec, and preserves split UTF-8 across chunks.
struct NativeSseParser {
    raw_buf: Vec<u8>,
}

impl NativeSseParser {
    fn new() -> Self {
        Self {
            raw_buf: Vec::new(),
        }
    }

    fn feed(&mut self, chunk: &[u8]) -> Vec<String> {
        self.raw_buf.extend_from_slice(chunk);

        let valid_up_to = match std::str::from_utf8(&self.raw_buf) {
            Ok(_) => self.raw_buf.len(),
            Err(error) => error.valid_up_to(),
        };
        if valid_up_to == 0 {
            return Vec::new();
        }

        let text = std::str::from_utf8(&self.raw_buf[..valid_up_to]).expect("valid UTF-8 prefix");
        let event_end = last_event_end(text);
        if event_end == 0 {
            return Vec::new();
        }

        let events_text = text[..event_end].to_string();
        self.raw_buf.drain(..event_end);
        parse_sse_events(&events_text)
    }
}

/// Byte index just past the last blank-line (event boundary) in `text`.
fn last_event_end(text: &str) -> usize {
    let bytes = text.as_bytes();
    let mut end = 0;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\n' {
            let mut j = i + 1;
            if j < bytes.len() && bytes[j] == b'\r' {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b'\n' {
                end = j + 1;
                i = j + 1;
                continue;
            }
        }
        i += 1;
    }
    end
}

/// Extract each complete SSE event's data payload from a buffer of whole
/// events. Multiple `data:` lines in one event are joined with "\n".
fn parse_sse_events(text: &str) -> Vec<String> {
    let mut events = Vec::new();
    let mut data_lines: Vec<String> = Vec::new();

    for raw_line in text.split('\n') {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if line.is_empty() {
            if !data_lines.is_empty() {
                events.push(data_lines.join("\n"));
                data_lines.clear();
            }
        } else if let Some(data) = line.strip_prefix("data:") {
            data_lines.push(data.trim_start().to_string());
        }
    }

    if !data_lines.is_empty() {
        events.push(data_lines.join("\n"));
    }

    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn tool_call(id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: arguments.to_string(),
        }
    }

    fn simple_request(messages: Vec<ChatMessage>) -> TurnRequest {
        TurnRequest {
            messages,
            temperature: -1.0,
            max_tokens: 128,
            tools: Vec::new(),
        }
    }

    fn client(base_url: &str) -> LlmClient<OpenAiBackend> {
        LlmClient::new_with_model(base_url, "test-model")
    }

    async fn mock_server(response: Vec<u8>) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buffer = [0_u8; 4096];
            loop {
                let read = stream.read(&mut buffer).await.unwrap();
                if read == 0 {
                    break;
                }
                let request = &buffer[..read];
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            let head =
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n";
            let _ = stream.write_all(head.as_bytes()).await;
            let _ = stream.write_all(&response).await;
            let _ = stream.shutdown().await;
        });
        format!("http://{address}")
    }

    async fn collect(
        mut rx: tokio::sync::mpsc::Receiver<NativeStreamEvent>,
    ) -> Vec<NativeStreamEvent> {
        let mut events = Vec::new();
        while let Some(event) = rx.recv().await {
            events.push(event);
        }
        events
    }

    fn sse(data: &str) -> String {
        format!("data: {data}\n\n")
    }

    fn sse_crlf(data: &str) -> String {
        format!("data: {data}\r\n\r\n")
    }

    // --- serialization ---

    #[test]
    fn assistant_tool_call_serializes_nested_with_null_content() {
        let message = ChatMessage::assistant_with_tool_calls(
            "",
            vec![tool_call("call_1", "get_weather", r#"{"city":"Berlin"}"#)],
        );
        let value = serde_json::to_value(message).unwrap();

        assert_eq!(value["role"], "assistant");
        assert!(value["content"].is_null());
        assert_eq!(value["tool_calls"][0]["id"], "call_1");
        assert_eq!(value["tool_calls"][0]["type"], "function");
        assert_eq!(value["tool_calls"][0]["function"]["name"], "get_weather");
        assert_eq!(
            value["tool_calls"][0]["function"]["arguments"],
            r#"{"city":"Berlin"}"#
        );
    }

    #[test]
    fn tool_result_serializes_role_and_id() {
        let message = ChatMessage::tool_result("call_1", "21 degrees");
        let value = serde_json::to_value(message).unwrap();

        assert_eq!(value["role"], "tool");
        assert_eq!(value["content"], "21 degrees");
        assert_eq!(value["tool_call_id"], "call_1");
        assert!(value.get("tool_calls").is_none());
    }

    #[test]
    fn legacy_message_serializes_without_tool_fields() {
        let value = serde_json::to_value(ChatMessage::user("hello")).unwrap();

        assert_eq!(value["role"], "user");
        assert_eq!(value["content"], "hello");
        assert!(value.get("tool_calls").is_none());
        assert!(value.get("tool_call_id").is_none());
    }

    #[test]
    fn tool_call_round_trips_nested_wire_shape() {
        let call = tool_call("id", "name", "{}");
        let value = serde_json::to_value(&call).unwrap();
        let back: ToolCall = serde_json::from_value(value).unwrap();
        assert_eq!(back, call);
    }

    // --- request body ---

    #[test]
    fn request_body_without_tools_uses_choice_none() {
        let backend = OpenAiBackend::new("http://example.com").with_model("m");
        let body = native_request_body(
            &backend,
            &simple_request(vec![ChatMessage::user("hi")]),
            false,
        );

        assert_eq!(body["tool_choice"], "none");
        assert!(body.get("tools").is_none());
        assert_eq!(body["stream"], false);
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn request_body_with_tools_nests_function_shape_and_auto_choice() {
        let backend = OpenAiBackend::new("http://example.com").with_model("m");
        let request = TurnRequest {
            messages: vec![ChatMessage::user("hi")],
            temperature: 0.5,
            max_tokens: 64,
            tools: vec![NativeToolDefinition {
                name: "get_weather".to_string(),
                description: "weather".to_string(),
                parameters: json!({"type": "object"}),
            }],
        };
        let body = native_request_body(&backend, &request, true);

        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(body["tools"][0]["function"]["parameters"]["type"], "object");
        assert_eq!(body["temperature"], 0.5);
        assert_eq!(body["stream"], true);
    }

    // --- complete_turn ---

    #[tokio::test]
    async fn complete_turn_returns_stop_response_with_usage() {
        let body = r#"{"choices":[{"finish_reason":"stop","message":{"content":"hi","reasoning_content":"think"}}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#;
        let url = mock_server(body.as_bytes().to_vec()).await;
        let response = client(&url)
            .complete_turn(simple_request(vec![ChatMessage::user("hi")]))
            .await
            .unwrap();

        assert_eq!(response.content, "hi");
        assert_eq!(response.reasoning, "think");
        assert_eq!(response.finish_reason, NativeFinishReason::Stop);
        assert!(response.tool_calls.is_empty());
        assert_eq!(response.usage.prompt_tokens, 10);
        assert_eq!(response.usage.total_tokens, 15);
    }

    #[tokio::test]
    async fn complete_turn_parses_tool_calls() {
        let body = r#"{"choices":[{"finish_reason":"tool_calls","message":{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Berlin\"}"}}]}}]}"#;
        let url = mock_server(body.as_bytes().to_vec()).await;
        let response = client(&url)
            .complete_turn(simple_request(vec![ChatMessage::user("hi")]))
            .await
            .unwrap();

        assert_eq!(response.finish_reason, NativeFinishReason::ToolCalls);
        assert_eq!(
            response.tool_calls,
            vec![tool_call("call_1", "get_weather", r#"{"city":"Berlin"}"#)]
        );
    }

    #[tokio::test]
    async fn complete_turn_errors_on_unsupported_finish_reason() {
        let body = r#"{"choices":[{"finish_reason":"length","message":{"content":"truncated"}}]}"#;
        let url = mock_server(body.as_bytes().to_vec()).await;
        let err = client(&url)
            .complete_turn(simple_request(vec![ChatMessage::user("hi")]))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("finish_reason 'length'"),
            "got: {err}"
        );
    }

    #[tokio::test]
    async fn complete_turn_errors_on_stop_with_tool_calls() {
        let body = r#"{"choices":[{"finish_reason":"stop","message":{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}"#;
        let url = mock_server(body.as_bytes().to_vec()).await;
        let err = client(&url)
            .complete_turn(simple_request(vec![ChatMessage::user("hi")]))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("finish_reason=stop"), "got: {err}");
    }

    // --- stream_turn ---

    #[tokio::test]
    async fn stream_turn_assembles_tool_call_fragments_by_sparse_index() {
        let mut body = String::new();
        body.push_str(&sse(r#"{"choices":[{"delta":{"tool_calls":[{"index":2,"id":"call_a","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}"#));
        body.push_str(&sse(r#"{"choices":[{"delta":{"tool_calls":[{"index":2,"function":{"arguments":"{\"city\":\""}}]},"finish_reason":null}]}"#));
        body.push_str(&sse(r#"{"choices":[{"delta":{"tool_calls":[{"index":2,"function":{"arguments":"Berlin\"}"}}]},"finish_reason":null}]}"#));
        body.push_str(&sse(
            r#"{"choices":[{"delta":{"content":""},"finish_reason":"tool_calls"}]}"#,
        ));
        body.push_str(&sse("[DONE]"));

        let url = mock_server(body.into_bytes()).await;
        let rx = client(&url)
            .stream_turn(simple_request(vec![ChatMessage::user("hi")]))
            .await
            .unwrap();
        let events = collect(rx).await;

        let last = events.last().unwrap();
        match last {
            NativeStreamEvent::Done(response) => {
                assert_eq!(response.finish_reason, NativeFinishReason::ToolCalls);
                assert_eq!(
                    response.tool_calls,
                    vec![tool_call("call_a", "get_weather", r#"{"city":"Berlin"}"#)]
                );
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_turn_handles_crlf_framing_and_usage_only_final_chunk() {
        let mut body = String::new();
        body.push_str(&sse_crlf(
            r#"{"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#,
        ));
        body.push_str(&sse_crlf(
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        ));
        body.push_str(&sse_crlf(
            r#"{"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":1,"total_tokens":8}}"#,
        ));
        body.push_str(&sse_crlf("[DONE]"));

        let url = mock_server(body.into_bytes()).await;
        let rx = client(&url)
            .stream_turn(simple_request(vec![ChatMessage::user("hi")]))
            .await
            .unwrap();
        let events = collect(rx).await;

        assert!(
            events
                .iter()
                .any(|e| *e == NativeStreamEvent::Content("hi".to_string()))
        );
        match events.last().unwrap() {
            NativeStreamEvent::Done(response) => {
                assert_eq!(response.content, "hi");
                assert_eq!(response.finish_reason, NativeFinishReason::Stop);
                assert_eq!(response.usage.total_tokens, 8);
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_turn_errors_on_duplicate_ids() {
        let mut body = String::new();
        body.push_str(&sse(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"dup","function":{"name":"a","arguments":""}}]},"finish_reason":null}]}"#));
        body.push_str(&sse(r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"dup","function":{"name":"b","arguments":""}}]},"finish_reason":null}]}"#));
        body.push_str(&sse(
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
        ));
        body.push_str(&sse("[DONE]"));

        let url = mock_server(body.into_bytes()).await;
        let rx = client(&url)
            .stream_turn(simple_request(vec![ChatMessage::user("hi")]))
            .await
            .unwrap();
        let events = collect(rx).await;

        match events.last().unwrap() {
            NativeStreamEvent::Error(message) => {
                assert!(message.contains("duplicate tool call id"), "got: {message}")
            }
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_turn_errors_on_missing_id() {
        let mut body = String::new();
        body.push_str(&sse(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"f","arguments":"{}"}}]},"finish_reason":null}]}"#));
        body.push_str(&sse(
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
        ));
        body.push_str(&sse("[DONE]"));

        let url = mock_server(body.into_bytes()).await;
        let rx = client(&url)
            .stream_turn(simple_request(vec![ChatMessage::user("hi")]))
            .await
            .unwrap();
        let events = collect(rx).await;

        match events.last().unwrap() {
            NativeStreamEvent::Error(message) => {
                assert!(message.contains("missing an id"), "got: {message}")
            }
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_turn_errors_on_truncated_stream_without_done() {
        let body = sse(r#"{"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}"#);
        let url = mock_server(body.into_bytes()).await;
        let rx = client(&url)
            .stream_turn(simple_request(vec![ChatMessage::user("hi")]))
            .await
            .unwrap();
        let events = collect(rx).await;

        match events.last().unwrap() {
            NativeStreamEvent::Error(message) => {
                assert!(message.contains("stream ended"), "got: {message}")
            }
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_turn_errors_when_done_alone_without_finish_reason() {
        let body = sse("[DONE]");
        let url = mock_server(body.into_bytes()).await;
        let rx = client(&url)
            .stream_turn(simple_request(vec![ChatMessage::user("hi")]))
            .await
            .unwrap();
        let events = collect(rx).await;

        match events.last().unwrap() {
            NativeStreamEvent::Error(message) => {
                assert!(message.contains("finish_reason"), "got: {message}")
            }
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_turn_errors_on_stop_with_tool_calls() {
        let mut body = String::new();
        body.push_str(&sse(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"f","arguments":"{}"}}]},"finish_reason":null}]}"#));
        body.push_str(&sse(r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#));
        body.push_str(&sse("[DONE]"));

        let url = mock_server(body.into_bytes()).await;
        let rx = client(&url)
            .stream_turn(simple_request(vec![ChatMessage::user("hi")]))
            .await
            .unwrap();
        let events = collect(rx).await;

        match events.last().unwrap() {
            NativeStreamEvent::Error(message) => {
                assert!(message.contains("stop but tool_calls"), "got: {message}")
            }
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_turn_rejects_provider_error_after_tool_calls_terminal() {
        let mut body = String::new();
        body.push_str(&sse(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"f","arguments":"{}"}}]},"finish_reason":null}]}"#));
        body.push_str(&sse(
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
        ));
        body.push_str(&sse(
            r#"{"error":{"message":"upstream exploded","type":"server_error"}}"#,
        ));
        body.push_str(&sse("[DONE]"));

        let url = mock_server(body.into_bytes()).await;
        let rx = client(&url)
            .stream_turn(simple_request(vec![ChatMessage::user("hi")]))
            .await
            .unwrap();
        let events = collect(rx).await;

        match events.last().unwrap() {
            NativeStreamEvent::Error(message) => {
                assert!(message.contains("upstream exploded"), "got: {message}")
            }
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_turn_rejects_conflicting_finish_reason_tool_calls_then_length() {
        let mut body = String::new();
        body.push_str(&sse(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"f","arguments":"{}"}}]},"finish_reason":null}]}"#));
        body.push_str(&sse(
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
        ));
        body.push_str(&sse(
            r#"{"choices":[{"delta":{},"finish_reason":"length"}]}"#,
        ));
        body.push_str(&sse("[DONE]"));

        let url = mock_server(body.into_bytes()).await;
        let rx = client(&url)
            .stream_turn(simple_request(vec![ChatMessage::user("hi")]))
            .await
            .unwrap();
        let events = collect(rx).await;

        match events.last().unwrap() {
            NativeStreamEvent::Error(message) => {
                assert!(
                    message.contains("conflicting finish_reason"),
                    "got: {message}"
                )
            }
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_turn_rejects_action_data_after_terminal() {
        let mut body = String::new();
        body.push_str(&sse(
            r#"{"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}"#,
        ));
        body.push_str(&sse(
            r#"{"choices":[{"delta":{"content":"sneaky extra"},"finish_reason":null}]}"#,
        ));
        body.push_str(&sse("[DONE]"));

        let url = mock_server(body.into_bytes()).await;
        let rx = client(&url)
            .stream_turn(simple_request(vec![ChatMessage::user("hi")]))
            .await
            .unwrap();
        let events = collect(rx).await;

        match events.last().unwrap() {
            NativeStreamEvent::Error(message) => {
                assert!(message.contains("after terminal"), "got: {message}")
            }
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_turn_rejects_multiple_choice_indices() {
        let mut body = String::new();
        body.push_str(&sse(r#"{"choices":[{"index":0,"delta":{"content":"a"},"finish_reason":null},{"index":1,"delta":{"content":"b"},"finish_reason":null}]}"#));
        body.push_str(&sse("[DONE]"));

        let url = mock_server(body.into_bytes()).await;
        let rx = client(&url)
            .stream_turn(simple_request(vec![ChatMessage::user("hi")]))
            .await
            .unwrap();
        let events = collect(rx).await;

        match events.last().unwrap() {
            NativeStreamEvent::Error(message) => {
                assert!(message.contains("completion choices"), "got: {message}")
            }
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn complete_turn_rejects_oversized_or_empty_id_tool_calls() {
        let body = r#"{"choices":[{"finish_reason":"tool_calls","message":{"tool_calls":[{"id":"","type":"function","function":{"name":"f","arguments":"{}"}}]}}]}"#;
        let url = mock_server(body.as_bytes().to_vec()).await;
        let err = client(&url)
            .complete_turn(simple_request(vec![ChatMessage::user("hi")]))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("missing an id"), "got: {err}");
    }

    // --- SSE parser ---

    #[tokio::test]
    async fn stream_turn_rejects_empty_or_whitespace_tool_names() {
        for name in ["", "  "] {
            let delta = serde_json::json!({"choices":[{"delta":{"tool_calls":[{
                "index":0,"id":"name-check","type":"function",
                "function":{"name":name,"arguments":"{}"}
            }]},"finish_reason":"tool_calls"}]});
            let body = format!("{}{}", sse(&delta.to_string()), sse("[DONE]"));
            let url = mock_server(body.into_bytes()).await;
            let events = collect(
                client(&url)
                    .stream_turn(simple_request(vec![ChatMessage::user("hi")]))
                    .await
                    .unwrap(),
            )
            .await;
            assert!(matches!(events.last(), Some(NativeStreamEvent::Error(_))));
            assert!(
                !events
                    .iter()
                    .any(|event| matches!(event, NativeStreamEvent::Done(_)))
            );
        }
    }

    #[test]
    fn sse_parser_joins_multiline_data() {
        let mut parser = NativeSseParser::new();
        let events = parser.feed(b"data: {\"a\":\ndata: 1}\n\n");
        assert_eq!(events, vec!["{\"a\":\n1}".to_string()]);
    }

    #[test]
    fn sse_parser_survives_split_utf8() {
        let mut parser = NativeSseParser::new();
        // "ü" is 0xC3 0xBC — split it across two feeds.
        let first = parser.feed("data: {\"x\":\"h\u{fc}".as_bytes());
        assert!(first.is_empty());
        let second = parser.feed("x\"}\n\n".as_bytes());
        assert_eq!(second, vec!["{\"x\":\"hüx\"}".to_string()]);
    }

    #[test]
    fn sse_parser_handles_crlf_and_done() {
        let mut parser = NativeSseParser::new();
        let events = parser.feed(b"data: [DONE]\r\n\r\n");
        assert_eq!(events, vec!["[DONE]".to_string()]);
    }

    #[test]
    fn native_finish_reason_mapping_is_strict() {
        assert_eq!(
            parse_native_finish_reason("stop").unwrap(),
            NativeFinishReason::Stop
        );
        assert_eq!(
            parse_native_finish_reason("tool_calls").unwrap(),
            NativeFinishReason::ToolCalls
        );
        assert!(parse_native_finish_reason("length").is_err());
        assert!(parse_native_finish_reason("content_filter").is_err());
    }

    #[test]
    fn token_usage_flows_through_native_response() {
        let _ = Value::Bool(true);
        // Keep this module's `TokenUsage` import exercised explicitly.
        let usage: TokenUsage =
            serde_json::from_str(r#"{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}"#)
                .unwrap();
        let mapped = LlmUsage::from(usage);
        assert_eq!(mapped.total_tokens, 5);
    }
}
