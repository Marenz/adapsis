use anyhow::{Result, anyhow};
use reqwest::{Client, Response, StatusCode, header::HeaderMap};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::error::Error;
use std::io::{self, Write};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

const MAX_LLM_RETRIES: usize = 5;
const MAX_SERVER_RETRY_DELAY: Duration = Duration::from_secs(15 * 60);

#[derive(Debug)]
enum LlmError {
    Api {
        status: StatusCode,
        body: String,
        retry_after: Option<Duration>,
    },
    Transport {
        message: String,
        retryable: bool,
    },
    Streaming(String),
    Decode(String),
}

impl LlmError {
    fn is_retryable(&self) -> bool {
        match self {
            Self::Api { status, .. } => matches!(
                *status,
                StatusCode::REQUEST_TIMEOUT
                    | StatusCode::CONFLICT
                    | StatusCode::TOO_EARLY
                    | StatusCode::TOO_MANY_REQUESTS
                    | StatusCode::INTERNAL_SERVER_ERROR
                    | StatusCode::BAD_GATEWAY
                    | StatusCode::SERVICE_UNAVAILABLE
                    | StatusCode::GATEWAY_TIMEOUT
            ),
            Self::Transport { retryable, .. } => *retryable,
            Self::Streaming(_) | Self::Decode(_) => false,
        }
    }

    fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::Api { retry_after, .. } => *retry_after,
            Self::Transport { .. } | Self::Streaming(_) | Self::Decode(_) => None,
        }
    }

    fn should_fallback_to_non_streaming(&self) -> bool {
        matches!(self, Self::Streaming(_))
    }
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Api {
                status,
                body,
                retry_after,
            } => {
                write!(f, "LLM API error {status}: {}", body.chars().take(500).collect::<String>())?;
                if let Some(delay) = retry_after {
                    write!(f, " (retry after {}s)", delay.as_secs())?;
                }
                Ok(())
            }
            Self::Transport { message, .. } => write!(f, "{message}"),
            Self::Streaming(message) => write!(f, "{message}"),
            Self::Decode(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for LlmError {}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
        }
    }

    pub fn tool(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Tool,
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LlmOutput {
    pub text: String,
    pub thinking: String,
    pub code: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Debug, Clone)]
pub struct LlmRequest {
    pub messages: Vec<ChatMessage>,
    pub temperature: f32,
    pub max_tokens: u32,
}

pub trait LlmBackend {
    async fn generate(&self, http: &Client, request: &LlmRequest) -> Result<LlmOutput>;
}

#[derive(Debug, Clone)]
pub struct OpenAiBackend {
    base_url: String,
    model: String,
    api_key: Option<String>,
}

#[allow(dead_code)]
impl OpenAiBackend {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: "default".to_string(),
            api_key: None,
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_api_key(mut self, api_key: Option<String>) -> Self {
        self.api_key = api_key;
        self
    }

    fn endpoint(&self) -> String {
        format!("{}/v1/chat/completions", self.base_url)
    }

    fn request_body(&self, request: &LlmRequest, stream: bool) -> Value {
        json!({
            "model": self.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": stream,
        })
    }

    fn apply_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(key) => req.bearer_auth(key),
            None => req,
        }
    }

    async fn send_request(&self, req: reqwest::RequestBuilder, context: &'static str) -> Result<Response> {
        let response = self
            .apply_auth(req)
            .send()
            .await
            .map_err(|error| anyhow!(LlmError::Transport {
                message: format!("{context}: {error}"),
                retryable: is_retryable_transport_error(&error),
            }))?;
        ensure_success(response).await
    }

    async fn generate_streaming(&self, http: &Client, request: &LlmRequest) -> Result<LlmOutput> {
        let req = http
            .post(self.endpoint())
            .json(&self.request_body(request, true));
        let response = self.send_request(req, "failed to send streaming chat completion request").await?;

        // Buffer raw bytes so multi-byte UTF-8 sequences that are split
        // across HTTP chunks are not corrupted by from_utf8_lossy.
        let mut raw_buf: Vec<u8> = Vec::new();
        let mut thinking_text = String::new();
        let mut content_text = String::new();
        let mut in_thinking = false;
        let mut stdout = io::stdout();

        let mut response = response;
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|error| anyhow!(LlmError::Streaming(format!("failed to read streaming response chunk: {error}"))))?
        {
            raw_buf.extend_from_slice(&chunk);

            // Convert as much valid UTF-8 as possible from the front of
            // the buffer, leaving any trailing incomplete sequence for the
            // next chunk to complete.
            let valid_up_to = match std::str::from_utf8(&raw_buf) {
                Ok(_) => raw_buf.len(),
                Err(e) => e.valid_up_to(),
            };
            if valid_up_to == 0 {
                continue;
            }

            let text = std::str::from_utf8(&raw_buf[..valid_up_to]).unwrap();
            // We work on complete SSE events delimited by "\n\n".
            // Only drain up to the last complete event boundary to avoid
            // splitting an event across iterations.
            let mut last_event_end = 0;
            let mut search_from = 0;
            while let Some(pos) = text[search_from..].find("\n\n") {
                last_event_end = search_from + pos + 2;
                search_from = last_event_end;
            }
            if last_event_end == 0 {
                continue; // no complete event yet
            }

            let events_str = &text[..last_event_end];
            // Process each complete SSE event
            for event in events_str.split("\n\n") {
                if event.is_empty() {
                    continue;
                }
                for data in sse_data_lines(event) {
                    if data == "[DONE]" {
                        stdout.flush().ok();
                        println!();
                        return Ok(build_output(thinking_text, content_text));
                    }

                    let chunk: ChatCompletionChunk = serde_json::from_str(&data)
                        .map_err(|error| anyhow!(LlmError::Streaming(format!("failed to parse SSE chunk: {error}; chunk={data}"))))?;

                    for choice in chunk.choices {
                        // Handle reasoning_content (Qwen thinking mode)
                        if let Some(reasoning) = choice.delta.reasoning_content {
                            if !in_thinking {
                                in_thinking = true;
                                print!("[thinking] ");
                            }
                            print!("{reasoning}");
                            stdout.flush().ok();
                            thinking_text.push_str(&reasoning);
                        }

                        // Handle regular content
                        if let Some(content) = choice.delta.content {
                            if in_thinking {
                                in_thinking = false;
                                println!();
                                print!("[code] ");
                            }
                            print!("{content}");
                            stdout.flush().ok();
                            content_text.push_str(&content);
                        }
                    }
                }
            }

            raw_buf.drain(..last_event_end);
        }

        stdout.flush().ok();
        println!();
        Ok(build_output(thinking_text, content_text))
    }

    async fn generate_non_streaming(&self, http: &Client, request: &LlmRequest) -> Result<LlmOutput> {
        let req = http
            .post(self.endpoint())
            .json(&self.request_body(request, false));
        let http_resp = self.send_request(req, "failed to send chat completion request").await?;
        let response: ChatCompletionResponse = http_resp
            .json()
            .await
            .map_err(|error| anyhow!(LlmError::Decode(format!("failed to deserialize chat completion response: {error}"))))?;

        let msg = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("chat completion response contained no choices"))?
            .message;

        let thinking = msg.reasoning_content.unwrap_or_default();
        let content = msg.content.unwrap_or_default();

        Ok(build_output(thinking, content))
    }
}

impl LlmBackend for OpenAiBackend {
    async fn generate(&self, http: &Client, request: &LlmRequest) -> Result<LlmOutput> {
        match self.generate_streaming(http, request).await {
            Ok(output) => Ok(output),
            Err(error) => {
                if error
                    .downcast_ref::<LlmError>()
                    .is_some_and(LlmError::should_fallback_to_non_streaming)
                {
                    warn!(error = %error, "streaming completion failed, falling back to non-streaming");
                    self.generate_non_streaming(http, request).await
                } else {
                    Err(error)
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct LlmClient<B = OpenAiBackend> {
    http: Client,
    backend: B,
    temperature: f32,
    max_tokens: u32,
}

#[allow(dead_code)]
impl LlmClient<OpenAiBackend> {
    pub fn new(base_url: &str) -> Self {
        Self {
            http: Client::new(),
            backend: OpenAiBackend::new(base_url),
            temperature: 0.7,
            max_tokens: 64000,
        }
    }

    pub fn new_with_model(base_url: &str, model: &str) -> Self {
        let max_tokens = if model.contains("opus") { 64000 } else { 32000 };
        Self {
            http: Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build().unwrap_or_else(|_| Client::new()),
            backend: OpenAiBackend::new(base_url).with_model(model),
            temperature: 0.7,
            max_tokens,
        }
    }

    pub fn new_with_model_and_key(base_url: &str, model: &str, api_key: Option<String>) -> Self {
        let max_tokens = if model.contains("opus") { 64000 } else { 32000 };
        Self {
            http: Client::builder()
                .timeout(std::time::Duration::from_secs(300)) // 5 min per LLM request
                .build().unwrap_or_else(|_| Client::new()),
            backend: OpenAiBackend::new(base_url).with_model(model).with_api_key(api_key),
            temperature: 0.7,
            max_tokens,
        }
    }
}

#[allow(dead_code)]
impl<B> LlmClient<B>
where
    B: LlmBackend,
{
    pub fn with_backend(backend: B) -> Self {
        Self {
            http: Client::new(),
            backend,
            temperature: 0.7,
            max_tokens: 128000,
        }
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub async fn generate(&self, messages: Vec<ChatMessage>) -> Result<LlmOutput> {
        info!("LLM request: {} messages, temp={}, max_tokens={}", messages.len(), self.temperature, self.max_tokens);
        for (i, msg) in messages.iter().enumerate() {
            let role = format!("{:?}", msg.role).to_lowercase();
            debug!("[msg {i}] {role} ({} chars):\n{}", msg.content.len(), msg.content);
        }

        let request = LlmRequest {
            messages,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
        };

        // Retry with exponential backoff on transient transport/server errors.
        let mut last_err = None;
        for attempt in 0..MAX_LLM_RETRIES {
            match self.backend.generate(&self.http, &request).await {
                Ok(output) => {
                    info!("LLM response: thinking={} chars, code={} chars, text={} chars",
                        output.thinking.len(), output.code.len(), output.text.len());
                    if !output.thinking.is_empty() {
                        debug!("[thinking]:\n{}", output.thinking);
                    }
                    if !output.code.is_empty() {
                        debug!("[code]:\n{}", output.code);
                    }
                    if !output.text.is_empty() {
                        debug!("[text]:\n{}", output.text);
                    }
                    return Ok(output);
                }
                Err(e) => {
                    let Some(llm_error) = e.downcast_ref::<LlmError>() else {
                        return Err(e);
                    };

                    if llm_error.is_retryable() && attempt + 1 < MAX_LLM_RETRIES {
                        let delay = llm_error
                            .retry_after()
                            .unwrap_or_else(|| default_retry_delay(attempt, llm_error));
                        eprintln!(
                            "[llm:retry] attempt {}/{} in {}ms: {}",
                            attempt + 2,
                            MAX_LLM_RETRIES,
                            delay.as_millis(),
                            llm_error
                        );
                        tokio::time::sleep(delay).await;
                        last_err = Some(e);
                        continue;
                    }

                    return Err(e);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow!("LLM request failed after {MAX_LLM_RETRIES} retries")))
    }
}

fn is_retryable_transport_error(error: &reqwest::Error) -> bool {
    error.is_connect()
        || error.is_timeout()
        || error.is_request()
        || error
            .source()
            .map(|source| {
                let text = source.to_string().to_ascii_lowercase();
                text.contains("connection")
                    || text.contains("timeout")
                    || text.contains("reset")
                    || text.contains("broken pipe")
                    || text.contains("temporarily unavailable")
            })
            .unwrap_or(false)
}

async fn ensure_success(response: Response) -> Result<Response> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }

    let headers = response.headers().clone();
    let body = response.text().await.unwrap_or_default();
    Err(anyhow!(LlmError::Api {
        status,
        retry_after: extract_retry_delay(&headers, &body),
        body,
    }))
}

fn default_retry_delay(attempt: usize, error: &LlmError) -> Duration {
    let is_rate_limited = matches!(
        error,
        LlmError::Api {
            status: StatusCode::TOO_MANY_REQUESTS,
            ..
        }
    );
    if is_rate_limited {
        // Rate limits typically last 30-60+ seconds; use a longer base delay.
        let seconds = 30_u64 * (1 << attempt.min(3));
        Duration::from_secs(seconds)
    } else {
        // Transient server errors: shorter exponential backoff.
        let seconds = 1_u64 << attempt.min(6);
        Duration::from_secs(seconds)
    }
}

fn extract_retry_delay(headers: &HeaderMap, body: &str) -> Option<Duration> {
    header_retry_delay(headers).or_else(|| body_retry_delay(body))
}

fn header_retry_delay(headers: &HeaderMap) -> Option<Duration> {
    if let Some(value) = header_value(headers, "retry-after") {
        if let Some(delay) = parse_retry_after_seconds(value) {
            return Some(delay);
        }
    }

    for name in ["x-ratelimit-reset", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"] {
        if let Some(value) = header_value(headers, name) {
            if let Some(delay) = parse_reset_timestamp(value) {
                return Some(delay);
            }
        }
    }

    None
}

fn header_value<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|value| value.to_str().ok()).map(str::trim)
}

fn body_retry_delay(body: &str) -> Option<Duration> {
    let json: Value = serde_json::from_str(body).ok()?;

    for key in ["retry_after_ms", "retryAfterMs"] {
        if let Some(ms) = find_numeric_field(&json, key) {
            return duration_from_millis(ms);
        }
    }

    for key in ["retry_after", "retryAfter"] {
        if let Some(seconds) = find_numeric_field(&json, key) {
            return duration_from_secs_f64(seconds);
        }
    }

    None
}

fn find_numeric_field(value: &Value, key: &str) -> Option<f64> {
    match value {
        Value::Object(map) => {
            if let Some(number) = map.get(key).and_then(json_number_to_f64) {
                return Some(number);
            }
            map.values().find_map(|entry| find_numeric_field(entry, key))
        }
        Value::Array(items) => items.iter().find_map(|entry| find_numeric_field(entry, key)),
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => None,
    }
}

fn json_number_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.trim().parse().ok(),
        Value::Null | Value::Bool(_) | Value::Array(_) | Value::Object(_) => None,
    }
}

fn parse_retry_after_seconds(raw: &str) -> Option<Duration> {
    duration_from_secs_f64(raw.parse().ok()?)
}

fn parse_reset_timestamp(raw: &str) -> Option<Duration> {
    let value: f64 = raw.parse().ok()?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs_f64();

    let seconds_until_reset = if value > 1_000_000_000_000.0 {
        (value / 1000.0) - now
    } else if value > 1_000_000_000.0 {
        value - now
    } else {
        value
    };

    duration_from_secs_f64(seconds_until_reset)
}

fn duration_from_secs_f64(seconds: f64) -> Option<Duration> {
    if !seconds.is_finite() || seconds <= 0.0 {
        return None;
    }

    let duration = Duration::from_secs_f64(seconds);
    (duration <= MAX_SERVER_RETRY_DELAY).then_some(duration)
}

fn duration_from_millis(milliseconds: f64) -> Option<Duration> {
    if !milliseconds.is_finite() || milliseconds <= 0.0 {
        return None;
    }

    let duration = Duration::from_secs_f64(milliseconds / 1000.0);
    (duration <= MAX_SERVER_RETRY_DELAY).then_some(duration)
}

#[derive(Debug, Deserialize)]
struct ChatCompletionChunk {
    choices: Vec<ChunkChoice>,
}

#[derive(Debug, Deserialize)]
struct ChunkChoice {
    delta: ChunkDelta,
}

#[derive(Debug, Deserialize)]
struct ChunkDelta {
    content: Option<String>,
    reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ResponseChoice>,
    #[serde(default)]
    usage: Option<TokenUsage>,
}

#[derive(Debug, Deserialize)]
struct TokenUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct ResponseChoice {
    message: ResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
}

fn sse_data_lines(event: &str) -> Vec<String> {
    event
        .lines()
        .filter_map(|line| line.strip_prefix("data:"))
        .map(|line| line.trim_start().to_string())
        .collect()
}

/// Build LlmOutput from separate thinking and content strings.
/// The content may contain <code> tags (if the model uses them), or may be
/// raw Forge code. We handle both cases.
fn build_output(thinking: String, content: String) -> LlmOutput {
    // First: strip <think> blocks from content to avoid think text contaminating code extraction.
    // Models like Sonnet interleave <think> and <code> blocks, and think blocks can contain
    // literal "<code>" text that confuses the parser.
    let think_blocks = extract_tag_contents(&content, "think");
    let combined_thinking = if !think_blocks.is_empty() {
        let think_text = think_blocks.join("\n\n");
        if thinking.is_empty() { think_text } else { format!("{thinking}\n\n{think_text}") }
    } else {
        thinking.clone()
    };
    let clean_content = strip_tags(&content, "think");
    let clean_content = strip_tags(&clean_content, "tool_call");
    let clean_content = strip_tags(&clean_content, "tool_result");

    // Extract code: prefer <code> blocks, but also scan for Forge operations anywhere
    let code_blocks = extract_tag_contents(&clean_content, "code");
    let code = if !code_blocks.is_empty() {
        code_blocks.join("\n\n")
    } else {
        // No <code> tags — scan for Forge operation lines in the full response.
        // Every Forge command starts with +, !, or ? (and `end` for module closing).
        // Extract contiguous blocks of such lines.
        let mut forge_lines: Vec<String> = Vec::new();
        let mut in_block = false;
        let mut block_is_greedy = false; // !plan set, !roadmap add, !opencode, +type eat continuation
        for line in clean_content.lines() {
            let trimmed = line.trim();
            let is_continuation = in_block && !trimmed.is_empty() && (
                block_is_greedy
                || trimmed.starts_with("- ") || trimmed.starts_with("* ")
                || line.starts_with("  ") || line.starts_with("\t")
                || trimmed.starts_with("//")
            );
            let is_forge = trimmed.starts_with('+') || trimmed.starts_with('!')
                || trimmed.starts_with('?') || trimmed == "end" || trimmed == "!done"
                || is_continuation
                || (in_block && trimmed.is_empty());
            if is_forge {
                // Check if this line starts a greedy block
                if trimmed.starts_with("!plan set") || trimmed.starts_with("!roadmap add")
                    || trimmed.starts_with("!opencode") || trimmed.starts_with("+type") {
                    block_is_greedy = true;
                } else if trimmed.starts_with('+') || trimmed.starts_with('!') || trimmed.starts_with('?') {
                    block_is_greedy = false; // New forge line resets greedy
                }
                forge_lines.push(line.to_string());
                in_block = true;
            } else if in_block && !trimmed.is_empty() {
                in_block = false;
                block_is_greedy = false;
            }
        }
        forge_lines.join("\n").trim().to_string()
    };

    let full_text = if thinking.is_empty() {
        content.clone()
    } else {
        format!("<think>\n{thinking}\n</think>\n{content}")
    };

    LlmOutput {
        text: full_text,
        thinking,
        code,
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    }
}

fn extract_tag_contents(text: &str, tag: &str) -> Vec<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut rest = text;
    let mut matches = Vec::new();

    while let Some(start) = rest.find(&open) {
        let after_open = &rest[start + open.len()..];
        if let Some(end) = after_open.find(&close) {
            let value = after_open[..end].trim();
            if !value.is_empty() {
                matches.push(value.to_string());
            }
            rest = &after_open[end + close.len()..];
        } else {
            // No closing tag — take everything after the opening tag
            let value = after_open.trim();
            if !value.is_empty() {
                matches.push(value.to_string());
            }
            break;
        }
    }

    matches
}

fn strip_tags(text: &str, tag: &str) -> String {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut result = text.to_string();
    while let Some(start) = result.find(&open) {
        if let Some(end) = result[start..].find(&close) {
            result.replace_range(start..start + end + close.len(), "");
        } else {
            break;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_retry_after_seconds_header() {
        let mut headers = HeaderMap::new();
        headers.insert("retry-after", "7".parse().unwrap());

        assert_eq!(header_retry_delay(&headers), Some(Duration::from_secs(7)));
    }

    #[test]
    fn parses_retry_after_from_json_body() {
        let body = r#"{"error":{"message":"slow down","retry_after_ms":1500}}"#;

        assert_eq!(body_retry_delay(body), Some(Duration::from_millis(1500)));
    }

    #[test]
    fn parses_rate_limit_reset_timestamp_header() {
        let reset_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 3;
        let mut headers = HeaderMap::new();
        headers.insert("x-ratelimit-reset", reset_at.to_string().parse().unwrap());

        let delay = header_retry_delay(&headers).unwrap();
        assert!(delay <= Duration::from_secs(3));
        assert!(delay >= Duration::from_secs(1));
    }

    #[test]
    fn marks_429_as_retryable() {
        let error = LlmError::Api {
            status: StatusCode::TOO_MANY_REQUESTS,
            body: "rate limit".to_string(),
            retry_after: Some(Duration::from_secs(9)),
        };

        assert!(error.is_retryable());
        assert_eq!(error.retry_after(), Some(Duration::from_secs(9)));
    }
}
