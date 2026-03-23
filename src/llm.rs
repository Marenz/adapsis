use anyhow::{Context, Result, anyhow};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::{self, Write};
use tracing::{debug, info, warn};

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

    async fn generate_streaming(&self, http: &Client, request: &LlmRequest) -> Result<LlmOutput> {
        let req = http
            .post(self.endpoint())
            .json(&self.request_body(request, true));
        let response = self.apply_auth(req)
            .send()
            .await
            .context("failed to send streaming chat completion request")?
            .error_for_status()
            .context("streaming chat completion request failed")?;

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
            .context("failed to read streaming response chunk")?
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
                        .with_context(|| format!("failed to parse SSE chunk: {data}"))?;

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
        let response: ChatCompletionResponse = self.apply_auth(req)
            .send()
            .await
            .context("failed to send chat completion request")?
            .error_for_status()
            .context("chat completion request failed")?
            .json()
            .await
            .context("failed to deserialize chat completion response")?;

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
                warn!(error = %error, "streaming completion failed, falling back to non-streaming");
                self.generate_non_streaming(http, request).await
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
        // Use higher token limit for capable models
        let max_tokens = if model.contains("opus") { 64000 } else { 32000 };
        Self {
            http: Client::new(),
            backend: OpenAiBackend::new(base_url).with_model(model),
            temperature: 0.7,
            max_tokens,
        }
    }

    pub fn new_with_model_and_key(base_url: &str, model: &str, api_key: Option<String>) -> Self {
        let max_tokens = if model.contains("opus") { 64000 } else { 32000 };
        Self {
            http: Client::new(),
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

        // Retry with exponential backoff on connection errors
        let mut last_err = None;
        for attempt in 0..3 {
            if attempt > 0 {
                let delay = std::time::Duration::from_millis(1000 * (1 << attempt));
                eprintln!("[llm:retry] attempt {}/3 after {}ms", attempt + 1, delay.as_millis());
                tokio::time::sleep(delay).await;
            }
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
                    let err_str = format!("{e}");
                    if err_str.contains("connection") || err_str.contains("Connection")
                        || err_str.contains("timeout") || err_str.contains("reset")
                        || err_str.contains("broken pipe")
                    {
                        eprintln!("[llm:error] {err_str} — will retry");
                        last_err = Some(e);
                        continue;
                    }
                    // Non-retryable error
                    return Err(e);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("LLM request failed after 3 retries")))
    }
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
                || trimmed.starts_with('?') || trimmed == "end" || trimmed == "DONE"
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
