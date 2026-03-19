use anyhow::{Context, Result, anyhow};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::io::{self, Write};
use tracing::warn;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
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
}

#[derive(Debug, Clone)]
pub struct LlmOutput {
    pub text: String,
    pub thinking: String,
    pub code: String,
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
}

#[allow(dead_code)]
impl OpenAiBackend {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: "default".to_string(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
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

    async fn generate_streaming(&self, http: &Client, request: &LlmRequest) -> Result<LlmOutput> {
        let response = http
            .post(self.endpoint())
            .json(&self.request_body(request, true))
            .send()
            .await
            .context("failed to send streaming chat completion request")?
            .error_for_status()
            .context("streaming chat completion request failed")?;

        let mut buffer = String::new();
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
            let chunk = String::from_utf8_lossy(&chunk);
            buffer.push_str(&chunk);

            while let Some(event_end) = buffer.find("\n\n") {
                let event = buffer[..event_end].to_string();
                buffer.drain(..event_end + 2);

                for data in sse_data_lines(&event) {
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
        }

        stdout.flush().ok();
        println!();
        Ok(build_output(thinking_text, content_text))
    }

    async fn generate_non_streaming(&self, http: &Client, request: &LlmRequest) -> Result<LlmOutput> {
        let response: ChatCompletionResponse = http
            .post(self.endpoint())
            .json(&self.request_body(request, false))
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
        let request = LlmRequest {
            messages,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
        };

        self.backend.generate(&self.http, &request).await
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
    // Try to extract <code> blocks from content first
    let code_blocks = extract_tag_contents(&content, "code");
    let code = if !code_blocks.is_empty() {
        code_blocks.join("\n\n")
    } else {
        // No <code> tags — the content itself might be code,
        // or it might also contain <think> tags (for models that don't use reasoning_content)
        let think_blocks = extract_tag_contents(&content, "think");
        if !think_blocks.is_empty() {
            // Model used <think> in content instead of reasoning_content
            let combined_thinking =
                if thinking.is_empty() { think_blocks.join("\n\n") } else { thinking.clone() };
            // Extract code blocks after removing think blocks
            let code_after_think = extract_tag_contents(&content, "code");
            if !code_after_think.is_empty() {
                return LlmOutput {
                    text: content,
                    thinking: combined_thinking,
                    code: code_after_think.join("\n\n"),
                };
            }
            // Strip think blocks from content to get the code
            let stripped = strip_tags(&content, "think");
            return LlmOutput {
                text: content,
                thinking: combined_thinking,
                code: stripped.trim().to_string(),
            };
        }
        // No tags at all — treat entire content as potential code
        content.clone()
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
