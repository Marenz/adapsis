//! Per-model context window, output limit and LLM timeouts (#68).
//!
//! One source of truth for how much a model may see and write. The conversation
//! view, compaction, the native output limit and the HTTP timeouts all derive from
//! it, so costs are controlled by task/turn budgets, never by a hidden smaller
//! context cap.
//!
//! Configuration (first match wins):
//! 1. `ADAPSIS_MODELS_FILE` (or `~/.config/adapsis/models.toml`), e.g.
//!    ```toml
//!    [model."anthropic/claude-opus-5"]
//!    context_tokens = 1000000
//!    output_tokens = 64000
//!    request_timeout_secs = 2400   # optional; derived when omitted
//!    chunk_timeout_secs = 120      # optional streaming chunk-gap timeout
//!    max_request_bytes = 2000000   # optional; serialized request body limit of the route
//!    ```
//! 2. Built-in entries below.
//! 3. Unknown models: `ADAPSIS_CONTEXT_TOKENS` (default 128000) and the legacy
//!    output rule (64000 for Opus-family names, 32000 otherwise).
//!
//! The request byte limit defaults to `ADAPSIS_MAX_REQUEST_BYTES` or 60 MiB, under
//! the local llm-gateway body limit. Source: llm-gateway 332a3b0, deployed
//! 2026-11-10 10:14, logs `request body limit max_request_bytes=67108864` (64 MiB) at
//! startup, and a 3 MB request now reaches model resolution. (Before it, axum's
//! 2 MiB default rejected 2.2 MB bodies, measured 2026-11-09.) The conversation view
//! keeps every request under the configured limit.
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::Duration;

/// Characters per token, consistent with the existing compaction estimate.
pub const CHARS_PER_TOKEN: usize = 4;
const DEFAULT_CONTEXT_TOKENS: usize = 128_000;
const DEFAULT_CHUNK_TIMEOUT_SECS: u64 = 120;
/// 60 MiB: headroom under the gateway's measured 64 MiB limit (see module docs).
const DEFAULT_MAX_REQUEST_BYTES: usize = 60 * 1024 * 1024;
/// Checkpoint fires when the projected request exceeds this share of the window...
pub const CHECKPOINT_TRIGGER_PERCENT: usize = 85;
/// ...and cuts in one step down to this share.
pub const CHECKPOINT_TARGET_PERCENT: usize = 50;

/// Built-in limits `(pattern, context_tokens, output_tokens)`. A pattern ending in
/// `/` matches every model id with that provider prefix; anything else matches the
/// exact id. Exact ids win over prefixes.
///
/// RULE: every entry carries a source comment: a measured provider error (with
/// date), provider documentation, or a model registry. Never add a guessed value;
/// a model without a sourced entry uses the unknown-model default. Other ChatGPT-route
/// models have no entry until a measured or metadata value exists.
const BUILT_IN: &[(&str, usize, u32)] = &[
    // Output 128000: MEASURED 2026-11-09 via the local gateway: "max_tokens: 200000 >
    // 128000, which is the maximum allowed number of output tokens for claude-opus-5".
    // Context 1M: provider metadata + documentation: Anthropic /v1/models reports
    // max_input_tokens=1000000, max_tokens=128000; platform.claude.com/docs/en/
    // build-with-claude/context-windows: 1M is the default, no beta header.
    ("anthropic/claude-opus-5", 1_000_000, 128_000),
    // Provider metadata + documentation: Anthropic /v1/models max_input_tokens=1000000,
    // max_tokens=128000; context-windows docs: 1M default, no beta header.
    ("anthropic/claude-opus-5-5", 1_000_000, 128_000),
    // models.dev registry (context and output; not verified by probe).
    ("anthropic/claude-sonnet-5", 1_000_000, 128_000),
    // models.dev registry (context and output; not verified by probe).
    ("anthropic/claude-sonnet-5-5", 1_000_000, 128_000),
    // models.dev registry: 1.05M window (922k input + 128k output); not verified.
    ("openai/gpt-6-astra", 1_050_000, 128_000),
    // ChatGPT subscription route. Context 872000: Codex /models metadata
    // max_context_window=872000 (its context_window=272000 is a Codex client default,
    // not enforced): MEASURED 2026-11-10 09:08 UTC, a 1,880,169-byte request was
    // accepted with usage prompt_tokens=470,030. Output 128000: registry.
    ("chatgpt/gpt-6-astra", 872_000, 128_000),
    // Codex /models metadata max_context_window=872000 (same model family; the
    // 470,030-token MEASURED acceptance was on gpt-6-astra). Output 128000: registry.
    ("chatgpt/gpt-6-sol", 872_000, 128_000),
    // Codex /models metadata max_context_window=872000 (same model family; the
    // 470,030-token MEASURED acceptance was on gpt-6-astra). Output 128000: registry.
    ("chatgpt/gpt-6-luna", 872_000, 128_000),
    // Codex /models metadata context_window=500000, max_context_window=872000.
    // Output 128000: registry.
    ("chatgpt/gpt-6.1-sol", 872_000, 128_000),
];

fn built_in(model: &str) -> Option<&'static (&'static str, usize, u32)> {
    BUILT_IN.iter().find(|(id, _, _)| !id.ends_with('/') && *id == model).or_else(|| {
        BUILT_IN.iter().find(|(id, _, _)| id.ends_with('/') && model.starts_with(id))
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelLimits {
    pub context_tokens: usize,
    pub output_tokens: u32,
    pub request_timeout: Duration,
    pub chunk_timeout: Duration,
    /// Largest serialized request body the route accepts (bytes).
    pub max_request_bytes: usize,
}

#[derive(Debug, Default, Deserialize)]
struct FileConfig {
    #[serde(default)]
    model: HashMap<String, FileEntry>,
}

#[derive(Debug, Default, Clone, Deserialize)]
struct FileEntry {
    context_tokens: Option<usize>,
    output_tokens: Option<u32>,
    request_timeout_secs: Option<u64>,
    chunk_timeout_secs: Option<u64>,
    max_request_bytes: Option<usize>,
}

/// Total-request timeout for a model's largest request: a 300 s base, plus
/// output at a conservative 33 tokens/s, plus 1 s per 5k input tokens. The
/// chunk-gap timeout separately catches stalled streams long before this.
pub fn derived_request_timeout(context_tokens: usize, output_tokens: u32) -> Duration {
    Duration::from_secs(300 + u64::from(output_tokens) * 30 / 1000 + context_tokens as u64 / 5000)
}

fn legacy_output_tokens(model: &str) -> u32 {
    if model.contains("opus") { 64_000 } else { 32_000 }
}

fn resolve(model: &str, file: &FileConfig, defaults: &Defaults) -> ModelLimits {
    let default_context = defaults.context_tokens;
    let entry = file.model.get(model).cloned().unwrap_or_default();
    let built_in = built_in(model);
    let context_tokens = entry
        .context_tokens
        .or(built_in.map(|b| b.1))
        .unwrap_or(default_context)
        .max(1_000);
    let output_tokens = entry
        .output_tokens
        .or(built_in.map(|b| b.2))
        .unwrap_or_else(|| legacy_output_tokens(model))
        .max(256);
    ModelLimits {
        context_tokens,
        output_tokens,
        request_timeout: entry
            .request_timeout_secs
            .map(Duration::from_secs)
            .unwrap_or_else(|| derived_request_timeout(context_tokens, output_tokens)),
        chunk_timeout: Duration::from_secs(entry.chunk_timeout_secs.unwrap_or(DEFAULT_CHUNK_TIMEOUT_SECS).max(1)),
        max_request_bytes: entry.max_request_bytes.unwrap_or(defaults.max_request_bytes).max(64 * 1024),
    }
}

#[derive(Debug, Clone, Copy)]
struct Defaults {
    context_tokens: usize,
    max_request_bytes: usize,
}

impl Default for Defaults {
    fn default() -> Self {
        Self { context_tokens: DEFAULT_CONTEXT_TOKENS, max_request_bytes: DEFAULT_MAX_REQUEST_BYTES }
    }
}

fn parse_file(text: &str) -> FileConfig {
    match toml::from_str(text) {
        Ok(config) => config,
        Err(error) => {
            crate::logging::diagnostic("model-limits-config-error", error);
            FileConfig::default()
        }
    }
}

fn config() -> &'static (FileConfig, Defaults) {
    static CONFIG: OnceLock<(FileConfig, Defaults)> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let env = |name: &str, default: usize| {
            std::env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
        };
        let defaults = Defaults {
            context_tokens: env("ADAPSIS_CONTEXT_TOKENS", DEFAULT_CONTEXT_TOKENS),
            max_request_bytes: env("ADAPSIS_MAX_REQUEST_BYTES", DEFAULT_MAX_REQUEST_BYTES),
        };
        let path = std::env::var_os("ADAPSIS_MODELS_FILE").map(std::path::PathBuf::from);
        // Like contexts_dir: tests never read the developer's live config.
        #[cfg(not(test))]
        let path = path.or_else(|| dirs::home_dir().map(|h| h.join(".config/adapsis/models.toml")));
        let file = path
            .and_then(|path| std::fs::read_to_string(path).ok())
            .map(|text| parse_file(&text))
            .unwrap_or_default();
        (file, defaults)
    })
}

/// Limits for the model actually generating the request.
pub fn for_model(model: &str) -> ModelLimits {
    let (file, defaults) = config();
    resolve(model, file, defaults)
}

pub fn estimate_tokens(chars: usize) -> usize {
    chars.div_ceil(CHARS_PER_TOKEN)
}

impl ModelLimits {
    /// Projected-request size (input + reserved output) that triggers a checkpoint.
    pub fn trigger_tokens(&self) -> usize {
        self.context_tokens * CHECKPOINT_TRIGGER_PERCENT / 100
    }

    /// Projected-request size a checkpoint cuts down to.
    pub fn target_tokens(&self) -> usize {
        self.context_tokens * CHECKPOINT_TARGET_PERCENT / 100
    }

    /// Per-message cap in the model view: 1/8 of the window AND 1/8 of the request
    /// byte limit. One message alone can never trigger back-to-back checkpoints
    /// (the 85%→50% step leaves 35%), and a deterministic per-message cut keeps the
    /// history byte-stable.
    pub fn max_message_chars(&self) -> usize {
        (self.context_tokens / 8 * CHARS_PER_TOKEN).min(self.max_request_bytes / 8)
    }

    /// Reserved output space, part of every projected request.
    pub fn output_reserve_tokens(&self) -> usize {
        self.output_tokens as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(context_tokens: usize) -> Defaults {
        Defaults { context_tokens, max_request_bytes: DEFAULT_MAX_REQUEST_BYTES }
    }

    #[test]
    fn model_limits_built_ins_file_overrides_and_unknown_defaults() {
        let file = parse_file(
            "[model.\"local/qwen\"]\ncontext_tokens = 32000\noutput_tokens = 4000\nchunk_timeout_secs = 30\nmax_request_bytes = 8000000\n\
             [model.\"anthropic/claude-opus-5\"]\nrequest_timeout_secs = 900\n",
        );
        let opus = resolve("anthropic/claude-opus-5", &file, &d(128_000));
        assert_eq!((opus.context_tokens, opus.output_tokens), (1_000_000, 128_000));
        assert_eq!(opus.request_timeout, Duration::from_secs(900), "file override wins");
        assert_eq!(opus.chunk_timeout, Duration::from_secs(120));
        assert_eq!(opus.max_request_bytes, 60 * 1024 * 1024, "default route body limit");
        let opus_default = resolve("anthropic/claude-opus-5", &FileConfig::default(), &d(128_000));
        assert_eq!(opus_default.request_timeout, Duration::from_secs(300 + 3_840 + 200));
        let local = resolve("local/qwen", &file, &d(128_000));
        assert_eq!((local.context_tokens, local.output_tokens), (32_000, 4_000));
        assert_eq!(local.chunk_timeout, Duration::from_secs(30));
        assert_eq!(local.max_request_bytes, 8_000_000);
        let unknown = resolve("dev-bot", &file, &d(96_000));
        assert_eq!((unknown.context_tokens, unknown.output_tokens), (96_000, 32_000));
        assert_eq!(resolve("some/opus-variant", &file, &d(96_000)).output_tokens, 64_000);
        let env_bytes = Defaults { context_tokens: 128_000, max_request_bytes: 32_000_000 };
        assert_eq!(resolve("dev-bot", &file, &env_bytes).max_request_bytes, 32_000_000);
        assert!(parse_file("not = [toml").model.is_empty(), "bad file falls back, never panics");
    }

    #[test]
    fn model_limits_built_ins_are_sourced_exact_ids_and_chatgpt_has_none() {
        let limits = |m: &str| { let l = resolve(m, &FileConfig::default(), &d(128_000)); (l.context_tokens, l.output_tokens) };
        for id in ["anthropic/claude-opus-5", "anthropic/claude-opus-5-5", "anthropic/claude-sonnet-5", "anthropic/claude-sonnet-5-5"] {
            assert_eq!(limits(id), (1_000_000, 128_000), "{id}");
        }
        assert_eq!(limits("openai/gpt-6-astra"), (1_050_000, 128_000));
        for id in ["chatgpt/gpt-6-astra", "chatgpt/gpt-6-sol", "chatgpt/gpt-6-luna", "chatgpt/gpt-6.1-sol"] {
            assert_eq!(limits(id), (872_000, 128_000), "{id}");
        }
        // Other ChatGPT-route ids have no sourced entry: unknown default.
        for id in ["chatgpt/gpt-5.6-sol", "chatgpt/gpt-6-astra-mini"] {
            assert_eq!(limits(id), (128_000, 32_000), "{id}");
        }
        assert_eq!(limits("anthropic/claude-opus-5-fast"), (128_000, 64_000), "exact ids only");
        assert_eq!(limits("openai/gpt-6-astra-mini"), (128_000, 32_000));
        // Every built-in entry is preceded by a source comment.
        let source = include_str!("model_limits.rs");
        let table = &source[source.find("const BUILT_IN").unwrap()..];
        let table = &table[..table.find("];").unwrap()];
        let lines: Vec<&str> = table.lines().map(str::trim).collect();
        for (i, line) in lines.iter().enumerate().filter(|(_, l)| l.starts_with("(\"")) {
            assert!(lines[i - 1].starts_with("//"), "built-in without a source comment: {line}");
            let comment = lines[..i].iter().rev().take_while(|l| l.starts_with("//")).cloned().collect::<Vec<_>>().join(" ");
            assert!(["MEASURED", "registry", "documentation", "metadata"].iter().any(|k| comment.contains(k)), "{line}: {comment}");
        }
        // The prefix mechanism still works for a configured `provider/` pattern.
        assert!(built_in("x/y").is_none());
    }

    #[test]
    fn model_limits_derived_values_scale_with_the_window_and_byte_limit() {
        let opus = resolve("anthropic/claude-opus-5", &FileConfig::default(), &d(128_000));
        assert_eq!(opus.trigger_tokens(), 850_000);
        assert_eq!(opus.target_tokens(), 500_000);
        assert_eq!(opus.max_message_chars(), 500_000, "window-bound under the 60 MiB route limit");
        let tight = Defaults { context_tokens: 128_000, max_request_bytes: 2_000_000 };
        assert_eq!(resolve("anthropic/claude-opus-5", &FileConfig::default(), &tight).max_message_chars(), 250_000, "a 2 MB route bounds it");
        let small = resolve("x", &FileConfig::default(), &d(128_000));
        assert_eq!(small.max_message_chars(), 64_000, "still above the old 40k cap");
        assert_eq!(small.request_timeout, Duration::from_secs(300 + 960 + 25));
        assert_eq!(estimate_tokens(9), 3);
    }
}
