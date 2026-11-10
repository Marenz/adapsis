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
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::Duration;

/// Serialized request bytes per provider-counted input token, used until a
/// conversation has its own measurement (`TokenCalibration`).
/// Source: MEASURED 2026-11-10 on the live admin DM (anthropic/claude-opus-5): a
/// request whose serialized view was ≈1.46 MB was counted by the provider as
/// 593,447 input tokens, i.e. ≈2.46 bytes/token. Tool-result JSON, code and German
/// text tokenize far denser than the former 4 chars/token assumption, under which
/// a history estimated at the 85% trigger would really have been ≈1.17M tokens.
pub const DEFAULT_BYTES_PER_TOKEN: f32 = 2.5;
/// Sane range for a measured ratio. Below 1.5 would mean almost one token per
/// byte (denser than base64 or digit runs); above 6 is sparser than plain English
/// prose (≈4–5). A measurement outside it is clamped, not trusted.
pub const MIN_BYTES_PER_TOKEN: f32 = 1.5;
pub const MAX_BYTES_PER_TOKEN: f32 = 6.0;
/// Requests smaller than this say too little about the ratio to replace a
/// previous measurement.
const MIN_CALIBRATION_TOKENS: u64 = 2_000;
/// Rough provider cost of one image in tokens. Images are projected separately and
/// kept out of the text ratio.
pub const IMAGE_TOKENS: usize = 1_600;
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

/// The lifetime a cache breakpoint asks the provider for (`cache_control.ttl`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTtl {
    FiveMinutes,
    OneHour,
}

impl CacheTtl {
    /// The wire value of `cache_control.ttl`.
    pub fn wire(self) -> &'static str {
        match self {
            Self::FiveMinutes => "5m",
            Self::OneHour => "1h",
        }
    }

    pub fn secs(self) -> u64 {
        match self {
            Self::FiveMinutes => 300,
            Self::OneHour => 3_600,
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "5m" => Some(Self::FiveMinutes),
            "1h" => Some(Self::OneHour),
            _ => None,
        }
    }
}

/// Built-in prompt-cache behaviour per route `(pattern, requested ttl, lifetime
/// in seconds)`, matched like `BUILT_IN`. The requested ttl is sent on every
/// cache breakpoint (None: a plain marker). The lifetime is only the default for
/// the cold-start rule: `cache_lifetime` learns the real value per route from
/// usage, and `models.toml` can replace it. A lifetime of None means UNKNOWN:
/// the route never cold-starts until usage has shown that its cache expires.
///
/// RULE: every entry carries a source comment (MEASURED, DOCS or UNVERIFIED with
/// the reason). Never add a guessed value; unknown is a valid entry.
/// Sources: research session 2026-11-10 (raw probe data was
/// /tmp/opencode/cacheprobe/results.jsonl), relayed by the manager.
const BUILT_IN_CACHE: &[(&str, Option<CacheTtl>, Option<u64>)] = &[
    // Owner decision 2026-11-10: request the 1-hour lifetime on Anthropic routes.
    // DOCS (platform.claude.com/docs/en/build-with-claude/prompt-caching) +
    // MEASURED on anthropic/claude-opus-5 via the gateway with ttl "1h": hit at
    // 1200 s, miss at 3900 s; works on the OAuth route without a beta header. A
    // read refreshes the lifetime, which counts from the START of the request.
    // (Default marker without ttl: 300 s, DOCS + MEASURED hit 240 s / miss 420 s.)
    ("anthropic/", Some(CacheTtl::OneHour), Some(3_600)),
    // DOCS: 1800 s is the OpenAI API minimum for GPT-5.6+ (developers.openai.com/
    // api/docs/guides/prompt-caching). MEASURED 2026-11-10 by the gateway worker
    // (llm-gateway #4, c4f0d36, which sends prompt_cache_key and session-id): still
    // readable after 3900 s, no expiry observed. Hits are PROBABILISTIC on this
    // route (see PROBABILISTIC_CACHE), so only hits can move the learned lifetime.
    ("chatgpt/", None, Some(1_800)),
    // DOCS only (developers.openai.com/api/docs/guides/prompt-caching, GPT-5.6+):
    // 1800 s, a hit refreshes. Not measured.
    ("openai/", None, Some(1_800)),
    // UNVERIFIED: conservative reading of the provider docs ("a few hours to a
    // few days"). Not measured.
    ("deepseek/", None, Some(3_600)),
    // DOCS: 300 s. Not measured.
    ("moonshot/", None, Some(300)),
    // UNKNOWN: the provider DOCS give no value.
    ("deepinfra/", None, None),
    // UNKNOWN: the provider DOCS give no value.
    ("xiaomi-mimo/", None, None),
];

/// Routes whose cache hits are probabilistic: a request that could have read the
/// cache often does not, at any pause. On such a route a miss is not evidence
/// about the lifetime (`cache_lifetime` records hits only) and one uncached
/// answer does not mean the cache is cold.
///
/// RULE: every entry carries a MEASURED source comment.
const PROBABILISTIC_CACHE: &[&str] = &[
    // MEASURED 2026-11-10 by the gateway worker (llm-gateway #4) with a stable
    // prompt_cache_key: about 60% hits at every gap from 12 s to 65 min (15 of 26
    // single reads; batches from 1 of 3 to 9 of 9 at 12 s). Live check after the
    // gateway deploy: four identical gpt-6-astra requests 12 s apart read 0, 2048,
    // 2048, 2048 of 2222 tokens. Reads are reported inside prompt_tokens (multiples
    // of 128); there is no write counter.
    "chatgpt/",
];

fn built_in_probabilistic(model: &str) -> bool {
    PROBABILISTIC_CACHE.iter().any(|prefix| model.starts_with(prefix))
}

fn built_in_cache(model: &str) -> (Option<CacheTtl>, Option<u64>) {
    let table = BUILT_IN_CACHE;
    table
        .iter()
        .find(|(id, _, _)| !id.ends_with('/') && *id == model)
        .or_else(|| table.iter().find(|(id, _, _)| id.ends_with('/') && model.starts_with(id)))
        .map_or((None, None), |(_, ttl, secs)| (*ttl, *secs))
}

/// History kept after a pause longer than the cache lifetime (cold start), in
/// tokens. Owner decision 2026-11-10; `ADAPSIS_COLD_START_TOKENS` or
/// `cold_start_tokens` in `models.toml` override it.
const DEFAULT_COLD_START_TOKENS: usize = 100_000;

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
    /// Lifetime requested on every cache breakpoint (`cache_control.ttl`); None
    /// sends a plain marker (the provider's default lifetime).
    pub cache_ttl: Option<CacheTtl>,
    /// Default prompt-cache lifetime in seconds for the cold-start rule: from
    /// `models.toml`, else the requested ttl, else the sourced built-in. None =
    /// unknown. Usage evidence overrides it (`cache_lifetime`).
    pub cache_ttl_secs: Option<u64>,
    /// False when `models.toml` sets `cache_ttl_secs = 0`: this model never
    /// cold-starts, whatever is learned.
    pub cold_start: bool,
    /// Cache hits on this route are probabilistic: misses are not evidence about
    /// the lifetime (built-in `PROBABILISTIC_CACHE`, or `models.toml`).
    pub cache_probabilistic: bool,
    /// History kept when a turn starts after a pause longer than the cache
    /// lifetime (tokens).
    pub cold_start_tokens: usize,
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
    /// "5m" or "1h": the lifetime to request on cache breakpoints.
    cache_ttl: Option<String>,
    cache_ttl_secs: Option<u64>,
    cache_probabilistic: Option<bool>,
    cold_start_tokens: Option<usize>,
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
    let (built_in_ttl, built_in_secs) = built_in_cache(model);
    let cache_ttl = match entry.cache_ttl.as_deref() {
        Some(value) => CacheTtl::parse(value).or_else(|| {
            crate::logging::diagnostic("model-limits-config-error", "cache_ttl must be \"5m\" or \"1h\"");
            built_in_ttl
        }),
        None => built_in_ttl,
    };
    ModelLimits {
        context_tokens,
        output_tokens,
        request_timeout: entry
            .request_timeout_secs
            .map(Duration::from_secs)
            .unwrap_or_else(|| derived_request_timeout(context_tokens, output_tokens)),
        chunk_timeout: Duration::from_secs(entry.chunk_timeout_secs.unwrap_or(DEFAULT_CHUNK_TIMEOUT_SECS).max(1)),
        max_request_bytes: entry.max_request_bytes.unwrap_or(defaults.max_request_bytes).max(64 * 1024),
        cache_ttl,
        cache_ttl_secs: match entry.cache_ttl_secs {
            Some(secs) => (secs > 0).then_some(secs),
            // The requested lifetime is what the provider keeps, when one is requested.
            None => cache_ttl.map(CacheTtl::secs).or(built_in_secs),
        },
        cold_start: entry.cache_ttl_secs != Some(0),
        cache_probabilistic: entry.cache_probabilistic.unwrap_or_else(|| built_in_probabilistic(model)),
        cold_start_tokens: entry.cold_start_tokens.unwrap_or(defaults.cold_start_tokens).max(1_000),
    }
}

#[derive(Debug, Clone, Copy)]
struct Defaults {
    context_tokens: usize,
    max_request_bytes: usize,
    cold_start_tokens: usize,
}

impl Default for Defaults {
    fn default() -> Self {
        Self {
            context_tokens: DEFAULT_CONTEXT_TOKENS,
            max_request_bytes: DEFAULT_MAX_REQUEST_BYTES,
            cold_start_tokens: DEFAULT_COLD_START_TOKENS,
        }
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
            cold_start_tokens: env("ADAPSIS_COLD_START_TOKENS", DEFAULT_COLD_START_TOKENS),
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

/// The last request of a conversation as the provider counted it: the basis for
/// projecting the next request's tokens from its serialized size. Persisted with
/// the conversation; the raw pair is stored and the ratio is clamped on read.
///
/// The ratio belongs to the model that counted it: another model has another
/// tokenizer, so a conversation whose generating model changes projects with the
/// default again until that model has answered once.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenCalibration {
    /// Serialized text bytes of the sent request (image data excluded).
    pub request_bytes: u64,
    /// Provider-counted input tokens for it (image allowance subtracted).
    pub input_tokens: u64,
    /// The generating model that counted them. Empty in records written before
    /// the field existed; those match no model and are re-learned.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub model: String,
}

impl TokenCalibration {
    /// A measurement from one answered request, or `None` when it carries no
    /// usable signal: no usage report (zero input tokens), an empty request, or
    /// too few text tokens. `input_tokens` must be the total input the provider
    /// processed (`LlmUsage::accounted_input_tokens`, which normalizes whether
    /// cache reads are inside or beside the prompt count).
    pub fn measure(model: &str, text_bytes: usize, images: usize, input_tokens: u64) -> Option<Self> {
        let text_tokens = input_tokens.saturating_sub((images * IMAGE_TOKENS) as u64);
        (text_bytes > 0 && text_tokens >= MIN_CALIBRATION_TOKENS).then(|| Self {
            request_bytes: text_bytes as u64,
            input_tokens: text_tokens,
            model: model.to_string(),
        })
    }

    /// Measured bytes per token, clamped to the sane range.
    pub fn bytes_per_token(&self) -> f32 {
        if self.input_tokens == 0 {
            return DEFAULT_BYTES_PER_TOKEN;
        }
        let ratio = self.request_bytes as f64 / self.input_tokens as f64;
        (ratio as f32).clamp(MIN_BYTES_PER_TOKEN, MAX_BYTES_PER_TOKEN)
    }
}

/// The ratio to project a request to `model` with: the conversation's
/// measurement if that model made it, else the default.
pub fn bytes_per_token(calibration: Option<&TokenCalibration>, model: &str) -> f32 {
    calibration
        .filter(|measured| !measured.model.is_empty() && measured.model == model)
        .map_or(DEFAULT_BYTES_PER_TOKEN, TokenCalibration::bytes_per_token)
}

/// Size of `text` as a JSON string in a request body: what the byte budgets, the
/// token projection and the compaction thresholds all count.
pub fn serialized_len(text: &str) -> usize {
    serde_json::to_string(text).map_or(text.len(), |json| json.len())
}

/// Projected tokens for `bytes` of serialized text at `bytes_per_token`.
pub fn tokens_for_bytes(bytes: usize, bytes_per_token: f32) -> usize {
    let ratio = if bytes_per_token.is_finite() { bytes_per_token } else { DEFAULT_BYTES_PER_TOKEN };
    (bytes as f64 / f64::from(ratio.clamp(MIN_BYTES_PER_TOKEN, MAX_BYTES_PER_TOKEN))).ceil() as usize
}

/// Serialized bytes that `tokens` stand for at `bytes_per_token`.
pub fn bytes_for_tokens(tokens: usize, bytes_per_token: f32) -> usize {
    let ratio = if bytes_per_token.is_finite() { bytes_per_token } else { DEFAULT_BYTES_PER_TOKEN };
    (tokens as f64 * f64::from(ratio.clamp(MIN_BYTES_PER_TOKEN, MAX_BYTES_PER_TOKEN))) as usize
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

    /// Per-message cap in the model view, in UTF-8 BYTES of the message text: 1/8
    /// of the window AND 1/8 of the request byte limit. One message alone can never
    /// trigger back-to-back checkpoints (the 85%→50% step leaves 35%). The cap uses
    /// the fixed default ratio, never a conversation's measured one: a
    /// deterministic per-message cut keeps the history byte-stable.
    pub fn max_message_bytes(&self) -> usize {
        bytes_for_tokens(self.context_tokens / 8, DEFAULT_BYTES_PER_TOKEN).min(self.max_request_bytes / 8)
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
        Defaults { context_tokens, ..Defaults::default() }
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
        let env_bytes = Defaults { max_request_bytes: 32_000_000, ..Defaults::default() };
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
        let table = &source[source.find("const BUILT_IN:").unwrap()..];
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
    fn model_limits_cache_lifetime_defaults_are_sourced_and_configurable() {
        let none = FileConfig::default();
        let cache = |id: &str, file: &FileConfig| {
            let limits = resolve(id, file, &d(128_000));
            (limits.cache_ttl, limits.cache_ttl_secs, limits.cold_start)
        };
        // Anthropic routes request the 1-hour lifetime; it is also the default
        // lifetime for the cold-start rule.
        for id in ["anthropic/claude-opus-5", "anthropic/claude-sonnet-5", "anthropic/some-future-model"] {
            assert_eq!(cache(id, &none), (Some(CacheTtl::OneHour), Some(3_600), true), "{id}");
        }
        // Sourced lifetimes without a requested ttl.
        assert_eq!(cache("openai/gpt-6-astra", &none), (None, Some(1_800), true));
        assert_eq!(cache("deepseek/deepseek-v4-flash", &none), (None, Some(3_600), true));
        assert_eq!(cache("moonshot/kimi", &none), (None, Some(300), true));
        // The ChatGPT route: a sourced lifetime, no requested ttl, probabilistic hits.
        for id in ["chatgpt/gpt-6-astra", "chatgpt/gpt-5.6-sol"] {
            assert_eq!(cache(id, &none), (None, Some(1_800), true), "{id}");
            assert!(resolve(id, &none, &d(128_000)).cache_probabilistic, "{id}");
        }
        // Unknown: listed-as-unknown routes, aliases and anything unlisted.
        for id in ["deepinfra/zai-org/GLM-5.2", "xiaomi-mimo/mimo-v2", "deepseek-v4-flash", "dev-bot", "claude-opus", "gemma4s"] {
            assert_eq!(cache(id, &none), (None, None, true), "{id}");
        }
        // Every other route is deterministic.
        for id in ["anthropic/claude-opus-5", "openai/gpt-6-astra", "deepseek/x", "moonshot/kimi", "deepinfra/x", "dev-bot"] {
            assert!(!resolve(id, &none, &d(128_000)).cache_probabilistic, "{id}");
        }
        assert_eq!((CacheTtl::FiveMinutes.wire(), CacheTtl::OneHour.wire()), ("5m", "1h"));
        assert_eq!(resolve("dev-bot", &none, &d(128_000)).cold_start_tokens, 100_000);
        // models.toml: requested ttl, default lifetime, cold-start size, and off.
        let file = parse_file(
            "[model.\"anthropic/claude-opus-5\"]\ncache_ttl = \"5m\"\ncold_start_tokens = 250000\n\
             [model.\"anthropic/claude-sonnet-5\"]\ncache_ttl_secs = 0\n\
             [model.\"local/qwen\"]\ncache_ttl_secs = 900\n\
             [model.\"anthropic/claude-opus-5-5\"]\ncache_ttl = \"2h\"\ncache_ttl_secs = 7200\n\
             [model.\"chatgpt/gpt-6-astra\"]\ncache_ttl = \"1h\"\ncache_probabilistic = false\n\
             [model.\"local/flaky\"]\ncache_ttl_secs = 600\ncache_probabilistic = true\n",
        );
        assert!(!resolve("chatgpt/gpt-6-astra", &file, &d(128_000)).cache_probabilistic, "models.toml overrides the built-in");
        assert!(resolve("local/flaky", &file, &d(128_000)).cache_probabilistic);
        assert!(resolve("chatgpt/gpt-6-sol", &file, &d(128_000)).cache_probabilistic);
        // The lifetime follows the requested ttl unless one is set explicitly.
        assert_eq!(cache("anthropic/claude-opus-5", &file), (Some(CacheTtl::FiveMinutes), Some(300), true));
        assert_eq!(resolve("anthropic/claude-opus-5", &file, &d(128_000)).cold_start_tokens, 250_000);
        // 0 switches cold starts off; the 1h marker is still requested.
        assert_eq!(cache("anthropic/claude-sonnet-5", &file), (Some(CacheTtl::OneHour), None, false));
        assert_eq!(cache("local/qwen", &file), (None, Some(900), true));
        // An invalid ttl is ignored (built-in kept); the explicit lifetime wins.
        assert_eq!(cache("anthropic/claude-opus-5-5", &file), (Some(CacheTtl::OneHour), Some(7_200), true));
        assert_eq!(cache("chatgpt/gpt-6-astra", &file), (Some(CacheTtl::OneHour), Some(3_600), true));
        let env = Defaults { cold_start_tokens: 40_000, ..Defaults::default() };
        assert_eq!(resolve("dev-bot", &none, &env).cold_start_tokens, 40_000);
        // Every built-in cache entry is preceded by a source comment.
        let source = include_str!("model_limits.rs");
        let table = &source[source.find("const BUILT_IN_CACHE").unwrap()..];
        let table = &table[..table.find("];").unwrap()];
        let lines: Vec<&str> = table.lines().map(str::trim).collect();
        let entries: Vec<usize> = (0..lines.len()).filter(|&i| lines[i].starts_with("(\"")).collect();
        assert_eq!(entries.len(), BUILT_IN_CACHE.len());
        for i in entries {
            let comment = lines[..i].iter().rev().take_while(|l| l.starts_with("//")).cloned().collect::<Vec<_>>().join(" ");
            assert!(["MEASURED", "DOCS", "UNVERIFIED"].iter().any(|k| comment.contains(k)), "{}: {comment}", lines[i]);
            // An unknown lifetime says so, and why.
            if lines[i].ends_with("None),") {
                assert!(comment.contains("UNKNOWN"), "{}: {comment}", lines[i]);
            }
        }
        // Every probabilistic route is backed by a measurement.
        let table = &source[source.find("const PROBABILISTIC_CACHE").unwrap()..];
        let table = &table[..table.find("];").unwrap()];
        let lines: Vec<&str> = table.lines().map(str::trim).collect();
        let entries: Vec<usize> = (0..lines.len()).filter(|&i| lines[i].starts_with('"')).collect();
        assert_eq!(entries.len(), PROBABILISTIC_CACHE.len());
        for i in entries {
            let comment = lines[..i].iter().rev().take_while(|l| l.starts_with("//")).cloned().collect::<Vec<_>>().join(" ");
            assert!(comment.contains("MEASURED"), "{}: {comment}", lines[i]);
        }
    }

    #[test]
    fn model_limits_derived_values_scale_with_the_window_and_byte_limit() {
        let opus = resolve("anthropic/claude-opus-5", &FileConfig::default(), &d(128_000));
        assert_eq!(opus.trigger_tokens(), 850_000);
        assert_eq!(opus.target_tokens(), 500_000);
        assert_eq!(opus.max_message_bytes(), 312_500, "1/8 window at the default 2.5 bytes/token");
        let tight = Defaults { max_request_bytes: 2_000_000, ..Defaults::default() };
        assert_eq!(resolve("anthropic/claude-opus-5", &FileConfig::default(), &tight).max_message_bytes(), 250_000, "a 2 MB route bounds it");
        let small = resolve("x", &FileConfig::default(), &d(128_000));
        assert_eq!(small.max_message_bytes(), 40_000);
        assert_eq!(small.request_timeout, Duration::from_secs(300 + 960 + 25));
    }

    #[test]
    fn calibration_default_is_conservative() {
        // No measurement: 2.5 bytes/token, i.e. MORE tokens per byte than the old
        // 4 chars/token guess, so a projection errs towards an early checkpoint.
        assert_eq!(bytes_per_token(None, "m"), DEFAULT_BYTES_PER_TOKEN);
        assert_eq!(DEFAULT_BYTES_PER_TOKEN, 2.5);
        assert_eq!(tokens_for_bytes(1_000_000, bytes_per_token(None, "m")), 400_000);
        assert_eq!(bytes_for_tokens(400_000, bytes_per_token(None, "m")), 1_000_000);
        assert_eq!(tokens_for_bytes(9, 2.5), 4, "rounds up");
        // A non-finite or out-of-range ratio never reaches the projection.
        assert_eq!(tokens_for_bytes(1_000, f32::NAN), 400);
        assert_eq!(tokens_for_bytes(6_000, 0.0), 4_000);
        assert_eq!(tokens_for_bytes(6_000, 100.0), 1_000);
    }

    #[test]
    fn calibration_learns_the_measured_ratio() {
        // The live admin-DM measurement: ≈1.46 MB counted as 593,447 input tokens.
        let measured = TokenCalibration::measure("m", 1_460_000, 0, 593_447).unwrap();
        let ratio = bytes_per_token(Some(&measured), "m");
        assert!((ratio - 2.46).abs() < 0.01, "{ratio}");
        assert_eq!(tokens_for_bytes(1_460_000, ratio), 593_447);
        // Images are taken out of the text ratio at their fixed projection cost.
        let with_images = TokenCalibration::measure("m", 40_000, 2, 10_000 + 2 * IMAGE_TOKENS as u64).unwrap();
        assert_eq!(with_images, TokenCalibration { request_bytes: 40_000, input_tokens: 10_000, model: "m".into() });
        assert_eq!(with_images.bytes_per_token(), 4.0);
    }

    #[test]
    fn calibration_belongs_to_the_model_that_counted_it() {
        // Another model has another tokenizer: its projection starts from the
        // default again instead of inheriting this ratio.
        let measured = TokenCalibration::measure("anthropic/claude-opus-5", 400_000, 0, 100_000).unwrap();
        assert_eq!(bytes_per_token(Some(&measured), "anthropic/claude-opus-5"), 4.0);
        assert_eq!(bytes_per_token(Some(&measured), "chatgpt/gpt-6-astra"), DEFAULT_BYTES_PER_TOKEN);
        assert_eq!(bytes_per_token(Some(&measured), ""), DEFAULT_BYTES_PER_TOKEN);
        // A record written before the model was stored matches no model.
        let old: TokenCalibration = serde_json::from_str(r#"{"request_bytes":400000,"input_tokens":100000}"#).unwrap();
        assert_eq!(old.model, "");
        for model in ["anthropic/claude-opus-5", ""] {
            assert_eq!(bytes_per_token(Some(&old), model), DEFAULT_BYTES_PER_TOKEN, "{model:?}");
        }
        assert_eq!(serialized_len("a\nb"), 6);
        assert_eq!(serialized_len("ä"), 4);
    }

    #[test]
    fn calibration_clamps_and_ignores_unusable_usage() {
        // Zero or missing input tokens, an empty request, or a request too small to
        // say anything: no measurement, the previous one (or the default) stays.
        assert_eq!(TokenCalibration::measure("m", 500_000, 0, 0), None);
        assert_eq!(TokenCalibration::measure("m", 0, 0, 50_000), None);
        assert_eq!(TokenCalibration::measure("m", 4_000, 0, 1_999), None);
        assert_eq!(TokenCalibration::measure("m", 40_000, 3, 3 * IMAGE_TOKENS as u64), None, "only image tokens");
        assert!(TokenCalibration::measure("m", 4_000, 0, 2_000).is_some());
        // Implausible ratios are clamped, not trusted.
        let dense = TokenCalibration::measure("m", 10_000, 0, 100_000).unwrap();
        assert_eq!(dense.bytes_per_token(), MIN_BYTES_PER_TOKEN);
        let sparse = TokenCalibration::measure("m", 10_000_000, 0, 100_000).unwrap();
        assert_eq!(sparse.bytes_per_token(), MAX_BYTES_PER_TOKEN);
        // A hand-edited or corrupt persisted record cannot divide by zero.
        let corrupt = TokenCalibration { request_bytes: 10, input_tokens: 0, model: "m".into() };
        assert_eq!(corrupt.bytes_per_token(), DEFAULT_BYTES_PER_TOKEN);
    }

    #[test]
    fn calibration_round_trips_through_serde() {
        let measured = TokenCalibration { request_bytes: 1_460_000, input_tokens: 593_447, model: "a/b".into() };
        let json = serde_json::to_string(&measured).unwrap();
        assert_eq!(json, r#"{"request_bytes":1460000,"input_tokens":593447,"model":"a/b"}"#);
        assert_eq!(serde_json::from_str::<TokenCalibration>(&json).unwrap(), measured);
    }
}
