//! Shared HTTP prompt and observability plumbing.
use super::AppConfig;

pub(super) fn main_context_prompts(
    config: &AppConfig,
    program_summary: &str,
    caller: Option<&str>,
) -> (
    String,
    String,
    crate::context_prompt::StablePrefixKey,
    String,
) {
    context_prompts(config, program_summary, caller, false)
}

pub(super) fn context_prompts(
    config: &AppConfig,
    program_summary: &str,
    caller: Option<&str>,
    native: bool,
) -> (
    String,
    String,
    crate::context_prompt::StablePrefixKey,
    String,
) {
    let model = config.llm_model.read().unwrap().clone();
    let available_models = if native {
        Vec::new()
    } else {
        config.permission_config.model_names()
    };
    let speaker = caller
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map_or_else(|| "local:api".to_string(), |s| format!("peer:{s}"));
    let inputs = crate::context_prompt::PromptInputs {
        context: "main",
        program_summary,
        llm_model: &model,
        available_models: &available_models,
        speaker: &speaker,
        authority: config
            .permission_config
            .authority(config.access_level, &model),
    };
    let (stable, key) = if native {
        crate::context_prompt::native_prefix_keyed(&inputs)
    } else {
        crate::context_prompt::stable_prefix_keyed(&inputs)
    };
    (
        stable,
        crate::context_prompt::volatile_suffix(&inputs),
        key,
        model,
    )
}

/// Observe the cache epoch for this request and mark the previous request's
/// history end as a cache breakpoint when the history only appended (#68).
pub(super) fn report_prompt_epoch(
    context: &str,
    key: &crate::context_prompt::StablePrefixKey,
    model: &str,
    stable: &str,
    log: &mut crate::context_log::ContextLog,
    checkpointed: bool,
) {
    match crate::context_log::observe_epoch(context, key, model, stable, log.history(), checkpointed) {
        crate::context_log::Epoch::Reset { reason } => {
            eprintln!("[llm:{context}] prompt-cache epoch reset: {reason}");
        }
        crate::context_log::Epoch::Continued { previous_len } => log.set_turn_boundary(previous_len),
    }
}

/// Recovered reference images (sent in the runtime tail) must fit both the image
/// budget and the route's remaining request headroom after the view and tail.
/// Newest-first order from the graph is kept; images that don't fit are dropped.
pub(super) fn fit_recovered_images(
    model: &str,
    view_bytes: usize,
    images: Vec<crate::attachment::Attachment>,
) -> Vec<crate::attachment::Attachment> {
    fit_images_to(&crate::model_limits::for_model(model), view_bytes, images)
}

fn fit_images_to(
    limits: &crate::model_limits::ModelLimits,
    view_bytes: usize,
    images: Vec<crate::attachment::Attachment>,
) -> Vec<crate::attachment::Attachment> {
    let policy = crate::session::conversation_view::ViewPolicy::from_limits(
        limits,
        "",
        crate::model_limits::DEFAULT_BYTES_PER_TOKEN,
    );
    let headroom = limits
        .max_request_bytes
        .saturating_sub(view_bytes + crate::session::conversation_view::ViewPolicy::TAIL_ALLOWANCE_BYTES);
    let mut budget = policy.image_bytes.min(headroom / 4 * 3);
    images
        .into_iter()
        .filter(|image| {
            let fits = image.len() <= budget;
            if fits {
                budget -= image.len();
            }
            fits
        })
        .collect()
}

/// The checkpointed model view of `context` for the generating `model` (#68),
/// with the stable system prompt installed. Call under the metadata lock; it
/// never awaits.
pub(super) fn checkpointed_view(
    conv: &mut crate::session::Conversation,
    model: &str,
    system: &str,
    known_tail_bytes: usize,
) -> crate::session::conversation_view::View {
    conv.set_primary_system(system.to_string());
    // Tokens are projected with this conversation's measured bytes-per-token.
    let policy = crate::session::conversation_view::ViewPolicy::for_model(
        model,
        system,
        known_tail_bytes,
        conv.bytes_per_token(),
    );
    conv.model_view(&policy)
}

/// The size of one request as sent, measured before the call so the provider's
/// usage report can calibrate the conversation's token projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SentRequest {
    /// Serialized bytes of the messages (plus tool schemas), image data excluded.
    text_bytes: usize,
    images: usize,
}

impl SentRequest {
    /// `extra_bytes`: serialized size of request parts outside the messages
    /// (native tool schemas).
    pub(crate) fn measure(messages: &[crate::llm::ChatMessage], extra_bytes: usize) -> Self {
        struct Count(usize);
        impl std::io::Write for Count {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0 += buf.len();
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut count = Count(0);
        let (mut images, mut image_bytes) = (0, 0);
        for message in messages {
            let _ = serde_json::to_writer(&mut count, message);
            if let crate::llm::ChatContent::Parts(parts) = &message.content {
                for part in parts {
                    if let crate::llm::ChatContentPart::ImageUrl { image_url } = part {
                        images += 1;
                        image_bytes += image_url.url.len();
                    }
                }
            }
        }
        Self { text_bytes: count.0.saturating_sub(image_bytes) + extra_bytes, images }
    }

    /// Store this request's measured bytes-per-token with the conversation. Not
    /// for a request the client had to shrink (`recovered`): the bytes measured
    /// here are then not what the provider counted. Usage reports without input
    /// tokens are ignored.
    pub(crate) fn calibrate(
        self,
        meta: &crate::session::SharedMeta,
        context: &str,
        recovered: bool,
        usage: &crate::llm::LlmUsage,
    ) {
        if recovered {
            return;
        }
        let Some(measured) = crate::model_limits::TokenCalibration::measure(
            self.text_bytes,
            self.images,
            usage.accounted_input_tokens(),
        ) else {
            return;
        };
        eprintln!(
            "[llm:{context}] token calibration: {} bytes / {} input tokens = {:.2} bytes/token",
            measured.request_bytes,
            measured.input_tokens,
            measured.bytes_per_token()
        );
        if let Ok(mut meta) = meta.lock() {
            meta.conversations.get_or_create(context).token_calibration = Some(measured);
        }
    }
}

/// Self-trigger (`!watch`) request for the `main` conversation (#68): the event is
/// stored as a user-role runtime notice (a role:"tool" message without a
/// tool_call_id is not a valid provider message) and the request is the bounded,
/// checkpointed view, never the full persisted history. Call under the lock.
pub(crate) fn self_trigger_messages(
    meta: &mut crate::session::SessionMeta,
    event: &str,
    model: &str,
) -> Vec<crate::llm::ChatMessage> {
    let conv = meta.conversations.get_or_create("main");
    conv.push_user(format!(
        "{}\n{event}",
        crate::session::conversation_view::RUNTIME_NOTICE_HEADER
    ));
    let system = conv
        .messages
        .first()
        .filter(|message| message.role == "system")
        .map(|message| message.content.clone())
        .unwrap_or_default();
    let policy =
        crate::session::conversation_view::ViewPolicy::for_model(model, &system, 0, conv.bytes_per_token());
    conv.model_view(&policy).messages
}

/// Measured serialized size of runtime-tail parts known before the view.
pub(super) fn tail_bytes(parts: &[&str]) -> usize {
    parts.iter().map(|part| serde_json::to_string(part).map_or(part.len(), |json| json.len())).sum()
}

/// Measured serialized size of the native tool schemas sent with every request.
pub(super) fn tool_schema_bytes(config: &AppConfig) -> usize {
    schema_bytes(
        super::native_tools::tool_definitions(config)
            .iter()
            .map(|tool| (tool.name.as_str(), tool.description.as_str(), &tool.parameters)),
    )
}

/// Serialized size of tool schemas given as `(name, description, parameters)`.
pub(super) fn schema_bytes<'a>(tools: impl Iterator<Item = (&'a str, &'a str, &'a serde_json::Value)>) -> usize {
    tools
        .map(|(name, description, parameters)| name.len() + description.len() + parameters.to_string().len() + 64)
        .sum()
}

/// After an LLM call (#68): if the client had to shrink a too-large request, or
/// the call failed as too large, advance the persisted anchor of `context` so the
/// conversation can never stay stuck; later turns start from the newest turn.
/// Returns whether a forced checkpoint was attempted (callers that do not save
/// at the end of the turn must then `notify_save`).
pub(crate) fn after_llm_call<T>(
    recovered: bool,
    result: &anyhow::Result<T>,
    meta: &crate::session::SharedMeta,
    context: &str,
) -> bool {
    let too_large = result.as_ref().err().is_some_and(crate::llm::is_too_large);
    if !(recovered || too_large) {
        return false;
    }
    let moved = meta
        .lock()
        .map(|mut meta| meta.conversations.get_or_create(context).force_view_checkpoint())
        .unwrap_or(false);
    eprintln!("[llm:{context}] request too large: forced view checkpoint (moved={moved}, retried_reduced={recovered})");
    true
}

pub(super) fn report_llm_usage(
    context: &str,
    started: std::time::Instant,
    usage: &crate::llm::LlmUsage,
) {
    eprintln!(
        "[llm:{context}] latency_ms={} usage: {}",
        started.elapsed().as_millis(),
        usage.summary()
    );
}

pub(super) fn persist_main_assistant(config: &AppConfig, content: String) {
    {
        let mut meta = config.meta.lock().unwrap();
        let conv = meta.conversations.get_or_create("main");
        // #68: persisted history is never trimmed; the model view checkpoints.
        conv.push_assistant(content);
    }
    config.notify_save();
}

#[cfg(test)]
mod tests {
    #[test]
    fn recovered_images_fit_the_route_headroom() {
        let image = |n: usize| crate::attachment::Attachment::from_bytes(vec![1u8; n], "image/png", "r.png").unwrap();
        let mut limits = crate::model_limits::for_model("dev-bot");
        // Default 60 MiB route: three normal photos fit.
        assert_eq!(super::fit_recovered_images("dev-bot", 100_000, vec![image(200_000), image(200_000), image(200_000)]).len(), 3);
        // A 2 MB route: images limited to ~450 KB raw in total.
        limits.max_request_bytes = 2_000_000;
        let kept = super::fit_images_to(&limits, 100_000, vec![image(200_000), image(200_000), image(200_000)]);
        assert_eq!(kept.len(), 2);
        // Headroom = limit − view − tail allowance; images must fit it as base64.
        let allowance = crate::session::conversation_view::ViewPolicy::TAIL_ALLOWANCE_BYTES;
        let full = 2_000_000 - allowance - 1_000;
        assert!(super::fit_images_to(&limits, full, vec![image(10_000)]).is_empty(), "no headroom");
        assert_eq!(super::fit_images_to(&limits, full - 20_000, vec![image(10_000)]).len(), 1, "fits once there is room");
    }
}

#[cfg(test)]
mod self_trigger_tests {
    #[test]
    fn watch_self_trigger_request_is_bounded_and_uses_a_user_notice() {
        let mut meta = crate::session::SessionMeta::new();
        {
            let conv = meta.conversations.get_or_create("main");
            conv.push_system("stable prefix");
            // ~2M chars: well past an unknown model's 128k-token window.
            for i in 0..200 {
                conv.push_user(format!("u{i} {}", "x".repeat(5_000)));
                conv.push_assistant(format!("a{i} {}", "y".repeat(5_000)));
            }
        }
        let messages = super::self_trigger_messages(&mut meta, "Watcher 'W.f' triggered: 1 -> 2", "unknown/test-model");
        let limits = crate::model_limits::for_model("unknown/test-model");
        let bytes: usize = messages.iter().map(|m| serde_json::to_string(m).unwrap().len()).sum();
        let tokens = crate::model_limits::tokens_for_bytes(bytes, crate::model_limits::DEFAULT_BYTES_PER_TOKEN);
        assert!(tokens < limits.context_tokens, "bounded: {bytes} bytes");
        let last = messages.last().unwrap();
        assert_eq!(last.role, crate::llm::ChatRole::User);
        let text = serde_json::to_string(last).unwrap();
        assert!(text.contains("Runtime notice") && text.contains("Watcher 'W.f' triggered"), "{text}");
        assert!(messages.iter().all(|m| m.role != crate::llm::ChatRole::Tool), "no tool message without a call id");
        let conv = meta.conversations.get("main").unwrap();
        assert_eq!(conv.messages.len(), 402, "persisted history untouched");
        assert!(conv.view_start > 0, "the checkpoint anchor was persisted");
    }
}

#[cfg(test)]
mod recovery_anchor_tests {
    #[test]
    fn too_large_failures_and_shrunk_retries_persist_a_forced_checkpoint() {
        let meta: crate::session::SharedMeta = std::sync::Arc::new(std::sync::Mutex::new(crate::session::SessionMeta::new()));
        let setup = |meta: &crate::session::SharedMeta| {
            let mut guard = meta.lock().unwrap();
            let conv = guard.conversations.get_or_create("ctx");
            *conv = crate::session::Conversation::new();
            conv.push_system("s");
            conv.push_user("old");
            conv.push_assistant("a");
            conv.push_user("newest");
        };
        let view_start = |meta: &crate::session::SharedMeta| meta.lock().unwrap().conversations.get("ctx").unwrap().view_start;
        setup(&meta);
        super::after_llm_call(false, &Ok::<(), anyhow::Error>(()), &meta, "ctx");
        assert_eq!(view_start(&meta), 0, "a normal call changes nothing");
        super::after_llm_call(true, &Ok::<(), anyhow::Error>(()), &meta, "ctx");
        assert_eq!(view_start(&meta), 3, "a shrunk retry persists the anchor at the newest turn");
        setup(&meta);
        let refused: anyhow::Result<()> = Err(crate::llm::test_too_large_error());
        super::after_llm_call(false, &refused, &meta, "ctx");
        assert_eq!(view_start(&meta), 3, "an unrecoverable too-large failure persists it too");
    }
}

#[cfg(test)]
mod calibration_tests {
    use super::SentRequest;
    use crate::llm::{ChatMessage, LlmUsage};

    fn meta() -> crate::session::SharedMeta {
        std::sync::Arc::new(std::sync::Mutex::new(crate::session::SessionMeta::new()))
    }

    fn usage(prompt: u64, read: Option<u64>, created: Option<u64>) -> LlmUsage {
        LlmUsage { prompt_tokens: prompt, cache_read_input_tokens: read, cache_creation_input_tokens: created, ..Default::default() }
    }

    fn ratio(meta: &crate::session::SharedMeta) -> f32 {
        meta.lock().unwrap().conversations.get_or_create("ctx").bytes_per_token()
    }

    #[test]
    fn calibration_measures_sent_text_bytes_without_image_data() {
        let text = vec![ChatMessage::system("s".repeat(1_000)), ChatMessage::user("u".repeat(2_000))];
        let plain = SentRequest::measure(&text, 0);
        let wire: usize = text.iter().map(|m| serde_json::to_string(m).unwrap().len()).sum();
        assert_eq!(plain, SentRequest { text_bytes: wire, images: 0 });
        assert_eq!(SentRequest::measure(&text, 500).text_bytes, wire + 500, "tool schemas are part of the request");
        // The same request with a 300 KB image: the base64 data is not text.
        let photo = crate::attachment::Attachment::from_bytes(vec![7u8; 300_000], "image/png", "p.png").unwrap();
        let with_image = vec![text[0].clone(), ChatMessage::user_with_attachments("u".repeat(2_000), &[photo])];
        let measured = SentRequest::measure(&with_image, 0);
        assert_eq!(measured.images, 1);
        assert!(measured.text_bytes < wire + 200, "image bytes excluded: {} vs {wire}", measured.text_bytes);
    }

    #[test]
    fn calibration_from_provider_usage_drives_the_next_view() {
        let meta = meta();
        let sent = SentRequest { text_bytes: 1_460_000, images: 0 };
        assert_eq!(ratio(&meta), crate::model_limits::DEFAULT_BYTES_PER_TOKEN);
        // Anthropic shape: the total processed input is prompt + read + created.
        sent.calibrate(&meta, "ctx", false, &usage(3_447, Some(0), Some(590_000)));
        assert!((ratio(&meta) - 2.46).abs() < 0.01, "{}", ratio(&meta));
        // ChatGPT shape: cache reads are inside prompt_tokens.
        sent.calibrate(&meta, "ctx", false, &usage(365_000, Some(300_000), None));
        assert_eq!(ratio(&meta), 4.0);
        // Missing/zero usage and a request the client shrank leave it unchanged.
        sent.calibrate(&meta, "ctx", false, &LlmUsage::default());
        sent.calibrate(&meta, "ctx", true, &usage(1_000_000, None, None));
        assert_eq!(ratio(&meta), 4.0);
        // Implausible reports are clamped.
        sent.calibrate(&meta, "ctx", false, &usage(5_000_000, None, None));
        assert_eq!(ratio(&meta), crate::model_limits::MIN_BYTES_PER_TOKEN);
        // The stored ratio is the one the next checkpointed view projects with.
        let mut guard = meta.lock().unwrap();
        let conv = guard.conversations.get_or_create("ctx");
        // ≈1.2 MB: under the trigger at the default ratio, far over it at 1.5.
        for i in 0..200 {
            conv.push_user(format!("q{i} {}", "u".repeat(3_000)));
            conv.push_assistant(format!("a{i} {}", "r".repeat(3_000)));
        }
        conv.push_user("newest");
        let mut uncalibrated = conv.clone();
        uncalibrated.token_calibration = None;
        assert!(!super::checkpointed_view(&mut uncalibrated, "anthropic/claude-opus-5", "s", 0).checkpointed);
        assert!(super::checkpointed_view(conv, "anthropic/claude-opus-5", "s", 0).checkpointed);
        // Each context keeps its own measurement.
        assert_eq!(guard.conversations.get_or_create("other").bytes_per_token(), crate::model_limits::DEFAULT_BYTES_PER_TOKEN);
    }
}
