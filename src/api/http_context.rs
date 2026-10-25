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

pub(super) fn report_prompt_epoch(
    context: &str,
    key: &crate::context_prompt::StablePrefixKey,
    model: &str,
    stable: &str,
    history: &[crate::llm::ChatMessage],
    compacted: bool,
) {
    if let crate::context_log::Epoch::Reset { reason } =
        crate::context_log::observe_epoch(context, key, model, stable, history, compacted)
    {
        eprintln!("[llm:{context}] prompt-cache epoch reset: {reason}");
    }
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
        conv.push_assistant(content);
        conv.trim(51);
    }
    config.notify_save();
}
