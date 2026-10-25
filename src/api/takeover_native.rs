//! Native tool protocol for write-capable `llm_takeover` turns (#85).
//!
//! `handle_llm_takeover` still owns everything around the model loop: the
//! per-context lock, Ladybug ingest of the user turn and attachments, recall,
//! canonical memories, image recovery, episode summaries, the identity/speaker
//! prompt composition and the user message in session history. This module only
//! decides eligibility and turns the shared [`NativeLoop`] outcome into the reply
//! string the platform callback delivers. There is no second round loop.
use super::llm_handlers::{write_log_file, EventSender};
use super::native_turn::{LoopEnd, NativeLoop, RoundBudget};
use super::AppConfig;
use crate::llm::ChatMessage;

/// Native only when the process opted in and the pinned snapshot for this
/// speaker/context is write-capable and non-guest. Guests and execute-only
/// speakers keep the legacy scoped protocol unchanged.
pub(super) fn native_snapshot(
    native_tools: bool,
    config: &AppConfig,
    context: &str,
) -> Option<AppConfig> {
    if !native_tools {
        return None;
    }
    let snapshot = super::native_turn::permission_snapshot_for(config, context);
    super::native_tools::native_tools_eligible(&snapshot).then_some(snapshot)
}

/// Assistant-turn writer into Ladybug, mirroring the legacy takeover loop.
pub(super) struct LadybugSink {
    graph: std::sync::Arc<crate::memory_graph::MemoryGraph>,
    context: String,
}

impl LadybugSink {
    pub(super) fn new(
        graph: std::sync::Arc<crate::memory_graph::MemoryGraph>,
        context: &str,
    ) -> Self {
        Self {
            graph,
            context: context.to_string(),
        }
    }

    pub(super) async fn assistant(&self, content: String) -> anyhow::Result<()> {
        super::llm_handlers::persist_takeover_message(
            &self.graph,
            &self.context,
            "agent:kronk".to_string(),
            "assistant",
            content,
            None,
        )
        .await
        .map(|_| ())
    }
}

const INCOMPLETE_NOTE: &str = "(This turn ended before the task was confirmed complete. The conversation and completed operations are saved.)";

pub(super) struct TakeoverRun<'a> {
    /// Pinned permission snapshot from [`native_snapshot`].
    pub config: AppConfig,
    pub context: &'a str,
    /// Model that generates (not the permission profile).
    pub generator: &'a str,
    pub messages: Vec<ChatMessage>,
    pub training_context: String,
    pub memory_graph: std::sync::Arc<crate::memory_graph::MemoryGraph>,
}

/// Drive one native takeover turn and return the user-visible reply.
pub(super) async fn run(turn: TakeoverRun<'_>) -> anyhow::Result<String> {
    let config = turn.config;
    // Immutable completion baseline (#66), captured before any round.
    let baseline = config.program.read().await.clone();
    let mut native = NativeLoop::new(
        turn.messages,
        turn.training_context,
        turn.context,
        false,
        Some(LadybugSink::new(turn.memory_graph, turn.context)),
    );
    native.generator = Some(turn.generator.to_string());
    let mut budget = RoundBudget::takeover();
    let tx = EventSender::broadcast_only(config.event_broadcast.clone());
    native.drive(&config, &baseline, &mut budget, &tx, false).await;
    if let Some(error) = native.fatal.take() {
        return Err(error);
    }
    let reply = render_reply(&native, budget.generations());
    eprintln!(
        "[llm_takeover:{}] native turn end={:?} generations={:?}",
        turn.context,
        native.end,
        budget.generations()
    );
    write_log_file(
        &config.log_file,
        "ai-text",
        &format!("[{}] {reply}", turn.context),
    )
    .await;
    config.notify_save();
    Ok(reply)
}

/// The reply is always the model's visible prose, never tool JSON. When finish
/// was not accepted, the person is told completion is unconfirmed.
fn render_reply(native: &NativeLoop, generations: Option<usize>) -> String {
    let prose = native.prose.clone();
    let append = |note: &str| {
        if prose.is_empty() {
            note.to_string()
        } else {
            format!("{prose}\n\n{note}")
        }
    };
    match &native.end {
        LoopEnd::Finished => prose,
        LoopEnd::Empty(error) => error.to_string(),
        LoopEnd::ProviderError(error) => {
            let detail: String = error.chars().take(120).collect();
            append(&format!(
                "Entschuldigung — ich habe gerade ein technisches Problem und konnte \
                 die Antwort nicht abschließen. Bitte versuch es gleich noch einmal. ({detail})"
            ))
        }
        LoopEnd::Exhausted
            if generations
                .is_some_and(|used| used >= super::llm_handlers::MAX_TAKEOVER_ITERATION_BUDGET) =>
        {
            append(super::llm_handlers::TAKEOVER_CALL_LIMIT_REPLY)
        }
        LoopEnd::Exhausted => append(INCOMPLETE_NOTE),
    }
}

#[cfg(test)]
#[path = "takeover_native_tests.rs"]
mod tests;
