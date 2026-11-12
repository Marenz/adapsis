//! Per-request accounting for one conversation (#68).
//!
//! A turn's view is built once (`turn_view`, the only place a cold start can
//! happen), by a turn that holds the per-context turn lock (`/api/ask`,
//! `/api/ask-stream` and `llm_takeover` all hold `takeover_context_lock` for the
//! whole turn), so the per-conversation `TurnProbe` belongs to exactly one turn
//! at a time. The `!watch` self-trigger holds no lock and is not a user turn: it
//! builds a plain view and only calibrates bytes-per-token.
//! Every request of the turn is then measured before it is sent
//! (`SentRequest::measure`) and accounted when it is answered
//! (`SentRequest::calibrate`): bytes-per-token, the real runtime tail, the pause
//! since the previous request, and prompt-cache hit/miss evidence per route.
use crate::cache_lifetime::{self, Outcome, TurnProbe};
use crate::session::ConversationManager;
use crate::session::conversation_view::{View, ViewPolicy};

/// One request as sent, measured before the call so the provider's usage report
/// can calibrate the conversation's projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SentRequest {
    /// The generating model: whose tokenizer counted, and whose cache was used.
    pub(super) model: String,
    /// Serialized bytes of the messages (plus tool schemas), image data excluded.
    pub(super) text_bytes: usize,
    pub(super) images: usize,
    /// When the request was sent (unix ms). Providers count a cache lifetime
    /// from the start of the request that wrote or read the entry.
    pub(super) started_ms: u64,
    /// False when the request was sent with another tool list than the turn's
    /// first request (the final status round has none): the provider caches the
    /// tools ahead of everything else, so no hit can be expected.
    pub(super) same_tools: bool,
}

/// Serialized bytes of `messages` without image data, and the image count.
fn text_bytes_and_images(messages: &[crate::llm::ChatMessage]) -> (usize, usize) {
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
    (count.0.saturating_sub(image_bytes), images)
}

pub(super) fn text_bytes(messages: &[crate::llm::ChatMessage]) -> usize {
    text_bytes_and_images(messages).0
}

impl SentRequest {
    /// `extra_bytes`: serialized size of request parts outside the messages
    /// (native tool schemas).
    pub(crate) fn measure(model: &str, messages: &[crate::llm::ChatMessage], extra_bytes: usize) -> Self {
        let (text, images) = text_bytes_and_images(messages);
        Self {
            model: model.to_string(),
            text_bytes: text + extra_bytes,
            images,
            started_ms: cache_lifetime::now_ms(),
            same_tools: true,
        }
    }

    /// This request carries another tool list than the turn's first request: its
    /// cache reads are neither a hit nor a miss. Sizes are still calibrated.
    pub(crate) fn with_other_tools(mut self) -> Self {
        self.same_tools = false;
        self
    }

    /// Account this answered request with its conversation.
    ///
    /// - Bytes-per-token: stored for this model unless the client had to shrink
    ///   the request (`recovered`: the bytes measured here are then not what the
    ///   provider counted) or the usage report has no usable input tokens.
    /// - Runtime tail: the first request of a turn beyond the turn's view.
    /// - Cache evidence: the pause since the previous request and whether the
    ///   cache hit, recorded per route when a hit could be expected.
    ///
    /// A shrunk or `refused` request is no cache evidence and is not assumed to
    /// have written the cache; a refused one does not count as the conversation's
    /// last request either. Requests that failed never get here.
    ///
    /// Never moves the view: a pause between two rounds of one turn is evidence,
    /// not a reason to drop context.
    pub(crate) fn calibrate(
        self,
        meta: &crate::session::SharedMeta,
        context: &str,
        recovered: bool,
        refused: bool,
        usage: &crate::llm::LlmUsage,
    ) {
        let Ok(mut meta) = meta.lock() else { return };
        let manager = &mut meta.conversations;
        let conv = manager.get_or_create(context);
        self.calibrate_tokens(conv, context, recovered, usage);
        let ratio = conv.bytes_per_token(&self.model);
        let pause_secs = conv.last_request_ms.map(|last| self.started_ms.saturating_sub(last) / 1000);
        if !refused {
            conv.last_request_ms = Some(self.started_ms);
        }
        let mut expected_tokens = None;
        if let Some(probe) = conv.turn_probe.as_mut() {
            if recovered || refused {
                // Nothing this turn can be expected to read what this request wrote.
                (probe.first_prefix_bytes, probe.turn_prefix_bytes) = (None, None);
            } else {
                expected_tokens = probe
                    .expected_prefix_bytes()
                    .filter(|_| self.same_tools)
                    .map(|bytes| crate::model_limits::tokens_for_bytes(bytes, ratio));
                if probe.requests == 0 {
                    // The request wrote the prefix the next turn would read, if
                    // it was built with explicit boundaries (ContextLog).
                    probe.wrote_prefix = probe.turn_prefix_bytes.is_some();
                    conv.tail_bytes = Some(self.text_bytes.saturating_sub(probe.view_bytes) as u64);
                }
            }
            probe.requests += 1;
        }
        let limits = crate::model_limits::for_model(&self.model);
        // On a probabilistic route an uncached answer is not a miss.
        let outcome =
            cache_lifetime::classify_for(expected_tokens, usage.cache_read_input_tokens, limits.cache_probabilistic);
        if let Some(pause) = pause_secs {
            manager.cache_lifetimes.record(&self.model, outcome, pause, self.started_ms);
        }
        let lifetime = manager.cache_lifetimes.lifetime(&self.model, &limits, self.started_ms);
        log_cache_metric(context, pause_secs, expected_tokens, usage, outcome, lifetime);
    }

    /// Bytes-per-token only, for a request that is not part of a user turn (the
    /// `!watch` self-trigger). Such a request holds no turn lock and carries no
    /// cache boundaries, so it must not touch the turn accounting: it is not the
    /// conversation's last request for the pause, measures no runtime tail and
    /// is no cache evidence.
    pub(crate) fn calibrate_tokens_only(
        self,
        meta: &crate::session::SharedMeta,
        context: &str,
        recovered: bool,
        usage: &crate::llm::LlmUsage,
    ) {
        if let Ok(mut meta) = meta.lock() {
            self.calibrate_tokens(meta.conversations.get_or_create(context), context, recovered, usage);
        }
    }

    fn calibrate_tokens(
        &self,
        conv: &mut crate::session::Conversation,
        context: &str,
        recovered: bool,
        usage: &crate::llm::LlmUsage,
    ) {
        if recovered {
            return;
        }
        let Some(measured) = crate::model_limits::TokenCalibration::measure(
            &self.model,
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
        conv.token_calibration = Some(measured);
    }
}

fn log_cache_metric(
    context: &str,
    pause_secs: Option<u64>,
    expected_tokens: Option<usize>,
    usage: &crate::llm::LlmUsage,
    outcome: Outcome,
    lifetime: Option<cache_lifetime::Lifetime>,
) {
    let number = |value: Option<u64>| value.map_or_else(|| "n/a".to_string(), |v| v.to_string());
    eprintln!(
        "[llm:{context}] cache: pause_s={} expected_tokens={} cached_read={} outcome={} lifetime_s={} source={}",
        number(pause_secs),
        number(expected_tokens.map(|tokens| tokens as u64)),
        number(usage.cache_read_input_tokens),
        outcome.label(),
        number(lifetime.map(|lifetime| lifetime.secs)),
        lifetime.map_or("unknown", |lifetime| lifetime.source.label()),
    );
}

/// The model view of `context` at the START OF A TURN, for the generating
/// `model`. This is the only place a cold start happens: when the pause since
/// the conversation's last answered request is at least the route's effective
/// cache lifetime, the cache is gone anyway, so the view is cut once to the
/// model's `cold_start_tokens` (never to more than a budget checkpoint would
/// leave) and grows append-only from there. An unknown lifetime or an unknown
/// pause (no earlier request recorded) never cold-starts. The persisted history
/// is untouched. Call under the metadata lock; never awaits.
///
/// `show_workspace`: whether this turn may see the conversation's workspace tool
/// batches (file content of the owner's repositories). Only a turn that has the
/// workspace tools itself does; for every other turn (another speaker class, a
/// request from another node, a legacy turn, a `!watch` event) those batches are
/// left out of the view whole, with a notice. The projection (checkpoint, cold
/// start) is taken on the unfiltered history, which is the larger one. A turn
/// that sees less than the previous one starts a new cache epoch by itself: its
/// history is not an extension of the previous request's.
/// NOT covered (#145): the owner's questions and the model's own prose in the
/// same conversation stay visible; if the model wrote "the file says X", X is
/// in the history.
pub(super) fn turn_view(
    manager: &mut ConversationManager,
    context: &str,
    model: &str,
    system: &str,
    known_tail_bytes: usize,
    now_ms: u64,
    show_workspace: bool,
) -> View {
    let limits = crate::model_limits::for_model(model);
    let lifetime = manager.cache_lifetimes.lifetime(model, &limits, now_ms);
    let conv = manager.get_or_create(context);
    let mut policy = ViewPolicy::for_model(model, system, known_tail_bytes, conv.projection(model));
    if !show_workspace {
        policy.hidden_tools = Some(super::native_tools::workspace::is_workspace_name);
    }
    let pause_secs = conv.last_request_ms.map(|last| now_ms.saturating_sub(last) / 1000);
    let mut cold = false;
    if let (Some(lifetime), Some(pause)) = (lifetime, pause_secs) {
        let from = conv.view_start;
        let keep = limits.cold_start_tokens.min(policy.target_tokens.saturating_sub(policy.overhead_tokens));
        if pause >= lifetime.secs {
            if let Some((before, after)) = conv.cold_start(&policy, keep) {
                cold = true;
                eprintln!(
                    "[llm:{context}] cold start: pause_s={pause} lifetime_s={} ({}) history_tokens={before}->{after} view_start={from}->{}",
                    lifetime.secs,
                    lifetime.source.label(),
                    conv.view_start
                );
            }
        }
    }
    let mut view = conv.model_view(&policy);
    view.checkpointed |= cold;
    conv.turn_probe = Some(TurnProbe {
        view_bytes: text_bytes(&view.messages),
        previous_wrote: conv.turn_probe.as_ref().is_some_and(|previous| previous.wrote_prefix),
        ..Default::default()
    });
    view
}

/// Tell the turn's accounting which cached prefix its requests can expect to
/// read (the stable prefix plus the history up to a cache boundary):
/// - the first request reads what the previous turn's first request wrote, if
///   the history only appended since (`previous_len`) and that request was
///   answered normally;
/// - later rounds read what this turn's first request writes: the history as
///   of turn start.
pub(super) fn expect_cached_prefix(
    manager: &mut ConversationManager,
    context: &str,
    stable: &str,
    history: &[crate::llm::ChatMessage],
    previous_len: Option<usize>,
) {
    let Some(probe) = manager.get_mut(context).and_then(|conv| conv.turn_probe.as_mut()) else { return };
    let stable_bytes = crate::model_limits::serialized_len(stable);
    let prefix = |upto: usize| {
        stable_bytes + text_bytes(&history[..crate::context_log::cached_history_len(history, upto)])
    };
    probe.first_prefix_bytes = previous_len.filter(|_| probe.previous_wrote).map(prefix);
    probe.turn_prefix_bytes = Some(prefix(history.len()));
}

#[cfg(test)]
#[path = "request_meter_tests.rs"]
mod tests;
