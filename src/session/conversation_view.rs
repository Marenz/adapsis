//! Checkpointed model view of a conversation (#68).
//!
//! The provider-visible history is append-only and byte-stable between explicit
//! checkpoints, so prompt caches keep hitting. A checkpoint fires only when the
//! projected request (input + reserved output) exceeds ~85% of the generating
//! model's window; it then advances `view_start` in ONE step so the projection
//! drops to ~50%, and the history grows append-only again. Persisted messages are
//! never removed; older content stays in the session and in Ladybug (episodes).
//!
//! Two budgets apply, whichever is tighter: projected tokens against the model
//! window, and serialized bytes (image base64 included) against the route's
//! request body limit. Tokens are projected from serialized text bytes with the
//! conversation's calibrated bytes-per-token ratio (measured from the provider's
//! usage report for the previous request; a conservative default before that),
//! plus a fixed allowance per image.
//!
//! Everything that decides what a message looks like in the view is a function of
//! the message and of the persisted checkpoint indices only: per-message caps are
//! constant per model, and image retention is anchored at `image_start`, which
//! moves only when the image budget overflows. No budget-dependent truncation.
use super::{ChatMessage, Conversation};

const IMAGE_COUNT: usize = 3;
/// Raw image byte budget ceiling; the policy may lower it to fit the route.
const IMAGE_BYTES: usize = 20 * 1024 * 1024;
use crate::model_limits::{IMAGE_TOKENS, tokens_for_bytes};
const OMITTED_NOTICE: &str = "older message(s) are omitted from this model context; the raw history is retained.";

/// Budget inputs for one request, all in estimated tokens.
#[derive(Debug, Clone, Copy)]
pub struct ViewPolicy {
    pub trigger_tokens: usize,
    pub target_tokens: usize,
    /// Everything outside the history: system prefix, runtime tail allowance and
    /// reserved output.
    pub overhead_tokens: usize,
    /// Per-message cap in UTF-8 bytes of the message text.
    pub max_message_bytes: usize,
    /// Serialized request byte budget (85% / 50% of the route body limit) and
    /// the bytes outside the history (system prefix, tail, tool schemas, framing).
    pub trigger_bytes: usize,
    pub target_bytes: usize,
    pub overhead_bytes: usize,
    /// Raw bytes of retained images (sent as base64, 4/3 larger).
    pub image_bytes: usize,
    /// Serialized text bytes per provider token for this conversation.
    pub bytes_per_token: f32,
}

/// What a conversation's previous requests measured, for projecting the next.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Projection {
    /// Serialized text bytes per provider token.
    pub bytes_per_token: f32,
    /// Real size of the previous turn's runtime tail (None: never measured).
    pub tail_bytes: Option<usize>,
}

impl ViewPolicy {
    /// Allowance for the part of the volatile runtime tail that is assembled only
    /// after the view (memories, episodes, activity, recovered images). Known
    /// parts (authority/runtime text, tool schemas) are measured and passed in.
    /// An estimate for the projection, not a cap; a request that still exceeds the
    /// route limit self-heals once (`llm::shrink_request` + forced checkpoint).
    pub const TAIL_ALLOWANCE_TOKENS: usize = 16_000;
    pub const TAIL_ALLOWANCE_BYTES: usize = 64 * 1024;

    /// `known_tail_bytes`: measured size of the tail parts known before the view
    /// (runtime/authority text, native tool schemas). `projection`: what the
    /// conversation's previous requests measured (`Conversation::projection`).
    pub fn for_model(model: &str, system: &str, known_tail_bytes: usize, projection: Projection) -> Self {
        let limits = crate::model_limits::for_model(model);
        Self::from_limits(&limits, system, projection.bytes_per_token).with_tail(
            known_tail_bytes,
            projection.tail_bytes.map(|tail| tail.min(limits.max_request_bytes / 4)),
        )
    }

    /// Account the runtime tail. With a measurement (the previous turn's real
    /// request minus its view: memories, episodes, activity, authority text and
    /// tool schemas together) it replaces the fixed allowance, with a quarter of
    /// margin because recall differs per turn; a larger known part still wins.
    /// Without one, the allowance covers the unknown part beside the known one.
    fn with_tail(mut self, known_tail_bytes: usize, measured_tail_bytes: Option<usize>) -> Self {
        let tail = match measured_tail_bytes {
            Some(measured) => {
                self.overhead_bytes -= Self::TAIL_ALLOWANCE_BYTES;
                self.overhead_tokens -= Self::TAIL_ALLOWANCE_TOKENS;
                (measured + measured / 4).max(known_tail_bytes)
            }
            None => known_tail_bytes,
        };
        self.overhead_bytes += tail;
        self.overhead_tokens += tokens_for_bytes(tail, self.bytes_per_token);
        self
    }

    pub fn from_limits(limits: &crate::model_limits::ModelLimits, system: &str, bytes_per_token: f32) -> Self {
        let max_bytes = limits.max_request_bytes;
        Self {
            bytes_per_token,
            trigger_tokens: limits.trigger_tokens(),
            target_tokens: limits.target_tokens(),
            overhead_tokens: tokens_for_bytes(json_len(system), bytes_per_token)
                + Self::TAIL_ALLOWANCE_TOKENS
                + limits.output_reserve_tokens(),
            max_message_bytes: limits.max_message_bytes(),
            trigger_bytes: max_bytes * crate::model_limits::CHECKPOINT_TRIGGER_PERCENT / 100,
            target_bytes: max_bytes * crate::model_limits::CHECKPOINT_TARGET_PERCENT / 100,
            overhead_bytes: json_len(system) + Self::TAIL_ALLOWANCE_BYTES,
            // At most 30% of the route limit in base64: 3 typical photos fit.
            image_bytes: IMAGE_BYTES.min(max_bytes * 3 / 10 * 3 / 4),
        }
    }

    /// Test helper: token budget only (byte budget effectively unlimited), at the
    /// default bytes-per-token ratio.
    #[cfg(test)]
    pub fn tokens_only(trigger_tokens: usize, target_tokens: usize, overhead_tokens: usize, max_message_bytes: usize) -> Self {
        Self {
            trigger_tokens, target_tokens, overhead_tokens, max_message_bytes,
            trigger_bytes: usize::MAX / 4, target_bytes: usize::MAX / 4, overhead_bytes: 0,
            image_bytes: IMAGE_BYTES,
            bytes_per_token: crate::model_limits::DEFAULT_BYTES_PER_TOKEN,
        }
    }

    fn over(&self, tokens: usize, bytes: usize) -> bool {
        self.overhead_tokens + tokens > self.trigger_tokens || self.overhead_bytes + bytes > self.trigger_bytes
    }

    fn within_target(&self, tokens: usize, bytes: usize) -> bool {
        self.overhead_tokens + tokens <= self.target_tokens && self.overhead_bytes + bytes <= self.target_bytes
    }
}

/// The model view of one over-long message: head and tail with a marker between
/// them that tells the model the message was cut and by how many characters. The
/// persisted message is never changed. The result has at most `max_bytes` UTF-8
/// bytes (cut on character boundaries) and depends only on the message and the
/// cap (stable prefix).
fn truncate_middle(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_string();
    }
    let floor = |mut index: usize| {
        while !value.is_char_boundary(index) {
            index -= 1;
        }
        index
    };
    let ceil = |mut index: usize| {
        while !value.is_char_boundary(index) {
            index += 1;
        }
        index
    };
    // Room is reserved for the widest possible count, so the cut never depends
    // on the marker's own length. Both templates are ASCII.
    let digits = value.len().to_string().len();
    let long = "\n[... {} characters omitted here by the runtime to fit the model context; the stored message is complete ...]\n";
    let short = "\n[... {} characters omitted ...]\n";
    let width = |template: &str| template.len() - 2 + digits;
    let template = if max_bytes >= 4 * width(long) {
        long
    } else if max_bytes >= width(short) + 16 {
        short
    } else {
        return value[..floor(max_bytes)].to_string();
    };
    let kept = max_bytes - width(template);
    let head = floor(kept / 2);
    let tail = ceil(value.len() - (kept - kept / 2));
    let omitted = value[head..tail].chars().count();
    let marker = template.replace("{}", &omitted.to_string());
    format!("{}{marker}{}", &value[..head], &value[tail..])
}

fn json_len(text: &str) -> usize {
    crate::model_limits::serialized_len(text)
}

/// Serialized size of an image part: a base64 data URL plus JSON framing.
fn image_part_bytes(raw: usize) -> usize {
    raw.div_ceil(3) * 4 + 64
}

/// A model view plus whether this request moved a checkpoint (history or image).
#[derive(Debug)]
pub struct View {
    pub messages: Vec<crate::llm::ChatMessage>,
    pub checkpointed: bool,
}

struct Group {
    start: usize,
    len: usize,
    tokens: usize,
    bytes: usize,
    user: bool,
    oversized_batch: bool,
}

/// Per-message JSON framing estimate (role, keys, cache_control).
const MESSAGE_FRAMING_BYTES: usize = 96;

/// UTF-8 bytes of a message's text and tool-call arguments (what the
/// per-message cap is compared with).
fn raw_bytes(message: &ChatMessage) -> usize {
    message.content.len()
        + message.tool_calls.as_ref().map_or(0, |calls| {
            calls.iter().map(|c| c.arguments.len() + c.name.len() + c.id.len()).sum()
        })
}

fn message_bytes(message: &ChatMessage, max_message_bytes: usize) -> usize {
    let content = if message.content.len() > max_message_bytes {
        json_len(&truncate_middle(&message.content, max_message_bytes))
    } else {
        json_len(&message.content)
    };
    let calls: usize = message.tool_calls.as_ref().map_or(0, |calls| {
        calls.iter().map(|c| json_len(&c.arguments) + c.name.len() + c.id.len() + 64).sum()
    });
    content + calls + message.tool_call_id.as_ref().map_or(0, |id| id.len() + 20) + MESSAGE_FRAMING_BYTES
}

/// Raw sizes of the images the view will send, per message index: the same
/// anchor rule and the same selection as `select_images` (IMAGE_COUNT in total
/// across the view within the byte budget, not per message), computed without
/// moving the anchor. A later checkpoint can only drop some, so the projection
/// stays conservative.
fn kept_image_sizes(conv: &Conversation, image_bytes: usize) -> std::collections::HashMap<usize, Vec<usize>> {
    let mut probe = conv.clone_indices();
    let (selected, _) = select_from(&mut probe, &conv.messages, image_bytes);
    selected.into_iter().enumerate()
        .filter(|(_, images)| !images.is_empty())
        .map(|(index, images)| (index, images.iter().map(|a| a.len()).collect()))
        .collect()
}

fn groups(conv: &Conversation, policy: &ViewPolicy) -> Vec<Group> {
    let max_message_bytes = policy.max_message_bytes;
    let kept_images = kept_image_sizes(conv, policy.image_bytes);
    let base = conv.messages.as_ptr() as usize;
    let size = std::mem::size_of::<ChatMessage>();
    super::native_history::complete_groups(&conv.messages)
        .into_iter()
        .map(|group| {
            let start = (group.as_ptr() as usize - base) / size;
            let batch = group.len() > 1;
            let oversized_batch = batch && group.iter().any(|m| raw_bytes(m) > max_message_bytes);
            // Serialized text bytes as sent; tokens follow from the calibrated ratio.
            let text_bytes: usize = group.iter().map(|m| message_bytes(m, max_message_bytes)).sum();
            // Exactly the images the view will send (newest 3 total), as base64.
            // They are projected at a fixed token cost, outside the text ratio.
            let image_raw: &[usize] = if batch { &[] } else { kept_images.get(&start).map_or(&[], Vec::as_slice) };
            Group {
                start,
                len: group.len(),
                tokens: tokens_for_bytes(text_bytes, policy.bytes_per_token) + image_raw.len() * IMAGE_TOKENS,
                bytes: text_bytes + image_raw.iter().map(|&raw| image_part_bytes(raw)).sum::<usize>(),
                user: !batch && group[0].role == "user",
                oversized_batch,
            }
        })
        .filter(|group| !(group.start == 0 && conv.messages[0].role == "system"))
        .collect()
}

/// Advance `view_start` once if the projection exceeds the trigger. The newest
/// group (the current turn) is never cut; the cut prefers a user-turn boundary.
fn checkpoint(conv: &mut Conversation, groups: &[Group], policy: &ViewPolicy) -> bool {
    let Some(last) = groups.last() else { return false };
    let start = conv.view_start.min(last.start);
    let visible: Vec<&Group> = groups.iter().filter(|g| g.start >= start).collect();
    let tokens: usize = visible.iter().map(|g| g.tokens).sum();
    let bytes: usize = visible.iter().map(|g| g.bytes).sum();
    if !policy.over(tokens, bytes) {
        conv.view_start = start;
        return false;
    }
    conv.view_start = cut_start(&visible, |tokens, bytes| policy.within_target(tokens, bytes));
    true
}

/// Where the view starts after dropping the oldest visible groups until
/// `fits(tokens, bytes)`: then on to the next user turn, so the view never
/// starts mid-turn. A group is a whole native batch, so batches are never
/// split, and the newest group always stays.
fn cut_start(visible: &[&Group], fits: impl Fn(usize, usize) -> bool) -> usize {
    let mut tokens: usize = visible.iter().map(|g| g.tokens).sum();
    let mut bytes: usize = visible.iter().map(|g| g.bytes).sum();
    let mut cut = visible.len() - 1;
    for (i, group) in visible.iter().enumerate().take(visible.len() - 1) {
        if fits(tokens, bytes) {
            cut = i;
            break;
        }
        tokens -= group.tokens;
        bytes -= group.bytes;
    }
    let cut = (cut..visible.len() - 1).find(|&i| visible[i].user).unwrap_or(cut);
    visible[cut].start
}

fn cold_start(conv: &mut Conversation, policy: &ViewPolicy, max_history_tokens: usize) -> Option<(usize, usize)> {
    let all = groups(conv, policy);
    let start = conv.view_start.min(all.last()?.start);
    let visible: Vec<&Group> = all.iter().filter(|g| g.start >= start).collect();
    let before: usize = visible.iter().map(|g| g.tokens).sum();
    if before <= max_history_tokens {
        return None;
    }
    let cut = cut_start(&visible, |tokens, _| tokens <= max_history_tokens);
    if cut <= conv.view_start {
        return None;
    }
    conv.view_start = cut;
    let after = visible.iter().filter(|g| g.start >= cut).map(|g| g.tokens).sum();
    Some((before, after))
}

/// Images retained in the view, anchored at `image_start`. On overflow the anchor
/// jumps to the newest image-bearing message (a stability break at most once per
/// `IMAGE_COUNT` new images, not once per image).
fn select_images(conv: &mut Conversation, image_bytes: usize) -> (Vec<Vec<crate::attachment::Attachment>>, bool) {
    let mut anchors = conv.clone_indices();
    let result = select_from(&mut anchors, &conv.messages, image_bytes);
    conv.image_start = anchors.image_start;
    result
}

fn select_from(conv: &mut Anchors, messages: &[ChatMessage], image_bytes: usize) -> (Vec<Vec<crate::attachment::Attachment>>, bool) {
    let images = |message: &ChatMessage| -> Vec<crate::attachment::Attachment> {
        if message.role != "user" {
            return Vec::new();
        }
        message.attachments.iter()
            .filter(|a| a.image_mime().is_some() && a.len() > 0)
            .cloned()
            .collect()
    };
    let from = conv.image_start.max(conv.view_start);
    let (count, bytes) = messages[from.min(messages.len())..].iter().map(images).flatten()
        .fold((0, 0), |(n, b), a| (n + 1, b + a.len()));
    let mut moved = false;
    if count > IMAGE_COUNT || bytes > image_bytes {
        if let Some(newest) = messages.iter().rposition(|m| !images(m).is_empty()) {
            conv.image_start = newest;
            moved = true;
        }
    } else {
        conv.image_start = from;
    }
    let (mut left, mut budget) = (IMAGE_COUNT, image_bytes);
    let kept = messages.iter().enumerate().map(|(i, message)| {
        if i < conv.image_start {
            return Vec::new();
        }
        images(message).into_iter().filter(|a| {
            let keep = left > 0 && a.len() <= budget;
            if keep {
                left -= 1;
                budget -= a.len();
            }
            keep
        }).collect()
    }).collect();
    (kept, moved)
}

/// The persisted checkpoint indices, separable from the message list.
struct Anchors {
    view_start: usize,
    image_start: usize,
}

impl Conversation {
    fn clone_indices(&self) -> Anchors {
        Anchors { view_start: self.view_start, image_start: self.image_start }
    }

    /// Bytes per token to project this conversation's next request to `model`
    /// with: its last measured request if `model` answered it, else the
    /// conservative default (another model, another tokenizer).
    pub fn bytes_per_token(&self, model: &str) -> f32 {
        crate::model_limits::bytes_per_token(self.token_calibration.as_ref(), model)
    }

    /// The measurements the next request to `model` is projected with.
    pub fn projection(&self, model: &str) -> Projection {
        Projection {
            bytes_per_token: self.bytes_per_token(model),
            tail_bytes: self.tail_bytes.map(|bytes| bytes as usize),
        }
    }

    /// Cold start (#68): the provider cache is gone, so keep at most
    /// `max_history_tokens` of history. Same mechanics as a checkpoint. Returns
    /// the projected history tokens (before, after) when the anchor moved.
    /// Only for the start of a turn: never call it between a turn's rounds.
    pub fn cold_start(&mut self, policy: &ViewPolicy, max_history_tokens: usize) -> Option<(usize, usize)> {
        cold_start(self, policy, max_history_tokens)
    }
}

/// Model view of one stored message. Secondary `system` messages (runtime
/// notifications, agent results) become user-role runtime notices: providers hoist
/// every system message ahead of history, which would break the cached prefix.
fn to_model(message: &ChatMessage, primary: bool) -> crate::llm::ChatMessage {
    if message.role == "system" && !primary {
        return crate::llm::ChatMessage::user(format!("{RUNTIME_NOTICE_HEADER}\n{}", message.content));
    }
    super::native_history::to_llm(message)
}

/// Marks a stored runtime notification inside the provider-visible history.
pub const RUNTIME_NOTICE_HEADER: &str = "[Runtime notice: generated by the Adapsis runtime, not by a speaker.]";

/// Self-heal after a request was refused as too large (#68): only the newest turn
/// remains visible; history grows append-only from there. Returns whether the
/// anchor moved. Batch integrity holds because the cut lands on a group start.
pub(super) fn force_checkpoint(conv: &mut Conversation) -> bool {
    let groups = super::native_history::complete_groups(&conv.messages);
    let base = conv.messages.as_ptr() as usize;
    let size = std::mem::size_of::<ChatMessage>();
    let newest_user = groups.iter().rev()
        .find(|g| g.len() == 1 && g[0].role == "user")
        .map(|g| (g.as_ptr() as usize - base) / size);
    match newest_user {
        Some(start) if start > conv.view_start => {
            conv.view_start = start;
            true
        }
        _ => false,
    }
}

pub(super) fn build(conv: &mut Conversation, policy: &ViewPolicy) -> View {
    if conv.messages.is_empty() {
        return View { messages: Vec::new(), checkpointed: false };
    }
    let all = groups(conv, policy);
    let moved_history = checkpoint(conv, &all, policy);
    let (images, moved_images) = select_images(conv, policy.image_bytes);
    let system = conv.messages.first().filter(|m| m.role == "system").cloned();
    let omitted = conv.view_start.saturating_sub(usize::from(system.is_some()));
    let mut result = Vec::new();
    if let Some(system) = system {
        result.push(system);
    }
    if omitted > 0 {
        result.push(ChatMessage { role: "user".into(), content: format!("[System: {omitted} {OMITTED_NOTICE}]"), ..Default::default() });
    }
    for group in all.iter().filter(|g| g.start >= conv.view_start && !g.oversized_batch) {
        let slice = &conv.messages[group.start..group.start + group.len];
        if group.len > 1 {
            result.extend(slice.iter().cloned());
            continue;
        }
        let message = &slice[0];
        let content = truncate_middle(&message.content, policy.max_message_bytes);
        let attachments = images[group.start].clone();
        if content.is_empty() && attachments.is_empty() {
            continue;
        }
        result.push(ChatMessage { role: message.role.clone(), content, attachments, ..Default::default() });
    }
    let primary = usize::from(conv.messages.first().is_some_and(|m| m.role == "system"));
    View {
        messages: result.iter().enumerate().map(|(i, m)| to_model(m, i < primary)).collect(),
        checkpointed: moved_history || moved_images,
    }
}

#[cfg(test)]
#[path = "conversation_view_tests.rs"]
mod tests;
