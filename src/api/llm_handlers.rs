//! LLM conversation handlers for AdapsisOS.
//!
//! Contains the iterative LLM loops that share the pattern:
//! build messages → call LLM → execute_code → feed back results.
//!
//! Handlers:
//!   `ask`               — non-streaming LLM handler
//!   `ask_stream`        — SSE streaming LLM handler
//!   `handle_llm_takeover` — Telegram/agent LLM handler

use super::{
    AppConfig, AskRequest, AskResponse, MutationResult, TestCaseResult, WorkingSet,
};
use super::execute::{
    build_plan_context, format_library_load_errors, AgentCompletionCallback, OperationResult,
};
use super::execute_code;

use axum::extract::State;
use axum::Json;
use anyhow::Context as _;

const TAKEOVER_CONTEXT_MAX_MESSAGES: usize = 64;
const TAKEOVER_CONTEXT_MAX_CHARS: usize = 60_000;
const TAKEOVER_MESSAGE_MAX_CHARS: usize = 40_000;
const COMPACTION_CHUNK_MAX_CHARS: usize = 120_000;
const COMPACTION_RETRY_SECS: u64 = 15 * 60;

#[derive(serde::Deserialize)]
struct CompactionOutput {
    summary: String,
    #[serde(default)]
    memories: Vec<CompactionMemory>,
}

#[derive(serde::Deserialize)]
struct CompactionMemory {
    content: String,
    memory_type: String,
    confidence: f64,
}

fn takeover_context_lock(context: &str) -> std::sync::Arc<tokio::sync::Mutex<()>> {
    static LOCKS: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, std::sync::Arc<tokio::sync::Mutex<()>>>>,
    > = std::sync::OnceLock::new();
    let locks = LOCKS.get_or_init(Default::default);
    let mut locks = locks.lock().unwrap();
    locks.entry(context.to_string()).or_default().clone()
}

fn compaction_attempt_due(context: &str) -> bool {
    static ATTEMPTS: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, std::time::Instant>>,
    > = std::sync::OnceLock::new();
    let attempts = ATTEMPTS.get_or_init(Default::default);
    let mut attempts = attempts.lock().unwrap();
    let now = std::time::Instant::now();
    if attempts
        .get(context)
        .is_some_and(|last| now.duration_since(*last).as_secs() < COMPACTION_RETRY_SECS)
    {
        return false;
    }
    attempts.insert(context.to_string(), now);
    true
}

/// Strip the `[user:<id>]` / `[user:<id> <name>]` sender prefix that a group
/// intake module prepends for the model's benefit.
///
/// Presentation only. The stripped text is what reaches Ladybug as message
/// content and what is embedded as the recall query, so the prefix must not end
/// up in either — but this function is deliberately NOT an identity source.
pub(crate) fn strip_sender_prefix(message: &str) -> &str {
    message
        .strip_prefix("[user:")
        .and_then(|rest| rest.split_once("] "))
        .map_or(message, |(_, content)| content)
}

/// Resolve the Ladybug principal for a turn.
///
/// Identity comes from the structured `speaker_id` in `source_metadata`, never
/// from the message text. It used to be parsed back out of the `[user:<id>] `
/// prose prefix, which worked only for as long as that prefix held nothing but
/// the id. Adding the display name (#38 item 4) turned `[user:47128798 Kata] `
/// into the principal `telegram:user:47128798 Kata` — a second, silently created
/// identity for the same person, splitting her messages across two `Principal`
/// nodes and re-fragmenting on every Telegram display-name change. Prose is not
/// an identity channel.
pub(crate) fn takeover_principal(context: &str, source_metadata: Option<&str>) -> String {
    source_metadata
        .and_then(|metadata| serde_json::from_str::<serde_json::Value>(metadata).ok())
        .and_then(|metadata| {
            metadata
                .get("speaker_id")
                .and_then(|value| value.as_str())
                .map(str::to_string)
        })
        .filter(|speaker| !speaker.is_empty())
        .unwrap_or_else(|| context_principal(context))
}

/// Identity for a turn that carries no speaker metadata.
///
/// A group context has no single speaker, so it resolves to a group-kind
/// principal rather than a fabricated user one: stripping `telegram:` off
/// `telegram:group:-100…` used to yield the nonsense principal
/// `telegram:user:group:-100…`.
fn context_principal(context: &str) -> String {
    if context.starts_with("telegram:group:") {
        return context.to_string();
    }
    context
        .strip_prefix("telegram:user:")
        .or_else(|| context.strip_prefix("telegram:"))
        .map_or_else(|| "user:unknown".to_string(), |id| format!("telegram:user:{id}"))
}

async fn persist_takeover_message(
    memory_graph: &std::sync::Arc<crate::memory_graph::MemoryGraph>,
    context: &str,
    principal_id: String,
    role: &str,
    content: String,
    source_metadata: Option<&str>,
) -> anyhow::Result<String> {
    let metadata = source_metadata
        .and_then(|value| serde_json::from_str::<serde_json::Value>(value).ok())
        .unwrap_or_default();
    let platform_message_id = metadata
        .get("message_id")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let message_id = platform_message_id.as_ref().map_or_else(
        || format!("message:{}", uuid::Uuid::new_v4()),
        |id| format!("message:{context}:{id}"),
    );
    let observed_at_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64;
    let created_at_ms = metadata
        .get("created_at_ms")
        .and_then(|value| value.as_i64())
        .unwrap_or(observed_at_ms);
    let speaker_name = metadata
        .get("speaker_name")
        .and_then(|value| value.as_str())
        .filter(|value| !value.is_empty())
        .unwrap_or(&principal_id)
        .to_string();
    let context_kind = if context.starts_with("telegram:group:") {
        "telegram_group"
    } else if context.starts_with("telegram:") {
        "telegram_direct"
    } else {
        "internal"
    };
    let source = crate::memory_graph::SourceMessage {
        id: message_id.clone(),
        platform_message_id,
        context_id: context.to_string(),
        context_kind: context_kind.to_string(),
        speaker_id: principal_id.clone(),
        speaker_name,
        role: role.to_string(),
        content,
        created_at_ms,
    };
    let graph = memory_graph.clone();
    tokio::task::spawn_blocking(move || graph.ingest_message(&source, "telegram:user:1815217"))
        .await
        .context("join Ladybug message write")??;
    Ok(message_id)
}

fn compaction_trigger_chars() -> usize {
    let context_tokens = std::env::var("ADAPSIS_CONTEXT_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(128_000);
    let percent = std::env::var("ADAPSIS_COMPACTION_PERCENT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(70)
        .min(100);
    let reserve = std::env::var("ADAPSIS_OUTPUT_RESERVE_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(32_000);
    let threshold_tokens = (context_tokens * percent / 100)
        .min(context_tokens.saturating_sub(reserve));
    threshold_tokens.saturating_mul(4)
}

fn parse_compaction_output(text: &str) -> anyhow::Result<CompactionOutput> {
    let start = text.find('{').context("compaction response contained no JSON object")?;
    let end = text.rfind('}').context("compaction response contained incomplete JSON")?;
    serde_json::from_str(&text[start..=end]).context("parse compaction JSON")
}

async fn compact_context_if_needed(
    graph: &std::sync::Arc<crate::memory_graph::MemoryGraph>,
    embedder: &std::sync::Arc<crate::memory_graph::MemoryEmbedder>,
    llm: &crate::llm::LlmClient,
    llm_model: &str,
    context: &str,
) -> anyhow::Result<bool> {
    let graph_for_read = graph.clone();
    let context_for_read = context.to_string();
    let trigger = compaction_trigger_chars();
    let pending = tokio::task::spawn_blocking(move || {
        graph_for_read.pending_context_messages(&context_for_read, trigger)
    })
    .await
    .context("join Ladybug compaction scan")??;
    let pending_chars: usize = pending.iter().map(|message| message.content.chars().count()).sum();
    eprintln!(
        "[memory:{context}] compaction scan: {} pending messages, {pending_chars}/{trigger} estimated characters",
        pending.len()
    );
    if pending_chars < trigger || pending.len() < 4 {
        return Ok(false);
    }

    let mut chunk = Vec::new();
    let mut chunk_chars = 0usize;
    for message in pending {
        let chars = message.content.chars().count();
        if !chunk.is_empty() && chunk_chars.saturating_add(chars) > COMPACTION_CHUNK_MAX_CHARS {
            break;
        }
        chunk_chars = chunk_chars.saturating_add(chars);
        chunk.push(message);
    }
    if let Some(boundary) = chunk.iter().rposition(|message| {
        message.role == "assistant" && message.content.contains("!done")
    }) {
        chunk.truncate(boundary + 1);
    }
    if chunk.len() < 4 {
        return Ok(false);
    }

    let transcript = chunk
        .iter()
        .map(|message| {
            format!(
                "[{} | {} | {}]\n{}",
                message.id, message.speaker_name, message.role, message.content
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let prompt = format!(
        "Compact this immutable conversation segment. Return one JSON object and no prose:\n\
         {{\"summary\":\"self-contained episode summary preserving decisions, rationale, preferences, technical discoveries, commitments, relationships, and unfinished work\",\
         \"memories\":[{{\"content\":\"concise source-backed index memory\",\"memory_type\":\"decision|preference|finding|commitment|relationship|work_state\",\"confidence\":0.0}}]}}\n\n\
         Extract only durable knowledge. Memories are pointers and may rely on their linked source messages for detail. Do not invent facts.\n\n{transcript}"
    );
    let output = llm
        .generate(vec![
            crate::llm::ChatMessage::system(
                "You create immutable episodic-memory checkpoints. Preserve provenance and return strict JSON.",
            ),
            crate::llm::ChatMessage::user(prompt),
        ])
        .await?;
    let extraction = parse_compaction_output(&output.text)?;
    anyhow::ensure!(!extraction.summary.trim().is_empty(), "compaction summary was empty");

    let source_message_ids: Vec<String> = chunk.iter().map(|message| message.id.clone()).collect();
    let created_at_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64;
    let episode = crate::memory_graph::EpisodeDraft {
        context_id: context.to_string(),
        summary: extraction.summary,
        source_message_ids: source_message_ids.clone(),
        model: llm_model.to_string(),
        created_at_ms,
    };
    let graph_for_episode = graph.clone();
    tokio::task::spawn_blocking(move || graph_for_episode.create_episode(&episode))
        .await
        .context("join Ladybug episode write")??;

    let memories: Vec<_> = extraction
        .memories
        .into_iter()
        .filter(|memory| !memory.content.trim().is_empty())
        .take(16)
        .collect();
    if memories.is_empty() {
        return Ok(true);
    }
    let contents: Vec<String> = memories.iter().map(|memory| memory.content.clone()).collect();
    let embedder = embedder.clone();
    let embeddings = tokio::task::spawn_blocking(move || embedder.embed_passages(&contents))
        .await
        .context("join FastEmbed compaction batch")??;
    let run_id = format!("extraction:{}", uuid::Uuid::new_v4());
    for (memory, embedding) in memories.into_iter().zip(embeddings) {
        let draft = crate::memory_graph::MemoryDraft {
            content: memory.content,
            memory_type: memory.memory_type,
            confidence: memory.confidence.clamp(0.0, 1.0),
            embedding,
            origin_context_ids: vec![context.to_string()],
            source_message_ids: source_message_ids.clone(),
            extraction_run_id: run_id.clone(),
            extraction_model: llm_model.to_string(),
            created_at_ms,
        };
        let graph = graph.clone();
        tokio::task::spawn_blocking(move || graph.create_index_memory(&draft))
            .await
            .context("join Ladybug memory write")??;
    }
    eprintln!(
        "[memory:{context}] immutable episode checkpointed: {} source messages",
        source_message_ids.len()
    );
    Ok(true)
}

/// Write a llm_takeover working set's program back into the live program using
/// per-function copy-on-write merge (issue #9), falling back to a full
/// overwrite if no base snapshot was captured. Holds the program write lock
/// only for the brief merge.
async fn write_back_takeover_program(
    program: &std::sync::Arc<tokio::sync::RwLock<crate::ast::Program>>,
    session: &WorkingSet,
) {
    let mut live = program.write().await;
    match &session.base_program {
        Some(base) => {
            live.merge_changed_from(base, &session.program);
        }
        None => *live = session.program.clone(),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// EventSender — unified SSE/broadcast/log sender
// ═══════════════════════════════════════════════════════════════════════

/// Channel wrapper that sends events to the broadcast channel (and optionally an mpsc
/// response channel used by the SSE streaming endpoint).  Both `/api/ask` and
/// `/api/ask-stream` use this so events always appear on `/api/events`.
///
/// The `log` method is the preferred single entry-point for emitting events: it
/// sends to the broadcast channel, the per-request mpsc (if present), writes to
/// the structured log file, and prints a short preview to stderr — all in one
/// call, replacing the previous pattern of separate `tx.send()` + `log_activity()`
/// + `eprintln!()` calls.
pub(super) struct EventSender {
    tx: Option<tokio::sync::mpsc::Sender<serde_json::Value>>,
    broadcast: tokio::sync::broadcast::Sender<String>,
    log_file: Option<std::sync::Arc<tokio::sync::Mutex<tokio::fs::File>>>,
}

impl EventSender {
    /// Broadcast-only sender (used by plain `/api/ask`).
    pub(super) fn broadcast_only(broadcast: tokio::sync::broadcast::Sender<String>) -> Self {
        Self { tx: None, broadcast, log_file: None }
    }

    /// Broadcast + per-request mpsc sender (used by `/api/ask-stream`).
    pub(super) fn with_mpsc(
        tx: tokio::sync::mpsc::Sender<serde_json::Value>,
        broadcast: tokio::sync::broadcast::Sender<String>,
        log_file: Option<std::sync::Arc<tokio::sync::Mutex<tokio::fs::File>>>,
    ) -> Self {
        Self { tx: Some(tx), broadcast, log_file }
    }

    /// Send a raw event value to broadcast + mpsc.
    pub(super) async fn send(&self, event: serde_json::Value) {
        let encoded = super::encode_broadcast_event(&event);
        let _ = self.broadcast.send(encoded);
        if let Some(tx) = &self.tx {
            let _ = tx.send(event).await;
        } else {
            // Yield so broadcast receivers get a chance to process the event.
            tokio::task::yield_now().await;
        }
    }

    /// Unified logging: broadcast event + write to log file + stderr preview.
    ///
    /// This replaces the separate `tx.send()` / `log_activity()` / `eprintln!()`
    /// pattern.  `event` is a short tag (e.g. "iter", "code", "feedback") and
    /// `detail` is the full text.  The method also constructs an appropriate
    /// broadcast JSON payload.
    pub(super) async fn log(&self, event: &str, detail: &str) {
        // 1. Write to structured log file
        write_log_file(&self.log_file, event, detail).await;

        // 2. Stderr: short preview
        let preview: String = detail.chars().take(200).collect();
        eprintln!("[{event}] {preview}");

        // 3. Broadcast (+ mpsc if present) — build a JSON event
        let json = match event {
            "iter" => serde_json::json!({"type": "iteration", "detail": detail}),
            "code" => serde_json::json!({"type": "code", "code": detail}),
            "think" => serde_json::json!({"type": "thinking", "text": detail}),
            "feedback" => serde_json::json!({"type": "feedback", "message": detail}),
            "ai-text" => serde_json::json!({"type": "text", "text": detail}),
            "user" => serde_json::json!({"type": "user", "text": detail}),
            "done" => serde_json::json!({"type": "done", "detail": detail}),
            "done-rejected" => serde_json::json!({"type": "result", "message": detail, "success": false}),
            "llm-error" => serde_json::json!({"type": "error", "message": detail}),
            _ => serde_json::json!({"type": event, "detail": detail}),
        };
        self.send(json).await;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Log helpers
// ═══════════════════════════════════════════════════════════════════════

/// Write a structured entry to the log file (if configured).
/// Shared by both `log_activity` and `EventSender::log`.
pub(super) async fn write_log_file(
    log_file: &Option<std::sync::Arc<tokio::sync::Mutex<tokio::fs::File>>>,
    event: &str,
    detail: &str,
) {
    if let Some(f) = log_file {
        use tokio::io::AsyncWriteExt;
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let h = (secs / 3600) % 24;
        let m = (secs / 60) % 60;
        let s = secs % 60;

        let line = match event {
            "iter" => format!("\n============================================================\n[{h:02}:{m:02}:{s:02}] {detail}\n============================================================\n"),
            "code" => format!("[{h:02}:{m:02}:{s:02}] CODE:\n{detail}\n"),
            "think" => format!("[{h:02}:{m:02}:{s:02}] THINK:\n{detail}\n"),
            "feedback" => {
                let has_err = detail.contains("ERROR") || detail.contains("FAIL") || detail.contains("Fix the errors");
                let prefix = if has_err { "FEEDBACK (ERRORS)" } else { "FEEDBACK (ok)" };
                format!("[{h:02}:{m:02}:{s:02}] {prefix}:\n{detail}\n")
            }
            "ai-text" => format!("[{h:02}:{m:02}:{s:02}] AI: {detail}\n"),
            "done" | "done-rejected" => format!("[{h:02}:{m:02}:{s:02}] >>> {event}: {detail}\n"),
            "llm-error" => format!("[{h:02}:{m:02}:{s:02}] !!! LLM ERROR: {detail}\n"),
            "user" => format!("[{h:02}:{m:02}:{s:02}] USER:\n{detail}\n"),
            _ => format!("[{h:02}:{m:02}:{s:02}] [{event}] {detail}\n"),
        };
        let mut f = f.lock().await;
        let _ = f.write_all(line.as_bytes()).await;
        let _ = f.flush().await;
    }
}

/// Write a training data entry (JSONL) for one iteration.
async fn log_training_data(
    training_log: &Option<std::sync::Arc<tokio::sync::Mutex<tokio::fs::File>>>,
    model: &str,
    context: &str,
    thinking: &str,
    code: &str,
    feedback: &[String],
    has_errors: bool,
    tests_passed: usize,
    tests_failed: usize,
) {
    let Some(f) = training_log else { return };
    use tokio::io::AsyncWriteExt;
    let entry = serde_json::json!({
        "model": model,
        "context": context,
        "thinking": thinking,
        "code": code,
        "outcome": if has_errors { "error" } else { "success" },
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
        "feedback": feedback,
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default().as_secs(),
    });
    let mut line = serde_json::to_string(&entry).unwrap_or_default();
    line.push('\n');
    let mut f = f.lock().await;
    let _ = f.write_all(line.as_bytes()).await;
    let _ = f.flush().await;
}

// ═══════════════════════════════════════════════════════════════════════
// ask — non-streaming LLM handler
// ═══════════════════════════════════════════════════════════════════════

pub async fn ask(
    State(config): State<AppConfig>,
    Json(req): Json<AskRequest>,
) -> Json<AskResponse> {
    config.install_handler_locals();
    eprintln!("\n[web:user] {}", req.message);
    let tx = EventSender::broadcast_only(config.event_broadcast.clone());
    tx.send(serde_json::json!({"type": "start", "message": req.message})).await;
    let llm = crate::llm::LlmClient::new_with_model_and_key(
        &config.llm_url, &config.llm_model.read().unwrap(), config.llm_api_key.clone(),
    );

    let max_iterations = config.max_iterations;
    let mut all_results: Vec<MutationResult> = vec![];
    let mut all_test_results: Vec<TestCaseResult> = vec![];
    let mut all_code = String::new();
    let mut reply_text = String::new();

    let system_prompt = {
        let base = crate::prompt::system_prompt();
        let builtins = crate::builtins::format_for_prompt();
        let identity = crate::prompt::adapsis_identity();
        let mesh = crate::prompt::mesh_topology()
            .map(|m| format!("\n\n{m}"))
            .unwrap_or_default();
        format!("{base}\n\n{builtins}\n\n{identity}{mesh}")
    };

    // Build messages from conversation history
    let mut session = config.snapshot_working_set().await;
    let mut messages = {
        // Ensure system prompt exists (brief mutable borrow)
        {
            let conv = session.meta.conversations.get_or_create("main");
            conv.set_primary_system(system_prompt);
        }
        // Build context string (immutable borrows)
        let (plan_ctx, needs_plan) = build_plan_context(&session.meta.plan);
        let plan_hint = if needs_plan {
            "\n\nYour previous plan is completed (or none exists). Create a new plan with !plan set for this task before writing code. You can update it anytime with !plan set / !plan done N."
        } else { "" };
        let load_errors_ctx = format_library_load_errors(&session.meta);
        // If a peer node is calling (caller set), frame the message as coming
        // from that peer over the VPN — not from the local admin user — so the
        // model addresses its reply to the peer and knows it's a node-to-node
        // exchange.
        let speaker_line = match req.caller.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            Some(caller) => format!(
                "Peer AdapsisOS node `{caller}` is contacting you over the VPN via /api/ask. \
                 Reply addressed to that peer (not to a local user): {}",
                req.message
            ),
            None => format!("User: {}", req.message),
        };
        let context = format!(
            "Working directory: {}\n{}{}{}\n{}{}",
            config.project_dir,
            crate::validator::program_summary_for_model(
                &session.program, &config.permission_config, config.access_level,
                &config.llm_model.read().unwrap(),
            ),
            load_errors_ctx,
            plan_ctx,
            speaker_line,
            plan_hint
        );
        // Push user message and get LLM messages (brief mutable borrow)
        let conv = session.meta.conversations.get_or_create("main");
        conv.push_user(&context);
        conv.to_llm_messages()
    };

    for iteration in 0..max_iterations {
        eprintln!("[web:iter {}/{}]", iteration + 1, max_iterations);

        // Call LLM
        let output = match llm.generate(messages.clone()).await {
            Ok(o) => o,
            Err(e) => {
                eprintln!("[web:error] LLM: {e}");
                tx.send(serde_json::json!({"type": "error", "message": format!("LLM error: {e}")})).await;
                reply_text.push_str(&format!("\n\nLLM error: {e}"));
                break;
            }
        };

        messages.push(crate::llm::ChatMessage::assistant(&output.text));

        let code = output.code.clone();

        // Build reply from thinking + prose
        let mut clean = output.text.clone();
        while let Some(s) = clean.find("<think>") {
            if let Some(e) = clean[s..].find("</think>") { clean.replace_range(s..s+e+8, ""); } else { break; }
        }
        while let Some(s) = clean.find("<code>") {
            if let Some(e) = clean[s..].find("</code>") { clean.replace_range(s..s+e+7, ""); } else { break; }
        }
        let clean = clean.trim();
        if !clean.is_empty() {
            tx.send(serde_json::json!({"type": "text", "text": clean})).await;
            if !reply_text.is_empty() { reply_text.push_str("\n\n"); }
            reply_text.push_str(clean);
        }
        if !output.thinking.is_empty() {
            eprintln!("[web:think] {}...", output.thinking.chars().take(100).collect::<String>());
            tx.send(serde_json::json!({"type": "thinking", "text": output.thinking})).await;
        }

        // Check for !done or no code (AI is asking a question / responding with text)
        if code.trim() == "!done" {
            eprintln!("[web:done] model said !done at iteration {}", iteration + 1);
            tx.send(serde_json::json!({"type": "done"})).await;
            break;
        }
        if code.is_empty() {
            // No code block = AI is responding with text only (question or explanation)
            eprintln!("[web:text-only] no code block, stopping");
            tx.send(serde_json::json!({"type": "done"})).await;
            break;
        }

        eprintln!("[web:code]\n{}", code.chars().take(200).collect::<String>());
        tx.send(serde_json::json!({"type": "code", "code": code})).await;
        if !all_code.is_empty() { all_code.push_str("\n\n// --- iteration ---\n"); }
        all_code.push_str(&code);

        // Execute code via shared execute_code() function
        let mut session = config.snapshot_working_set().await;
        let exec_result = execute_code(&code, &config, &mut session, None).await;

        // Send SSE events for results
        for r in &exec_result.mutation_results {
            tx.send(serde_json::json!({"type": "result", "message": r.message, "success": r.success})).await;
        }
        for t in &exec_result.test_results {
            tx.send(serde_json::json!({"type": "test", "pass": t.pass, "message": t.message})).await;
        }

        let iter_has_errors = exec_result.has_errors;

        // Handle opencode restart
        if exec_result.needs_opencode_restart {
            tokio::spawn(async {
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                let exe = std::env::args().next()
                    .map(std::path::PathBuf::from)
                    .and_then(|p| std::fs::canonicalize(&p).ok().or(Some(p)))
                    .unwrap_or_else(|| std::env::current_exe().unwrap_or_default());
                let args: Vec<String> = std::env::args().collect();
                let _ = exec::execvp(&exe, &args);
            });
        }

        all_results.extend(exec_result.mutation_results.clone());
        all_test_results.extend(exec_result.test_results.clone());
        // Write mutations back to tiers after each iteration
        config.write_back_working_set(&session).await;

        // Build feedback for next iteration
        if iter_has_errors {
            // Show ALL results (successes too), not just errors — otherwise
            // the model can't tell which parts already applied and may
            // re-submit code that succeeded.
            let details: Vec<String> = exec_result.mutation_results.iter()
                .map(|r| format!("{}: {}", if r.success { "OK" } else { "ERROR" }, r.message))
                .chain(exec_result.test_results.iter()
                    .map(|r| format!("{}: {}", if r.pass { "PASS" } else { "FAIL" }, r.message)))
                .collect();
            let feedback = format!(
                "Results:\n{}\n\nFix the errors and continue. Successful operations are already applied — do not re-submit them.",
                details.join("\n")
            );
            eprintln!("[web:feedback] → retrying");
            messages.push(crate::llm::ChatMessage::user(feedback));
        } else {
            // Success — tell the AI to continue or finish
            let results_summary: Vec<String> = exec_result.mutation_results.iter().map(|r| r.message.clone()).collect();
            let feedback = format!(
                "Results:\n{}\n\nIf the task is complete, respond with !done. Otherwise continue with the next step.",
                results_summary.join("\n")
            );
            messages.push(crate::llm::ChatMessage::user(feedback));
        }
    }

    // Save conversation — write assistant reply directly into meta tier
    {
        let summary = format!("{}\n{}", reply_text.chars().take(200).collect::<String>(),
            all_results.iter().map(|r| format!("{}: {}", if r.success {"OK"} else {"ERR"}, r.message)).collect::<Vec<_>>().join("\n"));
        let mut meta = config.meta.lock().unwrap();
        let conv = meta.conversations.get_or_create("main");
        conv.push_assistant(summary);
        conv.trim(50);
    }

    let has_errors = all_results.iter().any(|r| !r.success) || all_test_results.iter().any(|r| !r.pass);
    if has_errors {
        tx.send(serde_json::json!({"type": "error", "message": "request completed with errors"})).await;
    }
    tx.send(serde_json::json!({"type": "done"})).await;
    Json(AskResponse {
        reply: reply_text,
        code: all_code,
        results: all_results,
        test_results: all_test_results,
        has_errors,
    })
}

// ═══════════════════════════════════════════════════════════════════════
// ask_stream — SSE streaming LLM handler
// ═══════════════════════════════════════════════════════════════════════

/// SSE streaming version of /api/ask — streams events as they happen.
pub async fn ask_stream(
    State(config): State<AppConfig>,
    Json(req): Json<AskRequest>,
) -> axum::response::sse::Sse<impl futures::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>> {
    use axum::response::sse::KeepAlive;
    use tokio::sync::mpsc;

    let (raw_tx, mut rx) = mpsc::channel::<serde_json::Value>(100);

    // Spawn the processing loop
    let config_clone = config.clone();
    tokio::spawn(async move {
        config_clone.install_handler_locals();
        let tx = EventSender::with_mpsc(raw_tx, config_clone.event_broadcast.clone(), config_clone.log_file.clone());
        let llm = crate::llm::LlmClient::new_with_model_and_key(
            &config_clone.llm_url, &config_clone.llm_model.read().unwrap(), config_clone.llm_api_key.clone(),
        );

        let _ = tx.send(serde_json::json!({"type": "start", "message": req.message})).await;

        let system_prompt = {
            let base = crate::prompt::system_prompt();
            let builtins = crate::builtins::format_for_prompt();
            let identity = crate::prompt::adapsis_identity();
            let mesh = crate::prompt::mesh_topology()
                .map(|m| format!("\n\n{m}"))
                .unwrap_or_default();
            format!("{base}\n\n{builtins}\n\n{identity}{mesh}")
        };

            let mut messages = {
            // Tier 1: read program briefly for summary
            let program_summary = {
                let program = config_clone.program.read().await;
                crate::validator::program_summary_for_model(
                    &program, &config_clone.permission_config, config_clone.access_level,
                    &config_clone.llm_model.read().unwrap(),
                )
            };
            // Tier 3: read/write meta briefly for chat history + plan context
            // Note: guard must be dropped before any .await — std::sync::MutexGuard is not Send.
            let (context, msgs) = {
                let mut meta = config_clone.meta.lock().unwrap();
                let conv = meta.conversations.get_or_create("main");
                conv.set_primary_system(system_prompt);
                let (plan_ctx, needs_plan) = build_plan_context(&meta.plan);
                let plan_hint = if needs_plan {
                    "\n\nYour previous plan is completed (or none exists). Create a new plan with !plan set for this task before writing code. You can update it anytime with !plan set / !plan done N."
                } else { "" };
                let load_errors_ctx = format_library_load_errors(&meta);
                let speaker_line = match req.caller.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
                    Some(caller) => format!(
                        "Peer AdapsisOS node `{caller}` is contacting you over the VPN via /api/ask-stream. \
                         Reply addressed to that peer (not to a local user): {}",
                        req.message
                    ),
                    None => format!("User: {}", req.message),
                };
                let context = format!("Working directory: {}\n{}{}{}\n{}{}",
                    config_clone.project_dir,
                    program_summary,
                    load_errors_ctx,
                    plan_ctx, speaker_line, plan_hint);
                let conv = meta.conversations.get_or_create("main");
                conv.push_user(&context);
                let msgs = conv.to_llm_messages();
                // guard dropped here — before any .await
                (context, msgs)
            };
            tx.log("user", &context).await;
            msgs
        }; // All locks released before LLM call

        let max_iterations = config_clone.max_iterations;
        let mut last_context = req.message.clone();
        // Loop-breaker: track consecutive identical error feedback so we can
        // tell the model it's stuck instead of burning the iteration budget.
        let mut last_error_signature = String::new();
        let mut repeat_count = 0usize;
        for iteration in 0..max_iterations {
            // Check for injected messages and append to conversation
            {
                let mut queue = config_clone.message_queue.lock().await;
                for injected in queue.drain(..) {
                    tx.log("inject", &injected).await;
                    messages.push(crate::llm::ChatMessage::user(injected));
                }
            }

            tx.log("iter", &format!("iteration {}/{}", iteration + 1, max_iterations)).await;

            // === Streaming LLM call ===
            // generate_streaming() retries the HTTP connection internally.
            // Once connected, chunks flow through the channel — no "waiting"
            // timer needed because the chunks themselves are progress.
            let output = {
                let mut rx = match llm.generate_streaming(messages.clone()).await {
                    Ok(rx) => rx,
                    Err(e) => {
                        tx.log("llm-error", &format!("{e}")).await;
                        break;
                    }
                };

                // Forward incremental chunks to SSE
                let mut final_output = None;
                while let Some(chunk) = rx.recv().await {
                    match chunk {
                        crate::llm::StreamChunk::Thinking(text) => {
                            let _ = tx.send(serde_json::json!({"type": "thinking", "text": text})).await;
                        }
                        crate::llm::StreamChunk::Content(text) => {
                            let _ = tx.send(serde_json::json!({"type": "content", "text": text})).await;
                        }
                        crate::llm::StreamChunk::Done(output) => {
                            final_output = Some(output);
                        }
                    }
                }

                match final_output {
                    Some(o) => o,
                    None => {
                        // Channel closed without Done — stream error
                        tx.log("llm-error", "LLM stream ended without completing").await;
                        break;
                    }
                }
            };

            messages.push(crate::llm::ChatMessage::assistant(&output.text));

            // Log thinking (full text for the log file, already streamed incrementally above)
            if !output.thinking.is_empty() {
                write_log_file(&config_clone.log_file, "think", &output.thinking).await;
            }

            // Extract prose and send as a single text event
            let mut clean = output.text.clone();
            while let Some(s) = clean.find("<think>") { if let Some(e) = clean[s..].find("</think>") { clean.replace_range(s..s+e+8, ""); } else { break; } }
            while let Some(s) = clean.find("<code>") { if let Some(e) = clean[s..].find("</code>") { clean.replace_range(s..s+e+7, ""); } else { break; } }
            let clean = clean.trim();
            if !clean.is_empty() {
                write_log_file(&config_clone.log_file, "ai-text", clean).await;
                let _ = tx.send(serde_json::json!({"type": "text", "text": clean})).await;
            }

            let code = output.code.trim().to_string();

            // Empty code = AI responded with prose only, no operations
            if code.is_empty() {
                // Push a user message so the conversation doesn't end on assistant
                // (some models like MiMo require the last message to be from the user)
                messages.push(crate::llm::ChatMessage::user(
                    "Your response contained no Adapsis operations. Write code with +, !, or ? prefixes.".to_string()
                ));
                continue;
            }

            tx.log("code", &code).await;

            // Apply code — delegate to shared execute_code() for all ops
            // except !opencode (which needs the streaming subprocess handler).
            let mut session = config_clone.snapshot_working_set().await;
            let mut op_result = OperationResult::new();

            // Run everything through execute_code (mutations, tests, evals,
            // queries, watches, agents, plan, done, mock/stub, messages, etc.)
            let exec_result = execute_code(&code, &config_clone, &mut session, None).await;

            // Emit SSE events from execution results
            for r in &exec_result.mutation_results {
                if r.success {
                    op_result.ok(&r.message);
                } else {
                    op_result.error(&r.message);
                }
                let _ = tx.send(serde_json::json!({"type": "result", "message": r.message, "success": r.success})).await;
            }
            for t in &exec_result.test_results {
                if t.pass {
                    op_result.pass(&t.message);
                } else {
                    op_result.fail(&t.message);
                }
                let _ = tx.send(serde_json::json!({"type": "test", "pass": t.pass, "message": t.message})).await;
            }

            // Handle deferred restart and done acceptance from execute_code
            let needs_opencode_restart = exec_result.needs_opencode_restart;
            if exec_result.done_accepted {
                op_result.accepted_done = true;
            }

            // Write session mutations back to tiers after each iteration
            config_clone.write_back_working_set(&session).await;

            // Post-execution feedback: check inbox and list untested functions.
            // (Queries are already handled by execute_code and included in mutation_results.)
            {
                // Check for messages from agents addressed to main
                let inbox = crate::session::drain_messages(&mut session.meta, "main");
                if !inbox.is_empty() {
                    let inbox_text = inbox.iter()
                        .map(|m| format!("[from {}] {}", m.from, m.content))
                        .collect::<Vec<_>>().join("\n");
                    op_result.info(format!("Agent messages:\n{inbox_text}"));
                    let _ = tx.send(serde_json::json!({"type": "result", "message": format!("Messages from agents: {}", inbox.len()), "success": true})).await;
                }

                // Note untested functions in feedback so the AI knows what needs tests
                if session.program.require_modules {
                    let all_fns: Vec<String> = {
                        let mut fns = Vec::new();
                        for m in &session.program.modules {
                            for f in &m.functions {
                                let qname = format!("{}.{}", m.name, f.name);
                                if f.body.len() > 2 && !crate::session::is_function_tested(&session.program, &qname) {
                                    fns.push(qname);
                                }
                            }
                        }
                        fns
                    };
                    if !all_fns.is_empty() {
                        op_result.info(format!(
                            "Untested functions (blocked from !eval): {}. Example:\n+test {}\n  +with <param>=<value> -> expect <result>",
                            all_fns.join(", "),
                            all_fns[0]
                        ));
                    }
                }
            }

            // Build plan status summary for feedback
            let plan_summary = {
                let meta = config_clone.meta.lock().unwrap();
                let in_progress: Vec<_> = meta.plan.iter().filter(|s| matches!(s.status, crate::session::PlanStatus::InProgress)).collect();
                let pending: Vec<_> = meta.plan.iter().filter(|s| matches!(s.status, crate::session::PlanStatus::Pending)).collect();
                let failed: Vec<_> = meta.plan.iter().filter(|s| matches!(s.status, crate::session::PlanStatus::Failed)).collect();
                let total = meta.plan.len();
                let done = meta.plan.iter().filter(|s| matches!(s.status, crate::session::PlanStatus::Done)).count();
                if total == 0 {
                    "No plan set. Create one with !plan set.".to_string()
                } else if pending.is_empty() && in_progress.is_empty() && failed.is_empty() {
                    format!("All {total} plan steps completed. Verify everything works, then !done.")
                } else {
                    let mut msg = format!("Plan: {done}/{total} done.");
                    if !failed.is_empty() {
                        msg.push_str(&format!("\n  {} failed.", failed.len()));
                    }
                    if !in_progress.is_empty() {
                        for s in &in_progress {
                            msg.push_str(&format!("\n  Current: {}", s.description));
                        }
                        for s in pending.iter().take(1) {
                            msg.push_str(&format!("\n  Next: {}", s.description));
                        }
                    } else {
                        if let Some(first) = pending.first() {
                            msg.push_str(&format!("\n  Current: {}", first.description));
                        }
                        if let Some(next) = pending.get(1) {
                            msg.push_str(&format!("\n  Next: {}", next.description));
                        }
                    }
                    msg
                }
            };

            let feedback_details = &op_result.feedback;
            let has_errors = op_result.has_errors;
            let accepted_done = op_result.accepted_done;

            if has_errors {
                let errors: Vec<&str> = feedback_details.iter()
                    .filter(|d| d.starts_with("ERROR:") || d.starts_with("FAIL:") || d.contains("[FAILED]"))
                    .map(|s| s.as_str()).collect();
                // Detect repeated identical failures (same error set as last
                // iteration) and escalate the hint instead of repeating it.
                let error_signature = errors.join("\n");
                if !error_signature.is_empty() && error_signature == last_error_signature {
                    repeat_count += 1;
                } else {
                    repeat_count = 0;
                    last_error_signature = error_signature;
                }
                let stuck_hint = match repeat_count {
                    0 => "",
                    1 => "\n\nNote: this is the same error as your previous attempt. Re-read the error message carefully before retrying.",
                    _ => "\n\nYou have hit the exact same error multiple times in a row. STOP repeating the same approach. Use ?source to inspect the current state, ?symbols to check available functions/types, and try a structurally different solution.",
                };
                let feedback = format!(
                    "Results:\n{}\n\n{}\n\nFix the errors and continue.{}",
                    feedback_details.join("\n"),
                    plan_summary,
                    stuck_hint
                );
                tx.log("feedback", &format!("Errors found ({} issues), retrying...\n{feedback}", errors.len())).await;
                let current_model = config_clone.llm_model.read().unwrap().clone();
                log_training_data(&config_clone.training_log, &current_model, &last_context, &output.thinking, &code, feedback_details, true, op_result.tests_passed, op_result.tests_failed).await;
                last_context = feedback.clone();
                messages.push(crate::llm::ChatMessage::user(feedback));
            } else {
                let results_section = if feedback_details.is_empty() {
                    String::new()
                } else {
                    format!("Results:\n{}\n\n", feedback_details.join("\n"))
                };
                // Success resets the repeated-failure tracker.
                last_error_signature.clear();
                repeat_count = 0;
                let feedback = format!("{}{}", results_section, plan_summary);
                tx.log("feedback", &feedback).await;
                let current_model = config_clone.llm_model.read().unwrap().clone();
                log_training_data(&config_clone.training_log, &current_model, &last_context, &output.thinking, &code, feedback_details, false, op_result.tests_passed, op_result.tests_failed).await;
                last_context = feedback.clone();
                messages.push(crate::llm::ChatMessage::user(feedback));
                if accepted_done { break; }
            }

            // Handle deferred opencode restart
            if needs_opencode_restart {
                // Save session before restart
                {
                    let snap = config_clone.snapshot_working_set().await;
                    if let Some(path) = std::env::args().nth(std::env::args().position(|a| a == "--session").unwrap_or(999) + 1) {
                        let snap = crate::session::Session { program: snap.program, runtime: snap.runtime, meta: snap.meta, sandbox: snap.sandbox };
                        let _ = snap.save(std::path::Path::new(&path));
                    }
                }
                let exe = std::env::current_exe().unwrap_or_else(|_| {
                    std::env::args().next()
                        .map(std::path::PathBuf::from)
                        .unwrap_or_default()
                });
                let args: Vec<String> = std::env::args().collect();
                eprintln!("[ask_stream:opencode] restarting with binary: {}", exe.display());
                let err = exec::execvp(&exe, &args);
                eprintln!("[ask_stream:opencode] restart failed: {err}");
            }
        }

        let _ = tx.send(serde_json::json!({"type": "end"})).await;
    });

    // Convert channel to SSE stream
    let stream = async_stream::stream! {
        while let Some(event) = rx.recv().await {
            let data = serde_json::to_string(&event).unwrap_or_default();
            yield Ok(axum::response::sse::Event::default().data(data));
        }
    };

    axum::response::sse::Sse::new(stream).keep_alive(KeepAlive::default())
}

// ═══════════════════════════════════════════════════════════════════════
// handle_llm_takeover — Telegram/agent LLM handler
// ═══════════════════════════════════════════════════════════════════════

const DEFAULT_TAKEOVER_ITERATION_BUDGET: usize = 10;
const MAX_TAKEOVER_ITERATION_BUDGET: usize = 50;

pub(super) fn takeover_iteration_budget(text: &str) -> Option<usize> {
    let start = text.find("<iteration_budget>")? + "<iteration_budget>".len();
    let end = text[start..].find("</iteration_budget>")? + start;
    text[start..end].trim().parse().ok()
}

fn visible_takeover_prose(text: &str) -> String {
    let mut clean = text.to_string();
    while let Some(start) = clean.find("<think>") {
        if let Some(end) = clean[start..].find("</think>") {
            clean.replace_range(start..start + end + 8, "");
        } else {
            break;
        }
    }
    clean = clean.replace("</think>", "");
    while let Some(start) = clean.find("<iteration_budget>") {
        let Some(end) = clean[start..].find("</iteration_budget>") else {
            break;
        };
        clean.replace_range(start..start + end + "</iteration_budget>".len(), "");
    }
    if let Some(start) = clean.find("<code>") {
        clean.truncate(start);
    }
    clean.trim().trim_end_matches("!done").trim().to_string()
}

/// Handle a `llm_takeover` IO request: call LLM with per-context conversation
/// history, return text reply immediately, execute any code in background.
pub async fn handle_llm_takeover(
    context: String,
    message: String,
    attachment: Option<crate::attachment::Attachment>,
    source_metadata: Option<String>,
    memory_graph: std::sync::Arc<crate::memory_graph::MemoryGraph>,
    memory_embedder: std::sync::Arc<crate::memory_graph::MemoryEmbedder>,
    reply_fn: Option<String>,
    reply_arg: Option<String>,
    meta: crate::session::SharedMeta,
    program: std::sync::Arc<tokio::sync::RwLock<crate::ast::Program>>,
    runtime: crate::session::SharedRuntime,
    llm_url: &str,
    llm_model: &str,
    llm_key: Option<String>,
    io_sender: tokio::sync::mpsc::Sender<crate::coroutine::IoRequest>,
    task_registry: crate::coroutine::TaskRegistry,
    snap_registry: crate::coroutine::TaskSnapshotRegistry,
    opencode_lock: std::sync::Arc<tokio::sync::Mutex<()>>,
    opencode_git_dir: String,
    training_log: Option<std::sync::Arc<tokio::sync::Mutex<tokio::fs::File>>>,
    access_level: crate::permissions::AccessLevel,
    permission_config: std::sync::Arc<crate::permissions::PermissionConfig>,
    // Persistence + observability (issue: conversations were only saved on
    // mutations, and Telegram traffic never reached --log-file).
    save_notify: Option<tokio::sync::mpsc::Sender<()>>,
    log_file: Option<std::sync::Arc<tokio::sync::Mutex<tokio::fs::File>>>,
) -> anyhow::Result<String> {
    let context_lock = takeover_context_lock(&context);
    let _context_guard = context_lock.lock().await;
    let speaker_id = takeover_principal(&context, source_metadata.as_deref());
    let source_content = strip_sender_prefix(&message).to_string();
    let graph_message_id = persist_takeover_message(
        &memory_graph,
        &context,
        speaker_id.clone(),
        "user",
        source_content.clone(),
        source_metadata.as_deref(),
    )
    .await?;
    if let Some(source_attachment) = attachment.clone() {
        let graph = memory_graph.clone();
        let message_id = graph_message_id.clone();
        let platform_ref = source_metadata
            .as_deref()
            .and_then(|metadata| serde_json::from_str::<serde_json::Value>(metadata).ok())
            .and_then(|metadata| metadata.get("attachment_ref").and_then(|value| value.as_str()).map(str::to_string));
        tokio::task::spawn_blocking(move || {
            graph.ingest_attachment(&message_id, &source_attachment, platform_ref.as_deref())
        })
        .await
        .context("join Ladybug attachment write")??;
    }
    let llm = crate::llm::LlmClient::new_with_model_and_key(llm_url, llm_model, llm_key.clone());
    let compaction = if compaction_attempt_due(&context) {
        tokio::time::timeout(
            std::time::Duration::from_secs(30),
            compact_context_if_needed(
                &memory_graph,
                &memory_embedder,
                &llm,
                llm_model,
                &context,
            ),
        )
        .await
        .map_err(|_| anyhow::anyhow!("checkpoint generation timed out after 30 seconds"))
        .and_then(|result| result)
    } else {
        Ok(false)
    };
    if let Err(error) = compaction {
        eprintln!("[memory:{context}] compaction deferred: {error:#}");
        write_log_file(
            &log_file,
            "memory-compaction-error",
            &format!("[{context}] {error:#}"),
        )
        .await;
    }
    let recall_principal = speaker_id.clone();
    let recall_query = source_content.clone();
    let recall_graph = memory_graph.clone();
    let recall_embedder = memory_embedder.clone();
    let recalled = tokio::task::spawn_blocking(move || {
        let embedding = recall_embedder.embed_query(&recall_query)?;
        recall_graph.recall_authorized_hybrid(&recall_principal, &recall_query, &embedding, 5)
    })
    .await
    .context("join Ladybug recall")??;
    // Get or create conversation, update callback info, build messages
    //
    // Use the actual permission config + access level so the program summary
    // shown to the LLM is filtered correctly.  If the conversation has a
    // `permission_model` override, use that model's permissions instead (this
    // is how non-admin Telegram users get a restricted view).
    // A conversation's permission_model override (e.g. "gemma4s" for non-admin
    // Telegram users) determines BOTH the filtered program view shown to the LLM
    // AND the model identity that code execution is gated against. Without using
    // it for execution, the sandbox would be advisory only: a non-admin context
    // could emit mutations or !opencode that execute at the admin model's
    // permission level. We hoist it here so tmp_config below can enforce it.
    let perm_model_override = {
        let meta_guard = meta.lock().unwrap();
        meta_guard.conversations.get(&context)
            .and_then(|c| c.permission_model.clone())
    };
    let effective_model = perm_model_override.clone().unwrap_or_else(|| llm_model.to_string());
    let program_summary = {
        let prog = program.read().await;
        crate::validator::program_summary_for_model(
            &prog, &permission_config,
            access_level, &effective_model,
        )
    };
    let mut messages = {
        let mut meta_guard = meta.lock().unwrap();
        let conv = meta_guard.conversations.get_or_create(&context);

        // Update callback info if provided
        if reply_fn.is_some() {
            conv.reply_fn = reply_fn.clone();
        }
        if reply_arg.is_some() {
            conv.reply_arg = reply_arg.clone();
        }

        // Refresh the runtime prompt after upgrades while preserving chat history.
        let system = conv.system_prompt.clone().unwrap_or_else(|| {
            if perm_model_override.is_some() {
                    // Sandboxed (non-admin) context, e.g. a family member: use the
                    // persona instead of the Adapsis-programmer framing. The
                    // persona owns the identity/tone/safety; we still tell it how
                    // to actually invoke functions and the <code> reply mechanics,
                    // and show only the functions it may call.
                    let mesh = crate::prompt::mesh_topology()
                        .map(|m| format!("\n\n{m}"))
                        .unwrap_or_default();
                    format!(
                        "{}{}\n\n\
                         ## So führst du Aktionen aus (technisch)\n\
                         Wenn du etwas am Computer tun musst, rufe eine vorhandene Funktion in einem \
                         <code>…</code>-Block auf, z. B. <code>!eval Wolfi.install_app(\"firefox\")</code>. \
                         Kündige keine spätere Aktion nur im Text an: führe sie im selben Durchlauf im \
                         <code>-Block aus. Beende eine reine Textantwort immer mit <code>!done</code>. \
                         Beginne deine erste Antwort auf jede Nutzernachricht mit einer realistischen \
                         Schätzung <iteration_budget>N</iteration_budget>: 1 für eine reine Antwort, \
                         mehr für Aufgaben mit mehreren Aktionsschritten. Maximum ist 50. \
                         Der Nutzer sieht NUR den Text VOR dem ersten <code>-Block — schreibe also zuerst \
                         deine Antwort an Renate, dann den Code. Erfinde keine Funktionen; nutze nur die unten gelisteten. \
                         Lies bei Unklarheit den Quelltext mit ?source Modul.funktion.\n\n\
                         ## Verfügbare Funktionen\n{}\n\n\
                         Konversationskontext: '{context}'.",
                        crate::prompt::persona(),
                        mesh,
                        program_summary,
                    )
            } else {
                    let available_models = permission_config.model_names();
                    let models_line = if available_models.is_empty() {
                        String::new()
                    } else {
                        format!(
                            "Available models (switchable via llm_set_model): {}\n",
                            available_models.join(", "),
                        )
                    };
                    let mesh = crate::prompt::mesh_topology()
                        .map(|m| format!("\n\n{m}"))
                        .unwrap_or_default();
                    format!(
                        "{}\n\n{}\n\n{}{}\n\nCurrent model: {llm_model}\n\
                         {models_line}\
                         Current program state:\n{}\n\n\
                         You are in conversation context '{context}'. Respond naturally. \
                         If you need to do work (modify code, create modules, run tasks), include the \
                         Adapsis commands in your response. Do NOT use !agent for tasks that require \
                         IO execution (generating music, sending files, HTTP calls). Agents cannot run \
                         [io,async] functions. Instead, do IO tasks inline using !eval. Use !agent ONLY \
                         for pure code writing tasks (adding modules, functions, tests). \
                         IMPORTANT: The user only sees text BEFORE the first <code> block. \
                         Text after <code> blocks is discarded. Put your response to the user first, \
                          then the code. Do not narrate what the code does after the <code> block. \
                          Never promise an action for a later turn without executing it now. Every \
                          completed prose-only response must end with <code>!done</code>. \
                          Start your first response to each user message with a realistic action-round \
                          estimate: <iteration_budget>N</iteration_budget>. Use 1 for a prose-only answer \
                          and more for multi-step work. The maximum is 50. \
                         When you need to understand existing code, use ?source Module.function. \
                         Don't guess how things work — read the source. \
                         DESIGN PRINCIPLES: \
                         1. REUSE over CREATE. Before writing new code, check if existing functions \
                            already do what you need. Use !eval with existing functions directly. \
                         2. If you must write a new function, make it GENERIC and REUSABLE — parameterize \
                            everything (chat_id, caption, duration, etc). Add it to the appropriate \
                            existing module, not a new one. \
                         3. Never create one-off modules for single requests. \
                         4. Write +doc for every new function. \
                         5. Music generation takes ~60 seconds — do not retry if it seems slow.",
                        crate::prompt::system_prompt(),
                        crate::builtins::format_for_prompt(),
                        crate::prompt::adapsis_identity(),
                        mesh,
                        program_summary,
                    )
            }
        });
        conv.set_primary_system(system);

        // Add user message
        if let Some(attachment) = attachment {
            conv.push_user_with_attachment(&message, attachment);
        } else {
            conv.push_user(&message);
        }

        // Build LLM messages from conversation history
        conv.to_llm_messages_bounded(
            TAKEOVER_CONTEXT_MAX_MESSAGES,
            TAKEOVER_CONTEXT_MAX_CHARS,
            TAKEOVER_MESSAGE_MAX_CHARS,
        )
    }; // meta_guard dropped here

    let episode_graph = memory_graph.clone();
    let episode_context = context.clone();
    let episode_summaries = tokio::task::spawn_blocking(move || {
        episode_graph.episode_summaries(&episode_context, 6)
    })
    .await
    .context("join Ladybug episode recall")??;

    if !episode_summaries.is_empty() {
        messages.insert(
            usize::from(!messages.is_empty()),
            crate::llm::ChatMessage::system(format!(
                "Immutable summaries of older episodes in this context:\n- {}",
                episode_summaries.join("\n- ")
            )),
        );
    }

    if !recalled.is_empty() {
        let recall = recalled
            .iter()
            .map(|memory| format!("- [{} | {}] {}", memory.id, memory.citation, memory.content))
            .collect::<Vec<_>>()
            .join("\n");
        messages.insert(
            usize::from(!messages.is_empty()),
            crate::llm::ChatMessage::system(format!(
                "Authorized long-term memory recall. Treat these as source-backed pointers and inspect provenance before relying on uncertain details:\n{recall}"
            )),
        );
    }

    eprintln!("[llm_takeover:{context}] calling LLM with {} messages", messages.len());
    write_log_file(&log_file, "user", &format!("[{context}] {message}")).await;
    // Persist the conversation now that the user message is appended —
    // otherwise chats are lost on restart and the bot "resumes" from a
    // stale snapshot (the "will shutdown now!" greeting bug).
    if let Some(ref tx) = save_notify {
        let _ = tx.try_send(());
    }

    // Build a temporary AppConfig so we can reuse execute_code()
    let (self_trigger_tx, _self_trigger_rx) = tokio::sync::mpsc::channel::<String>(1);
    let tmp_config = AppConfig {
        program: program.clone(),
        meta: meta.clone(),
        llm_url: llm_url.to_string(),
        // Gate execution (mutations, !opencode, spawned agents) against the
        // effective model: the conversation's permission_model override when set,
        // otherwise the active model. This makes the non-admin sandbox enforced,
        // not just a filtered view. The conversational reply itself was already
        // generated above with the real model.
        llm_model: std::sync::Arc::new(std::sync::RwLock::new(effective_model.clone())),
        llm_api_key: llm_key.clone(),
        project_dir: ".".to_string(),
        io_sender: Some(io_sender.clone()),
        self_trigger: self_trigger_tx,
        task_registry: Some(task_registry.clone()),
        snapshot_registry: Some(snap_registry.clone()),
        log_file: log_file.clone(),
        training_log: training_log.clone(),
        jit_cache: crate::eval::new_jit_cache(),
        event_broadcast: tokio::sync::broadcast::channel(16).0,
        opencode_git_dir: opencode_git_dir.clone(),
        opencode_lock: opencode_lock.clone(),
        opencode_attach: None,
        message_queue: std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new())),
        max_iterations: 10,
        runtime: runtime.clone(),
        sessions: std::sync::Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new())),
        save_notify: save_notify.clone(),
        access_level,
        permission_config: permission_config.clone(),
    };

    // Build agent completion callback from conversation's reply info
    let agent_cb = {
        let meta_guard = meta.lock().unwrap();
        meta_guard.conversations.get(&context).and_then(|conv| {
            match (&conv.reply_fn, &conv.reply_arg) {
                (Some(rf), Some(ra)) => Some(AgentCompletionCallback {
                    context: context.clone(),
                    reply_fn: rf.clone(),
                    reply_arg: ra.clone(),
                    llm_url: llm_url.to_string(),
                    llm_model: llm_model.to_string(),
                    llm_key: llm_key.clone(),
                }),
                _ => None,
            }
        })
    };

    // Iterative loop: the model estimates its action budget on the first response;
    // a hard ceiling remains to contain runaway or no-progress loops.
    let mut llm_messages = messages;
    let mut reply_text = String::new();
    let mut final_summary_pending = false;
    let mut iteration_budget = DEFAULT_TAKEOVER_ITERATION_BUDGET;

    for iteration in 0..MAX_TAKEOVER_ITERATION_BUDGET {
        if iteration >= iteration_budget {
            break;
        }
        eprintln!("[llm_takeover:{context}] iteration {}/{}", iteration + 1, iteration_budget);

        let output = match llm.generate(llm_messages.clone()).await {
            Ok(o) => o,
            Err(e) => {
                eprintln!("[llm_takeover:{context}] LLM error: {e}");
                write_log_file(&log_file, "llm-error", &format!("[{context}] {e}")).await;
                // Never leave the user hanging: if we have no reply yet,
                // return a short apology instead of silently dropping the
                // message (the callback path sends whatever we return).
                if reply_text.is_empty() {
                    let detail: String = e.to_string().chars().take(120).collect();
                    reply_text = format!(
                        "Entschuldigung — ich habe gerade ein technisches Problem \
                         und konnte deine Nachricht nicht verarbeiten. Bitte \
                         versuch es gleich noch einmal. ({detail})"
                    );
                }
                break;
            }
        };

        if output.text.trim().is_empty() && output.code.trim().is_empty() {
            let error = "model returned an empty response";
            eprintln!("[llm_takeover:{context}] {error}");
            write_log_file(&log_file, "llm-error", &format!("[{context}] {error}")).await;
            if reply_text.is_empty() {
                reply_text = "Entschuldigung — das Modell hat gerade leer geantwortet. Die Nachricht und der vollständige Verlauf sind gespeichert; bitte versuch es noch einmal.".to_string();
            }
            break;
        }

        persist_takeover_message(
            &memory_graph,
            &context,
            "agent:kronk".to_string(),
            "assistant",
            output.text.clone(),
            None,
        )
        .await?;

        if iteration == 0 {
            if let Some(estimate) = takeover_iteration_budget(&output.text) {
                iteration_budget = estimate.clamp(1, MAX_TAKEOVER_ITERATION_BUDGET);
                eprintln!("[llm_takeover:{context}] model estimated {iteration_budget} iteration(s)");
            } else {
                eprintln!("[llm_takeover:{context}] no iteration estimate; using default {iteration_budget}");
            }
        }

        // Store assistant response in conversation
        {
            let mut meta_guard = meta.lock().unwrap();
            if let Some(conv) = meta_guard.conversations.get_mut(&context) {
                conv.push_assistant(&output.text);
            }
        }

        llm_messages.push(crate::llm::ChatMessage::assistant(&output.text));

        // Extract text reply (prose without code/commands)
        let prose = visible_takeover_prose(&output.text);

        if !prose.is_empty() {
            reply_text = prose.clone();
        }

        eprintln!("[llm_takeover:{context}] reply: {}...", prose.chars().take(80).collect::<String>());
        write_log_file(&log_file, "ai-text", &format!("[{context}] {}", output.text)).await;

        let code = output.code.trim().to_string();

        if code == "!done" {
            eprintln!("[llm_takeover:{context}] done at iteration {}", iteration + 1);
            break;
        }
        if code.is_empty() {
            let correction = "Your response contained no Adapsis operation. If you promised or need to do work, execute the next step now in a <code> block. If no action remains, end the response with <code>!done</code>. Do not defer work to a future turn.";
            eprintln!("[llm_takeover:{context}] prose-only at iteration {}, requesting an operation or explicit completion", iteration + 1);
            {
                let mut meta_guard = meta.lock().unwrap();
                if let Some(conv) = meta_guard.conversations.get_mut(&context) {
                    conv.push_user(correction);
                }
            }
            llm_messages.push(crate::llm::ChatMessage::user(correction.to_string()));
            if iteration + 1 >= iteration_budget {
                iteration_budget = (iteration_budget + 1).min(MAX_TAKEOVER_ITERATION_BUDGET);
            }
            continue;
        }

        // Check for !agent — spawn it and return prose (agent delivers callback later)
        if let Ok(ops) = crate::parser::parse(&code) {
            let has_agent = ops.iter().any(|op| matches!(op, crate::parser::Operation::Agent { .. }));
            if has_agent {
                eprintln!("[llm_takeover:{context}] !agent detected, spawning via execute_code and breaking");
                let base_program = program.read().await.clone();
                let mut session = WorkingSet {
                    program: base_program.clone(),
                    base_program: Some(base_program),
                    runtime: runtime.read().unwrap().clone(),
                    meta: meta.lock().unwrap().clone(),
                    sandbox: None,
                };
                execute_code(&code, &tmp_config, &mut session, agent_cb.clone()).await;
                // Write back mutations (per-function CoW merge, issue #9)
                write_back_takeover_program(&program, &session).await;
                if let Ok(mut rt) = runtime.write() {
                    rt.shared_vars.replace_from(session.runtime.shared_vars.snapshot());
                    rt.http_routes = session.runtime.http_routes.clone();
                    rt.failure_history = session.runtime.failure_history.clone();
                }
                *meta.lock().unwrap() = session.meta;
                tmp_config.notify_save();
                break;
            }
        }

        // Execute code inline
        let base_program = program.read().await.clone();
        let mut session = WorkingSet {
            program: base_program.clone(),
            base_program: Some(base_program),
            runtime: runtime.read().unwrap().clone(),
            meta: meta.lock().unwrap().clone(),
            sandbox: None,
        };
        let exec_result = execute_code(&code, &tmp_config, &mut session, agent_cb.clone()).await;

        // Write back mutations to shared state (per-function CoW merge, issue #9)
        write_back_takeover_program(&program, &session).await;
        if let Ok(mut rt) = runtime.write() {
            rt.shared_vars.replace_from(session.runtime.shared_vars.snapshot());
            rt.http_routes = session.runtime.http_routes.clone();
            rt.failure_history = session.runtime.failure_history.clone();
        }
        *meta.lock().unwrap() = session.meta;
        tmp_config.notify_save();

        // Build feedback and append to conversation
        let feedback: Vec<String> = exec_result.mutation_results.iter()
            .map(|r| format!("{}: {}", if r.success { "OK" } else { "ERROR" }, r.message))
            .chain(exec_result.test_results.iter()
                .map(|t| format!("{}: {}", if t.pass { "PASS" } else { "FAIL" }, t.message)))
            .collect();

        eprintln!("[llm_takeover:{context}] code results: {}", feedback.join("; "));

        let is_last_iteration = iteration + 1 >= iteration_budget;
        let feedback_msg = if exec_result.has_errors {
            if is_last_iteration {
                format!("Errors:\n{}\n\nThis is the last iteration. Tell the user where you got stuck and ask if they want you to continue.", feedback.join("\n"))
            } else {
                format!("Errors:\n{}\n\nFix and continue.", feedback.join("\n"))
            }
        } else {
            if is_last_iteration {
                format!("Results:\n{}\n\nThis is the last iteration. Summarize what you accomplished and what remains, then ask if the user wants you to continue.", feedback.join("\n"))
            } else {
                format!("Results:\n{}\n\nContinue or !done.", feedback.join("\n"))
            }
        };

        {
            let mut meta_guard = meta.lock().unwrap();
            if let Some(conv) = meta_guard.conversations.get_mut(&context) {
                conv.push_user(feedback_msg.clone());
            }
        }
        llm_messages.push(crate::llm::ChatMessage::user(feedback_msg));
        final_summary_pending = is_last_iteration;

        // Handle opencode restart if needed
        if exec_result.needs_opencode_restart {
            eprintln!("[llm_takeover:{context}] opencode restart triggered");
            let exe = std::env::current_exe().unwrap_or_default();
            let args: Vec<String> = std::env::args().collect();
            let _ = exec::execvp(&exe, &args);
        }
    }

    if final_summary_pending {
        eprintln!("[llm_takeover:{context}] requesting final summary after action limit");
        match llm.generate(llm_messages).await {
            Ok(output) => {
                let prose = visible_takeover_prose(&output.text);
                if !prose.is_empty() {
                    reply_text = prose;
                }
                {
                    let mut meta_guard = meta.lock().unwrap();
                    if let Some(conv) = meta_guard.conversations.get_mut(&context) {
                        conv.push_assistant(&output.text);
                    }
                }
                write_log_file(&log_file, "ai-text", &format!("[{context}] {}", output.text)).await;
            }
            Err(error) => {
                eprintln!("[llm_takeover:{context}] final summary LLM error: {error}");
            }
        }
    }

    // Persist assistant/feedback messages appended during the loop.
    if let Some(ref tx) = save_notify {
        let _ = tx.try_send(());
    }

    Ok(reply_text)
}
