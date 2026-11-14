//! Opt-in native turns. Provider completion and task completion are distinct.
//!
//! One driver ([`NativeLoop::drive`]) serves both HTTP ask/ask-stream and
//! write-capable `llm_takeover` conversations (#85). Callers differ only in how
//! they build the initial messages, which [`RoundBudget`] governs the loop, and
//! what they do with the outcome.
use super::llm_handlers::EventSender;
use super::{AppConfig, AskRequest, AskResponse, MutationResult};
use crate::llm::{
    ChatMessage, LlmClient, NativeResponse, NativeStreamEvent, ToolCall, TurnRequest,
};
use std::collections::HashSet;

/// HTTP turns (`main`) have no turn identity, so guest tools cannot be bound to a
/// speaker here: writers and execute-only speakers are native, guests stay legacy.
pub(super) fn eligible(config: &AppConfig) -> bool {
    use super::native_tools::{speaker_class, SpeakerClass};
    config.native_tools
        && matches!(
            speaker_class(&permission_snapshot(config)),
            SpeakerClass::Writer | SpeakerClass::ExecuteOnly
        )
}

/// Pin generation identity and intersect the context profile with the active
/// model. The dispatcher and prompt receive this same narrowed configuration.
pub(super) fn permission_snapshot(config: &AppConfig) -> AppConfig {
    permission_snapshot_for(config, "main")
}

/// [`permission_snapshot`] for an explicit conversation key. Takeover turns pass
/// their own context so its per-speaker `permission_model` narrows the turn.
pub(super) fn permission_snapshot_for(config: &AppConfig, context: &str) -> AppConfig {
    let mut snapshot = config.clone();
    let model = config.llm_model.read().unwrap().clone();
    snapshot.llm_model = std::sync::Arc::new(std::sync::RwLock::new(model.clone()));
    let context_profile = config
        .meta
        .lock()
        .unwrap()
        .conversations
        .get(context)
        .and_then(|conv| conv.permission_model.clone());
    let original = &config.permission_config;
    let principal_profile = config
        .turn
        .as_ref()
        .and_then(|turn| original.principals.get(&turn.principal).cloned());
    let profiles: Vec<String> = context_profile
        .into_iter()
        .chain(principal_profile)
        .collect();
    if profiles.is_empty() {
        return snapshot;
    }
    let groups: HashSet<String> = std::iter::once("user".to_string())
        .chain(original.groups.keys().cloned())
        .chain(
            original
                .model
                .values()
                .flat_map(|model| model.group_perms.keys().cloned()),
        )
        .collect();
    let group_perms = groups
        .into_iter()
        .map(|group| {
            let active = original.resolve_group(config.access_level, &model, &group);
            let effective = profiles.iter().fold(active, |level, profile| {
                level.min(original.resolve_group(config.access_level, profile, &group))
            });
            (group, effective)
        })
        .collect();
    let mut permissions = original.as_ref().clone();
    permissions.model.insert(
        model.clone(),
        crate::permissions::ModelPermissions {
            group_perms,
            opencode: original.can_opencode(config.access_level, &model)
                && profiles
                    .iter()
                    .all(|profile| original.can_opencode(config.access_level, profile)),
        },
    );
    // A named profile without a usable entry cannot be classified: pin the most
    // restricted class (a guest with no tools) instead of falling through to the
    // "unlisted model resolves to Execute" default.
    let unclassifiable = profiles.iter().any(|profile| {
        let listed = original.model.contains_key(profile) || original.model.contains_key("default");
        !listed && (!original.model.is_empty() || original.guest_profiles.contains_key(profile))
    });
    // Either guest profile makes this a guest turn (excluded from HTTP native turns).
    if unclassifiable {
        permissions.guest_profiles.insert(model.clone(), Vec::new());
    } else if let Some(tools) = profiles
        .iter()
        .find_map(|profile| original.guest_profiles.get(profile))
    {
        permissions
            .guest_profiles
            .insert(model.clone(), tools.clone());
    }
    snapshot.permission_config = std::sync::Arc::new(permissions);
    if let Some(turn) = &mut snapshot.turn {
        turn.may_write &= snapshot
            .permission_config
            .authority(snapshot.access_level, &model)
            .may_write;
    }
    snapshot
}

/// How a native turn ended. Callers render it for their own transport.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum LoopEnd {
    Finished,
    /// Budget exhausted without an accepted finish.
    Exhausted,
    ProviderError(String),
    /// Two consecutive empty responses (takeover budget only).
    Empty(&'static str),
    /// An interactive finish had no visible answer or certified publication.
    NoOutcome,
    /// The provider cut off more consecutive responses than are corrected.
    CutOff,
    Cancelled { cleanup_confirmed: bool },
}

impl LoopEnd {
    pub(super) fn category(&self) -> &'static str {
        match self {
            Self::Finished => "finished",
            Self::Exhausted => "exhausted",
            Self::ProviderError(_) => "provider-error",
            Self::Empty(_) => "empty-response",
            Self::NoOutcome => "no-outcome",
            Self::CutOff => "cut-off",
            Self::Cancelled { .. } => "cancelled",
        }
    }
}

pub(super) struct NativeLoop {
    pub(super) messages: Vec<ChatMessage>,
    seen: HashSet<String>,
    pub(super) response: AskResponse,
    pub(super) finished: bool,
    training_context: String,
    /// Conversation key whose session history receives complete batches.
    context: String,
    /// HTTP policy for `main`: persist the incomplete-turn note as an
    /// assistant turn. Takeover conversations report incompleteness only in the
    /// reply, as the legacy takeover loop does. Persisted history is never
    /// trimmed (#68): the checkpointed view bounds what the model sees.
    persist_incomplete_note: bool,
    /// Latest non-empty user-visible prose (takeover reply text).
    pub(super) prose: String,
    /// Runtime line appended to the visible reply when finish was accepted
    /// with an unsaved edit of this turn (#146).
    pub(super) unsaved_notice: Option<String>,
    ladybug: Option<super::takeover_native::LadybugSink>,
    pub(super) end: LoopEnd,
    /// Model that generates, when it differs from the pinned permission model
    /// (takeover: a principal/context profile names authority, not a provider).
    pub(super) generator: Option<String>,
    /// Ladybug write failure; aborts the turn like the legacy takeover loop.
    pub(super) fatal: Option<anyhow::Error>,
    outcome: super::native_outcome::Outcome,
    /// Turn-bound tool calls of this turn (restricted speakers; unlimited for writers).
    /// Created from the pinned snapshot's class at the first tool call.
    tool_budget: Option<std::sync::Arc<super::native_tools::TurnBudget>>,
    /// Workspace state of this turn, created at its first workspace tool call.
    workspace: Option<super::native_tools::workspace::Turn>,
    /// Consecutive responses the provider cut off (reset by any other response).
    cut_offs: usize,
    /// The visible prose (`prose`) is the fragment of a cut-off response.
    pub(super) prose_cut_off: bool,
}

/// A cut-off response is corrected this many times in a row; the next one ends the
/// turn. A model whose every answer is cut off would otherwise produce a full output
/// limit of tokens per round up to the round ceiling.
const MAX_CUT_OFF_CORRECTIONS: usize = 2;
pub(super) const CUT_OFF_TURN_ERROR: &str = "The model's responses were cut off at its output limit three times in a row, so this turn was stopped. Nothing from the cut-off responses was carried out.";
pub(super) const CUT_OFF_REPLY_MARKER: &str = "[cut off at the model's output limit]";

pub(super) async fn run(
    config: &AppConfig,
    req: &AskRequest,
    tx: &EventSender,
    streaming: bool,
) -> AskResponse {
    // Freeze model/permission identity as well as protocol for this entire turn.
    let mut config = permission_snapshot(config);
    // Once per turn: the tool list cannot change within it.
    super::native_tools::workspace::attach(&mut config).await;
    let baseline = config.program.read().await.clone();
    let workspace_line = super::native_tools::workspace::tail_line(Some(&config)).await;
    let messages = prepare(&config, req, &baseline, workspace_line);
    let mut turn = NativeLoop::new(messages, req.message.clone(), "main", true, None);
    let mut budget = RoundBudget::Fixed(config.max_iterations);
    turn.drive(&config, &baseline, &mut budget, tx, streaming).await;
    turn.response.completed = Some(turn.finished);
    turn.response
}

fn prepare(
    config: &AppConfig,
    req: &AskRequest,
    baseline: &crate::ast::Program,
    workspace_line: Option<String>,
) -> Vec<ChatMessage> {
    let model = config.llm_model.read().unwrap().clone();
    let summary = crate::validator::program_summary_for_model(
        baseline,
        &config.permission_config,
        config.access_level,
        &model,
    ) + &config.unsaved_tail(&model);
    let (stable, runtime, key, model) =
        super::http_context::context_prompts(config, &summary, req.caller.as_deref(), true);
    let key = key.with_tools(super::http_context::tool_schema_hash(config));
    let stable = super::native_tools::workspace::with_prompt(stable, Some(config));
    let messages = {
        let mut meta = config.meta.lock().unwrap();
        let conv = meta.conversations.get_or_create("main");
        conv.push_user(req.message.clone());
        let known = super::http_context::tail_bytes(&[&runtime]) + super::http_context::tool_schema_bytes(config);
        let view = super::http_context::checkpointed_view_for(&mut meta.conversations, "main", &model, &stable, known, Some(config));
        let mut log = crate::context_log::ContextLog::from_conversation(view.messages);
        log.push_runtime_text(runtime);
        if let Some(line) = workspace_line {
            log.push_runtime_text(line);
        }
        super::http_context::report_prompt_epoch(&mut meta.conversations, "main", &key, &model, &stable, &mut log, view.checkpointed);
        log.into_messages()
    };
    config.notify_save();
    messages
}

async fn generate(
    llm: &LlmClient<crate::llm::OpenAiBackend>,
    request: TurnRequest,
    tx: &EventSender,
    streaming: bool,
) -> Result<NativeResponse, String> {
    let flag = |e: anyhow::Error| {
        if crate::llm::is_too_large(&e) {
            llm.mark_recovered();
        }
        e.to_string()
    };
    if !streaming {
        return llm.complete_turn(request).await.map_err(flag);
    }
    let mut rx = llm.stream_turn(request).await.map_err(flag)?;
    let mut final_output = None;
    while let Some(event) = rx.recv().await {
        match event {
            NativeStreamEvent::Content(text) => {
                tx.send(serde_json::json!({"type":"content","text":text}))
                    .await
            }
            NativeStreamEvent::Thinking(text) => {
                tx.send(serde_json::json!({"type":"thinking","text":text}))
                    .await
            }
            NativeStreamEvent::Done(output) => final_output = Some(output),
            NativeStreamEvent::Error(error) => return Err(error),
        }
    }
    final_output.ok_or_else(|| {
        "stream ended without a validated terminal response; no calls executed".into()
    })
}

impl NativeLoop {
    pub(super) fn new(
        messages: Vec<ChatMessage>,
        training_context: String,
        context: &str,
        persist_incomplete_note: bool,
        ladybug: Option<super::takeover_native::LadybugSink>,
    ) -> Self {
        Self {
            messages,
            seen: HashSet::new(),
            finished: false,
            training_context,
            context: context.to_string(),
            persist_incomplete_note,
            prose: String::new(),
            unsaved_notice: None,
            ladybug,
            end: LoopEnd::Exhausted,
            generator: None,
            fatal: None,
            outcome: Default::default(),
            tool_budget: None,
            workspace: None,
            cut_offs: 0,
            prose_cut_off: false,
            response: AskResponse {
                reply: String::new(),
                code: String::new(),
                results: vec![],
                test_results: vec![],
                has_errors: false,
                completed: Some(false),
            },
        }
    }

    /// The one native round loop. Every caller pins `config` (model and
    /// permission snapshot) and `baseline` (immutable completion gate, #66)
    /// once per turn; neither is refreshed between rounds. Never executes
    /// `<code>` text and never falls back to the legacy protocol.
    pub(super) async fn drive(
        &mut self,
        config: &AppConfig,
        baseline: &crate::ast::Program,
        budget: &mut RoundBudget,
        tx: &EventSender,
        streaming: bool,
    ) {
        let model = self
            .generator
            .clone()
            .unwrap_or_else(|| config.llm_model.read().unwrap().clone());
        let llm = super::http_context::conversation_client(
            &config.llm_url,
            &model,
            config.llm_api_key.clone(),
            &self.context,
        );
        let mut iteration = 0;
        loop {
            let final_round = match budget.next(iteration) {
                RoundStep::Stop => break,
                RoundStep::Final => {
                    self.correct(FINAL_ROUND_CORRECTION);
                    true
                }
                RoundStep::Go => false,
            };
            tx.send(serde_json::json!({"type":"iteration", "n":iteration + 1,
                "detail":format!("iteration {}/{}", iteration + 1, budget.limit())}))
                .await;
            let started = std::time::Instant::now();
            let request = self.request(config, &model, !final_round);
            let max_tokens = request.max_tokens;
            // The final round sends no schemas, so measure this request's own.
            let schema_bytes = super::http_context::schema_bytes(
                request.tools.iter().map(|tool| (tool.name.as_str(), tool.description.as_str(), &tool.parameters)),
            );
            let sent = super::http_context::SentRequest::measure(&model, &request.messages, schema_bytes);
            // Without the turn's tools the provider cannot read the turn's cached prefix.
            let sent = if final_round { sent.with_other_tools() } else { sent };
            let result = generate(&llm, request, tx, streaming).await;
            // `generate` flags shrunk retries and too-large failures on the client.
            // Persist the forced checkpoint now: the provider-error branch below
            // ends the turn, and takeover turns save nothing else on that path.
            let recovered = llm.take_recovered();
            if super::http_context::after_llm_call(recovered, &Ok::<(), anyhow::Error>(()), &config.meta, &self.context) {
                config.notify_save();
            }
            let output = match result {
                Ok(output) => output,
                Err(error) => {
                    self.error(format!("Native provider error: {error}"), tx).await;
                    self.end = LoopEnd::ProviderError(error);
                    break;
                }
            };
            super::http_context::report_llm_usage(&self.context, started, &output.usage);
            // A refused native turn is a provider error (unsupported finish
            // reason) and never reaches this point.
            sent.calibrate(&config.meta, &self.context, recovered, false, &output.usage);
            if final_round {
                // The budget is spent: the status round is prose-only and can
                // never execute a tool, even if the provider ignores tool_choice.
                self.final_status(config, output, tx).await;
                break;
            }
            if output.finish_reason == crate::llm::NativeFinishReason::Length {
                if !self.cut_off(config, budget, iteration, output, max_tokens, tx).await {
                    break;
                }
                iteration += 1;
                continue;
            }
            self.cut_offs = 0;
            if output.content.trim().is_empty() && output.tool_calls.is_empty() {
                match budget.on_empty(iteration) {
                    EmptyStep::Ordinary => {}
                    EmptyStep::Retry => {
                        self.correct(NATIVE_EMPTY_RESPONSE_CORRECTION);
                        iteration += 1;
                        continue;
                    }
                    EmptyStep::Terminate(error) => {
                        self.end = LoopEnd::Empty(error);
                        break;
                    }
                }
            } else {
                budget.on_non_empty(&output.content);
            }
            if !self.round(config, baseline, budget, iteration, output, tx).await {
                break;
            }
            iteration += 1;
        }
        self.conclude(config, tx).await;
    }

    async fn final_status(&mut self, config: &AppConfig, output: NativeResponse, tx: &EventSender) {
        self.present(&output, tx).await;
        if output.finish_reason == crate::llm::NativeFinishReason::Length {
            // The final round has no correction budget: mark retained partial prose,
            // without asking for another generation. `present` just reset the flag.
            self.prose_cut_off |= !super::llm_handlers::visible_takeover_prose(&output.content).is_empty();
        }
        if let Err(error) = self.persist_ladybug(&output.content).await {
            self.fatal = Some(error);
            return;
        }
        if !output.tool_calls.is_empty() {
            self.error("Final status round returned tool calls after the budget was spent; none were executed.".into(), tx).await;
        }
        if !output.content.trim().is_empty() {
            self.messages.push(ChatMessage::assistant(&output.content));
            self.persist_assistant(config, output.content);
        }
    }

    /// `with_tools = false` sends no tool schemas (`tool_choice: none`).
    fn request(&self, config: &AppConfig, model: &str, with_tools: bool) -> TurnRequest {
        TurnRequest {
            messages: self.messages.clone(),
            // -1 = omit, matching the legacy client: claude-opus-5 rejects any
            // explicit temperature with 400 "temperature is deprecated".
            temperature: -1.0,
            // #68: the generating model's configured output limit.
            max_tokens: crate::model_limits::for_model(model).output_tokens,
            tools: super::native_tools::tool_definitions(config)
                .into_iter()
                .filter(|_| with_tools)
                .map(|tool| crate::llm::NativeToolDefinition {
                    name: tool.name.into(),
                    description: tool.description.into(),
                    parameters: tool.parameters,
                })
                .collect(),
        }
    }

    /// One validated provider response. Returns false when the loop must stop.
    async fn round(
        &mut self,
        config: &AppConfig,
        baseline: &crate::ast::Program,
        budget: &mut RoundBudget,
        iteration: usize,
        output: NativeResponse,
        tx: &EventSender,
    ) -> bool {
        self.present(&output, tx).await;
        if let Err(error) = self.persist_ladybug(&output.content).await {
            self.fatal = Some(error);
            return false;
        }
        if output.tool_calls.is_empty() {
            self.messages.push(ChatMessage::assistant(&output.content));
            self.persist_assistant(config, output.content);
            self.correct("No native tool call was supplied. Continue from completed results without replaying actions, or call finish alone to complete your answer or clarification.");
            budget.extend_for_correction(iteration);
            return true;
        }
        if let Err(error) = self.validate_envelope(&output.tool_calls) {
            self.error(format!("Native protocol error: {error}"), tx).await;
            self.correct(&format!("Rejected tool envelope; no calls executed: {error}. Use fresh unique call ids and finish alone."));
            budget.extend_for_correction(iteration);
            return true;
        }
        self.execute_batch(config, baseline, output, tx).await;
        if self.workspace.as_ref().is_some_and(|turn| turn.cancelled) {
            let cleanup_confirmed = self.workspace.as_ref().is_some_and(|turn| turn.cleanup_confirmed);
            let notice = if cleanup_confirmed { "Foreground cancellation requested; scope cleanup confirmed. No later calls were executed." }
                else { "Foreground cancellation requested, but cleanup could not be fully confirmed; processes may still be running. No later calls were executed. Do not replay the command." };
            self.error(notice.into(), tx).await;
            self.end = LoopEnd::Cancelled { cleanup_confirmed };
            return false;
        }
        if self.finished {
            self.end = LoopEnd::Finished;
            return false;
        }
        match self.outcome.take_rejection() {
            Some(super::native_outcome::Rejection::CorrectOnce) => {
                self.end = LoopEnd::NoOutcome;
                self.correct(super::native_outcome::CORRECTION);
                budget.extend_for_correction(iteration);
            }
            Some(super::native_outcome::Rejection::Stop) => {
                self.end = LoopEnd::NoOutcome;
                return false;
            }
            None => {}
        }
        true
    }

    /// The provider cut the response off. Nothing of it is executed (the response
    /// carries no tool calls); the partial text is kept as an ordinary assistant
    /// message and the model is told what happened. Returns false when the loop must stop.
    async fn cut_off(
        &mut self,
        config: &AppConfig,
        budget: &mut RoundBudget,
        iteration: usize,
        output: NativeResponse,
        max_tokens: u32,
        tx: &EventSender,
    ) -> bool {
        self.present(&output, tx).await;
        // If this fragment is what the person gets to read, it is marked as cut off.
        self.prose_cut_off |= !super::llm_handlers::visible_takeover_prose(&output.content).is_empty();
        if let Err(error) = self.persist_ladybug(&output.content).await {
            self.fatal = Some(error);
            return false;
        }
        self.error("Native response was cut off by the provider; nothing from it was executed.".into(), tx).await;
        if !output.content.trim().is_empty() {
            budget.on_non_empty(&output.content);
            self.messages.push(ChatMessage::assistant(&output.content));
            self.persist_assistant(config, output.content);
        }
        self.cut_offs += 1;
        if self.cut_offs > MAX_CUT_OFF_CORRECTIONS {
            self.error(CUT_OFF_TURN_ERROR.into(), tx).await;
            self.end = LoopEnd::CutOff;
            return false;
        }
        self.correct(&format!(
            "Your previous response was cut off before it was complete: the provider stopped it at the limit of {max_tokens} output tokens for one response, or at the context window. Nothing from it was executed, including any tool call you were writing. Continue from the last completed result and send large content in smaller pieces, each well under that limit."
        ));
        budget.extend_for_correction(iteration);
        true
    }

    async fn conclude(&mut self, config: &AppConfig, tx: &EventSender) {
        if !self.finished && self.fatal.is_none() {
            let no_outcome = self.end == LoopEnd::NoOutcome;
            let message = if no_outcome {
                super::native_outcome::INCOMPLETE
            } else if matches!(self.end, LoopEnd::Cancelled { .. }) {
                "Foreground cancellation requested; task completion is unconfirmed."
            } else { "Iteration limit or provider failure without successful finish; task completion is unconfirmed." };
            self.error(message.into(), tx).await;
            // HTTP records the outcome in `main`; takeover conversations only
            // tell the person in the reply, as the legacy takeover loop does.
            if self.persist_incomplete_note {
                self.persist_assistant(
                    config,
                    if no_outcome { super::native_outcome::INCOMPLETE.into() } else {
                        "Native turn ended without successful finish; task completion is unconfirmed.".into()
                    },
                );
            }
        }
        tx.send(serde_json::json!({"type":"done", "completed":self.finished, "has_errors":self.response.has_errors})).await;
    }

    async fn persist_ladybug(&self, content: &str) -> anyhow::Result<()> {
        match &self.ladybug {
            Some(sink) if !content.trim().is_empty() => sink.assistant(content.to_string()).await,
            _ => Ok(()),
        }
    }

    fn persist_assistant(&self, config: &AppConfig, content: String) {
        {
            let mut meta = config.meta.lock().unwrap();
            let conv = meta.conversations.get_or_create(&self.context);
            conv.push_assistant(content);
        }
        config.notify_save();
    }

    fn correct(&mut self, message: &str) {
        self.messages.push(ChatMessage::user(message));
    }

    async fn error(&mut self, message: String, tx: &EventSender) {
        self.response.has_errors = true;
        if !self.response.reply.is_empty() {
            self.response.reply.push_str("\n\n");
        }
        self.response.reply.push_str(&message);
        self.response.results.push(MutationResult {
            message: message.clone(),
            success: false,
        });
        tx.send(serde_json::json!({"type":"error", "message":message}))
            .await;
    }

    async fn present(&mut self, output: &NativeResponse, tx: &EventSender) {
        let prose = super::llm_handlers::visible_takeover_prose(&output.content);
        if !prose.is_empty() {
            self.prose = prose;
            if self.end == LoopEnd::NoOutcome {
                self.end = LoopEnd::Exhausted;
            }
            self.prose_cut_off = false;
        }
        if !output.content.is_empty() {
            if !self.response.reply.is_empty() {
                self.response.reply.push_str("\n\n");
            }
            self.response.reply.push_str(&output.content);
            tx.send(serde_json::json!({"type":"text", "text":output.content}))
                .await;
        }
    }

    fn validate_envelope(&self, calls: &[ToolCall]) -> Result<(), String> {
        let mut ids = HashSet::new();
        for call in calls {
            if call.id.trim().is_empty() || self.seen.contains(&call.id) || !ids.insert(&call.id) {
                return Err("empty or repeated call id".into());
            }
        }
        if calls.len() > 1 && calls.iter().any(|c| c.name == "finish") {
            return Err("finish must be the sole tool call".into());
        }
        Ok(())
    }

    async fn execute_batch(
        &mut self,
        config: &AppConfig,
        baseline: &crate::ast::Program,
        output: NativeResponse,
        tx: &EventSender,
    ) {
        let mut results = Vec::new();
        let mut successes = Vec::new();
        let test_start = self.response.test_results.len();
        let result_start = self.response.results.len();
        let validation = output.tool_calls.iter().find_map(|call| {
            super::native_tools::validate_native_tool_authorized(
                &call.name,
                &call.arguments,
                config,
            )
            .err()
            .map(|error| (call, error))
        });
        // A workspace preflight error is private content even when the same
        // refusal is correlated to ordinary peer calls of this skipped batch.
        // Without workspace tools these spellings are ordinary unknown names:
        // preserve that turn's pre-workspace refusal policy.
        let workspace_refusal = super::native_tools::workspace::available(config)
            && validation.as_ref().is_some_and(|(call, _)| super::native_tools::workspace::is_workspace_name(&call.name));
        self.messages.push(ChatMessage::assistant_with_tool_calls(
            &output.content,
            output.tool_calls.clone(),
        ));
        for call in &output.tool_calls {
            self.seen.insert(call.id.clone());
            let (result_before, test_before) = (self.response.results.len(), self.response.test_results.len());
            let content = if self.workspace.as_ref().is_some_and(|turn| turn.cancelled) {
                self.workspace_outcome(super::native_tools::workspace::Outcome::failed(&call.name, "earlier foreground operation cancelled; this call was not executed"), tx).await
            } else if let Some((_, error)) = &validation {
                let message = format!("Batch skipped before execution: {error}");
                let public = if workspace_refusal {
                    "Native tool batch refused before execution; no calls were executed.".into()
                } else { message.clone() };
                self.error(public, tx).await;
                serde_json::json!({"success":false,"error":message}).to_string()
            } else {
                self.execute_call(config, baseline, call, tx).await
            };
            self.messages
                .push(ChatMessage::tool_result(&call.id, &content));
            results.push(content);
            successes.push(validation.is_none()
                && self.response.results[result_before..].iter().all(|result| result.success)
                && self.response.test_results[test_before..].iter().all(|test| test.pass));
        }
        let model = config.llm_model.read().unwrap().clone();
        // Workspace calls are recorded without argument values and result content.
        use super::native_tools::workspace::{logged_call, logged_result};
        let (logged_calls, logged_results): (Vec<ToolCall>, Vec<String>) = if workspace_refusal || self.workspace.as_ref().is_some_and(|turn| turn.cancelled) {
            // Sanitize EVERY skipped call, including peer ids/args and feedback;
            // its raw result carries the workspace-origin validation detail too.
            let tools = super::native_tools::tool_definitions(config);
            output.tool_calls.iter().zip(&results).zip(successes).map(|((call, result), success)| {
                if workspace_refusal { super::native_tools::workspace::logged_skipped(call,result,&tools) }
                else { super::native_tools::workspace::logged_categorical(call,result,success,&tools) }
            }).unzip()
        } else {
            output.tool_calls.iter().zip(&results).zip(successes)
                .map(|((call, result), success)| (logged_call(call), logged_result(call, result, success))).unzip()
        };
        super::llm_handlers::log_training_data(
            &config.training_log,
            &model,
            &self.training_context,
            &output.reasoning,
            &serde_json::to_string(&logged_calls).unwrap_or_default(),
            &logged_results,
            self.response.results[result_start..]
                .iter()
                .any(|r| !r.success)
                || self.response.test_results[test_start..]
                    .iter()
                    .any(|t| !t.pass),
            self.response.test_results[test_start..]
                .iter()
                .filter(|t| t.pass)
                .count(),
            self.response.test_results[test_start..]
                .iter()
                .filter(|t| !t.pass)
                .count(),
        )
        .await;
        {
            let mut meta = config.meta.lock().unwrap();
            let conv = meta.conversations.get_or_create(&self.context);
            conv.push_native_batch(output.content, output.tool_calls, results);
        }
        config.notify_save();
    }

    async fn execute_call(
        &mut self,
        config: &AppConfig,
        baseline: &crate::ast::Program,
        call: &ToolCall,
        tx: &EventSender,
    ) -> String {
        let class = super::native_tools::speaker_class(config);
        let budget = self.tool_budget.get_or_insert_with(|| super::native_tools::turn_budget(&class)).clone();
        // Workspace tools take their own path (no working-set snapshot, no write-back, no
        // envelope). The branch comes AFTER the turn's tool budget exists and draws from
        // it for a restricted class. Today only the writer class has these tools and its
        // budget is unlimited, so the draw never refuses; it is here so that offering a
        // workspace tool to another class later cannot bypass that class's budget.
        if super::native_tools::workspace::is_tool(&call.name) && super::native_tools::workspace::available(config) {
            if class.restricted() {
                if let Err(refusal) = budget.take(&call.name) {
                    return self.workspace_outcome(super::native_tools::workspace::Outcome::failed(&call.name, refusal), tx).await;
                }
            }
            return self.workspace_call(config, call, tx).await;
        }
        let mut session = config.snapshot_working_set().await;
        let mut result = super::native_tools::execute_native_tool_in_turn(
            &call.name,
            &call.arguments,
            config,
            &mut session,
            baseline,
            &budget,
        )
        .await;
        config.write_back_working_set(&session).await;
        if call.name == "finish" {
            self.outcome.check_finish(&self.prose, &mut result);
        }
        let mut execution = result.execution;
        // Read-only inspectors carry their output in the structured envelope,
        // rather than pretending the query mutated the program in the executor.
        if let Ok(envelope) = serde_json::from_str::<serde_json::Value>(&result.content) {
            if let Some(text) = envelope["query_results"].as_str() {
                execution.mutation_results.push(MutationResult {
                    message: text.to_string(),
                    success: !execution.has_errors,
                });
            }
        }
        for item in &execution.mutation_results {
            tx.send(
                serde_json::json!({"type":"result","message":item.message,"success":item.success}),
            )
            .await;
        }
        for item in &execution.test_results {
            tx.send(serde_json::json!({"type":"test","message":item.message,"pass":item.pass}))
                .await;
        }
        self.finished = execution.done_accepted && !execution.has_errors;
        if self.finished {
            if let Some(notice) = execution.unsaved_notice.take() {
                if !self.response.reply.is_empty() {
                    self.response.reply.push_str("\n\n");
                }
                self.response.reply.push_str(&notice);
                tx.send(serde_json::json!({"type":"text", "text":notice})).await;
                self.unsaved_notice = Some(notice);
            }
        }
        self.response.has_errors |= execution.has_errors;
        self.response.results.extend(execution.mutation_results);
        self.response.test_results.extend(execution.test_results);
        result.content.to_string()
    }
}

impl NativeLoop {
    /// A workspace tool: no working-set snapshot and no write-back, because these
    /// tools never touch the Program. The result is plain text; success is the
    /// structural flag of the outcome, never read back out of the text.
    async fn workspace_call(&mut self, config: &AppConfig, call: &ToolCall, tx: &EventSender) -> String {
        use super::native_tools::workspace::{Outcome, Turn};
        if self.workspace.is_none() {
            let model = self.generator.clone().unwrap_or_else(|| config.llm_model.read().unwrap().clone());
            self.workspace = Turn::start(config, &self.context, &model);
        }
        let outcome = match &mut self.workspace {
            Some(turn) => turn.call(&call.name, &call.arguments).await,
            // The caller checked `available`; kept so no path executes without the gate.
            None => Outcome::failed(&call.name, "not available in this conversation"),
        };
        self.workspace_outcome(outcome, tx).await
    }

    /// Record a workspace call for the event stream and the response: the tool and
    /// whether it worked, nothing of what it returned (`/api/events` has subscribers
    /// that are not the owner). The text goes to the model only.
    async fn workspace_outcome(&mut self, outcome: super::native_tools::workspace::Outcome, tx: &EventSender) -> String {
        tx.send(serde_json::json!({"type":"result","message":outcome.label,"success":outcome.success})).await;
        self.response.has_errors |= !outcome.success;
        self.response.results.push(MutationResult { message: outcome.label, success: outcome.success });
        outcome.text
    }
}

const NATIVE_EMPTY_RESPONSE_CORRECTION: &str = "Your previous response was empty. Resume from the latest tool result without repeating completed or irreversible actions. Call the next necessary tool, or call finish alone if the task is complete.";
const FINAL_ROUND_CORRECTION: &str = "The round budget for this turn is spent and no tools are available in this final round. Reply in plain text only: tell the person what was done, what remains or where you are stuck, and ask whether to continue. Do not claim unconfirmed work is complete.";

pub(super) enum RoundStep {
    Go,
    /// One status round granted after the takeover estimate is exhausted.
    Final,
    Stop,
}

pub(super) enum EmptyStep {
    /// Treat like any response without tool calls (HTTP behaviour).
    Ordinary,
    Retry,
    Terminate(&'static str),
}

/// Round accounting. HTTP keeps its fixed `max_iterations`; takeover keeps the
/// legacy takeover semantics: model estimate (default 10), 50-generation hard
/// ceiling, protocol corrections extend an exhausted estimate while the ceiling
/// allows, one retry for an empty response, one final status round.
pub(super) enum RoundBudget {
    Fixed(usize),
    Takeover {
        estimate: usize,
        observed: super::llm_handlers::TakeoverIterationEstimate,
        empty: super::llm_handlers::EmptyResponseRetry,
        calls: super::llm_handlers::TakeoverCallBudget,
        final_round_used: bool,
    },
}

impl RoundBudget {
    pub(super) fn takeover() -> Self {
        Self::Takeover {
            estimate: super::llm_handlers::DEFAULT_TAKEOVER_ITERATION_BUDGET,
            observed: Default::default(),
            empty: Default::default(),
            calls: Default::default(),
            final_round_used: false,
        }
    }

    fn next(&mut self, iteration: usize) -> RoundStep {
        match self {
            Self::Fixed(max) if iteration < *max => RoundStep::Go,
            Self::Fixed(_) => RoundStep::Stop,
            Self::Takeover { estimate, calls, final_round_used, .. } => {
                if iteration < *estimate && calls.reserve() {
                    RoundStep::Go
                } else if !*final_round_used && calls.reserve() {
                    *final_round_used = true;
                    RoundStep::Final
                } else {
                    RoundStep::Stop
                }
            }
        }
    }

    fn limit(&self) -> usize {
        match self {
            Self::Fixed(max) => *max,
            Self::Takeover { estimate, .. } => *estimate,
        }
    }

    fn on_empty(&mut self, iteration: usize) -> EmptyStep {
        let Self::Takeover { estimate, empty, calls, .. } = self else {
            return EmptyStep::Ordinary;
        };
        let replacement =
            super::llm_handlers::prepare_empty_replacement(iteration, estimate, calls);
        match empty.on_empty(replacement) {
            super::llm_handlers::EmptyResponseAction::Retry(_) => EmptyStep::Retry,
            super::llm_handlers::EmptyResponseAction::Terminate(error) => EmptyStep::Terminate(error),
        }
    }

    fn on_non_empty(&mut self, content: &str) {
        if let Self::Takeover { estimate, observed, empty, .. } = self {
            empty.on_non_empty();
            // Tool-only responses still count as the first meaningful response.
            if let Some(Some(requested)) = observed.observe(content, "native") {
                *estimate =
                    requested.clamp(1, super::llm_handlers::MAX_TAKEOVER_ITERATION_BUDGET);
            }
        }
    }

    fn extend_for_correction(&mut self, iteration: usize) {
        if let Self::Takeover { estimate, calls, .. } = self {
            if iteration + 1 >= *estimate && calls.has_remaining() {
                *estimate = (*estimate + 1).min(super::llm_handlers::MAX_TAKEOVER_ITERATION_BUDGET);
            }
        }
    }

    pub(super) fn generations(&self) -> Option<usize> {
        match self {
            Self::Fixed(_) => None,
            Self::Takeover { calls, .. } => Some(calls.used()),
        }
    }
}
