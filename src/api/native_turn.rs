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

pub(super) fn eligible(config: &AppConfig) -> bool {
    config.native_tools && super::native_tools::native_tools_eligible(&permission_snapshot(config))
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
    // Either guest profile excludes this turn from the HTTP native protocol.
    if let Some(tools) = profiles
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
}

pub(super) struct NativeLoop {
    pub(super) messages: Vec<ChatMessage>,
    seen: HashSet<String>,
    pub(super) response: AskResponse,
    pub(super) finished: bool,
    training_context: String,
    /// Conversation key whose session history receives complete batches.
    context: String,
    /// HTTP policy for `main`: trim history and persist the incomplete-turn
    /// note as an assistant turn. Takeover conversations rely on the bounded
    /// view + Ladybug compaction and report incompleteness only in the reply,
    /// as the legacy takeover loop does.
    trim_history: bool,
    /// Latest non-empty user-visible prose (takeover reply text).
    pub(super) prose: String,
    ladybug: Option<super::takeover_native::LadybugSink>,
    pub(super) end: LoopEnd,
    /// Model that generates, when it differs from the pinned permission model
    /// (takeover: a principal/context profile names authority, not a provider).
    pub(super) generator: Option<String>,
    /// Ladybug write failure; aborts the turn like the legacy takeover loop.
    pub(super) fatal: Option<anyhow::Error>,
}

pub(super) async fn run(
    config: &AppConfig,
    req: &AskRequest,
    tx: &EventSender,
    streaming: bool,
) -> AskResponse {
    // Freeze model/permission identity as well as protocol for this entire turn.
    let config = permission_snapshot(config);
    let baseline = config.program.read().await.clone();
    let messages = prepare(&config, req, &baseline);
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
) -> Vec<ChatMessage> {
    let model = config.llm_model.read().unwrap().clone();
    let summary = crate::validator::program_summary_for_model(
        baseline,
        &config.permission_config,
        config.access_level,
        &model,
    );
    let (stable, runtime, key, model) =
        super::http_context::context_prompts(config, &summary, req.caller.as_deref(), true);
    let messages = {
        let mut meta = config.meta.lock().unwrap();
        let conv = meta.conversations.get_or_create("main");
        conv.set_primary_system(stable.clone());
        conv.push_user(req.message.clone());
        let mut log = crate::context_log::ContextLog::from_conversation(
            conv.to_llm_messages_bounded(100, 120_000, 16_000),
        );
        log.push_runtime_text(runtime);
        super::http_context::report_prompt_epoch(
            "main",
            &key,
            &model,
            &stable,
            log.history(),
            false,
        );
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
    if !streaming {
        return llm.complete_turn(request).await.map_err(|e| e.to_string());
    }
    let mut rx = llm.stream_turn(request).await.map_err(|e| e.to_string())?;
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
        trim_history: bool,
        ladybug: Option<super::takeover_native::LadybugSink>,
    ) -> Self {
        Self {
            messages,
            seen: HashSet::new(),
            finished: false,
            training_context,
            context: context.to_string(),
            trim_history,
            prose: String::new(),
            ladybug,
            end: LoopEnd::Exhausted,
            generator: None,
            fatal: None,
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
        let llm =
            LlmClient::new_with_model_and_key(&config.llm_url, &model, config.llm_api_key.clone());
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
            let output = match generate(&llm, self.request(config, !final_round), tx, streaming).await {
                Ok(output) => output,
                Err(error) => {
                    self.error(format!("Native provider error: {error}"), tx).await;
                    self.end = LoopEnd::ProviderError(error);
                    break;
                }
            };
            super::http_context::report_llm_usage(&self.context, started, &output.usage);
            if final_round {
                // The budget is spent: the status round is prose-only and can
                // never execute a tool, even if the provider ignores tool_choice.
                self.final_status(config, output, tx).await;
                break;
            }
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
    fn request(&self, config: &AppConfig, with_tools: bool) -> TurnRequest {
        TurnRequest {
            messages: self.messages.clone(),
            // -1 = omit, matching the legacy client: claude-opus-5 rejects any
            // explicit temperature with 400 "temperature is deprecated".
            temperature: -1.0,
            max_tokens: 8192,
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
        if self.finished {
            self.end = LoopEnd::Finished;
            return false;
        }
        true
    }

    async fn conclude(&mut self, config: &AppConfig, tx: &EventSender) {
        if !self.finished && self.fatal.is_none() {
            self.error("Iteration limit or provider failure without successful finish; task completion is unconfirmed.".into(), tx).await;
            // HTTP records the outcome in `main`; takeover conversations only
            // tell the person in the reply, as the legacy takeover loop does.
            if self.trim_history {
                self.persist_assistant(
                    config,
                    "Native turn ended without successful finish; task completion is unconfirmed.".into(),
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
            if self.trim_history {
                conv.trim(51);
            }
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
        let test_start = self.response.test_results.len();
        let result_start = self.response.results.len();
        let validation = output.tool_calls.iter().find_map(|call| {
            super::native_tools::validate_native_tool_authorized(
                &call.name,
                &call.arguments,
                config,
            )
            .err()
        });
        self.messages.push(ChatMessage::assistant_with_tool_calls(
            &output.content,
            output.tool_calls.clone(),
        ));
        for call in &output.tool_calls {
            self.seen.insert(call.id.clone());
            let content = if let Some(error) = &validation {
                let message = format!("Batch skipped before execution: {error}");
                self.error(message.clone(), tx).await;
                serde_json::json!({"success":false,"error":message}).to_string()
            } else {
                self.execute_call(config, baseline, call, tx).await
            };
            self.messages
                .push(ChatMessage::tool_result(&call.id, &content));
            results.push(content);
        }
        let model = config.llm_model.read().unwrap().clone();
        super::llm_handlers::log_training_data(
            &config.training_log,
            &model,
            &self.training_context,
            &output.reasoning,
            &serde_json::to_string(&output.tool_calls).unwrap_or_default(),
            &results,
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
            if self.trim_history {
                conv.trim(101);
            }
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
        let mut session = config.snapshot_working_set().await;
        let result = super::native_tools::execute_native_tool(
            &call.name,
            &call.arguments,
            config,
            &mut session,
            baseline,
        )
        .await;
        config.write_back_working_set(&session).await;
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
        self.response.has_errors |= execution.has_errors;
        self.response.results.extend(execution.mutation_results);
        self.response.test_results.extend(execution.test_results);
        result.content.to_string()
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
