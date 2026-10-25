//! Opt-in HTTP native turns. Provider completion and task completion are distinct.
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
    let mut snapshot = config.clone();
    let model = config.llm_model.read().unwrap().clone();
    snapshot.llm_model = std::sync::Arc::new(std::sync::RwLock::new(model.clone()));
    let context_profile = config
        .meta
        .lock()
        .unwrap()
        .conversations
        .get("main")
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

struct Turn {
    messages: Vec<ChatMessage>,
    seen: HashSet<String>,
    response: AskResponse,
    finished: bool,
    training_context: String,
}

pub(super) async fn run(
    config: &AppConfig,
    req: &AskRequest,
    tx: &EventSender,
    streaming: bool,
) -> AskResponse {
    // Freeze model/permission identity as well as protocol for this entire turn.
    let config = permission_snapshot(config);
    let model = config.llm_model.read().unwrap().clone();
    let baseline = config.program.read().await.clone();
    let mut turn = Turn {
        messages: prepare(&config, req, &baseline),
        seen: HashSet::new(),
        finished: false,
        training_context: req.message.clone(),
        response: AskResponse {
            reply: String::new(),
            code: String::new(),
            results: vec![],
            test_results: vec![],
            has_errors: false,
            completed: Some(false),
        },
    };
    let llm =
        LlmClient::new_with_model_and_key(&config.llm_url, &model, config.llm_api_key.clone());
    for iteration in 0..config.max_iterations {
        tx.send(serde_json::json!({"type":"iteration", "n":iteration + 1,
            "detail":format!("iteration {}/{}", iteration + 1, config.max_iterations)}))
            .await;
        let request = TurnRequest {
            messages: turn.messages.clone(),
            temperature: 0.7,
            max_tokens: 8192,
            tools: super::native_tools::tool_definitions(&config)
                .into_iter()
                .map(|tool| crate::llm::NativeToolDefinition {
                    name: tool.name.into(),
                    description: tool.description.into(),
                    parameters: tool.parameters,
                })
                .collect(),
        };
        let started = std::time::Instant::now();
        let output = match generate(&llm, request, tx, streaming).await {
            Ok(output) => output,
            Err(error) => {
                turn.error(format!("Native provider error: {error}"), tx)
                    .await;
                break;
            }
        };
        super::http_context::report_llm_usage("main", started, &output.usage);
        turn.present(&output, tx).await;
        if output.tool_calls.is_empty() {
            turn.messages.push(ChatMessage::assistant(&output.content));
            super::http_context::persist_main_assistant(&config, output.content);
            turn.correct("No native tool call was supplied. Continue from completed results without replaying actions, or call finish alone to complete your answer or clarification.");
            continue;
        }
        if let Err(error) = turn.validate_envelope(&output.tool_calls) {
            turn.error(format!("Native protocol error: {error}"), tx)
                .await;
            turn.correct(&format!("Rejected tool envelope; no calls executed: {error}. Use fresh unique call ids and finish alone."));
            continue;
        }
        turn.execute_batch(&config, &baseline, output, tx).await;
        if turn.finished {
            break;
        }
    }
    if !turn.finished {
        turn.error("Iteration limit or provider failure without successful finish; task completion is unconfirmed.".into(), tx).await;
        super::http_context::persist_main_assistant(
            &config,
            "Native turn ended without successful finish; task completion is unconfirmed.".into(),
        );
    }
    tx.send(serde_json::json!({"type":"done", "completed":turn.finished, "has_errors":turn.response.has_errors})).await;
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

impl Turn {
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
            let conv = meta.conversations.get_or_create("main");
            conv.push_native_batch(output.content, output.tool_calls, results);
            conv.trim(101);
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
