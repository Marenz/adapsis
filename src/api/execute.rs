//! Code execution pipeline for AdapsisOS.
//!
//! This module contains the core `execute_code` function that handles all
//! mutations, tests, evals, queries, watches, agents, and `!opencode`.

use crate::eval;

use super::{AppConfig, MutationResult, TestCaseResult, WorkingSet, EVAL_TIMEOUT_SECS};

pub(super) fn collect_opencode_tasks(ops: &[crate::parser::Operation]) -> Vec<String> {
    ops.iter()
        .filter_map(|op| match op {
            crate::parser::Operation::OpenCode(task) => Some(task.clone()),
            _ => None,
        })
        .collect()
}

/// Format the task registry for display.
pub fn format_tasks(registry: &Option<crate::coroutine::TaskRegistry>) -> String {
    let Some(reg) = registry else { return "No task registry (async not available).".to_string() };
    let tasks = reg.lock().unwrap();
    if tasks.is_empty() {
        return "No tasks.".to_string();
    }
    let mut out = String::new();
    let mut sorted: Vec<_> = tasks.values().collect();
    sorted.sort_by_key(|t| t.id);
    for t in sorted {
        out.push_str(&format!("  task {} [{}] — {}\n", t.id, t.function_name, t.status));
    }
    out
}

/// Format a detailed inspection of a single task, combining TaskInfo and TaskSnapshot.
pub fn format_inspect_task(
    task_registry: &Option<crate::coroutine::TaskRegistry>,
    snapshot_registry: &Option<crate::coroutine::TaskSnapshotRegistry>,
    task_id: i64,
) -> String {
    let Some(task_reg) = task_registry else {
        return "No task registry (async not available).".to_string();
    };
    let tasks = task_reg.lock().unwrap();
    let Some(info) = tasks.get(&task_id) else {
        return format!("No task with id {task_id}.");
    };

    let mut out = String::new();
    out.push_str(&format!("Task {}\n", info.id));
    out.push_str(&format!("  function: {}\n", info.function_name));
    out.push_str(&format!("  started:  {}\n", info.started_at));
    out.push_str(&format!("  status:   {}\n", info.status));

    if let Some(snap_reg) = snapshot_registry {
        if let Ok(snaps) = snap_reg.lock() {
            if let Some(snap) = snaps.get(&task_id) {
                if let Some(ref stmt_id) = snap.current_stmt_id {
                    out.push_str(&format!("  stmt:     {}\n", stmt_id));
                }
                out.push_str(&format!("  depth:    {}\n", snap.frame_depth));
                if snap.locals.is_empty() {
                    out.push_str("  locals:   (none)\n");
                } else {
                    out.push_str("  locals:\n");
                    for (name, val) in &snap.locals {
                        out.push_str(&format!("    {} = {}\n", name, val));
                    }
                }
            } else {
                out.push_str("  snapshot: (not yet captured)\n");
            }
        }
    } else {
        out.push_str("  snapshot: (registry not available)\n");
    }

    out
}

/// Parse `?inspect task N` queries, returning the task ID if matched.
pub fn parse_inspect_task_query(query: &str) -> Option<i64> {
    let parts: Vec<&str> = query.split_whitespace().collect();
    if parts.len() == 3 && parts[0] == "?inspect" && parts[1] == "task" {
        parts[2].parse::<i64>().ok()
    } else {
        None
    }
}

/// Build plan context string and whether the AI needs to create a new plan.
pub(super) fn build_plan_context(plan: &[crate::session::PlanStep]) -> (String, bool) {
    if plan.is_empty() {
        return (String::new(), true);
    }
    let all_done = plan.iter().all(|s| matches!(s.status, crate::session::PlanStatus::Done | crate::session::PlanStatus::Failed));
    let steps = plan.iter().enumerate().map(|(i, s)| {
        let icon = match s.status {
            crate::session::PlanStatus::Pending => "[ ]",
            crate::session::PlanStatus::InProgress => "[~]",
            crate::session::PlanStatus::Done => "[x]",
            crate::session::PlanStatus::Failed => "[!]",
        };
        format!("{} {}: {}", icon, i + 1, s.description)
    }).collect::<Vec<_>>().join("\n");
    (format!("\nCurrent plan:\n{steps}\n"), all_done)
}

/// Format library load errors for inclusion in the AI context.
/// Returns an empty string if there are no load errors.
pub(super) fn format_library_load_errors(meta: &crate::session::SessionMeta) -> String {
    if let Some(ref lib_state) = meta.library_state {
        if let Some(text) = lib_state.format_load_errors() {
            return format!("\nWARNING — Library module load failures:\n{text}Use `+await result:String = library_reload(\"\")` or `+await result:String = library_reload(\"ModuleName\")` to retry.\n");
        }
    }
    String::new()
}

// ═══════════════════════════════════════════════════════════════════════
// Operation dispatch helpers
// ═══════════════════════════════════════════════════════════════════════

/// Result of processing a single operation or a batch of operations.
pub(super) struct OperationResult {
    pub(super) feedback: Vec<String>,
    pub(super) has_errors: bool,
    pub(super) tests_passed: usize,
    pub(super) tests_failed: usize,
    /// Signals the main loop should stop after this iteration.
    pub(super) accepted_done: bool,
}

impl OperationResult {
    pub(super) fn new() -> Self {
        Self {
            feedback: Vec::new(),
            has_errors: false,
            tests_passed: 0,
            tests_failed: 0,
            accepted_done: false,
        }
    }

    pub(super) fn ok(&mut self, msg: impl Into<String>) {
        let entry = format!("OK: {}", msg.into());
        eprintln!("[op] {}", entry.chars().take(200).collect::<String>());
        self.feedback.push(entry);
    }

    /// Log to stderr but do NOT include in LLM feedback.
    /// Use for bookkeeping ops (mock registration, plan progress) where
    /// the information is already conveyed via plan_summary or is noise.
    pub(super) fn ok_silent(&self, msg: impl Into<String>) {
        let entry = format!("OK: {}", msg.into());
        eprintln!("[op] {}", entry.chars().take(200).collect::<String>());
    }

    pub(super) fn pass(&mut self, msg: impl Into<String>) {
        self.tests_passed += 1;
        let entry = format!("PASS: {}", msg.into());
        eprintln!("[op] {}", entry.chars().take(200).collect::<String>());
        self.feedback.push(entry);
    }

    pub(super) fn fail(&mut self, msg: impl Into<String>) {
        self.tests_failed += 1;
        self.has_errors = true;
        let entry = format!("FAIL: {}", msg.into());
        eprintln!("[op] {}", entry.chars().take(200).collect::<String>());
        self.feedback.push(entry);
    }

    pub(super) fn error(&mut self, msg: impl Into<String>) {
        self.has_errors = true;
        let entry = format!("ERROR: {}", msg.into());
        eprintln!("[op] {}", entry.chars().take(200).collect::<String>());
        self.feedback.push(entry);
    }

    pub(super) fn info(&mut self, msg: impl Into<String>) {
        let entry = msg.into();
        eprintln!("[op] {}", entry.chars().take(200).collect::<String>());
        self.feedback.push(entry);
    }
}

/// Result returned by `execute_code()` — no SSE events, just data.
pub struct CodeExecutionResult {
    pub mutation_results: Vec<MutationResult>,
    pub test_results: Vec<TestCaseResult>,
    pub has_errors: bool,
    /// True when `!opencode` succeeded and we need to `exec` into the new binary.
    pub needs_opencode_restart: bool,
    /// True when `!agent` was encountered — caller should know a background agent was spawned.
    pub agent_spawned: bool,
    /// Names of agents that were spawned.
    pub spawned_agent_names: Vec<String>,
    /// True when `!done` was accepted (all functions tested, task complete).
    pub done_accepted: bool,
}

impl CodeExecutionResult {
    fn new() -> Self {
        Self {
            mutation_results: Vec::new(),
            test_results: Vec::new(),
            has_errors: false,
            needs_opencode_restart: false,
            agent_spawned: false,
            spawned_agent_names: Vec::new(),
            done_accepted: false,
        }
    }

    fn push_ok(&mut self, msg: impl Into<String>) {
        self.mutation_results.push(MutationResult { message: msg.into(), success: true });
    }

    fn push_err(&mut self, msg: impl Into<String>) {
        let msg = msg.into();
        eprintln!("[execute_code:err] {}", msg.chars().take(200).collect::<String>());
        self.has_errors = true;
        self.mutation_results.push(MutationResult { message: msg, success: false });
    }

    fn push_test_pass(&mut self, msg: impl Into<String>) {
        self.test_results.push(TestCaseResult { message: msg.into(), pass: true });
    }

    fn push_test_fail(&mut self, msg: impl Into<String>) {
        self.has_errors = true;
        self.test_results.push(TestCaseResult { message: msg.into(), pass: false });
    }
}

/// Execute a block of Adapsis code against a working snapshot.
///
/// Optional callback info for agent completion notifications.
/// When set, spawned agents will call the LLM with the conversation context
/// after completion and deliver the result via the reply function.
#[derive(Clone)]
pub struct AgentCompletionCallback {
    /// Conversation context name (e.g. "telegram:1815217")
    pub context: String,
    /// Function to call with the result (e.g. "TelegramBot.send_reply")
    pub reply_fn: String,
    /// Argument to pass to reply_fn (e.g. "1815217")  
    pub reply_arg: String,
    /// LLM config for generating the summary
    pub llm_url: String,
    pub llm_model: String,
    pub llm_key: Option<String>,
}

/// All mutations, tests, evals, queries, watches, agents, and `!opencode` are
/// handled here.  SSE events are **not** sent — the caller is responsible for
/// inspecting the returned `CodeExecutionResult` and emitting any events it needs.
pub async fn execute_code(
    code: &str,
    config: &AppConfig,
    session: &mut WorkingSet,
    agent_callback: Option<AgentCompletionCallback>,
) -> CodeExecutionResult {
    let mut result = CodeExecutionResult::new();

    match crate::parser::parse(code) {
        Ok(ops) => {
            let opencode_tasks = collect_opencode_tasks(&ops);
            let mut needs_opencode_restart = false;

            // Remove duplicate function/type definitions before applying mutations
            let mut fns_removed = false;
            for op in &ops {
                match op {
                    crate::parser::Operation::Function(f) => {
                        session.program.functions.retain(|existing| existing.name != f.name);
                        fns_removed = true;
                    }
                    crate::parser::Operation::Type(t) => {
                        let name = t.name.clone();
                        session.program.types.retain(|existing: &crate::ast::TypeDecl| existing.name() != name);
                    }
                    _ => {}
                }
            }
            if fns_removed {
                session.program.rebuild_function_index();
            }

            // Handle !undo
            let has_undo = ops.iter().any(|op| matches!(op, crate::parser::Operation::Undo));
            if has_undo {
                if session.meta.revision > 0 {
                    let prev = session.meta.revision - 1;
                    match crate::session::rewind_to(&mut session.program, &mut session.runtime, &mut session.meta, &mut session.sandbox, prev) {
                        Ok(()) => result.push_ok(format!("Undone to rev {prev}")),
                        Err(e) => result.push_err(format!("Undo: {e}")),
                    }
                }
            }

            // Handle !plan
            let has_plan_ops = ops.iter().any(|op| matches!(op, crate::parser::Operation::Plan(_)));
            for op in &ops {
                if let crate::parser::Operation::Plan(action) = op {
                    match action {
                        crate::parser::PlanAction::Set(steps) => {
                            session.meta.plan = steps.iter().map(|s| crate::session::PlanStep {
                                description: s.clone(),
                                status: crate::session::PlanStatus::Pending,
                            }).collect();
                            result.push_ok(format!("Plan set: {} steps", steps.len()));
                        }
                        crate::parser::PlanAction::Progress(n) => {
                            let idx = n.saturating_sub(1);
                            if let Some(step) = session.meta.plan.get_mut(idx) {
                                step.status = crate::session::PlanStatus::Done;
                                result.push_ok(format!("Step {n} done: {}", step.description));
                            }
                        }
                        crate::parser::PlanAction::Fail(n) => {
                            let idx = n.saturating_sub(1);
                            if let Some(step) = session.meta.plan.get_mut(idx) {
                                step.status = crate::session::PlanStatus::Failed;
                                result.push_ok(format!("Step {n} failed: {}", step.description));
                            }
                        }
                        crate::parser::PlanAction::Show => {
                            let plan_str = session.meta.plan.iter().enumerate().map(|(i, s)| {
                                let icon = match s.status {
                                    crate::session::PlanStatus::Pending => "[ ]",
                                    crate::session::PlanStatus::InProgress => "[~]",
                                    crate::session::PlanStatus::Done => "[x]",
                                    crate::session::PlanStatus::Failed => "[!]",
                                };
                                format!("  {} {}: {}", icon, i + 1, s.description)
                            }).collect::<Vec<_>>().join("\n");
                            result.push_ok(if plan_str.is_empty() { "No plan set".to_string() } else { format!("Plan:\n{plan_str}") });
                        }
                    }
                }
            }
            if has_plan_ops {
                config.write_back_working_set(session).await;
            }

            // Apply mutations
            let has_mutations = ops.iter().any(|op| !matches!(op,
                crate::parser::Operation::Test(_) | crate::parser::Operation::Trace(_)
                | crate::parser::Operation::Eval(_) | crate::parser::Operation::Query(_)
                | crate::parser::Operation::Undo | crate::parser::Operation::Plan(_)
                | crate::parser::Operation::Watch { .. }
                | crate::parser::Operation::Agent { .. }
                | crate::parser::Operation::Message { .. }
                | crate::parser::Operation::Done
                | crate::parser::Operation::OpenCode(_)));

            if has_mutations {
                // Filter out !plan lines — already handled in the pre-pass above.
                // Without this, apply_to_tiers_async re-parses the raw code and
                // processes Plan operations a second time.
                let mutation_code: String = code.lines()
                    .filter(|line| {
                        let trimmed = line.trim();
                        !trimmed.starts_with("!plan ")
                            && trimmed != "!plan"
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                match crate::session::apply_to_tiers_async(&mut session.program, &mut session.runtime, &mut session.meta, &mut session.sandbox, &mutation_code, config.io_sender.as_ref()).await {
                    Ok(res) => {
                        config.write_back_working_set(session).await;
                        for (msg, ok) in res {
                            eprintln!("[execute_code:{}] {msg}", if ok { "ok" } else { "err" });
                            if !ok { result.has_errors = true; }
                            result.mutation_results.push(MutationResult { message: msg, success: ok });
                        }
                    }
                    Err(e) => {
                        result.push_err(format!("{e}"));
                    }
                }
            }

            // Handle tests, evals, queries, watches, agents, messages
            for op in &ops {
                match op {
                    crate::parser::Operation::Test(test) => {
                        let mut all_passed = true;
                        let needs_async = session.program.get_function(&test.function_name)
                            .is_some_and(|f| f.effects.iter().any(|e|
                                matches!(e, crate::ast::Effect::Io | crate::ast::Effect::Async)));

                        for case in &test.cases {
                            let case_result = if needs_async {
                                if let Some(sender) = &config.io_sender {
                                    let program = session.program.clone();
                                    let fn_name = test.function_name.clone();
                                    let case = case.clone();
                                    let mocks = session.meta.io_mocks.clone();
                                    let routes = session.runtime.http_routes.clone();
                                    let sender = sender.clone();
                                    crate::eval::eval_test_case_async(
                                        &program, &fn_name, &case, &mocks, sender, &routes,
                                    ).await
                                } else {
                                    crate::eval::eval_test_case_with_mocks(
                                        &session.program, &test.function_name, case, &session.meta.io_mocks, &session.runtime.http_routes,
                                    )
                                }
                            } else {
                                crate::eval::eval_test_case_with_mocks(
                                    &session.program, &test.function_name, case, &session.meta.io_mocks, &session.runtime.http_routes,
                                )
                            };
                            match case_result {
                                Ok(msg) => {
                                    eprintln!("[execute_code:pass] {msg}");
                                    result.push_test_pass(msg);
                                }
                                Err(e) => {
                                    all_passed = false;
                                    eprintln!("[execute_code:fail] {e}");
                                    result.push_test_fail(format!("{e}"));
                                }
                            }
                        }
                        if all_passed && !test.cases.is_empty() {
                            crate::session::store_test(&mut session.program, &test.function_name, &test.cases);
                        }
                    }
                    crate::parser::Operation::Eval(ev) => {
                        // Inline expression: evaluate directly
                        if let Some(ref expr) = ev.inline_expr {
                            // Check if expression needs async eval: IO builtins or [io,async] user functions
                            let needs_async_eval = crate::eval::expr_contains_io_builtin(expr)
                                || crate::eval::expr_calls_io_function(expr, &session.program);
                            if needs_async_eval {
                                if let Some(sender) = &config.io_sender {
                                    let program = session.program.clone();
                                    let program_mut = crate::eval::make_shared_program_mut(&program);
                                    let expr = expr.clone();
                                    let sender = sender.clone();
                                    let ctx = eval::EvalContext::new(config.runtime.clone(), config.meta.clone(), config.event_broadcast.clone(), &program, program_mut.clone());
                                    let eval_result = tokio::time::timeout(
                                        std::time::Duration::from_secs(EVAL_TIMEOUT_SECS),
                                        tokio::task::spawn_blocking(move || {
                                            ctx.install();
                                            crate::eval::eval_inline_expr_with_io(&program, &expr, sender)
                                        })
                                    ).await;
                                    let (msg, success) = match &eval_result {
                                        Ok(Ok(Ok(val))) => (format!("= {val}"), true),
                                        Ok(Ok(Err(e))) => { (format!("eval error: {e}"), false) }
                                        Ok(Err(e)) => { (format!("eval task error: {e}"), false) }
                                        Err(_) => { (format!("eval timed out after {EVAL_TIMEOUT_SECS}s"), false) }
                                    };
                                    eprintln!("[execute_code:eval] {msg}");
                                    if success {
                                        result.push_ok(msg);
                                    } else {
                                        result.push_err(msg);
                                    }
                                    if let Some(mutated) = crate::eval::read_back_program_mutations(&program_mut) {
                                        session.program = mutated.clone();
                                        *config.program.write().await = mutated;
                                    }
                                    config.sync_async_side_effects_into(session);
                                    continue;
                                }
                            }
                            match crate::eval::eval_inline_expr(&session.program, expr) {
                                Ok(val) => {
                                    let msg = format!("= {val}");
                                    eprintln!("[execute_code:eval] {msg}");
                                    result.push_ok(msg);
                                }
                                Err(e) => {
                                    let msg = format!("eval error: {e}");
                                    eprintln!("[execute_code:eval:err] {msg}");
                                    result.push_err(msg);
                                }
                            }
                            continue;
                        }

                        // Block eval of untested functions in AdapsisOS mode
                        if session.program.require_modules {
                            if let Some(func) = session.program.get_function(&ev.function_name) {
                                if func.body.len() > 2 && !crate::session::is_function_tested(&session.program, &ev.function_name) {
                                    result.push_err(format!(
                                        "function `{}` has {} statements but no passing tests. Write +test blocks first.",
                                        ev.function_name, func.body.len()
                                    ));
                                    continue;
                                }
                            }
                        }

                        let needs_async = session.program.get_function(&ev.function_name)
                            .is_some_and(|f| f.effects.iter().any(|e|
                                matches!(e, crate::ast::Effect::Io | crate::ast::Effect::Async)));

                        if needs_async {
                            if let Some(sender) = &config.io_sender {
                                let program = session.program.clone();
                                let program_mut = crate::eval::make_shared_program_mut(&program);
                                let fn_name = ev.function_name.clone();
                                let input = ev.input.clone();
                                let sender = sender.clone();
                                let ctx = eval::EvalContext::new(config.runtime.clone(), config.meta.clone(), config.event_broadcast.clone(), &program, program_mut.clone());
                                let eval_fn_name = ev.function_name.clone();
                                let eval_result = tokio::time::timeout(
                                    std::time::Duration::from_secs(EVAL_TIMEOUT_SECS),
                                    tokio::task::spawn_blocking(move || {
                                        ctx.install();
                                        let func = program.get_function(&fn_name)
                                            .ok_or_else(|| anyhow::anyhow!("function not found"))?;
                                        let handle = crate::coroutine::CoroutineHandle::new(sender);
                                        let mut env = crate::eval::Env::new_with_shared_interner(&program.shared_interner);
                                        env.populate_shared_from_program(&program);
                                        env.set("__coroutine_handle", crate::eval::Value::CoroutineHandle(handle));
                                        let input_val = crate::eval::eval_parser_expr_with_program(&input, &program)?;
                                        crate::eval::bind_input_to_params(&program, func, &input_val, &mut env);
                                        crate::eval::eval_function_body_pub(&program, &func.body, &mut env)
                                    })
                                ).await;
                                let (msg, success) = match &eval_result {
                                    Ok(Ok(Ok(val))) => (format!("eval {}() = {val}", eval_fn_name), true),
                                    Ok(Ok(Err(e))) => { (format!("eval error: {e}"), false) }
                                    Ok(Err(e)) => { (format!("eval task error: {e}"), false) }
                                    Err(_) => { (format!("eval {}() timed out after {EVAL_TIMEOUT_SECS}s", eval_fn_name), false) }
                                };
                                eprintln!("[execute_code:eval] {msg}");
                                if success {
                                    result.push_ok(msg);
                                } else {
                                    result.push_err(msg);
                                }
                                if let Some(mutated) = crate::eval::read_back_program_mutations(&program_mut) {
                                    session.program = mutated.clone();
                                    *config.program.write().await = mutated;
                                }
                                config.sync_async_side_effects_into(session);
                            }
                        } else {
                            match crate::eval::eval_compiled_or_interpreted_cached(&session.program, &ev.function_name, &ev.input, Some(&config.jit_cache), session.meta.revision) {
                                Ok((val, compiled)) => {
                                    let tag = if compiled { " [compiled]" } else { "" };
                                    let msg = format!("eval {}() = {val}{tag}", ev.function_name);
                                    eprintln!("[execute_code:eval] {msg}");
                                    result.push_ok(msg);
                                }
                                Err(e) => {
                                    let msg = format!("eval error: {e}");
                                    eprintln!("[execute_code:eval:err] {msg}");
                                    result.push_err(msg);
                                }
                            }
                        }
                    }
                    crate::parser::Operation::Query(query) => {
                        let response = if query.trim() == "?inbox" || query.trim().starts_with("?inbox") {
                            let msgs = crate::session::peek_messages(&session.meta, "main");
                            if msgs.is_empty() {
                                "No messages.".to_string()
                            } else {
                                msgs.iter().map(|m| format!("[{}] from {}: {}", m.timestamp, m.from, m.content)).collect::<Vec<_>>().join("\n")
                            }
                        } else if query.trim() == "?tasks" {
                            format_tasks(&config.task_registry)
                        } else if let Some(tid) = parse_inspect_task_query(query.trim()) {
                            format_inspect_task(&config.task_registry, &config.snapshot_registry, tid)
                        } else if query.trim() == "?library" {
                            crate::library::query_library(&session.program, session.meta.library_state.as_ref())
                        } else {
                            let table = crate::typeck::build_symbol_table(&session.program);
                            crate::typeck::handle_query(&session.program, &table, query, &session.runtime.http_routes)
                        };
                        result.push_ok(response);
                    }
                    crate::parser::Operation::Watch { function_name, args, interval_ms } => {
                        eprintln!("[execute_code:watch] {function_name}({args}) every {interval_ms}ms");
                        let fn_name = function_name.clone();
                        let fn_args = args.clone();
                        let interval = *interval_ms;
                        let watch_program = config.program.clone();
                        let watch_meta = config.meta.clone();
                        let trigger = config.self_trigger.clone();
                        let watch_jit_cache = config.jit_cache.clone();

                        tokio::spawn(async move {
                            let mut last_result = String::new();
                            loop {
                                tokio::time::sleep(std::time::Duration::from_millis(interval)).await;
                                let result = {
                                    let program = watch_program.read().await;
                                    let meta = watch_meta.lock().unwrap();
                                    let input_expr = if fn_args.trim().is_empty() {
                                        crate::parser::Expr::StructLiteral(vec![])
                                    } else {
                                        match crate::parser::parse_test_input(0, &fn_args) {
                                            Ok(expr) => expr,
                                            Err(_) => break,
                                        }
                                    };
                                    match crate::eval::eval_compiled_or_interpreted_cached(
                                        &program, &fn_name, &input_expr,
                                        Some(&watch_jit_cache), meta.revision,
                                    ) {
                                        Ok((r, _)) => r,
                                        Err(e) => format!("error: {e}"),
                                    }
                                };
                                if result != last_result && !last_result.is_empty() {
                                    eprintln!("[execute_code:watch:trigger] {fn_name} changed: {last_result} → {result}");
                                    let msg = format!("Watcher '{fn_name}' triggered: result changed from '{last_result}' to '{result}'");
                                    let _ = trigger.send(msg).await;
                                }
                                last_result = result;
                            }
                        });
                        result.push_ok(format!("Watching {function_name}({args}) every {interval_ms}ms"));
                    }
                    crate::parser::Operation::Agent { name, scope, task } => {
                        eprintln!("[execute_code:agent] spawning '{name}' scope={scope} task={}", task.chars().take(80).collect::<String>());

                        let agent_scope = crate::session::AgentScope::parse(scope);
                        let branch = crate::session::AgentBranch::fork_from_parts(name, agent_scope, task, &session.program, &session.runtime, &session.meta);
                        let program_summary = crate::validator::program_summary_compact(&session.program);
                        let agent_task = task.clone();
                        let agent_name = name.clone();
                        let llm_url = config.llm_url.clone();
                        let llm_model = config.llm_model.clone();
                        let llm_key = config.llm_api_key.clone();
                        let agent_program = config.program.clone();
                        let agent_meta = config.meta.clone();
                        let agent_runtime = config.runtime.clone();
                        let agent_callback = agent_callback.clone();
                        let agent_io_sender = config.io_sender.clone();

                        tokio::spawn(async move {
                            eprintln!("[agent:{agent_name}] starting");
                            let agent_llm = crate::llm::LlmClient::new_with_model_and_key(&llm_url, &llm_model, llm_key);

                            let scope_desc = match &branch.scope {
                                crate::session::AgentScope::ReadOnly =>
                                    "SCOPE: read-only. You CAN: write +test blocks, use !eval, use ?queries. You CANNOT: define new functions or types, modify existing code.".to_string(),
                                crate::session::AgentScope::NewOnly =>
                                    "SCOPE: new-only. You CAN: define NEW functions and types, write +test blocks, use !eval. You CANNOT: modify or replace existing functions.".to_string(),
                                crate::session::AgentScope::Module(m) =>
                                    format!("SCOPE: module {m}. You CAN: modify anything in module {m}, add new functions to it. You CANNOT: modify code outside module {m}."),
                                crate::session::AgentScope::Full =>
                                    "SCOPE: full. You can modify anything.".to_string(),
                            };
                            let agent_system = format!(
                                "{}\n\n{}\n\nYou are agent '{agent_name}'.\n{scope_desc}\n\nYour task:\n{agent_task}\n\nWork step by step. Always include a <code> block with Adapsis code. When done, respond with !done in a <code> block.",
                                crate::prompt::system_prompt(),
                                crate::builtins::format_for_prompt()
                            );

                            let mut agent_messages = vec![
                                crate::llm::ChatMessage::system(agent_system),
                                crate::llm::ChatMessage::user(format!("Program state:\n{program_summary}\n\nTask: {agent_task}")),
                            ];

                            let mut branch = branch;
                            for agent_iter in 0..10 {
                                {
                                    let mut meta = agent_meta.lock().unwrap();
                                    let inbox = meta.agent_mailbox.remove(&agent_name).unwrap_or_default();
                                    if !inbox.is_empty() {
                                        let inbox_text = inbox.iter()
                                            .map(|m| format!("[from {}] {}", m.from, m.content))
                                            .collect::<Vec<_>>().join("\n");
                                        eprintln!("[agent:{agent_name}] received {} messages", inbox.len());
                                        agent_messages.push(crate::llm::ChatMessage::user(
                                            format!("Messages received:\n{inbox_text}\n\nIncorporate this information and continue.")
                                        ));
                                    }
                                }

                                let output = match agent_llm.generate(agent_messages.clone()).await {
                                    Ok(o) => o,
                                    Err(e) => { eprintln!("[agent:{agent_name}] LLM error: {e}"); break; }
                                };

                                agent_messages.push(crate::llm::ChatMessage::assistant(&output.text));

                                let code = output.code.clone();
                                if code.trim() == "!done" || code.is_empty() {
                                    eprintln!("[agent:{agent_name}] done at iter {agent_iter}");
                                    break;
                                }

                                if let Ok(ops) = crate::parser::parse(&code) {
                                    for op in &ops {
                                        if let crate::parser::Operation::Message { to, content } = op {
                                            eprintln!("[agent:{agent_name}] !msg → {to}: {content}");
                                            let mut meta = agent_meta.lock().unwrap();
                                            let msg = crate::session::AgentMessage {
                                                from: agent_name.clone(),
                                                to: to.clone(),
                                                content: content.clone(),
                                                timestamp: crate::session::now(),
                                            };
                                            meta.agent_mailbox.entry(to.clone()).or_default().push(msg);
                                        }
                                    }
                                }

                                match branch.apply(&code) {
                                    Ok(results) => {
                                        let mut has_err = false;
                                        for (msg, ok) in &results {
                                            eprintln!("[agent:{agent_name}] {}: {msg}", if *ok {"ok"} else {"err"});
                                            if !*ok { has_err = true; }
                                        }
                                        let feedback = results.iter()
                                            .map(|(msg, ok)| format!("{}: {msg}", if *ok {"OK"} else {"ERROR"}))
                                            .collect::<Vec<_>>().join("\n");
                                        if has_err {
                                            agent_messages.push(crate::llm::ChatMessage::user(format!("Errors:\n{feedback}\nFix and continue.")));
                                        } else {
                                            agent_messages.push(crate::llm::ChatMessage::user(format!("Results:\n{feedback}\nContinue or !done.")));
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("[agent:{agent_name}] apply error: {e}");
                                        agent_messages.push(crate::llm::ChatMessage::user(format!("Error: {e}\nFix and continue.")));
                                    }
                                }
                            }

                            // Merge branch back
                            let mut program = agent_program.read().await.clone();
                            let mut runtime = agent_runtime.read().unwrap().clone();
                            let mut meta = agent_meta.lock().unwrap().clone();
                            let mut sandbox = None;
                            let conflicts = branch.merge_into_parts(&mut program, &mut runtime, &mut meta, &mut sandbox);
                            if conflicts.is_empty() {
                                eprintln!("[agent:{agent_name}] merged successfully");
                                meta.conversations.get_or_create("main").push_system(
                                    format!("Agent '{agent_name}' completed and merged successfully."),
                                );
                                if let Some(s) = meta.agent_log.iter_mut().rev().find(|s| s.name == agent_name && s.status == "running") {
                                    s.status = "merged".to_string();
                                    s.message = "completed and merged".to_string();
                                }
                            } else {
                                eprintln!("[agent:{agent_name}] merge conflicts: {:?}", conflicts);
                                meta.conversations.get_or_create("main").push_system(
                                    format!("Agent '{agent_name}' finished but had merge conflicts:\n{}", conflicts.join("\n")),
                                );
                                if let Some(s) = meta.agent_log.iter_mut().rev().find(|s| s.name == agent_name && s.status == "running") {
                                    s.status = "conflict".to_string();
                                    s.message = conflicts.join("; ");
                                }
                            }
                            *agent_program.write().await = program;
                            *agent_runtime.write().unwrap() = runtime;
                            *agent_meta.lock().unwrap() = meta;

                            // Notify conversation context if callback configured
                            if let Some(cb) = agent_callback {
                                let agent_result = if conflicts.is_empty() {
                                    format!("Agent '{agent_name}' completed and merged successfully.")
                                } else {
                                    format!("Agent '{agent_name}' had merge conflicts: {}", conflicts.join(", "))
                                };
                                eprintln!("[agent:{agent_name}] notifying context '{}'", cb.context);

                                // Append result to conversation as user message to prompt summary
                                {
                                    let mut meta_guard = agent_meta.lock().unwrap();
                                    if let Some(conv) = meta_guard.conversations.get_mut(&cb.context) {
                                        conv.push_user(format!("[System: {}] Summarize the result briefly for the user.", agent_result));
                                    }
                                }

                                // Call LLM for a user-facing summary
                                let summary_messages = {
                                    let meta_guard = agent_meta.lock().unwrap();
                                    if let Some(conv) = meta_guard.conversations.get(&cb.context) {
                                        conv.messages.iter().map(|m| match m.role.as_str() {
                                            "system" => crate::llm::ChatMessage::system(m.content.clone()),
                                            "assistant" => crate::llm::ChatMessage::assistant(&m.content),
                                            _ => crate::llm::ChatMessage::user(m.content.clone()),
                                        }).collect::<Vec<_>>()
                                    } else {
                                        vec![]
                                    }
                                };

                                if !summary_messages.is_empty() {
                                    let summary_llm = crate::llm::LlmClient::new_with_model_and_key(
                                        &cb.llm_url, &cb.llm_model, cb.llm_key,
                                    );
                                    eprintln!("[agent:{agent_name}] calling LLM for completion summary ({} messages)", summary_messages.len());
                                    match summary_llm.generate(summary_messages).await {
                                        Ok(output) => {
                                            let mut reply = output.text.clone();
                                            while let Some(s) = reply.find("<think>") {
                                                if let Some(e) = reply[s..].find("</think>") {
                                                    reply.replace_range(s..s + e + 8, "");
                                                } else { break; }
                                            }
                                            while let Some(s) = reply.find("<code>") {
                                                if let Some(e) = reply[s..].find("</code>") {
                                                    reply.replace_range(s..s + e + 7, "");
                                                } else { break; }
                                            }
                                            let reply = reply.trim().to_string();

                                            if !reply.is_empty() {
                                                // Store in conversation
                                                {
                                                    let mut meta_guard = agent_meta.lock().unwrap();
                                                    if let Some(conv) = meta_guard.conversations.get_mut(&cb.context) {
                                                        conv.push_assistant(&reply);
                                                    }
                                                }
                                                // Deliver via callback
                                                eprintln!("[agent:{agent_name}] delivering reply via {}({})", cb.reply_fn, cb.reply_arg);
                                                if let Some(sender) = agent_io_sender {
                                                    let (tx, _rx) = tokio::sync::oneshot::channel();
                                                    let _ = sender.send(crate::coroutine::IoRequest::Spawn {
                                                        function_name: cb.reply_fn,
                                                        args: vec![
                                                            crate::eval::Value::string(cb.reply_arg),
                                                            crate::eval::Value::string(reply),
                                                        ],
                                                        reply: tx,
                                                    }).await;
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            eprintln!("[agent:{agent_name}] summary LLM call failed: {e}");
                                        }
                                    }
                                }
                            }
                        });

                        result.push_ok(format!("Agent '{name}' spawned (background)"));
                        result.agent_spawned = true;
                        result.spawned_agent_names.push(name.clone());

                        session.meta.agent_log.push(crate::session::AgentStatus {
                            name: name.clone(),
                            task: task.chars().take(100).collect(),
                            scope: scope.clone(),
                            status: "running".to_string(),
                            message: String::new(),
                        });
                    }
                    crate::parser::Operation::Done => {
                        if session.program.require_modules {
                            let untested: Vec<String> = session
                                .program
                                .modules
                                .iter()
                                .flat_map(|m| {
                                    m.functions.iter().filter_map(|f| {
                                        let qname = format!("{}.{}", m.name, f.name);
                                        if f.body.len() > 2 && !crate::session::is_function_tested(&session.program, &qname) {
                                            Some(qname)
                                        } else {
                                            None
                                        }
                                    })
                                })
                                .collect();
                            if !untested.is_empty() {
                                result.push_err(format!(
                                    "Cannot accept !done: {} untested functions: {}. Write +test blocks for them.",
                                    untested.len(),
                                    untested.join(", ")
                                ));
                                continue;
                            }
                        }
                        eprintln!("[execute_code:done] accepted");
                        result.done_accepted = true;
                    }
                    crate::parser::Operation::Message { to, content } => {
                        eprintln!("[execute_code:msg] → {to}: {content}");
                        crate::session::send_agent_message(&mut session.meta, "main", to, content);
                        config.write_back_working_set(session).await;
                        result.push_ok(format!("Message sent to '{to}'"));
                    }
                    crate::parser::Operation::OpenCode(_) => {}
                    // Top-level statements: execute immediately
                    op @ (crate::parser::Operation::Call(_)
                    | crate::parser::Operation::Let(_)
                    | crate::parser::Operation::Set(_)
                    | crate::parser::Operation::Await(_)
                    | crate::parser::Operation::Spawn(_)
                    | crate::parser::Operation::If(_)
                    | crate::parser::Operation::While(_)
                    | crate::parser::Operation::Each(_)
                    | crate::parser::Operation::Match(_)
                    | crate::parser::Operation::Check(_)
                    | crate::parser::Operation::Branch(_)
                    | crate::parser::Operation::Return(_)) => {
                        match crate::validator::convert_statement_op(op) {
                            Ok(stmt) => {
                                let mut env = crate::eval::Env::new_with_shared_interner(&session.program.shared_interner);
                                env.populate_shared_from_program(&session.program);
                                if let Some(sender) = &config.io_sender {
                                    env.set("__coroutine_handle", crate::eval::Value::CoroutineHandle(
                                        crate::coroutine::CoroutineHandle::new(sender.clone())
                                    ));
                                }
                                match crate::eval::eval_function_body_pub(&session.program, &[stmt], &mut env) {
                                    Ok(val) => {
                                        let msg = format!("executed: {val}");
                                        eprintln!("[execute_code:exec] {msg}");
                                        result.push_ok(msg);
                                    }
                                    Err(e) => {
                                        let msg = format!("exec error: {e}");
                                        eprintln!("[execute_code:exec:err] {msg}");
                                        result.push_err(msg);
                                    }
                                }
                                config.sync_async_side_effects_into(session);
                            }
                            Err(e) => {
                                result.push_err(format!("statement error: {e}"));
                            }
                        }
                    }
                    crate::parser::Operation::Trace(trace) => {
                        match crate::eval::trace_function(&session.program, &trace.function_name, &trace.input) {
                            Ok(steps) => {
                                let trace_output = steps.iter()
                                    .map(|s| format!("  > {s}"))
                                    .collect::<Vec<_>>()
                                    .join("\n");
                                result.push_ok(format!("Trace {}:\n{trace_output}", trace.function_name));
                            }
                            Err(e) => {
                                result.push_err(format!("Trace error: {e}"));
                            }
                        }
                    }
                    _ => {}
                }
            }

            // Handle !opencode tasks
            for task in opencode_tasks {
                eprintln!("[execute_code:opencode] {task}");
                let oc_result = tokio::time::timeout(
                    std::time::Duration::from_secs(3600),
                    tokio::process::Command::new("opencode")
                        .arg("run").arg("--format").arg("json")
                        .arg("--attach").arg("http://localhost:4096")
                        .arg("--dir").arg(&config.project_dir)
                        .arg(task)
                        .current_dir(&config.project_dir)
                        .output()
                ).await;
                match oc_result {
                    Ok(Ok(output)) if output.status.success() => {
                        eprintln!("[execute_code:opencode:done] rebuilding...");
                        let build = tokio::process::Command::new("cargo")
                            .arg("build").arg("--release").current_dir(&config.project_dir).output().await;
                        match build {
                            Ok(b) if b.status.success() => {
                                result.push_ok("OpenCode + rebuild successful. Restart to apply.".to_string());
                                needs_opencode_restart = true;
                            }
                            _ => {
                                result.push_err("OpenCode done but build failed".to_string());
                            }
                        }
                    }
                    _ => {
                        result.push_err("OpenCode failed or timed out".to_string());
                    }
                }
            }

            if needs_opencode_restart {
                result.needs_opencode_restart = true;
            }
        }
        Err(e) => {
            result.push_err(format!("Parse error: {e}"));
        }
    }

    result
}
