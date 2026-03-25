//! Session management: mutation log, working history, save/load, revision control.
//!
//! Every change to the program is recorded as a numbered entry in the mutation log.
//! Evals, queries, and test results are recorded in the working history.
//! The program state can be reconstructed by replaying mutations 0..N.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::ast;
use crate::parser;
use crate::validator;

/// Runtime infrastructure state (Tier 2) — shared across async tasks via Arc<RwLock>.
/// Holds HTTP routes and shared variables, separate from the AST program state.
/// TODO(#9): Per-resource locks (each shared var gets its own Arc<RwLock<Value>>).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeState {
    pub http_routes: Vec<crate::ast::HttpRoute>,
    #[serde(skip)]
    pub shared_vars: HashMap<String, crate::eval::Value>,
    /// Roadmap mirror for builtin access during eval. Synced with SessionMeta.roadmap.
    #[serde(skip)]
    pub roadmap: Vec<RoadmapItem>,
    /// Plan mirror for builtin access during eval. Synced with SessionMeta.plan.
    #[serde(skip)]
    pub plan: Vec<PlanStep>,
    /// Agent mailbox mirror for builtin access during eval. Synced with SessionMeta.agent_mailbox.
    #[serde(skip)]
    pub agent_mailbox: HashMap<String, Vec<AgentMessage>>,
    /// Pending commands queued by IO builtins (watch_start, agent_spawn) for API-layer processing.
    #[serde(skip)]
    pub pending_commands: Vec<String>,
    /// IO mocks mirror for builtin access during eval. Synced with SessionMeta.io_mocks.
    #[serde(skip)]
    pub io_mocks: Vec<IoMock>,
}

/// Thread-safe handle to the runtime state.
pub type SharedRuntime = Arc<RwLock<RuntimeState>>;

/// A single entry in the mutation log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationEntry {
    pub revision: usize,
    pub timestamp: String,
    /// The raw Forge source for this mutation (so we can replay it)
    pub source: String,
    /// Human-readable summary of what changed
    pub summary: String,
    /// Whether this mutation was successfully applied
    pub success: bool,
}

/// A working history entry (non-mutation actions).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum HistoryEntry {
    Eval {
        revision: usize,
        function: String,
        input: String,
        result: String,
    },
    Test {
        revision: usize,
        function: String,
        passed: usize,
        failed: usize,
        details: Vec<String>,
    },
    Query {
        revision: usize,
        query: String,
        response: String,
    },
    Trace {
        revision: usize,
        function: String,
        steps: usize,
    },
    Note {
        revision: usize,
        text: String,
    },
}

/// Session metadata — mutation log, working history, conversation, and configuration.
/// Separated from the core program AST and runtime for Tier 3 modularity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    /// Append-only mutation log
    pub mutations: Vec<MutationEntry>,
    /// Working history (evals, tests, queries)
    pub history: Vec<HistoryEntry>,
    /// Current revision number (= number of successful mutations)
    pub revision: usize,
    /// All raw mutation sources (for replay)
    pub sources: Vec<String>,
    /// Conversation history for the LLM (persists across /api/ask calls)
    #[serde(default)]
    pub chat_messages: Vec<ChatMessage>,
    /// Active/completed agent statuses
    #[serde(default)]
    pub agent_log: Vec<AgentStatus>,
    /// Long-term roadmap — persists across sessions, the AI can modify it
    #[serde(default)]
    pub roadmap: Vec<RoadmapItem>,
    /// Current plan/goal tracking
    #[serde(default)]
    pub plan: Vec<PlanStep>,
    /// Agent message bus: agent_name → inbox of pending messages
    #[serde(default)]
    pub agent_mailbox: HashMap<String, Vec<AgentMessage>>,
    /// OpenCode session ID — reused across !opencode calls to maintain context
    #[serde(default)]
    pub opencode_session_id: Option<String>,
    /// IO mock table: (operation, url_pattern) -> response. Used during !test.
    #[serde(default)]
    pub io_mocks: Vec<IoMock>,
    /// Library state — tracks loaded modules and errors. Not serialized.
    #[serde(skip)]
    pub library_state: Option<crate::library::LibraryState>,
}

impl SessionMeta {
    pub fn new() -> Self {
        Self {
            mutations: Vec::new(),
            history: Vec::new(),
            revision: 0,
            sources: Vec::new(),
            chat_messages: Vec::new(),
            agent_log: Vec::new(),
            roadmap: Vec::new(),
            plan: Vec::new(),
            agent_mailbox: HashMap::new(),
            opencode_session_id: None,
            io_mocks: Vec::new(),
            library_state: None,
        }
    }
}

/// Snapshot of program + runtime state saved when entering a sandbox.
#[derive(Debug, Clone)]
pub struct SandboxState {
    pub original_program: ast::Program,
    pub original_runtime: RuntimeState,
    pub entered_at_revision: usize,
}

/// A Forge session — program state + runtime + metadata.
#[derive(Debug, Serialize, Deserialize)]
pub struct Session {
    /// Current program state
    pub program: ast::Program,
    /// Runtime state (Tier 2): HTTP routes, shared variables. Serialized with session.
    #[serde(default)]
    pub runtime: RuntimeState,
    /// Session metadata: mutation log, history, conversation, mocks, etc.
    #[serde(flatten)]
    pub meta: SessionMeta,
    /// Sandbox state — present when in sandbox mode. Not persisted.
    #[serde(skip)]
    pub sandbox: Option<SandboxState>,
}

/// A long-term roadmap item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoadmapItem {
    pub description: String,
    pub done: bool,
}

/// A mock IO response — matches operation + URL prefix, returns a fixed value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoMock {
    pub operation: String,      // e.g. "http_get", "http_post", "llm_call"
    pub patterns: Vec<String>,  // One pattern per argument position to match (contains check)
    pub response: String,       // value to return
}

/// A message sent between agents (or between main session and agents).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    pub from: String,
    pub to: String,
    pub content: String,
    pub timestamp: String,
}

/// A step in the AI's plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub description: String,
    pub status: PlanStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PlanStatus {
    Pending,
    InProgress,
    Done,
    Failed,
}

/// Status of an agent run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStatus {
    pub name: String,
    pub task: String,
    pub scope: String,
    pub status: String, // "running", "completed", "failed", "merged", "conflict"
    pub message: String,
}

/// Agent scope — what an agent is allowed to modify.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentScope {
    /// Can only add tests and evals, no mutations to program
    ReadOnly,
    /// Can only add new functions/types, can't modify existing
    NewOnly,
    /// Exclusive write to a specific module, read everywhere
    Module(String),
    /// Unrestricted
    Full,
}

impl AgentScope {
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "read-only" | "readonly" => AgentScope::ReadOnly,
            "new-only" | "newonly" => AgentScope::NewOnly,
            "full" => AgentScope::Full,
            s if s.starts_with("module ") => AgentScope::Module(s[7..].to_string()),
            s => AgentScope::Module(s.to_string()),
        }
    }

    /// Check if a mutation is allowed under this scope.
    pub fn allows_mutation(&self, op: &crate::parser::Operation) -> bool {
        match self {
            AgentScope::ReadOnly => {
                matches!(
                    op,
                    crate::parser::Operation::Test(_)
                        | crate::parser::Operation::Eval(_)
                        | crate::parser::Operation::Trace(_)
                        | crate::parser::Operation::Query(_)
                        | crate::parser::Operation::Message { .. }
                )
            }
            AgentScope::NewOnly => {
                match op {
                    crate::parser::Operation::Function(_)
                    | crate::parser::Operation::Type(_)
                    | crate::parser::Operation::Test(_)
                    | crate::parser::Operation::Eval(_)
                    | crate::parser::Operation::Query(_)
                    | crate::parser::Operation::Message { .. } => true,
                    crate::parser::Operation::Replace(_) => false, // can't modify existing
                    _ => false,
                }
            }
            AgentScope::Module(module_name) => {
                match op {
                    crate::parser::Operation::Test(_)
                    | crate::parser::Operation::Eval(_)
                    | crate::parser::Operation::Query(_)
                    | crate::parser::Operation::Message { .. } => true,
                    crate::parser::Operation::Move { target_module, .. } => {
                        target_module == module_name
                    }
                    // Allow all other mutations — the validator will check module membership
                    _ => true,
                }
            }
            AgentScope::Full => true,
        }
    }
}

/// An active agent branch.
#[derive(Debug)]
pub struct AgentBranch {
    pub name: String,
    pub scope: AgentScope,
    pub task: String,
    pub fork_revision: usize,
    pub program: ast::Program,
    pub mutations: Vec<String>,
    /// Forked copy of runtime state (isolated from main session).
    pub runtime_state: RuntimeState,
    /// Snapshot of shared_vars at fork time — used to detect changes during merge.
    runtime_state_snapshot: HashMap<String, crate::eval::Value>,
}

impl AgentBranch {
    /// Create a new branch forked from the current session state.
    pub fn fork(name: &str, scope: AgentScope, task: &str, session: &Session) -> Self {
        let runtime_state = session.runtime.clone();
        let snapshot = runtime_state.shared_vars.clone();
        Self {
            name: name.to_string(),
            scope,
            task: task.to_string(),
            fork_revision: session.meta.revision,
            program: session.program.clone(),
            mutations: Vec::new(),
            runtime_state,
            runtime_state_snapshot: snapshot,
        }
    }

    /// Access the forked runtime state.
    pub fn runtime(&self) -> &RuntimeState {
        &self.runtime_state
    }

    /// Apply a mutation to this branch (respecting scope).
    pub fn apply(&mut self, source: &str) -> Result<Vec<(String, bool)>> {
        let operations = crate::parser::parse(source)?;

        // Check scope
        for op in &operations {
            if !self.scope.allows_mutation(op) {
                return Ok(vec![(
                    format!(
                        "agent scope violation: {:?} not allowed in {:?}",
                        std::mem::discriminant(op),
                        self.scope
                    ),
                    false,
                )]);
            }
        }

        let mut results = Vec::new();
        for op in &operations {
            match op {
                crate::parser::Operation::Test(_)
                | crate::parser::Operation::Trace(_)
                | crate::parser::Operation::Eval(_)
                | crate::parser::Operation::Query(_)
                | crate::parser::Operation::Message { .. } => {}
                _ => match crate::validator::apply_and_validate(&mut self.program, op) {
                    Ok(msg) => results.push((msg, true)),
                    Err(e) => results.push((format!("{e}"), false)),
                },
            }
        }

        self.mutations.push(source.to_string());
        Ok(results)
    }

    /// Merge this branch's mutations back into a session.
    /// Returns list of conflicts (empty if clean merge).
    pub fn merge_into(self, session: &mut Session) -> Vec<String> {
        let mut conflicts = Vec::new();

        for source in &self.mutations {
            match session.apply(source) {
                Ok(results) => {
                    for (msg, ok) in &results {
                        if !ok {
                            conflicts.push(format!("merge conflict: {msg}"));
                        }
                    }
                }
                Err(e) => conflicts.push(format!("merge error: {e}")),
            }
        }

        // Merge shared_vars that changed during the branch's lifetime.
        // Only copy vars where the branch value differs from the fork-time snapshot.
        for (key, val) in &self.runtime_state.shared_vars {
            let changed = match self.runtime_state_snapshot.get(key) {
                Some(old_val) => format!("{old_val}") != format!("{val}"),
                None => true, // new key added during branch
            };
            if changed {
                session.runtime.shared_vars.insert(key.clone(), val.clone());
            }
        }

        conflicts
    }
}

/// A chat message for LLM conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl Session {
    pub fn new() -> Self {
        Self {
            program: ast::Program::default(),
            runtime: RuntimeState::default(),
            meta: SessionMeta::new(),
            sandbox: None,
        }
    }

    /// Send a message to an agent (or "main" for the main session).
    pub fn send_agent_message(&mut self, from: &str, to: &str, content: &str) {
        let msg = AgentMessage {
            from: from.to_string(),
            to: to.to_string(),
            content: content.to_string(),
            timestamp: now(),
        };
        self.meta.agent_mailbox
            .entry(to.to_string())
            .or_default()
            .push(msg);
    }

    /// Drain all pending messages for an agent.
    pub fn drain_messages(&mut self, agent_name: &str) -> Vec<AgentMessage> {
        self.meta.agent_mailbox.remove(agent_name).unwrap_or_default()
    }

    /// Peek at pending messages without removing them.
    pub fn peek_messages(&self, agent_name: &str) -> &[AgentMessage] {
        self.meta.agent_mailbox
            .get(agent_name)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Check whether a function is considered "tested":
    /// the function's AST `tests` field is non-empty and all test cases have `passed == true`.
    pub fn is_function_tested(&self, fn_name: &str) -> bool {
        if let Some(func) = self.program.get_function(fn_name) {
            !func.tests.is_empty() && func.tests.iter().all(|t| t.passed)
        } else {
            false
        }
    }

    /// Add or replace an HTTP route.
    pub fn add_route(&mut self, route: ast::HttpRoute) -> String {
        if let Some(existing) = self.runtime.http_routes.iter_mut()
            .find(|r| r.method == route.method && r.path == route.path)
        {
            let old_fn = existing.handler_fn.clone();
            existing.handler_fn = route.handler_fn.clone();
            format!("updated route {} {} -> `{}` (was `{old_fn}`)", route.method, route.path, route.handler_fn)
        } else {
            let msg = format!("added route {} {} -> `{}`", route.method, route.path, route.handler_fn);
            self.runtime.http_routes.push(route);
            msg
        }
    }

    /// Remove an HTTP route by method and path.
    pub fn remove_route(&mut self, method: &str, path: &str) -> Result<String> {
        let before = self.runtime.http_routes.len();
        let mut removed_handler = None;
        self.runtime.http_routes.retain(|r| {
            if r.method == method && r.path == path {
                removed_handler = Some(r.handler_fn.clone());
                false
            } else {
                true
            }
        });
        if self.runtime.http_routes.len() < before {
            Ok(format!("removed route {} {} (was -> `{}`)", method, path, removed_handler.unwrap_or_default()))
        } else {
            Err(anyhow!("no route found for {} {}", method, path))
        }
    }

    /// Remove all routes whose handler matches a function name.
    pub fn remove_routes_for_handler(&mut self, handler_name: &str) -> Vec<String> {
        let mut removed = Vec::new();
        self.runtime.http_routes.retain(|r| {
            if r.handler_fn == handler_name {
                removed.push(format!("{} {}", r.method, r.path));
                false
            } else {
                true
            }
        });
        removed
    }

    /// Get a snapshot of all HTTP routes.
    pub fn get_routes(&self) -> &[ast::HttpRoute] {
        &self.runtime.http_routes
    }

    /// Find a route by method and path.
    pub fn find_route(&self, method: &str, path: &str) -> Option<&ast::HttpRoute> {
        self.runtime.http_routes.iter().find(|r| r.method == method && r.path == path)
    }

    /// Handle a sandbox action (enter, merge, discard, status).
    fn handle_sandbox(&mut self, action: &parser::SandboxAction) -> (String, bool) {
        match action {
            parser::SandboxAction::Enter => {
                if self.sandbox.is_some() {
                    return ("already in sandbox mode — use !sandbox merge or !sandbox discard first".to_string(), false);
                }
                self.sandbox = Some(SandboxState {
                    original_program: self.program.clone(),
                    original_runtime: self.runtime.clone(),
                    entered_at_revision: self.meta.revision,
                });
                ("entered sandbox mode — mutations are isolated. Use !sandbox merge to keep changes or !sandbox discard to revert.".to_string(), true)
            }
            parser::SandboxAction::Merge => {
                if self.sandbox.is_none() {
                    return ("not in sandbox mode".to_string(), false);
                }
                let sandbox = self.sandbox.take().unwrap();
                let changes = self.meta.revision.saturating_sub(sandbox.entered_at_revision);
                (format!("sandbox merged — {changes} mutation(s) kept"), true)
            }
            parser::SandboxAction::Discard => {
                if self.sandbox.is_none() {
                    return ("not in sandbox mode".to_string(), false);
                }
                let sandbox = self.sandbox.take().unwrap();
                let discarded = self.meta.revision.saturating_sub(sandbox.entered_at_revision);
                self.program = sandbox.original_program;
                self.runtime = sandbox.original_runtime;
                self.meta.revision = sandbox.entered_at_revision;
                self.program.rebuild_function_index();
                (format!("sandbox discarded — reverted {discarded} mutation(s)"), true)
            }
            parser::SandboxAction::Status => {
                if let Some(ref sandbox) = self.sandbox {
                    let changes = self.meta.revision.saturating_sub(sandbox.entered_at_revision);
                    (format!("in sandbox mode (entered at revision {}, {changes} mutations since)", sandbox.entered_at_revision), true)
                } else {
                    ("not in sandbox mode".to_string(), true)
                }
            }
        }
    }

    /// Initialize shared variables from module declarations.
    /// For each +shared decl, if the key "Module.name" does not already exist
    /// in runtime.shared_vars, evaluate the default expression and insert it.
    pub fn init_shared_vars(&mut self) {
        for module in &self.program.modules {
            for sv in &module.shared_vars {
                let key = format!("{}.{}", module.name, sv.name);
                if !self.runtime.shared_vars.contains_key(&key) {
                    let value = crate::eval::eval_expr_standalone(&self.program, &sv.default)
                        .unwrap_or(crate::eval::Value::Int(0));
                    self.runtime.shared_vars.insert(key, value);
                }
            }
        }
    }

    /// Store test cases for a function in the AST.
    /// Populates the function's `tests` field (replace, not append).
    pub fn store_test(&mut self, fn_name: &str, cases: &[parser::TestCase]) {
        let ast_tests: Vec<ast::TestCase> = cases
            .iter()
            .map(|c| ast::TestCase {
                input: format_expr(&c.input),
                expected: format_expr(&c.expected),
                passed: true,
                matcher: c.matcher.as_ref().map(serialize_matcher),
                after_checks: c.after_checks.iter().map(|a| ast::AfterCheck {
                    target: a.target.clone(),
                    matcher: a.matcher.clone(),
                    value: a.value.clone(),
                }).collect(),
            })
            .collect();
        if let Some(func) = self.program.get_function_mut(fn_name) {
            func.tests = ast_tests.clone();
        }
        // Also store on qualified name(s) if bare name was given
        if !fn_name.contains('.') {
            let qnames: Vec<String> = self
                .program
                .modules
                .iter()
                .flat_map(|m| {
                    m.functions
                        .iter()
                        .filter(|f| f.name == fn_name)
                        .map(|f| format!("{}.{}", m.name, f.name))
                })
                .collect();
            for qn in &qnames {
                if let Some(func) = self.program.get_function_mut(qn) {
                    func.tests = ast_tests.clone();
                }
            }
        }
    }

    /// Extract the function names affected by an operation (for test invalidation).
    fn affected_function_names(op: &parser::Operation) -> Vec<String> {
        match op {
            parser::Operation::Module(m) => {
                m.body
                    .iter()
                    .filter_map(|body_op| {
                        if let parser::Operation::Function(f) = body_op {
                            Some(format!("{}.{}", m.name, f.name))
                        } else {
                            None
                        }
                    })
                    .collect()
            }
            parser::Operation::Replace(r) => {
                // Strip `.sN` suffix — e.g. "Mod.func.s1" → "Mod.func"
                let target = &r.target;
                let func_name = if let Some(dot_pos) = target.rfind('.') {
                    let suffix = &target[dot_pos + 1..];
                    if suffix.starts_with('s') && suffix[1..].parse::<usize>().is_ok() {
                        target[..dot_pos].to_string()
                    } else {
                        target.clone()
                    }
                } else {
                    target.clone()
                };
                vec![func_name]
            }
            _ => Vec::new(),
        }
    }

    /// Save pre-mutation backups of function bodies for affected functions
    /// that already have tests (for reject-on-fail).
    fn backup_affected_bodies(&self, op: &parser::Operation) -> HashMap<String, Vec<ast::Statement>> {
        let affected = Self::affected_function_names(op);
        let mut backups = HashMap::new();
        for name in &affected {
            if let Some(func) = self.program.get_function(name) {
                if !func.tests.is_empty() {
                    backups.insert(name.clone(), func.body.clone());
                }
            }
        }
        backups
    }

    /// Invalidate test status for affected functions, then re-run any stored
    /// tests. If any test fails, REVERT the function body to its pre-change
    /// state (reject-on-fail). Returns a list of (fn_name, passed, detail)
    /// for each re-run.
    ///
    /// `pre_backups` must be captured BEFORE `apply_and_validate` so the
    /// backup contains the original (pre-mutation) function body.
    fn invalidate_and_retest(
        &mut self,
        op: &parser::Operation,
        pre_backups: HashMap<String, Vec<ast::Statement>>,
    ) -> Vec<(String, bool, String)> {
        let affected = Self::affected_function_names(op);
        if affected.is_empty() {
            return Vec::new();
        }

        let backups = pre_backups;

        // Invalidate: reset AST passed flags
        for name in &affected {
            if let Some(func) = self.program.get_function_mut(name) {
                for t in &mut func.tests {
                    t.passed = false;
                }
            }
        }

        let mut retest_results = Vec::new();

        for name in &affected {
            // Read test cases from the function's AST (primary source of truth)
            let ast_cases: Vec<ast::TestCase> = self.program.get_function(name)
                .map(|f| f.tests.clone())
                .unwrap_or_default();

            if ast_cases.is_empty() {
                continue;
            }

            // Reconstruct test source and re-run
            let bare = name.rsplit('.').next().unwrap_or(name);
            let mut test_src = format!("!test {bare}\n");
            for case in &ast_cases {
                // Reconstruct the expect portion, including matcher syntax
                let expect_str = reconstruct_expect(&case.expected, case.matcher.as_deref());
                test_src.push_str(&format!("  +with {} -> expect {}\n", case.input, expect_str));
                for ac in &case.after_checks {
                    test_src.push_str(&format!("  +after {} {} \"{}\"\n", ac.target, ac.matcher, ac.value));
                }
            }

            match parser::parse(&test_src) {
                Ok(ops) => {
                    for test_op in &ops {
                        if let parser::Operation::Test(test) = test_op {
                            let mut all_passed = true;
                            for case in &test.cases {
                                match crate::eval::eval_test_case_with_mocks(
                                    &self.program,
                                    &test.function_name,
                                    case,
                                    &self.meta.io_mocks,
                                    &self.runtime.http_routes,
                                ) {
                                    Ok(msg) => {
                                        retest_results.push((
                                            name.clone(),
                                            true,
                                            format!("retest PASS: {msg}"),
                                        ));
                                    }
                                    Err(e) => {
                                        all_passed = false;
                                        retest_results.push((
                                            name.clone(),
                                            false,
                                            format!("retest FAIL: {e}"),
                                        ));
                                    }
                                }
                            }
                            if all_passed {
                                // Restore passed=true on AST tests after successful retest
                                if let Some(func) = self.program.get_function_mut(name) {
                                    for t in &mut func.tests {
                                        t.passed = true;
                                    }
                                }
                            } else {
                                // Reject: revert to the backed-up body and restore test flags
                                if let Some(old_body) = backups.get(name) {
                                    if let Some(func) = self.program.get_function_mut(name) {
                                        func.body = old_body.clone();
                                        for t in &mut func.tests {
                                            t.passed = true;
                                        }
                                    }
                                    retest_results.push((
                                        name.clone(),
                                        false,
                                        "REJECTED: replacement reverted because existing tests failed".to_string(),
                                    ));
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    // Parse error reconstructing tests — revert to be safe
                    if let Some(old_body) = backups.get(name) {
                        if let Some(func) = self.program.get_function_mut(name) {
                            func.body = old_body.clone();
                            for t in &mut func.tests {
                                t.passed = true;
                            }
                        }
                    }
                    retest_results.push((
                        name.clone(),
                        false,
                        format!("retest parse error (reverted): {e}"),
                    ));
                }
            }
        }

        retest_results
    }

    /// Apply a block of Forge source code as a mutation.
    /// Returns (results, new_revision) on success.
    pub fn apply(&mut self, source: &str) -> Result<Vec<(String, bool)>> {
        let operations = parser::parse(source)?;
        let mut results = Vec::new();
        let mut any_definition = false;

        for op in &operations {
            match op {
                parser::Operation::Test(test) => {
                    // Run tests and track which functions pass
                    let mut all_passed = true;
                    for case in &test.cases {
                        match crate::eval::eval_test_case_with_mocks(
                            &self.program,
                            &test.function_name,
                            case,
                            &self.meta.io_mocks,
                            &self.runtime.http_routes,
                        ) {
                            Ok(msg) => results.push((format!("PASS: {msg}"), true)),
                            Err(e) => {
                                all_passed = false;
                                results.push((format!("FAIL: {e}"), false));
                            }
                        }
                    }
                    if all_passed && !test.cases.is_empty() {
                        self.store_test(&test.function_name, &test.cases);
                    }
                }
                parser::Operation::Trace(_)
                | parser::Operation::Eval(_)
                | parser::Operation::Query(_) => {
                    // These don't modify program state — handled separately
                }
                parser::Operation::Plan(action) => match action {
                    parser::PlanAction::Set(steps) => {
                        self.meta.plan = steps
                            .iter()
                            .map(|s| PlanStep {
                                description: s.clone(),
                                status: PlanStatus::Pending,
                            })
                            .collect();
                        results.push((format!("Plan: {} steps", steps.len()), true));
                    }
                    parser::PlanAction::Progress(n) => {
                        if let Some(step) = self.meta.plan.get_mut(n.saturating_sub(1)) {
                            step.status = PlanStatus::Done;
                            results.push((format!("Step {n} done"), true));
                        }
                    }
                    parser::PlanAction::Fail(n) => {
                        if let Some(step) = self.meta.plan.get_mut(n.saturating_sub(1)) {
                            step.status = PlanStatus::Failed;
                            results.push((format!("Step {n} failed"), true));
                        }
                    }
                    parser::PlanAction::Show => {
                        let plan_str = self
                            .meta.plan
                            .iter()
                            .enumerate()
                            .map(|(i, s)| {
                                let icon = match s.status {
                                    PlanStatus::Pending => "[ ]",
                                    PlanStatus::InProgress => "[~]",
                                    PlanStatus::Done => "[x]",
                                    PlanStatus::Failed => "[!]",
                                };
                                format!("{} {}: {}", icon, i + 1, s.description)
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        results.push((
                            if plan_str.is_empty() {
                                "No plan set".to_string()
                            } else {
                                format!("Plan:\n{plan_str}")
                            },
                            true,
                        ));
                    }
                },
                parser::Operation::Roadmap(action) => { results.push(self.handle_roadmap(action)); }
                parser::Operation::Mock {
                    operation,
                    patterns,
                    response,
                } => {
                    let pattern_display = patterns.iter().map(|p| format!("\"{p}\"")).collect::<Vec<_>>().join(" ");
                    self.meta.io_mocks.push(IoMock {
                        operation: operation.clone(),
                        patterns: patterns.clone(),
                        response: response.clone(),
                    });
                    results.push((
                        format!(
                            "mock: {operation} {pattern_display} -> \"{}\"",
                            response.chars().take(50).collect::<String>()
                        ),
                        true,
                    ));
                }
                parser::Operation::Unmock => {
                    let count = self.meta.io_mocks.len();
                    self.meta.io_mocks.clear();
                    results.push((format!("cleared {count} mocks"), true));
                }
                parser::Operation::Route { method, path, handler_fn } => {
                    let route = ast::HttpRoute {
                        method: method.clone(),
                        path: path.clone(),
                        handler_fn: handler_fn.clone(),
                    };
                    let msg = self.add_route(route);
                    results.push((msg, true));
                }
                parser::Operation::RemoveRoute { method, path } => {
                    match self.remove_route(method, path) {
                        Ok(msg) => results.push((msg, true)),
                        Err(e) => results.push((format!("{e}"), false)),
                    }
                }
                parser::Operation::Sandbox(action) => {
                    let msg = self.handle_sandbox(action);
                    results.push(msg);
                }
                _ => {
                    any_definition = true;
                    let pre_backups = self.backup_affected_bodies(op);
                    match validator::apply_and_validate(&mut self.program, op) {
                        Ok(msg) => {
                            results.push((msg, true));
                            for (name, passed, detail) in self.invalidate_and_retest(op, pre_backups) {
                                results.push((detail, passed));
                                let _ = name;
                            }
                        }
                        Err(e) => results.push((format!("{e}"), false)),
                    }
                }
            }
        }

        let success = results.iter().all(|(_, ok)| *ok);
        let summary = if results.is_empty() {
            "no mutations".to_string()
        } else {
            results
                .iter()
                .map(|(msg, ok)| {
                    if *ok {
                        format!("OK: {msg}")
                    } else {
                        format!("ERR: {msg}")
                    }
                })
                .collect::<Vec<_>>()
                .join("; ")
        };

        if any_definition {
            self.meta.revision += 1;
            self.meta.mutations.push(MutationEntry {
                revision: self.meta.revision,
                timestamp: now(),
                source: source.to_string(),
                summary,
                success,
            });
            self.meta.sources.push(source.to_string());

            // Persist affected modules to the library
            let affected = crate::library::affected_module_names(&operations);
            eprintln!("[library] apply: any_definition={any_definition} success={success} affected={affected:?} lib_state={}", self.meta.library_state.is_some());
            if success && !affected.is_empty() {
                crate::library::persist_affected_modules(
                    &self.program,
                    &affected,
                    self.meta.library_state.as_ref(),
                );
            }
        }

        // Initialize any newly-declared shared variables
        self.init_shared_vars();

        Ok(results)
    }

    /// Apply a block of Forge source code, running async tests through the
    /// coroutine runtime when an `io_sender` is available. Falls back to
    /// mock-only execution otherwise. Must be called from within a tokio runtime.
    pub async fn apply_async(
        &mut self,
        source: &str,
        io_sender: Option<&tokio::sync::mpsc::Sender<crate::coroutine::IoRequest>>,
    ) -> Result<Vec<(String, bool)>> {
        let operations = parser::parse(source)?;
        let mut results = Vec::new();
        let mut any_definition = false;

        for op in &operations {
            match op {
                parser::Operation::Test(test) => {
                    let mut all_passed = true;
                    let needs_async = self
                        .program
                        .get_function(&test.function_name)
                        .is_some_and(|f| {
                            f.effects.iter().any(|e| {
                                matches!(e, ast::Effect::Async | ast::Effect::Io)
                            })
                        });

                    for case in &test.cases {
                        let case_result = if needs_async {
                            if let Some(sender) = io_sender {
                                crate::eval::eval_test_case_async(
                                    &self.program,
                                    &test.function_name,
                                    case,
                                    &self.meta.io_mocks,
                                    sender.clone(),
                                    &self.runtime.http_routes,
                                )
                                .await
                            } else {
                                crate::eval::eval_test_case_with_mocks(
                                    &self.program,
                                    &test.function_name,
                                    case,
                                    &self.meta.io_mocks,
                                    &self.runtime.http_routes,
                                )
                            }
                        } else {
                            crate::eval::eval_test_case_with_mocks(
                                &self.program,
                                &test.function_name,
                                case,
                                &self.meta.io_mocks,
                                &self.runtime.http_routes,
                            )
                        };

                        match case_result {
                            Ok(msg) => results.push((format!("PASS: {msg}"), true)),
                            Err(e) => {
                                all_passed = false;
                                results.push((format!("FAIL: {e}"), false));
                            }
                        }
                    }
                    if all_passed && !test.cases.is_empty() {
                        self.store_test(&test.function_name, &test.cases);
                    }
                }
                parser::Operation::Trace(_)
                | parser::Operation::Eval(_)
                | parser::Operation::Query(_) => {}
                parser::Operation::Plan(action) => match action {
                    parser::PlanAction::Set(steps) => {
                        self.meta.plan = steps
                            .iter()
                            .map(|s| PlanStep {
                                description: s.clone(),
                                status: PlanStatus::Pending,
                            })
                            .collect();
                        results.push((format!("Plan: {} steps", steps.len()), true));
                    }
                    parser::PlanAction::Progress(n) => {
                        if let Some(step) = self.meta.plan.get_mut(n.saturating_sub(1)) {
                            step.status = PlanStatus::Done;
                            results.push((format!("Step {n} done"), true));
                        }
                    }
                    parser::PlanAction::Fail(n) => {
                        if let Some(step) = self.meta.plan.get_mut(n.saturating_sub(1)) {
                            step.status = PlanStatus::Failed;
                            results.push((format!("Step {n} failed"), true));
                        }
                    }
                    parser::PlanAction::Show => {
                        let plan_str = self
                            .meta.plan
                            .iter()
                            .enumerate()
                            .map(|(i, s)| {
                                let icon = match s.status {
                                    PlanStatus::Pending => "[ ]",
                                    PlanStatus::InProgress => "[~]",
                                    PlanStatus::Done => "[x]",
                                    PlanStatus::Failed => "[!]",
                                };
                                format!("{} {}: {}", icon, i + 1, s.description)
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        results.push((
                            if plan_str.is_empty() {
                                "No plan set".to_string()
                            } else {
                                format!("Plan:\n{plan_str}")
                            },
                            true,
                        ));
                    }
                },
                parser::Operation::Roadmap(action) => { results.push(self.handle_roadmap(action)); }
                parser::Operation::Mock {
                    operation,
                    patterns,
                    response,
                } => {
                    let pattern_display = patterns.iter().map(|p| format!("\"{p}\"")).collect::<Vec<_>>().join(" ");
                    self.meta.io_mocks.push(IoMock {
                        operation: operation.clone(),
                        patterns: patterns.clone(),
                        response: response.clone(),
                    });
                    results.push((
                        format!(
                            "mock: {operation} {pattern_display} -> \"{}\"",
                            response.chars().take(50).collect::<String>()
                        ),
                        true,
                    ));
                }
                parser::Operation::Unmock => {
                    let count = self.meta.io_mocks.len();
                    self.meta.io_mocks.clear();
                    results.push((format!("cleared {count} mocks"), true));
                }
                parser::Operation::Route { method, path, handler_fn } => {
                    let route = ast::HttpRoute {
                        method: method.clone(),
                        path: path.clone(),
                        handler_fn: handler_fn.clone(),
                    };
                    let msg = self.add_route(route);
                    results.push((msg, true));
                }
                parser::Operation::RemoveRoute { method, path } => {
                    match self.remove_route(method, path) {
                        Ok(msg) => results.push((msg, true)),
                        Err(e) => results.push((format!("{e}"), false)),
                    }
                }
                parser::Operation::Sandbox(action) => {
                    let msg = self.handle_sandbox(action);
                    results.push(msg);
                }
                _ => {
                    any_definition = true;
                    let pre_backups = self.backup_affected_bodies(op);
                    match validator::apply_and_validate(&mut self.program, op) {
                        Ok(msg) => {
                            results.push((msg, true));
                            for (name, passed, detail) in self.invalidate_and_retest(op, pre_backups) {
                                results.push((detail, passed));
                                let _ = name;
                            }
                        }
                        Err(e) => results.push((format!("{e}"), false)),
                    }
                }
            }
        }

        let success = results.iter().all(|(_, ok)| *ok);
        let summary = if results.is_empty() {
            "no mutations".to_string()
        } else {
            results
                .iter()
                .map(|(msg, ok)| {
                    if *ok {
                        format!("OK: {msg}")
                    } else {
                        format!("ERR: {msg}")
                    }
                })
                .collect::<Vec<_>>()
                .join("; ")
        };

        if any_definition {
            self.meta.revision += 1;
            self.meta.mutations.push(MutationEntry {
                revision: self.meta.revision,
                timestamp: now(),
                source: source.to_string(),
                summary,
                success,
            });
            self.meta.sources.push(source.to_string());

            // Persist affected modules to the library
            let affected = crate::library::affected_module_names(&operations);
            eprintln!("[library] apply_async: any_definition={any_definition} success={success} affected={affected:?} lib_state={}", self.meta.library_state.is_some());
            if success && !affected.is_empty() {
                crate::library::persist_affected_modules(
                    &self.program,
                    &affected,
                    self.meta.library_state.as_ref(),
                );
            }
        }

        // Initialize any newly-declared shared variables
        self.init_shared_vars();

        Ok(results)
    }


    /// Get the parsed operations from a source string (for test/eval/query handling).
    pub fn parse_operations(&self, source: &str) -> Result<Vec<parser::Operation>> {
        parser::parse(source)
    }

    /// Record an eval in the working history.
    pub fn record_eval(&mut self, function: &str, input: &str, result: &str) {
        self.meta.history.push(HistoryEntry::Eval {
            revision: self.meta.revision,
            function: function.to_string(),
            input: input.to_string(),
            result: result.to_string(),
        });
    }

    /// Record test results in the working history.
    pub fn record_test(
        &mut self,
        function: &str,
        passed: usize,
        failed: usize,
        details: Vec<String>,
    ) {
        self.meta.history.push(HistoryEntry::Test {
            revision: self.meta.revision,
            function: function.to_string(),
            passed,
            failed,
            details,
        });
    }

    /// Record a query in the working history.
    pub fn record_query(&mut self, query: &str, response: &str) {
        self.meta.history.push(HistoryEntry::Query {
            revision: self.meta.revision,
            query: query.to_string(),
            response: response.to_string(),
        });
    }

    /// Record a trace in the working history.
    pub fn record_trace(&mut self, function: &str, steps: usize) {
        self.meta.history.push(HistoryEntry::Trace {
            revision: self.meta.revision,
            function: function.to_string(),
            steps,
        });
    }

    /// Replay mutations up to a specific revision, reconstructing program state.
    pub fn rewind_to(&mut self, target_revision: usize) -> Result<()> {
        if target_revision > self.meta.sources.len() {
            return Err(anyhow!(
                "revision {} doesn't exist (latest is {})",
                target_revision,
                self.meta.sources.len()
            ));
        }

        // Rebuild from scratch
        self.program = ast::Program::default();
        for source in &self.meta.sources[..target_revision] {
            let operations = parser::parse(source)?;
            for op in &operations {
                match op {
                    parser::Operation::Test(_)
                    | parser::Operation::Trace(_)
                    | parser::Operation::Eval(_)
                    | parser::Operation::Query(_) => {}
                    _ => {
                        let _ = validator::apply_and_validate(&mut self.program, op);
                    }
                }
            }
        }
        self.meta.revision = target_revision;
        Ok(())
    }

    /// Get recent history formatted for the LLM context.
    pub fn format_recent_history(&self, max_entries: usize) -> String {
        let mut out = String::new();
        out.push_str("=== Recent History ===\n");

        // Show last N mutations
        let start = self.meta.mutations.len().saturating_sub(max_entries);
        for entry in &self.meta.mutations[start..] {
            let status = if entry.success { "OK" } else { "ERR" };
            out.push_str(&format!(
                "[rev {}] {} — {}\n",
                entry.revision, status, entry.summary
            ));
        }

        // Show last N history entries
        let start = self.meta.history.len().saturating_sub(max_entries);
        for entry in &self.meta.history[start..] {
            match entry {
                HistoryEntry::Eval {
                    revision,
                    function,
                    result,
                    ..
                } => {
                    out.push_str(&format!("[rev {revision}] eval {function}() = {result}\n"));
                }
                HistoryEntry::Test {
                    revision,
                    function,
                    passed,
                    failed,
                    ..
                } => {
                    out.push_str(&format!(
                        "[rev {revision}] test {function}: {passed}P/{failed}F\n"
                    ));
                }
                HistoryEntry::Query {
                    revision, query, ..
                } => {
                    out.push_str(&format!("[rev {revision}] {query}\n"));
                }
                HistoryEntry::Trace {
                    revision,
                    function,
                    steps,
                } => {
                    out.push_str(&format!(
                        "[rev {revision}] trace {function}: {steps} steps\n"
                    ));
                }
                HistoryEntry::Note { revision, text } => {
                    out.push_str(&format!("[rev {revision}] note: {text}\n"));
                }
            }
        }

        out
    }

    /// Save session to a JSON file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load session from a JSON file and replay mutations.
    pub fn load(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let mut session: Session = serde_json::from_str(&json)?;

        // Rebuild function index (not serialized)
        session.program.rebuild_function_index();

        Ok(session)
    }

    fn handle_roadmap(&mut self, action: &parser::RoadmapAction) -> (String, bool) {
        let result = match action {
            parser::RoadmapAction::Show => {
                let items = self.meta.roadmap.iter().enumerate().map(|(i, item)| {
                    format!("{} {}: {}", if item.done { "[x]" } else { "[ ]" }, i + 1, item.description)
                }).collect::<Vec<_>>().join("\n");
                (if items.is_empty() { "Roadmap is empty.".to_string() } else { format!("Roadmap:\n{items}") }, true)
            }
            parser::RoadmapAction::Add(desc) => {
                self.meta.roadmap.push(RoadmapItem { description: desc.clone(), done: false });
                (format!("Roadmap: added \"{}\" (#{}).", desc, self.meta.roadmap.len()), true)
            }
            parser::RoadmapAction::Done(n) => {
                if let Some(item) = self.meta.roadmap.get_mut(n.saturating_sub(1)) {
                    item.done = true;
                    (format!("Roadmap: #{n} done."), true)
                } else { (format!("Roadmap: #{n} not found."), false) }
            }
            parser::RoadmapAction::Remove(n) => {
                let idx = n.saturating_sub(1);
                if idx < self.meta.roadmap.len() {
                    let removed = self.meta.roadmap.remove(idx);
                    (format!("Roadmap: removed \"{}\".", removed.description), true)
                } else { (format!("Roadmap: #{n} not found."), false) }
            }
        };
        // Keep runtime copy in sync so roadmap builtins see the same data.
        self.runtime.roadmap = self.meta.roadmap.clone();
        result
    }
}

/// Format a parser::Expr back into source-level syntax suitable for `+with` lines.
/// Public alias for testing.
pub fn format_expr_pub(expr: &parser::Expr) -> String {
    format_expr(expr)
}

fn format_expr(expr: &parser::Expr) -> String {
    match expr {
        parser::Expr::Int(n) => n.to_string(),
        parser::Expr::Float(f) => format!("{f}"),
        parser::Expr::Bool(b) => b.to_string(),
        parser::Expr::String(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
        parser::Expr::Ident(id) => id.clone(),
        parser::Expr::FieldAccess { base, field } => {
            format!("{}.{}", format_expr(base), field)
        }
        parser::Expr::Call { callee, args } => {
            let args_str = args.iter().map(|a| format_expr(a)).collect::<Vec<_>>().join(", ");
            format!("{}({})", format_expr(callee), args_str)
        }
        parser::Expr::Binary { op, left, right } => {
            let op_str = match op {
                parser::BinaryOp::Add => "+",
                parser::BinaryOp::Sub => "-",
                parser::BinaryOp::Mul => "*",
                parser::BinaryOp::Div => "/",
                parser::BinaryOp::Mod => "%",
                parser::BinaryOp::Gte => ">=",
                parser::BinaryOp::Lte => "<=",
                parser::BinaryOp::Eq => "==",
                parser::BinaryOp::Neq => "!=",
                parser::BinaryOp::Gt => ">",
                parser::BinaryOp::Lt => "<",
                parser::BinaryOp::And => "AND",
                parser::BinaryOp::Or => "OR",
            };
            format!("{} {} {}", format_expr(left), op_str, format_expr(right))
        }
        parser::Expr::Unary { op, expr: inner } => {
            let op_str = match op {
                parser::UnaryOp::Not => "NOT ",
                parser::UnaryOp::Neg => "-",
            };
            format!("{}{}", op_str, format_expr(inner))
        }
        parser::Expr::StructLiteral(fields) => {
            if fields.is_empty() {
                return String::new();
            }
            // Use flat key=value format only for simple top-level params
            // (no nested structs, no function calls as values). Otherwise
            // use brace struct syntax `{k: v, k: v}` which always round-trips.
            let all_simple = fields.iter().all(|f| {
                matches!(
                    &f.value,
                    parser::Expr::Int(_)
                        | parser::Expr::Float(_)
                        | parser::Expr::Bool(_)
                        | parser::Expr::String(_)
                        | parser::Expr::Ident(_)
                )
            });
            if all_simple {
                let parts: Vec<String> = fields
                    .iter()
                    .map(|f| format!("{}={}", f.name, format_expr(&f.value)))
                    .collect();
                parts.join(" ")
            } else {
                let parts: Vec<String> = fields
                    .iter()
                    .map(|f| format!("{}: {}", f.name, format_expr(&f.value)))
                    .collect();
                format!("{{{}}}", parts.join(", "))
            }
        }
        parser::Expr::Cast { expr: inner, .. } => format_expr(inner),
    }
}

/// Public wrapper for `reconstruct_expect` — used by `test_run` IO builtin in coroutine.rs.
pub fn reconstruct_expect_pub(expected: &str, matcher: Option<&str>) -> String {
    reconstruct_expect(expected, matcher)
}

/// Reconstruct the `expect` portion of a +with line from stored data.
/// If a matcher was used, emit the matcher syntax; otherwise emit the literal expected value.
fn reconstruct_expect(expected: &str, matcher: Option<&str>) -> String {
    match matcher {
        Some("AnyOk") => "Ok".to_string(),
        Some("AnyErr") => "Err".to_string(),
        Some(s) => {
            if let Some(val) = s.strip_prefix("contains:") {
                format!("contains(\"{}\")", val.replace('\\', "\\\\").replace('"', "\\\""))
            } else if let Some(val) = s.strip_prefix("starts_with:") {
                format!("starts_with(\"{}\")", val.replace('\\', "\\\\").replace('"', "\\\""))
            } else if let Some(val) = s.strip_prefix("ErrContaining:") {
                format!("Err(\"{}\")", val.replace('\\', "\\\\").replace('"', "\\\""))
            } else {
                expected.to_string()
            }
        }
        None => expected.to_string(),
    }
}

/// Serialize a TestMatcher to a string for session persistence.
fn serialize_matcher(m: &parser::TestMatcher) -> String {
    match m {
        parser::TestMatcher::Contains(s) => format!("contains:{s}"),
        parser::TestMatcher::StartsWith(s) => format!("starts_with:{s}"),
        parser::TestMatcher::AnyOk => "AnyOk".to_string(),
        parser::TestMatcher::AnyErr => "AnyErr".to_string(),
        parser::TestMatcher::ErrContaining(s) => format!("ErrContaining:{s}"),
    }
}

fn now() -> String {
    // Simple timestamp without external crate
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}s", dur.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_function(name: &str) -> ast::FunctionDecl {
        ast::FunctionDecl {
            id: "fn-1".to_string(),
            name: name.to_string(),
            params: vec![],
            return_type: ast::Type::Int,
            effects: vec![],
            body: vec![
                ast::Statement {
                    id: "s1".to_string(),
                    kind: ast::StatementKind::Let {
                        name: "x".to_string(),
                        ty: ast::Type::Int,
                        value: ast::Expr::Literal(ast::Literal::Int(1)),
                    },
                },
                ast::Statement {
                    id: "s2".to_string(),
                    kind: ast::StatementKind::Set {
                        name: "x".to_string(),
                        value: ast::Expr::Literal(ast::Literal::Int(2)),
                    },
                },
                ast::Statement {
                    id: "s3".to_string(),
                    kind: ast::StatementKind::Return {
                        value: ast::Expr::Literal(ast::Literal::Int(1)),
                    },
                },
            ],
            tests: vec![],
        }
    }

    #[test]
    fn store_test_marks_ast_tests_passed() {
        let mut session = Session::new();
        session.program.functions.push(std::sync::Arc::new(test_function("foo")));
        session.program.rebuild_function_index();

        let cases = vec![parser::TestCase {
            input: parser::Expr::StructLiteral(vec![]),
            expected: parser::Expr::Int(1),
            matcher: None,
            after_checks: vec![],
        }];

        session.store_test("foo", &cases);

        let func = session.program.get_function("foo").unwrap();
        assert_eq!(func.tests.len(), 1);
        assert!(func.tests[0].passed);
        assert!(session.is_function_tested("foo"));
    }

    #[test]
    fn invalidate_and_retest_restores_ast_test_pass_flags() {
        let mut session = Session::new();
        session.apply(
            "+fn foo ()->Int\n  +let x:Int = 1\n  +set x = 2\n  +return 1\n\n!test foo\n  +with -> expect 1\n",
        )
        .unwrap();

        let func = session.program.get_function("foo").unwrap();
        assert!(!func.tests.is_empty());
        assert!(func.tests.iter().all(|t| t.passed));

        let op = parser::Operation::Replace(parser::ReplaceMutation {
            target: "foo".to_string(),
            body: vec![
                parser::Operation::Let(parser::LetDecl {
                    name: "x".to_string(),
                    ty: parser::TypeExpr::Named("Int".to_string()),
                    expr: parser::Expr::Int(1),
                }),
                parser::Operation::Set(parser::SetDecl {
                    name: "x".to_string(),
                    expr: parser::Expr::Int(2),
                }),
                parser::Operation::Return(parser::ReturnDecl {
                    expr: parser::Expr::Int(1),
                }),
            ],
        });

        let pre_backups = session.backup_affected_bodies(&op);
        validator::apply_and_validate(&mut session.program, &op).unwrap();
        let results = session.invalidate_and_retest(&op, pre_backups);

        assert!(!results.is_empty());
        let func = session.program.get_function("foo").unwrap();
        assert!(func.tests.iter().all(|t| t.passed));
        assert!(session.is_function_tested("foo"));
    }

    // ═════════════════════════════════════════════════════════════════════
    // Architecture: SessionMeta serde roundtrip
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn session_meta_serde_roundtrip() {
        let mut meta = SessionMeta::new();
        meta.revision = 5;
        meta.roadmap.push(RoadmapItem { description: "build feature X".to_string(), done: false });
        meta.plan.push(PlanStep { description: "step 1".to_string(), status: PlanStatus::Pending });
        meta.chat_messages.push(ChatMessage { role: "user".to_string(), content: "hello".to_string() });

        let json = serde_json::to_string(&meta).expect("serialize");
        let restored: SessionMeta = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(restored.revision, 5);
        assert_eq!(restored.roadmap.len(), 1);
        assert_eq!(restored.roadmap[0].description, "build feature X");
        assert_eq!(restored.plan.len(), 1);
        assert_eq!(restored.chat_messages.len(), 1);
        // library_state is #[serde(skip)] — should be None after roundtrip
        assert!(restored.library_state.is_none());
    }

    #[test]
    fn session_serde_flatten_roundtrip() {
        let mut session = Session::new();
        session.meta.revision = 3;
        session.meta.roadmap.push(RoadmapItem { description: "test".to_string(), done: true });

        let json = serde_json::to_string(&session).expect("serialize");
        let restored: Session = serde_json::from_str(&json).expect("deserialize");

        // With #[serde(flatten)], meta fields appear at the top level in JSON
        assert_eq!(restored.meta.revision, 3);
        assert_eq!(restored.meta.roadmap.len(), 1);
        assert!(restored.meta.roadmap[0].done);
        // sandbox is #[serde(skip)] — should be None
        assert!(restored.sandbox.is_none());
    }

    #[test]
    fn session_serde_flatten_fields_at_top_level() {
        let mut session = Session::new();
        session.meta.revision = 7;
        let json = serde_json::to_string(&session).expect("serialize");
        // With flatten, "revision" should be a top-level key, not nested under "meta"
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(v.get("revision").is_some(), "revision should be top-level: {json}");
        assert!(v.get("meta").is_none(), "meta should not be a separate key: {json}");
    }

    // ═════════════════════════════════════════════════════════════════════
    // Architecture: AgentBranch fork/merge
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn agent_branch_fork_creates_independent_program() {
        let mut session = Session::new();
        session.apply("+fn greet ()->String\n  +return \"hello\"\n").unwrap();
        assert!(session.program.get_function("greet").is_some());

        let branch = AgentBranch::fork("test-agent", AgentScope::Full, "add a function", &session);
        assert!(branch.program.get_function("greet").is_some());
        assert_eq!(branch.fork_revision, session.meta.revision);
        assert!(branch.mutations.is_empty());
    }

    #[test]
    fn agent_branch_mutation_independent() {
        let mut session = Session::new();
        session.apply("+fn base ()->Int\n  +return 1\n").unwrap();

        let mut branch = AgentBranch::fork("agent", AgentScope::Full, "task", &session);
        branch.apply("+fn new_fn ()->Int\n  +return 2\n").unwrap();

        // Branch has the new function, session doesn't
        assert!(branch.program.get_function("new_fn").is_some());
        assert!(session.program.get_function("new_fn").is_none());
    }

    #[test]
    fn agent_branch_merge_applies_mutations() {
        let mut session = Session::new();
        session.apply("+fn base ()->Int\n  +return 1\n").unwrap();

        let mut branch = AgentBranch::fork("agent", AgentScope::Full, "task", &session);
        branch.apply("+fn added ()->Int\n  +return 42\n").unwrap();

        let conflicts = branch.merge_into(&mut session);
        assert!(conflicts.is_empty(), "expected no conflicts, got: {conflicts:?}");
        assert!(session.program.get_function("added").is_some());
    }

    #[test]
    fn agent_branch_fork_isolates_runtime() {
        let mut session = Session::new();
        session.runtime.shared_vars.insert("key".to_string(), crate::eval::Value::Int(10));

        let branch = AgentBranch::fork("agent", AgentScope::Full, "task", &session);
        assert!(matches!(
            branch.runtime_state.shared_vars.get("key"),
            Some(crate::eval::Value::Int(10))
        ));

        // Mutating session runtime doesn't affect branch
        session.runtime.shared_vars.insert("key".to_string(), crate::eval::Value::Int(99));
        assert!(matches!(
            branch.runtime_state.shared_vars.get("key"),
            Some(crate::eval::Value::Int(10))
        ), "branch should be unaffected by session mutation");
    }

    #[test]
    fn agent_branch_merge_propagates_shared_var_changes() {
        let mut session = Session::new();
        session.runtime.shared_vars.insert("counter".to_string(), crate::eval::Value::Int(0));

        let mut branch = AgentBranch::fork("agent", AgentScope::Full, "task", &session);
        // Modify shared var in branch
        branch.runtime_state.shared_vars.insert("counter".to_string(), crate::eval::Value::Int(5));

        let conflicts = branch.merge_into(&mut session);
        assert!(conflicts.is_empty());
        // Session should have the updated value
        assert!(matches!(
            session.runtime.shared_vars.get("counter"),
            Some(crate::eval::Value::Int(5))
        ));
    }

    #[test]
    fn agent_branch_merge_ignores_unchanged_vars() {
        let mut session = Session::new();
        session.runtime.shared_vars.insert("stable".to_string(), crate::eval::Value::Int(42));

        let branch = AgentBranch::fork("agent", AgentScope::Full, "task", &session);
        // Don't modify any shared vars

        // Change session's value before merge
        session.runtime.shared_vars.insert("stable".to_string(), crate::eval::Value::Int(100));

        let conflicts = branch.merge_into(&mut session);
        assert!(conflicts.is_empty());
        // Session value should stay at 100 (branch didn't change it)
        assert!(matches!(
            session.runtime.shared_vars.get("stable"),
            Some(crate::eval::Value::Int(100))
        ));
    }

    // ═════════════════════════════════════════════════════════════════════
    // Architecture: !sandbox enter/merge/discard
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn sandbox_enter_and_status() {
        let mut session = Session::new();
        // Not in sandbox initially
        let (msg, ok) = session.handle_sandbox(&parser::SandboxAction::Status);
        assert!(ok);
        assert!(msg.contains("not in sandbox"));

        // Enter sandbox
        let (msg, ok) = session.handle_sandbox(&parser::SandboxAction::Enter);
        assert!(ok, "enter should succeed: {msg}");
        assert!(session.sandbox.is_some());

        // Status should report active
        let (msg, ok) = session.handle_sandbox(&parser::SandboxAction::Status);
        assert!(ok);
        assert!(msg.contains("in sandbox mode"));
    }

    #[test]
    fn sandbox_double_enter_rejected() {
        let mut session = Session::new();
        session.handle_sandbox(&parser::SandboxAction::Enter);
        let (msg, ok) = session.handle_sandbox(&parser::SandboxAction::Enter);
        assert!(!ok, "double enter should fail");
        assert!(msg.contains("already in sandbox"));
    }

    #[test]
    fn sandbox_merge_keeps_changes() {
        let mut session = Session::new();
        session.apply("+fn base ()->Int\n  +return 1\n").unwrap();
        session.handle_sandbox(&parser::SandboxAction::Enter);

        // Add a function while in sandbox
        session.apply("+fn new_fn ()->Int\n  +return 42\n").unwrap();
        assert!(session.program.get_function("new_fn").is_some());

        // Merge — changes stay
        let (msg, ok) = session.handle_sandbox(&parser::SandboxAction::Merge);
        assert!(ok, "merge should succeed: {msg}");
        assert!(session.sandbox.is_none());
        assert!(session.program.get_function("new_fn").is_some(), "merged function should persist");
    }

    #[test]
    fn sandbox_discard_reverts_changes() {
        let mut session = Session::new();
        session.apply("+fn base ()->Int\n  +return 1\n").unwrap();
        let rev_before = session.meta.revision;

        session.handle_sandbox(&parser::SandboxAction::Enter);
        session.apply("+fn sandbox_fn ()->Int\n  +return 99\n").unwrap();
        assert!(session.program.get_function("sandbox_fn").is_some());

        // Discard — revert all sandbox changes
        let (msg, ok) = session.handle_sandbox(&parser::SandboxAction::Discard);
        assert!(ok, "discard should succeed: {msg}");
        assert!(session.sandbox.is_none());
        assert!(session.program.get_function("sandbox_fn").is_none(), "discarded function should be gone");
        assert_eq!(session.meta.revision, rev_before, "revision should be restored");
    }

    #[test]
    fn sandbox_merge_without_enter_fails() {
        let mut session = Session::new();
        let (_, ok) = session.handle_sandbox(&parser::SandboxAction::Merge);
        assert!(!ok, "merge without enter should fail");
    }

    #[test]
    fn sandbox_discard_without_enter_fails() {
        let mut session = Session::new();
        let (_, ok) = session.handle_sandbox(&parser::SandboxAction::Discard);
        assert!(!ok, "discard without enter should fail");
    }

    #[test]
    fn runtime_state_default_has_empty_plan() {
        let state = RuntimeState::default();
        assert!(state.plan.is_empty(), "default plan should be empty");
    }

    #[test]
    fn runtime_state_plan_field_round_trips_in_memory() {
        let mut state = RuntimeState::default();
        state.plan.push(PlanStep {
            description: "step one".to_string(),
            status: PlanStatus::Pending,
        });
        state.plan.push(PlanStep {
            description: "step two".to_string(),
            status: PlanStatus::Done,
        });
        assert_eq!(state.plan.len(), 2);
        assert_eq!(state.plan[0].description, "step one");
        assert_eq!(state.plan[1].status, PlanStatus::Done);
    }

    #[test]
    fn runtime_state_plan_skipped_in_serialization() {
        let mut state = RuntimeState::default();
        state.plan.push(PlanStep {
            description: "should not serialize".to_string(),
            status: PlanStatus::InProgress,
        });
        // Serialize — plan is #[serde(skip)] so it should not appear in JSON
        let json = serde_json::to_string(&state).expect("serialize");
        assert!(!json.contains("should not serialize"), "plan field should be skipped during serialization");
        // Deserialize — plan should default to empty
        let deserialized: RuntimeState = serde_json::from_str(&json).expect("deserialize");
        assert!(deserialized.plan.is_empty(), "deserialized plan should be empty (skipped)");
    }
}
