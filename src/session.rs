//! Session management: mutation log, working history, save/load, revision control.
//!
//! Every change to the program is recorded as a numbered entry in the mutation log.
//! Evals, queries, and test results are recorded in the working history.
//! The program state can be reconstructed by replaying mutations 0..N.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

use crate::ast;
use crate::parser;
use crate::validator;

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

/// A Forge session — program state + mutation log + working history.
#[derive(Debug, Serialize, Deserialize)]
pub struct Session {
    /// Current program state
    pub program: ast::Program,
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
    /// Functions that have been tested (passed at least one test). Eval/spawn blocked until tested.
    #[serde(default)]
    pub tested_functions: std::collections::HashSet<String>,
    /// Stored test cases — re-run automatically when functions change.
    /// Key is the function name; value is the list of stored cases.
    #[serde(default)]
    pub stored_tests: HashMap<String, Vec<StoredTestCase>>,
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

/// A stored test case — input and expected as source-level strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredTestCase {
    pub input: String,
    pub expected: String,
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
}

impl AgentBranch {
    /// Create a new branch forked from the current session state.
    pub fn fork(name: &str, scope: AgentScope, task: &str, session: &Session) -> Self {
        Self {
            name: name.to_string(),
            scope,
            task: task.to_string(),
            fork_revision: session.revision,
            program: session.program.clone(),
            mutations: Vec::new(),
        }
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
            mutations: Vec::new(),
            chat_messages: Vec::new(),
            agent_log: Vec::new(),
            plan: Vec::new(),
            history: Vec::new(),
            revision: 0,
            sources: Vec::new(),
            agent_mailbox: HashMap::new(),
            tested_functions: std::collections::HashSet::new(),
            stored_tests: HashMap::new(),
            opencode_session_id: None,
            io_mocks: Vec::new(),
            roadmap: Vec::new(),
            library_state: None,
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
        self.agent_mailbox
            .entry(to.to_string())
            .or_default()
            .push(msg);
    }

    /// Drain all pending messages for an agent.
    pub fn drain_messages(&mut self, agent_name: &str) -> Vec<AgentMessage> {
        self.agent_mailbox.remove(agent_name).unwrap_or_default()
    }

    /// Peek at pending messages without removing them.
    pub fn peek_messages(&self, agent_name: &str) -> &[AgentMessage] {
        self.agent_mailbox
            .get(agent_name)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Mark a function as tested, resolving bare names to qualified module names.
    pub fn mark_tested(&mut self, fn_name: &str) {
        self.tested_functions.insert(fn_name.to_string());
        if !fn_name.contains('.') {
            let qnames: Vec<String> = self.program.modules.iter()
                .flat_map(|m| m.functions.iter()
                    .filter(|f| f.name == fn_name)
                    .map(|f| format!("{}.{}", m.name, f.name)))
                .collect();
            for qn in qnames {
                self.tested_functions.insert(qn);
            }
        }
    }

    /// Store test cases for a function, keyed by both bare and qualified name.
    pub fn store_test(&mut self, fn_name: &str, cases: &[parser::TestCase]) {
        let stored: Vec<StoredTestCase> = cases
            .iter()
            .map(|c| StoredTestCase {
                input: format_expr(&c.input),
                expected: format_expr(&c.expected),
            })
            .collect();

        self.stored_tests
            .insert(fn_name.to_string(), stored.clone());

        // Also store under qualified name(s) if bare
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
            for qn in qnames {
                self.stored_tests.insert(qn, stored.clone());
            }
        }
    }

    /// Invalidate test status for affected functions, then re-run any stored
    /// tests. Returns a list of (fn_name, passed, detail) for each re-run.
    /// Must be called AFTER `apply_and_validate` so the function is already
    /// updated in the program.
    fn invalidate_and_retest(
        &mut self,
        op: &parser::Operation,
    ) -> Vec<(String, bool, String)> {
        // Collect affected function names
        let affected: Vec<String> = match op {
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
            parser::Operation::Replace(r) => vec![r.target.clone()],
            _ => return Vec::new(),
        };

        // Invalidate
        for name in &affected {
            self.tested_functions.remove(name);
        }

        let mut retest_results = Vec::new();

        for name in &affected {
            // Find stored tests: try exact name, then bare name
            let cases = self.stored_tests.get(name).cloned().or_else(|| {
                let bare = name.rsplit('.').next().unwrap_or(name);
                self.stored_tests.get(bare).cloned()
            });

            let cases = match cases {
                Some(c) if !c.is_empty() => c,
                _ => continue,
            };

            // Reconstruct test source and re-run
            let bare = name.rsplit('.').next().unwrap_or(name);
            let mut test_src = format!("!test {bare}\n");
            for case in &cases {
                test_src.push_str(&format!("  +with {} -> expect {}\n", case.input, case.expected));
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
                                    &self.io_mocks,
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
                                self.mark_tested(name);
                            }
                        }
                    }
                }
                Err(e) => {
                    retest_results.push((
                        name.clone(),
                        false,
                        format!("retest parse error: {e}"),
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
                            &self.io_mocks,
                        ) {
                            Ok(msg) => results.push((format!("PASS: {msg}"), true)),
                            Err(e) => {
                                all_passed = false;
                                results.push((format!("FAIL: {e}"), false));
                            }
                        }
                    }
                    if all_passed && !test.cases.is_empty() {
                        self.mark_tested(&test.function_name);
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
                        self.plan = steps
                            .iter()
                            .map(|s| PlanStep {
                                description: s.clone(),
                                status: PlanStatus::Pending,
                            })
                            .collect();
                        results.push((format!("Plan: {} steps", steps.len()), true));
                    }
                    parser::PlanAction::Progress(n) => {
                        if let Some(step) = self.plan.get_mut(n.saturating_sub(1)) {
                            step.status = PlanStatus::Done;
                            results.push((format!("Step {n} done"), true));
                        }
                    }
                    parser::PlanAction::Fail(n) => {
                        if let Some(step) = self.plan.get_mut(n.saturating_sub(1)) {
                            step.status = PlanStatus::Failed;
                            results.push((format!("Step {n} failed"), true));
                        }
                    }
                    parser::PlanAction::Show => {
                        let plan_str = self
                            .plan
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
                    self.io_mocks.push(IoMock {
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
                    let count = self.io_mocks.len();
                    self.io_mocks.clear();
                    results.push((format!("cleared {count} mocks"), true));
                }
                _ => {
                    any_definition = true;
                    match validator::apply_and_validate(&mut self.program, op) {
                        Ok(msg) => {
                            results.push((msg, true));
                            for (name, passed, detail) in self.invalidate_and_retest(op) {
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
            self.revision += 1;
            self.mutations.push(MutationEntry {
                revision: self.revision,
                timestamp: now(),
                source: source.to_string(),
                summary,
                success,
            });
            self.sources.push(source.to_string());

            // Persist affected modules to the library
            let affected = crate::library::affected_module_names(&operations);
            eprintln!("[library] apply: any_definition={any_definition} success={success} affected={affected:?} lib_state={}", self.library_state.is_some());
            if success && !affected.is_empty() {
                crate::library::persist_affected_modules(
                    &self.program,
                    &affected,
                    self.library_state.as_ref(),
                );
            }
        }

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
                                    &self.io_mocks,
                                    sender.clone(),
                                )
                                .await
                            } else {
                                crate::eval::eval_test_case_with_mocks(
                                    &self.program,
                                    &test.function_name,
                                    case,
                                    &self.io_mocks,
                                )
                            }
                        } else {
                            crate::eval::eval_test_case_with_mocks(
                                &self.program,
                                &test.function_name,
                                case,
                                &self.io_mocks,
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
                        self.mark_tested(&test.function_name);
                        self.store_test(&test.function_name, &test.cases);
                    }
                }
                parser::Operation::Trace(_)
                | parser::Operation::Eval(_)
                | parser::Operation::Query(_) => {}
                parser::Operation::Plan(action) => match action {
                    parser::PlanAction::Set(steps) => {
                        self.plan = steps
                            .iter()
                            .map(|s| PlanStep {
                                description: s.clone(),
                                status: PlanStatus::Pending,
                            })
                            .collect();
                        results.push((format!("Plan: {} steps", steps.len()), true));
                    }
                    parser::PlanAction::Progress(n) => {
                        if let Some(step) = self.plan.get_mut(n.saturating_sub(1)) {
                            step.status = PlanStatus::Done;
                            results.push((format!("Step {n} done"), true));
                        }
                    }
                    parser::PlanAction::Fail(n) => {
                        if let Some(step) = self.plan.get_mut(n.saturating_sub(1)) {
                            step.status = PlanStatus::Failed;
                            results.push((format!("Step {n} failed"), true));
                        }
                    }
                    parser::PlanAction::Show => {
                        let plan_str = self
                            .plan
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
                    self.io_mocks.push(IoMock {
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
                    let count = self.io_mocks.len();
                    self.io_mocks.clear();
                    results.push((format!("cleared {count} mocks"), true));
                }
                _ => {
                    any_definition = true;
                    match validator::apply_and_validate(&mut self.program, op) {
                        Ok(msg) => {
                            results.push((msg, true));
                            for (name, passed, detail) in self.invalidate_and_retest(op) {
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
            self.revision += 1;
            self.mutations.push(MutationEntry {
                revision: self.revision,
                timestamp: now(),
                source: source.to_string(),
                summary,
                success,
            });
            self.sources.push(source.to_string());

            // Persist affected modules to the library
            let affected = crate::library::affected_module_names(&operations);
            eprintln!("[library] apply_async: any_definition={any_definition} success={success} affected={affected:?} lib_state={}", self.library_state.is_some());
            if success && !affected.is_empty() {
                crate::library::persist_affected_modules(
                    &self.program,
                    &affected,
                    self.library_state.as_ref(),
                );
            }
        }

        Ok(results)
    }


    /// Get the parsed operations from a source string (for test/eval/query handling).
    pub fn parse_operations(&self, source: &str) -> Result<Vec<parser::Operation>> {
        parser::parse(source)
    }

    /// Record an eval in the working history.
    pub fn record_eval(&mut self, function: &str, input: &str, result: &str) {
        self.history.push(HistoryEntry::Eval {
            revision: self.revision,
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
        self.history.push(HistoryEntry::Test {
            revision: self.revision,
            function: function.to_string(),
            passed,
            failed,
            details,
        });
    }

    /// Record a query in the working history.
    pub fn record_query(&mut self, query: &str, response: &str) {
        self.history.push(HistoryEntry::Query {
            revision: self.revision,
            query: query.to_string(),
            response: response.to_string(),
        });
    }

    /// Record a trace in the working history.
    pub fn record_trace(&mut self, function: &str, steps: usize) {
        self.history.push(HistoryEntry::Trace {
            revision: self.revision,
            function: function.to_string(),
            steps,
        });
    }

    /// Replay mutations up to a specific revision, reconstructing program state.
    pub fn rewind_to(&mut self, target_revision: usize) -> Result<()> {
        if target_revision > self.sources.len() {
            return Err(anyhow!(
                "revision {} doesn't exist (latest is {})",
                target_revision,
                self.sources.len()
            ));
        }

        // Rebuild from scratch
        self.program = ast::Program::default();
        for source in &self.sources[..target_revision] {
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
        self.revision = target_revision;
        Ok(())
    }

    /// Get recent history formatted for the LLM context.
    pub fn format_recent_history(&self, max_entries: usize) -> String {
        let mut out = String::new();
        out.push_str("=== Recent History ===\n");

        // Show last N mutations
        let start = self.mutations.len().saturating_sub(max_entries);
        for entry in &self.mutations[start..] {
            let status = if entry.success { "OK" } else { "ERR" };
            out.push_str(&format!(
                "[rev {}] {} — {}\n",
                entry.revision, status, entry.summary
            ));
        }

        // Show last N history entries
        let start = self.history.len().saturating_sub(max_entries);
        for entry in &self.history[start..] {
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
        match action {
            parser::RoadmapAction::Show => {
                let items = self.roadmap.iter().enumerate().map(|(i, item)| {
                    format!("{} {}: {}", if item.done { "[x]" } else { "[ ]" }, i + 1, item.description)
                }).collect::<Vec<_>>().join("\n");
                (if items.is_empty() { "Roadmap is empty.".to_string() } else { format!("Roadmap:\n{items}") }, true)
            }
            parser::RoadmapAction::Add(desc) => {
                self.roadmap.push(RoadmapItem { description: desc.clone(), done: false });
                (format!("Roadmap: added \"{}\" (#{}).", desc, self.roadmap.len()), true)
            }
            parser::RoadmapAction::Done(n) => {
                if let Some(item) = self.roadmap.get_mut(n.saturating_sub(1)) {
                    item.done = true;
                    (format!("Roadmap: #{n} done."), true)
                } else { (format!("Roadmap: #{n} not found."), false) }
            }
            parser::RoadmapAction::Remove(n) => {
                let idx = n.saturating_sub(1);
                if idx < self.roadmap.len() {
                    let removed = self.roadmap.remove(idx);
                    (format!("Roadmap: removed \"{}\".", removed.description), true)
                } else { (format!("Roadmap: #{n} not found."), false) }
            }
        }
    }
}

/// Format a parser::Expr back into source-level syntax suitable for `+with` lines.
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
            let parts: Vec<String> = fields
                .iter()
                .map(|f| format!("{}={}", f.name, format_expr(&f.value)))
                .collect();
            parts.join(" ")
        }
        parser::Expr::Cast { expr: inner, .. } => format_expr(inner),
    }
}

fn now() -> String {
    // Simple timestamp without external crate
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}s", dur.as_secs())
}

