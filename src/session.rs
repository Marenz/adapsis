//! Session management: mutation log, working history, save/load, revision control.
//!
//! Every change to the program is recorded as a numbered entry in the mutation log.
//! Evals, queries, and test results are recorded in the working history.
//! The program state can be reconstructed by replaying mutations 0..N.

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
    #[serde(skip)]
    pub program: ast::Program,
    /// Append-only mutation log
    pub mutations: Vec<MutationEntry>,
    /// Working history (evals, tests, queries)
    pub history: Vec<HistoryEntry>,
    /// Current revision number (= number of successful mutations)
    pub revision: usize,
    /// All raw mutation sources (for replay)
    pub sources: Vec<String>,
}

impl Session {
    pub fn new() -> Self {
        Self {
            program: ast::Program::default(),
            mutations: Vec::new(),
            history: Vec::new(),
            revision: 0,
            sources: Vec::new(),
        }
    }

    /// Apply a block of Forge source code as a mutation.
    /// Returns (results, new_revision) on success.
    pub fn apply(&mut self, source: &str) -> Result<Vec<(String, bool)>> {
        let operations = parser::parse(source)?;
        let mut results = Vec::new();
        let mut any_definition = false;

        for op in &operations {
            match op {
                parser::Operation::Test(_)
                | parser::Operation::Trace(_)
                | parser::Operation::Eval(_)
                | parser::Operation::Query(_) => {
                    // These don't modify program state — handled separately
                }
                _ => {
                    any_definition = true;
                    match validator::apply_and_validate(&mut self.program, op) {
                        Ok(msg) => results.push((msg, true)),
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

        // Replay all mutations to reconstruct program state
        session.program = ast::Program::default();
        for source in &session.sources {
            let operations = parser::parse(source)?;
            for op in &operations {
                match op {
                    parser::Operation::Test(_)
                    | parser::Operation::Trace(_)
                    | parser::Operation::Eval(_)
                    | parser::Operation::Query(_) => {}
                    _ => {
                        let _ = validator::apply_and_validate(&mut session.program, op);
                    }
                }
            }
        }

        Ok(session)
    }
}

fn now() -> String {
    // Simple timestamp without external crate
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}s", dur.as_secs())
}
