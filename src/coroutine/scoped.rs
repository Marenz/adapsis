//! Turn-bound guest tools. No caller-supplied authority, callback or destination.
use super::*;
use anyhow::ensure;

pub const GUEST_TOOLS: &[&str] = &[
    "music_generate", "web_search", "web_read", "memory_remember", "memory_forget",
    "context_propose",
];

#[derive(Debug)]
pub enum Request {
    History { principal: String, context: String, limit: i64, before_ms: i64 },
    Music { context: String, caption: String, duration: i64, lyrics: String },
    Search { query: String },
    Read { url: String },
    Forget { principal: String, context: String, memory_id: String },
}

impl CoroutineHandle {
    pub(super) fn check_guest_tool(&self, op: &str, args: &[Value]) -> Result<()> {
        if let Some(tools) = self.turn.as_ref().and_then(|t| t.guest_tools.as_ref()) {
            ensure!(GUEST_TOOLS.contains(&op) && tools.iter().any(|t| t == op),
                "permission denied: {op} is not available in this guest conversation");
            if op == "memory_remember" {
                ensure!(args.get(1).is_none_or(|s| matches!(s, Value::String(v) if v.as_str() == "context" || v.is_empty())),
                    "guest memories must use context scope");
            }
        }
        Ok(())
    }

    pub(super) fn execute_scoped(&self, op: &str, args: &[Value]) -> Result<Option<Value>> {
        let request = match op {
            "conversation_history" => {
                let turn = self.require_turn(op)?;
                ensure!(crate::memory_graph::is_private_admin(&turn.context, &turn.principal),
                    "conversation_history is available only in the administrator's private Telegram chat");
                ensure!((1..=3).contains(&args.len()), "conversation_history(context[, limit, before_ms]) expects 1–3 arguments");
                let limit = optional_int(args, 1, 30)?;
                ensure!((1..=100).contains(&limit), "history limit must be between 1 and 100");
                let before_ms = optional_int(args, 2, i64::MAX)?;
                ensure!(before_ms > 0, "before_ms must be a positive Unix timestamp in milliseconds");
                Request::History { principal: turn.principal.clone(), context: text(args, 0, 200)?, limit, before_ms }
            }
            "music_generate" => {
                let turn = self.require_turn(op)?;
                ensure!((2..=3).contains(&args.len()), "music_generate(description, duration_seconds[, lyrics]) expects 2–3 arguments");
                let duration = optional_int(args, 1, 30)?;
                ensure!((10..=120).contains(&duration), "music duration must be between 10 and 120 seconds");
                Request::Music {
                    context: turn.context.clone(), caption: text(args, 0, 2000)?, duration,
                    lyrics: if args.len() == 3 { text_allow_empty(args, 2, 8000)? } else { String::new() },
                }
            }
            "web_search" => {
                self.require_turn(op)?;
                ensure!(args.len() == 1, "web_search(query) expects one argument");
                Request::Search { query: text(args, 0, 500)? }
            }
            "web_read" => {
                self.require_turn(op)?;
                ensure!(args.len() == 1, "web_read(url) expects one argument");
                Request::Read { url: text(args, 0, 4096)? }
            }
            "memory_forget" if self.turn.as_ref().is_some_and(|t| t.guest_tools.is_some()) => {
                let turn = self.require_turn(op)?;
                ensure!(args.len() == 1, "memory_forget(memory_id) expects one argument");
                Request::Forget {
                    principal: turn.principal.clone(), context: turn.context.clone(), memory_id: text(args, 0, 200)?,
                }
            }
            _ => return Ok(None),
        };
        let (tx, rx) = oneshot::channel();
        let result = self.send_and_wait(WaitReason::Running, IoRequest::Scoped { request, reply: tx }, rx)?;
        Ok(Some(Value::string(result)))
    }
}

fn text_allow_empty(args: &[Value], index: usize, max: usize) -> Result<String> {
    match args.get(index) {
        Some(Value::String(s)) if s.chars().count() <= max => Ok(s.as_ref().clone()),
        _ => bail!("argument {} must be a String of at most {max} characters", index + 1),
    }
}

fn text(args: &[Value], index: usize, max: usize) -> Result<String> {
    let value = text_allow_empty(args, index, max)?;
    ensure!(!value.trim().is_empty(), "argument {} must not be empty", index + 1);
    Ok(value)
}

fn optional_int(args: &[Value], index: usize, default: i64) -> Result<i64> {
    match args.get(index) {
        Some(Value::Int(value)) => Ok(*value),
        None => Ok(default),
        _ => bail!("argument {} must be an Int", index + 1),
    }
}

/// Check the entire block before any mutation, test, query or eval can run.
/// Guest calls deliberately accept literal arguments only: nested calls could
/// perform IO before the outer allowed tool checks its parameters.
pub fn validate_guest_code(code: &str, tools: &[String], program: &crate::ast::Program) -> Result<()> {
    for op in crate::parser::parse(code)? {
        match op {
            crate::parser::Operation::Done => {},
            crate::parser::Operation::Eval(ev) => {
                let Some(crate::parser::Expr::Call { callee, args }) = ev.inline_expr else {
                    bail!("guest tools require !eval tool_name(literal_arguments)");
                };
                let crate::parser::Expr::Ident(name) = *callee else {
                    bail!("guest tools must be native tool names, not module functions");
                };
                ensure!(GUEST_TOOLS.contains(&name.as_str()) && tools.contains(&name)
                    && program.get_function(&name).is_none(),
                    "permission denied: {name} is not an available guest tool");
                ensure!(args.iter().all(|a| matches!(a, crate::parser::Expr::String(_) | crate::parser::Expr::Int(_) | crate::parser::Expr::Bool(_) | crate::parser::Expr::Float(_))),
                    "guest tool arguments must be literal strings or numbers, not nested expressions");
            }
            _ => bail!("guest conversations may only call their listed tools with !eval or finish with !done"),
        }
    }
    Ok(())
}

pub fn prompt(tools: &[String]) -> String {
    let mut out = "Guest tools: use !eval tool_name(literal_arguments) in a <code> block. No module calls, nested expressions, shell, raw HTTP, local files, code changes or agents.\n".to_string();
    for tool in tools {
        if let Some(b) = crate::builtins::IO_BUILTINS.iter().find(|b| b.name == tool) {
            out.push_str(&format!("- {}: {}\n", b.short, b.long));
        }
    }
    out
}
