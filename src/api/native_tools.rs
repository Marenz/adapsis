//! Native JSON-schema tool dispatch over the existing permission-checked
//! runtime (issue #78).
//!
//! This module is the **only** entry point through which an LLM-facing
//! "native tool" may act on the program. Every tool is re-derived down to a
//! small, auditable set of Adapsis operations (`+module`, `+test`, `!eval`,
//! `!done`, and read-only inspection) and pushed through the *same* guarded
//! executor — [`execute_code_with_baseline`] — used by conversational turns, so
//! mutation semantics, the test gate, per-function CoW write-back and the
//! immutable completion baseline stay authoritative.
//!
//! # Contract with the caller
//!
//! [`execute_native_tool`] takes a caller-prepared [`WorkingSet`] snapshot and
//! an immutable [`Program`] baseline. It performs **no** top-level snapshot or
//! write-back itself: the caller must snapshot before *and* write the working
//! set back after **every** tool call (including `test`, whose persistence
//! mutates the AST via `store_test` but is not written back by the executor).
//! Batch callers must preflight **every** tool's arguments with
//! [`validate_native_tool`] (or [`validate_native_tool_authorized`]) before
//! executing any of them, so a malformed later tool cannot partially apply an
//! earlier valid one.
//!
//! # Permission posture
//!
//! The native path re-checks target-module permission itself and never inherits
//! an absent default. Legacy gaps in the *legacy* `?query`/`/api/*` paths are
//! out of scope here (#79) — this module only guarantees the native path does
//! not silently broaden them.
//!
//! * Eligibility (`native_tools_eligible`): non-guest, write-capable turns only.
//! * `inspect` `source`/`deps` → target module **Read**.
//! * `evaluate` → target module **Execute**.
//! * `test` → target module **Write** (test persistence mutates the AST).
//! * `mutate` → **Write** on every mentioned module (nested included).
//! * `finish` → `!done` against the *same* immutable baseline.
//!
//! Authority is read **only** from the caller-pinned config (`llm_model`) and
//! `turn` identity (`may_write`, `guest_tools`) — never from the live
//! conversation's mutable `permission_model`. The integrator's per-turn
//! permission snapshot is the single authority; this module consumes it, it
//! does not re-derive it.

use super::execute::{execute_code_with_baseline, CodeExecutionResult};
use super::{AppConfig, MutationResult, WorkingSet};
use crate::ast::Program;
use crate::permissions::PermissionLevel;
use crate::parser;
mod authored;
mod frontend;
mod literals;

// ── Bounds ─────────────────────────────────────────────────────────────────

/// Maximum accepted length (bytes) of a raw `arguments` JSON payload.
const MAX_ARGUMENTS_BYTES: usize = 64_000;
/// Maximum accepted length (bytes) of a `source` string (mutate/test) or `input`.
const MAX_SOURCE_BYTES: usize = 32_000;
/// Maximum accepted length (bytes) of a `target`/`function`/`kind` string.
const MAX_TARGET_BYTES: usize = 512;
/// Maximum length (chars) of a tool's primary `output` text.
const MAX_OUTPUT_CHARS: usize = 8_000;
/// Maximum number of mutation/test result entries serialized into the envelope.
const MAX_RESULT_MESSAGES: usize = 100;
/// Maximum length (chars) of a single result message inside the envelope.
const MAX_MESSAGE_CHARS: usize = 1_000;

// ── Public API types ───────────────────────────────────────────────────────

/// A JSON-schema tool definition exposed to the model.
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema object describing the tool's arguments.
    pub parameters: serde_json::Value,
}

/// Result of one native tool call: the raw executor result plus a bounded,
/// structured JSON envelope for the model/driver.
pub struct NativeToolResult {
    /// Underlying executor result (mutations, tests, done acceptance).
    pub execution: CodeExecutionResult,
    /// Single-line JSON envelope with `tool`, `success`, `has_errors`,
    /// `done_accepted`, `mutation_results`, `test_results`, `query_results`,
    /// `eval_results` and `output` (null where not applicable). Bounded.
    pub content: String,
}

// ── Eligibility ────────────────────────────────────────────────────────────

fn is_guest(config: &AppConfig) -> bool {
    config.permission_config.guest_profiles.contains_key(&model_name(config)) || config
        .turn
        .as_ref()
        .and_then(|t| t.guest_tools.as_ref())
        .is_some()
}

/// The pinned authority snapshot. `config.llm_model` is set by the caller
/// (integrator's per-turn `permission_snapshot`) to the intersected active
/// model; the native path reads it directly and applies **no** further model
/// re-derivation (no `principals`, no conversation `permission_model`) so it
/// cannot widen or drift from the single pinned authority (#79).
fn model_name(config: &AppConfig) -> String {
    config.llm_model.read().unwrap().clone()
}

fn resolve_module(config: &AppConfig, model: &str, module: &str) -> PermissionLevel {
    config
        .permission_config
        .resolve(config.access_level, model, module)
}

/// The first `.`-separated segment of a qualified name, used as the module for
/// permission resolution (matches the legacy `?source` check).
fn module_of(target: &str) -> &str {
    target.split('.').next().unwrap_or(target)
}

/// Whether this turn may use native tools at all: non-guest and write-capable.
///
/// Write-capability is resolved against the pinned authority snapshot
/// (`config.llm_model`, the caller's intersected active model) and further
/// narrowed by the turn identity (`turn.may_write`, `turn.guest_tools`). The
/// caller pins the single authority; this only ever narrows it, never widens.
pub fn native_tools_eligible(config: &AppConfig) -> bool {
    if is_guest(config) {
        return false;
    }
    if let Some(turn) = config.turn.as_ref() {
        if !turn.may_write {
            return false;
        }
    }
    let model = model_name(config);
    config
        .permission_config
        .authority(config.access_level, &model)
        .may_write
}

// ── Tool definitions ───────────────────────────────────────────────────────

/// The static set of native tool schemas. Pure metadata — exposure gating is the
/// caller's responsibility via [`native_tools_eligible`] (dispatch still rejects
/// guests regardless of what is exposed).
pub fn tool_definitions(_config: &AppConfig) -> Vec<ToolDefinition> {
    let obj = |required: &[&str], props: serde_json::Value| {
        serde_json::json!({
            "type": "object",
            "properties": props,
            "required": required,
            "additionalProperties": false,
        })
    };

    vec![
        ToolDefinition {
            name: "inspect".into(),
            description: "Read program metadata: symbols, source, dependencies, tasks, or the module library.".into(),
            parameters: obj(
                &["kind"],
                serde_json::json!({
                    "kind": {"type": "string", "enum": ["symbols", "source", "deps", "tasks", "library"]},
                    "target": {"type": "string", "description": "Qualified Module.member for source/deps; module name for symbols. Bare source/deps targets are rejected."},
                    "format": {"type": "string", "enum": ["adapsis", "python"], "description": "Source inspection only. Explicit python prints the supported subset; default adapsis preserves existing calls."},
                }),
            ),
        },
        ToolDefinition {
            name: "mutate".into(),
            description: "Declare or replace functions using Python with explicit module metadata, or legacy Adapsis declarations.".into(),
            parameters: obj(
                &["source"],
                serde_json::json!({
                    "source": {"type": "string", "description": "Typed Python-subset source or Adapsis +module source. Python user async calls require annotated await or explicit spawn; only a direct unshadowed IO builtin match has a no-await exception (io+async caller), admitted syntax that is not VM-portable after earlier IO in that function (#94). Prefer `response: Result[str] = await http_get(url)` then match response; never replay the function. Native bare union constructors are limited to same submission/target module; bound-field access is data, not a call reference. A Python function reusing another module's variant name is refused (tier resolution differs); pre-existing legacy collisions do not block unrelated functions. Shared declarations need an explicit annotation plus a data-only type-checked default. Native lifecycle/source/route declarations remain unsupported."},
                    "language": {"type": "string", "enum": ["adapsis", "python"], "description": "Choose python for the Python frontend. Omission preserves adapsis."},
                    "module": {"type": "string", "description": "Required with python: one module identifier. Forbidden with adapsis."}
                }),
            ),
        },
        ToolDefinition {
            name: "test".into(),
            description: "Run literal pytest-shaped Python assertions or Adapsis +test blocks; passing tests are persisted.".into(),
            parameters: obj(
                &["source"],
                serde_json::json!({
                    "source": {"type": "string", "description": "Python def test_* wrappers with literal assertions, or Adapsis +test source."},
                    "language": {"type": "string", "enum": ["adapsis", "python"], "description": "Default adapsis; explicit python uses Python assertion syntax."},
                    "module": {"type": "string", "description": "Required for Python local-name resolution; forbidden with adapsis."}
                }),
            ),
        },
        ToolDefinition {
            name: "evaluate".into(),
            description: "Evaluate a qualified function with an optional input expression.".into(),
            parameters: obj(
                &["function"],
                serde_json::json!({
                    "function": {"type": "string", "description": "Qualified Module.func name."},
                    "input": {"type": "string", "description": "Optional literal input or named literal fields. Function calls and variable references are rejected."},
                    "language": {"type": "string", "enum": ["adapsis", "python"], "description": "Literal input syntax, default adapsis; python accepts True/False and list literals."},
                }),
            ),
        },
        ToolDefinition {
            name: "finish".into(),
            description: "Signal task completion (!done) against the current turn's baseline.".into(),
            parameters: obj(&[], serde_json::json!({})),
        },
    ]
}

// ── Argument parsing (pure, side-effect free) ──────────────────────────────

enum InspectKind {
    Symbols,
    Source,
    Deps,
    Tasks,
    Library,
}

enum ParsedTool {
    Inspect { kind: InspectKind, target: Option<String>, format: frontend::SourceFormat },
    Mutate { source: String, modules: Vec<String>, python: bool, roles: Option<crate::python_frontend::NativeReferences> },
    PythonMutate { source: String, module: String, preflight: crate::python_frontend::NativePreflight },
    Test { source: String, targets: Vec<String>, python: bool },
    Evaluate { function: String, source: String, python: bool },
    Finish,
}

fn parse_args_object(arguments: &str) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    if arguments.len() > MAX_ARGUMENTS_BYTES {
        return Err(format!("arguments exceed {MAX_ARGUMENTS_BYTES} bytes"));
    }
    let value: serde_json::Value = serde_json::from_str(arguments)
        .map_err(|e| format!("arguments are not valid JSON: {e}"))?;
    match value {
        serde_json::Value::Object(map) => Ok(map),
        _ => Err("arguments must be a JSON object".to_string()),
    }
}

fn reject_extra_fields(
    map: &serde_json::Map<String, serde_json::Value>,
    allowed: &[&str],
) -> Result<(), String> {
    for key in map.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("unexpected field `{key}`"));
        }
    }
    Ok(())
}

fn take_string(
    map: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    required: bool,
    cap: usize,
) -> Result<Option<String>, String> {
    match map.get(key) {
        Some(serde_json::Value::String(s)) => {
            if s.len() > cap {
                return Err(format!("field `{key}` exceeds {cap} bytes"));
            }
            Ok(Some(s.clone()))
        }
        Some(_) => Err(format!("field `{key}` must be a string")),
        None if required => Err(format!("missing required field `{key}`")),
        None => Ok(None),
    }
}

fn is_identifier(segment: &str) -> bool {
    let mut chars = segment.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn is_qualified_function_name(name: &str) -> bool {
    name.contains('.') && name.split('.').all(is_identifier)
}

fn describe_op(op: &parser::Operation) -> &'static str {
    use parser::Operation::*;
    match op {
        Module(_) => "+module",
        Type(_) => "+type",
        Function(_) => "+fn",
        Let(_) => "+let",
        Set(_) => "+set",
        Call(_) => "+call",
        Check(_) => "+check",
        Branch(_) => "+branch",
        If(_) => "+if",
        Return(_) => "+return",
        Each(_) => "+each",
        While(_) => "+while",
        Match(_) => "+match",
        Await(_) => "+await",
        Spawn(_) => "+spawn",
        Replace(_) => "!replace",
        Test(_) => "+test",
        Trace(_) => "!trace",
        Eval(_) => "!eval",
        Move { .. } => "!move",
        Plan(_) => "!plan",
        Roadmap(_) => "!roadmap",
        Remove(_) => "!remove",
        RemoveRoute { .. } => "!remove route",
        Watch { .. } => "!watch",
        Undo => "!undo",
        Agent { .. } => "!agent",
        OpenCode(_) => "!opencode",
        Route { .. } => "+route",
        Message { .. } => "!msg",
        Mock { .. } => "!mock",
        Unmock => "!unmock",
        Stub { .. } => "!stub",
        Unstub => "!unstub",
        Done => "!done",
        Doc { .. } => "+doc",
        Query(_) => "query",
        SharedVar(_) => "+shared",
        Sandbox(_) => "!sandbox",
        Startup(_) => "+startup",
        Shutdown(_) => "+shutdown",
        SourceAdd(_) => "+source add",
        SourceRemove(_) => "+source remove",
        SourceReplace(_) => "+source replace",
        SourceList => "+source list",
        ModuleSource(_) => "+source",
        EventRegister(_) => "+event register",
        EventEmit(_) => "+event emit",
    }
}

fn collect_module_names(decl: &parser::ModuleDecl, out: &mut Vec<String>) {
    out.push(decl.name.clone());
    for op in &decl.body {
        if let parser::Operation::Module(nested) = op {
            collect_module_names(nested, out);
        }
    }
}

/// Validate a mutate source: only `+module` declarations whose bodies contain
/// `+fn`/`+type`/`+doc`/`+shared` (and nested `+module`). Functions are treated
/// as opaque data — declared `[io]`/`[async]` effects are permitted. Every other
/// operation kind (commands, queries, evals, tests, agents, lifecycle blocks,
/// source/event declarations, top-level statements, …) is rejected.
fn validate_mutate_ops(ops: &[parser::Operation]) -> Result<(), String> {
    if ops.is_empty() {
        return Err("mutate `source` is empty".to_string());
    }
    for op in ops {
        match op {
            parser::Operation::Module(decl) => validate_module_decl(decl)?,
            other => {
                return Err(format!(
                    "mutate forbids top-level `{}` (only +module declarations are allowed)",
                    describe_op(other)
                ))
            }
        }
    }
    Ok(())
}

fn validate_module_decl(decl: &parser::ModuleDecl) -> Result<(), String> {
    for op in &decl.body {
        match op {
            parser::Operation::Function(_)
            | parser::Operation::Type(_)
            | parser::Operation::Doc { .. }
            | parser::Operation::SharedVar(_) => {}
            parser::Operation::Module(nested) => validate_module_decl(nested)?,
            other => {
                return Err(format!(
                    "mutate forbids `{}` inside module `{}`",
                    describe_op(other),
                    decl.name
                ))
            }
        }
    }
    Ok(())
}

/// Validate a test source: only non-empty `+test` blocks for qualified function
/// names. No modules, no other operation kinds.
fn validate_test_ops(ops: &[parser::Operation]) -> Result<(), String> {
    if ops.is_empty() {
        return Err("test `source` is empty".to_string());
    }
    for op in ops {
        match op {
            parser::Operation::Test(t) => {
                if t.function_name.is_empty() {
                    return Err("test block has an empty function name".to_string());
                }
                if !is_qualified_function_name(&t.function_name) {
                    return Err(format!(
                        "test target `{}` must be a qualified function name (Module.func)",
                        t.function_name
                    ));
                }
                if t.cases.is_empty() {
                    return Err(format!("test block for `{}` has no cases", t.function_name));
                }
                for case in &t.cases {
                    authored::literal(&case.input)?;
                    // Matchers still evaluate their synthesized expected AST in
                    // the legacy executor. Bare Ok/Err may resolve user code.
                    authored::literal(&case.expected)?;
                    if !case.after_checks.is_empty() {
                        return Err("native test does not support global +after assertions".into());
                    }
                }
            }
            other => {
                return Err(format!(
                    "test tool admits only +test blocks, got `{}`",
                    describe_op(other)
                ))
            }
        }
    }
    Ok(())
}

/// Parse a tool name + arguments into a validated [`ParsedTool`], enforcing a
/// strict closed object, size caps and the per-tool static constraints
/// (op kinds, qualified names, injection-free lowering). Pure: touches no
/// program or permission state.
fn parse_tool(name: &str, arguments: &str) -> Result<ParsedTool, String> {
    let normalized = frontend::normalize(name, parse_args_object(arguments)?)?;
    let map = normalized.arguments;
    match name {
        "inspect" => {
            reject_extra_fields(&map, &["kind", "target"])?;
            let kind_str = take_string(&map, "kind", true, MAX_TARGET_BYTES)?.unwrap();
            let target = take_string(&map, "target", false, MAX_TARGET_BYTES)?;
            let kind = match kind_str.as_str() {
                "symbols" => InspectKind::Symbols,
                "source" => InspectKind::Source,
                "deps" => InspectKind::Deps,
                "tasks" => InspectKind::Tasks,
                "library" => InspectKind::Library,
                other => {
                    return Err(format!(
                        "inspect `kind` must be one of symbols|source|deps|tasks|library, got `{other}`"
                    ))
                }
            };
            match kind {
                InspectKind::Source | InspectKind::Deps => {
                    if !target.as_deref().is_some_and(is_qualified_function_name) {
                        return Err(format!("inspect `{kind_str}` requires a qualified Module.member target; bare module/function/type names are not supported"));
                    }
                }
                InspectKind::Symbols => {}
                InspectKind::Tasks | InspectKind::Library => {
                    if target.is_some() {
                        return Err(format!("inspect `{kind_str}` does not accept a `target`"));
                    }
                }
            }
            Ok(ParsedTool::Inspect { kind, target, format: normalized.format })
        }
        "mutate" if normalized.python => frontend::parse_python_mutate(&map),
        "mutate" => frontend::parse_mutate(&map, false),
        "test" => frontend::parse_tests(&map, normalized.python),
        "evaluate" => frontend::parse_evaluate(&map, normalized.python),
        "finish" => {
            reject_extra_fields(&map, &[])?;
            Ok(ParsedTool::Finish)
        }
        other => Err(format!("unknown native tool `{other}`")),
    }
}

/// Preflight a single tool's arguments without any side effects.
///
/// Validates only the argument schema and static constraints (closed object,
/// size caps, op kinds, qualified-name format, injection-free lowering). It
/// does **not** check target-module permissions or target existence. Batch
/// callers should preflight every tool before executing any of them.
pub fn validate_native_tool(name: &str, arguments: &str) -> Result<(), String> {
    parse_tool(name, arguments).map(|_| ())
}

/// Side-effect-free admission + permission preflight.
///
/// Runs [`validate_native_tool`], then the eligibility gate and the per-module
/// permission checks against the *same* `config` the execution will use. It
/// deliberately does **not** check target existence — a later `mutate` may have
/// just created the function a later `evaluate`/`test` targets. Existence (and
/// the executor's test gate) are resolved fresh in [`execute_native_tool`].
pub fn validate_native_tool_authorized(
    name: &str,
    arguments: &str,
    config: &AppConfig,
) -> Result<(), String> {
    if is_guest(config) {
        return Err("native tool dispatch denied: guest turns are not eligible".to_string());
    }
    if !native_tools_eligible(config) {
        return Err("native tool dispatch denied: not write-capable".to_string());
    }
    let parsed = parse_tool(name, arguments)?;
    let model = model_name(config);
    check_authorization(&parsed, config, &model)
}

/// Per-module permission checks for a parsed tool. Single source of truth —
/// shared by [`validate_native_tool_authorized`] and [`execute_native_tool`].
fn check_authorization(parsed: &ParsedTool, config: &AppConfig, model: &str) -> Result<(), String> {
    match parsed {
        ParsedTool::PythonMutate { module, preflight, .. } => {
            if resolve_module(config, model, module) < PermissionLevel::Write {
                return Err(format!("permission denied: cannot modify module '{module}' (need Write)"));
            }
            for target in &preflight.references {
                if resolve_module(config, model, module_of(target)) < PermissionLevel::Execute {
                    return Err(format!("permission denied for authored reference `{target}` (need Execute)"));
                }
            }
        }
        ParsedTool::Inspect { kind, target, .. } => match kind {
            InspectKind::Source | InspectKind::Deps => {
                let t = target.as_deref().unwrap_or("");
                let module = module_of(t);
                if resolve_module(config, &model, module) < PermissionLevel::Read {
                    return Err(format!(
                        "permission denied: cannot read module '{module}' (need Read)"
                    ));
                }
            }
            InspectKind::Symbols => {
                if let Some(t) = target {
                    let module = if t.contains('.') { module_of(t) } else { t.as_str() };
                    if resolve_module(config, &model, module) == PermissionLevel::None {
                        return Err(format!("permission denied: module '{module}' is not visible"));
                    }
                }
            }
            InspectKind::Tasks | InspectKind::Library => {}
        },
        ParsedTool::Mutate { modules, source, roles, .. } => {
            for m in modules {
                if resolve_module(config, &model, m) < PermissionLevel::Write {
                    return Err(format!(
                        "permission denied: cannot modify module '{m}' (need Write)"
                    ));
                }
            }
            let ops = parser::parse(source).map_err(|e| e.to_string())?;
            authored::check_operations(&ops, config, model, None, roles.as_ref())?;
        }
        ParsedTool::Test { targets, .. } => {
            for t in targets {
                let module = module_of(t);
                if resolve_module(config, &model, module) < PermissionLevel::Write {
                    return Err(format!(
                        "permission denied: cannot test function '{t}' (module '{module}' needs Write)"
                    ));
                }
            }
        }
        ParsedTool::Evaluate { function, .. } => {
            let module = module_of(function);
            if resolve_module(config, &model, module) < PermissionLevel::Execute {
                return Err(format!(
                    "permission denied: cannot evaluate function '{function}' (module '{module}' needs Execute)"
                ));
            }
        }
        ParsedTool::Finish => {}
    }
    Ok(())
}

// ── Envelope / result helpers ──────────────────────────────────────────────

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let t: String = s.chars().take(max).collect();
        format!("{t}\n…[truncated]")
    }
}

fn denied_result(tool: &str, msg: String) -> NativeToolResult {
    let execution = CodeExecutionResult {
        mutation_results: vec![MutationResult {
            message: msg.clone(),
            success: false,
        }],
        test_results: vec![],
        has_errors: true,
        needs_opencode_restart: false,
        agent_spawned: false,
        spawned_agent_names: vec![],
        done_accepted: false,
    };
    let content = build_envelope(tool, &execution, &msg, None, None);
    NativeToolResult {
        execution,
        content,
    }
}

fn summarize_mutations(code: &CodeExecutionResult) -> String {
    if code.mutation_results.is_empty() {
        "no mutations applied".to_string()
    } else {
        code.mutation_results
            .iter()
            .map(|r| format!("{}: {}", if r.success { "OK" } else { "ERROR" }, r.message))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn summarize_tests(code: &CodeExecutionResult) -> String {
    if code.test_results.is_empty() {
        "no tests run".to_string()
    } else {
        code.test_results
            .iter()
            .map(|t| format!("{}: {}", if t.pass { "PASS" } else { "FAIL" }, t.message))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

fn build_envelope(
    tool: &str,
    code: &CodeExecutionResult,
    output: &str,
    query_results: Option<&str>,
    eval_results: Option<&str>,
) -> String {
    let mutations: Vec<serde_json::Value> = code
        .mutation_results
        .iter()
        .take(MAX_RESULT_MESSAGES)
        .map(|m| {
            serde_json::json!({
                "message": truncate_chars(&m.message, MAX_MESSAGE_CHARS),
                "success": m.success,
            })
        })
        .collect();
    let tests: Vec<serde_json::Value> = code
        .test_results
        .iter()
        .take(MAX_RESULT_MESSAGES)
        .map(|t| {
            serde_json::json!({
                "message": truncate_chars(&t.message, MAX_MESSAGE_CHARS),
                "pass": t.pass,
            })
        })
        .collect();
    let json = serde_json::json!({
        "tool": tool,
        "success": !code.has_errors,
        "has_errors": code.has_errors,
        "done_accepted": code.done_accepted,
        "mutation_results": mutations,
        "test_results": tests,
        "query_results": query_results.map(|s| truncate_chars(s, MAX_OUTPUT_CHARS)),
        "eval_results": eval_results.map(|s| truncate_chars(s, MAX_OUTPUT_CHARS)),
        "output": truncate_chars(output, MAX_OUTPUT_CHARS),
    });
    json.to_string()
}

// ── Inspection (read-only, permission-filtered) ────────────────────────────

fn execute_inspect(
    config: &AppConfig,
    session: &WorkingSet,
    model: &str,
    kind: InspectKind,
    target: Option<String>,
    format: frontend::SourceFormat,
) -> NativeToolResult {
    let program = &session.program;
    let result: Result<String, String> = match kind {
        InspectKind::Tasks => inspect_tasks(config, model),
        InspectKind::Library => Ok(inspect_library(config, &model, program)),
        InspectKind::Symbols => match target {
            None => Ok(crate::validator::program_summary_for_model(
                program,
                &config.permission_config,
                config.access_level,
                &model,
            )),
            Some(t) => inspect_symbols_scoped(config, &model, program, &t),
        },
        InspectKind::Source => {
            let t = target.as_deref().unwrap_or("");
            match format {
                frontend::SourceFormat::Adapsis => Ok(crate::typeck::reconstruct_source(program, t)),
                frontend::SourceFormat::Python => frontend::print_source(config, model, program, t),
            }
        }
        InspectKind::Deps => {
            let t = target.as_deref().unwrap_or("");
            inspect_deps(config, &model, program, t)
        }
    };

    match result {
        Ok(output) => {
            let output = truncate_chars(&output, MAX_OUTPUT_CHARS);
            let execution = CodeExecutionResult {
                mutation_results: vec![],
                test_results: vec![],
                has_errors: false,
                needs_opencode_restart: false,
                agent_spawned: false,
                spawned_agent_names: vec![],
                done_accepted: false,
            };
            let content = build_envelope("inspect", &execution, &output, Some(&output), None);
            NativeToolResult {
                execution,
                content,
            }
        }
        Err(e) => denied_result("inspect", e),
    }
}

fn inspect_symbols_scoped(
    config: &AppConfig,
    model: &str,
    program: &Program,
    target: &str,
) -> Result<String, String> {
    if let Some(m) = program.modules.iter().find(|m| m.name == target) {
        let perm = resolve_module(config, model, target);
        if perm == PermissionLevel::None {
            return Err(format!("permission denied: module '{target}' is not visible"));
        }
        return Ok(format_module_contents(m, perm));
    }
    if target.contains('.') {
        let module = module_of(target);
        let perm = resolve_module(config, model, module);
        if perm == PermissionLevel::None {
            return Err(format!("permission denied: module '{module}' is not visible"));
        }
        if let Some(f) = program.get_function(target) {
            return Ok(format_function_sig(target, f, perm));
        }
        return Err(format!("no function `{target}` found"));
    }
    Err(format!(
        "unsupported symbols scope `{target}`: use a module name or a qualified Module.func"
    ))
}

fn format_module_contents(m: &crate::ast::Module, perm: PermissionLevel) -> String {
    let mut out = format!("Module {}:\n", m.name);
    if !m.types.is_empty() {
        out.push_str("  Types:\n");
        for t in &m.types {
            out.push_str(&format!("    {}\n", t.name()));
        }
    }
    out.push_str("  Functions:\n");
    for f in &m.functions {
        out.push_str(&format_function_sig_line(f, perm));
    }
    out
}

fn format_function_sig_line(f: &crate::ast::FunctionDecl, perm: PermissionLevel) -> String {
    let params = f
        .params
        .iter()
        .map(|p| format!("{}:{:?}", p.name, p.ty))
        .collect::<Vec<_>>()
        .join(", ");
    let effects = if f.effects.is_empty() {
        String::new()
    } else {
        format!(
            " [{}]",
            f.effects
                .iter()
                .map(|e| format!("{e:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let mut s = format!("    {} ({})->{:?}{}\n", f.name, params, f.return_type, effects);
    if perm >= PermissionLevel::Read {
        if let Some(doc) = &f.doc {
            s.push_str(&format!("        doc: {doc}\n"));
        }
    }
    s
}

fn format_function_sig(name: &str, f: &crate::ast::FunctionDecl, perm: PermissionLevel) -> String {
    let params = f
        .params
        .iter()
        .map(|p| format!("{}:{:?}", p.name, p.ty))
        .collect::<Vec<_>>()
        .join(", ");
    let mut out = format!("Function {name}:\n");
    out.push_str(&format!("  params: {params}\n"));
    out.push_str(&format!("  returns: {:?}\n", f.return_type));
    out.push_str(&format!("  effects: {:?}\n", f.effects));
    if perm >= PermissionLevel::Read {
        if let Some(doc) = &f.doc {
            out.push_str(&format!("  doc: {doc}\n"));
        }
    }
    out
}

fn callee_visible(config: &AppConfig, model: &str, program: &Program, callee: &str) -> bool {
    if callee.contains('.') {
        return resolve_module(config, model, module_of(callee)) > PermissionLevel::None;
    }
    if program.functions.iter().any(|f| f.name == callee) {
        return true;
    }
    let qualified = program.qualify_function_name(callee);
    if qualified.contains('.') {
        return resolve_module(config, model, module_of(&qualified)) > PermissionLevel::None;
    }
    true // builtin / external — not a module secret
}

fn inspect_deps(
    config: &AppConfig,
    model: &str,
    program: &Program,
    target: &str,
) -> Result<String, String> {
    let graph = crate::typeck::build_call_graph(program);
    let callees = match graph.callees.get(target) {
        Some(cs) if !cs.is_empty() => cs.clone(),
        _ => return Ok(format!("`{target}` calls no other functions")),
    };
    let mut visible = Vec::new();
    let mut hidden = 0usize;
    for c in &callees {
        if callee_visible(config, model, program, c) {
            visible.push(c.clone());
        } else {
            hidden += 1;
        }
    }
    visible.sort();
    let mut out = format!("`{target}` calls:");
    if visible.is_empty() {
        out.push_str(" (none visible)");
    } else {
        out.push_str(&format!(" {}", visible.join(", ")));
    }
    if hidden > 0 {
        out.push_str(&format!(" [{hidden} hidden]"));
    }
    Ok(out)
}

fn inspect_tasks(config: &AppConfig, model: &str) -> Result<String, String> {
    let Some(registry) = &config.task_registry else {
        return Ok("No task registry (async not available).".into());
    };
    let tasks = registry.lock().map_err(|_| "task registry unavailable")?;
    let mut visible: Vec<_> = tasks.values().filter(|task| {
        is_qualified_function_name(&task.function_name)
            && resolve_module(config, model, module_of(&task.function_name)) >= PermissionLevel::Read
    }).collect();
    visible.sort_by_key(|task| task.id);
    let rows: Vec<_> = visible.into_iter().map(|task| {
        let status = match task.status {
            crate::coroutine::WaitReason::Running => "running",
            crate::coroutine::WaitReason::Completed(_) => "completed",
            crate::coroutine::WaitReason::Failed(_) => "failed",
            _ => "waiting",
        };
        format!("task {} [{}] — {status}", task.id, task.function_name)
    }).collect();
    Ok(if rows.is_empty() { "No visible attributable tasks.".into() } else { rows.join("\n") })
}

fn inspect_library(config: &AppConfig, model: &str, program: &Program) -> String {
    let dir = crate::library::library_dir();
    let mut out = format!("Module library: {}\n", dir.display());
    let visible = |name: &str| resolve_module(config, model, name) > PermissionLevel::None;

    let mut loaded: Vec<&str> = program
        .modules
        .iter()
        .map(|m| m.name.as_str())
        .filter(|n| visible(n))
        .collect();
    loaded.sort();
    if loaded.is_empty() {
        out.push_str("No library modules visible to this turn.\n");
    } else {
        out.push_str(&format!("\nLoaded {} module(s):\n", loaded.len()));
        for name in &loaded {
            let m = program.modules.iter().find(|m| m.name == *name).unwrap();
            out.push_str(&format!(
                "  {name} — {} type(s), {} function(s)\n",
                m.types.len(),
                m.functions.len()
            ));
        }
    }

    if dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut files: Vec<String> = entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "ax"))
                .map(|e| {
                    e.path()
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("?")
                        .to_string()
                })
                .filter(|n| visible(n))
                .collect();
            files.sort();
            out.push_str(&format!("\nLibrary files on disk ({} visible):\n", files.len()));
            for f in files {
                out.push_str(&format!("  {f}\n"));
            }
        }
    }
    out
}

// ── Dispatch ───────────────────────────────────────────────────────────────

/// Admission order is load-bearing: authorize before signature-aware lowering,
/// then reauthorize canonical operations and check the actual runtime targets.
fn prepare_native_tool(
    name: &str,
    arguments: &str,
    config: &AppConfig,
    program: &Program,
) -> Result<(ParsedTool, String), String> {
    if is_guest(config) {
        return Err("native tool dispatch denied: guest turns are not eligible".to_string());
    }
    if !native_tools_eligible(config) {
        return Err("native tool dispatch denied: not write-capable".to_string());
    }
    let parsed = parse_tool(name, arguments)?;
    // Resolve the pinned permission model and re-check per-module permission
    // before dispatching. Authority is read only from the pinned config
    // (`llm_model`) and turn identity — never a live mutable conversation read,
    // so it cannot drift from the caller's per-turn permission snapshot (#79).
    let model = model_name(config);
    check_authorization(&parsed, config, &model)?;
    let parsed = frontend::finalize(parsed, config, program)?;
    check_authorization(&parsed, config, &model)?;
    authored::runtime_targets(&parsed, config, program)?;
    Ok((parsed, model))
}

/// Execute a single native tool against a caller-prepared working snapshot.
///
/// All tools run through [`execute_code_with_baseline`] with the *same*
/// immutable completion baseline, so the executor's mutation semantics, test
/// gate and per-function CoW write-back remain authoritative. Inspection is
/// read-only and permission-filtered. The caller snapshots before and writes
/// the working set back after **every** call.
pub async fn execute_native_tool(
    name: &str,
    arguments: &str,
    config: &AppConfig,
    session: &mut WorkingSet,
    completion_baseline: &Program,
) -> NativeToolResult {
    let (parsed, model) = match prepare_native_tool(name, arguments, config, &session.program) {
        Ok(prepared) => prepared,
        Err(error) => return denied_result(name, error),
    };
    match parsed {
        ParsedTool::PythonMutate { .. } => denied_result(name, "Python mutation was not finalized".into()),
        ParsedTool::Inspect { kind, target, format } => execute_inspect(config, session, &model, kind, target, format),
        ParsedTool::Mutate { source, .. } => {
            let code =
                execute_code_with_baseline(&source, config, session, None, Some(completion_baseline))
                    .await;
            let output = truncate_chars(&summarize_mutations(&code), MAX_OUTPUT_CHARS);
            let content = build_envelope(name, &code, &output, None, None);
            NativeToolResult {
                execution: code,
                content,
            }
        }
        ParsedTool::Test { source, .. } => {
            let code =
                execute_code_with_baseline(&source, config, session, None, Some(completion_baseline))
                    .await;
            let output = truncate_chars(&summarize_tests(&code), MAX_OUTPUT_CHARS);
            let content = build_envelope(name, &code, &output, None, None);
            NativeToolResult {
                execution: code,
                content,
            }
        }
        ParsedTool::Evaluate { source, .. } => {
            let code =
                execute_code_with_baseline(&source, config, session, None, Some(completion_baseline))
                    .await;
            let eval_msg = code
                .mutation_results
                .iter()
                .map(|r| r.message.clone())
                .collect::<Vec<_>>()
                .join("\n");
            let output = truncate_chars(&eval_msg, MAX_OUTPUT_CHARS);
            let content = build_envelope(name, &code, &output, None, Some(&output));
            NativeToolResult {
                execution: code,
                content,
            }
        }
        ParsedTool::Finish => {
            let code =
                execute_code_with_baseline("!done", config, session, None, Some(completion_baseline))
                    .await;
            let output = if code.done_accepted && !code.has_errors {
                "done accepted".to_string()
            } else {
                summarize_mutations(&code)
            };
            let output = truncate_chars(&output, MAX_OUTPUT_CHARS);
            let content = build_envelope(name, &code, &output, None, None);
            NativeToolResult {
                execution: code,
                content,
            }
        }
    }
}

#[cfg(test)]
#[path = "native_tools/tests.rs"]
mod tests;
