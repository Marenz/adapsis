//! Native `run`: execute an ad-hoc Python block once (#129, DESIGN L2).
//!
//! The block is lowered as the body of one function that is never declared, admitted by
//! the checks a `mutate` submission of that function passes, and executed on the
//! evaluation path of native `evaluate` (same blocking task, same coroutine handle bound
//! to the turn identity, same IO boundary). Three things differ from `mutate` + `evaluate`:
//!
//! * nothing is applied: the declaration never enters a Program, a library file, the
//!   mutation history or a replay record, and there is no test gate;
//! * the Program cannot change. A direct call to a program-changing builtin is refused at
//!   admission; the block's coroutine handle is frozen, so the same builtin reached through
//!   a called function or a spawned task is refused at dispatch, before it touches anything;
//! * the result is the value of `return` as written (no invocation normalization), capped
//!   at [`RUN_MAX_OUTPUT_CHARS`] with an explicit truncation notice.
use super::super::execute::io_eval;
use super::*;
use crate::reserved_names::RUN_SCOPE;

/// Maximum length (chars) of a `run` result. Sized for files, diffs and test logs; the
/// 8,000-character [`MAX_OUTPUT_CHARS`] of the other tools would cut those off.
pub(super) const RUN_MAX_OUTPUT_CHARS: usize = 100_000;
/// Name of the never-declared function a block is lowered as.
const FUNCTION: &str = "run_block";

/// A fully admitted block: the single-module, single-function operation shape every
/// shared mutation check accepts, plus where it resolves names.
pub(super) struct Block {
    pub operations: Vec<parser::Operation>,
    pub modules: Vec<String>,
    pub roles: crate::python_frontend::NativeReferences,
    module: String,
    function: String,
}

pub(super) fn definition(obj: impl Fn(&[&str], serde_json::Value) -> serde_json::Value) -> ToolDefinition {
    ToolDefinition {
        name: "run".into(),
        description: "Execute an ad-hoc typed Python block once: read a file, call functions, compute a value. Nothing is declared or persisted and there is no test gate. Use mutate to add or change code.".into(),
        parameters: obj(
            &["source"],
            serde_json::json!({
                "source": {"type": "string", "description": "Statements in the typed Python subset, exactly as inside an async function body: annotate every new binding (`text: str = await file_read(path)`), bind every await, and `return` the value you want back (rendered as text; without a returned value the call is an empty success). The block may await IO builtins and user functions. Call user functions as Module.function. Declarations (def/class) and imports are refused, and nothing a block runs can change the program (mutate, fn_replace, ... fail)."},
                "module": {"type": "string", "description": "Optional: one module identifier whose functions, types and shared variables resolve by bare name, as inside that module (needs Write on it). Omit to use qualified Module.function names only."},
                "language": {"type": "string", "enum": ["python"], "description": "Optional; run accepts Python only."}
            }),
        ),
    }
}

/// The `run` schema of a restricted speaker: no `module`, and the description states the
/// rule instead of offering what the dispatcher would refuse.
pub(super) fn restricted_definition(class: &SpeakerClass) -> ToolDefinition {
    let tools = class.guest_tools();
    let callable = match class {
        SpeakerClass::Guest(_) if tools.is_empty() => "pure builtins only (text, numbers, lists, maps, JSON, regex)".to_string(),
        SpeakerClass::Guest(_) => format!("pure builtins (text, numbers, lists, maps, JSON, regex) and your tools as awaited calls: {}", tools.join(", ")),
        _ => "pure builtins (text, numbers, lists, maps, JSON, regex) and functions of modules you may execute, as Module.function".to_string(),
    };
    let limits = class.limits();
    ToolDefinition {
        name: "run".into(),
        description: format!(
            "Execute a short typed Python block once, to combine several steps in one call. It may call {callable}. Nothing else: no files, shell, raw HTTP, code changes, shared variables or spawn. Limits per block: {} IO/tool calls, {} seconds, values up to {} MiB.",
            limits.io_calls.unwrap_or_default(),
            limits.wall_clock.map_or(0, |limit| limit.as_secs()),
            limits.value_bytes / (1024 * 1024),
        ),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Statements as inside an async function body: annotate every new binding (`hits: str = await web_search(\"...\")`), bind every await, and `return` the value you want back as text."},
            },
            "required": ["source"],
            "additionalProperties": false,
        }),
    }
}

/// Helpers a restricted block may not use: they start tasks or register handlers that
/// outlive the block.
const RESTRICTED_HELPERS: &[&str] = &["spawn", "source_add", "source_remove", "source_replace", "source_list", "event_register", "event_emit"];

/// The restricted call rule at admission (DESIGN L3): a block may call only what its
/// speaker can already call. Program-free, so a batch is refused before anything runs.
/// The interpreter enforces the same rule on the block's own frame (`eval/meter.rs`).
pub(super) fn restricted_admission(
    module: &str,
    preflight: &crate::python_frontend::NativePreflight,
    class: &SpeakerClass,
) -> Result<(), String> {
    let policy = crate::eval::meter::BlockPolicy {
        tools: class.guest_tools().iter().cloned().collect(),
        functions: Default::default(),
    };
    let refuse = |what: String| Err(crate::eval::meter::refusal(&what, &policy).to_string());
    if module != RUN_SCOPE {
        return refuse("a `module` scope".into());
    }
    if let Some(helper) = preflight.helper_calls.iter().find(|helper| RESTRICTED_HELPERS.contains(&helper.as_str())) {
        return refuse(format!("`{helper}`"));
    }
    if let Some(name) = preflight.bare_calls.iter().find(|name| {
        !crate::builtins::pure::allowed_in_restricted_block(name) && !policy.tools.contains(*name)
    }) {
        return refuse(format!("`{name}`"));
    }
    if let (SpeakerClass::Guest(_), Some(target)) = (class, preflight.references.first()) {
        return refuse(format!("function `{target}`"));
    }
    Ok(())
}

/// Program-free parsing and structural preflight (closed object, caps, syntax subset,
/// reference collection). No runtime signature is consulted here.
pub(super) fn parse(map: &serde_json::Map<String, serde_json::Value>) -> Result<ParsedTool, String> {
    reject_extra_fields(map, &["source", "module", "language"])?;
    if let Some(language) = take_string(map, "language", false, MAX_TARGET_BYTES)? {
        if language != "python" {
            return Err("run accepts Python only; omit `language` or set it to python".into());
        }
    }
    let source = take_string(map, "source", true, MAX_SOURCE_BYTES)?.unwrap();
    if source.trim().is_empty() {
        return Err("run `source` is empty".into());
    }
    let module = match take_string(map, "module", false, MAX_TARGET_BYTES)? {
        Some(module) if module == RUN_SCOPE => {
            return Err(format!("run `module` cannot be `{RUN_SCOPE}`: that name is reserved; omit `module` instead"));
        }
        Some(module) if !is_identifier(&module) || module.starts_with("__") => {
            return Err("run `module` must be one non-dunder identifier, without dots or paths".into());
        }
        Some(module) => module,
        None => RUN_SCOPE.to_string(),
    };
    let preflight = crate::python_frontend::preflight_block(&source, &module).map_err(diagnostic)?;
    if let Some(name) = preflight
        .bare_calls
        .iter()
        .find(|name| crate::builtins::program_effects::changes_program(name))
    {
        return Err(format!(
            "run cannot call `{name}`: a run block cannot change the program. \
             Use the mutate tool to declare, replace or remove code."
        ));
    }
    if module == RUN_SCOPE {
        unscoped_names(&preflight)?;
    }
    Ok(ParsedTool::PythonRun {
        source,
        module,
        preflight,
    })
}

/// Signature-aware admission at the per-call snapshot, after module/reference
/// authorization: the shared Python mutation admission, with block lowering.
pub(super) fn finalize(
    source: String,
    module: String,
    preflight: crate::python_frontend::NativePreflight,
    config: &AppConfig,
    program: &Program,
) -> Result<ParsedTool, String> {
    if module == RUN_SCOPE && program.modules.iter().any(|module| module.name == RUN_SCOPE) {
        // The validator cannot create such a module, but a hand-edited session file could
        // carry one; the reserved scope must never alias a real module.
        return Err(format!(
            "run without `module` is unavailable: a module named `{RUN_SCOPE}` exists although that \
             name is reserved for ad-hoc blocks. Pass `module` explicitly, or remove that module."
        ));
    }
    let function = function_name(program, &module);
    let admitted = frontend::admit_python(config, program, &module, &preflight, |signatures| {
        crate::python_frontend::lower_block_with_program(&source, &module, &function, signatures)
            .map_err(diagnostic)
    })
    .map_err(|error| error.replace(RUN_SCOPE, "<run>"))?;
    Ok(ParsedTool::Run(Block {
        operations: admitted.operations,
        modules: admitted.modules,
        roles: admitted.roles,
        module,
        function,
    }))
}

/// Without `module` the block resolves no name bare. The shared admission refuses these
/// shapes too; this states the fix in the tool's own terms, program-free, so a batch
/// preflight already catches it.
fn unscoped_names(preflight: &crate::python_frontend::NativePreflight) -> Result<(), String> {
    if let Some(name) = preflight.local_calls.first() {
        return Err(format!(
            "run: `{name}(...)` is not a builtin. Call user functions as Module.function, or pass \
             `module` to resolve that module's names bare."
        ));
    }
    if let Some(name) = preflight.value_references.iter().find(|name| *name != "None") {
        return Err(format!(
            "run: name `{name}` is not bound in this block. Bind it first with an annotation \
             (`{name}: T = ...`), or pass `module` to read that module's shared variables."
        ));
    }
    Ok(())
}

/// The block's function name only labels diagnostics and the evaluation frame; it is never
/// looked up. Avoid a real function of the scope module so no check can confuse the two.
fn function_name(program: &Program, module: &str) -> String {
    let taken = |name: &str| {
        program
            .modules
            .iter()
            .any(|candidate| candidate.name == module && candidate.functions.iter().any(|f| f.name == name))
    };
    (0usize..)
        .map(|n| if n == 0 { FUNCTION.to_string() } else { format!("{FUNCTION}_{n}") })
        .find(|name| !taken(name))
        .expect("an unused block name exists")
}

fn diagnostic(error: crate::python_frontend::Diagnostic) -> String {
    if error.module == RUN_SCOPE {
        format!("run:{}:{}: {}", error.line, error.column, error.message)
    } else {
        error.to_string()
    }
}

/// `Write` on the scope module is the mutate rule for the module a function is declared
/// in. Without `module` that is the level a new, unlisted module would need.
pub(super) fn write_denied(module: &str) -> String {
    if module == RUN_SCOPE {
        "permission denied: run without `module` needs Write on unlisted (user) modules; pass \
         `module` naming a module this turn may modify"
            .to_string()
    } else {
        format!("permission denied: cannot modify module '{module}' (need Write)")
    }
}

/// The validator checks `mutate` reaches (reserved `__` names, struct literals, effects)
/// and the reachable-graph checks of Python `evaluate`, on a declaration that stays
/// detached from every Program.
fn detached(block: &Block, program: &Program) -> Result<crate::ast::FunctionDecl, String> {
    let function = match block.operations.as_slice() {
        [parser::Operation::Module(module)] => match module.body.as_slice() {
            [parser::Operation::Function(function)] => function,
            _ => return Err("run block was not lowered to a single function".into()),
        },
        _ => return Err("run block was not lowered to a single module".into()),
    };
    let declaration = crate::validator::detached_module_function(program, &block.module, function)
        .map_err(|error| error.to_string().replace(RUN_SCOPE, "<run>"))?;
    crate::python_frontend::validate_await_contract_from(program, &declaration)?;
    crate::python_frontend::check_constructor_environment_from(program, &declaration.body)?;
    Ok(declaration)
}

pub(super) async fn execute(
    name: &str,
    block: Block,
    config: &AppConfig,
    session: &mut WorkingSet,
    class: &SpeakerClass,
    budget: &std::sync::Arc<TurnBudget>,
) -> NativeToolResult {
    let declaration = match detached(&block, &session.program) {
        Ok(declaration) => declaration,
        Err(error) => return denied_result(name, error),
    };
    let Some(sender) = config.io_sender.clone() else {
        return denied_result(name, "run requires the coroutine IO runtime, which this process does not provide".into());
    };
    let frame = format!("{}.{}", block.module, block.function);
    let body = declaration.body;
    // Limits for every class; for restricted speakers also the call policy of the
    // block's own frame: their tools and exactly the functions admission authorized.
    let context = crate::eval::meter::RunContext {
        limits: class.limits(),
        policy: class.restricted().then(|| crate::eval::meter::BlockPolicy {
            tools: class.guest_tools().iter().cloned().collect(),
            functions: block.roles.functions.iter().cloned().collect(),
        }),
        turn: budget.clone(),
    };
    let evaluation = io_eval::evaluate_with_io(
        config,
        &session.program,
        sender,
        io_eval::ProgramPolicy::Frozen,
        move |program, handle| crate::eval::eval_detached_block(program, &frame, &body, handle, context),
    )
    .await;
    // Shared variables, routes, roadmap and mocks a called function changed are live
    // already; refresh the snapshot so the caller's write-back keeps them, as evaluate
    // does. `session.program` is deliberately left as it was: nothing is merged back.
    config.sync_async_side_effects_into(session);
    let untouched = evaluation
        .after
        .as_ref()
        .is_some_and(|after| same_program(&evaluation.base, after));
    finish(name, evaluation.outcome, untouched)
}

/// Pointer identity of every declaration plus structural equality of the rest. The frozen
/// handle makes a difference unreachable; this is the backstop if a new program-changing
/// path is ever added without being classified.
pub(super) fn same_program(base: &Program, after: &Program) -> bool {
    fn functions(left: &[std::sync::Arc<crate::ast::FunctionDecl>], right: &[std::sync::Arc<crate::ast::FunctionDecl>]) -> bool {
        left.len() == right.len() && left.iter().zip(right).all(|(a, b)| std::sync::Arc::ptr_eq(a, b))
    }
    fn modules(left: &[crate::ast::Module], right: &[crate::ast::Module]) -> bool {
        left.len() == right.len()
            && left.iter().zip(right).all(|(a, b)| functions(&a.functions, &b.functions) && modules(&a.modules, &b.modules))
    }
    functions(&base.functions, &after.functions) && modules(&base.modules, &after.modules) && base == after
}

const PROGRAM_CHANGED: &str = "run failed its Program check: the Program this block ran on differs \
    from the one it started with, although a run block cannot change the program. The difference \
    was not written back. Any IO the block performed (files, HTTP, shell, shared variables) stays \
    done. Report this as a runtime bug.";

pub(super) fn finish(
    name: &str,
    outcome: Result<crate::eval::Value, io_eval::EvalFailure>,
    untouched: bool,
) -> NativeToolResult {
    // The evaluator has one "no value" value (what a function that falls off its end
    // yields). A block that returns it and a block that does not return both end with it,
    // so both are the same empty success and no field pretends to tell them apart.
    let (mut success, mut text) = match outcome {
        Ok(crate::eval::Value::None) => (true, String::new()),
        Ok(crate::eval::Value::String(text)) => (true, text.as_ref().clone()),
        Ok(value) => (true, value.to_string()),
        Err(failure) => (false, failure.describe("run")),
    };
    if !untouched {
        text = if text.is_empty() { PROGRAM_CHANGED.to_string() } else { format!("{PROGRAM_CHANGED}\nThe block ended with:\n{text}") };
        success = false;
    }
    let capped = cap_output(&text);
    // Source and result are content (#88): the same gates as native evaluate. The source
    // is never logged here; the result only when content logging is opted in.
    if crate::logging::content() {
        eprintln!("[native:run] {}", capped.text);
    }
    if !success {
        crate::logging::diagnostic("run-error", &capped.text);
    }
    let message = if success && text.is_empty() {
        "run completed with no value".to_string()
    } else {
        capped.text.clone()
    };
    let execution = CodeExecutionResult {
        mutation_results: vec![MutationResult { message, success }],
        test_results: vec![],
        has_errors: !success,
        needs_opencode_restart: false,
        agent_spawned: false,
        spawned_agent_names: vec![],
        done_accepted: false,
    };
    // The text appears once; the shared keys stay so every native envelope has one shape.
    let content = serde_json::json!({
        "tool": name,
        "success": success,
        "has_errors": !success,
        "done_accepted": false,
        "mutation_results": [],
        "test_results": [],
        "query_results": null,
        "eval_results": null,
        "truncated": capped.omitted > 0,
        "omitted_chars": capped.omitted,
        "output": capped.text,
    })
    .to_string();
    NativeToolResult { execution, content }
}

pub(super) struct Capped {
    pub text: String,
    pub omitted: usize,
}

/// Keep the first [`RUN_MAX_OUTPUT_CHARS`] characters and say exactly how much is missing.
pub(super) fn cap_output(text: &str) -> Capped {
    let total = text.chars().count();
    if total <= RUN_MAX_OUTPUT_CHARS {
        return Capped {
            text: text.to_string(),
            omitted: 0,
        };
    }
    let kept: String = text.chars().take(RUN_MAX_OUTPUT_CHARS).collect();
    let omitted = total - RUN_MAX_OUTPUT_CHARS;
    Capped {
        text: format!("{kept}\n…[truncated: {omitted} of {total} characters omitted]"),
        omitted,
    }
}
