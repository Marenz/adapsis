//! Explicit frontend selection before the unchanged guarded operation pipeline.
use super::*;
use serde_json::{Map, Value};

#[derive(Clone, Copy, Default)]
pub(super) enum SourceFormat {
    #[default]
    Adapsis,
    Python,
}

pub(super) struct Normalized {
    pub arguments: Map<String, Value>,
    pub python: bool,
    pub format: SourceFormat,
}

/// Only syntax and metadata are normalized here. A language selection never
/// changes authority; canonical output still traverses every native check.
pub(super) fn normalize(name: &str, mut map: Map<String, Value>) -> Result<Normalized, String> {
    let mut python = false;
    let mut format = SourceFormat::Adapsis;
    if matches!(name, "mutate" | "test" | "evaluate") {
        python = match map.remove("language") {
            None => false,
            Some(Value::String(value)) if value == "adapsis" => false,
            Some(Value::String(value)) if value == "python" => true,
            Some(_) => return Err("language must be adapsis or python".into()),
        };
    }
    if matches!(name, "mutate" | "test") {
        if python {
            let module = take_string(&map, "module", true, MAX_TARGET_BYTES)?.unwrap();
            if !is_identifier(&module) || module.starts_with("__") {
                return Err("python module metadata must be one non-dunder identifier, without dots or paths".into());
            }
            take_string(&map, "source", true, MAX_SOURCE_BYTES)?;
        } else if map.contains_key("module") {
            return Err("module metadata is accepted only with language=python; Adapsis source declares +module explicitly".into());
        }
    }
    if name == "inspect" {
        if let Some(value) = map.remove("format") {
            if map.get("kind").and_then(Value::as_str) != Some("source") {
                return Err("format is accepted only for inspect kind=source".into());
            }
            format = match value.as_str() {
                Some("adapsis") => SourceFormat::Adapsis,
                Some("python") => SourceFormat::Python,
                _ => return Err("source format must be adapsis or python".into()),
            };
        }
    }
    Ok(Normalized {
        arguments: map,
        python,
        format,
    })
}

pub(super) fn parse_mutate(map: &Map<String, Value>, python: bool) -> Result<ParsedTool, String> {
    reject_extra_fields(map, &["source"])?;
    let source = take_string(map, "source", true, MAX_SOURCE_BYTES)?.unwrap();
    if source.trim().is_empty() {
        return Err("mutate `source` is empty".into());
    }
    let ops =
        parser::parse(&source).map_err(|error| format!("mutate `source` parse error: {error}"))?;
    validate_mutate_ops(&ops)?;
    literals::check_new_shared_defaults(&ops)?;
    let mut modules = Vec::new();
    for op in &ops {
        if let parser::Operation::Module(module) = op {
            collect_module_names(module, &mut modules);
        }
    }
    Ok(ParsedTool::Mutate {
        source: crate::session::replay::Submission::adapsis(&source),
        operations: ops,
        modules,
        python,
        roles: None,
    })
}

pub(super) fn parse_python_mutate(map: &Map<String, Value>) -> Result<ParsedTool, String> {
    reject_extra_fields(map, &["source", "module"])?;
    let module = take_string(map, "module", true, MAX_TARGET_BYTES)?.unwrap();
    let source = take_string(map, "source", true, MAX_SOURCE_BYTES)?.unwrap();
    let preflight = crate::python_frontend::preflight_module(&source, &module)
        .map_err(|error| error.to_string())?;
    Ok(ParsedTool::PythonMutate {
        source,
        module,
        preflight,
    })
}

/// Invoked only after module/reference authorization, at the actual per-call
/// snapshot. No signature from an inaccessible module enters the type checker.
pub(super) fn finalize(
    parsed: ParsedTool,
    config: &AppConfig,
    program: &Program,
) -> Result<ParsedTool, String> {
    let ParsedTool::PythonMutate {
        source,
        module,
        preflight,
    } = parsed
    else {
        return Ok(parsed);
    };
    let model = model_name(config);
    let mut signatures = authorized_program(config, &model, program);
    let roles = crate::python_frontend::resolve_native_references(
        &preflight,
        &module,
        &signatures,
        program,
    )?;
    crate::python_frontend::retain_native_dependencies(&mut signatures, &roles)?;
    let lowered = crate::python_frontend::lower_module_with_program(&source, &module, &signatures)
        .map_err(|error| error.to_string())?;
    // Verify owner erasure before the canonical executor can consult full-state
    // schemas. The guard sees only full-state binding metadata and returns a
    // categorical refusal, never hidden field/type diagnostics.
    crate::python_frontend::check_native_type_bindings(
        &lowered.operations,
        &module,
        &roles.types,
        &roles.functions,
        &signatures,
        program,
    )?;
    validate_mutate_ops(&lowered.operations)?;
    literals::check_new_shared_defaults(&lowered.operations)?;
    let mut modules = Vec::new();
    for operation in &lowered.operations {
        if let parser::Operation::Module(module) = operation { collect_module_names(module, &mut modules); }
    }
    Ok(ParsedTool::Mutate {
        source: crate::session::replay::Submission::python(&module, &source),
        operations: lowered.operations, modules, python: true, roles: Some(roles),
    })
}

fn authorized_program(config: &AppConfig, model: &str, program: &Program) -> Program {
    let mut filtered = program.clone();
    filtered.functions.clear();
    filtered
        .modules
        .retain(|module| resolve_module(config, model, &module.name) >= PermissionLevel::Execute);
    for module in &mut filtered.modules {
        module.modules.clear();
    }
    filtered.rebuild_function_index();
    filtered
}

pub(super) fn print_source(
    config: &AppConfig,
    model: &str,
    program: &Program,
    target: &str,
) -> Result<String, String> {
    // Printing executes no defaults. Both admission and the type-aware printer
    // see only authorized metadata, independent of private signatures/defaults.
    let filtered = authorized_program(config, model, program);
    literals::require_python_entry(&filtered, target, super::scalar_entry::EntryUse::Inspect)?;
    if filtered.get_function(target).is_some() {
        let source = crate::python_frontend::print_function(&filtered, target)
            .map_err(|error| error.to_string())?;
        let module = module_of(target);
        let preflight =
            crate::python_frontend::preflight_printed_module(&source, module).map_err(|e| e.to_string())?;
        crate::python_frontend::resolve_native_references(&preflight, module, &filtered, program)?;
        Ok(source)
    } else {
        crate::python_frontend::print_type(&filtered, target).map_err(|error| error.to_string())
    }
}

pub(super) fn parse_tests(map: &Map<String, Value>, python: bool) -> Result<ParsedTool, String> {
    reject_extra_fields(map, if python { &["source", "module"] } else { &["source"] })?;
    let source = take_string(map, "source", true, MAX_SOURCE_BYTES)?.unwrap();
    if source.trim().is_empty() {
        return Err("test `source` is empty".into());
    }
    let (ops, submission) = if python {
        let module = take_string(map, "module", true, MAX_TARGET_BYTES)?.unwrap();
        let lowered = crate::python_frontend::lower_tests(&source, &module).map_err(|error| error.to_string())?;
        (lowered.operations, crate::session::replay::Submission::python(&module, &source))
    } else {
        (parser::parse(&source).map_err(|error| format!("test `source` parse error: {error}"))?,
            crate::session::replay::Submission::adapsis(&source))
    };
    validate_test_ops(&ops)?;
    let targets = ops
        .iter()
        .filter_map(|op| match op {
            parser::Operation::Test(test) => Some(test.function_name.clone()),
            _ => None,
        })
        .collect();
    Ok(ParsedTool::Test {
        source: submission,
        operations: ops,
        targets,
        python,
    })
}

pub(super) fn parse_evaluate(map: &Map<String, Value>, python: bool) -> Result<ParsedTool, String> {
    reject_extra_fields(map, &["function", "input"])?;
    let function = take_string(map, "function", true, MAX_TARGET_BYTES)?.unwrap();
    let input = take_string(map, "input", false, MAX_SOURCE_BYTES)?.unwrap_or_default();
    if !is_qualified_function_name(&function) {
        return Err(format!(
            "`function` must be a qualified function name (Module.func), got `{function}`"
        ));
    }
    if !python && (input.contains('\n') || input.contains('\r')) {
        return Err("evaluate `input` must not contain newlines".into());
    }
    let parsed = if python {
        crate::python_frontend::lower_input_expr(&input).map_err(|error| error.to_string())?
    } else if input.trim().is_empty() {
        parser::Expr::StructLiteral(vec![])
    } else {
        parser::parse_test_input(0, &input).map_err(|error| format!("evaluate `input` is not a valid expression: {error}"))?
    };
    authored::literal(&parsed)?;
    let operations = vec![parser::Operation::Eval(parser::EvalMutation { function_name: function.clone(), input: parsed, inline_expr: None })];
    let source = if python { crate::session::replay::Submission::python(module_of(&function), &input) }
        else { crate::session::replay::Submission::adapsis(&format!("!eval {function} {input}")) };
    Ok(ParsedTool::Evaluate {
        function,
        source,
        operations,
        python,
    })
}
