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
            let source = take_string(&map, "source", true, MAX_SOURCE_BYTES)?.unwrap();
            if name == "test" {
                let lowered = crate::python_frontend::lower_tests(&source, &module)
                    .map_err(|error| error.to_string())?;
                map.remove("module");
                map.insert("source".into(), Value::String(lowered.canonical_adapsis));
            }
        } else if map.contains_key("module") {
            return Err("module metadata is accepted only with language=python; Adapsis source declares +module explicitly".into());
        }
    } else if name == "evaluate" && python {
        let input = take_string(&map, "input", false, MAX_SOURCE_BYTES)?.unwrap_or_default();
        let input =
            crate::python_frontend::lower_input(&input).map_err(|error| error.to_string())?;
        map.insert("input".into(), Value::String(input));
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
        source,
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
    let mut parsed = parse_mutate(
        &serde_json::Map::from_iter([("source".into(), Value::String(lowered.canonical_adapsis))]),
        true,
    )?;
    if let ParsedTool::Mutate { roles: slot, .. } = &mut parsed {
        *slot = Some(roles);
    }
    Ok(parsed)
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
    // Full-state helper safety is categorical and independent of private
    // signatures. The type-aware printer receives only authorized signatures.
    let filtered = authorized_program(config, model, program);
    literals::require_python_entry(&filtered, target)?;
    if filtered.get_function(target).is_some() {
        let source = crate::python_frontend::print_function(&filtered, target)
            .map_err(|error| error.to_string())?;
        let module = module_of(target);
        let preflight =
            crate::python_frontend::preflight_module(&source, module).map_err(|e| e.to_string())?;
        crate::python_frontend::resolve_native_references(&preflight, module, &filtered, program)?;
        Ok(source)
    } else {
        crate::python_frontend::print_type(&filtered, target).map_err(|error| error.to_string())
    }
}

pub(super) fn parse_tests(map: &Map<String, Value>, python: bool) -> Result<ParsedTool, String> {
    reject_extra_fields(map, &["source"])?;
    let source = take_string(map, "source", true, MAX_SOURCE_BYTES)?.unwrap();
    if source.trim().is_empty() {
        return Err("test `source` is empty".into());
    }
    let ops =
        parser::parse(&source).map_err(|error| format!("test `source` parse error: {error}"))?;
    validate_test_ops(&ops)?;
    let targets = ops
        .iter()
        .filter_map(|op| match op {
            parser::Operation::Test(test) => Some(test.function_name.clone()),
            _ => None,
        })
        .collect();
    Ok(ParsedTool::Test {
        source,
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
    if input.contains('\n') || input.contains('\r') {
        return Err("evaluate `input` must not contain newlines".into());
    }
    if !input.trim().is_empty() {
        let parsed = parser::parse_test_input(0, &input)
            .map_err(|error| format!("evaluate `input` is not a valid expression: {error}"))?;
        authored::literal(&parsed)?;
    }
    let source = if input.trim().is_empty() {
        format!("!eval {function}")
    } else {
        format!("!eval {function} {input}")
    };
    let ops =
        parser::parse(&source).map_err(|error| format!("evaluate lowering failed: {error}"))?;
    match ops.as_slice() {
        [parser::Operation::Eval(eval)]
            if eval.function_name == function && eval.inline_expr.is_none() => {}
        _ => return Err("evaluate lowering produced unexpected operations (rejected)".into()),
    }
    Ok(ParsedTool::Evaluate {
        function,
        source,
        python,
    })
}
