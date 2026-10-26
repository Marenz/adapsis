//! Validate newly supplied executable syntax, never transitive trusted bodies.
use super::{AppConfig, PermissionLevel, Program, model_name, module_of, resolve_module};
use crate::parser::{Expr, Operation};

pub(super) use super::literals::literal;
type Runtime<'a> = (&'a Program, bool);
mod walk;

fn name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Ident(value) => Some(value.clone()),
        Expr::FieldAccess { base, field } => Some(format!("{}.{field}", name(base)?)),
        _ => None,
    }
}

fn reference(
    target: &str,
    config: &AppConfig,
    model: &str,
    program: Option<Runtime<'_>>,
    constructor: bool,
) -> Result<(), String> {
    if target.contains('.') {
        if resolve_module(config, model, module_of(target)) < PermissionLevel::Execute {
            return Err(format!(
                "permission denied for authored reference `{target}` (need Execute)"
            ));
        }
    } else {
        if constructor {
            return Ok(());
        }
        if !crate::builtins::is_builtin(target) && !matches!(target, "Ok" | "Err" | "Some") {
            return Err(format!(
                "native mutation requires qualified user calls (Module.func); bare `{target}` is not a builtin"
            ));
        }
        // Runtime tiers differ in builtin-vs-user lookup order. Never let a
        // nominal builtin spelling resolve to a denied user function.
        if let Some((program, python)) = program {
            if python {
                super::literals::require_unshadowed(program, target)?;
            }
            if program.get_function(target).is_some() {
                let qualified = program.qualify_function_name(target);
                if !qualified.contains('.')
                    || resolve_module(config, model, module_of(&qualified))
                        < PermissionLevel::Execute
                {
                    return Err(
                        "bare builtin spelling collides with an inaccessible user function".into(),
                    );
                }
            }
        }
    }
    Ok(())
}

pub(super) fn check_operations(
    ops: &[Operation],
    config: &AppConfig,
    model: &str,
    program: Option<Runtime<'_>>,
    roles: Option<&crate::python_frontend::NativeReferences>,
) -> Result<(), String> {
    walk::Checker {
        config,
        model,
        program,
        roles,
    }
    .operations(ops, "", &mut Default::default())
}

/// Existence checks are dispatch-time, after earlier batch mutations. Neutral
/// diagnostics avoid the legacy suggestion mechanism exposing hidden names.
pub(super) fn runtime_targets(
    parsed: &super::ParsedTool,
    config: &AppConfig,
    program: &Program,
) -> Result<(), String> {
    match parsed {
        super::ParsedTool::Evaluate {
            function,
            source,
            python,
        } => {
            exists(program, function)?;
            let ops = crate::parser::parse(source).map_err(|error| error.to_string())?;
            if *python {
                super::literals::require_python_entry(program, function)?;
                crate::python_frontend::validate_runtime_arguments(&ops, program)?;
            }
            for op in ops {
                if let Operation::Eval(eval) = op {
                    super::literals::check_runtime(&eval.input, program)?;
                }
            }
            Ok(())
        }
        super::ParsedTool::Test {
            targets,
            source,
            python,
        } => {
            targets
                .iter()
                .try_for_each(|target| exists(program, target))?;
            let ops = crate::parser::parse(source).map_err(|error| error.to_string())?;
            if *python {
                for target in targets {
                    super::literals::require_python_entry(program, target)?;
                }
                crate::python_frontend::validate_runtime_arguments(&ops, program)?;
            }
            for op in ops {
                if let Operation::Test(test) = op {
                    for case in test.cases {
                        super::literals::check_runtime(&case.input, program)?;
                        super::literals::check_runtime(&case.expected, program)?;
                    }
                }
            }
            Ok(())
        }
        super::ParsedTool::Mutate {
            source,
            python,
            roles,
            ..
        } => {
            let ops = crate::parser::parse(source).map_err(|e| e.to_string())?;
            check_operations(
                &ops,
                config,
                &model_name(config),
                Some((program, *python)),
                roles.as_ref(),
            )?;
            if *python {
                crate::python_frontend::validate_runtime_arguments(&ops, program)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn exists(program: &Program, target: &str) -> Result<(), String> {
    if program.get_function(target).is_some() {
        Ok(())
    } else {
        Err(format!("qualified function `{target}` was not found"))
    }
}
