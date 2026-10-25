//! Validate newly supplied executable syntax, never transitive trusted bodies.
use super::{AppConfig, PermissionLevel, Program, model_name, module_of, resolve_module};
use crate::parser::{Expr, Operation};

pub(super) use super::literals::literal;
type Runtime<'a> = (&'a Program, bool);

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
) -> Result<(), String> {
    if target.contains('.') {
        if resolve_module(config, model, module_of(target)) < PermissionLevel::Execute {
            return Err(format!(
                "permission denied for authored reference `{target}` (need Execute)"
            ));
        }
    } else {
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

fn expression(
    expr: &Expr,
    config: &AppConfig,
    model: &str,
    program: Option<Runtime<'_>>,
) -> Result<(), String> {
    match expr {
        Expr::Call { callee, args } => {
            let callee =
                name(callee).ok_or("native mutation does not support dynamic call targets")?;
            reference(&callee, config, model, program)?;
            args.iter()
                .try_for_each(|arg| expression(arg, config, model, program))
        }
        Expr::FieldAccess { base, .. } => {
            if let Some(target) = name(expr) {
                reference(&target, config, model, program)?;
            }
            expression(base, config, model, program)
        }
        Expr::Binary { left, right, .. } => {
            expression(left, config, model, program)?;
            expression(right, config, model, program)
        }
        Expr::Unary { expr, .. } | Expr::Cast { expr, .. } => {
            expression(expr, config, model, program)
        }
        Expr::StructLiteral(fields) => fields
            .iter()
            .try_for_each(|field| expression(&field.value, config, model, program)),
        _ => Ok(()),
    }
}

pub(super) fn check_operations(
    ops: &[Operation],
    config: &AppConfig,
    model: &str,
    program: Option<Runtime<'_>>,
) -> Result<(), String> {
    for op in ops {
        let check = |expr: &Expr| expression(expr, config, model, program);
        let body = |ops: &[Operation]| check_operations(ops, config, model, program);
        match op {
            Operation::Module(decl) => body(&decl.body)?,
            Operation::Function(decl) => body(&decl.body)?,
            Operation::SharedVar(decl) => {
                literal(&decl.default)?;
                if let Some((program, _)) = program {
                    super::literals::check_runtime(&decl.default, program)?;
                }
            }
            Operation::Type(_) | Operation::Doc { .. } => {}
            Operation::Let(decl) => check(&decl.expr)?,
            Operation::Set(decl) => {
                check(&decl.expr)?;
                if decl.name.contains('.') {
                    reference(&decl.name, config, model, program)?;
                }
            }
            Operation::Call(decl) => check(&decl.expr)?,
            Operation::Check(decl) => check(&decl.expr)?,
            Operation::Return(decl) => check(&decl.expr)?,
            Operation::Await(decl) => check(&decl.call)?,
            Operation::Spawn(decl) => check(&decl.call)?,
            Operation::If(decl) => {
                check(&decl.condition)?;
                body(&decl.then_body)?;
                body(&decl.else_body)?;
                for (expr, ops) in &decl.elif_branches {
                    check(expr)?;
                    body(ops)?;
                }
            }
            Operation::While(decl) => {
                check(&decl.condition)?;
                body(&decl.body)?;
            }
            Operation::Each(decl) => {
                check(&decl.collection)?;
                body(&decl.body)?;
            }
            Operation::Match(decl) => {
                check(&decl.expr)?;
                for arm in &decl.arms {
                    body(&arm.body)?;
                }
            }
            Operation::Branch(decl) => {
                check(&decl.expr)?;
                for arm in &decl.arms {
                    check(&arm.pattern)?;
                    reference(&arm.target, config, model, program)?;
                }
            }
            _ => {
                return Err(format!(
                    "native mutation does not support authored `{}` inside functions",
                    super::describe_op(op)
                ));
            }
        }
    }
    Ok(())
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
                super::literals::require_python_environment(program)?;
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
                super::literals::require_python_environment(program)?;
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
        super::ParsedTool::Mutate { source, python, .. } => {
            let ops = crate::parser::parse(source).map_err(|e| e.to_string())?;
            check_operations(&ops, config, &model_name(config), Some((program, *python)))?;
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
