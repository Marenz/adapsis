//! Conservative structural diagnostic for unresolved local/shared reads. This
//! never executes a function or claims to prove whole-program runtime validity.
use crate::ast;
use std::collections::HashSet;

pub(super) fn check_function(
    program: &ast::Program,
    module: &str,
    function: &ast::FunctionDecl,
) -> Result<(), String> {
    let mut names: HashSet<String> = function.params.iter().map(|p| p.name.clone()).collect();
    if let Some(module) = super::find_module(program, module) {
        names.extend(module.shared_vars.iter().map(|s| s.name.clone()));
    }
    body(program, &function.body, &mut names).map_err(|e| format!("{}: {e}", function.name))
}

fn expr(program: &ast::Program, value: &ast::Expr, names: &HashSet<String>) -> Result<(), String> {
    use ast::Expr::*;
    match value {
        Identifier(name)
            if !names.contains(name)
                && !matches!(name.as_str(), "true" | "false")
                && !program.is_union_variant(name) =>
        {
            Err(format!(
                "unbound identifier `{name}` in its AST scope (static source diagnostic)"
            ))
        }
        Identifier(_) | Literal(_) => Ok(()),
        Call(call) => args(program, &call.args, names),
        Binary { left, right, .. } => {
            expr(program, left, names)?;
            expr(program, right, names)
        }
        Unary { expr: value, .. } | FieldAccess { base: value, .. } => expr(program, value, names),
        StructInit { fields, .. } => {
            for field in fields {
                expr(program, &field.value, names)?;
            }
            Ok(())
        }
    }
}

fn args(
    program: &ast::Program,
    values: &[ast::Expr],
    names: &HashSet<String>,
) -> Result<(), String> {
    for value in values {
        expr(program, value, names)?;
    }
    Ok(())
}

fn pattern_names(pattern: &ast::MatchPattern, names: &mut HashSet<String>) {
    match pattern {
        ast::MatchPattern::Binding(name) => {
            names.insert(name.clone());
        }
        ast::MatchPattern::Variant { sub_patterns, .. } => {
            for pattern in sub_patterns {
                pattern_names(pattern, names);
            }
        }
        _ => {}
    }
}

fn body(
    program: &ast::Program,
    statements: &[ast::Statement],
    names: &mut HashSet<String>,
) -> Result<(), String> {
    use ast::StatementKind::*;
    for statement in statements {
        match &statement.kind {
            Let { name, value, .. } => {
                expr(program, value, names)?;
                names.insert(name.clone());
            }
            Set { name, value } => {
                expr(program, &ast::Expr::Identifier(name.clone()), names)?;
                expr(program, value, names)?;
            }
            Await { name, call, .. } => {
                args(program, &call.args, names)?;
                names.insert(name.clone());
            }
            Call { binding, call } | Spawn { binding, call } => {
                args(program, &call.args, names)?;
                if let Some(binding) = binding {
                    names.insert(binding.name.clone());
                }
            }
            Check { condition, .. } => expr(program, condition, names)?,
            Return { value } | Yield { value } => expr(program, value, names)?,
            Branch {
                condition,
                then_body,
                else_body,
            } => {
                expr(program, condition, names)?;
                body(program, then_body, &mut names.clone())?;
                body(program, else_body, &mut names.clone())?;
            }
            While {
                condition,
                body: inner,
            } => {
                expr(program, condition, names)?;
                body(program, inner, &mut names.clone())?;
            }
            Each {
                iterator,
                binding,
                body: inner,
            } => {
                expr(program, iterator, names)?;
                let mut local = names.clone();
                local.insert(binding.name.clone());
                body(program, inner, &mut local)?;
            }
            Match { expr: value, arms } => {
                expr(program, value, names)?;
                for arm in arms {
                    let mut local = names.clone();
                    local.extend(arm.bindings.iter().cloned());
                    if let Some(patterns) = &arm.patterns {
                        for pattern in patterns {
                            pattern_names(pattern, &mut local);
                        }
                    }
                    body(program, &arm.body, &mut local)?;
                }
            }
            Source(ast::SourceOp::Add {
                source_type: ast::SourceType::Timer(value),
                ..
            })
            | Source(ast::SourceOp::Replace {
                source_type: ast::SourceType::Timer(value),
                ..
            }) => expr(program, value, names)?,
            Event(ast::EventOp::Emit { value, .. }) => expr(program, value, names)?,
            Source(_) | Event(_) => {}
        }
    }
    Ok(())
}
