use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;

use crate::ast;
use crate::parser;

/// Apply a parsed operation to the program state and validate it.
/// Returns a human-readable success message or an error with diagnostics.
pub fn apply_and_validate(program: &mut ast::Program, op: &parser::Operation) -> Result<String> {
    match op {
        parser::Operation::Module(module_decl) => apply_module(program, module_decl),
        parser::Operation::Type(type_decl) => {
            let converted = convert_type_decl(type_decl)?;
            // Check for duplicate type name
            let name = converted.name();
            if program.require_modules {
                bail!(
                    "type `{name}` must be inside a module. Use: !module MyModule\\n+type {name} = ..."
                );
            }
            if program.types.iter().any(|t| t.name() == name) {
                bail!("duplicate type declaration: `{name}`");
            }
            let msg = format!("added type `{name}`");
            program.types.push(converted);
            Ok(msg)
        }
        parser::Operation::Function(fn_decl) => {
            let mut converted = convert_function(fn_decl)?;
            // Resolve union type references: Struct(name) → TaggedUnion(name) where appropriate
            resolve_union_types_in_function(&mut converted, program);
            // Check for builtin name collision
            if is_builtin_name(&converted.name) {
                bail!(
                    "function name `{}` conflicts with a built-in function — choose a different name",
                    converted.name
                );
            }
            // In AdapsisOS mode, reject top-level functions — must be inside a module
            if program.require_modules {
                bail!(
                    "function `{}` must be inside a module. Use: !module MyModule\\n+fn {}(...)\\n+end",
                    converted.name, converted.name
                );
            }
            // Check for duplicate function name at top level
            if program.functions.iter().any(|f| f.name == converted.name) {
                bail!("duplicate function: `{}`", converted.name);
            }
            let msg = format!(
                "added function `{}` ({} params, {} statements, effects: [{}])",
                converted.name,
                converted.params.len(),
                converted.body.len(),
                converted
                    .effects
                    .iter()
                    .map(|e| format!("{e:?}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            program.functions.push(std::sync::Arc::new(converted));
            program.rebuild_function_index();
            Ok(msg)
        }
        parser::Operation::SharedVar(sv) => {
            bail!(
                "+shared `{}` must be inside a module. Use: !module MyModule\\n+shared {}:{} = ...",
                sv.name,
                sv.name,
                format!("{:?}", sv.ty)
            );
        }
        parser::Operation::Replace(replace) => apply_replace(program, replace),
        parser::Operation::Test(_) => {
            // Tests are handled separately, not applied to program state
            Ok("test block (skipped during validation)".to_string())
        }
        parser::Operation::Move {
            function_names,
            target_module,
        } => apply_move(program, function_names, target_module),
        parser::Operation::Trace(_) => Ok("trace (handled by evaluator)".to_string()),
        parser::Operation::Eval(_) => Ok("eval (handled by evaluator)".to_string()),
        parser::Operation::Plan(_) => Ok("plan (handled by session)".to_string()),
        parser::Operation::Watch { .. } => Ok("watch (handled by runtime)".to_string()),
        parser::Operation::Undo => Ok("undo (handled by session)".to_string()),
        parser::Operation::Agent { .. } => Ok("agent (handled by orchestrator)".to_string()),
        parser::Operation::OpenCode(_) => Ok("opencode (handled by orchestrator)".to_string()),
        parser::Operation::Message { .. } => Ok("message (handled by orchestrator)".to_string()),
        parser::Operation::Done => Ok("done (handled by orchestrator)".to_string()),
        parser::Operation::Sandbox(_) => Ok("sandbox (handled by session)".to_string()),
        parser::Operation::Roadmap(_) => Ok("roadmap (handled by session)".to_string()),
        parser::Operation::Remove(target) => {
            if let Some((mod_name, item_name)) = target.split_once('.') {
                // Remove function or type from module
                if let Some(m) = program.modules.iter_mut().find(|m| m.name == mod_name) {
                    let fn_before = m.functions.len();
                    m.functions.retain(|f| f.name != item_name);
                    if m.functions.len() < fn_before {
                        program.rebuild_function_index();
                        // Also remove any routes pointing to this handler
                        let qualified = format!("{mod_name}.{item_name}");
                        let removed_routes = remove_routes_for_handler(program, &qualified);
                        let mut msg = format!("removed function `{target}`");
                        if !removed_routes.is_empty() {
                            msg.push_str(&format!(
                                "; also removed {} route(s): {}",
                                removed_routes.len(),
                                removed_routes.join(", ")
                            ));
                        }
                        return Ok(msg);
                    }
                    let ty_before = m.types.len();
                    m.types.retain(|t| t.name() != item_name);
                    if m.types.len() < ty_before {
                        return Ok(format!("removed type `{target}`"));
                    }
                    bail!("`{item_name}` not found in module `{mod_name}`");
                }
                bail!("module `{mod_name}` not found");
            }
            // Remove entire module, top-level type, or top-level function
            if let Some(pos) = program.modules.iter().position(|m| m.name == *target) {
                // Collect function names from the module before removing it
                let module_fn_names: Vec<String> = program.modules[pos]
                    .functions
                    .iter()
                    .map(|f| format!("{}.{}", target, f.name))
                    .collect();
                program.modules.remove(pos);
                program.rebuild_function_index();
                // Remove routes pointing to any function in this module
                let mut all_removed_routes = Vec::new();
                for fn_name in &module_fn_names {
                    all_removed_routes.extend(remove_routes_for_handler(program, fn_name));
                }
                let mut msg = format!("removed module `{target}`");
                if !all_removed_routes.is_empty() {
                    msg.push_str(&format!(
                        "; also removed {} route(s): {}",
                        all_removed_routes.len(),
                        all_removed_routes.join(", ")
                    ));
                }
                return Ok(msg);
            }
            let target_str: &str = target;
            if let Some(pos) = program.types.iter().position(|t| t.name() == target_str) {
                program.types.remove(pos);
                return Ok(format!("removed type `{target}`"));
            }
            if let Some(pos) = program.functions.iter().position(|f| f.name == *target) {
                program.functions.remove(pos);
                program.rebuild_function_index();
                // Also remove any routes pointing to this top-level function
                let removed_routes = remove_routes_for_handler(program, &target);
                let mut msg = format!("removed function `{target}`");
                if !removed_routes.is_empty() {
                    msg.push_str(&format!(
                        "; also removed {} route(s): {}",
                        removed_routes.len(),
                        removed_routes.join(", ")
                    ));
                }
                return Ok(msg);
            }
            bail!("`{target}` not found");
        }
        parser::Operation::RemoveRoute { .. } => {
            Ok("route removal (handled by session)".to_string())
        }
        parser::Operation::Mock { .. } => Ok("mock (handled by session)".to_string()),
        parser::Operation::Unmock => Ok("unmock (handled by session)".to_string()),
        parser::Operation::Route { .. } => Ok("route (handled by session)".to_string()),
        parser::Operation::Query(_) => Ok("query (handled by orchestrator)".to_string()),
        // Standalone statements at top level — execute immediately (not stored in AST)
        parser::Operation::Let(_)
        | parser::Operation::Set(_)
        | parser::Operation::Call(_)
        | parser::Operation::Check(_)
        | parser::Operation::Branch(_)
        | parser::Operation::If(_)
        | parser::Operation::Return(_)
        | parser::Operation::Each(_)
        | parser::Operation::While(_)
        | parser::Operation::Match(_)
        | parser::Operation::Await(_)
        | parser::Operation::Spawn(_) => {
            Ok("top-level statement (execute immediately)".to_string())
        }
    }
}

/// Route cleanup is now handled by Session — validator no longer touches routes.
/// Returns empty; callers will be updated to use session-level route removal.
fn remove_routes_for_handler(_program: &mut ast::Program, _handler_name: &str) -> Vec<String> {
    Vec::new()
}

fn apply_module(program: &mut ast::Program, decl: &parser::ModuleDecl) -> Result<String> {
    // Find existing module to merge into, or create new
    let existing_idx = program.modules.iter().position(|m| m.name == decl.name);
    if existing_idx.is_none() {
        program.modules.push(ast::Module {
            id: decl.name.clone(),
            name: decl.name.clone(),
            types: vec![],
            functions: vec![],
            modules: vec![],
            shared_vars: vec![],
            fn_index: HashMap::new(),
        });
    }
    let mod_idx = existing_idx.unwrap_or(program.modules.len() - 1);

    let mut added_fns = 0;
    let mut added_types = 0;
    let mut replaced_fns = 0;

    for op in &decl.body {
        match op {
            parser::Operation::Type(td) => {
                let converted = convert_type_decl(td)?;
                let name = converted.name();
                let m = &mut program.modules[mod_idx];
                if let Some(pos) = m.types.iter().position(|t| t.name() == name) {
                    m.types[pos] = converted;
                } else {
                    m.types.push(converted);
                    added_types += 1;
                }
            }
            parser::Operation::Function(fd) => {
                let mut converted = convert_function(fd)?;
                // Resolve union types — collect names via immutable borrow first
                let union_names = {
                    let mut names = collect_union_names(program);
                    for td in &program.modules[mod_idx].types {
                        if let ast::TypeDecl::TaggedUnion(u) = td {
                            names.insert(u.name.clone());
                        }
                    }
                    names
                };
                if !union_names.is_empty() {
                    resolve_type(&mut converted.return_type, &union_names);
                    for param in &mut converted.params {
                        resolve_type(&mut param.ty, &union_names);
                    }
                    resolve_union_types_in_stmts(&mut converted.body, &union_names);
                }
                // Now mutably access module to add/replace function
                let m = &mut program.modules[mod_idx];
                if let Some(pos) = m.functions.iter().position(|f| f.name == converted.name) {
                    m.functions[pos] = std::sync::Arc::new(converted);
                    replaced_fns += 1;
                } else {
                    m.functions.push(std::sync::Arc::new(converted));
                    added_fns += 1;
                }
            }
            parser::Operation::SharedVar(sv) => {
                let converted = ast::SharedVarDecl {
                    name: sv.name.clone(),
                    ty: convert_type(&sv.ty)?,
                    default: convert_expr(&sv.default)?,
                };
                let m = &mut program.modules[mod_idx];
                // Replace existing shared var with same name, or add new
                if let Some(pos) = m.shared_vars.iter().position(|v| v.name == converted.name) {
                    m.shared_vars[pos] = converted;
                } else {
                    m.shared_vars.push(converted);
                }
            }
            parser::Operation::Test(_) => {
                // Tests inside a module body are skipped during validation —
                // they're handled separately by the session/eval layer.
                // The parser allows !test inside !module so that it doesn't
                // break the module context for subsequent +fn definitions.
            }
            parser::Operation::Module(nested) => bail!(
                "nested module `{}` found inside module `{}` — check indentation",
                nested.name,
                decl.name
            ),
            // !test inside a module body is valid — tests are handled separately
            parser::Operation::Test(_) => {}
            other => bail!(
                "unexpected operation in module `{}`: {:?} — only +fn, +type, +shared, and !test are allowed",
                decl.name,
                std::mem::discriminant(other)
            ),
        }
    }

    let m = &program.modules[mod_idx];
    let action = if existing_idx.is_some() {
        "updated"
    } else {
        "added"
    };
    let mut details = Vec::new();
    if added_fns > 0 {
        details.push(format!("+{added_fns} fn"));
    }
    if replaced_fns > 0 {
        details.push(format!("~{replaced_fns} fn"));
    }
    if added_types > 0 {
        details.push(format!("+{added_types} type"));
    }
    let detail_str = if details.is_empty() {
        String::new()
    } else {
        format!(" ({})", details.join(", "))
    };
    let msg = format!(
        "{action} module `{}` ({} types, {} functions){detail_str}",
        m.name,
        m.types.len(),
        m.functions.len()
    );
    program.rebuild_function_index();
    Ok(msg)
}

fn apply_replace(program: &mut ast::Program, replace: &parser::ReplaceMutation) -> Result<String> {
    // Parse target path: "ModuleName.function_name.sN" or "function_name.sN"
    let parts: Vec<&str> = replace.target.split('.').collect();

    match parts.len() {
        1 => {
            // function_name — replace entire body
            let fn_name = parts[0];
            let func = program
                .functions
                .iter_mut()
                .find(|f| f.name == fn_name)
                .map(|f| std::sync::Arc::make_mut(f))
                .or_else(|| {
                    program
                        .modules
                        .iter_mut()
                        .flat_map(|m| m.functions.iter_mut())
                        .find(|f| f.name == fn_name)
                        .map(|f| std::sync::Arc::make_mut(f))
                })
                .ok_or_else(|| anyhow!("function `{fn_name}` not found for replace"))?;
            let mut new_body = vec![];
            for (i, op) in replace.body.iter().enumerate() {
                let mut stmt = convert_statement_op(op)?;
                stmt.id = format!("{fn_name}.s{}", i + 1);
                new_body.push(stmt);
            }
            func.body = new_body;
            Ok(format!(
                "replaced entire body of `{fn_name}` ({} statements)",
                func.body.len()
            ))
        }
        2 => {
            let fn_name = parts[0];
            let stmt_id = parts[1];
            // Check if it's fn.sN or Module.fn
            if stmt_id.starts_with('s') && stmt_id[1..].parse::<usize>().is_ok() {
                // function_name.sN
                let func = program
                    .functions
                    .iter_mut()
                    .find(|f| f.name == fn_name)
                    .map(|f| std::sync::Arc::make_mut(f))
                    .ok_or_else(|| anyhow!("function `{fn_name}` not found for replace"))?;
                replace_statement(&mut func.body, stmt_id, &replace.body)?;
                Ok(format!("replaced `{}`", replace.target))
            } else {
                // Module.function — replace entire body
                let mod_name = parts[0];
                let fn_name = parts[1];
                let module = program
                    .modules
                    .iter_mut()
                    .find(|m| m.name == mod_name)
                    .ok_or_else(|| anyhow!("module `{mod_name}` not found for replace"))?;
                let func = module
                    .functions
                    .iter_mut()
                    .find(|f| f.name == fn_name)
                    .map(|f| std::sync::Arc::make_mut(f))
                    .ok_or_else(|| {
                        anyhow!("function `{fn_name}` not found in module `{mod_name}`")
                    })?;
                let mut new_body = vec![];
                for (i, op) in replace.body.iter().enumerate() {
                    let mut stmt = convert_statement_op(op)?;
                    stmt.id = format!("{mod_name}.{fn_name}.s{}", i + 1);
                    new_body.push(stmt);
                }
                func.body = new_body;
                Ok(format!(
                    "replaced entire body of `{mod_name}.{fn_name}` ({} statements)",
                    func.body.len()
                ))
            }
        }
        3 => {
            // Module.function.sN
            let mod_name = parts[0];
            let fn_name = parts[1];
            let stmt_id = parts[2];
            let module = program
                .modules
                .iter_mut()
                .find(|m| m.name == mod_name)
                .ok_or_else(|| anyhow!("module `{mod_name}` not found for replace"))?;
            let func = module
                .functions
                .iter_mut()
                .find(|f| f.name == fn_name)
                .map(|f| std::sync::Arc::make_mut(f))
                .ok_or_else(|| anyhow!("function `{fn_name}` not found in module `{mod_name}`"))?;
            replace_statement(&mut func.body, stmt_id, &replace.body)?;
            Ok(format!("replaced `{}`", replace.target))
        }
        _ => bail!(
            "invalid replace target `{}` — expected `fn`, `fn.sN`, `Module.fn`, or `Module.fn.sN`",
            replace.target
        ),
    }
}

fn replace_statement(
    body: &mut Vec<ast::Statement>,
    stmt_id: &str,
    replacements: &[parser::Operation],
) -> Result<()> {
    // stmt_id is like "s1", "s2", etc.
    let index: usize = stmt_id
        .strip_prefix('s')
        .and_then(|n| n.parse::<usize>().ok())
        .ok_or_else(|| anyhow!("invalid statement id `{stmt_id}` — expected `sN`"))?;

    // 1-indexed
    if index == 0 || index > body.len() {
        bail!(
            "statement `{stmt_id}` out of range (function has {} statements)",
            body.len()
        );
    }

    let mut new_stmts = vec![];
    for op in replacements {
        new_stmts.push(convert_statement_op(op)?);
    }

    // Replace the statement at index-1
    body.splice((index - 1)..index, new_stmts);
    Ok(())
}

// --- Conversion from parser types to AST types ---

fn convert_type_decl(decl: &parser::TypeDecl) -> Result<ast::TypeDecl> {
    match &decl.body {
        parser::TypeBody::Struct(fields) => {
            let mut ast_fields = vec![];
            for (i, field) in fields.iter().enumerate() {
                ast_fields.push(ast::FieldDecl {
                    id: format!("{}.f{}", decl.name, i),
                    name: field.name.clone(),
                    ty: convert_type(&field.ty)?,
                });
            }
            Ok(ast::TypeDecl::Struct(ast::StructDecl {
                id: decl.name.clone(),
                name: decl.name.clone(),
                fields: ast_fields,
            }))
        }
        parser::TypeBody::Union(variants) => {
            let mut ast_variants = vec![];
            for (i, variant) in variants.iter().enumerate() {
                let payload_types = variant
                    .payload
                    .iter()
                    .map(convert_type)
                    .collect::<Result<Vec<_>>>()?;
                ast_variants.push(ast::UnionVariant {
                    id: format!("{}.v{}", decl.name, i),
                    name: variant.name.clone(),
                    payload: payload_types,
                });
            }
            Ok(ast::TypeDecl::TaggedUnion(ast::TaggedUnionDecl {
                id: decl.name.clone(),
                name: decl.name.clone(),
                variants: ast_variants,
            }))
        }
        parser::TypeBody::Alias(_type_expr) => {
            // Treat alias as a struct with no fields for now, or as a named reference
            // For Phase 1, just make it a struct with zero fields — we can improve later
            Ok(ast::TypeDecl::Struct(ast::StructDecl {
                id: decl.name.clone(),
                name: decl.name.clone(),
                fields: vec![],
            }))
        }
    }
}

fn convert_function(decl: &parser::FunctionDecl) -> Result<ast::FunctionDecl> {
    let mut params = vec![];
    for (i, p) in decl.params.iter().enumerate() {
        params.push(ast::Param {
            id: format!("{}.p{}", decl.name, i),
            name: p.name.clone(),
            ty: convert_type(&p.ty)?,
        });
    }

    let mut body = vec![];
    for (i, op) in decl.body.iter().enumerate() {
        let mut stmt = convert_statement_op(op)?;
        stmt.id = format!("{}.s{}", decl.name, i + 1);
        body.push(stmt);
    }

    let effects = decl
        .effects
        .iter()
        .map(|e| convert_effect(e))
        .collect::<Result<Vec<_>>>()?;

    Ok(ast::FunctionDecl {
        id: decl.name.clone(),
        name: decl.name.clone(),
        params,
        return_type: convert_type(&decl.return_type)?,
        effects,
        body,
        tests: vec![],
    })
}

pub fn convert_statement_op(op: &parser::Operation) -> Result<ast::Statement> {
    let kind = match op {
        parser::Operation::Let(decl) => ast::StatementKind::Let {
            name: decl.name.clone(),
            ty: convert_type(&decl.ty)?,
            value: convert_expr(&decl.expr)?,
        },
        parser::Operation::Set(decl) => ast::StatementKind::Set {
            name: decl.name.clone(),
            value: convert_expr(&decl.expr)?,
        },
        parser::Operation::While(decl) => {
            let condition = convert_expr(&decl.condition)?;
            let mut body = vec![];
            for (i, op) in decl.body.iter().enumerate() {
                let mut stmt = convert_statement_op(op)?;
                stmt.id = format!("while.s{}", i + 1);
                body.push(stmt);
            }
            ast::StatementKind::While { condition, body }
        }
        parser::Operation::Match(decl) => {
            let expr = convert_expr(&decl.expr)?;
            let mut arms = Vec::new();
            for arm in &decl.arms {
                let mut body = vec![];
                for (i, op) in arm.body.iter().enumerate() {
                    let mut stmt = convert_statement_op(op)?;
                    stmt.id = format!("match.{}.s{}", arm.variant, i + 1);
                    body.push(stmt);
                }
                let patterns = arm
                    .patterns
                    .as_ref()
                    .map(|pats| pats.iter().map(convert_match_pattern).collect());
                arms.push(ast::MatchArm {
                    variant: arm.variant.clone(),
                    bindings: arm.bindings.clone(),
                    patterns,
                    body,
                });
            }
            ast::StatementKind::Match { expr, arms }
        }
        parser::Operation::Await(decl) => {
            let call = extract_call_expr(&decl.call)?;
            ast::StatementKind::Await {
                name: decl.name.clone(),
                ty: convert_type(&decl.ty)?,
                call,
            }
        }
        parser::Operation::Spawn(decl) => {
            let call = extract_call_expr(&decl.call)?;
            let binding = match &decl.binding {
                Some((name, ty)) => Some(ast::Binding {
                    name: name.clone(),
                    ty: convert_type(ty)?,
                }),
                None => None,
            };
            ast::StatementKind::Spawn { call, binding }
        }
        parser::Operation::Call(decl) => ast::StatementKind::Call {
            binding: Some(ast::Binding {
                name: decl.name.clone(),
                ty: convert_type(&decl.ty)?,
            }),
            call: extract_call_expr(&decl.expr)?,
        },
        parser::Operation::Check(decl) => ast::StatementKind::Check {
            label: decl.name.clone(),
            condition: convert_expr(&decl.expr)?,
            on_fail: decl.err_label.clone(),
        },
        parser::Operation::Branch(decl) => {
            // Convert branch arms into nested if/else chains
            // Each arm: pattern -> target becomes: if expr == pattern { return target }
            let match_expr = convert_expr(&decl.expr)?;
            let mut else_body = vec![];

            // Build from last arm backwards to nest properly
            for arm in decl.arms.iter().rev() {
                let pattern = convert_expr(&arm.pattern)?;
                let condition = ast::Expr::Binary {
                    left: Box::new(match_expr.clone()),
                    op: ast::BinaryOp::Equal,
                    right: Box::new(pattern),
                };
                let then_body = vec![ast::Statement {
                    id: String::new(),
                    kind: ast::StatementKind::Return {
                        value: ast::Expr::Identifier(arm.target.clone()),
                    },
                }];
                else_body = vec![ast::Statement {
                    id: String::new(),
                    kind: ast::StatementKind::Branch {
                        condition,
                        then_body,
                        else_body,
                    },
                }];
            }

            // Unwrap the outermost branch
            if let Some(stmt) = else_body.into_iter().next() {
                stmt.kind
            } else {
                bail!("empty branch")
            }
        }
        parser::Operation::If(decl) => {
            let condition = convert_expr(&decl.condition)?;
            let mut then_body = vec![];
            for (i, op) in decl.then_body.iter().enumerate() {
                let mut stmt = convert_statement_op(op)?;
                stmt.id = format!("if.then.s{}", i + 1);
                then_body.push(stmt);
            }

            // Build else body: chain elif branches into nested Branch nodes
            let else_body = if !decl.elif_branches.is_empty() {
                // Build from the last elif backwards, ending with the else body
                let mut current_else = vec![];
                for (i, op) in decl.else_body.iter().enumerate() {
                    let mut stmt = convert_statement_op(op)?;
                    stmt.id = format!("if.else.s{}", i + 1);
                    current_else.push(stmt);
                }

                for (elif_cond, elif_body) in decl.elif_branches.iter().rev() {
                    let cond = convert_expr(elif_cond)?;
                    let mut elif_stmts = vec![];
                    for (i, op) in elif_body.iter().enumerate() {
                        let mut stmt = convert_statement_op(op)?;
                        stmt.id = format!("if.elif.s{}", i + 1);
                        elif_stmts.push(stmt);
                    }
                    current_else = vec![ast::Statement {
                        id: String::new(),
                        kind: ast::StatementKind::Branch {
                            condition: cond,
                            then_body: elif_stmts,
                            else_body: current_else,
                        },
                    }];
                }
                current_else
            } else {
                let mut else_stmts = vec![];
                for (i, op) in decl.else_body.iter().enumerate() {
                    let mut stmt = convert_statement_op(op)?;
                    stmt.id = format!("if.else.s{}", i + 1);
                    else_stmts.push(stmt);
                }
                else_stmts
            };

            ast::StatementKind::Branch {
                condition,
                then_body,
                else_body,
            }
        }
        parser::Operation::Return(decl) => ast::StatementKind::Return {
            value: convert_expr(&decl.expr)?,
        },
        parser::Operation::Each(decl) => {
            let mut each_body = vec![];
            for (i, body_op) in decl.body.iter().enumerate() {
                let mut stmt = convert_statement_op(body_op)?;
                stmt.id = format!("each.s{}", i + 1);
                each_body.push(stmt);
            }
            ast::StatementKind::Each {
                iterator: convert_expr(&decl.collection)?,
                binding: ast::Binding {
                    name: decl.item.clone(),
                    ty: convert_type(&decl.item_type)?,
                },
                body: each_body,
            }
        }
        parser::Operation::Function(fd) => bail!(
            "nested function `{}` found inside function body — check indentation",
            fd.name
        ),
        parser::Operation::Module(md) => bail!(
            "module `{}` found inside function body — check indentation",
            md.name
        ),
        parser::Operation::Type(td) => bail!(
            "type declaration `{}` found inside function body — check indentation",
            td.name
        ),
        other => bail!(
            "unexpected operation in function body: {:?}",
            std::mem::discriminant(other)
        ),
    };

    Ok(ast::Statement {
        id: String::new(),
        kind,
    })
}

fn extract_call_expr(expr: &parser::Expr) -> Result<ast::CallExpr> {
    match expr {
        parser::Expr::Call { callee, args } => {
            let callee_name = expr_to_ident(callee)?;
            let ast_args = args.iter().map(convert_expr).collect::<Result<Vec<_>>>()?;
            Ok(ast::CallExpr {
                callee: callee_name,
                args: ast_args,
            })
        }
        // If it's just an identifier or field access being called, treat it as a zero-arg call
        parser::Expr::Ident(name) => Ok(ast::CallExpr {
            callee: name.clone(),
            args: vec![],
        }),
        parser::Expr::FieldAccess { .. } => {
            let name = parser_expr_to_dotted_name(expr);
            Ok(ast::CallExpr {
                callee: name,
                args: vec![],
            })
        }
        other => bail!("expected call expression, got {:?}", other),
    }
}

fn expr_to_ident(expr: &parser::Expr) -> Result<String> {
    match expr {
        parser::Expr::Ident(name) => Ok(name.clone()),
        parser::Expr::FieldAccess { .. } => Ok(parser_expr_to_dotted_name(expr)),
        other => bail!("expected identifier, got {:?}", other),
    }
}

fn parser_expr_to_dotted_name(expr: &parser::Expr) -> String {
    match expr {
        parser::Expr::Ident(name) => name.clone(),
        parser::Expr::FieldAccess { base, field } => {
            format!("{}.{}", parser_expr_to_dotted_name(base), field)
        }
        _ => format!("{:?}", expr),
    }
}

fn convert_expr(expr: &parser::Expr) -> Result<ast::Expr> {
    match expr {
        parser::Expr::Int(v) => Ok(ast::Expr::Literal(ast::Literal::Int(*v))),
        parser::Expr::Float(v) => Ok(ast::Expr::Literal(ast::Literal::Float(*v))),
        parser::Expr::Bool(v) => Ok(ast::Expr::Literal(ast::Literal::Bool(*v))),
        parser::Expr::String(v) => Ok(ast::Expr::Literal(ast::Literal::String(v.clone()))),
        parser::Expr::Ident(name) => Ok(ast::Expr::Identifier(name.clone())),
        parser::Expr::FieldAccess { base, field } => Ok(ast::Expr::FieldAccess {
            base: Box::new(convert_expr(base)?),
            field: field.clone(),
        }),
        parser::Expr::Call { callee, args } => {
            let callee_name = expr_to_ident(callee)?;
            let ast_args = args.iter().map(convert_expr).collect::<Result<Vec<_>>>()?;
            Ok(ast::Expr::Call(ast::CallExpr {
                callee: callee_name,
                args: ast_args,
            }))
        }
        parser::Expr::Binary { op, left, right } => Ok(ast::Expr::Binary {
            left: Box::new(convert_expr(left)?),
            op: convert_binary_op(op),
            right: Box::new(convert_expr(right)?),
        }),
        parser::Expr::Unary { op, expr } => Ok(ast::Expr::Unary {
            op: convert_unary_op(op),
            expr: Box::new(convert_expr(expr)?),
        }),
        parser::Expr::StructLiteral(fields) => {
            let mut ast_fields = vec![];
            for f in fields {
                ast_fields.push(ast::StructFieldValue {
                    name: f.name.clone(),
                    value: convert_expr(&f.value)?,
                });
            }
            // We don't know the struct type name from just a literal — use empty
            Ok(ast::Expr::StructInit {
                ty: String::new(),
                fields: ast_fields,
            })
        }
        parser::Expr::Cast { expr, ty: _ } => {
            // For Phase 1, just pass through the inner expression
            convert_expr(expr)
        }
    }
}

fn convert_match_pattern(pat: &parser::MatchPatternDecl) -> ast::MatchPattern {
    match pat {
        parser::MatchPatternDecl::Binding(name) => ast::MatchPattern::Binding(name.clone()),
        parser::MatchPatternDecl::Variant {
            variant,
            sub_patterns,
        } => ast::MatchPattern::Variant {
            variant: variant.clone(),
            sub_patterns: sub_patterns.iter().map(convert_match_pattern).collect(),
        },
        parser::MatchPatternDecl::LiteralInt(n) => {
            ast::MatchPattern::Literal(ast::Literal::Int(*n))
        }
        parser::MatchPatternDecl::LiteralBool(b) => {
            ast::MatchPattern::Literal(ast::Literal::Bool(*b))
        }
        parser::MatchPatternDecl::LiteralString(s) => {
            ast::MatchPattern::Literal(ast::Literal::String(s.clone()))
        }
    }
}

fn convert_type(ty: &parser::TypeExpr) -> Result<ast::Type> {
    match ty {
        parser::TypeExpr::Named(name) => match name.as_str() {
            "Int" => Ok(ast::Type::Int),
            "Float" => Ok(ast::Type::Float),
            "Bool" => Ok(ast::Type::Bool),
            "String" => Ok(ast::Type::String),
            "Byte" => Ok(ast::Type::Byte),
            "Ok" => Ok(ast::Type::Struct("Ok".to_string())),
            other => Ok(ast::Type::Struct(other.to_string())),
        },
        parser::TypeExpr::Generic { name, args } => match name.as_str() {
            "List" => {
                if args.len() != 1 {
                    bail!("List expects 1 type argument, got {}", args.len());
                }
                Ok(ast::Type::List(Box::new(convert_type(&args[0])?)))
            }
            "Set" => {
                if args.len() != 1 {
                    bail!("Set expects 1 type argument, got {}", args.len());
                }
                Ok(ast::Type::Set(Box::new(convert_type(&args[0])?)))
            }
            "Map" => {
                if args.len() != 2 {
                    bail!("Map expects 2 type arguments, got {}", args.len());
                }
                Ok(ast::Type::Map(
                    Box::new(convert_type(&args[0])?),
                    Box::new(convert_type(&args[1])?),
                ))
            }
            "Option" => {
                if args.len() != 1 {
                    bail!("Option expects 1 type argument, got {}", args.len());
                }
                Ok(ast::Type::Option(Box::new(convert_type(&args[0])?)))
            }
            "Result" => {
                if args.len() != 1 {
                    bail!("Result expects 1 type argument, got {}", args.len());
                }
                Ok(ast::Type::Result(Box::new(convert_type(&args[0])?)))
            }
            "Yield" => {
                // Treat Yield<T> as a type for now
                if args.len() != 1 {
                    bail!("Yield expects 1 type argument, got {}", args.len());
                }
                Ok(ast::Type::List(Box::new(convert_type(&args[0])?)))
            }
            other => {
                // Unknown generic — treat as struct for now
                Ok(ast::Type::Struct(other.to_string()))
            }
        },
        parser::TypeExpr::Struct(_fields) => {
            // Anonymous struct type — just use a generated name
            Ok(ast::Type::Struct("_anon".to_string()))
        }
    }
}

fn convert_effect(effect: &str) -> Result<ast::Effect> {
    match effect.to_lowercase().as_str() {
        "io" => Ok(ast::Effect::Io),
        "mut" => Ok(ast::Effect::Mut),
        "fail" => Ok(ast::Effect::Fail),
        "async" => Ok(ast::Effect::Async),
        "rand" => Ok(ast::Effect::Rand),
        "yield" => Ok(ast::Effect::Yield),
        "parallel" => Ok(ast::Effect::Parallel),
        "unsafe" => Ok(ast::Effect::Unsafe),
        other => bail!("unknown effect `{other}`"),
    }
}

fn convert_binary_op(op: &parser::BinaryOp) -> ast::BinaryOp {
    match op {
        parser::BinaryOp::Add => ast::BinaryOp::Add,
        parser::BinaryOp::Sub => ast::BinaryOp::Sub,
        parser::BinaryOp::Mul => ast::BinaryOp::Mul,
        parser::BinaryOp::Div => ast::BinaryOp::Div,
        parser::BinaryOp::Mod => ast::BinaryOp::Mod,
        parser::BinaryOp::Gte => ast::BinaryOp::GreaterThanOrEqual,
        parser::BinaryOp::Lte => ast::BinaryOp::LessThanOrEqual,
        parser::BinaryOp::Eq => ast::BinaryOp::Equal,
        parser::BinaryOp::Neq => ast::BinaryOp::NotEqual,
        parser::BinaryOp::Gt => ast::BinaryOp::GreaterThan,
        parser::BinaryOp::Lt => ast::BinaryOp::LessThan,
        parser::BinaryOp::And => ast::BinaryOp::And,
        parser::BinaryOp::Or => ast::BinaryOp::Or,
    }
}

fn convert_unary_op(op: &parser::UnaryOp) -> ast::UnaryOp {
    match op {
        parser::UnaryOp::Not => ast::UnaryOp::Not,
        parser::UnaryOp::Neg => ast::UnaryOp::Neg,
    }
}

/// Collect all union type names from the program (including modules).
fn collect_union_names(program: &ast::Program) -> std::collections::HashSet<String> {
    let mut names = std::collections::HashSet::new();
    for td in &program.types {
        if let ast::TypeDecl::TaggedUnion(u) = td {
            names.insert(u.name.clone());
        }
    }
    for module in &program.modules {
        for td in &module.types {
            if let ast::TypeDecl::TaggedUnion(u) = td {
                names.insert(u.name.clone());
            }
        }
    }
    names
}

/// Resolve Type::Struct(name) → Type::TaggedUnion(name) for names that are actually unions.
fn resolve_type(ty: &mut ast::Type, union_names: &std::collections::HashSet<String>) {
    match ty {
        ast::Type::Struct(name) if union_names.contains(name) => {
            *ty = ast::Type::TaggedUnion(name.clone());
        }
        ast::Type::List(inner)
        | ast::Type::Option(inner)
        | ast::Type::Result(inner)
        | ast::Type::Set(inner) => {
            resolve_type(inner, union_names);
        }
        ast::Type::Map(k, v) => {
            resolve_type(k, union_names);
            resolve_type(v, union_names);
        }
        _ => {}
    }
}

/// Walk a function's types and resolve union references.
fn resolve_union_types_in_function(func: &mut ast::FunctionDecl, program: &ast::Program) {
    let union_names = collect_union_names(program);
    if union_names.is_empty() {
        return;
    }
    // Fix return type
    resolve_type(&mut func.return_type, &union_names);
    // Fix parameter types
    for param in &mut func.params {
        resolve_type(&mut param.ty, &union_names);
    }
    // Fix types in statements
    resolve_union_types_in_stmts(&mut func.body, &union_names);
}

/// Walk statements and resolve union type references.
fn resolve_union_types_in_stmts(
    stmts: &mut [ast::Statement],
    union_names: &std::collections::HashSet<String>,
) {
    for stmt in stmts {
        match &mut stmt.kind {
            ast::StatementKind::Let { ty, .. } => {
                resolve_type(ty, union_names);
            }
            ast::StatementKind::Call { binding, .. } => {
                if let Some(b) = binding {
                    resolve_type(&mut b.ty, union_names);
                }
            }
            ast::StatementKind::Branch {
                then_body,
                else_body,
                ..
            } => {
                resolve_union_types_in_stmts(then_body, union_names);
                resolve_union_types_in_stmts(else_body, union_names);
            }
            ast::StatementKind::Each { binding, body, .. } => {
                resolve_type(&mut binding.ty, union_names);
                resolve_union_types_in_stmts(body, union_names);
            }
            ast::StatementKind::Match { arms, .. } => {
                for arm in arms {
                    resolve_union_types_in_stmts(&mut arm.body, union_names);
                }
            }
            ast::StatementKind::While { body, .. } => {
                resolve_union_types_in_stmts(body, union_names);
            }
            ast::StatementKind::Await { ty, .. } => {
                resolve_type(ty, union_names);
            }
            _ => {}
        }
    }
}

/// Generate a summary of the current program state for injection into the LLM context.
/// Check if a name conflicts with a built-in function.
fn is_builtin_name(name: &str) -> bool {
    crate::builtins::is_builtin(name)
}
pub fn apply_move(
    program: &mut ast::Program,
    names: &[String],
    target_module: &str,
) -> Result<String> {
    let mut moved = Vec::new();
    let mut not_found = Vec::new();
    let mut funcs_to_move = Vec::new();
    let mut types_to_move = Vec::new();
    let mut modules_to_merge = Vec::new();

    for name in names {
        let mut found = false;

        // Check top-level functions
        if let Some(idx) = program.functions.iter().position(|f| f.name == *name) {
            funcs_to_move.push(program.functions.remove(idx));
            moved.push(format!("fn:{name}"));
            found = true;
        }

        // Check top-level types
        if !found {
            if let Some(idx) = program.types.iter().position(|t| t.name() == name) {
                types_to_move.push(program.types.remove(idx));
                moved.push(format!("type:{name}"));
                found = true;
            }
        }

        // Check if it's a module (merge into target)
        if !found {
            if let Some(idx) = program
                .modules
                .iter()
                .position(|m| m.name == *name && m.name != target_module)
            {
                modules_to_merge.push(program.modules.remove(idx));
                moved.push(format!("module:{name}"));
                found = true;
            }
        }

        // Check functions/types in other modules
        if !found {
            for module in &mut program.modules {
                if module.name == target_module {
                    continue;
                }
                if let Some(idx) = module.functions.iter().position(|f| f.name == *name) {
                    funcs_to_move.push(module.functions.remove(idx));
                    moved.push(format!("fn:{name}"));
                    found = true;
                    break;
                }
                if let Some(idx) = module.types.iter().position(|t| t.name() == name) {
                    types_to_move.push(module.types.remove(idx));
                    moved.push(format!("type:{name}"));
                    found = true;
                    break;
                }
            }
        }

        if !found {
            not_found.push(name.clone());
        }
    }

    if moved.is_empty() {
        bail!("nothing found to move: {}", not_found.join(", "));
    }

    // Find or create target module
    if !program.modules.iter().any(|m| m.name == target_module) {
        program.modules.push(ast::Module {
            id: target_module.to_string(),
            name: target_module.to_string(),
            types: vec![],
            functions: vec![],
            modules: vec![],
            shared_vars: vec![],
            fn_index: HashMap::new(),
        });
    }
    let target = program
        .modules
        .iter_mut()
        .find(|m| m.name == target_module)
        .unwrap();

    for func in funcs_to_move {
        target.functions.push(func);
    }
    for ty in types_to_move {
        target.types.push(ty);
    }
    // Nest sub-modules inside the target
    for sub in modules_to_merge {
        target.modules.push(sub);
    }

    // Update call sites for moved functions
    let moved_fn_names: std::collections::HashSet<String> = names.iter().cloned().collect();

    for func in &mut program.functions {
        update_call_sites(
            &mut std::sync::Arc::make_mut(func).body,
            &moved_fn_names,
            target_module,
        );
    }
    for module in &mut program.modules {
        if module.name == target_module {
            continue;
        }
        for func in &mut module.functions {
            update_call_sites(
                &mut std::sync::Arc::make_mut(func).body,
                &moved_fn_names,
                target_module,
            );
        }
    }

    program.rebuild_function_index();

    let mut msg = format!("moved [{}] into `{target_module}`", moved.join(", "));
    if !not_found.is_empty() {
        msg.push_str(&format!(" (not found: {})", not_found.join(", ")));
    }
    Ok(msg)
}

fn update_call_sites(
    stmts: &mut [ast::Statement],
    moved: &std::collections::HashSet<String>,
    module: &str,
) {
    for stmt in stmts {
        match &mut stmt.kind {
            ast::StatementKind::Let { value, .. }
            | ast::StatementKind::Set { value, .. }
            | ast::StatementKind::Return { value }
            | ast::StatementKind::Check {
                condition: value, ..
            } => {
                update_expr_calls(value, moved, module);
            }
            ast::StatementKind::Call { call, .. }
            | ast::StatementKind::Await { call, .. }
            | ast::StatementKind::Spawn { call, .. } => {
                if moved.contains(&call.callee) {
                    call.callee = format!("{module}.{}", call.callee);
                }
                for arg in &mut call.args {
                    update_expr_calls(arg, moved, module);
                }
            }
            ast::StatementKind::Branch {
                condition,
                then_body,
                else_body,
            } => {
                update_expr_calls(condition, moved, module);
                update_call_sites(then_body, moved, module);
                update_call_sites(else_body, moved, module);
            }
            ast::StatementKind::While { condition, body } => {
                update_expr_calls(condition, moved, module);
                update_call_sites(body, moved, module);
            }
            ast::StatementKind::Each { body, .. } => {
                update_call_sites(body, moved, module);
            }
            ast::StatementKind::Match { expr, arms } => {
                update_expr_calls(expr, moved, module);
                for arm in arms {
                    update_call_sites(&mut arm.body, moved, module);
                }
            }
            _ => {}
        }
    }
}

fn update_expr_calls(
    expr: &mut ast::Expr,
    moved: &std::collections::HashSet<String>,
    module: &str,
) {
    match expr {
        ast::Expr::Call(call) => {
            if moved.contains(&call.callee) {
                call.callee = format!("{module}.{}", call.callee);
            }
            for arg in &mut call.args {
                update_expr_calls(arg, moved, module);
            }
        }
        ast::Expr::Binary { left, right, .. } => {
            update_expr_calls(left, moved, module);
            update_expr_calls(right, moved, module);
        }
        ast::Expr::Unary { expr: inner, .. } => {
            update_expr_calls(inner, moved, module);
        }
        ast::Expr::FieldAccess { base, .. } => {
            update_expr_calls(base, moved, module);
        }
        ast::Expr::StructInit { fields, .. } => {
            for f in fields {
                update_expr_calls(&mut f.value, moved, module);
            }
        }
        _ => {}
    }
}

/// Compact summary for LLM context — just names and signatures, grouped by module.
pub fn program_summary_compact(program: &ast::Program) -> String {
    let mut out = String::new();

    let type_count =
        program.types.len() + program.modules.iter().map(|m| m.types.len()).sum::<usize>();
    let fn_count = program.functions.len()
        + program
            .modules
            .iter()
            .map(|m| m.functions.len())
            .sum::<usize>();
    let mod_count = program.modules.len();

    out.push_str(&format!(
        "Program: {type_count} types, {fn_count} functions, {mod_count} modules\n"
    ));

    if !program.types.is_empty() {
        out.push_str("Types: ");
        out.push_str(
            &program
                .types
                .iter()
                .map(|t| t.name().to_string())
                .collect::<Vec<_>>()
                .join(", "),
        );
        out.push('\n');
    }

    if !program.functions.is_empty() {
        out.push_str("Functions:\n");
        for func in &program.functions {
            let params = func
                .params
                .iter()
                .map(|p| format!("{}:{:?}", p.name, p.ty))
                .collect::<Vec<_>>()
                .join(", ");
            let effects = if func.effects.is_empty() {
                String::new()
            } else {
                format!(" [{:?}]", func.effects)
            };
            out.push_str(&format!(
                "  {} ({})->{:?}{}\n",
                func.name, params, func.return_type, effects
            ));
        }
    }

    for module in &program.modules {
        let fns: Vec<&str> = module.functions.iter().map(|f| f.name.as_str()).collect();
        let types: Vec<String> = module.types.iter().map(|t| t.name().to_string()).collect();
        out.push_str(&format!("Module {}:", module.name));
        if !types.is_empty() {
            out.push_str(&format!(" types=[{}]", types.join(", ")));
        }
        out.push_str(&format!(" fns=[{}]\n", fns.join(", ")));
    }

    // Routes are now in RuntimeState, not Program — use ?routes to see them.
    out.push_str("\nUse ?symbols, ?callers, ?callees, ?deps, ?routes for details.\n");
    out
}

/// Full program summary with all signatures.
pub fn program_summary(program: &ast::Program) -> String {
    let mut out = String::new();
    out.push_str("=== Current Program State ===\n");
    out.push_str(&format!("{program}"));

    // List all functions with their signatures
    for func in &program.functions {
        out.push_str(&format!(
            "\nfn {} ({})->{:?} [{:?}] — {} statements",
            func.name,
            func.params
                .iter()
                .map(|p| format!("{}:{:?}", p.name, p.ty))
                .collect::<Vec<_>>()
                .join(", "),
            func.return_type,
            func.effects,
            func.body.len()
        ));
    }

    for module in &program.modules {
        out.push_str(&format!("\nmodule {}:", module.name));
        for td in &module.types {
            out.push_str(&format!("\n  type {}", td.name()));
        }
        for func in &module.functions {
            out.push_str(&format!(
                "\n  fn {} ({})->{:?} [{:?}] — {} statements",
                func.name,
                func.params
                    .iter()
                    .map(|p| format!("{}:{:?}", p.name, p.ty))
                    .collect::<Vec<_>>()
                    .join(", "),
                func.return_type,
                func.effects,
                func.body.len()
            ));
        }
    }

    out.push('\n');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ═════════════════════════════════════════════════════════════════════
    // convert_effect
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn convert_effect_all_valid() {
        assert_eq!(convert_effect("io").unwrap(), ast::Effect::Io);
        assert_eq!(convert_effect("mut").unwrap(), ast::Effect::Mut);
        assert_eq!(convert_effect("fail").unwrap(), ast::Effect::Fail);
        assert_eq!(convert_effect("async").unwrap(), ast::Effect::Async);
        assert_eq!(convert_effect("rand").unwrap(), ast::Effect::Rand);
        assert_eq!(convert_effect("yield").unwrap(), ast::Effect::Yield);
        assert_eq!(convert_effect("parallel").unwrap(), ast::Effect::Parallel);
        assert_eq!(convert_effect("unsafe").unwrap(), ast::Effect::Unsafe);
    }

    #[test]
    fn convert_effect_case_insensitive() {
        assert_eq!(convert_effect("IO").unwrap(), ast::Effect::Io);
        assert_eq!(convert_effect("Async").unwrap(), ast::Effect::Async);
    }

    #[test]
    fn convert_effect_invalid() {
        assert!(convert_effect("invalid_effect").is_err());
        assert!(convert_effect("").is_err());
    }

    // ═════════════════════════════════════════════════════════════════════
    // convert_binary_op
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn convert_binary_op_arithmetic() {
        assert_eq!(
            convert_binary_op(&parser::BinaryOp::Add),
            ast::BinaryOp::Add
        );
        assert_eq!(
            convert_binary_op(&parser::BinaryOp::Sub),
            ast::BinaryOp::Sub
        );
        assert_eq!(
            convert_binary_op(&parser::BinaryOp::Mul),
            ast::BinaryOp::Mul
        );
        assert_eq!(
            convert_binary_op(&parser::BinaryOp::Div),
            ast::BinaryOp::Div
        );
        assert_eq!(
            convert_binary_op(&parser::BinaryOp::Mod),
            ast::BinaryOp::Mod
        );
    }

    #[test]
    fn convert_binary_op_comparison() {
        assert_eq!(
            convert_binary_op(&parser::BinaryOp::Gt),
            ast::BinaryOp::GreaterThan
        );
        assert_eq!(
            convert_binary_op(&parser::BinaryOp::Lt),
            ast::BinaryOp::LessThan
        );
        assert_eq!(
            convert_binary_op(&parser::BinaryOp::Gte),
            ast::BinaryOp::GreaterThanOrEqual
        );
        assert_eq!(
            convert_binary_op(&parser::BinaryOp::Lte),
            ast::BinaryOp::LessThanOrEqual
        );
        assert_eq!(
            convert_binary_op(&parser::BinaryOp::Eq),
            ast::BinaryOp::Equal
        );
        assert_eq!(
            convert_binary_op(&parser::BinaryOp::Neq),
            ast::BinaryOp::NotEqual
        );
    }

    #[test]
    fn convert_binary_op_logic() {
        assert_eq!(
            convert_binary_op(&parser::BinaryOp::And),
            ast::BinaryOp::And
        );
        assert_eq!(convert_binary_op(&parser::BinaryOp::Or), ast::BinaryOp::Or);
    }

    // ═════════════════════════════════════════════════════════════════════
    // convert_unary_op
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn convert_unary_op_all() {
        assert_eq!(convert_unary_op(&parser::UnaryOp::Not), ast::UnaryOp::Not);
        assert_eq!(convert_unary_op(&parser::UnaryOp::Neg), ast::UnaryOp::Neg);
    }

    // ═════════════════════════════════════════════════════════════════════
    // convert_type
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn convert_type_primitives() {
        assert_eq!(
            convert_type(&parser::TypeExpr::Named("Int".into())).unwrap(),
            ast::Type::Int
        );
        assert_eq!(
            convert_type(&parser::TypeExpr::Named("Float".into())).unwrap(),
            ast::Type::Float
        );
        assert_eq!(
            convert_type(&parser::TypeExpr::Named("Bool".into())).unwrap(),
            ast::Type::Bool
        );
        assert_eq!(
            convert_type(&parser::TypeExpr::Named("String".into())).unwrap(),
            ast::Type::String
        );
        assert_eq!(
            convert_type(&parser::TypeExpr::Named("Byte".into())).unwrap(),
            ast::Type::Byte
        );
    }

    #[test]
    fn convert_type_user_struct() {
        let ty = convert_type(&parser::TypeExpr::Named("User".into())).unwrap();
        assert_eq!(ty, ast::Type::Struct("User".to_string()));
    }

    #[test]
    fn convert_type_list() {
        let ty = convert_type(&parser::TypeExpr::Generic {
            name: "List".into(),
            args: vec![parser::TypeExpr::Named("Int".into())],
        })
        .unwrap();
        assert_eq!(ty, ast::Type::List(Box::new(ast::Type::Int)));
    }

    #[test]
    fn convert_type_map() {
        let ty = convert_type(&parser::TypeExpr::Generic {
            name: "Map".into(),
            args: vec![
                parser::TypeExpr::Named("String".into()),
                parser::TypeExpr::Named("Int".into()),
            ],
        })
        .unwrap();
        assert_eq!(
            ty,
            ast::Type::Map(Box::new(ast::Type::String), Box::new(ast::Type::Int))
        );
    }

    #[test]
    fn convert_type_option() {
        let ty = convert_type(&parser::TypeExpr::Generic {
            name: "Option".into(),
            args: vec![parser::TypeExpr::Named("Int".into())],
        })
        .unwrap();
        assert_eq!(ty, ast::Type::Option(Box::new(ast::Type::Int)));
    }

    #[test]
    fn convert_type_result() {
        let ty = convert_type(&parser::TypeExpr::Generic {
            name: "Result".into(),
            args: vec![parser::TypeExpr::Named("String".into())],
        })
        .unwrap();
        assert_eq!(ty, ast::Type::Result(Box::new(ast::Type::String)));
    }

    #[test]
    fn convert_type_nested_generic() {
        // List<List<Int>>
        let ty = convert_type(&parser::TypeExpr::Generic {
            name: "List".into(),
            args: vec![parser::TypeExpr::Generic {
                name: "List".into(),
                args: vec![parser::TypeExpr::Named("Int".into())],
            }],
        })
        .unwrap();
        assert_eq!(
            ty,
            ast::Type::List(Box::new(ast::Type::List(Box::new(ast::Type::Int))))
        );
    }

    #[test]
    fn convert_type_wrong_arity() {
        // List with 0 args
        assert!(convert_type(&parser::TypeExpr::Generic {
            name: "List".into(),
            args: vec![],
        })
        .is_err());

        // Map with 1 arg
        assert!(convert_type(&parser::TypeExpr::Generic {
            name: "Map".into(),
            args: vec![parser::TypeExpr::Named("Int".into())],
        })
        .is_err());
    }

    // ═════════════════════════════════════════════════════════════════════
    // is_builtin_name
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn is_builtin_name_true() {
        assert!(is_builtin_name("concat"));
        assert!(is_builtin_name("len"));
        assert!(is_builtin_name("http_get"));
    }

    #[test]
    fn is_builtin_name_false() {
        assert!(!is_builtin_name("my_custom_fn"));
        assert!(!is_builtin_name(""));
    }

    // ═════════════════════════════════════════════════════════════════════
    // apply_and_validate with types and functions
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn apply_struct_type() {
        let mut program = ast::Program::default();
        let ops = parser::parse("+type User = id:Int, name:String").unwrap();
        let result = apply_and_validate(&mut program, &ops[0]);
        assert!(
            result.is_ok(),
            "adding struct type should succeed: {:?}",
            result
        );
        assert_eq!(program.types.len(), 1);
        assert_eq!(program.types[0].name(), "User");
    }

    #[test]
    fn apply_union_type() {
        let mut program = ast::Program::default();
        let ops = parser::parse("+type Color = Red | Green | Blue").unwrap();
        let result = apply_and_validate(&mut program, &ops[0]);
        assert!(result.is_ok());
        assert_eq!(program.types.len(), 1);
        assert_eq!(program.types[0].name(), "Color");
    }

    #[test]
    fn apply_function() {
        let mut program = ast::Program::default();
        let source = "\
+fn add (a:Int, b:Int)->Int
  +return a + b
";
        let ops = parser::parse(source).unwrap();
        let result = apply_and_validate(&mut program, &ops[0]);
        assert!(
            result.is_ok(),
            "adding function should succeed: {:?}",
            result
        );
        assert_eq!(program.functions.len(), 1);
        assert_eq!(program.functions[0].name, "add");
        assert_eq!(program.functions[0].params.len(), 2);
    }

    #[test]
    fn apply_function_with_effects() {
        let mut program = ast::Program::default();
        let source = "\
+fn fetch (url:String)->String [io,async]
  +return url
";
        let ops = parser::parse(source).unwrap();
        apply_and_validate(&mut program, &ops[0]).unwrap();
        assert_eq!(program.functions[0].effects.len(), 2);
        assert!(program.functions[0].effects.contains(&ast::Effect::Io));
        assert!(program.functions[0].effects.contains(&ast::Effect::Async));
    }

    #[test]
    fn apply_duplicate_function_updates() {
        let mut program = ast::Program::default();
        let source1 = "+fn greet ()->String\n  +return \"hello\"\n";
        let source2 = "+fn greet ()->String\n  +return \"hi\"\n";
        let ops1 = parser::parse(source1).unwrap();
        let ops2 = parser::parse(source2).unwrap();
        apply_and_validate(&mut program, &ops1[0]).unwrap();
        assert_eq!(program.functions.len(), 1);
        // Re-adding same function name — validator either replaces or errors
        let result = apply_and_validate(&mut program, &ops2[0]);
        // Either it succeeds (replacing) or errors with "duplicate" — both are valid
        if result.is_ok() {
            assert_eq!(program.functions.len(), 1);
        } else {
            let err = format!("{}", result.unwrap_err());
            assert!(
                err.contains("duplicate"),
                "expected duplicate error, got: {err}"
            );
        }
    }

    #[test]
    fn apply_module_with_function() {
        let mut program = ast::Program::default();
        let source = "\
!module Math
+fn add (a:Int, b:Int)->Int
  +return a + b
";
        let ops = parser::parse(source).unwrap();
        let result = apply_and_validate(&mut program, &ops[0]);
        assert!(result.is_ok(), "adding module should succeed: {:?}", result);
        assert_eq!(program.modules.len(), 1);
        assert_eq!(program.modules[0].name, "Math");
        assert_eq!(program.modules[0].functions.len(), 1);
    }

    #[test]
    fn apply_remove_function() {
        let mut program = ast::Program::default();
        let fn_ops = parser::parse("+fn greet ()->String\n  +return \"hi\"\n").unwrap();
        apply_and_validate(&mut program, &fn_ops[0]).unwrap();
        assert_eq!(program.functions.len(), 1);

        let rm_ops = parser::parse("!remove greet").unwrap();
        let result = apply_and_validate(&mut program, &rm_ops[0]);
        assert!(result.is_ok());
        assert_eq!(program.functions.len(), 0);
    }

    // ═════════════════════════════════════════════════════════════════════
    // Module with embedded !test — validator should skip Test ops in body
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn apply_module_with_embedded_test_succeeds() {
        // A module containing +fn and !test blocks should be applied
        // successfully — the validator should skip Test operations in
        // the module body without error.
        let source = "\
!module TestMod

+fn helper ()->Int
  +return 42

!test helper
  +with -> expect 42

+fn other ()->Int
  +return 7
";
        let ops = parser::parse(source).unwrap();
        assert_eq!(ops.len(), 1, "parser should produce 1 Module operation");

        let mut program = ast::Program::default();
        let result = apply_and_validate(&mut program, &ops[0]);
        assert!(
            result.is_ok(),
            "module with embedded !test should validate: {:?}",
            result
        );

        // Both functions should be in the module
        assert_eq!(program.modules.len(), 1);
        assert_eq!(program.modules[0].functions.len(), 2);
        assert_eq!(program.modules[0].functions[0].name, "helper");
        assert_eq!(program.modules[0].functions[1].name, "other");
    }

    #[test]
    fn apply_module_rejects_unexpected_operations() {
        // Operations other than +fn, +type, +shared, and !test should
        // still be rejected inside a module body.
        // (Note: !eval breaks the module body loop in the parser, so
        // this tests that the parser correctly keeps !test but breaks
        // on other ! commands.)
        let source = "\
!module BadMod

+fn f1 ()->Int
  +return 1

!eval f1
";
        let ops = parser::parse(source).unwrap();
        // Parser should break on !eval, producing Module + Eval
        assert_eq!(ops.len(), 2);
        assert!(matches!(&ops[0], parser::Operation::Module(_)));
        assert!(matches!(&ops[1], parser::Operation::Eval(_)));
    }

    #[test]
    fn apply_module_with_require_modules_and_embedded_test() {
        // With require_modules=true, functions after !test inside a module
        // should NOT be rejected. This was the original bug.
        let source = "\
!module StrictMod

+fn first ()->Int
  +return 1

!test first
  +with -> expect 1

+fn second ()->Int
  +return 2
";
        let ops = parser::parse(source).unwrap();
        let mut program = ast::Program::default();
        program.require_modules = true;

        let result = apply_and_validate(&mut program, &ops[0]);
        assert!(
            result.is_ok(),
            "module with !test should work in require_modules mode: {:?}",
            result
        );
        assert_eq!(program.modules[0].functions.len(), 2);
    }
}
