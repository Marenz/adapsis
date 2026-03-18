use anyhow::{anyhow, bail, Result};

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
            if program.types.iter().any(|t| t.name() == name) {
                bail!("duplicate type declaration: `{name}`");
            }
            let msg = format!("added type `{name}`");
            program.types.push(converted);
            Ok(msg)
        }
        parser::Operation::Function(fn_decl) => {
            let converted = convert_function(fn_decl)?;
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
            program.functions.push(converted);
            Ok(msg)
        }
        parser::Operation::Replace(replace) => apply_replace(program, replace),
        parser::Operation::Test(_) => {
            // Tests are handled separately, not applied to program state
            Ok("test block (skipped during validation)".to_string())
        }
        parser::Operation::Trace(_) => Ok("trace (handled by evaluator)".to_string()),
        parser::Operation::Eval(_) => Ok("eval (handled by evaluator)".to_string()),
        parser::Operation::Query(_) => Ok("query (handled by orchestrator)".to_string()),
        // Standalone statements outside a function — invalid at top level
        parser::Operation::Let(_)
        | parser::Operation::Set(_)
        | parser::Operation::Call(_)
        | parser::Operation::Check(_)
        | parser::Operation::Branch(_)
        | parser::Operation::If(_)
        | parser::Operation::Return(_)
        | parser::Operation::Each(_)
        | parser::Operation::While(_) => {
            bail!("statement outside of function body")
        }
    }
}

fn apply_module(program: &mut ast::Program, decl: &parser::ModuleDecl) -> Result<String> {
    if program.modules.iter().any(|m| m.name == decl.name) {
        bail!("duplicate module: `{}`", decl.name);
    }

    let mut module = ast::Module {
        id: decl.name.clone(),
        name: decl.name.clone(),
        types: vec![],
        functions: vec![],
    };

    for op in &decl.body {
        match op {
            parser::Operation::Type(td) => {
                let converted = convert_type_decl(td)?;
                if module.types.iter().any(|t| t.name() == converted.name()) {
                    bail!(
                        "duplicate type `{}` in module `{}`",
                        converted.name(),
                        decl.name
                    );
                }
                module.types.push(converted);
            }
            parser::Operation::Function(fd) => {
                let converted = convert_function(fd)?;
                if module.functions.iter().any(|f| f.name == converted.name) {
                    bail!(
                        "duplicate function `{}` in module `{}`",
                        converted.name,
                        decl.name
                    );
                }
                module.functions.push(converted);
            }
            other => bail!(
                "unexpected operation in module `{}`: {:?}",
                decl.name,
                std::mem::discriminant(other)
            ),
        }
    }

    let msg = format!(
        "added module `{}` ({} types, {} functions)",
        module.name,
        module.types.len(),
        module.functions.len()
    );
    program.modules.push(module);
    Ok(msg)
}

fn apply_replace(program: &mut ast::Program, replace: &parser::ReplaceMutation) -> Result<String> {
    // Parse target path: "ModuleName.function_name.sN" or "function_name.sN"
    let parts: Vec<&str> = replace.target.split('.').collect();

    match parts.len() {
        2 => {
            // function_name.sN
            let fn_name = parts[0];
            let stmt_id = parts[1];
            let func = program
                .functions
                .iter_mut()
                .find(|f| f.name == fn_name)
                .ok_or_else(|| anyhow!("function `{fn_name}` not found for replace"))?;
            replace_statement(&mut func.body, stmt_id, &replace.body)?;
            Ok(format!("replaced `{}`", replace.target))
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
                .ok_or_else(|| {
                    anyhow!("function `{fn_name}` not found in module `{mod_name}` for replace")
                })?;
            replace_statement(&mut func.body, stmt_id, &replace.body)?;
            Ok(format!("replaced `{}`", replace.target))
        }
        _ => bail!(
            "invalid replace target `{}` — expected `fn.sN` or `module.fn.sN`",
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
                ast_variants.push(ast::UnionVariant {
                    id: format!("{}.v{}", decl.name, i),
                    name: variant.name.clone(),
                    payload: variant.payload.as_ref().map(convert_type).transpose()?,
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
    })
}

fn convert_statement_op(op: &parser::Operation) -> Result<ast::Statement> {
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
        other => bail!(
            "cannot convert {:?} to a statement",
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

/// Generate a summary of the current program state for injection into the LLM context.
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
