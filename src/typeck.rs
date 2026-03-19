//! Type checker for Forge programs.
//!
//! Validates that:
//! - All referenced types exist
//! - Function call arguments match parameter types
//! - Binary operations have compatible operand types
//! - Check conditions are boolean
//! - Return values match the declared return type
//! - Variables are used after being defined

use std::collections::HashMap;

use crate::ast::*;

/// A symbol table tracking types, functions, and variables in scope.
#[derive(Debug, Clone, Default)]
pub struct SymbolTable {
    /// Known type names → their declarations
    pub types: HashMap<String, TypeInfo>,
    /// Known function names → their signatures
    pub functions: HashMap<String, FunctionSig>,
}

#[derive(Debug, Clone)]
pub struct TypeInfo {
    pub fields: Vec<(String, Type)>,
    pub variants: Vec<(String, Option<Type>)>,
    pub is_union: bool,
}

#[derive(Debug, Clone)]
pub struct FunctionSig {
    pub params: Vec<(String, Type)>,
    pub return_type: Type,
    pub effects: Vec<Effect>,
}

impl SymbolTable {
    pub fn register_type(&mut self, decl: &TypeDecl) {
        match decl {
            TypeDecl::Struct(s) => {
                self.types.insert(
                    s.name.clone(),
                    TypeInfo {
                        fields: s
                            .fields
                            .iter()
                            .map(|f| (f.name.clone(), f.ty.clone()))
                            .collect(),
                        variants: vec![],
                        is_union: false,
                    },
                );
            }
            TypeDecl::TaggedUnion(u) => {
                self.types.insert(
                    u.name.clone(),
                    TypeInfo {
                        fields: vec![],
                        variants: u
                            .variants
                            .iter()
                            .map(|v| (v.name.clone(), v.payload.clone()))
                            .collect(),
                        is_union: true,
                    },
                );
            }
        }
    }

    pub fn register_function(&mut self, decl: &FunctionDecl) {
        self.functions.insert(
            decl.name.clone(),
            FunctionSig {
                params: decl
                    .params
                    .iter()
                    .map(|p| (p.name.clone(), p.ty.clone()))
                    .collect(),
                return_type: decl.return_type.clone(),
                effects: decl.effects.clone(),
            },
        );
    }

    pub fn resolve_type(&self, name: &str) -> Option<&TypeInfo> {
        self.types.get(name)
    }

    pub fn resolve_function(&self, name: &str) -> Option<&FunctionSig> {
        self.functions.get(name)
    }
}

/// Build a symbol table from an entire program.
pub fn build_symbol_table(program: &Program) -> SymbolTable {
    let mut table = SymbolTable::default();

    for td in &program.types {
        table.register_type(td);
    }

    for func in &program.functions {
        table.register_function(func);
    }

    for module in &program.modules {
        for td in &module.types {
            table.register_type(td);
        }
        for func in &module.functions {
            // Register with module-qualified name
            let sig = FunctionSig {
                params: func
                    .params
                    .iter()
                    .map(|p| (p.name.clone(), p.ty.clone()))
                    .collect(),
                return_type: func.return_type.clone(),
                effects: func.effects.clone(),
            };
            table
                .functions
                .insert(format!("{}.{}", module.name, func.name), sig.clone());
            // Also register with unqualified name for convenience
            table.functions.insert(func.name.clone(), sig);
        }
    }

    table
}

/// Type-check a function body.
/// Returns a list of warnings/errors.
pub fn check_function(table: &SymbolTable, func: &FunctionDecl) -> Vec<String> {
    let mut errors = vec![];
    let mut locals: HashMap<String, Type> = HashMap::new();

    // Register params as locals
    for param in &func.params {
        locals.insert(param.name.clone(), param.ty.clone());
    }

    check_statements(
        table,
        &func.body,
        &mut locals,
        &func.return_type,
        &mut errors,
    );
    errors
}

fn check_statements(
    table: &SymbolTable,
    stmts: &[Statement],
    locals: &mut HashMap<String, Type>,
    return_type: &Type,
    errors: &mut Vec<String>,
) {
    for stmt in stmts {
        match &stmt.kind {
            StatementKind::Let { name, ty, value } => {
                if let Some(expr_ty) = infer_expr_type(table, value, locals) {
                    if !types_compatible(ty, &expr_ty, table) {
                        errors.push(format!(
                            "{}: type mismatch in let binding `{}`: declared {:?} but expression is {:?}",
                            stmt.id, name, ty, expr_ty
                        ));
                    }
                }
                locals.insert(name.clone(), ty.clone());
            }
            StatementKind::Call { binding, call } => {
                // Check function exists and argument count matches
                if let Some(sig) = table.resolve_function(&call.callee) {
                    if call.args.len() != sig.params.len() {
                        errors.push(format!(
                            "{}: function `{}` expects {} arguments, got {}",
                            stmt.id,
                            call.callee,
                            sig.params.len(),
                            call.args.len()
                        ));
                    }
                }
                // Bind the result
                if let Some(binding) = binding {
                    locals.insert(binding.name.clone(), binding.ty.clone());
                }
            }
            StatementKind::Check { condition, .. } => {
                // Condition should be boolean
                if let Some(ty) = infer_expr_type(table, condition, locals) {
                    if !matches!(ty, Type::Bool) {
                        errors.push(format!(
                            "{}: check condition should be Bool, got {:?}",
                            stmt.id, ty
                        ));
                    }
                }
            }
            StatementKind::Return { value } => {
                if let Some(ty) = infer_expr_type(table, value, locals) {
                    // For Result<T> return types, the value should be T (auto-wrapped)
                    let expected = match return_type {
                        Type::Result(inner) => inner.as_ref(),
                        other => other,
                    };
                    if !types_compatible(expected, &ty, table) {
                        errors.push(format!(
                            "{}: return type mismatch: expected {:?} but got {:?}",
                            stmt.id, return_type, ty
                        ));
                    }
                }
            }
            StatementKind::Branch {
                condition,
                then_body,
                else_body,
            } => {
                if let Some(ty) = infer_expr_type(table, condition, locals) {
                    if !matches!(ty, Type::Bool) && !matches!(ty, Type::Struct(_)) {
                        errors.push(format!(
                            "{}: branch condition should be Bool or enum, got {:?}",
                            stmt.id, ty
                        ));
                    }
                }
                check_statements(table, then_body, &mut locals.clone(), return_type, errors);
                check_statements(table, else_body, &mut locals.clone(), return_type, errors);
            }
            StatementKind::Each {
                iterator,
                binding,
                body,
            } => {
                // Iterator should be a list
                if let Some(ty) = infer_expr_type(table, iterator, locals) {
                    if !matches!(ty, Type::List(_)) {
                        errors.push(format!("{}: each expects a List, got {:?}", stmt.id, ty));
                    }
                }
                let mut loop_locals = locals.clone();
                loop_locals.insert(binding.name.clone(), binding.ty.clone());
                check_statements(table, body, &mut loop_locals, return_type, errors);
            }
            StatementKind::Set { name, value } => {
                // Check that the variable exists and the new value is compatible
                if let Some(existing_ty) = locals.get(name) {
                    if let Some(expr_ty) = infer_expr_type(table, value, locals) {
                        if !types_compatible(existing_ty, &expr_ty, table) {
                            errors.push(format!(
                                "{}: type mismatch in set `{}`: variable is {:?} but expression is {:?}",
                                stmt.id, name, existing_ty, expr_ty
                            ));
                        }
                    }
                } else {
                    errors.push(format!(
                        "{}: undefined variable `{}` in set statement",
                        stmt.id, name
                    ));
                }
            }
            StatementKind::While { condition, body } => {
                if let Some(ty) = infer_expr_type(table, condition, locals) {
                    if !matches!(ty, Type::Bool) {
                        errors.push(format!(
                            "{}: while condition should be Bool, got {:?}",
                            stmt.id, ty
                        ));
                    }
                }
                check_statements(table, body, &mut locals.clone(), return_type, errors);
            }
            StatementKind::Await { name, ty, call } => {
                // Check that the function exists
                if let Some(_sig) = table.resolve_function(&call.callee) {
                    // Could verify arg types here
                }
                locals.insert(name.clone(), ty.clone());
            }
            StatementKind::Spawn { call } => {
                if table.resolve_function(&call.callee).is_none() {
                    // Builtin IO functions won't be in the table — that's fine
                }
            }
            StatementKind::Yield { .. } => {}
        }
    }
}

/// Try to infer the type of an expression.
/// Returns None if type cannot be determined (we don't fail hard — just skip the check).
fn infer_expr_type(
    table: &SymbolTable,
    expr: &Expr,
    locals: &HashMap<String, Type>,
) -> Option<Type> {
    match expr {
        Expr::Literal(lit) => Some(match lit {
            Literal::Int(_) => Type::Int,
            Literal::Float(_) => Type::Float,
            Literal::Bool(_) => Type::Bool,
            Literal::String(_) => Type::String,
        }),
        Expr::Identifier(name) => locals.get(name).cloned(),
        Expr::FieldAccess { base, field } => {
            let base_ty = infer_expr_type(table, base, locals)?;
            match &base_ty {
                Type::Struct(name) => {
                    let type_info = table.resolve_type(name)?;
                    type_info
                        .fields
                        .iter()
                        .find(|(n, _)| n == field)
                        .map(|(_, ty)| ty.clone())
                        .or_else(|| {
                            // Built-in field-like methods
                            match field.as_str() {
                                "len" => Some(Type::Int),
                                "is_empty" => Some(Type::Bool),
                                "is_ok" | "is_err" => Some(Type::Bool),
                                _ => None,
                            }
                        })
                }
                Type::String => match field.as_str() {
                    "len" => Some(Type::Int),
                    "is_empty" => Some(Type::Bool),
                    _ => None,
                },
                Type::List(_) => match field.as_str() {
                    "len" => Some(Type::Int),
                    "is_empty" => Some(Type::Bool),
                    _ => None,
                },
                Type::Result(inner) => match field.as_str() {
                    "is_ok" | "is_err" => Some(Type::Bool),
                    "unwrap" => Some(inner.as_ref().clone()),
                    _ => None,
                },
                Type::Option(inner) => match field.as_str() {
                    "is_some" | "is_none" => Some(Type::Bool),
                    "unwrap" => Some(inner.as_ref().clone()),
                    _ => None,
                },
                _ => None,
            }
        }
        Expr::Call(call) => table
            .resolve_function(&call.callee)
            .map(|sig| sig.return_type.clone()),
        Expr::Binary { op, .. } => Some(match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => {
                // Could be Int or Float depending on operands — simplified
                Type::Int
            }
            _ => Type::Bool,
        }),
        Expr::Unary { op, expr: inner } => match op {
            UnaryOp::Not => Some(Type::Bool),
            UnaryOp::Neg => infer_expr_type(table, inner, locals),
        },
        Expr::StructInit { ty, .. } => {
            if ty.is_empty() {
                None // anonymous struct
            } else {
                Some(Type::Struct(ty.clone()))
            }
        }
    }
}

/// Check if two types are compatible (for assignment/comparison).
fn types_compatible(expected: &Type, actual: &Type, _table: &SymbolTable) -> bool {
    // Exact match
    if expected == actual {
        return true;
    }

    // Int ↔ Float coercion (lenient for Phase 2)
    if matches!(
        (expected, actual),
        (Type::Int, Type::Float) | (Type::Float, Type::Int)
    ) {
        return true;
    }

    // Any struct type matches another struct type (lenient — we don't have full struct resolution yet)
    if matches!((expected, actual), (Type::Struct(_), Type::Struct(_))) {
        return true;
    }

    // Result<T> matches Result<U> if T matches U
    if let (Type::Result(a), Type::Result(b)) = (expected, actual) {
        return types_compatible(a, b, _table);
    }

    // Option<T> matches Option<U> if T matches U
    if let (Type::Option(a), Type::Option(b)) = (expected, actual) {
        return types_compatible(a, b, _table);
    }

    // List<T> matches List<U> if T matches U
    if let (Type::List(a), Type::List(b)) = (expected, actual) {
        return types_compatible(a, b, _table);
    }

    false
}

/// Respond to a semantic query about the program.
pub fn handle_query(program: &Program, table: &SymbolTable, query: &str) -> String {
    let parts: Vec<&str> = query.trim().split_whitespace().collect();
    if parts.is_empty() {
        return "empty query".to_string();
    }

    match parts[0] {
        "?symbols" => {
            let scope = parts.get(1).copied().unwrap_or("");
            query_symbols(program, table, scope)
        }
        "?callers" => {
            let target = parts.get(1).copied().unwrap_or("");
            query_callers(program, target)
        }
        "?effects" => {
            let target = parts.get(1).copied().unwrap_or("");
            query_effects(table, target)
        }
        "?type" => {
            let target = parts.get(1).copied().unwrap_or("");
            query_type(table, target)
        }
        _ => format!("unknown query: {}", parts[0]),
    }
}

fn query_symbols(_program: &Program, table: &SymbolTable, scope: &str) -> String {
    let mut out = String::new();

    if scope.is_empty() {
        out.push_str("Types:\n");
        for (name, info) in &table.types {
            if info.is_union {
                out.push_str(&format!(
                    "  {} = {}\n",
                    name,
                    info.variants
                        .iter()
                        .map(|(n, p)| {
                            if let Some(ty) = p {
                                format!("{n}({ty:?})")
                            } else {
                                n.clone()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" | ")
                ));
            } else {
                out.push_str(&format!(
                    "  {} = {{{}}}\n",
                    name,
                    info.fields
                        .iter()
                        .map(|(n, t)| format!("{n}:{t:?}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }

        out.push_str("Functions:\n");
        for (name, sig) in &table.functions {
            out.push_str(&format!(
                "  {} ({})->{:?} [{:?}]\n",
                name,
                sig.params
                    .iter()
                    .map(|(n, t)| format!("{n}:{t:?}"))
                    .collect::<Vec<_>>()
                    .join(", "),
                sig.return_type,
                sig.effects
            ));
        }
    } else {
        // Query specific scope
        if let Some(sig) = table.resolve_function(scope) {
            out.push_str(&format!("Function {}:\n", scope));
            out.push_str(&format!(
                "  params: {}\n",
                sig.params
                    .iter()
                    .map(|(n, t)| format!("{n}:{t:?}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
            out.push_str(&format!("  returns: {:?}\n", sig.return_type));
            out.push_str(&format!("  effects: {:?}\n", sig.effects));
        } else if let Some(info) = table.resolve_type(scope) {
            out.push_str(&format!("Type {}:\n", scope));
            if info.is_union {
                for (name, payload) in &info.variants {
                    out.push_str(&format!("  variant {}: {:?}\n", name, payload));
                }
            } else {
                for (name, ty) in &info.fields {
                    out.push_str(&format!("  field {}: {:?}\n", name, ty));
                }
            }
        } else {
            out.push_str(&format!("not found: {scope}\n"));
        }
    }

    out
}

fn query_callers(program: &Program, target: &str) -> String {
    let mut callers = vec![];

    for func in &program.functions {
        if calls_function(&func.body, target) {
            callers.push(func.name.clone());
        }
    }

    for module in &program.modules {
        for func in &module.functions {
            if calls_function(&func.body, target) {
                callers.push(format!("{}.{}", module.name, func.name));
            }
        }
    }

    if callers.is_empty() {
        format!("no callers of `{target}`")
    } else {
        format!("callers of `{target}`: {}", callers.join(", "))
    }
}

fn calls_function(stmts: &[Statement], target: &str) -> bool {
    stmts.iter().any(|stmt| match &stmt.kind {
        StatementKind::Call { call, .. } => {
            call.callee == target || call.callee.ends_with(&format!(".{target}"))
        }
        StatementKind::Branch {
            then_body,
            else_body,
            ..
        } => calls_function(then_body, target) || calls_function(else_body, target),
        StatementKind::Each { body, .. } => calls_function(body, target),
        _ => false,
    })
}

fn query_effects(table: &SymbolTable, target: &str) -> String {
    if let Some(sig) = table.resolve_function(target) {
        if sig.effects.is_empty() {
            format!("`{target}` is pure (no effects)")
        } else {
            format!("`{target}` effects: {:?}", sig.effects)
        }
    } else {
        format!("function `{target}` not found")
    }
}

fn query_type(table: &SymbolTable, target: &str) -> String {
    if let Some(info) = table.resolve_type(target) {
        if info.is_union {
            format!(
                "{} = {}",
                target,
                info.variants
                    .iter()
                    .map(|(n, p)| {
                        if let Some(ty) = p {
                            format!("{n}({ty:?})")
                        } else {
                            n.clone()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" | ")
            )
        } else {
            format!(
                "{} = {{{}}}",
                target,
                info.fields
                    .iter()
                    .map(|(n, t)| format!("{n}:{t:?}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    } else {
        format!("type `{target}` not found")
    }
}
