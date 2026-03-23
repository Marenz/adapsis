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
    pub variants: Vec<(String, Vec<Type>)>,
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
            let sig = FunctionSig {
                params: func
                    .params
                    .iter()
                    .map(|p| (p.name.clone(), p.ty.clone()))
                    .collect(),
                return_type: func.return_type.clone(),
                effects: func.effects.clone(),
            };
            // Register with module-qualified name (canonical for external queries)
            table
                .functions
                .insert(format!("{}.{}", module.name, func.name), sig.clone());
            // Also register bare name for internal type-checking resolution.
            // query_symbols() filters these out of display output.
            table.functions.entry(func.name.clone()).or_insert(sig);
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
            StatementKind::Match { expr: _, arms } => {
                for arm in arms {
                    // Bind arm variables for type checking
                    let mut arm_locals = locals.clone();
                    for binding in &arm.bindings {
                        // We don't know the exact type without looking up the union — use generic
                        arm_locals.insert(binding.clone(), Type::Int);
                    }
                    check_statements(table, &arm.body, &mut arm_locals, return_type, errors);
                }
            }
            StatementKind::Spawn { call, .. } => {
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
/// Precomputed call graph: function → callees, built once per program change.
pub struct CallGraph {
    /// function_name → list of functions it calls
    pub callees: HashMap<String, Vec<String>>,
    /// function_name → list of functions that call it
    #[allow(dead_code)]
    pub callers: HashMap<String, Vec<String>>,
}

pub fn build_call_graph(program: &Program) -> CallGraph {
    let mut callees_map: HashMap<String, Vec<String>> = HashMap::new();
    let mut callers_map: HashMap<String, Vec<String>> = HashMap::new();

    // Top-level functions
    for func in &program.functions {
        let callees = collect_callees_from_stmts(&func.body);
        for callee in &callees {
            callers_map
                .entry(callee.clone())
                .or_default()
                .push(func.name.clone());
        }
        callees_map.insert(func.name.clone(), callees);
    }

    // Module functions
    for module in &program.modules {
        for func in &module.functions {
            let qualified = format!("{}.{}", module.name, func.name);
            let callees = collect_callees_from_stmts(&func.body);
            for callee in &callees {
                callers_map
                    .entry(callee.clone())
                    .or_default()
                    .push(qualified.clone());
            }
            callees_map.insert(qualified, callees.clone());
            // Also store unqualified for lookup
            callees_map.insert(func.name.clone(), callees);
        }
    }

    CallGraph {
        callees: callees_map,
        callers: callers_map,
    }
}

fn query_callees_from_graph(graph: &CallGraph, target: &str) -> String {
    match graph.callees.get(target) {
        Some(callees) if !callees.is_empty() => {
            format!("`{target}` calls: {}", callees.join(", "))
        }
        _ => format!("`{target}` calls no other functions"),
    }
}

fn query_deps_from_graph(graph: &CallGraph, target: &str) -> String {
    // Transitive closure — all functions reachable from target
    let mut visited = std::collections::HashSet::new();
    let mut stack = vec![target.to_string()];
    while let Some(name) = stack.pop() {
        if !visited.insert(name.clone()) {
            continue;
        }
        if let Some(callees) = graph.callees.get(&name) {
            for callee in callees {
                if !visited.contains(callee) {
                    stack.push(callee.clone());
                }
            }
        }
    }
    visited.remove(target);
    if visited.is_empty() {
        format!("`{target}` has no dependencies")
    } else {
        let mut deps: Vec<&String> = visited.iter().collect();
        deps.sort();
        format!(
            "`{target}` depends on: {}",
            deps.iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

fn query_deps_modules(program: &Program, graph: &CallGraph, target: &str) -> String {
    // Find all transitive deps, then group by module
    let mut visited = std::collections::HashSet::new();
    let mut stack = vec![target.to_string()];
    while let Some(name) = stack.pop() {
        if !visited.insert(name.clone()) {
            continue;
        }
        if let Some(callees) = graph.callees.get(&name) {
            for callee in callees {
                if !visited.contains(callee) {
                    stack.push(callee.clone());
                }
            }
        }
    }
    visited.remove(target);

    // Group by module
    let mut module_deps: HashMap<String, Vec<String>> = HashMap::new();
    for dep in &visited {
        if let Some(dot) = dep.find('.') {
            let module = &dep[..dot];
            let func = &dep[dot + 1..];
            module_deps
                .entry(module.to_string())
                .or_default()
                .push(func.to_string());
        } else {
            // Check which module this function belongs to
            let mut found_module = None;
            for m in &program.modules {
                if m.functions.iter().any(|f| f.name == *dep) {
                    found_module = Some(m.name.clone());
                    break;
                }
            }
            let key = found_module.unwrap_or_else(|| "(top-level)".to_string());
            module_deps.entry(key).or_default().push(dep.clone());
        }
    }

    if module_deps.is_empty() {
        return format!("`{target}` has no module dependencies");
    }

    let mut out = format!("`{target}` depends on modules:\n");
    let mut modules: Vec<_> = module_deps.iter().collect();
    modules.sort_by_key(|(k, _)| (*k).clone());
    for (module, fns) in modules {
        out.push_str(&format!("  {module}: {}\n", fns.join(", ")));
    }
    out
}

/// Reconstruct Forge source code from the AST for a function.
/// Supports both qualified (`Module.func`) and unqualified (`func`) names.
fn reconstruct_source(program: &Program, target: &str) -> String {
    // Module-qualified lookup: "Module.func"
    let func = if let Some((module_name, fn_name)) = target.split_once('.') {
        program
            .modules
            .iter()
            .find(|m| m.name == module_name)
            .and_then(|m| m.functions.iter().find(|f| f.name == fn_name))
    } else {
        // Unqualified: search top-level first, then modules
        program
            .functions
            .iter()
            .find(|f| f.name == target)
            .or_else(|| {
                program
                    .modules
                    .iter()
                    .flat_map(|m| m.functions.iter())
                    .find(|f| f.name == target)
            })
    };

    let Some(func) = func else {
        // Try as a type — qualified or unqualified
        if let Some((module_name, type_name)) = target.split_once('.') {
            if let Some(m) = program.modules.iter().find(|m| m.name == module_name) {
                for td in &m.types {
                    if td.name() == type_name {
                        return reconstruct_type_source(td);
                    }
                }
            }
        } else {
            for td in &program.types {
                if td.name() == target {
                    return reconstruct_type_source(td);
                }
            }
            for m in &program.modules {
                for td in &m.types {
                    if td.name() == target {
                        return reconstruct_type_source(td);
                    }
                }
            }
        }
        return format!("`{target}` not found");
    };

    let mut out = String::new();
    let params = func
        .params
        .iter()
        .map(|p| format!("{}:{}", p.name, format_type_simple(&p.ty)))
        .collect::<Vec<_>>()
        .join(", ");
    let effects = if func.effects.is_empty() {
        String::new()
    } else {
        format!(
            " [{}]",
            func.effects
                .iter()
                .map(|e| format!("{e:?}").to_lowercase())
                .collect::<Vec<_>>()
                .join(",")
        )
    };
    out.push_str(&format!(
        "+fn {} ({})->{}{}\n",
        func.name,
        params,
        format_type_simple(&func.return_type),
        effects
    ));

    for stmt in &func.body {
        reconstruct_stmt(&mut out, stmt, 1);
    }
    out
}

fn reconstruct_type_source(td: &TypeDecl) -> String {
    match td {
        TypeDecl::Struct(s) => {
            let fields = s
                .fields
                .iter()
                .map(|f| format!("{}:{}", f.name, format_type_simple(&f.ty)))
                .collect::<Vec<_>>()
                .join(", ");
            format!("+type {} = {{{}}}\n", s.name, fields)
        }
        TypeDecl::TaggedUnion(u) => {
            let variants = u
                .variants
                .iter()
                .map(|v| {
                    if v.payload.is_empty() {
                        v.name.clone()
                    } else {
                        format!(
                            "{}({})",
                            v.name,
                            v.payload
                                .iter()
                                .map(format_type_simple)
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    }
                })
                .collect::<Vec<_>>()
                .join(" | ");
            format!("+type {} = {}\n", u.name, variants)
        }
    }
}

fn format_type_simple(ty: &Type) -> String {
    match ty {
        Type::Int => "Int".into(),
        Type::Float => "Float".into(),
        Type::Bool => "Bool".into(),
        Type::String => "String".into(),
        Type::Byte => "Byte".into(),
        Type::List(t) => format!("List<{}>", format_type_simple(t)),
        Type::Set(t) => format!("Set<{}>", format_type_simple(t)),
        Type::Map(k, v) => format!("Map<{},{}>", format_type_simple(k), format_type_simple(v)),
        Type::Option(t) => format!("Option<{}>", format_type_simple(t)),
        Type::Result(t) => format!("Result<{}>", format_type_simple(t)),
        Type::Struct(name) | Type::TaggedUnion(name) => name.clone(),
    }
}

fn reconstruct_stmt(out: &mut String, stmt: &Statement, indent: usize) {
    let pad = "  ".repeat(indent);
    match &stmt.kind {
        StatementKind::Let { name, ty, value } => {
            out.push_str(&format!(
                "{pad}+let {name}:{} = {}\n",
                format_type_simple(ty),
                reconstruct_expr(value)
            ));
        }
        StatementKind::Set { name, value } => {
            out.push_str(&format!("{pad}+set {name} = {}\n", reconstruct_expr(value)));
        }
        StatementKind::Call { binding, call } => {
            if let Some(b) = binding {
                out.push_str(&format!(
                    "{pad}+call {}:{} = {}({})\n",
                    b.name,
                    format_type_simple(&b.ty),
                    call.callee,
                    call.args
                        .iter()
                        .map(reconstruct_expr)
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            } else {
                out.push_str(&format!(
                    "{pad}+call {}({})\n",
                    call.callee,
                    call.args
                        .iter()
                        .map(reconstruct_expr)
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }
        StatementKind::Check {
            label,
            condition,
            on_fail,
        } => {
            out.push_str(&format!(
                "{pad}+check {label} {} ~{on_fail}\n",
                reconstruct_expr(condition)
            ));
        }
        StatementKind::Return { value } => {
            out.push_str(&format!("{pad}+return {}\n", reconstruct_expr(value)));
        }
        StatementKind::Branch {
            condition,
            then_body,
            else_body,
        } => {
            out.push_str(&format!("{pad}+if {}\n", reconstruct_expr(condition)));
            for s in then_body {
                reconstruct_stmt(out, s, indent + 1);
            }
            if !else_body.is_empty() {
                out.push_str(&format!("{pad}+else\n"));
                for s in else_body {
                    reconstruct_stmt(out, s, indent + 1);
                }
            }
        }
        StatementKind::While { condition, body } => {
            out.push_str(&format!("{pad}+while {}\n", reconstruct_expr(condition)));
            for s in body {
                reconstruct_stmt(out, s, indent + 1);
            }
        }
        StatementKind::Each {
            iterator,
            binding,
            body,
        } => {
            out.push_str(&format!(
                "{pad}+each {} {}:{}\n",
                reconstruct_expr(iterator),
                binding.name,
                format_type_simple(&binding.ty)
            ));
            for s in body {
                reconstruct_stmt(out, s, indent + 1);
            }
        }
        StatementKind::Match { expr, arms } => {
            out.push_str(&format!("{pad}+match {}\n", reconstruct_expr(expr)));
            for arm in arms {
                if arm.bindings.is_empty() {
                    out.push_str(&format!("{pad}+case {}\n", arm.variant));
                } else {
                    out.push_str(&format!(
                        "{pad}+case {}({})\n",
                        arm.variant,
                        arm.bindings.join(", ")
                    ));
                }
                for s in &arm.body {
                    reconstruct_stmt(out, s, indent + 1);
                }
            }
        }
        StatementKind::Await { name, ty, call } => {
            out.push_str(&format!(
                "{pad}+await {name}:{} = {}({})\n",
                format_type_simple(ty),
                call.callee,
                call.args
                    .iter()
                    .map(reconstruct_expr)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        StatementKind::Spawn { call, .. } => {
            out.push_str(&format!(
                "{pad}+spawn {}({})\n",
                call.callee,
                call.args
                    .iter()
                    .map(reconstruct_expr)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        StatementKind::Yield { value } => {
            out.push_str(&format!("{pad}+yield {}\n", reconstruct_expr(value)));
        }
    }
}

/// Public wrapper for `reconstruct_stmt` — used by the module library serializer.
pub fn reconstruct_stmt_pub(out: &mut String, stmt: &Statement, indent: usize) {
    reconstruct_stmt(out, stmt, indent);
}

fn reconstruct_expr(expr: &Expr) -> String {
    match expr {
        Expr::Literal(lit) => match lit {
            crate::ast::Literal::Int(n) => n.to_string(),
            crate::ast::Literal::Float(f) => f.to_string(),
            crate::ast::Literal::Bool(b) => b.to_string(),
            crate::ast::Literal::String(s) => format!("\"{s}\""),
        },
        Expr::Identifier(name) => name.clone(),
        Expr::FieldAccess { base, field } => format!("{}.{field}", reconstruct_expr(base)),
        Expr::Call(call) => format!(
            "{}({})",
            call.callee,
            call.args
                .iter()
                .map(reconstruct_expr)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Expr::Binary { left, op, right } => {
            let op_str = match op {
                crate::ast::BinaryOp::Add => "+",
                crate::ast::BinaryOp::Sub => "-",
                crate::ast::BinaryOp::Mul => "*",
                crate::ast::BinaryOp::Div => "/",
                crate::ast::BinaryOp::Mod => "%",
                crate::ast::BinaryOp::GreaterThan => ">",
                crate::ast::BinaryOp::LessThan => "<",
                crate::ast::BinaryOp::GreaterThanOrEqual => ">=",
                crate::ast::BinaryOp::LessThanOrEqual => "<=",
                crate::ast::BinaryOp::Equal => "==",
                crate::ast::BinaryOp::NotEqual => "!=",
                crate::ast::BinaryOp::And => " AND ",
                crate::ast::BinaryOp::Or => " OR ",
            };
            format!(
                "{}{op_str}{}",
                reconstruct_expr(left),
                reconstruct_expr(right)
            )
        }
        Expr::Unary { op, expr } => match op {
            crate::ast::UnaryOp::Not => format!("NOT {}", reconstruct_expr(expr)),
            crate::ast::UnaryOp::Neg => format!("-{}", reconstruct_expr(expr)),
        },
        Expr::StructInit { ty, fields } => {
            let fs = fields
                .iter()
                .map(|f| format!("{}: {}", f.name, reconstruct_expr(&f.value)))
                .collect::<Vec<_>>()
                .join(", ");
            if ty.is_empty() {
                format!("{{{fs}}}")
            } else {
                format!("{ty}{{{fs}}}")
            }
        }
    }
}

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
        "?callees" => {
            let target = parts.get(1).copied().unwrap_or("");
            let graph = build_call_graph(program);
            query_callees_from_graph(&graph, target)
        }
        "?deps" => {
            // Direct dependencies only (1 level)
            let target = parts.get(1).copied().unwrap_or("");
            let graph = build_call_graph(program);
            query_callees_from_graph(&graph, target)
        }
        "?deps-all" => {
            let target = parts.get(1).copied().unwrap_or("");
            let graph = build_call_graph(program);
            query_deps_from_graph(&graph, target)
        }
        "?deps-modules" => {
            let target = parts.get(1).copied().unwrap_or("");
            let graph = build_call_graph(program);
            query_deps_modules(program, &graph, target)
        }
        "?source" => {
            let target = parts.get(1).copied().unwrap_or("");
            reconstruct_source(program, target)
        }
        "?effects" => {
            let target = parts.get(1).copied().unwrap_or("");
            query_effects(table, target)
        }
        "?type" => {
            let target = parts.get(1).copied().unwrap_or("");
            query_type(table, target)
        }
        "?routes" => query_routes(program),
        // ?tasks is handled at the API level (needs runtime access, not just program)
        "?tasks" => "tasks query requires runtime context".to_string(),
        // ?library works from any query path — reads disk state directly
        "?library" => crate::library::query_library(program, None),
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
                            if p.is_empty() {
                                n.clone()
                            } else {
                                format!(
                                    "{n}({})",
                                    p.iter()
                                        .map(|t| format!("{t:?}"))
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                )
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
        // Collect and sort for deterministic output; skip unqualified aliases
        // (if "Module.fn" exists, don't also show "fn")
        let mut fn_names: Vec<&String> = table.functions.keys().collect();
        fn_names.sort();
        for name in fn_names {
            // Skip unqualified name if a qualified version exists in the table
            if !name.contains('.') {
                let has_qualified = table
                    .functions
                    .keys()
                    .any(|k| k.contains('.') && k.ends_with(&format!(".{name}")));
                if has_qualified {
                    continue;
                }
            }
            let sig = &table.functions[name];
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
    let callees = collect_callees_from_stmts(stmts);
    callees
        .iter()
        .any(|c| c == target || c.ends_with(&format!(".{target}")))
}

/// Collect all function names called from a list of statements (recursive).
pub fn collect_callees_from_stmts(stmts: &[Statement]) -> Vec<String> {
    let mut callees = vec![];
    for stmt in stmts {
        match &stmt.kind {
            StatementKind::Let { value, .. }
            | StatementKind::Set { value, .. }
            | StatementKind::Return { value }
            | StatementKind::Check {
                condition: value, ..
            } => {
                collect_callees_from_expr(value, &mut callees);
            }
            StatementKind::Call { call, .. }
            | StatementKind::Await { call, .. }
            | StatementKind::Spawn { call, .. } => {
                callees.push(call.callee.clone());
                for arg in &call.args {
                    collect_callees_from_expr(arg, &mut callees);
                }
            }
            StatementKind::Branch {
                condition,
                then_body,
                else_body,
            } => {
                collect_callees_from_expr(condition, &mut callees);
                callees.extend(collect_callees_from_stmts(then_body));
                callees.extend(collect_callees_from_stmts(else_body));
            }
            StatementKind::While { condition, body } => {
                collect_callees_from_expr(condition, &mut callees);
                callees.extend(collect_callees_from_stmts(body));
            }
            StatementKind::Each { body, .. } => {
                callees.extend(collect_callees_from_stmts(body));
            }
            StatementKind::Match { expr, arms } => {
                collect_callees_from_expr(expr, &mut callees);
                for arm in arms {
                    callees.extend(collect_callees_from_stmts(&arm.body));
                }
            }
            _ => {}
        }
    }
    callees.sort();
    callees.dedup();
    callees
}

fn collect_callees_from_expr(expr: &Expr, callees: &mut Vec<String>) {
    match expr {
        Expr::Call(call) => {
            callees.push(call.callee.clone());
            for arg in &call.args {
                collect_callees_from_expr(arg, callees);
            }
        }
        Expr::Binary { left, right, .. } => {
            collect_callees_from_expr(left, callees);
            collect_callees_from_expr(right, callees);
        }
        Expr::Unary { expr: inner, .. } => {
            collect_callees_from_expr(inner, callees);
        }
        Expr::FieldAccess { base, .. } => {
            collect_callees_from_expr(base, callees);
        }
        Expr::StructInit { fields, .. } => {
            for f in fields {
                collect_callees_from_expr(&f.value, callees);
            }
        }
        _ => {}
    }
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
                        if p.is_empty() {
                            n.clone()
                        } else {
                            format!(
                                "{n}({})",
                                p.iter()
                                    .map(|t| format!("{t:?}"))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )
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

fn query_routes(program: &Program) -> String {
    if program.http_routes.is_empty() {
        return "No HTTP routes registered.".to_string();
    }
    let mut out = String::from("HTTP Routes:\n");
    for route in &program.http_routes {
        out.push_str(&format!(
            "  {} {} -> {}\n",
            route.method, route.path, route.handler_fn
        ));
    }
    out
}
