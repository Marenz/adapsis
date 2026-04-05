//! Type checker for Adapsis programs.
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
            StatementKind::Source(op) => match op {
                SourceOp::Add {
                    source_type: SourceType::Timer(expr),
                    ..
                }
                | SourceOp::Replace {
                    source_type: SourceType::Timer(expr),
                    ..
                } => {
                    let _ = infer_expr_type(table, expr, locals);
                }
                _ => {}
            },
            StatementKind::Event(op) => {
                if let EventOp::Emit { value, .. } = op {
                    let _ = infer_expr_type(table, value, locals);
                }
            }
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

/// Reconstruct Adapsis source code from the AST for a function.
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

    // Include persisted tests in source reconstruction
    if !func.tests.is_empty() {
        out.push('\n');
        out.push_str(&format!("+test {}\n", func.name));
        for tc in &func.tests {
            let expect_part = reconstruct_test_expect(&tc.expected, tc.matcher.as_deref());
            out.push_str(&format!("  +with {} -> expect {}\n", tc.input, expect_part));
            for ac in &tc.after_checks {
                out.push_str(&format!(
                    "  +after {} {} \"{}\"\n",
                    ac.target, ac.matcher, ac.value
                ));
            }
        }
    }

    out
}

/// Reconstruct the expect portion of a test case, translating serialized
/// matcher strings back into Adapsis test syntax.
fn reconstruct_test_expect(expected: &str, matcher: Option<&str>) -> String {
    if let Some(m) = matcher {
        if m == "AnyOk" {
            return "Ok".to_string();
        } else if m == "AnyErr" {
            return "Err".to_string();
        } else if let Some(msg) = m.strip_prefix("ErrContaining:") {
            return format!("Err(\"{}\")", msg);
        } else if let Some(sub) = m.strip_prefix("contains:") {
            return format!("contains(\"{}\")", sub);
        } else if let Some(pre) = m.strip_prefix("starts_with:") {
            return format!("starts_with(\"{}\")", pre);
        }
    }
    expected.to_string()
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
        StatementKind::Source(op) => match op {
            SourceOp::Add {
                source_type,
                alias,
                handler,
            } => {
                out.push_str(&format!(
                    "{pad}+source add {} as {} -> {}\n",
                    format_source_type(source_type),
                    alias,
                    handler
                ));
            }
            SourceOp::Remove { alias } => {
                out.push_str(&format!("{pad}+source remove {}\n", alias));
            }
            SourceOp::Replace {
                alias,
                source_type,
                handler,
            } => {
                out.push_str(&format!(
                    "{pad}+source replace {} {} -> {}\n",
                    alias,
                    format_source_type(source_type),
                    handler
                ));
            }
            SourceOp::List => {
                out.push_str(&format!("{pad}+source list\n"));
            }
        },
        StatementKind::Event(op) => match op {
            EventOp::Register { name, payload_type } => {
                out.push_str(&format!(
                    "{pad}+event register {}({})\n",
                    name, payload_type
                ));
            }
            EventOp::Emit { name, value } => {
                out.push_str(&format!(
                    "{pad}+event emit {} {}\n",
                    name,
                    reconstruct_expr(value)
                ));
            }
        },
    }
}

fn format_source_type(st: &crate::ast::SourceType) -> String {
    match st {
        crate::ast::SourceType::Timer(expr) => format!("timer({})", reconstruct_expr(expr)),
        crate::ast::SourceType::Channel => "channel".to_string(),
        crate::ast::SourceType::Event(module, event) => format!("{}.{}", module, event),
    }
}

/// Public wrapper for `reconstruct_stmt` — used by the module library serializer.
pub fn reconstruct_stmt_pub(out: &mut String, stmt: &Statement, indent: usize) {
    reconstruct_stmt(out, stmt, indent);
}

pub fn reconstruct_expr(expr: &Expr) -> String {
    match expr {
        Expr::Literal(lit) => match lit {
            crate::ast::Literal::Int(n) => n.to_string(),
            crate::ast::Literal::Float(f) => f.to_string(),
            crate::ast::Literal::Bool(b) => b.to_string(),
            crate::ast::Literal::String(s) => crate::ast::format_string_literal(s),
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

pub fn handle_query(
    program: &Program,
    table: &SymbolTable,
    query: &str,
    http_routes: &[crate::ast::HttpRoute],
) -> String {
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
        "?routes" => query_routes(http_routes),
        // ?tasks is handled at the API level (needs runtime access, not just program)
        "?tasks" => "tasks query requires runtime context".to_string(),
        // ?inspect is handled at the API level (needs runtime snapshot registry)
        "?inspect" => "inspect query requires runtime context".to_string(),
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
        // Check if scope is a module name — list its contents
        if let Some(module) = _program.modules.iter().find(|m| m.name == scope) {
            out.push_str(&format!("Module {}:\n", scope));
            if !module.types.is_empty() {
                out.push_str("  Types:\n");
                for t in &module.types {
                    out.push_str(&format!("    {}\n", t.name()));
                }
            }
            out.push_str("  Functions:\n");
            for f in &module.functions {
                let effects: Vec<String> = f.effects.iter().map(|e| format!("{e:?}")).collect();
                let params: Vec<String> = f
                    .params
                    .iter()
                    .map(|p| format!("{}:{:?}", p.name, p.ty))
                    .collect();
                out.push_str(&format!(
                    "    {} ({})->{:?} [{}]\n",
                    f.name,
                    params.join(", "),
                    f.return_type,
                    effects.join(", ")
                ));
            }
        } else if let Some(sig) = table.resolve_function(scope) {
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

fn query_routes(http_routes: &[crate::ast::HttpRoute]) -> String {
    if http_routes.is_empty() {
        return "No HTTP routes registered.".to_string();
    }
    let mut out = String::from("HTTP Routes:\n");
    for route in http_routes {
        out.push_str(&format!(
            "  {} {} -> {}\n",
            route.method, route.path, route.handler_fn
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parser, validator};

    // ── Helpers ──────────────────────────────────────────────────────────

    /// Parse and validate Adapsis source into a Program.
    fn build_program(source: &str) -> Program {
        let ops = parser::parse(source).expect("parse failed");
        let mut program = Program::default();
        for op in &ops {
            match op {
                parser::Operation::Test(_) | parser::Operation::Eval(_) => {}
                _ => {
                    validator::apply_and_validate(&mut program, op).expect("validation failed");
                }
            }
        }
        program.rebuild_function_index();
        program
    }

    /// Build program + type-check a specific function. Returns errors.
    fn check(source: &str, fn_name: &str) -> Vec<String> {
        let program = build_program(source);
        let table = build_symbol_table(&program);
        let func = program
            .get_function(fn_name)
            .unwrap_or_else(|| panic!("function `{fn_name}` not found"));
        check_function(&table, func)
    }

    // ═════════════════════════════════════════════════════════════════════
    // 1. Symbol table construction
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn symbol_table_registers_types() {
        let program = build_program("+type User = id:Int, name:String");
        let table = build_symbol_table(&program);
        let info = table.resolve_type("User").expect("User type not found");
        assert!(!info.is_union);
        assert_eq!(info.fields.len(), 2);
        assert_eq!(info.fields[0].0, "id");
        assert_eq!(info.fields[1].0, "name");
    }

    #[test]
    fn symbol_table_registers_unions() {
        let program = build_program("+type Color = Red | Green | Blue");
        let table = build_symbol_table(&program);
        let info = table.resolve_type("Color").expect("Color type not found");
        assert!(info.is_union);
        assert_eq!(info.variants.len(), 3);
        assert_eq!(info.variants[0].0, "Red");
        assert_eq!(info.variants[1].0, "Green");
        assert_eq!(info.variants[2].0, "Blue");
    }

    #[test]
    fn symbol_table_registers_union_with_payloads() {
        let program = build_program("+type Shape = Circle(Float) | Rect(Float, Float) | Point");
        let table = build_symbol_table(&program);
        let info = table.resolve_type("Shape").unwrap();
        assert!(info.is_union);
        assert_eq!(info.variants[0].0, "Circle");
        assert_eq!(info.variants[0].1.len(), 1); // one Float payload
        assert_eq!(info.variants[1].0, "Rect");
        assert_eq!(info.variants[1].1.len(), 2); // two Float payloads
        assert_eq!(info.variants[2].0, "Point");
        assert!(info.variants[2].1.is_empty());
    }

    #[test]
    fn symbol_table_registers_functions() {
        let source = "\
+fn add (a:Int, b:Int)->Int
  +return a + b
";
        let program = build_program(source);
        let table = build_symbol_table(&program);
        let sig = table.resolve_function("add").expect("add not found");
        assert_eq!(sig.params.len(), 2);
        assert_eq!(sig.params[0].0, "a");
        assert!(matches!(sig.params[0].1, Type::Int));
        assert!(matches!(sig.return_type, Type::Int));
        assert!(sig.effects.is_empty());
    }

    #[test]
    fn symbol_table_registers_effects() {
        let source = "\
+fn fetch (url:String)->String [io,async]
  +return url
";
        let program = build_program(source);
        let table = build_symbol_table(&program);
        let sig = table.resolve_function("fetch").unwrap();
        assert_eq!(sig.effects.len(), 2);
        assert!(sig.effects.contains(&Effect::Io));
        assert!(sig.effects.contains(&Effect::Async));
    }

    #[test]
    fn symbol_table_module_functions_qualified() {
        let source = "\
+module Math
+fn add (a:Int, b:Int)->Int
  +return a + b
";
        let program = build_program(source);
        let table = build_symbol_table(&program);
        // Should be registered as Math.add
        assert!(table.resolve_function("Math.add").is_some());
        // Also registered as bare add for internal resolution
        assert!(table.resolve_function("add").is_some());
    }

    // ═════════════════════════════════════════════════════════════════════
    // 2. Type checking: struct field access
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn check_struct_field_access_ok() {
        let errors = check(
            "\
+type User = name:String, age:Int

+fn get_name (u:User)->String
  +return u.name
",
            "get_name",
        );
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
    }

    #[test]
    fn check_struct_field_wrong_return_type() {
        let errors = check(
            "\
+type User = name:String, age:Int

+fn get_name (u:User)->Int
  +return u.name
",
            "get_name",
        );
        // u.name is String but return type is Int — should flag mismatch
        assert!(
            !errors.is_empty(),
            "expected type mismatch error for String vs Int"
        );
        assert!(
            errors.iter().any(|e| e.contains("mismatch")),
            "errors: {errors:?}"
        );
    }

    // ═════════════════════════════════════════════════════════════════════
    // 3. Type checking: function call argument count
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn check_call_correct_arg_count() {
        let errors = check(
            "\
+fn add (a:Int, b:Int)->Int
  +return a + b

+fn use_add ()->Int
  +call result:Int = add(1, 2)
  +return result
",
            "use_add",
        );
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
    }

    #[test]
    fn check_call_wrong_arg_count() {
        let errors = check(
            "\
+fn add (a:Int, b:Int)->Int
  +return a + b

+fn use_add ()->Int
  +call result:Int = add(1)
  +return result
",
            "use_add",
        );
        assert!(!errors.is_empty(), "expected arg count mismatch");
        assert!(
            errors
                .iter()
                .any(|e| e.contains("expects 2 arguments, got 1")),
            "errors: {errors:?}"
        );
    }

    // ═════════════════════════════════════════════════════════════════════
    // 4. Type checking: let binding type mismatch
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn check_let_type_ok() {
        let errors = check(
            "\
+fn make_msg ()->String
  +let msg:String = \"hello\"
  +return msg
",
            "make_msg",
        );
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
    }

    #[test]
    fn check_let_type_mismatch() {
        let errors = check(
            "\
+fn bad ()->String
  +let x:String = 42
  +return x
",
            "bad",
        );
        assert!(!errors.is_empty(), "expected type mismatch in let");
        assert!(
            errors.iter().any(|e| e.contains("type mismatch")),
            "errors: {errors:?}"
        );
    }

    // ═════════════════════════════════════════════════════════════════════
    // 5. Type checking: return type mismatch
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn check_return_type_mismatch() {
        let errors = check(
            "\
+fn bad ()->Int
  +return \"hello\"
",
            "bad",
        );
        assert!(!errors.is_empty(), "expected return type mismatch");
        assert!(
            errors.iter().any(|e| e.contains("return type mismatch")),
            "errors: {errors:?}"
        );
    }

    #[test]
    fn check_return_result_inner_type_ok() {
        // Returning an Int from a Result<Int> function should be fine (auto-wrap)
        let errors = check(
            "\
+fn validate (x:Int)->Result<Int> [fail]
  +check pos x > 0 ~err_neg
  +return x
",
            "validate",
        );
        assert!(
            errors.is_empty(),
            "expected no errors for Result<Int> returning Int, got: {errors:?}"
        );
    }

    // ═════════════════════════════════════════════════════════════════════
    // 6. Type checking: check condition must be Bool
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn check_condition_is_bool() {
        let errors = check(
            "\
+fn validate (x:Int)->Result<Int> [fail]
  +check pos x > 0 ~err_neg
  +return x
",
            "validate",
        );
        // x > 0 is Bool — should be fine
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
    }

    // ═════════════════════════════════════════════════════════════════════
    // 7. Type checking: set variable type mismatch
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn check_set_type_ok() {
        let errors = check(
            "\
+fn inc ()->Int
  +let x:Int = 0
  +set x = x + 1
  +return x
",
            "inc",
        );
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
    }

    #[test]
    fn check_set_type_mismatch() {
        let errors = check(
            "\
+fn bad ()->Int
  +let x:Int = 0
  +set x = \"hello\"
  +return x
",
            "bad",
        );
        assert!(!errors.is_empty(), "expected set type mismatch");
        assert!(
            errors.iter().any(|e| e.contains("type mismatch in set")),
            "errors: {errors:?}"
        );
    }

    #[test]
    fn check_set_undefined_variable() {
        let errors = check(
            "\
+fn bad ()->Int
  +set x = 42
  +return 0
",
            "bad",
        );
        assert!(!errors.is_empty(), "expected undefined variable error");
        assert!(
            errors.iter().any(|e| e.contains("undefined variable")),
            "errors: {errors:?}"
        );
    }

    // ═════════════════════════════════════════════════════════════════════
    // 8. Type checking: while condition must be Bool
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn check_while_condition_ok() {
        let errors = check(
            "\
+fn count ()->Int
  +let i:Int = 0
  +while i < 10
    +set i = i + 1
  +end
  +return i
",
            "count",
        );
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
    }

    // ═════════════════════════════════════════════════════════════════════
    // 9. Call graph construction
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn call_graph_basic() {
        let source = "\
+fn double (x:Int)->Int
  +return x * 2

+fn quadruple (x:Int)->Int
  +call d:Int = double(x)
  +call r:Int = double(d)
  +return r
";
        let program = build_program(source);
        let graph = build_call_graph(&program);
        let callees = graph
            .callees
            .get("quadruple")
            .expect("quadruple not in graph");
        assert!(
            callees.contains(&"double".to_string()),
            "quadruple should call double"
        );
    }

    #[test]
    fn call_graph_no_calls() {
        let source = "\
+fn identity (x:Int)->Int
  +return x
";
        let program = build_program(source);
        let graph = build_call_graph(&program);
        let callees = graph.callees.get("identity").unwrap();
        assert!(callees.is_empty(), "identity should call nothing");
    }

    #[test]
    fn call_graph_module_functions() {
        let source = "\
+module Math
+fn add (a:Int, b:Int)->Int
  +return a + b
+fn sum3 (a:Int, b:Int, c:Int)->Int
  +call ab:Int = add(a, b)
  +call result:Int = add(ab, c)
  +return result
";
        let program = build_program(source);
        let graph = build_call_graph(&program);
        let callees = graph
            .callees
            .get("Math.sum3")
            .expect("Math.sum3 not in graph");
        assert!(!callees.is_empty(), "sum3 should call add");
    }

    // ═════════════════════════════════════════════════════════════════════
    // 10. Expression reconstruction
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn reconstruct_literal_int() {
        let expr = Expr::Literal(Literal::Int(42));
        assert_eq!(reconstruct_expr(&expr), "42");
    }

    #[test]
    fn reconstruct_literal_string() {
        let expr = Expr::Literal(Literal::String("hello world".to_string()));
        assert_eq!(reconstruct_expr(&expr), "\"hello world\"");
    }

    #[test]
    fn reconstruct_literal_bool() {
        assert_eq!(
            reconstruct_expr(&Expr::Literal(Literal::Bool(true))),
            "true"
        );
        assert_eq!(
            reconstruct_expr(&Expr::Literal(Literal::Bool(false))),
            "false"
        );
    }

    #[test]
    fn reconstruct_field_access() {
        let expr = Expr::FieldAccess {
            base: Box::new(Expr::Identifier("user".to_string())),
            field: "name".to_string(),
        };
        assert_eq!(reconstruct_expr(&expr), "user.name");
    }

    #[test]
    fn reconstruct_binary_expr() {
        let expr = Expr::Binary {
            left: Box::new(Expr::Identifier("a".to_string())),
            op: BinaryOp::Add,
            right: Box::new(Expr::Literal(Literal::Int(1))),
        };
        assert_eq!(reconstruct_expr(&expr), "a+1");
    }

    #[test]
    fn reconstruct_call_expr() {
        let expr = Expr::Call(CallExpr {
            callee: "concat".to_string(),
            args: vec![
                Expr::Literal(Literal::String("hello".to_string())),
                Expr::Literal(Literal::String(" world".to_string())),
            ],
        });
        assert_eq!(reconstruct_expr(&expr), r#"concat("hello", " world")"#);
    }

    // ═════════════════════════════════════════════════════════════════════
    // 11. Query handling (handle_query)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn query_symbols_lists_types_and_functions() {
        let source = "\
+type Point = x:Int, y:Int

+fn origin ()->Point
  +return {x: 0, y: 0}
";
        let program = build_program(source);
        let table = build_symbol_table(&program);
        let result = handle_query(&program, &table, "?symbols", &[]);
        assert!(result.contains("Point"), "should list Point type: {result}");
        assert!(
            result.contains("origin"),
            "should list origin function: {result}"
        );
    }

    #[test]
    fn query_callees() {
        let source = "\
+fn double (x:Int)->Int
  +return x * 2

+fn quad (x:Int)->Int
  +call d:Int = double(x)
  +call r:Int = double(d)
  +return r
";
        let program = build_program(source);
        let table = build_symbol_table(&program);
        let result = handle_query(&program, &table, "?callees quad", &[]);
        assert!(
            result.contains("double"),
            "quad should call double: {result}"
        );
    }

    #[test]
    fn query_empty() {
        let program = Program::default();
        let table = build_symbol_table(&program);
        let result = handle_query(&program, &table, "", &[]);
        assert_eq!(result, "empty query");
    }

    // ═════════════════════════════════════════════════════════════════════
    // 12. Result<T> and Option<T> generic type validation
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn types_compatible_result_match() {
        let table = SymbolTable::default();
        let r1 = Type::Result(Box::new(Type::Int));
        let r2 = Type::Result(Box::new(Type::Int));
        assert!(types_compatible(&r1, &r2, &table));
    }

    #[test]
    fn types_compatible_result_mismatch() {
        let table = SymbolTable::default();
        let r1 = Type::Result(Box::new(Type::Int));
        let r2 = Type::Result(Box::new(Type::String));
        assert!(!types_compatible(&r1, &r2, &table));
    }

    #[test]
    fn types_compatible_option_match() {
        let table = SymbolTable::default();
        let o1 = Type::Option(Box::new(Type::String));
        let o2 = Type::Option(Box::new(Type::String));
        assert!(types_compatible(&o1, &o2, &table));
    }

    #[test]
    fn types_compatible_list_match() {
        let table = SymbolTable::default();
        let l1 = Type::List(Box::new(Type::Int));
        let l2 = Type::List(Box::new(Type::Int));
        assert!(types_compatible(&l1, &l2, &table));
    }

    #[test]
    fn types_compatible_list_mismatch() {
        let table = SymbolTable::default();
        let l1 = Type::List(Box::new(Type::Int));
        let l2 = Type::List(Box::new(Type::String));
        assert!(!types_compatible(&l1, &l2, &table));
    }

    #[test]
    fn types_compatible_int_float_coercion() {
        let table = SymbolTable::default();
        assert!(types_compatible(&Type::Int, &Type::Float, &table));
        assert!(types_compatible(&Type::Float, &Type::Int, &table));
    }

    #[test]
    fn types_incompatible_int_string() {
        let table = SymbolTable::default();
        assert!(!types_compatible(&Type::Int, &Type::String, &table));
    }

    // ═════════════════════════════════════════════════════════════════════
    // 13. collect_callees_from_stmts
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn collect_callees_none() {
        let source = "\
+fn identity (x:Int)->Int
  +return x
";
        let program = build_program(source);
        let func = program.get_function("identity").unwrap();
        let callees = collect_callees_from_stmts(&func.body);
        assert!(callees.is_empty());
    }

    #[test]
    fn collect_callees_from_call() {
        let source = "\
+fn double (x:Int)->Int
  +return x * 2
+fn use_double (x:Int)->Int
  +call r:Int = double(x)
  +return r
";
        let program = build_program(source);
        let func = program.get_function("use_double").unwrap();
        let callees = collect_callees_from_stmts(&func.body);
        assert!(callees.contains(&"double".to_string()));
    }

    // ═════════════════════════════════════════════════════════════════════
    // Effect registration and querying
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn symbol_table_registers_fail_effect() {
        let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check pos x > 0 ~err_neg
  +return x
";
        let program = build_program(source);
        let table = build_symbol_table(&program);
        let sig = table.resolve_function("validate").unwrap();
        assert!(sig.effects.contains(&Effect::Fail));
        assert!(matches!(sig.return_type, Type::Result(_)));
    }

    #[test]
    fn symbol_table_registers_mut_effect() {
        let source = "\
+fn counter ()->Int [mut]
  +return 0
";
        let program = build_program(source);
        let table = build_symbol_table(&program);
        let sig = table.resolve_function("counter").unwrap();
        assert!(sig.effects.contains(&Effect::Mut));
    }

    #[test]
    fn symbol_table_no_effects_for_pure() {
        let source = "\
+fn add (a:Int, b:Int)->Int
  +return a + b
";
        let program = build_program(source);
        let table = build_symbol_table(&program);
        let sig = table.resolve_function("add").unwrap();
        assert!(
            sig.effects.is_empty(),
            "pure function should have no effects"
        );
    }

    // ═════════════════════════════════════════════════════════════════════
    // Match arm body type checking
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn check_match_arm_return_type_ok() {
        let source = "\
+type Color = Red | Green | Blue

+fn color_name (c:Color)->String
  +match c
  +case Red
    +return \"red\"
  +case Green
    +return \"green\"
  +case Blue
    +return \"blue\"
  +end
";
        let errors = check(source, "color_name");
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
    }

    #[test]
    fn check_match_arm_return_type_mismatch() {
        let source = "\
+type Color = Red | Green | Blue

+fn color_name (c:Color)->String
  +match c
  +case Red
    +return 42
  +case Green
    +return \"green\"
  +case Blue
    +return \"blue\"
  +end
";
        let errors = check(source, "color_name");
        assert!(
            !errors.is_empty(),
            "expected return type mismatch in Red arm"
        );
        assert!(
            errors.iter().any(|e| e.contains("return type mismatch")),
            "errors: {errors:?}"
        );
    }

    #[test]
    fn check_match_arm_let_binding_ok() {
        let source = "\
+type Shape = Circle(Float) | Rect(Float, Float)

+fn describe (s:Shape)->String
  +match s
  +case Circle(r)
    +let msg:String = \"circle\"
    +return msg
  +case Rect(w, h)
    +let msg:String = \"rect\"
    +return msg
  +end
";
        let errors = check(source, "describe");
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
    }

    // ═════════════════════════════════════════════════════════════════════
    // Branch (if/elif/else) body type checking
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn check_branch_both_arms_type_ok() {
        let source = "\
+fn abs_val (x:Int)->Int
  +if x >= 0
    +return x
  +else
    +let neg:Int = 0 - x
    +return neg
  +end
";
        let errors = check(source, "abs_val");
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
    }

    #[test]
    fn check_branch_return_mismatch_in_else() {
        let source = "\
+fn bad (x:Int)->Int
  +if x > 0
    +return x
  +else
    +return \"negative\"
  +end
";
        let errors = check(source, "bad");
        assert!(
            !errors.is_empty(),
            "expected return type mismatch in else arm"
        );
    }

    // ═════════════════════════════════════════════════════════════════════
    // Each loop type checking
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn check_each_body_type_ok() {
        let source = "\
+fn sum_list (items:List<Int>)->Int
  +let total:Int = 0
  +each items item:Int
    +set total = total + item
  +end
  +return total
";
        let errors = check(source, "sum_list");
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
    }

    #[test]
    fn check_each_set_mismatch_in_body() {
        let source = "\
+fn bad (items:List<Int>)->Int
  +let total:Int = 0
  +each items item:Int
    +set total = \"hello\"
  +end
  +return total
";
        let errors = check(source, "bad");
        assert!(
            !errors.is_empty(),
            "expected set type mismatch inside each body"
        );
        assert!(
            errors.iter().any(|e| e.contains("type mismatch in set")),
            "errors: {errors:?}"
        );
    }

    // ═════════════════════════════════════════════════════════════════════
    // Call with too many arguments
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn check_call_too_many_args() {
        let source = "\
+fn greet (name:String)->String
  +return name

+fn use_greet ()->String
  +call r:String = greet(\"alice\", \"extra\")
  +return r
";
        let errors = check(source, "use_greet");
        assert!(!errors.is_empty(), "expected arg count mismatch");
        assert!(
            errors
                .iter()
                .any(|e| e.contains("expects 1 arguments, got 2")),
            "errors: {errors:?}"
        );
    }

    // ═════════════════════════════════════════════════════════════════════
    // Complex program with multiple types and functions
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn check_complex_program_no_errors() {
        let source = "\
+type User = name:String, age:Int
+type Color = Red | Green | Blue

+fn create_user (name:String, age:Int)->User
  +let u:User = {name: name, age: age}
  +return u

+fn get_age (u:User)->Int
  +return u.age

+fn color_code (c:Color)->Int
  +match c
  +case Red
    +return 1
  +case Green
    +return 2
  +case Blue
    +return 3
  +end
";
        let program = build_program(source);
        let table = build_symbol_table(&program);
        for func in &program.functions {
            let errors = check_function(&table, func);
            assert!(errors.is_empty(), "errors in {}: {errors:?}", func.name);
        }
    }

    #[test]
    fn check_module_function_resolution() {
        let source = "\
+module Utils
+fn double (x:Int)->Int
  +return x * 2
";
        let program = build_program(source);
        let table = build_symbol_table(&program);
        // Both qualified and bare should resolve
        assert!(table.resolve_function("Utils.double").is_some());
        assert!(table.resolve_function("double").is_some());
        let sig = table.resolve_function("Utils.double").unwrap();
        assert_eq!(sig.params.len(), 1);
        assert!(matches!(sig.return_type, Type::Int));
    }

    // ═════════════════════════════════════════════════════════════════════
    // Query: ?source reconstruction roundtrip
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn query_source_returns_function_source() {
        let source = "\
+fn double (x:Int)->Int
  +let result:Int = x * 2
  +return result
";
        let program = build_program(source);
        let table = build_symbol_table(&program);
        let result = handle_query(&program, &table, "?source double", &[]);
        assert!(
            result.contains("+fn double"),
            "should contain function header: {result}"
        );
        assert!(
            result.contains("+return"),
            "should contain return stmt: {result}"
        );
    }

    #[test]
    fn query_source_unknown_function() {
        let program = build_program("+fn noop ()->Int\n  +return 0\n");
        let table = build_symbol_table(&program);
        let result = handle_query(&program, &table, "?source nonexistent", &[]);
        assert!(
            result.contains("not found") || result.contains("No match"),
            "should indicate not found: {result}"
        );
    }
}
