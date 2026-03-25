use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::ast;
use crate::compiler::CompiledProgram;
use crate::parser;

/// Cache for JIT-compiled programs, keyed by session revision.
/// When the revision matches, the compiled module is reused instead of recompiling.
pub type JitCache = Arc<Mutex<Option<(usize, CompiledProgram)>>>;

/// Create a new empty JIT cache.
pub fn new_jit_cache() -> JitCache {
    Arc::new(Mutex::new(None))
}

/// A runtime value during evaluation.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum Value {
    CoroutineHandle(crate::coroutine::CoroutineHandle),
    StateHandle(std::sync::Arc<std::sync::Mutex<Value>>),
    TaskHandle(crate::coroutine::TaskId),
    Union {
        variant: String,
        payload: Vec<Value>,
    },
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Struct(String, HashMap<String, Value>),
    List(Vec<Value>),
    Ok(Box<Value>),
    Err(String),
    None,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(v) => write!(f, "{v}"),
            Value::Float(v) => write!(f, "{v}"),
            Value::Bool(v) => write!(f, "{v}"),
            Value::String(v) => write!(f, "\"{v}\""),
            Value::Struct(name, fields) => {
                write!(f, "{name}{{")?;
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k}: {v}")?;
                }
                write!(f, "}}")
            }
            Value::List(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{item}")?;
                }
                write!(f, "]")
            }
            Value::Ok(v) => write!(f, "Ok({v})"),
            Value::Err(msg) => write!(f, "Err({msg})"),
            Value::None => write!(f, "None"),
            Value::Union { variant, payload } => {
                if payload.is_empty() {
                    write!(f, "{variant}")
                } else {
                    write!(f, "{variant}(")?;
                    for (i, v) in payload.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{v}")?;
                    }
                    write!(f, ")")
                }
            }
            Value::CoroutineHandle(_) => write!(f, "<coroutine>"),
            Value::TaskHandle(id) => write!(f, "<task:{id}>"),
            Value::StateHandle(s) => {
                let val = s.lock().unwrap();
                write!(f, "<state:{val}>")
            }
        }
    }
}

impl Value {
    fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Int(n) => *n != 0,
            Value::String(s) => !s.is_empty(),
            Value::None => false,
            Value::Err(_) => false,
            _ => true,
        }
    }

    /// Check structural equality for test assertions.
    fn matches(&self, other: &Value) -> bool {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => (a - b).abs() < f64::EPSILON,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Ok(a), Value::Ok(b)) => {
                // Ok(None) is a wildcard — matches any Ok value
                matches!(b.as_ref(), Value::None) || a.matches(b)
            }
            (Value::Err(a), Value::Err(b)) => a == b,
            (Value::None, Value::None) => true,
            (Value::Struct(n1, f1), Value::Struct(n2, f2)) => {
                // Allow empty name to match any struct name
                (n1.is_empty() || n2.is_empty() || n1 == n2)
                    && f1.len() == f2.len()
                    && f1
                        .iter()
                        .all(|(k, v)| f2.get(k).is_some_and(|v2| v.matches(v2)))
            }
            (Value::List(a), Value::List(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.matches(y))
            }
            (
                Value::Union {
                    variant: v1,
                    payload: p1,
                },
                Value::Union {
                    variant: v2,
                    payload: p2,
                },
            ) => v1 == v2 && p1.len() == p2.len() && p1.iter().zip(p2).all(|(a, b)| a.matches(b)),
            _ => false,
        }
    }
}

/// Evaluation environment with a scope stack.
/// Each scope is a HashMap of variable bindings. `+let` defines into the
/// top scope, `+set` mutates the nearest scope that already contains the
/// variable, and lookups walk the stack from top to bottom.
pub struct Env {
    scopes: Vec<HashMap<String, Value>>,
    /// Shared runtime state for +shared variable access.
    shared_runtime: Option<crate::session::SharedRuntime>,
    /// Local cache of shared vars (key = "Module.name") for borrow-friendly reads.
    shared_cache: HashMap<String, Value>,
}

impl Env {
    pub fn new() -> Self {
        let mut env = Self {
            scopes: vec![HashMap::new()],
            shared_runtime: None,
            shared_cache: HashMap::new(),
        };
        // Auto-pick up thread-local SharedRuntime if set
        SHARED_RUNTIME.with(|rt| {
            if let Some(rt) = rt.borrow().as_ref() {
                env.set_runtime(rt.clone());
            }
        });
        env
    }

    /// Attach shared runtime state for +shared variable access.
    pub fn set_runtime(&mut self, rt: crate::session::SharedRuntime) {
        // Pre-populate the local cache from the runtime's current shared_vars.
        if let Ok(state) = rt.read() {
            for (key, val) in &state.shared_vars {
                self.shared_cache.insert(key.clone(), val.clone());
            }
        }
        self.shared_runtime = Some(rt);
    }

    /// Populate the shared_cache directly from the program's module shared var
    /// declarations. This is a fallback for when SharedRuntime is not available
    /// (e.g. in tests or CLI mode). Evaluates each default expression.
    pub fn populate_shared_from_program(&mut self, program: &ast::Program) {
        for module in &program.modules {
            for sv in &module.shared_vars {
                let key = format!("{}.{}", module.name, sv.name);
                if !self.shared_cache.contains_key(&key) {
                    let value = eval_expr_standalone(program, &sv.default)
                        .unwrap_or(Value::Int(0));
                    self.shared_cache.insert(key, value);
                }
            }
        }
    }

    /// Push a new empty scope onto the stack.
    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    /// Pop the top scope. Panics if only the root scope remains.
    fn pop_scope(&mut self) {
        debug_assert!(self.scopes.len() > 1, "cannot pop the root scope");
        self.scopes.pop();
    }

    /// Define a new variable in the current (top) scope.
    /// Used by `+let` and parameter binding.
    pub fn set(&mut self, name: &str, value: Value) {
        self.scopes
            .last_mut()
            .expect("scope stack empty")
            .insert(name.to_string(), value);
    }

    /// Mutate an existing variable: walk scopes top-to-bottom, update the
    /// first scope that contains `name`. If not found in local scopes, check
    /// shared vars (keyed by "Module.name"). If still not found, insert into
    /// the top scope (same as `set`). Used by `+set`.
    fn set_existing(&mut self, name: &str, value: Value) {
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), value);
                return;
            }
        }
        // Check shared vars: derive "Module.name" key from current function context
        if !self.shared_cache.is_empty() {
            let module_name = FN_NAME_STACK.with(|s| {
                let stack = s.borrow();
                stack.last().and_then(|fn_name| fn_name.split_once('.').map(|(m, _)| m.to_string()))
            });
            if let Some(module) = module_name {
                let key = format!("{module}.{name}");
                if self.shared_cache.contains_key(&key) {
                    self.shared_cache.insert(key.clone(), value.clone());
                    // Write through to runtime
                    if let Some(rt) = &self.shared_runtime {
                        if let Ok(mut state) = rt.write() {
                            state.shared_vars.insert(key, value);
                        }
                    }
                    return;
                }
            }
        }
        // Not found anywhere — define in current scope
        self.set(name, value);
    }

    /// Look up a variable by walking scopes from top to bottom.
    /// Falls back to shared vars cache (keyed by "Module.name") if not found locally.
    fn get(&self, name: &str) -> Result<&Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(val) = scope.get(name) {
                return Ok(val);
            }
        }
        // Check shared vars: derive "Module.name" key from current function context
        if !self.shared_cache.is_empty() {
            let module_name = FN_NAME_STACK.with(|s| {
                let stack = s.borrow();
                stack.last().and_then(|fn_name| fn_name.split_once('.').map(|(m, _)| m.to_string()))
            });
            if let Some(module) = module_name {
                let key = format!("{module}.{name}");
                if let Some(val) = self.shared_cache.get(&key) {
                    return Ok(val);
                }
            }
        }
        Err(anyhow!("undefined variable `{name}`"))
    }

    /// Raw lookup (returns Option) — used for special variables like __coroutine_handle.
    fn get_raw(&self, name: &str) -> Option<&Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(val) = scope.get(name) {
                return Some(val);
            }
        }
        None
    }

    /// Flatten all visible bindings into display-string pairs for task inspection.
    /// Skips internal names (prefixed with `__`) and deduplicates across scopes
    /// (inner scopes shadow outer ones).
    pub fn snapshot_bindings(&self) -> Vec<(String, String)> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        // Walk from top (innermost) to bottom (outermost) so shadowed names are skipped.
        for scope in self.scopes.iter().rev() {
            for (name, val) in scope {
                if name.starts_with("__") {
                    continue;
                }
                if seen.insert(name.clone()) {
                    result.push((name.clone(), format!("{val}")));
                }
            }
        }
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }
}

/// Try to evaluate using the JIT compiler, falling back to the interpreter.
/// Returns (result_string, was_compiled).
///
/// If `cache` is provided along with a `revision`, the compiled module is reused
/// when the revision hasn't changed, avoiding recompilation on every eval call.
/// Find similar function names for "did you mean" suggestions.
pub fn suggest_similar(program: &ast::Program, name: &str) -> String {
    let bare = name.rsplit('.').next().unwrap_or(name);
    let mut candidates: Vec<String> = Vec::new();
    for m in &program.modules {
        for f in &m.functions {
            let qname = format!("{}.{}", m.name, f.name);
            if f.name == bare || f.name.contains(bare) || bare.contains(&f.name) {
                candidates.push(qname);
            }
        }
    }
    for f in &program.functions {
        if f.name.contains(bare) || bare.contains(&f.name) {
            candidates.push(f.name.clone());
        }
    }
    if candidates.is_empty() { String::new() }
    else { format!(". Did you mean: {}?", candidates.join(", ")) }
}

pub fn eval_compiled_or_interpreted(
    program: &ast::Program,
    function_name: &str,
    input: &parser::Expr,
) -> Result<(String, bool)> {
    eval_compiled_or_interpreted_cached(program, function_name, input, None, 0)
}

/// Like `eval_compiled_or_interpreted`, but with an optional JIT cache.
pub fn eval_compiled_or_interpreted_cached(
    program: &ast::Program,
    function_name: &str,
    input: &parser::Expr,
    cache: Option<&JitCache>,
    revision: usize,
) -> Result<(String, bool)> {
    // If it's a builtin, use eval_call_with_input directly
    if program.get_function(function_name).is_none() && crate::builtins::is_builtin(function_name) {
        let result = eval_call_with_input(program, function_name, input)?;
        return Ok((result, false));
    }

    let func = program
        .get_function(function_name)
        .ok_or_else(|| anyhow!("function `{function_name}` not found{}", crate::eval::suggest_similar(program, function_name)))?;

    // Try compiled path
    let returns_string = matches!(&func.return_type, ast::Type::String);
    if crate::compiler::is_compilable_function(func) {
        if let Ok(args) = input_to_i64_args(input, func) {
            // Try to use cached compiled module
            if let Some(jit_cache) = cache {
                if let Ok(mut guard) = jit_cache.lock() {
                    // Check if cache is valid for this revision
                    let needs_compile = match &*guard {
                        Some((cached_rev, _)) => *cached_rev != revision,
                        None => true,
                    };

                    if needs_compile {
                        // Compile and store in cache
                        if let Ok(compiled) = crate::compiler::compile(program) {
                            *guard = Some((revision, compiled));
                        }
                    }

                    // Try to use the cached compiled module
                    if let Some((_, ref mut compiled)) = *guard {
                        if returns_string {
                            if let Ok(result) = compiled.call_string(function_name, &args) {
                                return Ok((format!("\"{result}\""), true));
                            }
                        } else {
                            if let Ok(result) = compiled.call_i64(function_name, &args) {
                                return Ok((format!("{result}"), true));
                            }
                        }
                    }
                }
            } else {
                // No cache — compile fresh each time (original behavior)
                if let Ok(mut compiled) = crate::compiler::compile(program) {
                    if returns_string {
                        if let Ok(result) = compiled.call_string(function_name, &args) {
                            return Ok((format!("\"{result}\""), true));
                        }
                    } else {
                        if let Ok(result) = compiled.call_i64(function_name, &args) {
                            return Ok((format!("{result}"), true));
                        }
                    }
                }
            }
        }
    }

    // Fall back to interpreter
    let result = eval_call_with_input(program, function_name, input)?;
    Ok((result, false))
}

/// Try to convert a parser expression to i64 args for the compiler.
fn input_to_i64_args(input: &parser::Expr, func: &ast::FunctionDecl) -> Result<Vec<i64>> {
    match input {
        parser::Expr::Int(n) => Ok(vec![*n]),
        parser::Expr::Bool(b) => Ok(vec![*b as i64]),
        parser::Expr::StructLiteral(fields) => {
            // Match field names to param order
            let mut args = Vec::new();
            for param in &func.params {
                let field = fields
                    .iter()
                    .find(|f| f.name == param.name)
                    .ok_or_else(|| anyhow!("missing field `{}`", param.name))?;
                match &field.value {
                    parser::Expr::Int(n) => args.push(*n),
                    parser::Expr::Bool(b) => args.push(*b as i64),
                    parser::Expr::Float(f) => args.push(f.to_bits() as i64), // bit-cast
                    _ => bail!("unsupported arg type for compiled call"),
                }
            }
            Ok(args)
        }
        _ => bail!("unsupported input format for compiled call"),
    }
}

/// Evaluate a function call with given input, returning the result as a displayable string.
pub fn eval_call_with_input(
    program: &ast::Program,
    function_name: &str,
    input: &parser::Expr,
) -> Result<String> {
    // Try user-defined function first
    if let Some(func) = program.get_function(function_name) {
        let input_val = eval_parser_expr_standalone(input)?;
        let mut env = Env::new();
        bind_input_to_params(program, func, &input_val, &mut env);

        let returns_result = matches!(&func.return_type, ast::Type::Result(_));

        return match eval_function_body(program, &func.body, &mut env) {
            Ok(val) => {
                let result = if returns_result {
                    match &val {
                        Value::Ok(_) | Value::Err(_) => val,
                        _ => Value::Ok(Box::new(val)),
                    }
                } else {
                    val
                };
                Ok(format!("{result}"))
            }
            Err(e) => Ok(format!("Err({e})")),
        };
    }

    // Try as a builtin function
    if crate::builtins::is_builtin(function_name) {
        // Get args in order from the parser expression (not via HashMap which loses order)
        let args = match input {
            parser::Expr::StructLiteral(fields) => {
                // Preserve field order from the parser
                fields
                    .iter()
                    .map(|f| eval_parser_expr_standalone(&f.value))
                    .collect::<Result<Vec<_>>>()?
            }
            _ => {
                let input_val = eval_parser_expr_standalone(input)?;
                match &input_val {
                    Value::None => vec![],
                    other => vec![other.clone()],
                }
            }
        };
        let mut env = Env::new();
        let call = ast::CallExpr {
            callee: function_name.to_string(),
            args: vec![], // not used — we call eval_call_inner directly
        };
        // Build a fake call and evaluate using the builtin path
        match eval_call_inner_with_args(program, function_name, args, &mut env) {
            Ok(val) => Ok(format!("{val}")),
            Err(e) => Ok(format!("Err({e})")),
        }
    } else {
        bail!("function `{function_name}` not found (not a user function or builtin)")
    }
}

/// Evaluate an inline expression directly (not a function call).
/// Used for `!eval 1 + 2`, `!eval concat("a", "b")`, etc.
pub fn eval_inline_expr(program: &ast::Program, expr: &parser::Expr) -> Result<Value> {
    eval_parser_expr_with_program(expr, program)
}

/// Evaluate a test case against a function in the program.
/// Bind test input values to function parameters.
/// Handles three cases:
/// 1. Single param → bind directly
/// 2. Multi-param, all fields match param names → bind each field to its param
/// 3. Multi-param with struct params → distribute fields based on type definitions
pub fn bind_input_to_params(
    program: &ast::Program,
    func: &ast::FunctionDecl,
    input: &Value,
    env: &mut Env,
) {
    match (input, func.params.len()) {
        (_, 0) => {} // no params
        (Value::Struct(_, fields), n) if n == 1 => {
            // Single struct-typed param — pass the whole struct
            env.set(&func.params[0].name, input.clone());
            // Also expose fields directly for field access (e.g., input.name)
            for (k, v) in fields {
                env.set(k, v.clone());
            }
        }
        (Value::Struct(_, fields), _) => {
            // Multi-param function with struct input (key=value pairs)
            // First, check if all fields directly match param names
            let all_match = func.params.iter().all(|p| fields.contains_key(&p.name));

            if all_match {
                // Direct match: a=3 b=4 for (a:Int, b:Int)
                for param in &func.params {
                    if let Some(val) = fields.get(&param.name) {
                        env.set(&param.name, val.clone());
                    }
                }
            } else {
                // Check for positional fields (_0, _1, ...) from space-separated args
                let is_positional = fields.keys().any(|k| k.starts_with('_') && k[1..].parse::<usize>().is_ok());
                if is_positional && fields.len() == func.params.len() {
                    for (i, param) in func.params.iter().enumerate() {
                        if let Some(val) = fields.get(&format!("_{i}")) {
                            env.set(&param.name, val.clone());
                        }
                    }
                } else {
                    // Smart distribution: fields may belong to struct-typed params
                    // For each param, check if it's a struct type and collect matching fields
                    let mut used_fields: std::collections::HashSet<String> =
                        std::collections::HashSet::new();

                    for param in &func.params {
                        // Check if this param matches a field directly
                        if let Some(val) = fields.get(&param.name) {
                            env.set(&param.name, val.clone());
                            used_fields.insert(param.name.clone());
                            continue;
                        }

                        // Check if this param is a struct type — look up its field names
                        if let ast::Type::Struct(type_name) = &param.ty {
                            if let Some(type_fields) = get_struct_fields(program, type_name) {
                                // Collect input fields that match this struct's fields
                                let mut struct_fields = HashMap::new();
                                for (tf_name, _) in &type_fields {
                                    if let Some(val) = fields.get(tf_name) {
                                        struct_fields.insert(tf_name.clone(), val.clone());
                                        used_fields.insert(tf_name.clone());
                                    }
                                }
                                if !struct_fields.is_empty() {
                                    let struct_val = Value::Struct(type_name.clone(), struct_fields);
                                    env.set(&param.name, struct_val);
                                }
                            }
                        }
                    }
                }
            }
        }
        (_, 1) => {
            env.set(&func.params[0].name, input.clone());
        }
        _ => {
            // Multiple params but non-struct input — bind first param
            env.set(&func.params[0].name, input.clone());
        }
    }
}

/// Check if a name is a union variant constructor.
fn is_union_variant(program: &ast::Program, name: &str) -> bool {
    for td in &program.types {
        if let ast::TypeDecl::TaggedUnion(u) = td {
            for variant in &u.variants {
                if variant.name == name {
                    return true;
                }
            }
        }
    }
    for module in &program.modules {
        for td in &module.types {
            if let ast::TypeDecl::TaggedUnion(u) = td {
                for variant in &u.variants {
                    if variant.name == name {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Check if nested patterns match the payload values, binding variables on success.
fn match_nested_patterns(
    program: &ast::Program,
    patterns: &[ast::MatchPattern],
    values: &[Value],
    env: &mut Env,
) -> Result<bool> {
    for (pattern, value) in patterns.iter().zip(values.iter()) {
        if !match_single_pattern(program, pattern, value, env)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Match a single pattern against a value, binding variables on success.
fn match_single_pattern(
    program: &ast::Program,
    pattern: &ast::MatchPattern,
    value: &Value,
    env: &mut Env,
) -> Result<bool> {
    match pattern {
        ast::MatchPattern::Binding(name) => {
            if name != "_" {
                env.set(name, value.clone());
            }
            Ok(true)
        }
        ast::MatchPattern::Literal(lit) => {
            let expected = match lit {
                ast::Literal::Int(n) => Value::Int(*n),
                ast::Literal::Float(f) => Value::Float(*f),
                ast::Literal::Bool(b) => Value::Bool(*b),
                ast::Literal::String(s) => Value::String(s.clone()),
            };
            Ok(value.matches(&expected))
        }
        ast::MatchPattern::Variant {
            variant,
            sub_patterns,
        } => {
            if let Value::Union {
                variant: v,
                payload,
            } = value
            {
                if v == variant {
                    match_nested_patterns(program, sub_patterns, payload, env)
                } else {
                    Ok(false)
                }
            } else {
                Ok(false)
            }
        }
    }
}

/// Look up struct field names from the program's type declarations.
fn get_struct_fields(program: &ast::Program, type_name: &str) -> Option<Vec<(String, ast::Type)>> {
    for td in &program.types {
        if let ast::TypeDecl::Struct(s) = td {
            if s.name == type_name {
                return Some(
                    s.fields
                        .iter()
                        .map(|f| (f.name.clone(), f.ty.clone()))
                        .collect(),
                );
            }
        }
    }
    // Also check modules
    for module in &program.modules {
        for td in &module.types {
            if let ast::TypeDecl::Struct(s) = td {
                if s.name == type_name {
                    return Some(
                        s.fields
                            .iter()
                            .map(|f| (f.name.clone(), f.ty.clone()))
                            .collect(),
                    );
                }
            }
        }
    }
    None
}

pub fn eval_test_case(
    program: &ast::Program,
    function_name: &str,
    case: &parser::TestCase,
) -> Result<String> {
    eval_test_case_with_mocks(program, function_name, case, &[], &[])
}

pub fn eval_test_case_with_mocks(
    program: &ast::Program,
    function_name: &str,
    case: &parser::TestCase,
    mocks: &[crate::session::IoMock],
    http_routes: &[ast::HttpRoute],
) -> Result<String> {
    let func = program
        .get_function(function_name)
        .ok_or_else(|| anyhow!("function `{function_name}` not found{}", crate::eval::suggest_similar(program, function_name)))?;

    let has_async = func
        .effects
        .iter()
        .any(|e| matches!(e, ast::Effect::Async | ast::Effect::Io));

    if has_async {
        // Async function: spin up a temporary coroutine runtime so +await
        // operations execute through real IO (with mock fallback if provided).
        // This avoids "requires async context" and "no mock for ..." errors.
        return eval_test_case_with_runtime(program, function_name, case, mocks, http_routes);
    }

    let input = eval_parser_expr_with_program(&case.input, program)?;
    let expected = eval_parser_expr_with_program(&case.expected, program)?;

    let mut env = Env::new();
    env.populate_shared_from_program(program);
    bind_input_to_params(program, func, &input, &mut env);

    // Execute function body (use named variant so FN_NAME_STACK has the qualified name
    // for shared variable resolution in Env::get())
    let result = eval_function_body_named(program, function_name, &func.body, &mut env);

    let msg = check_test_result(result, &func.return_type, &input, &expected, case.matcher.as_ref())?;

    // Check +after assertions (sync path — program is immutable here, so
    // after_checks on routes/modules work; tasks/mocks not applicable in sync)
    for after in &case.after_checks {
        check_after(program, after, None, http_routes)?;
    }

    Ok(msg)
}

/// Run a test case for an async function by spinning up a temporary coroutine
/// runtime. Mocks are checked first; unmatched IO operations execute for real.
fn eval_test_case_with_runtime(
    program: &ast::Program,
    function_name: &str,
    case: &parser::TestCase,
    mocks: &[crate::session::IoMock],
    http_routes: &[ast::HttpRoute],
) -> Result<String> {
    let func = program
        .get_function(function_name)
        .ok_or_else(|| anyhow!("function `{function_name}` not found{}", crate::eval::suggest_similar(program, function_name)))?;

    let input = eval_parser_expr_with_program(&case.input, program)?;
    let expected = eval_parser_expr_with_program(&case.expected, program)?;
    let return_type = func.return_type.clone();
    let program = program.clone();
    let fn_name = function_name.to_string();
    let mocks = mocks.to_vec();
    let matcher = case.matcher.clone();
    let after_checks = case.after_checks.clone();
    let routes = http_routes.to_vec();
    let captured_runtime = SHARED_RUNTIME.with(|rt| rt.borrow().clone());

    // Spin up a temporary tokio runtime + coroutine IO loop on a dedicated
    // thread.  This works whether or not the caller is already inside a tokio
    // runtime (nested block_on is not allowed, so we always use a fresh thread).
    std::thread::spawn(move || {
        // Propagate SharedRuntime to the new thread
        set_shared_runtime(captured_runtime);
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|e| anyhow!("failed to create async test runtime: {e}"))?;

        rt.block_on(async {
            let (runtime, mut io_rx) = crate::coroutine::Runtime::new();
            let runtime = std::sync::Arc::new(runtime);
            let io_sender = runtime.io_sender();

            // Spawn the IO handler loop
            let rt_handle = runtime.clone();
            let io_loop = tokio::spawn(async move {
                while let Some(request) = io_rx.recv().await {
                    let rt = rt_handle.clone();
                    tokio::spawn(async move {
                        rt.handle_io(request).await;
                    });
                }
            });

            // Run the evaluation in a blocking task so blocking_send/blocking_recv
            // don't stall the tokio executor.
            let eval_result = tokio::task::spawn_blocking(move || {
                let func = program
                    .get_function(&fn_name)
                    .ok_or_else(|| anyhow!("function `{fn_name}` not found"))?;

                let mut env = Env::new();
                env.populate_shared_from_program(&program);

                // Create handle: mocks checked first, unmatched ops use real IO.
                let handle = if mocks.is_empty() {
                    crate::coroutine::CoroutineHandle::new(io_sender)
                } else {
                    crate::coroutine::CoroutineHandle::new_mock_with_sender(mocks, io_sender)
                };
                env.set("__coroutine_handle", Value::CoroutineHandle(handle));

                bind_input_to_params(&program, func, &input, &mut env);
                let result = eval_function_body(&program, &func.body, &mut env);
                let msg = check_test_result(result, &return_type, &input, &expected, matcher.as_ref())?;

                // Check +after assertions
                for after in &after_checks {
                    check_after(&program, after, None, &routes)?;
                }

                Ok::<String, anyhow::Error>(msg)
            })
            .await
            .map_err(|e| anyhow!("async test task panicked: {e}"))??;

            // Shut down the IO loop
            io_loop.abort();

            Ok(eval_result)
        })
    })
    .join()
    .map_err(|_| anyhow!("async test thread panicked"))?
}

/// Run a test case through the real coroutine runtime (for async functions with
/// real IO). Falls back to mock-only execution for sync functions.
/// Must be called from within a tokio runtime.
pub async fn eval_test_case_async(
    program: &ast::Program,
    function_name: &str,
    case: &parser::TestCase,
    mocks: &[crate::session::IoMock],
    io_sender: tokio::sync::mpsc::Sender<crate::coroutine::IoRequest>,
    http_routes: &[ast::HttpRoute],
) -> Result<String> {
    let func = program
        .get_function(function_name)
        .ok_or_else(|| anyhow!("function `{function_name}` not found{}", crate::eval::suggest_similar(program, function_name)))?;

    let has_async = func
        .effects
        .iter()
        .any(|e| matches!(e, ast::Effect::Async | ast::Effect::Io));

    // If the function isn't async, delegate to the sync path
    if !has_async {
        return eval_test_case_with_mocks(program, function_name, case, mocks, http_routes);
    }

    let input = eval_parser_expr_with_program(&case.input, program)?;
    let expected = eval_parser_expr_with_program(&case.expected, program)?;

    // Clone what we need for the blocking task
    let program = program.clone();
    let fn_name = function_name.to_string();
    let return_type = func.return_type.clone();
    let mocks = mocks.to_vec();
    let matcher = case.matcher.clone();
    let after_checks = case.after_checks.clone();
    let routes = http_routes.to_vec();
    let captured_rt = SHARED_RUNTIME.with(|rt| rt.borrow().clone());

    let eval_result = tokio::task::spawn_blocking(move || {
        // Propagate SharedRuntime to the blocking thread
        set_shared_runtime(captured_rt);

        let func = program
            .get_function(&fn_name)
            .ok_or_else(|| anyhow!("function `{fn_name}` not found"))?;

        let mut env = Env::new();
        env.populate_shared_from_program(&program);

        // Create handle: mocks are checked first, unmatched ops fall through
        // to real IO via the sender.
        let handle = if mocks.is_empty() {
            crate::coroutine::CoroutineHandle::new(io_sender)
        } else {
            crate::coroutine::CoroutineHandle::new_mock_with_sender(mocks, io_sender)
        };
        env.set("__coroutine_handle", Value::CoroutineHandle(handle));

        bind_input_to_params(&program, func, &input, &mut env);

        let result = eval_function_body(&program, &func.body, &mut env);
        let msg = check_test_result(result, &return_type, &input, &expected, matcher.as_ref())?;

        // Check +after assertions
        for after in &after_checks {
            check_after(&program, after, None, &routes)?;
        }

        Ok::<String, anyhow::Error>(msg)
    })
    .await
    .map_err(|e| anyhow!("async test task panicked: {e}"))??;

    Ok(eval_result)
}

/// Compare the eval result against expected, handling Result<T> wrapping.
/// When a `TestMatcher` is present, it overrides exact comparison.
fn check_test_result(
    result: Result<Value>,
    return_type: &ast::Type,
    input: &Value,
    expected: &Value,
    matcher: Option<&parser::TestMatcher>,
) -> Result<String> {
    let returns_result = matches!(return_type, ast::Type::Result(_));

    let actual = match result {
        Ok(val) => {
            if returns_result {
                // Wrap in Ok if not already a result type
                match &val {
                    Value::Ok(_) | Value::Err(_) => val,
                    _ => Value::Ok(Box::new(val)),
                }
            } else {
                val
            }
        }
        Err(err) => {
            // Check failures produce Err with the label
            let err_str = err.to_string();
            Value::Err(err_str)
        }
    };

    // If a matcher is present, use it instead of exact comparison
    if let Some(m) = matcher {
        // For string matchers (contains/starts_with), extract the raw string
        // content when the value is a String or Ok(String), avoiding the
        // Display quotes that wrap String values.
        let raw_text = match &actual {
            Value::String(s) => s.clone(),
            Value::Ok(inner) => match inner.as_ref() {
                Value::String(s) => s.clone(),
                other => format!("{other}"),
            },
            other => format!("{other}"),
        };
        let matched = match m {
            parser::TestMatcher::Contains(s) => {
                raw_text.contains(s.as_str())
            }
            parser::TestMatcher::StartsWith(s) => {
                raw_text.starts_with(s.as_str())
            }
            parser::TestMatcher::AnyOk => {
                matches!(&actual, Value::Ok(_))
            }
            parser::TestMatcher::AnyErr => {
                matches!(&actual, Value::Err(_))
            }
            parser::TestMatcher::ErrContaining(s) => {
                matches!(&actual, Value::Err(msg) if msg.contains(s.as_str()))
            }
        };
        if matched {
            Ok(format!("input={input} => {actual} (matcher: {m:?})"))
        } else {
            bail!("input={input} => {actual}, matcher {m:?} did not match")
        }
    } else {
        // Exact comparison
        if actual.matches(expected) {
            Ok(format!("input={input} => {actual} (expected {expected})"))
        } else {
            bail!("input={input} => {actual}, expected {expected}")
        }
    }
}

/// Check a single `+after` assertion against the current program state.
///
/// `mocks` is provided when the test has access to session-level IO mocks.
fn check_after(
    program: &ast::Program,
    after: &parser::AfterCheck,
    mocks: Option<&[crate::session::IoMock]>,
    http_routes: &[ast::HttpRoute],
) -> Result<()> {
    match after.target.as_str() {
        "routes" => {
            match after.matcher.as_str() {
                "contains" => {
                    let found = http_routes.iter().any(|r| {
                        r.path.contains(&after.value) || r.handler_fn.contains(&after.value)
                    });
                    if !found {
                        bail!("+after routes contains \"{}\": no matching route found (routes: {:?})",
                            after.value,
                            http_routes.iter().map(|r| format!("{} {} -> {}", r.method, r.path, r.handler_fn)).collect::<Vec<_>>());
                    }
                }
                other => bail!("+after routes: unknown matcher `{other}` (expected `contains`)"),
            }
        }
        "modules" => {
            match after.matcher.as_str() {
                "contains" => {
                    let found = program.modules.iter().any(|m| m.name.contains(&after.value));
                    if !found {
                        bail!("+after modules contains \"{}\": no matching module found (modules: {:?})",
                            after.value,
                            program.modules.iter().map(|m| &m.name).collect::<Vec<_>>());
                    }
                }
                other => bail!("+after modules: unknown matcher `{other}` (expected `contains`)"),
            }
        }
        "mocks" => {
            match after.matcher.as_str() {
                "contains" => {
                    if let Some(mocks) = mocks {
                        let found = mocks.iter().any(|m| {
                            m.operation.contains(&after.value)
                                || m.patterns.iter().any(|p| p.contains(&after.value))
                        });
                        if !found {
                            bail!("+after mocks contains \"{}\": no matching mock found", after.value);
                        }
                    } else {
                        bail!("+after mocks: mock state not available in this test context");
                    }
                }
                other => bail!("+after mocks: unknown matcher `{other}` (expected `contains`)"),
            }
        }
        "tasks" => {
            // Tasks are runtime-level and not directly inspectable from the program.
            // For now, this is a no-op with a warning.
            eprintln!("[test] +after tasks check skipped (tasks require live runtime context)");
        }
        other => bail!("+after: unknown target `{other}` (expected routes, modules, mocks, or tasks)"),
    }
    Ok(())
}

/// Public entry point for running a function body with an env.
pub fn eval_function_body_pub(
    program: &ast::Program,
    body: &[ast::Statement],
    env: &mut Env,
) -> Result<Value> {
    eval_function_body(program, body, env)
}

/// Evaluate an AST expression in isolation (no function context).
/// Used for evaluating +shared variable default values like `0`, `""`, `true`.
pub fn eval_expr_standalone(program: &ast::Program, expr: &ast::Expr) -> Result<Value> {
    let mut env = Env::new();
    eval_ast_expr(program, expr, &mut env)
}

/// Public entry point that also sets the top-level function name for snapshot tracking.
/// Use this for spawned tasks so `?inspect task N` shows the correct function name.
pub fn eval_function_body_named(
    program: &ast::Program,
    function_name: &str,
    body: &[ast::Statement],
    env: &mut Env,
) -> Result<Value> {
    FN_NAME_STACK.with(|s| s.borrow_mut().push(function_name.to_string()));
    let result = eval_function_body(program, body, env);
    FN_NAME_STACK.with(|s| s.borrow_mut().pop());
    result
}

fn eval_function_body(
    program: &ast::Program,
    body: &[ast::Statement],
    env: &mut Env,
) -> Result<Value> {
    for stmt in body {
        // Update task snapshot if this is a tracked coroutine task.
        if let Some(Value::CoroutineHandle(handle)) = env.get_raw("__coroutine_handle") {
            if handle.task_id.is_some() {
                let fn_name = FN_NAME_STACK.with(|s| {
                    s.borrow().last().cloned().unwrap_or_else(|| "<top>".to_string())
                });
                let depth = FN_NAME_STACK.with(|s| s.borrow().len());
                handle.update_snapshot(&fn_name, Some(stmt.id.clone()), depth, env);
            }
        }
        match &stmt.kind {
            ast::StatementKind::Let { name, value, .. } => {
                let val = eval_ast_expr(program, value, env)?;
                env.set(name, val);
            }
            ast::StatementKind::Call { binding, call } => {
                let wants_result = binding
                    .as_ref()
                    .is_some_and(|b| matches!(&b.ty, ast::Type::Result(_)));

                if wants_result {
                    // Binding is Result<T> — catch errors instead of propagating
                    match eval_call(program, call, env) {
                        Ok(val) => {
                            if let Some(binding) = binding {
                                // Wrap success in Ok
                                let wrapped = match &val {
                                    Value::Ok(_) | Value::Err(_) => val,
                                    _ => Value::Ok(Box::new(val)),
                                };
                                env.set(&binding.name, wrapped);
                            }
                        }
                        Err(e) => {
                            if let Some(binding) = binding {
                                env.set(&binding.name, Value::Err(e.to_string()));
                            }
                        }
                    }
                } else {
                    // Binding is plain T — errors propagate (like Rust's ?)
                    let val = eval_call(program, call, env)?;
                    // Unwrap Ok() if the function returned a Result but the caller wants plain T
                    let val = match val {
                        Value::Ok(inner) => *inner,
                        other => other,
                    };
                    if let Some(binding) = binding {
                        env.set(&binding.name, val);
                    }
                }
            }
            ast::StatementKind::Check {
                condition, on_fail, ..
            } => {
                let val = eval_ast_expr(program, condition, env)?;
                if !val.is_truthy() {
                    return Err(anyhow!("{on_fail}"));
                }
            }
            ast::StatementKind::Return { value } => {
                return eval_ast_expr(program, value, env);
            }
            ast::StatementKind::Branch {
                condition,
                then_body,
                else_body,
            } => {
                let val = eval_ast_expr(program, condition, env)?;
                let branch = if val.is_truthy() {
                    then_body
                } else {
                    else_body
                };
                env.push_scope();
                let result = eval_function_body(program, branch, env);
                env.pop_scope();
                let result = result?;
                // Only propagate if the branch did an explicit return (non-None)
                if !matches!(result, Value::None) {
                    return Ok(result);
                }
            }
            ast::StatementKind::Each {
                iterator,
                binding,
                body: each_body,
            } => {
                let iter_val = eval_ast_expr(program, iterator, env)?;
                match iter_val {
                    Value::List(items) => {
                        for item in items {
                            env.push_scope();
                            env.set(&binding.name, item);
                            let result = eval_function_body(program, each_body, env);
                            env.pop_scope();
                            match result {
                                Ok(val) => {
                                    // Propagate explicit +return from inside +each
                                    if !matches!(val, Value::None) {
                                        return Ok(val);
                                    }
                                }
                                Err(e) => return Err(e),
                            }
                        }
                    }
                    _ => bail!("each: expected list, got {}", iter_val),
                }
            }
            ast::StatementKind::Set { name, value } => {
                let val = eval_ast_expr(program, value, env)?;
                env.set_existing(name, val);
            }
            ast::StatementKind::Match { expr, arms } => {
                let val = eval_ast_expr(program, expr, env)?;
                match &val {
                    Value::Union { variant, payload } => {
                        let mut matched = false;
                        for arm in arms {
                            if arm.variant == *variant {
                                env.push_scope();
                                // Exact variant match — bind payload values
                                for (i, binding) in arm.bindings.iter().enumerate() {
                                    if binding != "_" {
                                        if let Some(pval) = payload.get(i) {
                                            env.set(binding, pval.clone());
                                        }
                                    }
                                }
                                // Check nested pattern guards
                                if let Some(ref patterns) = arm.patterns {
                                    if !match_nested_patterns(program, patterns, payload, env)? {
                                        env.pop_scope();
                                        continue; // nested pattern didn't match, try next arm
                                    }
                                }
                                let result = eval_function_body(program, &arm.body, env);
                                env.pop_scope();
                                let result = result?;
                                if !matches!(result, Value::None) {
                                    return Ok(result);
                                }
                                matched = true;
                                break;
                            } else if arm.variant == "_" {
                                env.push_scope();
                                // Wildcard/default case — bind the whole value if there's a binding
                                if arm.bindings.len() == 1 && arm.bindings[0] != "_" {
                                    env.set(&arm.bindings[0], val.clone());
                                }
                                let result = eval_function_body(program, &arm.body, env);
                                env.pop_scope();
                                let result = result?;
                                if !matches!(result, Value::None) {
                                    return Ok(result);
                                }
                                matched = true;
                                break;
                            }
                        }
                        if !matched {
                            bail!("+match: no arm matched variant `{variant}`");
                        }
                    }
                    // Treat Ok/Err as union variants for pattern matching
                    Value::Ok(inner) => {
                        let as_union = Value::Union { variant: "Ok".to_string(), payload: vec![inner.as_ref().clone()] };
                        let mut matched = false;
                        for arm in arms {
                            if arm.variant == "Ok" {
                                env.push_scope();
                                if let Some(binding) = arm.bindings.first() {
                                    if binding != "_" { env.set(binding, inner.as_ref().clone()); }
                                }
                                let result = eval_function_body(program, &arm.body, env);
                                env.pop_scope();
                                let result = result?;
                                if !matches!(result, Value::None) { return Ok(result); }
                                matched = true;
                                break;
                            } else if arm.variant == "_" {
                                env.push_scope();
                                if arm.bindings.len() == 1 && arm.bindings[0] != "_" {
                                    env.set(&arm.bindings[0], as_union.clone());
                                }
                                let result = eval_function_body(program, &arm.body, env);
                                env.pop_scope();
                                let result = result?;
                                if !matches!(result, Value::None) { return Ok(result); }
                                matched = true;
                                break;
                            }
                        }
                        if !matched { bail!("+match: no arm matched Ok"); }
                    }
                    Value::Err(msg) => {
                        let as_union = Value::Union { variant: "Err".to_string(), payload: vec![Value::String(msg.clone())] };
                        let mut matched = false;
                        for arm in arms {
                            if arm.variant == "Err" {
                                env.push_scope();
                                if let Some(binding) = arm.bindings.first() {
                                    if binding != "_" { env.set(binding, Value::String(msg.clone())); }
                                }
                                let result = eval_function_body(program, &arm.body, env);
                                env.pop_scope();
                                let result = result?;
                                if !matches!(result, Value::None) { return Ok(result); }
                                matched = true;
                                break;
                            } else if arm.variant == "_" {
                                env.push_scope();
                                if arm.bindings.len() == 1 && arm.bindings[0] != "_" {
                                    env.set(&arm.bindings[0], as_union.clone());
                                }
                                let result = eval_function_body(program, &arm.body, env);
                                env.pop_scope();
                                let result = result?;
                                if !matches!(result, Value::None) { return Ok(result); }
                                matched = true;
                                break;
                            }
                        }
                        if !matched { bail!("+match: no arm matched Err"); }
                    }
                    // Also match on None (Option type)
                    Value::None => {
                        let mut matched = false;
                        for arm in arms {
                            if arm.variant == "None" || arm.variant == "_" {
                                env.push_scope();
                                let result = eval_function_body(program, &arm.body, env);
                                env.pop_scope();
                                let result = result?;
                                if !matches!(result, Value::None) { return Ok(result); }
                                matched = true;
                                break;
                            }
                        }
                        if !matched { bail!("+match: no arm matched None"); }
                    }
                    _ => bail!("+match expects a union, Ok/Err, or None value, got {val}"),
                }
            }
            ast::StatementKind::Await { name, call, .. } => {
                // First check if the callee is a user-defined Forge function
                // If so, call it normally (it will use +await internally for IO)
                if program.get_function(&call.callee).is_some() {
                    let val = eval_call(program, call, env)?;
                    let val = match val {
                        Value::Ok(inner) => *inner,
                        other => other,
                    };
                    env.set(name, val);
                } else {
                    // Builtin IO operation
                    let handle = match env.get_raw("__coroutine_handle") {
                        Some(Value::CoroutineHandle(h)) => h.clone(),
                        _ => bail!("+await requires async context — use 'forge run-async'"),
                    };
                    let args: Vec<Value> = call
                        .args
                        .iter()
                        .map(|a| eval_ast_expr(program, a, env))
                        .collect::<Result<Vec<_>>>()?;
                    let result = handle.execute_await(&call.callee, &args)?;
                    env.set(name, result);
                }
            }
            ast::StatementKind::Spawn { call, binding } => {
                let handle = match env.get_raw("__coroutine_handle") {
                    Some(Value::CoroutineHandle(h)) => h.clone(),
                    _ => bail!("+spawn requires async context"),
                };
                let args: Vec<Value> = call
                    .args
                    .iter()
                    .map(|a| eval_ast_expr(program, a, env))
                    .collect::<Result<Vec<_>>>()?;
                let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
                let io_tx = handle.io_sender();
                let _ = io_tx.blocking_send(crate::coroutine::IoRequest::Spawn {
                    function_name: call.callee.clone(),
                    args,
                    reply: reply_tx,
                });
                // If there's a binding, wait for the task ID
                if let Some(b) = binding {
                    let task_id = reply_rx
                        .blocking_recv()
                        .map_err(|e| anyhow::anyhow!("spawn reply failed: {e}"))??;
                    env.set(&b.name, Value::TaskHandle(task_id));
                }
            }
            ast::StatementKind::While { condition, body } => {
                let mut iterations = 0;
                loop {
                    let cond = eval_ast_expr(program, condition, env)?;
                    if !cond.is_truthy() {
                        break;
                    }
                    iterations += 1;
                    if iterations > 10000 {
                        bail!("while loop exceeded 10000 iterations — possible infinite loop");
                    }
                    env.push_scope();
                    let result = eval_function_body(program, body, env);
                    env.pop_scope();
                    match result {
                        Ok(val) => {
                            // If the body returned a value (via +return), propagate it
                            if !matches!(val, Value::None) {
                                return Ok(val);
                            }
                        }
                        Err(e) => return Err(e),
                    }
                }
            }
            ast::StatementKind::Yield { value } => {
                let _val = eval_ast_expr(program, value, env)?;
            }
        }
    }

    // If no explicit return, return None
    Ok(Value::None)
}

std::thread_local! {
    static CALL_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Stack of function names for the current call chain (used by task snapshots).
    static FN_NAME_STACK: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
    /// Thread-local SharedRuntime so newly-created Env instances automatically
    /// have access to +shared variables without explicit plumbing.
    static SHARED_RUNTIME: std::cell::RefCell<Option<crate::session::SharedRuntime>> = std::cell::RefCell::new(None);
}

/// Set the thread-local SharedRuntime for +shared variable access.
/// Call this before any eval functions that need shared variable support.
pub fn set_shared_runtime(rt: Option<crate::session::SharedRuntime>) {
    SHARED_RUNTIME.with(|s| *s.borrow_mut() = rt);
}

const MAX_CALL_DEPTH: usize = 256;

fn eval_call(program: &ast::Program, call: &ast::CallExpr, env: &mut Env) -> Result<Value> {
    let depth = CALL_DEPTH.with(|d| {
        let v = d.get();
        d.set(v + 1);
        v
    });
    if depth >= MAX_CALL_DEPTH {
        CALL_DEPTH.with(|d| d.set(0));
        bail!(
            "maximum call depth ({MAX_CALL_DEPTH}) exceeded — possible infinite recursion in `{}`",
            call.callee
        );
    }

    let result = eval_call_inner(program, call, env);
    CALL_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    result
}

/// Evaluate a builtin call with pre-evaluated args. Public for !eval on builtins.
pub fn eval_call_inner_with_args(
    program: &ast::Program,
    callee: &str,
    args: Vec<Value>,
    env: &mut Env,
) -> Result<Value> {
    eval_builtin_or_user(program, callee, args, env)
}

fn eval_call_inner(program: &ast::Program, call: &ast::CallExpr, env: &mut Env) -> Result<Value> {
    let args: Vec<Value> = call
        .args
        .iter()
        .map(|a| eval_ast_expr(program, a, env))
        .collect::<Result<Vec<_>>>()?;

    eval_builtin_or_user(program, &call.callee, args, env)
}

fn eval_builtin_or_user(
    program: &ast::Program,
    callee: &str,
    args: Vec<Value>,
    env: &mut Env,
) -> Result<Value> {
    // User-defined union variants take priority over builtins
    // (e.g., user defines Maybe = Some(Int) | None — "Some" should create Union, not Ok)
    if is_union_variant(program, callee) {
        return Ok(Value::Union {
            variant: callee.to_string(),
            payload: args,
        });
    }

    // Check for built-in functions
    match callee {
        "Ok" => {
            if args.len() == 1 {
                Ok(Value::Ok(Box::new(args.into_iter().next().unwrap())))
            } else {
                Ok(Value::Ok(Box::new(Value::None)))
            }
        }
        "Err" => {
            if args.len() == 1 {
                match &args[0] {
                    Value::String(s) => Ok(Value::Err(s.clone())),
                    other => Ok(Value::Err(format!("{other}"))),
                }
            } else {
                Ok(Value::Err("unknown".to_string()))
            }
        }
        "Some" => {
            if args.len() == 1 {
                Ok(Value::Ok(Box::new(args.into_iter().next().unwrap())))
            } else {
                Ok(Value::Ok(Box::new(Value::None)))
            }
        }
        "len" | "length" => {
            if args.len() != 1 {
                bail!("len() expects 1 argument");
            }
            match &args[0] {
                Value::String(s) => Ok(Value::Int(s.len() as i64)),
                Value::List(l) => Ok(Value::Int(l.len() as i64)),
                _ => bail!("len() expects string or list"),
            }
        }
        "concat" => {
            let mut result = String::new();
            for arg in &args {
                match arg {
                    Value::String(s) => result.push_str(s),
                    other => result.push_str(&format!("{other}")),
                }
            }
            Ok(Value::String(result))
        }
        "to_string" | "str" => {
            if args.len() != 1 {
                bail!("to_string() expects 1 argument");
            }
            Ok(Value::String(format!("{}", args[0])))
        }
        "char_at" => {
            if args.len() != 2 {
                bail!("char_at(s, i) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::String(s), Value::Int(i)) => {
                    let i = *i as usize;
                    if i < s.len() {
                        Ok(Value::String(s[i..i + 1].to_string()))
                    } else {
                        bail!("char_at: index {i} out of bounds (len {})", s.len())
                    }
                }
                _ => bail!("char_at expects (String, Int)"),
            }
        }
        "substring" | "substr" => {
            if args.len() != 3 {
                bail!("substring(s, start, end) expects 3 arguments");
            }
            match (&args[0], &args[1], &args[2]) {
                (Value::String(s), Value::Int(start), Value::Int(end)) => {
                    let start = *start as usize;
                    let end = (*end as usize).min(s.len());
                    if start <= end && start <= s.len() {
                        Ok(Value::String(s[start..end].to_string()))
                    } else {
                        bail!(
                            "substring: invalid range {}..{} (len {})",
                            start,
                            end,
                            s.len()
                        )
                    }
                }
                _ => bail!("substring expects (String, Int, Int)"),
            }
        }
        "starts_with" => {
            if args.len() != 2 {
                bail!("starts_with(s, prefix) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::String(s), Value::String(prefix)) => {
                    Ok(Value::Bool(s.starts_with(prefix.as_str())))
                }
                _ => bail!("starts_with expects (String, String)"),
            }
        }
        "ends_with" => {
            if args.len() != 2 {
                bail!("ends_with(s, suffix) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::String(s), Value::String(suffix)) => {
                    Ok(Value::Bool(s.ends_with(suffix.as_str())))
                }
                _ => bail!("ends_with expects (String, String)"),
            }
        }
        "contains" => {
            if args.len() != 2 {
                bail!("contains(s, substr) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::String(s), Value::String(sub)) => Ok(Value::Bool(s.contains(sub.as_str()))),
                _ => bail!("contains expects (String, String)"),
            }
        }
        "regex_match" => {
            if args.len() != 2 {
                bail!("regex_match(pattern, text) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::String(pattern), Value::String(text)) => match regex::Regex::new(pattern) {
                    Ok(re) => Ok(Value::Bool(re.is_match(text))),
                    Err(_) => Ok(Value::Bool(false)),
                },
                _ => bail!("regex_match expects (String, String)"),
            }
        }
        "regex_replace" => {
            if args.len() != 3 {
                bail!("regex_replace(pattern, replacement, text) expects 3 arguments");
            }
            match (&args[0], &args[1], &args[2]) {
                (Value::String(pattern), Value::String(replacement), Value::String(text)) => {
                    match regex::Regex::new(pattern) {
                        Ok(re) => Ok(Value::String(
                            re.replace_all(text, replacement.as_str()).into_owned(),
                        )),
                        Err(e) => bail!("regex_replace: invalid pattern '{}': {}", pattern, e),
                    }
                }
                _ => bail!("regex_replace expects (String, String, String)"),
            }
        }
        "index_of" => {
            if args.len() != 2 {
                bail!("index_of(s, substr) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::String(s), Value::String(sub)) => match s.find(sub.as_str()) {
                    Some(i) => Ok(Value::Int(i as i64)),
                    None => Ok(Value::Int(-1)),
                },
                _ => bail!("index_of expects (String, String)"),
            }
        }
        "split" => {
            if args.len() != 2 {
                bail!("split(s, delim) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::String(s), Value::String(delim)) => {
                    let parts: Vec<Value> = s
                        .split(delim.as_str())
                        .map(|p| Value::String(p.to_string()))
                        .collect();
                    Ok(Value::List(parts))
                }
                _ => bail!("split expects (String, String)"),
            }
        }
        "trim" => {
            if args.len() != 1 {
                bail!("trim(s) expects 1 argument");
            }
            match &args[0] {
                Value::String(s) => Ok(Value::String(s.trim().to_string())),
                _ => bail!("trim expects String"),
            }
        }
        "json_get" => {
            if args.len() != 2 {
                bail!("json_get(json, key_path) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::String(json_str), Value::String(path)) => {
                    let parsed: serde_json::Value = serde_json::from_str(json_str)
                        .map_err(|e| anyhow!("json_get: invalid JSON: {e}"))?;
                    let mut current = &parsed;
                    for key in path.split('.') {
                        if key.is_empty() {
                            continue;
                        }
                        if let Ok(idx) = key.parse::<usize>() {
                            current = current
                                .get(idx)
                                .ok_or_else(|| anyhow!("json_get: index {idx} not found"))?;
                        } else {
                            current = current
                                .get(key)
                                .ok_or_else(|| anyhow!("json_get: key '{key}' not found"))?;
                        }
                    }
                    // Return the value as a string, stripping quotes from string values
                    match current {
                        serde_json::Value::String(s) => Ok(Value::String(s.clone())),
                        serde_json::Value::Number(n) => {
                            if let Some(i) = n.as_i64() {
                                Ok(Value::Int(i))
                            } else if let Some(f) = n.as_f64() {
                                Ok(Value::Float(f))
                            } else {
                                Ok(Value::String(n.to_string()))
                            }
                        }
                        serde_json::Value::Bool(b) => Ok(Value::Bool(*b)),
                        serde_json::Value::Null => Ok(Value::String("null".to_string())),
                        // Arrays and objects are returned as JSON strings
                        other => Ok(Value::String(other.to_string())),
                    }
                }
                _ => bail!("json_get expects (String, String)"),
            }
        }
        "json_array_len" => {
            if args.len() != 1 {
                bail!("json_array_len(json) expects 1 argument");
            }
            match &args[0] {
                Value::String(json_str) => {
                    let parsed: serde_json::Value = serde_json::from_str(json_str)
                        .map_err(|e| anyhow!("json_array_len: invalid JSON: {e}"))?;
                    match &parsed {
                        serde_json::Value::Array(arr) => Ok(Value::Int(arr.len() as i64)),
                        _ => bail!("json_array_len: expected JSON array, got {}", parsed),
                    }
                }
                _ => bail!("json_array_len expects String"),
            }
        }
        "json_escape" => {
            if args.len() != 1 {
                bail!("json_escape(s) expects 1 argument");
            }
            match &args[0] {
                Value::String(s) => {
                    let mut escaped = String::with_capacity(s.len());
                    for ch in s.chars() {
                        match ch {
                            '\\' => escaped.push_str("\\\\"),
                            '"' => escaped.push_str("\\\""),
                            '\n' => escaped.push_str("\\n"),
                            '\r' => escaped.push_str("\\r"),
                            '\t' => escaped.push_str("\\t"),
                            c => escaped.push(c),
                        }
                    }
                    Ok(Value::String(escaped))
                }
                _ => bail!("json_escape expects String"),
            }
        }
        "base64_encode" => {
            if args.len() != 1 {
                bail!("base64_encode(s) expects 1 argument");
            }
            match &args[0] {
                Value::String(s) => {
                    use base64::Engine;
                    let encoded = base64::engine::general_purpose::STANDARD.encode(s.as_bytes());
                    Ok(Value::String(encoded))
                }
                _ => bail!("base64_encode expects String"),
            }
        }
        // List operations
        "list" => {
            // list() creates empty list, list(a, b, c) creates list with items
            Ok(Value::List(args))
        }
        "push" => {
            if args.len() != 2 {
                bail!("push(list, item) expects 2 arguments");
            }
            match &args[0] {
                Value::List(items) => {
                    let mut new_items = items.clone();
                    new_items.push(args[1].clone());
                    Ok(Value::List(new_items))
                }
                _ => bail!("push expects (List, value)"),
            }
        }
        "get" => {
            if args.len() != 2 {
                bail!("get(list, index) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::List(items), Value::Int(i)) => {
                    let i = *i as usize;
                    if i < items.len() {
                        Ok(items[i].clone())
                    } else {
                        bail!("get: index {i} out of bounds (len {})", items.len())
                    }
                }
                _ => bail!("get expects (List, Int)"),
            }
        }
        "join" => {
            if args.len() != 2 {
                bail!("join(list, delim) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::List(items), Value::String(delim)) => {
                    let parts: Vec<String> = items.iter().map(|v| format!("{v}")).collect();
                    Ok(Value::String(parts.join(delim)))
                }
                _ => bail!("join expects (List, String)"),
            }
        }
        // Shared state operations
        "state" => {
            // state(initial_value) → StateHandle
            if args.len() != 1 {
                bail!("state(initial_value) expects 1 argument");
            }
            Ok(Value::StateHandle(std::sync::Arc::new(
                std::sync::Mutex::new(args.into_iter().next().unwrap()),
            )))
        }
        "get_state" => {
            if args.len() != 1 {
                bail!("get_state(handle) expects 1 argument");
            }
            match &args[0] {
                Value::StateHandle(s) => {
                    let val = s.lock().unwrap().clone();
                    Ok(val)
                }
                _ => bail!("get_state expects a StateHandle"),
            }
        }
        "set_state" => {
            if args.len() != 2 {
                bail!("set_state(handle, value) expects 2 arguments");
            }
            match &args[0] {
                Value::StateHandle(s) => {
                    let mut guard = s.lock().unwrap();
                    *guard = args[1].clone();
                    Ok(Value::Int(0))
                }
                _ => bail!("set_state expects (StateHandle, value)"),
            }
        }
        "abs" => {
            if args.len() != 1 {
                bail!("abs() expects 1 argument");
            }
            match &args[0] {
                Value::Int(n) => Ok(Value::Int(n.abs())),
                Value::Float(n) => Ok(Value::Float(n.abs())),
                _ => bail!("abs() expects number"),
            }
        }
        "sqrt" => {
            if args.len() != 1 {
                bail!("sqrt() expects 1 argument");
            }
            match &args[0] {
                Value::Int(n) => Ok(Value::Float((*n as f64).sqrt())),
                Value::Float(n) => Ok(Value::Float(n.sqrt())),
                _ => bail!("sqrt() expects number"),
            }
        }
        "pow" => {
            if args.len() != 2 {
                bail!("pow() expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.pow(*b as u32))),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a.powf(*b))),
                (Value::Int(a), Value::Float(b)) => Ok(Value::Float((*a as f64).powf(*b))),
                (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a.powi(*b as i32))),
                _ => bail!("pow() expects numbers"),
            }
        }
        "floor" => {
            if args.len() != 1 {
                bail!("floor() expects 1 argument");
            }
            match &args[0] {
                Value::Float(n) => Ok(Value::Int(n.floor() as i64)),
                Value::Int(n) => Ok(Value::Int(*n)),
                _ => bail!("floor() expects number"),
            }
        }
        "to_int" | "parse_int" | "int" => {
            if args.len() != 1 {
                bail!("to_int() expects 1 argument");
            }
            match &args[0] {
                Value::Int(n) => Ok(Value::Int(*n)),
                Value::Float(n) => Ok(Value::Int(*n as i64)),
                Value::Bool(b) => Ok(Value::Int(*b as i64)),
                Value::String(s) => match s.parse::<i64>() {
                    Ok(n) => Ok(Value::Int(n)),
                    Err(_) => bail!("to_int: cannot parse '{s}' as integer"),
                },
                _ => bail!("to_int() expects a value convertible to Int"),
            }
        }
        "digit_value" => {
            // digit_value("5") -> 5, digit_value("a") -> -1
            if args.len() != 1 {
                bail!("digit_value() expects 1 argument");
            }
            match &args[0] {
                Value::String(s) => {
                    if s.len() == 1 {
                        let ch = s.chars().next().unwrap();
                        if ch.is_ascii_digit() {
                            Ok(Value::Int((ch as i64) - ('0' as i64)))
                        } else {
                            Ok(Value::Int(-1))
                        }
                    } else {
                        Ok(Value::Int(-1))
                    }
                }
                _ => bail!("digit_value expects String"),
            }
        }
        "is_digit_char" => {
            if args.len() != 1 {
                bail!("is_digit_char() expects 1 argument");
            }
            match &args[0] {
                Value::String(s) => Ok(Value::Bool(
                    s.len() == 1 && s.chars().next().unwrap().is_ascii_digit(),
                )),
                _ => Ok(Value::Bool(false)),
            }
        }
        // Bitwise operations
        "bit_and" => {
            if args.len() != 2 {
                bail!("bit_and(a, b) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a & b)),
                _ => bail!("bit_and expects (Int, Int)"),
            }
        }
        "bit_or" => {
            if args.len() != 2 {
                bail!("bit_or(a, b) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a | b)),
                _ => bail!("bit_or expects (Int, Int)"),
            }
        }
        "bit_xor" => {
            if args.len() != 2 {
                bail!("bit_xor(a, b) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a ^ b)),
                _ => bail!("bit_xor expects (Int, Int)"),
            }
        }
        "bit_not" => {
            if args.len() != 1 {
                bail!("bit_not(a) expects 1 argument");
            }
            match &args[0] {
                Value::Int(a) => Ok(Value::Int(!a)),
                _ => bail!("bit_not expects Int"),
            }
        }
        "bit_shl" | "shl" => {
            if args.len() != 2 {
                bail!("bit_shl(a, n) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::Int(a), Value::Int(n)) => Ok(Value::Int(a << n)),
                _ => bail!("bit_shl expects (Int, Int)"),
            }
        }
        "bit_shr" | "shr" => {
            if args.len() != 2 {
                bail!("bit_shr(a, n) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::Int(a), Value::Int(n)) => Ok(Value::Int(a >> n)),
                _ => bail!("bit_shr expects (Int, Int)"),
            }
        }
        "left_rotate" | "rotl" => {
            if args.len() != 2 {
                bail!("left_rotate(val, n) expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::Int(val), Value::Int(n)) => {
                    // 32-bit left rotation
                    let v = *val as u32;
                    let n = (*n as u32) % 32;
                    Ok(Value::Int(v.rotate_left(n) as i64))
                }
                _ => bail!("left_rotate expects (Int, Int)"),
            }
        }
        "to_hex" => {
            if args.len() != 1 {
                bail!("to_hex(n) expects 1 argument");
            }
            match &args[0] {
                Value::Int(n) => Ok(Value::String(format!("{:08x}", *n as u32))),
                _ => bail!("to_hex expects Int"),
            }
        }
        "char_code" | "ord" => {
            if args.len() != 1 {
                bail!("char_code(ch) expects 1 argument");
            }
            match &args[0] {
                Value::String(s) => {
                    if s.len() == 1 {
                        Ok(Value::Int(s.bytes().next().unwrap() as i64))
                    } else {
                        bail!("char_code expects single character string")
                    }
                }
                _ => bail!("char_code expects String"),
            }
        }
        "from_char_code" | "chr" => {
            if args.len() != 1 {
                bail!("from_char_code(n) expects 1 argument");
            }
            match &args[0] {
                Value::Int(n) => Ok(Value::String(String::from(*n as u8 as char))),
                _ => bail!("from_char_code expects Int"),
            }
        }
        "u32_wrap" => {
            // Wrap to unsigned 32-bit
            if args.len() != 1 {
                bail!("u32_wrap(n) expects 1 argument");
            }
            match &args[0] {
                Value::Int(n) => Ok(Value::Int((*n as u32) as i64)),
                _ => bail!("u32_wrap expects Int"),
            }
        }
        "max" => {
            if args.len() != 2 {
                bail!("max() expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(*a.max(b))),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a.max(*b))),
                _ => bail!("max() expects two numbers of same type"),
            }
        }
        "min" => {
            if args.len() != 2 {
                bail!("min() expects 2 arguments");
            }
            match (&args[0], &args[1]) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(*a.min(b))),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a.min(*b))),
                _ => bail!("min() expects two numbers of same type"),
            }
        }
        _ => {
            // Try to find the function in the program and call it
            if let Some(func) = program.get_function(callee) {
                let mut call_env = Env::new();
                for (param, arg) in func.params.iter().zip(args) {
                    call_env.set(&param.name, arg);
                }
                // Propagate coroutine handle to called functions
                if let Some(handle) = env.get_raw("__coroutine_handle") {
                    call_env.set("__coroutine_handle", handle.clone());
                }
                // Track function name for task snapshot frames
                FN_NAME_STACK.with(|s| s.borrow_mut().push(callee.to_string()));
                let result = eval_function_body(program, &func.body, &mut call_env);
                FN_NAME_STACK.with(|s| s.borrow_mut().pop());
                result
            } else {
                // Check if it's a union variant constructor
                if is_union_variant(program, callee) {
                    return Ok(Value::Union {
                        variant: callee.to_string(),
                        payload: args,
                    });
                }
                // Check if it's an IO builtin called without +await
                if crate::builtins::is_io_builtin(callee) {
                    bail!(
                        "`{callee}` is an async IO operation, use: +await result:String = {callee}({})",
                        args.iter().enumerate().map(|(i, _)| format!("arg{i}")).collect::<Vec<_>>().join(", ")
                    )
                }
                bail!(
                    "undefined function `{}` (called with {} args)",
                    callee,
                    args.len()
                )
            }
        }
    }
}

fn eval_ast_expr(program: &ast::Program, expr: &ast::Expr, env: &mut Env) -> Result<Value> {
    match expr {
        ast::Expr::Literal(lit) => match lit {
            ast::Literal::Int(v) => Ok(Value::Int(*v)),
            ast::Literal::Float(v) => Ok(Value::Float(*v)),
            ast::Literal::Bool(v) => Ok(Value::Bool(*v)),
            ast::Literal::String(v) => Ok(Value::String(v.clone())),
        },
        ast::Expr::Identifier(name) => {
            match name.as_str() {
                "true" => Ok(Value::Bool(true)),
                "false" => Ok(Value::Bool(false)),
                _ => {
                    // Try variable first
                    if let Ok(val) = env.get(name) {
                        Ok(val.clone())
                    } else if is_union_variant(program, name) {
                        // No-payload union variant
                        Ok(Value::Union {
                            variant: name.clone(),
                            payload: vec![],
                        })
                    } else {
                        bail!("undefined variable `{name}`")
                    }
                }
            }
        }
        ast::Expr::FieldAccess { base, field } => {
            let base_val = eval_ast_expr(program, base, env)?;
            match &base_val {
                Value::Struct(_, fields) => fields
                    .get(field)
                    .cloned()
                    .ok_or_else(|| anyhow!("field `{field}` not found on {base_val}")),
                Value::Ok(inner) => {
                    // Special methods on Result values
                    match field.as_str() {
                        "is_ok" => Ok(Value::Bool(true)),
                        "is_err" => Ok(Value::Bool(false)),
                        "unwrap" => Ok(inner.as_ref().clone()),
                        _ => {
                            // Transparent field access on Ok values
                            match inner.as_ref() {
                                Value::Struct(_, fields) => fields
                                    .get(field)
                                    .cloned()
                                    .ok_or_else(|| anyhow!("field `{field}` not found")),
                                _ => bail!("cannot access field `{field}` on {base_val}"),
                            }
                        }
                    }
                }
                Value::Err(e) => match field.as_str() {
                    "is_ok" => Ok(Value::Bool(false)),
                    "is_err" => Ok(Value::Bool(true)),
                    "error" | "unwrap_err" => Ok(Value::String(e.clone())),
                    "unwrap" => bail!("unwrap on Err({e})"),
                    _ => bail!("cannot access field `{field}` on Err({e})"),
                },
                // Special methods
                _ => match field.as_str() {
                    "len" => match &base_val {
                        Value::String(s) => Ok(Value::Int(s.len() as i64)),
                        Value::List(l) => Ok(Value::Int(l.len() as i64)),
                        _ => bail!("`.len` not supported on {base_val}"),
                    },
                    "is_ok" => match &base_val {
                        Value::Ok(_) => Ok(Value::Bool(true)),
                        Value::Err(_) => Ok(Value::Bool(false)),
                        _ => Ok(Value::Bool(true)),
                    },
                    "is_err" => match &base_val {
                        Value::Err(_) => Ok(Value::Bool(true)),
                        _ => Ok(Value::Bool(false)),
                    },
                    "is_empty" => match &base_val {
                        Value::String(s) => Ok(Value::Bool(s.is_empty())),
                        Value::List(l) => Ok(Value::Bool(l.is_empty())),
                        _ => bail!("`.is_empty` not supported on {base_val}"),
                    },
                    "unwrap" => match &base_val {
                        Value::Ok(v) => Ok(v.as_ref().clone()),
                        Value::Err(e) => bail!("unwrap on Err({e})"),
                        other => Ok(other.clone()),
                    },
                    _ => bail!("cannot access field `{field}` on {base_val}"),
                },
            }
        }
        ast::Expr::Call(call) => eval_call(program, call, env),
        ast::Expr::Binary { left, op, right } => {
            // Short-circuit AND and OR
            if matches!(op, ast::BinaryOp::And) {
                let lhs = eval_ast_expr(program, left, env)?;
                if !lhs.is_truthy() {
                    return Ok(Value::Bool(false));
                }
                let rhs = eval_ast_expr(program, right, env)?;
                return Ok(Value::Bool(rhs.is_truthy()));
            }
            if matches!(op, ast::BinaryOp::Or) {
                let lhs = eval_ast_expr(program, left, env)?;
                if lhs.is_truthy() {
                    return Ok(Value::Bool(true));
                }
                let rhs = eval_ast_expr(program, right, env)?;
                return Ok(Value::Bool(rhs.is_truthy()));
            }
            let lhs = eval_ast_expr(program, left, env)?;
            let rhs = eval_ast_expr(program, right, env)?;
            eval_binary_op(&lhs, op, &rhs)
        }
        ast::Expr::Unary { op, expr } => {
            let val = eval_ast_expr(program, expr, env)?;
            match op {
                ast::UnaryOp::Not => Ok(Value::Bool(!val.is_truthy())),
                ast::UnaryOp::Neg => match val {
                    Value::Int(n) => Ok(Value::Int(-n)),
                    Value::Float(n) => Ok(Value::Float(-n)),
                    _ => bail!("cannot negate {val}"),
                },
            }
        }
        ast::Expr::StructInit { ty, fields } => {
            let mut field_map = HashMap::new();
            for f in fields {
                let val = eval_ast_expr(program, &f.value, env)?;
                field_map.insert(f.name.clone(), val);
            }
            Ok(Value::Struct(ty.clone(), field_map))
        }
    }
}

fn eval_binary_op(lhs: &Value, op: &ast::BinaryOp, rhs: &Value) -> Result<Value> {
    match op {
        ast::BinaryOp::Add => match (lhs, rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 + b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a + *b as f64)),
            _ => bail!("cannot add {lhs} + {rhs}"),
        },
        ast::BinaryOp::Sub => match (lhs, rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 - b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a - *b as f64)),
            _ => bail!("cannot subtract {lhs} - {rhs}"),
        },
        ast::BinaryOp::Mul => match (lhs, rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 * b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Float(a * *b as f64)),
            _ => bail!("cannot multiply {lhs} * {rhs}"),
        },
        ast::BinaryOp::Div => match (lhs, rhs) {
            (Value::Int(a), Value::Int(b)) => {
                if *b == 0 {
                    bail!("division by zero")
                }
                Ok(Value::Int(a / b))
            }
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 / b)),
            (Value::Float(a), Value::Int(b)) => {
                if *b == 0 {
                    bail!("division by zero")
                }
                Ok(Value::Float(a / *b as f64))
            }
            _ => bail!("cannot divide {lhs} / {rhs}"),
        },
        ast::BinaryOp::Mod => match (lhs, rhs) {
            (Value::Int(a), Value::Int(b)) => {
                if *b == 0 {
                    bail!("modulo by zero")
                }
                Ok(Value::Int(a % b))
            }
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a % b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 % b)),
            (Value::Float(a), Value::Int(b)) => {
                if *b == 0 {
                    bail!("modulo by zero")
                }
                Ok(Value::Float(a % *b as f64))
            }
            _ => bail!("cannot modulo {lhs} % {rhs}"),
        },
        ast::BinaryOp::And => Ok(Value::Bool(lhs.is_truthy() && rhs.is_truthy())),
        ast::BinaryOp::Or => Ok(Value::Bool(lhs.is_truthy() || rhs.is_truthy())),
        ast::BinaryOp::Equal => Ok(Value::Bool(lhs.matches(rhs))),
        ast::BinaryOp::NotEqual => Ok(Value::Bool(!lhs.matches(rhs))),
        ast::BinaryOp::GreaterThan => match (lhs, rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a > b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a > b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Bool((*a as f64) > *b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Bool(*a > *b as f64)),
            _ => bail!("cannot compare {lhs} > {rhs}"),
        },
        ast::BinaryOp::LessThan => match (lhs, rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a < b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a < b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Bool((*a as f64) < *b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Bool(*a < *b as f64)),
            _ => bail!("cannot compare {lhs} < {rhs}"),
        },
        ast::BinaryOp::GreaterThanOrEqual => match (lhs, rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a >= b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a >= b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Bool((*a as f64) >= *b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Bool(*a >= *b as f64)),
            _ => bail!("cannot compare {lhs} >= {rhs}"),
        },
        ast::BinaryOp::LessThanOrEqual => match (lhs, rhs) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a <= b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a <= b)),
            (Value::Int(a), Value::Float(b)) => Ok(Value::Bool((*a as f64) <= *b)),
            (Value::Float(a), Value::Int(b)) => Ok(Value::Bool(*a <= *b as f64)),
            _ => bail!("cannot compare {lhs} <= {rhs}"),
        },
    }
}

/// Evaluate a parser::Expr directly (for test case inputs/expected values).
/// Evaluate a parser expression with access to the program (can call user functions).
pub fn eval_parser_expr_with_program(expr: &parser::Expr, program: &ast::Program) -> Result<Value> {
    match expr {
        parser::Expr::Ident(name) => {
            // Check if this is a user-defined union variant first
            if is_union_variant(program, name) {
                return Ok(Value::Union {
                    variant: name.clone(),
                    payload: vec![],
                });
            }
            // Check if this is a zero-arg user function (e.g. initial_context_state)
            if let Some(func) = program.get_function(name) {
                if func.params.is_empty() {
                    // Reject functions with side effects in test expressions
                    let has_side_effects = func.effects.iter().any(|e| {
                        matches!(e, ast::Effect::Io | ast::Effect::Async | ast::Effect::Mut | ast::Effect::Unsafe)
                    });
                    if has_side_effects {
                        bail!(
                            "cannot call `{name}` in test expression: function has side effects {:?} — \
                             use !mock and an async test wrapper instead",
                            func.effects
                        );
                    }
                    let mut env = Env::new();
                    return eval_function_body(program, &func.body, &mut env);
                }
            }
            // Fall through to standalone handling
            eval_parser_expr_standalone(expr)
        }
        parser::Expr::Call { callee, args } => {
            let name = parser_callee_name(callee);
            // Check if it's a user-defined union constructor
            if is_union_variant(program, &name) {
                let payload = args
                    .iter()
                    .map(|a| eval_parser_expr_with_program(a, program))
                    .collect::<Result<Vec<_>>>()?;
                return Ok(Value::Union {
                    variant: name,
                    payload,
                });
            }
            // Try as user function
            if let Some(func) = program.get_function(&name) {
                // Reject functions with side effects in test expressions
                let has_side_effects = func.effects.iter().any(|e| {
                    matches!(e, ast::Effect::Io | ast::Effect::Async | ast::Effect::Mut | ast::Effect::Unsafe)
                });
                if has_side_effects {
                    bail!(
                        "cannot call `{name}` in test expression: function has side effects {:?} — \
                         use !mock and an async test wrapper instead",
                        func.effects
                    );
                }
                let eval_args: Vec<Value> = args
                    .iter()
                    .map(|a| eval_parser_expr_with_program(a, program))
                    .collect::<Result<Vec<_>>>()?;
                let mut env = Env::new();
                for (param, arg) in func.params.iter().zip(eval_args) {
                    env.set(&param.name, arg);
                }
                return eval_function_body(program, &func.body, &mut env);
            }
            // Try as builtin function (concat, len, to_string, etc.)
            // Skip Ok/Err/Some/None — these are handled by eval_parser_expr_standalone
            // with special parser-level semantics (e.g. Err(bare_ident) as error labels).
            if crate::builtins::is_builtin(&name)
                && !matches!(name.as_str(), "Ok" | "Err" | "Some" | "None")
            {
                let eval_args: Vec<Value> = args
                    .iter()
                    .map(|a| eval_parser_expr_with_program(a, program))
                    .collect::<Result<Vec<_>>>()?;
                let mut env = Env::new();
                return eval_builtin_or_user(program, &name, eval_args, &mut env);
            }
            // Fall through to standalone (handles union constructors, Ok, Err)
            eval_parser_expr_standalone(expr)
        }
        // StructLiteral needs program access so field values can call user functions
        parser::Expr::StructLiteral(fields) => {
            let mut field_map = HashMap::new();
            for f in fields {
                let val = eval_parser_expr_with_program(&f.value, program)?;
                field_map.insert(f.name.clone(), val);
            }
            Ok(Value::Struct(String::new(), field_map))
        }
        // Unary expressions need program access for their inner expression
        parser::Expr::Unary { op, expr: inner } => {
            let val = eval_parser_expr_with_program(inner, program)?;
            match op {
                parser::UnaryOp::Not => Ok(Value::Bool(!val.is_truthy())),
                parser::UnaryOp::Neg => match val {
                    Value::Int(n) => Ok(Value::Int(-n)),
                    Value::Float(n) => Ok(Value::Float(-n)),
                    _ => bail!("cannot negate {val}"),
                },
            }
        }
        // Binary expressions need program access for both sides
        parser::Expr::Binary { left, op, right } => {
            let l = eval_parser_expr_with_program(left, program)?;
            let r = eval_parser_expr_with_program(right, program)?;
            let ast_op = match op {
                parser::BinaryOp::Add => ast::BinaryOp::Add,
                parser::BinaryOp::Sub => ast::BinaryOp::Sub,
                parser::BinaryOp::Mul => ast::BinaryOp::Mul,
                parser::BinaryOp::Div => ast::BinaryOp::Div,
                parser::BinaryOp::Mod => ast::BinaryOp::Mod,
                parser::BinaryOp::Eq => ast::BinaryOp::Equal,
                parser::BinaryOp::Neq => ast::BinaryOp::NotEqual,
                parser::BinaryOp::Gt => ast::BinaryOp::GreaterThan,
                parser::BinaryOp::Lt => ast::BinaryOp::LessThan,
                parser::BinaryOp::Gte => ast::BinaryOp::GreaterThanOrEqual,
                parser::BinaryOp::Lte => ast::BinaryOp::LessThanOrEqual,
                parser::BinaryOp::And => ast::BinaryOp::And,
                parser::BinaryOp::Or => ast::BinaryOp::Or,
            };
            eval_binary_op(&l, &ast_op, &r)
        }
        // Everything else delegates to standalone
        _ => eval_parser_expr_standalone(expr),
    }
}

pub fn eval_parser_expr_standalone(expr: &parser::Expr) -> Result<Value> {
    match expr {
        parser::Expr::Int(v) => Ok(Value::Int(*v)),
        parser::Expr::Float(v) => Ok(Value::Float(*v)),
        parser::Expr::Bool(v) => Ok(Value::Bool(*v)),
        parser::Expr::String(v) => Ok(Value::String(v.clone())),
        parser::Expr::Ident(name) => {
            match name.as_str() {
                "true" => Ok(Value::Bool(true)),
                "false" => Ok(Value::Bool(false)),
                _ => {
                    // Uppercase names: union variants (including user-defined None, Some, Ok)
                    // Lowercase names: error labels for Err matching
                    if name.chars().next().is_some_and(|c| c.is_uppercase()) {
                        // Special cases for Result/Option builtins (used in test expectations)
                        match name.as_str() {
                            "Ok" => Ok(Value::Ok(Box::new(Value::None))),
                            "None" => Ok(Value::Union {
                                variant: "None".to_string(),
                                payload: vec![],
                            }),
                            _ => Ok(Value::Union {
                                variant: name.clone(),
                                payload: vec![],
                            }),
                        }
                    } else {
                        Ok(Value::Err(name.clone()))
                    }
                }
            }
        }
        parser::Expr::StructLiteral(fields) => {
            let mut field_map = HashMap::new();
            for f in fields {
                let val = eval_parser_expr_standalone(&f.value)?;
                field_map.insert(f.name.clone(), val);
            }
            Ok(Value::Struct(String::new(), field_map))
        }
        parser::Expr::Call { callee, args } => {
            // Handle Ok(...) and Err(...) constructors
            let name = parser_callee_name(callee);
            match name.as_str() {
                "Ok" => {
                    if args.len() == 1 {
                        let inner = eval_parser_expr_standalone(&args[0])?;
                        Ok(Value::Ok(Box::new(inner)))
                    } else {
                        Ok(Value::Ok(Box::new(Value::None)))
                    }
                }
                "Err" => {
                    if args.len() == 1 {
                        match &args[0] {
                            parser::Expr::Ident(label) => Ok(Value::Err(label.clone())),
                            other => {
                                let val = eval_parser_expr_standalone(other)?;
                                Ok(Value::Err(format!("{val}")))
                            }
                        }
                    } else {
                        Ok(Value::Err("unknown".to_string()))
                    }
                }
                "list" => {
                    let items = args
                        .iter()
                        .map(eval_parser_expr_standalone)
                        .collect::<Result<Vec<_>>>()?;
                    Ok(Value::List(items))
                }
                _ => {
                    // Treat as union variant constructor
                    let payload = args
                        .iter()
                        .map(eval_parser_expr_standalone)
                        .collect::<Result<Vec<_>>>()?;
                    Ok(Value::Union {
                        variant: name,
                        payload,
                    })
                }
            }
        }
        parser::Expr::FieldAccess { base, field } => {
            // For test expectations like result.name — just create a path identifier
            let base_name = parser_callee_name(base);
            Ok(Value::Err(format!("{base_name}.{field}")))
        }
        parser::Expr::Unary { op, expr } => {
            let val = eval_parser_expr_standalone(expr)?;
            match op {
                parser::UnaryOp::Not => Ok(Value::Bool(!val.is_truthy())),
                parser::UnaryOp::Neg => match val {
                    Value::Int(n) => Ok(Value::Int(-n)),
                    Value::Float(n) => Ok(Value::Float(-n)),
                    _ => bail!("cannot negate {val}"),
                },
            }
        }
        _ => bail!("cannot evaluate expression {:?} in test case", expr),
    }
}

fn parser_callee_name(expr: &parser::Expr) -> String {
    match expr {
        parser::Expr::Ident(name) => name.clone(),
        parser::Expr::FieldAccess { base, field } => {
            format!("{}.{}", parser_callee_name(base), field)
        }
        _ => format!("{:?}", expr),
    }
}

/// A single step in an execution trace.
#[derive(Debug, Clone)]
pub struct TraceStep {
    pub stmt_id: String,
    pub description: String,
    pub result: String,
    pub status: TraceStatus,
}

#[derive(Debug, Clone)]
pub enum TraceStatus {
    Pass,
    Fail,
    Return,
}

impl std::fmt::Display for TraceStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let marker = match self.status {
            TraceStatus::Pass => "pass",
            TraceStatus::Fail => "FAIL",
            TraceStatus::Return => "return",
        };
        write!(
            f,
            "{}: {} | {} | {}",
            self.stmt_id, self.description, self.result, marker
        )
    }
}

/// Execute a function with tracing, returning each step.
pub fn trace_function(
    program: &ast::Program,
    function_name: &str,
    input: &parser::Expr,
) -> Result<Vec<TraceStep>> {
    let func = program
        .get_function(function_name)
        .ok_or_else(|| anyhow!("function `{function_name}` not found{}", crate::eval::suggest_similar(program, function_name)))?;

    let input_val = eval_parser_expr_standalone(input)?;
    let mut env = Env::new();
    bind_input_to_params(program, func, &input_val, &mut env);

    let mut steps = vec![];
    trace_body(program, &func.body, &mut env, &mut steps);
    Ok(steps)
}

fn trace_body(
    program: &ast::Program,
    body: &[ast::Statement],
    env: &mut Env,
    steps: &mut Vec<TraceStep>,
) -> Option<Value> {
    for stmt in body {
        match &stmt.kind {
            ast::StatementKind::Let { name, value, ty } => {
                match eval_ast_expr(program, value, env) {
                    Ok(val) => {
                        steps.push(TraceStep {
                            stmt_id: stmt.id.clone(),
                            description: format!("let {name}:{ty:?}"),
                            result: format!("{val}"),
                            status: TraceStatus::Pass,
                        });
                        env.set(name, val);
                    }
                    Err(e) => {
                        steps.push(TraceStep {
                            stmt_id: stmt.id.clone(),
                            description: format!("let {name}:{ty:?}"),
                            result: format!("ERROR: {e}"),
                            status: TraceStatus::Fail,
                        });
                        return Some(Value::Err(e.to_string()));
                    }
                }
            }
            ast::StatementKind::Call { binding, call } => match eval_call(program, call, env) {
                Ok(val) => {
                    let desc = if let Some(b) = binding {
                        format!("call {}:{:?} = {}()", b.name, b.ty, call.callee)
                    } else {
                        format!("call {}()", call.callee)
                    };
                    steps.push(TraceStep {
                        stmt_id: stmt.id.clone(),
                        description: desc,
                        result: format!("{val}"),
                        status: TraceStatus::Pass,
                    });
                    if let Some(b) = binding {
                        env.set(&b.name, val);
                    }
                }
                Err(e) => {
                    steps.push(TraceStep {
                        stmt_id: stmt.id.clone(),
                        description: format!("call {}()", call.callee),
                        result: format!("ERROR: {e}"),
                        status: TraceStatus::Fail,
                    });
                    return Some(Value::Err(e.to_string()));
                }
            },
            ast::StatementKind::Check {
                label,
                condition,
                on_fail,
            } => match eval_ast_expr(program, condition, env) {
                Ok(val) => {
                    let is_true = val.is_truthy();
                    steps.push(TraceStep {
                        stmt_id: stmt.id.clone(),
                        description: format!("check {label}"),
                        result: format!("{val} = {is_true}"),
                        status: if is_true {
                            TraceStatus::Pass
                        } else {
                            TraceStatus::Fail
                        },
                    });
                    if !is_true {
                        return Some(Value::Err(on_fail.clone()));
                    }
                }
                Err(e) => {
                    steps.push(TraceStep {
                        stmt_id: stmt.id.clone(),
                        description: format!("check {label}"),
                        result: format!("ERROR: {e}"),
                        status: TraceStatus::Fail,
                    });
                    return Some(Value::Err(e.to_string()));
                }
            },
            ast::StatementKind::Return { value } => match eval_ast_expr(program, value, env) {
                Ok(val) => {
                    steps.push(TraceStep {
                        stmt_id: stmt.id.clone(),
                        description: "return".to_string(),
                        result: format!("{val}"),
                        status: TraceStatus::Return,
                    });
                    return Some(val);
                }
                Err(e) => {
                    steps.push(TraceStep {
                        stmt_id: stmt.id.clone(),
                        description: "return".to_string(),
                        result: format!("ERROR: {e}"),
                        status: TraceStatus::Fail,
                    });
                    return Some(Value::Err(e.to_string()));
                }
            },
            _ => {
                // Branch, Each, Yield — simplified tracing
                steps.push(TraceStep {
                    stmt_id: stmt.id.clone(),
                    description: format!("{:?}", std::mem::discriminant(&stmt.kind)),
                    result: "...".to_string(),
                    status: TraceStatus::Pass,
                });
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parser, session::IoMock, validator};

    /// Helper: parse Forge source and build a program from it.
    fn build_program(source: &str) -> ast::Program {
        let ops = parser::parse(source).expect("parse failed");
        let mut program = ast::Program::default();
        for op in &ops {
            match op {
                parser::Operation::Test(_) | parser::Operation::Eval(_) => {}
                _ => {
                    validator::apply_and_validate(&mut program, op)
                        .expect("validation failed");
                }
            }
        }
        program.rebuild_function_index();
        program
    }

    /// Helper: extract test cases from parsed source.
    fn extract_test_cases(source: &str) -> Vec<(String, parser::TestCase)> {
        let ops = parser::parse(source).expect("parse failed");
        let mut cases = Vec::new();
        for op in ops {
            if let parser::Operation::Test(test) = op {
                for case in test.cases {
                    cases.push((test.function_name.clone(), case));
                }
            }
        }
        cases
    }

    /// Helper: extract mocks from parsed source.
    fn extract_mocks(source: &str) -> Vec<IoMock> {
        let ops = parser::parse(source).expect("parse failed");
        let mut mocks = Vec::new();
        for op in ops {
            if let parser::Operation::Mock { operation, patterns, response } = op {
                mocks.push(IoMock { operation, patterns, response });
            }
        }
        mocks
    }

    // ── Sync function tests ───────────────────────────────────────────

    #[test]
    fn test_sync_function_passes() {
        let source = "\
+fn add (a:Int, b:Int)->Int
  +let sum:Int = a + b
  +return sum

!test add
  +with a=2 b=3 -> expect 5
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);

        let (fn_name, case) = &cases[0];
        let result = eval_test_case(&program, fn_name, case);
        assert!(result.is_ok(), "sync test should pass: {:?}", result);
        assert!(result.unwrap().contains("expected 5"));
    }

    #[test]
    fn test_sync_function_fails_on_wrong_expected() {
        let source = "\
+fn add (a:Int, b:Int)->Int
  +let sum:Int = a + b
  +return sum

!test add
  +with a=2 b=3 -> expect 99
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case(&program, fn_name, case);
        assert!(result.is_err(), "should fail when expected is wrong");
    }

    // ── Async function tests (mock-only path) ────────────────────────

    #[test]
    fn test_async_function_with_mock_http_get() {
        let source = "\
+fn fetch_data (url:String)->String [async]
  +await resp:String = http_get(url)
  +return resp
";
        let program = build_program(source);

        let test_source = "\
!test fetch_data
  +with url=\"https://example.com\" -> expect \"hello\"
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];

        let mocks = vec![IoMock {
            operation: "http_get".to_string(),
            patterns: vec!["example.com".to_string()],
            response: "hello".to_string(),
        }];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
        assert!(result.is_ok(), "async test with mock should pass: {:?}", result);
    }

    #[test]
    fn test_async_function_without_mock_runs_real_io() {
        // An async function tested without mocks should spin up a temporary
        // coroutine runtime and execute IO operations for real.
        let source = "\
+fn delayed_value ()->String [async]
  +await _:String = sleep(1)
  +return \"done\"
";
        let program = build_program(source);

        let test_source = "\
!test delayed_value
  +with -> expect \"done\"
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];

        // No mocks — should execute sleep(1ms) for real and return "done"
        let result = eval_test_case(&program, fn_name, case);
        assert!(result.is_ok(), "async test without mocks should run real IO: {:?}", result);
    }

    #[test]
    fn test_async_function_with_io_effect_gets_handle() {
        let source = "\
+fn fetch_data (url:String)->String [io]
  +await resp:String = http_get(url)
  +return resp
";
        let program = build_program(source);

        let test_source = "\
!test fetch_data
  +with url=\"https://example.com\" -> expect \"world\"
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];

        let mocks = vec![IoMock {
            operation: "http_get".to_string(),
            patterns: vec!["example.com".to_string()],
            response: "world".to_string(),
        }];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
        assert!(result.is_ok(), "io test with mock should pass: {:?}", result);
    }

    #[test]
    fn test_async_function_await_sleep_with_mock() {
        // +await on `sleep` is a builtin IO op — should be intercepted by mock
        let source = "\
+fn delayed_value (ms:Int)->String [async]
  +await _:String = sleep(ms)
  +return \"done\"
";
        let program = build_program(source);

        let test_source = "\
!test delayed_value
  +with ms=1000 -> expect \"done\"
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];

        // Mock sleep so it returns immediately without real delay
        let mocks = vec![IoMock {
            operation: "sleep".to_string(),
            patterns: vec!["1000".to_string()],
            response: "".to_string(),
        }];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
        assert!(result.is_ok(), "sleep mock test should pass: {:?}", result);
    }

    #[test]
    fn test_async_function_nested_await_propagates_handle() {
        // An async function that calls another user-defined async function
        // which itself does +await on a builtin — handle must propagate
        let source = "\
+fn inner_fetch (url:String)->String [async]
  +await resp:String = http_get(url)
  +return resp

+fn outer_fetch (url:String)->String [async]
  +await data:String = inner_fetch(url)
  +return data
";
        let program = build_program(source);

        let test_source = "\
!test outer_fetch
  +with url=\"https://api.test.com\" -> expect \"nested_ok\"
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];

        let mocks = vec![IoMock {
            operation: "http_get".to_string(),
            patterns: vec!["api.test.com".to_string()],
            response: "nested_ok".to_string(),
        }];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
        assert!(result.is_ok(), "nested async call should pass: {:?}", result);
    }

    // ── Mock JSON response tests ─────────────────────────────────────

    #[test]
    fn test_mock_http_get_json_consumed_by_json_get() {
        // Mock returns JSON, function uses json_get to extract a field
        let source = "\
+fn get_name (url:String)->String [async]
  +await body:String = http_get(url)
  +let name:String = json_get(body, \"name\")
  +return name
";
        let program = build_program(source);

        let test_source = "\
!test get_name
  +with url=\"https://api.example.com/user\" -> expect \"alice\"
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];

        // The mock response is valid JSON — note: IoMock.response is the
        // *decoded* string (no backslash escapes).
        let mocks = vec![IoMock {
            operation: "http_get".to_string(),
            patterns: vec!["api.example.com".to_string()],
            response: r#"{"name":"alice","age":30}"#.to_string(),
        }];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
        assert!(result.is_ok(), "json_get on mock JSON should pass: {:?}", result);
    }

    #[test]
    fn test_mock_http_get_json_consumed_by_json_array_len() {
        // Mock returns a JSON array, function uses json_array_len
        let source = "\
+fn count_items (url:String)->Int [async]
  +await body:String = http_get(url)
  +let count:Int = json_array_len(body)
  +return count
";
        let program = build_program(source);

        let test_source = "\
!test count_items
  +with url=\"https://api.example.com/items\" -> expect 3
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];

        let mocks = vec![IoMock {
            operation: "http_get".to_string(),
            patterns: vec!["api.example.com".to_string()],
            response: r#"[1,2,3]"#.to_string(),
        }];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
        assert!(result.is_ok(), "json_array_len on mock JSON should pass: {:?}", result);
    }

    #[test]
    fn test_mock_escape_decoding_via_parser() {
        // Verify that !mock strings are properly unescaped by the parser.
        // The Forge source text: !mock http_get "api.test" -> "{\"ok\":true,\"items\":[1,2]}"
        let source = "!mock http_get \"api.test\" -> \"{\\\"ok\\\":true,\\\"items\\\":[1,2]}\"";
        let mocks = extract_mocks(source);
        assert_eq!(mocks.len(), 1);
        assert_eq!(mocks[0].patterns, vec!["api.test"]);
        // After unescape, response should be valid JSON without backslashes
        assert_eq!(mocks[0].response, r#"{"ok":true,"items":[1,2]}"#);
        // Verify it's valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&mocks[0].response)
            .expect("mock response should be valid JSON");
        assert_eq!(parsed["ok"], true);
    }

    #[test]
    fn test_mock_json_end_to_end_with_parser_escaping() {
        // End-to-end: !mock with escaped JSON → function uses json_get
        let fn_source = "\
+fn check_status (url:String)->String [async]
  +await body:String = http_get(url)
  +let status:String = json_get(body, \"status\")
  +return status
";
        let program = build_program(fn_source);

        // This is how it would appear in a .ax file — escaped quotes
        // Forge source: !mock http_get "api.svc" -> "{\"status\":\"healthy\",\"uptime\":99}"
        let mock_source = "!mock http_get \"api.svc\" -> \"{\\\"status\\\":\\\"healthy\\\",\\\"uptime\\\":99}\"";
        let mocks = extract_mocks(mock_source);

        let test_source = "\
!test check_status
  +with url=\"https://api.svc.local/health\" -> expect \"healthy\"
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];

        let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
        assert!(result.is_ok(), "end-to-end mock JSON test should pass: {:?}", result);
    }

    // ── Orchestrator-style integration tests ────────────────────────
    // These simulate the exact flow: parse source with +fn, !mock, !test;
    // collect mocks; build program; run eval_test_case_with_mocks.

    /// Helper: simulate orchestrator flow — parse full source, collect mocks,
    /// build program, run each test case with mocks.
    fn run_orchestrator_style(source: &str) -> Vec<Result<String>> {
        let ops = parser::parse(source).expect("parse failed");
        let mut program = ast::Program::default();
        let mut test_ops = Vec::new();
        let mut io_mocks: Vec<IoMock> = Vec::new();

        for op in &ops {
            match op {
                parser::Operation::Test(test) => test_ops.push(test.clone()),
                parser::Operation::Mock { operation, patterns, response } => {
                    io_mocks.push(IoMock {
                        operation: operation.clone(),
                        patterns: patterns.clone(),
                        response: response.clone(),
                    });
                }
                parser::Operation::Unmock => { io_mocks.clear(); }
                parser::Operation::Eval(_) => {}
                _ => { let _ = validator::apply_and_validate(&mut program, op); }
            }
        }
        program.rebuild_function_index();

        let mut results = Vec::new();
        for test in &test_ops {
            for case in &test.cases {
                results.push(eval_test_case_with_mocks(
                    &program, &test.function_name, case, &io_mocks, &[],
                ));
            }
        }
        results
    }

    #[test]
    fn test_orchestrator_async_http_get_with_mock() {
        // Simulates LLM output containing +fn [async], !mock, !test
        let source = "\
+fn fetch_data (url:String)->String [async]
  +await resp:String = http_get(url)
  +return resp

!mock http_get \"example.com\" -> \"hello world\"

!test fetch_data
  +with url=\"https://example.com/api\" -> expect \"hello world\"
";
        let results = run_orchestrator_style(source);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok(), "orchestrator async http_get test should pass: {:?}", results[0]);
    }

    #[test]
    fn test_orchestrator_async_sleep_with_mock() {
        let source = "\
+fn delayed (ms:Int)->String [async]
  +await _:String = sleep(ms)
  +return \"done\"

!mock sleep \"500\" -> \"\"

!test delayed
  +with ms=500 -> expect \"done\"
";
        let results = run_orchestrator_style(source);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok(), "orchestrator async sleep test should pass: {:?}", results[0]);
    }

    #[test]
    fn test_orchestrator_nested_async_with_mock() {
        // wrapper -> inner_fetch -> http_get (all async, handle must propagate)
        let source = "\
+fn inner_fetch (url:String)->String [async]
  +await resp:String = http_get(url)
  +return resp

+fn wrapper (url:String)->String [async]
  +await data:String = inner_fetch(url)
  +return data

!mock http_get \"api.test\" -> \"nested result\"

!test wrapper
  +with url=\"https://api.test/v1\" -> expect \"nested result\"
";
        let results = run_orchestrator_style(source);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok(), "orchestrator nested async test should pass: {:?}", results[0]);
    }

    #[test]
    fn test_orchestrator_mock_json_escape_json_get_json_array_len() {
        // Proves: !mock with escaped JSON → parser decodes → json_get + json_array_len work
        // Forge source: !mock http_get "x" -> "{\"ok\":true,\"result\":[]}"
        let fn_source = "\
+fn check (url:String)->Int [async]
  +await body:String = http_get(url)
  +let arr:String = json_get(body, \"result\")
  +let count:Int = json_array_len(arr)
  +return count
";
        let mock_source = "!mock http_get \"x\" -> \"{\\\"ok\\\":true,\\\"result\\\":[]}\"";
        let test_source = "\
!test check
  +with url=\"x\" -> expect 0
";
        let full_source = format!("{fn_source}\n{mock_source}\n\n{test_source}");
        let results = run_orchestrator_style(&full_source);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok(), "mock JSON escape + json_get + json_array_len should pass: {:?}", results[0]);
    }

    // ── Async eval_test_case_async tests ─────────────────────────────

    #[tokio::test]
    async fn test_async_eval_test_case_with_mocks() {
        let source = "\
+fn fetch_data (url:String)->String [async]
  +await resp:String = http_get(url)
  +return resp
";
        let program = build_program(source);

        let test_source = "\
!test fetch_data
  +with url=\"https://example.com\" -> expect \"async_hello\"
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];

        let mocks = vec![IoMock {
            operation: "http_get".to_string(),
            patterns: vec!["example.com".to_string()],
            response: "async_hello".to_string(),
        }];

        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let result = eval_test_case_async(&program, fn_name, case, &mocks, tx, &[]).await;
        assert!(result.is_ok(), "async test case should pass: {:?}", result);
    }

    #[tokio::test]
    async fn test_session_apply_async_runs_async_tests_with_mocks() {
        // Simulate the full session flow: define an async function,
        // register mocks, then run !test — all through apply_async.
        let mut session = crate::session::Session::new();

        // Define async function
        let define_source = "\
+fn fetch_data (url:String)->String [async]
  +await resp:String = http_get(url)
  +return resp
";
        let results = session.apply_async(define_source, None).await;
        assert!(results.is_ok(), "define should succeed: {:?}", results);

        // Register mock
        let mock_source = "!mock http_get \"example.com\" -> \"mocked_response\"";
        let results = session.apply_async(mock_source, None).await;
        assert!(results.is_ok(), "mock should succeed: {:?}", results);

        // Run test — async function with mock, no io_sender needed (mock-only)
        let test_source = "\
!test fetch_data
  +with url=\"https://example.com/api\" -> expect \"mocked_response\"
";
        let results = session.apply_async(test_source, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].1, "async test with mock should pass: {:?}", results[0]);
        assert!(results[0].0.contains("PASS"), "result should be PASS: {:?}", results[0]);
    }

    #[tokio::test]
    async fn test_session_apply_async_nested_async_with_mocks() {
        // Nested async: wrapper -> inner_fetch -> http_get
        let mut session = crate::session::Session::new();

        let source = "\
+fn inner_fetch (url:String)->String [async]
  +await resp:String = http_get(url)
  +return resp

+fn wrapper (url:String)->String [async]
  +call data:String = inner_fetch(url)
  +return data
";
        let _ = session.apply_async(source, None).await;

        let mock_source = "!mock http_get \"api.test\" -> \"nested_result\"";
        let _ = session.apply_async(mock_source, None).await;

        let test_source = "\
!test wrapper
  +with url=\"https://api.test/v1\" -> expect \"nested_result\"
";
        let results = session.apply_async(test_source, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].1, "nested async test should pass: {:?}", results[0]);
    }

    #[tokio::test]
    async fn test_async_eval_delegates_sync_to_sync_path() {
        let source = "\
+fn add (a:Int, b:Int)->Int
  +let sum:Int = a + b
  +return sum
";
        let program = build_program(source);

        let test_source = "\
!test add
  +with a=10 b=20 -> expect 30
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];

        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let result = eval_test_case_async(&program, fn_name, case, &[], tx, &[]).await;
        assert!(result.is_ok(), "sync function via async path should pass: {:?}", result);
    }

    // ── UTF-8 regression tests ───────────────────────────────────────

    #[test]
    fn test_json_get_preserves_utf8() {
        // Verify json_get returns multi-byte UTF-8 characters intact
        // Use a wrapper that builds the JSON internally to avoid parser quoting issues
        let source = r#"
+fn get_cafe ()->String
  +let json:String = "{\"name\":\"café\"}"
  +return json_get(json, "name")
"#;
        let program = build_program(source);
        let input = parser::parse("!eval get_cafe")
            .unwrap()
            .into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .unwrap();
        let result = eval_call_with_input(&program, &input.function_name, &input.input).unwrap();
        assert_eq!(result, r#""café""#, "json_get should preserve UTF-8 chars");
    }

    #[test]
    fn test_json_escape_preserves_utf8() {
        // Verify json_escape passes multi-byte UTF-8 through unchanged
        let source = r#"
+fn escape_it (s:String)->String
  +return json_escape(s)
"#;
        let program = build_program(source);
        let input = parser::parse(r#"!eval escape_it s="café élève naïve""#)
            .unwrap()
            .into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .unwrap();
        let result = eval_call_with_input(&program, &input.function_name, &input.input).unwrap();
        assert_eq!(result, r#""café élève naïve""#, "json_escape should preserve UTF-8 chars");
    }

    #[test]
    fn test_value_display_utf8_string() {
        // Verify Value::String Display preserves multi-byte UTF-8
        let val = Value::String("café ☕ 日本語".to_string());
        let displayed = format!("{val}");
        assert_eq!(displayed, r#""café ☕ 日本語""#, "Value display should preserve UTF-8");
    }

    #[test]
    fn test_concat_preserves_utf8() {
        let source = r#"
+fn greet (name:String)->String
  +return concat("Bonjour, ", name)
"#;
        let program = build_program(source);
        let input = parser::parse(r#"!eval greet name="André""#)
            .unwrap()
            .into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .unwrap();
        let result = eval_call_with_input(&program, &input.function_name, &input.input).unwrap();
        assert_eq!(result, r#""Bonjour, André""#, "concat should preserve UTF-8");
    }

    #[test]
    fn test_unicode_string_literal_roundtrip() {
        // Full Unicode string literal: parse → eval → display must preserve
        // all multi-byte characters exactly.
        let source = "
+fn probe ()->String
  +return \"hé — 你好 ✓ ★\"
";
        let program = build_program(source);
        let input = parser::parse("!eval probe")
            .unwrap()
            .into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .unwrap();
        let result = eval_call_with_input(&program, &input.function_name, &input.input).unwrap();
        assert_eq!(result, "\"hé — 你好 ✓ ★\"",
            "Unicode string literal must survive parse/eval/display without mojibake");
        // Verify actual byte representation
        let inner = &result[1..result.len()-1]; // strip quotes
        assert_eq!(inner.as_bytes(), "hé — 你好 ✓ ★".as_bytes(),
            "UTF-8 byte representation must match");
    }

    #[test]
    fn test_unicode_json_serialization_roundtrip() {
        // Build JSON containing Unicode via json_escape, then extract via json_get.
        // This simulates what send_message_body does with Unicode text.
        let source = r#"
+fn build_json (text:String)->String
  +let escaped:String = json_escape(text)
  +let body:String = concat("{\"text\":\"", concat(escaped, "\"}"))
  +return body

+fn extract_text (json:String)->String
  +return json_get(json, "text")

+fn roundtrip (text:String)->String
  +let json:String = build_json(text)
  +return extract_text(json)
"#;
        let program = build_program(source);
        let input = parser::parse("!eval roundtrip text=\"café — 你好 ✓ ★\"")
            .unwrap()
            .into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .unwrap();
        let result = eval_call_with_input(&program, &input.function_name, &input.input).unwrap();
        assert_eq!(result, "\"café — 你好 ✓ ★\"",
            "Unicode text must survive json_escape → JSON embedding → json_get roundtrip");
    }

    #[tokio::test]
    async fn test_mocked_http_utf8_roundtrip() {
        // Simulate an http_get returning UTF-8 text via mock, then extracting it.
        let mut session = crate::session::Session::new();

        let source = "
+fn fetch_text (url:String)->String [async]
  +await resp:String = http_get(url)
  +return json_get(resp, \"text\")
";
        let _ = session.apply_async(source, None).await;

        // Mock returns JSON with Unicode content
        let mock_source = "!mock http_get \"unicode-test\" -> \"{\\\"text\\\":\\\"café — 你好 ✓\\\"}\"";
        let _ = session.apply_async(mock_source, None).await;

        let test_source = "
!test fetch_text
  +with url=\"https://unicode-test.example.com\" -> expect \"café — 你好 ✓\"
";
        let results = session.apply_async(test_source, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].1, "mocked HTTP UTF-8 test should pass: {:?}", results[0]);
        assert!(results[0].0.contains("PASS"),
            "mocked HTTP returning UTF-8 JSON should round-trip: {:?}", results[0]);
    }

    #[tokio::test]
    async fn test_mocked_llm_utf8_roundtrip() {
        // Simulate llm_call returning UTF-8 text via mock.
        let mut session = crate::session::Session::new();

        let source = "
+fn ask_llm (prompt:String)->String [async]
  +await reply:String = llm_call(prompt, \"echo\")
  +return reply
";
        let _ = session.apply_async(source, None).await;

        // Mock llm_call to return Unicode text
        let mock_source = "!mock llm_call \"test\" -> \"café — 你好 ✓ ★\"";
        let _ = session.apply_async(mock_source, None).await;

        let test_source = "
!test ask_llm
  +with prompt=\"test prompt\" -> expect \"café — 你好 ✓ ★\"
";
        let results = session.apply_async(test_source, None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].1, "mocked LLM UTF-8 test should pass: {:?}", results[0]);
    }

    // ── Pure function calls in test parameters ────────────────────────

    #[test]
    fn test_zero_arg_function_as_test_param_value() {
        // A function call with () used as a test parameter value
        // should call the function and use its return value
        let source = "\
+type Config = {host:String, port:Int}

+fn make_default ()->Config
  +let c:Config = {host: \"localhost\", port: 8080}
  +return c

+fn get_host (c:Config)->String
  +return c.host

!test get_host
  +with c=make_default() -> expect \"localhost\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_ok(), "zero-arg function call as param value should work: {:?}", result);
        assert!(result.unwrap().contains("expected \"localhost\""));
    }

    #[test]
    fn test_bare_function_name_as_test_param_value() {
        // A bare function name (no parens) for a zero-arg function should
        // be called, not turned into Err("make_default")
        let source = "\
+type Config = {host:String, port:Int}

+fn make_default ()->Config
  +let c:Config = {host: \"localhost\", port: 8080}
  +return c

+fn get_host (c:Config)->String
  +return c.host

!test get_host
  +with c=make_default -> expect \"localhost\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_ok(), "bare function name as param value should call function: {:?}", result);
        assert!(result.unwrap().contains("expected \"localhost\""));
    }

    #[test]
    fn test_function_call_with_args_in_test_param() {
        // A function call with arguments in a test parameter should work
        let source = "\
+type Config = {host:String, port:Int}

+fn make_config (h:String, p:Int)->Config
  +let c:Config = {host: h, port: p}
  +return c

+fn get_port (c:Config)->Int
  +return c.port

!test get_port
  +with c=make_config(\"example.com\", 3000) -> expect 3000
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_ok(), "function call with args as param value: {:?}", result);
        assert!(result.unwrap().contains("expected 3000"));
    }

    #[test]
    fn test_function_call_in_expected_value() {
        // Function calls should also work on the expected side of ->
        let source = "\
+type Config = {host:String, port:Int}

+fn make_default ()->Config
  +let c:Config = {host: \"localhost\", port: 8080}
  +return c

+fn identity (c:Config)->Config
  +return c

!test identity
  +with c=make_default() -> expect make_default()
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_ok(), "function call in expected value: {:?}", result);
        assert!(result.unwrap().contains("expected"));
    }

    // ── Positional (space-separated) args in !eval and +with ──────────

    #[test]
    fn test_eval_positional_multiple_strings() {
        let source = "\
+fn concat_two (a:String, b:String)->String
  +let result:String = concat(a, b)
  +return result
";
        let program = build_program(source);
        let eval_source = r#"!eval concat_two "hello" "world""#;
        let ops = parser::parse(eval_source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        let result = eval_compiled_or_interpreted(&program, &ev.function_name, &ev.input);
        assert!(result.is_ok(), "positional string args should work: {:?}", result);
        assert_eq!(result.unwrap().0, "\"helloworld\"");
    }

    #[test]
    fn test_eval_positional_mixed_types() {
        let source = "\
+fn show (a:String, b:Int)->String
  +let result:String = concat(a, to_string(b))
  +return result
";
        let program = build_program(source);
        let eval_source = r#"!eval show "count:" 42"#;
        let ops = parser::parse(eval_source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        let result = eval_compiled_or_interpreted(&program, &ev.function_name, &ev.input);
        assert!(result.is_ok(), "mixed positional args: {:?}", result);
        assert_eq!(result.unwrap().0, "\"count:42\"");
    }

    #[test]
    fn test_eval_positional_ints() {
        let source = "\
+fn add (a:Int, b:Int)->Int
  +let result:Int = a + b
  +return result
";
        let program = build_program(source);
        let eval_source = "!eval add 3 4";
        let ops = parser::parse(eval_source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        let result = eval_compiled_or_interpreted(&program, &ev.function_name, &ev.input);
        assert!(result.is_ok(), "positional int args: {:?}", result);
        assert_eq!(result.unwrap().0, "7");
    }

    #[test]
    fn test_with_positional_strings() {
        let source = "\
+fn concat_two (a:String, b:String)->String
  +let result:String = concat(a, b)
  +return result

!test concat_two
  +with \"hello\" \"world\" -> expect \"helloworld\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_ok(), "+with positional strings: {:?}", result);
        assert!(result.unwrap().contains("expected \"helloworld\""));
    }

    #[test]
    fn test_eval_builtin_positional_strings() {
        // Test that positional args also work for builtins
        let program = ast::Program::default();
        let eval_source = r#"!eval concat "foo" "bar""#;
        let ops = parser::parse(eval_source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        let result = eval_compiled_or_interpreted(&program, &ev.function_name, &ev.input);
        assert!(result.is_ok(), "builtin positional strings: {:?}", result);
        assert_eq!(result.unwrap().0, "\"foobar\"");
    }

    // ── Inline expression eval (!eval <expr>) ─────────────────────────

    #[test]
    fn test_eval_inline_arithmetic() {
        let program = ast::Program::default();
        let source = "!eval 1 + 2";
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some(), "should be inline expression");
        let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
        assert!(result.is_ok(), "inline 1+2: {:?}", result);
        assert_eq!(format!("{}", result.unwrap()), "3");
    }

    #[test]
    fn test_eval_inline_multiply_add() {
        let program = ast::Program::default();
        let source = "!eval 3 * 4 + 1";
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
        assert!(result.is_ok(), "inline 3*4+1: {:?}", result);
        assert_eq!(format!("{}", result.unwrap()), "13");
    }

    #[test]
    fn test_eval_inline_string_literal() {
        let program = ast::Program::default();
        let source = r#"!eval "hello""#;
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
        assert!(result.is_ok(), "inline string: {:?}", result);
        assert_eq!(format!("{}", result.unwrap()), "\"hello\"");
    }

    #[test]
    fn test_eval_inline_numeric_literal() {
        let program = ast::Program::default();
        let source = "!eval 42";
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
        assert!(result.is_ok(), "inline 42: {:?}", result);
        assert_eq!(format!("{}", result.unwrap()), "42");
    }

    #[test]
    fn test_eval_inline_boolean() {
        let program = ast::Program::default();
        let source = "!eval true";
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
        assert!(result.is_ok(), "inline true: {:?}", result);
        assert_eq!(format!("{}", result.unwrap()), "true");
    }

    #[test]
    fn test_eval_inline_comparison() {
        let program = ast::Program::default();
        let source = "!eval 3 > 2";
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
        assert!(result.is_ok(), "inline 3>2: {:?}", result);
        assert_eq!(format!("{}", result.unwrap()), "true");
    }

    #[test]
    fn test_eval_inline_concat_call() {
        let program = ast::Program::default();
        let source = r#"!eval concat("hello", " ", "world")"#;
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
        assert!(result.is_ok(), "inline concat: {:?}", result);
        assert_eq!(format!("{}", result.unwrap()), "\"hello world\"");
    }

    #[test]
    fn test_eval_inline_len_call() {
        let program = ast::Program::default();
        let source = r#"!eval len("hello")"#;
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
        assert!(result.is_ok(), "inline len: {:?}", result);
        assert_eq!(format!("{}", result.unwrap()), "5");
    }

    #[test]
    fn test_eval_inline_nested_calls() {
        let program = ast::Program::default();
        let source = r#"!eval len(concat("a", "b"))"#;
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
        assert!(result.is_ok(), "inline nested calls: {:?}", result);
        assert_eq!(format!("{}", result.unwrap()), "2");
    }

    #[test]
    fn test_eval_inline_struct_literal() {
        let program = ast::Program::default();
        let source = r#"!eval {name: "alice", age: 25}"#;
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
        assert!(result.is_ok(), "inline struct: {:?}", result);
        let val = result.unwrap();
        // Struct should contain the fields
        match val {
            Value::Struct(_, fields) => {
                assert!(matches!(fields.get("name"), Some(Value::String(s)) if s == "alice"), "expected name=alice");
                assert!(matches!(fields.get("age"), Some(Value::Int(25))), "expected age=25");
            }
            _ => panic!("expected struct, got {val}"),
        }
    }

    #[test]
    fn test_eval_inline_list_creation() {
        let program = ast::Program::default();
        let source = "!eval list(1, 2, 3)";
        let ops = parser::parse(source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
        assert!(result.is_ok(), "inline list: {:?}", result);
        assert_eq!(format!("{}", result.unwrap()), "[1, 2, 3]");
    }

    #[test]
    fn test_eval_inline_user_function_call() {
        // Inline expression calling a user-defined function
        let source = "\
+fn double (x:Int)->Int
  +let result:Int = x * 2
  +return result
";
        let program = build_program(source);
        let eval_source = "!eval double(5)";
        let ops = parser::parse(eval_source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_some());
        let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
        assert!(result.is_ok(), "inline user fn call: {:?}", result);
        assert_eq!(format!("{}", result.unwrap()), "10");
    }

    #[test]
    fn test_eval_func_name_still_works() {
        // Existing !eval func_name syntax should still work
        let source = "\
+fn greet ()->String
  +return \"hello\"
";
        let program = build_program(source);
        let eval_source = "!eval greet";
        let ops = parser::parse(eval_source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_none(), "bare function name should not be inline");
        let result = eval_compiled_or_interpreted(&program, &ev.function_name, &ev.input);
        assert!(result.is_ok(), "bare function name: {:?}", result);
        assert_eq!(result.unwrap().0, "\"hello\"");
    }

    #[test]
    fn test_eval_func_with_args_still_works() {
        // Existing !eval func_name arg1 arg2 syntax should still work
        let source = "\
+fn add (a:Int, b:Int)->Int
  +let result:Int = a + b
  +return result
";
        let program = build_program(source);
        let eval_source = "!eval add 3 4";
        let ops = parser::parse(eval_source).expect("parse should succeed");
        let ev = ops.into_iter()
            .find_map(|op| if let parser::Operation::Eval(ev) = op { Some(ev) } else { None })
            .expect("should have eval op");
        assert!(ev.inline_expr.is_none(), "func + args should not be inline");
        let result = eval_compiled_or_interpreted(&program, &ev.function_name, &ev.input);
        assert!(result.is_ok(), "func with args: {:?}", result);
        assert_eq!(result.unwrap().0, "7");
    }

    // ── Side-effect checks for function calls in test params ──────────

    #[test]
    fn test_impure_function_rejected_in_test_param_bare() {
        // A function with [io,async] effects used as a bare name in a test
        // param should produce a clear error, not a confusing runtime failure
        let source = "\
+type State = {count:Int, name:String}

+fn fetch_state ()->State [io,async]
  +await data:String = http_get(\"http://example.com\")
  +let s:State = {count: 0, name: data}
  +return s

+fn get_name (state:State)->String
  +return state.name

!test get_name
  +with state=fetch_state -> expect \"default\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_err(), "impure function should be rejected: {:?}", result);
        let err = result.unwrap_err().to_string();
        assert!(err.contains("side effects"), "error should mention side effects: {err}");
        assert!(err.contains("fetch_state"), "error should name the function: {err}");
    }

    #[test]
    fn test_impure_function_rejected_in_test_param_call() {
        // Same but with parens: state=fetch_state()
        let source = "\
+type State = {count:Int, name:String}

+fn fetch_state ()->State [io,async]
  +await data:String = http_get(\"http://example.com\")
  +let s:State = {count: 0, name: data}
  +return s

+fn get_name (state:State)->String
  +return state.name

!test get_name
  +with state=fetch_state() -> expect \"default\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_err(), "impure function call should be rejected: {:?}", result);
        let err = result.unwrap_err().to_string();
        assert!(err.contains("side effects"), "error should mention side effects: {err}");
    }

    #[test]
    fn test_fail_effect_allowed_in_test_param() {
        // [fail] is not a side effect — it should be allowed
        let source = "\
+type Config = {host:String, port:Int}

+fn validated_config (host:String, port:Int)->Config [fail]
  +check valid_port port > 0 ~err_invalid_port
  +let c:Config = {host: host, port: port}
  +return c

+fn get_host (cfg:Config)->String
  +return cfg.host

!test get_host
  +with cfg=validated_config(\"localhost\", 8080) -> expect \"localhost\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_ok(), "[fail] function should be allowed in test params: {:?}", result);
        assert!(result.unwrap().contains("expected \"localhost\""));
    }

    // ── Escaped quotes in test value strings ──────────────────────────

    #[test]
    fn test_escaped_quotes_in_key_value_string() {
        // Strings with escaped quotes in key=value test params should work
        let source = r#"
+fn identity (s:String)->String
  +return s

!test identity
  +with s="hello\"world" -> expect "hello\"world"
"#;
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_ok(), "escaped quotes in key=value string: {:?}", result);
    }

    #[test]
    fn test_escaped_quotes_multiple_string_args() {
        // Multiple string args with escaped quotes should parse correctly
        let source = r#"
+fn concat_two (a:String, b:String)->String
  +let result:String = concat(a, b)
  +return result

!test concat_two
  +with a="he\"llo" b="wo\"rld" -> expect "he\"llowo\"rld"
"#;
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_ok(), "escaped quotes in multiple args: {:?}", result);
    }

    #[test]
    fn test_newline_and_tab_escapes_in_test_value() {
        let source = r#"
+fn identity (s:String)->String
  +return s

!test identity
  +with s="line1\nline2" -> expect "line1\nline2"
"#;
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_ok(), "newline escape in test value: {:?}", result);
    }

    // ── IO builtin error message tests ──────────────────────────────

    #[test]
    fn test_io_builtin_without_await_gives_helpful_error() {
        // Calling an IO builtin like http_get without +await should give a
        // specific error, not "undefined function"
        let source = "\
+fn broken (url:String)->String
  +let resp:String = http_get(url)
  +return resp
";
        let program = build_program(source);
        let test_source = "\
!test broken
  +with url=\"http://example.com\" -> expect \"\"
";
        let cases = extract_test_cases(test_source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_err(), "should fail when IO builtin used without +await");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("async IO operation"), "error should mention async IO: {err}");
        assert!(err.contains("+await"), "error should suggest +await: {err}");
        assert!(!err.contains("undefined function"), "should NOT say undefined function: {err}");
    }

    // ── +each scoping tests ──────────────────────────────────────────

    #[test]
    fn test_each_return_propagation() {
        // +return inside +each should propagate to the enclosing function
        let source = "\
+type Message = {role:String, content:String}

+fn find_role (messages:List<Message>)->String
  +each messages msg:Message
    +return msg.role
  +end
  +return \"none\"

!test find_role
  +with messages=list({role: \"user\", content: \"hi\"}) -> expect \"user\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_ok(), "+return inside +each should propagate: {:?}", result);
    }

    #[test]
    fn test_each_field_access_and_let() {
        // +let with field access on the loop variable should work
        let source = "\
+type Message = {role:String, content:String}

+fn build_text (messages:List<Message>)->String
  +let result:String = \"\"
  +each messages msg:Message
    +let role:String = msg.role
    +set result = concat(result, role, \" \")
  +end
  +return result

!test build_text
  +with messages=list({role: \"user\", content: \"hi\"}, {role: \"bot\", content: \"hello\"}) -> expect \"user bot \"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
        assert!(result.is_ok(), "+let with field access in +each: {:?}", result);
    }

    // ── format_expr round-trip tests ──────────────────────────────────

    #[test]
    fn test_format_expr_struct_with_list_roundtrip() {
        // Bug: format_expr produced `messages=list()last_id=0` (missing separator)
        // for struct fields containing function calls, causing reparse failure.

        let source = "\
+type State = {messages:List<String>, last_id:Int}

+fn process (s:State)->Int
  +return s.last_id
";
        let program = build_program(source);

        // Simulate a stored test with struct input containing list()
        let test_source = "\
!test process
  +with s={messages: list(), last_id: 0} -> expect 0
";
        let cases = extract_test_cases(test_source);
        assert_eq!(cases.len(), 1);
        let (fn_name, case) = &cases[0];

        // First, verify the test passes directly
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_ok(), "direct test should pass: {:?}", result);

        // Now simulate the store/reparse cycle that invalidate_and_retest does
        let input_str = crate::session::format_expr_pub(&case.input);
        let expected_str = crate::session::format_expr_pub(&case.expected);

        // Reconstruct test source (same as invalidate_and_retest)
        let reconstructed = format!("!test process\n  +with {} -> expect {}\n", input_str, expected_str);

        // The reconstructed source must parse successfully
        let reparse = parser::parse(&reconstructed);
        assert!(reparse.is_ok(), "reconstructed test should parse: {:?} from source: {}", reparse.err(), reconstructed);

        // And the reparsed test should pass
        let reparsed_cases = extract_test_cases(&reconstructed);
        assert_eq!(reparsed_cases.len(), 1);
        let result = eval_test_case_with_mocks(&program, &reparsed_cases[0].0, &reparsed_cases[0].1, &[], &[]);
        assert!(result.is_ok(), "reparsed test should pass: {:?}", result);
    }

    // ── Test matchers ────────────────────────────────────────────────

    #[test]
    fn test_contains_matcher() {
        let source = "\
+fn greet (name:String)->String
  +return concat(\"hello \", name, \"!\")

!test greet
  +with name=\"world\" -> expect contains(\"hello\")
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let (fn_name, case) = &cases[0];
        assert!(case.matcher.is_some(), "should have a matcher");
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_ok(), "contains matcher should pass: {:?}", result);
    }

    #[test]
    fn test_contains_matcher_fails_when_not_present() {
        let source = "\
+fn greet (name:String)->String
  +return concat(\"hello \", name, \"!\")

!test greet
  +with name=\"world\" -> expect contains(\"goodbye\")
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_err(), "contains matcher should fail when substring absent");
    }

    #[test]
    fn test_starts_with_matcher() {
        let source = "\
+fn greet (name:String)->String
  +return concat(\"hello \", name)

!test greet
  +with name=\"world\" -> expect starts_with(\"hello\")
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_ok(), "starts_with matcher should pass: {:?}", result);
    }

    #[test]
    fn test_starts_with_matcher_fails() {
        let source = "\
+fn greet (name:String)->String
  +return concat(\"hello \", name)

!test greet
  +with name=\"world\" -> expect starts_with(\"goodbye\")
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_err(), "starts_with matcher should fail on wrong prefix");
    }

    #[test]
    fn test_any_ok_matcher() {
        let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x

!test validate
  +with x=5 -> expect Ok
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        assert!(matches!(case.matcher, Some(parser::TestMatcher::AnyOk)));
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_ok(), "AnyOk matcher should pass for Ok result: {:?}", result);
    }

    #[test]
    fn test_any_ok_matcher_fails_on_err() {
        let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x

!test validate
  +with x=-1 -> expect Ok
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_err(), "AnyOk matcher should fail on Err result");
    }

    #[test]
    fn test_any_err_matcher() {
        let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x

!test validate
  +with x=-1 -> expect Err
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        assert!(matches!(case.matcher, Some(parser::TestMatcher::AnyErr)));
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_ok(), "AnyErr matcher should pass for Err result: {:?}", result);
    }

    #[test]
    fn test_any_err_matcher_fails_on_ok() {
        let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x

!test validate
  +with x=5 -> expect Err
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_err(), "AnyErr matcher should fail on Ok result");
    }

    #[test]
    fn test_err_containing_matcher() {
        let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x

!test validate
  +with x=-1 -> expect Err(\"err_negative\")
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        assert!(matches!(case.matcher, Some(parser::TestMatcher::ErrContaining(_))));
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_ok(), "ErrContaining matcher should pass: {:?}", result);
    }

    #[test]
    fn test_err_containing_matcher_fails_on_wrong_msg() {
        let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x

!test validate
  +with x=-1 -> expect Err(\"err_something_else\")
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_err(), "ErrContaining should fail on wrong message");
    }

    // ── +after checks ────────────────────────────────────────────────

    #[test]
    fn test_after_routes_contains_pass() {
        // Build a program with a route, then test with +after routes contains
        let source = "\
+fn handler (body:String)->String
  +return \"ok\"

+route POST \"/chat\" -> handler

!test handler
  +with body=\"hello\" -> expect \"ok\"
  +after routes contains \"/chat\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        assert_eq!(cases.len(), 1);
        let (fn_name, case) = &cases[0];
        assert_eq!(case.after_checks.len(), 1);
        assert_eq!(case.after_checks[0].target, "routes");
        assert_eq!(case.after_checks[0].value, "/chat");

        // Routes are now in RuntimeState, pass them explicitly
        let routes = vec![crate::ast::HttpRoute {
            method: "POST".to_string(),
            path: "/chat".to_string(),
            handler_fn: "handler".to_string(),
        }];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &routes);
        assert!(result.is_ok(), "+after routes should pass: {:?}", result);
    }

    #[test]
    fn test_after_routes_contains_fail() {
        // Route doesn't exist — +after should fail
        let source = "\
+fn handler (body:String)->String
  +return \"ok\"

!test handler
  +with body=\"hello\" -> expect \"ok\"
  +after routes contains \"/nonexistent\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_err(), "+after routes should fail when route missing");
    }

    #[test]
    fn test_after_modules_contains_pass() {
        let source = "\
!module TestMod

+fn helper ()->Int
  +return 42

!test helper
  +with -> expect 42
  +after modules contains \"TestMod\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_ok(), "+after modules should pass: {:?}", result);
    }

    #[test]
    fn test_after_modules_contains_fail() {
        let source = "\
+fn helper ()->Int
  +return 42

!test helper
  +with -> expect 42
  +after modules contains \"NonExistent\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_err(), "+after modules should fail when module missing");
    }

    #[test]
    fn test_matchers_combined_with_after() {
        // Combine a matcher with +after checks
        let source = "\
+fn handler (body:String)->String
  +return concat(\"processed: \", body)

+route GET \"/api\" -> handler

!test handler
  +with body=\"test\" -> expect contains(\"processed\")
  +after routes contains \"/api\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        assert!(case.matcher.is_some());
        assert_eq!(case.after_checks.len(), 1);
        let routes = vec![crate::ast::HttpRoute {
            method: "GET".to_string(),
            path: "/api".to_string(),
            handler_fn: "handler".to_string(),
        }];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &routes);
        assert!(result.is_ok(), "matcher + after should pass: {:?}", result);
    }

    #[test]
    fn test_multiple_after_checks() {
        let source = "\
!module MyMod

+fn handler (body:String)->String
  +return \"ok\"

+route POST \"/webhook\" -> handler

!test handler
  +with body=\"x\" -> expect \"ok\"
  +after routes contains \"/webhook\"
  +after modules contains \"MyMod\"
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        assert_eq!(case.after_checks.len(), 2);
        let routes = vec![crate::ast::HttpRoute {
            method: "POST".to_string(),
            path: "/webhook".to_string(),
            handler_fn: "handler".to_string(),
        }];
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &routes);
        assert!(result.is_ok(), "multiple +after checks should pass: {:?}", result);
    }

    #[test]
    fn test_exact_err_still_works_without_matcher() {
        // Err(err_label) without quotes should still work as exact match (no matcher)
        let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x

!test validate
  +with x=-1 -> expect Err(err_negative)
";
        let program = build_program(source);
        let cases = extract_test_cases(source);
        let (fn_name, case) = &cases[0];
        // No matcher — this is the old exact match behavior
        assert!(case.matcher.is_none(), "Err(bare_ident) should not create a matcher");
        let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
        assert!(result.is_ok(), "exact Err match should still work: {:?}", result);
    }
}
