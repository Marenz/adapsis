use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;
use std::fmt;

use crate::ast;
use crate::parser;

/// A runtime value during evaluation.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum Value {
    CoroutineHandle(crate::coroutine::CoroutineHandle),
    StateHandle(std::sync::Arc<std::sync::Mutex<Value>>),
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

/// Evaluation environment (scope).
pub struct Env {
    pub vars: HashMap<String, Value>,
}

impl Env {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }

    pub fn set(&mut self, name: &str, value: Value) {
        self.vars.insert(name.to_string(), value);
    }

    fn get(&self, name: &str) -> Result<&Value> {
        self.vars
            .get(name)
            .ok_or_else(|| anyhow!("undefined variable `{name}`"))
    }
}

/// Try to evaluate using the JIT compiler, falling back to the interpreter.
/// Returns (result_string, was_compiled).
pub fn eval_compiled_or_interpreted(
    program: &ast::Program,
    function_name: &str,
    input: &parser::Expr,
) -> Result<(String, bool)> {
    // If it's a builtin, use eval_call_with_input directly
    if program.get_function(function_name).is_none() && crate::builtins::is_builtin(function_name) {
        let result = eval_call_with_input(program, function_name, input)?;
        return Ok((result, false));
    }

    let func = program
        .get_function(function_name)
        .ok_or_else(|| anyhow!("function `{function_name}` not found"))?;

    // Try compiled path
    if crate::compiler::is_compilable_function(func) {
        if let Ok(mut compiled) = crate::compiler::compile(program) {
            // Convert input to i64 args
            if let Ok(args) = input_to_i64_args(input, func) {
                if let Ok(result) = compiled.call_i64(function_name, &args) {
                    return Ok((format!("{result}"), true));
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
        let input_val = eval_parser_expr_standalone(input)?;
        // Convert input to args list
        let args = match &input_val {
            Value::Struct(_, fields) => {
                // Struct fields become positional args in order
                fields.values().cloned().collect::<Vec<_>>()
            }
            Value::None => vec![],
            other => vec![other.clone()],
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
    let func = program
        .get_function(function_name)
        .ok_or_else(|| anyhow!("function `{function_name}` not found"))?;

    let input = eval_parser_expr_with_program(&case.input, program)?;
    let expected = eval_parser_expr_with_program(&case.expected, program)?;

    let mut env = Env::new();
    bind_input_to_params(program, func, &input, &mut env);

    // Execute function body
    let result = eval_function_body(program, &func.body, &mut env);

    // If the function returns Result<T>, wrap successful returns in Ok()
    // and check failures produce Err()
    let returns_result = matches!(&func.return_type, ast::Type::Result(_));

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

    if actual.matches(&expected) {
        Ok(format!("input={input} => {actual} (expected {expected})"))
    } else {
        bail!("input={input} => {actual}, expected {expected}")
    }
}

/// Public entry point for running a function body with an env.
pub fn eval_function_body_pub(
    program: &ast::Program,
    body: &[ast::Statement],
    env: &mut Env,
) -> Result<Value> {
    eval_function_body(program, body, env)
}

fn eval_function_body(
    program: &ast::Program,
    body: &[ast::Statement],
    env: &mut Env,
) -> Result<Value> {
    for stmt in body {
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
                let result = eval_function_body(program, branch, env)?;
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
                            env.set(&binding.name, item);
                            // Execute body but don't return from each unless explicitly returned
                            match eval_function_body(program, each_body, env) {
                                Ok(_) => {}
                                Err(e) => return Err(e),
                            }
                        }
                    }
                    _ => bail!("each: expected list, got {}", iter_val),
                }
            }
            ast::StatementKind::Set { name, value } => {
                let val = eval_ast_expr(program, value, env)?;
                env.set(name, val);
            }
            ast::StatementKind::Match { expr, arms } => {
                let val = eval_ast_expr(program, expr, env)?;
                match &val {
                    Value::Union { variant, payload } => {
                        let mut matched = false;
                        for arm in arms {
                            if arm.variant == *variant {
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
                                        continue; // nested pattern didn't match, try next arm
                                    }
                                }
                                let result = eval_function_body(program, &arm.body, env)?;
                                if !matches!(result, Value::None) {
                                    return Ok(result);
                                }
                                matched = true;
                                break;
                            } else if arm.variant == "_" {
                                // Wildcard/default case — bind the whole value if there's a binding
                                if arm.bindings.len() == 1 && arm.bindings[0] != "_" {
                                    env.set(&arm.bindings[0], val.clone());
                                }
                                let result = eval_function_body(program, &arm.body, env)?;
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
                    _ => bail!("+match expects a union value, got {val}"),
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
                    let handle = match env.vars.get("__coroutine_handle") {
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
            ast::StatementKind::Spawn { call } => {
                let handle = match env.vars.get("__coroutine_handle") {
                    Some(Value::CoroutineHandle(h)) => h.clone(),
                    _ => bail!("+spawn requires async context"),
                };
                let args: Vec<Value> = call
                    .args
                    .iter()
                    .map(|a| eval_ast_expr(program, a, env))
                    .collect::<Result<Vec<_>>>()?;
                let io_tx = handle.io_sender();
                let _ = io_tx.blocking_send(crate::coroutine::IoRequest::Spawn {
                    function_name: call.callee.clone(),
                    args,
                });
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
                    match eval_function_body(program, body, env) {
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
}

const MAX_CALL_DEPTH: usize = 64;

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
                if let Some(handle) = env.vars.get("__coroutine_handle") {
                    call_env.set("__coroutine_handle", handle.clone());
                }
                eval_function_body(program, &func.body, &mut call_env)
            } else {
                // Check if it's a union variant constructor
                if is_union_variant(program, callee) {
                    return Ok(Value::Union {
                        variant: callee.to_string(),
                        payload: args,
                    });
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
            // Fall through to standalone (handles union constructors, Ok, Err)
            eval_parser_expr_standalone(expr)
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
        .ok_or_else(|| anyhow!("function `{function_name}` not found"))?;

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
