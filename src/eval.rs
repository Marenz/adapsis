use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;
use std::fmt;

use crate::ast;
use crate::parser;

/// A runtime value during evaluation.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum Value {
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
            _ => false,
        }
    }
}

/// Evaluation environment (scope).
struct Env {
    vars: HashMap<String, Value>,
}

impl Env {
    fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }

    fn set(&mut self, name: &str, value: Value) {
        self.vars.insert(name.to_string(), value);
    }

    fn get(&self, name: &str) -> Result<&Value> {
        self.vars
            .get(name)
            .ok_or_else(|| anyhow!("undefined variable `{name}`"))
    }
}

/// Evaluate a test case against a function in the program.
pub fn eval_test_case(
    program: &ast::Program,
    function_name: &str,
    case: &parser::TestCase,
) -> Result<String> {
    let func = program
        .get_function(function_name)
        .ok_or_else(|| anyhow!("function `{function_name}` not found"))?;

    let input = eval_parser_expr_standalone(&case.input)?;
    let expected = eval_parser_expr_standalone(&case.expected)?;

    // Set up environment with function params bound to input
    let mut env = Env::new();
    match (&input, func.params.len()) {
        (_, 0) => {} // no params
        (Value::Struct(_, fields), _) => {
            // If input is a struct, bind each param by name from the struct fields,
            // or bind the whole struct as the first param
            if func.params.len() == 1 {
                env.set(&func.params[0].name, input.clone());
                // Also expose fields directly for field access
                for (k, v) in fields {
                    env.set(k, v.clone());
                }
            } else {
                for param in &func.params {
                    if let Some(val) = fields.get(&param.name) {
                        env.set(&param.name, val.clone());
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
                // If the branch returned, propagate it
                return Ok(result);
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
            ast::StatementKind::Yield { value } => {
                // For Phase 1, yield just evaluates the expression (no real coroutine)
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

fn eval_call_inner(program: &ast::Program, call: &ast::CallExpr, env: &mut Env) -> Result<Value> {
    let args: Vec<Value> = call
        .args
        .iter()
        .map(|a| eval_ast_expr(program, a, env))
        .collect::<Result<Vec<_>>>()?;

    // Check for built-in functions
    match call.callee.as_str() {
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
            if let Some(func) = program.get_function(&call.callee) {
                let mut call_env = Env::new();
                for (param, arg) in func.params.iter().zip(args) {
                    call_env.set(&param.name, arg);
                }
                eval_function_body(program, &func.body, &mut call_env)
            } else {
                // Unknown function — return a placeholder
                bail!(
                    "undefined function `{}` (called with {} args)",
                    call.callee,
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
            // Handle special identifiers
            match name.as_str() {
                "true" => Ok(Value::Bool(true)),
                "false" => Ok(Value::Bool(false)),
                _ => env.get(name).cloned(),
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
fn eval_parser_expr_standalone(expr: &parser::Expr) -> Result<Value> {
    match expr {
        parser::Expr::Int(v) => Ok(Value::Int(*v)),
        parser::Expr::Float(v) => Ok(Value::Float(*v)),
        parser::Expr::Bool(v) => Ok(Value::Bool(*v)),
        parser::Expr::String(v) => Ok(Value::String(v.clone())),
        parser::Expr::Ident(name) => {
            // Handle special result constructors
            match name.as_str() {
                "Ok" => Ok(Value::Ok(Box::new(Value::None))),
                "None" => Ok(Value::None),
                // Treat unknown idents as error labels: Err(name)
                _ => Ok(Value::Err(name.clone())),
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
                _ => bail!("cannot evaluate call `{name}` in test case"),
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

    // Bind params
    match (&input_val, func.params.len()) {
        (_, 0) => {}
        (Value::Struct(_, fields), _) => {
            if func.params.len() == 1 {
                env.set(&func.params[0].name, input_val.clone());
                for (k, v) in fields {
                    env.set(k, v.clone());
                }
            } else {
                for param in &func.params {
                    if let Some(val) = fields.get(&param.name) {
                        env.set(&param.name, val.clone());
                    }
                }
            }
        }
        (_, 1) => {
            env.set(&func.params[0].name, input_val.clone());
        }
        _ => {
            env.set(&func.params[0].name, input_val.clone());
        }
    }

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
