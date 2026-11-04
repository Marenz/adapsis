//! Only actual typed NodeId fields are erased. Names, `id` struct fields,
//! callee spelling, docs, test metadata and effect/declaration order stay exact.
use crate::ast;
use std::sync::Arc;

#[cfg(test)]
pub(super) fn program(a: &ast::Program, b: &ast::Program) -> bool {
    let (mut a, mut b) = (a.clone(), b.clone());
    for program in [&mut a, &mut b] {
        for module in &mut program.modules {
            clear_module(module);
        }
        for function in &mut program.functions {
            clear_function(Arc::make_mut(function));
        }
        for ty in &mut program.types {
            clear_type(ty);
        }
    }
    exact(&a, &b)
}

pub(super) fn function(a: &ast::FunctionDecl, b: &ast::FunctionDecl) -> bool {
    let (mut a, mut b) = (a.clone(), b.clone());
    clear_function(&mut a);
    clear_function(&mut b);
    exact(&a, &b)
}

#[cfg(test)]
pub(super) fn function_difference(a: &ast::FunctionDecl, b: &ast::FunctionDecl) -> String {
    let (mut a, mut b) = (a.clone(), b.clone());
    clear_function(&mut a);
    clear_function(&mut b);
    difference(
        &serde_json::to_value(a).unwrap(),
        &serde_json::to_value(b).unwrap(),
        "function",
    )
    .unwrap_or_else(|| "unknown structural difference".into())
}

#[cfg(test)]
pub(super) fn module_difference(a: &ast::Module, b: &ast::Module) -> String {
    let (mut a, mut b) = (a.clone(), b.clone());
    clear_module(&mut a);
    clear_module(&mut b);
    difference(
        &serde_json::to_value(a).unwrap(),
        &serde_json::to_value(b).unwrap(),
        "module",
    )
    .unwrap_or_else(|| "unknown structural difference".into())
}

/// Report only a structural path, never expression/literal values.
#[cfg(test)]
fn difference(a: &serde_json::Value, b: &serde_json::Value, path: &str) -> Option<String> {
    use serde_json::Value;
    if a == b {
        return None;
    }
    match (a, b) {
        (Value::Object(a), Value::Object(b)) if a.keys().eq(b.keys()) => a
            .iter()
            .find_map(|(key, value)| difference(value, &b[key], &format!("{path}.{key}"))),
        (Value::Array(a), Value::Array(b)) if a.len() == b.len() => a
            .iter()
            .zip(b)
            .enumerate()
            .find_map(|(i, (a, b))| difference(a, b, &format!("{path}[{i}]"))),
        _ => Some(path.into()),
    }
}

pub(super) fn module(a: &ast::Module, b: &ast::Module) -> bool {
    let (mut a, mut b) = (a.clone(), b.clone());
    clear_module(&mut a);
    clear_module(&mut b);
    exact(&a, &b)
}

pub(super) fn ty(a: &ast::TypeDecl, b: &ast::TypeDecl) -> bool {
    let (mut a, mut b) = (a.clone(), b.clone());
    clear_type(&mut a);
    clear_type(&mut b);
    exact(&a, &b)
}

// Module::PartialEq intentionally omits fields such as routes/documentation.
// Compare the complete serialized structure AFTER clearing only typed NodeIds;
// never delete keys from the serialized value. Serialization errors fail closed.
fn exact<T: serde::Serialize>(a: &T, b: &T) -> bool {
    serde_json::to_value(a)
        .ok()
        .zip(serde_json::to_value(b).ok())
        .is_some_and(|(a, b)| a == b)
}

fn clear_module(module: &mut ast::Module) {
    module.id.clear();
    for function in &mut module.functions {
        clear_function(Arc::make_mut(function));
    }
    for ty in &mut module.types {
        clear_type(ty);
    }
    for module in &mut module.modules {
        clear_module(module);
    }
    for lifecycle in [&mut module.startup, &mut module.shutdown]
        .into_iter()
        .flatten()
    {
        lifecycle.id.clear();
        clear_body(&mut lifecycle.body);
    }
}

fn clear_type(ty: &mut ast::TypeDecl) {
    match ty {
        ast::TypeDecl::Struct(s) => {
            s.id.clear();
            for field in &mut s.fields {
                field.id.clear();
            }
        }
        ast::TypeDecl::TaggedUnion(u) => {
            u.id.clear();
            for variant in &mut u.variants {
                variant.id.clear();
            }
        }
    }
}

fn clear_function(function: &mut ast::FunctionDecl) {
    function.id.clear();
    for param in &mut function.params {
        param.id.clear();
    }
    clear_body(&mut function.body);
}

fn clear_body(body: &mut [ast::Statement]) {
    for statement in body {
        statement.id.clear();
        use ast::StatementKind::*;
        match &mut statement.kind {
            Branch {
                then_body,
                else_body,
                ..
            } => {
                clear_body(then_body);
                clear_body(else_body);
            }
            While { body, .. } | Each { body, .. } => clear_body(body),
            Match { arms, .. } => {
                for arm in arms {
                    clear_body(&mut arm.body);
                }
            }
            Let { .. }
            | Call { .. }
            | Check { .. }
            | Return { .. }
            | Set { .. }
            | Await { .. }
            | Spawn { .. }
            | Yield { .. }
            | Source(_)
            | Event(_) => {}
        }
    }
}
