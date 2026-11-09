//! Step A: structural validation independent of whole-source .ax emission/reparse.
//! This is NOT type checking, authority, or runtime validation. All existing
//! frontend/native/validator checks remain; legacy roundtrips are explicit
//! exporter/test oracles, not a required execution intermediate.
#[cfg(test)]
mod tests;
mod values;

use crate::parser as p;
use values::{bare, expr, name, text, ty};

type Checked = Result<(), String>;

#[derive(Clone, Copy)]
enum Place {
    Root,
    Module,
    Body,
}

pub(crate) fn validate(operations: &[p::Operation]) -> Checked {
    if operations.is_empty() {
        return Err("empty submission".into());
    }
    if operations
        .iter()
        .any(|op| matches!(op, p::Operation::Module(_)))
        && operations.len() != 1
    {
        return Err("Python module submission must contain exactly one module".into());
    }
    if operations
        .iter()
        .any(|op| matches!(op, p::Operation::Test(_)))
        && !operations
            .iter()
            .all(|op| matches!(op, p::Operation::Test(_)))
    {
        return Err("Python test submissions cannot contain declarations".into());
    }
    block(operations, Place::Root)
}

fn block(operations: &[p::Operation], place: Place) -> Checked {
    for (index, operation) in operations.iter().enumerate() {
        check(operation, place).map_err(|error| format!("operation {index}: {error}"))?;
    }
    Ok(())
}

fn check(op: &p::Operation, place: Place) -> Checked {
    use p::Operation::*;
    match (place, op) {
        (Place::Root, Module(module)) => {
            bare(&module.name)?;
            if let Some(doc) = &module.doc {
                text(doc)?;
            }
            block(&module.body, Place::Module)
        }
        (Place::Root | Place::Module, Function(function)) => function_decl(function, false),
        (Place::Root | Place::Module, Type(decl)) => type_decl(decl),
        (Place::Root, Test(test)) => values::test(test),
        (Place::Module, SharedVar(shared)) => {
            bare(&shared.name)?;
            ty(&shared.ty)?;
            expr(&shared.default)
        }
        (Place::Module, Startup(function)) => lifecycle(function, "startup"),
        (Place::Module, Shutdown(function)) => lifecycle(function, "shutdown"),
        (
            Place::Module,
            Route {
                method,
                path,
                handler_fn,
            },
        ) => {
            if method.is_empty() || !method.bytes().all(|b| b.is_ascii_uppercase()) {
                return Err("route method must be one uppercase token".into());
            }
            text(path)?;
            name(handler_fn)
        }
        (Place::Module, ModuleSource(source)) => module_source(source),
        (Place::Body, _) => statement(op),
        _ => Err("operation is not supported at this placement in Python IR".into()),
    }
}

fn lifecycle(function: &p::FunctionDecl, expected: &str) -> Checked {
    if function.name != expected
        || function.doc.is_some()
        || !matches!(&function.return_type, p::TypeExpr::Named(name) if name == "String")
    {
        return Err("unsupported lifecycle declaration metadata".into());
    }
    function_decl(function, true)
}

fn function_decl(function: &p::FunctionDecl, lifecycle: bool) -> Checked {
    bare(&function.name)?;
    let mut names = std::collections::HashSet::new();
    for param in &function.params {
        bare(&param.name)?;
        ty(&param.ty)?;
        if !names.insert(&param.name) {
            return Err("duplicate parameter".into());
        }
    }
    ty(&function.return_type)?;
    let mut effects = std::collections::HashSet::new();
    for effect in &function.effects {
        if !(effect == "async" || super::lower::is_effect_name(effect))
            || !effects.insert(effect.as_str())
        {
            return Err("unsupported or duplicate effect".into());
        }
    }
    if lifecycle
        && (!function.params.is_empty() || !effects.contains("io") || !effects.contains("async"))
    {
        return Err("lifecycle requires zero parameters and io,async effects".into());
    }
    if let Some(doc) = &function.doc {
        text(doc)?;
    }
    block(&function.body, Place::Body)
}

fn type_decl(decl: &p::TypeDecl) -> Checked {
    bare(&decl.name)?;
    let mut names = std::collections::HashSet::new();
    match &decl.body {
        p::TypeBody::Struct(fields) => {
            for field in fields {
                bare(&field.name)?;
                ty(&field.ty)?;
                if !names.insert(&field.name) {
                    return Err("duplicate struct field".into());
                }
            }
        }
        p::TypeBody::Union(variants) => {
            if variants.is_empty() {
                return Err("empty union".into());
            }
            for variant in variants {
                values::variant(&variant.name)?;
                if !names.insert(&variant.name) {
                    return Err("duplicate union variant".into());
                }
                for payload in &variant.payload {
                    ty(payload)?;
                }
            }
        }
        p::TypeBody::Alias(_) => return Err("type aliases are outside supported Python IR".into()),
    }
    // Reuse canonical conversion's field/payload structure checks; no Program
    // lookup, union reclassification, or caller authority is inferred here.
    crate::validator::convert_type_decl(decl)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

fn statement(op: &p::Operation) -> Checked {
    use p::Operation::*;
    match op {
        Let(v) => {
            bare(&v.name)?;
            ty(&v.ty)?;
            expr(&v.expr)
        }
        Set(v) => {
            bare(&v.name)?;
            expr(&v.expr)
        }
        Return(v) => expr(&v.expr),
        Call(v) => {
            bare(&v.name)?;
            ty(&v.ty)?;
            values::call(&v.expr)
        }
        Await(v) => {
            bare(&v.name)?;
            ty(&v.ty)?;
            values::call(&v.call)
        }
        Spawn(v) => {
            if let Some((binding, kind)) = &v.binding {
                bare(binding)?;
                ty(kind)?;
            }
            values::call(&v.call)
        }
        Check(v) => {
            bare(&v.name)?;
            bare(&v.err_label)?;
            expr(&v.expr)
        }
        If(v) => {
            expr(&v.condition)?;
            block(&v.then_body, Place::Body)?;
            for (condition, body) in &v.elif_branches {
                expr(condition)?;
                block(body, Place::Body)?;
            }
            block(&v.else_body, Place::Body)
        }
        While(v) => {
            expr(&v.condition)?;
            block(&v.body, Place::Body)
        }
        Each(v) => {
            expr(&v.collection)?;
            bare(&v.item)?;
            ty(&v.item_type)?;
            block(&v.body, Place::Body)
        }
        Match(v) => {
            expr(&v.expr)?;
            if v.arms.is_empty() {
                return Err("empty match".into());
            }
            for arm in &v.arms {
                values::arm(arm)?;
                block(&arm.body, Place::Body)?;
            }
            Ok(())
        }
        SourceAdd(v) => source_kind(&v.kind_text, &v.alias, &v.handler, op),
        SourceReplace(v) => source_kind(&v.kind_text, &v.alias, &v.handler, op),
        SourceRemove(alias) => bare(alias),
        SourceList => Ok(()),
        EventRegister(v) => {
            bare(&v.name)?;
            values::payload_type(&v.payload_type)
        }
        EventEmit(v) => {
            bare(&v.name)?;
            expr(&v.value)
        }
        _ => Err("unsupported operation in Python function body".into()),
    }
}

fn source_kind(kind: &str, alias: &str, handler: &str, op: &p::Operation) -> Checked {
    bare(alias)?;
    name(handler)?;
    if kind == "channel" {
    } else if let Some(timer) = kind
        .strip_prefix("timer(")
        .and_then(|s| s.strip_suffix(')'))
    {
        // Reuse the expression parser for this retained mini-language only;
        // there is no emitted/reparsed declaration or submission here.
        let parsed = p::parse_single_expr(timer).map_err(|_| "invalid timer expression")?;
        expr(&parsed)?;
        // parse_single_expr requires complete token consumption: punctuation
        // inside String nodes is data; unquoted delimiters/trailing tokens fail
        // its grammar. Keep only the legacy fragment's single-line boundary.
        if kind.contains(['\n', '\r']) {
            return Err("invalid source kind token".into());
        }
    } else {
        name(kind)?;
        if kind.split('.').count() != 2 {
            return Err("expected timer, channel or Module.event".into());
        }
    }
    crate::validator::convert_statement_op(op)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

fn module_source(source: &p::ModuleSourceDecl) -> Checked {
    bare(&source.name)?;
    bare(&source.source_type)?;
    name(&source.handler)?;
    let mut names = std::collections::HashSet::new();
    for (key, value) in &source.config {
        bare(key)?;
        if !names.insert(key)
            || value.is_empty()
            || value.chars().any(|c| {
                c.is_whitespace() || c.is_control() || matches!(c, '"' | '\'' | '=' | '>' | '<')
            })
        {
            return Err("source configuration requires unique keys and single-token values".into());
        }
    }
    Ok(())
}
