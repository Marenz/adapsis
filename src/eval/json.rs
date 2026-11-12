//! Pure JSON lookup. json_get and json_kind MUST share this exact path resolver;
//! compatibility includes skipped empty components and usize index parsing.
use super::Value;
use anyhow::{anyhow, bail, Result};
use serde_json::Value as Json;

fn arguments<'a>(args: &'a [Value], name: &str) -> Result<(Json, &'a str)> {
    if args.len() != 2 {
        bail!("{name}(json, key_path) expects 2 arguments");
    }
    let (Value::String(text), Value::String(path)) = (&args[0], &args[1]) else {
        bail!("{name} expects (String, String)");
    };
    let parsed = super::bounded::parse_json(text)?
        .map_err(|error| anyhow!("{name}: invalid JSON: {error}"))?;
    Ok((parsed, path.as_str()))
}

fn resolve<'a>(root: &'a Json, path: &str) -> Option<&'a Json> {
    let mut current = root;
    for key in path.split('.') {
        if key.is_empty() {
            continue;
        }
        current = if let Ok(index) = key.parse::<usize>() {
            current.get(index)
        } else {
            current.get(key)
        }?;
    }
    Some(current)
}

pub(super) fn get(args: &[Value]) -> Result<Value> {
    let (parsed, path) = arguments(args, "json_get")?;
    let Some(value) = resolve(&parsed, path) else {
        return Ok(Value::string(""));
    };
    // Preserve the original scalar conversions, missing sentinel and serialized
    // aggregate output. json_kind does not alter json_get's observable contract.
    Ok(match value {
        Json::String(text) => Value::string(text.clone()),
        Json::Number(number) => {
            if let Some(value) = number.as_i64() {
                Value::Int(value)
            } else if let Some(value) = number.as_f64() {
                Value::Float(value)
            } else {
                Value::string(number.to_string())
            }
        }
        Json::Bool(value) => Value::Bool(*value),
        Json::Null => Value::string("null"),
        other => Value::string(other.to_string()),
    })
}

pub(super) fn kind(args: &[Value]) -> Result<Value> {
    let (parsed, path) = arguments(args, "json_kind")?;
    let result = match resolve(&parsed, path) {
        None => "missing",
        Some(Json::Null) => "null",
        Some(Json::Bool(_)) => "bool",
        Some(Json::String(_)) => "string",
        Some(Json::Array(_)) => "array",
        Some(Json::Object(_)) => "object",
        Some(Json::Number(number)) => {
            if number.as_i64().is_some() {
                "int"
            } else if number.is_u64() {
                bail!("json_kind: integer outside signed 64-bit range; expected -9223372036854775808..9223372036854775807");
            } else if number.as_f64().is_some_and(f64::is_finite) {
                "float"
            } else {
                bail!("json_kind: number has no supported finite representation");
            }
        }
    };
    Ok(Value::string(result))
}

/// `json_valid(text)`: true iff the parse step shared by json_get/json_kind/
/// json_array_len succeeds. Total on String input -- malformed text, depth over
/// serde_json's recursion limit (more than 127 nested containers) and non-finite numbers such as 1e400 are `false`, never
/// an error. Validity is NOT representability: a represented u64 above i64::MAX is
/// valid here while json_kind still rejects it. Arity/type errors match json_kind.
pub(super) fn valid(args: &[Value]) -> Result<Value> {
    if args.len() != 1 {
        bail!("json_valid(json) expects 1 argument");
    }
    let Value::String(text) = &args[0] else {
        bail!("json_valid expects (String)");
    };
    Ok(Value::Bool(super::bounded::parse_json(text)?.is_ok()))
}

#[cfg(test)]
mod tests;
