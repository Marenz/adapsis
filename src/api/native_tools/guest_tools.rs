//! Direct native tools for guest profiles: the turn-bound builtins a guest could already
//! call with `!eval tool(literals)`, as JSON tools with the same literal arguments.
//!
//! Nothing is decided here. Each call dispatches the SAME builtin through a coroutine
//! handle bound to the turn identity, so the IO-boundary check against the guest's tool
//! list (`coroutine/scoped.rs`), the argument limits and every capability rule (one music
//! job, web fetch limits, the image-consent flag of `cad_prepare`) apply unchanged.
use super::*;
use crate::eval::Value;

enum Kind {
    Text,
    Int,
    Bool,
}

struct Field {
    name: &'static str,
    kind: Kind,
    required: bool,
    /// Used when a later optional field is given and this positional one is not.
    default: Option<&'static str>,
    help: &'static str,
}

const fn field(name: &'static str, kind: Kind, required: bool, default: Option<&'static str>, help: &'static str) -> Field {
    Field { name, kind, required, default, help }
}

/// Positional argument order of each builtin. The memory tools are not listed: they use
/// the existing memory tool definitions.
const SPECS: &[(&str, &[Field])] = &[
    ("music_generate", &[
        field("description", Kind::Text, true, None, "Genre, instruments, mood and vocal style."),
        field("duration_seconds", Kind::Int, true, None, "10 to 120."),
        field("lyrics", Kind::Text, false, None, "Optional lyrics; omit for an instrumental."),
    ]),
    ("web_search", &[field("query", Kind::Text, true, None, "Search terms.")]),
    ("web_read", &[field("url", Kind::Text, true, None, "A public http(s) URL.")]),
    ("context_propose", &[field("instructions", Kind::Text, true, None, "The FULL replacement text for this conversation's instructions.")]),
    ("cad_prepare", &[
        field("title", Kind::Text, true, None, "Short title of the object."),
        field("design_brief", Kind::Text, true, None, "Complete plain-language brief with dimensions and constraints."),
        field("material", Kind::Text, false, Some("PLA"), "PLA (default) or PETG."),
        field("use_recent_images", Kind::Bool, false, None, "true ONLY after the person agreed that up to three recent images of this chat are sent off-host to the external CAD author; default false."),
    ]),
    ("video_stabilize", &[]),
];

fn spec(name: &str) -> Option<&'static [Field]> {
    SPECS.iter().find(|(tool, _)| *tool == name).map(|(_, fields)| *fields)
}

pub(super) fn definition(name: &str) -> Option<ToolDefinition> {
    let fields = spec(name)?;
    let builtin = crate::builtins::IO_BUILTINS.iter().find(|builtin| builtin.name == name)?;
    let mut properties = serde_json::Map::new();
    for field in fields {
        let kind = match field.kind {
            Kind::Text => "string",
            Kind::Int => "integer",
            Kind::Bool => "boolean",
        };
        properties.insert(field.name.into(), serde_json::json!({"type": kind, "description": field.help}));
    }
    let required: Vec<&str> = fields.iter().filter(|field| field.required).map(|field| field.name).collect();
    Some(ToolDefinition {
        name: name.into(),
        description: format!("{}. {}", builtin.short, builtin.long),
        parameters: serde_json::json!({
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": false,
        }),
    })
}

/// Closed object to positional literal arguments. Trailing optional fields that were not
/// given are omitted, exactly as a shorter `!eval tool(...)` call would.
pub(super) fn parse(name: &str, map: &serde_json::Map<String, serde_json::Value>) -> Result<ParsedTool, String> {
    let (op, fields) = SPECS
        .iter()
        .find(|(tool, _)| *tool == name)
        .ok_or_else(|| format!("unknown native tool `{name}`"))?;
    let allowed: Vec<&str> = fields.iter().map(|field| field.name).collect();
    reject_extra_fields(map, &allowed)?;
    let last = fields.iter().rposition(|field| map.contains_key(field.name)).map_or(0, |index| index + 1);
    let mut args = Vec::new();
    for field in &fields[..last.max(fields.iter().filter(|field| field.required).count())] {
        let value = match (&field.kind, map.get(field.name)) {
            (Kind::Text, Some(serde_json::Value::String(text))) if text.len() <= MAX_ARGUMENTS_BYTES => Value::string(text.clone()),
            (Kind::Int, Some(serde_json::Value::Number(number))) if number.as_i64().is_some() => Value::Int(number.as_i64().unwrap_or_default()),
            (Kind::Bool, Some(serde_json::Value::Bool(flag))) => Value::Bool(*flag),
            (_, Some(_)) => return Err(format!("field `{}` of `{name}` has the wrong type", field.name)),
            (_, None) if field.required => return Err(format!("missing required field `{}`", field.name)),
            (Kind::Text, None) => match field.default {
                Some(default) => Value::string(default),
                None => return Err(format!("field `{}` of `{name}` must be given when a later field is", field.name)),
            },
            (_, None) => return Err(format!("field `{}` of `{name}` must be given when a later field is", field.name)),
        };
        args.push(value);
    }
    Ok(ParsedTool::TurnTool { op, args })
}

#[cfg(test)]
pub(super) fn names() -> Vec<&'static str> {
    SPECS.iter().map(|(name, _)| *name).collect()
}
