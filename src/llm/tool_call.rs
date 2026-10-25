//! Native tool-call wire types (issue #78).
//!
//! These are the public vocabulary the native transport (`llm::native`) and the
//! integration layer share. `ToolCall` serializes to the OpenAI nested
//! `function` shape so messages can round-trip the wire verbatim; the in-memory
//! representation stays flat for the dispatcher.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A single tool call, either authored by the model or passed back as a result.
///
/// On the wire this is the OpenAI shape:
/// `{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

impl Serialize for ToolCall {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;

        #[derive(Serialize)]
        struct Function<'a> {
            name: &'a str,
            arguments: &'a str,
        }

        let mut state = serializer.serialize_struct("ToolCall", 3)?;
        state.serialize_field("id", &self.id)?;
        state.serialize_field("type", "function")?;
        state.serialize_field(
            "function",
            &Function {
                name: &self.name,
                arguments: &self.arguments,
            },
        )?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for ToolCall {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Wire {
            id: String,
            function: WireFunction,
        }

        #[derive(Deserialize)]
        struct WireFunction {
            name: String,
            #[serde(default)]
            arguments: String,
        }

        let wire = Wire::deserialize(deserializer)?;
        Ok(ToolCall {
            id: wire.id,
            name: wire.function.name,
            arguments: wire.function.arguments,
        })
    }
}

/// A tool the client advertises to the model for a native turn.
///
/// `parameters` is a JSON Schema object describing the tool's arguments. The
/// wire nests it under `{"type": "function", "function": {...}}` (built in
/// `native::native_request_body`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Terminal reason a native turn ended with.
///
/// Any other finish reason (or a missing one) is a transport error: the native
/// layer refuses to hand back a result it cannot certify.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeFinishReason {
    Stop,
    ToolCalls,
}
