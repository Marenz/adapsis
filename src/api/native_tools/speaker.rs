//! Speaker classes of the native protocol (DESIGN L3, owner decision of 2026-11-10).
//!
//! Every native tool list is derived from the class of the pinned turn snapshot:
//!
//! * [`SpeakerClass::Writer`]: may change code. The code tools, and the workspace tools
//!   (`workspace.rs`) when the node has a workspace registry.
//! * [`SpeakerClass::ExecuteOnly`]: `inspect`, `evaluate`, restricted `run`, `finish`,
//!   and the turn-bound memory tools when the turn has an identity.
//! * [`SpeakerClass::Guest`]: one direct tool per profile entry, restricted `run`, `finish`.
//!
//! A turn that cannot be classified with certainty is a guest with an EMPTY tool list,
//! never execute-only or writer.
use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::api) enum SpeakerClass {
    Writer,
    ExecuteOnly,
    Guest(Vec<String>),
}

impl SpeakerClass {
    /// Guests and execute-only speakers: the author of a `run` block is untrusted.
    pub(in crate::api) fn restricted(&self) -> bool {
        !matches!(self, Self::Writer)
    }

    pub(super) fn limits(&self) -> crate::run_limits::RunLimits {
        if self.restricted() {
            crate::run_limits::RunLimits::restricted()
        } else {
            crate::run_limits::RunLimits::owner()
        }
    }

    pub(super) fn guest_tools(&self) -> &[String] {
        match self {
            Self::Guest(tools) => tools,
            _ => &[],
        }
    }
}

const MOST_RESTRICTED: SpeakerClass = SpeakerClass::Guest(Vec::new());

/// Classify the pinned snapshot. Reads only `config` (the caller's per-turn pin): the
/// intersected permission model and the turn identity.
///
/// The two guest signals are the profile entry the snapshot carries for the pinned model
/// and the tool list of the turn identity (which the IO boundary enforces). They must
/// agree. If only one is present the turn is not classifiable and gets no tools; if both
/// are present the tools are their intersection, limited to the known guest tools.
pub(in crate::api) fn speaker_class(config: &AppConfig) -> SpeakerClass {
    let model = model_name(config);
    let profile = config.permission_config.guest_profiles.get(&model);
    let turn = config.turn.as_ref().and_then(|turn| turn.guest_tools.as_ref());
    match (profile, turn) {
        (Some(profile), Some(turn)) => SpeakerClass::Guest(
            crate::coroutine::scoped::GUEST_TOOLS
                .iter()
                .filter(|tool| profile.iter().any(|t| t == *tool) && turn.iter().any(|t| t == *tool))
                .map(|tool| tool.to_string())
                .collect(),
        ),
        (None, None) => {
            let authority = config.permission_config.authority(config.access_level, &model);
            let turn_writes = config.turn.as_ref().is_none_or(|turn| turn.may_write);
            if authority.may_write && turn_writes {
                SpeakerClass::Writer
            } else if authority.may_execute {
                SpeakerClass::ExecuteOnly
            } else {
                MOST_RESTRICTED
            }
        }
        _ => MOST_RESTRICTED,
    }
}

const WRITER_ONLY: &[&str] = &["mutate", "test"];
const EXECUTE_ONLY_TOOLS: &[&str] = &["inspect", "evaluate", "run", "finish"];
const MEMORY_TOOLS: &[&str] = &["memory_inspect", "memory_remember", "memory_correct", "memory_forget"];

pub(super) fn is_memory_tool(name: &str) -> bool {
    MEMORY_TOOLS.contains(&name)
}

/// Whether `name` is a tool this class may call. The refusal keeps the wording the
/// write-only gate always had, and names the tool.
pub(super) fn admit_tool(class: &SpeakerClass, name: &str) -> Result<(), String> {
    let known = EXECUTE_ONLY_TOOLS.contains(&name)
        || WRITER_ONLY.contains(&name)
        || is_memory_tool(name)
        || crate::coroutine::scoped::GUEST_TOOLS.contains(&name);
    match class {
        // Unknown names are refused by the parser, as before. Workspace tool names are
        // deliberately not in `known`: for the other classes they are names that do not exist.
        SpeakerClass::Writer => Ok(()),
        SpeakerClass::ExecuteOnly if EXECUTE_ONLY_TOOLS.contains(&name) || is_memory_tool(name) => Ok(()),
        SpeakerClass::Guest(tools) if matches!(name, "run" | "finish") || tools.iter().any(|tool| tool == name) => Ok(()),
        _ if !known => Err(format!("unknown native tool `{name}`")),
        SpeakerClass::ExecuteOnly => Err(format!(
            "native tool dispatch denied: not write-capable (`{name}` is not available to this speaker)"
        )),
        SpeakerClass::Guest(_) => Err(format!(
            "native tool dispatch denied: guest turns are not eligible for `{name}`"
        )),
    }
}

/// The tool schemas of one class, in a stable order.
pub(super) fn definitions(config: &AppConfig, all: Vec<ToolDefinition>) -> Vec<ToolDefinition> {
    let class = speaker_class(config);
    match &class {
        // Repository workspaces: this class only, and only where `workspace::available`
        // says so (owner-private conversation, node with a registry; #129).
        SpeakerClass::Writer if workspace::available(config) => all.into_iter().chain(workspace::definitions()).collect(),
        SpeakerClass::Writer => all,
        SpeakerClass::ExecuteOnly => all
            .into_iter()
            .filter(|tool| admit_tool(&class, &tool.name).is_ok())
            .map(|tool| if tool.name == "run" { run::restricted_definition(&class) } else { tool })
            .collect(),
        SpeakerClass::Guest(tools) => {
            let mut memory: Vec<ToolDefinition> = all.into_iter().filter(|tool| is_memory_tool(&tool.name)).collect();
            let mut out = Vec::new();
            for name in tools {
                if let Some(index) = memory.iter().position(|tool| &tool.name == name) {
                    out.push(memory.remove(index));
                } else if let Some(tool) = guest_tools::definition(name) {
                    out.push(tool);
                }
            }
            out.push(run::restricted_definition(&class));
            out.push(ToolDefinition {
                name: "finish".into(),
                description: "Signal that this turn is complete.".into(),
                parameters: serde_json::json!({"type":"object","properties":{},"required":[],"additionalProperties":false}),
            });
            out
        }
    }
}
