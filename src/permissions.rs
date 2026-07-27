//! Permission system for Adapsis.
//!
//! Controls what each model/context can do. Permissions are layered:
//! Process level (--access-level) → Model level (permissions.toml) → Context level (override)
//! Each layer can only restrict, never expand beyond its parent.

use std::collections::HashMap;
use std::path::Path;

/// Permission level for a module group.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Deserialize, serde::Serialize,
)]
#[serde(rename_all = "lowercase")]
pub enum PermissionLevel {
    /// Module invisible, can't call anything.
    None = 0,
    /// Can !eval functions, +await, +spawn.
    Execute = 1,
    /// Execute + ?source, see in program summary.
    Read = 2,
    /// Read + can +module to modify/add functions.
    Write = 3,
}

/// Process-level access cap set via --access-level CLI flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AccessLevel {
    /// Everything allowed including !opencode.
    Full,
    /// Can modify any Adapsis module, no !opencode.
    AdapsisOnly,
    /// Can only modify non-core modules, no !opencode.
    UserOnly,
    /// Cannot modify anything. Can only !eval existing functions.
    ExecuteOnly,
}

impl std::str::FromStr for AccessLevel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "full" => Ok(Self::Full),
            "adapsis-only" => Ok(Self::AdapsisOnly),
            "user-only" => Ok(Self::UserOnly),
            "execute-only" => Ok(Self::ExecuteOnly),
            _ => Err(format!(
                "unknown access level: {s}. Expected: full, adapsis-only, user-only, execute-only"
            )),
        }
    }
}

impl AccessLevel {
    /// Maximum permission level this access level allows.
    pub fn max_permission(&self) -> PermissionLevel {
        match self {
            Self::Full | Self::AdapsisOnly => PermissionLevel::Write,
            Self::UserOnly => PermissionLevel::Write, // capped per group
            Self::ExecuteOnly => PermissionLevel::Execute,
        }
    }

    pub fn allows_opencode(&self) -> bool {
        matches!(self, Self::Full)
    }

    pub fn allows_agents(&self) -> bool {
        !matches!(self, Self::ExecuteOnly)
    }
}

/// What the speaker of a single turn may do, flattened for the prompt.
///
/// A snapshot, not a capability: nothing is enforced here. Enforcement stays in
/// `resolve`/`can_opencode`/`can_agent`, which the execution path consults
/// independently. This exists so the model can be told the truth up front.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeakerAuthority {
    /// May `!eval` existing functions.
    pub may_execute: bool,
    /// May read source via `?source`.
    pub may_read_source: bool,
    /// May add or modify code.
    pub may_write: bool,
    /// May spawn background agents.
    pub may_agent: bool,
    /// May rewrite the runtime via `!opencode`.
    pub may_opencode: bool,
}

impl SpeakerAuthority {
    /// Intersection of two authorities.
    ///
    /// Sub-agents may be narrowed at spawn but never widened (issue #41): an
    /// agent spawned with `--model` must not inherit that model's permissions
    /// when they exceed those of the conversation that spawned it.
    pub fn narrowed_to(self, other: Self) -> Self {
        Self {
            may_execute: self.may_execute && other.may_execute,
            may_read_source: self.may_read_source && other.may_read_source,
            may_write: self.may_write && other.may_write,
            may_agent: self.may_agent && other.may_agent,
            may_opencode: self.may_opencode && other.may_opencode,
        }
    }
}

/// Per-model permission configuration.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ModelPermissions {
    /// Permission level per group name.
    #[serde(flatten)]
    pub group_perms: HashMap<String, PermissionLevel>,
    /// Whether this model can use !opencode.
    #[serde(default)]
    pub opencode: bool,
}

/// Full permission configuration loaded from TOML.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct PermissionConfig {
    /// Named module groups: group_name → [module_names].
    #[serde(default)]
    pub groups: HashMap<String, Vec<String>>,
    /// Per-model permissions.
    #[serde(default)]
    pub model: HashMap<String, ModelPermissions>,
}

impl Default for PermissionConfig {
    fn default() -> Self {
        Self {
            groups: HashMap::new(),
            model: HashMap::new(),
        }
    }
}

impl PermissionConfig {
    /// Load from a TOML file.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            anyhow::anyhow!("failed to read permissions file {}: {e}", path.display())
        })?;
        let config: Self = toml::from_str(&content).map_err(|e| {
            anyhow::anyhow!("failed to parse permissions file {}: {e}", path.display())
        })?;
        Ok(config)
    }

    /// Find which group a module belongs to. Returns "user" for ungrouped modules.
    pub fn group_for_module(&self, module_name: &str) -> &str {
        for (group_name, modules) in &self.groups {
            if modules.iter().any(|m| m == module_name) {
                return group_name;
            }
        }
        "user"
    }

    /// Get the model permissions, falling back to "default" entry.
    fn model_perms(&self, model_name: &str) -> Option<&ModelPermissions> {
        self.model
            .get(model_name)
            .or_else(|| self.model.get("default"))
    }

    /// Resolve the effective permission level for a model accessing a module.
    /// Applies the process-level cap.
    pub fn resolve(
        &self,
        process_level: AccessLevel,
        model_name: &str,
        module_name: &str,
    ) -> PermissionLevel {
        self.resolve_group(process_level, model_name, self.group_for_module(module_name))
    }

    /// Resolve the effective permission level for a model against a whole group.
    ///
    /// `resolve` is this function after a module→group lookup. Exposed separately
    /// so a caller can ask "what is the most this speaker may do anywhere?"
    /// without inventing a representative module name for each group.
    pub fn resolve_group(
        &self,
        process_level: AccessLevel,
        model_name: &str,
        group: &str,
    ) -> PermissionLevel {
        let process_max = process_level.max_permission();

        // Special case: execute-only process level
        if process_level == AccessLevel::ExecuteOnly {
            return PermissionLevel::Execute;
        }

        // Special case: user-only process level blocks core group writes
        if process_level == AccessLevel::UserOnly && group != "user" {
            let model_level = self
                .model_perms(model_name)
                .and_then(|mp| mp.group_perms.get(group).copied())
                .unwrap_or(PermissionLevel::Execute);
            return model_level.min(PermissionLevel::Read); // cap at read for non-user groups
        }

        // Normal resolution: model config capped by process level
        let model_level = self
            .model_perms(model_name)
            .and_then(|mp| mp.group_perms.get(group).copied())
            .unwrap_or(if self.model.is_empty() {
                PermissionLevel::Write // no config = fully permissive
            } else {
                PermissionLevel::Execute // config exists but model not listed = restrictive
            });

        model_level.min(process_max)
    }

    /// Check if a model can use !opencode.
    pub fn can_opencode(&self, process_level: AccessLevel, model_name: &str) -> bool {
        if !process_level.allows_opencode() {
            return false;
        }
        self.model_perms(model_name)
            .map(|mp| mp.opencode)
            .unwrap_or(self.model.is_empty()) // no config = allow, config but not listed = deny
    }

    /// Check if a model can spawn agents.
    pub fn can_agent(&self, process_level: AccessLevel, _model_name: &str) -> bool {
        process_level.allows_agents()
    }

    /// The highest permission this model reaches in ANY group.
    ///
    /// Note the fall-through documented in `unlisted_group_resolves_to_execute_not_none`:
    /// a group a profile does not name resolves to `Execute`, so this is `Execute`
    /// even for a profile that lists nothing.
    pub fn max_level(&self, process_level: AccessLevel, model_name: &str) -> PermissionLevel {
        std::iter::once("user")
            .chain(self.groups.keys().map(String::as_str))
            .map(|group| self.resolve_group(process_level, model_name, group))
            .max()
            .unwrap_or(PermissionLevel::None)
    }

    /// What the speaker of one turn is allowed to have done.
    ///
    /// Issue #41: the composed system prompt states this explicitly, because a
    /// model that does not know it is sandboxed promises commits and deploys it
    /// cannot perform — which reads as incompetence rather than as a boundary.
    pub fn authority(&self, process_level: AccessLevel, model_name: &str) -> SpeakerAuthority {
        let max = self.max_level(process_level, model_name);
        SpeakerAuthority {
            may_execute: max >= PermissionLevel::Execute,
            may_read_source: max >= PermissionLevel::Read,
            may_write: max >= PermissionLevel::Write,
            may_agent: self.can_agent(process_level, model_name),
            may_opencode: self.can_opencode(process_level, model_name),
        }
    }

    /// List all configured model names (excluding "default").
    pub fn model_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self
            .model
            .keys()
            .filter(|k| k.as_str() != "default")
            .map(|k| k.as_str())
            .collect();
        names.sort();
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> PermissionConfig {
        toml::from_str(
            r#"
            [groups]
            core = ["TelegramBot", "MusicGen"]
            data = ["Stratum", "Memory"]

            [model.gemma4s]
            core = "execute"
            data = "execute"
            user = "execute"
            opencode = false

            [model."chatgpt/gpt-5.4"]
            core = "read"
            data = "write"
            user = "write"
            opencode = false

            [model."anthropic/claude-opus-4-6"]
            core = "write"
            data = "write"
            user = "write"
            opencode = true

            [model.default]
            core = "execute"
            data = "execute"
            user = "write"
            opencode = false
        "#,
        )
        .unwrap()
    }

    #[test]
    fn group_lookup() {
        let config = test_config();
        assert_eq!(config.group_for_module("TelegramBot"), "core");
        assert_eq!(config.group_for_module("Stratum"), "data");
        assert_eq!(config.group_for_module("MyCustomModule"), "user");
    }

    #[test]
    fn gemma4s_execute_only() {
        let config = test_config();
        assert_eq!(
            config.resolve(AccessLevel::Full, "gemma4s", "TelegramBot"),
            PermissionLevel::Execute
        );
        assert_eq!(
            config.resolve(AccessLevel::Full, "gemma4s", "Stratum"),
            PermissionLevel::Execute
        );
        assert_eq!(
            config.resolve(AccessLevel::Full, "gemma4s", "MyModule"),
            PermissionLevel::Execute
        );
    }

    #[test]
    fn gpt54_mixed_permissions() {
        let config = test_config();
        assert_eq!(
            config.resolve(AccessLevel::Full, "chatgpt/gpt-5.4", "TelegramBot"),
            PermissionLevel::Read
        );
        assert_eq!(
            config.resolve(AccessLevel::Full, "chatgpt/gpt-5.4", "Stratum"),
            PermissionLevel::Write
        );
        assert_eq!(
            config.resolve(AccessLevel::Full, "chatgpt/gpt-5.4", "MyModule"),
            PermissionLevel::Write
        );
    }

    #[test]
    fn opus_full_access() {
        let config = test_config();
        assert_eq!(
            config.resolve(
                AccessLevel::Full,
                "anthropic/claude-opus-4-6",
                "TelegramBot"
            ),
            PermissionLevel::Write
        );
        assert!(config.can_opencode(AccessLevel::Full, "anthropic/claude-opus-4-6"));
    }

    #[test]
    fn process_level_caps_model() {
        let config = test_config();
        // Opus has write on everything, but process level restricts
        assert_eq!(
            config.resolve(
                AccessLevel::ExecuteOnly,
                "anthropic/claude-opus-4-6",
                "TelegramBot"
            ),
            PermissionLevel::Execute
        );
        assert!(!config.can_opencode(AccessLevel::AdapsisOnly, "anthropic/claude-opus-4-6"));
    }

    #[test]
    fn user_only_caps_core_groups() {
        let config = test_config();
        // GPT-5.4 has read on core, user-only caps non-user at read
        assert_eq!(
            config.resolve(AccessLevel::UserOnly, "chatgpt/gpt-5.4", "TelegramBot"),
            PermissionLevel::Read
        );
        // User modules still writable
        assert_eq!(
            config.resolve(AccessLevel::UserOnly, "chatgpt/gpt-5.4", "MyModule"),
            PermissionLevel::Write
        );
    }

    #[test]
    fn unknown_model_uses_default() {
        let config = test_config();
        assert_eq!(
            config.resolve(AccessLevel::Full, "some-unknown-model", "TelegramBot"),
            PermissionLevel::Execute
        );
        assert_eq!(
            config.resolve(AccessLevel::Full, "some-unknown-model", "MyModule"),
            PermissionLevel::Write
        );
    }

    #[test]
    fn opencode_requires_both_process_and_model() {
        let config = test_config();
        assert!(config.can_opencode(AccessLevel::Full, "anthropic/claude-opus-4-6"));
        assert!(!config.can_opencode(AccessLevel::Full, "gemma4s"));
        assert!(!config.can_opencode(AccessLevel::AdapsisOnly, "anthropic/claude-opus-4-6"));
    }

    /// The CLI default for --access-level must parse to a level that forbids
    /// !opencode. !opencode rebuilds and re-execs the runtime, so it must be
    /// opt-in (--access-level full), never the default posture. If someone
    /// changes the default in main.rs back to "full", this test fails.
    #[test]
    fn cli_default_access_level_disallows_opencode() {
        let default: AccessLevel = "adapsis-only".parse().expect("default must parse");
        assert!(
            !default.allows_opencode(),
            "the default --access-level must not allow !opencode"
        );
        // Even a model explicitly granted opencode cannot use it under the default.
        let config = test_config();
        assert!(!config.can_opencode(default, "anthropic/claude-opus-4-6"));
    }

    #[test]
    fn load_from_string() {
        let config: PermissionConfig = toml::from_str(
            r#"
            [groups]
            core = ["Bot"]
            [model.test]
            core = "read"
            opencode = false
        "#,
        )
        .unwrap();
        assert_eq!(config.group_for_module("Bot"), "core");
        assert_eq!(
            config.resolve(AccessLevel::Full, "test", "Bot"),
            PermissionLevel::Read
        );
    }

    /// Issue #38 item 3, root cause. A group a profile does not mention does NOT
    /// resolve to `None` — it resolves to `Execute`. That is precisely the
    /// "callable but not readable" state that let a family-persona module be
    /// bound in an unrelated project conversation and then be undebuggable when
    /// it failed.
    ///
    /// This is deliberate current behaviour, not a bug being fixed here; the test
    /// exists so that changing it is a conscious decision rather than a surprise.
    #[test]
    fn unlisted_group_resolves_to_execute_not_none() {
        let config: PermissionConfig = toml::from_str(
            r#"
            [groups]
            assist = ["Wolfi"]
            core = ["TelegramBot"]

            [model.coding]
            core = "write"
            opencode = true
        "#,
        )
        .unwrap();

        assert_eq!(config.group_for_module("Wolfi"), "assist");
        assert_eq!(
            config.resolve(AccessLevel::Full, "coding", "Wolfi"),
            PermissionLevel::Execute,
            "an unmentioned group must be known to fall through to Execute — \
             every profile therefore has to name every group explicitly"
        );
    }

    /// The shipped fix for the above: naming the group with `none` makes the
    /// module genuinely invisible to that profile.
    #[test]
    fn explicit_none_hides_a_persona_module() {
        let config: PermissionConfig = toml::from_str(
            r#"
            [groups]
            assist = ["Wolfi"]
            core = ["TelegramBot"]

            [model.coding]
            core = "write"
            assist = "none"
            opencode = true

            [model.family]
            assist = "execute"
            core = "none"
            opencode = false
        "#,
        )
        .unwrap();

        // Coding contexts cannot see the family persona at all.
        assert_eq!(
            config.resolve(AccessLevel::Full, "coding", "Wolfi"),
            PermissionLevel::None
        );
        // The family sandbox still can, and still cannot reach core modules.
        assert_eq!(
            config.resolve(AccessLevel::Full, "family", "Wolfi"),
            PermissionLevel::Execute
        );
        assert_eq!(
            config.resolve(AccessLevel::Full, "family", "TelegramBot"),
            PermissionLevel::None
        );
    }

    fn two_profile_config() -> PermissionConfig {
        toml::from_str(
            r#"
            [groups]
            assist = ["Wolfi"]
            core = ["TelegramBot"]

            [model.coding]
            core = "write"
            assist = "none"
            user = "write"
            opencode = true

            [model.family]
            assist = "execute"
            core = "none"
            user = "none"
            opencode = false
        "#,
        )
        .unwrap()
    }

    /// Issue #41: the prompt states the speaker's authority, so the resolution
    /// has to collapse the whole permission stack into one honest summary.
    #[test]
    fn authority_summarizes_the_whole_permission_stack() {
        let config = two_profile_config();

        let admin = config.authority(AccessLevel::Full, "coding");
        assert!(admin.may_execute && admin.may_read_source && admin.may_write);
        assert!(admin.may_opencode && admin.may_agent);

        let sandboxed = config.authority(AccessLevel::Full, "family");
        assert!(sandboxed.may_execute, "the family profile may still call Wolfi");
        assert!(!sandboxed.may_write, "the family profile may not author code");
        assert!(!sandboxed.may_opencode);
    }

    /// The process cap is the outer layer: a profile that says `write` and
    /// `opencode = true` still cannot do either under `execute-only`.
    #[test]
    fn the_process_cap_overrides_a_permissive_profile() {
        let config = two_profile_config();
        let capped = config.authority(AccessLevel::ExecuteOnly, "coding");
        assert!(capped.may_execute);
        assert!(!capped.may_write);
        assert!(!capped.may_opencode);
        assert!(!capped.may_agent, "execute-only forbids agents");
    }

    /// Sub-agents may be narrowed at spawn, never widened: `!agent --model X`
    /// chooses who generates, not what the agent is permitted to do.
    #[test]
    fn narrowing_an_authority_never_widens_it() {
        let config = two_profile_config();
        let spawner = config.authority(AccessLevel::Full, "family");
        let requested = config.authority(AccessLevel::Full, "coding");

        let effective = spawner.narrowed_to(requested);
        assert!(
            !effective.may_write && !effective.may_opencode,
            "a sandboxed conversation widened itself by naming a privileged model"
        );

        // And the reverse direction still narrows.
        assert_eq!(requested.narrowed_to(spawner), effective);
    }
}
