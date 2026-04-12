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
        let process_max = process_level.max_permission();

        // Special case: execute-only process level
        if process_level == AccessLevel::ExecuteOnly {
            return PermissionLevel::Execute;
        }

        // Special case: user-only process level blocks core group writes
        let group = self.group_for_module(module_name);
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
}
