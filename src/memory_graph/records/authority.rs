//! Record-only permission ceiling is not identity authentication.
//! Production ingress attestation issuance is deliberately absent.
use crate::{
    coroutine::TurnIdentity,
    permissions::{AccessLevel, PermissionConfig, PermissionLevel},
};

#[derive(Clone)]
pub(crate) struct RecordCeiling {
    principal: String,
    context: String,
    permissions: PermissionConfig,
    process: AccessLevel,
    profiles: Vec<String>,
}

impl RecordCeiling {
    pub(crate) fn capture(
        turn: &TurnIdentity,
        permissions: &PermissionConfig,
        process: AccessLevel,
        actual_model: Option<&str>,
        context_profile: Option<&str>,
    ) -> Option<Self> {
        let actual = actual_model.filter(|s| !s.trim().is_empty())?;
        if turn.principal.is_empty() || turn.context.is_empty() || turn.guest_tools.is_some() {
            return None;
        }
        let mut profiles = vec![actual.to_string()];
        if let Some(profile) = context_profile {
            profiles.push(profile.to_string());
        }
        if let Some(profile) = permissions.principals.get(&turn.principal) {
            profiles.push(profile.clone());
        }
        if profiles.iter().any(|profile| {
            profile.trim().is_empty() || permissions.guest_profiles.contains_key(profile)
        }) {
            return None;
        }
        Some(Self {
            principal: turn.principal.clone(),
            context: turn.context.clone(),
            permissions: permissions.clone(),
            process,
            profiles,
        })
    }

    fn level(&self, turn: &TurnIdentity, group: &str) -> PermissionLevel {
        if self.principal != turn.principal
            || self.context != turn.context
            || turn.guest_tools.is_some()
            || !self.permissions.groups.contains_key(group)
        {
            return PermissionLevel::None;
        }
        self.profiles
            .iter()
            .map(|profile| self.permissions.resolve_group(self.process, profile, group))
            .min()
            .unwrap_or(PermissionLevel::None)
    }
}

/// Cannot be decoded, built by Adapsis, or issued from TurnIdentity metadata.
#[derive(Clone)]
pub(crate) struct TrustedCaller {
    pinned: RecordCeiling,
}

impl TrustedCaller {
    #[cfg(test)]
    pub(crate) fn issue_for_test(pinned: RecordCeiling) -> Self {
        Self { pinned }
    }

    pub(crate) fn level(
        &self,
        turn: &TurnIdentity,
        current: &RecordCeiling,
        group: &str,
    ) -> PermissionLevel {
        self.pinned
            .level(turn, group)
            .min(current.level(turn, group))
    }
}
