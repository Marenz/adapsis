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

    /// Add another profile that may only lower the ceiling. A guest or empty
    /// profile refuses the whole ceiling rather than being ignored.
    pub(crate) fn narrowed(mut self, profile: Option<&str>) -> Option<Self> {
        if let Some(profile) = profile {
            if profile.trim().is_empty() || self.permissions.guest_profiles.contains_key(profile) {
                return None;
            }
            self.profiles.push(profile.to_string());
        }
        Some(self)
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
///
/// `live` is shared with the issuing turn and cleared when that turn ends. Every
/// binding and every backend operation checks it, so a handle that outlives its
/// turn (stashed, inherited by a delayed task) is denied even if it escapes.
#[derive(Clone)]
pub(crate) struct TrustedCaller {
    pinned: RecordCeiling,
    live: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl TrustedCaller {
    pub(crate) fn from_ingress(
        claim: &crate::ingress::ClaimedOrigin,
        pinned: RecordCeiling,
    ) -> Option<Self> {
        if pinned.principal != claim.principal() || pinned.context != claim.context() {
            return None;
        }
        Some(Self {
            pinned,
            live: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true)),
        })
    }
    #[cfg(test)]
    pub(crate) fn issue_for_test(pinned: RecordCeiling) -> Self {
        Self {
            pinned,
            live: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true)),
        }
    }

    pub(crate) fn is_live(&self) -> bool {
        self.live.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Irreversible for this attestation; a later turn needs a fresh claim.
    pub(crate) fn revoke(&self) {
        self.live.store(false, std::sync::atomic::Ordering::SeqCst);
    }

    pub(crate) fn level(
        &self,
        turn: &TurnIdentity,
        current: &RecordCeiling,
        group: &str,
    ) -> PermissionLevel {
        if !self.is_live() {
            return PermissionLevel::None;
        }
        self.pinned
            .level(turn, group)
            .min(current.level(turn, group))
    }
}
