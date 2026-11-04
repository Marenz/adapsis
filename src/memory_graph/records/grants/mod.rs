//! Operator control plane only. Registry generations are not task lifecycle state.
mod config;
pub(crate) mod provision;
#[cfg(test)]
mod tests;
use super::StoreError;
use crate::memory_graph::{execute, string_value};
pub(crate) use config::{Grant, GrantConfig};
use lbug::Connection;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Entry {
    grant: Grant,
    enabled: bool,
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct Registry {
    version: u32,
    digest: Option<String>,
    entries: Vec<Entry>,
}

impl Registry {
    pub(super) fn empty() -> Self {
        Self {
            version: 1,
            digest: None,
            entries: Vec::new(),
        }
    }

    pub(super) fn apply(&self, config: &GrantConfig) -> Result<Self, StoreError> {
        let config = config
            .clone()
            .normalize()
            .map_err(|_| StoreError::InvalidRequest)?;
        let mut next = self.clone();
        for entry in &mut next.entries {
            entry.enabled = false;
        }
        for grant in &config.grants {
            if let Some(index) = self.entries.iter().position(|e| e.grant.id == grant.id) {
                let old = &self.entries[index];
                if !old.grant.same_identity(grant)
                    || grant.generation < old.grant.generation
                    || (grant.generation == old.grant.generation
                        && (!old.enabled || grant != &old.grant))
                {
                    return Err(StoreError::Denied);
                }
                next.entries[index] = Entry {
                    grant: grant.clone(),
                    enabled: true,
                };
            } else {
                if next.entries.len() >= config::MAX_HISTORY {
                    return Err(StoreError::Denied);
                }
                next.entries.push(Entry {
                    grant: grant.clone(),
                    enabled: true,
                });
            }
        }
        next.entries.sort_by(|a, b| a.grant.id.cmp(&b.grant.id));
        next.digest = Some(digest(&config)?);
        Ok(next)
    }

    pub(super) fn revoke_all(&self) -> Self {
        let mut revoked = self.clone();
        for entry in &mut revoked.entries {
            entry.enabled = false;
        }
        revoked.digest = None;
        revoked
    }

    pub(super) fn matching(&self, principal: &str, context: &str) -> Option<&Grant> {
        self.entries
            .iter()
            .find(|e| e.enabled && e.grant.principal == principal && e.grant.context == context)
            .map(|e| &e.grant)
    }

    pub(super) fn contains(&self, binding: &super::RecordBinding, write: bool) -> bool {
        self.matching(&binding.principal, &binding.context)
            .is_some_and(|grant| {
                grant.id == binding.grant_id
                    && grant.namespace == binding.namespace
                    && Some(grant.generation) == binding.generation
                    && (!write || grant.writable())
            })
    }
}

pub(super) fn load(connection: &Connection<'_>) -> Result<Registry, StoreError> {
    let mut rows = connection
        .query("MATCH (r:RecordGrantRegistry {id:'registry'}) RETURN r.version,r.payload")
        .map_err(|_| StoreError::Unavailable)?;
    let row = rows.next().ok_or(StoreError::Unavailable)?;
    if row.len() != 2
        || rows.next().is_some()
        || crate::memory_graph::int64_value(&row[0]).ok() != Some(1)
    {
        return Err(StoreError::Unavailable);
    }
    let text = string_value(&row[1]).map_err(|_| StoreError::Unavailable)?;
    if text.len() > 512 * 1024 {
        return Err(StoreError::Unavailable);
    }
    let registry: Registry = serde_json::from_str(&text).map_err(|_| StoreError::Unavailable)?;
    if registry.version != 1 || registry.entries.len() > config::MAX_HISTORY {
        return Err(StoreError::Unavailable);
    }
    let mut ids = std::collections::BTreeSet::new();
    let mut turns = std::collections::BTreeSet::new();
    for entry in &registry.entries {
        let validated = GrantConfig {
            version: 1,
            grants: vec![entry.grant.clone()],
        }
        .normalize()
        .map_err(|_| StoreError::Unavailable)?;
        if validated.grants[0] != entry.grant
            || !ids.insert(&entry.grant.id)
            || (entry.enabled && !turns.insert((&entry.grant.principal, &entry.grant.context)))
        {
            return Err(StoreError::Unavailable);
        }
    }
    let active = GrantConfig {
        version: 1,
        grants: registry
            .entries
            .iter()
            .filter(|entry| entry.enabled)
            .map(|entry| entry.grant.clone())
            .collect(),
    }
    .normalize()
    .map_err(|_| StoreError::Unavailable)?;
    match &registry.digest {
        Some(stored) if stored == &digest(&active)? => {}
        None if active.grants.is_empty() => {}
        _ => return Err(StoreError::Unavailable),
    }
    Ok(registry)
}

fn digest(config: &GrantConfig) -> Result<String, StoreError> {
    use sha2::Digest;
    Ok(format!(
        "{:x}",
        sha2::Sha256::digest(serde_json::to_vec(config).map_err(|_| StoreError::InvalidRequest)?)
    ))
}

pub(super) fn save(connection: &Connection<'_>, registry: &Registry) -> Result<(), StoreError> {
    execute(
        connection,
        "MERGE (r:RecordGrantRegistry {id:'registry'}) SET r.version=1,r.payload=$payload",
        vec![(
            "payload",
            serde_json::to_string(registry)
                .map_err(|_| StoreError::Unavailable)?
                .into(),
        )],
    )
    .map_err(|_| StoreError::Unavailable)
}

/// Caller holds the shared DB lock. Full content is read/validated, never mtime-only.
pub(super) fn refresh(
    connection: &Connection<'_>,
    path: &std::path::Path,
) -> Result<Registry, StoreError> {
    let current = load(connection)?;
    let next = GrantConfig::read(path)
        .ok()
        .and_then(|config| current.apply(&config).ok());
    let desired = next.clone().unwrap_or_else(|| current.revoke_all());
    if current != desired {
        connection
            .query("BEGIN TRANSACTION")
            .map_err(|_| StoreError::Unavailable)?;
        if let Err(error) = save(connection, &desired) {
            let _ = connection.query("ROLLBACK");
            return Err(error);
        }
        connection
            .query("COMMIT")
            .map_err(|_| StoreError::OutcomeUnknown)?;
        connection
            .query("CHECKPOINT")
            .map_err(|_| StoreError::OutcomeUnknown)?;
    }
    next.ok_or(StoreError::Denied)
}
