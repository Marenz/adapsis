use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, io::Read, path::Path};

pub(super) const MAX_CONFIG_BYTES: usize = 64 * 1024;
pub(super) const MAX_ACTIVE_GRANTS: usize = 32;
pub(super) const MAX_HISTORY: usize = 256;

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Grant {
    pub id: String,
    pub generation: i64,
    pub namespace: String,
    pub principal: String,
    pub context: String,
    pub permission_group: String,
    pub rights: Vec<Right>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Right {
    Read,
    Write,
}

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GrantConfig {
    pub version: u32,
    pub grants: Vec<Grant>,
}

fn exact(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.trim() == value
        && !value.chars().any(|c| c.is_control() || c == '*')
}

impl GrantConfig {
    pub(crate) fn parse(text: &str) -> Result<Self, &'static str> {
        if text.len() > MAX_CONFIG_BYTES {
            return Err("record grant config exceeds byte limit");
        }
        let config: Self = toml::from_str(text).map_err(|_| "invalid record grant config")?;
        config.normalize()
    }

    pub(super) fn normalize(mut self) -> Result<Self, &'static str> {
        let config = &mut self;
        if config.version != 1 || config.grants.len() > MAX_ACTIVE_GRANTS {
            return Err("unsupported record grant version or count");
        }
        let mut ids = BTreeSet::new();
        let mut turns = BTreeSet::new();
        for grant in &mut config.grants {
            if ![
                &grant.id,
                &grant.namespace,
                &grant.principal,
                &grant.context,
                &grant.permission_group,
            ]
            .iter()
            .all(|s| exact(s))
                || matches!(grant.context.as_str(), "main" | "default")
                || grant.generation <= 0
                || !ids.insert(grant.id.clone())
                || !turns.insert((grant.principal.clone(), grant.context.clone()))
            {
                return Err("invalid or ambiguous exact record grant");
            }
            grant.rights.sort();
            if grant.rights != [Right::Read] && grant.rights != [Right::Read, Right::Write] {
                return Err("record rights require read, optionally write, without duplicates");
            }
        }
        config.grants.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(self)
    }

    pub(crate) fn read(path: &Path) -> Result<Self, &'static str> {
        let file = std::fs::File::open(path).map_err(|_| "record grant config unavailable")?;
        if !file
            .metadata()
            .map_err(|_| "record grant config metadata unavailable")?
            .is_file()
        {
            return Err("record grant config must be a regular file");
        }
        let mut bytes = Vec::new();
        file.take((MAX_CONFIG_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|_| "record grant config unreadable")?;
        let text = std::str::from_utf8(&bytes).map_err(|_| "record grant config must be UTF-8")?;
        Self::parse(text)
    }
}

impl std::fmt::Debug for Grant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RecordGrant(<redacted>)")
    }
}
impl std::fmt::Debug for GrantConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RecordGrantConfig(<redacted>)")
    }
}

impl Grant {
    pub(super) fn same_identity(&self, other: &Self) -> bool {
        self.namespace == other.namespace
            && self.principal == other.principal
            && self.context == other.context
    }
    pub(crate) fn writable(&self) -> bool {
        self.rights.contains(&Right::Write)
    }
}
