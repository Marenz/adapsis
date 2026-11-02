use super::StoreError;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub(crate) const MAX_BATCH_BYTES: usize = 128 * 1024;
pub(super) const MAX_KEYS: usize = 32;
pub(super) const MAX_PUTS: usize = 16;
pub(crate) const MAX_KEY_BYTES: usize = 128;
pub(super) const MAX_FIELDS: usize = 32;
pub(super) const MAX_FIELD_BYTES: usize = 64;
pub(super) const MAX_VALUE_BYTES: usize = 4096;
pub(super) const MAX_ROW_BYTES: usize = 16 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Field {
    pub name: String,
    pub value: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Expected {
    Absent {},
    Revision { revision: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Condition {
    pub key: String,
    pub expected: Expected,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Put {
    pub key: String,
    pub fields: Vec<Field>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Batch {
    pub version: u32,
    pub conditions: Vec<Condition>,
    pub puts: Vec<Put>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadRequest {
    version: u32,
    keys: Vec<String>,
}

pub(super) fn token(value: &str) -> bool {
    !value.is_empty() && value.len() <= MAX_KEY_BYTES
}

pub(super) fn read_keys(text: &str) -> Result<Vec<String>, StoreError> {
    if text.len() > MAX_BATCH_BYTES {
        return Err(StoreError::InvalidRequest);
    }
    let request: ReadRequest =
        serde_json::from_str(text).map_err(|_| StoreError::InvalidRequest)?;
    if request.version != 1 || request.keys.is_empty() || request.keys.len() > MAX_KEYS {
        return Err(StoreError::InvalidRequest);
    }
    unique_keys(request.keys.iter().map(String::as_str))?;
    Ok(request.keys)
}

fn unique_keys<'a>(keys: impl Iterator<Item = &'a str>) -> Result<(), StoreError> {
    let mut seen = BTreeSet::new();
    for key in keys {
        if !token(key) || !seen.insert(key) {
            return Err(StoreError::InvalidRequest);
        }
    }
    Ok(())
}

pub(super) fn fields_valid(fields: &[Field]) -> bool {
    let mut seen = BTreeSet::new();
    !fields.is_empty()
        && fields.len() <= MAX_FIELDS
        && fields.iter().all(|field| {
            !field.name.is_empty()
                && field.name.len() <= MAX_FIELD_BYTES
                && field.value.len() <= MAX_VALUE_BYTES
                && seen.insert(&field.name)
        })
        && serde_json::to_vec(fields).is_ok_and(|v| v.len() <= MAX_ROW_BYTES)
}

pub(super) fn revision(value: &str) -> Option<i64> {
    value
        .parse::<i64>()
        .ok()
        .filter(|v| *v > 0 && v.to_string() == value)
}

impl Batch {
    pub(super) fn parse(text: &str) -> Result<Self, StoreError> {
        if text.len() > MAX_BATCH_BYTES {
            return Err(StoreError::InvalidRequest);
        }
        let mut batch: Self = serde_json::from_str(text).map_err(|_| StoreError::InvalidRequest)?;
        if batch.version != 1
            || batch.conditions.is_empty()
            || batch.conditions.len() > MAX_KEYS
            || batch.puts.is_empty()
            || batch.puts.len() > MAX_PUTS
        {
            return Err(StoreError::InvalidRequest);
        }
        unique_keys(batch.conditions.iter().map(|c| c.key.as_str()))?;
        unique_keys(batch.puts.iter().map(|p| p.key.as_str()))?;
        for condition in &batch.conditions {
            if let Expected::Revision { revision: value } = &condition.expected {
                if revision(value).is_none() {
                    return Err(StoreError::InvalidRequest);
                }
            }
        }
        for put in &mut batch.puts {
            if !batch.conditions.iter().any(|c| c.key == put.key) || !fields_valid(&put.fields) {
                return Err(StoreError::InvalidRequest);
            }
            put.fields.sort_by(|a, b| a.name.cmp(&b.name));
        }
        batch.conditions.sort_by(|a, b| a.key.cmp(&b.key));
        batch.puts.sort_by(|a, b| a.key.cmp(&b.key));
        Ok(batch)
    }

    pub(super) fn hash(&self) -> Result<String, StoreError> {
        let canonical = serde_json::to_vec(self).map_err(|_| StoreError::InvalidRequest)?;
        Ok(format!("{:x}", Sha256::digest(canonical)))
    }
}

pub(super) fn identity(parts: &[&str]) -> String {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update((part.len() as u64).to_be_bytes());
        hash.update(part.as_bytes());
    }
    format!("{:x}", hash.finalize())
}
