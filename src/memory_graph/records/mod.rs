//! Dormant generic record/CAS backend. Only tests can initialize or grant access.
//! No task policy, caller-selected namespace, IO builtin or startup activation.
#![allow(dead_code)]

mod envelope;
mod storage;
#[cfg(test)]
mod tests;

use super::MemoryGraph;
use crate::coroutine::TurnIdentity;
pub(crate) use envelope::Field;
use envelope::{Batch, Expected, identity};
use lbug::Connection;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::{Arc, MutexGuard, TryLockError};
use std::time::{Duration, Instant};

const LOCK_WAIT: Duration = Duration::from_secs(5);
const TX_BUDGET: Duration = Duration::from_secs(5);
const QUERY_TIMEOUT_MS: u64 = 2000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StoreError {
    Denied,
    InvalidRequest,
    Conflict,
    IdempotencyMismatch,
    Unavailable,
    OutcomeUnknown,
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Denied => "record_access_denied",
            Self::InvalidRequest => "invalid_record_request",
            Self::Conflict => "record_version_conflict",
            Self::IdempotencyMismatch => "record_request_id_mismatch",
            Self::Unavailable => "record_store_unavailable",
            Self::OutcomeUnknown => "record_commit_outcome_unknown",
        })
    }
}

impl std::error::Error for StoreError {}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Record {
    pub key: String,
    pub revision: String,
    pub fields: Vec<Field>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Revision {
    pub key: String,
    pub revision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Receipt {
    pub version: u32,
    pub request_id: String,
    pub revisions: Vec<Revision>,
}

/// Private fields and no production constructor: not a serializable bearer token.
#[derive(Clone)]
pub(crate) struct RecordBinding {
    store: Arc<()>,
    namespace: String,
    principal: String,
    context: String,
    grant_id: String,
    write: bool,
}

#[derive(Clone)]
pub(crate) struct RecordStore {
    graph: MemoryGraph,
    identity: Arc<()>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault {
    None,
    #[cfg(test)]
    BeforeCommit,
    #[cfg(test)]
    AfterCommit,
}

impl RecordStore {
    fn authorize<'a>(
        &self,
        binding: Option<&'a RecordBinding>,
        turn: Option<&TurnIdentity>,
        write: bool,
    ) -> Result<&'a RecordBinding, StoreError> {
        let binding = binding.ok_or(StoreError::Denied)?;
        let turn = turn.ok_or(StoreError::Denied)?;
        if !Arc::ptr_eq(&self.identity, &binding.store)
            || turn.guest_tools.is_some()
            || binding.principal != turn.principal
            || binding.context != turn.context
            || (write && (!binding.write || !turn.may_write))
        {
            return Err(StoreError::Denied);
        }
        Ok(binding)
    }

    fn lock(&self) -> Result<MutexGuard<'_, ()>, StoreError> {
        let start = Instant::now();
        loop {
            match self.graph.write_lock.try_lock() {
                Ok(guard) => return Ok(guard),
                Err(TryLockError::Poisoned(_)) => return Err(StoreError::Unavailable),
                Err(TryLockError::WouldBlock) if start.elapsed() < LOCK_WAIT => {
                    std::thread::sleep(Duration::from_millis(2));
                }
                Err(_) => return Err(StoreError::Unavailable),
            }
        }
    }

    fn connection(&self) -> Result<Connection<'_>, StoreError> {
        let connection =
            Connection::new(&self.graph.database).map_err(|_| StoreError::Unavailable)?;
        connection.set_query_timeout(QUERY_TIMEOUT_MS);
        Ok(connection)
    }

    pub(crate) fn read(
        &self,
        binding: Option<&RecordBinding>,
        turn: Option<&TurnIdentity>,
        request: &str,
    ) -> Result<Vec<Option<Record>>, StoreError> {
        let binding = self.authorize(binding, turn, false)?;
        let keys = envelope::read_keys(request)?;
        let _guard = self.lock()?;
        let connection = self.connection()?;
        let start = Instant::now();
        connection
            .query("BEGIN TRANSACTION READ ONLY")
            .map_err(|_| StoreError::Unavailable)?;
        let result = keys
            .iter()
            .map(|key| {
                deadline(start)?;
                storage::read_row(&connection, &binding.namespace, key)
            })
            .collect();
        finish_read(&connection, result)
    }

    pub(crate) fn receipt(
        &self,
        binding: Option<&RecordBinding>,
        turn: Option<&TurnIdentity>,
        request_id: &str,
    ) -> Result<Option<Receipt>, StoreError> {
        let binding = self.authorize(binding, turn, true)?;
        if !envelope::token(request_id) {
            return Err(StoreError::InvalidRequest);
        }
        let _guard = self.lock()?;
        let connection = self.connection()?;
        storage::read_receipt(&connection, binding, request_id)
            .map(|receipt| receipt.map(|(_, receipt)| receipt))
    }

    pub(crate) fn commit(
        &self,
        binding: Option<&RecordBinding>,
        turn: Option<&TurnIdentity>,
        request_id: &str,
        request: &str,
    ) -> Result<Receipt, StoreError> {
        self.commit_inner(binding, turn, request_id, request, Fault::None)
    }

    fn commit_inner(
        &self,
        binding: Option<&RecordBinding>,
        turn: Option<&TurnIdentity>,
        request_id: &str,
        request: &str,
        fault: Fault,
    ) -> Result<Receipt, StoreError> {
        let binding = self.authorize(binding, turn, true)?;
        if !envelope::token(request_id) {
            return Err(StoreError::InvalidRequest);
        }
        let batch = Batch::parse(request)?;
        let hash = batch.hash()?; // Validation/canonicalization before taking DB lock.
        let _guard = self.lock()?;
        let connection = self.connection()?;
        let start = Instant::now();
        connection
            .query("BEGIN TRANSACTION")
            .map_err(|_| StoreError::Unavailable)?;
        let result = apply_batch(&connection, binding, request_id, &batch, &hash, start);
        #[cfg(test)]
        let result = if fault == Fault::BeforeCommit && result.is_ok() {
            Err(StoreError::Unavailable)
        } else {
            result
        };
        let receipt = match result {
            Ok(receipt) => receipt,
            Err(error) => {
                connection
                    .query("ROLLBACK")
                    .map_err(|_| StoreError::OutcomeUnknown)?;
                return Err(error);
            }
        };
        // COMMIT or post-COMMIT failure is ambiguous. Never substitute a new ID.
        connection
            .query("COMMIT")
            .map_err(|_| StoreError::OutcomeUnknown)?;
        #[cfg(test)]
        if fault == Fault::AfterCommit {
            return Err(StoreError::OutcomeUnknown);
        }
        let _ = fault;
        connection
            .query("CHECKPOINT")
            .map_err(|_| StoreError::OutcomeUnknown)?;
        Ok(receipt)
    }
}

fn deadline(start: Instant) -> Result<(), StoreError> {
    if start.elapsed() >= TX_BUDGET {
        Err(StoreError::Unavailable)
    } else {
        Ok(())
    }
}

fn finish_read<T>(
    connection: &Connection<'_>,
    result: Result<T, StoreError>,
) -> Result<T, StoreError> {
    match result {
        Ok(value) => {
            connection
                .query("COMMIT")
                .map_err(|_| StoreError::Unavailable)?;
            Ok(value)
        }
        Err(error) => {
            let _ = connection.query("ROLLBACK");
            Err(error)
        }
    }
}

fn apply_batch(
    connection: &Connection<'_>,
    binding: &RecordBinding,
    request_id: &str,
    batch: &Batch,
    hash: &str,
    start: Instant,
) -> Result<Receipt, StoreError> {
    if let Some((stored_hash, receipt)) = storage::read_receipt(connection, binding, request_id)? {
        return if stored_hash == hash {
            Ok(receipt)
        } else {
            Err(StoreError::IdempotencyMismatch)
        };
    }
    let mut revisions = BTreeMap::new();
    for condition in &batch.conditions {
        deadline(start)?;
        let row = storage::read_row(connection, &binding.namespace, &condition.key)?;
        let revision = match (&condition.expected, row) {
            (Expected::Absent {}, None) => 0,
            (Expected::Revision { revision }, Some(row)) if revision == &row.revision => {
                envelope::revision(revision).ok_or(StoreError::Unavailable)?
            }
            _ => return Err(StoreError::Conflict),
        };
        revisions.insert(condition.key.as_str(), revision);
    }
    let mut receipt = Receipt {
        version: 1,
        request_id: request_id.into(),
        revisions: Vec::new(),
    };
    for put in &batch.puts {
        deadline(start)?;
        let revision = revisions[put.key.as_str()]
            .checked_add(1)
            .ok_or(StoreError::Unavailable)?;
        storage::put_row(connection, &binding.namespace, put, revision)?;
        receipt.revisions.push(Revision {
            key: put.key.clone(),
            revision: revision.to_string(),
        });
    }
    deadline(start)?;
    storage::put_receipt(connection, binding, hash, &receipt)?;
    Ok(receipt)
}

#[cfg(test)]
impl RecordStore {
    fn initialize_for_test(graph: MemoryGraph) -> Result<Self, StoreError> {
        let store = Self {
            graph,
            identity: Arc::new(()),
        };
        {
            let _guard = store.lock()?;
            let connection = store.connection()?;
            storage::initialize(&connection)?;
        }
        Ok(store)
    }

    fn grant_for_test(
        &self,
        namespace: &str,
        turn: &TurnIdentity,
        grant_id: &str,
        write: bool,
    ) -> RecordBinding {
        assert!(
            [namespace, &turn.principal, &turn.context, grant_id]
                .iter()
                .all(|s| envelope::token(s))
        );
        RecordBinding {
            store: self.identity.clone(),
            namespace: namespace.into(),
            principal: turn.principal.clone(),
            context: turn.context.clone(),
            grant_id: grant_id.into(),
            write,
        }
    }
}
