use super::{GrantConfig, Registry, load, save};
use crate::memory_graph::records::{StoreError, storage};
use lbug::{Connection, Database, SystemConfig};
use serde::Serialize;
use std::path::Path;

#[derive(Serialize)]
pub(crate) struct Report {
    dry_run: bool,
    migration_required: bool,
    active_grants: usize,
}

fn has_registry(connection: &Connection<'_>) -> Result<bool, StoreError> {
    let tables = connection
        .query("CALL show_tables() RETURN name")
        .map_err(|_| StoreError::Unavailable)?;
    for row in tables {
        if row
            .first()
            .and_then(|v| crate::memory_graph::string_value(v).ok())
            .as_deref()
            == Some("RecordGrantRegistry")
        {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(crate) fn apply(
    database: &Path,
    config: &GrantConfig,
    dry_run: bool,
) -> Result<Report, StoreError> {
    // Required explicit EXISTING database: neither dry-run nor apply creates a DB.
    if !database.is_file() {
        return Err(StoreError::Unavailable);
    }
    let database = Database::new(
        database,
        SystemConfig::default()
            .read_only(dry_run)
            .enable_checksums(true)
            .throw_on_wal_replay_failure(true),
    )
    .map_err(|_| StoreError::Unavailable)?;
    let connection = Connection::new(&database).map_err(|_| StoreError::Unavailable)?;
    let exists = has_registry(&connection)?;
    let current = if exists {
        load(&connection)?
    } else {
        Registry::empty()
    };
    let desired = current.apply(config)?; // Validate ALL changes before any schema/data write.
    if !dry_run {
        connection
            .query("BEGIN TRANSACTION")
            .map_err(|_| StoreError::Unavailable)?;
        let result = (|| {
            storage::initialize(&connection)?;
            connection.query("CREATE NODE TABLE IF NOT EXISTS RecordGrantRegistry(id STRING, version INT64, payload STRING, PRIMARY KEY(id))").map_err(|_| StoreError::Unavailable)?;
            save(&connection, &desired)
        })();
        if let Err(error) = result {
            connection
                .query("ROLLBACK")
                .map_err(|_| StoreError::OutcomeUnknown)?;
            return Err(error);
        }
        connection
            .query("COMMIT")
            .map_err(|_| StoreError::OutcomeUnknown)?;
        connection
            .query("CHECKPOINT")
            .map_err(|_| StoreError::OutcomeUnknown)?;
    }
    Ok(Report {
        dry_run,
        migration_required: !exists,
        active_grants: config.grants.len(),
    })
}
