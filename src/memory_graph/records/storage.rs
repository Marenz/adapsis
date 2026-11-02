use super::{Field, Receipt, Record, RecordBinding, StoreError, envelope, identity};
use crate::memory_graph::{execute, int64_value, string_value};
use lbug::{Connection, Value};

#[cfg(test)]
pub(super) fn initialize(connection: &Connection<'_>) -> Result<(), StoreError> {
    connection.query(
        "CREATE NODE TABLE IF NOT EXISTS ScopedRecord(id STRING, namespace STRING, record_key STRING, revision INT64, fields STRING, PRIMARY KEY(id));
         CREATE NODE TABLE IF NOT EXISTS ScopedRecordReceipt(id STRING, namespace STRING, principal STRING, context STRING, grant_id STRING, batch_hash STRING, receipt STRING, PRIMARY KEY(id));"
    ).map_err(|_| StoreError::Unavailable)?;
    Ok(())
}

pub(super) fn read_row(
    connection: &Connection<'_>,
    namespace: &str,
    key: &str,
) -> Result<Option<Record>, StoreError> {
    let id = identity(&[namespace, key]);
    let mut statement = connection.prepare(
        "MATCH (r:ScopedRecord {id: $id}) RETURN r.namespace, r.record_key, r.revision, r.fields"
    ).map_err(|_| StoreError::Unavailable)?;
    let mut rows = connection
        .execute(&mut statement, vec![("id", id.into())])
        .map_err(|_| StoreError::Unavailable)?;
    let Some(row) = rows.next() else {
        return Ok(None);
    };
    let mut decode = || -> anyhow::Result<Record> {
        anyhow::ensure!(row.len() == 4 && rows.next().is_none(), "record shape");
        anyhow::ensure!(
            string_value(&row[0])? == namespace && string_value(&row[1])? == key,
            "record scope"
        );
        let revision = int64_value(&row[2])?;
        anyhow::ensure!(revision > 0, "record revision");
        let encoded = string_value(&row[3])?;
        anyhow::ensure!(encoded.len() <= envelope::MAX_ROW_BYTES, "record size");
        let fields: Vec<Field> = serde_json::from_str(&encoded)?;
        anyhow::ensure!(envelope::fields_valid(&fields), "record fields");
        Ok(Record {
            key: key.into(),
            revision: revision.to_string(),
            fields,
        })
    };
    decode().map(Some).map_err(|_| StoreError::Unavailable)
}

pub(super) fn put_row(
    connection: &Connection<'_>,
    namespace: &str,
    put: &envelope::Put,
    revision: i64,
) -> Result<(), StoreError> {
    execute(connection,
        "MERGE (r:ScopedRecord {id: $id}) SET r.namespace = $namespace, r.record_key = $key, r.revision = $revision, r.fields = $fields",
        vec![
            ("id", identity(&[namespace, &put.key]).into()), ("namespace", namespace.into()),
            ("key", put.key.clone().into()), ("revision", Value::Int64(revision)),
            ("fields", serde_json::to_string(&put.fields).map_err(|_| StoreError::InvalidRequest)?.into()),
        ],
    ).map_err(|_| StoreError::Unavailable)
}

fn receipt_id(binding: &RecordBinding, request_id: &str) -> String {
    identity(&[&binding.namespace, &binding.principal, request_id])
}

pub(super) fn read_receipt(
    connection: &Connection<'_>,
    binding: &RecordBinding,
    request_id: &str,
) -> Result<Option<(String, Receipt)>, StoreError> {
    let mut statement = connection.prepare(
        "MATCH (r:ScopedRecordReceipt {id: $id}) RETURN r.namespace, r.principal, r.context, r.grant_id, r.batch_hash, r.receipt"
    ).map_err(|_| StoreError::Unavailable)?;
    let mut rows = connection
        .execute(
            &mut statement,
            vec![("id", receipt_id(binding, request_id).into())],
        )
        .map_err(|_| StoreError::Unavailable)?;
    let Some(row) = rows.next() else {
        return Ok(None);
    };
    if row.len() != 6 || rows.next().is_some() {
        return Err(StoreError::Unavailable);
    }
    for (index, expected) in [
        &binding.namespace,
        &binding.principal,
        &binding.context,
        &binding.grant_id,
    ]
    .iter()
    .enumerate()
    {
        if string_value(&row[index]).map_err(|_| StoreError::Unavailable)? != **expected {
            return Err(StoreError::Denied);
        }
    }
    let hash = string_value(&row[4]).map_err(|_| StoreError::Unavailable)?;
    let encoded = string_value(&row[5]).map_err(|_| StoreError::Unavailable)?;
    if encoded.len() > envelope::MAX_BATCH_BYTES {
        return Err(StoreError::Unavailable);
    }
    let receipt: Receipt = serde_json::from_str(&encoded).map_err(|_| StoreError::Unavailable)?;
    if receipt.version != 1
        || receipt.request_id != request_id
        || receipt.revisions.len() > envelope::MAX_PUTS
    {
        return Err(StoreError::Unavailable);
    }
    Ok(Some((hash, receipt)))
}

pub(super) fn put_receipt(
    connection: &Connection<'_>,
    binding: &RecordBinding,
    hash: &str,
    receipt: &Receipt,
) -> Result<(), StoreError> {
    execute(connection,
        "CREATE (:ScopedRecordReceipt {id: $id, namespace: $namespace, principal: $principal, context: $context, grant_id: $grant_id, batch_hash: $hash, receipt: $receipt})",
        vec![
            ("id", receipt_id(binding, &receipt.request_id).into()), ("namespace", binding.namespace.clone().into()),
            ("principal", binding.principal.clone().into()), ("context", binding.context.clone().into()),
            ("grant_id", binding.grant_id.clone().into()), ("hash", hash.into()),
            ("receipt", serde_json::to_string(receipt).map_err(|_| StoreError::Unavailable)?.into()),
        ],
    ).map_err(|_| StoreError::Unavailable)
}
