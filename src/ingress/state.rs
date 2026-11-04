use super::*;
use crate::{
    coroutine::TurnIdentity,
    memory_graph::records::{Field, Record, RecordBinding, RecordStore, StoreError},
};
use serde_json::json;
use std::collections::BTreeMap;
use telegram_poll::Event;

#[derive(Clone)]
pub(super) struct IntakeStore {
    store: RecordStore,
    binding: RecordBinding,
    caller: TurnIdentity,
    instance: String,
    #[cfg(test)]
    admission_gate: Option<Arc<std::sync::Barrier>>,
}
pub(super) struct Pending {
    pub events: Vec<Event>,
    pub unknown: usize,
}

fn failure(error: StoreError) -> IngressError {
    match error {
        StoreError::OutcomeUnknown => IngressError::StateUnknown,
        StoreError::Conflict => IngressError::Conflict,
        _ => IngressError::StateUnavailable,
    }
}
fn fields(row: &Record) -> BTreeMap<String, String> {
    row.fields
        .iter()
        .map(|f| (f.name.clone(), f.value.clone()))
        .collect()
}
fn number(values: &BTreeMap<String, String>, key: &str) -> Result<i64, IngressError> {
    let text = values.get(key).ok_or(IngressError::StateUnavailable)?;
    let value = text
        .parse::<i64>()
        .map_err(|_| IngressError::StateUnavailable)?;
    if value < 0 || value.to_string() != *text {
        return Err(IngressError::StateUnavailable);
    }
    Ok(value)
}
fn row(key: String, values: BTreeMap<String, String>) -> Record {
    Record {
        key,
        revision: String::new(),
        fields: values
            .into_iter()
            .map(|(name, value)| Field { name, value })
            .collect(),
    }
}
fn event_key(id: i64) -> String {
    format!("event:{id}")
}
fn chunk_key(id: i64, index: usize) -> String {
    format!("event:{id}:chunk:{index}")
}

fn root_shape(values: &BTreeMap<String, String>) -> Result<(), IngressError> {
    let count = number(values, "count")?;
    let epoch = number(values, "epoch")?;
    if values.get("schema").map(String::as_str) != Some("ingress_v1")
        || count > EVENT_LIMIT as i64
        || number(values, "bytes")? > SOURCE_BYTES as i64
        || values.len() != 8 + count as usize
        || epoch < 1
        || epoch > INCARNATION_LIMIT
        || number(values, "cursor")? < number(values, "seed")?
        || values
            .get("owner")
            .and_then(|owner| uuid::Uuid::parse_str(owner).ok())
            .is_none()
        || values
            .get("identity")
            .is_none_or(|identity| identity.len() != 64)
    {
        return Err(IngressError::StateUnavailable);
    }
    let mut previous = None;
    for index in 0..count {
        let id = number(values, &format!("id{index}"))?;
        if id >= number(values, "cursor")? || previous.is_some_and(|p| p >= id) {
            return Err(IngressError::StateUnavailable);
        }
        previous = Some(id);
    }
    Ok(())
}

impl IntakeStore {
    #[cfg(test)]
    pub(super) fn fixture(store: RecordStore, namespace: &str, source: &str) -> Self {
        let caller = TurnIdentity {
            principal: format!("service:ingress:{source}"),
            context: format!("source:{source}"),
            may_write: true,
            guest_tools: None,
        };
        let binding = store.grant_for_test(namespace, &caller, "intake-only", true);
        Self {
            store,
            binding,
            caller,
            instance: uuid::Uuid::new_v4().to_string(),
            admission_gate: None,
        }
    }

    fn read(&self, keys: &[String]) -> Result<Vec<Option<Record>>, IngressError> {
        self.store
            .read(
                Some(&self.binding),
                Some(&self.caller),
                &json!({"version":1,"keys":keys}).to_string(),
            )
            .map_err(failure)
    }
    fn one(&self, key: &str) -> Result<Option<Record>, IngressError> {
        Ok(self.read(&[key.to_string()])?.remove(0))
    }

    fn commit(
        &self,
        id: &str,
        checks: &[(String, Option<String>)],
        puts: &[Record],
    ) -> Result<(), IngressError> {
        let conditions=checks.iter().map(|(key,revision)|json!({"key":key,"expected":match revision {
            Some(revision)=>json!({"kind":"revision","revision":revision}),None=>json!({"kind":"absent"})
        }})).collect::<Vec<_>>();
        let puts = puts
            .iter()
            .map(|row| json!({"key":row.key,"fields":row.fields}))
            .collect::<Vec<_>>();
        self.store
            .commit(
                Some(&self.binding),
                Some(&self.caller),
                id,
                &json!({"version":1,"conditions":conditions,"puts":puts}).to_string(),
            )
            .map_err(failure)?;
        Ok(())
    }

    fn root(&self, epoch: i64) -> Result<(Record, BTreeMap<String, String>), IngressError> {
        let root = self.one("source")?.ok_or(IngressError::StateUnavailable)?;
        let values = fields(&root);
        root_shape(&values)?;
        if number(&values, "epoch")? != epoch || values.get("owner") != Some(&self.instance) {
            return Err(IngressError::StaleSource);
        }
        Ok((root, values))
    }

    pub(super) fn open(&self, identity: &str, seed: i64) -> Result<i64, IngressError> {
        let previous = self.one("source")?;
        let mut values = if let Some(root) = &previous {
            let values = fields(root);
            root_shape(&values)?;
            values
        } else {
            BTreeMap::from([
                ("schema".into(), "ingress_v1".into()),
                ("identity".into(), identity.into()),
                ("seed".into(), seed.to_string()),
                ("cursor".into(), seed.to_string()),
                ("count".into(), "0".into()),
                ("bytes".into(), "0".into()),
                ("epoch".into(), "0".into()),
            ])
        };
        if values.get("identity").map(String::as_str) != Some(identity)
            || number(&values, "seed")? != seed
        {
            return Err(IngressError::Mismatch);
        }
        let epoch = number(&values, "epoch")?
            .checked_add(1)
            .ok_or(IngressError::Capacity)?;
        if epoch > INCARNATION_LIMIT {
            return Err(IngressError::Capacity);
        }
        values.insert("epoch".into(), epoch.to_string());
        values.insert("owner".into(), self.instance.clone());
        self.commit(
            &format!("boot:{epoch}:{}", self.instance),
            &[("source".into(), previous.map(|r| r.revision))],
            &[row("source".into(), values)],
        )?;
        self.recover_started(epoch)?;
        Ok(epoch)
    }

    pub(super) fn cursor(&self, epoch: i64) -> Result<i64, IngressError> {
        let (_, root) = self.root(epoch)?;
        if number(&root, "count")? >= EVENT_LIMIT as i64
            || number(&root, "bytes")? >= SOURCE_BYTES as i64
        {
            return Err(IngressError::Capacity);
        }
        number(&root, "cursor")
    }

    pub(super) fn accept(&self, epoch: i64, event: &Event) -> Result<(), IngressError> {
        if event.raw.len() > EVENT_BYTES
            || event.id < 0
            || event.id == i64::MAX
            || telegram_poll::digest(&event.raw) != event.digest
        {
            return Err(IngressError::Protocol);
        }
        let (root, mut values) = self.root(epoch)?;
        let key = event_key(event.id);
        if let Some(existing) = self.one(&key)? {
            return if fields(&existing).get("digest") == Some(&event.digest) {
                Ok(())
            } else {
                Err(IngressError::AlteredReplay)
            };
        }
        if event.id < number(&values, "cursor")? {
            return Err(IngressError::AlteredReplay);
        }
        let count = number(&values, "count")?;
        let size = number(&values, "bytes")? + event.raw.len() as i64;
        if count >= EVENT_LIMIT as i64 || size > SOURCE_BYTES as i64 {
            return Err(IngressError::Capacity);
        }
        let chunks = payload_chunks(event.id, &event.raw);
        let metadata = row(
            key.clone(),
            BTreeMap::from([
                ("id".into(), event.id.to_string()),
                ("digest".into(), event.digest.clone()),
                ("status".into(), "accepted".into()),
                ("epoch".into(), epoch.to_string()),
                ("chunks".into(), chunks.len().to_string()),
            ]),
        );
        values.insert(format!("id{count}"), event.id.to_string());
        values.insert("count".into(), (count + 1).to_string());
        values.insert("bytes".into(), size.to_string());
        values.insert("cursor".into(), (event.id + 1).to_string());
        let mut checks = vec![("source".into(), Some(root.revision)), (key, None)];
        checks.extend(chunks.iter().map(|r| (r.key.clone(), None)));
        let mut puts = vec![row("source".into(), values), metadata];
        puts.extend(chunks);
        self.commit(&format!("accept:{}", event.id), &checks, &puts)
    }

    fn event_rows(&self, epoch: i64) -> Result<(Record, Vec<Record>), IngressError> {
        let (root, values) = self.root(epoch)?;
        let count = number(&values, "count")?;
        if count == 0 {
            return Ok((root, Vec::new()));
        }
        let keys = (0..count)
            .map(|i| number(&values, &format!("id{i}")).map(event_key))
            .collect::<Result<Vec<_>, _>>()?;
        let rows = self
            .read(&keys)?
            .into_iter()
            .map(|r| r.ok_or(IngressError::StateUnavailable))
            .collect::<Result<Vec<_>, _>>()?;
        Ok((root, rows))
    }

    fn recover_started(&self, epoch: i64) -> Result<(), IngressError> {
        let (root, events) = self.event_rows(epoch)?;
        for event in events {
            let mut values = fields(&event);
            if values.get("status").map(String::as_str) == Some("started") {
                if number(&values, "epoch")? >= epoch {
                    return Err(IngressError::StateUnavailable);
                }
                values.insert("status".into(), "unknown".into());
                let id = number(&values, "id")?;
                self.commit(
                    &format!("unknown:{id}"),
                    &[
                        (root.key.clone(), Some(root.revision.clone())),
                        (event.key.clone(), Some(event.revision)),
                    ],
                    &[row(event.key, values)],
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn pending(&self, epoch: i64) -> Result<Pending, IngressError> {
        let (_, events) = self.event_rows(epoch)?;
        let mut pending = Pending {
            events: Vec::new(),
            unknown: 0,
        };
        for event in events {
            let values = fields(&event);
            match values.get("status").map(String::as_str) {
                Some("accepted") if pending.events.len() < PAGE_LIMIT => {
                    pending.events.push(self.payload(&values)?)
                }
                Some("accepted") | Some("started") => {}
                Some("unknown") => pending.unknown += 1,
                _ => return Err(IngressError::StateUnavailable),
            }
        }
        Ok(pending)
    }

    fn payload(&self, values: &BTreeMap<String, String>) -> Result<Event, IngressError> {
        let id = number(values, "id")?;
        let count = number(values, "chunks")? as usize;
        if count == 0 || count > 6 {
            return Err(IngressError::StateUnavailable);
        }
        let keys = (0..count).map(|i| chunk_key(id, i)).collect::<Vec<_>>();
        let mut raw = String::new();
        for chunk in self.read(&keys)? {
            let chunk = fields(&chunk.ok_or(IngressError::StateUnavailable)?);
            if chunk.is_empty() || chunk.len() > 3 {
                return Err(IngressError::StateUnavailable);
            }
            for i in 0..chunk.len() {
                raw.push_str(
                    chunk
                        .get(&format!("part{i}"))
                        .ok_or(IngressError::StateUnavailable)?,
                );
            }
        }
        let event = telegram_poll::parse_event(&raw)?;
        if event.id != id || values.get("digest") != Some(&event.digest) {
            return Err(IngressError::StateUnavailable);
        }
        Ok(event)
    }

    pub(super) fn start(&self, epoch: i64, id: i64, digest: &str) -> Result<(), IngressError> {
        let (root, _) = self.root(epoch)?;
        let event = self
            .one(&event_key(id))?
            .ok_or(IngressError::StateUnavailable)?;
        let mut values = fields(&event);
        if values.get("digest").map(String::as_str) != Some(digest) {
            return Err(IngressError::Mismatch);
        }
        if values.get("status").map(String::as_str) != Some("accepted") {
            return Err(IngressError::AlreadyStarted);
        }
        #[cfg(test)]
        if let Some(barrier) = &self.admission_gate {
            barrier.wait();
        }
        values.insert("status".into(), "started".into());
        values.insert("epoch".into(), epoch.to_string());
        self.commit(
            &format!("start:{id}:{}", uuid::Uuid::new_v4()),
            &[
                (root.key, Some(root.revision)),
                (event.key.clone(), Some(event.revision)),
            ],
            &[row(event.key, values)],
        )
    }

    #[cfg(test)]
    pub(super) fn gate_admission(&mut self, barrier: Arc<std::sync::Barrier>) {
        self.admission_gate = Some(barrier);
    }
}

fn payload_chunks(id: i64, text: &str) -> Vec<Record> {
    let mut parts = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let mut end = (start + 2048).min(text.len());
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        parts.push(text[start..end].to_string());
        start = end;
    }
    parts
        .chunks(3)
        .enumerate()
        .map(|(index, parts)| {
            row(
                chunk_key(id, index),
                parts
                    .iter()
                    .enumerate()
                    .map(|(i, text)| (format!("part{i}"), text.clone()))
                    .collect(),
            )
        })
        .collect()
}
