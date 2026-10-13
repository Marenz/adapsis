//! Transactional memory correction and lifecycle transitions.
use super::*;
use anyhow::bail;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryStatus {
    Active,
    Superseded,
    Disputed,
    Forgotten,
}

impl MemoryStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Superseded => "superseded",
            Self::Disputed => "disputed",
            Self::Forgotten => "forgotten",
        }
    }

    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "active" => Ok(Self::Active),
            "superseded" => Ok(Self::Superseded),
            "disputed" => Ok(Self::Disputed),
            "forgotten" => Ok(Self::Forgotten),
            other => Err(anyhow!("unknown memory lifecycle status {other:?}")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemporalValidity {
    pub valid_from_ms: Option<i64>,
    pub valid_until_ms: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct CorrectionDraft {
    pub target_id: String,
    pub replacement: String,
    pub context_id: String,
    pub principal_id: String,
    pub may_write: bool,
    pub embedding: Vec<f32>,
    pub valid_from_ms: Option<i64>,
    pub valid_until_ms: Option<i64>,
    pub created_at_ms: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InheritedScope {
    Context,
    Global,
}

#[derive(Debug, Clone, Copy)]
struct CorrectionTarget {
    status: MemoryStatus,
    scope: InheritedScope,
}

enum CorrectionMode {
    Replace,
    Conflict(Vec<String>),
    Resolve(Vec<String>),
}

impl MemoryGraph {
    /// Create an explicit assertion that corrects an authorized memory.
    ///
    /// `None` deliberately combines absent, foreign, non-contributable, forgotten,
    /// and already-resolved historical targets so this action is not an id probe.
    pub fn correct_memory(&self, draft: &CorrectionDraft) -> Result<Option<String>> {
        self.correct_memory_impl(draft, false)
    }

    fn correct_memory_impl(
        &self,
        draft: &CorrectionDraft,
        fail_after_create: bool,
    ) -> Result<Option<String>> {
        let target_id = draft.target_id.trim();
        ensure!(
            !target_id.is_empty(),
            "memory_correct requires an old memory id"
        );
        ensure!(
            !draft.context_id.is_empty(),
            "memory_correct requires a bound context"
        );
        ensure!(
            !draft.principal_id.is_empty(),
            "memory_correct requires a bound principal"
        );
        ensure!(
            draft.embedding.len() == 384,
            "memory correction embedding must have 384 dimensions"
        );
        if let (Some(from), Some(until)) = (draft.valid_from_ms, draft.valid_until_ms) {
            ensure!(
                from <= until,
                "memory_correct valid_from_ms ({from}) must be <= valid_until_ms ({until})"
            );
        }
        let replacement = draft.replacement.trim();
        ensure!(
            !replacement.is_empty(),
            "memory_correct replacement cannot be empty"
        );
        let normalized_replacement = normalize_replacement(replacement);

        let _guard = self
            .write_lock
            .lock()
            .map_err(|_| anyhow!("memory correction lock is poisoned"))?;
        let connection = Connection::new(&self.database)?;
        connection.query("BEGIN TRANSACTION")?;
        let result = (|| {
            let Some(target) = authorized_target(&connection, draft, target_id)? else {
                return Ok(None);
            };
            let correction_id = correction_id(
                draft,
                target_id,
                &normalized_replacement,
                target.scope,
            );
            if correction_exists(&connection, &correction_id, target_id)? {
                return Ok(Some(correction_id.clone()));
            }
            let mode = correction_mode(&connection, draft, target_id, target)?;
            let Some(mode) = mode else {
                return Ok(None);
            };
            create_correction_node(&connection, &correction_id, target_id, replacement, draft)?;
            if fail_after_create {
                bail!("injected correction failure after replacement creation");
            }
            apply_correction_mode(&connection, &correction_id, target_id, mode)?;
            Ok(Some(correction_id))
        })();
        finish_transaction(&connection, result)
    }

    #[cfg(test)]
    fn correct_memory_failing_after_create(
        &self,
        draft: &CorrectionDraft,
    ) -> Result<Option<String>> {
        self.correct_memory_impl(draft, true)
    }
}

fn normalize_replacement(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn hash_part(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value);
}

fn correction_id(
    draft: &CorrectionDraft,
    target_id: &str,
    replacement: &str,
    scope: InheritedScope,
) -> String {
    let mut hasher = Sha256::new();
    let scope_key = match scope {
        InheritedScope::Context => draft.context_id.as_bytes(),
        InheritedScope::Global => GLOBAL_GROUP_ID.as_bytes(),
    };
    for value in [
        draft.principal_id.as_bytes(),
        scope_key,
        target_id.as_bytes(),
        replacement.as_bytes(),
    ] {
        hash_part(&mut hasher, value);
    }
    for value in [draft.valid_from_ms, draft.valid_until_ms] {
        match value {
            Some(value) => {
                hasher.update([1]);
                hasher.update(value.to_be_bytes());
            }
            None => hasher.update([0]),
        }
    }
    format!("memory:correction:{:x}", hasher.finalize())
}

fn authorized_target(
    connection: &Connection<'_>,
    draft: &CorrectionDraft,
    target_id: &str,
) -> Result<Option<CorrectionTarget>> {
    let mut statement = connection.prepare(
        "MATCH (principal:Principal {id: $principal_id})-[membership:MEMBER_OF]->(access_group:AccessGroup)<-[:GOVERNED_BY]-(memory:Memory {id: $memory_id})
         WHERE membership.can_read = true AND membership.can_contribute = true
           AND NOT (memory)-[:DENIED_TO]->(principal)
         WITH memory, count(DISTINCT access_group) AS accessible_groups
         MATCH (memory)-[:GOVERNED_BY]->(required_group:AccessGroup)
         WITH memory, accessible_groups, count(DISTINCT required_group) AS required_groups,
              collect(DISTINCT required_group.id) AS group_ids
         WHERE accessible_groups = required_groups AND required_groups = 1
           AND (list_contains(group_ids, $group_id) OR list_contains(group_ids, $global_group_id))
         RETURN memory.status, list_contains(group_ids, $global_group_id)",
    )?;
    let Some(row) = connection
        .execute(
            &mut statement,
            vec![
                ("principal_id", draft.principal_id.clone().into()),
                ("memory_id", target_id.into()),
                ("group_id", context_group_id(&draft.context_id).into()),
                ("global_group_id", GLOBAL_GROUP_ID.into()),
            ],
        )?
        .next()
    else {
        return Ok(None);
    };
    let status = MemoryStatus::parse(&string_value(&row[0])?)?;
    if status == MemoryStatus::Forgotten {
        return Ok(None);
    }
    let scope = if bool_value(&row[1])? {
        if !draft.may_write {
            return Ok(None);
        }
        InheritedScope::Global
    } else {
        InheritedScope::Context
    };
    Ok(Some(CorrectionTarget { status, scope }))
}

fn correction_exists(
    connection: &Connection<'_>,
    correction_id: &str,
    target_id: &str,
) -> Result<bool> {
    let mut statement = connection.prepare(
        "MATCH (correction:Memory {id: $correction_id})-[:SUPERSEDES]->(target:Memory {id: $target_id}) RETURN count(correction)",
    )?;
    let count = connection
        .execute(
            &mut statement,
            vec![
                ("correction_id", correction_id.into()),
                ("target_id", target_id.into()),
            ],
        )?
        .next()
        .and_then(|row| row.first().and_then(|value| int64_value(value).ok()))
        .unwrap_or(0);
    Ok(count > 0)
}

fn correction_mode(
    connection: &Connection<'_>,
    draft: &CorrectionDraft,
    target_id: &str,
    target: CorrectionTarget,
) -> Result<Option<CorrectionMode>> {
    match target.status {
        MemoryStatus::Active => {
            let successors =
                active_or_disputed_successors(connection, draft, target_id, target.scope)?;
            if successors.is_empty() {
                Ok(Some(CorrectionMode::Replace))
            } else {
                Ok(Some(CorrectionMode::Conflict(successors)))
            }
        }
        MemoryStatus::Superseded => {
            let successors =
                active_or_disputed_successors(connection, draft, target_id, target.scope)?;
            Ok((!successors.is_empty()).then_some(CorrectionMode::Conflict(successors)))
        }
        MemoryStatus::Disputed => {
            let cluster = authorized_dispute_cluster(connection, draft, target_id, target.scope)?;
            Ok(Some(CorrectionMode::Resolve(cluster)))
        }
        MemoryStatus::Forgotten => Ok(None),
    }
}

fn active_or_disputed_successors(
    connection: &Connection<'_>,
    draft: &CorrectionDraft,
    target_id: &str,
    inherited_scope: InheritedScope,
) -> Result<Vec<String>> {
    let mut statement = connection.prepare(
        "MATCH (successor:Memory)-[:SUPERSEDES]->(target:Memory {id: $target_id})
         WHERE successor.status = 'active' OR successor.status = 'disputed'
         RETURN successor.id ORDER BY successor.id",
    )?;
    let ids = connection
        .execute(&mut statement, vec![("target_id", target_id.into())])?
        .map(|row| string_value(&row[0]))
        .collect::<Result<Vec<_>>>()?;
    filter_authorized_related(connection, draft, ids, inherited_scope)
}

fn authorized_dispute_cluster(
    connection: &Connection<'_>,
    draft: &CorrectionDraft,
    target_id: &str,
    inherited_scope: InheritedScope,
) -> Result<Vec<String>> {
    let mut ids = BTreeSet::from([target_id.to_string()]);
    for query in [
        "MATCH (target:Memory {id: $target_id})-[:CONTRADICTS]->(related:Memory) RETURN related.id",
        "MATCH (related:Memory)-[:CONTRADICTS]->(target:Memory {id: $target_id}) RETURN related.id",
    ] {
        let mut statement = connection.prepare(query)?;
        for row in connection.execute(&mut statement, vec![("target_id", target_id.into())])? {
            ids.insert(string_value(&row[0])?);
        }
    }
    filter_authorized_related(
        connection,
        draft,
        ids.into_iter().collect(),
        inherited_scope,
    )
}

fn filter_authorized_related(
    connection: &Connection<'_>,
    draft: &CorrectionDraft,
    ids: Vec<String>,
    inherited_scope: InheritedScope,
) -> Result<Vec<String>> {
    let mut result = Vec::new();
    for id in ids {
        if authorized_target(connection, draft, &id)?
            .is_some_and(|target| target.scope == inherited_scope)
        {
            result.push(id);
        }
    }
    result.sort();
    result.dedup();
    Ok(result)
}

fn create_correction_node(
    connection: &Connection<'_>,
    correction_id: &str,
    target_id: &str,
    replacement: &str,
    draft: &CorrectionDraft,
) -> Result<()> {
    create_memory_node(
        connection,
        correction_id,
        replacement,
        ASSERTED_MEMORY_TYPE,
        true,
        1.0,
        &draft.embedding,
        draft.created_at_ms,
    )?;
    execute(
        connection,
        "MATCH (correction:Memory {id: $correction_id}), (target:Memory {id: $target_id})-[:ORIGINATED_IN]->(origin:Context)
         CREATE (correction)-[:ORIGINATED_IN]->(origin)",
        vec![
            ("correction_id", correction_id.into()),
            ("target_id", target_id.into()),
        ],
    )?;
    execute(
        connection,
        "MATCH (correction:Memory {id: $correction_id}), (target:Memory {id: $target_id})-[:GOVERNED_BY]->(governance:AccessGroup)
         CREATE (correction)-[:GOVERNED_BY]->(governance)",
        vec![
            ("correction_id", correction_id.into()),
            ("target_id", target_id.into()),
        ],
    )?;
    execute(
        connection,
        "MATCH (correction:Memory {id: $correction_id}), (principal:Principal {id: $principal_id})
         CREATE (correction)-[:ASSERTED_BY]->(principal)",
        vec![
            ("correction_id", correction_id.into()),
            ("principal_id", draft.principal_id.clone().into()),
        ],
    )?;
    add_supersedes(connection, correction_id, target_id)?;
    if draft.valid_from_ms.is_some() || draft.valid_until_ms.is_some() {
        let validity_id = format!("validity:{}", correction_id.trim_start_matches("memory:"));
        execute(
            connection,
            "CREATE (validity:TemporalValidity {id: $validity_id, valid_from_ms: $valid_from_ms, valid_until_ms: $valid_until_ms})",
            vec![
                ("validity_id", validity_id.clone().into()),
                (
                    "valid_from_ms",
                    draft
                        .valid_from_ms
                        .map(Value::Int64)
                        .unwrap_or(Value::Null(LogicalType::Int64)),
                ),
                (
                    "valid_until_ms",
                    draft
                        .valid_until_ms
                        .map(Value::Int64)
                        .unwrap_or(Value::Null(LogicalType::Int64)),
                ),
            ],
        )?;
        execute(
            connection,
            "MATCH (correction:Memory {id: $correction_id}), (validity:TemporalValidity {id: $validity_id})
             CREATE (correction)-[:HAS_VALIDITY]->(validity)",
            vec![
                ("correction_id", correction_id.into()),
                ("validity_id", validity_id.into()),
            ],
        )?;
    }
    Ok(())
}

fn apply_correction_mode(
    connection: &Connection<'_>,
    correction_id: &str,
    target_id: &str,
    mode: CorrectionMode,
) -> Result<()> {
    match mode {
        CorrectionMode::Replace => set_status(connection, target_id, MemoryStatus::Superseded),
        CorrectionMode::Conflict(successors) => {
            set_status(connection, target_id, MemoryStatus::Superseded)?;
            set_status(connection, correction_id, MemoryStatus::Disputed)?;
            for successor in successors {
                set_status(connection, &successor, MemoryStatus::Disputed)?;
                add_contradiction(connection, correction_id, &successor)?;
            }
            Ok(())
        }
        CorrectionMode::Resolve(cluster) => {
            for disputed in cluster {
                set_status(connection, &disputed, MemoryStatus::Superseded)?;
                if disputed != target_id {
                    add_supersedes(connection, correction_id, &disputed)?;
                }
            }
            Ok(())
        }
    }
}

fn set_status(connection: &Connection<'_>, memory_id: &str, status: MemoryStatus) -> Result<()> {
    execute(
        connection,
        "MATCH (memory:Memory {id: $memory_id}) SET memory.status = $status",
        vec![
            ("memory_id", memory_id.into()),
            ("status", status.as_str().into()),
        ],
    )
}

fn add_supersedes(connection: &Connection<'_>, newer: &str, older: &str) -> Result<()> {
    execute(
        connection,
        "MATCH (newer:Memory {id: $newer}), (older:Memory {id: $older}) MERGE (newer)-[:SUPERSEDES]->(older)",
        vec![("newer", newer.into()), ("older", older.into())],
    )
}

fn add_contradiction(connection: &Connection<'_>, left: &str, right: &str) -> Result<()> {
    for (from, to) in [(left, right), (right, left)] {
        execute(
            connection,
            "MATCH (from:Memory {id: $from}), (to:Memory {id: $to}) MERGE (from)-[:CONTRADICTS]->(to)",
            vec![("from", from.into()), ("to", to.into())],
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedding(index: usize) -> Vec<f32> {
        let mut embedding = vec![0.0; 384];
        embedding[index] = 1.0;
        embedding
    }

    fn ingest(
        graph: &MemoryGraph,
        context: &str,
        principal: &str,
        message_id: &str,
        admin: &str,
    ) -> Result<()> {
        graph.ingest_message(
            &SourceMessage {
                id: message_id.into(),
                platform_message_id: Some(message_id.into()),
                context_id: context.into(),
                context_kind: "test".into(),
                speaker_id: principal.into(),
                speaker_name: principal.into(),
                role: "user".into(),
                content: format!("source {message_id}"),
                created_at_ms: 10,
            },
            admin,
        )
    }

    fn extracted(
        graph: &MemoryGraph,
        context: &str,
        message_id: &str,
        content: &str,
    ) -> Result<String> {
        graph.create_index_memory(&MemoryDraft {
            content: content.into(),
            memory_type: "fact".into(),
            confidence: 0.8,
            embedding: embedding(0),
            origin_context_ids: vec![context.into()],
            source_message_ids: vec![message_id.into()],
            extraction_run_id: format!("run:{message_id}"),
            extraction_model: "fixture".into(),
            created_at_ms: 20,
        })
    }

    fn extracted_with_id(
        graph: &MemoryGraph,
        id: &str,
        context: &str,
        message_id: &str,
        content: &str,
    ) -> Result<String> {
        let draft = MemoryDraft {
            content: content.into(),
            memory_type: "fact".into(),
            confidence: 0.8,
            embedding: embedding(0),
            origin_context_ids: vec![context.into()],
            source_message_ids: vec![message_id.into()],
            extraction_run_id: format!("run:{message_id}"),
            extraction_model: "fixture".into(),
            created_at_ms: 20,
        };
        let connection = Connection::new(&graph.database)?;
        connection.query("BEGIN TRANSACTION")?;
        finish_transaction(
            &connection,
            graph.create_index_memory_inner(&connection, id, &draft),
        )?;
        Ok(id.to_string())
    }

    fn correction(
        target_id: &str,
        replacement: &str,
        context_id: &str,
        principal_id: &str,
    ) -> CorrectionDraft {
        CorrectionDraft {
            target_id: target_id.into(),
            replacement: replacement.into(),
            context_id: context_id.into(),
            principal_id: principal_id.into(),
            may_write: false,
            embedding: embedding(1),
            valid_from_ms: None,
            valid_until_ms: None,
            created_at_ms: 30,
        }
    }

    fn canonical(
        graph: &MemoryGraph,
        context: &str,
        principal: &str,
        content: &str,
        scope: MemoryScope,
    ) -> Result<String> {
        graph.remember_canonical(&CanonicalDraft {
            content: content.into(),
            context_id: context.into(),
            principal_id: principal.into(),
            embedding: embedding(2),
            scope,
            created_at_ms: 20,
        })
    }

    fn status(graph: &MemoryGraph, memory_id: &str) -> Result<MemoryStatus> {
        let connection = Connection::new(&graph.database)?;
        let mut statement =
            connection.prepare("MATCH (memory:Memory {id: $memory_id}) RETURN memory.status")?;
        let row = connection
            .execute(&mut statement, vec![("memory_id", memory_id.into())])?
            .next()
            .context("missing test memory")?;
        MemoryStatus::parse(&string_value(&row[0])?)
    }

    fn memory_count(graph: &MemoryGraph, memory_id: &str) -> Result<i64> {
        let connection = Connection::new(&graph.database)?;
        let mut statement =
            connection.prepare("MATCH (memory:Memory {id: $memory_id}) RETURN count(memory)")?;
        Ok(connection
            .execute(&mut statement, vec![("memory_id", memory_id.into())])?
            .next()
            .and_then(|row| row.first().and_then(|value| int64_value(value).ok()))
            .unwrap_or(0))
    }

    fn validity(graph: &MemoryGraph, memory_id: &str) -> Result<Option<TemporalValidity>> {
        let connection = Connection::new(&graph.database)?;
        let mut statement = connection.prepare(
            "MATCH (memory:Memory {id: $memory_id})-[:HAS_VALIDITY]->(validity:TemporalValidity)
             RETURN validity.valid_from_ms, validity.valid_until_ms",
        )?;
        connection
            .execute(&mut statement, vec![("memory_id", memory_id.into())])?
            .next()
            .map(|row| {
                Ok(TemporalValidity {
                    valid_from_ms: match &row[0] {
                        Value::Null(_) => None,
                        value => Some(int64_value(value)?),
                    },
                    valid_until_ms: match &row[1] {
                        Value::Null(_) => None,
                        value => Some(int64_value(value)?),
                    },
                })
            })
            .transpose()
    }

    #[test]
    fn extracted_correction_is_canonical_and_preserves_the_source() -> Result<()> {
        let graph = MemoryGraph::in_memory()?;
        ingest(&graph, "context:a", "speaker", "message:a", "admin")?;
        let old = extracted(&graph, "context:a", "message:a", "Old address")?;
        let new = graph
            .correct_memory(&correction(&old, "New address", "context:a", "speaker"))?
            .expect("authorized correction");

        assert_eq!(status(&graph, &old)?, MemoryStatus::Superseded);
        assert_eq!(status(&graph, &new)?, MemoryStatus::Active);
        let exact = graph.inspect("speaker", "context:a", &new, None)?;
        assert_eq!(exact.canonical.len(), 1);
        assert_eq!(exact.canonical[0].content, "New address");
        assert_eq!(exact.canonical[0].supersedes, [old.clone()]);
        assert_eq!(exact.canonical[0].asserted_by.as_deref(), Some("speaker"));
        assert_eq!(exact.canonical[0].origin_contexts, ["context:a"]);

        let connection = Connection::new(&graph.database)?;
        let mut statement = connection
            .prepare("MATCH (message:Message {id: 'message:a'}) RETURN message.content")?;
        let row = connection.execute(&mut statement, vec![])?.next().unwrap();
        assert_eq!(string_value(&row[0])?, "source message:a");
        Ok(())
    }

    #[test]
    fn canonical_correction_and_rollback_create_new_audit_nodes() -> Result<()> {
        let graph = MemoryGraph::in_memory()?;
        ingest(&graph, "context:a", "speaker", "message:a", "admin")?;
        let original = canonical(
            &graph,
            "context:a",
            "speaker",
            "Prefer blue",
            MemoryScope::Context,
        )?;
        let corrected = graph
            .correct_memory(&correction(
                &original,
                "Prefer green",
                "context:a",
                "speaker",
            ))?
            .unwrap();
        let rollback = graph
            .correct_memory(&correction(
                &corrected,
                "Prefer blue",
                "context:a",
                "speaker",
            ))?
            .unwrap();

        assert_ne!(rollback, original);
        assert_ne!(rollback, corrected);
        assert_eq!(status(&graph, &original)?, MemoryStatus::Superseded);
        assert_eq!(status(&graph, &corrected)?, MemoryStatus::Superseded);
        assert_eq!(status(&graph, &rollback)?, MemoryStatus::Active);
        let exact = graph.inspect("speaker", "context:a", &rollback, None)?;
        assert_eq!(exact.canonical[0].supersedes, [corrected]);
        assert_eq!(exact.canonical[0].content, "Prefer blue");
        Ok(())
    }

    #[test]
    fn exact_retry_is_idempotent_and_survives_reopen() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("memory.lbug");
        let draft;
        let first;
        {
            let graph = MemoryGraph::open(&path)?;
            ingest(&graph, "context:a", "speaker", "message:a", "admin")?;
            let target = extracted(&graph, "context:a", "message:a", "old")?;
            draft = correction(
                &target,
                "  normalized\n replacement  ",
                "context:a",
                "speaker",
            );
            first = graph.correct_memory(&draft)?.unwrap();
            let mut equivalent = draft.clone();
            equivalent.replacement = "normalized replacement".into();
            assert_eq!(
                graph.correct_memory(&equivalent)?.as_deref(),
                Some(first.as_str())
            );
            assert_eq!(memory_count(&graph, &first)?, 1);
            assert_eq!(
                graph.inspect("speaker", "context:a", &first, None)?.canonical[0].content,
                "normalized\n replacement"
            );
        }
        let reopened = MemoryGraph::open(&path)?;
        assert_eq!(
            reopened.correct_memory(&draft)?.as_deref(),
            Some(first.as_str())
        );
        assert_eq!(memory_count(&reopened, &first)?, 1);
        Ok(())
    }

    #[test]
    fn global_retry_from_another_authorized_context_reuses_the_active_successor() -> Result<()> {
        let graph = MemoryGraph::in_memory()?;
        ingest(&graph, "context:a", "speaker", "message:a", "admin")?;
        ingest(&graph, "context:b", "speaker", "message:b", "admin")?;
        let target = canonical(
            &graph,
            "context:a",
            "speaker",
            "global old",
            MemoryScope::Global,
        )?;
        let mut from_a = correction(&target, "global new", "context:a", "speaker");
        from_a.may_write = true;
        let first = graph.correct_memory(&from_a)?.unwrap();

        let mut from_b = correction(&target, "global new", "context:b", "speaker");
        from_b.may_write = true;
        let retry = graph.correct_memory(&from_b)?.unwrap();

        assert_eq!(retry, first);
        assert_eq!(memory_count(&graph, &first)?, 1);
        assert_eq!(status(&graph, &first)?, MemoryStatus::Active);
        let exact = graph.inspect("speaker", "context:b", &first, None)?;
        assert_eq!(exact.canonical.len(), 1);
        assert_eq!(exact.canonical[0].status, MemoryStatus::Active);
        assert!(exact.canonical[0].contradicts.is_empty());
        let target_audit = graph.inspect("speaker", "context:b", &target, None)?;
        assert_eq!(target_audit.canonical[0].superseded_by, [first]);
        Ok(())
    }

    fn conflict_order(first: &str, second: &str) -> Result<(String, String, MemoryGraph)> {
        let graph = MemoryGraph::in_memory()?;
        ingest(&graph, "context:a", "speaker", "message:a", "admin")?;
        let target = extracted_with_id(
            &graph,
            "memory:fixed-conflict-target",
            "context:a",
            "message:a",
            "old",
        )?;
        let first_id = graph
            .correct_memory(&correction(&target, first, "context:a", "speaker"))?
            .unwrap();
        let second_id = graph
            .correct_memory(&correction(&target, second, "context:a", "speaker"))?
            .unwrap();
        let (a, b) = if first == "A" {
            (first_id, second_id)
        } else {
            (second_id, first_id)
        };
        Ok((a, b, graph))
    }

    #[test]
    fn conflicting_orders_have_stable_ids_and_disputed_resolution() -> Result<()> {
        let (a_first, b_first, graph) = conflict_order("A", "B")?;
        let (a_second, b_second, _) = conflict_order("B", "A")?;
        assert_eq!(
            (a_first.as_str(), b_first.as_str()),
            (a_second.as_str(), b_second.as_str())
        );
        assert_eq!(status(&graph, &a_first)?, MemoryStatus::Disputed);
        assert_eq!(status(&graph, &b_first)?, MemoryStatus::Disputed);
        assert!(graph
            .canonical_for_context("speaker", "context:a", 40, 8_000)?
            .is_empty());

        let disputed = graph.inspect("speaker", "context:a", &a_first, None)?;
        assert_eq!(disputed.canonical[0].status, MemoryStatus::Disputed);
        assert_eq!(disputed.canonical[0].contradicts, [b_first.clone()]);
        let resolution = graph
            .correct_memory(&correction(&a_first, "Resolved", "context:a", "speaker"))?
            .unwrap();
        assert_eq!(status(&graph, &a_first)?, MemoryStatus::Superseded);
        assert_eq!(status(&graph, &b_first)?, MemoryStatus::Superseded);
        assert_eq!(status(&graph, &resolution)?, MemoryStatus::Active);
        let resolved = graph.inspect("speaker", "context:a", &resolution, None)?;
        assert_eq!(
            resolved.canonical[0]
                .supersedes
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([a_first, b_first])
        );
        assert_eq!(
            graph.canonical_for_context("speaker", "context:a", 40, 8_000)?[0].id,
            resolution
        );
        Ok(())
    }

    #[test]
    fn concurrent_different_corrections_serialize_to_a_dispute() -> Result<()> {
        let graph = MemoryGraph::in_memory()?;
        ingest(&graph, "context:a", "speaker", "message:a", "admin")?;
        let target = extracted(&graph, "context:a", "message:a", "old")?;
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
        let mut workers = Vec::new();
        for replacement in ["A", "B"] {
            let graph = graph.clone();
            let barrier = barrier.clone();
            let draft = correction(&target, replacement, "context:a", "speaker");
            workers.push(std::thread::spawn(move || {
                barrier.wait();
                graph.correct_memory(&draft).unwrap().unwrap()
            }));
        }
        barrier.wait();
        let ids = workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();
        assert_ne!(ids[0], ids[1]);
        assert!(ids
            .iter()
            .all(|id| status(&graph, id).unwrap() == MemoryStatus::Disputed));
        Ok(())
    }

    #[test]
    fn temporal_validity_is_explicit_and_invalid_intervals_do_not_write() -> Result<()> {
        let graph = MemoryGraph::in_memory()?;
        ingest(&graph, "context:a", "speaker", "message:a", "admin")?;
        let first = extracted(&graph, "context:a", "message:a", "old")?;
        let without = graph
            .correct_memory(&correction(&first, "without dates", "context:a", "speaker"))?
            .unwrap();
        assert_eq!(validity(&graph, &without)?, None);

        let mut with_dates = correction(&without, "with dates", "context:a", "speaker");
        with_dates.valid_from_ms = Some(100);
        with_dates.valid_until_ms = Some(200);
        let dated = graph.correct_memory(&with_dates)?.unwrap();
        assert_eq!(
            validity(&graph, &dated)?,
            Some(TemporalValidity {
                valid_from_ms: Some(100),
                valid_until_ms: Some(200),
            })
        );
        let rendered = render_inspection(&graph.inspect("speaker", "context:a", &dated, None)?);
        assert!(rendered.contains("valid_from_ms=100 valid_until_ms=200"));

        let mut invalid = correction(&dated, "invalid", "context:a", "speaker");
        invalid.valid_from_ms = Some(300);
        invalid.valid_until_ms = Some(200);
        assert!(graph.correct_memory(&invalid).is_err());
        assert_eq!(status(&graph, &dated)?, MemoryStatus::Active);
        assert_eq!(
            memory_count(
                &graph,
                &correction_id(&invalid, &dated, "invalid", InheritedScope::Context),
            )?,
            0
        );
        Ok(())
    }

    #[test]
    fn correction_authorization_inherits_scope_without_widening() -> Result<()> {
        let graph = MemoryGraph::in_memory()?;
        ingest(&graph, "context:a", "owner", "message:a", "admin")?;
        ingest(&graph, "context:a", "participant", "message:p", "admin")?;
        ingest(&graph, "context:b", "participant", "message:b", "admin")?;
        let context_memory = extracted(&graph, "context:a", "message:a", "context fact")?;
        assert!(graph
            .correct_memory(&correction(
                &context_memory,
                "participant correction",
                "context:a",
                "participant",
            ))?
            .is_some());

        let foreign = canonical(
            &graph,
            "context:a",
            "owner",
            "foreign",
            MemoryScope::Context,
        )?;
        assert!(graph
            .correct_memory(&correction(
                &foreign,
                "wrong context",
                "context:b",
                "participant",
            ))?
            .is_none());
        let mut admin_foreign = correction(&foreign, "admin wrong context", "context:b", "admin");
        admin_foreign.may_write = true;
        assert!(graph.correct_memory(&admin_foreign)?.is_none());

        let global = canonical(&graph, "context:a", "owner", "global", MemoryScope::Global)?;
        let mut global_attempt = correction(&global, "new global", "context:b", "participant");
        assert!(graph.correct_memory(&global_attempt)?.is_none());
        global_attempt.may_write = true;
        let corrected_global = graph.correct_memory(&global_attempt)?.unwrap();
        let exact = graph.inspect("participant", "context:b", &corrected_global, None)?;
        assert_eq!(exact.canonical[0].scopes, ["global"]);
        assert_eq!(exact.canonical[0].origin_contexts, ["global"]);

        let multi_context = graph.create_index_memory(&MemoryDraft {
            content: "multi-context inference".into(),
            memory_type: "fact".into(),
            confidence: 0.8,
            embedding: embedding(0),
            origin_context_ids: vec!["context:a".into(), "context:b".into()],
            source_message_ids: vec!["message:a".into(), "message:b".into()],
            extraction_run_id: "run:multi".into(),
            extraction_model: "fixture".into(),
            created_at_ms: 30,
        })?;
        assert!(graph
            .correct_memory(&correction(
                &multi_context,
                "must not narrow governance",
                "context:a",
                "participant",
            ))?
            .is_none());
        Ok(())
    }

    #[test]
    fn failed_transaction_rolls_back_replacement_and_target_status() -> Result<()> {
        let graph = MemoryGraph::in_memory()?;
        ingest(&graph, "context:a", "speaker", "message:a", "admin")?;
        let target = extracted(&graph, "context:a", "message:a", "old")?;
        let draft = correction(&target, "new", "context:a", "speaker");
        let expected_id = correction_id(&draft, &target, "new", InheritedScope::Context);
        assert!(graph.correct_memory_failing_after_create(&draft).is_err());
        assert_eq!(status(&graph, &target)?, MemoryStatus::Active);
        assert_eq!(memory_count(&graph, &expected_id)?, 0);
        Ok(())
    }

    #[test]
    fn opening_a_v1_database_adds_temporal_schema_v2() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("v1.lbug");
        {
            let database = Database::new(&path, SystemConfig::default())?;
            let connection = Connection::new(&database)?;
            connection.query(
                "CREATE NODE TABLE SchemaVersion(version INT64, PRIMARY KEY(version));
                 CREATE NODE TABLE Memory(id STRING, content STRING, memory_type STRING, canonical BOOL, status STRING, confidence DOUBLE, embedding FLOAT[384], created_at_ms INT64, PRIMARY KEY(id));
                 MERGE (:SchemaVersion {version: 1});",
            )?;
        }
        let graph = MemoryGraph::open(&path)?;
        let connection = Connection::new(&graph.database)?;
        let versions =
            connection.query("MATCH (version:SchemaVersion) RETURN max(version.version)")?;
        assert!(versions.to_string().contains('2'));
        let validity =
            connection.query("MATCH (validity:TemporalValidity) RETURN count(validity)")?;
        assert!(validity.to_string().contains('0'));
        Ok(())
    }
}
