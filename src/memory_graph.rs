use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context as _, Result, anyhow, ensure};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use lbug::{Connection, Database, LogicalType, SystemConfig, Value};
use sha2::{Digest, Sha256};

const SCHEMA_VERSION: i64 = 1;

/// The one access group every principal joins.
///
/// Ladybug authorizes a memory when the reader is a member of *every* group
/// governing it, so a memory governed by this group alone is readable from any
/// context. "Global" is therefore a governance choice inside the existing model,
/// not a second concept bolted onto it — which is why there is no `global`
/// column on `Memory`.
pub const GLOBAL_GROUP_ID: &str = "access:global";

/// `memory_type` of an explicitly remembered memory.
///
/// Every other value in this column names a *category the extractor inferred*
/// ("finding", "preference", "inference", …). This one records how the memory
/// arrived instead: somebody said "remember this".
pub const ASSERTED_MEMORY_TYPE: &str = "asserted";

fn context_group_id(context_id: &str) -> String {
    format!("access:{context_id}")
}

/// Fallback admin principal, kept only so an un-flagged process behaves as it
/// did before `--admin-id` existed. New deployments should pass the flag.
const DEFAULT_ADMIN_PRINCIPAL: &str = "telegram:user:1815217";

static ADMIN_PRINCIPAL: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Set the principal that receives Manage membership in every context's access
/// group, i.e. the one identity with cross-context recall.
///
/// Process-level and set once at startup from `--admin-id`: it does not vary per
/// request, and threading it through every `AppConfig` literal would invite the
/// one thing that must not happen — two ingest paths disagreeing about who the
/// admin is. Ignored if called twice.
pub fn set_admin_principal(principal_id: &str) {
    let _ = ADMIN_PRINCIPAL.set(principal_id.to_string());
}

/// The configured admin principal, or the historical literal when unset.
pub fn admin_principal() -> &'static str {
    ADMIN_PRINCIPAL.get().map_or(DEFAULT_ADMIN_PRINCIPAL, String::as_str)
}

#[derive(Clone)]
pub struct MemoryGraph {
    database: Arc<Database>,
    write_lock: Arc<Mutex<()>>,
}

pub struct MemoryEmbedder {
    model: Mutex<TextEmbedding>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SourceMessage {
    pub id: String,
    pub platform_message_id: Option<String>,
    pub context_id: String,
    pub context_kind: String,
    pub speaker_id: String,
    pub speaker_name: String,
    pub role: String,
    pub content: String,
    pub created_at_ms: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryDraft {
    pub content: String,
    pub memory_type: String,
    pub confidence: f64,
    pub embedding: Vec<f32>,
    pub origin_context_ids: Vec<String>,
    pub source_message_ids: Vec<String>,
    pub extraction_run_id: String,
    pub extraction_model: String,
    pub created_at_ms: i64,
}

/// How far an explicitly remembered memory reaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryScope {
    /// Readable only from the context it was asserted in.
    Context,
    /// Readable from every context, via [`GLOBAL_GROUP_ID`].
    Global,
}

/// An asserted memory: somebody said "remember this".
///
/// Deliberately not a [`MemoryDraft`]. An extraction is an *observation* and
/// carries a confidence, an extraction run and the messages it was inferred
/// from. An assertion has none of those — its provenance is "this principal
/// said so, in this context, at this time", so it links `ASSERTED_BY` instead
/// of `DERIVED_FROM`/`EXTRACTED_BY` and its confidence is fixed at 1.0.
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalDraft {
    pub content: String,
    pub context_id: String,
    pub principal_id: String,
    pub embedding: Vec<f32>,
    pub scope: MemoryScope,
    pub created_at_ms: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EpisodeDraft {
    pub context_id: String,
    pub summary: String,
    pub source_message_ids: Vec<String>,
    pub model: String,
    pub created_at_ms: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphMessage {
    pub id: String,
    pub role: String,
    pub content: String,
    pub speaker_name: String,
    pub created_at_ms: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecalledMemory {
    pub id: String,
    pub content: String,
    pub memory_type: String,
    pub canonical: bool,
    pub confidence: f64,
    pub citation: String,
}

impl MemoryGraph {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let database = Database::new(
            path,
            SystemConfig::default()
                .enable_checksums(true)
                .throw_on_wal_replay_failure(true),
        )
        .context("open Ladybug memory database")?;
        let graph = Self {
            database: Arc::new(database),
            write_lock: Arc::new(Mutex::new(())),
        };
        graph.initialize_schema()?;
        Ok(graph)
    }

    pub fn recover(path: impl AsRef<Path>) -> Result<()> {
        {
            let database = Database::new(
                path.as_ref(),
                SystemConfig::default()
                    .enable_checksums(true)
                    .throw_on_wal_replay_failure(false),
            )
            .context("open Ladybug database for WAL recovery")?;
            let connection = Connection::new(&database)?;
            connection.query("CHECKPOINT")?;
        }
        Self::open(path).context("verify recovered Ladybug database")?;
        Ok(())
    }

    #[cfg(test)]
    fn in_memory() -> Result<Self> {
        let database = Database::in_memory(SystemConfig::default())?;
        let graph = Self {
            database: Arc::new(database),
            write_lock: Arc::new(Mutex::new(())),
        };
        graph.initialize_schema()?;
        Ok(graph)
    }

    fn initialize_schema(&self) -> Result<()> {
        let _guard = self.write_lock.lock().unwrap();
        let connection = Connection::new(&self.database)?;
        connection.query(
            "CREATE NODE TABLE IF NOT EXISTS SchemaVersion(version INT64, PRIMARY KEY(version));
             CREATE NODE TABLE IF NOT EXISTS Principal(id STRING, kind STRING, display_name STRING, created_at_ms INT64, PRIMARY KEY(id));
             CREATE NODE TABLE IF NOT EXISTS Context(id STRING, kind STRING, display_name STRING, created_at_ms INT64, PRIMARY KEY(id));
             CREATE NODE TABLE IF NOT EXISTS AccessGroup(id STRING, display_name STRING, created_at_ms INT64, PRIMARY KEY(id));
             CREATE NODE TABLE IF NOT EXISTS Message(id STRING, platform_message_id STRING, role STRING, content STRING, created_at_ms INT64, PRIMARY KEY(id));
             CREATE NODE TABLE IF NOT EXISTS Episode(id STRING, summary STRING, start_message_id STRING, end_message_id STRING, status STRING, model STRING, created_at_ms INT64, PRIMARY KEY(id));
             CREATE NODE TABLE IF NOT EXISTS Memory(id STRING, content STRING, memory_type STRING, canonical BOOL, status STRING, confidence DOUBLE, embedding FLOAT[384], created_at_ms INT64, PRIMARY KEY(id));
             CREATE NODE TABLE IF NOT EXISTS ExtractionRun(id STRING, model STRING, created_at_ms INT64, PRIMARY KEY(id));
             CREATE NODE TABLE IF NOT EXISTS Entity(id STRING, kind STRING, canonical_name STRING, aliases_json STRING, created_at_ms INT64, PRIMARY KEY(id));
             CREATE NODE TABLE IF NOT EXISTS Attachment(id STRING, sha256 STRING, mime_type STRING, storage_path STRING, platform_ref STRING, created_at_ms INT64, PRIMARY KEY(id));
             CREATE REL TABLE IF NOT EXISTS USES_ACCESS_GROUP(FROM Context TO AccessGroup);
             CREATE REL TABLE IF NOT EXISTS MEMBER_OF(FROM Principal TO AccessGroup, can_read BOOL, can_contribute BOOL, can_manage BOOL, valid_from_ms INT64);
             CREATE REL TABLE IF NOT EXISTS IN_CONTEXT(FROM Message TO Context);
             CREATE REL TABLE IF NOT EXISTS SAID_BY(FROM Message TO Principal);
             CREATE REL TABLE IF NOT EXISTS ORIGINATED_IN(FROM Memory TO Context);
             CREATE REL TABLE IF NOT EXISTS GOVERNED_BY(FROM Memory TO AccessGroup);
             CREATE REL TABLE IF NOT EXISTS DERIVED_FROM(FROM Memory TO Message);
             CREATE REL TABLE IF NOT EXISTS EXTRACTED_BY(FROM Memory TO ExtractionRun);
             CREATE REL TABLE IF NOT EXISTS SUMMARIZES(FROM Episode TO Context);
             CREATE REL TABLE IF NOT EXISTS CONTAINS(FROM Episode TO Message);
             CREATE REL TABLE IF NOT EXISTS MENTIONS(FROM Memory TO Entity);
             CREATE REL TABLE IF NOT EXISTS HAS_ATTACHMENT(FROM Message TO Attachment);
             CREATE REL TABLE IF NOT EXISTS SUPERSEDES(FROM Memory TO Memory);
             CREATE REL TABLE IF NOT EXISTS CONTRADICTS(FROM Memory TO Memory);
             CREATE REL TABLE IF NOT EXISTS ASSERTED_BY(FROM Memory TO Principal);
             CREATE REL TABLE IF NOT EXISTS DENIED_TO(FROM Memory TO Principal);",
        )?;
        let mut statement = connection.prepare(
            "MERGE (version:SchemaVersion {version: $version}) RETURN version.version",
        )?;
        connection.execute(&mut statement, vec![("version", Value::Int64(SCHEMA_VERSION))])?;
        Ok(())
    }

    pub fn ingest_message(&self, message: &SourceMessage, admin_id: &str) -> Result<()> {
        let _guard = self.write_lock.lock().unwrap();
        let connection = Connection::new(&self.database)?;
        connection.query("BEGIN TRANSACTION")?;
        let result = self.ingest_message_inner(&connection, message, admin_id);
        finish_transaction(&connection, result)
    }

    pub fn has_message(&self, message_id: &str) -> Result<bool> {
        let connection = Connection::new(&self.database)?;
        let mut statement = connection.prepare(
            "MATCH (message:Message {id: $message_id}) RETURN count(message)",
        )?;
        let count = connection
            .execute(
                &mut statement,
                vec![("message_id", message_id.into())],
            )?
            .next()
            .and_then(|row| row.first().and_then(|value| int64_value(value).ok()))
            .unwrap_or(0);
        Ok(count > 0)
    }

    pub fn ingest_attachment(
        &self,
        message_id: &str,
        attachment: &crate::attachment::Attachment,
        platform_ref: Option<&str>,
    ) -> Result<String> {
        let bytes = attachment.bytes().context("read attachment for durable storage")?;
        let sha256 = format!("{:x}", Sha256::digest(&bytes));
        let directory = dirs::data_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("adapsis")
            .join("attachments");
        std::fs::create_dir_all(&directory).context("create attachment store")?;
        let storage_path = directory.join(&sha256);
        if !storage_path.exists() {
            std::fs::write(&storage_path, bytes).context("write content-addressed attachment")?;
        }

        let _guard = self.write_lock.lock().unwrap();
        let connection = Connection::new(&self.database)?;
        connection.query("BEGIN TRANSACTION")?;
        let result = (|| {
            execute(
                &connection,
                "MERGE (attachment:Attachment {id: $id}) ON CREATE SET attachment.sha256 = $sha256, attachment.mime_type = $mime_type, attachment.storage_path = $storage_path, attachment.platform_ref = $platform_ref, attachment.created_at_ms = $created_at_ms",
                vec![
                    ("id", format!("attachment:{sha256}").into()),
                    ("sha256", sha256.clone().into()),
                    ("mime_type", attachment.mime_type.clone().into()),
                    ("storage_path", storage_path.display().to_string().into()),
                    ("platform_ref", platform_ref.unwrap_or_default().into()),
                    ("created_at_ms", unix_time_ms().into()),
                ],
            )?;
            execute(
                &connection,
                "MATCH (message:Message {id: $message_id}), (attachment:Attachment {id: $attachment_id}) MERGE (message)-[:HAS_ATTACHMENT]->(attachment)",
                vec![
                    ("message_id", message_id.into()),
                    ("attachment_id", format!("attachment:{sha256}").into()),
                ],
            )
        })();
        finish_transaction(&connection, result)?;
        Ok(sha256)
    }

    fn ingest_message_inner(
        &self,
        connection: &Connection<'_>,
        message: &SourceMessage,
        admin_id: &str,
    ) -> Result<()> {
        ensure!(!message.id.is_empty(), "message id cannot be empty");
        ensure!(!message.context_id.is_empty(), "context id cannot be empty");
        ensure!(!message.speaker_id.is_empty(), "speaker id cannot be empty");

        let group_id = context_group_id(&message.context_id);
        execute(
            connection,
            "MERGE (context:Context {id: $id}) ON CREATE SET context.kind = $kind, context.display_name = $id, context.created_at_ms = $created_at_ms",
            vec![
                ("id", message.context_id.clone().into()),
                ("kind", message.context_kind.clone().into()),
                ("created_at_ms", message.created_at_ms.into()),
            ],
        )?;
        execute(
            connection,
            "MERGE (access_group:AccessGroup {id: $id}) ON CREATE SET access_group.display_name = $display_name, access_group.created_at_ms = $created_at_ms",
            vec![
                ("id", group_id.clone().into()),
                ("display_name", format!("{} memory", message.context_id).into()),
                ("created_at_ms", message.created_at_ms.into()),
            ],
        )?;
        execute(
            connection,
            "MATCH (context:Context {id: $context_id}), (access_group:AccessGroup {id: $group_id}) MERGE (context)-[:USES_ACCESS_GROUP]->(access_group)",
            vec![
                ("context_id", message.context_id.clone().into()),
                ("group_id", group_id.clone().into()),
            ],
        )?;

        self.ensure_global_group(connection, message.created_at_ms)?;

        self.upsert_principal(connection, &message.speaker_id, &message.speaker_name, message.created_at_ms)?;
        self.upsert_principal(connection, admin_id, admin_id, message.created_at_ms)?;
        self.upsert_membership(connection, &message.speaker_id, &group_id, false, message.created_at_ms)?;
        self.upsert_membership(connection, admin_id, &group_id, true, message.created_at_ms)?;
        // Everyone joins the global group, which is what makes a globally-scoped
        // memory readable everywhere without a second ACL mechanism.
        self.upsert_membership(connection, &message.speaker_id, GLOBAL_GROUP_ID, false, message.created_at_ms)?;
        self.upsert_membership(connection, admin_id, GLOBAL_GROUP_ID, true, message.created_at_ms)?;

        execute(
            connection,
            "MERGE (message:Message {id: $id}) ON CREATE SET message.platform_message_id = $platform_message_id, message.role = $role, message.content = $content, message.created_at_ms = $created_at_ms",
            vec![
                ("id", message.id.clone().into()),
                (
                    "platform_message_id",
                    message.platform_message_id.clone().unwrap_or_default().into(),
                ),
                ("role", message.role.clone().into()),
                ("content", message.content.clone().into()),
                ("created_at_ms", message.created_at_ms.into()),
            ],
        )?;
        execute(
            connection,
            "MATCH (message:Message {id: $message_id}), (context:Context {id: $context_id}) MERGE (message)-[:IN_CONTEXT]->(context)",
            vec![
                ("message_id", message.id.clone().into()),
                ("context_id", message.context_id.clone().into()),
            ],
        )?;
        execute(
            connection,
            "MATCH (message:Message {id: $message_id}), (speaker:Principal {id: $speaker_id}) MERGE (message)-[:SAID_BY]->(speaker)",
            vec![
                ("message_id", message.id.clone().into()),
                ("speaker_id", message.speaker_id.clone().into()),
            ],
        )?;
        Ok(())
    }

    fn upsert_principal(
        &self,
        connection: &Connection<'_>,
        id: &str,
        display_name: &str,
        created_at_ms: i64,
    ) -> Result<()> {
        // A context principal (a group speaking as itself, when a turn carries no
        // per-speaker metadata) must not be recorded as a person.
        let kind = if id.starts_with("telegram:group:") { "group" } else { "user" };
        execute(
            connection,
            "MERGE (principal:Principal {id: $id}) ON CREATE SET principal.kind = $kind, principal.display_name = $display_name, principal.created_at_ms = $created_at_ms",
            vec![
                ("id", id.into()),
                ("kind", kind.into()),
                ("display_name", display_name.into()),
                ("created_at_ms", created_at_ms.into()),
            ],
        )
    }

    fn upsert_membership(
        &self,
        connection: &Connection<'_>,
        principal_id: &str,
        group_id: &str,
        manage: bool,
        valid_from_ms: i64,
    ) -> Result<()> {
        execute(
            connection,
            "MATCH (principal:Principal {id: $principal_id}), (access_group:AccessGroup {id: $group_id}) MERGE (principal)-[membership:MEMBER_OF]->(access_group) ON CREATE SET membership.can_read = true, membership.can_contribute = true, membership.can_manage = $can_manage, membership.valid_from_ms = $valid_from_ms ON MATCH SET membership.can_manage = membership.can_manage OR $can_manage",
            vec![
                ("principal_id", principal_id.into()),
                ("group_id", group_id.into()),
                ("can_manage", Value::Bool(manage)),
                ("valid_from_ms", valid_from_ms.into()),
            ],
        )
    }

    fn ensure_global_group(&self, connection: &Connection<'_>, created_at_ms: i64) -> Result<()> {
        execute(
            connection,
            "MERGE (access_group:AccessGroup {id: $id}) ON CREATE SET access_group.display_name = $display_name, access_group.created_at_ms = $created_at_ms",
            vec![
                ("id", GLOBAL_GROUP_ID.into()),
                ("display_name", "global memory".into()),
                ("created_at_ms", created_at_ms.into()),
            ],
        )
    }

    pub fn create_index_memory(&self, draft: &MemoryDraft) -> Result<String> {
        ensure!(!draft.origin_context_ids.is_empty(), "memory requires an origin context");
        ensure!(!draft.source_message_ids.is_empty(), "memory requires source messages");
        let id = format!("memory:{}", uuid::Uuid::new_v4());
        let _guard = self.write_lock.lock().unwrap();
        let connection = Connection::new(&self.database)?;
        connection.query("BEGIN TRANSACTION")?;
        let result = self.create_index_memory_inner(&connection, &id, draft);
        finish_transaction(&connection, result)?;
        Ok(id)
    }

    pub fn create_episode(&self, draft: &EpisodeDraft) -> Result<String> {
        ensure!(!draft.source_message_ids.is_empty(), "episode requires source messages");
        let id = format!("episode:{}", uuid::Uuid::new_v4());
        let start_message_id = draft.source_message_ids.first().unwrap();
        let end_message_id = draft.source_message_ids.last().unwrap();
        let _guard = self.write_lock.lock().unwrap();
        let connection = Connection::new(&self.database)?;
        connection.query("BEGIN TRANSACTION")?;
        let result = (|| {
            execute(
                &connection,
                "CREATE (:Episode {id: $id, summary: $summary, start_message_id: $start_message_id, end_message_id: $end_message_id, status: 'complete', model: $model, created_at_ms: $created_at_ms})",
                vec![
                    ("id", id.clone().into()),
                    ("summary", draft.summary.clone().into()),
                    ("start_message_id", start_message_id.clone().into()),
                    ("end_message_id", end_message_id.clone().into()),
                    ("model", draft.model.clone().into()),
                    ("created_at_ms", draft.created_at_ms.into()),
                ],
            )?;
            execute(
                &connection,
                "MATCH (episode:Episode {id: $episode_id}), (context:Context {id: $context_id}) CREATE (episode)-[:SUMMARIZES]->(context)",
                vec![
                    ("episode_id", id.clone().into()),
                    ("context_id", draft.context_id.clone().into()),
                ],
            )?;
            for message_id in &draft.source_message_ids {
                execute(
                    &connection,
                    "MATCH (episode:Episode {id: $episode_id}), (message:Message {id: $message_id}) CREATE (episode)-[:CONTAINS]->(message)",
                    vec![
                        ("episode_id", id.clone().into()),
                        ("message_id", message_id.clone().into()),
                    ],
                )?;
            }
            Ok(())
        })();
        finish_transaction(&connection, result)?;
        Ok(id)
    }

    pub fn pending_context_messages(
        &self,
        context_id: &str,
        max_chars: usize,
    ) -> Result<Vec<GraphMessage>> {
        let connection = Connection::new(&self.database)?;
        let mut end_statement = connection.prepare(
            "MATCH (episode:Episode)-[:SUMMARIZES]->(context:Context {id: $context_id}) RETURN episode.end_message_id ORDER BY episode.created_at_ms DESC LIMIT 1",
        )?;
        let last_end = connection
            .execute(&mut end_statement, vec![("context_id", context_id.into())])?
            .next()
            .and_then(|row| row.first().and_then(|value| string_value(value).ok()));
        let mut statement = connection.prepare(
            "MATCH (message:Message)-[:IN_CONTEXT]->(context:Context {id: $context_id}), (message)-[:SAID_BY]->(speaker:Principal) RETURN message.id, message.role, message.content, speaker.display_name, message.created_at_ms ORDER BY message.created_at_ms, message.id",
        )?;
        let rows = connection.execute(&mut statement, vec![("context_id", context_id.into())])?;
        let mut past_checkpoint = last_end.is_none();
        let mut messages = Vec::new();
        let mut chars = 0usize;
        for row in rows {
            let id = string_value(&row[0])?;
            if !past_checkpoint {
                if Some(&id) == last_end.as_ref() {
                    past_checkpoint = true;
                }
                continue;
            }
            let content = string_value(&row[2])?;
            let content_chars = content.chars().count();
            let reached_limit = !messages.is_empty() && chars.saturating_add(content_chars) > max_chars;
            chars = chars.saturating_add(content_chars);
            messages.push(GraphMessage {
                id,
                role: string_value(&row[1])?,
                content,
                speaker_name: string_value(&row[3])?,
                created_at_ms: int64_value(&row[4])?,
            });
            if reached_limit {
                break;
            }
        }
        Ok(messages)
    }

    pub fn episode_summaries(&self, context_id: &str, limit: i64) -> Result<Vec<String>> {
        let connection = Connection::new(&self.database)?;
        let mut statement = connection.prepare(
            "MATCH (episode:Episode)-[:SUMMARIZES]->(context:Context {id: $context_id}) WHERE episode.status = 'complete' RETURN episode.summary ORDER BY episode.created_at_ms DESC LIMIT $limit",
        )?;
        connection
            .execute(
                &mut statement,
                vec![("context_id", context_id.into()), ("limit", limit.max(0).into())],
            )?
            .map(|row| string_value(&row[0]))
            .collect()
    }

    pub fn describe(&self) -> Result<String> {
        let connection = Connection::new(&self.database)?;
        let message_total = connection.query("MATCH (message:Message) RETURN count(message) AS messages")?;
        let episode_total = connection.query("MATCH (episode:Episode) RETURN count(episode) AS episodes")?;
        let memory_total = connection.query("MATCH (memory:Memory) RETURN count(memory) AS memories")?;
        let contexts = connection.query(
            "MATCH (message:Message)-[:IN_CONTEXT]->(context:Context) RETURN context.id, count(message) AS messages, sum(size(message.content)) AS characters ORDER BY messages DESC",
        )?;
        let episodes = connection.query(
            "MATCH (episode:Episode)-[:SUMMARIZES]->(context:Context) RETURN context.id, episode.id, episode.start_message_id, episode.end_message_id, size(episode.summary) AS summary_characters ORDER BY episode.created_at_ms",
        )?;
        Ok(format!(
            "Totals:\n{message_total}{episode_total}{memory_total}\nContexts:\n{contexts}\nEpisodes:\n{episodes}"
        ))
    }

    pub fn authorized_cypher(&self, principal_id: &str, query: &str) -> Result<String> {
        validate_authorized_cypher(query)?;
        let connection = Connection::new(&self.database)?;
        let mut ids_statement = connection.prepare(
            "MATCH (principal:Principal {id: $principal_id})-[membership:MEMBER_OF]->(access_group:AccessGroup)<-[:GOVERNED_BY]-(memory:Memory) WHERE membership.can_read = true AND memory.status = 'active' AND NOT (memory)-[:DENIED_TO]->(principal) WITH memory, count(DISTINCT access_group) AS accessible_groups MATCH (memory)-[:GOVERNED_BY]->(required_group:AccessGroup) WITH memory, accessible_groups, count(DISTINCT required_group) AS required_groups WHERE accessible_groups = required_groups RETURN memory.id",
        )?;
        let memory_ids: Vec<String> = connection
            .execute(
                &mut ids_statement,
                vec![("principal_id", principal_id.into())],
            )?
            .map(|row| string_value(&row[0]))
            .collect::<Result<_>>()?;
        let mut statement = connection.prepare(query)?;
        ensure!(statement.is_read_only(), "memory Cypher must be read-only");
        let mut output = String::new();
        for memory_id in memory_ids {
            let result = connection.execute(
                &mut statement,
                vec![("memory_id", memory_id.into())],
            )?;
            let rendered = result.to_string();
            if !rendered.lines().skip(1).all(str::is_empty) {
                output.push_str(&rendered);
            }
            if output.len() >= 40_000 {
                output.truncate(40_000);
                output.push_str("\n[results truncated]");
                break;
            }
        }
        Ok(output)
    }

    fn create_index_memory_inner(
        &self,
        connection: &Connection<'_>,
        id: &str,
        draft: &MemoryDraft,
    ) -> Result<()> {
        ensure!(draft.embedding.len() == 384, "memory embedding must have 384 dimensions");
        // `canonical` stays false on this path by construction: extraction infers,
        // it does not assert. Only `remember_canonical` sets the flag, which is
        // what keeps an inference from quietly overwriting something a person
        // stated outright.
        create_memory_node(
            connection,
            id,
            &draft.content,
            &draft.memory_type,
            false,
            draft.confidence,
            &draft.embedding,
            draft.created_at_ms,
        )?;
        execute(
            connection,
            "MERGE (run:ExtractionRun {id: $id}) ON CREATE SET run.model = $model, run.created_at_ms = $created_at_ms",
            vec![
                ("id", draft.extraction_run_id.clone().into()),
                ("model", draft.extraction_model.clone().into()),
                ("created_at_ms", draft.created_at_ms.into()),
            ],
        )?;
        execute(
            connection,
            "MATCH (memory:Memory {id: $memory_id}), (run:ExtractionRun {id: $run_id}) CREATE (memory)-[:EXTRACTED_BY]->(run)",
            vec![
                ("memory_id", id.into()),
                ("run_id", draft.extraction_run_id.clone().into()),
            ],
        )?;

        for context_id in &draft.origin_context_ids {
            let group_id = context_group_id(context_id);
            execute(
                connection,
                "MATCH (memory:Memory {id: $memory_id}), (context:Context {id: $context_id}), (access_group:AccessGroup {id: $group_id}) CREATE (memory)-[:ORIGINATED_IN]->(context), (memory)-[:GOVERNED_BY]->(access_group)",
                vec![
                    ("memory_id", id.into()),
                    ("context_id", context_id.clone().into()),
                    ("group_id", group_id.into()),
                ],
            )?;
        }
        for message_id in &draft.source_message_ids {
            execute(
                connection,
                "MATCH (memory:Memory {id: $memory_id}), (message:Message {id: $message_id}) CREATE (memory)-[:DERIVED_FROM]->(message)",
                vec![
                    ("memory_id", id.into()),
                    ("message_id", message_id.clone().into()),
                ],
            )?;
        }
        Ok(())
    }

    /// Record an explicitly asserted memory and return its id.
    ///
    /// The context and principal are the *turn's* — the caller cannot name them
    /// (see `coroutine::TurnIdentity`), so a memory cannot be filed against a
    /// conversation the speaker is not in.
    pub fn remember_canonical(&self, draft: &CanonicalDraft) -> Result<String> {
        ensure!(!draft.content.trim().is_empty(), "a remembered memory cannot be empty");
        ensure!(!draft.context_id.is_empty(), "a remembered memory requires a context");
        ensure!(!draft.principal_id.is_empty(), "a remembered memory requires a principal");
        let id = format!("memory:{}", uuid::Uuid::new_v4());
        let group_id = match draft.scope {
            MemoryScope::Context => context_group_id(&draft.context_id),
            MemoryScope::Global => GLOBAL_GROUP_ID.to_string(),
        };

        let _guard = self.write_lock.lock().unwrap();
        let connection = Connection::new(&self.database)?;
        connection.query("BEGIN TRANSACTION")?;
        let result = (|| {
            self.ensure_global_group(&connection, draft.created_at_ms)?;
            // A context that has never carried a message has no group yet, and a
            // `MATCH` against a missing group silently writes nothing — the memory
            // would exist ungoverned and be recallable by nobody.
            execute(
                &connection,
                "MERGE (context:Context {id: $id}) ON CREATE SET context.kind = 'conversation', context.display_name = $id, context.created_at_ms = $created_at_ms",
                vec![
                    ("id", draft.context_id.clone().into()),
                    ("created_at_ms", draft.created_at_ms.into()),
                ],
            )?;
            execute(
                &connection,
                "MERGE (access_group:AccessGroup {id: $id}) ON CREATE SET access_group.display_name = $id, access_group.created_at_ms = $created_at_ms",
                vec![
                    ("id", group_id.clone().into()),
                    ("created_at_ms", draft.created_at_ms.into()),
                ],
            )?;
            self.upsert_principal(
                &connection,
                &draft.principal_id,
                &draft.principal_id,
                draft.created_at_ms,
            )?;
            // Asserting into a context makes you a member of it; otherwise a
            // speaker could write a memory they are then not authorized to read.
            self.upsert_membership(
                &connection,
                &draft.principal_id,
                &group_id,
                false,
                draft.created_at_ms,
            )?;
            create_memory_node(
                &connection,
                &id,
                draft.content.trim(),
                ASSERTED_MEMORY_TYPE,
                true,
                1.0,
                &draft.embedding,
                draft.created_at_ms,
            )?;
            execute(
                &connection,
                "MATCH (memory:Memory {id: $memory_id}), (context:Context {id: $context_id}), (access_group:AccessGroup {id: $group_id}) CREATE (memory)-[:ORIGINATED_IN]->(context), (memory)-[:GOVERNED_BY]->(access_group)",
                vec![
                    ("memory_id", id.clone().into()),
                    ("context_id", draft.context_id.clone().into()),
                    ("group_id", group_id.clone().into()),
                ],
            )?;
            execute(
                &connection,
                "MATCH (memory:Memory {id: $memory_id}), (principal:Principal {id: $principal_id}) CREATE (memory)-[:ASSERTED_BY]->(principal)",
                vec![
                    ("memory_id", id.clone().into()),
                    ("principal_id", draft.principal_id.clone().into()),
                ],
            )
        })();
        finish_transaction(&connection, result)?;
        Ok(id)
    }

    /// Soft-delete a memory the principal is authorized to contribute to.
    ///
    /// `status` moves to `forgotten` rather than the node being deleted, so the
    /// assertion and its provenance survive an accidental forget. Returns false
    /// when the id does not exist or is not the principal's to forget — the
    /// caller must not be able to tell those apart, or forgetting becomes a probe
    /// for which memories other conversations hold.
    pub fn forget_memory(&self, principal_id: &str, memory_id: &str) -> Result<bool> {
        ensure!(!memory_id.trim().is_empty(), "memory_forget requires a memory id");
        let memory_id = memory_id.trim();
        let _guard = self.write_lock.lock().unwrap();
        let connection = Connection::new(&self.database)?;
        if !self.may_contribute_to(&connection, principal_id, memory_id)? {
            return Ok(false);
        }
        connection.query("BEGIN TRANSACTION")?;
        let result = execute(
            &connection,
            "MATCH (memory:Memory {id: $memory_id}) SET memory.status = 'forgotten'",
            vec![("memory_id", memory_id.into())],
        );
        finish_transaction(&connection, result)?;
        Ok(true)
    }

    /// The same all-governing-groups test recall uses, but on the contribute bit.
    fn may_contribute_to(
        &self,
        connection: &Connection<'_>,
        principal_id: &str,
        memory_id: &str,
    ) -> Result<bool> {
        let mut statement = connection.prepare(
            "MATCH (principal:Principal {id: $principal_id})-[membership:MEMBER_OF]->(access_group:AccessGroup)<-[:GOVERNED_BY]-(memory:Memory {id: $memory_id})
             WHERE membership.can_contribute = true AND memory.status = 'active' AND NOT (memory)-[:DENIED_TO]->(principal)
             WITH memory, count(DISTINCT access_group) AS accessible_groups
             MATCH (memory)-[:GOVERNED_BY]->(required_group:AccessGroup)
             WITH memory, accessible_groups, count(DISTINCT required_group) AS required_groups
             WHERE accessible_groups = required_groups
             RETURN count(memory)",
        )?;
        let count = connection
            .execute(
                &mut statement,
                vec![
                    ("principal_id", principal_id.into()),
                    ("memory_id", memory_id.into()),
                ],
            )?
            .next()
            .and_then(|row| row.first().and_then(|value| int64_value(value).ok()))
            .unwrap_or(0);
        Ok(count > 0)
    }

    /// Every canonical memory this turn must always know, newest first.
    ///
    /// Not ranked and not filtered by the current message: an extraction is an
    /// observation that competes for the top-5 recall slots, but an assertion is
    /// an instruction. "Remember I prefer large text" must not depend on today's
    /// message being lexically similar to it. Bounded by count and characters so
    /// an unbounded set cannot crowd out the conversation.
    pub fn canonical_for_context(
        &self,
        principal_id: &str,
        context_id: &str,
        max_count: usize,
        max_chars: usize,
    ) -> Result<Vec<RecalledMemory>> {
        if max_count == 0 || max_chars == 0 {
            return Ok(Vec::new());
        }
        let connection = Connection::new(&self.database)?;
        let mut statement = connection.prepare(
            "MATCH (principal:Principal {id: $principal_id})-[membership:MEMBER_OF]->(access_group:AccessGroup)<-[:GOVERNED_BY]-(memory:Memory)
             WHERE membership.can_read = true AND memory.status = 'active' AND memory.canonical = true
               AND NOT (memory)-[:DENIED_TO]->(principal)
             WITH memory, count(DISTINCT access_group) AS accessible_groups
             MATCH (memory)-[:GOVERNED_BY]->(required_group:AccessGroup)
             WITH memory, accessible_groups, count(DISTINCT required_group) AS required_groups,
                  collect(DISTINCT required_group.id) AS group_ids
             WHERE accessible_groups = required_groups
               AND (list_contains(group_ids, $group_id) OR list_contains(group_ids, $global_group_id))
             RETURN memory.id, memory.content, memory.memory_type, memory.canonical, memory.confidence
             ORDER BY memory.created_at_ms DESC LIMIT $limit",
        )?;
        let result = connection.execute(
            &mut statement,
            vec![
                ("principal_id", principal_id.into()),
                ("group_id", context_group_id(context_id).into()),
                ("global_group_id", GLOBAL_GROUP_ID.into()),
                ("limit", (max_count as i64).into()),
            ],
        )?;
        let mut budget = max_chars;
        let mut memories = Vec::new();
        for row in result {
            if row.len() != 5 {
                return Err(anyhow!("unexpected Ladybug canonical row width: {}", row.len()));
            }
            let content = string_value(&row[1])?;
            // Drop from the oldest end: the newest assertion is the one most
            // likely to be a correction of an older one.
            if content.len() > budget {
                break;
            }
            budget -= content.len();
            memories.push(RecalledMemory {
                id: string_value(&row[0])?,
                content,
                memory_type: string_value(&row[2])?,
                canonical: bool_value(&row[3])?,
                confidence: double_value(&row[4])?,
                citation: String::new(),
            });
        }
        // Selected newest-first so truncation drops the oldest; rendered
        // oldest-first so the model reads them in the order they were asserted.
        memories.reverse();
        Ok(memories)
    }

    /// Ranked recall over *extracted* memories only.
    ///
    /// Canonical memories are excluded here because
    /// [`Self::canonical_for_context`] injects them unconditionally — leaving
    /// them in would spend top-5 recall slots re-stating text that is already in
    /// the prompt verbatim.
    pub fn recall_authorized(&self, principal_id: &str, limit: i64) -> Result<Vec<RecalledMemory>> {
        let connection = Connection::new(&self.database)?;
        let mut statement = connection.prepare(
            "MATCH (principal:Principal {id: $principal_id})-[membership:MEMBER_OF]->(access_group:AccessGroup)<-[:GOVERNED_BY]-(memory:Memory)
             WHERE membership.can_read = true AND memory.status = 'active' AND memory.canonical = false
               AND NOT (memory)-[:DENIED_TO]->(principal)
             WITH memory, count(DISTINCT access_group) AS accessible_groups
             MATCH (memory)-[:GOVERNED_BY]->(required_group:AccessGroup)
             WITH memory, accessible_groups, count(DISTINCT required_group) AS required_groups
             WHERE accessible_groups = required_groups
             RETURN memory.id, memory.content, memory.memory_type, memory.canonical, memory.confidence
             ORDER BY memory.created_at_ms DESC LIMIT $limit",
        )?;
        let result = connection.execute(
            &mut statement,
            vec![
                ("principal_id", principal_id.into()),
                ("limit", limit.max(0).into()),
            ],
        )?;
        let mut memories: Vec<RecalledMemory> = result
            .map(|row| {
                if row.len() != 5 {
                    return Err(anyhow!("unexpected Ladybug recall row width: {}", row.len()));
                }
                Ok(RecalledMemory {
                    id: string_value(&row[0])?,
                    content: string_value(&row[1])?,
                    memory_type: string_value(&row[2])?,
                    canonical: bool_value(&row[3])?,
                    confidence: double_value(&row[4])?,
                    citation: String::new(),
                })
            })
            .collect::<Result<_>>()?;
        self.add_citations(&mut memories)?;
        Ok(memories)
    }

    pub fn recall_authorized_semantic(
        &self,
        principal_id: &str,
        embedding: &[f32],
        limit: i64,
    ) -> Result<Vec<RecalledMemory>> {
        ensure!(embedding.len() == 384, "query embedding must have 384 dimensions");
        let connection = Connection::new(&self.database)?;
        let mut statement = connection.prepare(
            "MATCH (principal:Principal {id: $principal_id})-[membership:MEMBER_OF]->(access_group:AccessGroup)<-[:GOVERNED_BY]-(memory:Memory)
             WHERE membership.can_read = true AND memory.status = 'active' AND memory.canonical = false
               AND NOT (memory)-[:DENIED_TO]->(principal)
             WITH memory, count(DISTINCT access_group) AS accessible_groups
             MATCH (memory)-[:GOVERNED_BY]->(required_group:AccessGroup)
             WITH memory, accessible_groups, count(DISTINCT required_group) AS required_groups
             WHERE accessible_groups = required_groups
             RETURN memory.id, memory.content, memory.memory_type, memory.canonical, memory.confidence, array_cosine_similarity(memory.embedding, $embedding) AS similarity
             ORDER BY similarity DESC, memory.created_at_ms DESC LIMIT $limit",
        )?;
        let result = connection.execute(
            &mut statement,
            vec![
                ("principal_id", principal_id.into()),
                ("embedding", embedding_value(embedding)?),
                ("limit", limit.max(0).into()),
            ],
        )?;
        let mut memories: Vec<RecalledMemory> = result
            .map(|row| {
                if row.len() != 6 {
                    return Err(anyhow!("unexpected Ladybug semantic recall row width: {}", row.len()));
                }
                Ok(RecalledMemory {
                    id: string_value(&row[0])?,
                    content: string_value(&row[1])?,
                    memory_type: string_value(&row[2])?,
                    canonical: bool_value(&row[3])?,
                    confidence: double_value(&row[4])?,
                    citation: String::new(),
                })
            })
            .collect::<Result<_>>()?;
        self.add_citations(&mut memories)?;
        Ok(memories)
    }

    pub fn recall_authorized_hybrid(
        &self,
        principal_id: &str,
        query: &str,
        embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<RecalledMemory>> {
        let mut recalled = self.recall_authorized_semantic(principal_id, embedding, limit as i64)?;
        let query_terms: std::collections::HashSet<String> = query
            .split(|character: char| !character.is_alphanumeric())
            .filter(|term| term.chars().count() >= 3)
            .map(str::to_lowercase)
            .collect();
        if query_terms.is_empty() || recalled.len() >= limit {
            return Ok(recalled);
        }
        for candidate in self.recall_authorized(principal_id, 200)? {
            if recalled.iter().any(|memory| memory.id == candidate.id) {
                continue;
            }
            let candidate_terms: std::collections::HashSet<String> = candidate
                .content
                .split(|character: char| !character.is_alphanumeric())
                .map(str::to_lowercase)
                .collect();
            if !query_terms.is_disjoint(&candidate_terms) {
                recalled.push(candidate);
                if recalled.len() >= limit {
                    break;
                }
            }
        }
        Ok(recalled)
    }

    fn add_citations(&self, memories: &mut [RecalledMemory]) -> Result<()> {
        let connection = Connection::new(&self.database)?;
        let mut statement = connection.prepare(
            "MATCH (memory:Memory {id: $memory_id})-[:DERIVED_FROM]->(message:Message)-[:IN_CONTEXT]->(context:Context), (message)-[:SAID_BY]->(speaker:Principal) RETURN context.id, speaker.display_name, message.created_at_ms, message.platform_message_id, message.id ORDER BY message.created_at_ms LIMIT 3",
        )?;
        for memory in memories {
            let rows = connection.execute(
                &mut statement,
                vec![("memory_id", memory.id.clone().into())],
            )?;
            let mut sources = Vec::new();
            for row in rows {
                let platform_id = string_value(&row[3])?;
                let source_id = if platform_id.is_empty() {
                    string_value(&row[4])?
                } else {
                    platform_id
                };
                sources.push(format!(
                    "{}; {}; {}; {}",
                    string_value(&row[0])?,
                    string_value(&row[1])?,
                    int64_value(&row[2])?,
                    source_id
                ));
            }
            memory.citation = sources.join(" | ");
        }
        Ok(())
    }
}

impl MemoryEmbedder {
    pub fn new() -> Result<Self> {
        let options = TextInitOptions::new(EmbeddingModel::MultilingualE5Small)
            .with_show_download_progress(false);
        let model = TextEmbedding::try_new(options).context("initialize multilingual FastEmbed model")?;
        Ok(Self {
            model: Mutex::new(model),
        })
    }

    pub fn embed_passages(&self, passages: &[String]) -> Result<Vec<Vec<f32>>> {
        let inputs: Vec<String> = passages
            .iter()
            .map(|passage| format!("passage: {passage}"))
            .collect();
        self.model
            .lock()
            .unwrap()
            .embed(inputs, None)
            .context("embed memory passages")
    }

    pub fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        self.model
            .lock()
            .unwrap()
            .embed(vec![format!("query: {query}")], None)
            .context("embed memory query")?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("FastEmbed returned no query embedding"))
    }
}

/// The single `CREATE (:Memory …)` in the codebase.
///
/// `canonical` and `confidence` used to be string literals inside the extraction
/// path's Cypher, which is why the flag existed in the schema for so long without
/// ever being true. Both are parameters now so there is exactly one place that
/// decides them.
#[allow(clippy::too_many_arguments)]
fn create_memory_node(
    connection: &Connection<'_>,
    id: &str,
    content: &str,
    memory_type: &str,
    canonical: bool,
    confidence: f64,
    embedding: &[f32],
    created_at_ms: i64,
) -> Result<()> {
    ensure!(embedding.len() == 384, "memory embedding must have 384 dimensions");
    execute(
        connection,
        "CREATE (:Memory {id: $id, content: $content, memory_type: $memory_type, canonical: $canonical, status: 'active', confidence: $confidence, embedding: $embedding, created_at_ms: $created_at_ms})",
        vec![
            ("id", id.into()),
            ("content", content.into()),
            ("memory_type", memory_type.into()),
            ("canonical", Value::Bool(canonical)),
            ("confidence", confidence.into()),
            ("embedding", embedding_value(embedding)?),
            ("created_at_ms", created_at_ms.into()),
        ],
    )
}

fn execute(connection: &Connection<'_>, query: &str, params: Vec<(&str, Value)>) -> Result<()> {
    let mut statement = connection.prepare(query)?;
    connection.execute(&mut statement, params)?;
    Ok(())
}

fn embedding_value(embedding: &[f32]) -> Result<Value> {
    ensure!(embedding.len() == 384, "embedding must have 384 dimensions");
    Ok(Value::Array(
        LogicalType::Float,
        embedding.iter().copied().map(Value::Float).collect(),
    ))
}

/// Wall-clock milliseconds, for callers that mint a memory outside this module.
pub fn now_ms() -> i64 {
    unix_time_ms()
}

fn unix_time_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64
}

fn validate_authorized_cypher(query: &str) -> Result<()> {
    let normalized = query.trim().trim_end_matches(';').trim();
    let uppercase = normalized.to_ascii_uppercase();
    ensure!(
        uppercase.starts_with("MATCH (MEMORY:MEMORY {ID: $MEMORY_ID})"),
        "memory Cypher must start with MATCH (memory:Memory {{id: $memory_id}})"
    );
    ensure!(uppercase.matches("MATCH").count() == 1, "memory Cypher permits one MATCH clause");
    ensure!(!uppercase.contains("<-"), "memory Cypher permits outgoing provenance traversals only");
    ensure!(!uppercase.contains('*'), "memory Cypher does not permit variable-length traversals");
    let return_offset = uppercase.find(" RETURN ").context("memory Cypher requires RETURN")?;
    ensure!(
        !uppercase[..return_offset].contains(','),
        "memory Cypher does not permit independent comma-separated patterns"
    );
    for forbidden in [
        " CREATE ", " MERGE ", " SET ", " DELETE ", " DETACH ", " DROP ", " ALTER ",
        " COPY ", " LOAD ", " CALL ", " SUPERSEDES", " CONTRADICTS", " DENIED_TO",
    ] {
        ensure!(!uppercase.contains(forbidden), "memory Cypher contains forbidden operation {forbidden}");
    }
    Ok(())
}

fn finish_transaction(connection: &Connection<'_>, result: Result<()>) -> Result<()> {
    match result {
        Ok(()) => {
            connection.query("COMMIT")?;
            connection.query("CHECKPOINT")?;
            Ok(())
        }
        Err(error) => {
            let _ = connection.query("ROLLBACK");
            Err(error)
        }
    }
}

fn string_value(value: &Value) -> Result<String> {
    match value {
        Value::String(value) => Ok(value.clone()),
        other => Err(anyhow!("expected string from Ladybug, got {other:?}")),
    }
}

fn bool_value(value: &Value) -> Result<bool> {
    match value {
        Value::Bool(value) => Ok(*value),
        other => Err(anyhow!("expected bool from Ladybug, got {other:?}")),
    }
}

fn double_value(value: &Value) -> Result<f64> {
    match value {
        Value::Double(value) => Ok(*value),
        other => Err(anyhow!("expected double from Ladybug, got {other:?}")),
    }
}

fn int64_value(value: &Value) -> Result<i64> {
    match value {
        Value::Int64(value) => Ok(*value),
        other => Err(anyhow!("expected int64 from Ladybug, got {other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message(id: &str, context_id: &str, speaker_id: &str) -> SourceMessage {
        SourceMessage {
            id: id.to_string(),
            platform_message_id: Some(id.to_string()),
            context_id: context_id.to_string(),
            context_kind: "telegram_group".to_string(),
            speaker_id: speaker_id.to_string(),
            speaker_name: speaker_id.to_string(),
            role: "user".to_string(),
            content: format!("source {id}"),
            created_at_ms: 100,
        }
    }

    #[test]
    fn memory_is_authorized_through_its_context_group() -> Result<()> {
        let graph = MemoryGraph::in_memory()?;
        graph.ingest_message(&message("m1", "chronica", "kata"), "marenz")?;
        let memory_id = graph.create_index_memory(&MemoryDraft {
            content: "The mobile header overflows".to_string(),
            memory_type: "finding".to_string(),
            confidence: 0.9,
            embedding: vec![1.0; 384],
            origin_context_ids: vec!["chronica".to_string()],
            source_message_ids: vec!["m1".to_string()],
            extraction_run_id: "run:1".to_string(),
            extraction_model: "dev-bot".to_string(),
            created_at_ms: 200,
        })?;

        let recalled = graph.recall_authorized("kata", 10)?;
        assert_eq!(recalled[0].id, memory_id);
        assert!(recalled[0].citation.contains("chronica; kata; 100; m1"));
        assert_eq!(
            graph.recall_authorized_semantic("kata", &vec![1.0; 384], 10)?[0].id,
            memory_id
        );
        assert_eq!(
            graph.recall_authorized_hybrid("kata", "mobile header", &vec![1.0; 384], 10)?[0].id,
            memory_id
        );
        let cypher = "MATCH (memory:Memory {id: $memory_id})-[:DERIVED_FROM]->(message:Message) RETURN memory.id, message.content";
        assert!(graph.authorized_cypher("kata", cypher)?.contains("source m1"));
        assert!(graph.authorized_cypher("sven", cypher)?.is_empty());
        assert_eq!(graph.recall_authorized("marenz", 10)?[0].id, memory_id);
        assert!(graph.recall_authorized("sven", 10)?.is_empty());
        Ok(())
    }

    #[test]
    fn cross_context_memory_requires_all_source_groups() -> Result<()> {
        let graph = MemoryGraph::in_memory()?;
        graph.ingest_message(&message("m1", "chronica", "kata"), "marenz")?;
        graph.ingest_message(&message("m2", "private", "sven"), "marenz")?;
        graph.create_index_memory(&MemoryDraft {
            content: "Combined conclusion".to_string(),
            memory_type: "inference".to_string(),
            confidence: 0.8,
            embedding: vec![1.0; 384],
            origin_context_ids: vec!["chronica".to_string(), "private".to_string()],
            source_message_ids: vec!["m1".to_string(), "m2".to_string()],
            extraction_run_id: "run:2".to_string(),
            extraction_model: "dev-bot".to_string(),
            created_at_ms: 200,
        })?;

        assert!(graph.recall_authorized("kata", 10)?.is_empty());
        assert!(graph.recall_authorized("sven", 10)?.is_empty());
        assert_eq!(graph.recall_authorized("marenz", 10)?.len(), 1);
        Ok(())
    }

    #[test]
    fn episode_checkpoint_advances_without_deleting_messages() -> Result<()> {
        let graph = MemoryGraph::in_memory()?;
        for index in 0..5 {
            let mut source = message(&format!("m{index}"), "chronica", "kata");
            source.created_at_ms = index;
            graph.ingest_message(&source, "marenz")?;
        }
        assert_eq!(graph.pending_context_messages("chronica", 10_000)?.len(), 5);
        assert_eq!(
            graph.pending_context_messages("chronica", 12)?.len(),
            2,
            "the first message crossing the threshold signals that more context exists"
        );

        graph.create_episode(&EpisodeDraft {
            context_id: "chronica".to_string(),
            summary: "First task completed".to_string(),
            source_message_ids: vec!["m0".to_string(), "m1".to_string(), "m2".to_string()],
            model: "dev-bot".to_string(),
            created_at_ms: 10,
        })?;

        let pending = graph.pending_context_messages("chronica", 10_000)?;
        assert_eq!(pending.iter().map(|message| message.id.as_str()).collect::<Vec<_>>(), vec!["m3", "m4"]);
        assert_eq!(graph.episode_summaries("chronica", 10)?, vec!["First task completed"]);
        Ok(())
    }

    fn asserted(content: &str, context_id: &str, principal_id: &str, scope: MemoryScope) -> CanonicalDraft {
        CanonicalDraft {
            content: content.to_string(),
            context_id: context_id.to_string(),
            principal_id: principal_id.to_string(),
            embedding: vec![0.5; 384],
            scope,
            created_at_ms: 300,
        }
    }

    #[test]
    fn a_remembered_memory_is_canonical_and_scoped_to_its_context() -> Result<()> {
        let graph = MemoryGraph::in_memory()?;
        graph.ingest_message(&message("m1", "chronica", "kata"), "marenz")?;
        graph.ingest_message(&message("m2", "family", "renate"), "marenz")?;

        let id = graph.remember_canonical(&asserted(
            "Kata prefers metric units",
            "chronica",
            "kata",
            MemoryScope::Context,
        ))?;

        let mine = graph.canonical_for_context("kata", "chronica", 10, 10_000)?;
        assert_eq!(mine.len(), 1);
        assert_eq!(mine[0].id, id);
        assert!(mine[0].canonical, "an asserted memory must be canonical");
        assert_eq!(mine[0].confidence, 1.0, "an assertion is not an estimate");
        assert_eq!(mine[0].memory_type, ASSERTED_MEMORY_TYPE);

        // The defect this replaces: a note taken in one conversation surfacing in
        // an unrelated one.
        assert!(
            graph.canonical_for_context("renate", "family", 10, 10_000)?.is_empty(),
            "a context-scoped memory must not leak into another context"
        );
        Ok(())
    }

    #[test]
    fn a_global_memory_is_readable_from_every_context() -> Result<()> {
        let graph = MemoryGraph::in_memory()?;
        graph.ingest_message(&message("m1", "chronica", "kata"), "marenz")?;
        graph.ingest_message(&message("m2", "family", "renate"), "marenz")?;

        graph.remember_canonical(&asserted(
            "Always answer in German",
            "chronica",
            "kata",
            MemoryScope::Global,
        ))?;

        assert_eq!(graph.canonical_for_context("kata", "chronica", 10, 10_000)?.len(), 1);
        assert_eq!(
            graph.canonical_for_context("renate", "family", 10, 10_000)?.len(),
            1,
            "global scope is membership in the one group everybody joins"
        );
        Ok(())
    }

    #[test]
    fn canonical_memories_bypass_ranking_but_not_the_budget() -> Result<()> {
        let graph = MemoryGraph::in_memory()?;
        graph.ingest_message(&message("m1", "chronica", "kata"), "marenz")?;
        for index in 0..5 {
            let mut draft = asserted(
                &format!("assertion number {index}"),
                "chronica",
                "kata",
                MemoryScope::Context,
            );
            draft.created_at_ms = 300 + index;
            graph.remember_canonical(&draft)?;
        }

        let all = graph.canonical_for_context("kata", "chronica", 10, 10_000)?;
        assert_eq!(all.len(), 5, "canonical memories are not similarity-ranked away");
        assert_eq!(all[0].content, "assertion number 0", "rendered oldest first");
        assert_eq!(all[4].content, "assertion number 4");

        let capped = graph.canonical_for_context("kata", "chronica", 2, 10_000)?;
        assert_eq!(capped.len(), 2);
        assert_eq!(
            capped.iter().map(|m| m.content.as_str()).collect::<Vec<_>>(),
            vec!["assertion number 3", "assertion number 4"],
            "the count bound drops the oldest, keeping the newest correction"
        );

        let starved = graph.canonical_for_context("kata", "chronica", 10, 40)?;
        assert_eq!(starved.len(), 2, "the character budget bounds it too");
        Ok(())
    }

    #[test]
    fn an_extraction_never_lands_in_the_canonical_set() -> Result<()> {
        let graph = MemoryGraph::in_memory()?;
        graph.ingest_message(&message("m1", "chronica", "kata"), "marenz")?;
        graph.create_index_memory(&MemoryDraft {
            content: "The mobile header overflows".to_string(),
            memory_type: "finding".to_string(),
            confidence: 0.9,
            embedding: vec![1.0; 384],
            origin_context_ids: vec!["chronica".to_string()],
            source_message_ids: vec!["m1".to_string()],
            extraction_run_id: "run:1".to_string(),
            extraction_model: "dev-bot".to_string(),
            created_at_ms: 200,
        })?;
        let remembered = graph.remember_canonical(&asserted(
            "Ship on Fridays",
            "chronica",
            "kata",
            MemoryScope::Context,
        ))?;

        let canonical = graph.canonical_for_context("kata", "chronica", 10, 10_000)?;
        assert_eq!(canonical.len(), 1, "inference must not enter the asserted set");
        assert_eq!(canonical[0].id, remembered);

        // And the reverse: the asserted memory does not also burn a ranked slot.
        let ranked = graph.recall_authorized("kata", 10)?;
        assert_eq!(ranked.len(), 1);
        assert_ne!(ranked[0].id, remembered);
        assert!(
            graph.recall_authorized_semantic("kata", &vec![0.5; 384], 10)?
                .iter()
                .all(|memory| memory.id != remembered),
            "a canonical memory is injected verbatim, so it must not be recalled twice"
        );
        Ok(())
    }

    #[test]
    fn forgetting_is_a_status_change_an_outsider_cannot_make() -> Result<()> {
        let graph = MemoryGraph::in_memory()?;
        graph.ingest_message(&message("m1", "chronica", "kata"), "marenz")?;
        graph.ingest_message(&message("m2", "family", "renate"), "marenz")?;
        let id = graph.remember_canonical(&asserted(
            "Kata prefers metric units",
            "chronica",
            "kata",
            MemoryScope::Context,
        ))?;

        assert!(
            !graph.forget_memory("renate", &id)?,
            "a principal outside the governing group cannot forget it"
        );
        assert_eq!(graph.canonical_for_context("kata", "chronica", 10, 10_000)?.len(), 1);

        assert!(graph.forget_memory("kata", &id)?);
        assert!(graph.canonical_for_context("kata", "chronica", 10, 10_000)?.is_empty());
        // Soft delete: the node and its provenance survive the status change.
        assert!(!graph.forget_memory("kata", &id)?, "forgetting twice is not an error path");
        assert!(graph.forget_memory("kata", "memory:does-not-exist").is_ok());
        Ok(())
    }

    #[test]
    fn authorized_cypher_is_anchored_and_read_only() -> Result<()> {
        assert!(validate_authorized_cypher(
            "MATCH (memory:Memory {id: $memory_id})-[:DERIVED_FROM]->(message:Message) RETURN memory.id, message.content"
        ).is_ok());
        assert!(validate_authorized_cypher("MATCH (memory:Memory) RETURN memory").is_err());
        assert!(validate_authorized_cypher(
            "MATCH (memory:Memory {id: $memory_id}) MATCH (other:Memory) RETURN other"
        ).is_err());
        assert!(validate_authorized_cypher(
            "MATCH (memory:Memory {id: $memory_id})-[:MENTIONS]->(entity)<-[:MENTIONS]-(other) RETURN other"
        ).is_err());
        assert!(validate_authorized_cypher(
            "MATCH (memory:Memory {id: $memory_id}) DELETE memory"
        ).is_err());
        Ok(())
    }
}
