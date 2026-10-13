//! Deterministic, network-free Ladybug memory quality evaluation.
use super::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

const FIXTURE_SCHEMA_VERSION: u32 = 1;
const REPORT_SCHEMA_VERSION: u32 = 1;
const TOKEN_ESTIMATE_DEFINITION: &str =
    "ceil((prompt_chars + output_chars) / 4 Unicode scalar values)";
const DEFAULT_FIXTURE: &str = "tests/fixtures/memory-quality/v1/corpus.json";
const DEFAULT_SNAPSHOT: &str = "tests/fixtures/memory-quality/v1/extraction-snapshot.json";
const DEFAULT_BASELINE: &str = "tests/fixtures/memory-quality/v1/baseline.json";
const FIXTURE_ADMIN: &str = "principal:fixture-operator";

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CorpusFixture {
    schema_version: u32,
    fixture_id: String,
    description: String,
    coverage: Vec<String>,
    embeddings: BTreeMap<String, SparseEmbedding>,
    messages: Vec<MessageFixture>,
    attachment_references: Vec<AttachmentReferenceFixture>,
    extracted_memories: Vec<ExtractedMemoryFixture>,
    canonical_memories: Vec<CanonicalMemoryFixture>,
    noise_series: Vec<NoiseSeriesFixture>,
    denials: Vec<DenialFixture>,
    queries: Vec<QueryFixture>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct SparseEmbedding {
    dimensions: usize,
    values: Vec<(usize, f32)>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct MessageFixture {
    key: String,
    context_id: String,
    context_kind: String,
    speaker_id: String,
    speaker_name: String,
    role: String,
    content: String,
    created_at_ms: i64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct AttachmentReferenceFixture {
    message_key: String,
    mime_type: String,
    reference_kind: String,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Freshness {
    Current,
    Stale,
    Neutral,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExtractedMemoryFixture {
    key: String,
    content: String,
    memory_type: String,
    confidence: f64,
    embedding_key: String,
    origin_context_ids: Vec<String>,
    source_message_keys: Vec<String>,
    expected_citation_message_keys: Vec<String>,
    extraction_run_id: String,
    extraction_model: String,
    created_at_ms: i64,
    freshness: Freshness,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FixtureScope {
    Context,
    Global,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalMemoryFixture {
    key: String,
    content: String,
    context_id: String,
    principal_id: String,
    embedding_key: String,
    scope: FixtureScope,
    created_at_ms: i64,
    freshness: Freshness,
    correction_group: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct NoiseSeriesFixture {
    key_prefix: String,
    count: usize,
    content_prefix: String,
    context_id: String,
    source_message_key: String,
    embedding_key: String,
    created_at_ms_start: i64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct DenialFixture {
    memory_key: String,
    principal_id: String,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum QuerySurface {
    Inspect,
    RecallHybrid,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct QueryFixture {
    id: String,
    surface: QuerySurface,
    principal_id: String,
    context_id: String,
    query: String,
    embedding_key: Option<String>,
    exact_memory_key: Option<String>,
    limit: Option<usize>,
    expected_memory_keys: Vec<String>,
    forbidden_memory_keys: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExtractionSnapshot {
    schema_version: u32,
    snapshot_id: String,
    fixture_id: String,
    provider_label: String,
    gold_fact_keys: Vec<String>,
    candidates: Vec<ExtractionCandidate>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExtractionCandidate {
    id: String,
    content: String,
    matched_gold_key: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Baseline {
    schema_version: u32,
    baseline_id: String,
    fixture_id: String,
    extraction_snapshot_id: String,
    expected: QualityMetrics,
    reference_latency: LatencyMetrics,
    thresholds: Thresholds,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Thresholds {
    min_retrieval_precision: f64,
    min_retrieval_recall: f64,
    max_unauthorized_recall_count: usize,
    max_unsupported_citation_count: usize,
    max_stale_memory_rate: f64,
    min_extraction_precision: f64,
    min_extraction_recall: f64,
    max_prompt_chars: usize,
    max_output_chars: usize,
    max_estimated_tokens: usize,
    max_latency_p95_us: u64,
    max_latency_max_us: u64,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualityMetrics {
    pub retrieval: RetrievalMetrics,
    pub extraction: ExtractionMetrics,
    pub cost: CostMetrics,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalMetrics {
    pub query_count: usize,
    pub returned_count: usize,
    pub true_positive_count: usize,
    pub false_positive_count: usize,
    pub false_negative_count: usize,
    pub precision: f64,
    pub recall: f64,
    pub unauthorized_recall_count: usize,
    pub unsupported_citation_count: usize,
    pub stale_memory_count: usize,
    pub freshness_scored_count: usize,
    pub stale_memory_rate: f64,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractionMetrics {
    pub gold_fact_count: usize,
    pub candidate_count: usize,
    pub supported_candidate_count: usize,
    pub recovered_gold_count: usize,
    pub precision: f64,
    pub recall: f64,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CostMetrics {
    pub prompt_chars: usize,
    pub canonical_prompt_chars: usize,
    pub ranked_recall_prompt_chars: usize,
    pub output_chars: usize,
    pub estimated_tokens: usize,
    pub token_estimate_definition: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct LatencyMetrics {
    pub unit: String,
    pub sample_count: usize,
    pub p50: u64,
    pub p95: u64,
    pub max: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CaseResult {
    pub id: String,
    pub surface: String,
    pub expected_memory_keys: Vec<String>,
    pub returned_memory_keys: Vec<String>,
    pub true_positive_count: usize,
    pub false_positive_count: usize,
    pub false_negative_count: usize,
    pub unauthorized_recall_count: usize,
    pub unsupported_citation_count: usize,
    pub prompt_chars: usize,
    pub canonical_prompt_chars: usize,
    pub ranked_recall_prompt_chars: usize,
    pub output_chars: usize,
    pub bounded_scan_truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BaselineComparison {
    pub baseline_id: String,
    pub exact_quality_match: bool,
    pub reference_latency: LatencyMetrics,
    pub passed: bool,
    pub violations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct QualityReport {
    pub schema_version: u32,
    pub fixture_id: String,
    pub extraction_snapshot_id: String,
    pub extraction_provider_label: String,
    pub quality: QualityMetrics,
    pub latency: LatencyMetrics,
    pub thresholds: Thresholds,
    pub baseline_comparison: BaselineComparison,
    pub coverage: Vec<String>,
    pub cases: Vec<CaseResult>,
}

struct SeededCorpus {
    graph: MemoryGraph,
    id_by_key: BTreeMap<String, String>,
    key_by_id: BTreeMap<String, String>,
    freshness_by_key: BTreeMap<String, Freshness>,
    citation_ids_by_key: BTreeMap<String, BTreeSet<String>>,
}

struct ReturnedMemory {
    key: String,
    canonical: bool,
    citation_source_ids: Vec<String>,
}

struct RetrievalEvaluation {
    metrics: RetrievalMetrics,
    canonical_prompt_chars: usize,
    ranked_recall_prompt_chars: usize,
    output_chars: usize,
    latencies_us: Vec<u64>,
    cases: Vec<CaseResult>,
}

pub fn run_cli(
    fixture: Option<&Path>,
    snapshot: Option<&Path>,
    baseline: Option<&Path>,
    report_path: Option<&Path>,
) -> Result<()> {
    let fixture = fixture
        .map(PathBuf::from)
        .unwrap_or_else(|| repository_path(DEFAULT_FIXTURE));
    let snapshot = snapshot
        .map(PathBuf::from)
        .unwrap_or_else(|| repository_path(DEFAULT_SNAPSHOT));
    let baseline = baseline
        .map(PathBuf::from)
        .unwrap_or_else(|| repository_path(DEFAULT_BASELINE));
    let report = evaluate_paths(&fixture, &snapshot, &baseline)?;
    let json = report_json(&report)?;
    println!("{json}");
    if let Some(path) = report_path {
        std::fs::write(path, format!("{json}\n"))
            .with_context(|| format!("write memory quality report {}", path.display()))?;
        eprintln!("machine report: {}", path.display());
    }
    print_human_summary(&report);
    enforce_gate(&report)
}

fn repository_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn evaluate_paths(fixture: &Path, snapshot: &Path, baseline: &Path) -> Result<QualityReport> {
    let corpus: CorpusFixture = read_json(fixture, "memory quality corpus")?;
    let snapshot: ExtractionSnapshot = read_json(snapshot, "memory extraction snapshot")?;
    let baseline: Baseline = read_json(baseline, "memory quality baseline")?;
    evaluate(&corpus, &snapshot, &baseline)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path, label: &str) -> Result<T> {
    let bytes = std::fs::read(path).with_context(|| format!("read {label} {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {label} {}", path.display()))
}

fn evaluate(
    corpus: &CorpusFixture,
    snapshot: &ExtractionSnapshot,
    baseline: &Baseline,
) -> Result<QualityReport> {
    validate_inputs(corpus, snapshot, baseline)?;
    let seeded = seed_corpus(corpus)?;
    let retrieval = evaluate_retrieval(corpus, &seeded)?;
    let extraction = evaluate_extraction(snapshot);
    let prompt_chars = retrieval
        .canonical_prompt_chars
        .saturating_add(retrieval.ranked_recall_prompt_chars);
    let total_chars = prompt_chars.saturating_add(retrieval.output_chars);
    let quality = QualityMetrics {
        retrieval: retrieval.metrics,
        extraction,
        cost: CostMetrics {
            prompt_chars,
            canonical_prompt_chars: retrieval.canonical_prompt_chars,
            ranked_recall_prompt_chars: retrieval.ranked_recall_prompt_chars,
            output_chars: retrieval.output_chars,
            estimated_tokens: total_chars.saturating_add(3) / 4,
            token_estimate_definition: TOKEN_ESTIMATE_DEFINITION.to_string(),
        },
    };
    let latency = latency_metrics(retrieval.latencies_us);
    let violations = threshold_violations(&quality, &latency, &baseline.thresholds);
    Ok(QualityReport {
        schema_version: REPORT_SCHEMA_VERSION,
        fixture_id: corpus.fixture_id.clone(),
        extraction_snapshot_id: snapshot.snapshot_id.clone(),
        extraction_provider_label: snapshot.provider_label.clone(),
        baseline_comparison: BaselineComparison {
            baseline_id: baseline.baseline_id.clone(),
            exact_quality_match: quality == baseline.expected,
            reference_latency: baseline.reference_latency.clone(),
            passed: violations.is_empty(),
            violations,
        },
        quality,
        latency,
        thresholds: baseline.thresholds.clone(),
        coverage: corpus.coverage.clone(),
        cases: retrieval.cases,
    })
}

fn validate_inputs(
    corpus: &CorpusFixture,
    snapshot: &ExtractionSnapshot,
    baseline: &Baseline,
) -> Result<()> {
    ensure!(
        corpus.schema_version == FIXTURE_SCHEMA_VERSION,
        "unsupported corpus schema {}",
        corpus.schema_version
    );
    ensure!(
        !corpus.description.trim().is_empty(),
        "corpus description cannot be empty"
    );
    ensure!(
        snapshot.schema_version == FIXTURE_SCHEMA_VERSION,
        "unsupported extraction snapshot schema {}",
        snapshot.schema_version
    );
    ensure!(
        baseline.schema_version == FIXTURE_SCHEMA_VERSION,
        "unsupported baseline schema {}",
        baseline.schema_version
    );
    ensure!(
        snapshot.fixture_id == corpus.fixture_id,
        "snapshot fixture_id does not match corpus"
    );
    ensure!(
        baseline.fixture_id == corpus.fixture_id,
        "baseline fixture_id does not match corpus"
    );
    ensure!(
        baseline.extraction_snapshot_id == snapshot.snapshot_id,
        "baseline snapshot id does not match extraction snapshot"
    );
    validate_embeddings(corpus)?;
    validate_fixture_keys(corpus)?;
    validate_extraction_snapshot(snapshot)?;
    validate_coverage(corpus)?;
    Ok(())
}

fn validate_extraction_snapshot(snapshot: &ExtractionSnapshot) -> Result<()> {
    let gold = unique_keys(
        snapshot.gold_fact_keys.iter().map(String::as_str),
        "gold fact",
    )?;
    let _candidate_ids = unique_keys(
        snapshot
            .candidates
            .iter()
            .map(|candidate| candidate.id.as_str()),
        "extraction candidate",
    )?;
    for candidate in &snapshot.candidates {
        ensure!(
            !candidate.content.trim().is_empty(),
            "extraction candidate {} has empty content",
            candidate.id
        );
        if let Some(key) = &candidate.matched_gold_key {
            ensure!(
                gold.contains(key),
                "extraction candidate {} names unknown gold fact {key}",
                candidate.id
            );
        }
    }
    Ok(())
}

fn validate_embeddings(corpus: &CorpusFixture) -> Result<()> {
    ensure!(
        !corpus.embeddings.is_empty(),
        "corpus requires deterministic embeddings"
    );
    for (key, embedding) in &corpus.embeddings {
        ensure!(
            embedding.dimensions == 384,
            "embedding {key} must declare 384 dimensions"
        );
        let mut seen = BTreeSet::new();
        for &(index, value) in &embedding.values {
            ensure!(
                index < embedding.dimensions,
                "embedding {key} index {index} is out of range"
            );
            ensure!(
                value.is_finite(),
                "embedding {key} contains a non-finite value"
            );
            ensure!(seen.insert(index), "embedding {key} repeats index {index}");
        }
    }
    Ok(())
}

fn validate_fixture_keys(corpus: &CorpusFixture) -> Result<()> {
    let message_keys = unique_keys(
        corpus.messages.iter().map(|message| message.key.as_str()),
        "message",
    )?;
    let mut memory_keys = unique_keys(
        corpus
            .extracted_memories
            .iter()
            .map(|memory| memory.key.as_str()),
        "memory",
    )?;
    for memory in &corpus.canonical_memories {
        ensure!(
            memory_keys.insert(memory.key.clone()),
            "duplicate memory key {}",
            memory.key
        );
    }
    for memory in &corpus.extracted_memories {
        ensure!(
            corpus.embeddings.contains_key(&memory.embedding_key),
            "memory {} has unknown embedding",
            memory.key
        );
        for source in &memory.source_message_keys {
            ensure!(
                message_keys.contains(source),
                "memory {} has unknown source {source}",
                memory.key
            );
        }
        for source in &memory.expected_citation_message_keys {
            ensure!(
                memory.source_message_keys.contains(source),
                "memory {} expects a citation outside its sources",
                memory.key
            );
        }
    }
    for memory in &corpus.canonical_memories {
        ensure!(
            corpus.embeddings.contains_key(&memory.embedding_key),
            "canonical {} has unknown embedding",
            memory.key
        );
    }
    for query in &corpus.queries {
        if let Some(key) = &query.embedding_key {
            ensure!(
                corpus.embeddings.contains_key(key),
                "query {} has unknown embedding",
                query.id
            );
        }
        if let Some(key) = &query.exact_memory_key {
            ensure!(
                memory_keys.contains(key),
                "query {} has unknown exact memory",
                query.id
            );
        }
    }
    Ok(())
}

fn unique_keys<'a>(keys: impl Iterator<Item = &'a str>, label: &str) -> Result<BTreeSet<String>> {
    let mut result = BTreeSet::new();
    for key in keys {
        ensure!(!key.is_empty(), "{label} key cannot be empty");
        ensure!(
            result.insert(key.to_string()),
            "duplicate {label} key {key}"
        );
    }
    Ok(result)
}

fn validate_coverage(corpus: &CorpusFixture) -> Result<()> {
    const REQUIRED: &[&str] = &[
        "multilingual-de-en",
        "paraphrase",
        "long-gap",
        "stable-principal-rename",
        "shared-group",
        "canonical-prompt-accounting",
        "canonical-preference",
        "correction-stale",
        "attachment-reference",
        "adversarial-cross-context",
        "same-principal-current-context-scope",
        "admin-current-context-scope",
        "exact-denial",
        "citation-support",
        "bounded-scan",
    ];
    let declared = corpus
        .coverage
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    for required in REQUIRED {
        ensure!(
            declared.contains(required),
            "corpus is missing required coverage tag {required}"
        );
    }
    ensure!(
        corpus.noise_series.iter().any(|series| series.count > 200),
        "bounded-scan coverage requires more than 200 candidates"
    );
    ensure!(
        !corpus.attachment_references.is_empty(),
        "attachment-reference coverage requires metadata"
    );
    ensure!(
        !corpus.denials.is_empty(),
        "exact-denial coverage requires a denial edge"
    );
    ensure!(corpus.canonical_memories.iter().any(|memory| memory.freshness == Freshness::Stale && memory.correction_group.is_some()), "correction-stale coverage requires an annotated stale assertion");
    Ok(())
}

fn seed_corpus(corpus: &CorpusFixture) -> Result<SeededCorpus> {
    let graph = MemoryGraph::in_memory()?;
    for message in &corpus.messages {
        graph.ingest_message(&source_message(message), FIXTURE_ADMIN)?;
    }
    for attachment in &corpus.attachment_references {
        seed_attachment_reference(&graph, attachment)?;
    }
    verify_attachment_references(&graph, &corpus.attachment_references)?;
    let mut seeded = SeededCorpus {
        graph,
        id_by_key: BTreeMap::new(),
        key_by_id: BTreeMap::new(),
        freshness_by_key: BTreeMap::new(),
        citation_ids_by_key: BTreeMap::new(),
    };
    seed_extracted_memories(corpus, &mut seeded)?;
    seed_canonical_memories(corpus, &mut seeded)?;
    seed_noise(corpus, &mut seeded)?;
    apply_denials(corpus, &seeded)?;
    Ok(seeded)
}

fn source_message(message: &MessageFixture) -> SourceMessage {
    SourceMessage {
        id: format!("fixture-message:{}", message.key),
        platform_message_id: Some(format!("fixture-platform:{}", message.key)),
        context_id: message.context_id.clone(),
        context_kind: message.context_kind.clone(),
        speaker_id: message.speaker_id.clone(),
        speaker_name: message.speaker_name.clone(),
        role: message.role.clone(),
        content: message.content.clone(),
        created_at_ms: message.created_at_ms,
    }
}

fn seed_attachment_reference(
    graph: &MemoryGraph,
    fixture: &AttachmentReferenceFixture,
) -> Result<()> {
    let connection = Connection::new(&graph.database)?;
    let message_id = format!("fixture-message:{}", fixture.message_key);
    let attachment_id = format!("fixture-attachment:{}", fixture.message_key);
    let reference = serde_json::json!({"kind": fixture.reference_kind}).to_string();
    execute(
        &connection,
        "MATCH (message:Message {id: $message_id}) CREATE (attachment:Attachment {id: $attachment_id, sha256: 'synthetic-not-a-hash', mime_type: $mime_type, storage_path: '', platform_ref: $platform_ref, created_at_ms: 0}) CREATE (message)-[:HAS_ATTACHMENT]->(attachment)",
        vec![
            ("message_id", message_id.into()),
            ("attachment_id", attachment_id.into()),
            ("mime_type", fixture.mime_type.clone().into()),
            ("platform_ref", reference.into()),
        ],
    )
}

fn verify_attachment_references(
    graph: &MemoryGraph,
    fixtures: &[AttachmentReferenceFixture],
) -> Result<()> {
    let connection = Connection::new(&graph.database)?;
    let mut statement = connection.prepare(
        "MATCH (message:Message {id: $message_id})-[:HAS_ATTACHMENT]->(attachment:Attachment) RETURN count(attachment)",
    )?;
    for fixture in fixtures {
        let count = connection
            .execute(
                &mut statement,
                vec![(
                    "message_id",
                    format!("fixture-message:{}", fixture.message_key).into(),
                )],
            )?
            .next()
            .and_then(|row| row.first().and_then(|value| int64_value(value).ok()))
            .unwrap_or(0);
        ensure!(
            count == 1,
            "fixture message {} must have exactly one attachment reference",
            fixture.message_key
        );
    }
    Ok(())
}

fn seed_extracted_memories(corpus: &CorpusFixture, seeded: &mut SeededCorpus) -> Result<()> {
    for fixture in &corpus.extracted_memories {
        let draft = MemoryDraft {
            content: fixture.content.clone(),
            memory_type: fixture.memory_type.clone(),
            confidence: fixture.confidence,
            embedding: embedding(corpus, &fixture.embedding_key)?,
            origin_context_ids: fixture.origin_context_ids.clone(),
            source_message_ids: fixture
                .source_message_keys
                .iter()
                .map(|key| format!("fixture-message:{key}"))
                .collect(),
            extraction_run_id: format!("fixture-run:{}", fixture.extraction_run_id),
            extraction_model: fixture.extraction_model.clone(),
            created_at_ms: fixture.created_at_ms,
        };
        let id = seeded.graph.create_index_memory(&draft)?;
        register_memory(seeded, &fixture.key, id, fixture.freshness);
        seeded.citation_ids_by_key.insert(
            fixture.key.clone(),
            fixture
                .expected_citation_message_keys
                .iter()
                .map(|key| format!("fixture-platform:{key}"))
                .collect(),
        );
    }
    Ok(())
}

fn seed_canonical_memories(corpus: &CorpusFixture, seeded: &mut SeededCorpus) -> Result<()> {
    for fixture in &corpus.canonical_memories {
        let draft = CanonicalDraft {
            content: fixture.content.clone(),
            context_id: fixture.context_id.clone(),
            principal_id: fixture.principal_id.clone(),
            embedding: embedding(corpus, &fixture.embedding_key)?,
            scope: match fixture.scope {
                FixtureScope::Context => MemoryScope::Context,
                FixtureScope::Global => MemoryScope::Global,
            },
            created_at_ms: fixture.created_at_ms,
        };
        let id = seeded.graph.remember_canonical(&draft)?;
        register_memory(seeded, &fixture.key, id, fixture.freshness);
    }
    Ok(())
}

fn seed_noise(corpus: &CorpusFixture, seeded: &mut SeededCorpus) -> Result<()> {
    for series in &corpus.noise_series {
        for index in 0..series.count {
            let key = format!("{}-{index:03}", series.key_prefix);
            let draft = MemoryDraft {
                content: format!("{} {index:03}", series.content_prefix),
                memory_type: "fixture-noise".to_string(),
                confidence: 0.5,
                embedding: embedding(corpus, &series.embedding_key)?,
                origin_context_ids: vec![series.context_id.clone()],
                source_message_ids: vec![format!("fixture-message:{}", series.source_message_key)],
                extraction_run_id: format!("fixture-run:{}", series.key_prefix),
                extraction_model: "fixture-noise-generator-v1".to_string(),
                created_at_ms: series.created_at_ms_start + index as i64,
            };
            let id = seeded.graph.create_index_memory(&draft)?;
            register_memory(seeded, &key, id, Freshness::Neutral);
            seeded.citation_ids_by_key.insert(
                key,
                BTreeSet::from([format!("fixture-platform:{}", series.source_message_key)]),
            );
        }
    }
    Ok(())
}

fn register_memory(seeded: &mut SeededCorpus, key: &str, id: String, freshness: Freshness) {
    seeded.id_by_key.insert(key.to_string(), id.clone());
    seeded.key_by_id.insert(id, key.to_string());
    seeded.freshness_by_key.insert(key.to_string(), freshness);
}

fn apply_denials(corpus: &CorpusFixture, seeded: &SeededCorpus) -> Result<()> {
    let connection = Connection::new(&seeded.graph.database)?;
    for denial in &corpus.denials {
        let memory_id = seeded
            .id_by_key
            .get(&denial.memory_key)
            .with_context(|| format!("unknown denial memory {}", denial.memory_key))?;
        execute(
            &connection,
            "MATCH (memory:Memory {id: $memory_id}), (principal:Principal {id: $principal_id}) MERGE (memory)-[:DENIED_TO]->(principal)",
            vec![
                ("memory_id", memory_id.clone().into()),
                ("principal_id", denial.principal_id.clone().into()),
            ],
        )?;
    }
    Ok(())
}

fn embedding(corpus: &CorpusFixture, key: &str) -> Result<Vec<f32>> {
    let sparse = corpus
        .embeddings
        .get(key)
        .with_context(|| format!("unknown embedding {key}"))?;
    let mut values = vec![0.0; sparse.dimensions];
    for &(index, value) in &sparse.values {
        values[index] = value;
    }
    Ok(values)
}

fn evaluate_retrieval(
    corpus: &CorpusFixture,
    seeded: &SeededCorpus,
) -> Result<RetrievalEvaluation> {
    let mut aggregate = RetrievalMetrics {
        query_count: corpus.queries.len(),
        returned_count: 0,
        true_positive_count: 0,
        false_positive_count: 0,
        false_negative_count: 0,
        precision: 0.0,
        recall: 0.0,
        unauthorized_recall_count: 0,
        unsupported_citation_count: 0,
        stale_memory_count: 0,
        freshness_scored_count: 0,
        stale_memory_rate: 0.0,
    };
    let mut canonical_prompt_chars = 0usize;
    let mut ranked_recall_prompt_chars = 0usize;
    let mut output_chars = 0usize;
    let mut latencies = Vec::new();
    let mut cases = Vec::new();
    for query in &corpus.queries {
        let started = Instant::now();
        let (returned, canonical_prompt, ranked_prompt, output, bounded) =
            run_query(corpus, seeded, query)?;
        latencies.push(started.elapsed().as_micros().min(u128::from(u64::MAX)) as u64);
        let case = score_case(
            query,
            &returned,
            seeded,
            canonical_prompt,
            ranked_prompt,
            output,
            bounded,
        );
        aggregate_case(&mut aggregate, &case, &returned, seeded);
        canonical_prompt_chars += canonical_prompt;
        ranked_recall_prompt_chars += ranked_prompt;
        output_chars += output;
        cases.push(case);
    }
    aggregate.precision = ratio(
        aggregate.true_positive_count,
        aggregate.true_positive_count + aggregate.false_positive_count,
    );
    aggregate.recall = ratio(
        aggregate.true_positive_count,
        aggregate.true_positive_count + aggregate.false_negative_count,
    );
    aggregate.stale_memory_rate = ratio(
        aggregate.stale_memory_count,
        aggregate.freshness_scored_count,
    );
    Ok(RetrievalEvaluation {
        metrics: aggregate,
        canonical_prompt_chars,
        ranked_recall_prompt_chars,
        output_chars,
        latencies_us: latencies,
        cases,
    })
}

fn run_query(
    corpus: &CorpusFixture,
    seeded: &SeededCorpus,
    query: &QueryFixture,
) -> Result<(Vec<ReturnedMemory>, usize, usize, usize, bool)> {
    let canonical = seeded.graph.canonical_for_context(
        &query.principal_id,
        &query.context_id,
        CANONICAL_MEMORY_LIMIT,
        DEFAULT_CANONICAL_MEMORY_CHAR_BUDGET,
    )?;
    let canonical_prompt_chars = render_canonical_prompt(&canonical).chars().count();
    let query_text = query
        .exact_memory_key
        .as_ref()
        .map(|key| {
            seeded
                .id_by_key
                .get(key)
                .cloned()
                .with_context(|| format!("query {} exact key is not seeded", query.id))
        })
        .transpose()?
        .unwrap_or_else(|| query.query.clone());
    let query_embedding = query
        .embedding_key
        .as_ref()
        .map(|key| embedding(corpus, key))
        .transpose()?;
    match query.surface {
        QuerySurface::Inspect => {
            let inspection = seeded.graph.inspect(
                &query.principal_id,
                &query.context_id,
                &query_text,
                query_embedding.as_deref(),
            )?;
            let output = inspect::render_inspection(&inspection);
            let bounded = inspection.candidate_scan_truncated;
            let returned = inspection
                .canonical
                .iter()
                .chain(&inspection.extracted)
                .map(|memory| returned_inspection_memory(memory, seeded))
                .collect::<Result<Vec<_>>>()?;
            Ok((
                returned,
                canonical_prompt_chars,
                0,
                output.chars().count(),
                bounded,
            ))
        }
        QuerySurface::RecallHybrid => {
            let vector = query_embedding.context("recall_hybrid query requires an embedding")?;
            let recalled = seeded.graph.recall_authorized_hybrid(
                &query.principal_id,
                &query_text,
                &vector,
                query.limit.unwrap_or(5),
            )?;
            let prompt = render_recall_prompt(&recalled);
            let returned = recalled
                .iter()
                .map(|memory| returned_recalled_memory(memory, seeded))
                .collect::<Result<Vec<_>>>()?;
            Ok((
                returned,
                canonical_prompt_chars,
                prompt.chars().count(),
                0,
                false,
            ))
        }
    }
}

fn returned_inspection_memory(
    memory: &inspect::InspectedMemory,
    seeded: &SeededCorpus,
) -> Result<ReturnedMemory> {
    let key = seeded
        .key_by_id
        .get(&memory.id)
        .cloned()
        .with_context(|| format!("inspection returned unseeded memory {}", memory.id))?;
    Ok(ReturnedMemory {
        key,
        canonical: memory.canonical,
        citation_source_ids: memory
            .citations
            .iter()
            .map(|citation| citation.source_id.clone())
            .collect(),
    })
}

fn returned_recalled_memory(
    memory: &RecalledMemory,
    seeded: &SeededCorpus,
) -> Result<ReturnedMemory> {
    let key = seeded
        .key_by_id
        .get(&memory.id)
        .cloned()
        .with_context(|| format!("recall returned unseeded memory {}", memory.id))?;
    let citations = if memory.citation.is_empty() {
        Vec::new()
    } else {
        memory
            .citation
            .split(" | ")
            .map(|citation| citation.rsplit("; ").next().unwrap_or_default().to_string())
            .collect()
    };
    Ok(ReturnedMemory {
        key,
        canonical: memory.canonical,
        citation_source_ids: citations,
    })
}

fn score_case(
    query: &QueryFixture,
    returned: &[ReturnedMemory],
    seeded: &SeededCorpus,
    canonical_prompt_chars: usize,
    ranked_recall_prompt_chars: usize,
    output_chars: usize,
    bounded_scan_truncated: bool,
) -> CaseResult {
    let expected = query
        .expected_memory_keys
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let returned_keys = returned
        .iter()
        .map(|memory| memory.key.clone())
        .collect::<BTreeSet<_>>();
    let forbidden = query
        .forbidden_memory_keys
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let true_positive_count = returned_keys.intersection(&expected).count();
    let false_positive_count = returned_keys.difference(&expected).count();
    let false_negative_count = expected.difference(&returned_keys).count();
    let unauthorized_recall_count = returned_keys.intersection(&forbidden).count();
    let unsupported_citation_count = unsupported_citations(returned, &seeded.citation_ids_by_key);
    CaseResult {
        id: query.id.clone(),
        surface: match query.surface {
            QuerySurface::Inspect => "inspect",
            QuerySurface::RecallHybrid => "recall_hybrid",
        }
        .to_string(),
        expected_memory_keys: expected.into_iter().collect(),
        returned_memory_keys: returned.iter().map(|memory| memory.key.clone()).collect(),
        true_positive_count,
        false_positive_count,
        false_negative_count,
        unauthorized_recall_count,
        unsupported_citation_count,
        prompt_chars: canonical_prompt_chars.saturating_add(ranked_recall_prompt_chars),
        canonical_prompt_chars,
        ranked_recall_prompt_chars,
        output_chars,
        bounded_scan_truncated,
    }
}

fn unsupported_citations(
    returned: &[ReturnedMemory],
    expected_by_key: &BTreeMap<String, BTreeSet<String>>,
) -> usize {
    returned
        .iter()
        .filter(|memory| !memory.canonical)
        .map(|memory| {
            let Some(expected) = expected_by_key.get(&memory.key) else {
                return 1;
            };
            if memory.citation_source_ids.is_empty() {
                return 1;
            }
            memory
                .citation_source_ids
                .iter()
                .filter(|source| !expected.contains(*source))
                .count()
        })
        .sum()
}

fn aggregate_case(
    aggregate: &mut RetrievalMetrics,
    case: &CaseResult,
    returned: &[ReturnedMemory],
    seeded: &SeededCorpus,
) {
    aggregate.returned_count += returned.len();
    aggregate.true_positive_count += case.true_positive_count;
    aggregate.false_positive_count += case.false_positive_count;
    aggregate.false_negative_count += case.false_negative_count;
    aggregate.unauthorized_recall_count += case.unauthorized_recall_count;
    aggregate.unsupported_citation_count += case.unsupported_citation_count;
    for memory in returned {
        match seeded
            .freshness_by_key
            .get(&memory.key)
            .copied()
            .unwrap_or(Freshness::Neutral)
        {
            Freshness::Current => aggregate.freshness_scored_count += 1,
            Freshness::Stale => {
                aggregate.freshness_scored_count += 1;
                aggregate.stale_memory_count += 1;
            }
            Freshness::Neutral => {}
        }
    }
}

fn evaluate_extraction(snapshot: &ExtractionSnapshot) -> ExtractionMetrics {
    let gold = snapshot
        .gold_fact_keys
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let supported_candidate_count = snapshot
        .candidates
        .iter()
        .filter(|candidate| {
            candidate
                .matched_gold_key
                .as_ref()
                .is_some_and(|key| gold.contains(key))
        })
        .count();
    let recovered = snapshot
        .candidates
        .iter()
        .filter_map(|candidate| candidate.matched_gold_key.as_ref())
        .filter(|key| gold.contains(*key))
        .cloned()
        .collect::<BTreeSet<_>>();
    ExtractionMetrics {
        gold_fact_count: gold.len(),
        candidate_count: snapshot.candidates.len(),
        supported_candidate_count,
        recovered_gold_count: recovered.len(),
        precision: ratio(supported_candidate_count, snapshot.candidates.len()),
        recall: ratio(recovered.len(), gold.len()),
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        return 1.0;
    }
    ((numerator as f64 / denominator as f64) * 1_000_000.0).round() / 1_000_000.0
}

fn latency_metrics(mut samples: Vec<u64>) -> LatencyMetrics {
    samples.sort_unstable();
    LatencyMetrics {
        unit: "microseconds".to_string(),
        sample_count: samples.len(),
        p50: percentile(&samples, 50),
        p95: percentile(&samples, 95),
        max: samples.last().copied().unwrap_or(0),
    }
}

fn percentile(samples: &[u64], percent: usize) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let rank = (percent.saturating_mul(samples.len()).saturating_add(99) / 100).max(1);
    samples[rank.saturating_sub(1).min(samples.len() - 1)]
}

fn threshold_violations(
    quality: &QualityMetrics,
    latency: &LatencyMetrics,
    thresholds: &Thresholds,
) -> Vec<String> {
    let mut violations = Vec::new();
    check_min(
        &mut violations,
        "retrieval.precision",
        quality.retrieval.precision,
        thresholds.min_retrieval_precision,
    );
    check_min(
        &mut violations,
        "retrieval.recall",
        quality.retrieval.recall,
        thresholds.min_retrieval_recall,
    );
    check_max_usize(
        &mut violations,
        "retrieval.unauthorized_recall_count",
        quality.retrieval.unauthorized_recall_count,
        thresholds.max_unauthorized_recall_count,
    );
    check_max_usize(
        &mut violations,
        "retrieval.unsupported_citation_count",
        quality.retrieval.unsupported_citation_count,
        thresholds.max_unsupported_citation_count,
    );
    check_max(
        &mut violations,
        "retrieval.stale_memory_rate",
        quality.retrieval.stale_memory_rate,
        thresholds.max_stale_memory_rate,
    );
    check_min(
        &mut violations,
        "extraction.precision",
        quality.extraction.precision,
        thresholds.min_extraction_precision,
    );
    check_min(
        &mut violations,
        "extraction.recall",
        quality.extraction.recall,
        thresholds.min_extraction_recall,
    );
    check_max_usize(
        &mut violations,
        "cost.prompt_chars",
        quality.cost.prompt_chars,
        thresholds.max_prompt_chars,
    );
    check_max_usize(
        &mut violations,
        "cost.output_chars",
        quality.cost.output_chars,
        thresholds.max_output_chars,
    );
    check_max_usize(
        &mut violations,
        "cost.estimated_tokens",
        quality.cost.estimated_tokens,
        thresholds.max_estimated_tokens,
    );
    if latency.p95 > thresholds.max_latency_p95_us {
        violations.push(format!(
            "latency.p95={} exceeds {} microseconds",
            latency.p95, thresholds.max_latency_p95_us
        ));
    }
    if latency.max > thresholds.max_latency_max_us {
        violations.push(format!(
            "latency.max={} exceeds {} microseconds",
            latency.max, thresholds.max_latency_max_us
        ));
    }
    violations
}

fn check_min(violations: &mut Vec<String>, name: &str, actual: f64, threshold: f64) {
    if actual < threshold {
        violations.push(format!("{name}={actual:.6} is below {threshold:.6}"));
    }
}

fn check_max(violations: &mut Vec<String>, name: &str, actual: f64, threshold: f64) {
    if actual > threshold {
        violations.push(format!("{name}={actual:.6} exceeds {threshold:.6}"));
    }
}

fn check_max_usize(violations: &mut Vec<String>, name: &str, actual: usize, threshold: usize) {
    if actual > threshold {
        violations.push(format!("{name}={actual} exceeds {threshold}"));
    }
}

fn report_json(report: &QualityReport) -> Result<String> {
    serde_json::to_string_pretty(report).context("serialize memory quality report")
}

fn print_human_summary(report: &QualityReport) {
    let status = if report.baseline_comparison.passed {
        "PASS"
    } else {
        "FAIL"
    };
    eprintln!(
        "memory quality {status}: retrieval p/r {:.3}/{:.3}, unauthorized {}, unsupported citations {}, stale {:.3}; extraction p/r {:.3}/{:.3}; chars prompt/output {}/{}, estimated tokens {}; latency us p50/p95/max {}/{}/{}",
        report.quality.retrieval.precision,
        report.quality.retrieval.recall,
        report.quality.retrieval.unauthorized_recall_count,
        report.quality.retrieval.unsupported_citation_count,
        report.quality.retrieval.stale_memory_rate,
        report.quality.extraction.precision,
        report.quality.extraction.recall,
        report.quality.cost.prompt_chars,
        report.quality.cost.output_chars,
        report.quality.cost.estimated_tokens,
        report.latency.p50,
        report.latency.p95,
        report.latency.max,
    );
    if !report.baseline_comparison.exact_quality_match {
        eprintln!(
            "quality metrics differ from baseline {}; thresholds still decide the gate",
            report.baseline_comparison.baseline_id
        );
    }
    for violation in &report.baseline_comparison.violations {
        eprintln!("violation: {violation}");
    }
}

fn enforce_gate(report: &QualityReport) -> Result<()> {
    ensure!(
        report.baseline_comparison.passed,
        "memory quality thresholds failed: {}",
        report.baseline_comparison.violations.join("; ")
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixtures() -> (CorpusFixture, ExtractionSnapshot, Baseline) {
        (
            serde_json::from_str(include_str!(
                "../../tests/fixtures/memory-quality/v1/corpus.json"
            ))
            .unwrap(),
            serde_json::from_str(include_str!(
                "../../tests/fixtures/memory-quality/v1/extraction-snapshot.json"
            ))
            .unwrap(),
            serde_json::from_str(include_str!(
                "../../tests/fixtures/memory-quality/v1/baseline.json"
            ))
            .unwrap(),
        )
    }

    fn principal_only_can_read(
        seeded: &SeededCorpus,
        principal_id: &str,
        memory_key: &str,
    ) -> Result<bool> {
        let memory_id = seeded
            .id_by_key
            .get(memory_key)
            .with_context(|| format!("missing fixture memory {memory_key}"))?;
        let connection = Connection::new(&seeded.graph.database)?;
        let mut statement = connection.prepare(
            "MATCH (principal:Principal {id: $principal_id})-[membership:MEMBER_OF]->(access_group:AccessGroup)<-[:GOVERNED_BY]-(memory:Memory {id: $memory_id})
             WHERE membership.can_read = true AND memory.status = 'active'
               AND NOT (memory)-[:DENIED_TO]->(principal)
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
                    ("memory_id", memory_id.clone().into()),
                ],
            )?
            .next()
            .and_then(|row| row.first().and_then(|value| int64_value(value).ok()))
            .unwrap_or(0);
        Ok(count > 0)
    }

    #[test]
    fn versioned_fixture_covers_required_scenarios() -> Result<()> {
        let (corpus, snapshot, baseline) = fixtures();
        validate_inputs(&corpus, &snapshot, &baseline)?;
        let renamed = corpus
            .messages
            .iter()
            .filter(|message| message.speaker_id == "principal:traveler-a")
            .map(|message| message.speaker_name.as_str())
            .collect::<BTreeSet<_>>();
        assert!(
            renamed.len() >= 2,
            "stable principal rename fixture is not discriminating"
        );
        let oldest = corpus
            .messages
            .iter()
            .map(|message| message.created_at_ms)
            .min()
            .unwrap();
        let newest = corpus
            .messages
            .iter()
            .map(|message| message.created_at_ms)
            .max()
            .unwrap();
        assert!(
            newest - oldest > 30_000_000_000,
            "long-gap fixture is too short"
        );
        assert!(
            corpus
                .queries
                .iter()
                .any(|query| !query.forbidden_memory_keys.is_empty())
        );
        for (case_id, forbidden_key) in [
            ("same-principal-cross-context-exact", "timezone-current"),
            ("admin-cross-context-exact", "attachment-handle"),
        ] {
            let query = corpus
                .queries
                .iter()
                .find(|query| query.id == case_id)
                .unwrap_or_else(|| panic!("missing context-scope fixture case {case_id}"));
            assert!(
                query.expected_memory_keys.is_empty(),
                "{case_id} must expect no foreign-context memory"
            );
            assert_eq!(query.forbidden_memory_keys, [forbidden_key]);
        }
        Ok(())
    }

    #[test]
    fn deterministic_harness_meets_acl_citation_and_performance_gates() -> Result<()> {
        let (corpus, snapshot, baseline) = fixtures();
        let mut first = evaluate(&corpus, &snapshot, &baseline)?;
        let mut second = evaluate(&corpus, &snapshot, &baseline)?;
        assert_eq!(first.quality.retrieval.unauthorized_recall_count, 0);
        assert_eq!(first.thresholds.max_unauthorized_recall_count, 0);
        assert_eq!(first.quality.retrieval.unsupported_citation_count, 0);
        assert!(first.cases.iter().any(|case| case.bounded_scan_truncated));
        for case_id in [
            "same-principal-cross-context-exact",
            "admin-cross-context-exact",
        ] {
            let case = first
                .cases
                .iter()
                .find(|case| case.id == case_id)
                .unwrap_or_else(|| panic!("missing evaluated context-scope case {case_id}"));
            assert!(case.returned_memory_keys.is_empty());
            assert_eq!(case.unauthorized_recall_count, 0);
        }
        let canonical_cost_case = first
            .cases
            .iter()
            .find(|case| case.id == "travel-german-paraphrase")
            .expect("missing canonical prompt accounting case");
        assert!(canonical_cost_case.canonical_prompt_chars > 0);
        assert_eq!(
            canonical_cost_case.prompt_chars,
            canonical_cost_case
                .canonical_prompt_chars
                .saturating_add(canonical_cost_case.ranked_recall_prompt_chars)
        );
        assert!(first.quality.cost.canonical_prompt_chars > 0);
        assert_eq!(
            first.quality.cost.prompt_chars,
            first
                .quality
                .cost
                .canonical_prompt_chars
                .saturating_add(first.quality.cost.ranked_recall_prompt_chars)
        );
        assert!(
            first.baseline_comparison.passed,
            "{:?}",
            first.baseline_comparison.violations
        );
        assert!(first.baseline_comparison.exact_quality_match);
        first.latency = LatencyMetrics {
            unit: "microseconds".into(),
            sample_count: 0,
            p50: 0,
            p95: 0,
            max: 0,
        };
        second.latency = first.latency.clone();
        assert_eq!(report_json(&first)?, report_json(&second)?);
        let json = report_json(&first)?;
        assert!(json.find("\"schema_version\"").unwrap() < json.find("\"quality\"").unwrap());
        assert_eq!(
            first.quality.cost.token_estimate_definition,
            TOKEN_ESTIMATE_DEFINITION
        );
        Ok(())
    }

    #[test]
    fn exact_context_scope_cases_discriminate_the_context_clause() -> Result<()> {
        let (corpus, _, _) = fixtures();
        let seeded = seed_corpus(&corpus)?;
        for case_id in [
            "same-principal-cross-context-exact",
            "admin-cross-context-exact",
        ] {
            let query = corpus
                .queries
                .iter()
                .find(|query| query.id == case_id)
                .unwrap_or_else(|| panic!("missing context-scope fixture case {case_id}"));
            let foreign_key = query
                .forbidden_memory_keys
                .first()
                .expect("context-scope case needs a forbidden memory");
            assert!(
                principal_only_can_read(&seeded, &query.principal_id, foreign_key)?,
                "{case_id} does not discriminate the current-context clause"
            );
            let (returned, _, _, _, _) = run_query(&corpus, &seeded, query)?;
            assert!(
                returned.is_empty(),
                "{case_id} leaked a foreign-context memory"
            );
        }
        Ok(())
    }

    #[test]
    fn unsupported_citations_are_counted() {
        let returned = vec![ReturnedMemory {
            key: "fact".into(),
            canonical: false,
            citation_source_ids: vec!["fixture-platform:wrong".into()],
        }];
        let expected = BTreeMap::from([(
            "fact".into(),
            BTreeSet::from(["fixture-platform:right".into()]),
        )]);
        assert_eq!(unsupported_citations(&returned, &expected), 1);
    }

    #[test]
    fn extraction_metrics_are_independent_from_retrieval_metrics() -> Result<()> {
        let (corpus, mut snapshot, baseline) = fixtures();
        let report = evaluate(&corpus, &snapshot, &baseline)?;
        let retrieval = report.quality.retrieval.clone();
        snapshot.candidates.clear();
        let changed = evaluate_extraction(&snapshot);
        assert_ne!(report.quality.extraction, changed);
        assert_eq!(report.quality.retrieval, retrieval);
        Ok(())
    }

    #[test]
    fn threshold_failure_is_a_nonzero_gate_error() -> Result<()> {
        let (corpus, snapshot, baseline) = fixtures();
        let mut report = evaluate(&corpus, &snapshot, &baseline)?;
        report.quality.retrieval.unauthorized_recall_count = 1;
        report.baseline_comparison.violations =
            threshold_violations(&report.quality, &report.latency, &report.thresholds);
        report.baseline_comparison.passed = report.baseline_comparison.violations.is_empty();
        assert!(enforce_gate(&report).is_err());
        assert!(
            report
                .baseline_comparison
                .violations
                .iter()
                .any(|violation| violation.contains("unauthorized_recall_count"))
        );
        Ok(())
    }

    #[test]
    fn writing_a_report_never_rewrites_the_baseline() -> Result<()> {
        let (corpus, snapshot, baseline) = fixtures();
        let report = evaluate(&corpus, &snapshot, &baseline)?;
        let directory = tempfile::tempdir()?;
        let baseline_path = directory.path().join("baseline.json");
        let report_path = directory.path().join("report.json");
        std::fs::write(
            &baseline_path,
            include_bytes!("../../tests/fixtures/memory-quality/v1/baseline.json"),
        )?;
        let before = std::fs::read(&baseline_path)?;
        std::fs::write(&report_path, report_json(&report)?)?;
        assert_eq!(std::fs::read(&baseline_path)?, before);
        assert!(report_path.exists());
        Ok(())
    }
}
