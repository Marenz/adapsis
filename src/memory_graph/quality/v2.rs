//! Schema-v2 prompt-selection quality gate.
use super::super::*;
use crate::memory_graph::prompt_budget::{
    SEMANTIC_EQUIVALENCE_MIN_COSINE, compatibility_guard_allows,
};
use anyhow::{Context as _, Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const SCHEMA_VERSION: u32 = 2;
const DEFAULT_FIXTURE: &str = "tests/fixtures/memory-quality/v2/corpus.json";
const DEFAULT_SNAPSHOT: &str = "tests/fixtures/memory-quality/v2/extraction-snapshot.json";
const DEFAULT_BASELINE: &str = "tests/fixtures/memory-quality/v2/baseline.json";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Corpus {
    schema_version: u32,
    fixture_id: String,
    description: String,
    coverage: Vec<String>,
    model_evidence: ModelEvidence,
    embeddings: BTreeMap<String, SparseEmbedding>,
    messages: Vec<MessageFixture>,
    records: Vec<RecordFixture>,
    cases: Vec<CaseFixture>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ModelEvidence {
    model_id: String,
    model_revision: String,
    runtime: String,
    measured_at: String,
    source_note: String,
    manually_reviewed: bool,
    pairs: Vec<EvidencePair>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct EvidencePair {
    id: String,
    left: String,
    right: String,
    cosine: f64,
    expected_equivalent: bool,
    expected_guard_compatible: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SparseEmbedding {
    dimensions: usize,
    values: Vec<(usize, f32)>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MessageFixture {
    key: String,
    context_id: String,
    principal_id: String,
    created_at_ms: i64,
}

#[derive(Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Channel {
    Canonical,
    Ranked,
}

#[derive(Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Freshness {
    Current,
    Stale,
    Neutral,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RecordFixture {
    key: String,
    channel: Channel,
    content: String,
    padding_chars: usize,
    context_id: String,
    principal_id: String,
    embedding_key: String,
    source_message_key: Option<String>,
    correction_of: Option<String>,
    created_at_ms: i64,
    freshness: Freshness,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedStats {
    collapsed: usize,
    omitted_by_count: usize,
    omitted_by_chars: usize,
    omitted_beyond_scan: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CaseFixture {
    id: String,
    channel: Channel,
    principal_id: String,
    context_id: String,
    query: String,
    query_embedding_key: Option<String>,
    max_count: usize,
    max_row_chars: usize,
    expected_representative_keys: Vec<String>,
    forbidden_keys: Vec<String>,
    equivalence_groups: Vec<Vec<String>>,
    must_coexist: Vec<Vec<String>>,
    expected_stats: ExpectedStats,
    calibrates_ranked_budget: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    schema_version: u32,
    snapshot_id: String,
    fixture_id: String,
    provider_label: String,
    precision: f64,
    recall: f64,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Thresholds {
    min_retrieval_recall: f64,
    max_semantic_duplicate_count: usize,
    max_correction_precedence_violations: usize,
    max_contradiction_false_merge_count: usize,
    max_acl_leak_count: usize,
    max_unsupported_citation_count: usize,
    max_stale_memory_count: usize,
    max_omission_mismatch_count: usize,
    max_canonical_prompt_chars: usize,
    max_ranked_prompt_chars: usize,
    selected_semantic_cosine: f64,
    selected_ranked_row_chars: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Baseline {
    schema_version: u32,
    baseline_id: String,
    fixture_id: String,
    extraction_snapshot_id: String,
    thresholds: Thresholds,
    cosine_candidates: Vec<f64>,
    ranked_row_char_candidates: Vec<usize>,
}

struct Seeded {
    graph: MemoryGraph,
    id_by_key: BTreeMap<String, String>,
    key_by_id: BTreeMap<String, String>,
    record_by_key: BTreeMap<String, usize>,
}

#[derive(Default, Serialize)]
struct Metrics {
    case_count: usize,
    expected_representative_count: usize,
    selected_representative_count: usize,
    recovered_representative_count: usize,
    retrieval_recall: f64,
    semantic_duplicate_count: usize,
    correction_precedence_violations: usize,
    contradiction_false_merge_count: usize,
    acl_leak_count: usize,
    unsupported_citation_count: usize,
    stale_memory_count: usize,
    omission_mismatch_count: usize,
    canonical_prompt_chars: usize,
    ranked_prompt_chars: usize,
}

#[derive(Serialize)]
struct CaseReport {
    id: String,
    channel: String,
    max_count: usize,
    max_row_chars: usize,
    expected_representative_keys: Vec<String>,
    selected_representative_keys: Vec<String>,
    selection: PromptSelectionStatsReport,
    rendered_prompt_chars: usize,
}

#[derive(Serialize)]
struct PromptSelectionStatsReport {
    active_authorized_records: usize,
    candidate_records_scanned: usize,
    selected_representatives: usize,
    equivalent_records_collapsed: usize,
    omitted_by_count: usize,
    omitted_by_chars: usize,
    omitted_beyond_scan: usize,
    selected_row_chars: usize,
}

#[derive(Serialize)]
struct CosineSweepRow {
    threshold: f64,
    missed_equivalents: usize,
    false_merges: usize,
}

#[derive(Serialize)]
struct BudgetSweepRow {
    row_chars: usize,
    missed_expected_representatives: usize,
    rendered_prompt_chars: usize,
}

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    fixture_id: String,
    extraction_snapshot_id: String,
    extraction_provider_label: String,
    baseline_id: String,
    description: String,
    coverage: Vec<String>,
    model_evidence: ModelEvidence,
    thresholds: Thresholds,
    metrics: Metrics,
    cosine_sweep: Vec<CosineSweepRow>,
    ranked_row_char_sweep: Vec<BudgetSweepRow>,
    extraction_precision: f64,
    extraction_recall: f64,
    passed: bool,
    violations: Vec<String>,
    cases: Vec<CaseReport>,
}

pub(super) fn run_cli(
    fixture: Option<&Path>,
    snapshot: Option<&Path>,
    baseline: Option<&Path>,
    report_path: Option<&Path>,
) -> Result<()> {
    let fixture = selected_path(fixture, DEFAULT_FIXTURE);
    let snapshot = selected_path(snapshot, DEFAULT_SNAPSHOT);
    let baseline = selected_path(baseline, DEFAULT_BASELINE);
    let report = evaluate_paths(&fixture, &snapshot, &baseline)?;
    let json = serde_json::to_string_pretty(&report)?;
    println!("{json}");
    if let Some(path) = report_path {
        std::fs::write(path, format!("{json}\n"))
            .with_context(|| format!("write memory quality report {}", path.display()))?;
    }
    eprintln!(
        "memory quality v2: cases={} recall={:.6} duplicate={} precedence={} contradiction={} acl={} citation={} stale={} omission={} passed={}",
        report.metrics.case_count,
        report.metrics.retrieval_recall,
        report.metrics.semantic_duplicate_count,
        report.metrics.correction_precedence_violations,
        report.metrics.contradiction_false_merge_count,
        report.metrics.acl_leak_count,
        report.metrics.unsupported_citation_count,
        report.metrics.stale_memory_count,
        report.metrics.omission_mismatch_count,
        report.passed,
    );
    ensure!(
        report.passed,
        "memory quality v2 gate failed: {}",
        report.violations.join("; ")
    );
    Ok(())
}

fn selected_path(path: Option<&Path>, default: &str) -> PathBuf {
    path.map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(default))
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path, label: &str) -> Result<T> {
    let bytes = std::fs::read(path).with_context(|| format!("read {label} {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {label} {}", path.display()))
}

fn evaluate_paths(fixture: &Path, snapshot: &Path, baseline: &Path) -> Result<Report> {
    let corpus: Corpus = read_json(fixture, "memory quality v2 corpus")?;
    let snapshot: Snapshot = read_json(snapshot, "memory quality v2 extraction snapshot")?;
    let baseline: Baseline = read_json(baseline, "memory quality v2 baseline")?;
    validate(&corpus, &snapshot, &baseline)?;
    let seeded = seed(&corpus)?;
    evaluate(&corpus, &snapshot, &baseline, &seeded)
}

fn validate(corpus: &Corpus, snapshot: &Snapshot, baseline: &Baseline) -> Result<()> {
    ensure!(
        corpus.schema_version == SCHEMA_VERSION,
        "unsupported v2 corpus schema"
    );
    ensure!(
        snapshot.schema_version == SCHEMA_VERSION,
        "unsupported v2 snapshot schema"
    );
    ensure!(
        baseline.schema_version == SCHEMA_VERSION,
        "unsupported v2 baseline schema"
    );
    ensure!(
        snapshot.fixture_id == corpus.fixture_id,
        "snapshot fixture mismatch"
    );
    ensure!(
        baseline.fixture_id == corpus.fixture_id,
        "baseline fixture mismatch"
    );
    ensure!(
        baseline.extraction_snapshot_id == snapshot.snapshot_id,
        "snapshot id mismatch"
    );
    ensure!(
        !corpus.description.trim().is_empty(),
        "v2 description cannot be empty"
    );
    ensure!(
        corpus.model_evidence.manually_reviewed,
        "model evidence must be manually reviewed"
    );
    ensure!(
        baseline.thresholds.selected_semantic_cosine as f32 == SEMANTIC_EQUIVALENCE_MIN_COSINE,
        "baseline cosine does not match production constant"
    );
    ensure!(
        baseline.thresholds.selected_ranked_row_chars == DEFAULT_RANKED_MEMORY_CHAR_BUDGET,
        "baseline ranked budget does not match production default"
    );
    for (key, embedding) in &corpus.embeddings {
        ensure!(
            embedding.dimensions == 384,
            "embedding {key} must have 384 dimensions"
        );
        ensure!(
            embedding
                .values
                .iter()
                .all(|(index, value)| *index < 384 && value.is_finite()),
            "embedding {key} is invalid"
        );
    }
    for pair in &corpus.model_evidence.pairs {
        ensure!(
            compatibility_guard_allows(&pair.left, &pair.right) == pair.expected_guard_compatible,
            "frozen evidence {} guard expectation drifted",
            pair.id
        );
    }
    Ok(())
}

fn embedding(corpus: &Corpus, key: &str) -> Result<Vec<f32>> {
    let fixture = corpus
        .embeddings
        .get(key)
        .with_context(|| format!("missing embedding {key}"))?;
    let mut result = vec![0.0; fixture.dimensions];
    for &(index, value) in &fixture.values {
        result[index] = value;
    }
    Ok(result)
}

fn seed(corpus: &Corpus) -> Result<Seeded> {
    let graph = MemoryGraph::in_memory()?;
    for message in &corpus.messages {
        graph.ingest_message(
            &SourceMessage {
                id: message.key.clone(),
                platform_message_id: Some(message.key.clone()),
                context_id: message.context_id.clone(),
                context_kind: "synthetic".into(),
                speaker_id: message.principal_id.clone(),
                speaker_name: message.principal_id.clone(),
                role: "user".into(),
                content: format!("synthetic source {}", message.key),
                created_at_ms: message.created_at_ms,
            },
            "principal:fixture-admin",
        )?;
    }
    let mut seeded = Seeded {
        graph,
        id_by_key: BTreeMap::new(),
        key_by_id: BTreeMap::new(),
        record_by_key: BTreeMap::new(),
    };
    for (index, record) in corpus.records.iter().enumerate() {
        let id = seed_record(corpus, &seeded, record)?;
        seeded.id_by_key.insert(record.key.clone(), id.clone());
        seeded.key_by_id.insert(id, record.key.clone());
        seeded.record_by_key.insert(record.key.clone(), index);
    }
    Ok(seeded)
}

fn seed_record(corpus: &Corpus, seeded: &Seeded, record: &RecordFixture) -> Result<String> {
    let content = format!("{}{}", record.content, "x".repeat(record.padding_chars));
    let vector = embedding(corpus, &record.embedding_key)?;
    match (record.channel, record.correction_of.as_deref()) {
        (Channel::Canonical, Some(target_key)) => {
            let target_id = seeded.id_by_key.get(target_key).with_context(|| {
                format!("correction {} precedes target {target_key}", record.key)
            })?;
            seeded
                .graph
                .correct_memory(&CorrectionDraft {
                    target_id: target_id.clone(),
                    replacement: content,
                    context_id: record.context_id.clone(),
                    principal_id: record.principal_id.clone(),
                    may_write: true,
                    embedding: vector,
                    valid_from_ms: None,
                    valid_until_ms: None,
                    created_at_ms: record.created_at_ms,
                })?
                .with_context(|| format!("correction {} was unauthorized", record.key))
        }
        (Channel::Canonical, None) => seeded.graph.remember_canonical(&CanonicalDraft {
            content,
            context_id: record.context_id.clone(),
            principal_id: record.principal_id.clone(),
            embedding: vector,
            scope: MemoryScope::Context,
            created_at_ms: record.created_at_ms,
        }),
        (Channel::Ranked, None) => seeded.graph.create_index_memory(&MemoryDraft {
            content,
            memory_type: "synthetic".into(),
            confidence: 0.9,
            embedding: vector,
            origin_context_ids: vec![record.context_id.clone()],
            source_message_ids: vec![
                record
                    .source_message_key
                    .clone()
                    .with_context(|| format!("ranked record {} needs a source", record.key))?,
            ],
            extraction_run_id: format!("run:{}", record.key),
            extraction_model: "fixture-v2".into(),
            created_at_ms: record.created_at_ms,
        }),
        (Channel::Ranked, Some(_)) => {
            anyhow::bail!("ranked record {} cannot be a correction", record.key)
        }
    }
}

fn evaluate(
    corpus: &Corpus,
    snapshot: &Snapshot,
    baseline: &Baseline,
    seeded: &Seeded,
) -> Result<Report> {
    let mut metrics = Metrics::default();
    let mut cases = Vec::new();
    for case in &corpus.cases {
        let (selection, selected_keys, rendered_chars) =
            run_case(corpus, seeded, case, case.max_row_chars)?;
        score_case(
            corpus,
            seeded,
            case,
            &selection,
            &selected_keys,
            rendered_chars,
            &mut metrics,
        )?;
        cases.push(case_report(case, selection, selected_keys, rendered_chars));
    }
    metrics.case_count = cases.len();
    metrics.retrieval_recall = ratio(
        metrics.recovered_representative_count,
        metrics.expected_representative_count,
    );
    let cosine_sweep = cosine_sweep(corpus, baseline);
    let ranked_row_char_sweep = budget_sweep(corpus, seeded, baseline)?;
    let violations = violations(
        &metrics,
        &baseline.thresholds,
        &cosine_sweep,
        &ranked_row_char_sweep,
    );
    Ok(Report {
        schema_version: SCHEMA_VERSION,
        fixture_id: corpus.fixture_id.clone(),
        extraction_snapshot_id: snapshot.snapshot_id.clone(),
        extraction_provider_label: snapshot.provider_label.clone(),
        baseline_id: baseline.baseline_id.clone(),
        description: corpus.description.clone(),
        coverage: corpus.coverage.clone(),
        model_evidence: ModelEvidence {
            model_id: corpus.model_evidence.model_id.clone(),
            model_revision: corpus.model_evidence.model_revision.clone(),
            runtime: corpus.model_evidence.runtime.clone(),
            measured_at: corpus.model_evidence.measured_at.clone(),
            source_note: corpus.model_evidence.source_note.clone(),
            manually_reviewed: corpus.model_evidence.manually_reviewed,
            pairs: corpus
                .model_evidence
                .pairs
                .iter()
                .map(|pair| EvidencePair {
                    id: pair.id.clone(),
                    left: pair.left.clone(),
                    right: pair.right.clone(),
                    cosine: pair.cosine,
                    expected_equivalent: pair.expected_equivalent,
                    expected_guard_compatible: pair.expected_guard_compatible,
                })
                .collect(),
        },
        thresholds: Thresholds {
            min_retrieval_recall: baseline.thresholds.min_retrieval_recall,
            max_semantic_duplicate_count: baseline.thresholds.max_semantic_duplicate_count,
            max_correction_precedence_violations: baseline
                .thresholds
                .max_correction_precedence_violations,
            max_contradiction_false_merge_count: baseline
                .thresholds
                .max_contradiction_false_merge_count,
            max_acl_leak_count: baseline.thresholds.max_acl_leak_count,
            max_unsupported_citation_count: baseline.thresholds.max_unsupported_citation_count,
            max_stale_memory_count: baseline.thresholds.max_stale_memory_count,
            max_omission_mismatch_count: baseline.thresholds.max_omission_mismatch_count,
            max_canonical_prompt_chars: baseline.thresholds.max_canonical_prompt_chars,
            max_ranked_prompt_chars: baseline.thresholds.max_ranked_prompt_chars,
            selected_semantic_cosine: baseline.thresholds.selected_semantic_cosine,
            selected_ranked_row_chars: baseline.thresholds.selected_ranked_row_chars,
        },
        metrics,
        cosine_sweep,
        ranked_row_char_sweep,
        extraction_precision: snapshot.precision,
        extraction_recall: snapshot.recall,
        passed: violations.is_empty(),
        violations,
        cases,
    })
}

fn run_case(
    corpus: &Corpus,
    seeded: &Seeded,
    case: &CaseFixture,
    row_chars: usize,
) -> Result<(PromptMemorySelection, Vec<String>, usize)> {
    let selection = match case.channel {
        Channel::Canonical => seeded.graph.canonical_prompt_selection(
            &case.principal_id,
            &case.context_id,
            case.max_count,
            row_chars,
        )?,
        Channel::Ranked => seeded.graph.ranked_prompt_selection(
            &case.principal_id,
            &case.query,
            &embedding(
                corpus,
                case.query_embedding_key
                    .as_deref()
                    .context("ranked case needs query embedding")?,
            )?,
            case.max_count,
            row_chars,
        )?,
    };
    let selected_keys =
        selection
            .memories
            .iter()
            .map(|memory| {
                seeded.key_by_id.get(&memory.id).cloned().with_context(|| {
                    format!("case {} returned unknown memory {}", case.id, memory.id)
                })
            })
            .collect::<Result<Vec<_>>>()?;
    let rendered = match case.channel {
        Channel::Canonical => render_canonical_selection(&selection),
        Channel::Ranked => render_recall_selection(&selection),
    };
    Ok((selection, selected_keys, rendered.chars().count()))
}

fn score_case(
    corpus: &Corpus,
    seeded: &Seeded,
    case: &CaseFixture,
    selection: &PromptMemorySelection,
    selected: &[String],
    rendered_chars: usize,
    metrics: &mut Metrics,
) -> Result<()> {
    let expected = case
        .expected_representative_keys
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let selected_set = selected.iter().cloned().collect::<BTreeSet<_>>();
    metrics.expected_representative_count += expected.len();
    metrics.selected_representative_count += selected.len();
    metrics.recovered_representative_count += expected.intersection(&selected_set).count();
    metrics.acl_leak_count += case
        .forbidden_keys
        .iter()
        .filter(|key| selected_set.contains(*key))
        .count();
    metrics.correction_precedence_violations += case
        .forbidden_keys
        .iter()
        .filter(|key| {
            record(corpus, seeded, key).is_ok_and(|record| record.freshness == Freshness::Stale)
        })
        .filter(|key| selected_set.contains(*key))
        .count();
    metrics.stale_memory_count += selected
        .iter()
        .filter(|key| {
            record(corpus, seeded, key).is_ok_and(|record| record.freshness == Freshness::Stale)
        })
        .count();
    for group in &case.equivalence_groups {
        metrics.semantic_duplicate_count += group
            .iter()
            .filter(|key| selected_set.contains(*key))
            .count()
            .saturating_sub(1);
    }
    for group in &case.must_coexist {
        metrics.contradiction_false_merge_count += group
            .iter()
            .filter(|key| !selected_set.contains(*key))
            .count();
    }
    if case.channel == Channel::Ranked {
        metrics.unsupported_citation_count +=
            unsupported_citations(corpus, seeded, selection, selected)?;
        metrics.ranked_prompt_chars += rendered_chars;
    } else {
        metrics.canonical_prompt_chars += rendered_chars;
    }
    let stats = &selection.stats;
    metrics.omission_mismatch_count += usize::from(
        stats.equivalent_records_collapsed != case.expected_stats.collapsed
            || stats.omitted_by_count != case.expected_stats.omitted_by_count
            || stats.omitted_by_chars != case.expected_stats.omitted_by_chars
            || stats.omitted_beyond_scan != case.expected_stats.omitted_beyond_scan,
    );
    Ok(())
}

fn record<'a>(corpus: &'a Corpus, seeded: &Seeded, key: &str) -> Result<&'a RecordFixture> {
    let index = seeded
        .record_by_key
        .get(key)
        .with_context(|| format!("unknown record key {key}"))?;
    Ok(&corpus.records[*index])
}

fn unsupported_citations(
    corpus: &Corpus,
    seeded: &Seeded,
    selection: &PromptMemorySelection,
    selected: &[String],
) -> Result<usize> {
    let mut unsupported = 0;
    for (memory, key) in selection.memories.iter().zip(selected) {
        let expected = record(corpus, seeded, key)?.source_message_key.as_deref();
        let cited = memory
            .citation
            .split(" | ")
            .filter_map(|citation| citation.rsplit("; ").next())
            .collect::<BTreeSet<_>>();
        if expected.is_none() || cited != BTreeSet::from([expected.unwrap()]) {
            unsupported += 1;
        }
    }
    Ok(unsupported)
}

fn case_report(
    case: &CaseFixture,
    selection: PromptMemorySelection,
    selected_representative_keys: Vec<String>,
    rendered_prompt_chars: usize,
) -> CaseReport {
    let stats = selection.stats;
    CaseReport {
        id: case.id.clone(),
        channel: match case.channel {
            Channel::Canonical => "canonical",
            Channel::Ranked => "ranked",
        }
        .into(),
        max_count: case.max_count,
        max_row_chars: case.max_row_chars,
        expected_representative_keys: case.expected_representative_keys.clone(),
        selected_representative_keys,
        selection: PromptSelectionStatsReport {
            active_authorized_records: stats.active_authorized_records,
            candidate_records_scanned: stats.candidate_records_scanned,
            selected_representatives: stats.selected_representatives,
            equivalent_records_collapsed: stats.equivalent_records_collapsed,
            omitted_by_count: stats.omitted_by_count,
            omitted_by_chars: stats.omitted_by_chars,
            omitted_beyond_scan: stats.omitted_beyond_scan,
            selected_row_chars: stats.selected_row_chars,
        },
        rendered_prompt_chars,
    }
}

fn cosine_sweep(corpus: &Corpus, baseline: &Baseline) -> Vec<CosineSweepRow> {
    baseline
        .cosine_candidates
        .iter()
        .map(|&threshold| {
            let mut missed = 0;
            let mut false_merges = 0;
            for pair in &corpus.model_evidence.pairs {
                let predicted =
                    compatibility_guard_allows(&pair.left, &pair.right) && pair.cosine >= threshold;
                missed += usize::from(pair.expected_equivalent && !predicted);
                false_merges += usize::from(!pair.expected_equivalent && predicted);
            }
            CosineSweepRow {
                threshold,
                missed_equivalents: missed,
                false_merges,
            }
        })
        .collect()
}

fn budget_sweep(
    corpus: &Corpus,
    seeded: &Seeded,
    baseline: &Baseline,
) -> Result<Vec<BudgetSweepRow>> {
    baseline
        .ranked_row_char_candidates
        .iter()
        .map(|&row_chars| {
            let mut missed = 0;
            let mut rendered = 0;
            for case in corpus
                .cases
                .iter()
                .filter(|case| case.calibrates_ranked_budget)
            {
                let (_, selected, chars) = run_case(corpus, seeded, case, row_chars)?;
                let selected = selected.into_iter().collect::<BTreeSet<_>>();
                missed += case
                    .expected_representative_keys
                    .iter()
                    .filter(|key| !selected.contains(*key))
                    .count();
                rendered += chars;
            }
            Ok(BudgetSweepRow {
                row_chars,
                missed_expected_representatives: missed,
                rendered_prompt_chars: rendered,
            })
        })
        .collect()
}

fn violations(
    metrics: &Metrics,
    thresholds: &Thresholds,
    cosine: &[CosineSweepRow],
    budgets: &[BudgetSweepRow],
) -> Vec<String> {
    let mut result = Vec::new();
    check_min(
        &mut result,
        "retrieval_recall",
        metrics.retrieval_recall,
        thresholds.min_retrieval_recall,
    );
    check_max(
        &mut result,
        "semantic_duplicate_count",
        metrics.semantic_duplicate_count,
        thresholds.max_semantic_duplicate_count,
    );
    check_max(
        &mut result,
        "correction_precedence_violations",
        metrics.correction_precedence_violations,
        thresholds.max_correction_precedence_violations,
    );
    check_max(
        &mut result,
        "contradiction_false_merge_count",
        metrics.contradiction_false_merge_count,
        thresholds.max_contradiction_false_merge_count,
    );
    check_max(
        &mut result,
        "acl_leak_count",
        metrics.acl_leak_count,
        thresholds.max_acl_leak_count,
    );
    check_max(
        &mut result,
        "unsupported_citation_count",
        metrics.unsupported_citation_count,
        thresholds.max_unsupported_citation_count,
    );
    check_max(
        &mut result,
        "stale_memory_count",
        metrics.stale_memory_count,
        thresholds.max_stale_memory_count,
    );
    check_max(
        &mut result,
        "omission_mismatch_count",
        metrics.omission_mismatch_count,
        thresholds.max_omission_mismatch_count,
    );
    check_max(
        &mut result,
        "canonical_prompt_chars",
        metrics.canonical_prompt_chars,
        thresholds.max_canonical_prompt_chars,
    );
    check_max(
        &mut result,
        "ranked_prompt_chars",
        metrics.ranked_prompt_chars,
        thresholds.max_ranked_prompt_chars,
    );
    let selected = cosine
        .iter()
        .find(|row| row.threshold == thresholds.selected_semantic_cosine);
    if selected.is_none_or(|row| row.missed_equivalents > 0 || row.false_merges > 0) {
        result.push("selected cosine threshold fails frozen evidence".into());
    }
    let higher_clean = cosine.iter().any(|row| {
        row.threshold > thresholds.selected_semantic_cosine
            && row.missed_equivalents == 0
            && row.false_merges == 0
    });
    if higher_clean {
        result.push("selected cosine threshold is not the highest zero-error candidate".into());
    }
    let selected_budget = budgets
        .iter()
        .find(|row| row.row_chars == thresholds.selected_ranked_row_chars);
    if selected_budget.is_none_or(|row| row.missed_expected_representatives > 0) {
        result.push("selected ranked row budget loses expected recall".into());
    }
    let lower_complete = budgets.iter().any(|row| {
        row.row_chars < thresholds.selected_ranked_row_chars
            && row.missed_expected_representatives == 0
    });
    if lower_complete {
        result.push("selected ranked row budget is not the smallest complete candidate".into());
    }
    result
}

fn check_max(result: &mut Vec<String>, name: &str, actual: usize, maximum: usize) {
    if actual > maximum {
        result.push(format!("{name}={actual} exceeds {maximum}"));
    }
}

fn check_min(result: &mut Vec<String>, name: &str, actual: f64, minimum: f64) {
    if actual < minimum {
        result.push(format!("{name}={actual:.6} is below {minimum:.6}"));
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        1.0
    } else {
        (numerator as f64 / denominator as f64 * 1_000_000.0).round() / 1_000_000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v2_gate_passes_and_is_byte_stable() -> Result<()> {
        let fixture = selected_path(None, DEFAULT_FIXTURE);
        let snapshot = selected_path(None, DEFAULT_SNAPSHOT);
        let baseline = selected_path(None, DEFAULT_BASELINE);
        let first = evaluate_paths(&fixture, &snapshot, &baseline)?;
        let second = evaluate_paths(&fixture, &snapshot, &baseline)?;
        assert!(first.passed, "{:?}", first.violations);
        assert_eq!(
            serde_json::to_string_pretty(&first)?,
            serde_json::to_string_pretty(&second)?
        );
        assert_eq!(first.metrics.semantic_duplicate_count, 0);
        assert_eq!(first.metrics.correction_precedence_violations, 0);
        assert_eq!(first.metrics.contradiction_false_merge_count, 0);
        assert_eq!(first.metrics.acl_leak_count, 0);
        assert_eq!(first.metrics.unsupported_citation_count, 0);
        assert_eq!(first.metrics.stale_memory_count, 0);
        Ok(())
    }
}
