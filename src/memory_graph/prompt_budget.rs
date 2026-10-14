//! Deterministic, read-only prompt selection and semantic deduplication.
use super::*;
use std::cmp::Ordering;
use std::collections::BTreeSet;

pub(crate) const CANONICAL_SCAN_LIMIT: usize = 256;
pub(crate) const RANKED_SCAN_MIN: usize = 32;
pub(crate) const RANKED_SCAN_MAX: usize = 64;

// Calibrated against the frozen multilingual-e5-small evidence in the v2 quality fixture.
pub(crate) const SEMANTIC_EQUIVALENCE_MIN_COSINE: f32 = 0.84;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PromptSelectionStats {
    pub active_authorized_records: usize,
    pub candidate_records_scanned: usize,
    pub selected_representatives: usize,
    pub equivalent_records_collapsed: usize,
    pub omitted_by_count: usize,
    pub omitted_by_chars: usize,
    pub omitted_beyond_scan: usize,
    pub selected_row_chars: usize,
    pub max_count: usize,
    pub max_row_chars: usize,
}

impl PromptSelectionStats {
    pub fn has_omissions(&self) -> bool {
        self.equivalent_records_collapsed > 0
            || self.omitted_by_count > 0
            || self.omitted_by_chars > 0
            || self.omitted_beyond_scan > 0
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct PromptMemorySelection {
    pub memories: Vec<RecalledMemory>,
    pub stats: PromptSelectionStats,
}

impl PromptMemorySelection {
    pub fn should_render(&self) -> bool {
        !self.memories.is_empty() || self.stats.has_omissions()
    }
}

#[derive(Clone)]
struct Candidate {
    memory: RecalledMemory,
    embedding: Vec<f32>,
    governance: Vec<String>,
    created_at_ms: i64,
    similarity: f64,
}

struct Cluster {
    representative: Candidate,
    best_similarity: f64,
    equivalent_records: usize,
}

pub(super) struct InspectionDedupRecord {
    pub id: String,
    pub content: String,
    pub canonical: bool,
    pub confidence: f64,
    pub memory_type: String,
    pub embedding: Vec<f32>,
    pub governance: Vec<String>,
    pub created_at_ms: i64,
    pub score: f64,
}

pub(super) fn select_inspection_records(
    records: Vec<InspectionDedupRecord>,
    total: usize,
    max_count: usize,
    canonical_order: bool,
    ranked: bool,
) -> PromptMemorySelection {
    let candidates = records
        .into_iter()
        .map(|record| Candidate {
            memory: RecalledMemory {
                id: record.id,
                content: record.content,
                memory_type: record.memory_type,
                canonical: record.canonical,
                confidence: record.confidence,
                citation: String::new(),
            },
            embedding: record.embedding,
            governance: record.governance,
            created_at_ms: record.created_at_ms,
            similarity: record.score,
        })
        .collect::<Vec<_>>();
    if !ranked {
        return select_candidates(candidates, total, max_count, usize::MAX, canonical_order);
    }
    let scanned = candidates.len();
    let mut clusters = cluster_ranked(candidates);
    let collapsed = scanned.saturating_sub(clusters.len());
    clusters.sort_by(|left, right| {
        right
            .best_similarity
            .total_cmp(&left.best_similarity)
            .then_with(|| newest_first(&left.representative, &right.representative))
    });
    let mut selection = select_clusters(clusters, total, max_count, usize::MAX, false, scanned);
    selection.stats.equivalent_records_collapsed = collapsed;
    selection
}

impl MemoryGraph {
    pub fn canonical_prompt_selection(
        &self,
        principal_id: &str,
        context_id: &str,
        max_count: usize,
        max_row_chars: usize,
    ) -> Result<PromptMemorySelection> {
        let total = self.prompt_authorized_count(principal_id, Some(context_id), true)?;
        let mut candidates = self.canonical_prompt_candidates(principal_id, context_id)?;
        candidates.sort_by(newest_first);
        Ok(select_candidates(
            candidates,
            total,
            max_count,
            max_row_chars,
            true,
        ))
    }

    pub fn ranked_prompt_selection(
        &self,
        principal_id: &str,
        _query: &str,
        embedding: &[f32],
        max_count: usize,
        max_row_chars: usize,
    ) -> Result<PromptMemorySelection> {
        ensure!(
            embedding.len() == 384,
            "query embedding must have 384 dimensions"
        );
        let total = self.prompt_authorized_count(principal_id, None, false)?;
        let scan_limit = max_count
            .saturating_mul(8)
            .max(RANKED_SCAN_MIN)
            .min(RANKED_SCAN_MAX);
        let mut candidates = self.ranked_prompt_candidates(principal_id, embedding, scan_limit)?;
        let mut recalled = candidates
            .iter()
            .map(|candidate| candidate.memory.clone())
            .collect::<Vec<_>>();
        self.add_citations(&mut recalled)?;
        for (candidate, memory) in candidates.iter_mut().zip(recalled) {
            candidate.memory = memory;
        }
        let scanned = candidates.len();
        let mut clusters = cluster_ranked(candidates);
        let collapsed = scanned.saturating_sub(clusters.len());
        clusters.sort_by(|left, right| {
            right
                .best_similarity
                .total_cmp(&left.best_similarity)
                .then_with(|| newest_first(&left.representative, &right.representative))
        });
        let mut selected =
            select_clusters(clusters, total, max_count, max_row_chars, false, scanned);
        selected.stats.equivalent_records_collapsed = collapsed;
        Ok(selected)
    }

    pub(crate) fn exact_equivalent_ids(
        &self,
        principal_id: &str,
        context_id: &str,
        memory_id: &str,
        canonical: bool,
    ) -> Result<(Vec<String>, PromptSelectionStats)> {
        let scan_limit = if canonical {
            CANONICAL_SCAN_LIMIT
        } else {
            RANKED_SCAN_MAX
        };
        let total = self.prompt_authorized_count(principal_id, Some(context_id), canonical)?;
        let candidates =
            self.context_prompt_candidates(principal_id, context_id, canonical, scan_limit)?;
        let Some(target) =
            self.context_exact_prompt_candidate(principal_id, context_id, memory_id, canonical)?
        else {
            return Ok((Vec::new(), PromptSelectionStats::default()));
        };
        let scanned = candidates.len();
        let mut equivalent_ids = candidates
            .into_iter()
            .filter(|candidate| candidate.memory.id != memory_id && equivalent(&target, candidate))
            .map(|candidate| candidate.memory.id)
            .collect::<Vec<_>>();
        equivalent_ids.sort();
        Ok((
            equivalent_ids.clone(),
            PromptSelectionStats {
                active_authorized_records: total,
                candidate_records_scanned: scanned,
                selected_representatives: 1,
                equivalent_records_collapsed: equivalent_ids.len(),
                omitted_beyond_scan: total.saturating_sub(scanned),
                max_count: scan_limit,
                max_row_chars: usize::MAX,
                ..Default::default()
            },
        ))
    }

    fn canonical_prompt_candidates(
        &self,
        principal_id: &str,
        context_id: &str,
    ) -> Result<Vec<Candidate>> {
        let query = context_authorized_memory_query(
            "AND memory.canonical = true
             RETURN memory.id, memory.content, memory.memory_type, memory.canonical,
                    memory.confidence, memory.created_at_ms, memory.embedding, group_ids
             ORDER BY memory.created_at_ms DESC, memory.id DESC LIMIT $limit",
        );
        self.prompt_candidates(
            &query,
            context_params_for_prompt(principal_id, context_id, CANONICAL_SCAN_LIMIT),
            None,
        )
    }

    fn context_prompt_candidates(
        &self,
        principal_id: &str,
        context_id: &str,
        canonical: bool,
        limit: usize,
    ) -> Result<Vec<Candidate>> {
        let tail = "AND memory.canonical = $canonical
                    RETURN memory.id, memory.content, memory.memory_type, memory.canonical,
                           memory.confidence, memory.created_at_ms, memory.embedding, group_ids
                    ORDER BY memory.created_at_ms DESC, memory.id DESC LIMIT $limit";
        let query = context_authorized_memory_query(tail);
        let mut params = context_params_for_prompt(principal_id, context_id, limit);
        params.push(("canonical", Value::Bool(canonical)));
        self.prompt_candidates(&query, params, None)
    }

    fn context_exact_prompt_candidate(
        &self,
        principal_id: &str,
        context_id: &str,
        memory_id: &str,
        canonical: bool,
    ) -> Result<Option<Candidate>> {
        let query = context_authorized_memory_query(
            "AND memory.id = $memory_id AND memory.canonical = $canonical
             RETURN memory.id, memory.content, memory.memory_type, memory.canonical,
                    memory.confidence, memory.created_at_ms, memory.embedding, group_ids LIMIT 1",
        );
        let params = vec![
            ("principal_id", principal_id.into()),
            ("group_id", context_group_id(context_id).into()),
            ("global_group_id", GLOBAL_GROUP_ID.into()),
            ("memory_id", memory_id.into()),
            ("canonical", Value::Bool(canonical)),
        ];
        Ok(self
            .prompt_candidates(&query, params, None)?
            .into_iter()
            .next())
    }

    fn ranked_prompt_candidates(
        &self,
        principal_id: &str,
        embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<Candidate>> {
        let connection = Connection::new(&self.database)?;
        let mut statement = connection.prepare(
            "MATCH (principal:Principal {id: $principal_id})-[membership:MEMBER_OF]->(access_group:AccessGroup)<-[:GOVERNED_BY]-(memory:Memory)
             WHERE membership.can_read = true AND memory.status = 'active' AND memory.canonical = false
               AND NOT (memory)-[:DENIED_TO]->(principal)
             WITH memory, count(DISTINCT access_group) AS accessible_groups
             MATCH (memory)-[:GOVERNED_BY]->(required_group:AccessGroup)
             WITH memory, accessible_groups, count(DISTINCT required_group) AS required_groups,
                  collect(DISTINCT required_group.id) AS group_ids
             WHERE accessible_groups = required_groups
             RETURN memory.id, memory.content, memory.memory_type, memory.canonical,
                    memory.confidence, memory.created_at_ms, memory.embedding, group_ids,
                    array_cosine_similarity(memory.embedding, $embedding) AS similarity
             ORDER BY similarity DESC, memory.created_at_ms DESC, memory.id ASC LIMIT $limit",
        )?;
        let rows = connection.execute(
            &mut statement,
            vec![
                ("principal_id", principal_id.into()),
                ("embedding", embedding_value(embedding)?),
                ("limit", (limit as i64).into()),
            ],
        )?;
        rows.map(|row| prompt_candidate_from_row(&row, Some(8)))
            .collect()
    }

    fn prompt_candidates(
        &self,
        query: &str,
        params: Vec<(&str, Value)>,
        similarity_index: Option<usize>,
    ) -> Result<Vec<Candidate>> {
        let connection = Connection::new(&self.database)?;
        let mut statement = connection.prepare(query)?;
        connection
            .execute(&mut statement, params)?
            .map(|row| prompt_candidate_from_row(&row, similarity_index))
            .collect()
    }

    fn prompt_authorized_count(
        &self,
        principal_id: &str,
        context_id: Option<&str>,
        canonical: bool,
    ) -> Result<usize> {
        let connection = Connection::new(&self.database)?;
        let (query, params) = if let Some(context_id) = context_id {
            (
                context_authorized_memory_query(
                    "AND memory.canonical = $canonical RETURN count(memory)",
                ),
                vec![
                    ("principal_id", principal_id.into()),
                    ("group_id", context_group_id(context_id).into()),
                    ("global_group_id", GLOBAL_GROUP_ID.into()),
                    ("canonical", Value::Bool(canonical)),
                ],
            )
        } else {
            (
                "MATCH (principal:Principal {id: $principal_id})-[membership:MEMBER_OF]->(access_group:AccessGroup)<-[:GOVERNED_BY]-(memory:Memory)
                 WHERE membership.can_read = true AND memory.status = 'active' AND memory.canonical = $canonical
                   AND NOT (memory)-[:DENIED_TO]->(principal)
                 WITH memory, count(DISTINCT access_group) AS accessible_groups
                 MATCH (memory)-[:GOVERNED_BY]->(required_group:AccessGroup)
                 WITH memory, accessible_groups, count(DISTINCT required_group) AS required_groups
                 WHERE accessible_groups = required_groups RETURN count(memory)".to_string(),
                vec![
                    ("principal_id", principal_id.into()),
                    ("canonical", Value::Bool(canonical)),
                ],
            )
        };
        let mut statement = connection.prepare(&query)?;
        Ok(connection
            .execute(&mut statement, params)?
            .next()
            .and_then(|row| row.first().and_then(|value| int64_value(value).ok()))
            .unwrap_or(0)
            .max(0) as usize)
    }
}

fn context_params_for_prompt<'a>(
    principal_id: &'a str,
    context_id: &str,
    limit: usize,
) -> Vec<(&'a str, Value)> {
    vec![
        ("principal_id", principal_id.into()),
        ("group_id", context_group_id(context_id).into()),
        ("global_group_id", GLOBAL_GROUP_ID.into()),
        ("limit", (limit as i64).into()),
    ]
}

fn prompt_candidate_from_row(row: &[Value], similarity_index: Option<usize>) -> Result<Candidate> {
    ensure!(
        row.len() >= 8,
        "unexpected prompt candidate row width: {}",
        row.len()
    );
    Ok(Candidate {
        memory: RecalledMemory {
            id: string_value(&row[0])?,
            content: string_value(&row[1])?,
            memory_type: string_value(&row[2])?,
            canonical: bool_value(&row[3])?,
            confidence: double_value(&row[4])?,
            citation: String::new(),
        },
        created_at_ms: int64_value(&row[5])?,
        embedding: float_array(&row[6])?,
        governance: string_list(&row[7])?,
        similarity: similarity_index.map_or(Ok(0.0), |index| similarity_value(&row[index]))?,
    })
}

pub(super) fn similarity_value(value: &Value) -> Result<f64> {
    match value {
        Value::Double(value) => Ok(*value),
        Value::Float(value) => Ok(f64::from(*value)),
        other => Err(anyhow!(
            "expected floating-point similarity from Ladybug, got {other:?}"
        )),
    }
}

pub(super) fn float_array(value: &Value) -> Result<Vec<f32>> {
    match value {
        Value::Array(LogicalType::Float, values) => values
            .iter()
            .map(|value| match value {
                Value::Float(value) => Ok(*value),
                other => Err(anyhow!("expected float embedding entry, got {other:?}")),
            })
            .collect(),
        other => Err(anyhow!("expected float embedding array, got {other:?}")),
    }
}

pub(super) fn string_list(value: &Value) -> Result<Vec<String>> {
    let mut result = match value {
        Value::List(LogicalType::String, values) => values
            .iter()
            .map(string_value)
            .collect::<Result<Vec<_>>>()?,
        other => return Err(anyhow!("expected governance string list, got {other:?}")),
    };
    result.sort();
    result.dedup();
    Ok(result)
}

fn newest_first(left: &Candidate, right: &Candidate) -> Ordering {
    right
        .created_at_ms
        .cmp(&left.created_at_ms)
        .then_with(|| {
            right
                .memory
                .id
                .starts_with("memory:correction:")
                .cmp(&left.memory.id.starts_with("memory:correction:"))
        })
        .then_with(|| right.memory.id.cmp(&left.memory.id))
}

fn cluster_ranked(mut candidates: Vec<Candidate>) -> Vec<Cluster> {
    candidates.sort_by(newest_first);
    let mut clusters: Vec<Cluster> = Vec::new();
    for candidate in candidates {
        if let Some(cluster) = clusters
            .iter_mut()
            .find(|cluster| equivalent(&cluster.representative, &candidate))
        {
            cluster.best_similarity = cluster.best_similarity.max(candidate.similarity);
            cluster.equivalent_records += 1;
        } else {
            let best_similarity = candidate.similarity;
            clusters.push(Cluster {
                representative: candidate,
                best_similarity,
                equivalent_records: 0,
            });
        }
    }
    clusters
}

fn select_candidates(
    mut candidates: Vec<Candidate>,
    total: usize,
    max_count: usize,
    max_row_chars: usize,
    canonical: bool,
) -> PromptMemorySelection {
    let scanned = candidates.len();
    candidates.sort_by(newest_first);
    let clusters = cluster_ranked(candidates);
    let collapsed = scanned.saturating_sub(clusters.len());
    let mut selection = select_clusters(
        clusters,
        total,
        max_count,
        max_row_chars,
        canonical,
        scanned,
    );
    selection.stats.equivalent_records_collapsed = collapsed;
    selection.stats.candidate_records_scanned = scanned;
    selection
}

fn select_clusters(
    clusters: Vec<Cluster>,
    total: usize,
    max_count: usize,
    max_row_chars: usize,
    canonical: bool,
    scanned: usize,
) -> PromptMemorySelection {
    let mut memories = Vec::new();
    let mut used: usize = 0;
    let mut omitted_by_count = 0;
    let mut omitted_by_chars = 0;
    let mut collapsed = 0;
    let mut char_exhausted = false;
    for (index, cluster) in clusters.into_iter().enumerate() {
        collapsed += cluster.equivalent_records;
        if index >= max_count {
            omitted_by_count += 1;
            continue;
        }
        let row_chars = rendered_row_chars(&cluster.representative.memory);
        if char_exhausted || used.saturating_add(row_chars) > max_row_chars {
            char_exhausted = true;
            omitted_by_chars += 1;
            continue;
        }
        used += row_chars;
        memories.push(cluster.representative.memory);
    }
    if canonical {
        memories.reverse();
    }
    PromptMemorySelection {
        stats: PromptSelectionStats {
            active_authorized_records: total,
            candidate_records_scanned: scanned,
            selected_representatives: memories.len(),
            equivalent_records_collapsed: collapsed,
            omitted_by_count,
            omitted_by_chars,
            omitted_beyond_scan: total.saturating_sub(scanned),
            selected_row_chars: used,
            max_count,
            max_row_chars,
            ..Default::default()
        },
        memories,
    }
}

pub(crate) fn rendered_row_chars(memory: &RecalledMemory) -> usize {
    if memory.canonical {
        format!("- [{}] {}\n", memory.id, memory.content)
            .chars()
            .count()
    } else {
        format!(
            "- [{} | {}] {}\n",
            memory.id, memory.citation, memory.content
        )
        .chars()
        .count()
    }
}

fn equivalent(left: &Candidate, right: &Candidate) -> bool {
    if left.memory.canonical != right.memory.canonical || left.governance != right.governance {
        return false;
    }
    let left_normalized = normalize_text(&left.memory.content);
    let right_normalized = normalize_text(&right.memory.content);
    if left_normalized == right_normalized {
        return true;
    }
    contradiction_compatible(&left.memory.content, &right.memory.content)
        && cosine(&left.embedding, &right.embedding) >= SEMANTIC_EQUIVALENCE_MIN_COSINE
}

fn normalize_text(value: &str) -> String {
    value
        .chars()
        .flat_map(char::to_lowercase)
        .map(|character| {
            if character.is_alphanumeric() {
                character
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn contradiction_compatible(left: &str, right: &str) -> bool {
    let Some(left_literals) = preserved_literals(left) else {
        return false;
    };
    let Some(right_literals) = preserved_literals(right) else {
        return false;
    };
    left_literals == right_literals && polarity(left) == polarity(right)
}

pub(crate) fn compatibility_guard_allows(left: &str, right: &str) -> bool {
    normalize_text(left) == normalize_text(right) || contradiction_compatible(left, right)
}

fn polarity(value: &str) -> BTreeSet<String> {
    const NEGATIVE: &[(&str, &str)] = &[
        ("no", "neg"),
        ("not", "neg"),
        ("never", "neg"),
        ("without", "neg"),
        ("don't", "neg"),
        ("cannot", "neg"),
        ("can't", "neg"),
        ("nie", "neg"),
        ("nigdy", "neg"),
        ("bez", "neg"),
        ("nicht", "neg"),
        ("nie", "neg"),
        ("kein", "neg"),
        ("keine", "neg"),
        ("ohne", "neg"),
        ("nunca", "neg"),
        ("sin", "neg"),
    ];
    let normalized = value.to_lowercase();
    normalized
        .split(|character: char| !character.is_alphabetic() && character != '\'')
        .filter_map(|token| {
            NEGATIVE
                .iter()
                .find(|(word, _)| token == *word)
                .map(|(_, class)| (*class).to_string())
        })
        .collect()
}

fn preserved_literals(value: &str) -> Option<BTreeSet<String>> {
    let mut result = BTreeSet::new();
    let mut quote: Option<char> = None;
    let mut quoted = String::new();
    for character in value.chars() {
        if let Some(delimiter) = quote {
            if character == delimiter {
                result.insert(format!("quote:{}", quoted.to_lowercase()));
                quoted.clear();
                quote = None;
            } else {
                quoted.push(character);
            }
        } else if matches!(character, '\'' | '"' | '`') {
            quote = Some(character);
        }
    }
    if quote.is_some() {
        return None;
    }
    let tokens = value.split_whitespace().collect::<Vec<_>>();
    for (index, raw) in tokens.iter().enumerate() {
        let token = raw.trim_matches(|character: char| {
            !character.is_alphanumeric()
                && !matches!(character, ':' | '/' | '.' | '-' | '_' | '@' | '#')
        });
        if token.is_empty() {
            continue;
        }
        let lower = token.to_lowercase();
        if token.chars().any(|character| character.is_ascii_digit()) {
            result.insert(format!("literal:{lower}"));
        } else if lower.starts_with("http://")
            || lower.starts_with("https://")
            || lower.starts_with("www.")
        {
            result.insert(format!("url:{lower}"));
        } else if token.starts_with('@')
            || token.starts_with('#')
            || token.chars().filter(|c| c.is_alphabetic()).count() >= 2
                && token
                    .chars()
                    .filter(|c| c.is_alphabetic())
                    .all(char::is_uppercase)
            || index > 0 && explicitly_names(tokens[index - 1])
        {
            result.insert(format!("name:{token}"));
        }
    }
    Some(result)
}

fn explicitly_names(previous: &str) -> bool {
    matches!(
        previous
            .trim_matches(|character: char| !character.is_alphabetic())
            .to_lowercase()
            .as_str(),
        "named" | "called" | "imieniem" | "namens" | "llamado" | "llamada"
    )
}

fn cosine(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return -1.0;
    }
    let mut dot = 0.0f64;
    let mut left_norm = 0.0f64;
    let mut right_norm = 0.0f64;
    for (&left, &right) in left.iter().zip(right) {
        dot += f64::from(left) * f64::from(right);
        left_norm += f64::from(left) * f64::from(left);
        right_norm += f64::from(right) * f64::from(right);
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return -1.0;
    }
    (dot / (left_norm.sqrt() * right_norm.sqrt())) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(content: &str, embedding: Vec<f32>, created_at_ms: i64) -> Candidate {
        Candidate {
            memory: RecalledMemory {
                id: format!("memory:{created_at_ms}"),
                content: content.into(),
                memory_type: "asserted".into(),
                canonical: true,
                confidence: 1.0,
                citation: String::new(),
            },
            embedding,
            governance: vec!["access:test".into()],
            created_at_ms,
            similarity: 1.0,
        }
    }

    fn vector(first: f32, second: f32) -> Vec<f32> {
        let mut result = vec![0.0; 384];
        result[0] = first;
        result[1] = second;
        result
    }

    fn source(id: &str, context_id: &str, created_at_ms: i64) -> SourceMessage {
        SourceMessage {
            id: id.into(),
            platform_message_id: Some(id.into()),
            context_id: context_id.into(),
            context_kind: "telegram_group".into(),
            speaker_id: "speaker".into(),
            speaker_name: "Speaker".into(),
            role: "user".into(),
            content: format!("source {id}"),
            created_at_ms,
        }
    }

    #[test]
    fn semantic_equivalence_is_guarded_against_contradictions_and_literals() {
        let base = candidate(
            "Use dark mode at 18:30 with version v2",
            vector(1.0, 0.0),
            1,
        );
        for text in [
            "Do not use dark mode at 18:30 with version v2",
            "Use dark mode at 19:30 with version v2",
            "Use dark mode at 18:30 with version v3",
            "Use dark mode at 18:30 with version v2 for someone named Bob",
            "Use dark mode at 18:30 with version v2 from https://other.example",
            "Use dark mode at 18:30 with version v2 and label \"beta\"",
        ] {
            assert!(
                !equivalent(&base, &candidate(text, vector(1.0, 0.0), 2)),
                "merged {text}"
            );
        }
    }

    #[test]
    fn translated_equivalents_collapse_but_translated_negation_does_not() {
        let english = candidate("I enjoy quiet mornings", vector(1.0, 0.0), 1);
        let polish = candidate("Lubię spokojne poranki", vector(0.99, 0.01), 2);
        assert!(equivalent(&english, &polish));
        let negative = candidate("Nie lubię spokojnych poranków", vector(0.99, 0.01), 3);
        assert!(!equivalent(&english, &negative));
    }

    #[test]
    fn budget_is_a_strict_newest_prefix_without_shorter_backfill() {
        let newest = candidate("x".repeat(100).as_str(), vector(1.0, 0.0), 3);
        let middle = candidate("medium", vector(0.0, 1.0), 2);
        let oldest = candidate("short", vector(0.7, 0.7), 1);
        let selection = select_candidates(vec![newest, middle, oldest], 3, 3, 30, true);
        assert!(selection.memories.is_empty());
        assert_eq!(selection.stats.omitted_by_chars, 3);
    }

    #[test]
    fn clustering_does_not_chain_through_an_intermediate_record() {
        let newest = candidate("Calm morning preference", vector(0.68, 0.7332121), 3);
        let middle = candidate("Quiet daytime preference", vector(0.9, 0.4358899), 2);
        let oldest = candidate("Peaceful routine preference", vector(1.0, 0.0), 1);
        let selection = select_candidates(vec![newest, middle, oldest], 3, 10, 8_000, true);
        assert_eq!(selection.memories.len(), 2);
        assert_eq!(selection.stats.equivalent_records_collapsed, 1);
    }

    #[test]
    fn count_and_character_boundaries_apply_after_dedup() {
        let newest = candidate("Repeated preference.", vector(1.0, 0.0), 3);
        let duplicate = candidate("Repeated preference!", vector(1.0, 0.0), 2);
        let distinct = candidate("Distinct preference.", vector(0.0, 1.0), 1);
        let row_chars = rendered_row_chars(&newest.memory);

        let exact = select_candidates(
            vec![newest.clone(), duplicate.clone(), distinct.clone()],
            3,
            1,
            row_chars,
            true,
        );
        assert_eq!(exact.memories.len(), 1);
        assert_eq!(exact.stats.equivalent_records_collapsed, 1);
        assert_eq!(exact.stats.omitted_by_count, 1);
        assert_eq!(exact.stats.omitted_by_chars, 0);

        let short = select_candidates(vec![newest, duplicate, distinct], 3, 1, row_chars - 1, true);
        assert!(short.memories.is_empty());
        assert_eq!(short.stats.equivalent_records_collapsed, 1);
        assert_eq!(short.stats.omitted_by_count, 1);
        assert_eq!(short.stats.omitted_by_chars, 1);
        let rendered = render_canonical_selection(&short);
        assert!(rendered.contains("1 equivalent record(s) collapsed"));
        assert!(rendered.contains("omitted 1 by count, 1 by row-character budget"));
    }

    #[test]
    fn canonical_selection_collapses_only_exact_governance() -> Result<()> {
        let graph = MemoryGraph::in_memory()?;
        graph.ingest_message(&source("m1", "context:a", 1), "admin")?;
        let older = graph.remember_canonical(&CanonicalDraft {
            content: "Remember quiet mornings.".into(),
            context_id: "context:a".into(),
            principal_id: "speaker".into(),
            embedding: vector(1.0, 0.0),
            scope: MemoryScope::Context,
            created_at_ms: 10,
        })?;
        let newer = graph.remember_canonical(&CanonicalDraft {
            content: "Remember quiet mornings!".into(),
            context_id: "context:a".into(),
            principal_id: "speaker".into(),
            embedding: vector(1.0, 0.0),
            scope: MemoryScope::Context,
            created_at_ms: 20,
        })?;
        let global = graph.remember_canonical(&CanonicalDraft {
            content: "Remember quiet mornings.".into(),
            context_id: "context:a".into(),
            principal_id: "speaker".into(),
            embedding: vector(1.0, 0.0),
            scope: MemoryScope::Global,
            created_at_ms: 30,
        })?;

        let selection = graph.canonical_prompt_selection("speaker", "context:a", 40, 8_000)?;
        let ids = selection
            .memories
            .iter()
            .map(|memory| memory.id.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ids, vec![newer.as_str(), global.as_str()]);
        assert!(!ids.contains(&older.as_str()));
        assert_eq!(selection.stats.equivalent_records_collapsed, 1);
        assert_eq!(selection.stats.active_authorized_records, 3);
        Ok(())
    }

    #[test]
    fn ranked_selection_keeps_representative_provenance() -> Result<()> {
        let graph = MemoryGraph::in_memory()?;
        graph.ingest_message(&source("older-source", "context:a", 1), "admin")?;
        graph.ingest_message(&source("newer-source", "context:a", 2), "admin")?;
        for (source_id, run, created_at_ms) in [
            ("older-source", "run:old", 10),
            ("newer-source", "run:new", 20),
        ] {
            graph.create_index_memory(&MemoryDraft {
                content: "The header needs more spacing.".into(),
                memory_type: "finding".into(),
                confidence: 0.9,
                embedding: vector(1.0, 0.0),
                origin_context_ids: vec!["context:a".into()],
                source_message_ids: vec![source_id.into()],
                extraction_run_id: run.into(),
                extraction_model: "fixture".into(),
                created_at_ms,
            })?;
        }

        let selection = graph.ranked_prompt_selection(
            "speaker",
            "header spacing",
            &vector(1.0, 0.0),
            5,
            4_000,
        )?;
        assert_eq!(selection.memories.len(), 1);
        assert_eq!(selection.stats.equivalent_records_collapsed, 1);
        assert!(selection.memories[0].citation.contains("newer-source"));
        assert!(!selection.memories[0].citation.contains("older-source"));
        Ok(())
    }
}
