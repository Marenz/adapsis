//! Provider prompt-cache lifetime per model route, and what one conversation turn
//! needs to observe it (#68 cold start).
//!
//! Provider caches expire after a pause, and the lifetime differs per provider,
//! so nothing is hard-coded here: `model_limits` supplies a sourced default (or
//! none), and this module learns from real usage. For every answered request
//! where a cache hit could be expected, the pause since the previous request of
//! the same conversation and the outcome (hit or miss) are recorded per route.
//!
//! Evidence rules:
//! - A hit is hard evidence: the cache lived at least that long. A hit at or past
//!   the default raises the lifetime to just past that pause, and every recorded
//!   miss at a shorter or equal pause is discarded.
//! - A miss is soft evidence (providers also evict early). One miss never
//!   changes the lifetime; the SECOND shortest recorded miss is a confirmed
//!   upper bound. It lowers the lifetime when it is under the default, and it is
//!   the lifetime of a route that has no default. Two misses at or below the
//!   longest hit are newer than that hit (a later hit would have discarded
//!   them), so they replace it.
//! - Between a hit and a longer confirmed miss the shorter value is used: a
//!   needless cold start costs context once, a wrongly kept view re-writes the
//!   whole history. Pauses inside a turn and small conversations (which never
//!   cold-start) keep probing longer pauses at little cost.
//! - Misses expire after `MISS_MAX_AGE_MS`, so a provider that lengthens its
//!   lifetime is relearned. Pauses under `MIN_LIFETIME_SECS` say nothing.
//! - Routes that do not report cache reads separately are never recorded.
//! - A pause is measured between the START times of two requests of the same
//!   conversation: the provider counts the lifetime from the start of the request
//!   that wrote or read the entry, and a read refreshes it (Anthropic DOCS and
//!   MEASURED 2026-11-10).
//! - A request that failed, was refused or had to be shrunk is no evidence, and
//!   it is not assumed to have written the cache (MEASURED 2026-11-10: a refused
//!   request reports a cache write that the next identical request cannot read).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Sane range for a learned lifetime. No provider documents a cache lifetime
/// under a minute; beyond a week a conversation is cold whatever was observed.
pub const MIN_LIFETIME_SECS: u64 = 60;
pub const MAX_LIFETIME_SECS: u64 = 7 * 24 * 3600;
/// Recorded misses older than this no longer count (14 days).
const MISS_MAX_AGE_MS: u64 = 14 * 24 * 3600 * 1000;
/// A cacheable prefix smaller than this says too little: providers have a minimum
/// cacheable size (Anthropic DOCS: 512 tokens on current models, more on older
/// ones) and the projection of a small prefix is coarse.
pub const MIN_OBSERVABLE_TOKENS: usize = 8_000;
/// Routes tracked at most (model switches are rare; this only bounds the file).
const MAX_ROUTES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Miss {
    pub pause_secs: u64,
    pub observed_ms: u64,
}

/// What real usage has shown about one model route's cache lifetime.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheEvidence {
    /// Longest pause after which the cache still hit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub longest_hit_secs: Option<u64>,
    /// The two shortest pauses that missed, ascending.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub misses: Vec<Miss>,
    /// Audit counters: observations recorded in total.
    #[serde(default)]
    pub hit_count: u64,
    #[serde(default)]
    pub miss_count: u64,
}

/// Where an effective lifetime comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// Derived from recorded hits and misses.
    Learned,
    /// The route's default: `models.toml`, the requested ttl, or the sourced
    /// built-in.
    Default,
}

impl Source {
    pub fn label(self) -> &'static str {
        match self {
            Self::Learned => "learned",
            Self::Default => "default",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lifetime {
    pub secs: u64,
    pub source: Source,
}

/// Outcome of one request with respect to the prompt cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Hit,
    Miss,
    /// Some cache was read, but neither most nor almost none of the prefix.
    Ambiguous,
    /// The route does not report cache reads separately.
    Unobservable,
    /// No hit could be expected: first turn, after a checkpoint or prefix
    /// change, a reduced request, or a prefix too small to be cached.
    NotExpected,
}

impl Outcome {
    pub fn label(self) -> &'static str {
        match self {
            Self::Hit => "hit",
            Self::Miss => "miss",
            Self::Ambiguous => "ambiguous",
            Self::Unobservable => "unobservable",
            Self::NotExpected => "not-expected",
        }
    }
}

/// Classify a request: `expected_tokens` is the cacheable prefix the previous
/// request left behind (None when no hit can be expected), `cached_read` the
/// provider's cache-read counter (None when the route does not report it).
/// A hit reads at least half of the expected prefix, a miss at most a tenth.
pub fn classify(expected_tokens: Option<usize>, cached_read: Option<u64>) -> Outcome {
    let Some(expected) = expected_tokens.filter(|&tokens| tokens >= MIN_OBSERVABLE_TOKENS) else {
        return Outcome::NotExpected;
    };
    let Some(read) = cached_read else {
        return Outcome::Unobservable;
    };
    let expected = expected as u64;
    if read.saturating_mul(2) >= expected {
        Outcome::Hit
    } else if read.saturating_mul(10) <= expected {
        Outcome::Miss
    } else {
        Outcome::Ambiguous
    }
}

impl CacheEvidence {
    fn fresh_misses(&self, now_ms: u64) -> impl Iterator<Item = &Miss> {
        self.misses.iter().filter(move |miss| now_ms.saturating_sub(miss.observed_ms) <= MISS_MAX_AGE_MS)
    }

    /// Record one classified request. Only `Hit` and `Miss` are evidence.
    /// Returns whether anything changed.
    pub fn record(&mut self, outcome: Outcome, pause_secs: u64, now_ms: u64) -> bool {
        match outcome {
            Outcome::Hit => {
                self.hit_count += 1;
                self.longest_hit_secs = Some(self.longest_hit_secs.map_or(pause_secs, |hit| hit.max(pause_secs)));
                // The cache demonstrably lives this long: shorter misses were not expiry.
                self.misses.retain(|miss| miss.pause_secs > pause_secs);
                true
            }
            Outcome::Miss if pause_secs >= MIN_LIFETIME_SECS => {
                self.miss_count += 1;
                self.misses.retain(|miss| now_ms.saturating_sub(miss.observed_ms) <= MISS_MAX_AGE_MS);
                self.misses.push(Miss { pause_secs, observed_ms: now_ms });
                self.misses.sort_by_key(|miss| miss.pause_secs);
                self.misses.truncate(2);
                // Two misses at or below the longest hit are newer than it.
                if let (Some(hit), Some(second)) = (self.longest_hit_secs, self.misses.get(1)) {
                    if second.pause_secs <= hit {
                        self.longest_hit_secs = None;
                    }
                }
                true
            }
            _ => false,
        }
    }

    /// The lifetime to act on, given the route's sourced `default` (if any).
    /// None means unknown: there is no default and no confirmed expiry.
    pub fn effective(&self, default: Option<u64>, now_ms: u64) -> Option<Lifetime> {
        let learned = |secs: u64| Lifetime { secs: secs.clamp(MIN_LIFETIME_SECS, MAX_LIFETIME_SECS), source: Source::Learned };
        let upper = self.fresh_misses(now_ms).nth(1).map(|miss| miss.pause_secs);
        let Some(default) = default else {
            // Hits alone are only a lower bound: nothing shows the cache expires.
            return upper.map(learned);
        };
        Some(match (self.longest_hit_secs, upper) {
            // A hit at or past the default proves the cache lives longer.
            (Some(hit), _) if hit >= default => learned(hit + 1),
            // Confirmed expiry before the default.
            (_, Some(upper)) if upper < default => learned(upper),
            _ => Lifetime { secs: default, source: Source::Default },
        })
    }
}

/// Learned evidence for every model route, persisted with the session.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RouteLifetimes {
    routes: HashMap<String, CacheEvidence>,
}

impl RouteLifetimes {
    pub fn is_empty(&self) -> bool {
        self.routes.is_empty()
    }

    #[cfg(test)]
    pub fn evidence(&self, model: &str) -> Option<&CacheEvidence> {
        self.routes.get(model)
    }

    pub fn record(&mut self, model: &str, outcome: Outcome, pause_secs: u64, now_ms: u64) -> bool {
        if !matches!(outcome, Outcome::Hit | Outcome::Miss) {
            return false;
        }
        if !self.routes.contains_key(model) && self.routes.len() >= MAX_ROUTES {
            return false;
        }
        self.routes.entry(model.to_string()).or_default().record(outcome, pause_secs, now_ms)
    }

    /// Effective lifetime for the generating model's route: evidence over the
    /// model's default. None = unknown, or cold starts are switched off for the
    /// model (`cache_ttl_secs = 0` in `models.toml`).
    pub fn lifetime(&self, model: &str, limits: &crate::model_limits::ModelLimits, now_ms: u64) -> Option<Lifetime> {
        if !limits.cold_start {
            return None;
        }
        match self.routes.get(model) {
            Some(evidence) => evidence.effective(limits.cache_ttl_secs, now_ms),
            None => limits.cache_ttl_secs.map(|secs| Lifetime { secs, source: Source::Default }),
        }
    }
}

/// What the current turn of a conversation needs to account its requests:
/// set when the turn's view is built, read when a request is answered. In-memory
/// only (a restart is a prefix reset for the epoch tracker anyway).
///
/// One per conversation, and never shared by two turns: only turns that hold the
/// per-context turn lock for their whole duration create and update it (`ask`,
/// `ask_stream`, `llm_takeover`). The `!watch` self-trigger, which can run while
/// an `ask` turn on `main` is in flight, does not touch it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TurnProbe {
    /// Serialized text bytes of the view at turn start; the first request's bytes
    /// beyond it are the runtime tail.
    pub view_bytes: usize,
    /// Cacheable prefix (bytes) the turn's FIRST request can read from the
    /// cache: None unless the history only appended since the previous request.
    pub first_prefix_bytes: Option<usize>,
    /// Cacheable prefix (bytes) later requests of the turn can read: the stable
    /// prefix and history as of turn start, written by the first request.
    pub turn_prefix_bytes: Option<usize>,
    /// Answered requests of this turn so far.
    pub requests: u32,
    /// The previous turn's first request was answered normally, so it wrote the
    /// prefix this turn's first request would read.
    pub previous_wrote: bool,
    /// This turn's first request was answered normally (not failed, refused or
    /// shrunk): it wrote the turn-start prefix.
    pub wrote_prefix: bool,
}

impl TurnProbe {
    /// The prefix a hit would cover for the next answered request of the turn.
    pub fn expected_prefix_bytes(&self) -> Option<usize> {
        if self.requests == 0 { self.first_prefix_bytes } else { self.turn_prefix_bytes }
    }
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY_MS: u64 = 24 * 3600 * 1000;

    fn limits(ttl: Option<u64>, cold_start: bool) -> crate::model_limits::ModelLimits {
        let mut limits = crate::model_limits::for_model("unknown/cache-test");
        limits.cache_ttl_secs = ttl;
        limits.cold_start = cold_start;
        limits
    }

    #[test]
    fn cache_lifetime_classifies_hits_misses_and_unobservable_requests() {
        assert_eq!(classify(Some(100_000), Some(95_000)), Outcome::Hit);
        assert_eq!(classify(Some(100_000), Some(50_000)), Outcome::Hit);
        assert_eq!(classify(Some(100_000), Some(0)), Outcome::Miss);
        assert_eq!(classify(Some(100_000), Some(10_000)), Outcome::Miss);
        assert_eq!(classify(Some(100_000), Some(30_000)), Outcome::Ambiguous);
        // The route does not report cache reads: nothing can be learned.
        assert_eq!(classify(Some(100_000), None), Outcome::Unobservable);
        // First turn / after a checkpoint / prefix change: no expectation.
        assert_eq!(classify(None, Some(0)), Outcome::NotExpected);
        // A prefix that may be under the provider's minimum cacheable size.
        assert_eq!(classify(Some(MIN_OBSERVABLE_TOKENS - 1), Some(0)), Outcome::NotExpected);
        assert_eq!(classify(Some(MIN_OBSERVABLE_TOKENS), Some(0)), Outcome::Miss);
    }

    #[test]
    fn cache_lifetime_unknown_without_default_or_confirmed_expiry() {
        let mut evidence = CacheEvidence::default();
        assert_eq!(evidence.effective(None, 0), None);
        assert_eq!(evidence.effective(Some(300), 0), Some(Lifetime { secs: 300, source: Source::Default }));
        // Hits alone are a lower bound, not a lifetime.
        evidence.record(Outcome::Hit, 900, 0);
        assert_eq!(evidence.effective(None, 0), None);
        // One miss is not confirmed.
        evidence.record(Outcome::Miss, 4_000, 0);
        assert_eq!(evidence.effective(None, 0), None);
        // A second miss confirms an upper bound even without a default.
        evidence.record(Outcome::Miss, 5_000, 0);
        assert_eq!(evidence.effective(None, 0), Some(Lifetime { secs: 5_000, source: Source::Learned }));
    }

    #[test]
    fn cache_lifetime_learned_longer_than_the_default() {
        let mut evidence = CacheEvidence::default();
        // A hit below the default changes nothing.
        evidence.record(Outcome::Hit, 120, 0);
        assert_eq!(evidence.effective(Some(300), 0).unwrap(), Lifetime { secs: 300, source: Source::Default });
        // A hit after 50 minutes proves the cache outlives the 300 s default.
        evidence.record(Outcome::Hit, 3_000, 0);
        assert_eq!(evidence.effective(Some(300), 0).unwrap(), Lifetime { secs: 3_001, source: Source::Learned });
        // A shorter hit never lowers it.
        evidence.record(Outcome::Hit, 400, 0);
        assert_eq!(evidence.longest_hit_secs, Some(3_000));
        assert_eq!((evidence.hit_count, evidence.miss_count), (3, 0));
    }

    #[test]
    fn cache_lifetime_learned_shorter_than_the_default_needs_two_misses() {
        let mut evidence = CacheEvidence::default();
        // One early miss (provider eviction, another backend) is not believed.
        assert!(evidence.record(Outcome::Miss, 200, 0));
        assert_eq!(evidence.effective(Some(3_600), 0).unwrap(), Lifetime { secs: 3_600, source: Source::Default });
        // The second miss confirms: the second shortest pause is the bound.
        evidence.record(Outcome::Miss, 260, 0);
        assert_eq!(evidence.effective(Some(3_600), 0).unwrap(), Lifetime { secs: 260, source: Source::Learned });
        // It tightens from above as more misses arrive; only two are kept.
        evidence.record(Outcome::Miss, 230, 0);
        assert_eq!(evidence.misses.iter().map(|m| m.pause_secs).collect::<Vec<_>>(), vec![200, 230]);
        assert_eq!(evidence.effective(Some(3_600), 0).unwrap().secs, 230);
        // Pauses under a minute say nothing about the lifetime.
        assert!(!evidence.record(Outcome::Miss, 20, 0));
        assert_eq!(evidence.effective(Some(3_600), 0).unwrap().secs, 230);
        // Clamped to the sane range.
        let mut long = CacheEvidence::default();
        long.record(Outcome::Hit, 30 * 24 * 3600, 0);
        assert_eq!(long.effective(Some(300), 0).unwrap().secs, MAX_LIFETIME_SECS);
    }

    #[test]
    fn cache_lifetime_hits_and_misses_correct_each_other() {
        let mut evidence = CacheEvidence::default();
        evidence.record(Outcome::Miss, 400, 0);
        evidence.record(Outcome::Miss, 500, 0);
        assert_eq!(evidence.effective(Some(3_600), 0).unwrap().secs, 500);
        // Misses past the default only confirm it; they never lengthen it.
        assert_eq!(evidence.effective(Some(300), 0).unwrap(), Lifetime { secs: 300, source: Source::Default });
        // A later hit after 450 s is hard evidence: the 400 s miss was not expiry.
        evidence.record(Outcome::Hit, 450, 0);
        assert_eq!(evidence.misses.iter().map(|m| m.pause_secs).collect::<Vec<_>>(), vec![500]);
        assert_eq!(evidence.effective(Some(300), 0).unwrap(), Lifetime { secs: 451, source: Source::Learned });
        assert_eq!(evidence.effective(Some(3_600), 0).unwrap().source, Source::Default, "one miss left: unconfirmed");
        // Between a hit and a longer confirmed miss, the hit decides.
        evidence.record(Outcome::Miss, 520, 0);
        assert_eq!(evidence.effective(Some(300), 0).unwrap().secs, 451);
        assert_eq!(evidence.effective(None, 0).unwrap().secs, 520, "without a default the confirmed miss is all there is");
        // The provider shortens its lifetime: one miss under the old hit is not
        // enough, two are, and they replace the stale hit.
        let mut shrunk = CacheEvidence::default();
        shrunk.record(Outcome::Hit, 3_000, 0);
        shrunk.record(Outcome::Miss, 600, 0);
        assert_eq!(shrunk.effective(Some(300), 0).unwrap().secs, 3_001);
        shrunk.record(Outcome::Miss, 700, 0);
        assert_eq!(shrunk.longest_hit_secs, None);
        assert_eq!(shrunk.effective(Some(300), 0).unwrap(), Lifetime { secs: 300, source: Source::Default });
        assert_eq!(shrunk.effective(Some(3_600), 0).unwrap(), Lifetime { secs: 700, source: Source::Learned });
    }

    #[test]
    fn cache_lifetime_misses_expire_so_a_longer_lifetime_is_relearned() {
        let mut evidence = CacheEvidence::default();
        evidence.record(Outcome::Miss, 400, 0);
        evidence.record(Outcome::Miss, 420, DAY_MS);
        assert_eq!(evidence.effective(Some(3_600), 2 * DAY_MS).unwrap().secs, 420);
        // 14 days after the first miss only one is fresh: back to the default.
        assert_eq!(evidence.effective(Some(3_600), 14 * DAY_MS + 1).unwrap(), Lifetime { secs: 3_600, source: Source::Default });
        // Recording drops the expired entries.
        evidence.record(Outcome::Miss, 4_000, 16 * DAY_MS);
        assert_eq!(evidence.misses.iter().map(|m| m.pause_secs).collect::<Vec<_>>(), vec![4_000]);
    }

    #[test]
    fn cache_lifetime_routes_are_separate_and_non_observable_routes_stay_on_the_default() {
        let mut routes = RouteLifetimes::default();
        let anthropic = limits(Some(300), true);
        let unknown = limits(None, true);
        // Nothing learned: the default, or unknown.
        assert_eq!(routes.lifetime("a/opus", &anthropic, 0).unwrap(), Lifetime { secs: 300, source: Source::Default });
        assert_eq!(routes.lifetime("b/other", &unknown, 0), None);
        // A route that cannot report cache reads never records anything.
        for pause in [100, 1_000, 10_000] {
            assert!(!routes.record("a/opus", classify(Some(500_000), None), pause, 0));
            assert!(!routes.record("a/opus", Outcome::Ambiguous, pause, 0));
            assert!(!routes.record("a/opus", Outcome::NotExpected, pause, 0));
        }
        assert!(routes.is_empty());
        assert_eq!(routes.lifetime("a/opus", &anthropic, 0).unwrap().source, Source::Default);
        // Evidence on one route does not leak to another.
        routes.record("a/opus", Outcome::Hit, 1_800, 0);
        assert_eq!(routes.lifetime("a/opus", &anthropic, 0).unwrap().secs, 1_801);
        assert_eq!(routes.lifetime("a/sonnet", &anthropic, 0).unwrap().secs, 300);
        assert_eq!(routes.lifetime("b/other", &unknown, 0), None);
        // A configured default is still subject to evidence (the 1,800 s hit).
        assert_eq!(routes.lifetime("a/opus", &limits(Some(600), true), 0).unwrap(), Lifetime { secs: 1_801, source: Source::Learned });
        assert_eq!(routes.lifetime("a/opus", &limits(Some(7_200), true), 0).unwrap(), Lifetime { secs: 7_200, source: Source::Default });
        // Cold starts switched off for a model: no lifetime, whatever was learned.
        assert_eq!(routes.lifetime("a/opus", &limits(None, false), 0), None);
        assert_eq!(routes.lifetime("a/opus", &limits(Some(300), false), 0), None);
        // The table is bounded.
        for i in 0..2 * MAX_ROUTES {
            routes.record(&format!("r/{i}"), Outcome::Hit, 100, 0);
        }
        assert_eq!(routes.routes.len(), MAX_ROUTES);
    }

    #[test]
    fn cache_lifetime_evidence_round_trips_and_old_data_loads() {
        let mut routes = RouteLifetimes::default();
        routes.record("anthropic/claude-opus-5", Outcome::Hit, 240, 1_000);
        routes.record("anthropic/claude-opus-5", Outcome::Miss, 1_380, 2_000);
        let json = serde_json::to_string(&routes).unwrap();
        assert_eq!(json, r#"{"anthropic/claude-opus-5":{"longest_hit_secs":240,"misses":[{"pause_secs":1380,"observed_ms":2000}],"hit_count":1,"miss_count":1}}"#);
        assert_eq!(serde_json::from_str::<RouteLifetimes>(&json).unwrap(), routes);
        let empty: CacheEvidence = serde_json::from_str("{}").unwrap();
        assert_eq!(empty, CacheEvidence::default());
    }

    #[test]
    fn cache_lifetime_turn_probe_expects_the_right_prefix_per_request() {
        let mut probe = TurnProbe { first_prefix_bytes: None, turn_prefix_bytes: Some(900), ..Default::default() };
        assert_eq!(probe.expected_prefix_bytes(), None, "first request after a reset: no hit expected");
        probe.requests = 1;
        assert_eq!(probe.expected_prefix_bytes(), Some(900), "later rounds read what the first wrote");
        let continued = TurnProbe { first_prefix_bytes: Some(700), turn_prefix_bytes: Some(900), ..Default::default() };
        assert_eq!(continued.expected_prefix_bytes(), Some(700));
    }
}
