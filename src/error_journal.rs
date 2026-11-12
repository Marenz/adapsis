//! In-process runtime error journal (watchdog feed).
//!
//! Failure sites call [`record`] with one line; nothing here changes their behaviour.
//! Entries live in a bounded ring and per-fingerprint counters, and are exposed to
//! module code only through the read-only `runtime_errors(after_seq)` builtin.
//!
//! Messages are REDACTED before storage: URLs keep scheme/host/port only, Telegram
//! bot tokens and long secret-looking runs are masked, and every message is capped
//! at 1 KiB. Callers pass error text only, never request bodies or chat text. The
//! journal is still untrusted data: an error may quote attacker-controlled input,
//! so consumers (the Watchdog) must treat it as data, never as instructions.
use regex::Regex;
use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, OnceLock};

/// Ring size. Older entries drop off; fingerprint counters keep the totals.
pub const RING_CAPACITY: usize = 500;
/// Distinct fingerprints retained; the least recently seen is evicted beyond this.
pub const FINGERPRINT_CAPACITY: usize = 2000;
/// Redacted message cap in bytes (UTF-8 safe).
pub const MESSAGE_CAP: usize = 1024;
/// Most entries one `runtime_errors` call returns (newest kept).
pub const QUERY_LIMIT: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    Library,
    Route,
    Task,
    Provider,
    Session,
    Panic,
    Startup,
    /// A guard refused an operation (shell policy, destructive guard, permission).
    Refusal,
}

impl Category {
    fn label(self) -> &'static str {
        match self {
            Self::Library => "library",
            Self::Route => "route",
            Self::Task => "task",
            Self::Provider => "provider",
            Self::Session => "session",
            Self::Panic => "panic",
            Self::Startup => "startup",
            Self::Refusal => "refusal",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Entry {
    pub seq: u64,
    /// Unix seconds.
    pub time: u64,
    pub category: Category,
    pub source: String,
    pub fingerprint: String,
    pub message: String,
    /// Occurrences of this fingerprint so far, including this one.
    pub count: u64,
    pub first_seen: u64,
    pub last_seen: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Counter {
    pub category: Category,
    pub source: String,
    pub count: u64,
    pub first_seen: u64,
    pub last_seen: u64,
    pub last_seq: u64,
}

#[derive(Default)]
pub struct Journal {
    next_seq: u64,
    ring: VecDeque<Entry>,
    counters: HashMap<String, Counter>,
}

impl Journal {
    pub fn record_at(&mut self, now: u64, category: Category, source: &str, message: &str) -> u64 {
        self.record_keyed(now, category, source, message, None)
    }

    /// `key`: fingerprint input that replaces the message (used where the message
    /// differs only in numbers that matter, such as a panic's line).
    fn record_keyed(&mut self, now: u64, category: Category, source: &str, message: &str, key: Option<&str>) -> u64 {
        let source = redact(source);
        let message = redact(message);
        let fingerprint = match key {
            Some(key) => exact_fingerprint(category, key),
            None => fingerprint(category, &source, &message),
        };
        self.next_seq += 1;
        let seq = self.next_seq;
        let counter = self.counters.entry(fingerprint.clone()).or_insert_with(|| Counter {
            category,
            source: source.clone(),
            count: 0,
            first_seen: now,
            last_seen: now,
            last_seq: seq,
        });
        counter.count += 1;
        counter.last_seen = now;
        counter.last_seq = seq;
        let entry = Entry {
            seq,
            time: now,
            category,
            source,
            fingerprint,
            message,
            count: counter.count,
            first_seen: counter.first_seen,
            last_seen: now,
        };
        if self.ring.len() == RING_CAPACITY {
            self.ring.pop_front();
        }
        self.ring.push_back(entry);
        self.evict_counters();
        seq
    }

    fn evict_counters(&mut self) {
        while self.counters.len() > FINGERPRINT_CAPACITY {
            let Some(oldest) = self
                .counters
                .iter()
                .min_by_key(|(_, c)| c.last_seq)
                .map(|(k, _)| k.clone())
            else {
                return;
            };
            self.counters.remove(&oldest);
        }
    }

    /// JSON: `{"now", "last_seq", "dropped", "entries":[…seq > after_seq…], "fingerprints":{…}}`.
    /// `dropped` is true when the ring no longer holds every entry after `after_seq`.
    /// `now` (unix s) gives module code a clock for rate limits.
    pub fn query(&self, after_seq: u64, now: u64) -> String {
        let mut entries: Vec<&Entry> = self.ring.iter().filter(|e| e.seq > after_seq).collect();
        let oldest_held = self.ring.front().map_or(self.next_seq + 1, |e| e.seq);
        let mut dropped = self.next_seq > after_seq && oldest_held > after_seq + 1;
        if entries.len() > QUERY_LIMIT {
            entries.drain(..entries.len() - QUERY_LIMIT);
            dropped = true;
        }
        serde_json::json!({
            // Unique per query: lets the Watchdog own its busy latch (owner token).
            "snapshot_id": uuid::Uuid::new_v4().simple().to_string(),
            "now": now,
            "last_seq": self.next_seq,
            "dropped": dropped,
            "entries": entries,
            "fingerprints": self.counters,
        })
        .to_string()
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.ring.len()
    }
}

fn journal() -> &'static Mutex<Journal> {
    static JOURNAL: OnceLock<Mutex<Journal>> = OnceLock::new();
    JOURNAL.get_or_init(|| Mutex::new(Journal::default()))
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The single hook API. Never panics and never blocks on a poisoned lock.
/// Returns the entry's sequence number (callers normally ignore it).
pub fn record(category: Category, source: impl std::fmt::Display, message: impl std::fmt::Display) -> u64 {
    let (source, message) = (source.to_string(), message.to_string());
    let mut guard = match journal().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    guard.record_at(now(), category, &source, &message)
}

/// Route handler failure, journaled structurally so the guarantee does not depend on
/// handler hygiene: occurrences of the request body (raw or escaped) become
/// `<request body>`, then only the first line (the error's kind) is kept, prefixed
/// with the route. `record` redacts what remains. Short bodies are not substituted.
pub fn record_route_error(route: &str, handler: &str, error: &str, body: &str) -> u64 {
    let mut text = error.to_string();
    if body.len() >= 4 {
        let escaped = format!("{body:?}");
        let escaped = &escaped[1..escaped.len() - 1];
        text = text.replace(body, "<request body>");
        if escaped != body {
            text = text.replace(escaped, "<request body>");
        }
    }
    let kind = text.lines().map(str::trim).find(|line| !line.is_empty()).unwrap_or("error");
    record(Category::Route, handler, format!("{route}: {kind}"))
}

/// The process panic hook (#134). The panic MESSAGE is never journaled: Rust's
/// slice and index panics quote the value (`byte index 1 is not a char boundary;
/// it is inside 'ä' of `<user text>``), and the journal feeds `runtime_errors`,
/// the Watchdog's triage prompt and the admin conversation. Only the source
/// location and the payload's length are recorded. The message itself goes to
/// the content-gated diagnostic log (`ADAPSIS_LOG_CONTENT=1`).
pub fn panic_hook(info: &std::panic::PanicHookInfo<'_>) {
    crate::logging::diagnostic("panic", info);
    let payload = info.payload();
    let payload_bytes = payload
        .downcast_ref::<&str>()
        .map(|text| text.len())
        .or_else(|| payload.downcast_ref::<String>().map(String::len));
    let location = info.location().map(|at| (at.file(), at.line(), at.column()));
    record_panic(location, payload_bytes);
    eprintln!("{}", std::backtrace::Backtrace::force_capture());
}

/// Journal a panic by location and payload length. Never waits for the lock (the
/// panicking thread may hold it). One fingerprint per source line, so two panic
/// sites in one file are two findings and one site with changing data is one.
pub fn record_panic(location: Option<(&str, u32, u32)>, payload_bytes: Option<usize>) {
    let place = location.map_or_else(|| "unknown location".to_string(), |(file, line, column)| format!("{file}:{line}:{column}"));
    let payload = payload_bytes.map_or_else(|| "non-text payload".to_string(), |bytes| format!("message of {bytes} bytes not recorded"));
    if let Ok(mut guard) = journal().try_lock() {
        guard.record_keyed(now(), Category::Panic, "panic", &format!("panic at {place} ({payload})"), Some(&place));
    }
}

/// A panic inside a spawned task, as text for the journal and the task registry.
/// `JoinError`'s own Display quotes the panic message (tokio >= 1.4x: "task N
/// panicked with message \"...\""), so it is never used for a panic.
pub fn describe_join_error(error: tokio::task::JoinError) -> String {
    if !error.is_panic() {
        return "was cancelled".to_string();
    }
    let payload = error.into_panic();
    let bytes = payload
        .downcast_ref::<&str>()
        .map(|text| text.len())
        .or_else(|| payload.downcast_ref::<String>().map(String::len));
    match bytes {
        Some(bytes) => format!("panicked (message of {bytes} bytes not recorded; see the panic entry for the location)"),
        None => "panicked (non-text payload)".to_string(),
    }
}

/// Read-only JSON snapshot for the `runtime_errors` builtin.
pub fn query(after_seq: u64) -> String {
    match journal().lock() {
        Ok(guard) => guard.query(after_seq, now()),
        Err(poisoned) => poisoned.into_inner().query(after_seq, now()),
    }
}

/// Fingerprint = category + source + message with numbers, hex runs and quoted
/// values collapsed, so `task 17 failed` and `task 18 failed` group together.
pub fn fingerprint(category: Category, source: &str, message: &str) -> String {
    static NUMBERS: OnceLock<Regex> = OnceLock::new();
    let numbers = NUMBERS.get_or_init(|| Regex::new(r"(?i)\b(?:0x)?[0-9a-f]*[0-9][0-9a-f]*\b").unwrap());
    let normalize = |text: &str| numbers.replace_all(text, "#").into_owned();
    let raw = format!("{}|{}|{}", category.label(), normalize(source), normalize(message));
    use sha2::{Digest, Sha256};
    format!("{}:{:x}", category.label(), Sha256::digest(raw.as_bytes()))[..category.label().len() + 17].to_string()
}

/// Fingerprint from an exact key (no number collapsing).
fn exact_fingerprint(category: Category, key: &str) -> String {
    use sha2::{Digest, Sha256};
    let raw = format!("{}|exact|{key}", category.label());
    format!("{}:{:x}", category.label(), Sha256::digest(raw.as_bytes()))[..category.label().len() + 17].to_string()
}

/// Redaction for journal storage: host-only URLs, masked tokens, 1 KiB cap.
pub fn redact(text: &str) -> String {
    static URL: OnceLock<Regex> = OnceLock::new();
    static TOKEN: OnceLock<Regex> = OnceLock::new();
    static SECRET: OnceLock<Regex> = OnceLock::new();
    let url = URL.get_or_init(|| Regex::new(r#"(?i)\b[a-z][a-z0-9+.-]*://[^\s"'<>`)\]]+"#).unwrap());
    // Telegram bot tokens may appear without a URL scheme (`bot123:ABC…`).
    let token = TOKEN.get_or_init(|| Regex::new(r"\b(?:bot)?\d{5,}:[A-Za-z0-9_-]{20,}").unwrap());
    // Long opaque runs (API keys, bearer tokens) are not diagnostically useful.
    // Identifiers stay readable: only runs that also carry several digits are masked.
    let secret = SECRET.get_or_init(|| Regex::new(r"[A-Za-z0-9_\-]{32,}").unwrap());
    let text = url.replace_all(text, |caps: &regex::Captures| crate::coroutine::host_only(&caps[0]));
    let text = token.replace_all(&text, "<token>");
    let text = secret.replace_all(&text, |caps: &regex::Captures| {
        let run = &caps[0];
        if run.bytes().filter(u8::is_ascii_digit).count() >= 4 { "<redacted>".to_string() } else { run.to_string() }
    });
    cap(&text, MESSAGE_CAP)
}

fn cap(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        return text.to_string();
    }
    let mut end = limit.saturating_sub(3);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &text[..end])
}

#[cfg(test)]
#[path = "error_journal_tests.rs"]
pub(crate) mod tests;
#[cfg(test)]
#[path = "error_journal_hook_tests.rs"]
mod hook_tests;
