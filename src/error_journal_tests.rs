use super::*;

fn parsed(journal: &Journal, after: u64) -> serde_json::Value {
    serde_json::from_str(&journal.query(after, 1_000)).unwrap()
}

#[test]
fn error_journal_ring_is_bounded_and_counters_keep_totals() {
    let mut journal = Journal::default();
    for i in 0..(RING_CAPACITY + 25) {
        journal.record_at(i as u64, Category::Task, "Mod.loop", &format!("task {i} failed"));
    }
    assert_eq!(journal.len(), RING_CAPACITY);
    let value = parsed(&journal, 0);
    assert_eq!(value["last_seq"], (RING_CAPACITY + 25) as u64);
    assert_eq!(value["dropped"], true, "entries before the ring start were lost");
    assert_eq!(value["entries"].as_array().unwrap().len(), QUERY_LIMIT);
    let counters = value["fingerprints"].as_object().unwrap();
    assert_eq!(counters.len(), 1, "numbers are normalized into one fingerprint");
    assert_eq!(counters.values().next().unwrap()["count"], (RING_CAPACITY + 25) as u64);
}

#[test]
fn error_journal_dedupes_by_fingerprint_and_separates_sources() {
    let mut journal = Journal::default();
    journal.record_at(10, Category::Library, "Mod", "failed to load line 3");
    journal.record_at(20, Category::Library, "Mod", "failed to load line 9");
    journal.record_at(30, Category::Library, "Other", "failed to load line 3");
    let value = parsed(&journal, 0);
    let entries = value["entries"].as_array().unwrap();
    assert_eq!(entries[0]["fingerprint"], entries[1]["fingerprint"]);
    assert_ne!(entries[0]["fingerprint"], entries[2]["fingerprint"]);
    assert_eq!(entries[1]["count"], 2);
    assert_eq!(entries[1]["first_seen"], 10);
    assert_eq!(entries[1]["last_seen"], 20);
    assert_eq!(value["fingerprints"].as_object().unwrap().len(), 2);
    assert_ne!(
        fingerprint(Category::Library, "Mod", "x"),
        fingerprint(Category::Route, "Mod", "x"),
        "category is part of the fingerprint"
    );
}

#[test]
fn error_journal_after_seq_returns_only_newer_entries() {
    let mut journal = Journal::default();
    let first = journal.record_at(1, Category::Session, "save", "disk full");
    journal.record_at(2, Category::Session, "save", "disk full");
    let value = parsed(&journal, first);
    let entries = value["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0]["seq"], first + 1);
    assert_eq!(value["dropped"], false);
    assert!(parsed(&journal, 99)["entries"].as_array().unwrap().is_empty());
}

/// Token-shaped fixtures are assembled at runtime so no source literal matches a real
/// credential pattern (secret scanners), while the redaction tests see the same shapes.
pub(crate) fn synthetic_bot_token(id: &str, filler: char) -> String {
    format!("{id}:{}", filler.to_string().repeat(35))
}

/// An API-key-shaped string: provider prefix + digit-bearing opaque run (>= 32 chars).
pub(crate) fn synthetic_api_key() -> String {
    let body: String = (0..36).map(|i| if i % 3 == 0 { char::from(b'0' + (i % 10) as u8) } else { 'q' }).collect();
    ["sk", "fixture", &body].join("-")
}

#[test]
fn error_journal_redacts_urls_to_host_tokens_and_caps_messages() {
    let token = synthetic_bot_token(&"9".repeat(9), 'Z');
    let message = format!(
        "error sending request for url (https://api.telegram.org/{}{token}/getUpdates?offset=7&token=SECRETQ)",
        "bot"
    );
    let redacted = redact(&message);
    assert!(redacted.contains("https://api.telegram.org"), "{redacted}");
    let filler = "Z".repeat(35);
    for leak in ["999999999", filler.as_str(), "getUpdates", "offset", "SECRETQ"] {
        assert!(!redacted.contains(leak), "{leak} leaked: {redacted}");
    }
    let schemeless_token = format!("{}{}", "bot", synthetic_bot_token(&"8".repeat(9), 'Y'));
    let schemeless = redact(&format!("token {schemeless_token} rejected"));
    assert!(!schemeless.contains(&"Y".repeat(35)) && schemeless.contains("<token>"), "{schemeless}");
    let userinfo = redact("POST http://user:pass@10.0.0.1:3002/api/x?k=v failed");
    assert!(userinfo.contains("http://10.0.0.1:3002") && !userinfo.contains("pass"), "{userinfo}");
    let long = redact(&"é".repeat(2000));
    assert!(long.len() <= MESSAGE_CAP && long.ends_with("..."));
    let key = synthetic_api_key();
    let opaque = redact(&format!("Bearer {key} rejected"));
    assert!(!opaque.contains(&key) && opaque.contains("<redacted>"), "{opaque}");
    let identifier = "TelegramBot.telegram_migration_private_library_roundtrip";
    assert_eq!(redact(identifier), identifier, "long identifiers stay readable");
}

#[test]
fn error_journal_global_record_and_query_roundtrip() {
    let marker = "Journal.global_roundtrip_marker";
    let seq = record(Category::Refusal, &marker, "shell policy refused");
    let value: serde_json::Value = serde_json::from_str(&query(seq - 1)).unwrap();
    assert!(value["entries"].as_array().unwrap().iter().any(|e| e["source"] == marker
        && e["category"] == "refusal"));
}

#[test]
fn error_journal_route_errors_keep_only_route_and_first_line_without_body() {
    let body = "{\"message\":{\"text\":\"ROUTE_SECRET_CHAT\"}}";
    let error = format!("handler failed on {body:?}\ncaused by: {body}\nmore detail");
    let seq = record_route_error("POST /hook", "Route.handler_unit", &error, body);
    let value: serde_json::Value = serde_json::from_str(&query(seq - 1)).unwrap();
    let entry = value["entries"].as_array().unwrap().iter()
        .find(|e| e["source"] == "Route.handler_unit").unwrap().clone();
    let message = entry["message"].as_str().unwrap();
    assert_eq!(message, "POST /hook: handler failed on \"<request body>\"");
    assert_eq!(entry["category"], "route");
    // Short bodies are not substituted (they would mangle unrelated text).
    let seq = record_route_error("POST /hook", "Route.handler_short", "bad id", "id");
    let value: serde_json::Value = serde_json::from_str(&query(seq - 1)).unwrap();
    assert!(value["entries"].as_array().unwrap().iter()
        .any(|e| e["source"] == "Route.handler_short" && e["message"] == "POST /hook: bad id"));
    assert!(value.get("snapshot_id").and_then(|v| v.as_str()).is_some_and(|s| s.len() == 32));
}
