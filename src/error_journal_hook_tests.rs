//! Each feeding hook writes exactly the categorized, redacted entry it should.
//! The journal is process-global, so assertions use unique sources and seq bounds.
use super::*;
use crate::coroutine::{CoroutineHandle, IoRequest, Runtime, ShellPolicy, TurnIdentity};
use crate::eval::Value;

fn last_seq() -> u64 {
    let value: serde_json::Value = serde_json::from_str(&query(u64::MAX - 1)).unwrap();
    value["last_seq"].as_u64().unwrap()
}

fn entries_since(seq: u64) -> Vec<serde_json::Value> {
    let value: serde_json::Value = serde_json::from_str(&query(seq)).unwrap();
    value["entries"].as_array().unwrap().clone()
}

fn find<'a>(entries: &'a [serde_json::Value], category: &str, source: &str) -> Option<&'a serde_json::Value> {
    entries.iter().find(|e| e["category"] == category && e["source"] == source)
}

#[test]
fn library_reload_error_is_journaled_with_module_source() {
    let before = last_seq();
    let dir = crate::library::ensure_library_dir().unwrap();
    let path = dir.join("JournalHookBroken.ax");
    std::fs::write(&path, "+module JournalHookBroken\n+fn broken (\n").unwrap();
    let state = crate::library::LibraryState::new();
    let mut program = crate::ast::Program::default();
    let result = crate::library::reload_module(&mut program, "JournalHookBroken", Some(&state));
    let _ = std::fs::remove_file(&path);
    assert!(result.is_err());
    let entries = entries_since(before);
    assert!(find(&entries, "library", "JournalHookBroken").is_some(), "{entries:#?}");
}

#[test]
fn library_save_error_is_journaled() {
    let before = last_seq();
    crate::library::LibraryState::new().record_error("failed to persist module `JournalSave`: disk full".into());
    let entries = entries_since(before);
    // Concurrent tests also record `library` entries: match this one by its message.
    assert!(entries.iter().any(|e| e["category"] == "library" && e["source"] == "library"
        && e["message"].as_str().unwrap_or("").contains("JournalSave")), "{entries:#?}");
}

#[test]
fn identical_route_reregistration_at_startup_is_not_journaled() {
    // #113: a persisted session already holds the route; re-adding it at startup
    // returns an `updated route …` outcome that must not look like an error.
    let route = || crate::ast::HttpRoute {
        method: "POST".into(),
        path: "/journal/same".into(),
        handler_fn: "JournalRoutes.same".into(),
    };
    let mut session = crate::session::Session::new();
    session.runtime.http_routes.push(route());
    session.program.pending_routes.push(route());
    let before = last_seq();
    crate::register_library_routes(&mut session);
    assert_eq!(session.runtime.http_routes.len(), 1);
    let entries = entries_since(before);
    assert!(
        entries.iter().all(|e| !e["message"].as_str().unwrap_or("").contains("/journal/same")),
        "route outcome journaled: {entries:#?}"
    );
}

#[tokio::test]
async fn spawned_task_panic_is_journaled_as_task_failure() {
    let before = last_seq();
    let registry: crate::coroutine::TaskRegistry = Default::default();
    let task = tokio::task::spawn_blocking(|| panic!("journal hook synthetic panic"));
    crate::monitor_spawned_task(task, registry, 7, "JournalTask.boom".into()).await;
    let entries = entries_since(before);
    let entry = find(&entries, "task", "JournalTask.boom").expect("task entry");
    assert!(entry["message"].as_str().unwrap().contains("panicked"), "{entry}");
}

#[test]
fn guest_boundary_refusal_is_journaled_without_arguments() {
    let before = last_seq();
    let (tx, _rx) = tokio::sync::mpsc::channel(1);
    let guest = TurnIdentity {
        context: "telegram:user:1".into(),
        principal: "telegram:user:1".into(),
        may_write: false,
        guest_tools: Some(vec!["web_search".into()]),
    };
    let handle = CoroutineHandle::new(tx).with_turn(Some(guest));
    let secret_path = "/home/guest-journal-secret-path";
    assert!(handle.execute_await("read_file", &[Value::string(secret_path)]).is_err());
    let entries = entries_since(before);
    let entry = find(&entries, "refusal", "guest_tool").expect("refusal entry");
    assert!(entry["message"].as_str().unwrap().contains("read_file"));
    assert!(entries.iter().all(|e| !e.to_string().contains(secret_path)));
}

#[tokio::test]
async fn shell_policy_refusal_is_journaled_without_the_command() {
    let before = last_seq();
    let (mut runtime, _rx) = Runtime::new();
    runtime.shell_policy = ShellPolicy::parse("deny");
    let (reply, rx) = tokio::sync::oneshot::channel();
    runtime
        .handle_io(IoRequest::ShellExec { command: "echo journal-shell-secret-arg".into(), reply })
        .await;
    assert!(rx.await.unwrap().is_err());
    let entries = entries_since(before);
    assert!(find(&entries, "refusal", "shell_exec").is_some(), "{entries:#?}");
    assert!(entries.iter().all(|e| !e.to_string().contains("journal-shell-secret-arg")));
}

#[test]
fn boot_time_library_load_failure_is_journaled_once_without_source_text() {
    let dir = tempfile::tempdir().unwrap();
    let secret = "BOOT_SOURCE_CANARY_41";
    std::fs::write(
        dir.path().join("BootBroken.ax"),
        format!("+module BootBroken\n+fn ok ()->Int\n  +return 1\n+fn bad (x:Int\n  +return \"{secret}\"\n"),
    )
    .unwrap();
    std::fs::write(dir.path().join("BootFine.ax"), "+module BootFine\n+fn one ()->Int\n  +return 1\n").unwrap();
    let before = last_seq();
    let mut program = crate::ast::Program::default();
    let mut state = crate::library::LibraryState::new();
    crate::library::load_module_files(&mut program, dir.path(), &mut state);
    assert_eq!(state.loaded_modules, vec!["BootFine".to_string()]);
    let entries = entries_since(before);
    // The journal is process-global and other tests run concurrently: count every
    // entry that mentions the module anywhere, so a raw duplicate would be caught.
    let library: Vec<_> = entries.iter().filter(|e| e.to_string().contains("BootBroken")).collect();
    assert_eq!(library.len(), 1, "exactly one entry for the broken module: {library:#?}");
    assert_eq!(library[0]["category"], "library");
    assert_eq!(library[0]["source"], "BootBroken");
    let message = library[0]["message"].as_str().unwrap();
    assert!(message.starts_with("failed to load BootBroken.ax at startup"), "{message}");
    assert!(entries.iter().all(|e| !e.to_string().contains(secret) && !e.to_string().contains("+fn")));
}
