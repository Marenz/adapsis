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
    registry.lock().unwrap().insert(7, crate::coroutine::TaskInfo {
        id: 7, function_name: "JournalTask.boom".into(), status: crate::coroutine::WaitReason::Running,
        started_at: "0".into(),
    });
    // tokio's JoinError quotes the panic message; the journal and the task
    // registry must not (#134).
    let task = tokio::task::spawn_blocking(|| panic!("journal hook synthetic panic TASK_PANIC_SENTINEL"));
    crate::monitor_spawned_task(task, registry.clone(), 7, "JournalTask.boom".into()).await;
    let entries = entries_since(before);
    let entry = find(&entries, "task", "JournalTask.boom").expect("task entry");
    let message = entry["message"].as_str().unwrap();
    assert_eq!(message, "panicked (message of 48 bytes not recorded; see the panic entry for the location)");
    assert!(!serde_json::to_string(&entries).unwrap().contains("TASK_PANIC_SENTINEL"));
    let crate::coroutine::WaitReason::Failed(stored) = registry.lock().unwrap().get(&7).unwrap().status.clone() else { panic!("failed") };
    assert!(stored.contains("panicked") && !stored.contains("TASK_PANIC_SENTINEL"), "{stored}");
    // A cancelled task is described without a payload too.
    let cancelled = tokio::spawn(std::future::pending::<()>());
    cancelled.abort();
    assert_eq!(crate::error_journal::describe_join_error(cancelled.await.unwrap_err()), "was cancelled");
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

#[test]
fn managed_python_load_failure_is_journaled_with_name_and_line_only() {
    let dir = tempfile::tempdir().unwrap();
    let secret = "MANAGED_SOURCE_CANARY_77";
    std::fs::write(
        dir.path().join("library-sources.json"),
        r#"{"version":1,"modules":{"JournalPy":{"format":"python","dependencies":[]}}}"#,
    )
    .unwrap();
    std::fs::write(
        dir.path().join("JournalPy.py"),
        format!("def ok() -> int:\n    return 1\n\ndef bad(x: int -> str:\n    return \"{secret}\"\n"),
    )
    .unwrap();
    let before = last_seq();
    let mut program = crate::ast::Program::default();
    let mut state = crate::library::LibraryState::new();
    crate::library::load_module_files(&mut program, dir.path(), &mut state);
    assert!(state.require_authoritative_sources().is_err());
    let entries = entries_since(before);
    let library: Vec<_> = entries.iter().filter(|e| e.to_string().contains("JournalPy")).collect();
    assert_eq!(library.len(), 1, "exactly one entry for the managed failure: {library:#?}");
    assert_eq!(library[0]["category"], "library");
    assert_eq!(library[0]["source"], "library-sources");
    let message = library[0]["message"].as_str().unwrap();
    assert!(message.contains("JournalPy.py") && message.contains("at line"), "{message}");
    assert!(entries.iter().all(|e| !e.to_string().contains(secret) && !e.to_string().contains("def bad")));
}

#[test]
fn managed_startup_rerun_journals_one_entry_per_module_without_case_content() {
    let dir = tempfile::tempdir().unwrap();
    let secret = "RERUN_CASE_CANARY_55";
    std::fs::write(
        dir.path().join("library-sources.json"),
        r#"{"version":1,"modules":{"JournalRerun":{"format":"python","dependencies":[]}}}"#,
    )
    .unwrap();
    // Three failing cases in two functions of one module.
    std::fs::write(
        dir.path().join("JournalRerun.py"),
        format!("def a(x: str) -> str:\n    return x\n\ndef b(x: str) -> str:\n    return x\n\n@stored_test(\"JournalRerun.a\")\ndef test_case_1() -> None:\n    assert a(\"{secret}\") == \"no\"\n\n@stored_test(\"JournalRerun.a\")\ndef test_case_2() -> None:\n    assert a(\"x\") == \"no\"\n\n@stored_test(\"JournalRerun.b\")\ndef test_case_3() -> None:\n    assert b(\"x\") == \"no\"\n"),
    )
    .unwrap();
    let before = last_seq();
    let mut program = crate::ast::Program::default();
    let mut state = crate::library::LibraryState::new();
    crate::library::load_module_files(&mut program, dir.path(), &mut state);
    state.require_authoritative_sources().unwrap();
    let entries = entries_since(before);
    let ours: Vec<_> = entries.iter().filter(|e| e.to_string().contains("JournalRerun")).collect();
    assert_eq!(ours.len(), 1, "one entry per module, not per case: {ours:#?}");
    assert_eq!(ours[0]["category"], "library");
    assert_eq!(ours[0]["source"], "JournalRerun");
    let message = ours[0]["message"].as_str().unwrap();
    assert!(message.contains("stay untested"), "{message}");
    assert!(entries.iter().all(|e| !e.to_string().contains(secret)));
}
