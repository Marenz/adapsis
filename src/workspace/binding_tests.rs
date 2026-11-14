use super::*;
use crate::workspace::read::FileFacts;
use crate::workspace::registry::Repository;
use std::sync::{Arc, Barrier, Mutex};

fn store() -> (tempfile::TempDir, BindingStore) {
    let temp = tempfile::tempdir().unwrap();
    let store = BindingStore::at(&temp.path().join("state"));
    (temp, store)
}

/// A workspace value for `repo`/`name`; its worktree does not have to exist.
fn workspace(temp: &tempfile::TempDir, repo: &str, name: &str) -> Workspace {
    let repository = Repository {
        name: repo.to_string(),
        checkout: temp.path().join("checkout"),
        worktree_parent: temp.path().join("worktrees").join(repo),
        base: "main".to_string(),
        read_roots: Vec::new(),
        env: Default::default(),
    };
    let path = temp.path().join("worktrees").join(repo).join(name);
    std::fs::create_dir_all(&path).unwrap();
    Workspace::at(&repository, name, &path, "main").unwrap()
}

fn facts(size: u64) -> FileFacts {
    FileFacts { size, mtime_ns: 7, sha256: Some("ab".repeat(32)) }
}

/// Every other conversation is idle.
fn all_idle(_: &str) -> Option<IdleGuard> {
    Some(Box::new(()))
}

fn nobody_idle(_: &str) -> Option<IdleGuard> {
    None
}

fn entries(store: &BindingStore) -> Vec<String> {
    let mut names: Vec<String> =
        std::fs::read_dir(store.root().join("_bindings")).unwrap().map(|entry| entry.unwrap().file_name().into_string().unwrap()).collect();
    names.sort();
    names
}

fn journal(after: u64) -> serde_json::Value {
    serde_json::from_str(&crate::error_journal::query(after)).unwrap()
}

#[test]
fn a_binding_survives_a_new_store_and_keeps_its_ledger() {
    let (temp, store) = store();
    let one = workspace(&temp, "adapsis", "fix-115");
    assert_eq!(store.load("telegram:100000001").unwrap(), None, "no state directory yet");
    store.release("telegram:100000001", None).unwrap();
    assert!(!store.save_ledger("telegram:100000001", &one, &ReadLedger::default()).unwrap(), "nothing to save into");
    assert_eq!(store.claim("telegram:100000001", &one, &nobody_idle).unwrap(), Claimed { taken: false });
    let mut ledger = ReadLedger::default();
    ledger.record("src/main.rs", facts(10));
    assert!(store.save_ledger("telegram:100000001", &one, &ledger).unwrap());
    // A restart: nothing but the directory is shared.
    let again = BindingStore::at(&temp.path().join("state"));
    let loaded = again.load("telegram:100000001").unwrap().expect("stored");
    assert_eq!((loaded.repo.as_str(), loaded.workspace.as_str()), ("adapsis", "fix-115"));
    assert_eq!(loaded.ledger.recorded("src/main.rs"), Some(&facts(10)));
    // One record, no temporary file, and a name that says nothing about the conversation.
    let names = entries(&again);
    assert_eq!(names.len(), 1, "{names:?}");
    assert!(names[0].len() == 37 && names[0].ends_with(".json") && names[0][..32].bytes().all(|b| b.is_ascii_hexdigit()), "{names:?}");
    assert!(!names[0].contains("telegram") && !names[0].contains("100000001"));
    // Claiming another workspace replaces the record and empties the ledger.
    let two = workspace(&temp, "adapsis", "other");
    again.claim("telegram:100000001", &two, &nobody_idle).unwrap();
    assert_eq!(again.load("telegram:100000001").unwrap().unwrap(), Binding::new("adapsis", "other"));
    assert!(!again.save_ledger("telegram:100000001", &one, &ledger).unwrap(), "the old workspace is no longer this conversation's");
    assert!(again.holders("adapsis", "fix-115").unwrap().is_empty() && entries(&again).len() == 1);
    // Release only what is named.
    again.release("telegram:100000001", Some(&one)).unwrap();
    assert!(again.load("telegram:100000001").unwrap().is_some(), "a newer binding is not released for an old workspace");
    again.release("telegram:100000001", Some(&two)).unwrap();
    assert_eq!(again.load("telegram:100000001").unwrap(), None);
    again.release("telegram:100000001", None).unwrap();
    use std::os::unix::fs::PermissionsExt;
    assert_eq!(std::fs::metadata(temp.path().join("state/_bindings")).unwrap().permissions().mode() & 0o777, 0o700);
}

#[test]
fn conversation_keys_never_share_a_record() {
    let (temp, store) = store();
    let long = "k".repeat(300);
    let keys = ["telegram:1", "telegram_1", "telegram/1", "../../etc/passwd", "a/../b", "..", ".", "ключ", long.as_str()];
    for (index, key) in keys.iter().enumerate() {
        store.claim(key, &workspace(&temp, "repo", &format!("ws-{index}")), &nobody_idle).unwrap();
    }
    for (index, key) in keys.iter().enumerate() {
        assert_eq!(store.load(key).unwrap().unwrap().workspace, format!("ws-{index}"), "{key}");
        assert_eq!(store.holders("repo", &format!("ws-{index}")).unwrap(), [key.to_string()]);
    }
    assert_eq!(entries(&store).len(), keys.len());
    assert!(std::fs::read_dir(store.root().join("_bindings")).unwrap().all(|e| e.unwrap().file_type().unwrap().is_file()));
    for bad in ["", &"x".repeat(513)] {
        assert!(matches!(store.claim(bad, &workspace(&temp, "r", "w"), &all_idle), Err(Error::Binding(_))));
        assert!(matches!(store.load(bad), Err(Error::Binding(_))));
    }
}

/// DESIGN T4. A busy holder keeps its workspace; an idle one loses it to exactly one
/// claimant, which starts with an empty ledger.
#[test]
fn a_claim_takes_a_workspace_only_from_idle_holders() {
    let (temp, store) = store();
    let one = workspace(&temp, "adapsis", "one");
    store.claim("ctx-a", &one, &nobody_idle).unwrap();
    let mut ledger = ReadLedger::default();
    ledger.record("a.rs", facts(1));
    store.save_ledger("ctx-a", &one, &ledger).unwrap();
    store.claim("ctx-c", &workspace(&temp, "gateway", "one"), &nobody_idle).unwrap();

    let refused = store.claim("ctx-b", &one, &nobody_idle).unwrap_err();
    assert_eq!(
        refused.to_string(),
        "workspace adapsis/one is in use by another conversation that is working right now; nothing was changed. Try again when it has finished"
    );
    assert!(!refused.to_string().contains("ctx-a"), "the holder is not named");
    assert_eq!(store.holders("adapsis", "one").unwrap(), ["ctx-a"]);
    assert_eq!(store.load("ctx-a").unwrap().unwrap().ledger, ledger);
    assert_eq!(store.load("ctx-b").unwrap(), None);

    let asked = Mutex::new(Vec::new());
    let taken = store.claim("ctx-b", &one, &|holder| {
        asked.lock().unwrap().push(holder.to_string());
        Some(Box::new(()))
    });
    assert_eq!(taken.unwrap(), Claimed { taken: true });
    assert_eq!(*asked.lock().unwrap(), ["ctx-a"], "only the holders of this workspace are asked");
    assert_eq!(store.holders("adapsis", "one").unwrap(), ["ctx-b"]);
    assert_eq!(store.load("ctx-a").unwrap(), None);
    assert!(store.load("ctx-b").unwrap().unwrap().ledger.is_empty());
    assert_eq!(store.holders("gateway", "one").unwrap(), ["ctx-c"]);
    // The old holder's late ledger write does not bring its record back.
    assert!(!store.save_ledger("ctx-a", &one, &ledger).unwrap());
    assert_eq!(store.holders("adapsis", "one").unwrap(), ["ctx-b"]);
}

/// The reviewer's race: two conversations claim one UNBOUND workspace at the same time.
/// The claims are one transaction each under the workspace's reservation, so the second
/// finds the first as holder; that holder is in its turn, so the second is refused.
#[test]
fn two_claims_on_an_unbound_workspace_leave_one_holder() {
    let (temp, store) = store();
    // Many other records, so both scans overlap if nothing serializes them.
    for index in 0..300 {
        store.put(&format!("padding-{index}"), &Binding::new("other", "other")).unwrap();
    }
    let one = workspace(&temp, "demo", "claim-race");
    for trial in 0..32 {
        let start = Arc::new(Barrier::new(2));
        let claimants: Vec<_> = ["a", "b"]
            .into_iter()
            .map(|who| {
                let (store, one, start) = (store.clone(), one.clone(), start.clone());
                let context = format!("claim-{who}-{trial}");
                std::thread::spawn(move || {
                    start.wait();
                    // Both are in their turns: neither is idle for the other.
                    (context.clone(), store.claim(&context, &one, &nobody_idle))
                })
            })
            .collect();
        let results: Vec<_> = claimants.into_iter().map(|thread| thread.join().unwrap()).collect();
        let winners: Vec<&String> = results.iter().filter(|(_, result)| result.is_ok()).map(|(context, _)| context).collect();
        assert_eq!(winners.len(), 1, "trial {trial}: {results:?}");
        assert_eq!(store.holders("demo", "claim-race").unwrap(), [winners[0].clone()], "trial {trial}");
        for (context, result) in &results {
            if result.is_err() {
                assert_eq!(store.load(context).unwrap(), None, "the loser has no record");
            }
            store.release(context, None).unwrap();
        }
    }
}

/// The order inside a claim: holders idle, their records removed durably, the claimant
/// published, with the holders held throughout. An error (or a crash) between removal
/// and publication leaves the workspace bound to nobody, never to two.
#[test]
fn a_claim_removes_old_holders_before_it_publishes_and_holds_them_until_then() {
    let (temp, mut store) = store();
    let one = workspace(&temp, "adapsis", "one");
    for (fail_at, old_holder_kept, new_holder) in [(Step::BeforeRemoval, true, false), (Step::AfterRemoval, false, false), (Step::AfterPublish, false, true)] {
        store.hook = None;
        store.release("new", None).unwrap();
        store.put("old", &Binding::new("adapsis", "one")).unwrap();
        // The old holder's guard, as `turn_lock::idle` would give it: alive while held.
        let held = Arc::new(());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (probe, log, directory) = (Arc::downgrade(&held), seen.clone(), store.root().join("_bindings"));
        store.hook = Some(Arc::new(move |step| {
            let files = std::fs::read_dir(&directory).unwrap().count();
            log.lock().unwrap().push((step, files, probe.strong_count() > 1));
            if step == fail_at { Err(std::io::Error::other("injected")) } else { Ok(()) }
        }));
        let guard = Mutex::new(Some(held));
        let result = store.claim("new", &one, &|_| guard.lock().unwrap().as_ref().map(|held| Box::new(held.clone()) as IdleGuard));
        assert!(matches!(result, Err(Error::Binding(_))), "{fail_at:?}: {result:?}");
        let seen = seen.lock().unwrap().clone();
        // At every step the old holder was still held; the file counts show the order:
        // one record, then none, then the claimant's.
        let expected = [(Step::BeforeRemoval, 1, true), (Step::AfterRemoval, 0, true), (Step::AfterPublish, 1, true)];
        let reached = expected.iter().position(|(step, ..)| *step == fail_at).unwrap() + 1;
        assert_eq!(seen, expected[..reached], "{fail_at:?}");
        assert_eq!(Arc::strong_count(guard.lock().unwrap().as_ref().unwrap()), 1, "the guard is released after the claim");
        store.hook = None;
        let holders = store.holders("adapsis", "one").unwrap();
        let expected: Vec<&str> = [old_holder_kept.then_some("old"), new_holder.then_some("new")].into_iter().flatten().collect();
        assert_eq!(holders, expected, "{fail_at:?}");
        assert!(holders.len() <= 1, "never two holders");
    }
    // Unbound after the middle failure: either conversation claims it again.
    store.release("new", None).unwrap();
    assert_eq!(store.claim("old", &one, &nobody_idle).unwrap(), Claimed { taken: false });
}

/// Fail closed: an entry of `_bindings` that is not an attributable record blocks every
/// claim and every holder query, and the error names that file (by its hash name, never
/// by a conversation key) so the owner can repair or remove it.
#[test]
fn entries_that_cannot_be_attributed_fail_closed_and_name_the_file() {
    let (temp, store) = store();
    let one = workspace(&temp, "adapsis", "one");
    store.claim("telegram:group:-5550001", &one, &nobody_idle).unwrap();
    let directory = store.root().join("_bindings");
    let path = directory.join(&entries(&store)[0]);
    let original = std::fs::read(&path).unwrap();
    let secret = "CANARY_BINDING_TEXT";
    let before = journal(u64::MAX - 1)["last_seq"].as_u64().unwrap();
    let unrelated = workspace(&temp, "gateway", "two");
    let check = |file: &std::path::Path, reason: &str, own: Option<&str>| {
        let mut errors = vec![store.holders("adapsis", "one").unwrap_err(), store.holders("other", "x").unwrap_err()];
        errors.push(store.claim("someone-else", &unrelated, &all_idle).unwrap_err());
        errors.push(store.claim("someone-else", &one, &all_idle).unwrap_err());
        errors.extend(own.map(|context| store.load(context).unwrap_err()));
        errors.extend(own.map(|context| store.save_ledger(context, &one, &ReadLedger::default()).unwrap_err()));
        for error in errors {
            let text = error.to_string();
            assert!(matches!(error, Error::Binding(_)), "{text}");
            let close = if reason.is_empty() { "" } else { ")" };
            assert!(text.starts_with(&format!("the workspace binding record {} cannot be used ({reason}{close}", file.display())), "{text}");
            assert!(text.contains("the owner removes that file"), "{text}");
            assert!(!text.contains(secret) && !text.contains("telegram") && !text.contains("5550001"), "{text}");
        }
        assert_eq!(store.load("someone-else").unwrap(), None, "nothing was claimed");
    };

    // Not JSON.
    std::fs::write(&path, format!("{{ {secret}")).unwrap();
    check(&path, "it is not a binding record", Some("telegram:group:-5550001"));
    // Another version.
    let mut newer: serde_json::Value = serde_json::from_slice(&original).unwrap();
    newer["version"] = 2.into();
    std::fs::write(&path, newer.to_string()).unwrap();
    check(&path, "it has version 2, this build reads version 1", Some("telegram:group:-5550001"));
    // Too large to be a binding.
    std::fs::write(&path, vec![b' '; BINDING_MAX_BYTES as usize + 1]).unwrap();
    check(&path, "invalid data", Some("telegram:group:-5550001"));
    // The reviewer's case: a valid record whose content names another conversation.
    let mut foreign: serde_json::Value = serde_json::from_slice(&original).unwrap();
    foreign["context"] = "ctx-b".into();
    std::fs::write(&path, foreign.to_string()).unwrap();
    check(&path, "its file name does not belong to the conversation it names", Some("telegram:group:-5550001"));
    // One record, many failing calls: one journal entry, with no path, key or content.
    let ours = |journal: serde_json::Value| -> Vec<serde_json::Value> {
        journal["entries"].as_array().unwrap().iter().filter(|entry| entry["source"] == "binding").cloned().collect()
    };
    let first = ours(journal(before));
    assert_eq!(first.len(), 1, "{first:?}");
    assert_eq!((first[0]["category"].as_str(), first[0]["message"].as_str()), (Some("workspace"), Some("workspace binding record unreadable")));
    assert!(!first[0].to_string().contains("_bindings") && !first[0].to_string().contains("telegram"), "{first:?}");

    // Strangers in the directory: a copy under another name, a stray file, a
    // directory, a symlink. Each is named; none is skipped.
    std::fs::write(&path, &original).unwrap();
    assert_eq!(store.holders("adapsis", "one").unwrap(), ["telegram:group:-5550001"]);
    let strangers: [(&str, &str, Box<dyn Fn(&std::path::Path)>); 4] = [
        ("copy.json", "its file name does not belong to the conversation it names", Box::new(|at| std::fs::write(at, &original).unwrap())),
        ("notes.txt", "it is not a binding record", Box::new(|at| std::fs::write(at, "hello").unwrap())),
        ("subdirectory", "", Box::new(|at| std::fs::create_dir(at).unwrap())),
        ("link.json", "", Box::new(|at| std::os::unix::fs::symlink(&path, at).unwrap())),
    ];
    for (name, reason, make) in strangers {
        let stranger = directory.join(name);
        make(&stranger);
        check(&stranger, reason, None);
        let _ = std::fs::remove_file(&stranger).or_else(|_| std::fs::remove_dir(&stranger));
    }
    assert_eq!(ours(journal(before)).len(), 5, "one entry per file");
    // Our own in-flight temporary files are not records.
    std::fs::write(directory.join(".tmp-4242-7"), "partial").unwrap();
    assert_eq!(store.holders("adapsis", "one").unwrap(), ["telegram:group:-5550001"]);
    store.claim("someone-else", &unrelated, &all_idle).unwrap();

    // A symlink in place of a conversation's own record is never followed.
    std::fs::remove_file(&path).unwrap();
    let outside = store.root().join("outside.json");
    std::fs::write(&outside, &original).unwrap();
    std::os::unix::fs::symlink(&outside, &path).unwrap();
    let linked = store.load("telegram:group:-5550001").unwrap_err().to_string();
    assert!(linked.starts_with(&format!("the workspace binding record {} cannot be used (", path.display())), "{linked}");
    assert!(store.claim("telegram:group:-5550001", &one, &all_idle).is_err());
    assert_eq!(std::fs::read(&outside).unwrap(), original, "nothing was written through the link");
}

/// Removal is made durable before anything is published on top of it.
#[test]
fn removing_a_record_syncs_the_directory() {
    let (temp, store) = store();
    let one = workspace(&temp, "adapsis", "one");
    store.claim("ctx-a", &one, &nobody_idle).unwrap();
    let directory = crate::workspace::state_io::Dir::root(&store.root().join("_bindings"), false).unwrap().unwrap();
    let name = std::ffi::OsString::from(&entries(&store)[0]);
    directory.remove_file_synced(&name).unwrap();
    directory.remove_file_synced(&name).unwrap();
    assert!(entries(&store).is_empty());
    // On a directory that is gone the sync itself fails: the error is not swallowed.
    std::fs::remove_dir(store.root().join("_bindings")).unwrap();
    assert!(store.release("ctx-a", None).is_ok(), "no directory, nothing to release");
}
