//! #128: under a manifest the single writer enrols new modules, the manifest
//! follows the code, and every manifest it commits loads with the files on disk.
use super::source::MANIFEST_FILE;
use super::source_tests::{directory, flush, load, python_library, PYTHON};
use crate::ast;

const B: &str = "def twice(x: int) -> int:\n    a: int = A.value(x)\n    b: int = A.value(a)\n    return b\n\n@stored_test(\"B.twice\")\ndef test_case_0() -> None:\n    assert B.twice(1) == 5\n";

/// A Python-authored module entering the live Program the way a mutation does.
fn add_python(program: &mut ast::Program, module: &str, text: &str) {
    let lowered = crate::python_frontend::lower_python_module(text, module, program).unwrap();
    for operation in &lowered.declarations.operations {
        crate::validator::apply_and_validate(program, operation).unwrap();
    }
    for test in &lowered.tests {
        for case in &test.cases {
            crate::eval::eval_test_case(program, &test.function_name, case).unwrap();
        }
        crate::session::store_test_payload(program, &test.function_name, &test.cases, true);
    }
}

fn apply_ax(program: &mut ast::Program, source: &str) {
    for operation in crate::parser::parse(source).unwrap() {
        crate::validator::apply_and_validate(program, &operation).unwrap();
    }
}

/// A module no mutation path produces: one function named like a builtin,
/// which the Python printer refuses.
fn unprintable(name: &str) -> ast::Module {
    serde_json::from_value(serde_json::json!({
        "id": name, "name": name, "types": [], "modules": [],
        "functions": [{"id": "f", "name": "eval", "params": [], "return_type": "Int", "effects": [],
            "body": [{"id": "r", "kind": {"Return": {"value": {"Literal": {"Int": 1}}}}}]}]
    }))
    .unwrap()
}

fn manifest(dir: &std::path::Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(dir.join(MANIFEST_FILE)).unwrap()).unwrap()
}

fn manifest_bytes(dir: &std::path::Path) -> Vec<u8> {
    std::fs::read(dir.join(MANIFEST_FILE)).unwrap()
}

fn names(dir: &std::path::Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

fn journal(source: &str) -> Vec<String> {
    let value: serde_json::Value = serde_json::from_str(&crate::error_journal::query(0)).unwrap();
    value["entries"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|entry| entry["source"] == source)
        .map(|entry| entry["message"].as_str().unwrap().to_string())
        .collect()
}

fn queue(program: &ast::Program, state: &super::LibraryState, name: &str) -> (String, bool) {
    let affected = std::collections::BTreeSet::from([name.to_string()]);
    let mut results = super::persist_affected_modules(program, &affected, Some(state));
    assert_eq!(results.len(), 1, "{results:?}");
    results.remove(0)
}

#[test]
fn managed_enrol_new_module_is_saved_listed_and_loads_after_restart() {
    use std::os::unix::fs::PermissionsExt;
    let dir = python_library();
    let (mut program, state) = load(dir.path());
    add_python(&mut program, "B", B);
    let (notice, ok) = queue(&program, &state, "B");
    // The acknowledgement inside the mutation is neutral: it cannot know the
    // save outcome (the entry point reports that after its write-back).
    assert!(ok && notice == "module `B` changed in memory (new module)", "{notice}");
    assert!(!dir.path().join("B.py").exists(), "a mutation never writes source itself");
    assert!(state.pending_source_names().contains("B"));

    let receipt = flush(&program, &state, dir.path());
    assert!(receipt.failures.is_empty(), "{:?}", receipt.failures);
    assert_eq!(receipt.enrolled, vec!["B".to_string()]);
    assert!(receipt.saved_names.contains(&"B".to_string()));
    let file = dir.path().join("B.py");
    assert_eq!(std::fs::metadata(&file).unwrap().permissions().mode() & 0o777, 0o600);
    let listed = manifest(dir.path());
    assert_eq!(listed["modules"]["B"]["format"], "python");
    assert_eq!(listed["modules"]["B"]["dependencies"], serde_json::json!(["A"]));
    assert_eq!(listed["modules"]["A"]["dependencies"], serde_json::json!([]));
    assert!(state.managed_module_names().contains("B"));
    assert!(state.pending_source_names().is_empty());
    assert_eq!(names(dir.path()), ["A.py", "B.py", MANIFEST_FILE]);

    // The next save changes nothing: the manifest already follows the code.
    let bytes = manifest_bytes(dir.path());
    let again = flush(&program, &state, dir.path());
    assert!(again.failures.is_empty() && again.enrolled.is_empty() && again.notes.is_empty());
    assert_eq!(manifest_bytes(dir.path()), bytes);

    // A new process with an empty session: source alone restores the module,
    // and its stored case earns its pass flag in the mock-only startup re-run.
    let (restarted, _) = load(dir.path());
    let twice = restarted.get_function("B.twice").unwrap();
    assert_eq!(twice.tests.len(), 1);
    assert!(twice.tests[0].passed);
}

#[test]
fn managed_enrol_listed_module_gaining_a_dependency_updates_the_manifest_and_saves() {
    let dir = directory();
    std::fs::write(
        dir.path().join(MANIFEST_FILE),
        r#"{"version":1,"modules":{"A":{"format":"python","dependencies":[]},"B":{"format":"python","dependencies":[]}}}"#,
    )
    .unwrap();
    std::fs::write(dir.path().join("A.py"), PYTHON).unwrap();
    std::fs::write(dir.path().join("B.py"), "def twice(x: int) -> int:\n    return x\n").unwrap();
    let (mut program, state) = load(dir.path());
    let replaced = crate::python_frontend::lower_python_module(
        "def twice(x: int) -> int:\n    return A.value(x)\n",
        "B",
        &{
            let mut catalog = program.clone();
            catalog.modules.retain(|module| module.name != "B");
            catalog.rebuild_function_index();
            catalog
        },
    )
    .unwrap();
    for operation in &replaced.declarations.operations {
        crate::validator::apply_and_validate(&mut program, operation).unwrap();
    }
    let receipt = flush(&program, &state, dir.path());
    assert!(receipt.failures.is_empty(), "{:?}", receipt.failures);
    assert!(receipt.enrolled.is_empty());
    assert!(receipt.notes.iter().any(|note| note.contains("dependencies of `B` are now [A]")), "{:?}", receipt.notes);
    assert_eq!(manifest(dir.path())["modules"]["B"]["dependencies"], serde_json::json!(["A"]));
    assert!(std::fs::read_to_string(dir.path().join("B.py")).unwrap().contains("A.value(x)"));
    let (restarted, _) = load(dir.path());
    assert!(restarted.get_function("B.twice").is_some());
}

fn two_module_library(b_source: &str, b_dependencies: &str) -> tempfile::TempDir {
    let dir = directory();
    std::fs::write(
        dir.path().join(MANIFEST_FILE),
        format!(r#"{{"version":1,"modules":{{"A":{{"format":"python","dependencies":[]}},"B":{{"format":"python","dependencies":{b_dependencies}}}}}}}"#),
    )
    .unwrap();
    std::fs::write(dir.path().join("A.py"), PYTHON).unwrap();
    std::fs::write(dir.path().join("B.py"), b_source).unwrap();
    dir
}

/// The manifest and the source files are separate renames. A dependency is
/// removed from the manifest only after the source that no longer needs it is
/// on disk; the other order would refuse the next start after a crash between.
#[test]
fn managed_dependency_is_removed_only_after_the_source_without_it_is_saved() {
    let uses_a = "def twice(x: int) -> int:\n    return A.value(x)\n";
    let dir = two_module_library(uses_a, r#"["A"]"#);
    let (mut program, state) = load(dir.path());
    apply_ax(&mut program, "+module B\n+fn twice (x:Int)->Int\n  +return x\n+end\n");
    let bytes = manifest_bytes(dir.path());
    // Before the saves nothing is removed: the source on disk still calls A.
    {
        let _transaction = state.source_transaction.lock().unwrap();
        let before = super::source_enrol::grow(&program, &state, dir.path()).unwrap();
        assert!(before.enrolled.is_empty() && before.notes.is_empty() && before.failures.is_empty());
    }
    assert_eq!(manifest_bytes(dir.path()), bytes);
    assert_eq!(std::fs::read_to_string(dir.path().join("B.py")).unwrap(), uses_a);
    load(dir.path());
    // The whole save: source first, then the manifest.
    let receipt = flush(&program, &state, dir.path());
    assert!(receipt.failures.is_empty(), "{:?}", receipt.failures);
    assert!(!std::fs::read_to_string(dir.path().join("B.py")).unwrap().contains("A.value"));
    assert_eq!(manifest(dir.path())["modules"]["B"]["dependencies"], serde_json::json!([]));
    assert!(receipt.notes.iter().any(|note| note.contains("dependencies of `B` are now []")), "{:?}", receipt.notes);
    let (restarted, _) = load(dir.path());
    assert!(restarted.get_function("B.twice").is_some());
    // A removal whose source save failed is not made.
    let dir = two_module_library(uses_a, r#"["A"]"#);
    let (mut program, state) = load(dir.path());
    let chain = " + 1".repeat(130);
    apply_ax(&mut program, &format!("+module B\n+fn twice (x:Int)->Int\n  +return x{chain}\n+end\n"));
    let bytes = manifest_bytes(dir.path());
    let receipt = flush(&program, &state, dir.path());
    assert_eq!(receipt.failures.len(), 1, "{:?}", receipt.failures);
    assert_eq!(manifest_bytes(dir.path()), bytes);
    load(dir.path());
}

/// A new module that uses a function another module gained in the same save
/// window. Listing it before that module's source is on disk would refuse the
/// next start after a crash, so the first pass leaves it out and the second
/// pass of the same save enrols it.
#[test]
fn managed_enrol_waits_until_the_sources_it_uses_are_on_disk() {
    let dir = python_library();
    let (mut program, state) = load(dir.path());
    apply_ax(&mut program, "+module A\n+fn fresh (x:Int)->Int\n  +return x\n+end\n");
    add_python(&mut program, "B", "def use(x: int) -> int:\n    return A.fresh(x)\n");
    let bytes = manifest_bytes(dir.path());
    {
        let _transaction = state.source_transaction.lock().unwrap();
        let first = super::source_enrol::grow(&program, &state, dir.path()).unwrap();
        assert!(first.enrolled.is_empty());
        assert!(first.refused["B"].detail.contains("does not load yet together with the module sources on disk"), "{:?}", first.refused);
    }
    assert_eq!(manifest_bytes(dir.path()), bytes);
    assert_eq!(names(dir.path()), ["A.py", MANIFEST_FILE], "nothing is written for a module that is not listed");
    load(dir.path());
    let receipt = flush(&program, &state, dir.path());
    assert!(receipt.failures.is_empty(), "{:?}", receipt.failures);
    assert_eq!(receipt.enrolled, vec!["B".to_string()]);
    assert!(state.pending_source_names().is_empty());
    let (restarted, _) = load(dir.path());
    assert!(restarted.get_function("B.use").is_some() && restarted.get_function("A.fresh").is_some());
}

/// A recorded dependency that the other module no longer needs must not turn
/// a new, legal dependency in the opposite direction into a refused cycle.
#[test]
fn managed_dependency_direction_can_be_reversed_in_one_save() {
    let dir = two_module_library("def twice(x: int) -> int:\n    return A.value(x)\n", r#"["A"]"#);
    let (mut program, state) = load(dir.path());
    apply_ax(&mut program, "+module B\n+fn twice (x:Int)->Int\n  +return x\n+end\n");
    apply_ax(&mut program, "+module A\n+fn back (x:Int)->Int\n  +return B.twice(x)\n+end\n");
    let receipt = flush(&program, &state, dir.path());
    assert!(receipt.failures.is_empty(), "{:?}", receipt.failures);
    let listed = manifest(dir.path());
    assert_eq!(listed["modules"]["A"]["dependencies"], serde_json::json!(["B"]));
    assert_eq!(listed["modules"]["B"]["dependencies"], serde_json::json!([]));
    assert!(state.pending_source_names().is_empty());
    let (restarted, _) = load(dir.path());
    assert!(restarted.get_function("A.back").is_some());
}

#[test]
fn managed_enrol_cycle_is_refused_and_the_notice_names_both_modules() {
    let dir = python_library();
    let (mut program, state) = load(dir.path());
    add_python(&mut program, "B", B);
    apply_ax(&mut program, "+module A\n+fn back (x:Int)->Int\n  +return B.twice(x)\n+end\n");
    let bytes = manifest_bytes(dir.path());

    let receipt = flush(&program, &state, dir.path());
    // The speaker learns it from the save outcome of the mutation, for either
    // side: the record names both modules and the result line says how to fix it.
    for (name, partner) in [("A", "B"), ("B", "A")] {
        let entry = &receipt.unsaved[name];
        assert_eq!(entry.kind, super::UnsavedKind::Cycle(vec![partner.to_string()]), "{name}");
        let line = entry.result_line(name);
        assert!(line.starts_with(&format!("module `{name}` NOT saved: ")), "{line}");
        assert!(line.contains("call each other") && line.contains("third module"), "{line}");
        assert!(line.contains(&format!("unless you break the cycle between `{name}` and `{partner}`")), "{line}");
        assert!(line.contains("live in memory and will be lost at the next restart"), "{line}");
        assert!(super::fixable(&receipt.unsaved, name));
    }
    assert!(receipt.enrolled.is_empty());
    let cycle: Vec<_> = receipt.failures.iter().filter(|failure| failure.contains("call each other")).collect();
    assert_eq!(cycle.len(), 2, "{:?}", receipt.failures);
    assert!(cycle.iter().all(|failure| failure.contains("`A`") && failure.contains("`B`")));
    assert_eq!(manifest_bytes(dir.path()), bytes);
    assert!(!dir.path().join("B.py").exists());
    assert_eq!(std::fs::read_to_string(dir.path().join("A.py")).unwrap(), PYTHON);
    let pending = state.pending_source_names();
    assert!(pending.contains("A") && pending.contains("B"), "{pending:?}");

    // Breaking the cycle makes the same pending work durable.
    apply_ax(&mut program, "!remove A.back");
    let receipt = flush(&program, &state, dir.path());
    assert!(receipt.failures.is_empty(), "{:?}", receipt.failures);
    assert_eq!(receipt.enrolled, vec!["B".to_string()]);
    assert!(state.pending_source_names().is_empty());
    assert!(receipt.unsaved.is_empty() && state.unsaved().is_empty(), "the record is cleared by the flush that saves");
}

#[test]
fn managed_enrol_unprintable_new_module_is_reported_and_not_listed() {
    let dir = python_library();
    let (mut program, state) = load(dir.path());
    program.modules.push(unprintable("Odd"));
    program.rebuild_function_index();
    let bytes = manifest_bytes(dir.path());
    let receipt = flush(&program, &state, dir.path());
    assert!(receipt.enrolled.is_empty());
    assert_eq!(receipt.failures.len(), 1, "{:?}", receipt.failures);
    assert!(receipt.failures[0].contains("`Odd`") && receipt.failures[0].contains("has NO source"));
    assert!(receipt.failures[0].contains("dropped at the next restart"));
    assert_eq!(manifest_bytes(dir.path()), bytes);
    assert_eq!(names(dir.path()), ["A.py", MANIFEST_FILE]);
    assert!(state.pending_source_names().contains("Odd"));
    assert!(!state.managed_module_names().contains("Odd"));
}

#[test]
fn managed_enrol_moves_unlisted_leftover_files_aside_and_never_deletes_them() {
    let dir = python_library();
    std::fs::write(dir.path().join("B.py"), "def stray() -> int:\n    return 7\n").unwrap();
    std::fs::write(dir.path().join("B.ax"), "+module B\n").unwrap();
    let (mut program, state) = load(dir.path());
    assert!(program.get_function("B.stray").is_none(), "unlisted files are never loaded");
    add_python(&mut program, "B", B);
    let receipt = flush(&program, &state, dir.path());
    assert!(receipt.failures.is_empty(), "{:?}", receipt.failures);
    assert_eq!(receipt.enrolled, vec!["B".to_string()]);
    assert_eq!(receipt.notes.iter().filter(|note| note.contains("moved aside")).count(), 2, "{:?}", receipt.notes);
    assert!(std::fs::read_to_string(dir.path().join("B.py")).unwrap().contains("def twice"));
    let orphans: Vec<String> = names(dir.path()).into_iter().filter(|name| name.contains(".orphan-")).collect();
    assert_eq!(orphans.len(), 2, "{orphans:?}");
    let stray = orphans.iter().find(|name| name.starts_with("B.py.orphan-")).unwrap();
    assert!(std::fs::read_to_string(dir.path().join(stray)).unwrap().contains("def stray"));
    assert!(orphans.iter().any(|name| name.starts_with("B.ax.orphan-")));
    // Orphans are not sources: the next start loads A and B and nothing else.
    let (restarted, restarted_state) = load(dir.path());
    assert!(restarted.get_function("B.twice").is_some() && restarted.get_function("B.stray").is_none());
    assert_eq!(restarted_state.managed_module_names().len(), 2);
}

#[test]
fn managed_enrol_writes_nothing_when_the_manifest_changed_on_disk() {
    let dir = python_library();
    let (mut program, state) = load(dir.path());
    add_python(&mut program, "B", B);
    let external = r#"{"version":1,"modules":{"A":{"format":"python","dependencies":[]}}} "#;
    std::fs::write(dir.path().join(MANIFEST_FILE), external).unwrap();
    let receipt = flush(&program, &state, dir.path());
    assert!(receipt.enrolled.is_empty() && receipt.saved_names.is_empty());
    assert_eq!(receipt.failures.len(), 1, "{:?}", receipt.failures);
    assert!(receipt.failures[0].contains("no source files saved"));
    assert_eq!(std::fs::read_to_string(dir.path().join(MANIFEST_FILE)).unwrap(), external);
    assert_eq!(names(dir.path()), ["A.py", MANIFEST_FILE]);
    assert!(state.pending_source_names().contains("A"));
}

/// Crash after step (1), the source file, before step (2), the manifest.
#[test]
fn managed_enrol_crash_after_source_before_manifest_never_refuses_the_next_start() {
    let reference = python_library();
    let (mut program, state) = load(reference.path());
    add_python(&mut program, "B", B);
    assert!(flush(&program, &state, reference.path()).failures.is_empty());
    let printed = std::fs::read_to_string(reference.path().join("B.py")).unwrap();

    // (a) The session was not saved either: the file is an ignored leftover.
    let dir = python_library();
    std::fs::write(dir.path().join("B.py"), &printed).unwrap();
    let (fresh, fresh_state) = load(dir.path());
    assert!(fresh.get_function("B.twice").is_none());
    assert!(fresh.get_function("A.value").is_some());
    assert!(!fresh_state.managed_module_names().contains("B"));

    // (b) The session has the module: startup adopts it and reuses the file.
    let mut cached = program.clone();
    let mut adopting = super::LibraryState::new();
    super::load_module_files(&mut cached, dir.path(), &mut adopting);
    adopting.require_authoritative_sources().unwrap();
    assert!(cached.get_function("B.twice").is_some());
    assert!(adopting.managed_module_names().contains("B"));
    assert_eq!(manifest(dir.path())["modules"]["B"]["dependencies"], serde_json::json!(["A"]));
    assert_eq!(std::fs::read_to_string(dir.path().join("B.py")).unwrap(), printed);
    assert!(names(dir.path()).iter().all(|name| !name.contains(".orphan-")), "identical content is reused");
}

/// Crash after step (2), the manifest, before step (3), the session.
#[test]
fn managed_enrol_crash_after_manifest_before_session_restores_from_source() {
    let dir = python_library();
    let (mut program, state) = load(dir.path());
    let before_mutation = program.clone();
    add_python(&mut program, "B", B);
    assert!(flush(&program, &state, dir.path()).failures.is_empty());
    // The session file still holds the Program from before the mutation.
    let mut stale = before_mutation;
    let mut restarted = super::LibraryState::new();
    super::load_module_files(&mut stale, dir.path(), &mut restarted);
    restarted.require_authoritative_sources().unwrap();
    assert!(stale.get_function("B.twice").is_some());
    assert!(stale.get_function("B.twice").unwrap().tests[0].passed);
}

#[test]
fn managed_startup_adopts_a_printable_cached_module_with_one_journal_entry() {
    let dir = python_library();
    let (mut cached, _) = load(dir.path());
    add_python(&mut cached, "AdoptedOnce", &B.replace("B.twice", "AdoptedOnce.twice"));
    let mut state = super::LibraryState::new();
    super::load_module_files(&mut cached, dir.path(), &mut state);
    state.require_authoritative_sources().unwrap();
    assert!(cached.get_function("AdoptedOnce.twice").is_some());
    assert!(dir.path().join("AdoptedOnce.py").is_file());
    assert_eq!(manifest(dir.path())["modules"]["AdoptedOnce"]["dependencies"], serde_json::json!(["A"]));
    assert!(state.dropped_module_names().is_empty());
    let entries = journal("AdoptedOnce");
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert!(entries[0].contains("adopted into the source manifest at startup"));
    // The process after that finds an ordinary listed module.
    let (restarted, _) = load(dir.path());
    assert!(restarted.get_function("AdoptedOnce.twice").is_some());
    assert_eq!(journal("AdoptedOnce").len(), 1);
}

#[test]
fn managed_startup_drops_an_unprintable_cached_module_into_a_recovery_file() {
    let dir = python_library();
    let (mut cached, _) = load(dir.path());
    cached.modules.push(unprintable("DroppedOnce"));
    cached.rebuild_function_index();
    let bytes = manifest_bytes(dir.path());
    let mut state = super::LibraryState::new();
    super::load_module_files(&mut cached, dir.path(), &mut state);
    state.require_authoritative_sources().unwrap();
    assert!(cached.modules.iter().all(|module| module.name != "DroppedOnce"));
    assert!(cached.get_function("DroppedOnce.eval").is_none());
    assert!(cached.get_function("A.value").is_some());
    assert_eq!(manifest_bytes(dir.path()), bytes);
    assert!(state.dropped_module_names().contains("DroppedOnce"));
    let recovery = names(&dir.path().join("recovery"));
    assert_eq!(recovery.len(), 1, "{recovery:?}");
    assert!(recovery[0].starts_with("DroppedOnce-") && recovery[0].ends_with(".json"));
    let kept: ast::Module =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("recovery").join(&recovery[0])).unwrap()).unwrap();
    assert_eq!(kept.functions[0].name, "eval");
    let entries = journal("DroppedOnce");
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert!(entries[0].contains("dropped at startup") && entries[0].contains("recovery/DroppedOnce-"));
    // The recovery folder is never scanned for sources.
    let (restarted, restarted_state) = load(dir.path());
    assert!(restarted.modules.iter().all(|module| module.name != "DroppedOnce"));
    assert_eq!(restarted_state.managed_module_names().len(), 1);
}

/// Review of a488baa: adoption enrolled the printable module (source and
/// manifest written) and THEN failed on the recovery file of the unprintable
/// one, so a refused start had changed source and manifest. The recovery file
/// is now written before any enrolment commit.
#[test]
fn managed_startup_refused_for_an_unwritable_recovery_file_changes_no_source_and_no_manifest() {
    let dir = directory();
    let before = r#"{"version":1,"modules":{}}"#;
    std::fs::write(dir.path().join(MANIFEST_FILE), before).unwrap();
    std::fs::write(dir.path().join("recovery"), "a regular file prevents recovery writes").unwrap();
    let mut cached = ast::Program::default();
    apply_ax(&mut cached, "+module Good\n+fn one ()->Int\n  +return 1\n+end\n");
    cached.modules.push(unprintable("Bad"));
    cached.rebuild_function_index();
    let recovery_cache = serde_json::to_string(&cached).unwrap();
    let mut state = super::LibraryState::new();
    super::load_module_files(&mut cached, dir.path(), &mut state);
    let error = format!("{:#}", state.require_authoritative_sources().unwrap_err());
    assert!(error.contains("startup refused") && error.contains("`Bad`") && error.contains("recovery file could not be written"), "{error}");
    assert_eq!(std::fs::read_to_string(dir.path().join(MANIFEST_FILE)).unwrap(), before);
    assert_eq!(names(dir.path()), [MANIFEST_FILE, "recovery"], "no source file was written");
    assert_eq!(serde_json::to_string(&cached).unwrap(), recovery_cache, "the session cache is the recovery material");
    // The same cache once the recovery folder is usable: Good is adopted, Bad is
    // dropped into a recovery file, and the start goes on.
    std::fs::remove_file(dir.path().join("recovery")).unwrap();
    let mut state = super::LibraryState::new();
    super::load_module_files(&mut cached, dir.path(), &mut state);
    state.require_authoritative_sources().unwrap();
    assert!(cached.get_function("Good.one").is_some() && cached.modules.iter().all(|module| module.name != "Bad"));
    assert!(dir.path().join("Good.py").is_file());
    assert_eq!(names(&dir.path().join("recovery")).len(), 1);
    assert!(names(&dir.path().join("recovery"))[0].starts_with("Bad-"));
}

/// Adoption that needs no recovery file does not depend on the recovery folder.
#[test]
fn managed_startup_adopts_without_touching_the_recovery_folder_when_nothing_is_dropped() {
    let dir = directory();
    std::fs::write(dir.path().join(MANIFEST_FILE), r#"{"version":1,"modules":{}}"#).unwrap();
    std::fs::write(dir.path().join("recovery"), "a regular file prevents recovery writes").unwrap();
    let mut cached = ast::Program::default();
    apply_ax(&mut cached, "+module Good\n+fn one ()->Int\n  +return 1\n+end\n");
    let mut state = super::LibraryState::new();
    super::load_module_files(&mut cached, dir.path(), &mut state);
    state.require_authoritative_sources().unwrap();
    assert!(dir.path().join("Good.py").is_file());
    assert_eq!(manifest(dir.path())["modules"]["Good"]["format"], "python");
    assert_eq!(std::fs::read_to_string(dir.path().join("recovery")).unwrap(), "a regular file prevents recovery writes");
}

#[test]
fn managed_scratch_and_reserved_modules_are_never_enrolled() {
    let dir = python_library();
    let (mut program, state) = load(dir.path());
    let declaration = crate::parser::parse("+module RunToolScratch\n").unwrap();
    assert!(crate::validator::apply_and_validate(&mut program, &declaration[0]).unwrap_err().to_string().contains("reserved"));
    let mut internal = unprintable("__internal");
    internal.functions.clear();
    program.modules.push(internal);
    // Represent pre-existing/cached scratch state directly: normal declarations
    // are now reserved by the run tool. This fixture must still prove that the
    // source writer excludes such state, without weakening declaration admission.
    let mut fixture = ast::Program::default();
    apply_ax(&mut fixture, "+module FixtureEnrolScratch\n+fn probe ()->Int\n  +return 1\n+end\n");
    let mut scratch = fixture.modules.remove(0);
    scratch.name = "RunToolScratch".into();
    program.modules.push(scratch);
    program.rebuild_function_index();
    let bytes = manifest_bytes(dir.path());
    let affected = std::collections::BTreeSet::from(["RunToolScratch".to_string(), "__internal".to_string()]);
    assert!(super::persist_affected_modules(&program, &affected, Some(&state)).is_empty());
    let receipt = flush(&program, &state, dir.path());
    assert!(receipt.failures.is_empty(), "{:?}", receipt.failures);
    assert!(receipt.enrolled.is_empty());
    assert_eq!(manifest_bytes(dir.path()), bytes);
    assert_eq!(names(dir.path()), ["A.py", MANIFEST_FILE]);
    assert!(!state.pending_source_names().contains("RunToolScratch"));
    // At startup they leave the cache without a recovery file or a refusal.
    let mut restarted = super::LibraryState::new();
    super::load_module_files(&mut program, dir.path(), &mut restarted);
    restarted.require_authoritative_sources().unwrap();
    assert_eq!(program.modules.iter().map(|module| module.name.as_str()).collect::<Vec<_>>(), ["A"]);
    assert!(!dir.path().join("recovery").exists());
    assert!(restarted.dropped_module_names().is_empty());
}

#[test]
fn managed_declared_dependencies_equal_to_the_code_are_never_rewritten() {
    // Hand-written manifest bytes (compact, own key order) with exactly the
    // dependencies the code has: the first save after a switch must keep them.
    let dir = directory();
    let text = r#"{"version":1,"modules":{"Bot":{"format":"python","dependencies":["Fmt"]},"Fmt":{"format":"python","dependencies":[]},"A":{"format":"python","dependencies":[]}}}"#;
    std::fs::write(dir.path().join(MANIFEST_FILE), text).unwrap();
    std::fs::write(dir.path().join("A.py"), PYTHON).unwrap();
    std::fs::write(dir.path().join("Fmt.py"), "def bold(text: str) -> str:\n    return concat(\"*\", text)\n").unwrap();
    std::fs::write(dir.path().join("Bot.py"), "def greet(name: str) -> str:\n    return Fmt.bold(name)\n").unwrap();
    let (program, state) = load(dir.path());
    for module in ["A", "Bot", "Fmt"] {
        let derived = crate::python_frontend::module_dependencies(&program, module).unwrap();
        let declared: std::collections::BTreeSet<String> = manifest(dir.path())["modules"][module]["dependencies"]
            .as_array()
            .unwrap()
            .iter()
            .map(|name| name.as_str().unwrap().to_string())
            .collect();
        assert_eq!(derived, declared, "{module}");
    }
    let receipt = flush(&program, &state, dir.path());
    assert!(receipt.failures.is_empty(), "{:?}", receipt.failures);
    assert!(receipt.enrolled.is_empty() && receipt.notes.is_empty(), "{:?}", receipt.notes);
    assert_eq!(std::fs::read_to_string(dir.path().join(MANIFEST_FILE)).unwrap(), text);
}

#[test]
fn managed_port_fixture_dependencies_are_telegrambot_on_tgformat_only() {
    // The sanitized port of the two live Telegram modules: what the writer
    // derives must be what the migration manifest declares for them.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/telegram-python-port/python");
    let mut program = ast::Program::default();
    for module in ["TgFormat", "TelegramBot"] {
        let text = std::fs::read_to_string(root.join(format!("{module}.py"))).unwrap();
        let lowered = crate::python_frontend::lower_python_module(&text, module, &program).unwrap();
        for operation in &lowered.declarations.operations {
            crate::validator::apply_and_validate(&mut program, operation).unwrap();
        }
    }
    let of = |module: &str| crate::python_frontend::module_dependencies(&program, module).unwrap().into_iter().collect::<Vec<_>>();
    assert_eq!(of("TelegramBot"), ["TgFormat"]);
    assert!(of("TgFormat").is_empty());
}

#[test]
fn managed_save_refused_for_nesting_says_the_printed_form_is_too_deep() {
    let dir = python_library();
    let (mut program, state) = load(dir.path());
    // 130 chained operators (possible through the unguarded `.ax` path): 131
    // levels as written, 261 once the printer has put every binary operation
    // in parentheses. Stored source may scan at 250.
    let chain = " + 1".repeat(130);
    apply_ax(&mut program, &format!("+module A\n+fn long (x:Int)->Int\n  +return x{chain}\n+end\n"));
    let receipt = flush(&program, &state, dir.path());
    assert_eq!(receipt.failures.len(), 1, "{:?}", receipt.failures);
    let failure = &receipt.failures[0];
    assert!(failure.contains("module `A`") && failure.contains("printed form"), "{failure}");
    assert!(failure.contains("exceeds the nesting limit of 250 levels for stored source"), "{failure}");
    assert!(failure.contains("about 124 chained operators"), "{failure}");
    assert!(failure.contains("Split the expression"), "{failure}");
    assert_eq!(std::fs::read_to_string(dir.path().join("A.py")).unwrap(), PYTHON);
    assert!(state.pending_source_names().contains("A"));
    // A chain at the submission limit saves, and loads again.
    let chain = " + 1".repeat(99);
    apply_ax(&mut program, &format!("+module A\n+fn long (x:Int)->Int\n  +return x{chain}\n+end\n"));
    assert!(flush(&program, &state, dir.path()).failures.is_empty());
    assert!(load(dir.path()).0.get_function("A.long").is_some());
}

#[test]
fn managed_function_removal_is_an_ordinary_edit_that_stays_gone_after_restart() {
    let dir = python_library();
    let source = format!("def other(x: int) -> int:\n    return x\n\n{PYTHON}");
    std::fs::write(dir.path().join("A.py"), &source).unwrap();
    let (program, state) = load(dir.path());
    let mut session = crate::session::Session::new();
    session.program = program;
    session.meta.library_state = Some(state.clone());
    let results = session.apply("!remove A.other").unwrap();
    assert!(results.iter().all(|(_, ok)| *ok), "{results:?}");
    assert!(session.program.get_function("A.other").is_none());
    let receipt = flush(&session.program, &state, dir.path());
    assert!(receipt.failures.is_empty(), "{:?}", receipt.failures);
    let saved = std::fs::read_to_string(dir.path().join("A.py")).unwrap();
    assert!(!saved.contains("def other") && saved.contains("def value"), "{saved}");
    let (restarted, _) = load(dir.path());
    assert!(restarted.get_function("A.other").is_none());
    assert_eq!(restarted.get_function("A.value").unwrap().tests.len(), 2);
}

fn managed_session(dir: &std::path::Path) -> crate::session::Session {
    let (program, state) = load(dir);
    let mut session = crate::session::Session::new();
    session.program = program;
    session.meta.library_state = Some(state);
    session
}

fn unchanged(session: &crate::session::Session, program: &str, revision: usize, sources: usize) {
    assert_eq!(serde_json::to_string(&session.program).unwrap(), program);
    assert_eq!(session.meta.revision, revision);
    assert_eq!(session.meta.sources.len(), sources);
}

#[test]
fn managed_whole_module_removal_is_refused_before_any_change_naming_126() {
    let dir = python_library();
    let mut session = managed_session(dir.path());
    let (program, revision, sources) =
        (serde_json::to_string(&session.program).unwrap(), session.meta.revision, session.meta.sources.len());
    let error = session.apply("!remove A").unwrap_err().to_string();
    assert!(error.contains("#126") && error.contains("`A`") && error.contains("nothing was changed"), "{error}");
    unchanged(&session, &program, revision, sources);
    assert_eq!(std::fs::read_to_string(dir.path().join("A.py")).unwrap(), PYTHON);
}

#[test]
fn managed_adapsis_source_module_creation_is_refused_before_any_change() {
    let dir = python_library();
    let mut session = managed_session(dir.path());
    let (program, revision, sources) =
        (serde_json::to_string(&session.program).unwrap(), session.meta.revision, session.meta.sources.len());
    for source in [
        "+module Fresh\n+fn one ()->Int\n  +return 1\n+end\n",
        // A valid edit of a managed module does not carry a new module with it.
        "+module A\n+fn extra ()->Int\n  +return 1\n+end\n+module Fresh\n+fn one ()->Int\n  +return 1\n+end\n",
        "!move A.value Fresh",
    ] {
        let error = session.apply(source).unwrap_err().to_string();
        assert!(error.contains("`Fresh` was not created"), "{error}");
        assert!(error.contains("this library stores Python source; create the module with a Python mutation"), "{error}");
        unchanged(&session, &program, revision, sources);
    }
    assert!(session.meta.library_state.as_ref().unwrap().pending_source_names().is_empty());
    // Editing an existing managed module with Adapsis source is unchanged.
    let results = session.apply("+module A\n+fn extra ()->Int\n  +return 1\n+end\n").unwrap();
    assert!(results.iter().all(|(_, ok)| *ok), "{results:?}");
    assert!(session.program.get_function("A.extra").is_some());
}

#[test]
fn managed_python_module_creation_is_accepted_and_enrolled_by_the_next_save() {
    let dir = python_library();
    let mut session = managed_session(dir.path());
    let state = session.meta.library_state.clone().unwrap();
    let lowered = crate::python_frontend::lower_python_module(B, "B", &session.program).unwrap();
    let results = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(session.apply_operations_async(
            &lowered.declarations.operations,
            &crate::session::replay::Submission::python("B", B),
            None,
        ))
        .unwrap();
    assert!(results.iter().all(|(_, ok)| *ok), "{results:?}");
    assert!(results.iter().any(|(message, ok)| *ok && message == "module `B` changed in memory (new module)"), "{results:?}");
    assert!(session.program.get_function("B.twice").is_some());
    assert!(state.pending_source_names().contains("B"));
    let receipt = flush(&session.program, &state, dir.path());
    assert!(receipt.failures.is_empty(), "{:?}", receipt.failures);
    assert_eq!(receipt.enrolled, vec!["B".to_string()]);
}

#[test]
fn managed_agent_branch_cannot_create_a_module_from_adapsis_source() {
    let dir = python_library();
    let mut session = managed_session(dir.path());
    let program = serde_json::to_string(&session.program).unwrap();
    let fresh = "+module Fresh\n+fn one ()->Int\n  +return 1\n+end\n";
    // In the branch: refused as a result the agent sees, branch unchanged.
    let mut branch = crate::session::AgentBranch::fork("worker", crate::session::AgentScope::Full, "task", &session);
    let results = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(branch.apply_async(fresh, None))
        .unwrap();
    assert_eq!(results.len(), 1, "{results:?}");
    assert!(!results[0].1 && results[0].0.contains("create the module with a Python mutation"), "{results:?}");
    assert_eq!(serde_json::to_string(&branch.program).unwrap(), program);
    assert!(branch.mutations.is_empty(), "a refused block is not replayed at merge");
    // At the merge (replay through apply_to_tiers) the same check holds.
    let mut meta = session.meta.clone();
    let mut runtime = session.runtime.clone();
    let mut sandbox = None;
    let error = crate::session::apply_to_tiers(&mut session.program, &mut runtime, &mut meta, &mut sandbox, fresh)
        .unwrap_err()
        .to_string();
    assert!(error.contains("create the module with a Python mutation"), "{error}");
    assert_eq!(serde_json::to_string(&session.program).unwrap(), program);
}

/// The in-language mutation builtins write the live Program directly; they get
/// the same refusals, from the session's library state on this thread.
#[test]
fn managed_mutation_builtins_refuse_creation_and_whole_module_removal() {
    use crate::eval::Value;
    let dir = python_library();
    let (program, state) = load(dir.path());
    let before = serde_json::to_string(&program).unwrap();
    let live = std::sync::Arc::new(std::sync::RwLock::new(program));
    crate::shared_state::set_shared_program_mut(Some(live.clone()));
    let mut meta = crate::session::SessionMeta::new();
    meta.library_state = Some(state);
    crate::shared_state::set_shared_meta(Some(std::sync::Arc::new(std::sync::Mutex::new(meta))));
    let handle = crate::coroutine::CoroutineHandle::new_mock(vec![]);
    let text = |name: &str| Value::string(name);
    let refused = |operation: &str, arguments: &[Value], expected: &str| {
        let error = handle.execute_await(operation, arguments).unwrap_err().to_string();
        assert!(error.contains(expected), "{operation}: {error}");
        assert_eq!(serde_json::to_string(&*live.read().unwrap()).unwrap(), before, "{operation}");
    };
    let create = "create the module with a Python mutation";
    refused("module_create", &[text("Fresh")], create);
    refused("mutate", &[text("+module Fresh\n+fn one ()->Int\n  +return 1\n+end\n")], create);
    refused("move_symbols", &[text("A.value"), text("Fresh")], create);
    refused("module_remove", &[text("A")], "#126");
    refused("mutate", &[text("!remove A")], "#126");
    crate::shared_state::set_shared_meta(None);
    crate::shared_state::set_shared_program_mut(None);
}

// ── One save, several source files (the renames are separate) ──────────────

fn restart_loads(dir: &std::path::Path) -> bool {
    let mut fresh = ast::Program::default();
    let mut state = super::LibraryState::new();
    super::load_module_files(&mut fresh, dir, &mut state);
    state.require_authoritative_sources().is_ok()
}

/// B starts to use a function A gains in the same save, and A cannot be saved.
/// Writing B alone would leave a set the next start refuses, so B waits.
#[test]
fn managed_dependent_is_not_saved_while_the_module_it_needs_cannot_be_saved() {
    let uses_a = "def twice(x: int) -> int:\n    return A.value(x)\n";
    let dir = two_module_library(uses_a, r#"["A"]"#);
    let (mut program, state) = load(dir.path());
    let chain = " + 1".repeat(130);
    apply_ax(&mut program, &format!("+module A\n+fn fresh (x:Int)->Int\n  +return x\n+end\n+fn long (x:Int)->Int\n  +return x{chain}\n+end\n"));
    apply_ax(&mut program, "+module B\n+fn twice (x:Int)->Int\n  +return A.fresh(x)\n+end\n");
    let receipt = flush(&program, &state, dir.path());
    assert!(receipt.saved_names.is_empty(), "{:?}", receipt.saved_names);
    assert_eq!(receipt.failures.len(), 2, "{:?}", receipt.failures);
    let blocked = receipt.failures.iter().find(|failure| failure.contains("module `B`")).unwrap();
    assert!(blocked.contains("would not load at the next start") && blocked.contains("it waits for `A`, whose source is not saved"), "{blocked}");
    assert!(receipt.failures.iter().any(|failure| failure.contains("module `A`") && failure.contains("printed form")));
    assert_eq!(std::fs::read_to_string(dir.path().join("B.py")).unwrap(), uses_a);
    assert_eq!(std::fs::read_to_string(dir.path().join("A.py")).unwrap(), PYTHON);
    let pending = state.pending_source_names();
    assert!(pending.contains("A") && pending.contains("B"));
    assert!(restart_loads(dir.path()), "the set on disk still loads");
    // Once A can be saved, the same pending work becomes durable.
    apply_ax(&mut program, "!remove A.long");
    let receipt = flush(&program, &state, dir.path());
    assert!(receipt.failures.is_empty(), "{:?}", receipt.failures);
    assert!(state.pending_source_names().is_empty());
    let (restarted, _) = load(dir.path());
    assert!(restarted.get_function("B.twice").is_some() && restarted.get_function("A.fresh").is_some());
}

/// Every state the disk passes through during one save must load. The check
/// replays the save's renames in the order the writer chose (dependencies
/// first, then whatever became possible), from the texts before and after.
fn every_intermediate_state_loads(before: &std::path::Path, after: &std::path::Path, order: &[&str]) {
    let stage = directory();
    for entry in std::fs::read_dir(before).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), stage.path().join(entry.file_name())).unwrap();
    }
    assert!(restart_loads(stage.path()));
    for name in order {
        std::fs::copy(after.join(format!("{name}.py")), stage.path().join(format!("{name}.py"))).unwrap();
        assert!(restart_loads(stage.path()), "a crash right after writing `{name}` leaves a set that does not load");
    }
}

fn snapshot(dir: &std::path::Path) -> tempfile::TempDir {
    let copy = directory();
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), copy.path().join(entry.file_name())).unwrap();
    }
    copy
}

fn modified(dir: &std::path::Path, name: &str) -> std::time::SystemTime {
    std::fs::metadata(dir.join(format!("{name}.py"))).unwrap().modified().unwrap()
}

#[test]
fn managed_save_writes_a_module_that_gains_a_function_before_its_new_caller() {
    let dir = two_module_library("def twice(x: int) -> int:\n    return A.value(x)\n", r#"["A"]"#);
    let (mut program, state) = load(dir.path());
    let before = snapshot(dir.path());
    apply_ax(&mut program, "+module A\n+fn fresh (x:Int)->Int\n  +return x\n+end\n");
    apply_ax(&mut program, "+module B\n+fn twice (x:Int)->Int\n  +return A.fresh(x)\n+end\n");
    let receipt = flush(&program, &state, dir.path());
    assert!(receipt.failures.is_empty(), "{:?}", receipt.failures);
    assert_eq!(receipt.saved_names, ["A", "B"]);
    assert!(modified(dir.path(), "A") <= modified(dir.path(), "B"));
    every_intermediate_state_loads(before.path(), dir.path(), &["A", "B"]);
}

#[test]
fn managed_save_writes_a_caller_that_stops_using_a_function_before_the_module_drops_it() {
    let dir = two_module_library("def twice(x: int) -> int:\n    return A.other(x)\n", r#"["A"]"#);
    std::fs::write(dir.path().join("A.py"), format!("def other(x: int) -> int:\n    return x\n\n{PYTHON}")).unwrap();
    let (mut program, state) = load(dir.path());
    let before = snapshot(dir.path());
    apply_ax(&mut program, "+module B\n+fn twice (x:Int)->Int\n  +return x\n+end\n");
    apply_ax(&mut program, "!remove A.other");
    let receipt = flush(&program, &state, dir.path());
    assert!(receipt.failures.is_empty(), "{:?}", receipt.failures);
    // Dependencies are tried first; A cannot go first here, so B does.
    assert_eq!(receipt.saved_names, ["B", "A"]);
    every_intermediate_state_loads(before.path(), dir.path(), &["B", "A"]);
    assert!(!std::fs::read_to_string(dir.path().join("A.py")).unwrap().contains("def other"));
    assert!(receipt.notes.iter().all(|note| !note.contains("only load together")), "{:?}", receipt.notes);
}

/// The remaining gap: functions move between two modules in both directions.
/// The complete new set loads, but neither file can be written first. Both are
/// written, the log says so, and the result loads.
#[test]
fn managed_save_of_a_move_in_both_directions_is_written_and_logged_as_the_remaining_gap() {
    let dir = directory();
    std::fs::write(
        dir.path().join(MANIFEST_FILE),
        r#"{"version":1,"modules":{"A":{"format":"python","dependencies":[]},"B":{"format":"python","dependencies":[]},"C":{"format":"python","dependencies":["A","B"]}}}"#,
    )
    .unwrap();
    std::fs::write(dir.path().join("A.py"), "def left(x: int) -> int:\n    return x\n\ndef keep_a(x: int) -> int:\n    return x\n").unwrap();
    std::fs::write(dir.path().join("B.py"), "def right(x: int) -> int:\n    return x\n\ndef keep_b(x: int) -> int:\n    return x\n").unwrap();
    std::fs::write(dir.path().join("C.py"), "def both(x: int) -> int:\n    return A.left(B.right(x))\n").unwrap();
    let (mut program, state) = load(dir.path());
    // `left` moves A -> B, `right` moves B -> A; C follows.
    apply_ax(&mut program, "!remove A.left");
    apply_ax(&mut program, "+module B\n+fn left (x:Int)->Int\n  +return x\n+end\n");
    apply_ax(&mut program, "!remove B.right");
    apply_ax(&mut program, "+module A\n+fn right (x:Int)->Int\n  +return x\n+end\n");
    apply_ax(&mut program, "+module C\n+fn both (x:Int)->Int\n  +return B.left(A.right(x))\n+end\n");
    let receipt = flush(&program, &state, dir.path());
    assert!(receipt.failures.is_empty(), "{:?}", receipt.failures);
    let gap: Vec<_> = receipt.notes.iter().filter(|note| note.contains("only load together")).collect();
    assert_eq!(gap.len(), 1, "{:?}", receipt.notes);
    assert!(gap[0].contains("`A`") && gap[0].contains("`B`") && gap[0].contains("`C`"), "{}", gap[0]);
    assert!(state.pending_source_names().is_empty());
    let (restarted, _) = load(dir.path());
    assert!(restarted.get_function("B.left").is_some() && restarted.get_function("A.right").is_some());
    assert!(restarted.get_function("A.left").is_none());
}

#[test]
fn managed_save_that_changes_no_file_rewrites_nothing() {
    let dir = two_module_library("def twice(x: int) -> int:\n    return A.value(x)\n", r#"["A"]"#);
    let (program, state) = load(dir.path());
    // The first save brings hand-written files into the printer's form.
    assert!(flush(&program, &state, dir.path()).failures.is_empty());
    let stamps = (modified(dir.path(), "A"), modified(dir.path(), "B"));
    let bytes = (std::fs::read(dir.path().join("A.py")).unwrap(), std::fs::read(dir.path().join("B.py")).unwrap());
    let receipt = flush(&program, &state, dir.path());
    assert!(receipt.failures.is_empty() && receipt.notes.is_empty());
    assert_eq!(receipt.saved_names, ["A", "B"]);
    assert_eq!((modified(dir.path(), "A"), modified(dir.path(), "B")), stamps);
    assert_eq!((std::fs::read(dir.path().join("A.py")).unwrap(), std::fs::read(dir.path().join("B.py")).unwrap()), bytes);
    assert_eq!(names(dir.path()), ["A.py", "B.py", MANIFEST_FILE]);
}
