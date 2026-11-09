use super::source::{ManagedSources, MANIFEST_FILE};

fn assert_retained_policy_guard(empty: bool, dangling: bool) {
    let dir = python_library();
    if empty {
        std::fs::write(
            dir.path().join(MANIFEST_FILE),
            r#"{"version":1,"modules":{}}"#,
        )
        .unwrap();
    }
    let (mut program, state) = load(dir.path());
    let legacy = "+module Extra\n+fn old ()->Int\n  +return 7\n+end\n";
    for op in crate::parser::parse(legacy).unwrap() {
        crate::validator::apply_and_validate(&mut program, &op).unwrap();
    }
    std::fs::write(dir.path().join("Extra.ax"), legacy).unwrap();
    std::fs::remove_file(dir.path().join(MANIFEST_FILE)).unwrap();
    if dangling {
        std::os::unix::fs::symlink(
            dir.path().join("missing-policy"),
            dir.path().join(MANIFEST_FILE),
        )
        .unwrap();
    }
    assert_refuses_legacy_entry_points(&mut program, &state, dir.path(), legacy);
}

fn assert_refuses_legacy_entry_points(
    program: &mut crate::ast::Program,
    state: &super::LibraryState,
    dir: &std::path::Path,
    legacy: &str,
) {
    let before = serde_json::to_string(program).unwrap();
    let hashes = state.file_hashes.lock().unwrap().clone();
    let pending = state.pending_source_names();
    for name in ["", "Extra"] {
        assert!(
            super::reload_module_from_dir(program, name, Some(state), dir).is_err(),
            "policy must refuse reload {name:?}"
        );
        assert_eq!(serde_json::to_string(program).unwrap(), before);
        assert_eq!(*state.file_hashes.lock().unwrap(), hashes);
        assert_eq!(state.pending_source_names(), pending);
    }
    std::fs::remove_file(dir.join("Extra.ax")).unwrap();
    assert!(super::persist_program_module_to_dir(program, "Extra", Some(state), dir).is_err());
    assert!(!dir.join("Extra.ax").exists());
    assert_eq!(*state.file_hashes.lock().unwrap(), hashes);
    assert_eq!(state.pending_source_names(), pending);
    std::fs::write(dir.join("Extra.ax"), legacy).unwrap();
    let receipt = flush(program, state, dir);
    assert!(!receipt.failures.is_empty());
    assert_eq!(
        std::fs::read_to_string(dir.join("Extra.ax")).unwrap(),
        legacy
    );
    assert_eq!(*state.file_hashes.lock().unwrap(), hashes);
}

#[test]
fn managed_policy_empty_removed_never_reverts_to_legacy() {
    assert_retained_policy_guard(true, false);
}

#[test]
fn managed_policy_empty_dangling_never_reverts_to_legacy() {
    assert_retained_policy_guard(true, true);
}

#[test]
fn managed_policy_nonempty_removed_never_reverts_to_legacy() {
    assert_retained_policy_guard(false, false);
}

#[test]
fn managed_policy_nonempty_dangling_never_reverts_to_legacy() {
    assert_retained_policy_guard(false, true);
}

#[test]
fn managed_policy_new_dangling_blocks_previously_legacy_entry_points() {
    let dir = directory();
    let (mut program, state) = load(dir.path());
    let legacy = "+module Extra\n+fn old ()->Int\n  +return 7\n+end\n";
    for op in crate::parser::parse(legacy).unwrap() {
        crate::validator::apply_and_validate(&mut program, &op).unwrap();
    }
    std::fs::write(dir.path().join("Extra.ax"), legacy).unwrap();
    std::os::unix::fs::symlink(
        dir.path().join("missing-policy"),
        dir.path().join(MANIFEST_FILE),
    )
    .unwrap();
    assert_refuses_legacy_entry_points(&mut program, &state, dir.path(), legacy);
}

#[test]
fn managed_policy_never_managed_legacy_control_remains_writable() {
    let dir = directory();
    let (mut program, state) = load(dir.path());
    for op in crate::parser::parse("+module Extra\n+fn old ()->Int\n  +return 7\n+end\n").unwrap() {
        crate::validator::apply_and_validate(&mut program, &op).unwrap();
    }
    super::persist_program_module_to_dir(&program, "Extra", Some(&state), dir.path()).unwrap();
    assert!(dir.path().join("Extra.ax").exists());
    super::reload_module_from_dir(&mut program, "Extra", Some(&state), dir.path()).unwrap();
}

fn directory() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

#[test]
fn managed_manifest_orders_closed_dependencies() {
    let dir = directory();
    std::fs::write(
        dir.path().join(MANIFEST_FILE),
        r#"{
        "version":1,"modules":{
            "A":{"format":"python","dependencies":["Z"]},
            "Z":{"format":"python","dependencies":[]}
        }}"#,
    )
    .unwrap();
    let managed = ManagedSources::read(dir.path()).unwrap().unwrap();
    assert_eq!(managed.order, ["Z", "A"]);
    assert_eq!(
        managed
            .dependency_names("A")
            .into_iter()
            .collect::<Vec<_>>(),
        ["Z"]
    );
}

#[test]
fn managed_manifest_refuses_missing_cycles_duplicates_paths_and_unknown_fields() {
    let dir = directory();
    for manifest in [
        r#"{"version":1,"modules":{"A":{"format":"python","dependencies":["Missing"]}}}"#,
        r#"{"version":1,"modules":{"A":{"format":"python","dependencies":["A"]}}}"#,
        r#"{"version":1,"modules":{"../A":{"format":"python","dependencies":[]}}}"#,
        r#"{"version":2,"modules":{}}"#,
        r#"{"version":1,"modules":{},"guess":true}"#,
        r#"{"version":1,"modules":{"A":{"format":"python","dependencies":[]},"A":{"format":"legacy_ax","dependencies":[]}}}"#,
        r#"{"version":1,"modules":{"A":{"format":"python","dependencies":["B","B"]},"B":{"format":"python","dependencies":[]}}}"#,
    ] {
        std::fs::write(dir.path().join(MANIFEST_FILE), manifest).unwrap();
        assert!(ManagedSources::read(dir.path()).is_err());
    }
}

#[test]
fn managed_unreadable_manifest_is_not_absence_or_implicit_legacy_mode() {
    let dir = directory();
    assert!(ManagedSources::read(dir.path()).unwrap().is_none());
    std::os::unix::fs::symlink(
        dir.path().join("missing-policy"),
        dir.path().join(MANIFEST_FILE),
    )
    .unwrap();
    assert!(ManagedSources::read(dir.path()).is_err());
    let mut program = crate::ast::Program::default();
    let mut state = super::LibraryState::new();
    std::fs::write(
        dir.path().join("Legacy.ax"),
        "+module Legacy\n+fn old ()->Int\n  +return 1\n+end\n",
    )
    .unwrap();
    super::load_module_files(&mut program, dir.path(), &mut state);
    assert!(state.require_authoritative_sources().is_err());
    assert!(program.modules.is_empty());
}

#[test]
fn managed_manifest_collisions_and_drift_are_not_format_preferences() {
    let dir = directory();
    let manifest = r#"{"version":1,"modules":{"A":{"format":"python","dependencies":[]}}}"#;
    std::fs::write(dir.path().join(MANIFEST_FILE), manifest).unwrap();
    let managed = ManagedSources::read(dir.path()).unwrap().unwrap();
    std::fs::write(dir.path().join("A.py"), "python").unwrap();
    std::fs::write(dir.path().join("A.ax"), "legacy").unwrap();
    assert!(managed.verify_manifest(dir.path()).is_err());
    assert!(ManagedSources::read(dir.path()).is_err());
    std::fs::remove_file(dir.path().join("A.ax")).unwrap();
    std::fs::write(dir.path().join(MANIFEST_FILE), format!("{manifest}\n")).unwrap();
    assert!(managed.verify_manifest(dir.path()).is_err());
}

#[test]
fn managed_source_write_preserves_external_edit_and_manifest_drift() {
    let dir = directory();
    std::fs::write(
        dir.path().join(MANIFEST_FILE),
        r#"{"version":1,"modules":{"A":{"format":"python","dependencies":[]}}}"#,
    )
    .unwrap();
    let managed = ManagedSources::read(dir.path()).unwrap().unwrap();
    let path = dir.path().join("A.py");
    std::fs::write(&path, "old").unwrap();
    let hash = super::source_hash("old");
    std::fs::write(&path, "external").unwrap();
    assert!(super::source::write_checked(dir.path(), "A.py", "new", &hash, &managed).is_err());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "external");
    std::fs::write(&path, "old").unwrap();
    std::fs::write(dir.path().join(MANIFEST_FILE), "broken").unwrap();
    assert!(super::source::write_checked(dir.path(), "A.py", "new", &hash, &managed).is_err());
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "old");
}

#[test]
fn managed_source_write_is_owner_only_and_leaves_no_temporary_files() {
    use std::os::unix::fs::PermissionsExt;
    let dir = directory();
    std::fs::write(
        dir.path().join(MANIFEST_FILE),
        r#"{"version":1,"modules":{"A":{"format":"python","dependencies":[]}}}"#,
    )
    .unwrap();
    let managed = ManagedSources::read(dir.path()).unwrap().unwrap();
    let path = dir.path().join("A.py");
    std::fs::write(&path, "old").unwrap();
    super::source::write_checked(
        dir.path(),
        "A.py",
        "new",
        &super::source_hash("old"),
        &managed,
    )
    .unwrap();
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "new");
    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 2);
}

const PYTHON: &str = "def value(x: int) -> int:\n    a: int = x + 1\n    b: int = a + 1\n    return b\n\n@stored_test(\"A.value\")\ndef test_case_0() -> None:\n    assert A.value(1) == 3\n\n@stored_test(\"A.value\")\ndef test_case_1() -> None:\n    assert A.value(2) == 4\n";

fn python_library() -> tempfile::TempDir {
    let dir = directory();
    std::fs::write(
        dir.path().join(MANIFEST_FILE),
        r#"{"version":1,"modules":{"A":{"format":"python","dependencies":[]}}}"#,
    )
    .unwrap();
    std::fs::write(dir.path().join("A.py"), PYTHON).unwrap();
    dir
}

fn load(dir: &std::path::Path) -> (crate::ast::Program, super::LibraryState) {
    let mut program = crate::ast::Program::default();
    let mut state = super::LibraryState::new();
    super::load_module_files(&mut program, dir, &mut state);
    state.require_authoritative_sources().unwrap();
    (program, state)
}

fn run_cases(program: &crate::ast::Program, text: &str) {
    let source =
        crate::python_frontend::lower_python_module(text, "A", &crate::ast::Program::default())
            .unwrap();
    assert_eq!(source.tests.len(), 1);
    assert_eq!(source.tests[0].cases.len(), 2);
    for case in &source.tests[0].cases {
        crate::eval::eval_test_case(program, "A.value", case).unwrap();
    }
}

fn flush(
    program: &crate::ast::Program,
    state: &super::LibraryState,
    dir: &std::path::Path,
) -> super::SaveReceipt {
    let live = std::sync::Arc::new(tokio::sync::RwLock::new(program.clone()));
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(super::source_flush::flush_live_sources_in(
            live,
            Some(state.clone()),
            dir.to_path_buf(),
        ))
        .unwrap()
}

#[test]
fn managed_python_load_modify_save_restart_preserves_cases_without_forging_passed() {
    let dir = python_library();
    let (mut program, state) = load(dir.path());
    let tests = program.get_function("A.value").unwrap().tests.clone();
    assert_eq!(tests.len(), 2);
    assert!(tests.iter().all(|case| !case.passed));
    run_cases(&program, PYTHON);
    // Real frontend + validator mutation, not replacement source installed by hand.
    let modified = PYTHON
        .replace("a + 1", "a + 2")
        .replace("== 3", "== 4")
        .replace("A.value(2) == 4", "A.value(2) == 5");
    let lowered = crate::python_frontend::lower_python_module(
        &modified,
        "A",
        &crate::ast::Program::default(),
    )
    .unwrap();
    for op in &lowered.declarations.operations {
        crate::validator::apply_and_validate(&mut program, op).unwrap();
    }
    for test in &lowered.tests {
        for case in &test.cases {
            crate::eval::eval_test_case(&program, &test.function_name, case).unwrap();
        }
        crate::session::store_test(&mut program, &test.function_name, &test.cases);
    }
    assert!(flush(&program, &state, dir.path()).failures.is_empty());
    assert!(!dir.path().join("A.ax").exists());
    let text = std::fs::read_to_string(dir.path().join("A.py")).unwrap();
    assert_ne!(text, PYTHON);
    let (restarted, _) = load(dir.path());
    run_cases(&restarted, &text);
    let reloaded_tests = &restarted.get_function("A.value").unwrap().tests;
    assert_eq!(reloaded_tests.len(), 2);
    assert!(reloaded_tests.iter().all(|case| !case.passed));
    assert_eq!(reloaded_tests[0].expected, "4");
    assert_eq!(reloaded_tests[1].expected, "5");
    // A fresh test-harness process uses the actual authoritative loader rather
    // than inheriting the parent's Program, LibraryState or passing-test flags.
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            "library::source_tests::managed_restart_child_loads_python_and_runs_cases",
            "--nocapture",
        ])
        .env("ADAPSIS_TEST_MANAGED_SOURCE_DIR", dir.path())
        .env("HOME", dir.path())
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("MANAGED_RESTART_TWO_CASES_PASSED"));
}

#[test]
fn managed_python_loaded_cases_carry_typed_payload_and_invalid_typed_refuses_save() {
    let dir = python_library();
    let (mut program, state) = load(dir.path());
    let tests = &program.get_function("A.value").unwrap().tests;
    assert!(tests
        .iter()
        .all(|case| case.typed.is_some() && !case.passed));
    for case in tests {
        case.typed.as_ref().unwrap().case().unwrap();
    }
    // Display strings disagree with the typed payload: the printer must emit
    // the validated typed case, never the legacy string fallback.
    let mut function = (*program.get_function("A.value").unwrap()).clone();
    function.tests[0].expected = "999".into();
    let printed =
        crate::python_frontend::print_python_module(&program_with(&program, function.clone()), "A")
            .unwrap();
    assert!(printed.contains("== 3") && !printed.contains("999"));
    // A present but invalid typed payload is an error, not a silent fallback.
    let mut json = serde_json::to_value(&function.tests[0]).unwrap();
    json["typed"]["version"] = serde_json::json!(2);
    function.tests[0] = serde_json::from_value(json).unwrap();
    program = program_with(&program, function);
    let receipt = flush(&program, &state, dir.path());
    assert_eq!(receipt.failures.len(), 1);
    assert!(state.pending_source_names().contains("A"));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("A.py")).unwrap(),
        PYTHON
    );
}

fn program_with(
    program: &crate::ast::Program,
    function: crate::ast::FunctionDecl,
) -> crate::ast::Program {
    let mut program = program.clone();
    let module = program
        .modules
        .iter_mut()
        .find(|module| module.name == "A")
        .unwrap();
    let slot = module
        .functions
        .iter_mut()
        .find(|f| f.name == function.name)
        .unwrap();
    *slot = std::sync::Arc::new(function);
    program.rebuild_function_index();
    program
}

#[test]
#[ignore = "invoked only by the isolated parent roundtrip test in a fresh process"]
fn managed_restart_child_loads_python_and_runs_cases() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("ADAPSIS_TEST_MANAGED_SOURCE_DIR")
            .expect("isolated parent must supply source directory"),
    );
    let (program, _) = load(&dir);
    let text = std::fs::read_to_string(dir.join("A.py")).unwrap();
    let function = program.get_function("A.value").unwrap();
    assert_eq!(function.tests.len(), 2);
    assert!(function.tests.iter().all(|case| !case.passed));
    assert_eq!(function.tests[0].expected, "4");
    assert_eq!(function.tests[1].expected, "5");
    run_cases(&program, &text);
    println!("MANAGED_RESTART_TWO_CASES_PASSED");
}

#[test]
fn managed_python_bad_source_preserves_cache_and_blocks_startup_and_autosave() {
    let dir = python_library();
    let (mut program, _) = load(dir.path());
    let recovery = serde_json::to_string(&program).unwrap();
    std::fs::write(dir.path().join("A.py"), "def broken(: SECRET_PAYLOAD").unwrap();
    let mut state = super::LibraryState::new();
    super::load_module_files(&mut program, dir.path(), &mut state);
    let error = state
        .require_authoritative_sources()
        .unwrap_err()
        .to_string();
    assert!(error.contains("startup refused"));
    assert!(!error.contains("SECRET_PAYLOAD"));
    assert_eq!(serde_json::to_string(&program).unwrap(), recovery);
    assert!(super::persist_program_module_to_dir(&program, "A", Some(&state), dir.path()).is_err());
    assert_eq!(
        std::fs::read_to_string(dir.path().join("A.py")).unwrap(),
        "def broken(: SECRET_PAYLOAD"
    );
}

#[test]
fn managed_python_load_ignores_stale_session_callees_and_commits_whole_set() {
    let dir = python_library();
    let (mut program, _) = load(dir.path());
    let recovery = serde_json::to_string(&program).unwrap();
    std::fs::write(
        dir.path().join(MANIFEST_FILE),
        r#"{"version":1,"modules":{
        "A":{"format":"python","dependencies":["Z"]},
        "Z":{"format":"python","dependencies":[]}}}"#,
    )
    .unwrap();
    std::fs::write(
        dir.path().join("A.py"),
        "def value(x: int) -> int:\n    return Z.inc(x)\n",
    )
    .unwrap();
    std::fs::write(
        dir.path().join("Z.py"),
        "def inc(x: int) -> int:\n    return x + 10\n",
    )
    .unwrap();
    let mut state = super::LibraryState::new();
    super::load_module_files(&mut program, dir.path(), &mut state);
    state.require_authoritative_sources().unwrap();
    assert!(program.get_function("Z.inc").is_some());
    assert_ne!(serde_json::to_string(&program).unwrap(), recovery);
    let good = serde_json::to_string(&program).unwrap();
    // Remove dependency declaration: stale Z in memory cannot make the file valid.
    std::fs::write(
        dir.path().join(MANIFEST_FILE),
        r#"{"version":1,"modules":{
        "A":{"format":"python","dependencies":[]},
        "Z":{"format":"python","dependencies":[]}}}"#,
    )
    .unwrap();
    let mut restarted_state = super::LibraryState::new();
    super::load_module_files(&mut program, dir.path(), &mut restarted_state);
    assert!(restarted_state.require_authoritative_sources().is_err());
    assert_eq!(serde_json::to_string(&program).unwrap(), good);
    assert!(super::persist_program_module_to_dir(&program, "A", Some(&state), dir.path()).is_err());
}

#[test]
fn managed_python_unsupported_test_refuses_entire_save() {
    let dir = python_library();
    let (mut program, state) = load(dir.path());
    // A legacy (untyped) record whose display string is unsupported.
    program.get_function_mut("A.value").unwrap().tests[0].typed = None;
    program.get_function_mut("A.value").unwrap().tests[0].input =
        "arbitrary_unknown_helper()".to_string();
    let receipt = flush(&program, &state, dir.path());
    assert_eq!(receipt.failures.len(), 1);
    assert!(receipt.saved_names.is_empty());
    assert!(state.pending_source_names().contains("A"));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("A.py")).unwrap(),
        PYTHON
    );
    assert_eq!(program.get_function("A.value").unwrap().tests.len(), 2);
}

#[test]
fn managed_python_all_hot_reload_refused_and_startup_route_counts_stay_exact() {
    let dir = python_library();
    let routed = format!(
        "@route(\"POST\", \"/hook\")\ndef handle(body: str) -> str:\n    return body\n\n{PYTHON}"
    );
    std::fs::write(dir.path().join("A.py"), &routed).unwrap();
    let (mut program, state) = load(dir.path());
    assert_eq!(program.pending_routes.len(), 1);
    assert_eq!(program.modules[0].routes.len(), 1);
    let registered = std::mem::take(&mut program.pending_routes);
    for _ in 0..3 {
        let error =
            super::reload_module_from_dir(&mut program, "A", Some(&state), dir.path()).unwrap_err();
        assert!(error.to_string().contains("requires restart"));
        assert!(program.pending_routes.is_empty());
        assert_eq!(program.modules[0].routes, registered);
        assert_eq!(program.get_function("A.value").unwrap().tests.len(), 2);
    }
    let good = serde_json::to_string(&program).unwrap();
    std::fs::write(
        dir.path().join("A.py"),
        routed.replace("/hook", "/new-hook"),
    )
    .unwrap();
    let error =
        super::reload_module_from_dir(&mut program, "", Some(&state), dir.path()).unwrap_err();
    assert!(error.to_string().contains("requires restart"));
    assert_eq!(serde_json::to_string(&program).unwrap(), good);
    assert!(program.pending_routes.is_empty());
    assert_eq!(flush(&program, &state, dir.path()).failures.len(), 1);
    let (fresh, _) = load(dir.path());
    assert_eq!(fresh.pending_routes.len(), 1);
    assert_eq!(fresh.pending_routes[0].path, "/new-hook");
}

#[test]
fn managed_python_valid_source_wins_over_cache_and_partial_temp_is_ignored() {
    let dir = python_library();
    let (mut program, _) = load(dir.path());
    program.get_function_mut("A.value").unwrap().doc = Some("stale session cache".into());
    std::fs::write(dir.path().join(".A.py.999.0.tmp"), "partial invalid bytes").unwrap();
    let mut state = super::LibraryState::new();
    super::load_module_files(&mut program, dir.path(), &mut state);
    state.require_authoritative_sources().unwrap();
    assert!(program.get_function("A.value").unwrap().doc.is_none());
    assert_eq!(program.get_function("A.value").unwrap().tests.len(), 2);
    assert_eq!(
        std::fs::read_to_string(dir.path().join("A.py")).unwrap(),
        PYTHON
    );
    std::fs::write(dir.path().join("A.ax"), "+module A\n").unwrap();
    let good = serde_json::to_string(&program).unwrap();
    let mut collision_state = super::LibraryState::new();
    super::load_module_files(&mut program, dir.path(), &mut collision_state);
    assert!(collision_state.require_authoritative_sources().is_err());
    assert_eq!(serde_json::to_string(&program).unwrap(), good);
}

#[test]
fn managed_python_source_commit_survives_failed_session_checkpoint() {
    let dir = python_library();
    let (mut program, state) = load(dir.path());
    program.get_function_mut("A.value").unwrap().doc = Some("new authoritative revision".into());
    assert!(flush(&program, &state, dir.path()).failures.is_empty());
    let mut session = crate::session::Session::new();
    session.program = program;
    session.meta.library_state = Some(state);
    assert!(session
        .save(&dir.path().join("missing-parent/session.json"))
        .is_err());
    let (restarted, _) = load(dir.path());
    assert_eq!(
        restarted.get_function("A.value").unwrap().doc.as_deref(),
        Some("new authoritative revision")
    );
    assert_eq!(restarted.get_function("A.value").unwrap().tests.len(), 2);
}

#[test]
fn managed_legacy_import_refuses_duplicate_members_without_overwriting_source() {
    let dir = directory();
    std::fs::write(
        dir.path().join(MANIFEST_FILE),
        r#"{"version":1,"modules":{"A":{"format":"legacy_ax","dependencies":[]}}}"#,
    )
    .unwrap();
    let source = "+module A\n+fn value (x:Int)->Int\n  +return x + 1\n+end\n+test value\n  +with 1 -> expect 2\n";
    std::fs::write(dir.path().join("A.ax"), source).unwrap();
    let (mut program, state) = load(dir.path());
    assert_eq!(program.get_function("A.value").unwrap().tests.len(), 1);
    assert!(!program.get_function("A.value").unwrap().tests[0].passed);
    let good = serde_json::to_string(&program).unwrap();
    let duplicate = format!("{source}+fn value (x:Int)->Int\n  +return x + 2\n+end\n");
    std::fs::write(dir.path().join("A.ax"), &duplicate).unwrap();
    let mut restarted_state = super::LibraryState::new();
    super::load_module_files(&mut program, dir.path(), &mut restarted_state);
    assert!(restarted_state.require_authoritative_sources().is_err());
    assert_eq!(serde_json::to_string(&program).unwrap(), good);
    assert!(super::persist_program_module_to_dir(&program, "A", Some(&state), dir.path()).is_err());
    assert_eq!(
        std::fs::read_to_string(dir.path().join("A.ax")).unwrap(),
        duplicate
    );
    assert!(!dir.path().join("A.py").exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn managed_delayed_flush_reads_current_live_program_and_old_snapshot_cannot_save() {
    use std::sync::Arc;
    let dir = python_library();
    let (program, state) = load(dir.path());
    let old_snapshot = program.clone();
    let live = Arc::new(tokio::sync::RwLock::new(program));
    let mut publishing = live.write().await;
    let (queued_tx, queued_rx) = tokio::sync::oneshot::channel();
    let worker_live = live.clone();
    let worker_state = state.clone();
    let path = dir.path().to_path_buf();
    let delayed_flush = tokio::spawn(async move {
        queued_tx.send(()).unwrap();
        super::source_flush::flush_live_sources_in(worker_live, Some(worker_state), path).await
    });
    queued_rx.await.unwrap();
    publishing.get_function_mut("A.value").unwrap().doc = Some("new live revision".into());
    drop(publishing);
    let receipt = delayed_flush.await.unwrap().unwrap();
    assert!(receipt.failures.is_empty());
    assert_eq!(
        receipt
            .program_snapshot
            .get_function("A.value")
            .unwrap()
            .doc
            .as_deref(),
        Some("new live revision")
    );
    let current = std::fs::read_to_string(dir.path().join("A.py")).unwrap();
    assert!(current.contains("new live revision"));
    assert!(
        super::persist_program_module_to_dir(&old_snapshot, "A", Some(&state), dir.path()).is_err()
    );
    assert_eq!(
        std::fs::read_to_string(dir.path().join("A.py")).unwrap(),
        current
    );
    assert!(state.pending_source_names().is_empty());
}

#[test]
fn managed_disjoint_mutations_queue_without_writing_and_flush_the_merged_live_program() {
    let dir = python_library();
    let source = format!("def other(x: int) -> int:\n    return x\n\n{PYTHON}");
    std::fs::write(dir.path().join("A.py"), &source).unwrap();
    let (base, state) = load(dir.path());
    let mut left = base.clone();
    let mut right = base.clone();
    left.get_function_mut("A.value").unwrap().doc = Some("left modification".into());
    right.get_function_mut("A.other").unwrap().doc = Some("right modification".into());
    let affected = std::collections::BTreeSet::from(["A".to_string()]);
    for snapshot in [&left, &right] {
        let results = super::persist_affected_modules(snapshot, &affected, Some(&state));
        assert_eq!(results.len(), 1);
        assert!(results[0].1);
        assert!(results[0].0.contains("not yet durable"));
        assert_eq!(
            std::fs::read_to_string(dir.path().join("A.py")).unwrap(),
            source
        );
    }
    assert!(state.pending_source_names().contains("A"));
    let mut live = base.clone();
    live.merge_changed_from(&base, &left);
    live.merge_changed_from(&base, &right);
    let receipt = flush(&live, &state, dir.path());
    assert!(receipt.failures.is_empty());
    let (restarted, _) = load(dir.path());
    assert_eq!(
        restarted.get_function("A.value").unwrap().doc.as_deref(),
        Some("left modification")
    );
    assert_eq!(
        restarted.get_function("A.other").unwrap().doc.as_deref(),
        Some("right modification")
    );
    assert!(state.pending_source_names().is_empty());
}

#[test]
fn managed_failed_save_keeps_pending_recovery_cache_without_making_it_source_authority() {
    let dir = python_library();
    let (mut program, state) = load(dir.path());
    // A legacy (untyped) record whose display string is unsupported.
    program.get_function_mut("A.value").unwrap().tests[0].typed = None;
    program.get_function_mut("A.value").unwrap().tests[0].input =
        "unsupported_saved_helper()".into();
    for _ in 0..2 {
        let receipt = flush(&program, &state, dir.path());
        assert_eq!(receipt.failures.len(), 1);
        assert!(state.pending_source_names().contains("A"));
        assert_eq!(
            receipt
                .program_snapshot
                .get_function("A.value")
                .unwrap()
                .tests[0]
                .input,
            "unsupported_saved_helper()"
        );
    }
    let mut session = crate::session::Session::new();
    session.program = program;
    session.meta.library_state = Some(state);
    let path = dir.path().join("recovery.json");
    session.save(&path).unwrap();
    let recovery = std::fs::read(&path).unwrap();
    let mut restarted = crate::session::Session::load(&path).unwrap();
    assert_eq!(
        restarted.program.get_function("A.value").unwrap().tests[0].input,
        "unsupported_saved_helper()"
    );
    let mut new_state = super::LibraryState::new();
    super::load_module_files(&mut restarted.program, dir.path(), &mut new_state);
    new_state.require_authoritative_sources().unwrap();
    assert_ne!(
        restarted.program.get_function("A.value").unwrap().tests[0].input,
        "unsupported_saved_helper()"
    );
    assert_eq!(std::fs::read(&path).unwrap(), recovery);
    assert_eq!(
        std::fs::read_to_string(dir.path().join("A.py")).unwrap(),
        PYTHON
    );
}

#[test]
fn managed_bounded_source_write_failure_keeps_original_and_pending_state() {
    let dir = python_library();
    let (mut program, state) = load(dir.path());
    program.get_function_mut("A.value").unwrap().doc =
        Some("x".repeat(super::source::MAX_SOURCE_BYTES + 1));
    let receipt = flush(&program, &state, dir.path());
    assert_eq!(receipt.failures.len(), 1);
    assert!(state.pending_source_names().contains("A"));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("A.py")).unwrap(),
        PYTHON
    );
    assert_eq!(
        std::fs::read_dir(dir.path()).unwrap().count(),
        2,
        "budget refusal must happen before even a partial temporary source is created"
    );
    assert!(!receipt
        .program_snapshot
        .get_function("A.value")
        .unwrap()
        .doc
        .as_ref()
        .unwrap()
        .is_empty());
}

#[test]
fn managed_source_authority_cannot_be_overwritten_by_an_unmanaged_legacy_file() {
    let dir = python_library();
    std::fs::write(dir.path().join("Legacy.ax"), "+module A\n+fn value (x:Int)->Int\n  +return 999\n+end\n+module Legacy\n+fn harmless ()->Int\n  +return 1\n+end\n").unwrap();
    let (program, state) = load(dir.path());
    assert!(program.get_function("Legacy.harmless").is_none());
    assert_eq!(program.get_function("A.value").unwrap().tests.len(), 2);
    run_cases(&program, PYTHON);
    assert!(state
        .load_errors
        .lock()
        .unwrap()
        .iter()
        .any(|error| error.module_name == "Legacy" && error.error.contains("not listed")));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("A.py")).unwrap(),
        PYTHON
    );
}

#[test]
fn managed_whitelist_refuses_unlisted_cached_code_and_ignores_route_hijack_files() {
    let dir = python_library();
    let routed = format!(
        "@route(\"POST\", \"/hook\")\ndef handle(body: str) -> str:\n    return body\n\n{PYTHON}"
    );
    std::fs::write(dir.path().join("A.py"), &routed).unwrap();
    std::fs::write(dir.path().join("Legacy.ax"), "+module Legacy\n+fn steal (body:String)->String\n  +return body\n+end\n+route POST /hook -> steal\n").unwrap();
    let (mut program, state) = load(dir.path());
    assert_eq!(program.pending_routes.len(), 1);
    assert_eq!(program.pending_routes[0].handler_fn, "A.handle");
    assert!(program.get_function("Legacy.steal").is_none());
    for op in crate::parser::parse("+module Cached\n+fn old ()->Int\n  +return 9\n+end\n").unwrap()
    {
        crate::validator::apply_and_validate(&mut program, &op).unwrap();
    }
    let recovery = serde_json::to_string(&program).unwrap();
    let mut fresh_state = super::LibraryState::new();
    super::load_module_files(&mut program, dir.path(), &mut fresh_state);
    let error = fresh_state.require_authoritative_sources().unwrap_err();
    assert!(format!("{error:#}").contains("Cached"));
    assert_eq!(serde_json::to_string(&program).unwrap(), recovery);
    assert!(
        super::persist_program_module_to_dir(&program, "Cached", Some(&state), dir.path()).is_err()
    );
    assert!(!dir.path().join("Cached.ax").exists());
    assert_eq!(
        std::fs::read_to_string(dir.path().join("A.py")).unwrap(),
        routed
    );
}

#[test]
fn managed_rejected_signature_change_restores_whole_function_live_and_on_disk() {
    let dir = python_library();
    let (program, state) = load(dir.path());
    let original = program.get_function("A.value").unwrap().clone();
    let mut session = crate::session::Session::new();
    session.program = program;
    session.meta.library_state = Some(state.clone());
    // Changes params, return type and body; the stored one-argument cases fail.
    let results = session
        .apply("+module A\n+fn value (x:Int, y:Int)->String\n  +return \"changed\"\n+end\n")
        .unwrap();
    assert!(
        results
            .iter()
            .any(|(message, ok)| !ok && message.contains("REJECTED")),
        "{results:?}"
    );
    let live = session.program.get_function("A.value").unwrap();
    assert_eq!(live.params, original.params);
    assert_eq!(live.return_type, original.return_type);
    assert_eq!(live.effects, original.effects);
    assert_eq!(live.body, original.body);
    assert_eq!(
        live.tests, original.tests,
        "pass flags are restored, not forged"
    );
    let receipt = flush(&session.program, &state, dir.path());
    assert!(receipt.failures.is_empty(), "{:?}", receipt.failures);
    let (reloaded, _) = load(dir.path());
    let disk = reloaded.get_function("A.value").unwrap();
    assert_eq!(disk.params, original.params);
    assert_eq!(disk.return_type, original.return_type);
    assert_eq!(disk.body, original.body);
}

#[test]
fn managed_legacy_import_with_bare_local_calls_saves_qualified_python_and_counts_sites() {
    let dir = directory();
    std::fs::write(
        dir.path().join(MANIFEST_FILE),
        r#"{"version":1,"modules":{"Q":{"format":"python","dependencies":[]}}}"#,
    )
    .unwrap();
    // Valid Python with qualified local calls loads; then the live Program is
    // given bare local calls, as legacy .ax modules commonly carry.
    std::fs::write(
        dir.path().join("Q.py"),
        "def g(x: int) -> int:\n    return x + 10\n\ndef f(x: int) -> int:\n    return Q.g(x)\n",
    )
    .unwrap();
    let (mut program, state) = load(dir.path());
    for op in crate::parser::parse("+module Q\n+fn f (x:Int)->Int\n  +return g(x)\n+end\n").unwrap()
    {
        crate::validator::apply_and_validate(&mut program, &op).unwrap();
    }
    let receipt = flush(&program, &state, dir.path());
    assert!(receipt.failures.is_empty(), "{:?}", receipt.failures);
    assert_eq!(receipt.qualified_call_sites, vec![("Q".to_string(), 1)]);
    let (reloaded, _) = load(dir.path());
    let crate::ast::StatementKind::Return {
        value: crate::ast::Expr::Call(call),
    } = &reloaded.get_function("Q.f").unwrap().body[0].kind
    else {
        panic!("unexpected reloaded shape")
    };
    assert_eq!(call.callee, "Q.g");
}

#[test]
fn managed_flush_refuses_qualification_when_live_resolution_differs_from_catalog() {
    let dir = python_library();
    let (mut program, state) = load(dir.path());
    // An unlisted live module earlier in order also defines `value`: the closed
    // catalog proves A.value, but the stored bare call ran against the live set.
    for op in crate::parser::parse("+module Early\n+fn value (x:Int)->Int\n  +return x\n+end\n+module A\n+fn caller (x:Int)->Int\n  +return value(x)\n+end\n").unwrap() {
        crate::validator::apply_and_validate(&mut program, &op).unwrap();
    }
    let early = program
        .modules
        .iter()
        .position(|m| m.name == "Early")
        .unwrap();
    let moved = program.modules.remove(early);
    program.modules.insert(0, moved);
    program.rebuild_function_index();
    let receipt = flush(&program, &state, dir.path());
    // `Early` is unlisted, so its snapshot write is refused too; the managed
    // module must fail on the live-resolution proof specifically.
    assert!(
        receipt.failures.iter().any(|f| f.contains("module `A`")
            && f.contains("resolves to another declaration in the live program")),
        "{:?}",
        receipt.failures
    );
    assert!(!receipt.saved_names.contains(&"A".to_string()));
    assert!(receipt.qualified_call_sites.is_empty());
    assert!(state.pending_source_names().contains("A"));
    assert_eq!(
        std::fs::read_to_string(dir.path().join("A.py")).unwrap(),
        PYTHON
    );
}

#[test]
fn managed_legacy_ax_bare_self_call_persists_despite_earlier_same_name_module() {
    let dir = directory();
    // `Alpha` (python) loads before `Mod` (legacy_ax) and also declares `foo`.
    std::fs::write(
        dir.path().join(MANIFEST_FILE),
        r#"{"version":1,"modules":{"Alpha":{"format":"python","dependencies":[]},"Mod":{"format":"legacy_ax","dependencies":[]}}}"#,
    )
    .unwrap();
    std::fs::write(
        dir.path().join("Alpha.py"),
        "def foo(x: int) -> int:\n    return x\n",
    )
    .unwrap();
    let legacy = "+module Mod\n+fn foo (x:Int)->Int\n  +return x + 1\n+end\n+fn bar (x:Int)->Int\n  +return foo(x)\n+end\n";
    std::fs::write(dir.path().join("Mod.ax"), legacy).unwrap();
    let (mut program, state) = load(dir.path());
    assert_eq!(program.modules[0].name, "Alpha");
    program.get_function_mut("Mod.bar").unwrap().doc = Some("edited".into());
    let receipt = flush(&program, &state, dir.path());
    assert!(receipt.failures.is_empty(), "{:?}", receipt.failures);
    assert!(receipt.saved_names.contains(&"Mod".to_string()));
    assert!(receipt.qualified_call_sites.is_empty() && receipt.order_dependent.is_empty());
    assert!(!state.pending_source_names().contains("Mod"));
    let saved = std::fs::read_to_string(dir.path().join("Mod.ax")).unwrap();
    assert!(
        saved.contains("edited") && saved.contains("+return foo(x)"),
        "{saved}"
    );
}
