//! Rewind/undo through the real API entry points under a source manifest.
//! History has no startup baseline for disk-loaded managed modules, so both
//! entry points must refuse before touching Program, meta or source files.
use super::*;

const PYTHON: &str = "def value(x: int) -> int:\n    a: int = x + 1\n    b: int = a + 1\n    return b\n\n@stored_test(\"RwA.value\")\ndef test_case_0() -> None:\n    assert RwA.value(1) == 3\n";

struct Fixture {
    _dir: tempfile::TempDir,
    source: std::path::PathBuf,
    config: AppConfig,
}

/// A config whose live Program/LibraryState were loaded from a private library.
/// `manifest` None means a legacy (unmanifested) library.
async fn fixture(manifest: Option<&str>) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    if let Some(manifest) = manifest {
        std::fs::write(dir.path().join("library-sources.json"), manifest).unwrap();
    }
    let source = dir.path().join("RwA.py");
    std::fs::write(&source, PYTHON).unwrap();
    let mut program = crate::ast::Program::default();
    let state = crate::library::load_library_from_dir_for_tests(&mut program, dir.path());
    let config = test_config();
    *config.program.write().await = program;
    {
        let mut meta = config.meta.lock().unwrap();
        meta.library_state = Some(state);
        // A reconstructible revision exists, so both entry points would
        // otherwise rewind.
        let history = "+module RwHistory\n+fn h ()->Int\n  +return 1\n+end\n";
        meta.sources
            .push(crate::session::replay::ReplayEntry::Legacy(history.into()));
        meta.mutations.push(crate::session::MutationEntry {
            revision: 1,
            timestamp: String::new(),
            source: history.into(),
            origin: crate::session::replay::SourceOrigin::Adapsis,
            summary: String::new(),
            success: true,
        });
        meta.revision = 1;
    }
    Fixture {
        _dir: dir,
        source,
        config,
    }
}

struct Snapshot {
    program: String,
    revision: usize,
    sources: usize,
    bytes: Vec<u8>,
    pending: std::collections::BTreeSet<String>,
}

async fn snapshot(f: &Fixture) -> Snapshot {
    let program = serde_json::to_string(&*f.config.program.read().await).unwrap();
    let meta = f.config.meta.lock().unwrap();
    Snapshot {
        program,
        revision: meta.revision,
        sources: meta.sources.len(),
        bytes: std::fs::read(&f.source).unwrap(),
        pending: meta.library_state.as_ref().unwrap().pending_source_names(),
    }
}

async fn assert_unchanged(f: &Fixture, before: &Snapshot) {
    let after = snapshot(f).await;
    assert_eq!(after.program, before.program);
    assert_eq!(after.revision, before.revision);
    assert_eq!(after.sources, before.sources);
    assert_eq!(after.bytes, before.bytes);
    assert_eq!(after.pending, before.pending);
}

const MANAGED: &str = r#"{"version":1,"modules":{"RwA":{"format":"python","dependencies":[]}}}"#;
const EMPTY: &str = r#"{"version":1,"modules":{}}"#;

#[tokio::test]
async fn managed_rewind_api_refuses_before_any_state_change() {
    for manifest in [MANAGED, EMPTY] {
        let f = fixture(Some(manifest)).await;
        let before = snapshot(&f).await;
        let Json(response) =
            rewind(State(f.config.clone()), Json(RewindRequest { revision: 0 })).await;
        assert!(!response.success);
        assert_eq!(response.message, crate::library::REWIND_REFUSAL);
        assert_eq!(response.revision, before.revision);
        assert_unchanged(&f, &before).await;
    }
}

#[tokio::test]
async fn managed_undo_refuses_before_any_state_change() {
    for manifest in [MANAGED, EMPTY] {
        let f = fixture(Some(manifest)).await;
        let before = snapshot(&f).await;
        let mut session = f.config.snapshot_working_set().await;
        let result = execute::execute_code("!undo", &f.config, &mut session, None).await;
        assert!(result.has_errors);
        assert!(result.mutation_results.iter().any(
            |r| !r.success && r.message == format!("Undo: {}", crate::library::REWIND_REFUSAL)
        ));
        assert!(!result
            .mutation_results
            .iter()
            .any(|r| r.success && r.message.contains("Undone")));
        // The working set must not have been rewound either.
        assert_eq!(
            serde_json::to_string(&session.program).unwrap(),
            before.program
        );
        assert_eq!(session.meta.revision, before.revision);
        assert_unchanged(&f, &before).await;
    }
}

#[tokio::test]
async fn unmanifested_rewind_still_rewinds() {
    let f = fixture(None).await;
    {
        let state = f.config.meta.lock().unwrap();
        assert!(state
            .library_state
            .as_ref()
            .unwrap()
            .managed_module_names()
            .is_empty());
    }
    let Json(response) = rewind(State(f.config.clone()), Json(RewindRequest { revision: 1 })).await;
    assert!(response.success, "{}", response.message);
    assert_eq!(response.revision, 1);
    assert!(f
        .config
        .program
        .read()
        .await
        .get_function("RwHistory.h")
        .is_some());
    let Json(response) = rewind(State(f.config.clone()), Json(RewindRequest { revision: 0 })).await;
    assert!(response.success, "{}", response.message);
    assert_eq!(response.revision, 0);
    assert!(f
        .config
        .program
        .read()
        .await
        .get_function("RwHistory.h")
        .is_none());
}
