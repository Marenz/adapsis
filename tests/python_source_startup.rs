//! Process-level proof that a cached module cannot bypass failed authoritative
//! source loading, that a refused start exits with the dedicated code, and that
//! a module which exists only in the session cache is adopted or dropped
//! instead of refusing the start (#128). All paths and the would-be startup
//! effect are isolated.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// `library::EXIT_SOURCE_REFUSED` (EX_CONFIG): the unit must not restart on it.
const REFUSED: i32 = 78;

#[test]
fn managed_python_broken_source_refuses_new_process_before_startup_io() {
    assert_startup_refusal(Fixture { source: "def broken(: PRIVATE_SOURCE", ..Fixture::default() });
}

#[test]
fn managed_unusable_regular_file_root_blocks_startup() {
    assert_startup_refusal(Fixture { root_kind: "file", ..Fixture::default() });
}

#[test]
fn managed_unusable_inaccessible_root_blocks_startup() {
    assert_startup_refusal(Fixture { root_kind: "inaccessible", ..Fixture::default() });
}

#[test]
fn managed_unusable_root_creation_failure_blocks_startup() {
    assert_startup_refusal(Fixture { root_kind: "parent-file", ..Fixture::default() });
}

#[test]
fn managed_cached_top_level_function_refuses_with_exit_78() {
    // Cached top-level code has no module to live in: still a refusal, with
    // the dedicated exit code so the service does not restart in a loop.
    let run = start(Fixture { top_level: true, ..Fixture::default() });
    assert_eq!(run.code, Some(REFUSED), "{}", run.stderr);
    assert!(run.stderr.contains("startup refused"), "{}", run.stderr);
    assert!(!run.marker_exists && run.session_unchanged);
}

#[test]
fn managed_cached_printable_module_is_adopted_and_startup_continues() {
    // A module that exists only in the session cache no longer refuses the
    // start. It is written as source, listed in the manifest, and the process
    // goes on (here until the deliberately unusable memory database).
    let run = start(Fixture { cached_module: "Unlisted", guard_database: true, ..Fixture::default() });
    assert_ne!(run.code, Some(REFUSED), "{}", run.stderr);
    assert!(!run.stderr.contains("startup refused"), "{}", run.stderr);
    assert!(run.stderr.contains("adopted cached module `Unlisted`"), "{}", run.stderr);
    assert!(run.modules.join("Unlisted.py").is_file());
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(run.modules.join("library-sources.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(manifest["modules"]["Unlisted"]["format"], "python");
    assert!(manifest["modules"]["A"].is_object());
    assert!(!run.marker_exists, "lifecycle code must not run before initialization");
}

#[test]
fn managed_cached_unprintable_module_is_dropped_with_a_recovery_file() {
    // A function named like a builtin cannot be printed as Python: the module
    // is removed before it can run, kept in a recovery file, and startup goes on.
    let run = start(Fixture {
        cached_module: "Unlisted",
        unprintable: true,
        guard_database: true,
        ..Fixture::default()
    });
    assert_ne!(run.code, Some(REFUSED), "{}", run.stderr);
    assert!(!run.stderr.contains("startup refused"), "{}", run.stderr);
    assert!(run.stderr.contains("dropped cached module `Unlisted`"), "{}", run.stderr);
    assert!(!run.modules.join("Unlisted.py").exists());
    let recovery: Vec<_> = std::fs::read_dir(run.modules.join("recovery"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert_eq!(recovery.len(), 1);
    assert!(recovery[0].to_string_lossy().starts_with("Unlisted-"));
    let manifest = std::fs::read_to_string(run.modules.join("library-sources.json")).unwrap();
    assert!(!manifest.contains("Unlisted"));
    assert!(!run.marker_exists);
}

/// An unreadable session is a refused start too: exit 78 (no restart loop),
/// the file named and left untouched, nothing initialized.
#[test]
fn unreadable_session_file_refuses_the_start_with_exit_78() {
    let complete = serde_json::to_string(&cached_startup(&Fixture::default(), std::path::Path::new("/nonexistent/marker"))).unwrap();
    let deep = format!("{}{}", "[".repeat(5000), "]".repeat(5000));
    for (name, text) in [("truncated", &complete[..complete.len() / 2]), ("too deep", deep.as_str()), ("empty", "")] {
        let dir = tempfile::tempdir().unwrap();
        let session = dir.path().join("session.json");
        std::fs::write(&session, text).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_adapsis"))
            .args(["os", "--session", session.to_str().unwrap(), "--port", "0", "--model", "offline-source-test", "--url", "http://127.0.0.1:9"])
            .current_dir(dir.path())
            .env("HOME", dir.path())
            .env("ADAPSIS_MODULES_DIR", dir.path().join("modules"))
            .env("ADAPSIS_MEMORY_DB", dir.path().join("memory.lbug"))
            .env("ADAPSIS_CONTEXTS_DIR", dir.path().join("contexts"))
            .env("ADAPSIS_MESH_FILE", dir.path().join("mesh.md"))
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(REFUSED), "{name}: {stderr}");
        assert!(stderr.contains("startup refused: session file") && stderr.contains(session.to_str().unwrap()), "{name}: {stderr}");
        assert!(stderr.contains("Restore the last backup copy"), "{name}: {stderr}");
        assert_eq!(std::fs::read_to_string(&session).unwrap(), text, "{name}: the file is untouched");
        assert!(!dir.path().join("memory.lbug").exists() && !dir.path().join("modules").exists(), "{name}: nothing was initialized");
    }
}

fn assert_startup_refusal(fixture: Fixture) {
    let run = start(fixture);
    assert_eq!(run.code, Some(REFUSED), "refused starts exit with 78: {}", run.stderr);
    assert!(run.stderr.contains("startup refused"), "{}", run.stderr);
    assert!(!run.stderr.contains("PRIVATE_SOURCE"));
    assert!(!run.marker_exists);
    assert!(!run.memory_exists);
    assert!(run.session_unchanged);
}

struct Fixture {
    cached_module: &'static str,
    source: &'static str,
    root_kind: &'static str,
    /// Give the cached module a function the Python printer refuses.
    unprintable: bool,
    /// Add a cached top-level function.
    top_level: bool,
    /// Point the memory database below a regular file, so a start that passes
    /// the library gate stops there, offline, before any embedding setup.
    guard_database: bool,
}

impl Default for Fixture {
    fn default() -> Self {
        Fixture {
            cached_module: "A",
            source: "def value() -> int:\n    return 1\n",
            root_kind: "directory",
            unprintable: false,
            top_level: false,
            guard_database: false,
        }
    }
}

struct Run {
    code: Option<i32>,
    stderr: String,
    modules: std::path::PathBuf,
    marker_exists: bool,
    memory_exists: bool,
    session_unchanged: bool,
    _dir: tempfile::TempDir,
}

fn one_liner(name: &str) -> serde_json::Value {
    serde_json::json!({"id": name, "name": name, "params": [], "return_type": "Int", "effects": [],
        "body": [{"id": "r", "kind": {"Return": {"value": {"Literal": {"Int": 1}}}}}]})
}

fn cached_startup(fixture: &Fixture, marker: &std::path::Path) -> serde_json::Value {
    let functions = if fixture.unprintable { vec![one_liner("eval")] } else { Vec::new() };
    let top_level = if fixture.top_level { vec![one_liner("loose")] } else { Vec::new() };
    serde_json::json!({
        "program": {
            "modules": [{
                "id":"cached-module", "name":fixture.cached_module, "types":[], "functions":functions, "modules":[],
                "startup": {"id":"cached-startup", "effects":["Io","Async"], "body":[{
                    "id":"startup-write", "kind":{"Await":{
                        "name":"written", "ty":"String", "call":{
                            "callee":"file_write", "args":[
                                {"Literal":{"String":marker.to_str().unwrap()}},
                                {"Literal":{"String":"unexpected startup effect"}}
                            ]
                        }
                    }}
                }]}
            }],
            "functions":top_level, "types":[]
        },
        "mutations":[], "history":[], "revision":0, "sources":[]
    })
}

fn start(fixture: Fixture) -> Run {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let modules = dir.path().join("modules");
    std::fs::create_dir(&modules).unwrap();
    let marker = dir.path().join("startup-ran");
    let memory = dir.path().join("memory.lbug");
    let cache = cached_startup(&fixture, &marker);
    let session = dir.path().join("session.json");
    let bytes = serde_json::to_vec(&cache).unwrap();
    std::fs::write(&session, &bytes).unwrap();
    std::fs::write(
        modules.join("library-sources.json"),
        r#"{"version":1,"modules":{"A":{"format":"python","dependencies":[]}}}"#,
    )
    .unwrap();
    std::fs::write(modules.join("A.py"), fixture.source).unwrap();
    let root_kind = fixture.root_kind;
    let library_root = match root_kind {
        "file" | "parent-file" => {
            let file = dir.path().join("not-directory");
            std::fs::write(&file, "source-root fixture").unwrap();
            if root_kind == "file" {
                file
            } else {
                file.join("modules")
            }
        }
        "inaccessible" => {
            std::fs::set_permissions(&modules, std::fs::Permissions::from_mode(0o000)).unwrap();
            modules.clone()
        }
        _ => modules.clone(),
    };
    // A broken implementation must not proceed to embeddings/provider setup.
    // This private DB guard makes that wrong path fail immediately and offline.
    let db_guard = dir.path().join("db-guard");
    std::fs::write(&db_guard, "not a directory").unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_adapsis"))
        .args([
            "os",
            "--session",
            session.to_str().unwrap(),
            "--port",
            "0",
            "--model",
            "offline-source-test",
            "--url",
            "http://127.0.0.1:9",
        ])
        .current_dir(dir.path())
        .env("HOME", dir.path())
        .env("ADAPSIS_MODULES_DIR", &library_root)
        .env(
            "ADAPSIS_MEMORY_DB",
            if root_kind == "directory" && !fixture.guard_database {
                memory.clone()
            } else {
                db_guard.join("memory.lbug")
            },
        )
        .env("ADAPSIS_CONTEXTS_DIR", dir.path().join("contexts"))
        .env("ADAPSIS_MESH_FILE", dir.path().join("mesh.md"))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while child.try_wait().unwrap().is_none() {
        if Instant::now() >= deadline {
            child.kill().unwrap();
            let _ = child.wait();
            panic!("OS neither refused nor stopped at the database guard before initialization");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let output = child.wait_with_output().unwrap();
    if root_kind == "inaccessible" {
        std::fs::set_permissions(&modules, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    assert!(!output.status.success());
    Run {
        code: output.status.code(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        marker_exists: marker.exists(),
        memory_exists: memory.exists(),
        session_unchanged: std::fs::read(&session).unwrap() == bytes,
        modules,
        _dir: dir,
    }
}
