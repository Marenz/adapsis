//! Process-level proof that a cached module cannot bypass failed authoritative
//! source loading. All paths and the would-be startup effect are isolated.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[test]
fn managed_python_broken_source_refuses_new_process_before_startup_io() {
    assert_startup_refusal("A", "def broken(: PRIVATE_SOURCE");
}

#[test]
fn managed_unlisted_cached_module_refuses_new_process_before_startup_io() {
    assert_startup_refusal("Unlisted", "def value() -> int:\n    return 1\n");
}

#[test]
fn managed_unusable_regular_file_root_blocks_startup() {
    assert_startup_refusal_with_root("A", "def value() -> int:\n    return 1\n", "file");
}

#[test]
fn managed_unusable_inaccessible_root_blocks_startup() {
    assert_startup_refusal_with_root("A", "def value() -> int:\n    return 1\n", "inaccessible");
}

#[test]
fn managed_unusable_root_creation_failure_blocks_startup() {
    assert_startup_refusal_with_root("A", "def value() -> int:\n    return 1\n", "parent-file");
}

fn assert_startup_refusal(cached_module: &str, source: &str) {
    assert_startup_refusal_with_root(cached_module, source, "directory");
}

fn cached_startup(cached_module: &str, marker: &std::path::Path) -> serde_json::Value {
    serde_json::json!({
        "program": {
            "modules": [{
                "id":"cached-module", "name":cached_module, "types":[], "functions":[], "modules":[],
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
            "functions":[], "types":[]
        },
        "mutations":[], "history":[], "revision":0, "sources":[]
    })
}

fn assert_startup_refusal_with_root(cached_module: &str, source: &str, root_kind: &str) {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let modules = dir.path().join("modules");
    std::fs::create_dir(&modules).unwrap();
    let marker = dir.path().join("startup-ran");
    let memory = dir.path().join("memory.lbug");
    let cache = cached_startup(cached_module, &marker);
    let session = dir.path().join("session.json");
    let bytes = serde_json::to_vec(&cache).unwrap();
    std::fs::write(&session, &bytes).unwrap();
    std::fs::write(
        modules.join("library-sources.json"),
        r#"{"version":1,"modules":{"A":{"format":"python","dependencies":[]}}}"#,
    )
    .unwrap();
    std::fs::write(modules.join("A.py"), source).unwrap();
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
            if root_kind == "directory" {
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
            panic!("OS did not refuse broken authoritative source before initialization");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let output = child.wait_with_output().unwrap();
    if root_kind == "inaccessible" {
        std::fs::set_permissions(&modules, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    assert!(!output.status.success());
    let diagnostic = String::from_utf8_lossy(&output.stderr);
    assert!(diagnostic.contains("startup refused"), "{diagnostic}");
    assert!(!diagnostic.contains("PRIVATE_SOURCE"));
    if cached_module != "A" {
        assert!(diagnostic.contains(cached_module), "{diagnostic}");
    }
    assert!(!marker.exists());
    assert!(!memory.exists());
    assert_eq!(std::fs::read(&session).unwrap(), bytes);
}
