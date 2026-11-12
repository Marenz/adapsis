//! One nesting limit (2026-11-12): whatever the submission guard accepts (100
//! levels) must be storable, replayable, readable again and durable as managed
//! source. No layer behind the guard may have a lower limit of its own.
//!
//! Every scenario runs on a 2 MiB thread, the smallest stack any of these paths
//! has in the service (tokio workers and blocking threads; the main thread has
//! more). A stack overflow aborts the test binary, so "the test passed" is the
//! measurement.
use super::*;
use serde_json::json;

const SHAPES: [&str; 6] = ["operators", "calls", "parentheses", "not", "blocks", "elif"];
const CHILD: &str = "api::tests::storage_depth_tests::storage_depth_child_reads_the_session_in_a_fresh_process";

/// Run `scenario` on a 2 MiB thread with its own runtime, like a worker.
fn on_worker_stack<F, Fut>(scenario: F)
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()>,
{
    std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(scenario());
        })
        .unwrap()
        .join()
        .expect("the process survived and the scenario passed");
}

/// Function `f` of one shape whose nesting scans at exactly `n` levels (the
/// same generators as the #133 tests), written as a submission (bare calls).
fn shape(name: &str, n: usize) -> String {
    match name {
        "operators" => format!("def f(x: int) -> int:\n    return x{}\n", " + 1".repeat(n - 1)),
        "calls" => format!("def g(x: int) -> int:\n    return x + 1\n\ndef f(x: int) -> int:\n    return {}x{}\n", "g(".repeat(n - 1), ")".repeat(n - 1)),
        "parentheses" => format!("def f(x: int) -> int:\n    return {}x{}\n", "(".repeat(n - 1), ")".repeat(n - 1)),
        "not" => format!("def f(x: bool) -> bool:\n    return {}x\n", "not ".repeat(2 * ((n - 1) / 2))),
        "blocks" => {
            let mut source = String::from("def f(x: bool) -> int:\n");
            for level in 0..n - 1 {
                source.push_str(&format!("{}if x:\n", " ".repeat(level + 1)));
            }
            source.push_str(&format!("{}return 7\n return 0\n", " ".repeat(n)));
            source
        }
        "elif" => {
            let mut source = String::from("def f(x: int) -> int:\n    if x == 0:\n        return 0\n");
            for branch in 1..=n - 2 {
                source.push_str(&format!("    elif x == {branch}:\n        return {}\n", branch * 2));
            }
            source.push_str("    return -1\n");
            source
        }
        other => panic!("unknown shape {other}"),
    }
}

/// (argument, expected result as printed) for `shape(name, n)`.
fn sample(name: &str, n: usize) -> (String, String) {
    match name {
        "operators" | "calls" => ("1".into(), n.to_string()),
        "parentheses" => ("7".into(), "7".into()),
        "not" => ("True".into(), "true".into()),
        "blocks" => ("True".into(), "7".into()),
        _ => ((n - 2).to_string(), ((n - 2) * 2).to_string()),
    }
}

fn python_literal(printed: &str) -> String {
    match printed {
        "true" => "True".into(),
        "false" => "False".into(),
        other => other.into(),
    }
}

async fn tool(name: &str, arguments: serde_json::Value, config: &AppConfig, ws: &mut WorkingSet, baseline: &crate::ast::Program) -> crate::api::native_tools::NativeToolResult {
    crate::api::native_tools::execute_native_tool(name, &arguments.to_string(), config, ws, baseline).await
}

fn program_json(program: &crate::ast::Program) -> String {
    serde_json::to_string(program).unwrap()
}

fn stored_cases(program: &crate::ast::Program, function: &str) -> Vec<crate::parser::TestCase> {
    program.get_function(function).unwrap().tests.iter().map(|case| case.typed.as_ref().expect("typed case").case().unwrap()).collect()
}

/// (d): mutate, test, evaluate, store, replay, read in a fresh process, rewind,
/// save as managed source, load at startup. All six shapes at the limit.
#[test]
fn every_shape_at_the_submission_limit_is_stored_replayed_read_again_and_saved_as_source() {
    on_worker_stack(|| async {
        let limit = crate::python_frontend::MAX_NESTING;
        for name in SHAPES {
            let module = format!("Depth{}", &name[..1].to_uppercase()) + &name[1..];
            let function = format!("{module}.f");
            let (input, expected) = sample(name, limit);
            let config = test_config();
            let mut ws = config.snapshot_working_set().await;
            let baseline = ws.program.clone();

            // Admission and execution through the native tools.
            crate::session::replay::limits::DEEPEST_SEEN.with(|seen| seen.set(0));
            let result = tool("mutate", json!({"language":"python","module":module,"source":shape(name, limit)}), &config, &mut ws, &baseline).await;
            assert!(!result.execution.has_errors, "{name} mutate: {}", result.content.chars().take(300).collect::<String>());
            let test = format!("def test_f() -> None:\n    assert f({input}) == {}\n", python_literal(&expected));
            let result = tool("test", json!({"language":"python","module":module,"source":test}), &config, &mut ws, &baseline).await;
            assert!(!result.execution.has_errors && result.execution.test_results.iter().all(|t| t.pass), "{name} test: {}", result.content.chars().take(300).collect::<String>());
            let result = tool("evaluate", json!({"language":"python","function":function,"input":input}), &config, &mut ws, &baseline).await;
            assert!(!result.execution.has_errors && result.content.contains(&expected), "{name} evaluate: {}", result.content.chars().take(300).collect::<String>());
            // Inspection prints the stored function back, in both formats.
            for format in ["python", "adapsis"] {
                let result = tool("inspect", json!({"kind":"source","target":function,"format":format}), &config, &mut ws, &baseline).await;
                assert!(!result.execution.has_errors && result.content.contains("f"), "{name} inspect {format}: {}", result.content.chars().take(300).collect::<String>());
            }
            let record_depth = crate::session::replay::limits::DEEPEST_SEEN.with(|seen| seen.get());
            assert!(record_depth <= crate::session::replay::limits::MAX_JSON_DEPTH);

            // One level more is refused at admission, with the #133 text.
            let stored = ws.program.get_function(&function).map(|stored| stored as *const crate::ast::FunctionDecl);
            let result = tool("mutate", json!({"language":"python","module":module,"source":shape(name, limit + 1)}), &config, &mut ws, &baseline).await;
            assert!(result.execution.has_errors && result.content.contains("nesting too deep: more than 100 levels"), "{name}: {}", result.content.chars().take(300).collect::<String>());
            assert_eq!(stored, ws.program.get_function(&function).map(|stored| stored as *const crate::ast::FunctionDecl), "{name}: stored function untouched");

            // Session file: written here, read back here and in a fresh process.
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("session.json");
            let session = crate::session::Session { program: ws.program.clone(), runtime: ws.runtime.clone(), meta: ws.meta.clone(), sandbox: None };
            assert!(session.meta.sources.iter().any(|entry| matches!(entry, crate::session::replay::ReplayEntry::Python(_))), "{name}: typed replay record stored");
            session.save(&path).unwrap();
            let text = std::fs::read_to_string(&path).unwrap();
            let file_depth = crate::session::deep_read::json_depth(&text);
            assert!(file_depth <= crate::session::deep_read::MAX_DEPTH);
            let reloaded = crate::session::Session::load(&path).unwrap();
            assert_eq!(program_json(&reloaded.program), program_json(&session.program), "{name}");
            assert_eq!(reloaded.meta.sources.len(), session.meta.sources.len());
            let child = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--ignored", "--exact", CHILD, "--nocapture"])
                .env("ADAPSIS_TEST_DEPTH_SESSION", &path)
                .env("ADAPSIS_TEST_DEPTH_FUNCTION", &function)
                .env("HOME", dir.path())
                .current_dir(dir.path())
                .output()
                .unwrap();
            let output = String::from_utf8_lossy(&child.stdout).into_owned() + &String::from_utf8_lossy(&child.stderr);
            assert!(child.status.success() && output.contains("STORAGE_DEPTH_CHILD_OK"), "{name}: {}", output.chars().rev().take(600).collect::<String>().chars().rev().collect::<String>());

            // Managed source: the writer enrols the module, a start loads it.
            let library = tempfile::tempdir().unwrap();
            std::fs::write(library.path().join("library-sources.json"), r#"{"version":1,"modules":{}}"#).unwrap();
            let state = crate::library::load_library_from_dir_for_tests(&mut crate::ast::Program::default(), library.path());
            state.require_authoritative_sources().unwrap();
            let live = std::sync::Arc::new(tokio::sync::RwLock::new(ws.program.clone()));
            let printed = crate::python_frontend::printed_nesting_for_tests(&ws.program, ws.program.modules.iter().find(|m| m.name == module).unwrap()).unwrap();
            let receipt = crate::library::flush_live_sources_to_dir_for_tests(live, state, library.path()).await;
            assert!(receipt.failures.is_empty(), "{name}: {:?}", receipt.failures);
            assert_eq!(receipt.enrolled, vec![module.clone()], "{name}");
            let mut started = crate::ast::Program::default();
            let started_state = crate::library::load_library_from_dir_for_tests(&mut started, library.path());
            started_state.require_authoritative_sources().unwrap_or_else(|error| panic!("{name}: {error:#}"));
            let loaded = started.get_function(&function).unwrap();
            assert!(!loaded.tests.is_empty() && loaded.tests.iter().all(|case| case.passed), "{name}: stored case earned its flag at startup");
            let case = stored_cases(&started, &function).remove(0);
            let (value, _) = crate::eval::eval_compiled_or_interpreted_cached(&started, &function, &case.input, Some(&crate::eval::new_jit_cache()), 1).unwrap();
            assert_eq!(value, expected, "{name}: evaluated from the loaded source");
            // The session saved after that start is readable as well.
            let after = crate::session::Session { program: receipt.program_snapshot, runtime: Default::default(), meta: crate::session::SessionMeta::new(), sandbox: None };
            after.save(&path).unwrap();
            crate::session::Session::load(&path).unwrap();
            eprintln!("STORAGE-DEPTH {name} at {limit}: replay record json depth {record_depth}, session file json depth {file_depth} ({} bytes), printed source scans at {printed}", text.len());
        }
    });
}

#[test]
#[ignore = "invoked only by the parent test, in a fresh process"]
fn storage_depth_child_reads_the_session_in_a_fresh_process() {
    let path = std::path::PathBuf::from(std::env::var_os("ADAPSIS_TEST_DEPTH_SESSION").expect("parent supplies the session"));
    let function = std::env::var("ADAPSIS_TEST_DEPTH_FUNCTION").unwrap();
    // `Session::load` validates every typed replay record and stored case.
    let mut session = crate::session::Session::load(&path).unwrap();
    let cases = stored_cases(&session.program, &function);
    assert!(!cases.is_empty());
    for case in &cases {
        crate::eval::eval_test_case(&session.program, &function, case).unwrap();
    }
    let before = program_json(&session.program);
    let copy = session.program.clone();
    drop(copy);
    // Rewind rebuilds the Program from the replay records (no manifest here).
    let latest = session.meta.sources.len();
    session.rewind_to(latest).unwrap();
    assert!(session.program.get_function(&function).is_some());
    session.rewind_to(1).unwrap();
    let rebuilt = session.program.get_function(&function).expect("revision 1 holds the function");
    assert!(before.contains(&format!("\"name\":\"{}\"", rebuilt.name)));
    for case in &cases {
        crate::eval::eval_test_case(&session.program, &function, case).unwrap();
    }
    println!("STORAGE_DEPTH_CHILD_OK");
}

/// A managed module written by hand at REAL depth 250 (the stored-source limit):
/// qualified calls, a stored case and a route handler.
fn stored_file(name: &str, module: &str, n: usize) -> (String, String, String) {
    let (input, expected) = sample(name, n);
    let function = shape(name, n).replace("g(", &format!("{module}.g(")).replace(&format!("def {module}.g("), "def g(");
    let handler = if name == "not" {
        format!("def handle(body: str) -> str:\n    if {module}.f(True):\n        return \"true\"\n    return \"false\"\n")
    } else {
        format!("def handle(body: str) -> str:\n    return to_string({module}.f({input}))\n")
    };
    let source = format!(
        "{function}\n@route(\"POST\", \"/deep\")\n{handler}\n@stored_test(\"{module}.f\")\ndef test_case_0() -> None:\n    assert {module}.f({input}) == {}\n",
        python_literal(&expected)
    );
    (source, input, expected)
}

fn write_library(module: &str, source: &str) -> tempfile::TempDir {
    let library = tempfile::tempdir().unwrap();
    std::fs::write(library.path().join("library-sources.json"), format!(r#"{{"version":1,"modules":{{"{module}":{{"format":"python","dependencies":[]}}}}}}"#)).unwrap();
    std::fs::write(library.path().join(format!("{module}.py")), source).unwrap();
    library
}

fn start(library: &std::path::Path) -> anyhow::Result<(crate::ast::Program, crate::library::LibraryState)> {
    let mut program = crate::ast::Program::default();
    let state = crate::library::load_library_from_dir_for_tests(&mut program, library);
    state.require_authoritative_sources()?;
    Ok((program, state))
}

/// The factor between the stored-source limit and the smallest overflow must
/// be shown, not assumed. For every shape: the deepest hand-written managed
/// file a start accepts, through every evaluation path of this branch, on a
/// 2 MiB stack; and one level more, refused cleanly.
///
/// A start also PRINTS every module it loads (the same check a save does), so
/// a file is accepted only if its printed form scans within the stored limit:
/// block shapes reach real depth 250, expression shapes about 124.
#[test]
fn hand_edited_managed_source_at_the_stored_limit_runs_on_every_path_without_overflow() {
    on_worker_stack(|| async {
        let limit = crate::python_frontend::MAX_STORED_NESTING;
        for name in SHAPES {
            let module = format!("Stored{}", &name[..1].to_uppercase()) + &name[1..];
            let function = format!("{module}.f");
            // Deepest file a start accepts; everything above it up to the
            // limit (and one beyond) is refused before anything runs.
            let mut deepest = 0;
            for n in (3..=limit + 1).rev() {
                let library = write_library(&module, &stored_file(name, &module, n).0);
                match start(library.path()) {
                    Ok(_) => {
                        deepest = n;
                        break;
                    }
                    Err(error) => assert!(format!("{error:#}").contains("failed module/test validation"), "{name} at {n}: {error:#}"),
                }
            }
            assert!(deepest <= limit, "{name}: {deepest} loaded beyond the stored limit");
            let expected_depth = match name {
                "blocks" | "parentheses" => limit,
                "elif" => limit - 1,
                // Printed with parentheses / qualified: about half.
                _ => limit / 2,
            };
            assert!(deepest.abs_diff(expected_depth) <= 3, "{name}: deepest accepted file has {deepest} levels, expected about {expected_depth}");
            let (source, _input, expected) = stored_file(name, &module, deepest);
            let library = write_library(&module, &source);

            // Startup load, including the mock-only re-run of the stored case
            // (tree-walker).
            let (mut program, state) = start(library.path()).unwrap();
            let loaded = program.get_function(&function).unwrap();
            assert!(loaded.tests.iter().all(|case| case.passed) && loaded.tests.len() == 1, "{name}: stored case re-run at startup");

            // Tree-walker directly (the tier spawned tasks and tests use).
            let case = stored_cases(&program, &function).remove(0);
            crate::eval::eval_test_case(&program, &function, &case).unwrap_or_else(|error| panic!("{name}: {error:#}"));
            // JIT admission and compile, VM admission and execution, tree fallback.
            let (value, jit) = crate::eval::eval_compiled_or_interpreted_cached(&program, &function, &case.input, Some(&crate::eval::new_jit_cache()), 1)
                .unwrap_or_else(|error| panic!("{name}: {error:#}"));
            assert_eq!(value, expected, "{name}");
            let vm = matches!(crate::eval::try_vm_execute_for_tests(&program, &function, &case.input), Some(Ok(_)));

            // Route handler: the real router; the handler runs on a tokio
            // blocking thread (2 MiB).
            let config = test_config();
            let routes = std::mem::take(&mut program.pending_routes);
            assert_eq!(routes.len(), 1);
            config.runtime.write().unwrap().http_routes.extend(routes);
            *config.program.write().await = program.clone();
            let response = router_with_llm(config.clone())
                .oneshot(axum::http::Request::builder().method("POST").uri("/deep").body(axum::body::Body::from("x")).unwrap())
                .await
                .unwrap();
            let status = response.status();
            let body = String::from_utf8(axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap().to_vec()).unwrap();
            assert_eq!((status, body.as_str()), (axum::http::StatusCode::OK, expected.as_str()), "{name}");

            // Queries, clone, drop, session write and read.
            let table = crate::typeck::build_symbol_table(&program);
            let listing = crate::typeck::handle_query(&program, &table, &format!("?source {function}"), &[]);
            assert!(listing.contains("+fn f"), "{name}: {}", listing.chars().take(200).collect::<String>());
            drop(table);
            let copy = program.clone();
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("session.json");
            let session = crate::session::Session { program: copy, runtime: Default::default(), meta: crate::session::SessionMeta::new(), sandbox: None };
            session.save(&path).unwrap();
            let file_depth = crate::session::deep_read::json_depth(&std::fs::read_to_string(&path).unwrap());
            let reloaded = crate::session::Session::load(&path).unwrap();
            assert_eq!(program_json(&reloaded.program), program_json(&session.program), "{name}");
            assert!(file_depth <= crate::session::replay::limits::MAX_JSON_DEPTH, "{name}: {file_depth} must also fit the typed-record budget");
            drop(reloaded);
            drop(session);

            // A later Python edit of this module passes the typed-record budget
            // (it budgets the whole module before and after).
            *config.program.write().await = program.clone();
            let mut ws = config.snapshot_working_set().await;
            let baseline = ws.program.clone();
            let result = tool("mutate", json!({"language":"python","module":module,"source":"def extra() -> int:\n    return 1\n"}), &config, &mut ws, &baseline).await;
            assert!(!result.execution.has_errors, "{name}: {}", result.content.chars().take(300).collect::<String>());

            // The writer saves what a start accepted (a start prints it too),
            // and the next start loads the result.
            let live = std::sync::Arc::new(tokio::sync::RwLock::new(ws.program.clone()));
            let receipt = crate::library::flush_live_sources_to_dir_for_tests(live, state, library.path()).await;
            assert!(receipt.failures.is_empty(), "{name}: {:?}", receipt.failures);
            let (restarted, _) = start(library.path()).unwrap();
            assert!(restarted.get_function(&format!("{module}.extra")).is_some() && restarted.get_function(&function).is_some());
            eprintln!("STORED-DEPTH {name}: deepest managed file a start accepts has {deepest} levels; load, stored-case re-run, tree-walker, dispatch (jit={jit}, vm={vm}), route handler (200), Python edit, source save and restart all ran on 2 MiB; session json depth {file_depth}");
        }
    });
}

/// A submission arriving on the same thread right after a managed load or a
/// managed save is still limited to 100.
#[test]
fn stored_source_limit_does_not_reach_submissions_after_a_managed_load_on_the_same_thread() {
    on_worker_stack(|| async {
        // 120 levels: more than a submission may have, and its printed form
        // (241) still fits the stored limit.
        let (source, _, _) = stored_file("operators", "Scope", 120);
        let library = write_library("Scope", &source);
        let config = test_config();
        let submission = shape("operators", 101);
        let refused = |result: crate::api::native_tools::NativeToolResult| {
            assert!(result.execution.has_errors && result.content.contains("nesting too deep: more than 100 levels"), "{}", result.content.chars().take(300).collect::<String>());
        };
        for _ in 0..2 {
            assert_eq!(crate::python_frontend::nesting_limit_for_tests(), 100);
            let mut program = crate::ast::Program::default();
            let state = crate::library::load_library_from_dir_for_tests(&mut program, library.path());
            state.require_authoritative_sources().unwrap();
            assert!(program.get_function("Scope.f").is_some(), "120 levels load as stored source");
            assert_eq!(crate::python_frontend::nesting_limit_for_tests(), 100);
            *config.program.write().await = program.clone();
            let mut ws = config.snapshot_working_set().await;
            let baseline = ws.program.clone();
            refused(tool("mutate", json!({"language":"python","module":"Scope","source":submission}), &config, &mut ws, &baseline).await);
            refused(tool("mutate", json!({"language":"python","module":"ScopeNew","source":submission}), &config, &mut ws, &baseline).await);
            refused(tool("test", json!({"language":"python","module":"Scope","source":format!("def test_f() -> None:\n    assert f(1) == 1{}\n", " + 1".repeat(101))}), &config, &mut ws, &baseline).await);
            refused(tool("evaluate", json!({"language":"python","function":"Scope.f","input":format!("1{}", " + 1".repeat(101))}), &config, &mut ws, &baseline).await);
            // The same 120-level text as a submission: refused.
            refused(tool("mutate", json!({"language":"python","module":"Scope","source":shape("operators", 120)}), &config, &mut ws, &baseline).await);
            // Source inspection prints with the stored limit; the next
            // submission is still limited to 100.
            let inspected = tool("inspect", json!({"kind":"source","target":"Scope.f","format":"python"}), &config, &mut ws, &baseline).await;
            assert!(!inspected.execution.has_errors, "{}", inspected.content.chars().take(300).collect::<String>());
            assert_eq!(crate::python_frontend::nesting_limit_for_tests(), 100);
            refused(tool("mutate", json!({"language":"python","module":"Scope","source":submission}), &config, &mut ws, &baseline).await);
            // A managed save on this runtime (the print round trip uses the
            // stored limit), then a submission again.
            let live = std::sync::Arc::new(tokio::sync::RwLock::new(program));
            let receipt = crate::library::flush_live_sources_to_dir_for_tests(live, state, library.path()).await;
            assert!(receipt.failures.is_empty(), "{:?}", receipt.failures);
            assert_eq!(crate::python_frontend::nesting_limit_for_tests(), 100);
            refused(tool("mutate", json!({"language":"python","module":"Scope","source":submission}), &config, &mut ws, &baseline).await);
        }
    });
}

/// Regressions of the two reproduced paths around the old limits: both left a
/// session file the next start could not read.
#[test]
fn deep_declarations_from_adapsis_source_and_from_managed_files_leave_a_readable_session() {
    on_worker_stack(|| async {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.json");
        // (1) An Adapsis-source mutation with 70 chained operators.
        let chain = " + 1".repeat(70);
        let mut session = crate::session::Session::new();
        let results = session.apply_async(&format!("+module DeepAx\n+fn f (x:Int)->Int\n  +return x{chain}\n+end\n"), None).await.unwrap();
        assert!(results.iter().all(|(_, ok)| *ok), "{results:?}");
        session.save(&path).unwrap();
        let depth = crate::session::deep_read::json_depth(&std::fs::read_to_string(&path).unwrap());
        assert!(depth > 127, "deeper than serde_json's default limit: {depth}");
        let reloaded = crate::session::Session::load(&path).unwrap();
        assert_eq!(program_json(&reloaded.program), program_json(&session.program));
        // (2) A managed file with a 40-branch `elif` chain, loaded at startup,
        // then the session the autosave writes.
        let (source, _, _) = stored_file("elif", "DeepElif", 42);
        let library = write_library("DeepElif", &source);
        let mut program = crate::ast::Program::default();
        let state = crate::library::load_library_from_dir_for_tests(&mut program, library.path());
        state.require_authoritative_sources().unwrap();
        let live = std::sync::Arc::new(tokio::sync::RwLock::new(program));
        let receipt = crate::library::flush_live_sources_to_dir_for_tests(live, state, library.path()).await;
        assert!(receipt.failures.is_empty(), "{:?}", receipt.failures);
        let session = crate::session::Session { program: receipt.program_snapshot, runtime: Default::default(), meta: crate::session::SessionMeta::new(), sandbox: None };
        session.save(&path).unwrap();
        let depth = crate::session::deep_read::json_depth(&std::fs::read_to_string(&path).unwrap());
        assert!(depth > 127, "{depth}");
        let reloaded = crate::session::Session::load(&path).unwrap();
        assert!(reloaded.program.get_function("DeepElif.f").is_some());
    });
}

/// The JIT compiles top-level functions only, so no managed (module) function
/// reaches its code generator. A legacy session can hold top-level functions:
/// the same depths through JIT admission, compilation and execution.
#[test]
fn jit_tier_handles_top_level_functions_at_the_stored_depth() {
    on_worker_stack(|| async {
        let depth = crate::python_frontend::MAX_STORED_NESTING;
        let mut blocks = String::from("+fn deep_blocks (x:Int)->Int\n");
        for level in 1..depth {
            blocks.push_str(&format!("{}+if x > 0\n", " ".repeat(level)));
        }
        blocks.push_str(&format!("{}+return 7\n", " ".repeat(depth)));
        for level in (1..depth).rev() {
            blocks.push_str(&format!("{}+end\n", " ".repeat(level)));
        }
        blocks.push_str(" +return 0\n+end\n");
        let source = format!("+fn deep_operators (x:Int)->Int\n  +return x{}\n+end\n{blocks}", " + 1".repeat(depth - 1));
        let mut program = crate::ast::Program::default();
        for operation in crate::parser::parse(&source).unwrap() {
            crate::validator::apply_and_validate(&mut program, &operation).unwrap();
        }
        let cache = crate::eval::new_jit_cache();
        let input = crate::parser::parse_expr_pub(1, "1").unwrap();
        let mut tiers = Vec::new();
        for (function, expected) in [("deep_operators", depth.to_string()), ("deep_blocks", "7".to_string())] {
            let (value, jit) = crate::eval::eval_compiled_or_interpreted_cached(&program, function, &input, Some(&cache), 1).unwrap();
            assert_eq!(value, expected, "{function}");
            tiers.push(format!("{function} jit={jit}"));
        }
        eprintln!("STORED-DEPTH jit at {depth}: {}", tiers.join(", "));
        assert!(tiers[0].ends_with("jit=true"), "the operator chain is compiled natively: {tiers:?}");
    });
}
