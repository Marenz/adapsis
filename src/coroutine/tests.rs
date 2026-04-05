use super::*;

/// Helper: set up SharedMeta and SharedRuntime with empty roadmap/plan
/// and install them as thread-locals, returning the handle and meta for assertions.
fn setup_roadmap_runtime() -> (CoroutineHandle, crate::session::SharedMeta) {
    let rt = std::sync::Arc::new(std::sync::RwLock::new(
        crate::session::RuntimeState::default(),
    ));
    let mut initial_meta = crate::session::SessionMeta::new();
    // Seed a successful mutation so roadmap_done_checked passes
    initial_meta.mutations.push(crate::session::MutationEntry {
        revision: 1,
        timestamp: String::new(),
        source: String::new(),
        summary: "test setup".to_string(),
        success: true,
    });
    let meta = std::sync::Arc::new(std::sync::Mutex::new(initial_meta));
    crate::shared_state::set_shared_runtime(Some(rt.clone()));
    crate::shared_state::set_shared_meta(Some(meta.clone()));
    let handle = CoroutineHandle::new_mock(vec![]);
    (handle, meta)
}

fn unwrap_string(v: Value) -> String {
    match v {
        Value::String(s) => s.as_ref().clone(),
        other => panic!("expected String, got {other}"),
    }
}

#[test]
fn roadmap_list_empty() {
    let (handle, _rt) = setup_roadmap_runtime();
    let result = unwrap_string(handle.execute_await("roadmap_list", &[]).unwrap());
    assert_eq!(result, "Roadmap is empty.");
}

#[test]
fn roadmap_add_and_list() {
    let (handle, meta) = setup_roadmap_runtime();

    // Add an item
    let result = unwrap_string(
        handle
            .execute_await("roadmap_add", &[Value::string("Build feature X")])
            .unwrap(),
    );
    assert_eq!(result, "Build feature X");

    // Verify it's in the meta state
    assert_eq!(meta.lock().unwrap().roadmap.len(), 1);
    assert_eq!(
        meta.lock().unwrap().roadmap[0].description,
        "Build feature X"
    );
    assert!(!meta.lock().unwrap().roadmap[0].done);

    // List should show it
    let list = unwrap_string(handle.execute_await("roadmap_list", &[]).unwrap());
    assert!(
        list.contains("Build feature X"),
        "list should contain the item: {list}"
    );
    assert!(list.contains("[ ] 1:"), "item should be unchecked: {list}");
}

#[test]
fn roadmap_add_empty_fails() {
    let (handle, _rt) = setup_roadmap_runtime();
    let result = handle.execute_await("roadmap_add", &[Value::string("  ")]);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("must not be empty"),
        "should reject empty description"
    );
}

#[test]
fn roadmap_add_no_args_fails() {
    let (handle, _rt) = setup_roadmap_runtime();
    let result = handle.execute_await("roadmap_add", &[]);
    assert!(result.is_err());
}

#[test]
fn roadmap_done_marks_item() {
    let (handle, meta) = setup_roadmap_runtime();

    // Add two items
    handle
        .execute_await("roadmap_add", &[Value::string("Item A")])
        .unwrap();
    handle
        .execute_await("roadmap_add", &[Value::string("Item B")])
        .unwrap();

    // Mark item 2 as done
    let result = unwrap_string(
        handle
            .execute_await("roadmap_done", &[Value::Int(2)])
            .unwrap(),
    );
    assert!(result.contains("#2 done"), "confirmation: {result}");

    // Verify state
    assert!(!meta.lock().unwrap().roadmap[0].done);
    assert!(meta.lock().unwrap().roadmap[1].done);

    // List should show [x] for item 2
    let list = unwrap_string(handle.execute_await("roadmap_list", &[]).unwrap());
    assert!(list.contains("[ ] 1: Item A"), "A unchecked: {list}");
    assert!(list.contains("[x] 2: Item B"), "B checked: {list}");
}

#[test]
fn roadmap_done_out_of_bounds() {
    let (handle, _rt) = setup_roadmap_runtime();
    handle
        .execute_await("roadmap_add", &[Value::string("Only item")])
        .unwrap();

    let result = handle.execute_await("roadmap_done", &[Value::Int(5)]);
    assert!(result.is_err());
    assert!(
        result.unwrap_err().to_string().contains("not found"),
        "should error on out-of-bounds index"
    );
}

#[test]
fn roadmap_done_zero_index_fails() {
    let (handle, _rt) = setup_roadmap_runtime();
    let result = handle.execute_await("roadmap_done", &[Value::Int(0)]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains(">= 1"));
}

#[test]
fn roadmap_done_wrong_type_fails() {
    let (handle, _rt) = setup_roadmap_runtime();
    let result = handle.execute_await("roadmap_done", &[Value::string("1")]);
    assert!(result.is_err());
}

#[test]
fn roadmap_done_warns_without_mutations() {
    // Fresh meta with NO mutations — roadmap_done should succeed but warn
    let rt = std::sync::Arc::new(std::sync::RwLock::new(
        crate::session::RuntimeState::default(),
    ));
    let meta = std::sync::Arc::new(std::sync::Mutex::new(crate::session::SessionMeta::new()));
    crate::shared_state::set_shared_runtime(Some(rt));
    crate::shared_state::set_shared_meta(Some(meta.clone()));
    let handle = CoroutineHandle::new_mock(vec![]);

    handle
        .execute_await("roadmap_add", &[Value::string("Suspicious item")])
        .unwrap();
    let result = unwrap_string(
        handle
            .execute_await("roadmap_done", &[Value::Int(1)])
            .unwrap(),
    );
    assert!(
        result.contains("WARNING"),
        "should warn without mutations: {result}"
    );
    assert!(
        result.contains("#1 done"),
        "should still mark done: {result}"
    );
    assert!(
        meta.lock().unwrap().roadmap[0].done,
        "item should be marked done"
    );
}

// ═════════════════════════════════════════════════════════════════════
// Plan IO builtins
// ═════════════════════════════════════════════════════════════════════

#[test]
fn plan_show_empty() {
    let (handle, _rt) = setup_roadmap_runtime();
    let result = unwrap_string(handle.execute_await("plan_show", &[]).unwrap());
    assert_eq!(result, "No plan set.");
}

#[test]
fn plan_set_and_show() {
    let (handle, meta) = setup_roadmap_runtime();

    let result = unwrap_string(
        handle
            .execute_await(
                "plan_set",
                &[Value::string("Parse input\nValidate data\nStore results")],
            )
            .unwrap(),
    );
    assert_eq!(result, "Plan set with 3 steps.");

    // Verify state
    assert_eq!(meta.lock().unwrap().plan.len(), 3);
    assert_eq!(meta.lock().unwrap().plan[0].description, "Parse input");
    assert_eq!(meta.lock().unwrap().plan[1].description, "Validate data");
    assert_eq!(meta.lock().unwrap().plan[2].description, "Store results");

    // Show should list all steps
    let show = unwrap_string(handle.execute_await("plan_show", &[]).unwrap());
    assert!(show.contains("[ ] 1: Parse input"), "step 1: {show}");
    assert!(show.contains("[ ] 2: Validate data"), "step 2: {show}");
    assert!(show.contains("[ ] 3: Store results"), "step 3: {show}");
}

#[test]
fn plan_set_empty_fails() {
    let (handle, _rt) = setup_roadmap_runtime();
    let result = handle.execute_await("plan_set", &[Value::string("  \n  \n  ")]);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("must not be empty"),
        "should reject empty steps"
    );
}

#[test]
fn plan_set_no_args_fails() {
    let (handle, _rt) = setup_roadmap_runtime();
    let result = handle.execute_await("plan_set", &[]);
    assert!(result.is_err());
}

#[test]
fn plan_set_skips_blank_lines() {
    let (handle, meta) = setup_roadmap_runtime();

    let result = unwrap_string(
        handle
            .execute_await("plan_set", &[Value::string("Step A\n\n  \nStep B")])
            .unwrap(),
    );
    assert_eq!(result, "Plan set with 2 steps.");
    assert_eq!(meta.lock().unwrap().plan.len(), 2);
    assert_eq!(meta.lock().unwrap().plan[0].description, "Step A");
    assert_eq!(meta.lock().unwrap().plan[1].description, "Step B");
}

#[test]
fn plan_done_marks_step() {
    let (handle, meta) = setup_roadmap_runtime();

    handle
        .execute_await("plan_set", &[Value::string("Alpha\nBravo")])
        .unwrap();

    let result = unwrap_string(handle.execute_await("plan_done", &[Value::Int(2)]).unwrap());
    assert_eq!(result, "Plan: step 2 done.");

    // Verify state
    assert_eq!(
        meta.lock().unwrap().plan[0].status,
        crate::session::PlanStatus::Pending
    );
    assert_eq!(
        meta.lock().unwrap().plan[1].status,
        crate::session::PlanStatus::Done
    );

    // Show should reflect [x]
    let show = unwrap_string(handle.execute_await("plan_show", &[]).unwrap());
    assert!(show.contains("[ ] 1: Alpha"), "Alpha pending: {show}");
    assert!(show.contains("[x] 2: Bravo"), "Bravo done: {show}");
}

#[test]
fn plan_done_out_of_bounds() {
    let (handle, _rt) = setup_roadmap_runtime();
    handle
        .execute_await("plan_set", &[Value::string("Only step")])
        .unwrap();

    let result = handle.execute_await("plan_done", &[Value::Int(5)]);
    assert!(result.is_err());
    assert!(
        result.unwrap_err().to_string().contains("not found"),
        "should error on out-of-bounds index"
    );
}

#[test]
fn plan_done_zero_index_fails() {
    let (handle, _rt) = setup_roadmap_runtime();
    let result = handle.execute_await("plan_done", &[Value::Int(0)]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains(">= 1"));
}

#[test]
fn plan_done_wrong_type_fails() {
    let (handle, _rt) = setup_roadmap_runtime();
    let result = handle.execute_await("plan_done", &[Value::string("1")]);
    assert!(result.is_err());
}

#[test]
fn plan_fail_marks_step() {
    let (handle, meta) = setup_roadmap_runtime();

    handle
        .execute_await("plan_set", &[Value::string("First\nSecond\nThird")])
        .unwrap();

    let result = unwrap_string(handle.execute_await("plan_fail", &[Value::Int(1)]).unwrap());
    assert_eq!(result, "Plan: step 1 failed.");

    // Verify state
    assert_eq!(
        meta.lock().unwrap().plan[0].status,
        crate::session::PlanStatus::Failed
    );
    assert_eq!(
        meta.lock().unwrap().plan[1].status,
        crate::session::PlanStatus::Pending
    );

    // Show should reflect [!]
    let show = unwrap_string(handle.execute_await("plan_show", &[]).unwrap());
    assert!(show.contains("[!] 1: First"), "First failed: {show}");
    assert!(show.contains("[ ] 2: Second"), "Second pending: {show}");
}

#[test]
fn plan_fail_out_of_bounds() {
    let (handle, _rt) = setup_roadmap_runtime();
    handle
        .execute_await("plan_set", &[Value::string("Only step")])
        .unwrap();

    let result = handle.execute_await("plan_fail", &[Value::Int(3)]);
    assert!(result.is_err());
    assert!(
        result.unwrap_err().to_string().contains("not found"),
        "should error on out-of-bounds index"
    );
}

#[test]
fn plan_fail_zero_index_fails() {
    let (handle, _rt) = setup_roadmap_runtime();
    let result = handle.execute_await("plan_fail", &[Value::Int(0)]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains(">= 1"));
}

#[test]
fn plan_fail_wrong_type_fails() {
    let (handle, _rt) = setup_roadmap_runtime();
    let result = handle.execute_await("plan_fail", &[Value::string("1")]);
    assert!(result.is_err());
}

#[test]
fn plan_set_replaces_existing() {
    let (handle, meta) = setup_roadmap_runtime();

    handle
        .execute_await("plan_set", &[Value::string("Old step 1\nOld step 2")])
        .unwrap();
    assert_eq!(meta.lock().unwrap().plan.len(), 2);

    handle
        .execute_await("plan_set", &[Value::string("New step")])
        .unwrap();
    assert_eq!(meta.lock().unwrap().plan.len(), 1);
    assert_eq!(meta.lock().unwrap().plan[0].description, "New step");
}

// ═════════════════════════════════════════════════════════════════════
// Query IO builtins
// ═════════════════════════════════════════════════════════════════════

/// Helper: build a program from Adapsis source and install it as the thread-local.
fn setup_query_runtime(source: &str) -> CoroutineHandle {
    let ops = crate::parser::parse(source).expect("parse failed");
    let mut program = crate::ast::Program::default();
    for op in &ops {
        match op {
            crate::parser::Operation::Test(_) | crate::parser::Operation::Eval(_) => {}
            _ => {
                crate::validator::apply_and_validate(&mut program, op).expect("validation failed");
            }
        }
    }
    program.rebuild_function_index();
    crate::shared_state::set_shared_program(Some(std::sync::Arc::new(program)));

    // Also set up a runtime for query_routes/query_tasks
    let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState {
        http_routes: vec![],
        shared_vars: std::collections::HashMap::new(),
        ..Default::default()
    }));
    crate::shared_state::set_shared_runtime(Some(rt));

    CoroutineHandle::new_mock(vec![])
}

#[test]
fn query_symbols_empty_program() {
    let handle = setup_query_runtime("");
    let result = unwrap_string(handle.execute_await("query_symbols", &[]).unwrap());
    assert!(
        result.contains("Types:"),
        "should contain Types header: {result}"
    );
    assert!(
        result.contains("Functions:"),
        "should contain Functions header: {result}"
    );
}

#[test]
fn query_symbols_with_function() {
    let handle = setup_query_runtime(
        "+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end",
    );
    let result = unwrap_string(handle.execute_await("query_symbols", &[]).unwrap());
    assert!(
        result.contains("greet"),
        "should list greet function: {result}"
    );
}

#[test]
fn query_symbols_detail_existing() {
    let handle = setup_query_runtime(
        "+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end",
    );
    let result = unwrap_string(
        handle
            .execute_await("query_symbols_detail", &[Value::string("greet")])
            .unwrap(),
    );
    assert!(
        result.contains("greet"),
        "should show greet details: {result}"
    );
    assert!(
        result.contains("params") || result.contains("String"),
        "should show parameter info: {result}"
    );
}

#[test]
fn query_symbols_detail_not_found() {
    let handle = setup_query_runtime("");
    let result = unwrap_string(
        handle
            .execute_await("query_symbols_detail", &[Value::string("nonexistent")])
            .unwrap(),
    );
    assert!(
        result.contains("not found"),
        "should say not found: {result}"
    );
}

#[test]
fn query_symbols_detail_wrong_type_fails() {
    let handle = setup_query_runtime("");
    let result = handle.execute_await("query_symbols_detail", &[Value::Int(42)]);
    assert!(result.is_err(), "should fail with non-String arg");
}

#[test]
fn query_source_existing_function() {
    let handle = setup_query_runtime(
        "+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end",
    );
    let result = unwrap_string(
        handle
            .execute_await("query_source", &[Value::string("greet")])
            .unwrap(),
    );
    assert!(
        result.contains("+fn greet"),
        "should contain function definition: {result}"
    );
    assert!(
        result.contains("concat"),
        "should contain function body: {result}"
    );
}

#[test]
fn query_source_not_found() {
    let handle = setup_query_runtime("");
    let result = unwrap_string(
        handle
            .execute_await("query_source", &[Value::string("missing")])
            .unwrap(),
    );
    assert!(
        result.contains("not found"),
        "should say not found: {result}"
    );
}

#[test]
fn query_source_wrong_type_fails() {
    let handle = setup_query_runtime("");
    let result = handle.execute_await("query_source", &[Value::Int(1)]);
    assert!(result.is_err(), "should fail with non-String arg");
}

#[test]
fn query_callers_no_callers() {
    let handle = setup_query_runtime(
        "+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end",
    );
    let result = unwrap_string(
        handle
            .execute_await("query_callers", &[Value::string("greet")])
            .unwrap(),
    );
    assert!(
        result.contains("no callers"),
        "should say no callers: {result}"
    );
}

#[test]
fn query_callers_with_caller() {
    let handle = setup_query_runtime(
        "+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end\n\
             +fn main ()->String\n  +let x:String = greet(\"world\")\n  +return x\n+end",
    );
    let result = unwrap_string(
        handle
            .execute_await("query_callers", &[Value::string("greet")])
            .unwrap(),
    );
    assert!(
        result.contains("main"),
        "should list main as caller: {result}"
    );
}

#[test]
fn query_callees_lists_calls() {
    let handle = setup_query_runtime(
        "+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end\n\
             +fn main ()->String\n  +let x:String = greet(\"world\")\n  +return x\n+end",
    );
    let result = unwrap_string(
        handle
            .execute_await("query_callees", &[Value::string("main")])
            .unwrap(),
    );
    assert!(
        result.contains("greet"),
        "should list greet as callee: {result}"
    );
}

#[test]
fn query_callees_wrong_type_fails() {
    let handle = setup_query_runtime("");
    let result = handle.execute_await("query_callees", &[Value::Int(1)]);
    assert!(result.is_err(), "should fail with non-String arg");
}

#[test]
fn query_deps_same_as_callees() {
    let handle = setup_query_runtime(
        "+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end\n\
             +fn main ()->String\n  +let x:String = greet(\"world\")\n  +return x\n+end",
    );
    let result = unwrap_string(
        handle
            .execute_await("query_deps", &[Value::string("main")])
            .unwrap(),
    );
    assert!(
        result.contains("greet"),
        "should list greet as dependency: {result}"
    );
}

#[test]
fn query_deps_all_transitive() {
    let handle = setup_query_runtime(
        "+fn a ()->String\n  +return \"hello\"\n+end\n\
             +fn b ()->String\n  +let x:String = a()\n  +return x\n+end\n\
             +fn c ()->String\n  +let x:String = b()\n  +return x\n+end",
    );
    let result = unwrap_string(
        handle
            .execute_await("query_deps_all", &[Value::string("c")])
            .unwrap(),
    );
    // c -> b -> a, so both a and b should appear
    assert!(
        result.contains("a"),
        "should include transitive dep 'a': {result}"
    );
    assert!(
        result.contains("b"),
        "should include direct dep 'b': {result}"
    );
}

#[test]
fn query_deps_all_no_deps() {
    let handle = setup_query_runtime("+fn a ()->String\n  +return \"hello\"\n+end");
    let result = unwrap_string(
        handle
            .execute_await("query_deps_all", &[Value::string("a")])
            .unwrap(),
    );
    assert!(
        result.contains("no dependencies"),
        "should say no dependencies: {result}"
    );
}

#[test]
fn query_routes_empty() {
    let handle = setup_query_runtime("");
    let result = unwrap_string(handle.execute_await("query_routes", &[]).unwrap());
    assert!(
        result.contains("No HTTP routes"),
        "should say no routes: {result}"
    );
}

#[test]
fn query_routes_with_routes() {
    // Set up runtime with routes
    let program = crate::ast::Program::default();
    crate::shared_state::set_shared_program(Some(std::sync::Arc::new(program)));
    let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState {
        http_routes: vec![crate::ast::HttpRoute {
            method: "GET".to_string(),
            path: "/health".to_string(),
            handler_fn: "health_check".to_string(),
        }],
        shared_vars: std::collections::HashMap::new(),
        ..Default::default()
    }));
    crate::shared_state::set_shared_runtime(Some(rt));
    let handle = CoroutineHandle::new_mock(vec![]);

    let result = unwrap_string(handle.execute_await("query_routes", &[]).unwrap());
    assert!(
        result.contains("GET"),
        "should contain GET method: {result}"
    );
    assert!(
        result.contains("/health"),
        "should contain /health path: {result}"
    );
    assert!(
        result.contains("health_check"),
        "should contain handler: {result}"
    );
}

#[test]
fn query_tasks_no_registry() {
    let handle = setup_query_runtime("");
    let result = unwrap_string(handle.execute_await("query_tasks", &[]).unwrap());
    // Mock handles don't have a task registry
    assert!(
        result.contains("No task registry") || result.contains("No tasks"),
        "should indicate no tasks available: {result}"
    );
}

#[test]
fn query_tasks_with_registry() {
    let registry: TaskRegistry = std::sync::Arc::new(std::sync::Mutex::new(HashMap::new()));
    // Add a task
    registry.lock().unwrap().insert(
        1,
        TaskInfo {
            id: 1,
            function_name: "my_task".to_string(),
            status: WaitReason::Running,
            started_at: "2025-01-01T00:00:00Z".to_string(),
        },
    );

    let (tx, _) = mpsc::channel(1);
    let handle = CoroutineHandle {
        io_tx: tx,
        task_id: None,
        task_registry: Some(registry),
        snapshot_registry: None,
        mocks: Some(vec![]),
        stubs: None,
    };

    let result = unwrap_string(handle.execute_await("query_tasks", &[]).unwrap());
    assert!(result.contains("my_task"), "should show task: {result}");
    assert!(result.contains("running"), "should show status: {result}");
}

#[test]
fn query_inbox_matches_query_output() {
    let (handle, meta) = setup_roadmap_runtime();
    {
        let mut meta = meta.lock().unwrap();
        crate::session::send_agent_message(&mut meta, "agent1", "main", "hello");
        crate::session::send_agent_message(&mut meta, "agent2", "main", "status update");
    }

    let result = unwrap_string(handle.execute_await("query_inbox", &[]).unwrap());
    assert!(result.contains("from agent1: hello"), "got: {result}");
    assert!(
        result.contains("from agent2: status update"),
        "got: {result}"
    );
}

#[test]
fn query_inbox_with_args_fails() {
    let (handle, _meta) = setup_roadmap_runtime();
    let result = handle.execute_await("query_inbox", &[Value::string("unexpected")]);
    assert!(result.is_err(), "should fail with extra args");
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("expects no arguments")
    );
}

#[test]
fn inbox_clear_removes_messages() {
    let (handle, meta) = setup_roadmap_runtime();
    {
        let mut meta = meta.lock().unwrap();
        crate::session::send_agent_message(&mut meta, "agent1", "main", "first");
        crate::session::send_agent_message(&mut meta, "agent2", "main", "second");
    }
    let rt = crate::shared_state::get_shared_runtime().unwrap();
    rt.write().unwrap().agent_mailbox = meta.lock().unwrap().agent_mailbox.clone();

    let result = unwrap_string(handle.execute_await("inbox_clear", &[]).unwrap());
    assert_eq!(result, "cleared 2 messages");
    assert!(meta.lock().unwrap().agent_mailbox.get("main").is_none());
    assert!(rt.read().unwrap().agent_mailbox.get("main").is_none());
}

#[test]
fn inbox_clear_with_args_fails() {
    let (handle, _meta) = setup_roadmap_runtime();
    let result = handle.execute_await("inbox_clear", &[Value::string("unexpected")]);
    assert!(result.is_err(), "should fail with extra args");
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("expects no arguments")
    );
}

#[test]
fn query_library_returns_string() {
    let handle = setup_query_runtime("");
    let result = unwrap_string(handle.execute_await("query_library", &[]).unwrap());
    assert!(
        result.contains("Module library"),
        "should contain library info: {result}"
    );
}

#[test]
fn failure_history_returns_recent_failures() {
    let (handle, _meta) = setup_roadmap_runtime();
    let rt = crate::shared_state::get_shared_runtime().unwrap();
    if let Ok(mut state) = rt.write() {
        crate::session::record_failure(&mut state, "undefined variable `user_id`");
        crate::session::record_failure(&mut state, "type mismatch in let binding");
    }
    let result = unwrap_string(handle.execute_await("failure_history", &[]).unwrap());
    assert!(
        result.contains("type mismatch in let binding"),
        "got: {result}"
    );
    assert!(
        result.contains("undefined variable `user_id`"),
        "got: {result}"
    );
}

#[test]
fn failure_patterns_groups_similar_errors() {
    let (handle, _meta) = setup_roadmap_runtime();
    let rt = crate::shared_state::get_shared_runtime().unwrap();
    if let Ok(mut state) = rt.write() {
        crate::session::record_failure(&mut state, "undefined variable `user_id`");
        crate::session::record_failure(&mut state, "undefined variable `account_id`");
        crate::session::record_failure(&mut state, "parse error: unexpected +end");
    }
    let result = unwrap_string(handle.execute_await("failure_patterns", &[]).unwrap());
    assert!(
        result.contains("2x undefined variable errors"),
        "got: {result}"
    );
    assert!(result.contains("latest: `account_id`"), "got: {result}");
    assert!(result.contains("1x parse errors"), "got: {result}");
}

#[test]
fn failure_patterns_with_args_fails() {
    let (handle, _meta) = setup_roadmap_runtime();
    let result = handle.execute_await("failure_patterns", &[Value::string("unexpected")]);
    assert!(result.is_err(), "should fail with extra args");
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("expects no arguments")
    );
}

#[test]
fn clear_failure_history_clears_entries() {
    let (handle, _meta) = setup_roadmap_runtime();
    let rt = crate::shared_state::get_shared_runtime().unwrap();
    if let Ok(mut state) = rt.write() {
        crate::session::record_failure(&mut state, "undefined variable `user_id`");
    }
    let result = unwrap_string(handle.execute_await("clear_failure_history", &[]).unwrap());
    assert_eq!(result, "cleared");
    let history = unwrap_string(handle.execute_await("failure_history", &[]).unwrap());
    assert_eq!(history, "No recent mutation failures.");
}

#[test]
fn query_no_program_errors() {
    // Clear the thread-local program
    crate::shared_state::set_shared_program(None);
    let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState {
        http_routes: vec![],
        shared_vars: std::collections::HashMap::new(),
        ..Default::default()
    }));
    crate::shared_state::set_shared_runtime(Some(rt));
    let handle = CoroutineHandle::new_mock(vec![]);

    // All query builtins that need the program should error
    let result = handle.execute_await("query_symbols", &[]);
    assert!(result.is_err(), "query_symbols should fail without program");
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("program not available"),
        "should mention program not available"
    );

    let result = handle.execute_await("query_source", &[Value::string("x")]);
    assert!(result.is_err(), "query_source should fail without program");

    let result = handle.execute_await("query_library", &[]);
    assert!(result.is_err(), "query_library should fail without program");
}

// ═════════════════════════════════════════════════════════════════════
// Mutation IO builtins
// ═════════════════════════════════════════════════════════════════════

/// Helper: build a program from Adapsis source and install both read-only
/// and mutable program thread-locals. Returns the handle and the mutable
/// program Arc for post-mutation assertions.
fn setup_mutation_runtime(
    source: &str,
) -> (
    CoroutineHandle,
    std::sync::Arc<std::sync::RwLock<crate::ast::Program>>,
) {
    let ops = crate::parser::parse(source).expect("parse failed");
    let mut program = crate::ast::Program::default();
    for op in &ops {
        match op {
            crate::parser::Operation::Test(_) | crate::parser::Operation::Eval(_) => {}
            _ => {
                crate::validator::apply_and_validate(&mut program, op).expect("validation failed");
            }
        }
    }
    program.rebuild_function_index();
    crate::shared_state::set_shared_program(Some(std::sync::Arc::new(program.clone())));

    let program_mut = std::sync::Arc::new(std::sync::RwLock::new(program));
    crate::shared_state::set_shared_program_mut(Some(program_mut.clone()));

    let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState {
        http_routes: vec![],
        shared_vars: std::collections::HashMap::new(),
        ..Default::default()
    }));
    crate::shared_state::set_shared_runtime(Some(rt));

    (CoroutineHandle::new_mock(vec![]), program_mut)
}

// ── mutate ──

#[test]
fn mutate_add_function() {
    let (handle, prog) = setup_mutation_runtime("");
    let code = "+fn hello ()->String\n  +return \"hi\"\n+end";
    let result = unwrap_string(
        handle
            .execute_await("mutate", &[Value::string(code)])
            .unwrap(),
    );
    assert!(
        result.contains("Applied 1 mutation"),
        "should report 1 mutation: {result}"
    );
    assert!(
        result.contains("hello"),
        "should mention function name: {result}"
    );

    // Verify function was actually added
    let p = prog.read().unwrap();
    assert!(
        p.get_function("hello").is_some(),
        "hello function should exist in program"
    );
}

#[test]
fn mutate_add_module_with_functions() {
    let (handle, prog) = setup_mutation_runtime("");
    let code = "+module Greeter\n+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end";
    let result = unwrap_string(
        handle
            .execute_await("mutate", &[Value::string(code)])
            .unwrap(),
    );
    assert!(
        result.contains("Applied"),
        "should report mutations: {result}"
    );

    let p = prog.read().unwrap();
    assert!(
        p.get_function("Greeter.greet").is_some(),
        "Greeter.greet should exist"
    );
}

#[test]
fn mutate_empty_code_fails() {
    let (handle, _prog) = setup_mutation_runtime("");
    let result = handle.execute_await("mutate", &[Value::string("")]);
    assert!(result.is_err(), "mutate with empty code should fail");
    assert!(
        result.unwrap_err().to_string().contains("empty"),
        "error should mention empty"
    );
}

#[test]
fn mutate_invalid_syntax_fails() {
    let (handle, _prog) = setup_mutation_runtime("");
    let result = handle.execute_await("mutate", &[Value::string("+fn")]);
    assert!(result.is_err(), "mutate with invalid syntax should fail");
}

#[test]
fn mutate_no_args_fails() {
    let (handle, _prog) = setup_mutation_runtime("");
    let result = handle.execute_await("mutate", &[]);
    assert!(result.is_err(), "mutate with no args should fail");
}

#[test]
fn mutate_no_program_fails() {
    crate::shared_state::set_shared_program_mut(None);
    let handle = CoroutineHandle::new_mock(vec![]);
    let result = handle.execute_await(
        "mutate",
        &[Value::string("+fn x ()->Int\n  +return 1\n+end")],
    );
    assert!(result.is_err(), "mutate without program should fail");
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("program not available")
    );
}

// ── fn_remove ──

#[test]
fn fn_remove_top_level() {
    let (handle, prog) = setup_mutation_runtime("+fn hello ()->String\n  +return \"hi\"\n+end");
    // Verify function exists
    assert!(prog.read().unwrap().get_function("hello").is_some());

    let result = unwrap_string(
        handle
            .execute_await("fn_remove", &[Value::string("hello")])
            .unwrap(),
    );
    assert_eq!(result, "Removed hello");

    // Verify function was removed
    assert!(prog.read().unwrap().get_function("hello").is_none());
}

#[test]
fn fn_remove_from_module() {
    let (handle, prog) = setup_mutation_runtime(
        "+module MyMod\n+fn greet (name:String)->String\n  +return name\n+end",
    );
    assert!(prog.read().unwrap().get_function("MyMod.greet").is_some());

    let result = unwrap_string(
        handle
            .execute_await("fn_remove", &[Value::string("MyMod.greet")])
            .unwrap(),
    );
    assert_eq!(result, "Removed MyMod.greet");
    assert!(prog.read().unwrap().get_function("MyMod.greet").is_none());
}

#[test]
fn fn_remove_not_found() {
    let (handle, _prog) = setup_mutation_runtime("");
    let result = handle.execute_await("fn_remove", &[Value::string("nonexistent")]);
    assert!(
        result.is_err(),
        "fn_remove should fail for missing function"
    );
    assert!(result.unwrap_err().to_string().contains("not found"));
}

#[test]
fn fn_remove_wrong_type_fails() {
    let (handle, _prog) = setup_mutation_runtime("");
    let result = handle.execute_await("fn_remove", &[Value::Int(42)]);
    assert!(result.is_err(), "fn_remove should fail with non-String arg");
}

#[test]
fn fn_remove_module_not_found() {
    let (handle, _prog) = setup_mutation_runtime("");
    let result = handle.execute_await("fn_remove", &[Value::string("NoModule.func")]);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("module `NoModule` not found")
    );
}

// ── type_remove ──

#[test]
fn type_remove_top_level() {
    let (handle, prog) = setup_mutation_runtime("+type Color = Red | Green | Blue");
    assert!(!prog.read().unwrap().types.is_empty());

    let result = unwrap_string(
        handle
            .execute_await("type_remove", &[Value::string("Color")])
            .unwrap(),
    );
    assert_eq!(result, "Removed Color");
    assert!(prog.read().unwrap().types.is_empty());
}

#[test]
fn type_remove_not_found() {
    let (handle, _prog) = setup_mutation_runtime("");
    let result = handle.execute_await("type_remove", &[Value::string("Missing")]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}

#[test]
fn type_remove_wrong_type_fails() {
    let (handle, _prog) = setup_mutation_runtime("");
    let result = handle.execute_await("type_remove", &[Value::Int(1)]);
    assert!(result.is_err());
}

// ── module_remove ──

#[test]
fn module_remove_existing() {
    let (handle, prog) =
        setup_mutation_runtime("+module MyMod\n+fn hello ()->String\n  +return \"hi\"\n+end");
    assert!(!prog.read().unwrap().modules.is_empty());

    let result = unwrap_string(
        handle
            .execute_await("module_remove", &[Value::string("MyMod")])
            .unwrap(),
    );
    assert_eq!(result, "Removed module MyMod");
    assert!(prog.read().unwrap().modules.is_empty());
    assert!(prog.read().unwrap().get_function("MyMod.hello").is_none());
}

#[test]
fn module_remove_not_found() {
    let (handle, _prog) = setup_mutation_runtime("");
    let result = handle.execute_await("module_remove", &[Value::string("NoModule")]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}

#[test]
fn module_remove_wrong_type_fails() {
    let (handle, _prog) = setup_mutation_runtime("");
    let result = handle.execute_await("module_remove", &[Value::Int(1)]);
    assert!(result.is_err());
}

// ── Mutation builtins update read-only snapshot ──

#[test]
fn mutate_updates_query_snapshot() {
    let (handle, _prog) = setup_mutation_runtime("");

    // Add a function via mutate
    handle
        .execute_await(
            "mutate",
            &[Value::string("+fn test_func ()->Int\n  +return 42\n+end")],
        )
        .unwrap();

    // Query builtins should see the new function via updated snapshot
    let result = unwrap_string(handle.execute_await("query_symbols", &[]).unwrap());
    assert!(
        result.contains("test_func"),
        "query_symbols should see mutated function: {result}"
    );
}

#[test]
fn fn_remove_updates_query_snapshot() {
    let (handle, _prog) =
        setup_mutation_runtime("+fn to_remove ()->String\n  +return \"bye\"\n+end");

    // Remove the function
    handle
        .execute_await("fn_remove", &[Value::string("to_remove")])
        .unwrap();

    // Query builtins should no longer see it
    let result = unwrap_string(handle.execute_await("query_symbols", &[]).unwrap());
    assert!(
        !result.contains("to_remove"),
        "query_symbols should not see removed function: {result}"
    );
}

// ═════════════════════════════════════════════════════════════════════
// Query alias builtins — verify aliases work with Program access
// ═════════════════════════════════════════════════════════════════════

#[test]
fn symbols_list_alias_works() {
    let handle = setup_query_runtime(
        "+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end",
    );
    let result = unwrap_string(handle.execute_await("symbols_list", &[]).unwrap());
    assert!(
        result.contains("greet"),
        "symbols_list alias should list greet function: {result}"
    );
}

#[test]
fn source_get_alias_works() {
    let handle = setup_query_runtime(
        "+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end",
    );
    let result = unwrap_string(
        handle
            .execute_await("source_get", &[Value::string("greet")])
            .unwrap(),
    );
    assert!(
        result.contains("+fn greet"),
        "source_get alias should return source: {result}"
    );
}

#[test]
fn callers_get_alias_works() {
    let handle = setup_query_runtime(
        "+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end\n\
             +fn main ()->String\n  +let x:String = greet(\"world\")\n  +return x\n+end",
    );
    let result = unwrap_string(
        handle
            .execute_await("callers_get", &[Value::string("greet")])
            .unwrap(),
    );
    assert!(
        result.contains("main"),
        "callers_get alias should list main as caller: {result}"
    );
}

#[test]
fn callees_get_alias_works() {
    let handle = setup_query_runtime(
        "+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end\n\
             +fn main ()->String\n  +let x:String = greet(\"world\")\n  +return x\n+end",
    );
    let result = unwrap_string(
        handle
            .execute_await("callees_get", &[Value::string("main")])
            .unwrap(),
    );
    assert!(
        result.contains("greet"),
        "callees_get alias should list greet as callee: {result}"
    );
}

#[test]
fn deps_get_alias_works() {
    let handle = setup_query_runtime(
        "+fn a ()->String\n  +return \"hello\"\n+end\n\
             +fn b ()->String\n  +let x:String = a()\n  +return x\n+end\n\
             +fn c ()->String\n  +let x:String = b()\n  +return x\n+end",
    );
    let result = unwrap_string(
        handle
            .execute_await("deps_get", &[Value::string("c")])
            .unwrap(),
    );
    assert!(
        result.contains("a"),
        "deps_get alias should include transitive dep 'a': {result}"
    );
    assert!(
        result.contains("b"),
        "deps_get alias should include direct dep 'b': {result}"
    );
}

#[test]
fn routes_list_alias_works() {
    let program = crate::ast::Program::default();
    crate::shared_state::set_shared_program(Some(std::sync::Arc::new(program)));
    let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState {
        http_routes: vec![crate::ast::HttpRoute {
            method: "POST".to_string(),
            path: "/api/data".to_string(),
            handler_fn: "handle_data".to_string(),
        }],
        shared_vars: std::collections::HashMap::new(),
        ..Default::default()
    }));
    crate::shared_state::set_shared_runtime(Some(rt));
    let handle = CoroutineHandle::new_mock(vec![]);

    let result = unwrap_string(handle.execute_await("routes_list", &[]).unwrap());
    assert!(
        result.contains("POST"),
        "routes_list alias should contain POST method: {result}"
    );
    assert!(
        result.contains("/api/data"),
        "routes_list alias should contain path: {result}"
    );
}

#[test]
fn alias_no_program_errors() {
    // Clear the thread-local program
    crate::shared_state::set_shared_program(None);
    let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState {
        http_routes: vec![],
        shared_vars: std::collections::HashMap::new(),
        ..Default::default()
    }));
    crate::shared_state::set_shared_runtime(Some(rt));
    let handle = CoroutineHandle::new_mock(vec![]);

    // All alias builtins that need the program should error
    let result = handle.execute_await("symbols_list", &[]);
    assert!(result.is_err(), "symbols_list should fail without program");
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("program not available"),
        "should mention program not available"
    );

    let result = handle.execute_await("source_get", &[Value::string("x")]);
    assert!(result.is_err(), "source_get should fail without program");

    let result = handle.execute_await("callers_get", &[Value::string("x")]);
    assert!(result.is_err(), "callers_get should fail without program");

    let result = handle.execute_await("callees_get", &[Value::string("x")]);
    assert!(result.is_err(), "callees_get should fail without program");

    let result = handle.execute_await("deps_get", &[Value::string("x")]);
    assert!(result.is_err(), "deps_get should fail without program");

    let result = handle.execute_await("routes_list", &[]);
    assert!(result.is_err(), "routes_list should fail without program");
}

// ═════════════════════════════════════════════════════════════════════
// move_symbols builtin
// ═════════════════════════════════════════════════════════════════════

#[test]
fn move_symbols_moves_function() {
    let (handle, prog) = setup_mutation_runtime("+fn helper ()->String\n  +return \"hi\"\n+end");
    assert!(prog.read().unwrap().get_function("helper").is_some());

    let result = unwrap_string(
        handle
            .execute_await(
                "move_symbols",
                &[Value::string("helper"), Value::string("Utils")],
            )
            .unwrap(),
    );
    assert!(result.contains("moved"), "should confirm move: {result}");
    assert!(
        result.contains("Utils"),
        "should mention target module: {result}"
    );

    // Function should now be in Utils module, not top-level
    let p = prog.read().unwrap();
    assert!(
        p.get_function("Utils.helper").is_some(),
        "helper should be in Utils"
    );
    assert!(
        p.functions.iter().all(|f| f.name != "helper"),
        "helper should not be in top-level functions"
    );
}

#[test]
fn move_symbols_multiple_comma_separated() {
    let (handle, prog) = setup_mutation_runtime(
        "+fn foo ()->Int\n  +return 1\n+end\n\
             +fn bar ()->Int\n  +return 2\n+end",
    );
    let result = unwrap_string(
        handle
            .execute_await(
                "move_symbols",
                &[Value::string("foo, bar"), Value::string("Helpers")],
            )
            .unwrap(),
    );
    assert!(result.contains("moved"), "should confirm move: {result}");
    let p = prog.read().unwrap();
    assert!(p.get_function("Helpers.foo").is_some());
    assert!(p.get_function("Helpers.bar").is_some());
}

#[test]
fn move_symbols_empty_symbols_fails() {
    let (handle, _prog) = setup_mutation_runtime("");
    let result = handle.execute_await(
        "move_symbols",
        &[Value::string(""), Value::string("Target")],
    );
    assert!(result.is_err(), "should fail with empty symbols");
}

#[test]
fn move_symbols_empty_target_fails() {
    let (handle, _prog) = setup_mutation_runtime("+fn x ()->Int\n  +return 1\n+end");
    let result = handle.execute_await("move_symbols", &[Value::string("x"), Value::string("")]);
    assert!(result.is_err(), "should fail with empty target");
}

#[test]
fn move_symbols_not_found_fails() {
    let (handle, _prog) = setup_mutation_runtime("");
    let result = handle.execute_await(
        "move_symbols",
        &[Value::string("nonexistent"), Value::string("Target")],
    );
    assert!(result.is_err(), "should fail when symbol not found");
}

#[test]
fn move_symbols_no_args_fails() {
    let (handle, _prog) = setup_mutation_runtime("");
    let result = handle.execute_await("move_symbols", &[]);
    assert!(result.is_err(), "should fail with no args");
}

#[test]
fn move_symbols_no_program_fails() {
    crate::shared_state::set_shared_program_mut(None);
    let handle = CoroutineHandle::new_mock(vec![]);
    let result = handle.execute_await(
        "move_symbols",
        &[Value::string("x"), Value::string("Target")],
    );
    assert!(result.is_err(), "should fail without program");
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("program not available")
    );
}

// ═════════════════════════════════════════════════════════════════════
// trace_run builtin
// ═════════════════════════════════════════════════════════════════════

#[test]
fn trace_run_simple_function() {
    let handle = setup_query_runtime(
        "+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end",
    );
    let result = unwrap_string(
        handle
            .execute_await(
                "trace_run",
                &[Value::string("greet"), Value::string("\"world\"")],
            )
            .unwrap(),
    );
    assert!(
        result.contains("Trace of greet"),
        "should have trace header: {result}"
    );
    assert!(
        result.contains("return"),
        "should have a return step: {result}"
    );
}

#[test]
fn trace_run_no_args() {
    let handle = setup_query_runtime("+fn get_one ()->Int\n  +return 1\n+end");
    let result = unwrap_string(
        handle
            .execute_await("trace_run", &[Value::string("get_one"), Value::string("")])
            .unwrap(),
    );
    assert!(
        result.contains("Trace of get_one"),
        "should trace: {result}"
    );
}

#[test]
fn trace_run_function_not_found() {
    let handle = setup_query_runtime("");
    let result = handle.execute_await(
        "trace_run",
        &[Value::string("nonexistent"), Value::string("")],
    );
    assert!(result.is_err(), "should fail for missing function");
    assert!(result.unwrap_err().to_string().contains("not found"));
}

#[test]
fn trace_run_wrong_type_fails() {
    let handle = setup_query_runtime("");
    let result = handle.execute_await("trace_run", &[Value::Int(42)]);
    assert!(result.is_err(), "should fail with non-String fn_name");
}

#[test]
fn trace_run_no_program_fails() {
    crate::shared_state::set_shared_program(None);
    let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState {
        http_routes: vec![],
        shared_vars: std::collections::HashMap::new(),
        ..Default::default()
    }));
    crate::shared_state::set_shared_runtime(Some(rt));
    let handle = CoroutineHandle::new_mock(vec![]);
    let result = handle.execute_await("trace_run", &[Value::string("x"), Value::string("")]);
    assert!(result.is_err(), "should fail without program");
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("program not available")
    );
}

// ═════════════════════════════════════════════════════════════════════
// msg_send builtin
// ═════════════════════════════════════════════════════════════════════

#[test]
fn msg_send_delivers_message() {
    let (handle, _meta) = setup_roadmap_runtime();
    let rt = crate::shared_state::get_shared_runtime().unwrap();
    let result = unwrap_string(
        handle
            .execute_await(
                "msg_send",
                &[Value::string("agent1"), Value::string("hello from main")],
            )
            .unwrap(),
    );
    assert!(
        result.contains("Message sent to 'agent1'"),
        "confirmation: {result}"
    );

    // Verify message is in the mailbox (agent_mailbox is still in RuntimeState)
    let state = rt.read().unwrap();
    let inbox = state.agent_mailbox.get("agent1").unwrap();
    assert_eq!(inbox.len(), 1);
    assert_eq!(inbox[0].content, "hello from main");
    assert_eq!(inbox[0].from, "main");
    assert_eq!(inbox[0].to, "agent1");
}

#[test]
fn msg_send_multiple_messages() {
    let (handle, _meta) = setup_roadmap_runtime();
    let rt = crate::shared_state::get_shared_runtime().unwrap();
    handle
        .execute_await(
            "msg_send",
            &[Value::string("agent1"), Value::string("first")],
        )
        .unwrap();
    handle
        .execute_await(
            "msg_send",
            &[Value::string("agent1"), Value::string("second")],
        )
        .unwrap();

    let state = rt.read().unwrap();
    let inbox = state.agent_mailbox.get("agent1").unwrap();
    assert_eq!(inbox.len(), 2);
    assert_eq!(inbox[0].content, "first");
    assert_eq!(inbox[1].content, "second");
}

#[test]
fn msg_send_syncs_meta_mailbox() {
    let (handle, meta) = setup_roadmap_runtime();
    handle
        .execute_await(
            "msg_send",
            &[Value::string("agent1"), Value::string("hello from main")],
        )
        .unwrap();

    let meta = meta.lock().unwrap();
    let inbox = meta.agent_mailbox.get("agent1").unwrap();
    assert_eq!(inbox.len(), 1);
    assert_eq!(inbox[0].content, "hello from main");
}

#[test]
fn msg_send_empty_target_fails() {
    let (handle, _rt) = setup_roadmap_runtime();
    let result = handle.execute_await("msg_send", &[Value::string(""), Value::string("hello")]);
    assert!(result.is_err(), "should fail with empty target");
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("must not be empty")
    );
}

#[test]
fn msg_send_no_args_fails() {
    let (handle, _rt) = setup_roadmap_runtime();
    let result = handle.execute_await("msg_send", &[]);
    assert!(result.is_err(), "should fail with no args");
}

#[test]
fn msg_send_wrong_type_fails() {
    let (handle, _rt) = setup_roadmap_runtime();
    let result = handle.execute_await("msg_send", &[Value::Int(42)]);
    assert!(result.is_err(), "should fail with non-String target");
}

#[test]
fn msg_send_missing_message_fails() {
    let (handle, _rt) = setup_roadmap_runtime();
    let result = handle.execute_await("msg_send", &[Value::string("agent1")]);
    assert!(result.is_err(), "should fail without message arg");
}

// ═════════════════════════════════════════════════════════════════════
// inbox_read builtin
// ═════════════════════════════════════════════════════════════════════

#[test]
fn inbox_read_returns_json_and_clears_inbox() {
    let (handle, meta) = setup_roadmap_runtime();
    {
        let mut meta = meta.lock().unwrap();
        crate::session::send_agent_message(&mut meta, "agent1", "main", "first");
        crate::session::send_agent_message(&mut meta, "agent2", "main", "second");
    }
    let rt = crate::shared_state::get_shared_runtime().unwrap();
    rt.write().unwrap().agent_mailbox = meta.lock().unwrap().agent_mailbox.clone();

    let result = unwrap_string(handle.execute_await("inbox_read", &[]).unwrap());
    assert_eq!(result, "[\"first\",\"second\"]");
    assert!(meta.lock().unwrap().agent_mailbox.get("main").is_none());
    assert!(rt.read().unwrap().agent_mailbox.get("main").is_none());

    let second = unwrap_string(handle.execute_await("inbox_read", &[]).unwrap());
    assert_eq!(second, "[]");
}

#[test]
fn inbox_read_mock_intercepts_and_preserves_inbox() {
    let (_handle, meta) = setup_roadmap_runtime();
    {
        let mut meta = meta.lock().unwrap();
        crate::session::send_agent_message(&mut meta, "agent1", "main", "first");
    }
    let handle = CoroutineHandle::new_mock(vec![crate::session::IoMock {
        operation: "inbox_read".to_string(),
        patterns: vec!["".to_string()],
        response: "[\"mocked\"]".to_string(),
    }]);

    let result = unwrap_string(handle.execute_await("inbox_read", &[]).unwrap());
    assert_eq!(result, "[\"mocked\"]");
    let meta = meta.lock().unwrap();
    let inbox = meta.agent_mailbox.get("main").unwrap();
    assert_eq!(inbox.len(), 1);
    assert_eq!(inbox[0].content, "first");
}

#[test]
fn inbox_read_with_args_fails() {
    let (handle, _meta) = setup_roadmap_runtime();
    let result = handle.execute_await("inbox_read", &[Value::string("unexpected")]);
    assert!(result.is_err(), "should fail with extra args");
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("expects no arguments")
    );
}

// ═════════════════════════════════════════════════════════════════════
// watch_start builtin
// ═════════════════════════════════════════════════════════════════════

#[test]
fn watch_start_queues_command() {
    // Set up with a function to watch
    let (handle, _prog) = setup_mutation_runtime("+fn checker ()->Int\n  +return 42\n+end");
    let result = unwrap_string(
        handle
            .execute_await("watch_start", &[Value::string("checker"), Value::Int(1000)])
            .unwrap(),
    );
    assert!(
        result.contains("Watching checker"),
        "confirmation: {result}"
    );
    assert!(
        result.contains("1000ms"),
        "should mention interval: {result}"
    );

    // Verify command was queued
    let rt = crate::shared_state::get_shared_runtime().unwrap();
    let state = rt.read().unwrap();
    assert_eq!(state.pending_commands.len(), 1);
    assert!(state.pending_commands[0].contains("!watch checker 1000"));
}

#[test]
fn watch_start_function_not_found_fails() {
    let handle = setup_query_runtime("");
    let result = handle.execute_await(
        "watch_start",
        &[Value::string("nonexistent"), Value::Int(1000)],
    );
    assert!(result.is_err(), "should fail for missing function");
    assert!(result.unwrap_err().to_string().contains("not found"));
}

#[test]
fn watch_start_zero_interval_fails() {
    let handle = setup_query_runtime("+fn x ()->Int\n  +return 1\n+end");
    let result = handle.execute_await("watch_start", &[Value::string("x"), Value::Int(0)]);
    assert!(result.is_err(), "should fail with zero interval");
}

#[test]
fn watch_start_negative_interval_fails() {
    let handle = setup_query_runtime("+fn x ()->Int\n  +return 1\n+end");
    let result = handle.execute_await("watch_start", &[Value::string("x"), Value::Int(-100)]);
    assert!(result.is_err(), "should fail with negative interval");
}

#[test]
fn watch_start_wrong_type_fn_name_fails() {
    let (handle, _) = setup_roadmap_runtime();
    let result = handle.execute_await("watch_start", &[Value::Int(42)]);
    assert!(result.is_err(), "should fail with non-String fn_name");
}

#[test]
fn watch_start_empty_fn_name_fails() {
    let handle = setup_query_runtime("+fn x ()->Int\n  +return 1\n+end");
    let result = handle.execute_await("watch_start", &[Value::string(""), Value::Int(1000)]);
    assert!(result.is_err(), "should fail with empty fn_name");
}

// ═════════════════════════════════════════════════════════════════════
// agent_spawn builtin
// ═════════════════════════════════════════════════════════════════════

#[test]
fn agent_spawn_queues_command() {
    let (handle, _rt) = setup_roadmap_runtime();
    let result = unwrap_string(
        handle
            .execute_await(
                "agent_spawn",
                &[
                    Value::string("worker1"),
                    Value::string("new-only"),
                    Value::string("Build a calculator module"),
                ],
            )
            .unwrap(),
    );
    assert!(
        result.contains("Agent 'worker1' spawned"),
        "confirmation: {result}"
    );
    assert!(
        result.contains("new-only"),
        "should mention scope: {result}"
    );

    // Verify command was queued
    let rt = crate::shared_state::get_shared_runtime().unwrap();
    let state = rt.read().unwrap();
    assert_eq!(state.pending_commands.len(), 1);
    assert!(state.pending_commands[0].contains("!agent worker1"));
    assert!(state.pending_commands[0].contains("--scope new-only"));
    assert!(state.pending_commands[0].contains("Build a calculator module"));
}

#[test]
fn agent_spawn_empty_name_fails() {
    let (handle, _rt) = setup_roadmap_runtime();
    let result = handle.execute_await(
        "agent_spawn",
        &[
            Value::string(""),
            Value::string("full"),
            Value::string("do something"),
        ],
    );
    assert!(result.is_err(), "should fail with empty name");
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("must not be empty")
    );
}

#[test]
fn agent_spawn_empty_task_fails() {
    let (handle, _rt) = setup_roadmap_runtime();
    let result = handle.execute_await(
        "agent_spawn",
        &[
            Value::string("worker"),
            Value::string("full"),
            Value::string(""),
        ],
    );
    assert!(result.is_err(), "should fail with empty task");
}

#[test]
fn agent_spawn_no_args_fails() {
    let (handle, _rt) = setup_roadmap_runtime();
    let result = handle.execute_await("agent_spawn", &[]);
    assert!(result.is_err(), "should fail with no args");
}

#[test]
fn agent_spawn_missing_task_fails() {
    let (handle, _rt) = setup_roadmap_runtime();
    let result = handle.execute_await(
        "agent_spawn",
        &[Value::string("worker"), Value::string("full")],
    );
    assert!(result.is_err(), "should fail without task arg");
}

// ═════════════════════════════════════════════════════════════════════
// New builtins are registered
// ═════════════════════════════════════════════════════════════════════

#[test]
fn new_io_builtins_registered() {
    for name in &[
        "move_symbols",
        "watch_start",
        "agent_spawn",
        "msg_send",
        "query_inbox",
        "inbox_read",
        "inbox_clear",
        "trace_run",
        "route_list",
        "route_add",
        "route_remove",
        "undo",
        "sandbox_enter",
        "sandbox_merge",
        "sandbox_discard",
        "mock_set",
        "mock_clear",
        "sse_send",
        "failure_history",
        "clear_failure_history",
        "failure_patterns",
        "module_create",
        "test_run",
        "fn_replace",
    ] {
        assert!(
            crate::builtins::is_io_builtin(name),
            "IO builtin '{name}' should be registered"
        );
        assert!(
            crate::builtins::is_builtin(name),
            "'{name}' should also return true for is_builtin"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════
// route_list builtin
// ═════════════════════════════════════════════════════════════════════

#[test]
fn route_list_empty() {
    let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState {
        http_routes: vec![],
        ..Default::default()
    }));
    crate::shared_state::set_shared_runtime(Some(rt));
    let handle = CoroutineHandle::new_mock(vec![]);
    let result = unwrap_string(handle.execute_await("route_list", &[]).unwrap());
    assert_eq!(result, "No routes registered.");
}

#[test]
fn route_list_with_routes() {
    let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState {
        http_routes: vec![
            crate::ast::HttpRoute {
                method: "GET".into(),
                path: "/api/foo".into(),
                handler_fn: "Mod.foo".into(),
            },
            crate::ast::HttpRoute {
                method: "POST".into(),
                path: "/api/bar".into(),
                handler_fn: "Mod.bar".into(),
            },
        ],
        ..Default::default()
    }));
    crate::shared_state::set_shared_runtime(Some(rt));
    let handle = CoroutineHandle::new_mock(vec![]);
    let result = unwrap_string(handle.execute_await("route_list", &[]).unwrap());
    assert!(
        result.contains("GET /api/foo -> `Mod.foo`"),
        "should list first route: {result}"
    );
    assert!(
        result.contains("POST /api/bar -> `Mod.bar`"),
        "should list second route: {result}"
    );
}

// ═════════════════════════════════════════════════════════════════════
// route_add builtin
// ═════════════════════════════════════════════════════════════════════

#[test]
fn route_add_new_route() {
    let rt = std::sync::Arc::new(std::sync::RwLock::new(
        crate::session::RuntimeState::default(),
    ));
    crate::shared_state::set_shared_runtime(Some(rt.clone()));
    let handle = CoroutineHandle::new_mock(vec![]);
    let result = unwrap_string(
        handle
            .execute_await(
                "route_add",
                &[
                    Value::string("POST"),
                    Value::string("/api/test"),
                    Value::string("Handler.test"),
                ],
            )
            .unwrap(),
    );
    assert!(
        result.contains("added route POST /api/test"),
        "should confirm add: {result}"
    );
    assert_eq!(rt.read().unwrap().http_routes.len(), 1);
    assert_eq!(rt.read().unwrap().http_routes[0].handler_fn, "Handler.test");
}

#[test]
fn route_add_upserts_existing() {
    let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState {
        http_routes: vec![crate::ast::HttpRoute {
            method: "GET".into(),
            path: "/api/data".into(),
            handler_fn: "Old.handler".into(),
        }],
        ..Default::default()
    }));
    crate::shared_state::set_shared_runtime(Some(rt.clone()));
    let handle = CoroutineHandle::new_mock(vec![]);
    let result = unwrap_string(
        handle
            .execute_await(
                "route_add",
                &[
                    Value::string("GET"),
                    Value::string("/api/data"),
                    Value::string("New.handler"),
                ],
            )
            .unwrap(),
    );
    assert!(
        result.contains("updated route"),
        "should say updated: {result}"
    );
    assert!(
        result.contains("Old.handler"),
        "should mention old handler: {result}"
    );
    assert_eq!(rt.read().unwrap().http_routes.len(), 1);
    assert_eq!(rt.read().unwrap().http_routes[0].handler_fn, "New.handler");
}

#[test]
fn route_add_invalid_method() {
    let rt = std::sync::Arc::new(std::sync::RwLock::new(
        crate::session::RuntimeState::default(),
    ));
    crate::shared_state::set_shared_runtime(Some(rt));
    let handle = CoroutineHandle::new_mock(vec![]);
    let result = handle.execute_await(
        "route_add",
        &[
            Value::string("FOOBAR"),
            Value::string("/api/x"),
            Value::string("H.x"),
        ],
    );
    assert!(result.is_err(), "invalid method should fail");
    assert!(
        result.unwrap_err().to_string().contains("method must be"),
        "should mention valid methods"
    );
}

#[test]
fn route_add_invalid_path() {
    let rt = std::sync::Arc::new(std::sync::RwLock::new(
        crate::session::RuntimeState::default(),
    ));
    crate::shared_state::set_shared_runtime(Some(rt));
    let handle = CoroutineHandle::new_mock(vec![]);
    let result = handle.execute_await(
        "route_add",
        &[
            Value::string("GET"),
            Value::string("no-leading-slash"),
            Value::string("H.x"),
        ],
    );
    assert!(result.is_err(), "path without / should fail");
}

// ═════════════════════════════════════════════════════════════════════
// route_remove builtin
// ═════════════════════════════════════════════════════════════════════

#[test]
fn route_remove_existing() {
    let rt = std::sync::Arc::new(std::sync::RwLock::new(crate::session::RuntimeState {
        http_routes: vec![crate::ast::HttpRoute {
            method: "POST".into(),
            path: "/api/rm".into(),
            handler_fn: "Rm.handler".into(),
        }],
        ..Default::default()
    }));
    crate::shared_state::set_shared_runtime(Some(rt.clone()));
    let handle = CoroutineHandle::new_mock(vec![]);
    let result = unwrap_string(
        handle
            .execute_await(
                "route_remove",
                &[Value::string("POST"), Value::string("/api/rm")],
            )
            .unwrap(),
    );
    assert!(
        result.contains("removed route"),
        "should confirm removal: {result}"
    );
    assert_eq!(rt.read().unwrap().http_routes.len(), 0);
}

#[test]
fn route_remove_not_found() {
    let rt = std::sync::Arc::new(std::sync::RwLock::new(
        crate::session::RuntimeState::default(),
    ));
    crate::shared_state::set_shared_runtime(Some(rt));
    let handle = CoroutineHandle::new_mock(vec![]);
    let result = handle.execute_await(
        "route_remove",
        &[Value::string("GET"), Value::string("/nonexistent")],
    );
    assert!(result.is_err(), "removing nonexistent route should fail");
    assert!(
        result.unwrap_err().to_string().contains("no route found"),
        "should say no route found"
    );
}

// ═════════════════════════════════════════════════════════════════════
// undo builtin
// ═════════════════════════════════════════════════════════════════════

#[test]
fn undo_queues_command() {
    let rt = std::sync::Arc::new(std::sync::RwLock::new(
        crate::session::RuntimeState::default(),
    ));
    crate::shared_state::set_shared_runtime(Some(rt.clone()));
    let handle = CoroutineHandle::new_mock(vec![]);
    let result = unwrap_string(handle.execute_await("undo", &[]).unwrap());
    assert!(
        result.contains("Undo queued"),
        "should confirm queued: {result}"
    );
    let state = rt.read().unwrap();
    assert_eq!(state.pending_commands.len(), 1);
    assert_eq!(state.pending_commands[0], "!undo");
}

// ═════════════════════════════════════════════════════════════════════
// sandbox builtins
// ═════════════════════════════════════════════════════════════════════

#[test]
fn sandbox_enter_queues_command() {
    let rt = std::sync::Arc::new(std::sync::RwLock::new(
        crate::session::RuntimeState::default(),
    ));
    crate::shared_state::set_shared_runtime(Some(rt.clone()));
    let handle = CoroutineHandle::new_mock(vec![]);
    let result = unwrap_string(handle.execute_await("sandbox_enter", &[]).unwrap());
    assert!(
        result.contains("Sandbox enter queued"),
        "should confirm queued: {result}"
    );
    assert_eq!(rt.read().unwrap().pending_commands[0], "!sandbox enter");
}

#[test]
fn sandbox_merge_queues_command() {
    let rt = std::sync::Arc::new(std::sync::RwLock::new(
        crate::session::RuntimeState::default(),
    ));
    crate::shared_state::set_shared_runtime(Some(rt.clone()));
    let handle = CoroutineHandle::new_mock(vec![]);
    let result = unwrap_string(handle.execute_await("sandbox_merge", &[]).unwrap());
    assert!(
        result.contains("Sandbox merge queued"),
        "should confirm queued: {result}"
    );
    assert_eq!(rt.read().unwrap().pending_commands[0], "!sandbox merge");
}

#[test]
fn sandbox_discard_queues_command() {
    let rt = std::sync::Arc::new(std::sync::RwLock::new(
        crate::session::RuntimeState::default(),
    ));
    crate::shared_state::set_shared_runtime(Some(rt.clone()));
    let handle = CoroutineHandle::new_mock(vec![]);
    let result = unwrap_string(handle.execute_await("sandbox_discard", &[]).unwrap());
    assert!(
        result.contains("Sandbox discard queued"),
        "should confirm queued: {result}"
    );
    assert_eq!(rt.read().unwrap().pending_commands[0], "!sandbox discard");
}

// ═════════════════════════════════════════════════════════════════════
// mock_set and mock_clear builtins
// ═════════════════════════════════════════════════════════════════════

#[test]
fn mock_set_adds_mock() {
    let rt = std::sync::Arc::new(std::sync::RwLock::new(
        crate::session::RuntimeState::default(),
    ));
    let meta = std::sync::Arc::new(std::sync::Mutex::new(crate::session::SessionMeta::new()));
    crate::shared_state::set_shared_runtime(Some(rt));
    crate::shared_state::set_shared_meta(Some(meta.clone()));
    let handle = CoroutineHandle::new_mock(vec![]);
    let result = unwrap_string(
        handle
            .execute_await(
                "mock_set",
                &[
                    Value::string("http_get"),
                    Value::string("example.com"),
                    Value::string("mock response body"),
                ],
            )
            .unwrap(),
    );
    assert!(
        result.contains("mock: http_get"),
        "should confirm mock: {result}"
    );
    let m = meta.lock().unwrap();
    assert_eq!(m.io_mocks.len(), 1);
    assert_eq!(m.io_mocks[0].operation, "http_get");
    assert_eq!(m.io_mocks[0].response, "mock response body");
}

#[test]
fn mock_set_empty_operation_fails() {
    let rt = std::sync::Arc::new(std::sync::RwLock::new(
        crate::session::RuntimeState::default(),
    ));
    let meta = std::sync::Arc::new(std::sync::Mutex::new(crate::session::SessionMeta::new()));
    crate::shared_state::set_shared_runtime(Some(rt));
    crate::shared_state::set_shared_meta(Some(meta));
    let handle = CoroutineHandle::new_mock(vec![]);
    let result = handle.execute_await(
        "mock_set",
        &[
            Value::string(""),
            Value::string("pattern"),
            Value::string("response"),
        ],
    );
    assert!(result.is_err(), "empty operation should fail");
}

#[test]
fn mock_clear_clears_all_mocks() {
    let rt = std::sync::Arc::new(std::sync::RwLock::new(
        crate::session::RuntimeState::default(),
    ));
    let mut initial_meta = crate::session::SessionMeta::new();
    initial_meta.io_mocks = vec![
        crate::session::IoMock {
            operation: "http_get".into(),
            patterns: vec!["x".into()],
            response: "y".into(),
        },
        crate::session::IoMock {
            operation: "http_post".into(),
            patterns: vec![],
            response: "z".into(),
        },
    ];
    let meta = std::sync::Arc::new(std::sync::Mutex::new(initial_meta));
    crate::shared_state::set_shared_runtime(Some(rt));
    crate::shared_state::set_shared_meta(Some(meta.clone()));
    let handle = CoroutineHandle::new_mock(vec![]);
    let result = unwrap_string(handle.execute_await("mock_clear", &[]).unwrap());
    assert!(
        result.contains("cleared 2 mocks"),
        "should report count: {result}"
    );
    assert_eq!(meta.lock().unwrap().io_mocks.len(), 0);
}

#[test]
fn mock_clear_empty_returns_zero() {
    let rt = std::sync::Arc::new(std::sync::RwLock::new(
        crate::session::RuntimeState::default(),
    ));
    let meta = std::sync::Arc::new(std::sync::Mutex::new(crate::session::SessionMeta::new()));
    crate::shared_state::set_shared_runtime(Some(rt));
    crate::shared_state::set_shared_meta(Some(meta));
    let handle = CoroutineHandle::new_mock(vec![]);
    let result = unwrap_string(handle.execute_await("mock_clear", &[]).unwrap());
    assert!(
        result.contains("cleared 0 mocks"),
        "should report 0: {result}"
    );
}

#[test]
fn sse_send_sends_json_event() {
    let (tx, mut rx) = tokio::sync::broadcast::channel(16);
    crate::shared_state::set_shared_event_broadcast(Some(tx));
    let handle = CoroutineHandle::new_mock(vec![]);

    let result = unwrap_string(
        handle
            .execute_await(
                "sse_send",
                &[Value::string("mutation"), Value::string("updated module")],
            )
            .unwrap(),
    );
    assert_eq!(result, "sent");
    let payload = rx.try_recv().unwrap();
    assert_eq!(
        payload,
        "{\"data\":\"updated module\",\"type\":\"mutation\"}"
    );
}

#[test]
fn sse_send_without_sender_fails() {
    crate::shared_state::set_shared_event_broadcast(None);
    let handle = CoroutineHandle::new_mock(vec![]);
    let result = handle.execute_await(
        "sse_send",
        &[Value::string("mutation"), Value::string("updated module")],
    );
    assert!(result.is_err(), "missing sender should fail");
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("no event broadcast available")
    );
}

// ═════════════════════════════════════════════════════════════════════
// module_create builtin
// ═════════════════════════════════════════════════════════════════════

#[test]
fn module_create_new_module() {
    let (handle, prog) = setup_mutation_runtime("");
    let result = unwrap_string(
        handle
            .execute_await("module_create", &[Value::string("MyMod")])
            .unwrap(),
    );
    assert!(
        result.contains("created module"),
        "should confirm creation: {result}"
    );
    let p = prog.read().unwrap();
    assert!(
        p.modules.iter().any(|m| m.name == "MyMod"),
        "module should exist"
    );
}

#[test]
fn module_create_already_exists() {
    let (handle, _prog) = setup_mutation_runtime("+module Existing");
    let result = unwrap_string(
        handle
            .execute_await("module_create", &[Value::string("Existing")])
            .unwrap(),
    );
    assert!(
        result.contains("already exists"),
        "should say already exists: {result}"
    );
}

#[test]
fn module_create_lowercase_fails() {
    let (handle, _prog) = setup_mutation_runtime("");
    let result = handle.execute_await("module_create", &[Value::string("lowercase")]);
    assert!(result.is_err(), "lowercase module name should fail");
    assert!(
        result.unwrap_err().to_string().contains("uppercase"),
        "should mention uppercase"
    );
}

#[test]
fn module_create_empty_name_fails() {
    let (handle, _prog) = setup_mutation_runtime("");
    let result = handle.execute_await("module_create", &[Value::string("")]);
    assert!(result.is_err(), "empty name should fail");
}

// ═════════════════════════════════════════════════════════════════════
// test_run builtin
// ═════════════════════════════════════════════════════════════════════

#[test]
fn test_run_with_stored_tests() {
    // Build a program with a function that has stored tests
    let source = "+fn double (x:Int)->Int\n  +return x * 2\n+end\n\
                      +test double\n  +with 3 -> expect 6\n  +with 5 -> expect 10\n";
    let ops = crate::parser::parse(source).unwrap();
    let mut program = crate::ast::Program::default();
    for op in &ops {
        match op {
            crate::parser::Operation::Test(test) => {
                // Store the tests on the function
                if let Some(func) = program.get_function_mut(&test.function_name) {
                    func.tests = test
                        .cases
                        .iter()
                        .map(|c| crate::ast::TestCase {
                            input: crate::session::format_expr_pub(&c.input),
                            expected: crate::session::format_expr_pub(&c.expected),
                            passed: true,
                            matcher: None,
                            after_checks: vec![],
                        })
                        .collect();
                }
            }
            _ => {
                crate::validator::apply_and_validate(&mut program, op).unwrap();
            }
        }
    }
    program.rebuild_function_index();
    crate::shared_state::set_shared_program(Some(std::sync::Arc::new(program)));
    let rt = std::sync::Arc::new(std::sync::RwLock::new(
        crate::session::RuntimeState::default(),
    ));
    crate::shared_state::set_shared_runtime(Some(rt));
    let handle = CoroutineHandle::new_mock(vec![]);

    let result = unwrap_string(
        handle
            .execute_await("test_run", &[Value::string("double")])
            .unwrap(),
    );
    assert!(
        result.contains("PASS"),
        "should have passing tests: {result}"
    );
    // Each test case should appear
    let pass_count = result.matches("PASS").count();
    assert_eq!(pass_count, 2, "should have 2 passing tests: {result}");
}

#[test]
fn test_run_no_stored_tests() {
    let (handle, _prog) = setup_mutation_runtime("+fn foo ()->Int\n  +return 1\n+end");
    let result = unwrap_string(
        handle
            .execute_await("test_run", &[Value::string("foo")])
            .unwrap(),
    );
    assert!(
        result.contains("no stored tests"),
        "should say no tests: {result}"
    );
}

#[test]
fn test_run_function_not_found() {
    let (handle, _prog) = setup_mutation_runtime("");
    let result = handle.execute_await("test_run", &[Value::string("nonexistent")]);
    assert!(result.is_err(), "nonexistent function should fail");
    assert!(
        result.unwrap_err().to_string().contains("not found"),
        "should say not found"
    );
}

// ═════════════════════════════════════════════════════════════════════
// fn_replace builtin
// ═════════════════════════════════════════════════════════════════════

#[test]
fn fn_replace_single_statement() {
    let (handle, prog) = setup_mutation_runtime(
        "+fn greet (name:String)->String\n  +return concat(\"Hello \", name)\n+end",
    );
    let result = unwrap_string(
        handle
            .execute_await(
                "fn_replace",
                &[
                    Value::string("greet.s1"),
                    Value::string("  +return concat(\"Hi \", name)"),
                ],
            )
            .unwrap(),
    );
    // Should succeed
    assert!(!result.is_empty(), "should return a message: {result}");
    // Verify the function was modified
    let p = prog.read().unwrap();
    let func = p.get_function("greet").expect("greet should still exist");
    // The body should have the replaced statement
    assert_eq!(func.body.len(), 1, "should still have 1 statement");
}

#[test]
fn fn_replace_empty_target_fails() {
    let (handle, _prog) = setup_mutation_runtime("+fn dummy ()->Int\n  +return 1\n+end");
    let result = handle.execute_await(
        "fn_replace",
        &[Value::string(""), Value::string("  +return 2")],
    );
    assert!(result.is_err(), "empty target should fail");
}

#[test]
fn fn_replace_empty_code_fails() {
    let (handle, _prog) = setup_mutation_runtime("+fn dummy ()->Int\n  +return 1\n+end");
    let result = handle.execute_await(
        "fn_replace",
        &[Value::string("dummy.s1"), Value::string("")],
    );
    assert!(result.is_err(), "empty code should fail");
}

#[test]
fn fn_replace_nonexistent_function_fails() {
    let (handle, _prog) = setup_mutation_runtime("");
    let result = handle.execute_await(
        "fn_replace",
        &[
            Value::string("nonexistent.s1"),
            Value::string("  +return 42"),
        ],
    );
    assert!(
        result.is_err(),
        "replacing in nonexistent function should fail"
    );
}

// ── library_reload ──

#[test]
fn library_reload_nonexistent_module_fails() {
    let (handle, _prog) = setup_mutation_runtime("");
    let result = handle.execute_await("library_reload", &[Value::string("NonExistentModule99999")]);
    assert!(result.is_err(), "should fail for nonexistent module");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("not found") || err.contains("could not"),
        "error should mention not found: {err}"
    );
}

#[test]
fn library_reload_empty_name_reloads_all() {
    let (handle, _prog) = setup_mutation_runtime("");
    // With empty name, it reloads all modules from the library dir.
    // This should succeed even if the library dir is empty.
    let result = handle.execute_await("library_reload", &[Value::string("")]);
    assert!(result.is_ok(), "should succeed with empty name: {result:?}");
    let msg = unwrap_string(result.unwrap());
    assert!(msg.contains("Reloaded"), "should report reloaded: {msg}");
}

// ── library_errors ──

#[test]
fn library_errors_no_errors() {
    let (handle, _rt) = setup_roadmap_runtime();
    let result = unwrap_string(handle.execute_await("library_errors", &[]).unwrap());
    assert_eq!(result, "No library errors.");
}

#[test]
fn library_errors_with_load_errors() {
    let (handle, _meta) = setup_roadmap_runtime();
    let rt = crate::shared_state::get_shared_runtime().unwrap();
    // Add some library load errors to the runtime state
    if let Ok(mut state) = rt.write() {
        state.library_load_errors = vec![
            ("BadModule".to_string(), "parse error on line 5".to_string()),
            (
                "BrokenMod".to_string(),
                "no +module declaration found".to_string(),
            ),
        ];
    }
    let result = unwrap_string(handle.execute_await("library_errors", &[]).unwrap());
    assert!(
        result.contains("Load errors (2):"),
        "should show load error count: {result}"
    );
    assert!(
        result.contains("BadModule: parse error on line 5"),
        "should show first error: {result}"
    );
    assert!(
        result.contains("BrokenMod: no +module declaration found"),
        "should show second error: {result}"
    );
}

#[test]
fn library_errors_with_general_errors() {
    let (handle, _meta) = setup_roadmap_runtime();
    let rt = crate::shared_state::get_shared_runtime().unwrap();
    if let Ok(mut state) = rt.write() {
        state.library_errors = vec!["failed to persist module `Foo`: disk full".to_string()];
    }
    let result = unwrap_string(handle.execute_await("library_errors", &[]).unwrap());
    assert!(
        result.contains("Errors this session (1):"),
        "should show session error count: {result}"
    );
    assert!(
        result.contains("failed to persist module `Foo`: disk full"),
        "should show the error: {result}"
    );
}

#[test]
fn library_errors_with_both_error_types() {
    let (handle, _meta) = setup_roadmap_runtime();
    let rt = crate::shared_state::get_shared_runtime().unwrap();
    if let Ok(mut state) = rt.write() {
        state.library_load_errors = vec![("FailedMod".to_string(), "syntax error".to_string())];
        state.library_errors = vec!["could not read library dir".to_string()];
    }
    let result = unwrap_string(handle.execute_await("library_errors", &[]).unwrap());
    assert!(
        result.contains("Load errors (1):"),
        "should show load errors: {result}"
    );
    assert!(
        result.contains("FailedMod: syntax error"),
        "should show load error detail: {result}"
    );
    assert!(
        result.contains("Errors this session (1):"),
        "should show session errors: {result}"
    );
    assert!(
        result.contains("could not read library dir"),
        "should show session error detail: {result}"
    );
}

// ── run_module_startups ──

#[test]
fn run_module_startups_no_modules() {
    let (handle, _prog) = setup_mutation_runtime("");
    let result = handle.execute_await("run_module_startups", &[]);
    assert!(result.is_ok(), "should succeed with no modules: {result:?}");
    let msg = unwrap_string(result.unwrap());
    assert!(
        msg.contains("no modules have startup blocks"),
        "should say no startups: {msg}"
    );
}

#[test]
fn run_module_startups_with_startup_block() {
    let source = r#"
+module Svc
+startup [io,async]
  +return "started"
+fn greet ()->String
  +return "hello"
"#;
    let (handle, _prog) = setup_mutation_runtime(source);
    let result = handle.execute_await("run_module_startups", &[]);
    assert!(result.is_ok(), "should succeed: {result:?}");
    let msg = unwrap_string(result.unwrap());
    assert!(
        msg.contains("executed 1 startup"),
        "should report 1 startup: {msg}"
    );
    assert!(
        msg.contains("Svc.startup"),
        "should mention Svc.startup: {msg}"
    );
}

#[test]
fn run_module_startups_sorted_alphabetically() {
    let source = r#"
+module Zebra
+startup [io,async]
  +return "z"
+module Alpha
+startup [io,async]
  +return "a"
"#;
    let (handle, _prog) = setup_mutation_runtime(source);
    let result = handle.execute_await("run_module_startups", &[]);
    assert!(result.is_ok(), "should succeed: {result:?}");
    let msg = unwrap_string(result.unwrap());
    assert!(
        msg.contains("executed 2 startup"),
        "should report 2 startups: {msg}"
    );
    // Alpha should come before Zebra in the output
    let alpha_pos = msg.find("Alpha.startup").expect("should contain Alpha");
    let zebra_pos = msg.find("Zebra.startup").expect("should contain Zebra");
    assert!(
        alpha_pos < zebra_pos,
        "Alpha should come before Zebra: {msg}"
    );
}

#[test]
fn run_module_startups_with_source_decl() {
    let source = r#"
+module Poller
+source heartbeat timer interval=5000 -> on_tick
+fn on_tick ()->String
  +return "tick"
"#;
    let (handle, _prog) = setup_mutation_runtime(source);
    let result = handle.execute_await("run_module_startups", &[]);
    assert!(result.is_ok(), "should succeed: {result:?}");
    let msg = unwrap_string(result.unwrap());
    // No startup blocks, but source registration should still be attempted
    // However with mock handle, the IoRequest won't be processed — that's fine,
    // the mock handle's send will succeed silently
    assert!(
        msg.contains("no modules have startup blocks") || msg.contains("source"),
        "should handle no startups or report source: {msg}"
    );
}

// ── query_startups ──

#[test]
fn query_startups_no_modules() {
    let (handle, _prog) = setup_mutation_runtime("");
    let result = handle.execute_await("query_startups", &[]);
    assert!(result.is_ok());
    let msg = unwrap_string(result.unwrap());
    assert!(
        msg.contains("No modules have startup or shutdown blocks"),
        "msg: {msg}"
    );
}

#[test]
fn query_startups_with_startup() {
    let source = r#"
+module Svc
+startup [io,async]
  +return "started"
+fn greet ()->String
  +return "hi"
"#;
    let (handle, _prog) = setup_mutation_runtime(source);
    let result = handle.execute_await("query_startups", &[]);
    assert!(result.is_ok());
    let msg = unwrap_string(result.unwrap());
    assert!(
        msg.contains("Svc: +startup [io,async]"),
        "should list Svc startup: {msg}"
    );
    assert!(
        msg.contains("1 statement"),
        "should show statement count: {msg}"
    );
}

#[test]
fn query_startups_with_both() {
    let source = r#"
+module Svc
+startup [io,async]
  +return "started"
+shutdown [io,async]
  +return "stopped"
+fn greet ()->String
  +return "hi"
"#;
    let (handle, _prog) = setup_mutation_runtime(source);
    let result = handle.execute_await("query_startups", &[]);
    assert!(result.is_ok());
    let msg = unwrap_string(result.unwrap());
    assert!(msg.contains("Svc: +startup"), "should list startup: {msg}");
    assert!(
        msg.contains("Svc: +shutdown"),
        "should list shutdown: {msg}"
    );
}

#[test]
fn query_startups_multiple_modules() {
    let source = r#"
+module Alpha
+startup [io,async]
  +return "a"
+module Beta
+startup [io,async]
  +return "b1"
  +return "b2"
"#;
    let (handle, _prog) = setup_mutation_runtime(source);
    let result = handle.execute_await("query_startups", &[]);
    assert!(result.is_ok());
    let msg = unwrap_string(result.unwrap());
    assert!(msg.contains("Alpha: +startup"), "should list Alpha: {msg}");
    assert!(msg.contains("Beta: +startup"), "should list Beta: {msg}");
    assert!(msg.contains("1 statement"), "Alpha has 1 statement: {msg}");
    assert!(msg.contains("2 statements"), "Beta has 2 statements: {msg}");
}
