use super::*;
use crate::{
    api::takeover_dispatch::{Host, Input},
    coroutine::{CoroutineHandle, IoRequest, Runtime},
    eval,
    memory_graph::{MemoryEmbedder, MemoryGraph},
    permissions::{AccessLevel, ModelPermissions, PermissionConfig, PermissionLevel},
    session::SessionMeta,
};
use std::collections::HashMap;

struct Provider {
    url: String,
    requests: Arc<Mutex<Vec<serde_json::Value>>>,
    server: tokio::task::JoinHandle<()>,
}
/// Every distinct Flow.write evaluation result any provider request carried (conversations share
/// history, so results repeat across requests). Negative tests must show the write
/// was actually attempted and refused, not silently skipped.
fn tool_results(provider: &Provider) -> Vec<String> {
    let mut results = Vec::new();
    for request in provider.requests.lock().unwrap().iter() {
        for message in request["messages"].as_array().unwrap() {
            let content = message["content"].to_string();
            // Only the Flow.write evaluations; finish acknowledgements are noise here.
            if message["role"] == "tool"
                && content.contains("eval Flow.write")
                && !results.contains(&content)
            {
                results.push(content);
            }
        }
    }
    results
}

fn evaluate_result(provider: &Provider) -> String {
    tool_results(provider).join("\n")
}

impl Drop for Provider {
    fn drop(&mut self) {
        self.server.abort();
    }
}

async fn provider(fail: bool) -> Provider {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let recorded = requests.clone();
    let router=axum::Router::new().route("/v1/chat/completions",axum::routing::post(move|axum::Json(body):axum::Json<serde_json::Value>| {
        let recorded=recorded.clone();
        async move {
            let (round, after_tool) = {
                let mut log = recorded.lock().unwrap();
                let after_tool = body["messages"].as_array().and_then(|m| m.last()).is_some_and(|m| m["role"] == "tool");
                log.push(body);
                (log.len(), after_tool)
            };
            if fail { return (StatusCode::BAD_REQUEST,axum::Json(json!({"error":{"message":"fixture provider rejection"}}))).into_response(); }
            // Stateless per turn: each turn evaluates Flow.write once, then finishes
            // alone (finish must be the sole call in its response).
            let call = if !after_tool {
                json!({"id":format!("write-{round}"),"type":"function","function":{"name":"evaluate","arguments":json!({"language":"python","function":"Flow.write","input":"'one'"}).to_string()}})
            } else {
                json!({"id":format!("finish-{round}"),"type":"function","function":{"name":"finish","arguments":"{}"}})
            };
            axum::Json(json!({"choices":[{"index":0,"message":{"role":"assistant","content":"stored","tool_calls":[call]},
                "finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}})).into_response()
        }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    Provider {
        url,
        requests,
        server,
    }
}

fn permissions(actual: PermissionLevel, context: PermissionLevel) -> Arc<PermissionConfig> {
    let mut config = PermissionConfig::default();
    config.groups.insert("flow".into(), vec!["Flow".into()]);
    config.groups.insert("records".into(), vec![]);
    for (model, level) in [("actual", actual), ("context", context)] {
        config.model.insert(
            model.into(),
            ModelPermissions {
                group_perms: HashMap::from([
                    ("flow".into(), PermissionLevel::Write),
                    ("records".into(), level),
                ]),
                opencode: false,
            },
        );
    }
    Arc::new(config)
}

struct Harness {
    _directory: tempfile::TempDir,
    graph: Arc<MemoryGraph>,
    source: PollSource,
    _telegram: Server,
    program: Arc<tokio::sync::RwLock<crate::ast::Program>>,
    meta: crate::session::SharedMeta,
    runtime: crate::session::SharedRuntime,
    io: Arc<Runtime>,
    receiver: Option<tokio::sync::mpsc::Receiver<IoRequest>>,
    store: RecordStore,
}

impl Harness {
    async fn new(text: &str, profile: bool) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("db.lbug");
        drop(RecordStore::graph_for_test(&path).unwrap());
        let grant_file = directory.path().join("grants.toml");
        std::fs::write(&grant_file,"version=1\n[[grants]]\nid='owner'\ngeneration=1\nnamespace='personal'\nprincipal='telegram:user:7'\ncontext='telegram:7'\npermission_group='records'\nrights=['read','write']\n").unwrap();
        let grants = crate::memory_graph::records::grants::GrantConfig::read(&grant_file).unwrap();
        crate::memory_graph::records::grants::provision::apply(&path, &grants, false).unwrap();
        let graph = Arc::new(MemoryGraph::open(&path).unwrap());
        let store = RecordStore::configured(graph.as_ref().clone(), grant_file).unwrap();
        let mut raw: serde_json::Value = serde_json::from_str(&event(1, text)).unwrap();
        if profile {
            raw["permission_profile"] = json!("context");
        }
        let telegram = start_server(99, vec![page(&[raw.to_string()])]).await;
        let mut source_config = config(&telegram, "pipeline", 99);
        source_config.handler = "Flow.entry".into();
        source_config.routes[0].reply_handler = "Flow.reply".into();
        let source = PollSource::fixture(
            source_config,
            RecordStore::initialize_for_test(graph.as_ref().clone()).unwrap(),
            "intake",
        )
        .unwrap();
        let lowered = crate::python_frontend::lower_module(
            include_str!("../../../tests/fixtures/ingress/pipeline.py"),
            "Flow",
        )
        .unwrap();
        let mut program = crate::ast::Program::default();
        program.require_modules = true;
        for operation in &lowered.operations {
            crate::validator::apply_and_validate(&mut program, operation).unwrap();
        }
        program.rebuild_function_index();
        crate::python_frontend::validate_runtime_arguments(&lowered.operations, &program).unwrap();
        crate::python_frontend::print_module(&program, "Flow").unwrap();
        let (io, receiver) = Runtime::new();
        Self {
            _directory: directory,
            graph,
            source,
            _telegram: telegram,
            program: Arc::new(tokio::sync::RwLock::new(program)),
            meta: Arc::new(Mutex::new(SessionMeta::new())),
            runtime: Default::default(),
            io: Arc::new(io),
            receiver: Some(receiver),
            store,
        }
    }

    fn serve(
        &mut self,
        provider: &Provider,
        permissions: Arc<PermissionConfig>,
    ) -> tokio::task::JoinHandle<()> {
        let mut receiver = self.receiver.take().unwrap();
        let io = self.io.clone();
        let program = self.program.clone();
        let runtime = self.runtime.clone();
        let meta = self.meta.clone();
        let graph = self.graph.clone();
        let store = self.store.clone();
        let url = provider.url.clone();
        tokio::spawn(async move {
            while let Some(request) = receiver.recv().await {
                match request {
                    IoRequest::Spawn {
                        function_name,
                        args,
                        origin,
                        reply,
                    } => crate::spawn_dispatch::dispatch(
                        crate::spawn_dispatch::Context {
                            io: io.clone(),
                            program: program.clone(),
                            runtime: runtime.clone(),
                            meta: meta.clone(),
                        },
                        function_name,
                        args,
                        origin,
                        reply,
                    ),
                    IoRequest::LlmTakeover {
                        context,
                        message,
                        attachment,
                        source_metadata,
                        reply_fn,
                        reply_arg,
                        permission_model,
                        origin,
                        reply,
                    } => crate::api::takeover_dispatch::dispatch(
                        Host {
                            meta: meta.clone(),
                            program: program.clone(),
                            runtime: runtime.clone(),
                            url: url.clone(),
                            model: "actual".into(),
                            key: None,
                            sender: io.io_sender(),
                            tasks: io.task_registry.clone(),
                            snapshots: io.snapshot_registry.clone(),
                            opencode_lock: Arc::new(tokio::sync::Mutex::new(())),
                            opencode_dir: ".".into(),
                            training: None,
                            log: None,
                            access: AccessLevel::Full,
                            permissions: permissions.clone(),
                            save: None,
                            native: true,
                            graph: graph.clone(),
                            embedder: Arc::new(MemoryEmbedder::deterministic()),
                            record_store: Some(store.clone()),
                        },
                        Input {
                            context,
                            message,
                            attachment,
                            metadata: source_metadata,
                            reply_fn,
                            reply_arg,
                            permission_model,
                            origin,
                        },
                        reply,
                    ),
                    other => io.handle_io(other).await,
                }
            }
        })
    }

    async fn entry(&self, attested: bool) -> String {
        let delivery = self.source.receive_once().await.unwrap().events.remove(0);
        let admitted = self.source.admit(delivery).await.unwrap();
        let handle = CoroutineHandle::new(self.io.io_sender()).with_origin(if attested {
            admitted.origin()
        } else {
            None
        });
        let program = self.program.read().await.clone();
        let payload = admitted.payload;
        tokio::task::spawn_blocking(move || {
            let input = crate::parser::parse_single_expr(&serde_json::to_string(&payload).unwrap())
                .unwrap();
            let value = eval::eval_async_function(&program, "Flow.entry", &input, handle).unwrap();
            let eval::Value::String(value) = value else {
                panic!()
            };
            value.as_ref().clone()
        })
        .await
        .unwrap()
    }

    async fn wait_workers(&self, count: usize) {
        tokio::time::timeout(std::time::Duration::from_secs(30), async {
            loop {
                let done = {
                    let tasks = self.io.task_registry.lock().unwrap();
                    tasks.len() == count
                        && tasks.values().all(|info| {
                            matches!(
                                info.status,
                                crate::coroutine::WaitReason::Completed(_)
                                    | crate::coroutine::WaitReason::Failed(_)
                            )
                        })
                };
                if done {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
    }

    fn stored(&self) -> bool {
        self.store
            .observe_row_for_test("personal", "proof-row")
            .is_some()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ingress_pipeline_actual_spawn_takeover_commits_once() {
    let provider = provider(false).await;
    let mut harness = Harness::new("duplicate", false).await;
    let service = harness.serve(
        &provider,
        permissions(PermissionLevel::Write, PermissionLevel::Write),
    );
    assert_eq!(harness.entry(true).await, "queued");
    harness.wait_workers(2).await;
    assert!(harness.stored());
    let row = harness
        .store
        .observe_row_for_test("personal", "proof-row")
        .unwrap();
    assert_eq!(row.revision, "1");
    assert_eq!(row.fields[0].value, "one");
    assert!(
        harness
            .store
            .observe_receipt_for_test(
                "personal",
                "telegram:user:7",
                "telegram:7",
                "owner",
                "fixture-write"
            )
            .is_some()
    );
    assert_eq!(harness.source.state.status_for_test(1), "turn_finished");
    // One claimed turn wrote. The duplicate worker either lost the admission race
    // (refused before the provider) or found the origin already spent and ran as
    // origin-free legacy work, whose write was denied. Never a second stored write.
    let results = tool_results(&provider);
    assert_eq!(
        results.iter().filter(|r| r.contains("stored")).count(),
        1,
        "{results:?}"
    );
    assert!(
        results
            .iter()
            .all(|r| r.contains("stored") || r.contains("record_access_denied")),
        "{results:?}"
    );
    let history = serde_json::to_string(&*harness.meta.lock().unwrap()).unwrap();
    assert!(!history.contains("fixture-token"));
    assert!(!history.contains("InvocationOrigin"));
    for request in provider.requests.lock().unwrap().iter() {
        assert!(!request.to_string().contains("fixture-token"));
    }
    service.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ingress_pipeline_identical_unattested_json_cannot_bind() {
    let provider = provider(false).await;
    let mut harness = Harness::new("store", false).await;
    let service = harness.serve(
        &provider,
        permissions(PermissionLevel::Write, PermissionLevel::Write),
    );
    harness.entry(false).await;
    harness.wait_workers(1).await;
    assert!(!harness.stored());
    assert_eq!(provider.requests.lock().unwrap().len(), 2);
    assert!(evaluate_result(&provider).contains("record_access_denied"));
    service.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ingress_pipeline_actual_and_context_ceiling_both_restrict() {
    for (actual, context) in [
        (PermissionLevel::None, PermissionLevel::Write),
        (PermissionLevel::Write, PermissionLevel::Read),
    ] {
        let provider = provider(false).await;
        let mut harness = Harness::new("store", true).await;
        let service = harness.serve(&provider, permissions(actual, context));
        harness.entry(true).await;
        harness.wait_workers(1).await;
        assert!(!harness.stored());
        assert_eq!(provider.requests.lock().unwrap().len(), 2);
        assert!(evaluate_result(&provider).contains("record_access_denied"));
        service.abort();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ingress_pipeline_provider_failure_is_unknown() {
    let provider = provider(true).await;
    let mut harness = Harness::new("store", false).await;
    let service = harness.serve(
        &provider,
        permissions(PermissionLevel::Write, PermissionLevel::Write),
    );
    harness.entry(true).await;
    harness.wait_workers(1).await;
    assert!(!harness.stored());
    assert_eq!(
        harness
            .source
            .state
            .pending(harness.source.epoch)
            .unwrap()
            .unknown,
        1
    );
    service.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ingress_pipeline_mismatched_binding_refused_before_side_effects() {
    let provider = provider(false).await;
    let mut harness = Harness::new("mismatch", false).await;
    let service = harness.serve(
        &provider,
        permissions(PermissionLevel::Write, PermissionLevel::Write),
    );
    harness.entry(true).await;
    harness.wait_workers(1).await;
    assert!(!harness.stored());
    assert!(provider.requests.lock().unwrap().is_empty());
    // Refused before the durable claim and before any conversation is touched.
    assert_eq!(harness.source.state.status_for_test(1), "started");
    let meta = harness.meta.lock().unwrap();
    assert!(meta.conversations.get("telegram:8").is_none());
    assert!(meta.conversations.get("telegram:7").is_none());
    drop(meta);
    let failure: Vec<String> = harness
        .io
        .task_registry
        .lock()
        .unwrap()
        .values()
        .map(|task| format!("{}", task.status))
        .collect();
    assert_eq!(failure, ["failed"]);
    service.abort();
}

async fn worker_result(harness: &Harness, function: &str, handle: CoroutineHandle) -> String {
    let program = harness.program.read().await.clone();
    let function = function.to_string();
    tokio::task::spawn_blocking(move || {
        let input = crate::parser::parse_single_expr("\"late\"").unwrap();
        match eval::eval_async_function(&program, &function, &input, handle).unwrap() {
            eval::Value::String(value) => value.as_ref().clone(),
            other => panic!("unexpected {other:?}"),
        }
    })
    .await
    .unwrap()
}

/// B1: a record-bearing handle minted by the real verified turn, carried into a
/// delayed worker after the turn ended, is denied (revocation), and no row exists.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ingress_pipeline_escaped_handle_is_revoked_after_turn() {
    let provider = provider(false).await;
    let mut harness = Harness::new("store", false).await;
    let service = harness.serve(
        &provider,
        permissions(PermissionLevel::Write, PermissionLevel::Write),
    );
    harness.entry(true).await;
    harness.wait_workers(1).await;
    assert!(harness.stored());
    assert_eq!(harness.source.state.status_for_test(1), "turn_finished");
    let escaped = crate::api::record_context::escaped::take(&harness.store);
    assert!(
        !escaped.is_empty(),
        "the verified turn must have minted a bound handle"
    );
    for handle in escaped {
        assert_eq!(
            worker_result(&harness, "Flow.write_late", handle).await,
            "record_access_denied"
        );
    }
    assert!(
        harness
            .store
            .observe_row_for_test("personal", "late-row")
            .is_none()
    );
    service.abort();
}

/// Non-blocking 1: the first takeover consumes the origin; a descendant spawned
/// afterwards to another context runs as origin-free legacy work without records.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ingress_pipeline_spent_origin_descendant_runs_legacy() {
    let provider = provider(false).await;
    let mut harness = Harness::new("descendant", false).await;
    let service = harness.serve(
        &provider,
        permissions(PermissionLevel::Write, PermissionLevel::Write),
    );
    harness.entry(true).await;
    harness.wait_workers(2).await;
    assert_eq!(harness.source.state.status_for_test(1), "turn_finished");
    let results = tool_results(&provider);
    assert_eq!(
        results.iter().filter(|r| r.contains("stored")).count(),
        1,
        "{results:?}"
    );
    assert_eq!(
        results
            .iter()
            .filter(|r| r.contains("record_access_denied"))
            .count(),
        1,
        "{results:?}"
    );
    // The descendant's turn ran (provider reached for telegram:9), not refused.
    assert!(
        harness
            .meta
            .lock()
            .unwrap()
            .conversations
            .get("telegram:9")
            .is_some()
    );
    let statuses: Vec<String> = harness
        .io
        .task_registry
        .lock()
        .unwrap()
        .values()
        .map(|task| format!("{}", task.status))
        .collect();
    assert!(
        statuses.iter().all(|status| status == "done"),
        "{statuses:?}"
    );
    service.abort();
}

/// B2: a verified turn dispatched with a narrowing profile keeps it even when a
/// concurrent origin-less admin dispatch resets the shared conversation field
/// before the verified turn acquires the context lock.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ingress_pipeline_turn_profile_survives_concurrent_profile_reset() {
    let provider = provider(false).await;
    let mut harness = Harness::new("store", true).await;
    let service = harness.serve(
        &provider,
        permissions(PermissionLevel::Write, PermissionLevel::Execute),
    );
    let lock = crate::api::llm_handlers::takeover_context_lock("telegram:7");
    let held = lock.lock().await;
    harness.entry(true).await;
    let profile = |harness: &Harness| {
        harness
            .meta
            .lock()
            .unwrap()
            .conversations
            .get("telegram:7")
            .map(|c| c.permission_model.clone())
    };
    let wait = |want: Option<Option<String>>| {
        let harness = &harness;
        async move {
            tokio::time::timeout(std::time::Duration::from_secs(30), async {
                while profile(harness) != want {
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                }
            })
            .await
            .unwrap()
        }
    };
    // B has claimed the event and pinned "context", and now waits on the lock.
    wait(Some(Some("context".into()))).await;
    assert_eq!(harness.source.state.status_for_test(1), "takeover_started");
    let (reply, admin) = tokio::sync::oneshot::channel();
    harness
        .io
        .io_sender()
        .send(IoRequest::LlmTakeover {
            context: "telegram:7".into(),
            message: "admin".into(),
            attachment: None,
            source_metadata: Some(r#"{"speaker_id":"telegram:user:1"}"#.into()),
            reply_fn: None,
            reply_arg: None,
            permission_model: None,
            origin: None,
            reply,
        })
        .await
        .unwrap();
    wait(Some(None)).await;
    drop(held);
    harness.wait_workers(1).await;
    let _ = tokio::time::timeout(std::time::Duration::from_secs(30), admin).await;
    assert!(!harness.stored());
    let results = tool_results(&provider);
    assert!(
        !results.is_empty() && results.iter().all(|r| r.contains("record_access_denied")),
        "{results:?}"
    );
    service.abort();
}

/// Burn-on-mismatch: the first takeover attempt spends the origin even when its
/// binding is refused. A later, correctly bound call from the same handle runs as
/// origin-free legacy work (write denied, no row), and the event — never claimed —
/// ends `unknown` after the source is reopened.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ingress_pipeline_mismatch_burns_origin_for_later_calls() {
    let provider = provider(false).await;
    let mut harness = Harness::new("burn", false).await;
    let service = harness.serve(
        &provider,
        permissions(PermissionLevel::Write, PermissionLevel::Write),
    );
    let delivery = harness
        .source
        .receive_once()
        .await
        .unwrap()
        .events
        .remove(0);
    let admitted = harness.source.admit(delivery).await.unwrap();
    let handle = CoroutineHandle::new(harness.io.io_sender()).with_origin(admitted.origin());
    let payload = serde_json::to_string(&admitted.payload).unwrap();
    let program = harness.program.read().await.clone();
    let call = |function: &'static str| {
        let (program, handle, payload) = (program.clone(), handle.clone(), payload.clone());
        tokio::task::spawn_blocking(move || {
            let input = crate::parser::parse_single_expr(&payload).unwrap();
            eval::eval_async_function(&program, function, &input, handle)
        })
    };
    let first = call("Flow.misbound").await.unwrap();
    assert!(
        first
            .unwrap_err()
            .to_string()
            .contains("verified ingress admission refused")
    );
    assert!(provider.requests.lock().unwrap().is_empty());
    assert!(call("Flow.worker").await.unwrap().is_ok());
    assert!(!harness.stored());
    let results = tool_results(&provider);
    assert!(
        !results.is_empty() && results.iter().all(|r| r.contains("record_access_denied")),
        "{results:?}"
    );
    assert_eq!(harness.source.state.status_for_test(1), "started");
    let mut reopened_config = config(&harness._telegram, "pipeline", 99);
    reopened_config.handler = "Flow.entry".into();
    reopened_config.routes[0].reply_handler = "Flow.reply".into();
    let reopened = PollSource::fixture(
        reopened_config,
        RecordStore::initialize_for_test(harness.graph.as_ref().clone()).unwrap(),
        "intake",
    )
    .unwrap();
    assert_eq!(reopened.state.status_for_test(1), "unknown");
    service.abort();
}
