//! Offline mock-provider takeover fixtures (#85). No live provider, gateway,
//! FastEmbed model, user library, contexts directory or memory database.
use crate::session::{SessionMeta, SharedMeta};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
#[path = "takeover_logging_tests.rs"]
mod logging;
#[path = "takeover_restricted_tests.rs"]
mod restricted;
#[path = "takeover_cache_tests.rs"]
mod cache;
#[path = "takeover_workspace_tests.rs"]
mod workspace;

const ADMIN: &str = "telegram:user:100000001";

fn call(id: &str, name: &str, arguments: Value) -> Value {
    json!({"id":id,"type":"function","function":{"name":name,"arguments":arguments.to_string()}})
}

fn response(content: &str, calls: Vec<Value>) -> Value {
    json!({"choices":[{"index":0,"message":{"role":"assistant","content":content,"tool_calls":calls},
        "finish_reason":if calls.is_empty() { "stop" } else { "tool_calls" }}],
        "usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}})
}

fn tool(id: &str, name: &str, arguments: Value) -> Value {
    response("", vec![call(id, name, arguments)])
}

fn python(id: &str, name: &str, module: &str, source: &str) -> Value {
    tool(id, name, json!({"language":"python","module":module,"source":source}))
}

const ADD_TWO: &str = "def add_two(x: int) -> int:\n    first: int = x + 1\n    second: int = first + 1\n    return second\n";
const TEST_ADD_TWO: &str = "def test_add_two() -> None:\n    assert add_two(3) == 5\n";

struct Harness {
    meta: SharedMeta,
    program: Arc<tokio::sync::RwLock<crate::ast::Program>>,
    graph: Arc<crate::memory_graph::MemoryGraph>,
    permissions: Arc<crate::permissions::PermissionConfig>,
    native_tools: bool,
    log_file: Option<Arc<tokio::sync::Mutex<tokio::fs::File>>>,
    /// The generating model (its limits decide the requested cache lifetime).
    model: &'static str,
    /// Workspace site of the node (`None`: no workspace tools).
    workspaces: Option<Arc<crate::api::native_tools::workspace::Site>>,
}

struct Provider {
    url: String,
    requests: Arc<Mutex<Vec<Value>>>,
    server: tokio::task::JoinHandle<()>,
}

impl Drop for Provider {
    fn drop(&mut self) {
        self.server.abort();
    }
}

/// The legacy client streams; the native takeover driver does not.
fn sse_or_json(output: &Value, streaming: bool) -> axum::response::Response {
    use axum::response::IntoResponse;
    if !streaming {
        return axum::Json(output.clone()).into_response();
    }
    let delta = json!({"content": output["choices"][0]["message"]["content"]});
    let chunk = json!({"choices":[{"index":0,"delta":delta,"finish_reason":null}]});
    // Legacy fixtures answer without tool calls, so this is "stop" unless a test
    // scripts another terminal reason (a refusal).
    let end = json!({"choices":[{"index":0,"delta":{},"finish_reason":output["choices"][0]["finish_reason"]}],"usage":output["usage"]});
    (
        [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
        format!("data: {chunk}\n\ndata: {end}\n\ndata: [DONE]\n\n"),
    )
        .into_response()
}

async fn provider(outputs: Vec<Value>) -> Provider {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let recorded = requests.clone();
    let app = axum::Router::new().route(
        "/v1/chat/completions",
        axum::routing::post(move |axum::Json(body): axum::Json<Value>| {
            let recorded = recorded.clone();
            let outputs = outputs.clone();
            async move {
                let mut requests = recorded.lock().unwrap();
                let output = outputs
                    .get(requests.len())
                    .cloned()
                    .unwrap_or_else(|| response("unexpected extra request", vec![]));
                let streaming = body["stream"] == true;
                requests.push(body);
                // `{"http_status": N}` scripts a provider failure (400: not retried).
                if let Some(status) = output["http_status"].as_u64() {
                    use axum::response::IntoResponse;
                    let status = axum::http::StatusCode::from_u16(status as u16).unwrap();
                    return (status, r#"{"error":{"message":"scripted provider failure"}}"#).into_response();
                }
                sse_or_json(&output, streaming)
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    Provider { url, requests, server }
}

impl Harness {
    fn new(native_tools: bool) -> Self {
        Self {
            meta: Arc::new(Mutex::new(SessionMeta::new())),
            // Production enables the module requirement and with it the
            // `!done` test gate; the fixtures must exercise the same gate.
            program: Arc::new(tokio::sync::RwLock::new({
                let mut program = crate::ast::Program::default();
                program.require_modules = true;
                program
            })),
            graph: Arc::new(crate::memory_graph::MemoryGraph::in_memory().unwrap()),
            permissions: Arc::new(crate::permissions::PermissionConfig::default()),
            native_tools,
            log_file: None,
            model: "takeover-fixture",
            workspaces: None,
        }
    }

    async fn turn(&self, provider: &Provider, context: &str, speaker: &str, text: &str) -> String {
        let (io_tx, _io_rx) = tokio::sync::mpsc::channel(8);
        self.turn_with_io(provider, io_tx, context, speaker, text).await
    }

    async fn turn_with_io(
        &self,
        provider: &Provider,
        io_tx: tokio::sync::mpsc::Sender<crate::coroutine::IoRequest>,
        context: &str,
        speaker: &str,
        text: &str,
    ) -> String {
        let metadata = json!({"speaker_id": speaker}).to_string();
        let future = super::super::handle_llm_takeover(
            context.into(),
            text.into(),
            None,
            Some(metadata),
            self.graph.clone(),
            Arc::new(crate::memory_graph::MemoryEmbedder::deterministic()),
            Some("FixtureBot.send_reply".into()),
            Some("42".into()),
            self.meta.clone(),
            self.program.clone(),
            Default::default(),
            &provider.url,
            self.model,
            None,
            io_tx,
            Default::default(),
            Default::default(),
            Arc::new(tokio::sync::Mutex::new(())),
            ".".into(),
            None,
            crate::permissions::AccessLevel::Full,
            self.permissions.clone(),
            None,
            self.log_file.clone(),
            self.native_tools,
            self.workspaces.clone(),
        );
        tokio::time::timeout(std::time::Duration::from_secs(30), future)
            .await
            .expect("takeover fixture timed out")
            .expect("takeover turn failed")
    }

    fn set_profile(&self, context: &str, profile: Option<&str>) {
        let mut meta = self.meta.lock().unwrap();
        meta.conversations.get_or_create(context).permission_model = profile.map(str::to_string);
    }
}

fn system_text(request: &Value) -> String {
    request["messages"][0]["content"].as_str().unwrap_or_default().to_string()
}

fn all_text(request: &Value) -> String {
    request["messages"].to_string()
}

#[tokio::test]
async fn native_takeover_python_mutate_test_evaluate_finish_delivers_prose() {
    let context = "telegram:100000001";
    let module = "TakeoverPyLifecycle";
    let dir = crate::context_prompt::contexts_dir().unwrap();
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        crate::context_prompt::identity_path(&dir, context),
        "You are TakeoverFixtureIdentity for this admin DM.",
    )
    .unwrap();
    let harness = Harness::new(true);
    let embedder = crate::memory_graph::MemoryEmbedder::deterministic();
    harness
        .graph
        .remember_canonical(&crate::memory_graph::CanonicalDraft {
            content: "CanonicalFixturePreference: answer tersely".into(),
            context_id: context.into(),
            principal_id: ADMIN.into(),
            embedding: embedder.embed_passages(&["answer tersely".into()]).unwrap().remove(0),
            scope: crate::memory_graph::MemoryScope::Context,
            created_at_ms: 1,
        })
        .unwrap();
    let provider = provider(vec![
        python("mutate", "mutate", module, ADD_TWO),
        tool("early", "finish", json!({})),
        python("test", "test", module, TEST_ADD_TWO),
        tool(
            "evaluate",
            "evaluate",
            json!({"language":"python","function":format!("{module}.add_two"),"input":"3"}),
        ),
        response("add_two(3) is 5, tested.", vec![call("finish", "finish", json!({}))]),
    ])
    .await;
    let reply = harness.turn(&provider, context, ADMIN, "Build add_two").await;
    let requests = provider.requests.lock().unwrap().clone();
    assert_eq!(requests.len(), 5, "premature finish must not end the turn: {}",
        requests.last().map(all_text).unwrap_or_default());
    assert_eq!(reply, "add_two(3) is 5, tested.");
    let system = system_text(&requests[0]);
    assert!(system.contains("Fixed native tool protocol"));
    assert!(system.contains("Conversation turn mechanics"));
    assert!(system.contains("TakeoverFixtureIdentity"), "identity file must compose");
    let speaker = all_text(&requests[0]);
    assert!(speaker.contains("Spawn background agents (`!agent`): no"), "native turns cannot spawn agents");
    assert!(speaker.contains("Change the runtime itself (`!opencode`): no"));
    assert!(speaker.contains("Add or modify code: yes"));
    assert!(!system.contains("NEVER use <tool_call>"), "no legacy language prompt");
    assert_eq!(requests[0]["tools"].as_array().unwrap().len(), 10);
    assert!(requests.iter().all(|r| r.get("temperature").is_none()));
    assert!(requests.iter().all(|r| r["model"] == "takeover-fixture"));
    let first = all_text(&requests[0]);
    assert!(first.contains("CanonicalFixturePreference"), "canonical memory injected");
    assert!(first.contains("Build add_two"));
    // Premature finish was refused with the test obligation, then satisfied.
    let after_early = all_text(&requests[2]);
    assert!(after_early.contains("\"tool_call_id\":\"early\""));
    assert!(after_early.contains("Cannot accept !done: 1 untested functions"), "{after_early}");
    assert!(all_text(&requests[4]).contains('5'));

    let program = harness.program.read().await;
    assert!(program.modules.iter().any(|m| m.name == module));
    drop(program);
    let history = harness
        .graph
        .conversation_history(ADMIN, context, 100, i64::MAX)
        .unwrap();
    assert!(history.contains("Build add_two") && history.contains("add_two(3) is 5, tested."), "{history}");

    // Session round-trip (restart): complete native batches survive and are
    // replayed to the provider on the next turn of the same context.
    let saved = serde_json::to_string(&harness.meta.lock().unwrap().conversations).unwrap();
    let restored = Harness::new(true);
    restored.meta.lock().unwrap().conversations = serde_json::from_str(&saved).unwrap();
    *restored.program.write().await = harness.program.read().await.clone();
    let next = provider_after_restart().await;
    let reply = restored.turn(&next, context, ADMIN, "What did you build?").await;
    assert_eq!(reply, "I built add_two.");
    let replay = all_text(&next.requests.lock().unwrap()[0]);
    for id in ["mutate", "early", "test", "evaluate", "finish"] {
        assert!(replay.contains(&format!("\"tool_call_id\":\"{id}\"")), "{id} missing: {replay}");
    }
    assert!(replay.contains("Build add_two") && replay.contains("What did you build?"));
}

async fn provider_after_restart() -> Provider {
    provider(vec![response("I built add_two.", vec![call("done", "finish", json!({}))])]).await
}

#[tokio::test]
async fn native_takeover_exhaustion_is_reported_truthfully() {
    let context = "telegram:group:-100000085";
    let module = "TakeoverPyExhausted";
    let late = "TakeoverPyFinalRoundLate";
    let harness = Harness::new(true);
    let provider = provider(vec![
        response(
            "<iteration_budget>1</iteration_budget>Starting.",
            vec![call("mutate", "mutate", json!({"language":"python","module":module,"source":ADD_TWO}))],
        ),
        // The final status round ignores the prose-only instruction and tries
        // to mutate again; it must not execute (L2).
        response(
            "I wrote add_two but it is untested.",
            vec![call("late", "mutate", json!({"language":"python","module":late,"source":ADD_TWO}))],
        ),
        response("Picking up again.", vec![call("done", "finish", json!({}))]),
    ])
    .await;
    let reply = harness.turn(&provider, context, ADMIN, "Build it").await;
    assert!(reply.starts_with("I wrote add_two but it is untested."), "{reply}");
    assert!(reply.contains("before the task was confirmed complete"), "{reply}");
    assert!(!reply.contains("iteration_budget") && !reply.contains("tool_call"));
    {
        let requests = provider.requests.lock().unwrap();
        assert_eq!(requests.len(), 2, "estimate 1 plus one final status round");
        assert!(all_text(&requests[1]).contains("no tools are available in this final round"));
        assert!(requests[1].get("tools").is_none(), "final round sends no tool schemas");
        assert_eq!(requests[1]["tool_choice"], "none");
    }
    let program = harness.program.read().await;
    assert!(program.modules.iter().any(|m| m.name == module));
    assert!(!program.modules.iter().any(|m| m.name == late), "final-round call executed");
    drop(program);

    // L1: the incomplete-turn note is reply-only; it is not persisted as an
    // assistant turn, so the next turn's request does not carry it.
    let reply = harness.turn(&provider, context, ADMIN, "Continue").await;
    assert_eq!(reply, "Picking up again.");
    let next = all_text(&provider.requests.lock().unwrap()[2]);
    assert!(!next.contains("task completion is unconfirmed"), "{next}");
    assert!(!next.contains("before the task was confirmed complete"));
    assert!(next.contains("I wrote add_two but it is untested."));
    assert!(!next.contains("\"tool_call_id\":\"late\""), "skipped call must not enter history");
}

#[tokio::test]
async fn native_takeover_double_empty_response_terminates_with_the_existing_error() {
    let harness = Harness::new(true);
    let provider = provider(vec![response("", vec![]), response("", vec![])]).await;
    let reply = harness.turn(&provider, "telegram:100000001", ADMIN, "Hi").await;
    assert_eq!(reply, super::super::llm_handlers::EMPTY_RESPONSE_ERROR);
    assert_eq!(provider.requests.lock().unwrap().len(), 2);
}

/// Without `--native-tools` nothing changes for guests and execute-only speakers: they
/// keep the legacy `<code>` protocol. (With the flag they are native; see `restricted`.)
#[tokio::test]
async fn guest_and_non_write_takeover_turns_stay_legacy_without_the_native_flag() {
    let mut harness = Harness::new(false);
    harness.permissions = restricted::permissions();
    let guest = provider(vec![response("Hallo!\n<code>!done</code>", vec![])]).await;
    let reply = harness.turn(&guest, restricted::GUEST_DM, restricted::GUEST, "Hi").await;
    assert_eq!(reply, "Hallo!");
    let context = "telegram:group:-100000086";
    harness.set_profile(context, Some("viewer"));
    let viewer = provider(vec![response("Nur Lesen.\n<code>!done</code>", vec![])]).await;
    let reply = harness.turn(&viewer, context, "telegram:user:100000778", "Hi").await;
    assert_eq!(reply, "Nur Lesen.");
    for request in guest.requests.lock().unwrap().iter().chain(viewer.requests.lock().unwrap().iter()) {
        assert!(request.get("tools").is_none(), "legacy turns expose no native tools");
        assert!(!system_text(request).contains("Fixed native tool protocol"));
    }
}

#[tokio::test]
async fn takeover_without_native_flag_is_unchanged_legacy() {
    let harness = Harness::new(false);
    let provider = provider(vec![response("Legacy reply.\n<code>!done</code>", vec![])]).await;
    let reply = harness.turn(&provider, "telegram:100000001", ADMIN, "Hi").await;
    assert_eq!(reply, "Legacy reply.");
    let requests = provider.requests.lock().unwrap();
    assert!(requests[0].get("tools").is_none());
    assert!(!system_text(&requests[0]).contains("Fixed native tool protocol"));
}

/// Images already in the conversation reach the native request as image parts.
/// (Seeded into history rather than passed as a new attachment so the test does
/// not write the content-addressed store under the user's data directory.)
#[tokio::test]
async fn native_takeover_carries_conversation_images_to_the_provider() {
    let context = "telegram:group:-100000087";
    let harness = Harness::new(true);
    let png = b"\x89PNG\r\n\x1a\nfixture".to_vec();
    let image = crate::attachment::Attachment::from_bytes(png, "image/png", "fixture.png").unwrap();
    harness
        .meta
        .lock()
        .unwrap()
        .conversations
        .get_or_create(context)
        .push_user_with_attachment("look at this", image);
    let provider = provider(vec![response("A fixture image.", vec![call("done", "finish", json!({}))])]).await;
    let reply = harness.turn(&provider, context, ADMIN, "What is it?").await;
    assert_eq!(reply, "A fixture image.");
    let request = all_text(&provider.requests.lock().unwrap()[0]);
    assert!(request.contains("image_url") && request.contains("data:image/png;base64"), "{request}");
}

// ── Native memory tools (#87) ──────────────────────────────────────────────
//
// The native memory tools dispatch the turn-bound builtins through the same
// coroutine runtime, so these fixtures service the IO requests with an in-memory
// Ladybug graph and the deterministic embedder, exactly as main.rs does.

fn spawn_memory_io(
    graph: Arc<crate::memory_graph::MemoryGraph>,
    embedder: Arc<crate::memory_graph::MemoryEmbedder>,
) -> tokio::sync::mpsc::Sender<crate::coroutine::IoRequest> {
    use crate::coroutine::IoRequest;
    let (tx, mut rx) = tokio::sync::mpsc::channel::<IoRequest>(32);
    let worker_tx = tx.clone();
    tokio::spawn(async move {
        while let Some(request) = rx.recv().await {
            let graph = graph.clone();
            let embedder = embedder.clone();
            let sender = worker_tx.clone();
            match request {
                IoRequest::MemoryInspect {
                    context_id,
                    principal_id,
                    query,
                    reply,
                } => {
                    let result = tokio::task::spawn_blocking(move || {
                        let embedding = if query.trim().is_empty()
                            || query.trim().starts_with("memory:")
                        {
                            None
                        } else {
                            embedder.embed_query(&query).ok()
                        };
                        graph
                            .inspect(&principal_id, &context_id, &query, embedding.as_deref())
                            .map(|inspection| crate::memory_graph::render_inspection(&inspection))
                    })
                    .await
                    .unwrap_or_else(|join| Err(anyhow::anyhow!("inspect join: {join}")));
                    let _ = reply.send(result);
                }
                IoRequest::MemoryRemember {
                    context_id,
                    principal_id,
                    content,
                    scope,
                    reply,
                } => {
                    let result = tokio::task::spawn_blocking(move || {
                        let embedding = embedder
                            .embed_passages(&[content.clone()])?
                            .into_iter()
                            .next()
                            .ok_or_else(|| anyhow::anyhow!("no embedding"))?;
                        graph
                            .remember_canonical(&crate::memory_graph::CanonicalDraft {
                                content,
                                context_id,
                                principal_id,
                                embedding,
                                scope,
                                created_at_ms: crate::memory_graph::now_ms(),
                            })
                            .map(|id| format!("remembered as {id}"))
                    })
                    .await
                    .unwrap_or_else(|join| Err(anyhow::anyhow!("remember join: {join}")));
                    let _ = reply.send(result);
                }
                IoRequest::MemoryForget {
                    principal_id,
                    memory_id,
                    reply,
                } => {
                    let result = tokio::task::spawn_blocking(move || {
                        graph.forget_memory(&principal_id, &memory_id).map(|forgotten| {
                            if forgotten {
                                format!("forgot {memory_id}")
                            } else {
                                format!("nothing forgotten: {memory_id} is not a memory this conversation can change")
                            }
                        })
                    })
                    .await
                    .unwrap_or_else(|join| Err(anyhow::anyhow!("forget join: {join}")));
                    let _ = reply.send(result);
                }
                IoRequest::Scoped { request, reply } => {
                    let result =
                        crate::capabilities::dispatch(request, graph, embedder, sender).await;
                    let _ = reply.send(result);
                }
                _ => {}
            }
        }
    });
    tx
}

fn remember(
    harness: &Harness,
    embedder: &crate::memory_graph::MemoryEmbedder,
    content: &str,
    context: &str,
    principal: &str,
    scope: crate::memory_graph::MemoryScope,
) -> String {
    harness
        .graph
        .remember_canonical(&crate::memory_graph::CanonicalDraft {
            content: content.into(),
            context_id: context.into(),
            principal_id: principal.into(),
            embedding: embedder.embed_passages(&[content.into()]).unwrap().remove(0),
            scope,
            created_at_ms: 1,
        })
        .unwrap()
}

#[tokio::test]
async fn native_takeover_memory_inspect_then_correct_authorized_memory() {
    let context = "telegram:100000001";
    let harness = Harness::new(true);
    let embedder = crate::memory_graph::MemoryEmbedder::deterministic();
    let wrong = remember(
        &harness,
        &embedder,
        "Opus 5 does not exist",
        context,
        ADMIN,
        crate::memory_graph::MemoryScope::Context,
    );
    let io = spawn_memory_io(harness.graph.clone(), Arc::new(embedder));
    let provider = provider(vec![
        tool("inspect", "memory_inspect", json!({})),
        tool(
            "correct",
            "memory_correct",
            json!({"old_id": wrong, "replacement": "Opus 5 exists"}),
        ),
        response("Corrected.", vec![call("done", "finish", json!({}))]),
    ])
    .await;
    let reply = harness
        .turn_with_io(&provider, io, context, ADMIN, "Opus 5 does not exist")
        .await;
    assert_eq!(reply, "Corrected.");
    let active = harness
        .graph
        .canonical_for_context(ADMIN, context, 40, 8_000)
        .unwrap();
    assert!(
        active.iter().any(|m| m.content == "Opus 5 exists"),
        "{active:?}"
    );
    assert!(!active.iter().any(|m| m.content == "Opus 5 does not exist"));
}

/// #129: a `run` block in a native takeover turn executes with the turn's identity, so a
/// turn-bound builtin inside it sees this conversation's memories (and only those), and
/// nothing about the block is declared or recorded as a mutation.
#[tokio::test]
async fn native_takeover_run_block_uses_the_turn_identity_and_persists_nothing() {
    let context = "telegram:100000001";
    let harness = Harness::new(true);
    let embedder = crate::memory_graph::MemoryEmbedder::deterministic();
    remember(&harness, &embedder, "RunBlockVisibleNote", context, ADMIN, crate::memory_graph::MemoryScope::Context);
    remember(&harness, &embedder, "RunBlockForeignNote", "telegram:group:-100000099", "telegram:user:100000555",
        crate::memory_graph::MemoryScope::Context);
    let io = spawn_memory_io(harness.graph.clone(), Arc::new(embedder));
    let provider = provider(vec![
        tool("block", "run", json!({"source": "seen: str = await memory_inspect()\nreturn seen\n"})),
        response("Looked.", vec![call("done", "finish", json!({}))]),
    ])
    .await;
    let reply = harness.turn_with_io(&provider, io, context, ADMIN, "What do you remember?").await;
    assert_eq!(reply, "Looked.");
    let requests = provider.requests.lock().unwrap().clone();
    let names: Vec<&str> = requests[0]["tools"].as_array().unwrap().iter()
        .map(|tool| tool["function"]["name"].as_str().unwrap()).collect();
    assert_eq!(&names[..6], ["inspect", "mutate", "test", "evaluate", "run", "finish"]);
    let delivered = all_text(&requests[1]);
    assert!(delivered.contains("\"tool_call_id\":\"block\""), "{delivered}");
    assert!(delivered.contains("RunBlockVisibleNote"), "{delivered}");
    assert!(!delivered.contains("RunBlockForeignNote"), "{delivered}");
    assert!(harness.program.read().await.modules.is_empty());
    let meta = harness.meta.lock().unwrap();
    assert_eq!(meta.revision, 0);
    assert!(meta.mutations.is_empty() && meta.sources.is_empty());
}

#[tokio::test]
async fn native_takeover_memory_inspect_other_context_id_is_indistinguishable() {
    let context = "telegram:100000001";
    let harness = Harness::new(true);
    let embedder = crate::memory_graph::MemoryEmbedder::deterministic();
    let foreign = remember(
        &harness,
        &embedder,
        "secret from another chat",
        "telegram:group:-100000099",
        "telegram:user:999999",
        crate::memory_graph::MemoryScope::Context,
    );
    let io = spawn_memory_io(harness.graph.clone(), Arc::new(embedder));
    let provider = provider(vec![
        tool("look", "memory_inspect", json!({"query": foreign})),
        response("done", vec![call("done", "finish", json!({}))]),
    ])
    .await;
    harness
        .turn_with_io(&provider, io, context, ADMIN, "look it up")
        .await;
    let request = all_text(&provider.requests.lock().unwrap()[1]);
    assert!(request.contains("indistinguishable"), "{request}");
    assert!(!request.contains("secret from another chat"), "foreign content leaked: {request}");
}

#[tokio::test]
async fn native_takeover_memory_remember_global_scope_works_for_write_turn() {
    let context = "telegram:100000001";
    let harness = Harness::new(true);
    let embedder = crate::memory_graph::MemoryEmbedder::deterministic();
    let io = spawn_memory_io(harness.graph.clone(), Arc::new(embedder));
    let provider = provider(vec![
        tool(
            "remember",
            "memory_remember",
            json!({"content": "Prefers dark mode everywhere", "scope": "global"}),
        ),
        response("Remembered globally.", vec![call("done", "finish", json!({}))]),
    ])
    .await;
    let reply = harness
        .turn_with_io(&provider, io, context, ADMIN, "remember this globally")
        .await;
    assert_eq!(reply, "Remembered globally.");
    let active = harness
        .graph
        .canonical_for_context(ADMIN, context, 40, 8_000)
        .unwrap();
    assert!(
        active.iter().any(|m| m.content == "Prefers dark mode everywhere"),
        "{active:?}"
    );
}

#[tokio::test]
async fn native_takeover_memory_forget_foreign_and_absent_are_indistinguishable() {
    let context = "telegram:100000001";
    let harness = Harness::new(true);
    let embedder = crate::memory_graph::MemoryEmbedder::deterministic();
    // A memory this conversation cannot change (foreign context).
    let foreign = remember(
        &harness,
        &embedder,
        "secret from another chat",
        "telegram:group:-100000099",
        "telegram:user:999999",
        crate::memory_graph::MemoryScope::Context,
    );
    let io = spawn_memory_io(harness.graph.clone(), Arc::new(embedder));
    let provider = provider(vec![
        response(
            "",
            vec![
                call("forget-foreign", "memory_forget", json!({"id": foreign})),
                call("forget-absent", "memory_forget", json!({"id": "memory:absent-xyz"})),
            ],
        ),
        response("done", vec![call("done", "finish", json!({}))]),
    ])
    .await;
    harness
        .turn_with_io(&provider, io, context, ADMIN, "forget both")
        .await;
    let round = all_text(&provider.requests.lock().unwrap()[1]);
    // "Not yours" and "does not exist" read identically (same wording, echoing
    // only the caller-supplied id) so forgetting cannot probe other
    // conversations' memory ids.
    let foreign_msg = format!("nothing forgotten: {foreign} is not a memory this conversation can change");
    let absent_msg =
        "nothing forgotten: memory:absent-xyz is not a memory this conversation can change".to_string();
    assert!(round.contains(&foreign_msg), "{round}");
    assert!(round.contains(&absent_msg), "{round}");
    assert!(!round.contains("does not exist"), "{round}");
    assert!(!round.contains("not yours"), "{round}");
}

/// #68: across real takeover turns the provider-visible history only appends:
/// each request's messages before the runtime envelope are a byte prefix of the
/// next request's, and the previous history end carries a cache breakpoint.
#[tokio::test]
async fn takeover_history_is_append_only_across_turns_with_turn_boundary_breakpoint() {
    let harness = Harness::new(true);
    let outputs: Vec<Value> = (0..4).flat_map(|i| vec![
        tool(&format!("look{i}"), "inspect", json!({"kind":"symbols"})),
        response(&format!("answer {i}"), vec![call(&format!("fin{i}"), "finish", json!({}))]),
    ]).collect();
    let provider = provider(outputs).await;
    for i in 0..4 {
        harness.turn(&provider, "telegram:group:-88068", ADMIN, &format!("question {i}")).await;
    }
    let requests = provider.requests.lock().unwrap().clone();
    // First request of each turn: requests 0, 2, 4, 6.
    let firsts: Vec<&Value> = requests.iter().step_by(2).collect();
    assert_eq!(firsts.len(), 4);
    let strip = |m: &Value| { let mut m = m.clone(); m.as_object_mut().unwrap().remove("cache_control"); m.to_string() };
    for pair in firsts.windows(2) {
        let (old, new) = (pair[0]["messages"].as_array().unwrap(), pair[1]["messages"].as_array().unwrap());
        // Everything but the trailing runtime envelope must be an unchanged prefix.
        let old_history = &old[..old.len() - 1];
        for (a, b) in old_history.iter().zip(new.iter()) {
            assert_eq!(strip(a), strip(b), "history changed between turns");
        }
        assert!(new.len() > old.len(), "history grew");
        // The previous request's history boundary (its last assistant message) is
        // a breakpoint in the next request too.
        if let Some(boundary) = old_history.iter().rposition(|m| m["role"] == "assistant") {
            assert!(old[boundary].get("cache_control").is_some(), "history boundary missing at {boundary}");
            assert!(new[boundary].get("cache_control").is_some(), "turn boundary breakpoint missing at {boundary}");
        }
    }
    for request in &requests {
        let messages = request["messages"].as_array().unwrap();
        assert_eq!(crate::context_log::gateway_boundary_errors(messages), Vec::<String>::new());
    }
}

fn boundary_roles(request: &Value) -> Vec<(usize, String)> {
    request["messages"].as_array().unwrap().iter().enumerate()
        .filter(|(_, m)| m.get("cache_control").is_some())
        .map(|(i, m)| (i, m["role"].as_str().unwrap().to_string()))
        .collect()
}

/// Regression for the deployed dd8cea4 build, through the real takeover loop:
/// a native turn succeeds, the next turn's provider call fails (no assistant
/// message is stored), and the turn after that must still be accepted. The old
/// placement marked the `finish` tool result and the failed turn's user message,
/// which the gateway merges into one provider message and rejects (HTTP 400),
/// for every following request of that context.
#[tokio::test]
async fn native_takeover_request_after_a_failed_turn_has_gateway_valid_cache_boundaries() {
    let harness = Harness::new(true);
    let context = "telegram:group:-88069";
    let provider = provider(vec![
        tool("look", "inspect", json!({"kind":"symbols"})),
        response("first answer", vec![call("fin0", "finish", json!({}))]),
        json!({"http_status": 400}),
        json!({"http_status": 400}),
        response("recovered", vec![call("fin1", "finish", json!({}))]),
        response("still fine", vec![call("fin2", "finish", json!({}))]),
    ]).await;
    assert_eq!(harness.turn(&provider, context, ADMIN, "first question").await, "first answer");
    // Two failed turns in a row: each leaves only its user message behind.
    for failed in ["second question", "third question"] {
        let reply = harness.turn(&provider, context, ADMIN, failed).await;
        assert!(reply.contains("technisches Problem"), "{reply}");
    }
    assert_eq!(harness.turn(&provider, context, ADMIN, "fourth question").await, "recovered");
    assert_eq!(harness.turn(&provider, context, ADMIN, "fifth question").await, "still fine");
    let requests = provider.requests.lock().unwrap().clone();
    assert_eq!(requests.len(), 6);
    for (index, request) in requests.iter().enumerate() {
        let messages = request["messages"].as_array().unwrap();
        // The gateway's merge rule and the Adapsis rule (no boundary on a tool or
        // user history message, at most one per merged provider message).
        assert_eq!(crate::context_log::gateway_boundary_errors(messages), Vec::<String>::new(), "request {index}");
        assert!(messages.last().unwrap().get("cache_control").is_none(), "request {index}: envelope");
    }
    // Requests 2-4 follow the native turn: the stored history is
    // user, assistant(inspect), tool, assistant(finish), tool, then the failed
    // turns' user messages. The only history boundary is the finish batch.
    for index in [2, 3, 4] {
        let roles: Vec<String> = requests[index]["messages"].as_array().unwrap().iter().map(|m| m["role"].as_str().unwrap().to_string()).collect();
        assert_eq!(&roles[..6], ["system", "user", "assistant", "tool", "assistant", "tool"], "request {index}");
        assert!(roles[6..].iter().all(|role| role == "user"), "request {index}: {roles:?}");
        assert_eq!(boundary_roles(&requests[index]), vec![(0, "system".to_string()), (4, "assistant".to_string())], "request {index}");
    }
    // The history only appended across the failures (the epoch continued, so the
    // turn-boundary path ran), and after recovery the boundary moves on.
    let strip = |m: &Value| { let mut m = m.clone(); m.as_object_mut().unwrap().remove("cache_control"); m.to_string() };
    let (third, fourth) = (requests[3]["messages"].as_array().unwrap(), requests[4]["messages"].as_array().unwrap());
    for (a, b) in third[..third.len() - 1].iter().zip(fourth.iter()) {
        assert_eq!(strip(a), strip(b));
    }
    let last = boundary_roles(&requests[5]);
    assert_eq!(last.len(), 3, "stable prefix, previous boundary, new history end: {last:?}");
    assert!(last[1..].iter().all(|(_, role)| role == "assistant"));
    assert_eq!(last[1].0, 4, "the previous request's boundary is read again");
}

/// The same failure sequence on the legacy (non-native) takeover protocol.
#[tokio::test]
async fn legacy_takeover_requests_after_failed_turns_have_gateway_valid_cache_boundaries() {
    let harness = Harness::new(false);
    let context = "telegram:group:-88070";
    let provider = provider(vec![
        response("first answer<code>!done</code>", vec![]),
        json!({"http_status": 400}),
        json!({"http_status": 400}),
        response("recovered<code>!done</code>", vec![]),
        response("still fine<code>!done</code>", vec![]),
    ]).await;
    harness.turn(&provider, context, ADMIN, "first question").await;
    for failed in ["second question", "third question"] {
        assert!(harness.turn(&provider, context, ADMIN, failed).await.contains("technisches Problem"));
    }
    harness.turn(&provider, context, ADMIN, "fourth question").await;
    harness.turn(&provider, context, ADMIN, "fifth question").await;
    let requests = provider.requests.lock().unwrap().clone();
    assert_eq!(requests.len(), 5);
    for (index, request) in requests.iter().enumerate() {
        assert!(request.get("tools").is_none(), "legacy protocol");
        let messages = request["messages"].as_array().unwrap();
        assert_eq!(crate::context_log::gateway_boundary_errors(messages), Vec::<String>::new(), "request {index}");
    }
    // After two failed turns the old placement marked both failed user messages.
    let roles: Vec<String> = requests[3]["messages"].as_array().unwrap().iter().map(|m| m["role"].as_str().unwrap().to_string()).collect();
    assert_eq!(roles, ["system", "user", "assistant", "user", "user", "user"]);
    assert_eq!(boundary_roles(&requests[3]), vec![(0, "system".to_string()), (2, "assistant".to_string())]);
}

/// The wire request of a real takeover turn carries the 1-hour lifetime on every
/// cache boundary for an Anthropic model and no ttl for other models, on both
/// protocols, and the gateway rules accept it.
#[tokio::test]
async fn takeover_requests_carry_the_one_hour_cache_lifetime_only_for_models_that_ask_for_it() {
    for (native, model, ttl) in [
        (true, "anthropic/claude-opus-5", Some("1h")),
        (false, "anthropic/claude-opus-5", Some("1h")),
        (true, "takeover-fixture", None),
        (true, "chatgpt/gpt-6-astra", None),
    ] {
        let mut harness = Harness::new(native);
        harness.model = model;
        let context = format!("telegram:group:-8807{}{}", u8::from(native), model.len());
        let answer = |text: &str, id: &str| if native {
            response(text, vec![call(id, "finish", json!({}))])
        } else {
            response(&format!("{text}<code>!done</code>"), vec![])
        };
        let provider = provider(vec![answer("one", "f0"), answer("two", "f1"), answer("three", "f2")]).await;
        for question in ["first", "second", "third"] {
            harness.turn(&provider, &context, ADMIN, question).await;
        }
        let requests = provider.requests.lock().unwrap().clone();
        assert_eq!(requests.len(), 3, "{model} native={native}");
        for (index, request) in requests.iter().enumerate() {
            let messages = request["messages"].as_array().unwrap();
            assert_eq!(crate::context_log::gateway_boundary_errors(messages), Vec::<String>::new(), "{model} request {index}");
            let markers: Vec<&Value> = messages.iter().filter_map(|m| m.get("cache_control")).collect();
            // Stable prefix always; history boundaries from the second turn on.
            assert_eq!(markers.len(), [1, 2, 3][index], "{model} native={native} request {index}");
            for marker in markers {
                assert_eq!(marker["type"], "ephemeral");
                assert_eq!(marker.get("ttl").and_then(Value::as_str), ttl, "{model} native={native} request {index}");
            }
        }
    }
}
