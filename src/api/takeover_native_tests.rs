//! Offline mock-provider takeover fixtures (#85). No live provider, gateway,
//! FastEmbed model, user library, contexts directory or memory database.
use crate::session::{SessionMeta, SharedMeta};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

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
    let end = json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":output["usage"]});
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
        }
    }

    async fn turn(&self, provider: &Provider, context: &str, speaker: &str, text: &str) -> String {
        let (io_tx, _io_rx) = tokio::sync::mpsc::channel(8);
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
            "takeover-fixture",
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
            None,
            self.native_tools,
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
    assert_eq!(requests[0]["tools"].as_array().unwrap().len(), 5);
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

#[tokio::test]
async fn guest_and_non_write_takeover_turns_stay_on_the_legacy_protocol() {
    let mut harness = Harness::new(true);
    harness.permissions = Arc::new(
        toml::from_str(
            r#"
        [principals]
        "telegram:user:100000777" = "guest"
        [guest_profiles]
        guest = []
        [model.guest]
        user = "execute"
        [model.viewer]
        user = "execute"
        [model.takeover-fixture]
        user = "write"
    "#,
        )
        .unwrap(),
    );
    // Guest principal (server-bound speaker mapping).
    let guest = provider(vec![response("Hallo!\n<code>!done</code>", vec![])]).await;
    let reply = harness
        .turn(&guest, "telegram:user:100000777", "telegram:user:100000777", "Hi")
        .await;
    assert_eq!(reply, "Hallo!");
    // Execute-only speaker selected by the intake module's permission_model.
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
