//! Offline HTTP fixtures: no provider/model-quality or live-gateway claims.
use super::*;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

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

fn provider_body(output: &Value, streaming: bool) -> axum::response::Response {
    if !streaming {
        return Json(output.clone()).into_response();
    }
    let mut delta = output["choices"][0]["message"].clone();
    if let Some(calls) = delta["tool_calls"].as_array_mut() {
        for (index, call) in calls.iter_mut().enumerate() {
            call["index"] = json!(index);
        }
    }
    let chunk = json!({"choices":[{"index":0,"delta":delta,"finish_reason":null}]});
    let terminal = json!({"choices":[{"index":0,"delta":{},"finish_reason":output["choices"][0]["finish_reason"]}],"usage":output["usage"]});
    let ending = if output["truncated"] == true {
        String::new()
    } else {
        format!("data: {terminal}\n\ndata: [DONE]\n\n")
    };
    (
        [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
        format!("data: {chunk}\n\n{ending}"),
    )
        .into_response()
}

struct Fixture {
    config: AppConfig,
    requests: Arc<Mutex<Vec<Value>>>,
    server: tokio::task::JoinHandle<()>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}

impl Fixture {
    async fn new(outputs: Vec<Value>, prelude: &str) -> Self {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = requests.clone();
        let budget = outputs.len();
        let app = axum::Router::new().route(
            "/v1/chat/completions",
            axum::routing::post(move |Json(body): Json<Value>| {
                let recorded = recorded.clone();
                let outputs = outputs.clone();
                async move {
                    let mut requests = recorded.lock().unwrap();
                    let output = outputs
                        .get(requests.len())
                        .cloned()
                        .unwrap_or_else(|| response("unexpected", vec![]));
                    let streaming = body["stream"] == true;
                    requests.push(body);
                    provider_body(&output, streaming)
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let mut config = test_config();
        config.native_tools = true;
        config.llm_url = format!("http://{address}");
        config.max_iterations = budget;
        *config.llm_model.write().unwrap() = "native-fixture".into();
        if !prelude.is_empty() {
            let mut ws = config.snapshot_working_set().await;
            let result = execute_code(prelude, &config, &mut ws, None).await;
            assert!(!result.has_errors, "{:?}", result.mutation_results);
            config.write_back_working_set(&ws).await;
        }
        config.program.write().await.require_modules = true;
        Self {
            config,
            requests,
            server,
        }
    }

    async fn run(&self, streaming: bool) -> AskResponse {
        tokio::time::timeout(std::time::Duration::from_secs(15), async {
            let request = Json(AskRequest {
                message: "Build and verify the function".into(),
                caller: None,
            });
            if !streaming {
                return llm_handlers::ask(State(self.config.clone()), request)
                    .await
                    .0;
            }
            let response = llm_handlers::ask_stream(State(self.config.clone()), request)
                .await
                .into_response();
            let bytes = axum::body::to_bytes(response.into_body(), 4 * 1024 * 1024)
                .await
                .unwrap();
            let body = String::from_utf8(bytes.to_vec()).unwrap();
            let events: Vec<Value> = body
                .lines()
                .filter_map(|line| {
                    line.strip_prefix("data:")
                        .and_then(|s| serde_json::from_str(s.trim()).ok())
                })
                .collect();
            AskResponse {
                completed: events
                    .iter()
                    .rev()
                    .find(|e| e["type"] == "done")
                    .and_then(|e| e["completed"].as_bool()),
                has_errors: events.iter().any(|e| {
                    e["type"] == "error" || e["success"] == false || e["has_errors"] == true
                }),
                reply: body,
                code: String::new(),
                test_results: vec![],
                results: events
                    .iter()
                    .filter(|e| e["type"] == "result")
                    .map(|e| MutationResult {
                        message: e["message"].as_str().unwrap_or_default().into(),
                        success: e["success"] == true,
                    })
                    .collect(),
            }
        })
        .await
        .expect("native fixture timed out")
    }
}

fn source(module: &str) -> String {
    format!(
        "+module {module}\n+fn add (x:Int)->Int\n +let a:Int = x+1\n +let b:Int = a+1\n +return b\n+end"
    )
}

async fn lifecycle(streaming: bool, module: &str) {
    let fixture = Fixture::new(
        vec![
            tool("inspect", "inspect", json!({"kind":"symbols"})),
            tool("mutate", "mutate", json!({"source":source(module)})),
            tool("early", "finish", json!({})),
            tool(
                "test",
                "test",
                json!({"source":format!("+test {module}.add\n +with x=3 -> expect 5\n")}),
            ),
            tool(
                "evaluate",
                "evaluate",
                json!({"function":format!("{module}.add"),"input":"3"}),
            ),
            response("Verified five.", vec![call("finish", "finish", json!({}))]),
        ],
        "",
    )
    .await;
    let result = fixture.run(streaming).await;
    assert_eq!(
        result.completed,
        Some(true),
        "results={:?}; reply={}",
        result.results,
        result.reply
    );
    assert!(
        result
            .results
            .iter()
            .any(|r| !r.success && r.message.contains(module)),
        "{:?}",
        result.results
    );
    assert!(
        result
            .results
            .iter()
            .any(|r| r.success && r.message.contains('5')),
        "{:?}",
        result.results
    );
    let requests = fixture.requests.lock().unwrap();
    assert_eq!(requests.len(), 6);
    assert_eq!(requests[0]["tools"].as_array().unwrap().len(), 5);
    assert!(
        requests[0]["messages"][0]["content"]
            .as_str()
            .unwrap()
            .contains("Fixed native tool protocol")
    );
    assert!(
        !requests[0]["messages"][0]["content"]
            .as_str()
            .unwrap()
            .contains("NEVER use <tool_call>")
    );
    let persisted = fixture
        .config
        .meta
        .lock()
        .unwrap()
        .conversations
        .get("main")
        .unwrap()
        .to_llm_messages();
    assert_eq!(
        persisted
            .iter()
            .filter(|m| m.role == crate::llm::ChatRole::Tool)
            .count(),
        6
    );
    assert!(
        persisted
            .iter()
            .any(|m| m.tool_call_id.as_deref() == Some("finish"))
    );
}

#[tokio::test]
async fn native_http_inspect_mutate_test_evaluate_finish() {
    lifecycle(false, "NativeHttpLifecycle").await;
}

#[tokio::test]
async fn native_stream_inspect_mutate_test_evaluate_finish() {
    lifecycle(true, "NativeStreamLifecycle").await;
}

#[tokio::test]
async fn native_prose_and_legacy_code_do_not_complete_or_execute() {
    let fixture = Fixture::new(
        vec![response(
            "<code>+module MustNotExist\n+fn no()->Int\n+return 1\n+end\n!done</code>",
            vec![],
        )],
        "",
    )
    .await;
    let result = fixture.run(false).await;
    assert!(result.has_errors && result.reply.contains("completion is unconfirmed"));
    assert!(
        !fixture
            .config
            .program
            .read()
            .await
            .modules
            .iter()
            .any(|module| module.name == "MustNotExist")
    );
}

#[tokio::test]
async fn native_calls_take_precedence_over_embedded_legacy_code() {
    let fixture = Fixture::new(
        vec![response(
            "<code>+module DoubleRunForbidden\n+fn no()->Int\n+return 1\n+end</code>",
            vec![call("finish", "finish", json!({}))],
        )],
        "",
    )
    .await;
    assert!(!fixture.run(false).await.has_errors);
    assert!(
        !fixture
            .config
            .program
            .read()
            .await
            .modules
            .iter()
            .any(|module| module.name == "DoubleRunForbidden")
    );
}

#[tokio::test]
async fn native_truncated_stream_never_applies_partial_calls() {
    let mut output = tool(
        "mutate",
        "mutate",
        json!({"source":source("TruncatedNative")}),
    );
    output["truncated"] = json!(true);
    let fixture = Fixture::new(vec![output], "").await;
    let response = fixture.run(true).await;
    assert!(response.has_errors);
    assert!(response.reply.contains("\"type\":\"iteration\""));
    assert_eq!(response.completed, Some(false));
    assert!(
        !fixture
            .config
            .program
            .read()
            .await
            .modules
            .iter()
            .any(|module| module.name == "TruncatedNative")
    );
}

#[tokio::test]
async fn native_schema_failure_skips_whole_batch_with_correlated_errors() {
    for bad in [
        call("bad", "unknown", json!({})),
        call("bad", "inspect", json!({"kind":"symbols","extra":true})),
    ] {
        let fixture = Fixture::new(
            vec![response(
                "",
                vec![
                    call(
                        "good",
                        "mutate",
                        json!({"source":source("SchemaBatchForbidden")}),
                    ),
                    bad,
                ],
            )],
            "",
        )
        .await;
        assert!(fixture.run(false).await.has_errors);
        assert!(
            !fixture
                .config
                .program
                .read()
                .await
                .modules
                .iter()
                .any(|module| module.name == "SchemaBatchForbidden")
        );
        let messages = fixture
            .config
            .meta
            .lock()
            .unwrap()
            .conversations
            .get("main")
            .unwrap()
            .to_llm_messages();
        assert_eq!(
            messages
                .iter()
                .filter(|m| m.role == crate::llm::ChatRole::Tool)
                .count(),
            2
        );
    }
}

#[tokio::test]
async fn native_duplicate_id_is_not_replayed_but_new_id_same_args_is() {
    let fixture = Fixture::new(
        vec![
            tool("one", "inspect", json!({"kind":"symbols"})),
            tool("one", "inspect", json!({"kind":"symbols"})),
            tool("two", "inspect", json!({"kind":"symbols"})),
            tool("finish", "finish", json!({})),
        ],
        "",
    )
    .await;
    assert!(fixture.run(false).await.has_errors);
    let messages = fixture
        .config
        .meta
        .lock()
        .unwrap()
        .conversations
        .get("main")
        .unwrap()
        .to_llm_messages();
    assert_eq!(
        messages
            .iter()
            .filter(|m| m.tool_call_id.as_deref() == Some("one"))
            .count(),
        1
    );
    assert_eq!(
        messages
            .iter()
            .filter(|m| m.tool_call_id.as_deref() == Some("two"))
            .count(),
        1
    );
}

#[tokio::test]
async fn native_finish_cannot_share_a_batch_and_old_debt_does_not_block() {
    let fixture = Fixture::new(
        vec![
            response(
                "",
                vec![
                    call("inspect", "inspect", json!({"kind":"symbols"})),
                    call("finish", "finish", json!({})),
                ],
            ),
            tool("new-finish", "finish", json!({})),
        ],
        &source("NativeOldDebt"),
    )
    .await;
    let result = fixture.run(false).await;
    assert!(result.has_errors);
    assert!(!result.reply.contains("completion is unconfirmed"));
    let messages = fixture
        .config
        .meta
        .lock()
        .unwrap()
        .conversations
        .get("main")
        .unwrap()
        .to_llm_messages();
    assert_eq!(
        messages
            .iter()
            .filter(|m| m.role == crate::llm::ChatRole::Tool)
            .count(),
        1
    );
}

#[test]
fn native_mode_is_opt_in_and_excludes_execute_only_and_guests() {
    let mut config = test_config();
    assert!(!super::super::native_turn::eligible(&config));
    config.native_tools = true;
    assert!(super::super::native_turn::eligible(&config));
    config.access_level = crate::permissions::AccessLevel::ExecuteOnly;
    assert!(!super::super::native_turn::eligible(&config));
    config.access_level = crate::permissions::AccessLevel::Full;
    Arc::make_mut(&mut config.permission_config)
        .guest_profiles
        .insert(String::new(), vec![]);
    assert!(!super::super::native_turn::eligible(&config));
}

#[tokio::test]
async fn native_denied_target_preflight_prevents_earlier_mutation() {
    let mut fixture = Fixture::new(
        vec![response(
            "",
            vec![
                call(
                    "good",
                    "mutate",
                    json!({"source":source("NativeAllowedButSkipped")}),
                ),
                call(
                    "denied",
                    "inspect",
                    json!({"kind":"source","target":"NativePrivate"}),
                ),
            ],
        )],
        "",
    )
    .await;
    fixture.config.permission_config = Arc::new(
        toml::from_str(
            r#"
        [groups]
        secret = ["NativePrivate"]
        [model.native-fixture]
        user = "write"
        secret = "none"
    "#,
        )
        .unwrap(),
    );
    assert!(fixture.run(false).await.has_errors);
    assert!(
        !fixture
            .config
            .program
            .read()
            .await
            .modules
            .iter()
            .any(|module| module.name == "NativeAllowedButSkipped")
    );
}

#[tokio::test]
async fn native_batch_allows_mutate_then_inspect_created_target() {
    let fixture = Fixture::new(
        vec![
            response(
                "",
                vec![
                    call(
                        "create",
                        "mutate",
                        json!({"source":source("NativeDependentBatch")}),
                    ),
                    call(
                        "read",
                        "inspect",
                        json!({"kind":"source","target":"NativeDependentBatch.add"}),
                    ),
                ],
            ),
            tool("finish", "finish", json!({})),
        ],
        "",
    )
    .await;
    let result = fixture.run(false).await;
    assert!(
        result
            .results
            .iter()
            .any(|r| r.success && r.message.contains("+fn")),
        "{:?}",
        result.results
    );
    assert!(
        result.has_errors,
        "the new function still owes tests at finish"
    );
}

#[tokio::test]
async fn native_context_profile_intersects_model_permissions_without_widening() {
    for (active, profile) in [("write", "read"), ("read", "write")] {
        let mut fixture = Fixture::new(
            vec![tool(
                "blocked",
                "mutate",
                json!({"source":source("NativeContextPrivate")}),
            )],
            "",
        )
        .await;
        fixture.config.permission_config = Arc::new(
            toml::from_str(&format!(
                r#"
            [groups]
            secret = ["NativeContextPrivate"]
            [model.native-fixture]
            user = "write"
            secret = "{active}"
            [model.context-profile]
            user = "write"
            secret = "{profile}"
        "#
            ))
            .unwrap(),
        );
        fixture
            .config
            .meta
            .lock()
            .unwrap()
            .conversations
            .get_or_create("main")
            .permission_model = Some("context-profile".into());
        let result = fixture.run(false).await;
        assert!(result.has_errors);
        assert_eq!(result.completed, Some(false));
        assert!(
            !fixture
                .config
                .program
                .read()
                .await
                .modules
                .iter()
                .any(|module| module.name == "NativeContextPrivate")
        );
        assert_eq!(
            fixture.requests.lock().unwrap()[0]["model"],
            "native-fixture"
        );
    }
}

#[test]
fn native_permission_snapshot_respects_default_profiles_and_is_pinned() {
    let mut config = test_config();
    config.native_tools = true;
    *config.llm_model.write().unwrap() = "unlisted-generation-model".into();
    config.permission_config = Arc::new(
        toml::from_str(
            r#"
        [groups]
        hidden = ["Private"]
        [model.default]
        user = "read"
        hidden = "none"
        [model.permissive]
        user = "write"
        hidden = "write"
    "#,
        )
        .unwrap(),
    );
    config
        .meta
        .lock()
        .unwrap()
        .conversations
        .get_or_create("main")
        .permission_model = Some("permissive".into());
    let snapshot = super::super::native_turn::permission_snapshot(&config);
    assert!(!super::super::native_turn::eligible(&config));
    *config.llm_model.write().unwrap() = "permissive".into();
    assert_eq!(
        *snapshot.llm_model.read().unwrap(),
        "unlisted-generation-model"
    );
    assert_eq!(
        snapshot.permission_config.resolve(
            snapshot.access_level,
            "unlisted-generation-model",
            "Private"
        ),
        crate::permissions::PermissionLevel::None
    );
    assert_eq!(
        snapshot.permission_config.resolve(
            snapshot.access_level,
            "unlisted-generation-model",
            "Ungrouped"
        ),
        crate::permissions::PermissionLevel::Read
    );
}

#[test]
fn native_principal_profile_cannot_widen_the_active_model() {
    let mut config = test_config();
    config.native_tools = true;
    *config.llm_model.write().unwrap() = "restricted".into();
    config.permission_config = Arc::new(
        toml::from_str(
            r#"
        [principals]
        "server-bound:alice" = "powerful"
        [groups]
        secret = ["Private"]
        [model.restricted]
        user = "write"
        secret = "none"
        [model.powerful]
        user = "write"
        secret = "write"
    "#,
        )
        .unwrap(),
    );
    config.turn = Some(crate::coroutine::TurnIdentity {
        context: "main".into(),
        principal: "server-bound:alice".into(),
        may_write: true,
        guest_tools: None,
    });
    let snapshot = super::super::native_turn::permission_snapshot(&config);
    assert!(super::super::native_tools::native_tools_eligible(&snapshot));
    assert!(
        super::super::native_tools::validate_native_tool_authorized(
            "inspect",
            r#"{"kind":"source","target":"Private"}"#,
            &snapshot
        )
        .is_err()
    );
}

#[tokio::test]
async fn native_missing_targets_are_neutral_in_http_and_sse_results() {
    for streaming in [false, true] {
        let suffix = if streaming { "Stream" } else { "Http" };
        let private = format!("NativeDiagPrivate{suffix}");
        let public = format!("NativeDiagPublic{suffix}");
        let prelude = format!(
            "+module {private}\n+fn secret ()->String\n +return \"SECRET_DIAGNOSTIC\"\n+end\n+module {public}\n+fn echo (x:String)->String\n +return x\n+end"
        );
        let mut fixture = Fixture::new(
            vec![
                tool(
                    "eval",
                    "evaluate",
                    json!({"function":format!("{public}.secret")}),
                ),
                tool(
                    "test",
                    "test",
                    json!({"source":format!("+test {public}.secret\n +with -> expect \"no\"\n")}),
                ),
            ],
            &prelude,
        )
        .await;
        fixture.config.permission_config = Arc::new(
            toml::from_str(&format!(
                r#"
            [groups]
            hidden = ["{private}"]
            [model.native-fixture]
            hidden = "none"
            user = "write"
        "#
            ))
            .unwrap(),
        );
        let response = fixture.run(streaming).await;
        assert_eq!(response.completed, Some(false));
        let wire = serde_json::to_string(&response).unwrap();
        assert!(
            !wire.contains(&private) && !wire.contains("SECRET_DIAGNOSTIC"),
            "{wire}"
        );
    }
}
