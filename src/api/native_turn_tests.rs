//! Offline HTTP fixtures: no provider/model-quality or live-gateway claims.
use super::*;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
#[path = "python_turn_tests.rs"]
mod python_turn_tests;
#[path = "native_repair_turn_tests.rs"]
mod native_repair_turn_tests;
#[path = "run_turn_tests.rs"]
mod run_turn_tests;
#[path = "source_save_tests.rs"]
mod source_save_tests;

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
    assert_eq!(requests[0]["tools"].as_array().unwrap().len(), 6);
    // #68: the output limit is the generating model's configured limit, not 8192.
    let expected = crate::model_limits::for_model(requests[0]["model"].as_str().unwrap()).output_tokens;
    assert!(expected >= 32_000);
    assert!(requests.iter().all(|r| r["max_tokens"] == expected), "{:?}", requests[0]["max_tokens"]);
    // Regression: an explicit temperature makes claude-opus-5 reject every
    // native turn (400 "temperature is deprecated"); omit it like legacy.
    assert!(
        requests.iter().all(|r| r.get("temperature").is_none()),
        "native request bodies must omit temperature"
    );
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
fn native_mode_is_opt_in_and_http_excludes_guests() {
    use super::super::native_tools::{speaker_class, SpeakerClass};
    let mut config = test_config();
    assert!(!super::super::native_turn::eligible(&config));
    config.native_tools = true;
    assert!(super::super::native_turn::eligible(&config));
    assert_eq!(speaker_class(&config), SpeakerClass::Writer);
    // Execute-only speakers are native on HTTP too, with their own tool list (DESIGN L3).
    config.access_level = crate::permissions::AccessLevel::ExecuteOnly;
    assert!(super::super::native_turn::eligible(&config));
    assert_eq!(speaker_class(&config), SpeakerClass::ExecuteOnly);
    // Guests stay on the legacy protocol on HTTP: there is no turn identity to bind
    // their tools to. Without one a guest profile is the most restricted class.
    config.access_level = crate::permissions::AccessLevel::Full;
    Arc::make_mut(&mut config.permission_config)
        .guest_profiles
        .insert(String::new(), vec!["web_search".into()]);
    assert!(!super::super::native_turn::eligible(&config));
    assert_eq!(speaker_class(&config), SpeakerClass::Guest(vec![]));
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
    // The active model's default profile caps the turn at Read: an execute-only
    // speaker, native with the execute-only tool list, never a writer.
    assert!(super::super::native_turn::eligible(&config));
    assert_eq!(
        super::super::native_tools::speaker_class(&snapshot),
        super::super::native_tools::SpeakerClass::ExecuteOnly
    );
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

/// Decision 4 at the snapshot: a named profile without a usable entry pins the most
/// restricted class instead of falling through to "unlisted model resolves to Execute".
#[test]
fn a_missing_or_malformed_profile_pins_the_most_restricted_class() {
    use super::super::native_tools::{speaker_class, SpeakerClass};
    let permissions = |extra: &str| -> Arc<crate::permissions::PermissionConfig> {
        Arc::new(toml::from_str(&format!("[model.listed]\nuser = \"execute\"\n[model.native-fixture]\nuser = \"write\"\n{extra}")).unwrap())
    };
    for (label, config_text, profile, expected) in [
        ("listed profile", "", Some("listed"), SpeakerClass::ExecuteOnly),
        ("no profile at all", "", None, SpeakerClass::Writer),
        ("profile without an entry", "", Some("no-such-profile"), SpeakerClass::Guest(vec![])),
        ("guest profile without model permissions", "[guest_profiles]\nhalf = [\"web_search\"]\n", Some("half"), SpeakerClass::Guest(vec![])),
        ("a default entry makes every name listed", "[model.default]\nuser = \"execute\"\n", Some("anything"), SpeakerClass::ExecuteOnly),
    ] {
        let mut config = test_config();
        config.native_tools = true;
        *config.llm_model.write().unwrap() = "native-fixture".into();
        config.permission_config = permissions(config_text);
        config.meta.lock().unwrap().conversations.get_or_create("main").permission_model = profile.map(str::to_string);
        let snapshot = super::super::native_turn::permission_snapshot(&config);
        assert_eq!(speaker_class(&snapshot), expected, "{label}");
        // HTTP: only writers and execute-only speakers are native; the rest stay legacy.
        assert_eq!(
            super::super::native_turn::eligible(&config),
            matches!(expected, SpeakerClass::Writer | SpeakerClass::ExecuteOnly),
            "{label}"
        );
    }
}

/// The native loop records each answered request's size against the provider's
/// reported input tokens, per conversation, for the next view's projection.
#[tokio::test]
async fn native_turn_calibrates_the_token_projection_from_provider_usage() {
    for streaming in [false, true] {
        let mut answer = tool("done", "finish", json!({}));
        // Anthropic wire shape: total processed input = prompt + read + created.
        answer["usage"] = json!({"prompt_tokens":1_000,"completion_tokens":5,"total_tokens":1_005,
            "cache_read_input_tokens":0,"cache_creation_input_tokens":3_000});
        let fixture = Fixture::new(vec![answer], "").await;
        fixture.run(streaming).await;
        let requests = fixture.requests.lock().unwrap();
        let sent: usize = requests[0]["messages"].as_array().unwrap().iter().map(|m| m.to_string().len()).sum();
        let tools: usize = requests[0]["tools"].to_string().len();
        let meta = fixture.config.meta.lock().unwrap();
        let measured = meta.conversations.get("main").unwrap().token_calibration.clone().expect("calibrated");
        assert_eq!(measured.input_tokens, 4_000, "streaming={streaming}");
        // Messages plus tool schemas, within the schema framing estimate.
        let expected = (sent + tools) as u64;
        assert!(measured.request_bytes.abs_diff(expected) * 20 <= expected, "streaming={streaming}: {} vs {expected}", measured.request_bytes);
        assert_eq!(measured.model, "native-fixture");
        let ratio = meta.conversations.get("main").unwrap().bytes_per_token("native-fixture");
        assert!((crate::model_limits::MIN_BYTES_PER_TOKEN..=crate::model_limits::MAX_BYTES_PER_TOKEN).contains(&ratio));
    }
}

/// The cache epoch of an HTTP native turn carries its tool list (`prepare` puts
/// `tool_schema_hash` into the prefix key): the hash is stable for one class, differs
/// between classes and is never the "no tools" value of a legacy turn. The `main` epoch
/// tracker is process-global, so the reset itself is asserted on the tracker
/// (`context_log`) and through the takeover loop (`takeover_cache_tests`).
#[test]
fn http_native_turns_key_their_cache_epoch_by_tool_list() {
    let snapshot = |profile: Option<&str>| {
        let mut config = test_config();
        config.native_tools = true;
        *config.llm_model.write().unwrap() = "native-fixture".into();
        config.permission_config = Arc::new(toml::from_str("[model.listed]\nuser = \"execute\"\n[model.native-fixture]\nuser = \"write\"\n").unwrap());
        config.meta.lock().unwrap().conversations.get_or_create("main").permission_model = profile.map(str::to_string);
        super::super::native_turn::permission_snapshot(&config)
    };
    let hash = |profile| super::super::http_context::tool_schema_hash(&snapshot(profile));
    let (writer, execute_only) = (hash(None), hash(Some("listed")));
    assert_eq!(writer, hash(None));
    assert_eq!(execute_only, hash(Some("listed")));
    assert_ne!(writer, execute_only);
    assert!(writer != 0 && execute_only != 0);
}

/// A response the provider cut off (output limit) executes nothing, even when a tool
/// call in it looks complete. The model is told, keeps its partial text, and the turn
/// goes on. Both transports.
#[tokio::test]
async fn a_response_cut_off_at_the_output_limit_executes_nothing_and_is_corrected() {
    for streaming in [false, true] {
        let module = format!("NtCutOff{}", if streaming { "Stream" } else { "Plain" });
        let mut cut = tool("cut", "mutate", json!({"source": format!("+module {module}\n+fn one ()->Int\n  +return 1\n+end")}));
        cut["choices"][0]["finish_reason"] = json!("length");
        cut["choices"][0]["message"]["content"] = json!("I will now write the module");
        let fixture = Fixture::new(vec![cut, response("Done in pieces.", vec![call("done", "finish", json!({}))])], "").await;
        let response = fixture.run(streaming).await;
        assert_eq!(response.completed, Some(true), "streaming={streaming}: {:?}", response.results);
        // The cut-off is reported as an error of the turn (the stream carries it as an error event).
        assert!(response.has_errors, "streaming={streaming}");
        if !streaming {
            assert!(response.results.iter().any(|result| !result.success && result.message.contains("cut off by the provider")), "{:?}", response.results);
        }
        assert!(!fixture.config.program.read().await.modules.iter().any(|m| m.name == module), "the cut-off call ran");
        let requests = fixture.requests.lock().unwrap().clone();
        assert_eq!(requests.len(), 2, "streaming={streaming}");
        let messages = requests[1]["messages"].as_array().unwrap();
        let limit = crate::model_limits::for_model("native-fixture").output_tokens;
        let correction = messages.last().unwrap()["content"].as_str().unwrap();
        assert!(correction.contains("cut off before it was complete") && correction.contains(&format!("{limit} output tokens")), "{correction}");
        assert!(correction.contains("Nothing from it was executed") && correction.contains("smaller pieces"), "{correction}");
        let partial = &messages[messages.len() - 2];
        assert_eq!((partial["role"].as_str(), partial["content"].as_str()), (Some("assistant"), Some("I will now write the module")));
        assert!(partial.get("tool_calls").is_none_or(|calls| calls.as_array().is_none_or(Vec::is_empty)), "{partial}");
        assert!(messages.iter().all(|message| message["tool_call_id"] != "cut"), "no result exists for a call that never ran");
        // The stored history holds no half batch either.
        let meta = fixture.config.meta.lock().unwrap();
        let stored = &meta.conversations.get("main").unwrap().messages;
        assert!(stored.iter().all(|message| message.tool_calls.as_ref().is_none_or(|calls| calls.iter().all(|c| c.id != "cut"))));
    }
}

/// `/api/ask` on `main`, on a node with a workspace registry. A request from this node
/// has the workspace tools; a request from another node has none and is not shown the
/// workspace batches of earlier local turns; the next local turn sees them again. What
/// the event stream, the response and the training log record of a workspace call
/// carries no path, pattern or content.
#[tokio::test]
async fn http_main_has_workspace_tools_only_for_a_local_request_and_hides_their_history_from_others() {
    use crate::api::origin::Origin;
    let repo = crate::workspace::test_support::RepoFixture::new();
    let canary = "HTTP_CANARY_CONTENT_7731";
    crate::workspace::test_support::write(&repo.repository.checkout, "notes-7731.txt", format!("{canary}\n"));
    crate::workspace::test_support::commit_all(&repo.repository.checkout, "notes");
    let site = repo.site("http-main");
    let fixture = Fixture::new(
        vec![
            // Turn 1, local.
            tool("open", "workspace_open", json!({"repo": "http-main", "name": "h1"})),
            tool("rd", "read", json!({"path": "notes-7731.txt"})),
            response("Read it.", vec![call("done", "finish", json!({}))]),
            // Turn 2, from another node.
            tool("forged", "read", json!({"path": "notes-7731.txt"})),
            response("Peer done.", vec![call("done2", "finish", json!({}))]),
            // Turn 3, local again.
            tool("close", "workspace_close", json!({})),
            response("Closed.", vec![call("done3", "finish", json!({}))]),
        ],
        "",
    )
    .await;
    let log = tempfile::NamedTempFile::new().unwrap();
    let mut config = fixture.config.clone();
    config.workspaces = Some(site.clone());
    config.max_iterations = 4;
    config.training_log = Some(Arc::new(tokio::sync::Mutex::new(tokio::fs::File::create(log.path()).await.unwrap())));
    // Everything any subscriber of `/api/events` would see.
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let (mut events, sink) = (config.event_broadcast.subscribe(), seen.clone());
    let collector = tokio::spawn(async move {
        loop {
            match events.recv().await {
                Ok(event) => sink.lock().unwrap().push(event),
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => panic!("the test lost events"),
                Err(_) => break,
            }
        }
    });
    let ask = |origin: Option<Origin>, message: &str| {
        let mut config = config.clone();
        config.origin = origin;
        let request = Json(AskRequest { message: message.into(), caller: None });
        async move { tokio::time::timeout(std::time::Duration::from_secs(30), llm_handlers::ask(State(config), request)).await.unwrap().0 }
    };
    let remote = Some(Origin::Remote("10.0.0.4".parse().unwrap()));

    let local = ask(Some(Origin::Local), "read the notes").await;
    assert_eq!((local.completed, local.has_errors), (Some(true), false), "{:?}", local.results);
    let peer = ask(remote, "what is in the notes?").await;
    assert_eq!(peer.completed, Some(true), "{:?}", peer.results);
    let again = ask(Some(Origin::Local), "close it").await;
    assert_eq!((again.completed, again.has_errors), (Some(true), false), "{:?}", again.results);

    let requests = fixture.requests.lock().unwrap().clone();
    assert_eq!(requests.len(), 7);
    let names = |request: &Value| -> Vec<String> { request["tools"].as_array().unwrap().iter().map(|tool| tool["function"]["name"].as_str().unwrap().to_string()).collect() };
    let result = |index: usize, id: &str| requests[index]["messages"].as_array().unwrap().iter().find(|m| m["tool_call_id"] == id).unwrap()["content"].as_str().unwrap().to_string();
    // Local: the tools, the section, the tail line, plain-text results.
    assert!(names(&requests[0]).iter().any(|name| name == "workspace_open") && names(&requests[0]).len() == 6 + 7, "{:?}", names(&requests[0]));
    assert!(requests[0]["messages"][0]["content"].as_str().unwrap().contains("## Workspace tools"));
    assert!(requests[0]["messages"].to_string().contains("Workspace of this conversation: none. Registered repositories: http-main."));
    assert!(result(1, "open").starts_with("opened workspace http-main/h1\npath: "), "{}", result(1, "open"));
    assert_eq!(result(2, "rd"), format!("notes-7731.txt: lines 1-1 of 1 (25 bytes)\n1: {canary}"));
    // From another node: no tools, no section, no tail line, the forged call is no tool,
    // and the earlier workspace batches are not in the request.
    for request in &requests[3..5] {
        let sent = request.to_string();
        assert_eq!(names(request).len(), 6, "{:?}", names(request));
        assert!(!sent.contains("orkspace of this conversation") && !sent.contains("## Workspace tools") && !sent.contains("workspace_open"), "{sent}");
        assert!(!sent.contains(canary), "file content reached a request from another node");
        assert!(sent.contains("[System: 2 earlier tool exchange(s) of this conversation are not shown in this turn.]"), "{sent}");
        // The uncovered remainder (#145): the owner's question and the model's prose.
        assert!(sent.contains("read the notes") && sent.contains("Read it."));
    }
    assert!(!requests[3].to_string().contains("notes-7731.txt"), "not even the path, before the peer itself names it");
    assert!(result(4, "forged").contains("unknown native tool `read`"), "{}", result(4, "forged"));
    assert!(peer.results.iter().any(|line| !line.success && line.message.contains("unknown native tool `read`")));
    // Local again: everything is back, including the first turn's results byte for byte.
    assert!(names(&requests[5]).iter().any(|name| name == "read"));
    assert_eq!(requests[5]["messages"].as_array().unwrap().iter().find(|m| m["tool_call_id"] == "rd").unwrap()["content"], result(2, "rd").as_str());
    assert!(!requests[5].to_string().contains("are not shown in this turn"));
    assert!(requests[5]["messages"].to_string().contains("Workspace of this conversation: http-main/h1 at "));
    assert!(result(6, "close").starts_with("closed workspace http-main/h1;"));
    for request in &requests {
        assert_eq!(crate::context_log::gateway_boundary_errors(request["messages"].as_array().unwrap()), Vec::<String>::new());
    }
    // The stored history is complete throughout.
    {
        let meta = fixture.config.meta.lock().unwrap();
        let stored = &meta.conversations.get("main").unwrap().messages;
        assert!(stored.iter().any(|message| message.content.contains(canary)) && stored.iter().filter(|message| message.tool_calls.is_some()).count() == 7);
    }
    assert!(site.bindings.load("main").unwrap().is_none() && !repo.repository.worktree_parent.join("h1").exists());

    // Response and event stream: the tool and whether it worked, nothing else.
    let lines: Vec<(&str, bool)> = local.results.iter().chain(&again.results).map(|line| (line.message.as_str(), line.success)).collect();
    assert_eq!(lines, [("`workspace_open`: ok", true), ("`read`: ok", true), ("`workspace_close`: ok", true)]);
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    collector.abort();
    let stream = seen.lock().unwrap().join("\n");
    assert!(stream.contains("`read`: ok") && stream.contains("`workspace_open`: ok"), "{stream}");
    assert!(!stream.contains(canary) && !stream.contains("notes-7731") && !stream.contains("worktrees"), "{stream}");
    // Training log: static names/sizes and structural feedback, no header or content.
    let training = tokio::fs::read_to_string(log.path()).await.unwrap();
    let entries: Vec<Value> = training.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
    let recorded = |index: usize| {
        let entry = &entries[index];
        let calls: Vec<Value> = serde_json::from_str(entry["code"].as_str().unwrap()).unwrap();
        (calls[0]["function"]["arguments"].as_str().unwrap().to_string(), entry["feedback"][0].as_str().unwrap().to_string(), entry["outcome"].as_str().unwrap().to_string())
    };
    let (args, feedback, outcome) = recorded(1);
    assert_eq!(serde_json::from_str::<Value>(&args).unwrap(), json!({"fields":{"path":{"bytes":16}},"unknown_fields":{"count":0,"bytes":0}}));
    assert_eq!(serde_json::from_str::<Value>(&feedback).unwrap(), json!({"tool":"read","success":true,"result_bytes":69}));
    assert_eq!(outcome, "success");
    assert_eq!(serde_json::from_str::<Value>(&recorded(0).0).unwrap()["fields"], json!({"name":{"bytes":4},"repo":{"bytes":11}}));
    assert_eq!(recorded(3).2, "error");
    assert!(!training.contains(canary) && !training.contains("notes-7731.txt"), "{training}");
}

/// The actual automatic training sink, including a mixed batch and preflight
/// failures. Patterns, paths, file headers/bodies, unknown JSON keys and malformed
/// argument text never survive in a workspace training record.
#[tokio::test]
async fn workspace_training_sink_records_only_structural_metadata() {
    let repo = crate::workspace::test_support::RepoFixture::new();
    let root = "CANARY_DIRECTORY_123";
    let file = format!("{root}/CANARY_GLOB_PATTERN_123.txt");
    let body = "CANARY_READ_BODY_123 CANARY_TOKEN_123\n";
    crate::workspace::test_support::write(&repo.repository.checkout, &file, body);
    crate::workspace::test_support::commit_all(&repo.repository.checkout, "fixture");
    let mut malformed = tool("bad", "read", json!({}));
    malformed["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json!("{\"path\":\"CANARY_MALFORMED_123");
    let mut fixture = Fixture::new(vec![
        tool("o", "workspace_open", json!({"repo":"train","name":"metadata"})),
        response("", vec![
            call("r", "read", json!({"path":file})),
            call("s", "search", json!({"pattern":"CANARY_TOKEN_123","path":root})),
            call("g", "glob", json!({"pattern":"CANARY_GLOB_PATTERN*","path":root})),
            call("i", "inspect", json!({"kind":"symbols"})),
        ]),
        tool("e", "read", json!({"path":"CANARY_FAILED_PATH_123"})),
        tool("u", "search", json!({"pattern":"CANARY_REJECTED_PATTERN_123","CANARY_UNKNOWN_KEY_123":"CANARY_UNKNOWN_VALUE_123"})),
        malformed,
        response("Done.", vec![call("f", "finish", json!({}))]),
    ], "").await;
    let log = tempfile::NamedTempFile::new().unwrap();
    fixture.config.workspaces = Some(repo.site("train"));
    fixture.config.origin = Some(crate::api::origin::Origin::Local);
    fixture.config.training_log = Some(Arc::new(tokio::sync::Mutex::new(tokio::fs::File::create(log.path()).await.unwrap())));
    let answer = fixture.run(false).await;
    assert_eq!(answer.completed, Some(true));
    let training = tokio::fs::read_to_string(log.path()).await.unwrap();
    assert!(!training.contains("CANARY"), "{training}");
    let entries: Vec<Value> = training.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
    assert_eq!(entries.len(), 6);
    let calls = |index: usize| serde_json::from_str::<Vec<Value>>(entries[index]["code"].as_str().unwrap()).unwrap();
    let feedback = |index: usize, part: usize| serde_json::from_str::<Value>(entries[index]["feedback"][part].as_str().unwrap()).unwrap();
    let mixed = calls(1);
    assert_eq!(mixed.iter().map(|c| c["function"]["name"].as_str().unwrap()).collect::<Vec<_>>(), ["read","search","glob","inspect"]);
    for (index, tool) in [(0,"read"), (1,"search"), (2,"glob")] {
        let meta = feedback(1, index);
        assert_eq!(meta["tool"], tool);
        assert_eq!(meta["success"], true);
        assert!(meta["result_bytes"].as_u64().unwrap() > 0);
        assert_eq!(meta.as_object().unwrap().len(), 3);
    }
    assert_eq!(feedback(1, 3)["tool"], "inspect", "other tools keep their envelope in the mixed batch");
    assert!(feedback(1, 3).get("query_results").is_some());
    for index in [2, 3, 4] {
        assert_eq!(entries[index]["outcome"], "error");
        assert_eq!(feedback(index, 0)["success"], false);
    }
    let unknown: Value = serde_json::from_str(calls(3)[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(unknown["argument_fields"], 2);
    assert!(unknown["argument_bytes"].as_u64().unwrap() > 0);
    let malformed: Value = serde_json::from_str(calls(4)[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert!(malformed["argument_bytes"].as_u64().unwrap() > 0);
    assert_eq!(malformed["argument_fields"], Value::Null);
    // The model did receive the sensitive content; redaction is at the sink.
    let requests = fixture.requests.lock().unwrap();
    assert!(requests[2].to_string().contains("CANARY_READ_BODY_123"));
    assert!(requests[3].to_string().contains("CANARY_FAILED_PATH_123"));
}

/// Workspace-origin preflight errors are shared with every skipped call of a
/// mixed batch. Neither an ordinary peer's training policy nor generic error
/// reporting may forward that private diagnosis. Both orders (including the
/// workspace failure last), malformed JSON and oversized paths; no execution.
#[tokio::test]
async fn workspace_preflight_failures_in_mixed_batches_remain_private() {
    for workspace_first in [true, false] {
        for (case, arguments, diagnosis) in [
            ("unknown-key", json!({"path":"README.md","CANARY_INPUT_MARKER_17":"value"}).to_string(), "unexpected field `CANARY_INPUT_MARKER_17`"),
            ("malformed", "{\"path\":\"CANARY_MALFORMED_17".into(), "arguments are not valid JSON"),
            ("path", json!({"path":format!("CANARY_PATH_17{}", "p".repeat(4096))}).to_string(), "field `path` exceeds 4096 bytes"),
        ] {
            let repo = crate::workspace::test_support::RepoFixture::new();
            let mut workspace_call = call("CANARY_WORKSPACE_ID_17", "read", json!({}));
            workspace_call["function"]["arguments"] = json!(arguments);
            // This is an admissible inspect call; its args/id deliberately carry
            // content too, so the shared-refusal record must sanitize peer calls.
            let peer = call("CANARY_PEER_ID_17", "inspect", json!({"kind":"symbols","target":"CANARY_PEER_TARGET_17"}));
            // A third call would mutate state if any part of this batch ran.
            let mutation = call("CANARY_MUTATION_ID_17", "mutate", json!({"source":"+module CANARY_ShouldNotExist"}));
            let mut calls = if workspace_first { vec![workspace_call, peer] } else { vec![peer, workspace_call] };
            calls.insert(0, mutation);
            let mut fixture = Fixture::new(vec![response("", calls), response("Done.", vec![call("finish", "finish", json!({}))])], "").await;
            let training_file = tempfile::NamedTempFile::new().unwrap();
            fixture.config.workspaces = Some(repo.site("mixed-refusal"));
            fixture.config.origin = Some(crate::api::origin::Origin::Local);
            fixture.config.training_log = Some(Arc::new(tokio::sync::Mutex::new(tokio::fs::File::create(training_file.path()).await.unwrap())));
            let mut events = fixture.config.event_broadcast.subscribe();
            let answer = fixture.run(false).await;
            assert_eq!(answer.completed, Some(true), "{case}/{workspace_first}");
            assert!(answer.has_errors);
            assert!(!fixture.config.program.read().await.modules.iter().any(|module| module.name == "CANARY_ShouldNotExist"));
            let response_text = serde_json::to_string(&answer).unwrap();
            assert!(!response_text.contains("CANARY"), "{case}/{workspace_first}: {response_text}");
            let failed: Vec<_> = answer.results.iter().filter(|result| !result.success).collect();
            assert_eq!(failed.len(), 3);
            assert!(failed.iter().all(|result| result.message == "Native tool batch refused before execution; no calls were executed."));
            while let Ok(event) = events.try_recv() {
                assert!(!event.contains("CANARY"), "{case}/{workspace_first}: {event}");
            }
            let training = tokio::fs::read_to_string(training_file.path()).await.unwrap();
            assert!(!training.contains("CANARY"), "{case}/{workspace_first}: {training}");
            let rows: Vec<Value> = training.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
            assert_eq!(rows.len(), 2);
            assert_eq!(rows[0]["outcome"], "error");
            let recorded: Vec<Value> = serde_json::from_str(rows[0]["code"].as_str().unwrap()).unwrap();
            assert_eq!(recorded.len(), 3);
            for (index, call) in recorded.iter().enumerate() {
                assert_eq!(call["id"], "skipped");
                let arguments: Value = serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
                assert!(arguments["argument_bytes"].as_u64().unwrap() > 0);
                let feedback: Value = serde_json::from_str(rows[0]["feedback"][index].as_str().unwrap()).unwrap();
                assert_eq!(feedback["tool"], call["function"]["name"]);
                assert_eq!(feedback["success"], false, "all skipped calls fail");
                assert!(feedback["result_bytes"].as_u64().unwrap() > 0);
                assert_eq!(feedback.as_object().unwrap().len(), 3);
            }
            // Diagnosis is still actionable in the private model exchange, for
            // each correlated result (not reconstructed by parsing prose).
            let requests = fixture.requests.lock().unwrap();
            assert_eq!(requests.len(), 2);
            let replies: Vec<_> = requests[1]["messages"].as_array().unwrap().iter().filter(|m| m["role"] == "tool").collect();
            assert_eq!(replies.len(), 3);
            for reply in replies {
                let detail: Value = serde_json::from_str(reply["content"].as_str().unwrap()).unwrap();
                assert_eq!(detail["success"], false);
                assert!(detail["error"].as_str().unwrap().contains(diagnosis), "{case}/{workspace_first}: {detail}");
            }
        }
    }
}

/// The real HTTP layer. `Origin` comes from the TCP peer: a direct request from this
/// node is local; the same request through a reverse proxy on this node (the Caddy
/// case: loopback peer, forwarding header) is not, whatever the header claims.
#[tokio::test]
async fn the_http_layer_gives_workspace_tools_to_direct_local_requests_only() {
    let repo = crate::workspace::test_support::RepoFixture::new();
    let finish = |id: &str| response("ok", vec![call(id, "finish", json!({}))]);
    let fixture = Fixture::new(vec![finish("a"), finish("b"), finish("c"), finish("d")], "").await;
    let mut config = fixture.config.clone();
    config.workspaces = Some(repo.site("http-origin"));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = crate::api::router_with_llm(config).into_make_service_with_connect_info::<std::net::SocketAddr>();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let client = reqwest::Client::new();
    for (headers, local) in [
        (vec![], true),
        (vec![("X-Forwarded-For", "127.0.0.1")], false),
        (vec![("Forwarded", "for=127.0.0.1")], false),
        (vec![("X-Real-IP", "::1")], false),
    ] {
        let mut request = client.post(format!("http://{address}/api/ask")).json(&json!({"message": "hello"}));
        for (name, value) in &headers {
            request = request.header(*name, *value);
        }
        let answer: Value = request.send().await.unwrap().json().await.unwrap();
        assert_eq!(answer["completed"], true, "{answer}");
        let requests = fixture.requests.lock().unwrap();
        let sent = requests.last().unwrap();
        let has = sent["tools"].as_array().unwrap().iter().any(|tool| tool["function"]["name"] == "workspace_open");
        assert_eq!(has, local, "{headers:?}");
        assert_eq!(sent.to_string().contains("## Workspace tools"), local, "{headers:?}");
        assert_eq!(sent.to_string().contains("Workspace of this conversation"), local, "{headers:?}");
    }
    server.abort();
}

/// What drops a turn's future in production: `/api/ask` when its client disconnects
/// (the handler future IS the turn). The blocking call it started (here `workspace_open`,
/// held in a git hook) finishes and binds; until then the conversation's work lock is
/// held, and the model never gets a second request.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_disconnected_ask_client_drops_the_turn_but_its_blocking_call_finishes_under_the_work_lock() {
    use std::os::unix::fs::PermissionsExt;
    use tokio::io::AsyncWriteExt;
    let repo = crate::workspace::test_support::RepoFixture::new();
    let site = repo.site("http-drop");
    let (ready, release) = (repo.top.join("hook-ready"), repo.top.join("hook-release"));
    let hook = repo.repository.checkout.join(".git/hooks/post-checkout");
    std::fs::write(&hook, format!("#!/bin/sh\necho ready > '{}'\nwhile [ ! -f '{}' ]; do /usr/bin/sleep 0.01; done\n", ready.display(), release.display())).unwrap();
    std::fs::set_permissions(&hook, std::fs::Permissions::from_mode(0o700)).unwrap();
    let fixture = Fixture::new(
        vec![tool("open", "workspace_open", json!({"repo": "http-drop", "name": "dropped"})), response("never sent", vec![call("done", "finish", json!({}))])],
        "",
    )
    .await;
    let mut config = fixture.config.clone();
    config.workspaces = Some(site.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let app = crate::api::router_with_llm(config).into_make_service_with_connect_info::<std::net::SocketAddr>();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let body = json!({"message": "open it"}).to_string();
    let mut client = tokio::net::TcpStream::connect(address).await.unwrap();
    client
        .write_all(format!("POST /api/ask HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).as_bytes())
        .await
        .unwrap();
    let wait = |what: &'static str, done: &dyn Fn() -> bool| {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while !done() {
            assert!(std::time::Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    };
    wait("the hook", &|| ready.exists());
    assert!(crate::api::turn_lock::work_now(site.bindings.root(), "main").is_none(), "the blocking call holds the conversation's work lock");
    drop(client);
    // The turn's future is dropped: nothing is written back to the model any more.
    // Its blocking call still runs, so the binding is not there yet and the lock is held.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    assert!(site.bindings.load("main").unwrap().is_none());
    assert!(crate::api::turn_lock::work_now(site.bindings.root(), "main").is_none());
    std::fs::write(&release, b"go").unwrap();
    wait("the late binding", &|| site.bindings.load("main").unwrap().is_some());
    assert_eq!(site.bindings.holders("http-drop", "dropped").unwrap(), ["main"], "finished and bound, once");
    wait("the work lock", &|| crate::api::turn_lock::work_now(site.bindings.root(), "main").is_some());
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    assert_eq!(fixture.requests.lock().unwrap().len(), 1, "the dropped turn asked the model nothing more");
    server.abort();
}

/// Three responses cut off in a row end an HTTP turn too: no fourth generation, nothing
/// executed, a named error, `completed: false`. Both transports.
#[tokio::test]
async fn three_cut_off_responses_in_a_row_end_an_http_turn() {
    for streaming in [false, true] {
        let cut = |text: &str| {
            let mut output = tool("cut", "mutate", json!({"source": "+module NtCutOffThrice"}));
            output["choices"][0]["finish_reason"] = json!("length");
            output["choices"][0]["message"]["content"] = json!(text);
            output
        };
        let fixture = Fixture::new(vec![cut("one"), cut("two"), cut("three"), response("never", vec![call("done", "finish", json!({}))]), response("never", vec![])], "").await;
        let response = fixture.run(streaming).await;
        assert_eq!(response.completed, Some(false), "streaming={streaming}");
        assert!(response.has_errors);
        assert_eq!(fixture.requests.lock().unwrap().len(), 3, "streaming={streaming}: no fourth generation");
        if !streaming {
            assert!(response.results.iter().any(|line| !line.success && line.message.starts_with("The model's responses were cut off at its output limit three times in a row")), "{:?}", response.results);
        }
        assert!(!fixture.config.program.read().await.modules.iter().any(|m| m.name == "NtCutOffThrice"));
    }
}
