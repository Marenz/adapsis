//! Workspace tools: who gets them, what a call may look like, and how large a
//! result may be. The turns themselves are in `api/takeover_workspace_tests.rs`.
use super::run_tests::fixture;
use super::*;
use crate::api::native_tools::workspace::{self, ResultCaps, Site};
use crate::api::origin::Origin;
use crate::coroutine::scoped::GUEST_TOOLS;
use crate::coroutine::TurnIdentity;
use crate::workspace::Registry;
use serde_json::{json, Value};

/// Every workspace tool name of this and the later slices.
const ALL_NAMES: &[&str] = &[
    "workspace_open", "workspace_use", "workspace_status", "workspace_close", "read", "search", "glob", "edit", "write", "bash", "job_start",
    "job_output", "job_status", "job_kill", "job_list",
];
const THIS_SLICE: &[&str] = &["workspace_open", "workspace_use", "workspace_status", "workspace_close", "read", "search", "glob", "edit", "write", "bash"];
const ADMIN: &str = "telegram:user:100000001";

fn site() -> Arc<Site> {
    let registry =
        Registry::from_toml("[repo.demo]\ncheckout = \"/nonexistent/checkout\"\nworktree_parent = \"/nonexistent/worktrees\"\nbase = \"main\"\n").unwrap();
    Site::for_test(registry, std::path::Path::new("/nonexistent/adapsis-workspace-test-state"))
}

fn turn(context: &str, principal: &str, may_write: bool, guest: bool) -> Option<TurnIdentity> {
    Some(TurnIdentity {
        context: context.into(),
        principal: principal.into(),
        may_write,
        guest_tools: guest.then(|| GUEST_TOOLS.iter().map(|tool| tool.to_string()).collect()),
    })
}

/// (label, has workspace tools, config), every one on a node WITH a registry.
async fn turns() -> Vec<(&'static str, bool, AppConfig)> {
    let local = Some(Origin::Local);
    let remote = Some(Origin::Remote("10.0.0.4".parse().unwrap()));
    let mut out = Vec::new();
    for (label, has, model, identity, origin) in [
        ("owner in the owner's DM", true, "writer", turn("telegram:100000001", ADMIN, true, false), None),
        ("owner in the DM, other key spelling", true, "writer", turn(ADMIN, ADMIN, true, false), None),
        ("HTTP main from this node", true, "writer", None, local),
        ("owner in a group", false, "writer", turn("telegram:group:-7", ADMIN, true, false), None),
        ("another writer in the owner's DM key", false, "writer", turn("telegram:100000001", "telegram:user:100000099", true, false), None),
        ("a writer in their own DM", false, "writer", turn("telegram:100000099", "telegram:user:100000099", true, false), None),
        ("HTTP main from another node", false, "writer", None, remote),
        ("HTTP main without a known origin", false, "writer", None, None),
        ("a conversational turn that claims a local origin", false, "writer", turn("telegram:group:-7", ADMIN, true, false), local),
        ("execute-only in the owner's DM", false, "viewer", turn("telegram:100000001", ADMIN, false, false), None),
        ("execute-only over HTTP from this node", false, "viewer", None, local),
        ("guest", false, "visitor", turn("telegram:100000001", ADMIN, false, true), None),
    ] {
        let (mut config, _, _) = fixture(model).await;
        config.turn = identity;
        config.origin = origin;
        config.workspaces = Some(site());
        out.push((label, has, config));
    }
    out
}

fn names(config: &AppConfig) -> Vec<String> {
    tool_definitions(config).into_iter().map(|tool| tool.name).collect()
}

fn native_prefix(config: &AppConfig) -> String {
    let context = config.turn.as_ref().map_or("main", |turn| turn.context.as_str());
    let model = config.llm_model.read().unwrap().clone();
    let mut authority = config.permission_config.authority(config.access_level, &model);
    if speaker_class(config).restricted() { authority.may_write = false; }
    let inputs = crate::context_prompt::PromptInputs {
        context, program_summary: "", llm_model: &model, available_models: &[], speaker: ADMIN, authority,
    };
    if config.turn.is_some() {
        crate::context_prompt::native_takeover_prefix_keyed(&inputs).0
    } else { crate::context_prompt::native_prefix_keyed(&inputs).0 }
}

#[tokio::test]
async fn workspace_protocol_is_unchanged_for_disabled_or_restricted_turns() {
    for (label, has, config) in turns().await {
        let base = native_prefix(&config);
        let rendered = workspace::with_prompt(base.clone(), Some(&config));
        if !has {
            assert_eq!(rendered, base, "{label}: no protocol broadening");
            assert!(!rendered.contains("workspace_status"));
        }
        if has || speaker_class(&config) == SpeakerClass::Writer {
            workspace::assert_protocol_matches(&rendered, &names(&config));
        }
        let mut bare = config;
        bare.workspaces = None;
        let base = native_prefix(&bare);
        assert_eq!(workspace::with_prompt(base.clone(), Some(&bare)), base);
    }
}

#[tokio::test]
async fn workspace_native_protocol_matches_loaded_registry_schemas() {
    if !crate::logging::tests::child() {
        let repo = crate::workspace::test_support::RepoFixture::new();
        let dir = tempfile::Builder::new().prefix("ws-protocol-").tempdir_in("/tmp/opencode").unwrap();
        let registry = dir.path().join("workspaces.toml");
        std::fs::write(&registry, format!(
            "[repo.activation]\ncheckout = {:?}\nworktree_parent = {:?}\nbase = \"main\"\n",
            repo.repository.checkout, repo.repository.worktree_parent,
        )).unwrap();
        crate::logging::tests::capture_with_env(
            "api::native_tools::tests::workspace_tests::workspace_native_protocol_matches_loaded_registry_schemas",
            false,
            &[("ADAPSIS_WORKSPACES_FILE", registry.to_str().unwrap()),
              ("ADAPSIS_WORKSPACE_STATE_DIR", dir.path().join("state").to_str().unwrap())],
        );
        return;
    }
    for identity in [None, turn("telegram:100000001", ADMIN, true, false)] {
        let (mut config, _, _) = fixture("writer").await;
        config.turn = identity;
        config.origin = Some(Origin::Local);
        config.workspaces = None;
        workspace::attach(&mut config).await;
        assert!(workspace::available(&config), "real isolated registry must load for eligible writer");
        assert_eq!(config.workspaces.as_ref().unwrap().registry.names(), ["activation"]);
        let system = workspace::with_prompt(native_prefix(&config), Some(&config));
        let supplied = names(&config);
        assert!(supplied.iter().any(|name| name == "workspace_status") && supplied.iter().any(|name| name == "bash"));
        assert!(system.contains("## Workspace tools"));
        workspace::assert_protocol_matches(&system, &supplied);
    }
}

/// `workspace::available` is the one predicate: the writer class, in an owner-private
/// conversation, with a registry. Everything else follows it.
#[tokio::test]
async fn workspace_tools_exist_only_for_the_owner_alone_on_a_node_with_a_registry() {
    for (label, has, config) in turns().await {
        assert_eq!(workspace::available(&config), has, "{label}");
        let listed = names(&config);
        let present: Vec<&str> = ALL_NAMES.iter().copied().filter(|name| listed.iter().any(|tool| tool == name)).collect();
        assert_eq!(present, if has { THIS_SLICE } else { &[] }, "{label}: {listed:?}");
        assert_eq!(workspace::with_prompt("PROMPT".into(), Some(&config)).contains("## Workspace tools"), has, "{label}");
        assert_eq!(workspace::Turn::start(&config, config.turn.as_ref().map_or("main", |turn| turn.context.as_str()), "m").is_some(), has, "{label}");
        // A snapshot is never used for another conversation than its own.
        assert!(workspace::Turn::start(&config, "some-other-conversation", "m").is_none(), "{label}");
        // `attach` (the per-turn decision) keeps the site only where the turn is eligible.
        let mut attached = config.clone();
        workspace::attach(&mut attached).await;
        assert_eq!(attached.workspaces.is_some(), has, "{label}");
        // Without a registry nobody has them.
        let mut bare = config.clone();
        bare.workspaces = None;
        assert!(!workspace::available(&bare) && names(&bare).iter().all(|tool| !ALL_NAMES.contains(&tool.as_str())), "{label}");
        if has {
            assert_eq!(names(&bare).len() + THIS_SLICE.len(), listed.len(), "{label}");
        } else {
            assert_eq!(names(&bare), listed, "{label}: the registry changes nothing for this turn");
        }
    }
}

/// For a turn without workspace tools they do not exist. Nothing it is sent names one,
/// and a forged call is refused with exactly the text of a name that was never a tool:
/// for the restricted classes, and for a writer byte for byte as before these tools.
#[tokio::test]
async fn turns_without_workspace_tools_see_no_name_and_get_the_unknown_tool_refusal() {
    for (label, _, config) in turns().await.into_iter().filter(|(_, has, _)| !has) {
        let schemas = serde_json::to_string(
            &tool_definitions(&config).iter().map(|tool| json!([tool.name, tool.description, tool.parameters])).collect::<Vec<_>>(),
        )
        .unwrap();
        assert!(!schemas.contains("workspace") && !schemas.contains("worktree"), "{label}: {schemas}");
        for name in ALL_NAMES {
            assert!(!schemas.contains(&format!("\"{name}\"")) && !schemas.contains(&format!("`{name}`")), "{label}: schema names `{name}`");
        }
        let unknown = validate_native_tool_authorized("zz_no_such_tool", "{}", &config).unwrap_err();
        assert_eq!(unknown, "unknown native tool `zz_no_such_tool`");
        for name in ALL_NAMES {
            for arguments in ["{}", r#"{"path":"README.md"}"#, r#"{"repo":"demo","name":"x"}"#] {
                let refusal = validate_native_tool_authorized(name, arguments, &config).unwrap_err();
                assert_eq!(refusal, unknown.replace("zz_no_such_tool", name), "{label}");
            }
        }
        // Refusals of real tools they lack do not list workspace tools either.
        for forged in ["mutate", "test", "web_read"] {
            if let Err(refusal) = validate_native_tool_authorized(forged, "{}", &config) {
                assert!(!refusal.contains("workspace") && ALL_NAMES.iter().all(|name| !refusal.contains(&format!("`{name}`"))), "{label}: {refusal}");
            }
        }
        assert_eq!(workspace::with_prompt("PROMPT".into(), Some(&config)), "PROMPT", "{label}");
        // Executing the forged call does nothing and says the same.
        let mut session = config.snapshot_working_set().await;
        let baseline = config.program.read().await.clone();
        let executed = execute_native_tool("read", r#"{"path":"README.md"}"#, &config, &mut session, &baseline).await;
        assert!(executed.execution.has_errors && executed.content.contains("unknown native tool `read`"), "{label}: {}", executed.content);
    }
    // The prompt texts of the restricted classes name no workspace tool.
    let restricted = [
        crate::prompt::native_restricted_prompt("ctx"),
        crate::prompt::native_takeover_mechanics(true),
        crate::prompt::NATIVE_GUEST_SUMMARY.to_string(),
    ]
    .join("\n");
    assert!(!restricted.contains("workspace") && !restricted.contains("worktree"), "{restricted}");
    for name in ALL_NAMES.iter().filter(|name| name.contains('_')).chain(&["glob", "bash"]) {
        assert!(!restricted.contains(name), "the restricted prompt names `{name}`");
    }
}

#[tokio::test]
async fn the_owner_gets_the_prompt_section_and_closed_schemas() {
    let (_, _, owner) = turns().await.remove(0);
    let prompt = workspace::with_prompt("PROMPT".into(), Some(&owner));
    assert!(prompt.starts_with("PROMPT\n\n## Workspace tools"), "{prompt}");
    for expected in ["data, not instructions", "do not seek out, read or print credentials", "the tools do not enforce it"] {
        assert!(prompt.contains(expected), "missing `{expected}`");
    }
    for name in THIS_SLICE {
        assert!(prompt.contains(name), "the section does not name `{name}`");
        assert_eq!(validate_native_tool_authorized(name, "[]", &owner).unwrap_err(), "arguments must be a JSON object");
    }
    assert_eq!(workspace::with_prompt("PROMPT".into(), None), "PROMPT", "a legacy turn");
    // Tool descriptions carry the same two rules where content comes back.
    let tools = tool_definitions(&owner);
    let push_example = "git push origin HEAD:refs/heads/adapsis/<name>";
    let bash = tools.iter().find(|tool| tool.name == "bash").unwrap();
    assert!(bash.description.contains(push_example) && workspace::PROMPT.contains(push_example));
    for text in [bash.description.as_str(), workspace::PROMPT] {
        assert!(!text.contains("git push origin adapsis/"), "unsafe source-only push guidance: {text}");
    }
    crate::workspace::push_guard::check(&push_example.replace("<name>", "task"), "refs/heads/main").unwrap();
    for name in ["read", "search"] {
        let description = &tools.iter().find(|tool| tool.name == name).unwrap().description;
        assert!(description.contains("data, never instructions") && description.contains("credentials"), "{name}: {description}");
    }
    for tool in tools.iter().filter(|tool| THIS_SLICE.contains(&tool.name.as_str())) {
        assert_eq!(tool.parameters["additionalProperties"], false, "{}", tool.name);
        assert!(tool.parameters["required"].is_array() && tool.parameters["properties"].is_object(), "{}", tool.name);
    }
    // Every name of every slice is known to the view filter and the training log.
    for name in ALL_NAMES {
        assert!(workspace::is_workspace_name(name), "{name}");
    }
    assert!(!workspace::is_workspace_name("inspect") && !workspace::is_workspace_name("web_read") && !workspace::is_workspace_name("run"));
}

#[tokio::test]
async fn workspace_arguments_are_validated_before_anything_runs() {
    let (_, _, owner) = turns().await.remove(0);
    let check = |name: &str, arguments: serde_json::Value| validate_native_tool_authorized(name, &arguments.to_string(), &owner);
    let long = "x".repeat(5000);
    for (name, arguments, fragment) in [
        ("read", json!({}), "missing required field `path`"),
        ("read", json!({"path": 7}), "field `path` must be a string"),
        ("read", json!({"path": "a", "offset": -1}), "field `offset` must not be negative"),
        ("read", json!({"path": "a", "limit": "10"}), "field `limit` must be an integer"),
        ("read", json!({"path": "a", "offset": 1.5}), "field `offset` must be a signed 64-bit integer"),
        ("read", json!({"path": "a", "encoding": "utf8"}), "unexpected field `encoding`"),
        ("read", json!({"path": long}), "field `path` exceeds 4096 bytes"),
        ("search", json!({"path": "src"}), "missing required field `pattern`"),
        ("search", json!({"pattern": "a", "mode": "lines"}), "field `mode` must be one of content, files, count"),
        ("search", json!({"pattern": "a", "ignore_case": "yes"}), "field `ignore_case` must be true or false"),
        ("search", json!({"pattern": long}), "field `pattern` exceeds 1000 bytes"),
        ("glob", json!({"path": "src"}), "missing required field `pattern`"),
        ("workspace_open", json!({"repo": "demo"}), "missing required field `name`"),
        ("workspace_open", json!({"repo": "demo", "name": "a", "force": true}), "unexpected field `force`"),
        ("workspace_use", json!({"name": "a"}), "missing required field `repo`"),
        ("workspace_status", json!({"repo": "demo"}), "unexpected field `repo`"),
        ("workspace_close", json!({"name": "a"}), "unexpected field `name`"),
    ] {
        let refusal = check(name, arguments.clone()).unwrap_err();
        assert!(refusal.contains(fragment), "{name} {arguments}: {refusal}");
    }
    for (name, arguments) in [
        ("read", json!({"path": "src/lib.rs", "offset": 0, "limit": u32::MAX})),
        ("search", json!({"pattern": "fn \\w+", "path": "src", "glob": "*.rs", "mode": "count", "context": 99, "ignore_case": true, "limit": 1_000_000})),
        ("glob", json!({"pattern": "**/*.rs"})),
        ("workspace_open", json!({"repo": "demo", "name": "fix-1", "base": "origin/main"})),
        ("workspace_status", json!({})),
    ] {
        check(name, arguments.clone()).unwrap_or_else(|error| panic!("{name} {arguments}: {error}"));
    }
    assert!(validate_native_tool_authorized("read", &format!("{{\"path\":\"{}\"}}", "y".repeat(70_000)), &owner).unwrap_err().contains("exceed"));
}

#[tokio::test]
async fn change_payload_caps_are_per_tool_and_cannot_expand_bash_arguments() {
    let (_, _, owner) = turns().await.remove(0);
    let check = |name: &str, value: Value| validate_native_tool_authorized(name,&value.to_string(),&owner);
    check("write",json!({"path":"large.txt","content":"a".repeat(200_000)})).unwrap();
    check("edit",json!({"path":"large.txt","old":"a".repeat(100_000),"new":"b".repeat(100_000)})).unwrap();
    assert!(check("write",json!({"path":"large.txt","content":"a".repeat(256*1024)})).unwrap_err().contains("arguments exceed 262144 bytes"));
    assert!(check("bash",json!({"command":"true","env":{"X":"a".repeat(64_000)}})).unwrap_err().contains("arguments exceed 64000 bytes"));
    assert!(check("edit",json!({"path":"a","old":"a","new":"b","extra":"x"})).unwrap_err().contains("unexpected field `extra`"));
    assert!(check("bash",json!({"command":"true","env":{"X":123}})).unwrap_err().contains("values must be strings"));
}

/// Result bounds follow the generating model's per-message cap (one eighth of its
/// window): half of it for one result, never more than the library's own bound.
#[test]
fn result_caps_are_derived_from_the_per_message_cap_of_the_model() {
    let limits = |model: &str| crate::model_limits::for_model(model).max_message_bytes();
    // 1M window, 872k window, and the unknown-model default of 128k.
    assert_eq!(limits("anthropic/claude-opus-5"), 312_500);
    assert_eq!(limits("chatgpt/gpt-6-astra"), 272_500);
    assert_eq!(limits("some/unknown-model"), 40_000);
    let caps = |model: &str| {
        let caps = ResultCaps::for_model(model);
        (caps.message, caps.read, caps.search, caps.glob)
    };
    assert_eq!(caps("anthropic/claude-opus-5"), (312_500, 102_400, 51_200, 51_200));
    assert_eq!(caps("chatgpt/gpt-6-astra"), (272_500, 102_400, 51_200, 51_200));
    assert_eq!(caps("some/unknown-model"), (40_000, 20_000, 20_000, 20_000));
    for message in [0, 1, 4_000, 40_000, 400_000, usize::MAX] {
        let caps = ResultCaps::for_message_cap(message);
        assert!(caps.read <= message / 2 + 1 && caps.search <= message / 2 + 1 && caps.glob <= message / 2 + 1, "{message}");
        assert!(caps.read <= crate::workspace::limits::READ_MAX_BYTES);
    }
}

/// An invalid registry file means no workspace tools, and one journal entry per distinct
/// problem (not one per turn), without the file's text.
#[test]
fn an_invalid_registry_means_no_workspace_tools_and_is_journaled_once_per_problem() {
    let secret = "/home/someone/CANARY_REGISTRY_PATH";
    let invalid = || Registry::from_toml(&format!("[repo.Bad_Name]\ncheckout = \"{secret}\"\nworktree_parent = \"relative\"\nbase = \"main\"\n"));
    let query = |after: u64| serde_json::from_str::<serde_json::Value>(&crate::error_journal::query(after)).unwrap();
    let ours = |after: u64| query(after)["entries"].as_array().unwrap().iter().filter(|entry| entry["category"] == "workspace" && entry["source"] == "registry").count();
    let before = query(u64::MAX - 1)["last_seq"].as_u64().unwrap();
    for _ in 0..5 {
        assert!(workspace::decide(invalid()).is_none());
    }
    assert_eq!(ours(before), 1, "five turns with the same broken file");
    assert!(workspace::decide(Registry::from_toml("[repo.other-problem]\ncheckout = \"relative\"\nworktree_parent = \"/b\"\nbase = \"main\"\n")).is_none());
    assert_eq!(ours(before), 2, "another problem is reported");
    assert!(workspace::decide(Ok(Registry::default())).is_none(), "an empty registry");
    assert!(workspace::decide(Registry::from_toml("[repo.ok]\ncheckout = \"/a\"\nworktree_parent = \"/b\"\nbase = \"main\"\n")).is_some());
    assert_eq!(ours(before), 2);
    let journal = query(before).to_string();
    assert!(!journal.contains("CANARY") && !journal.contains("Bad_Name"), "{journal}");
}

/// Every workspace name, including future slices, uses metadata-only training
/// records: known STATIC field names/sizes, unknown-field counts/combined sizes,
/// execution success and result bytes. No correlation id, key, value or prose.
#[test]
fn the_training_log_form_of_a_workspace_call_has_no_values_and_no_content() {
    use crate::llm::ToolCall;
    let call = |name: &str, arguments: serde_json::Value| ToolCall { id: "CANARY_CORRELATION_ID".into(), name: name.into(), arguments: arguments.to_string() };
    for name in ALL_NAMES {
        let original = call(name, json!({"path": "secrets/CANARY_PATH.env", "content": "CANARY_VALUE=hunter2", "offset": 12,
            "CANARY_UNKNOWN_KEY": {"nested": "CANARY_NESTED_VALUE"}}));
        let logged = workspace::logged_call(&original);
        assert_eq!((logged.id.as_str(), logged.name.as_str()), ("workspace", *name));
        let args: Value = serde_json::from_str(&logged.arguments).unwrap();
        assert!(args["unknown_fields"]["count"].as_u64().unwrap() >= 2, "{name}: {args}");
        assert!(args["unknown_fields"]["bytes"].as_u64().unwrap() > 0);
        if *name == "read" {
            assert_eq!(args["fields"], json!({"path": {"bytes": 25}, "offset": {"bytes": 2}}));
        }
        assert!(!logged.arguments.contains("CANARY") && !logged.arguments.contains("hunter2"), "{name}: {logged:?}");
        let raw = "CANARY_HEADER: lines 1-2\n1: CANARY_VALUE=hunter2\n2: more";
        for success in [true, false] {
            let result: Value = serde_json::from_str(&workspace::logged_result(&original, raw, success)).unwrap();
            assert_eq!(result, json!({"tool": name, "success": success, "result_bytes": raw.len()}));
        }
    }
    let broken = ToolCall { id: "c2".into(), name: "write".into(), arguments: "{\"content\": \"CANARY_VALUE".into() };
    assert_eq!(workspace::logged_call(&broken).arguments, r#"{"unparsed_bytes":25}"#);
    assert_eq!(serde_json::from_str::<Value>(&workspace::logged_result(&broken, "CANARY_ERROR", false)).unwrap(),
        json!({"tool": "write", "success": false, "result_bytes": 12}));
    let other = call("mutate", json!({"source": "+module M"}));
    assert_eq!(workspace::logged_call(&other), other);
    assert_eq!(workspace::logged_result(&other, "{\"success\":true}", true), "{\"success\":true}");
    // A shared workspace refusal overrides a peer's ordinary content policy:
    // arbitrary peer ids, keys, values and even invented names stay out.
    let tools = vec![ToolDefinition { name: "inspect".into(), description: String::new(), parameters: json!({}) }];
    for name in ["inspect", "CANARY_INVENTED_TOOL"] {
        let peer = call(name, json!({"CANARY_PEER_KEY":"CANARY_PEER_VALUE"}));
        let (logged, feedback) = workspace::logged_skipped(&peer, "CANARY_SHARED_ERROR", &tools);
        let text = format!("{logged:?} {feedback}");
        assert!(!text.contains("CANARY"), "{text}");
        assert_eq!(logged.name, if name == "inspect" { "inspect" } else { "unknown" });
        assert_eq!(serde_json::from_str::<Value>(&feedback).unwrap()["success"], false);
    }
}

/// The one path that copies stored session history into the memory graph (the operator
/// command `adapsis memory-migrate`) leaves out the results of workspace tool batches.
/// Other tool results are imported as before (reported for #145, not changed here).
#[test]
fn memory_migration_leaves_out_workspace_tool_results() {
    use crate::llm::ToolCall;
    let call = |id: &str, name: &str| ToolCall { id: id.into(), name: name.into(), arguments: "{}".into() };
    let mut session = crate::session::Session::new();
    {
        let conv = session.meta.conversations.get_or_create("telegram:100000001");
        conv.push_user("read the notes");
        conv.push_native_batch("Reading.".into(), vec![call("c1", "read")], vec!["notes.txt: MIGRATE_CANARY_FILE".into()]);
        // Ids are unique within a batch/turn, not over a whole conversation.
        conv.push_user("inspect now");
        conv.push_native_batch("Inspecting.".into(), vec![call("c1", "inspect")], vec!["MIGRATE_REUSED_ID_RESULT".into()]);
        conv.push_native_batch("Mixed.".into(), vec![call("mix1", "glob"), call("mix2", "inspect")], vec!["MIGRATE_CANARY_GLOB".into(), "MIGRATE_CANARY_MIXED".into()]);
        conv.push_native_batch("".into(), vec![call("mix2", "evaluate")], vec!["MIGRATE_OTHER_TOOL_RESULT".into()]);
        // Reusing an id in ordinary text cannot make it a tool result.
        conv.push_user("ordinary text mentioning c1 and mix2");
        conv.push_assistant("The notes say hello.");
        assert_eq!(workspace::result_positions(&conv.messages), [2, 7, 8].into_iter().collect());
        // Malformed/orphan batches follow the model-history policy, not id matching.
        conv.messages.push(crate::session::ChatMessage { role: "tool".into(), content: "MIGRATE_CANARY_ORPHAN".into(), tool_call_id: Some("c1".into()), ..Default::default() });
        let last = conv.messages.len() - 1;
        assert!(workspace::result_positions(&conv.messages).contains(&last));
        let malformed_start = conv.messages.len();
        conv.push_native_batch("MIGRATE_CANARY_MALFORMED".into(), vec![call("dup", "read"), call("dup", "inspect")],
            vec!["MIGRATE_CANARY_DUP_READ".into(), "MIGRATE_CANARY_DUP_INSPECT".into()]);
        let omitted = workspace::result_positions(&conv.messages);
        assert!((malformed_start..conv.messages.len()).all(|index| omitted.contains(&index)));
    }
    let directory = tempfile::tempdir().unwrap();
    let (session_path, database) = (directory.path().join("session.json"), directory.path().join("memory.lbug"));
    session.save(&session_path).unwrap();
    crate::migrate_conversations_to_memory(&session_path, &database).unwrap();
    let graph = crate::memory_graph::MemoryGraph::open(&database).unwrap();
    let imported: Vec<String> = graph.pending_context_messages("telegram:100000001", usize::MAX).unwrap().into_iter().map(|message| message.content).collect();
    let all = imported.join("\n");
    assert!(!all.contains("MIGRATE_CANARY"), "{all}");
    for kept in ["read the notes", "The notes say hello.", "MIGRATE_OTHER_TOOL_RESULT", "MIGRATE_REUSED_ID_RESULT", "ordinary text mentioning c1 and mix2"] {
        assert!(all.contains(kept), "missing {kept}: {all}");
    }
}
