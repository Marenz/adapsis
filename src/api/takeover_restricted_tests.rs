//! Native takeover turns for speakers who cannot change code (DESIGN L3, #129): guests
//! and execute-only speakers get their own tool lists, a restricted `run`, and a per-turn
//! tool budget. Mock provider; turn-bound tools are answered by an in-memory recorder, so
//! no web request, music job or CAD author is ever started.
use super::*;

pub(super) const GUEST: &str = "telegram:user:100000777";
pub(super) const GUEST_DM: &str = "telegram:user:100000777";
const VIEWER: &str = "telegram:user:100000778";
const VIEWER_GROUP: &str = "telegram:group:-100000086";

pub(super) fn permissions() -> Arc<crate::permissions::PermissionConfig> {
    Arc::new(
        toml::from_str(
            r#"
        [principals]
        "telegram:user:100000777" = "guest"
        "telegram:user:100000779" = "ghost"
        "telegram:user:100000780" = "viewer"
        "telegram:user:100000781" = "memguest"
        [guest_profiles]
        guest = ["web_search", "music_generate", "memory_inspect", "cad_prepare"]
        memguest = ["memory_remember"]
        [model.memguest]
        user = "none"
        lib = "none"
        secret = "none"
        [groups]
        lib = ["RestrictedLib"]
        secret = ["RestrictedSecret"]
        [model.guest]
        user = "none"
        lib = "none"
        secret = "none"
        [model.viewer]
        user = "execute"
        lib = "execute"
        secret = "none"
        [model.takeover-fixture]
        user = "write"
        lib = "write"
        secret = "write"
    "#,
        )
        .unwrap(),
    )
}

const LIBRARY: &str = "+module RestrictedLib\n+fn double (x:Int)->Int\n  +return x * 2\n+end\n+module RestrictedSecret\n+fn token ()->String\n  +return \"hidden-token\"\n+end\n";

async fn harness() -> Harness {
    let mut harness = Harness::new(true);
    harness.permissions = permissions();
    let mut program = crate::ast::Program::default();
    for op in crate::parser::parse(LIBRARY).unwrap() {
        crate::validator::apply_and_validate(&mut program, &op).unwrap();
    }
    program.require_modules = true;
    *harness.program.write().await = program;
    harness
}

type Seen = Arc<Mutex<Vec<String>>>;

/// Records every turn-bound request and answers it; nothing leaves the process.
fn recording_io() -> (tokio::sync::mpsc::Sender<crate::coroutine::IoRequest>, Seen) {
    use crate::coroutine::IoRequest;
    let (tx, mut rx) = tokio::sync::mpsc::channel::<IoRequest>(64);
    let seen = Seen::default();
    let log = seen.clone();
    tokio::spawn(async move {
        while let Some(request) = rx.recv().await {
            match request {
                IoRequest::Scoped { request, reply } => {
                    let text = format!("{request:?}");
                    log.lock().unwrap().push(text.clone());
                    let _ = reply.send(Ok(format!("result of {}", text.split_whitespace().next().unwrap_or_default())));
                }
                IoRequest::MemoryInspect { context_id, principal_id, reply, .. } => {
                    log.lock().unwrap().push(format!("MemoryInspect {context_id} {principal_id}"));
                    let _ = reply.send(Ok("no memories".into()));
                }
                IoRequest::MemoryRemember { reply, .. } => {
                    log.lock().unwrap().push("MemoryRemember".into());
                    let _ = reply.send(Ok("remembered".into()));
                }
                _ => log.lock().unwrap().push("UNEXPECTED".into()),
            }
        }
    });
    (tx, seen)
}

fn tool_names(request: &Value) -> Vec<String> {
    request["tools"]
        .as_array()
        .map(|tools| tools.iter().map(|tool| tool["function"]["name"].as_str().unwrap().to_string()).collect())
        .unwrap_or_default()
}

fn result_of(request: &Value, id: &str) -> Value {
    let message = request["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["tool_call_id"] == id)
        .unwrap_or_else(|| panic!("no result for call {id}: {}", request["messages"]));
    serde_json::from_str(message["content"].as_str().unwrap()).unwrap()
}

fn run(id: &str, source: &str) -> Value {
    call(id, "run", json!({"source": source}))
}

#[tokio::test]
async fn a_guest_turn_is_native_with_exactly_the_profile_tools_run_and_finish() {
    let harness = harness().await;
    let (io, seen) = recording_io();
    let provider = provider(vec![
        response("", vec![
            call("search", "web_search", json!({"query": "adapsis"})),
            run("block", "hits: str = await web_search(\"runtime\")\nn: int = len(split(hits, \" \"))\nreturn concat(to_string(n), \" words\")\n"),
        ]),
        response("", vec![
            // Everything outside the list is refused, in any form.
            call("forged-mutate", "mutate", json!({"source": "+module GuestMade"})),
            call("forged-evaluate", "evaluate", json!({"function": "RestrictedLib.double", "input": "2"})),
            call("forged-inspect", "inspect", json!({"kind": "symbols"})),
            call("not-in-profile", "web_read", json!({"url": "http://example.org/"})),
        ]),
        response("", vec![run("user-fn", "return RestrictedLib.double(2)\n")]),
        response("", vec![run("file", "text: str = await file_read(\"/etc/hostname\")\nreturn text\n")]),
        response("", vec![run("other-tool", "page: str = await web_read(\"http://example.org/\")\nreturn page\n")]),
        response("", vec![run("host", "return home_dir()\n")]),
        response("", vec![run("scope", "return counter\n")]),
        response("Found it.", vec![call("done", "finish", json!({}))]),
    ])
    .await;
    let reply = harness.turn_with_io(&provider, io, GUEST_DM, GUEST, "search please").await;
    assert_eq!(reply, "Found it.");
    let requests = provider.requests.lock().unwrap().clone();
    assert_eq!(tool_names(&requests[0]), ["music_generate", "web_search", "memory_inspect", "cad_prepare", "run", "finish"]);
    let system = system_text(&requests[0]);
    assert!(system.contains("Fixed native tool protocol") && system.contains("Combining steps: run"), "{system}");
    for absent in ["mutate", "inspect", "evaluate", "!eval", "Preferred workflow", "memory_remember"] {
        assert!(!system.contains(absent), "the guest prompt names `{absent}`");
    }
    let tail = all_text(&requests[0]);
    assert!(tail.contains("Add or modify code: no"), "{tail}");
    assert!(tail.contains("exactly the native tools supplied"), "{tail}");
    assert!(!tail.contains("Guest tools: use !eval"), "legacy guest instructions leaked into a native turn");
    let run_tool = requests[0]["tools"].as_array().unwrap().iter().find(|tool| tool["function"]["name"] == "run").unwrap();
    assert!(run_tool["function"]["parameters"]["properties"].get("module").is_none());
    assert!(run_tool["function"]["description"].as_str().unwrap().contains("music_generate, web_search, memory_inspect, cad_prepare"));

    assert_eq!(result_of(&requests[1], "search")["success"], true);
    assert_eq!(result_of(&requests[1], "block")["output"], "3 words");
    // One refused call skips its whole batch, as for every native batch.
    for id in ["forged-mutate", "forged-evaluate", "forged-inspect", "not-in-profile"] {
        let refused = requests[2]["messages"].to_string();
        assert!(refused.contains(&format!("\\\"tool_call_id\\\":\\\"{id}\\\"")) || refused.contains(id), "{id}");
    }
    assert!(requests[2]["messages"].to_string().contains("guest turns are not eligible for `mutate`"));
    for (index, id, fragment) in [
        (3, "user-fn", "function `RestrictedLib.double` is not available in this conversation"),
        (4, "file", "`file_read` is not available in this conversation"),
        (5, "other-tool", "`web_read` is not available in this conversation"),
        (6, "host", "`home_dir` is not available in this conversation"),
        (7, "scope", "name `counter` is not bound in this block"),
    ] {
        let result = requests[index]["messages"].to_string();
        assert!(result.contains(id) && result.contains(fragment), "{id}: {result}");
    }
    // Exactly the two admitted searches crossed the IO boundary, bound to no caller argument.
    assert_eq!(*seen.lock().unwrap(), ["Search { query: \"adapsis\" }", "Search { query: \"runtime\" }"]);
    assert!(harness.program.read().await.modules.iter().all(|m| m.name != "GuestMade"));
    assert_eq!(harness.meta.lock().unwrap().revision, 0);
}

#[tokio::test]
async fn an_execute_only_turn_is_native_with_inspect_evaluate_run_finish_and_memory() {
    let harness = harness().await;
    harness.set_profile(VIEWER_GROUP, Some("viewer"));
    let (io, seen) = recording_io();
    let provider = provider(vec![
        response("", vec![
            call("eval", "evaluate", json!({"function": "RestrictedLib.double", "input": "21"})),
            run("block", "a: int = RestrictedLib.double(4)\nreturn a + 1\n"),
            call("mem", "memory_inspect", json!({})),
        ]),
        response("", vec![call("forged", "mutate", json!({"source": "+module ViewerMade"}))]),
        response("", vec![run("secret", "return RestrictedSecret.token()\n")]),
        response("", vec![call("secret-eval", "evaluate", json!({"function": "RestrictedSecret.token"}))]),
        response("", vec![run("io", "out: str = await shell_exec(\"id\")\nreturn out\n")]),
        response("", vec![run("spawn", "spawn(RestrictedLib.double(1))\nreturn 1\n")]),
        response("", vec![run("module", "return 1\n"), call("scoped", "run", json!({"source": "return double(1)\n", "module": "RestrictedLib"}))]),
        response("Done.", vec![call("done", "finish", json!({}))]),
    ])
    .await;
    let reply = harness.turn_with_io(&provider, io, VIEWER_GROUP, VIEWER, "compute").await;
    assert_eq!(reply, "Done.");
    let requests = provider.requests.lock().unwrap().clone();
    assert_eq!(
        tool_names(&requests[0]),
        ["inspect", "evaluate", "run", "finish", "memory_inspect", "memory_remember", "memory_correct", "memory_forget"]
    );
    let system = system_text(&requests[0]);
    assert!(system.contains("Combining steps: run") && !system.contains("Preferred workflow"), "{system}");
    assert!(result_of(&requests[1], "eval")["output"].as_str().unwrap().contains("= 42"));
    assert_eq!(result_of(&requests[1], "block")["output"], "9");
    assert_eq!(result_of(&requests[1], "mem")["success"], true);
    for (index, fragment) in [
        (2, "not write-capable (`mutate` is not available to this speaker)"),
        (3, "permission denied for authored reference `RestrictedSecret.token` (need Execute)"),
        (4, "module 'RestrictedSecret' needs Execute"),
        (5, "`shell_exec` is not available in this conversation"),
        (6, "`spawn` is not available in this conversation"),
        (7, "a `module` scope is not available in this conversation"),
    ] {
        let result = requests[index]["messages"].to_string();
        assert!(result.contains(fragment), "{fragment}: {result}");
        assert!(!result.contains("hidden-token"));
    }
    assert_eq!(*seen.lock().unwrap(), [format!("MemoryInspect {VIEWER_GROUP} {VIEWER}")]);
    assert!(harness.program.read().await.modules.iter().all(|m| m.name != "ViewerMade"));
}

/// Decision 4: a turn that cannot be classified with certainty is a guest with an empty
/// tool list. Three ways to be unclassifiable, each with only `run` (pure) and `finish`.
#[tokio::test]
async fn an_unclassifiable_turn_gets_the_most_restricted_class() {
    for (label, speaker, context, context_profile) in [
        // The principal maps to a profile that has no entry at all.
        ("missing profile entry", "telegram:user:100000779", "telegram:user:100000779", None),
        // The context names a profile that does not exist.
        ("missing context profile", VIEWER, "telegram:group:-100000087", Some("no-such-profile")),
        // The principal's profile (non-guest) and the context's profile (guest) disagree
        // about whether this is a guest turn.
        ("guest signals disagree", "telegram:user:100000780", "telegram:group:-100000088", Some("guest")),
    ] {
        let harness = harness().await;
        harness.set_profile(context, context_profile);
        let (io, seen) = recording_io();
        let provider = provider(vec![
            response("", vec![run("pure", "return concat(\"a\", to_string(1 + 1))\n")]),
            response("", vec![call("tool", "web_search", json!({"query": "x"}))]),
            response("", vec![run("in-block", "hits: str = await web_search(\"x\")\nreturn hits\n")]),
            response("", vec![call("eval", "evaluate", json!({"function": "RestrictedLib.double", "input": "2"}))]),
            response("", vec![run("user-fn", "return RestrictedLib.double(2)\n")]),
            response("Nothing more.", vec![call("done", "finish", json!({}))]),
        ])
        .await;
        let reply = harness.turn_with_io(&provider, io, context, speaker, "hello").await;
        assert_eq!(reply, "Nothing more.", "{label}");
        let requests = provider.requests.lock().unwrap().clone();
        assert_eq!(tool_names(&requests[0]), ["run", "finish"], "{label}");
        assert_eq!(result_of(&requests[1], "pure")["output"], "a2", "{label}");
        for (index, fragment) in [
            (2, "guest turns are not eligible for `web_search`"),
            (3, "`web_search` is not available in this conversation"),
            (4, "guest turns are not eligible for `evaluate`"),
            (5, "function `RestrictedLib.double` is not available in this conversation"),
        ] {
            let result = requests[index]["messages"].to_string();
            assert!(result.contains(fragment), "{label}: {fragment}: {result}");
        }
        assert!(seen.lock().unwrap().is_empty(), "{label}: {:?}", seen.lock().unwrap());
        assert!(all_text(&requests[0]).contains("Add or modify code: no"), "{label}");
    }
}

/// The turn budget is shared by direct tool calls and calls inside blocks. Legacy turns
/// have no such bound; this limit is new (DESIGN L3).
#[tokio::test]
async fn a_restricted_turn_stops_at_its_tool_call_budget() {
    let limit = crate::run_limits::RunLimits::restricted().turn_tool_calls.unwrap();
    assert_eq!(limit, 24);
    let harness = harness().await;
    let (io, seen) = recording_io();
    let direct = |from: usize, to: usize| -> Vec<Value> {
        (from..to).map(|n| call(&format!("s{n}"), "web_search", json!({"query": format!("q{n}")}))).collect()
    };
    let block = "a: str = await web_search(\"b1\")\nb: str = await web_search(\"b2\")\nc: str = await web_search(\"b3\")\nd: str = await web_search(\"b4\")\nreturn d\n";
    let provider = provider(vec![
        response("", direct(0, 20)),
        // 20 used: the block may make four calls, and does.
        response("", vec![run("block", block)]),
        // 24 used: a direct call and a block are both refused, with nothing sent.
        response("", vec![call("over", "web_search", json!({"query": "over"}))]),
        response("", vec![run("block-over", "x: str = await web_search(\"over-in-block\")\nreturn x\n")]),
        response("", vec![run("pure", "return 1 + 1\n")]),
        response("That is all.", vec![call("done", "finish", json!({}))]),
    ])
    .await;
    let reply = harness.turn_with_io(&provider, io, GUEST_DM, GUEST, "search a lot").await;
    assert_eq!(reply, "That is all.");
    let requests = provider.requests.lock().unwrap().clone();
    assert_eq!(result_of(&requests[2], "block")["output"], "result of Search");
    for (index, id) in [(3, "over"), (4, "block-over")] {
        let result = result_of(&requests[index], id);
        assert_eq!(result["success"], false, "{id}");
        assert!(result["output"].as_str().unwrap().contains("this turn has used its 24 tool calls"), "{id}: {result}");
    }
    // Pure computation needs no tool call and still works.
    assert_eq!(result_of(&requests[5], "pure")["output"], "2");
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 24, "{seen:?}");
    assert!(seen.iter().all(|request| !request.contains("over")));
}

/// A block cannot multiply a costly tool or run more IO than its per-block limit; what it
/// already did stays done and is reported as such.
#[tokio::test]
async fn a_block_is_held_to_its_io_and_costly_tool_limits() {
    let harness = harness().await;
    let (io, seen) = recording_io();
    let searches: String = (0..9).map(|n| format!("s{n}: str = await web_search(\"q{n}\")\n")).collect();
    let provider = provider(vec![
        response("", vec![run("music", "a: str = await music_generate(\"calm piano\", 30)\nb: str = await music_generate(\"loud drums\", 30)\nreturn b\n")]),
        response("", vec![run("many", &format!("{searches}return s8\n"))]),
        // Consent flag reaches the capability exactly as written; nothing defaults it to true.
        response("", vec![
            call("cad", "cad_prepare", json!({"title": "hook", "design_brief": "a wall hook, 40 mm"})),
            call("cad-images", "cad_prepare", json!({"title": "hook", "design_brief": "like the photo", "use_recent_images": true})),
        ]),
        response("Started.", vec![call("done", "finish", json!({}))]),
    ])
    .await;
    harness.turn_with_io(&provider, io, GUEST_DM, GUEST, "make things").await;
    let requests = provider.requests.lock().unwrap().clone();
    let music = result_of(&requests[1], "music");
    assert_eq!(music["success"], false);
    assert!(music["output"].as_str().unwrap().contains("`music_generate` may be called 1 time(s) per block"), "{music}");
    let many = result_of(&requests[2], "many");
    assert!(many["output"].as_str().unwrap().contains("IO call 9 of this block; the limit for this conversation is 8"), "{many}");
    let seen = seen.lock().unwrap();
    let music_jobs: Vec<&String> = seen.iter().filter(|request| request.starts_with("Music")).collect();
    assert_eq!(music_jobs.len(), 1, "{seen:?}");
    assert!(music_jobs[0].contains("calm piano") && music_jobs[0].contains(GUEST_DM), "{music_jobs:?}");
    assert_eq!(seen.iter().filter(|request| request.starts_with("Search")).count(), 8, "{seen:?}");
    let cad: Vec<&String> = seen.iter().filter(|request| request.starts_with("Cad")).collect();
    assert_eq!(cad.len(), 2, "{seen:?}");
    assert!(cad[0].contains("use_recent_images: false") && cad[0].contains("material: \"PLA\""), "{}", cad[0]);
    assert!(cad[1].contains("use_recent_images: true") && cad[1].contains(GUEST), "{}", cad[1]);
}

/// A global canonical memory is injected into every conversation, the owner's included,
/// so a speaker who cannot change code may not assert one. The builtin decides, so the
/// native tool and the legacy `!eval` path give the same answer; context scope still works.
#[tokio::test]
async fn global_memories_are_refused_for_speakers_who_cannot_change_code() {
    const MEMGUEST: &str = "telegram:user:100000781";
    for native in [true, false] {
        for (label, speaker, context, profile, refusal) in [
            ("execute-only", VIEWER, "telegram:group:-100000090", Some("viewer"), "global memories require a write-capable speaker"),
            ("guest", MEMGUEST, MEMGUEST, None, "guest memories must use context scope"),
        ] {
            let mut harness = harness().await;
            harness.native_tools = native;
            harness.set_profile(context, profile);
            let io = spawn_memory_io(harness.graph.clone(), Arc::new(crate::memory_graph::MemoryEmbedder::deterministic()));
            let here = format!("ContextNote {label} {native}");
            let everywhere = format!("GlobalNote {label} {native}");
            let outputs = if native {
                vec![
                    tool("ctx", "memory_remember", json!({"content": here})),
                    tool("glob", "memory_remember", json!({"content": everywhere, "scope": "global"})),
                    response("Noted.", vec![call("done", "finish", json!({}))]),
                ]
            } else {
                vec![
                    response(&format!("<code>!eval memory_remember(\"{here}\")</code>"), vec![]),
                    response(&format!("<code>!eval memory_remember(\"{everywhere}\", \"global\")</code>"), vec![]),
                    response("Noted.\n<code>!done</code>", vec![]),
                ]
            };
            let provider = provider(outputs).await;
            let reply = harness.turn_with_io(&provider, io, context, speaker, "remember this").await;
            assert_eq!(reply, "Noted.", "{label} native={native}");
            let requests = provider.requests.lock().unwrap().clone();
            assert_eq!(requests.len(), 3, "{label} native={native}");
            assert_eq!(requests[0].get("tools").is_some(), native, "{label}");
            // The context-scoped note was accepted, the global one refused with its reason.
            assert!(all_text(&requests[1]).contains("remembered as"), "{label} native={native}: {}", all_text(&requests[1]));
            let after_global = all_text(&requests[2]);
            assert!(after_global.contains(refusal), "{label} native={native}: {after_global}");
            let stored: Vec<String> = harness
                .graph
                .canonical_for_context(speaker, context, 40, 8_000)
                .unwrap()
                .into_iter()
                .map(|memory| memory.content)
                .collect();
            assert_eq!(stored, [here], "{label} native={native}");
        }
    }
}

/// The write-capable speaker is unchanged on the legacy path too (the native path is
/// `native_takeover_memory_remember_global_scope_works_for_write_turn`).
#[tokio::test]
async fn a_write_capable_legacy_turn_still_remembers_globally() {
    let context = "telegram:100000001";
    let mut harness = harness().await;
    harness.native_tools = false;
    let io = spawn_memory_io(harness.graph.clone(), Arc::new(crate::memory_graph::MemoryEmbedder::deterministic()));
    let provider = provider(vec![
        response("<code>!eval memory_remember(\"OwnerGlobalNote\", \"global\")</code>", vec![]),
        response("Noted.\n<code>!done</code>", vec![]),
    ])
    .await;
    assert_eq!(harness.turn_with_io(&provider, io, context, ADMIN, "remember globally").await, "Noted.");
    let requests = provider.requests.lock().unwrap().clone();
    assert!(all_text(&requests[1]).contains("remembered as"), "{}", all_text(&requests[1]));
    let stored = harness.graph.canonical_for_context(ADMIN, context, 40, 8_000).unwrap();
    assert!(stored.iter().any(|memory| memory.content == "OwnerGlobalNote"), "{stored:?}");
}
