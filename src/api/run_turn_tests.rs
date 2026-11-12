//! Native `run` inside real HTTP turns (#129): the existing batch rules apply unchanged
//! (validate all, then execute sequentially; duplicate ids; `finish` alone), a block sees
//! what an earlier call of the same batch declared, and training data is recorded exactly
//! as for every other native tool. Mock provider, in-memory IO.
use super::*;

const TWICE: &str = "def twice(x: int) -> int:\n    return x * 2\n";

fn python_mutate(id: &str, module: &str) -> Value {
    call(id, "mutate", json!({"language":"python","module":module,"source":TWICE}))
}

fn run_call(id: &str, source: &str) -> Value {
    call(id, "run", json!({"source": source}))
}

/// Every `file_write` is recorded and acknowledged; nothing touches the filesystem.
fn recording_io(fixture: &mut Fixture) -> Arc<Mutex<Vec<String>>> {
    let (sender, mut receiver) = tokio::sync::mpsc::channel(8);
    fixture.config.io_sender = Some(sender);
    let log = Arc::new(Mutex::new(Vec::new()));
    let seen = log.clone();
    tokio::spawn(async move {
        while let Some(request) = receiver.recv().await {
            if let crate::coroutine::IoRequest::FileWrite { path, data, reply } = request {
                seen.lock().unwrap().push(format!("{path}={data}"));
                let _ = reply.send(Ok(()));
            }
        }
    });
    log
}

fn tool_results(requests: &[Value], id: &str) -> Vec<Value> {
    let Some(last) = requests.last() else { return vec![] };
    last["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|message| message["tool_call_id"] == id)
        .map(|message| serde_json::from_str(message["content"].as_str().unwrap()).unwrap())
        .collect()
}

async fn batch_after_mutate(streaming: bool, module: &str) {
    let mut fixture = Fixture::new(
        vec![
            response(
                "",
                vec![
                    python_mutate("declare", module),
                    run_call("use", &format!("n: int = {module}.twice(21)\nreturn n\n")),
                ],
            ),
            tool("done", "finish", json!({})),
        ],
        "",
    )
    .await;
    recording_io(&mut fixture);
    let result = fixture.run(streaming).await;
    assert_eq!(result.completed, Some(true), "results={:?}; reply={}", result.results, result.reply);
    assert!(result.results.iter().any(|r| r.success && r.message == "42"), "{:?}", result.results);
    let requests = fixture.requests.lock().unwrap().clone();
    let names: Vec<&str> = requests[0]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["function"]["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["inspect", "mutate", "test", "evaluate", "run", "finish"]);
    // The model receives the value within the turn, in the tool result of its own call.
    let delivered = tool_results(&requests, "use");
    assert_eq!(delivered.len(), 1);
    assert_eq!(delivered[0]["output"], "42");
    assert_eq!(delivered[0]["success"], true);
    // Only the mutate is a mutation: one revision, one replay record, one module, one function.
    let meta = fixture.config.meta.lock().unwrap();
    assert_eq!(meta.revision, 1);
    assert_eq!(meta.mutations.len(), 1);
    assert_eq!(meta.sources.len(), 1);
    assert!(!serde_json::to_string(&meta.sources).unwrap().contains("twice(21)"));
    drop(meta);
    let program = fixture.config.program.read().await;
    assert_eq!(program.modules.len(), 1);
    assert_eq!(program.modules[0].name, module);
    assert_eq!(program.modules[0].functions.len(), 1);
    assert_eq!(program.modules[0].functions[0].name, "twice");
}

#[tokio::test]
async fn native_http_run_sees_a_function_declared_earlier_in_the_same_batch() {
    batch_after_mutate(false, "RunTurnHttp").await;
}

#[tokio::test]
async fn native_stream_run_sees_a_function_declared_earlier_in_the_same_batch() {
    batch_after_mutate(true, "RunTurnStream").await;
}

#[tokio::test]
async fn native_run_refusal_skips_the_whole_batch_before_any_call_executes() {
    for (index, bad) in [
        "import os\nreturn 1\n",
        "text = await file_read(\"/tmp/x\")\nreturn text\n",
        "done: str = await mutate(\"+module RunTurnInjected\")\nreturn done\n",
        "def helper() -> int:\n    return 1\n",
    ]
    .into_iter()
    .enumerate()
    {
        let module = format!("RunTurnSkipped{index}");
        let mut fixture = Fixture::new(
            vec![response(
                "",
                vec![
                    run_call("first", "written: str = await file_write(\"fixture://skipped\", \"x\")\nreturn written\n"),
                    python_mutate("declare", &module),
                    run_call("bad", bad),
                ],
            )],
            "",
        )
        .await;
        let log = recording_io(&mut fixture);
        assert!(fixture.run(false).await.has_errors);
        assert!(log.lock().unwrap().is_empty(), "{bad:?}: an earlier run of a refused batch executed");
        assert!(fixture.config.program.read().await.modules.is_empty(), "{bad:?}");
        let messages = fixture.config.meta.lock().unwrap().conversations.get("main").unwrap().to_llm_messages();
        let skipped: Vec<_> = messages.iter().filter(|m| m.role == crate::llm::ChatRole::Tool).collect();
        assert_eq!(skipped.len(), 3);
        assert!(skipped.iter().all(|m| serde_json::to_string(&m.content).unwrap().contains("Batch skipped before execution")), "{bad:?}");
    }
}

#[tokio::test]
async fn native_run_follows_the_duplicate_id_and_finish_rules() {
    let write = "written: str = await file_write(\"fixture://count\", \"x\")\nreturn 1\n";
    let mut fixture = Fixture::new(
        vec![
            // finish may not share a batch: nothing in it executes.
            response("", vec![run_call("shared", write), call("early", "finish", json!({}))]),
            tool("once", "run", json!({"source": write})),
            // The same id again is rejected without replaying the block.
            tool("once", "run", json!({"source": write})),
            // The same block under a new id is a new call.
            tool("again", "run", json!({"source": write})),
            tool("done", "finish", json!({})),
        ],
        "",
    )
    .await;
    let log = recording_io(&mut fixture);
    let result = fixture.run(false).await;
    assert_eq!(result.completed, Some(true), "{:?}", result.results);
    assert!(result.has_errors, "the rejected rounds are reported");
    assert_eq!(log.lock().unwrap().len(), 2, "{:?}", log.lock().unwrap());
    let messages = fixture.config.meta.lock().unwrap().conversations.get("main").unwrap().to_llm_messages();
    for (id, count) in [("shared", 0), ("early", 0), ("once", 1), ("again", 1), ("done", 1)] {
        assert_eq!(messages.iter().filter(|m| m.tool_call_id.as_deref() == Some(id)).count(), count, "{id}");
    }
    assert!(fixture.config.program.read().await.modules.is_empty());
    assert_eq!(fixture.config.meta.lock().unwrap().revision, 0);
}

/// The training log is a deliberate full-content artifact (not the journal). A native
/// batch records its tool calls and results there whatever the tool is; `run` adds no
/// second record and is not left out.
#[tokio::test]
async fn native_run_is_recorded_for_training_like_every_other_native_tool() {
    let mut fixture = Fixture::new(
        vec![
            response("", vec![
                call("eval", "inspect", json!({"kind":"symbols"})),
                run_call("block", "return \"TRAINING_RUN_RESULT\"\n"),
            ]),
            tool("done", "finish", json!({})),
        ],
        "",
    )
    .await;
    recording_io(&mut fixture);
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("training.jsonl");
    fixture.config.training_log = Some(Arc::new(tokio::sync::Mutex::new(
        tokio::fs::File::create(&path).await.unwrap(),
    )));
    assert_eq!(fixture.run(false).await.completed, Some(true));
    let recorded = tokio::fs::read_to_string(&path).await.unwrap();
    let entries: Vec<Value> = recorded.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
    assert_eq!(entries.len(), 2, "one entry per native batch: {recorded}");
    let batch = &entries[0];
    assert_eq!(batch["outcome"], "success");
    let code = batch["code"].as_str().unwrap();
    assert!(code.contains("\\\"inspect\\\"") || code.contains("\"name\":\"inspect\""), "{code}");
    assert!(code.contains("TRAINING_RUN_RESULT"), "the run call (its source) is recorded: {code}");
    let feedback = batch["feedback"].as_array().unwrap();
    assert_eq!(feedback.len(), 2, "one result per call");
    assert!(feedback[1].as_str().unwrap().contains("TRAINING_RUN_RESULT"));
    assert_eq!(recorded.matches("\\\"tool\\\":\\\"run\\\"").count(), 1, "{recorded}");
}

/// DESIGN L3 on HTTP: an execute-only speaker is native with `inspect`, `evaluate`,
/// restricted `run` and `finish` (no memory tools: HTTP turns have no turn identity).
/// A forged `mutate` skips its batch; nothing the turn does changes code.
#[tokio::test]
async fn native_http_execute_only_turn_has_its_own_tool_list_and_a_restricted_run() {
    let module = "RunTurnViewer";
    let mut fixture = Fixture::new(
        vec![
            response("", vec![
                call("eval", "evaluate", json!({"function": format!("{module}.twice"), "input": "4"})),
                run_call("block", &format!("n: int = {module}.twice(5)\nreturn concat(\"n=\", to_string(n))\n")),
                call("look", "inspect", json!({"kind": "symbols"})),
            ]),
            response("", vec![python_mutate("forged", "RunTurnViewerMade")]),
            response("", vec![run_call("io", "t: str = await file_read(\"/etc/hostname\")\nreturn t\n")]),
            response("", vec![call("scoped", "run", json!({"source": "return twice(1)\n", "module": module}))]),
            tool("done", "finish", json!({})),
        ],
        &format!("+module {module}\n+fn twice (x:Int)->Int\n  +return x * 2\n+end\n"),
    )
    .await;
    let log = recording_io(&mut fixture);
    fixture.config.access_level = crate::permissions::AccessLevel::ExecuteOnly;
    assert!(super::super::super::native_turn::eligible(&fixture.config));
    let result = fixture.run(false).await;
    assert_eq!(result.completed, Some(true), "{:?}", result.results);
    let requests = fixture.requests.lock().unwrap().clone();
    let names: Vec<&str> = requests[0]["tools"].as_array().unwrap().iter().map(|tool| tool["function"]["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["inspect", "evaluate", "run", "finish"]);
    let system = requests[0]["messages"][0]["content"].as_str().unwrap();
    assert!(system.contains("Fixed native tool protocol") && system.contains("Combining steps: run"), "{system}");
    assert!(!system.contains("Preferred workflow") && !system.contains("mutate"), "{system}");
    assert!(tool_results(&requests[1..2], "eval")[0]["output"].as_str().unwrap().contains("= 8"));
    assert_eq!(tool_results(&requests[1..2], "block")[0]["output"], "n=10");
    for (index, fragment) in [
        (2, "not write-capable (`mutate` is not available to this speaker)"),
        (3, "`file_read` is not available in this conversation"),
        (4, "a `module` scope is not available in this conversation"),
    ] {
        let text = requests[index]["messages"].to_string();
        assert!(text.contains(fragment), "{fragment}: {text}");
    }
    assert!(log.lock().unwrap().is_empty());
    let program = fixture.config.program.read().await;
    assert_eq!(program.modules.len(), 1);
    assert_eq!(program.modules[0].functions.len(), 1);
    assert_eq!(fixture.config.meta.lock().unwrap().revision, 1, "only the fixture prelude is a mutation");
}
