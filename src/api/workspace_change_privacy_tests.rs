use super::*;

#[tokio::test]
async fn workspace_changes_keep_payloads_commands_output_and_failures_out_of_public_and_training_sinks()
 {
    let repo = crate::workspace::test_support::RepoFixture::new();
    let site = repo.site("change-privacy");
    let workspace = crate::workspace::open(
        site.registry.get("change-privacy").unwrap(),
        "private",
        None,
    )
    .unwrap();
    site.bindings.claim("main", &workspace, &|_| None).unwrap();
    let file = "CANARY_CHANGE_FILE.txt";
    let mut fixture = Fixture::new(vec![
        response("",vec![
            call("write","write",json!({"path":file,"content":"CANARY_WRITE_BODY\n"})),
            call("edit","edit",json!({"path":file,"old":"WRITE_BODY","new":"EDIT_BODY"})),
            call("read","read",json!({"path":file})),
        ]),
        tool("command","bash",json!({"command":"printf '%s\\n' CANARY_COMMAND_OUTPUT","env":{"CANARY_ENV_NAME":"CANARY_ENV_VALUE"}})),
        tool("failure","bash",json!({"command":"printf '%s\\n' '{\"success\":true,\"CANARY_COMMAND_ERROR\":\"data\"}'; exit 9"})),
        tool("bad-edit","edit",json!({"path":file,"old":"no such text","new":"CANARY_NEW_TEXT"})),
        response("",vec![
            call("ordinary","inspect",json!({"kind":"symbols"})),
            call("bad-write","write",json!({"path":file,"content":"CANARY_REFUSED_VALUE","CANARY_EXTRA_KEY":"CANARY_UNKNOWN"})),
        ]),
        response("Done.",vec![call("finish","finish",json!({}))]),
    ],"").await;
    fixture.config.workspaces = Some(site);
    fixture.config.origin = Some(crate::api::origin::Origin::Local);
    let log = tempfile::NamedTempFile::new().unwrap();
    fixture.config.training_log = Some(Arc::new(tokio::sync::Mutex::new(
        tokio::fs::File::create(log.path()).await.unwrap(),
    )));
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let (mut events, sink) = (fixture.config.event_broadcast.subscribe(), seen.clone());
    let collector = tokio::spawn(async move {
        while let Ok(event) = events.recv().await {
            sink.lock().unwrap().push(event);
        }
    });
    let answer = fixture.run(false).await;
    assert_eq!(answer.completed, Some(true));
    assert!(answer.has_errors);
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    collector.abort();
    let training = tokio::fs::read_to_string(log.path()).await.unwrap();
    for text in [
        training.as_str(),
        &serde_json::to_string(&answer).unwrap(),
        &seen.lock().unwrap().join("\n"),
    ] {
        assert!(!text.contains("CANARY"), "{text}");
    }
    let entries: Vec<Value> = training
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let feedback = |row: usize, index: usize| {
        serde_json::from_str::<Value>(entries[row]["feedback"][index].as_str().unwrap()).unwrap()
    };
    assert_eq!(feedback(0, 0)["tool"], "write");
    assert_eq!(feedback(0, 1)["tool"], "edit");
    assert_eq!(feedback(1, 0)["success"], true);
    assert_eq!(feedback(2, 0)["success"], false);
    for n in [0, 1] {
        assert_eq!(feedback(4, n)["success"], false);
    }
    let requests = fixture.requests.lock().unwrap();
    assert!(
        requests[1].to_string().contains("CANARY_EDIT_BODY"),
        "the model gets the file content"
    );
    assert!(requests[2].to_string().contains("CANARY_COMMAND_OUTPUT"));
    assert!(requests[3].to_string().contains("CANARY_COMMAND_ERROR"));
    assert!(
        requests[5]
            .to_string()
            .contains("unexpected field `CANARY_EXTRA_KEY`")
    );
    assert_eq!(
        std::fs::read_to_string(workspace.path.join(file)).unwrap(),
        "CANARY_EDIT_BODY\n"
    );
}
