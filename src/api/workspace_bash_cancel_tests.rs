//! Real HTTP cancellation and client disconnect while a foreground command is running.
use super::*;
use crate::workspace::{self, test_support::RepoFixture};

async fn until(what: &str, done: impl Fn() -> bool) {
    let end = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while !done() {
        assert!(std::time::Instant::now() < end, "timed out: {what}");
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
}

async fn cancel_and_collect(address: std::net::SocketAddr, mut raw: tokio::net::TcpStream) {
    let reply: Value = reqwest::Client::new()
        .post(format!("http://{address}/api/workspace/cancel"))
        .json(&json!({"context":"main"}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        reply["status"],
        "cancellation requested; stopped state is not yet confirmed"
    );
    use tokio::io::AsyncReadExt;
    let mut buffer = Vec::new();
    tokio::time::timeout(
        std::time::Duration::from_secs(15),
        raw.read_to_end(&mut buffer),
    )
    .await
    .unwrap()
    .unwrap();
    let text = String::from_utf8_lossy(&buffer);
    assert!(
        text.contains("\"completed\":false") && text.contains("cancellation requested"),
        "{text}"
    );
}

async fn assert_training(file: &std::path::Path, disconnect: bool) {
    let training = tokio::fs::read_to_string(file).await.unwrap();
    assert!(
        !training.contains("CANARY"),
        "cancelled peer args/ids must not leak: {training}"
    );
    if !disconnect {
        let row: Value = serde_json::from_str(training.trim()).unwrap();
        let flags: Vec<bool> = row["feedback"]
            .as_array()
            .unwrap()
            .iter()
            .map(|text| {
                serde_json::from_str::<Value>(text.as_str().unwrap()).unwrap()["success"]
                    .as_bool()
                    .unwrap()
            })
            .collect();
        assert_eq!(
            flags,
            [true, false, false, false],
            "success before cancellation is preserved; skipped peers fail"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn foreground_bash_stops_on_local_http_cancel_and_client_disconnect() {
    use tokio::io::AsyncWriteExt;
    for disconnect in [false, true] {
        let repo = RepoFixture::new();
        let site = repo.site("http-bash");
        let workspace =
            workspace::open(site.registry.get("http-bash").unwrap(), "running", None).unwrap();
        site.bindings.claim("main", &workspace, &|_| None).unwrap();
        let command = "printf '%s' $$ > running.pid; printf 'command started\\n'; sleep 60";
        let mut fixture = Fixture::new(
            vec![
                response(
                    "",
                    vec![
                        call("before", "read", json!({"path":"README.md"})),
                        call("bash", "bash", json!({"command":command})),
                        call(
                            "CANARY_PEER_ID",
                            "inspect",
                            json!({"kind":"symbols","target":"CANARY_PEER_TARGET"}),
                        ),
                        call(
                            "later",
                            "write",
                            json!({"path":"must-not-exist","content":"CANARY_PEER_VALUE"}),
                        ),
                    ],
                ),
                response("never requested", vec![call("finish", "finish", json!({}))]),
            ],
            "",
        )
        .await;
        fixture.config.workspaces = Some(site.clone());
        let training_file = tempfile::NamedTempFile::new().unwrap();
        fixture.config.training_log = Some(Arc::new(tokio::sync::Mutex::new(
            tokio::fs::File::create(training_file.path()).await.unwrap(),
        )));
        let app = crate::api::router_with_llm(fixture.config.clone())
            .into_make_service_with_connect_info::<std::net::SocketAddr>();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let body = json!({"message":"run a foreground command"}).to_string();
        let mut raw = tokio::net::TcpStream::connect(address).await.unwrap();
        raw.write_all(format!("POST /api/ask HTTP/1.1\r\nHost: {address}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()).as_bytes()).await.unwrap();
        let pid_file = workspace.path.join("running.pid");
        until("command startup", || {
            std::fs::read_to_string(&pid_file).is_ok_and(|text| !text.is_empty())
        })
        .await;
        let pid: i32 = std::fs::read_to_string(&pid_file).unwrap().parse().unwrap();
        assert!(
            crate::api::turn_lock::work_now(site.bindings.root(), "main").is_none(),
            "worker owns the work lock while bash runs"
        );
        if disconnect {
            drop(raw);
        } else {
            cancel_and_collect(address, raw).await;
        }
        until("command process gone", || {
            // SAFETY: signal 0 observes process existence only.
            unsafe { libc::kill(pid, 0) != 0 }
        })
        .await;
        until("work cleanup complete", || {
            crate::api::turn_lock::work_now(site.bindings.root(), "main").is_some()
        })
        .await;
        assert!(
            !workspace.path.join("must-not-exist").exists(),
            "later calls never execute after cancel/drop"
        );
        assert_eq!(
            fixture.requests.lock().unwrap().len(),
            1,
            "no replay or follow-up generation"
        );
        assert_eq!(
            site.bindings.holders("http-bash", "running").unwrap(),
            ["main"]
        );
        assert_training(training_file.path(), disconnect).await;
        server.abort();
    }
}
