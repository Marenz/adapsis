use super::*;

fn unwrap_string(value: Value) -> String {
    match value {
        Value::String(value) => value.as_ref().clone(),
        other => panic!("expected String, got {other}"),
    }
}

async fn recording_server(
    status: &str,
    response_body: String,
) -> (String, tokio::sync::oneshot::Receiver<String>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (request_tx, request_rx) = tokio::sync::oneshot::channel();
    let status = status.to_string();
    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut buffer = [0_u8; 4096];
        loop {
            let read = stream.read(&mut buffer).await.unwrap();
            if read == 0 {
                break;
            }
            request.extend_from_slice(&buffer[..read]);
            let Some(headers_end) = request.windows(4).position(|window| window == b"\r\n\r\n")
            else {
                continue;
            };
            let headers = String::from_utf8_lossy(&request[..headers_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or(0);
            if request.len() >= headers_end + 4 + content_length {
                break;
            }
        }
        let _ = request_tx.send(String::from_utf8_lossy(&request).into_owned());
        let response = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response_body.len(),
            response_body
        );
        let _ = stream.write_all(response.as_bytes()).await;
    });
    (format!("http://{address}/items/7"), request_rx)
}

async fn execute_http_request(args: Vec<Value>) -> Value {
    let (runtime, mut io_rx) = Runtime::new();
    let handle = CoroutineHandle::new(runtime.io_sender());
    let io_task = tokio::spawn(async move {
        let request = io_rx.recv().await.expect("http_request request");
        runtime.handle_io(request).await;
    });
    let value = tokio::task::spawn_blocking(move || handle.execute_await("http_request", &args))
        .await
        .unwrap()
        .unwrap();
    io_task.await.unwrap();
    value
}

#[tokio::test(flavor = "multi_thread")]
async fn http_request_applies_method_headers_and_body_through_runtime() {
    let (url, request_rx) = recording_server("200 OK", "accepted".into()).await;
    let value = execute_http_request(vec![
        Value::string("put"),
        Value::string(url),
        Value::string("X-Test: yes\nContent-Type: application/json"),
        Value::string("{\"enabled\":true}"),
    ])
    .await;
    match value {
        Value::Ok(value) => assert_eq!(unwrap_string(*value), "accepted"),
        other => panic!("expected Ok response, got {other:?}"),
    }
    let request = request_rx.await.unwrap();
    let lowercase = request.to_ascii_lowercase();
    assert!(request.starts_with("PUT /items/7 HTTP/1.1"), "{request}");
    assert!(lowercase.contains("x-test: yes"), "{request}");
    assert!(request.ends_with("{\"enabled\":true}"), "{request}");
}

#[tokio::test(flavor = "multi_thread")]
async fn http_request_policy_errors_are_actionable_and_catchable() {
    for (method, headers, expected) in [
        ("TRACE", "", "unsupported method `TRACE`"),
        ("GET", "not a header", "malformed header on line 1"),
    ] {
        let value = execute_http_request(vec![
            Value::string(method),
            Value::string("http://127.0.0.1:1/not-contacted"),
            Value::string(headers),
            Value::string(""),
        ])
        .await;
        match value {
            Value::Err(error) => assert!(error.contains(expected), "{error}"),
            other => panic!("expected catchable policy error, got {other:?}"),
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn http_request_transport_failure_is_actionable_and_catchable() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let target = format!("http://{address}/gone");
    let value = execute_http_request(vec![
        Value::string("GET"),
        Value::string(&target),
        Value::string(""),
        Value::string(""),
    ])
    .await;
    match value {
        Value::Err(error) => {
            assert!(
                error.contains(&format!("http_request GET {target} failed")),
                "{error}"
            );
            assert!(error.len() < 1024, "transport error was not bounded");
        }
        other => panic!("expected catchable transport error, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn http_request_success_body_over_limit_is_rejected() {
    let (url, _request_rx) = recording_server("200 OK", "x".repeat(2 * 1024 * 1024 + 1)).await;
    let value = execute_http_request(vec![
        Value::string("GET"),
        Value::string(url),
        Value::string(""),
        Value::string("ignored"),
    ])
    .await;
    match value {
        Value::Err(error) => {
            assert!(
                error.contains("response exceeded the 2097152-byte limit"),
                "{error}"
            );
            assert!(
                error.len() < 1024,
                "oversized-response error was not bounded"
            );
        }
        other => panic!("expected catchable oversized-response error, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn http_request_non_success_error_body_is_bounded() {
    let oversized_error = format!("remote rejected request: {}", "x".repeat(70 * 1024));
    let (url, _request_rx) = recording_server("413 Payload Too Large", oversized_error).await;
    let value = execute_http_request(vec![
        Value::string("DELETE"),
        Value::string(url),
        Value::string(""),
        Value::string(""),
    ])
    .await;
    match value {
        Value::Err(error) => {
            assert!(error.contains("HTTP 413 Payload Too Large"), "{error}");
            assert!(error.contains("remote rejected request"), "{error}");
            assert!(error.contains("response body truncated"), "{error}");
            assert!(error.len() < 66 * 1024, "error body was not bounded");
        }
        other => panic!("expected catchable HTTP error, got {other:?}"),
    }
}
