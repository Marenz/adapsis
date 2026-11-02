//! #116: HTTP builtin errors never contain the request URL (bot tokens, userinfo, query
//! secrets). Loopback only: a closed port for transport failures and a tiny local server
//! that promises a body and hangs up for the body-read failures. No external network and
//! no credentials; the "token" is a sentinel inside a loopback URL.
use super::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const PATH: &str = "/bot123456:SECRET-TOKEN_abc/getUpdates?k=SECRET&offset=5";
const FORBIDDEN: [&str; 6] = ["bot123456", "SECRET", "getUpdates", "offset", "k=", "/bot"];

async fn closed_port_url() -> (String, String) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap().to_string();
    drop(listener);
    (
        format!("http://{address}{PATH}"),
        format!("http://{address}"),
    )
}

/// Answers every request with a 200 that promises 100 bytes and delivers 5.
async fn truncating_server() -> (String, String) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap().to_string();
    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            tokio::spawn(async move {
                let mut request = Vec::new();
                let mut buffer = [0_u8; 4096];
                loop {
                    let Ok(read) = stream.read(&mut buffer).await else {
                        return;
                    };
                    if read == 0 {
                        return;
                    }
                    request.extend_from_slice(&buffer[..read]);
                    if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                        let headers = String::from_utf8_lossy(&request[..end]).to_ascii_lowercase();
                        let length = headers
                            .lines()
                            .find_map(|l| {
                                l.strip_prefix("content-length:")?
                                    .trim()
                                    .parse::<usize>()
                                    .ok()
                            })
                            .unwrap_or(0);
                        if request.len() >= end + 4 + length {
                            break;
                        }
                    }
                }
                let _ = stream
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\nConnection: close\r\n\r\nshort",
                    )
                    .await;
                let _ = stream.shutdown().await;
            });
        }
    });
    (
        format!("http://{address}{PATH}"),
        format!("http://{address}"),
    )
}

fn assert_clean(label: &str, rendered: &[String], host: &str) {
    for text in rendered {
        for forbidden in FORBIDDEN {
            assert!(
                !text.contains(forbidden),
                "{label}: `{forbidden}` leaked: {text}"
            );
        }
        assert!(
            text.contains(host),
            "{label}: host `{host}` missing: {text}"
        );
    }
}

fn both_forms<E: std::fmt::Display + std::fmt::Debug>(
    error: &anyhow::Error,
    _marker: Option<E>,
) -> Vec<String> {
    vec![
        format!("{error}"),
        format!("{error:#}"),
        format!("{error:?}"),
    ]
}

async fn request<T>(make: impl FnOnce(oneshot::Sender<Result<T>>) -> IoRequest) -> Result<T> {
    let (runtime, _io_rx) = Runtime::new();
    let (tx, rx) = oneshot::channel();
    runtime.handle_io(make(tx)).await;
    rx.await.unwrap()
}

macro_rules! check {
    ($label:expr, $url:expr, $host:expr, $result:expr) => {{
        let error = $result
            .await
            .err()
            .unwrap_or_else(|| panic!("{}: expected an error", $label));
        assert_clean($label, &both_forms::<String>(&error, None), $host);
        error
    }};
}

#[tokio::test(flavor = "multi_thread")]
async fn transport_failures_never_contain_the_url_for_every_http_builtin() {
    let (url, host) = closed_port_url().await;
    let e = check!(
        "get",
        url,
        &host,
        request(|reply| IoRequest::HttpGet {
            url: url.clone(),
            reply
        })
    );
    // Actionable: the kind, the class and the host-only target, in that order.
    assert_eq!(
        format!("{e}"),
        format!("error sending request: connection failed ({host})")
    );
    check!(
        "post",
        url,
        &host,
        request(|reply| IoRequest::HttpPost {
            url: url.clone(),
            body: "{}".into(),
            content_type: "application/json".into(),
            reply,
        })
    );
    check!(
        "post_binary",
        url,
        &host,
        request(|reply| IoRequest::HttpPostBinary {
            url: url.clone(),
            body: "x".into(),
            content_type: "text/plain".into(),
            reply,
        })
    );
    check!(
        "get_binary",
        url,
        &host,
        request(|reply| IoRequest::HttpPostBinary {
            url: url.clone(),
            body: String::new(),
            content_type: "application/octet-stream".into(),
            reply,
        })
    );
    check!(
        "upload_bytes",
        url,
        &host,
        request(|reply| IoRequest::HttpUploadBytes {
            url: url.clone(),
            bytes: vec![1, 2, 3],
            file_name: "f.bin".into(),
            mime_type: "application/octet-stream".into(),
            file_field: "file".into(),
            extra_fields: vec![("k".into(), "v".into())],
            reply,
        })
    );
    let file = std::env::temp_dir().join(format!("adapsis-116-{}.txt", std::process::id()));
    std::fs::write(&file, "data").unwrap();
    check!(
        "upload",
        url,
        &host,
        request(|reply| IoRequest::HttpUpload {
            url: url.clone(),
            file_path: file.to_string_lossy().into_owned(),
            file_field: "file".into(),
            extra_fields: vec![],
            reply,
        })
    );
    let _ = std::fs::remove_file(&file);
}

#[tokio::test(flavor = "multi_thread")]
async fn body_read_failures_never_contain_the_url() {
    let (url, host) = truncating_server().await;
    check!(
        "get body",
        url,
        &host,
        request(|reply| IoRequest::HttpGet {
            url: url.clone(),
            reply
        })
    );
    check!(
        "post body",
        url,
        &host,
        request(|reply| IoRequest::HttpPost {
            url: url.clone(),
            body: "{}".into(),
            content_type: "application/json".into(),
            reply,
        })
    );
    check!(
        "binary body",
        url,
        &host,
        request(|reply| IoRequest::HttpPostBinary {
            url: url.clone(),
            body: "x".into(),
            content_type: "text/plain".into(),
            reply,
        })
    );
    check!(
        "upload_bytes body",
        url,
        &host,
        request(|reply| IoRequest::HttpUploadBytes {
            url: url.clone(),
            bytes: vec![1],
            file_name: "f".into(),
            mime_type: "text/plain".into(),
            file_field: "file".into(),
            extra_fields: vec![],
            reply,
        })
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn builtin_error_values_are_catchable_and_clean() {
    // Through the evaluator-facing builtins: `http_get`/`http_post` return an Err VALUE.
    let (url, host) = closed_port_url().await;
    for (name, args) in [
        ("http_get", vec![Value::string(&url)]),
        (
            "http_post",
            vec![
                Value::string(&url),
                Value::string("{}"),
                Value::string("application/json"),
            ],
        ),
    ] {
        let (runtime, mut io_rx) = Runtime::new();
        let handle = CoroutineHandle::new(runtime.io_sender());
        let io_task = tokio::spawn(async move {
            let request = io_rx.recv().await.expect("request");
            runtime.handle_io(request).await;
        });
        let value = tokio::task::spawn_blocking(move || handle.execute_await(name, &args))
            .await
            .unwrap()
            .unwrap();
        io_task.await.unwrap();
        match value {
            Value::Err(text) => assert_clean(name, &[text], &host),
            other => panic!("{name}: expected Err value, got {other:?}"),
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn http_request_errors_name_the_host_only() {
    async fn run(url: &str) -> String {
        let (runtime, mut io_rx) = Runtime::new();
        let handle = CoroutineHandle::new(runtime.io_sender());
        let io_task = tokio::spawn(async move {
            let request = io_rx.recv().await.expect("request");
            runtime.handle_io(request).await;
        });
        let args = vec![
            Value::string("GET"),
            Value::string(url),
            Value::string(""),
            Value::string(""),
        ];
        let value =
            tokio::task::spawn_blocking(move || handle.execute_await("http_request", &args))
                .await
                .unwrap()
                .unwrap();
        io_task.await.unwrap();
        match value {
            Value::Err(text) => text,
            other => panic!("expected Err value, got {other:?}"),
        }
    }
    let (url, host) = closed_port_url().await;
    let text = run(&url).await;
    assert_clean("request transport", &[text.clone()], &host);
    assert!(text.contains("connection failed"), "{text}");
    // Malformed, scheme-less and non-http inputs are never echoed (a scheme-less
    // `bot123456:SECRET` even parses with the scheme `bot123456`).
    for raw in [
        "not a url with SECRET",
        "bot123456:SECRET-TOKEN_abc/getUpdates?k=SECRET",
        "ftp://user:SECRET@host/bot123456:SECRET",
        "file:///etc/SECRET",
    ] {
        let text = run(raw).await;
        for forbidden in FORBIDDEN {
            assert!(
                !text.contains(forbidden),
                "`{forbidden}` leaked for {raw}: {text}"
            );
        }
        assert!(text.starts_with("http_request: "), "{text}");
    }
    // A response that hangs up mid-body fails in the read loop, also host-only.
    let (cut_url, cut_host) = truncating_server().await;
    let text = run(&cut_url).await;
    assert_clean("request body", &[text.clone()], &cut_host);
    assert!(text.contains("could not read the response"), "{text}");
    assert!(text.ends_with("response decode failed"), "{text}");
    // A non-success status names the host only.
    let (ok_url, ok_host) = {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap().to_string();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buffer = [0_u8; 4096];
            let _ = stream.read(&mut buffer).await;
            let _ = stream
                .write_all(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 4\r\nConnection: close\r\n\r\ngone",
                )
                .await;
        });
        (
            format!("http://{address}{PATH}"),
            format!("http://{address}"),
        )
    };
    let text = run(&ok_url).await;
    assert_clean("request status", &[text.clone()], &ok_host);
    assert!(
        text.contains("returned HTTP 404") && text.contains("gone"),
        "{text}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn successful_requests_are_unchanged() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            tokio::spawn(async move {
                let mut buffer = [0_u8; 4096];
                let _ = stream.read(&mut buffer).await;
                let _ = stream
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok",
                    )
                    .await;
            });
        }
    });
    let url = format!("http://{address}{PATH}");
    assert_eq!(
        request(|reply| IoRequest::HttpGet {
            url: url.clone(),
            reply
        })
        .await
        .unwrap(),
        "ok"
    );
    assert_eq!(
        request(|reply| IoRequest::HttpPost {
            url: url.clone(),
            body: "{}".into(),
            content_type: "application/json".into(),
            reply
        })
        .await
        .unwrap(),
        "ok"
    );
    assert_eq!(
        request(|reply| IoRequest::HttpUploadBytes {
            url,
            bytes: vec![1],
            file_name: "f".into(),
            mime_type: "text/plain".into(),
            file_field: "file".into(),
            extra_fields: vec![],
            reply,
        })
        .await
        .unwrap(),
        "ok"
    );
}
