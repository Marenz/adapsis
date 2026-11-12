use super::*;
use axum::{
    body::Body,
    extract::ConnectInfo,
    http::{Request as HttpRequest, StatusCode},
};
use tower::ServiceExt;

#[tokio::test]
async fn only_direct_loopback_can_cancel_an_existing_entry() {
    let repo = crate::workspace::test_support::RepoFixture::new();
    let site = repo.site("cancel");
    let mut config = crate::api::tests::test_config();
    config.workspaces = Some(site.clone());
    let app = crate::api::router_with_llm(config);
    for (peer, forwarded, expected) in [
        ("10.0.0.4:1234", false, StatusCode::FORBIDDEN),
        ("127.0.0.1:1234", true, StatusCode::FORBIDDEN),
        ("[::1]:1234", true, StatusCode::FORBIDDEN),
        ("127.5.6.7:1234", false, StatusCode::ACCEPTED),
        ("[::1]:1234", false, StatusCode::ACCEPTED),
        ("[::ffff:127.0.0.1]:1234", false, StatusCode::ACCEPTED),
    ] {
        let active = Active::register(site.bindings.root(), "CANARY_CONTEXT").unwrap();
        let mut request = HttpRequest::builder()
            .method("POST")
            .uri("/api/workspace/cancel")
            .header("Content-Type", "application/json");
        if forwarded {
            request = request.header("X-Forwarded-For", "127.0.0.1");
        }
        let mut request = request
            .body(Body::from(r#"{"context":"CANARY_CONTEXT"}"#))
            .unwrap();
        request
            .extensions_mut()
            .insert(ConnectInfo(peer.parse::<std::net::SocketAddr>().unwrap()));
        let reply = app.clone().oneshot(request).await.unwrap();
        assert_eq!(reply.status(), expected, "{peer}/{forwarded}");
        assert_eq!(
            active.cancel.is_cancelled(),
            expected == StatusCode::ACCEPTED
        );
        let bytes = axum::body::to_bytes(reply.into_body(), 4096).await.unwrap();
        assert!(!String::from_utf8_lossy(&bytes).contains("CANARY_CONTEXT"));
        drop(active);
    }
    let active = Active::register(site.bindings.root(), "unknown-peer").unwrap();
    let reply = app
        .clone()
        .oneshot(
            HttpRequest::builder()
                .method("POST")
                .uri("/api/workspace/cancel")
                .header("Content-Type", "application/json")
                .body(Body::from(r#"{"context":"unknown-peer"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(reply.status(), StatusCode::FORBIDDEN);
    assert!(
        !active.cancel.is_cancelled(),
        "missing ConnectInfo never authorizes cancellation"
    );
    drop(active);
    let mut request = HttpRequest::builder()
        .method("POST")
        .uri("/api/workspace/cancel")
        .header("Content-Type", "application/json")
        .body(Body::from(r#"{"context":"not-running"}"#))
        .unwrap();
    request.extensions_mut().insert(ConnectInfo(
        "127.0.0.1:1234".parse::<std::net::SocketAddr>().unwrap(),
    ));
    assert_eq!(
        app.oneshot(request).await.unwrap().status(),
        StatusCode::NOT_FOUND
    );
    assert!(
        !request_for_test(site.bindings.root(), "not-running"),
        "cancel never creates a runtime entry"
    );
}

fn request_for_test(root: &Path, context: &str) -> bool {
    super::request(root, context)
}

#[test]
fn dropping_the_async_guard_requests_cancel_but_worker_keeps_entry_until_cleanup() {
    let root = tempfile::tempdir().unwrap();
    let active = Active::register(root.path(), "x").unwrap();
    let drop_guard = OnDrop(active.cancel.clone());
    assert!(Active::register(root.path(), "x").is_err());
    drop(drop_guard);
    assert!(active.cancel.is_cancelled());
    assert!(
        request(root.path(), "x"),
        "stopping is still an active operation"
    );
    drop(active);
    assert!(!request(root.path(), "x"));
    assert!(!request(root.path(), "y"));
}
