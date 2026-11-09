use super::*;
use crate::coroutine::Runtime;

#[tokio::test]
async fn records_bridge_backend_rejects_direct_requests_and_redacts_debug() {
    let directory = tempfile::tempdir().unwrap();
    let store = RecordStore::disk_for_test(&directory.path().join("records.lbug")).unwrap();
    let turn = TurnIdentity {
        principal: "owner".into(),
        context: "context".into(),
        may_write: true,
        guest_tools: None,
    };
    let grant = store.grant_for_test("namespace", &turn, "grant", true);
    let access = Access {
        store,
        binding: grant,
    };
    let (runtime, _) = Runtime::new();
    let different = TurnIdentity {
        principal: "other".into(),
        ..turn.clone()
    };
    for (access, turn) in [
        (None, Some(turn.clone())),
        (Some(std::sync::Arc::new(access.clone())), Some(different)),
    ] {
        let request = Request {
            access,
            turn,
            operation: Operation::Receipt("PRIVATE-REQUEST-ID".into()),
        };
        assert_eq!(format!("{request:?}"), "RecordRequest(<redacted>)");
        let (reply, receiver) = oneshot::channel();
        runtime
            .handle_io(IoRequest::Records { request, reply })
            .await;
        assert_eq!(receiver.await.unwrap().unwrap(), Err(StoreError::Denied));
    }
    assert_eq!(format!("{access:?}"), "RecordAccess(<bound>)");
}

#[test]
fn records_bridge_default_handle_and_mocks_do_not_grant_access() {
    let mock = crate::session::IoMock {
        operation: "records_read".into(),
        patterns: vec![],
        response: "forged success".into(),
    };
    let handle = CoroutineHandle::new_mock(vec![mock]);
    assert!(
        matches!(handle.execute_await("records_read",&[Value::string("{}")]).unwrap(),Value::Err(e) if e=="record_access_denied")
    );
}
