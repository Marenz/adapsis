use super::*;
use serde_json::{Value as Json, json};
use std::path::Path;
use std::sync::Barrier;

fn disk_graph(path: &Path) -> MemoryGraph {
    RecordStore::graph_for_test(path).unwrap()
}

fn turn() -> TurnIdentity {
    TurnIdentity {
        context: "context".into(),
        principal: "owner".into(),
        may_write: true,
        guest_tools: None,
    }
}

fn fixture() -> (tempfile::TempDir, RecordStore, RecordBinding, TurnIdentity) {
    let directory = tempfile::tempdir().unwrap();
    let store =
        RecordStore::initialize_for_test(disk_graph(&directory.path().join("records.lbug")))
            .unwrap();
    let turn = turn();
    let grant = store.grant_for_test("namespace", &turn, "grant-v1", true);
    (directory, store, grant, turn)
}

fn create(keys: &[&str], value: &str) -> Json {
    json!({"version":1,
        "conditions":keys.iter().map(|key| json!({"key":key,"expected":{"kind":"absent"}})).collect::<Vec<_>>(),
        "puts":keys.iter().map(|key| json!({"key":key,"fields":[{"name":"value","value":value}]})).collect::<Vec<_>>()})
}

fn read(
    store: &RecordStore,
    grant: &RecordBinding,
    turn: &TurnIdentity,
    keys: &[&str],
) -> Vec<Option<Record>> {
    store
        .read(
            Some(grant),
            Some(turn),
            &json!({"version":1,"keys":keys}).to_string(),
        )
        .unwrap()
}

#[test]
fn record_envelope_limits_and_canonical_hash() {
    let base = create(&["a", "b"], "value");
    let mut reordered = base.clone();
    reordered["conditions"].as_array_mut().unwrap().reverse();
    reordered["puts"].as_array_mut().unwrap().reverse();
    assert_eq!(
        Batch::parse(&base.to_string()).unwrap().hash(),
        Batch::parse(&reordered.to_string()).unwrap().hash()
    );
    let mut invalid = Vec::new();
    let mut value = base.clone();
    value["version"] = json!(2);
    invalid.push(value);
    let mut value = base.clone();
    value["namespace"] = json!("foreign");
    invalid.push(value);
    let mut value = base.clone();
    value["conditions"][1] = value["conditions"][0].clone();
    invalid.push(value);
    let mut value = base.clone();
    value["puts"][1] = value["puts"][0].clone();
    invalid.push(value);
    let mut value = base.clone();
    value["puts"][0]["key"] = json!("");
    invalid.push(value);
    let mut value = base.clone();
    value["puts"][0]["key"] = json!("unguarded");
    invalid.push(value);
    let mut value = base.clone();
    value["puts"][0]["fields"] = json!([{"name":"x","value":"a"},{"name":"x","value":"b"}]);
    invalid.push(value);
    let mut value = base.clone();
    value["puts"][0]["fields"][0]["value"] = json!(17);
    invalid.push(value);
    let mut value = base.clone();
    value["puts"][0]["fields"][0]["value"] = json!("x".repeat(envelope::MAX_VALUE_BYTES + 1));
    invalid.push(value);
    let mut value = base.clone();
    value["puts"][0]["fields"][0]["name"] = json!("x".repeat(envelope::MAX_FIELD_BYTES + 1));
    invalid.push(value);
    let mut value = base.clone();
    value["conditions"][0]["key"] = json!("x".repeat(envelope::MAX_KEY_BYTES + 1));
    invalid.push(value);
    let mut value = base.clone();
    value["puts"][0]["fields"] = json!(
        (0..33)
            .map(|i| json!({"name":i.to_string(),"value":""}))
            .collect::<Vec<_>>()
    );
    invalid.push(value);
    let mut value = base.clone();
    value["puts"][0]["fields"] = json!(
        (0..5)
            .map(|i| json!({"name":i.to_string(),"value":"x".repeat(4096)}))
            .collect::<Vec<_>>()
    );
    invalid.push(value);
    let mut value = base.clone();
    value["conditions"][0]["expected"] = json!({"kind":"revision","revision":"01"});
    invalid.push(value);
    let mut value = base.clone();
    value["conditions"][0]["expected"] = json!({"kind":"absent","revision":"1"});
    invalid.push(value);
    for (index, value) in invalid.into_iter().enumerate() {
        assert!(
            matches!(
                Batch::parse(&value.to_string()),
                Err(StoreError::InvalidRequest)
            ),
            "invalid-envelope case {index}"
        );
    }
    for raw in [
        "{",
        "{\"version\":1,\"version\":1,\"conditions\":[],\"puts\":[]}",
    ] {
        assert!(matches!(Batch::parse(raw), Err(StoreError::InvalidRequest)));
    }
    assert!(Batch::parse(&" ".repeat(envelope::MAX_BATCH_BYTES + 1)).is_err());
    assert!(envelope::read_keys(&json!({"version":1,"keys":vec!["a";33]}).to_string()).is_err());
    assert!(envelope::read_keys(&json!({"version":1,"keys":["a","a"]}).to_string()).is_err());
}

#[test]
fn records_invalid_bounded_requests_do_not_take_database_lock() {
    let (_dir, store, grant, turn) = fixture();
    let _guard = store.graph.write_lock.lock().unwrap();
    let keys: Vec<_> = (0..17).map(|i| format!("key-{i}")).collect();
    let refs: Vec<_> = keys.iter().map(String::as_str).collect();
    let too_many_puts = create(&refs, "value");
    let mut too_many_conditions = create(&["key"], "value");
    too_many_conditions["conditions"] = json!(
        (0..33)
            .map(|i| json!({
                "key":format!("key-{i}"),"expected":{"kind":"absent"}
            }))
            .collect::<Vec<_>>()
    );
    let start = Instant::now();
    for request in [
        too_many_puts.to_string(),
        too_many_conditions.to_string(),
        " ".repeat(envelope::MAX_BATCH_BYTES + 1),
    ] {
        assert_eq!(
            store.commit(Some(&grant), Some(&turn), "invalid", &request),
            Err(StoreError::InvalidRequest)
        );
    }
    assert!(start.elapsed() < Duration::from_secs(1));
}

#[test]
fn records_explicit_initialization_and_denied_authority() {
    let directory = tempfile::tempdir().unwrap();
    let graph = disk_graph(&directory.path().join("records.lbug"));
    assert!(
        Connection::new(&graph.database)
            .unwrap()
            .query("MATCH (r:ScopedRecord) RETURN r.id")
            .is_err()
    );
    let store = RecordStore::initialize_for_test(graph).unwrap();
    let turn = turn();
    let grant = store.grant_for_test("namespace", &turn, "grant", true);
    let batch = create(&["a"], "private").to_string();
    for (binding, caller) in [(None, Some(&turn)), (Some(&grant), None)] {
        assert_eq!(
            store.commit(binding, caller, "request", &batch),
            Err(StoreError::Denied)
        );
        assert_eq!(
            store.receipt(binding, caller, "request"),
            Err(StoreError::Denied)
        );
    }
    for altered in [
        TurnIdentity {
            principal: "other".into(),
            ..turn.clone()
        },
        TurnIdentity {
            context: "other".into(),
            ..turn.clone()
        },
        TurnIdentity {
            guest_tools: Some(vec![]),
            ..turn.clone()
        },
        TurnIdentity {
            may_write: false,
            ..turn.clone()
        },
    ] {
        assert_eq!(
            store.commit(Some(&grant), Some(&altered), "request", &batch),
            Err(StoreError::Denied)
        );
        assert_eq!(
            store.receipt(Some(&grant), Some(&altered), "request"),
            Err(StoreError::Denied)
        );
    }
    let read_only = store.grant_for_test("namespace", &turn, "readonly", false);
    assert_eq!(
        store.commit(Some(&read_only), Some(&turn), "r", &batch),
        Err(StoreError::Denied)
    );
    assert_eq!(
        store.receipt(Some(&read_only), Some(&turn), "r"),
        Err(StoreError::Denied)
    );
    assert_eq!(read(&store, &read_only, &turn, &["a"]), vec![None]);
    store
        .commit(Some(&grant), Some(&turn), "request", &batch)
        .unwrap();
    let foreign = store.grant_for_test("foreign", &turn, "foreign", true);
    assert_eq!(read(&store, &foreign, &turn, &["a"]), vec![None]);
    assert_eq!(
        store
            .receipt(Some(&foreign), Some(&turn), "request")
            .unwrap(),
        None
    );
    assert_eq!(
        store.commit(
            Some(&grant),
            Some(&turn),
            "invalid",
            "{\"namespace\":\"foreign\"}"
        ),
        Err(StoreError::InvalidRequest)
    );
    let (_other_dir, other_store, _, _) = fixture();
    assert_eq!(
        other_store.receipt(Some(&grant), Some(&turn), "request"),
        Err(StoreError::Denied)
    );
}

#[test]
fn records_idempotency_is_canonical_principal_and_grant_bound() {
    let (_dir, store, grant, turn) = fixture();
    let mut batch = create(&["a", "b"], "first");
    let first = store
        .commit(Some(&grant), Some(&turn), "request", &batch.to_string())
        .unwrap();
    batch["puts"].as_array_mut().unwrap().reverse();
    batch["conditions"].as_array_mut().unwrap().reverse();
    assert_eq!(
        store
            .commit(Some(&grant), Some(&turn), "request", &batch.to_string())
            .unwrap(),
        first
    );
    assert_eq!(
        read(&store, &grant, &turn, &["a"])[0]
            .as_ref()
            .unwrap()
            .revision,
        "1"
    );
    assert_eq!(
        store.commit(
            Some(&grant),
            Some(&turn),
            "request",
            &create(&["a", "b"], "changed").to_string()
        ),
        Err(StoreError::IdempotencyMismatch)
    );
    let other = TurnIdentity {
        principal: "other".into(),
        ..turn.clone()
    };
    let other_grant = store.grant_for_test("namespace", &other, "other", true);
    assert_eq!(
        store
            .receipt(Some(&other_grant), Some(&other), "request")
            .unwrap(),
        None
    );
    assert_eq!(
        store.commit(
            Some(&other_grant),
            Some(&other),
            "request",
            &batch.to_string()
        ),
        Err(StoreError::Conflict)
    );
    for (caller, grant_id) in [
        (turn.clone(), "new-grant"),
        (
            TurnIdentity {
                context: "elsewhere".into(),
                ..turn.clone()
            },
            "grant-v1",
        ),
    ] {
        let new_grant = store.grant_for_test("namespace", &caller, grant_id, true);
        assert_eq!(
            store.receipt(Some(&new_grant), Some(&caller), "request"),
            Err(StoreError::Denied)
        );
        assert_eq!(
            store.commit(
                Some(&new_grant),
                Some(&caller),
                "request",
                &batch.to_string()
            ),
            Err(StoreError::Denied)
        );
    }
}

#[test]
fn records_parallel_disjoint_writes_and_atomic_conflict() {
    let (_dir, store, grant, turn) = fixture();
    for contested in [false, true] {
        let barrier = Barrier::new(2);
        let results = std::thread::scope(|scope| {
            let threads: Vec<_> = (0..2)
                .map(|index| {
                    let (store, grant, turn, barrier) = (&store, &grant, &turn, &barrier);
                    scope.spawn(move || {
                        let own = format!("row-{contested}-{index}");
                        let resource = if contested {
                            "contested".to_string()
                        } else {
                            format!("resource-{index}")
                        };
                        assert_eq!(read(store, grant, turn, &[&resource]), vec![None]);
                        barrier.wait(); // Real concurrent attempts following the same absent read.
                        let request = format!("request-{contested}-{index}");
                        let result = store.commit(
                            Some(grant),
                            Some(turn),
                            &request,
                            &create(&[&own, &resource], "value").to_string(),
                        );
                        (own, request, result)
                    })
                })
                .collect();
            threads
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(
            results
                .iter()
                .filter(|(_, _, result)| result.is_ok())
                .count(),
            if contested { 1 } else { 2 }
        );
        for (own, request, result) in results {
            if contested && result.is_err() {
                assert_eq!(result, Err(StoreError::Conflict));
                assert_eq!(read(&store, &grant, &turn, &[&own]), vec![None]);
                assert_eq!(
                    store.receipt(Some(&grant), Some(&turn), &request).unwrap(),
                    None
                );
            }
        }
    }
}

#[test]
fn records_stale_check_only_conditions_and_precommit_rollback() {
    let (_dir, store, grant, turn) = fixture();
    store
        .commit(
            Some(&grant),
            Some(&turn),
            "initial",
            &create(&["a"], "first").to_string(),
        )
        .unwrap();
    let mut update = create(&["a"], "second");
    update["conditions"][0]["expected"] = json!({"kind":"revision","revision":"1"});
    store
        .commit(Some(&grant), Some(&turn), "update", &update.to_string())
        .unwrap();
    assert_eq!(
        store.commit(Some(&grant), Some(&turn), "stale", &update.to_string()),
        Err(StoreError::Conflict)
    );
    let mut guarded = create(&["new"], "value");
    guarded["conditions"]
        .as_array_mut()
        .unwrap()
        .push(json!({"key":"a","expected":{"kind":"revision","revision":"1"}}));
    assert_eq!(
        store.commit(Some(&grant), Some(&turn), "guarded", &guarded.to_string()),
        Err(StoreError::Conflict)
    );
    assert_eq!(read(&store, &grant, &turn, &["new"]), vec![None]);
    let failing = create(&["x", "y"], "rolled back").to_string();
    assert_eq!(
        store.commit_inner(
            Some(&grant),
            Some(&turn),
            "rollback",
            &failing,
            Fault::BeforeCommit
        ),
        Err(StoreError::Unavailable)
    );
    assert_eq!(read(&store, &grant, &turn, &["x", "y"]), vec![None, None]);
    assert_eq!(
        store
            .receipt(Some(&grant), Some(&turn), "rollback")
            .unwrap(),
        None
    );
}

#[test]
fn records_disk_reopen_preserves_rows_and_receipts() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("records.lbug");
    let turn = turn();
    let batch = create(&["a", "b"], "durable").to_string();
    let receipt;
    {
        let store = RecordStore::initialize_for_test(disk_graph(&path)).unwrap();
        let grant = store.grant_for_test("namespace", &turn, "grant", true);
        receipt = store
            .commit(Some(&grant), Some(&turn), "request", &batch)
            .unwrap();
        // This fault models lost acknowledgment, not a process crash.
        assert_eq!(
            store.commit_inner(
                Some(&grant),
                Some(&turn),
                "ambiguous",
                &create(&["c"], "committed").to_string(),
                Fault::AfterCommit
            ),
            Err(StoreError::OutcomeUnknown)
        );
    }
    let store = RecordStore::initialize_for_test(disk_graph(&path)).unwrap();
    let grant = store.grant_for_test("namespace", &turn, "grant", true);
    assert_eq!(
        store.receipt(Some(&grant), Some(&turn), "request").unwrap(),
        Some(receipt.clone())
    );
    assert_eq!(
        store
            .commit(Some(&grant), Some(&turn), "request", &batch)
            .unwrap(),
        receipt
    );
    assert_eq!(
        read(&store, &grant, &turn, &["a", "b", "c"])
            .iter()
            .filter(|r| r.as_ref().is_some_and(|r| r.revision == "1"))
            .count(),
        3
    );
    assert!(
        store
            .receipt(Some(&grant), Some(&turn), "ambiguous")
            .unwrap()
            .is_some()
    );
    store
        .commit(
            Some(&grant),
            Some(&turn),
            "ambiguous",
            &create(&["c"], "committed").to_string(),
        )
        .unwrap();
    assert_eq!(
        read(&store, &grant, &turn, &["c"])[0]
            .as_ref()
            .unwrap()
            .revision,
        "1"
    );
}

#[test]
fn records_bounded_batch_releases_lock_promptly() {
    let (_dir, store, grant, turn) = fixture();
    let keys: Vec<_> = (0..envelope::MAX_PUTS)
        .map(|i| format!("row-{i}"))
        .collect();
    let refs: Vec<_> = keys.iter().map(String::as_str).collect();
    let batch = create(&refs, &"x".repeat(envelope::MAX_VALUE_BYTES)).to_string();
    let start = Instant::now();
    store
        .commit(Some(&grant), Some(&turn), "bounded", &batch)
        .unwrap();
    assert!(
        start.elapsed() < Duration::from_secs(15),
        "practical bounded-work check, not an OS I/O guarantee"
    );
    assert!(store.graph.write_lock.try_lock().is_ok());
    assert_eq!(read(&store, &grant, &turn, &refs).len(), envelope::MAX_PUTS);
}
