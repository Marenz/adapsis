use super::*;
use crate::coroutine::TurnIdentity;
use crate::memory_graph::records::RecordStore;
use crate::memory_graph::records::authority::{RecordCeiling, TrustedCaller};
use crate::permissions::PermissionLevel;

fn ceiling(turn: &TurnIdentity, level: PermissionLevel) -> RecordCeiling {
    let mut permissions = crate::permissions::PermissionConfig::default();
    permissions
        .groups
        .insert("personal".into(), vec!["Fixture".into()]);
    permissions.model.insert(
        "actual".into(),
        crate::permissions::ModelPermissions {
            group_perms: std::collections::HashMap::from([("personal".into(), level)]),
            opencode: false,
        },
    );
    RecordCeiling::capture(
        turn,
        &permissions,
        crate::permissions::AccessLevel::Full,
        Some("actual"),
        None,
    )
    .unwrap()
}

impl RecordStore {
    fn bind_for_test(
        &self,
        turn: &TurnIdentity,
        level: PermissionLevel,
    ) -> Result<crate::memory_graph::records::RecordBinding, StoreError> {
        let ceiling = ceiling(turn, level);
        self.bind(
            turn,
            Some(&TrustedCaller::issue_for_test(ceiling.clone())),
            Some(&ceiling),
        )
    }
}

fn text(generation: i64, rights: &str) -> String {
    format!(
        "version=1\n[[grants]]\nid='personal'\ngeneration={generation}\nnamespace='opaque-ns'\nprincipal='owner'\ncontext='private-owner'\npermission_group='personal'\nrights={rights}\n"
    )
}

fn turn() -> TurnIdentity {
    TurnIdentity {
        principal: "owner".into(),
        context: "private-owner".into(),
        may_write: true,
        guest_tools: None,
    }
}

fn seed() -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("existing.lbug");
    drop(RecordStore::graph_for_test(&path).unwrap());
    let file = dir.path().join("grants.toml");
    std::fs::write(&file, text(1, "['read','write']")).unwrap();
    provision::apply(&path, &GrantConfig::read(&file).unwrap(), false).unwrap();
    (dir, path, file)
}

fn open(path: &std::path::Path, file: &std::path::Path) -> RecordStore {
    RecordStore::configured(
        RecordStore::graph_for_test(path).unwrap(),
        file.to_path_buf(),
    )
    .unwrap()
}

const BATCH: &str = r#"{"version":1,"conditions":[{"key":"a","expected":{"kind":"absent"}}],"puts":[{"key":"a","fields":[{"name":"value","value":"fixture"}]}]}"#;

#[test]
fn record_grants_config_is_strict_bounded_and_unambiguous() {
    assert!(GrantConfig::parse(&text(1, "['read','write']")).is_ok());
    for bad in [
        text(0, "['read']"),
        text(1, "['write']"),
        text(1, "['read','read']"),
        text(1, "['all']"),
        text(1, "['read']").replace("private-owner", "main"),
        text(1, "['read']").replace("owner'", "*'"),
        format!("{}unknown=true", text(1, "['read']")),
        " ".repeat(config::MAX_CONFIG_BYTES + 1),
    ] {
        assert!(GrantConfig::parse(&bad).is_err());
    }
    let duplicate = format!(
        "{}{}",
        text(1, "['read']"),
        text(1, "['read']")
            .replace("version=1\n", "")
            .replace("id='personal'", "id='second'")
    );
    assert!(GrantConfig::parse(&duplicate).is_err());
    let mut excessive = "version=1\n".to_string();
    for index in 0..33 {
        excessive.push_str(
            &text(1, "['read']")
                .replace("version=1\n", "")
                .replace("id='personal'", &format!("id='id-{index}'"))
                .replace(
                    "context='private-owner'",
                    &format!("context='context-{index}'"),
                ),
        );
    }
    assert!(GrantConfig::parse(&excessive).is_err());
}

#[test]
fn record_grants_dry_run_never_creates_or_changes_database() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("missing.lbug");
    let config = GrantConfig::parse(&text(1, "['read']")).unwrap();
    assert!(provision::apply(&path, &config, true).is_err());
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    drop(RecordStore::graph_for_test(&path).unwrap());
    let before = std::fs::read(&path).unwrap();
    let before_files = std::fs::read_dir(dir.path()).unwrap().count();
    provision::apply(&path, &config, true).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), before_files);
    let graph = RecordStore::graph_for_test(&path).unwrap();
    assert!(load(&Connection::new(&graph.database).unwrap()).is_err());
}

#[test]
fn record_grants_rotation_revokes_old_handles_and_preserves_receipts() {
    let (_dir, path, file) = seed();
    let store = open(&path, &file);
    let turn = turn();
    let old = store.bind_for_test(&turn, PermissionLevel::Write).unwrap();
    let receipt = store
        .commit(Some(&old), Some(&turn), "operation", BATCH)
        .unwrap();
    std::fs::write(&file, text(2, "['read','write']")).unwrap();
    let new = store.bind_for_test(&turn, PermissionLevel::Write).unwrap();
    assert_eq!(
        store.receipt(Some(&old.clone()), Some(&turn), "operation"),
        Err(StoreError::Denied)
    );
    assert_eq!(
        store.receipt(Some(&new), Some(&turn), "operation").unwrap(),
        Some(receipt.clone())
    );
    assert_eq!(
        store
            .commit(Some(&new), Some(&turn), "operation", BATCH)
            .unwrap(),
        receipt
    );
}

#[test]
fn record_grants_restart_tombstone_and_generation_floor_reject_rollback() {
    let (_dir, path, file) = seed();
    let turn = turn();
    {
        let store = open(&path, &file);
        let old = store.bind_for_test(&turn, PermissionLevel::Write).unwrap();
        std::fs::write(&file, text(2, "['read','write']")).unwrap();
        store.bind_for_test(&turn, PermissionLevel::Write).unwrap();
        std::fs::write(&file, "version=1\ngrants=[]\n").unwrap();
        assert_eq!(
            store.receipt(Some(&old), Some(&turn), "missing"),
            Err(StoreError::Denied)
        );
    }
    let store = open(&path, &file);
    for generation in [1, 2] {
        std::fs::write(&file, text(generation, "['read','write']")).unwrap();
        assert!(store.bind_for_test(&turn, PermissionLevel::Write).is_err());
    }
    std::fs::write(&file, text(3, "['read','write']")).unwrap();
    assert!(store.bind_for_test(&turn, PermissionLevel::Write).is_ok());
}

#[test]
fn record_grants_changed_identity_or_same_generation_rights_never_partially_activate() {
    let (_dir, path, file) = seed();
    for changed in [
        text(1, "['read']"),
        text(2, "['read']").replace("opaque-ns", "foreign"),
        text(2, "['read']").replace("principal='owner'", "principal='other'"),
        text(2, "['read']").replace("private-owner", "elsewhere"),
    ] {
        let invalid = GrantConfig::parse(&changed).unwrap();
        assert!(provision::apply(&path, &invalid, false).is_err());
    }
    let store = open(&path, &file);
    assert!(store.bind_for_test(&turn(), PermissionLevel::Write).is_ok());
}

#[test]
fn record_grants_missing_bad_file_and_named_group_ceilings_fail_closed() {
    let (_dir, path, file) = seed();
    let store = open(&path, &file);
    let turn = turn();
    assert!(store.bind_for_test(&turn, PermissionLevel::None).is_err());
    assert!(
        store
            .bind_for_test(&turn, PermissionLevel::Execute)
            .is_err()
    );
    let read = store.bind_for_test(&turn, PermissionLevel::Read).unwrap();
    assert_eq!(
        store.receipt(Some(&read), Some(&turn), "x"),
        Err(StoreError::Denied)
    );
    let old = store.bind_for_test(&turn, PermissionLevel::Write).unwrap();
    std::fs::remove_file(&file).unwrap();
    assert_eq!(
        store.receipt(Some(&old), Some(&turn), "x"),
        Err(StoreError::Denied)
    );
    std::fs::write(&file, text(2, "['read','write']")).unwrap();
    let old = store.bind_for_test(&turn, PermissionLevel::Write).unwrap();
    std::fs::write(&file, "invalid TOML").unwrap();
    assert_eq!(
        store.receipt(Some(&old), Some(&turn), "x"),
        Err(StoreError::Denied)
    );
}

#[test]
fn record_grants_metadata_and_permission_ceiling_are_not_attestation() {
    let (_dir, path, file) = seed();
    let store = open(&path, &file);
    let forged = turn(); // Even exact owner/context metadata and may_write are insufficient.
    let current = ceiling(&forged, PermissionLevel::Write);
    assert!(matches!(
        store.bind(&forged, None, Some(&current)),
        Err(StoreError::Denied)
    ));
    let handle = crate::coroutine::CoroutineHandle::new(tokio::sync::mpsc::channel(1).0)
        .with_turn(Some(forged.clone()));
    assert!(
        matches!(handle.execute_await("records_receipt",&[crate::eval::Value::string("id")]).unwrap(),crate::eval::Value::Err(e) if e=="record_access_denied")
    );
    let pinned = ceiling(&forged, PermissionLevel::Read);
    let token = TrustedCaller::issue_for_test(pinned);
    let bound = store.bind(&forged, Some(&token), Some(&current)).unwrap();
    assert_eq!(
        store.receipt(Some(&bound), Some(&forged), "id"),
        Err(StoreError::Denied)
    );
    let mut guest = crate::permissions::PermissionConfig::default();
    guest.guest_profiles.insert("actual".into(), vec![]);
    assert!(
        RecordCeiling::capture(
            &forged,
            &guest,
            crate::permissions::AccessLevel::Full,
            Some("actual"),
            None
        )
        .is_none()
    );
}

#[test]
fn record_grants_later_invalid_entry_cannot_activate_earlier_valid_change() {
    let (_dir, path, _file) = seed();
    let second = text(1, "['read','write']")
        .replace("version=1\n", "")
        .replace("id='personal'", "id='second'")
        .replace("context='private-owner'", "context='second-context'");
    let valid = format!("{}{}", text(1, "['read','write']"), second);
    provision::apply(&path, &GrantConfig::parse(&valid).unwrap(), false).unwrap();
    let invalid = format!(
        "{}{}",
        text(2, "['read']"),
        second
            .replace("generation=1", "generation=2")
            .replace("opaque-ns", "changed-identity")
    );
    assert!(provision::apply(&path, &GrantConfig::parse(&invalid).unwrap(), false).is_err());
    let graph = RecordStore::graph_for_test(&path).unwrap();
    let registry = load(&Connection::new(&graph.database).unwrap()).unwrap();
    let unchanged = registry.matching("owner", "private-owner").unwrap();
    assert_eq!(unchanged.generation, 1);
    assert!(unchanged.writable());
}

#[test]
fn record_grants_pins_actual_context_principal_and_current_executor_intersection() {
    use crate::permissions::{AccessLevel, ModelPermissions, PermissionConfig};
    let turn = turn();
    for (actual, context, principal, expected) in [
        (
            PermissionLevel::None,
            PermissionLevel::Write,
            PermissionLevel::Write,
            PermissionLevel::None,
        ),
        (
            PermissionLevel::Write,
            PermissionLevel::None,
            PermissionLevel::Write,
            PermissionLevel::None,
        ),
        (
            PermissionLevel::Write,
            PermissionLevel::Write,
            PermissionLevel::Read,
            PermissionLevel::Read,
        ),
        (
            PermissionLevel::Write,
            PermissionLevel::Write,
            PermissionLevel::Write,
            PermissionLevel::Write,
        ),
    ] {
        let mut permissions = PermissionConfig::default();
        permissions
            .groups
            .insert("personal".into(), vec!["Fixture".into()]);
        permissions
            .principals
            .insert(turn.principal.clone(), "principal".into());
        for (name, level) in [
            ("actual", actual),
            ("context", context),
            ("principal", principal),
        ] {
            permissions.model.insert(
                name.into(),
                ModelPermissions {
                    group_perms: std::collections::HashMap::from([("personal".into(), level)]),
                    opencode: false,
                },
            );
        }
        let pinned = RecordCeiling::capture(
            &turn,
            &permissions,
            AccessLevel::Full,
            Some("actual"),
            Some("context"),
        )
        .unwrap();
        let token = TrustedCaller::issue_for_test(pinned.clone());
        assert_eq!(token.level(&turn, &pinned, "personal"), expected);
        // Mid-turn permission/profile changes can narrow but never widen the pin.
        for profile in permissions.model.values_mut() {
            profile
                .group_perms
                .insert("personal".into(), PermissionLevel::Write);
        }
        let current = RecordCeiling::capture(
            &turn,
            &permissions,
            AccessLevel::Full,
            Some("actual"),
            Some("context"),
        )
        .unwrap();
        assert_eq!(token.level(&turn, &current, "personal"), expected);
        assert_eq!(
            token.level(&turn, &current, "unknown"),
            PermissionLevel::None
        );
        let restricted = RecordCeiling::capture(
            &turn,
            &permissions,
            AccessLevel::ExecuteOnly,
            Some("actual"),
            None,
        )
        .unwrap();
        assert!(token.level(&turn, &restricted, "personal") <= PermissionLevel::Execute);
        assert!(
            RecordCeiling::capture(&turn, &permissions, AccessLevel::Full, None, None).is_none()
        );
    }
}

#[test]
fn record_grants_revocation_and_commit_linearize_under_one_lock() {
    let (_dir, path, file) = seed();
    let store = open(&path, &file);
    let turn = turn();
    let binding = store.bind_for_test(&turn, PermissionLevel::Write).unwrap();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let mut committing = store.clone();
    committing.authorization_barrier = Some(barrier.clone());
    let worker_turn = turn.clone();
    let worker_binding = binding.clone();
    let commit = std::thread::spawn(move || {
        committing.commit(Some(&worker_binding), Some(&worker_turn), "accepted", BATCH)
    });
    barrier.wait(); // The REAL commit holds the lock after grant revalidation.
    std::fs::write(&file, "version=1\ngrants=[]\n").unwrap();
    let revoking = store.clone();
    let revoke_turn = turn.clone();
    let (started, waiting) = std::sync::mpsc::channel();
    let revoke = std::thread::spawn(move || {
        started.send(()).unwrap();
        revoking
            .bind_for_test(&revoke_turn, PermissionLevel::Write)
            .is_err()
    });
    waiting.recv().unwrap();
    assert!(store.graph.write_lock.try_lock().is_err());
    barrier.wait();
    assert!(commit.join().unwrap().is_ok());
    assert!(revoke.join().unwrap());
    assert_eq!(
        store.commit(Some(&binding), Some(&turn), "after-revoke", BATCH),
        Err(StoreError::Denied)
    );
    std::fs::write(&file, text(2, "['read','write']")).unwrap();
    let new = store.bind_for_test(&turn, PermissionLevel::Write).unwrap();
    assert!(
        store
            .receipt(Some(&new), Some(&turn), "accepted")
            .unwrap()
            .is_some()
    );
}

#[test]
fn record_grants_provision_preserves_memory_and_dry_run_existing_registry_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("memory.lbug");
    {
        let graph = RecordStore::graph_for_test(&path).unwrap();
        graph.initialize_schema().unwrap();
        Connection::new(&graph.database)
            .unwrap()
            .query("CREATE (:Memory {id:'sentinel',content:'preserve'})")
            .unwrap();
    }
    let config = GrantConfig::parse(&text(1, "['read','write']")).unwrap();
    provision::apply(&path, &config, false).unwrap();
    {
        let graph = RecordStore::graph_for_test(&path).unwrap();
        let connection = Connection::new(&graph.database).unwrap();
        let row = connection
            .query("MATCH (m:Memory {id:'sentinel'}) RETURN m.content")
            .unwrap()
            .next()
            .unwrap();
        assert_eq!(string_value(&row[0]).unwrap(), "preserve");
    }
    let before = std::fs::read(&path).unwrap();
    provision::apply(
        &path,
        &GrantConfig::parse(&text(2, "['read']")).unwrap(),
        true,
    )
    .unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), before);
}
