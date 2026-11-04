use std::process::Command;

fn snapshot(path: &std::path::Path) -> std::collections::BTreeMap<std::path::PathBuf, Vec<u8>> {
    std::fs::read_dir(path)
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (
                entry.file_name().into(),
                std::fs::read(entry.path()).unwrap(),
            )
        })
        .collect()
}

const CONFIG: &str = "version=1\n[[grants]]\nid='fixture'\ngeneration=1\nnamespace='fixture-ns'\nprincipal='fixture-owner'\ncontext='fixture-private'\npermission_group='fixture'\nrights=['read','write']\n";

#[test]
fn record_grants_cli_dry_run_missing_database_creates_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let grants = directory.path().join("grants.toml");
    let database = directory.path().join("missing.lbug");
    std::fs::write(&grants, CONFIG).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_adapsis"))
        .arg("records-provision")
        .arg("--database")
        .arg(&database)
        .arg("--grants")
        .arg(&grants)
        .arg("--dry-run")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!database.exists());
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    assert_eq!(std::fs::read_to_string(grants).unwrap(), CONFIG);
}

#[test]
fn record_grants_cli_dry_run_existing_database_is_byte_and_schema_read_only() {
    let directory = tempfile::tempdir().unwrap();
    let grants = directory.path().join("grants.toml");
    let database = directory.path().join("existing.lbug");
    std::fs::write(&grants, CONFIG).unwrap();
    {
        let db = lbug::Database::new(
            &database,
            lbug::SystemConfig::default().buffer_pool_size(64 * 1024 * 1024),
        )
        .unwrap();
        lbug::Connection::new(&db)
            .unwrap()
            .query("CREATE NODE TABLE Sentinel(id STRING, PRIMARY KEY(id))")
            .unwrap();
    }
    let before = std::fs::read(&database).unwrap();
    let directory_before = snapshot(directory.path());
    let count = std::fs::read_dir(directory.path()).unwrap().count();
    let output = Command::new(env!("CARGO_BIN_EXE_adapsis"))
        .arg("records-provision")
        .arg("--database")
        .arg(&database)
        .arg("--grants")
        .arg(&grants)
        .arg("--dry-run")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["dry_run"], true);
    assert_eq!(report["migration_required"], true);
    assert_eq!(std::fs::read(&database).unwrap(), before);
    assert_eq!(snapshot(directory.path()), directory_before);
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), count);
    let db = lbug::Database::new(&database, lbug::SystemConfig::default().read_only(true)).unwrap();
    assert!(
        lbug::Connection::new(&db)
            .unwrap()
            .query("MATCH (r:RecordGrantRegistry) RETURN r.id")
            .is_err()
    );
}
