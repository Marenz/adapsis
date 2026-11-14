//! Foreground scope naming and startup cleanup. Only units with THIS state
//! root's digest AND this process kind's prefix are ever considered. Like the
//! binding locks this relies on one active daemon per state root.
use super::{Error, Result, systemd};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

pub(crate) fn root_digest(root: &Path) -> String {
    use std::os::unix::ffi::OsStrExt;
    let root = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    Sha256::digest(root.as_os_str().as_bytes())[..16]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub fn prefix(root: &Path) -> String {
    format!(
        "{}-{}",
        super::run::default_unit_prefix(),
        root_digest(root)
    )
}

/// Memoized per root, not per turn/Site: a new turn must never sweep another
/// command that is running in THIS daemon. The call runs on a blocking thread.
pub fn ready(root: &Path) -> Result<String> {
    type Slot = Arc<OnceLock<Result<String>>>;
    static ROOTS: OnceLock<Mutex<HashMap<PathBuf, Slot>>> = OnceLock::new();
    let slot = {
        let mut roots = ROOTS
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        roots.entry(root.to_path_buf()).or_default().clone()
    };
    slot.get_or_init(|| {
        let outcome = sweep(root);
        if outcome.is_err() {
            crate::error_journal::record(crate::error_journal::Category::Workspace,"foreground","startup foreground cleanup could not be confirmed; bash is unavailable until resolved and restarted");
        }
        outcome
    }).clone()
}

fn refused() -> Error {
    Error::Systemd { action: "foreground startup cleanup".into(), detail: "previous foreground scope cleanup is uncertain; no command was started. Restore the user manager and inspect this state root's scopes before restarting".into() }
}

fn owned(unit: &str, prefix: &str) -> bool {
    let Some(tail) = unit
        .strip_prefix(&format!("{prefix}-run-"))
        .and_then(|s| s.strip_suffix(".scope"))
    else {
        return false;
    };
    tail.split_once('-').is_some_and(|(pid, sequence)| {
        !pid.is_empty()
            && !sequence.is_empty()
            && pid.bytes().all(|b| b.is_ascii_digit())
            && sequence.bytes().all(|b| b.is_ascii_digit())
    })
}

fn active(properties: &HashMap<String, String>) -> Result<bool> {
    match (
        properties.get("LoadState").map(String::as_str),
        properties.get("ActiveState").map(String::as_str),
    ) {
        (Some("not-found"), Some("inactive")) => Ok(false),
        (Some("loaded"), Some("inactive" | "failed")) => Ok(false),
        (Some("loaded"), Some("active" | "activating" | "deactivating" | "reloading")) => Ok(true),
        _ => Err(refused()),
    }
}

/// An inactive label is not proof if the attributed scope cgroup still has
/// members. Never inspect a guessed or broader cgroup subtree.
fn stopped(properties: &HashMap<String, String>, unit: &str) -> Result<()> {
    if active(properties)? {
        return Err(refused());
    }
    if properties.get("LoadState").map(String::as_str) == Some("not-found") {
        return Ok(());
    }
    let group = properties.get("ControlGroup").ok_or_else(refused)?;
    if group.is_empty() {
        return Ok(());
    }
    let relative = Path::new(group).strip_prefix("/").map_err(|_| refused())?;
    if relative.file_name().and_then(|name| name.to_str()) != Some(unit)
        || !relative
            .components()
            .all(|part| matches!(part, std::path::Component::Normal(_)))
    {
        return Err(refused());
    }
    match std::fs::read_to_string(
        Path::new("/sys/fs/cgroup")
            .join(relative)
            .join("cgroup.events"),
    ) {
        Ok(events) if events.lines().any(|line| line == "populated 0") => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        _ => Err(refused()),
    }
}

fn sweep(root: &Path) -> Result<String> {
    if !systemd::available() {
        // Foreground process-group fallback remains supported, explicitly.
        return Ok("no user manager was reachable for the startup scope sweep; earlier scopes could not be inspected, and this command uses the library's reported containment".into());
    }
    let prefix = prefix(root);
    let listing = systemd::systemctl(
        &[
            "list-units",
            "--all",
            "--no-legend",
            "--plain",
            "--full",
            &format!("{prefix}-run-*.scope"),
        ],
        "list foreground scopes",
    )
    .map_err(|_| refused())?;
    let units: Vec<_> = listing
        .stdout_text()
        .lines()
        .filter_map(|line| line.split_whitespace().next().map(str::to_string))
        .collect();
    // Validate the ENTIRE listing before signalling anything. Unexpected data
    // never broadens a sweep or licenses a claim that cleanup succeeded.
    if units.iter().any(|unit| !owned(unit, &prefix)) {
        return Err(refused());
    }
    for unit in units {
        let before = systemd::show(&unit, &["LoadState", "ActiveState", "ControlGroup"])
            .map_err(|_| refused())?;
        if !active(&before)? {
            stopped(&before, &unit)?;
            continue;
        }
        systemd::kill(&unit, "SIGKILL").map_err(|_| refused())?;
        let until = std::time::Instant::now() + super::limits::RUN_DRAIN_GRACE;
        loop {
            let after = systemd::show(&unit, &["LoadState", "ActiveState", "ControlGroup"])
                .map_err(|_| refused())?;
            if !active(&after)? {
                stopped(&after, &unit)?;
                break;
            }
            if std::time::Instant::now() >= until {
                return Err(refused());
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }
    Ok(String::new())
}

/// Once at daemon bootstrap, before any foreground tool can start. Per-root
/// memoization is shared with calls made later from per-turn Site snapshots.
pub async fn startup() {
    let result = tokio::task::spawn_blocking(|| ready(&super::job_store::state_root())).await;
    if result.is_err() {
        crate::error_journal::record(
            crate::error_journal::Category::Workspace,
            "foreground",
            "startup foreground cleanup could not be confirmed; bash is unavailable until the state is resolved and the daemon restarted",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sweep_names_are_exactly_root_scoped() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let (a, b) = (prefix(a.path()), prefix(b.path()));
        assert_ne!(a, b);
        assert!(owned(&format!("{a}-run-123-9.scope"), &a));
        for unit in [
            format!("{b}-run-123-9.scope"),
            "adapsis-run-123-9.scope".into(),
            format!("{a}-run-bad-9.scope"),
            format!("{a}-run-123-9.service"),
            format!("{a}-run-123-9.scope.extra"),
        ] {
            assert!(!owned(&unit, &a), "{unit}");
        }
        for (load, state, expected) in [
            ("loaded", "active", Some(true)),
            ("loaded", "inactive", Some(false)),
            ("not-found", "inactive", Some(false)),
            ("not-found", "active", None),
            ("error", "inactive", None),
            ("loaded", "unknown", None),
            ("", "", None),
        ] {
            let data = HashMap::from([
                ("LoadState".into(), load.into()),
                ("ActiveState".into(), state.into()),
            ]);
            assert_eq!(active(&data).ok(), expected, "{load}/{state}");
        }
    }

    #[test]
    fn startup_sweep_stops_only_the_matching_state_roots_real_scope() {
        if !systemd::available() {
            return;
        }
        let fixture = super::super::test_support::Fixture::plain();
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let start = |root: &Path, file: &str| {
            let workspace = fixture.workspace.clone();
            let (prefix, command) = (prefix(root), format!("printf ready > {file}; sleep 60"));
            let cancel = super::super::Cancel::new();
            let held = cancel.clone();
            let task = std::thread::spawn(move || {
                let policy = crate::coroutine::ShellPolicy::Unrestricted;
                let mut options = super::super::run::RunOptions::new(&policy);
                options.unit_prefix = &prefix;
                options.containment = super::super::Containment::Scope;
                options.cancel = Some(&held);
                super::super::run_with(
                    &workspace,
                    &super::super::RunRequest::new(&command),
                    &options,
                )
            });
            (task, cancel)
        };
        let (ta, _ca) = start(a.path(), "a-ready");
        let (tb, cb) = start(b.path(), "b-ready");
        let until = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while !fixture.root.join("a-ready").exists() || !fixture.root.join("b-ready").exists() {
            assert!(std::time::Instant::now() < until);
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(ready(a.path()).is_ok());
        let result = ta.join().unwrap().unwrap();
        assert_ne!(result.exit, super::super::Exit::Code(0));
        assert!(
            !tb.is_finished(),
            "another root's command must be untouched"
        );
        // Once initialization finished, another Site/turn of this daemon must
        // not sweep its new, legitimate command on that same root.
        let (tc, cc) = start(a.path(), "c-ready");
        let until = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while !fixture.root.join("c-ready").exists() {
            assert!(std::time::Instant::now() < until);
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(ready(a.path()).is_ok());
        assert!(
            !tc.is_finished(),
            "startup sweep is memoized, not repeated per turn"
        );
        cc.cancel();
        let _ = tc.join().unwrap().unwrap();
        cb.cancel();
        let _ = tb.join().unwrap().unwrap();
    }

    #[test]
    fn stopped_state_needs_a_known_empty_or_gone_scope() {
        let mut data = HashMap::from([
            ("LoadState".into(), "loaded".into()),
            ("ActiveState".into(), "inactive".into()),
        ]);
        assert!(
            stopped(&data, "example.scope").is_err(),
            "missing cgroup identity is uncertain"
        );
        data.insert("ControlGroup".into(), "/wrong-unit.scope".into());
        assert!(
            stopped(&data, "example.scope").is_err(),
            "another unit's cgroup proves nothing"
        );
        data.insert("ControlGroup".into(), String::new());
        assert!(stopped(&data, "example.scope").is_ok());
        data.insert("ActiveState".into(), "active".into());
        assert!(stopped(&data, "example.scope").is_err());
    }
}
