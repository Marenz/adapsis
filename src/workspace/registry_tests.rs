use std::ffi::OsString;

use super::error::Error;
use super::registry::{Registry, registry_path};

const VALID: &str = r#"
[repo.adapsis]
checkout = "/home/owner/Projects/adapsis"
worktree_parent = "/home/owner/Projects/adapsis-workspaces"
base = "origin/master"
read_roots = ["/home/owner/Projects/llm-gateway"]

[repo.gateway]
checkout = "/srv/gateway"
worktree_parent = "/srv/gateway-workspaces"
base = "main"
"#;

fn problems(text: &str) -> String {
    match Registry::from_toml(text) {
        Err(Error::Registry(text)) => text,
        other => panic!("expected a registry error, got {other:?}"),
    }
}

#[test]
fn registry_loads_valid_entries() {
    let registry = Registry::from_toml(VALID).unwrap();
    assert_eq!(registry.names(), vec!["adapsis", "gateway"]);
    let adapsis = registry.get("adapsis").unwrap();
    assert_eq!(adapsis.checkout.to_str(), Some("/home/owner/Projects/adapsis"));
    assert_eq!(adapsis.worktree_parent.to_str(), Some("/home/owner/Projects/adapsis-workspaces"));
    assert_eq!(adapsis.base, "origin/master");
    assert_eq!(adapsis.read_roots.len(), 1);
    assert!(registry.get("gateway").unwrap().read_roots.is_empty());
}

#[test]
fn registry_missing_file_means_no_repositories() {
    let temp = tempfile::tempdir().unwrap();
    let registry = Registry::load_from(&temp.path().join("absent.toml")).unwrap();
    assert!(registry.is_empty());
    let error = registry.get("adapsis").unwrap_err().to_string();
    assert!(error.contains("no repository `adapsis`") && error.contains("none are registered"), "{error}");
}

#[test]
fn registry_empty_file_is_empty() {
    assert!(Registry::from_toml("").unwrap().is_empty());
}

#[test]
fn registry_unknown_repository_lists_the_known_ones() {
    let registry = Registry::from_toml(VALID).unwrap();
    let error = registry.get("nope").unwrap_err().to_string();
    assert!(error.contains("registered: adapsis, gateway"), "{error}");
}

#[test]
fn registry_reports_every_problem_at_once() {
    let text = problems(
        r#"
[repo.Bad_Name]
checkout = "relative/path"
worktree_parent = "~/worktrees"
base = "-evil"
read_roots = ["/", "also/relative"]

[repo.nested]
checkout = "/srv/repo"
worktree_parent = "/srv/repo/worktrees"
base = "main"
"#,
    );
    for expected in [
        "repo `Bad_Name`: name",
        "checkout `relative/path` must be an absolute path",
        "worktree_parent `~/worktrees` must be an absolute path (`~` is not expanded)",
        "base `-evil` must not start with a dash",
        "read_roots[0] is `/`",
        "read_roots entry `also/relative` must be an absolute path",
        "repo `nested`: worktree_parent must not be the checkout or a directory inside it",
    ] {
        assert!(text.contains(expected), "missing `{expected}` in: {text}");
    }
}

#[test]
fn registry_rejects_unknown_keys_and_missing_fields() {
    let unknown = problems("[repo.a]\ncheckout = \"/a\"\nworktree_parent = \"/b\"\nbase = \"main\"\nforce = true\n");
    assert!(unknown.contains("unknown field `force`"), "{unknown}");
    let missing = problems("[repo.a]\ncheckout = \"/a\"\nbase = \"main\"\n");
    assert!(missing.contains("missing field `worktree_parent`"), "{missing}");
    let top = problems("[repos.a]\ncheckout = \"/a\"\n");
    assert!(top.contains("unknown field `repos`") && top.contains("[repo.<name>]"), "{top}");
    let syntax = problems("[repo.a\ncheckout = ");
    assert!(syntax.contains("not valid"), "{syntax}");
}

#[test]
fn registry_rejects_dot_components_and_range_bases() {
    let text = problems("[repo.a]\ncheckout = \"/srv/../etc\"\nworktree_parent = \"/b\"\nbase = \"a..b\"\n");
    assert!(text.contains("must not contain `.` or `..`"), "{text}");
    assert!(text.contains("must name one revision, not a range"), "{text}");
}

#[test]
fn registry_unreadable_or_wrong_kind_of_file_is_an_error() {
    let temp = tempfile::tempdir().unwrap();
    let error = Registry::load_from(temp.path()).unwrap_err().to_string();
    assert!(error.contains("is not a regular file"), "{error}");
    let invalid = temp.path().join("workspaces.toml");
    std::fs::write(&invalid, "[repo.a]\ncheckout = 1\n").unwrap();
    let error = Registry::load_from(&invalid).unwrap_err().to_string();
    assert!(error.contains("workspaces.toml") && error.contains("not valid"), "{error}");
    std::fs::write(&invalid, vec![b'#'; 300 * 1024]).unwrap();
    let error = Registry::load_from(&invalid).unwrap_err().to_string();
    assert!(error.contains("the limit is"), "{error}");
}

#[test]
fn registry_loads_from_a_file() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("workspaces.toml");
    std::fs::write(&path, VALID).unwrap();
    assert_eq!(Registry::load_from(&path).unwrap().names().len(), 2);
}

#[test]
fn registry_path_prefers_the_override_and_never_uses_home_in_tests() {
    let explicit = registry_path(Some(OsString::from("/etc/adapsis/ws.toml")), Some(OsString::from("/home/owner")));
    assert_eq!(explicit.unwrap().to_str(), Some("/etc/adapsis/ws.toml"));
    // Without the override a test build must not point at the real file.
    let default = registry_path(None, Some(OsString::from("/home/owner"))).unwrap();
    assert!(!default.starts_with("/home/owner"), "{default:?}");
    assert!(default.starts_with(std::env::temp_dir()), "{default:?}");
    let empty = registry_path(Some(OsString::new()), None).unwrap();
    assert!(empty.starts_with(std::env::temp_dir()), "{empty:?}");
}

#[test]
fn registry_load_in_tests_reads_no_real_file() {
    // The default test path does not exist, so this is the empty registry
    // unless a developer exported the override on purpose.
    if std::env::var_os(super::registry::REGISTRY_ENV).is_none() {
        assert!(Registry::load().unwrap().is_empty());
    }
}

#[test]
fn registry_environment_is_a_validated_string_table() {
    let configured = format!("{VALID}\n[repo.gateway.env]\nCARGO_TARGET_DIR = '/shared/target'\nLANG = 'de_DE.UTF-8'\n");
    let registry = Registry::from_toml(&configured).unwrap();
    assert!(registry.get("adapsis").unwrap().env.is_empty());
    assert_eq!(registry.get("gateway").unwrap().env["CARGO_TARGET_DIR"], "/shared/target");
    for (suffix, expected) in [
        ("[repo.gateway.env]\n'BAD NAME' = 'x'", "is not valid"),
        ("[repo.gateway.env]\nX = 17", "string"),
        ("[repo.gateway.env]\nX = \"a\\u0000b\"", "NUL byte"),
        ("[repo.gateway]\nunknown = true", "duplicate"),
    ] {
        assert!(problems(&format!("{VALID}\n{suffix}")).contains(expected), "{suffix}");
    }
    let many = (0..33).map(|i| format!("V{i} = 'x'\n")).collect::<String>();
    assert!(problems(&format!("{VALID}\n[repo.gateway.env]\n{many}")).contains("the limit is 32"));
}
