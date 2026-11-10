use std::path::Path;

use super::error::{Error, Rule};
use super::resolve::{Access, Area, Kind, Resolved, resolve};
use super::secure_open::{Strategy, open_file_beneath, stat_beneath};
use super::test_support::{Fixture, symlink, write};

/// The tree every resolver test (and the property test) runs against.
///
/// ```text
/// top/ws/            workspace root
///   a.txt  dir/b.txt  dir/sub/  .git/config
///   link_in -> dir/b.txt          link_dir_in -> dir
///   abs_in -> <root>/a.txt        dir/link_root -> ..
///   link_out -> ../outside/secret.txt
///   link_dir_out -> ../outside    link_up -> ..
///   link_extra -> ../extra/e.txt  link_extra_dir -> ../extra
///   loop1 <-> loop2               dangling -> nowhere
/// top/outside/secret.txt
/// top/extra/e.txt  extra/escape -> ../outside/secret.txt  extra/back -> ../ws/a.txt
/// top/alias -> ws
/// ```
pub fn tree() -> Fixture {
    let fixture = Fixture::plain();
    fixture.write("a.txt", "alpha\n");
    fixture.write("dir/b.txt", "beta\n");
    fixture.write(".git/config", "[core]\n");
    std::fs::create_dir_all(fixture.root.join("dir/sub")).unwrap();
    write(&fixture.outside, "secret.txt", "SECRET\n");
    write(&fixture.extra, "e.txt", "extra\n");
    let root = &fixture.root;
    symlink("dir/b.txt", root.join("link_in"));
    symlink("dir", root.join("link_dir_in"));
    symlink(root.join("a.txt"), root.join("abs_in"));
    symlink("..", root.join("dir/link_root"));
    symlink("../outside/secret.txt", root.join("link_out"));
    symlink("../outside", root.join("link_dir_out"));
    symlink("..", root.join("link_up"));
    symlink("../extra/e.txt", root.join("link_extra"));
    symlink("../extra", root.join("link_extra_dir"));
    symlink("loop2", root.join("loop1"));
    symlink("loop1", root.join("loop2"));
    symlink("nowhere", root.join("dangling"));
    symlink("../outside/secret.txt", fixture.extra.join("escape"));
    symlink("../ws/a.txt", fixture.extra.join("back"));
    symlink("ws", fixture.top.join("alias"));
    fixture
}

fn read(fixture: &Fixture, path: &str) -> Result<Resolved, Error> {
    resolve(&fixture.workspace, path, Access::Read)
}

fn write_mode(fixture: &Fixture, path: &str) -> Result<Resolved, Error> {
    resolve(&fixture.workspace, path, Access::Write)
}

fn rule(result: Result<Resolved, Error>) -> Rule {
    match result {
        Err(Error::Refused { rule, .. }) => rule,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

/// A refusal must not describe the filesystem outside the workspace.
fn assert_discreet(fixture: &Fixture, supplied: &str, error: &Error) {
    let text = error.to_string();
    for secret in ["SECRET", "secret.txt", "nowhere", "/outside", "outside/"] {
        assert!(supplied.contains(secret) || !text.contains(secret), "`{supplied}` leaked `{secret}`: {text}");
    }
    let top = fixture.top.to_str().unwrap();
    assert!(supplied.contains(top) || !text.contains(top), "`{supplied}` leaked the real location: {text}");
}

#[test]
fn resolve_accepts_paths_inside_the_root() {
    let fixture = tree();
    for (path, display, kind) in [
        ("a.txt", "a.txt", Kind::File),
        ("dir/b.txt", "dir/b.txt", Kind::File),
        ("./dir/../a.txt", "a.txt", Kind::File),
        ("dir//b.txt", "dir/b.txt", Kind::File),
        ("dir/sub/../../dir/./b.txt", "dir/b.txt", Kind::File),
        (".", ".", Kind::Directory),
        ("dir/", "dir", Kind::Directory),
        ("dir/sub/..", "dir", Kind::Directory),
    ] {
        let resolved = read(&fixture, path).unwrap_or_else(|error| panic!("{path}: {error}"));
        assert_eq!((resolved.display.as_str(), resolved.kind, resolved.area), (display, kind, Area::Root), "{path}");
        assert_eq!(resolved.real, fixture.root.join(display).components().collect::<std::path::PathBuf>(), "{path}");
        assert_eq!(resolved.base.join(&resolved.relative), resolved.real, "{path}");
    }
}

#[test]
fn resolve_refuses_parent_escapes() {
    let fixture = tree();
    for path in ["..", "../outside/secret.txt", "dir/../../outside/secret.txt", "../ws/a.txt", "missing/../../x", "dir/sub/../../../ws"] {
        for access in [Access::Read, Access::Write] {
            let error = resolve(&fixture.workspace, path, access).unwrap_err();
            assert_eq!(error.rule(), Some(Rule::ParentEscape), "{path}: {error}");
            assert_discreet(&fixture, path, &error);
            assert!(error.to_string().contains("`..` leaves the workspace"), "{error}");
        }
    }
    // `..` applied to a file is an error, not a way to its directory.
    assert!(matches!(read(&fixture, "a.txt/.."), Err(Error::WrongKind { .. })));
    assert!(matches!(read(&fixture, "a.txt/x"), Err(Error::WrongKind { .. })));
}

#[test]
fn resolve_absolute_paths_must_end_inside() {
    let fixture = tree();
    let root = fixture.root.to_str().unwrap();
    let top = fixture.top.to_str().unwrap();
    let inside = read(&fixture, &format!("{root}/dir/b.txt")).unwrap();
    assert_eq!(inside.display, "dir/b.txt");
    assert_eq!(read(&fixture, root).unwrap().display, ".");
    assert_eq!(write_mode(&fixture, &format!("{root}/new.txt")).unwrap().kind, Kind::Missing);
    // Through an alias of the root that only resolution can see.
    let aliased = read(&fixture, &format!("{top}/alias/a.txt")).unwrap();
    assert_eq!((aliased.display.as_str(), aliased.real.clone()), ("a.txt", fixture.root.join("a.txt")));
    assert_eq!(write_mode(&fixture, &format!("{top}/alias/dir/new.txt")).unwrap().display, "dir/new.txt");

    for path in [
        "/etc/passwd".to_string(),
        "/".to_string(),
        format!("{top}/outside/secret.txt"),
        format!("{top}/outside/missing"),
        format!("{top}/outside/../ws/a.txt"),
        format!("{top}/nonexistent/../ws/a.txt"),
        format!("{root}-sibling/a.txt"),
        "/proc/self/environ".to_string(),
    ] {
        for access in [Access::Read, Access::Write] {
            let error = resolve(&fixture.workspace, &path, access).unwrap_err();
            assert_eq!(error.rule(), Some(Rule::OutsideWorkspace), "{path}: {error}");
        }
    }
    // An absolute spelling of the root gets no extra rights.
    assert_eq!(rule(read(&fixture, &format!("{root}/../outside/secret.txt"))), Rule::ParentEscape);
    assert_eq!(rule(read(&fixture, &format!("{root}/link_out"))), Rule::SymlinkOutside);
}

#[test]
fn resolve_outside_answers_do_not_depend_on_what_exists_there() {
    let fixture = tree();
    let top = fixture.top.to_str().unwrap();
    let existing = read(&fixture, &format!("{top}/outside/secret.txt")).unwrap_err();
    let missing = read(&fixture, &format!("{top}/outside/no-such-file")).unwrap_err();
    let through_file = read(&fixture, &format!("{top}/outside/secret.txt/x")).unwrap_err();
    for error in [&existing, &missing, &through_file] {
        assert_eq!(error.rule(), Some(Rule::OutsideWorkspace), "{error}");
    }
    let strip = |error: &Error| error.to_string().split("refused:").nth(1).map(str::to_string);
    assert_eq!(strip(&existing), strip(&missing));
    assert_eq!(strip(&existing), strip(&through_file));
}

#[test]
fn resolve_symlink_to_a_file_outside_is_refused_for_read_and_write() {
    let fixture = tree();
    let error = read(&fixture, "link_out").unwrap_err();
    assert_eq!(error.rule(), Some(Rule::SymlinkOutside));
    assert_discreet(&fixture, "link_out", &error);
    assert_eq!(error.to_string(), "path `link_out` refused: a symlink on it leads outside the workspace and its readable roots");
    let error = write_mode(&fixture, "link_out").unwrap_err();
    assert_eq!(error.rule(), Some(Rule::WriteThroughSymlink));
    assert_discreet(&fixture, "link_out", &error);
}

#[test]
fn resolve_symlink_to_a_directory_outside_is_refused_with_anything_under_it() {
    let fixture = tree();
    for path in [
        "link_dir_out",
        "link_dir_out/secret.txt",
        "link_dir_out/new.txt",
        "link_dir_out/../ws/a.txt",
        "link_up/outside/secret.txt",
        "link_up/ws/a.txt",
        "dir/link_root/link_dir_out/secret.txt",
    ] {
        let error = read(&fixture, path).unwrap_err();
        assert_eq!(error.rule(), Some(Rule::SymlinkOutside), "{path}: {error}");
        assert_discreet(&fixture, path, &error);
        let error = write_mode(&fixture, path).unwrap_err();
        assert!(matches!(error.rule(), Some(Rule::SymlinkOutside | Rule::WriteThroughSymlink)), "{path}: {error}");
        assert_discreet(&fixture, path, &error);
    }
}

#[test]
fn resolve_follows_symlinks_that_stay_inside() {
    let fixture = tree();
    for (path, display) in [
        ("link_in", "dir/b.txt"),
        ("link_dir_in/b.txt", "dir/b.txt"),
        ("abs_in", "a.txt"),
        ("dir/link_root/a.txt", "a.txt"),
        ("dir/link_root/link_dir_in/link_root/dir/b.txt", "dir/b.txt"),
    ] {
        let resolved = read(&fixture, path).unwrap_or_else(|error| panic!("{path}: {error}"));
        assert_eq!((resolved.display.as_str(), resolved.kind), (display, Kind::File), "{path}");
    }
    // A symlinked parent directory is fine for writing; a final symlink never is.
    let through_parent = write_mode(&fixture, "link_dir_in/b.txt").unwrap();
    assert_eq!((through_parent.display.as_str(), through_parent.kind), ("dir/b.txt", Kind::File));
    let created = write_mode(&fixture, "link_dir_in/new.txt").unwrap();
    assert_eq!((created.real.clone(), created.kind), (fixture.root.join("dir/new.txt"), Kind::Missing));
    for path in ["link_in", "abs_in", "link_dir_in", "dir/link_root", "dangling", "loop1"] {
        assert_eq!(rule(write_mode(&fixture, path)), Rule::WriteThroughSymlink, "{path}");
    }
}

#[test]
fn resolve_write_mode_allows_files_that_do_not_exist_yet() {
    let fixture = tree();
    let fresh = write_mode(&fixture, "dir/new.rs").unwrap();
    assert_eq!((fresh.kind, fresh.display.as_str()), (Kind::Missing, "dir/new.rs"));
    let deep = write_mode(&fixture, "dir/new/deeper/file.rs").unwrap();
    assert_eq!((deep.kind, deep.real.clone()), (Kind::Missing, fixture.root.join("dir/new/deeper/file.rs")));
    assert_eq!(write_mode(&fixture, "a.txt").unwrap().kind, Kind::File);
    assert_eq!(write_mode(&fixture, "dir").unwrap().kind, Kind::Directory);
    for path in ["dir/new.rs", "dir/new/deeper/file.rs", "dangling", "dangling/x"] {
        assert!(matches!(read(&fixture, path), Err(Error::NotFound { .. })), "{path}");
    }
    assert!(matches!(write_mode(&fixture, "a.txt/under-a-file"), Err(Error::WrongKind { .. })));
}

#[test]
fn resolve_missing_components_do_not_hide_what_follows_them() {
    let fixture = tree();
    // `..` cancels the missing component; the rest is on real ground again.
    assert_eq!(write_mode(&fixture, "nope/../a.txt").unwrap().kind, Kind::File);
    assert_eq!(read(&fixture, "nope/../dir/b.txt").unwrap().display, "dir/b.txt");
    assert_eq!(rule(write_mode(&fixture, "nope/../link_out")), Rule::WriteThroughSymlink);
    assert_eq!(rule(read(&fixture, "nope/../link_dir_out/secret.txt")), Rule::SymlinkOutside);
    assert_eq!(rule(write_mode(&fixture, "nope/deeper/../../link_dir_out/new.txt")), Rule::SymlinkOutside);
    assert_eq!(rule(write_mode(&fixture, "nope/../.git/config")), Rule::WriteGitDir);
    assert_eq!(rule(read(&fixture, "nope/../../outside/secret.txt")), Rule::ParentEscape);
}

#[test]
fn resolve_refuses_writes_under_git() {
    let fixture = tree();
    for path in [".git", ".git/config", ".git/hooks/pre-commit", ".GIT/config", "dir/.git/config", "dir/sub/.git", "link_dir_in/.git/x"] {
        let error = write_mode(&fixture, path).unwrap_err();
        assert_eq!(error.rule(), Some(Rule::WriteGitDir), "{path}: {error}");
        assert!(error.to_string().contains("writes under `.git` are not allowed"), "{error}");
    }
    assert_eq!(read(&fixture, ".git/config").unwrap().kind, Kind::File, "reading git's files is allowed");
    assert_eq!(write_mode(&fixture, ".gitignore").unwrap().kind, Kind::Missing);
    assert_eq!(write_mode(&fixture, "dir/.github/workflows/ci.yml").unwrap().kind, Kind::Missing);
}

#[test]
fn resolve_extra_roots_are_readable_but_never_writable() {
    let fixture = tree();
    let extra = fixture.extra.to_str().unwrap();
    for path in ["link_extra".to_string(), "link_extra_dir/e.txt".to_string(), format!("{extra}/e.txt")] {
        let resolved = read(&fixture, &path).unwrap_or_else(|error| panic!("{path}: {error}"));
        assert_eq!((resolved.area, resolved.kind), (Area::ReadRoot, Kind::File), "{path}");
        assert_eq!(resolved.real, fixture.extra.join("e.txt"));
        assert_eq!((resolved.base.clone(), resolved.relative.as_path()), (fixture.extra.clone(), Path::new("e.txt")));
        assert_eq!(resolved.display, format!("{extra}/e.txt"));
    }
    assert_eq!(rule(write_mode(&fixture, "link_extra")), Rule::WriteThroughSymlink);
    for path in ["link_extra_dir/e.txt".to_string(), "link_extra_dir/new.txt".to_string(), format!("{extra}/e.txt"), format!("{extra}/new/file")] {
        let error = write_mode(&fixture, &path).unwrap_err();
        assert_eq!(error.rule(), Some(Rule::WriteReadOnlyRoot), "{path}: {error}");
        assert!(error.to_string().contains("only the workspace itself is writable"), "{error}");
    }
    // A link in the extra root is held to the same rule as one in the workspace.
    assert_eq!(rule(read(&fixture, "link_extra_dir/escape")), Rule::SymlinkOutside);
    assert_eq!(read(&fixture, "link_extra_dir/back").unwrap().area, Area::Root);
    assert_eq!(rule(read(&fixture, "link_extra_dir/..")), Rule::ParentEscape);
    assert_eq!(rule(read(&fixture, &format!("{extra}/../outside/secret.txt"))), Rule::OutsideWorkspace);
    assert!(matches!(read(&fixture, &format!("{extra}/absent")), Err(Error::NotFound { .. })));
}

#[test]
fn resolve_handles_unicode_and_long_components() {
    let fixture = tree();
    let name = "dir/ünïcödé-文件 with space.txt";
    assert_eq!(write_mode(&fixture, name).unwrap().kind, Kind::Missing);
    fixture.write(name, "x\n");
    let resolved = read(&fixture, name).unwrap();
    assert_eq!((resolved.kind, resolved.display.as_str()), (Kind::File, name));

    let longest = "x".repeat(255);
    assert_eq!(write_mode(&fixture, &longest).unwrap().kind, Kind::Missing);
    let multibyte = "é".repeat(128);
    assert_eq!(rule(write_mode(&fixture, &multibyte)), Rule::ComponentTooLong, "256 bytes in 128 characters");
    assert_eq!(rule(read(&fixture, &format!("dir/{}", "x".repeat(256)))), Rule::ComponentTooLong);
    let deep = "d/".repeat(2000);
    assert_eq!(write_mode(&fixture, &deep).unwrap().kind, Kind::Missing);
    assert_eq!(rule(read(&fixture, &"d/".repeat(2049))), Rule::PathTooLong);
    assert_eq!(rule(read(&fixture, "")), Rule::EmptyPath);
    assert_eq!(rule(read(&fixture, "a\0.txt")), Rule::NulByte);
    // The echo of an over-long path is bounded.
    let error = read(&fixture, &"y".repeat(5000)).unwrap_err().to_string();
    assert!(error.chars().count() < 400, "{}", error.len());
}

#[test]
fn resolve_symlink_loops_and_dangling_links_end_cleanly() {
    let fixture = tree();
    assert_eq!(rule(read(&fixture, "loop1")), Rule::SymlinkLoop);
    assert_eq!(rule(read(&fixture, "loop2/x")), Rule::SymlinkLoop);
    assert!(matches!(read(&fixture, "dangling"), Err(Error::NotFound { .. })));
    let under = write_mode(&fixture, "dangling/x").unwrap();
    assert_eq!((under.kind, under.real.clone()), (Kind::Missing, fixture.root.join("nowhere/x")));
}

#[test]
fn resolve_messages_echo_only_what_the_caller_supplied() {
    let fixture = tree();
    for path in ["link_out", "link_dir_out/secret.txt", "link_up/x", "..", "../x", "link_extra_dir/escape", "loop1", "dangling"] {
        for access in [Access::Read, Access::Write] {
            if let Err(error) = resolve(&fixture.workspace, path, access) {
                assert_discreet(&fixture, path, &error);
                assert!(error.to_string().contains(path), "the caller's own path is named: {error}");
            }
        }
    }
    let control = read(&fixture, "a\u{1b}[31m\nb").unwrap_err().to_string();
    assert!(!control.contains('\u{1b}') && !control.contains('\n'), "{control:?}");
}

// --- races: what happens when the tree changes after resolution ------------

const STRATEGIES: [Strategy; 3] = [Strategy::Auto, Strategy::Openat2, Strategy::Walk];

#[test]
fn race_ancestor_replaced_by_a_symlink_is_not_followed() {
    for strategy in STRATEGIES {
        let fixture = tree();
        write(&fixture.outside, "b.txt", "SECRET\n");
        let resolved = read(&fixture, "dir/b.txt").unwrap();
        // Between resolving and opening, `dir` becomes a link to the outside.
        std::fs::rename(fixture.root.join("dir"), fixture.root.join("dir-moved")).unwrap();
        symlink("../outside", fixture.root.join("dir"));
        let error = resolved.open_file_with(strategy).unwrap_err();
        assert_eq!(error.rule(), Some(Rule::ChangedToSymlink), "{strategy:?}: {error}");
        assert!(error.to_string().contains("became a symlink"), "{error}");
        let raw = open_file_beneath(&resolved.base, &resolved.relative, strategy).unwrap_err();
        assert_eq!(raw.raw_os_error(), Some(libc::ELOOP), "{strategy:?}");
        assert_eq!(stat_beneath(&resolved.base, &resolved.relative, strategy).unwrap_err().raw_os_error(), Some(libc::ELOOP));
        // Resolving again applies the rules to the new tree.
        assert_eq!(rule(read(&fixture, "dir/b.txt")), Rule::SymlinkOutside);
    }
}

#[test]
fn race_final_component_replaced_by_a_symlink_is_not_followed() {
    for strategy in STRATEGIES {
        let fixture = tree();
        let resolved = read(&fixture, "a.txt").unwrap();
        std::fs::remove_file(fixture.root.join("a.txt")).unwrap();
        symlink("../outside/secret.txt", fixture.root.join("a.txt"));
        let error = resolved.open_file_with(strategy).unwrap_err();
        assert_eq!(error.rule(), Some(Rule::ChangedToSymlink), "{strategy:?}: {error}");
        // `stat_beneath` describes the link itself, it does not follow it.
        let metadata = stat_beneath(&resolved.base, &resolved.relative, strategy).unwrap();
        assert!(metadata.file_type().is_symlink(), "{strategy:?}");
    }
}

#[test]
fn race_both_open_strategies_agree_on_ordinary_files() {
    use std::io::Read;
    let fixture = tree();
    for strategy in STRATEGIES {
        let resolved = read(&fixture, "dir/b.txt").unwrap();
        let (mut file, metadata) = resolved.open_file_with(strategy).unwrap();
        let mut text = String::new();
        file.read_to_string(&mut text).unwrap();
        assert_eq!((text.as_str(), metadata.len()), ("beta\n", 5), "{strategy:?}");
        assert!(stat_beneath(&fixture.root, Path::new("dir/sub"), strategy).unwrap().is_dir());
        let missing = open_file_beneath(&fixture.root, Path::new("dir/absent"), strategy).unwrap_err();
        assert_eq!(missing.kind(), std::io::ErrorKind::NotFound, "{strategy:?}");
        for invalid in ["", "../a.txt", "/etc/passwd", "dir/../a.txt"] {
            let error = open_file_beneath(&fixture.root, Path::new(invalid), strategy).unwrap_err();
            assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput, "{strategy:?} {invalid:?}");
        }
    }
}

#[test]
fn race_file_replaced_by_a_fifo_or_directory_does_not_hang() {
    for strategy in STRATEGIES {
        let fixture = tree();
        let resolved = read(&fixture, "a.txt").unwrap();
        std::fs::remove_file(fixture.root.join("a.txt")).unwrap();
        let fifo = std::ffi::CString::new(fixture.root.join("a.txt").to_str().unwrap()).unwrap();
        // SAFETY: a valid NUL-terminated path; mkfifo touches no memory.
        assert_eq!(unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) }, 0);
        // Opening a FIFO for reading blocks until a writer appears; this must not.
        let error = resolved.open_file_with(strategy).unwrap_err();
        assert!(matches!(error, Error::WrongKind { expected: "a regular file", .. }), "{strategy:?}: {error:?}");

        let directory = read(&fixture, "dir/b.txt").unwrap();
        std::fs::remove_file(fixture.root.join("dir/b.txt")).unwrap();
        std::fs::create_dir(fixture.root.join("dir/b.txt")).unwrap();
        let error = directory.open_file_with(strategy).unwrap_err();
        assert!(matches!(error, Error::WrongKind { .. }), "{strategy:?}: {error:?}");
    }
}

#[test]
fn race_file_removed_after_resolution_is_not_found() {
    let fixture = tree();
    let resolved = read(&fixture, "dir/b.txt").unwrap();
    std::fs::remove_dir_all(fixture.root.join("dir")).unwrap();
    for strategy in STRATEGIES {
        assert!(matches!(resolved.open_file_with(strategy), Err(Error::NotFound { .. })), "{strategy:?}");
    }
}

#[test]
fn race_fallback_is_chosen_when_openat2_is_unavailable() {
    use super::secure_open::{OPENAT2_UNAVAILABLE, openat2_unusable};
    use std::sync::atomic::Ordering;
    // Old kernels answer ENOSYS, sandboxes EPERM; a real refusal is not a reason to retry.
    for unusable in [libc::ENOSYS, libc::EPERM, libc::EINVAL, libc::E2BIG, libc::EAGAIN] {
        assert!(openat2_unusable(&std::io::Error::from_raw_os_error(unusable)), "{unusable}");
    }
    for real in [libc::ELOOP, libc::ENOENT, libc::EACCES, libc::EXDEV, libc::ENOTDIR] {
        assert!(!openat2_unusable(&std::io::Error::from_raw_os_error(real)), "{real}");
    }
    // With the "unavailable" latch set, Auto goes straight to the walk and
    // keeps the same contract. (Other tests running meanwhile are unaffected:
    // both paths give the same answers.)
    let fixture = tree();
    let ordinary = read(&fixture, "dir/b.txt").unwrap();
    let swapped = read(&fixture, "a.txt").unwrap();
    std::fs::remove_file(fixture.root.join("a.txt")).unwrap();
    symlink("../outside/secret.txt", fixture.root.join("a.txt"));
    let before = OPENAT2_UNAVAILABLE.swap(true, Ordering::Relaxed);
    let opened = ordinary.open_file_with(Strategy::Auto);
    let refused = swapped.open_file_with(Strategy::Auto);
    OPENAT2_UNAVAILABLE.store(before, Ordering::Relaxed);
    assert_eq!(opened.unwrap().1.len(), 5);
    assert_eq!(refused.unwrap_err().rule(), Some(Rule::ChangedToSymlink));
}

/// A directory that keeps flipping between the real one and a symlink to the
/// outside, while every read tool runs against it. What is guaranteed: no
/// tool ever returns content or names that exist only outside. What is not:
/// which of "found", "refused" or "not found" a particular call gets.
#[test]
fn race_flipping_directory_never_yields_outside_content() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    let fixture = tree();
    write(&fixture.outside, "b.txt", "SECRET\n");
    write(&fixture.outside, "only-outside.txt", "SECRET\n");
    symlink("../outside", fixture.root.join("park_link"));
    let root = fixture.root.clone();
    let stop = Arc::new(AtomicBool::new(false));
    let flips = Arc::new(AtomicU64::new(0));
    let flipper = {
        let (stop, flips) = (stop.clone(), flips.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let _ = std::fs::rename(root.join("dir"), root.join("park_real"));
                let _ = std::fs::rename(root.join("park_link"), root.join("dir"));
                std::thread::yield_now();
                let _ = std::fs::rename(root.join("dir"), root.join("park_link"));
                let _ = std::fs::rename(root.join("park_real"), root.join("dir"));
                flips.fetch_add(1, Ordering::Relaxed);
            }
        })
    };
    // Run until the race has demonstrably happened often, not for a fixed
    // time: a loaded machine gets more time, never a weaker test.
    let give_up = Instant::now() + Duration::from_secs(120);
    let (mut found, mut rounds) = (0u64, 0u64);
    while (rounds < 400 || found < 20 || flips.load(Ordering::Relaxed) < 400) && Instant::now() < give_up {
        rounds += 1;
        if let Ok(result) = super::read::read(&fixture.workspace, "dir/b.txt", None, None) {
            assert_eq!(result.lines[0].text, "beta", "read returned the outside file");
            found += 1;
        }
        for strategy in [Strategy::Openat2, Strategy::Walk] {
            let Ok(resolved) = read(&fixture, "dir/b.txt") else { continue };
            if let Ok((file, _)) = resolved.open_file_with(strategy) {
                assert_eq!(std::io::read_to_string(file).unwrap(), "beta\n", "{strategy:?} opened the outside file");
            }
        }
        if rounds % 8 == 0 {
            let listed = super::glob::glob(&fixture.workspace, "**", ".").unwrap();
            assert!(!listed.render().contains("only-outside"), "glob listed a name from outside: {}", listed.render());
            let hits = super::search::search(&fixture.workspace, &super::search::SearchRequest::new("SECRET")).unwrap();
            assert_eq!(hits.shown, 0, "search matched outside content: {}", hits.render());
        }
    }
    stop.store(true, Ordering::Relaxed);
    flipper.join().unwrap();
    let flips = flips.load(Ordering::Relaxed);
    assert!(flips >= 400 && rounds >= 400, "the race did not really run: {flips} flips, {rounds} rounds");
    assert!(found >= 20, "the real file was readable only {found} times in {rounds} rounds");
}
