//! Property-style test of the resolver: random paths over an alphabet that
//! contains every escape ingredient (`..`, `.`, symlinks to the outside, to
//! the extra root, back into the root, loops, absolute prefixes) never resolve
//! outside the allowed areas, in either access mode.

use std::path::{Component, Path};

use super::error::Error;
use super::resolve::{Access, Area, Kind, resolve};
use super::resolve_tests::tree;
use super::secure_open::Strategy;
use super::test_support::Fixture;

/// Deterministic xorshift generator; the seed makes every failure repeatable.
struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[(self.next() % items.len() as u64) as usize]
    }
}

const COMPONENTS: &[&str] = &[
    "..", "..", ".", "a.txt", "dir", "sub", "b.txt", "new", ".git", "config", "link_in", "link_dir_in", "abs_in", "link_root", "link_out",
    "link_dir_out", "link_up", "link_extra", "link_extra_dir", "escape", "back", "loop1", "dangling", "outside", "secret.txt", "extra", "ws",
    "e.txt", "alias", "",
];

fn random_path(random: &mut Random, prefixes: &[&str]) -> String {
    let mut path = random.pick(prefixes).to_string();
    for index in 0..(1 + random.next() % 8) {
        if index > 0 || !path.is_empty() {
            path.push('/');
        }
        path.push_str(random.pick(COMPONENTS));
    }
    path
}

/// The real location must be symlink-free as far as it exists: resolving it
/// with the operating system's own rules gives the same location.
fn assert_real(real: &Path, path: &str) {
    let mut existing = real.to_path_buf();
    while std::fs::symlink_metadata(&existing).is_err() {
        assert!(existing.pop(), "{path}: no existing ancestor of {real:?}");
    }
    assert_eq!(std::fs::canonicalize(&existing).unwrap(), existing, "{path}: {real:?} is not a real path");
    assert!(real.components().all(|part| matches!(part, Component::Normal(_) | Component::RootDir)), "{path}: {real:?}");
}

/// An independent oracle for an accepted relative path: the operating system's
/// own resolution. Where it can resolve the path it must arrive at the same
/// place, and no prefix of the caller's path may lead outside the allowed
/// areas on the way (a path must not leave and come back).
fn assert_agrees_with_the_kernel(fixture: &Fixture, path: &str, real: &Path) {
    if Path::new(path).is_absolute() {
        return;
    }
    if let Ok(kernel) = std::fs::canonicalize(fixture.root.join(path)) {
        assert_eq!(kernel, real, "{path}: the kernel resolves it elsewhere");
    }
    let mut prefix = fixture.root.clone();
    for part in Path::new(path).components() {
        prefix.push(part);
        if let Ok(reached) = std::fs::canonicalize(&prefix) {
            assert!(
                reached.starts_with(&fixture.root) || reached.starts_with(&fixture.extra),
                "{path}: accepted although its prefix {prefix:?} leads outside"
            );
        }
    }
}

fn check(fixture: &Fixture, path: &str, access: Access) -> bool {
    let (root, extra) = (&fixture.root, &fixture.extra);
    let resolved = match resolve(&fixture.workspace, path, access) {
        Ok(resolved) => resolved,
        Err(error) => {
            let text = error.to_string();
            assert!(!text.contains("SECRET"), "{path}: {text}");
            if !path.contains("secret") {
                assert!(!text.contains("secret"), "{path} leaked a target: {text}");
            }
            assert!(!matches!(error, Error::Git { .. } | Error::Registry(_)), "{path}: {error:?}");
            return false;
        }
    };
    assert_real(&resolved.real, path);
    assert_agrees_with_the_kernel(fixture, path, &resolved.real);
    let in_root = resolved.real.starts_with(root);
    let in_extra = resolved.real.starts_with(extra);
    assert_eq!(resolved.base.join(&resolved.relative), resolved.real, "{path}");
    match access {
        Access::Read => {
            assert!(in_root || in_extra, "{path} read-resolved outside: {:?}", resolved.real);
            assert_eq!(resolved.area == Area::Root, in_root, "{path}");
            assert_ne!(resolved.kind, Kind::Missing, "{path}");
            // Whatever is opened is the file at the real location, and its
            // content is never the outside file's.
            if resolved.kind == Kind::File {
                for strategy in [Strategy::Openat2, Strategy::Walk] {
                    let (file, _) = resolved.open_file_with(strategy).unwrap_or_else(|error| panic!("{path}: {error}"));
                    let text = std::io::read_to_string(file).unwrap();
                    assert!(!text.contains("SECRET"), "{path} read the outside file");
                }
            }
        }
        Access::Write => {
            assert!(in_root, "{path} write-resolved outside the root: {:?}", resolved.real);
            assert_eq!(resolved.area, Area::Root, "{path}");
            assert!(!resolved.relative.components().any(|part| part.as_os_str() == ".git"), "{path}: {:?}", resolved.real);
            if let Ok(metadata) = std::fs::symlink_metadata(&resolved.real) {
                assert!(!metadata.file_type().is_symlink(), "{path} write-resolved to a symlink");
            }
        }
    }
    true
}

#[test]
fn property_random_paths_never_resolve_outside() {
    let fixture = tree();
    let root = fixture.root.to_str().unwrap().to_string();
    let top = fixture.top.to_str().unwrap().to_string();
    let extra = fixture.extra.to_str().unwrap().to_string();
    let alias = format!("{top}/alias");
    let prefixes: [&str; 14] = ["", "", "", "", "/", "//", &root, &top, &extra, &alias, "/etc", "/proc/self", "./", "../"];
    let mut random = Random(0x9E37_79B9_7F4A_7C15);
    let (mut accepted_reads, mut accepted_writes, mut total) = (0u32, 0u32, 0u32);
    for _ in 0..30_000 {
        let path = random_path(&mut random, &prefixes);
        total += 1;
        accepted_reads += u32::from(check(&fixture, &path, Access::Read));
        accepted_writes += u32::from(check(&fixture, &path, Access::Write));
    }
    // The test only means something if both outcomes actually occur.
    assert!(accepted_reads > 500 && accepted_reads < total - 500, "reads accepted: {accepted_reads} of {total}");
    assert!(accepted_writes > 500 && accepted_writes < total - 500, "writes accepted: {accepted_writes} of {total}");
}

#[test]
fn property_random_bytes_never_panic() {
    let fixture = tree();
    let mut random = Random(0xD1B5_4A32_D192_ED03);
    let alphabet: Vec<char> = "/.a\\~ \t\n\0é文-_:*?[]{}$`'\"|;&<>%".chars().collect();
    for _ in 0..20_000 {
        let length = random.next() % 40;
        let path: String = (0..length).map(|_| alphabet[(random.next() % alphabet.len() as u64) as usize]).collect();
        for access in [Access::Read, Access::Write] {
            if let Ok(resolved) = resolve(&fixture.workspace, &path, access) {
                assert!(resolved.real.starts_with(&fixture.root) || resolved.real.starts_with(&fixture.extra), "{path:?}");
            }
        }
    }
}
