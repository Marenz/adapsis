//! Opt-in authoritative source policy. The manifest, not filename preference or
//! session provenance, chooses a module's format. Source bytes remain the conflict
//! baseline; the session AST is only a recovery cache.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

pub(super) const MANIFEST_FILE: &str = "library-sources.json";
pub(super) const MAX_SOURCE_BYTES: usize = 2 * 1024 * 1024;
const MAX_MANIFEST_BYTES: usize = 64 * 1024;

/// Dangling policy links remain present; inspection failures never select legacy.
pub(super) fn manifest_present(dir: &Path) -> Result<bool> {
    let root = std::fs::metadata(dir).context("could not inspect source library directory")?;
    if !root.is_dir() {
        bail!("source library path is not a directory");
    }
    match std::fs::symlink_metadata(dir.join(MANIFEST_FILE)) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => {
            Err(error).context("could not inspect source manifest; refusing legacy fallback")
        }
    }
}

pub(super) fn read_source(path: &Path) -> std::io::Result<String> {
    read_bounded(path, MAX_SOURCE_BYTES)
}

pub(super) fn diagnostic_name(name: &str) -> String {
    name.chars()
        .take(128)
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn read_bounded(path: &Path, limit: usize) -> std::io::Result<String> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "source byte budget exceeded",
        ));
    }
    String::from_utf8(bytes)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "source must be UTF-8"))
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(super) enum SourceFormat {
    Python,
    LegacyAx,
}

impl SourceFormat {
    pub(super) fn extension(self) -> &'static str {
        match self {
            Self::Python => "py",
            Self::LegacyAx => "ax",
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct ModuleSource {
    pub(super) format: SourceFormat,
    pub(super) dependencies: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(super) struct SourceManifest {
    version: u32,
    #[serde(deserialize_with = "unique_modules")]
    pub(super) modules: BTreeMap<String, ModuleSource>,
}

fn unique_modules<'de, D>(
    deserializer: D,
) -> std::result::Result<BTreeMap<String, ModuleSource>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct Unique;
    impl<'de> serde::de::Visitor<'de> for Unique {
        type Value = BTreeMap<String, ModuleSource>;
        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a map with unique module names")
        }
        fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
        where
            A: serde::de::MapAccess<'de>,
        {
            let mut result = BTreeMap::new();
            while let Some((name, source)) = map.next_entry::<String, ModuleSource>()? {
                if result.insert(name, source).is_some() {
                    return Err(serde::de::Error::custom("duplicate source module name"));
                }
            }
            Ok(result)
        }
    }
    deserializer.deserialize_map(Unique)
}

#[derive(Clone, Debug)]
pub(super) struct ManagedSources {
    pub(super) manifest: SourceManifest,
    pub(super) manifest_hash: String,
    pub(super) order: Vec<String>,
}

fn valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    name.len() <= 128
        && chars
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Modules the managed writer never enrols: runtime scratch space and the
/// reserved `__` prefix (#124). They are not source and never reach a manifest.
pub(super) fn never_enrolled(name: &str) -> bool {
    name == "RunToolScratch" || name.starts_with("__") || !valid_name(name)
}

impl SourceManifest {
    /// Same module set and the same dependency SETS (declared order is free).
    pub(super) fn same_policy(&self, other: &Self) -> bool {
        self.version == other.version
            && self.modules.len() == other.modules.len()
            && self.modules.iter().all(|(name, source)| {
                other.modules.get(name).is_some_and(|theirs| {
                    theirs.format == source.format
                        && theirs.dependencies.iter().collect::<BTreeSet<_>>()
                            == source.dependencies.iter().collect::<BTreeSet<_>>()
                })
            })
    }
}

impl ManagedSources {
    /// Writer-side replacement of the manifest. Order: the caller has already
    /// written every new source file; this rename is the commit point. The
    /// loaded manifest must still be the one on disk (external edits win).
    pub(super) fn write_updated(&self, dir: &Path, manifest: SourceManifest) -> Result<Self> {
        let order = dependency_order(&manifest)?;
        let mut text = serde_json::to_string_pretty(&manifest)
            .context("could not serialize the source manifest")?;
        text.push('\n');
        if text.len() > MAX_MANIFEST_BYTES {
            bail!("source manifest would exceed its 64 KiB budget; manifest unchanged");
        }
        let updated = Self { manifest, manifest_hash: super::source_hash(&text), order };
        updated.check_files(dir)?;
        self.verify_manifest(dir)?;
        let target = dir.join(MANIFEST_FILE);
        let tmp = unique_temp(dir, MANIFEST_FILE);
        let mut file = OpenOptions::new().write(true).create_new(true).mode(0o600).open(&tmp)
            .context("could not create the manifest temporary file; manifest unchanged")?;
        let result = (|| {
            file.write_all(text.as_bytes())?;
            file.sync_all()?;
            self.verify_manifest(dir)?;
            std::fs::rename(&tmp, &target)?;
            File::open(dir)?.sync_all().context(
                "manifest rename succeeded but directory sync failed; reconcile and restart before retrying",
            )?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        result.map(|()| updated)
    }

    pub(super) fn read(dir: &Path) -> Result<Option<Self>> {
        if !manifest_present(dir)? {
            return Ok(None);
        }
        let path = dir.join(MANIFEST_FILE);
        let text = read_bounded(&path, MAX_MANIFEST_BYTES)
            .context("source manifest exists but is unreadable; refusing legacy fallback")?;
        let manifest: SourceManifest = serde_json::from_str(&text)
            .context("invalid library-sources.json; expected a versioned module source manifest")?;
        if manifest.version != 1 {
            bail!("unsupported source manifest version; expected version 1");
        }
        let order = dependency_order(&manifest)?;
        let result = Self {
            manifest,
            manifest_hash: super::source_hash(&text),
            order,
        };
        result.check_files(dir)?;
        Ok(Some(result))
    }

    pub(super) fn verify_manifest(&self, dir: &Path) -> Result<()> {
        let text = read_bounded(&dir.join(MANIFEST_FILE), MAX_MANIFEST_BYTES)
            .context("source manifest unavailable; reconcile it and restart before saving")?;
        if super::source_hash(&text) != self.manifest_hash {
            bail!("source manifest changed; no source was saved. Reconcile and restart the server");
        }
        self.check_files(dir)
    }

    fn check_files(&self, dir: &Path) -> Result<()> {
        for name in self.manifest.modules.keys() {
            if dir.join(format!("{name}.py")).exists() && dir.join(format!("{name}.ax")).exists() {
                bail!("module `{name}` has both .py and .ax sources; move rollback inputs outside the active library");
            }
        }
        Ok(())
    }

    pub(super) fn dependency_names(&self, name: &str) -> BTreeSet<String> {
        let mut names = BTreeSet::new();
        let mut pending = self.manifest.modules[name].dependencies.clone();
        while let Some(name) = pending.pop() {
            if names.insert(name.clone()) {
                pending.extend(self.manifest.modules[&name].dependencies.iter().cloned());
            }
        }
        names
    }
}

impl ManagedSources {
    /// A manifest that is not on disk (yet), for checks before a commit.
    pub(super) fn trial(manifest: SourceManifest) -> Result<Self> {
        let order = dependency_order(&manifest)?;
        Ok(Self { manifest, manifest_hash: String::new(), order })
    }
}

fn dependency_order(manifest: &SourceManifest) -> Result<Vec<String>> {
    if manifest.modules.len() > 128 {
        bail!("source manifest exceeds the 128-module budget");
    }
    for (name, source) in &manifest.modules {
        if !valid_name(name) {
            bail!("invalid module name in source manifest; use an ASCII identifier");
        }
        let mut seen = BTreeSet::new();
        for dependency in &source.dependencies {
            if !manifest.modules.contains_key(dependency) {
                bail!("module `{name}` has a dependency outside the closed source manifest");
            }
            if !seen.insert(dependency) {
                bail!("module `{name}` repeats a dependency in the source manifest");
            }
        }
    }
    let mut remaining: BTreeSet<_> = manifest.modules.keys().cloned().collect();
    let mut loaded = BTreeSet::new();
    let mut order = Vec::new();
    while !remaining.is_empty() {
        let ready = remaining
            .iter()
            .find(|name| {
                manifest.modules[*name]
                    .dependencies
                    .iter()
                    .all(|dep| loaded.contains(dep))
            })
            .cloned();
        let Some(name) = ready else {
            bail!("source manifest has a dependency cycle; cyclic source loading is not supported");
        };
        remaining.remove(&name);
        loaded.insert(name.clone());
        order.push(name);
    }
    Ok(order)
}

fn unique_temp(dir: &Path, filename: &str) -> std::path::PathBuf {
    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
    let suffix = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
    dir.join(format!(".{filename}.{}.new{suffix}.tmp", std::process::id()))
}

/// First source file of a module that is being enrolled. It is written BEFORE
/// the manifest lists it: until then the file is unlisted and ignored by a
/// start. An unlisted leftover of the same name (crash between this write and
/// the manifest, or a stray file) is never authoritative under the whitelist:
/// identical content is reused, anything else is moved aside, never deleted.
/// Returns the names of files that were moved aside.
pub(super) fn write_new_source(dir: &Path, name: &str, source: &str) -> Result<Vec<String>> {
    if source.len() > MAX_SOURCE_BYTES {
        bail!("serialized source exceeds the 2 MiB budget; disk unchanged");
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis())
        .unwrap_or_default();
    let mut moved = Vec::new();
    let target = dir.join(format!("{name}.py"));
    let mut reuse = false;
    for extension in ["py", "ax"] {
        let existing = dir.join(format!("{name}.{extension}"));
        match std::fs::symlink_metadata(&existing) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error).context("could not inspect an existing source file"),
            Ok(_) => {}
        }
        if extension == "py" && read_source(&existing).is_ok_and(|text| text == source) {
            reuse = true;
            continue;
        }
        let aside = format!("{name}.{extension}.orphan-{stamp}");
        std::fs::rename(&existing, dir.join(&aside))
            .context("could not move an unlisted leftover source file aside")?;
        moved.push(aside);
    }
    if reuse {
        return Ok(moved);
    }
    let tmp = unique_temp(dir, &format!("{name}.py"));
    let mut file = OpenOptions::new().write(true).create_new(true).mode(0o600).open(&tmp)
        .context("could not create the new source temporary file")?;
    let result = (|| {
        file.write_all(source.as_bytes())?;
        file.sync_all()?;
        std::fs::rename(&tmp, &target)?;
        File::open(dir)?.sync_all().context("new source rename succeeded but directory sync failed")?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result.map(|()| moved)
}

/// A unique owner-only sibling is renamed only after both source and manifest
/// baselines are rechecked. This is NOT cross-process compare-and-swap: an
/// uncooperative external writer can still race the final check and rename.
pub(super) fn write_checked(
    dir: &Path,
    filename: &str,
    source: &str,
    expected_hash: &str,
    managed: &ManagedSources,
) -> Result<()> {
    if source.len() > MAX_SOURCE_BYTES {
        bail!("serialized source exceeds the 2 MiB budget; disk unchanged");
    }
    managed.verify_manifest(dir)?;
    let target = dir.join(filename);
    verify_source(&target, expected_hash)?;
    if super::source_hash(source) == expected_hash {
        return Ok(());
    }
    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
    let suffix = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
    let tmp = dir.join(format!(".{filename}.{}.{}.tmp", std::process::id(), suffix));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&tmp)
        .context("could not create unique source temporary file; original source unchanged")?;
    let result = (|| {
        file.write_all(source.as_bytes())?;
        file.sync_all()?;
        managed.verify_manifest(dir)?;
        verify_source(&target, expected_hash)?;
        std::fs::rename(&tmp, &target)?;
        File::open(dir)?
            .sync_all()
            .context("source rename succeeded but directory sync failed; reconcile and restart before retrying")?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

fn verify_source(path: &Path, expected: &str) -> Result<()> {
    let bytes = read_source(path).context(
        "authoritative source unavailable; restore/reconcile it and restart before saving",
    )?;
    if super::source_hash(&bytes) != expected {
        bail!("authoritative source changed externally; disk wins. Preserve recovery checkpoint, reconcile, and restart");
    }
    Ok(())
}
