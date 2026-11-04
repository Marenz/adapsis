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
use serde::Deserialize;

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

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
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

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ModuleSource {
    pub(super) format: SourceFormat,
    pub(super) dependencies: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
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

impl ManagedSources {
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
