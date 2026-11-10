//! `write`: create a file, or replace the whole content of one the model has
//! read.
//!
//! Creating is free: the path must be inside the workspace and must not
//! exist; missing directories are created. Replacing an existing file is held
//! to the same rule as `edit`: the file must be, byte for byte, what the
//! model read. The content is written as given (no line-ending conversion),
//! atomically, through `secure_write`.

use super::edit::{facts_of, load_for_change};
use super::error::{Error, Result};
use super::ledger::ReadLedger;
use super::lifecycle::Workspace;
use super::limits::{SCAN_MAX_BYTES, WRITE_MAX_BYTES};
use super::resolve::{Access, Kind, resolve};
use super::secure_write::Replace;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteResult {
    /// Workspace-relative path of the written file.
    pub path: String,
    /// The file did not exist before.
    pub created: bool,
    pub bytes: u64,
    pub lines: u64,
}

impl WriteResult {
    pub fn render(&self) -> String {
        let verb = if self.created { "created" } else { "replaced the content of" };
        format!("{verb} {} ({} bytes, {} lines)", self.path, self.bytes, self.lines)
    }
}

pub fn write(workspace: &Workspace, ledger: &mut ReadLedger, path: &str, content: &str) -> Result<WriteResult> {
    if content.len() > WRITE_MAX_BYTES {
        return Err(Error::InvalidArgument(format!(
            "the content is {} bytes; one write accepts at most {WRITE_MAX_BYTES}. Write the file in parts with a command, or generate it",
            content.len()
        )));
    }
    let resolved = resolve(workspace, path, Access::Write)?;
    let how = match resolved.kind {
        Kind::Missing => Replace::CreateNew,
        Kind::File => {
            // The old content is only needed for its identity; a file too
            // large to hash was never recorded as read and is refused here.
            let loaded = load_for_change(&resolved, ledger, SCAN_MAX_BYTES)?;
            Replace::Overwrite { expect: Some(loaded.stamp) }
        }
        Kind::Directory => return Err(Error::WrongKind { path: resolved.supplied, expected: "a file (it is a directory)" }),
        Kind::Other => return Err(Error::WrongKind { path: resolved.supplied, expected: "a regular file" }),
    };
    let written = resolved.write_content(content.as_bytes(), how)?;
    ledger.record(&resolved.display, facts_of(content.as_bytes(), &written));
    let lines = content.lines().count() as u64;
    Ok(WriteResult { path: resolved.display, created: written.created, bytes: written.size, lines })
}
