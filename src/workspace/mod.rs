//! Workspace tools: reading, searching and listing a repository through a git
//! worktree that belongs to one task (DESIGN L2, T4; issue #129).
//!
//! This is the library half. Nothing here is wired into the native tool list,
//! a turn or a conversation yet; that is the next slice, and until it lands
//! most of this module is reachable only from its tests.
//!
//! * `registry`: which repositories exist (`workspaces.toml`).
//! * `lifecycle`: `open` / `find` / `status` / `close` of a workspace.
//! * `resolve` + `secure_open`: the path rules, and opening what they allow.
//! * `read`, `search`, `glob`: the read-only tools. Each returns a plain
//!   struct with a `render()` that is the text the model sees.
//! * `secure_write`, `edit`, `write`: the write side. A file is replaced
//!   atomically and never through a symlink; an existing file is changed only
//!   if it is what the model read.
//! * `run`: one shell command in the workspace, bounded in time and output.
//! * `ledger`: what was read, so a later edit can require "read and unchanged".
//! * `limits`: every cap, as a named policy value.
//!
//! Scope of the safety rules (DESIGN N5): they keep the file tools inside the
//! workspace against mistakes. They are not a sandbox.

// Wired into the native tools by the next slice; until then only tests call in.
#![allow(dead_code)]

pub mod edit;
pub mod error;
pub mod git;
pub mod glob;
pub mod ledger;
pub mod lifecycle;
pub mod limits;
pub mod read;
pub mod registry;
pub mod resolve;
pub mod run;
mod scan;
pub mod search;
pub mod secure_open;
pub mod secure_write;
mod walk;
pub mod write;

#[cfg(test)]
mod test_support;

#[cfg(test)]
mod edit_tests;
#[cfg(test)]
mod git_tests;
#[cfg(test)]
mod lifecycle_tests;
#[cfg(test)]
mod property_tests;
#[cfg(test)]
mod read_tests;
#[cfg(test)]
mod registry_tests;
#[cfg(test)]
mod resolve_tests;
#[cfg(test)]
mod run_tests;
#[cfg(test)]
mod search_tests;
#[cfg(test)]
mod secure_write_tests;

#[allow(unused_imports)]
pub use edit::{EditResult, edit};
#[allow(unused_imports)]
pub use error::{Error, Result, Rule};
#[allow(unused_imports)]
pub use glob::{GlobEntry, GlobResult, glob};
#[allow(unused_imports)]
pub use ledger::{Freshness, ReadLedger};
#[allow(unused_imports)]
pub use lifecycle::{Workspace, WorkspaceStatus, close, find, open, status};
#[allow(unused_imports)]
pub use read::{FileFacts, ReadLine, ReadResult, read};
#[allow(unused_imports)]
pub use registry::{Registry, Repository};
#[allow(unused_imports)]
pub use resolve::{Access, Area, Kind, Resolved, resolve};
#[allow(unused_imports)]
pub use search::{SearchLine, SearchMode, SearchRequest, SearchResult, search};
#[allow(unused_imports)]
pub use run::{Exit, RunRequest, RunResult, run, run_with};
#[allow(unused_imports)]
pub use walk::Stop;
#[allow(unused_imports)]
pub use write::{WriteResult, write};
