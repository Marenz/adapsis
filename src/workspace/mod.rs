//! Workspace tools: reading, searching and listing a repository through a git
//! worktree that belongs to one task (DESIGN L2, T4; issue #129).
//!
//! This is the library half. `api/native_tools/workspace.rs` wires the
//! lifecycle and the read-only tools into owner turns; the write side, commands
//! and jobs are wired by later slices and until then reachable only from tests.
//!
//! * `registry`: which repositories exist (`workspaces.toml`).
//! * `lifecycle`: `open` / `find` / `status` / `close` of a workspace.
//! * `resolve` + `secure_open`: the path rules, and opening what they allow.
//! * `read`, `search`, `glob`: the read-only tools. Each returns a plain
//!   struct with a `render()` that is the text the model sees.
//! * `secure_write`, `edit`, `write`: the write side. A file is replaced
//!   atomically and never through a symlink; an existing file is changed only
//!   if it is what the model read.
//! * `run` + `contain`: one shell command in the workspace, bounded in time
//!   and output, cancellable, contained in a systemd scope where possible.
//! * `jobs` + `job_store` + `job_output`: commands that outlive the call and
//!   a restart of this process, as systemd user services with their state
//!   on disk.
//! * `systemd`: the few things asked of the user manager.
//! * `ledger`: what was read, so a later edit can require "read and unchanged".
//! * `binding`: which workspace a conversation works in, with its ledger, on disk.
//! * `limits`: every cap, as a named policy value.
//!
//! Scope of the safety rules (DESIGN N5): they keep the file tools inside the
//! workspace against mistakes. They are not a sandbox.

// Wired into the native tools by the next slice; until then only tests call in.
#![allow(dead_code)]

mod contain;
mod adapter;
mod reservation;
mod state_io;
pub mod binding;
pub mod edit;
pub mod error;
pub mod git;
pub mod glob;
pub mod job_output;
pub mod job_store;
pub mod jobs;
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
pub mod systemd;
mod walk;
pub mod write;

#[cfg(test)]
pub(crate) mod test_support;

#[cfg(test)]
mod edit_tests;
#[cfg(test)]
mod git_tests;
#[cfg(test)]
mod jobs_tests;
#[cfg(test)]
mod repair_tests;
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
pub use binding::{Binding, BindingStore};
#[allow(unused_imports)]
pub use edit::{EditResult, edit};
#[allow(unused_imports)]
pub use error::{Error, Result, Rule};
#[allow(unused_imports)]
pub use glob::{GlobEntry, GlobResult, glob, glob_within};
#[allow(unused_imports)]
pub use ledger::{Freshness, ReadLedger};
#[allow(unused_imports)]
pub use lifecycle::{Workspace, WorkspaceStatus, close, find, list, open, status};
#[allow(unused_imports)]
pub use read::{FileFacts, ReadLine, ReadResult, read, read_within};
#[allow(unused_imports)]
pub use registry::{Registry, Repository};
#[allow(unused_imports)]
pub use resolve::{Access, Area, Kind, Resolved, resolve};
#[allow(unused_imports)]
pub use search::{SearchLine, SearchMode, SearchRequest, SearchResult, search};
#[allow(unused_imports)]
pub use contain::Cancel;
#[allow(unused_imports)]
pub use job_output::{JobOutput, job_output, render_jobs};
#[allow(unused_imports)]
pub use job_store::{JobRecord, JobState};
#[allow(unused_imports)]
pub use jobs::{JobEntry, JobRequest, JobStatus, JobStore, job_kill, job_list, job_start, job_status};
#[allow(unused_imports)]
pub use run::{Contained, Containment, Exit, RunOptions, RunRequest, RunResult, run, run_with};
#[allow(unused_imports)]
pub use walk::Stop;
#[allow(unused_imports)]
pub use write::{WriteResult, write};
