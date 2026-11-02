//! Dormant structural identity/scope model, not persistence or authorization.
//! No runtime entry point consumes this catalog. See docs/task-foundation.md.
#![allow(dead_code)] // Intentional until a separately reviewed integration.

mod ids;
pub(crate) use ids::{DurableTaskId, ProjectId, TaskRunId, WorkspaceId};

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;

const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Workspace {
    pub(crate) id: WorkspaceId,
    pub(crate) label: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Project {
    pub(crate) id: ProjectId,
    pub(crate) workspace_id: WorkspaceId,
    pub(crate) label: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DurableTask {
    pub(crate) id: DurableTaskId,
    pub(crate) workspace_id: WorkspaceId,
    pub(crate) project_id: Option<ProjectId>,
    pub(crate) title: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TaskRun {
    pub(crate) id: TaskRunId,
    pub(crate) task_id: DurableTaskId,
}

/// Untrusted staging data. Only FoundationCatalog represents validated links.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FoundationSnapshot {
    pub(crate) version: u32,
    pub(crate) workspaces: Vec<Workspace>,
    pub(crate) projects: Vec<Project>,
    pub(crate) tasks: Vec<DurableTask>,
    pub(crate) runs: Vec<TaskRun>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ValidationError {
    UnsupportedVersion,
    DuplicateWorkspace,
    DuplicateProject,
    DuplicateTask,
    DuplicateRun,
    MissingWorkspace,
    MissingProject,
    MissingTask,
    CrossWorkspaceProject,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::UnsupportedVersion => "unsupported task foundation version; expected version 1",
            Self::DuplicateWorkspace => {
                "duplicate workspace identity; require unique workspace IDs"
            }
            Self::DuplicateProject => "duplicate project identity; require unique project IDs",
            Self::DuplicateTask => "duplicate task identity; require unique task IDs",
            Self::DuplicateRun => "duplicate run identity; require unique run IDs",
            Self::MissingWorkspace => {
                "missing workspace reference; include the referenced workspace"
            }
            Self::MissingProject => "missing project reference; include the referenced project",
            Self::MissingTask => "missing task reference; include the referenced task",
            Self::CrossWorkspaceProject => {
                "cross-workspace task/project link; require the same workspace"
            }
        })
    }
}

impl std::error::Error for ValidationError {}

/// Immutable validated structure. This does not establish anyone's access rights.
/// Deserialization always validates the complete snapshot before publishing it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "FoundationSnapshot", into = "FoundationSnapshot")]
pub(crate) struct FoundationCatalog(FoundationSnapshot);

impl TryFrom<FoundationSnapshot> for FoundationCatalog {
    type Error = ValidationError;

    fn try_from(snapshot: FoundationSnapshot) -> Result<Self, Self::Error> {
        use ValidationError::*;
        if snapshot.version != FORMAT_VERSION {
            return Err(UnsupportedVersion);
        }
        unique(snapshot.workspaces.iter().map(|v| v.id), DuplicateWorkspace)?;
        unique(snapshot.projects.iter().map(|v| v.id), DuplicateProject)?;
        unique(snapshot.tasks.iter().map(|v| v.id), DuplicateTask)?;
        unique(snapshot.runs.iter().map(|v| v.id), DuplicateRun)?;
        let workspace_exists = |id| snapshot.workspaces.iter().any(|w| w.id == id);
        for project in &snapshot.projects {
            if !workspace_exists(project.workspace_id) {
                return Err(MissingWorkspace);
            }
        }
        for task in &snapshot.tasks {
            if !workspace_exists(task.workspace_id) {
                return Err(MissingWorkspace);
            }
            if let Some(id) = task.project_id {
                let project = snapshot
                    .projects
                    .iter()
                    .find(|p| p.id == id)
                    .ok_or(MissingProject)?;
                if project.workspace_id != task.workspace_id {
                    return Err(CrossWorkspaceProject);
                }
            }
        }
        for run in &snapshot.runs {
            if !snapshot.tasks.iter().any(|t| t.id == run.task_id) {
                return Err(MissingTask);
            }
        }
        Ok(Self(snapshot))
    }
}

fn unique<T: Eq + std::hash::Hash>(
    ids: impl Iterator<Item = T>,
    error: ValidationError,
) -> Result<(), ValidationError> {
    let mut seen = HashSet::new();
    for id in ids {
        if !seen.insert(id) {
            return Err(error);
        }
    }
    Ok(())
}

impl From<FoundationCatalog> for FoundationSnapshot {
    fn from(catalog: FoundationCatalog) -> Self {
        catalog.0
    }
}

impl FoundationCatalog {
    pub(crate) fn snapshot(&self) -> &FoundationSnapshot {
        &self.0
    }

    pub(crate) fn run_workspace(&self, id: TaskRunId) -> Option<WorkspaceId> {
        let run = self.0.runs.iter().find(|r| r.id == id)?;
        self.0
            .tasks
            .iter()
            .find(|t| t.id == run.task_id)
            .map(|t| t.workspace_id)
    }
}

#[cfg(test)]
mod tests;
