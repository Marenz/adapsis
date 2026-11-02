use super::*;

fn fixture() -> FoundationSnapshot {
    let workspace = Workspace {
        id: WorkspaceId::new(),
        label: "same label".into(),
    };
    let other = Workspace {
        id: WorkspaceId::new(),
        label: workspace.label.clone(),
    };
    let project = Project {
        id: ProjectId::new(),
        workspace_id: workspace.id,
        label: "project".into(),
    };
    let task = DurableTask {
        id: DurableTaskId::new(),
        workspace_id: workspace.id,
        project_id: Some(project.id),
        title: "private title".into(),
    };
    let runs = (0..2)
        .map(|_| TaskRun {
            id: TaskRunId::new(),
            task_id: task.id,
        })
        .collect();
    FoundationSnapshot {
        version: FORMAT_VERSION,
        workspaces: vec![workspace, other],
        projects: vec![project],
        tasks: vec![task],
        runs,
    }
}

#[test]
fn roundtrip_preserves_all_identities_and_links() {
    let snapshot = fixture();
    let catalog = FoundationCatalog::try_from(snapshot.clone()).unwrap();
    let json = serde_json::to_string(&catalog).unwrap();
    let restored: FoundationCatalog = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.snapshot(), &snapshot);
    assert_eq!(serde_json::to_string(&restored).unwrap(), json);
    assert_ne!(snapshot.workspaces[0].id, snapshot.workspaces[1].id);
    assert_ne!(snapshot.runs[0].id, snapshot.runs[1].id);
    for run in &snapshot.runs {
        assert_eq!(
            restored.run_workspace(run.id),
            Some(snapshot.tasks[0].workspace_id)
        );
    }
    assert_eq!(restored.run_workspace(TaskRunId::new()), None);
}

#[test]
fn rename_and_reorder_do_not_reassign_identity() {
    let mut snapshot = fixture();
    let original = snapshot.clone();
    snapshot.workspaces[0].label = "renamed".into();
    snapshot.projects[0].label = "renamed project".into();
    snapshot.tasks[0].title = "renamed task".into();
    snapshot.workspaces.reverse();
    snapshot.runs.reverse();
    let catalog = FoundationCatalog::try_from(snapshot).unwrap();
    assert_eq!(catalog.snapshot().projects[0].id, original.projects[0].id);
    assert_eq!(catalog.snapshot().tasks[0].id, original.tasks[0].id);
    for run in original.runs {
        assert_eq!(
            catalog.run_workspace(run.id),
            Some(original.workspaces[0].id)
        );
    }
}

#[test]
fn projectless_tasks_and_empty_catalog_are_valid_without_default_workspace() {
    let mut snapshot = fixture();
    snapshot.tasks[0].project_id = None;
    snapshot.projects.clear();
    assert!(FoundationCatalog::try_from(snapshot).is_ok());
    let empty = FoundationSnapshot {
        version: FORMAT_VERSION,
        workspaces: vec![],
        projects: vec![],
        tasks: vec![],
        runs: vec![],
    };
    assert_eq!(
        FoundationCatalog::try_from(empty.clone())
            .unwrap()
            .snapshot(),
        &empty
    );
}

fn rejected(snapshot: FoundationSnapshot, expected: ValidationError) {
    // Both construction paths reject the entire candidate, not just its bad row.
    assert_eq!(
        FoundationCatalog::try_from(snapshot.clone()).unwrap_err(),
        expected
    );
    let json = serde_json::to_string(&snapshot).unwrap();
    let error = serde_json::from_str::<FoundationCatalog>(&json)
        .unwrap_err()
        .to_string();
    assert!(error.contains(&expected.to_string()));
    assert!(!error.contains("private title"));
}

#[test]
fn duplicate_ids_of_every_kind_are_rejected() {
    let mut s = fixture();
    s.workspaces.push(s.workspaces[0].clone());
    rejected(s, ValidationError::DuplicateWorkspace);
    let mut s = fixture();
    s.projects.push(s.projects[0].clone());
    rejected(s, ValidationError::DuplicateProject);
    let mut s = fixture();
    s.tasks.push(s.tasks[0].clone());
    rejected(s, ValidationError::DuplicateTask);
    let mut s = fixture();
    s.runs.push(s.runs[0].clone());
    rejected(s, ValidationError::DuplicateRun);
}

#[test]
fn dangling_links_and_cross_workspace_projects_are_rejected() {
    let mut s = fixture();
    s.projects[0].workspace_id = WorkspaceId::new();
    rejected(s, ValidationError::MissingWorkspace);
    let mut s = fixture();
    s.tasks[0].workspace_id = WorkspaceId::new();
    rejected(s, ValidationError::MissingWorkspace);
    let mut s = fixture();
    s.tasks[0].project_id = Some(ProjectId::new());
    rejected(s, ValidationError::MissingProject);
    let mut s = fixture();
    s.runs[0].task_id = DurableTaskId::new();
    rejected(s, ValidationError::MissingTask);
    let mut s = fixture();
    s.tasks[0].workspace_id = s.workspaces[1].id;
    rejected(s, ValidationError::CrossWorkspaceProject);
}

#[test]
fn invalid_candidate_cannot_mutate_previously_validated_catalog() {
    let catalog = FoundationCatalog::try_from(fixture()).unwrap();
    let before = serde_json::to_string(&catalog).unwrap();
    let mut candidate = catalog.snapshot().clone();
    candidate.tasks[0].workspace_id = candidate.workspaces[1].id;
    rejected(candidate, ValidationError::CrossWorkspaceProject);
    assert_eq!(serde_json::to_string(&catalog).unwrap(), before);
}

#[test]
fn version_and_required_fields_are_not_defaulted() {
    let mut s = fixture();
    s.version = FORMAT_VERSION + 1;
    rejected(s, ValidationError::UnsupportedVersion);
    let value = serde_json::to_value(fixture()).unwrap();
    for field in ["version", "workspaces", "projects", "tasks", "runs"] {
        let mut candidate = value.clone();
        candidate.as_object_mut().unwrap().remove(field);
        assert!(serde_json::from_value::<FoundationCatalog>(candidate).is_err());
    }
    let mut missing_id = value.clone();
    missing_id["tasks"][0].as_object_mut().unwrap().remove("id");
    assert!(serde_json::from_value::<FoundationCatalog>(missing_id).is_err());
    let mut run_scope = value;
    run_scope["runs"][0]["workspace_id"] = run_scope["workspaces"][1]["id"].clone();
    assert!(serde_json::from_value::<FoundationCatalog>(run_scope).is_err());
}

#[test]
fn ids_reject_wrong_kind_noncanonical_and_non_v4_encodings() {
    let encoded = [
        serde_json::to_value(WorkspaceId::new()).unwrap(),
        serde_json::to_value(ProjectId::new()).unwrap(),
        serde_json::to_value(DurableTaskId::new()).unwrap(),
        serde_json::to_value(TaskRunId::new()).unwrap(),
    ];
    for (index, value) in encoded.iter().enumerate() {
        assert_eq!(
            serde_json::from_value::<WorkspaceId>(value.clone()).is_ok(),
            index == 0
        );
        assert_eq!(
            serde_json::from_value::<ProjectId>(value.clone()).is_ok(),
            index == 1
        );
        assert_eq!(
            serde_json::from_value::<DurableTaskId>(value.clone()).is_ok(),
            index == 2
        );
        assert_eq!(
            serde_json::from_value::<TaskRunId>(value.clone()).is_ok(),
            index == 3
        );
    }
    for raw in [
        "",
        "private title",
        "ws_00000000-0000-0000-0000-000000000000",
        "ws_abcdefab-cdef-4abc-8abc-abcdefabcdef ",
        "ws_ABCDEFAB-CDEF-4ABC-8ABC-ABCDEFABCDEF",
        "ws_abcdefabcdef4abc8abcabcdefabcdef",
        "ws_abcdefab-cdef-1abc-8abc-abcdefabcdef",
        "ws_abcdefab-cdef-4abc-0abc-abcdefabcdef",
    ] {
        let error = serde_json::from_value::<WorkspaceId>(raw.into()).unwrap_err();
        assert_eq!(error.to_string(), "invalid ws_identity");
    }
}
