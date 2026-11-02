# Test-only constructors/drivers, appended to TaskDomain in an isolated runtime.

def scenario_catalog(case: str) -> Catalog:
    workspace: Workspace = Workspace(id="w", label="same name")
    other: Workspace = Workspace(id="other", label="same name")
    project: Project = Project(id="p", workspace_id="w", label="project")
    task: Task = Task(id="t", workspace_id="w", project=InTaskProject("p"), title="private title", state=TaskActive)
    run: Run = Run(id="r", task_id="t", state=RunRunning)
    retry: Run = Run(id="retry", task_id="t", state=RunPending)
    workspaces: list[Workspace] = [workspace, other]
    projects: list[Project] = [project]
    tasks: list[Task] = [task]
    runs: list[Run] = [run, retry]
    if case == "projectless":
        projects = []
        projectless_task: Task = Task(id="t", workspace_id="w", project=NoTaskProject, title="help", state=TaskReady)
        tasks = [projectless_task]
    elif case == "empty":
        workspaces = []
        projects = []
        tasks = []
        runs = []
    elif case == "duplicate_workspace":
        workspaces = [workspace, workspace]
    elif case == "duplicate_project":
        projects = [project, project]
    elif case == "duplicate_task":
        tasks = [task, task]
    elif case == "duplicate_run":
        runs = [run, run]
    elif case == "empty_workspace":
        empty_workspace: Workspace = Workspace(id="", label="private title")
        workspaces = [empty_workspace]
    elif case == "empty_project":
        empty_project: Project = Project(id="", workspace_id="w", label="private title")
        projects = [empty_project]
    elif case == "empty_task":
        empty_task: Task = Task(id="", workspace_id="w", project=NoTaskProject, title="private title", state=TaskReady)
        tasks = [empty_task]
    elif case == "empty_run":
        empty_run: Run = Run(id="", task_id="t", state=RunPending)
        runs = [empty_run]
    elif case == "missing_project_workspace":
        missing_workspace_project: Project = Project(id="p", workspace_id="absent", label="private title")
        projects = [missing_workspace_project]
    elif case == "missing_task_workspace":
        missing_workspace_task: Task = Task(id="t", workspace_id="absent", project=NoTaskProject, title="private title", state=TaskReady)
        tasks = [missing_workspace_task]
    elif case == "missing_project":
        projects = []
    elif case == "cross_workspace":
        cross_workspace_task: Task = Task(id="t", workspace_id="other", project=InTaskProject("p"), title="private title", state=TaskReady)
        tasks = [cross_workspace_task]
    elif case == "missing_run_task":
        tasks = []
    elif case == "renamed":
        renamed_workspace: Workspace = Workspace(id="w", label="renamed")
        renamed_project: Project = Project(id="p", workspace_id="w", label="renamed")
        renamed_task: Task = Task(id="t", workspace_id="w", project=InTaskProject("p"), title="renamed", state=TaskActive)
        workspaces = [other, renamed_workspace]
        projects = [renamed_project]
        tasks = [renamed_task]
        runs = [retry, run]
    elif case == "cross_kind_reuse":
        same_workspace: Workspace = Workspace(id="same", label="workspace")
        same_project: Project = Project(id="same", workspace_id="same", label="project")
        same_task: Task = Task(id="same", workspace_id="same", project=InTaskProject("same"), title="task", state=TaskReady)
        same_run: Run = Run(id="same", task_id="same", state=RunPending)
        workspaces = [same_workspace]
        projects = [same_project]
        tasks = [same_task]
        runs = [same_run]
    return Catalog(workspaces=workspaces, projects=projects, tasks=tasks, runs=runs)

def scenario_validation(case: str) -> str:
    match validate_catalog(scenario_catalog(case)):
        case CatalogValid():
            return "valid"
        case CatalogInvalid(reason):
            return reason

def scenario_scope(case: str, run_id: str) -> str:
    match run_workspace(scenario_catalog(case), run_id):
        case RunWorkspace(workspace_id):
            return workspace_id
        case RunScopeInvalid(reason):
            return reason

def scenario_resource(left_key: str, left_write: bool, right_key: str, right_write: bool) -> str:
    match resources_conflict(left_key, left_write, right_key, right_write):
        case ResourceIndependent():
            return "independent"
        case ResourceConflict():
            return "conflict"
        case ResourceInvalid(reason):
            return reason

def scenario_task_pause() -> bool:
    match task_next_state(TaskActive, TaskRequestPause):
        case TaskAccepted(requested):
            match requested:
                case TaskPauseRequested():
                    match task_next_state(requested, TaskSafeBoundary):
                        case TaskAccepted(acknowledged):
                            match acknowledged:
                                case TaskPaused():
                                    match task_next_state(acknowledged, TaskResume):
                                        case TaskAccepted(resumed):
                                            match resumed:
                                                case TaskReady():
                                                    return True
                                                case _:
                                                    return False
                                        case _:
                                            return False
                                case _:
                                    return False
                        case _:
                            return False
                case _:
                    return False
        case _:
            return False

def scenario_task_distinctions() -> bool:
    blocked: bool = False
    review: bool = False
    match task_next_state(TaskActive, TaskBlock):
        case TaskAccepted(state):
            match state:
                case TaskBlocked():
                    blocked = True
                case _:
                    return False
        case _:
            return False
    match task_next_state(TaskActive, TaskRequestReview):
        case TaskAccepted(state):
            match state:
                case TaskAwaitingReview():
                    review = True
                case _:
                    return False
        case _:
            return False
    return blocked and review

def scenario_task_cancel() -> bool:
    match task_next_state(TaskPaused, TaskRequestCancel):
        case TaskAccepted(requested):
            match requested:
                case TaskCancelRequested():
                    match task_next_state(requested, TaskConfirmCancel):
                        case TaskAccepted(confirmed):
                            match confirmed:
                                case TaskCancelled():
                                    return True
                                case _:
                                    return False
                        case _:
                            return False
                case _:
                    return False
        case _:
            return False

def scenario_task_completion_refused() -> bool:
    states: list[TaskState] = [TaskReady, TaskActive, TaskPauseRequested, TaskPaused, TaskBlocked, TaskAwaitingReview, TaskCancelRequested]
    for state in states:
        match task_next_state(state, TaskAccept):
            case TaskRefused(reason):
                if reason != "acceptance_integration_required":
                    return False
            case _:
                return False
    return True

def scenario_terminal_tasks_immutable() -> bool:
    states: list[TaskState] = [TaskCompleted, TaskCancelled]
    events: list[TaskEvent] = [TaskStart, TaskRequestPause, TaskSafeBoundary, TaskResume, TaskBlock, TaskRequestReview, TaskAccept, TaskRequestCancel, TaskConfirmCancel]
    for state in states:
        for event in events:
            match task_next_state(state, event):
                case TaskRefused(reason):
                    if reason != "terminal_task":
                        return False
                case _:
                    return False
    return True

def scenario_terminal_runs_immutable() -> bool:
    states: list[RunState] = [RunSucceeded, RunFailed, RunCancelled]
    events: list[RunEvent] = [RunStart, RunRequestPause, RunSafeBoundary, RunResume, RunSucceed, RunFail, RunRequestCancel, RunConfirmCancel]
    for state in states:
        for event in events:
            match run_next_state(state, event):
                case RunRefused(reason):
                    if reason != "terminal_run":
                        return False
                case _:
                    return False
    return True

def scenario_run_pause() -> bool:
    match run_next_state(RunRunning, RunRequestPause):
        case RunAccepted(requested):
            match requested:
                case RunPausePending():
                    match run_next_state(requested, RunSafeBoundary):
                        case RunAccepted(acknowledged):
                            match acknowledged:
                                case RunSuspended():
                                    match run_next_state(acknowledged, RunResume):
                                        case RunAccepted(resumed):
                                            match resumed:
                                                case RunPending():
                                                    return True
                                                case _:
                                                    return False
                                        case _:
                                            return False
                                case _:
                                    return False
                        case _:
                            return False
                case _:
                    return False
        case _:
            return False

def scenario_failure_keeps_task() -> bool:
    task: Task = Task(id="task", workspace_id="workspace", project=NoTaskProject, title="work", state=TaskActive)
    match run_next_state(RunRunning, RunFail):
        case RunAccepted(state):
            match state:
                case RunFailed():
                    match task.state:
                        case TaskActive():
                            return True
                        case _:
                            return False
                case _:
                    return False
        case _:
            return False

def scenario_unrequested_acknowledgments_refused() -> bool:
    match task_next_state(TaskActive, TaskSafeBoundary):
        case TaskRefused(reason):
            if reason != "invalid_task_transition":
                return False
        case _:
            return False

    match run_next_state(RunRunning, RunSafeBoundary):
        case RunRefused(reason):
            if reason != "invalid_run_transition":
                return False
        case _:
            return False
    match task_next_state(TaskActive, TaskConfirmCancel):
        case TaskRefused(reason):
            if reason != "invalid_task_transition":
                return False
        case _:
            return False
    match run_next_state(RunRunning, RunConfirmCancel):
        case RunRefused(reason):
            return reason == "invalid_run_transition"
        case _:
            return False

def scenario_no_task_event_completes() -> bool:
    states: list[TaskState] = [TaskReady, TaskActive, TaskPauseRequested, TaskPaused, TaskBlocked, TaskAwaitingReview, TaskCompleted, TaskCancelRequested, TaskCancelled]
    events: list[TaskEvent] = [TaskStart, TaskRequestPause, TaskSafeBoundary, TaskResume, TaskBlock, TaskRequestReview, TaskAccept, TaskRequestCancel, TaskConfirmCancel]
    count: int = 0
    for state in states:
        for event in events:
            match task_next_state(state, event):
                case TaskAccepted(next_state):
                    match next_state:
                        case TaskCompleted():
                            return False
                        case _:
                            count = count + 1
                case TaskRefused(reason):
                    count = count + 1
    return count == 81

def scenario_run_outcome(request: str, outcome: str) -> str:
    current: RunState = RunRunning
    if request == "pause":
        current = RunPausePending
    elif request == "cancel":
        current = RunCancelPending
    event: RunEvent = RunSucceed
    if outcome == "fail":
        event = RunFail
    elif outcome == "cancel":
        event = RunConfirmCancel
    match run_next_state(current, event):
        case RunAccepted(next_state):
            match next_state:
                case RunSucceeded():
                    return "succeeded"
                case RunFailed():
                    return "failed"
                case RunCancelled():
                    return "cancelled"
                case _:
                    return "unexpected"
        case RunRefused(reason):
            return reason
