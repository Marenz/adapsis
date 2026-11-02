"""Pure task-domain policy prototype. No storage, executor, authority or tools."""

@union
class TaskProject:
    NoTaskProject: tuple[()]
    InTaskProject: tuple[str]

@union
class TaskState:
    TaskReady: tuple[()]
    TaskActive: tuple[()]
    TaskPauseRequested: tuple[()]
    TaskPaused: tuple[()]
    TaskBlocked: tuple[()]
    TaskAwaitingReview: tuple[()]
    TaskCompleted: tuple[()]
    TaskCancelRequested: tuple[()]
    TaskCancelled: tuple[()]

@union
class RunState:
    RunPending: tuple[()]
    RunRunning: tuple[()]
    RunPausePending: tuple[()]
    RunSuspended: tuple[()]
    RunSucceeded: tuple[()]
    RunFailed: tuple[()]
    RunCancelPending: tuple[()]
    RunCancelled: tuple[()]

@union
class TaskEvent:
    TaskStart: tuple[()]
    TaskRequestPause: tuple[()]
    TaskSafeBoundary: tuple[()]
    TaskResume: tuple[()]
    TaskBlock: tuple[()]
    TaskRequestReview: tuple[()]
    TaskAccept: tuple[()]
    TaskRequestCancel: tuple[()]
    TaskConfirmCancel: tuple[()]

@union
class RunEvent:
    RunStart: tuple[()]
    RunRequestPause: tuple[()]
    RunSafeBoundary: tuple[()]
    RunResume: tuple[()]
    RunSucceed: tuple[()]
    RunFail: tuple[()]
    RunRequestCancel: tuple[()]
    RunConfirmCancel: tuple[()]

@union
class TaskTransition:
    TaskAccepted: tuple[TaskState]
    TaskRefused: tuple[str]

@union
class RunTransition:
    RunAccepted: tuple[RunState]
    RunRefused: tuple[str]

@union
class CatalogValidation:
    CatalogValid: tuple[()]
    CatalogInvalid: tuple[str]

@union
class RunScope:
    RunWorkspace: tuple[str]
    RunScopeInvalid: tuple[str]

@union
class ResourceDecision:
    ResourceIndependent: tuple[()]
    ResourceConflict: tuple[()]
    ResourceInvalid: tuple[str]

@dataclass
class Workspace:
    id: str
    label: str

@dataclass
class Project:
    id: str
    workspace_id: str
    label: str

@dataclass
class Task:
    id: str
    workspace_id: str
    project: TaskProject
    title: str
    state: TaskState

@dataclass
class Run:
    id: str
    task_id: str
    state: RunState

@dataclass
class Catalog:
    workspaces: list[Workspace]
    projects: list[Project]
    tasks: list[Task]
    runs: list[Run]

def validate_workspaces(workspaces: list[Workspace]) -> CatalogValidation:
    for workspace in workspaces:
        if workspace.id == "":
            return CatalogInvalid("empty_workspace_id")
        count: int = 0
        for candidate in workspaces:
            if candidate.id == workspace.id:
                count = count + 1
        if count != 1:
            return CatalogInvalid("duplicate_workspace_id")
    return CatalogValid

def validate_projects(projects: list[Project], workspaces: list[Workspace]) -> CatalogValidation:
    for project in projects:
        if project.id == "":
            return CatalogInvalid("empty_project_id")
        count: int = 0
        for candidate in projects:
            if candidate.id == project.id:
                count = count + 1
        if count != 1:
            return CatalogInvalid("duplicate_project_id")
        found: bool = False
        for workspace in workspaces:
            if workspace.id == project.workspace_id:
                found = True
        if not found:
            return CatalogInvalid("missing_project_workspace")
    return CatalogValid

def validate_task_project(task: Task, projects: list[Project]) -> CatalogValidation:
    match task.project:
        case NoTaskProject():
            return CatalogValid
        case InTaskProject(project_id):
            for project in projects:
                if project.id == project_id:
                    if project.workspace_id != task.workspace_id:
                        return CatalogInvalid("cross_workspace_project")
                    return CatalogValid
            return CatalogInvalid("missing_task_project")

def validate_tasks(tasks: list[Task], workspaces: list[Workspace], projects: list[Project]) -> CatalogValidation:
    for task in tasks:
        if task.id == "":
            return CatalogInvalid("empty_task_id")
        count: int = 0
        for candidate in tasks:
            if candidate.id == task.id:
                count = count + 1
        if count != 1:
            return CatalogInvalid("duplicate_task_id")
        found: bool = False
        for workspace in workspaces:
            if workspace.id == task.workspace_id:
                found = True
        if not found:
            return CatalogInvalid("missing_task_workspace")
        checked: CatalogValidation = validate_task_project(task, projects)
        match checked:
            case CatalogValid():
                found = True
            case CatalogInvalid(reason):
                return CatalogInvalid(reason)
    return CatalogValid

def validate_runs(runs: list[Run], tasks: list[Task]) -> CatalogValidation:
    for run in runs:
        if run.id == "":
            return CatalogInvalid("empty_run_id")
        count: int = 0
        for candidate in runs:
            if candidate.id == run.id:
                count = count + 1
        if count != 1:
            return CatalogInvalid("duplicate_run_id")
        found: bool = False
        for task in tasks:
            if task.id == run.task_id:
                found = True
        if not found:
            return CatalogInvalid("missing_run_task")
    return CatalogValid

def validate_catalog(catalog: Catalog) -> CatalogValidation:
    match validate_workspaces(catalog.workspaces):
        case CatalogInvalid(reason):
            return CatalogInvalid(reason)
        case CatalogValid():
            match validate_projects(catalog.projects, catalog.workspaces):
                case CatalogInvalid(reason):
                    return CatalogInvalid(reason)
                case CatalogValid():
                    match validate_tasks(catalog.tasks, catalog.workspaces, catalog.projects):
                        case CatalogInvalid(reason):
                            return CatalogInvalid(reason)
                        case CatalogValid():
                            return validate_runs(catalog.runs, catalog.tasks)

def run_workspace(catalog: Catalog, run_id: str) -> RunScope:
    match validate_catalog(catalog):
        case CatalogInvalid(reason):
            return RunScopeInvalid(reason)
        case CatalogValid():
            for run in catalog.runs:
                if run.id == run_id:
                    for task in catalog.tasks:
                        if task.id == run.task_id:
                            return RunWorkspace(task.workspace_id)
            return RunScopeInvalid("missing_run")

def task_next_state(current: TaskState, event: TaskEvent) -> TaskTransition:
    match current:
        case TaskCompleted():
            return TaskRefused("terminal_task")
        case TaskCancelled():
            return TaskRefused("terminal_task")
        case _:
            match event:
                case TaskAccept():
                    return TaskRefused("acceptance_integration_required")
                case TaskRequestCancel():
                    return TaskAccepted(TaskCancelRequested)
                case _:
                    return task_nonterminal_transition(current, event)

def task_nonterminal_transition(current: TaskState, event: TaskEvent) -> TaskTransition:
    match current:
        case TaskReady():
            match event:
                case TaskStart():
                    return TaskAccepted(TaskActive)
                case TaskRequestPause():
                    return TaskAccepted(TaskPauseRequested)
                case _:
                    return TaskRefused("invalid_task_transition")
        case TaskActive():
            match event:
                case TaskRequestPause():
                    return TaskAccepted(TaskPauseRequested)
                case TaskBlock():
                    return TaskAccepted(TaskBlocked)
                case TaskRequestReview():
                    return TaskAccepted(TaskAwaitingReview)
                case _:
                    return TaskRefused("invalid_task_transition")
        case TaskPauseRequested():
            match event:
                case TaskSafeBoundary():
                    return TaskAccepted(TaskPaused)
                case _:
                    return TaskRefused("invalid_task_transition")
        case TaskPaused():
            match event:
                case TaskResume():
                    return TaskAccepted(TaskReady)
                case _:
                    return TaskRefused("invalid_task_transition")
        case TaskBlocked():
            match event:
                case TaskResume():
                    return TaskAccepted(TaskReady)
                case TaskRequestPause():
                    return TaskAccepted(TaskPauseRequested)
                case _:
                    return TaskRefused("invalid_task_transition")
        case TaskAwaitingReview():
            match event:
                case TaskResume():
                    return TaskAccepted(TaskReady)
                case TaskRequestPause():
                    return TaskAccepted(TaskPauseRequested)
                case _:
                    return TaskRefused("invalid_task_transition")
        case TaskCancelRequested():
            match event:
                case TaskConfirmCancel():
                    return TaskAccepted(TaskCancelled)
                case _:
                    return TaskRefused("invalid_task_transition")
        case _:
            return TaskRefused("terminal_task")

def run_next_state(current: RunState, event: RunEvent) -> RunTransition:
    match current:
        case RunSucceeded():
            return RunRefused("terminal_run")
        case RunFailed():
            return RunRefused("terminal_run")
        case RunCancelled():
            return RunRefused("terminal_run")
        case _:
            match event:
                case RunRequestCancel():
                    return RunAccepted(RunCancelPending)
                case _:
                    return run_nonterminal_transition(current, event)

def run_nonterminal_transition(current: RunState, event: RunEvent) -> RunTransition:
    match current:
        case RunPending():
            match event:
                case RunStart():
                    return RunAccepted(RunRunning)
                case RunRequestPause():
                    return RunAccepted(RunPausePending)
                case _:
                    return RunRefused("invalid_run_transition")
        case RunRunning():
            match event:
                case RunRequestPause():
                    return RunAccepted(RunPausePending)
                case RunSucceed():
                    return RunAccepted(RunSucceeded)
                case RunFail():
                    return RunAccepted(RunFailed)
                case _:
                    return RunRefused("invalid_run_transition")
        case RunPausePending():
            match event:
                case RunSafeBoundary():
                    return RunAccepted(RunSuspended)
                case RunSucceed():
                    return RunAccepted(RunSucceeded)
                case RunFail():
                    return RunAccepted(RunFailed)
                case _:
                    return RunRefused("invalid_run_transition")
        case RunSuspended():
            match event:
                case RunResume():
                    return RunAccepted(RunPending)
                case _:
                    return RunRefused("invalid_run_transition")
        case RunCancelPending():
            match event:
                case RunConfirmCancel():
                    return RunAccepted(RunCancelled)
                case RunSucceed():
                    return RunAccepted(RunSucceeded)
                case RunFail():
                    return RunAccepted(RunFailed)
                case _:
                    return RunRefused("invalid_run_transition")
        case _:
            return RunRefused("terminal_run")

def resources_conflict(left_key: str, left_write: bool, right_key: str, right_write: bool) -> ResourceDecision:
    if left_key == "" or right_key == "":
        return ResourceInvalid("empty_resource_id")
    if left_key == right_key:
        if left_write or right_write:
            return ResourceConflict
    return ResourceIndependent
