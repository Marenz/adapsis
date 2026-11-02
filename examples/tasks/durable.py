"""Task workflows over generic record IO. Binding is test-only; no live task-storage capability."""

@union
class DurableOutcome:
    DurableCommitted: tuple[str]
    DurableReceipt: tuple[str]
    DurableAbsent: tuple[()]
    DurableConflict: tuple[()]
    DurableDenied: tuple[()]
    DurableInvalid: tuple[str]
    DurableUnavailable: tuple[()]
    DurableOutcomeUnknown: tuple[()]
    DurableRequestMismatch: tuple[()]

@union
class DurablePlan:
    DurablePlanReady: tuple[str]
    DurablePlanRefused: tuple[str]

def failure(reason: str) -> DurableOutcome:
    if reason == "record_version_conflict":
        return DurableConflict
    if reason == "record_access_denied":
        return DurableDenied
    if reason == "record_commit_outcome_unknown":
        return DurableOutcomeUnknown
    if reason == "record_request_id_mismatch":
        return DurableRequestMismatch
    if reason == "invalid_record_request":
        return DurableInvalid("invalid_record_request")
    return DurableUnavailable

def make_row(key: str, names: list[str], values: list[str]) -> RecordRow:
    fields: list[RecordField] = []
    index: int = 0
    for name in names:
        value: str = get(values, index)
        item: RecordField = RecordField(name=name, value=value)
        fields = push(fields, item)
        index = index + 1
    return RecordRow(key=key, revision="", exists=False, fields=fields)

def init_plan(rows: list[RecordRow], workspace_id: str, project_id: str, task_id: str, run_id: str, title: str) -> DurablePlan:
    workspace: Workspace = Workspace(id=workspace_id, label="")
    projects: list[Project] = []
    task: Task = Task(id=task_id, workspace_id=workspace_id, project=NoTaskProject, title=title, state=TaskReady)
    if project_id != "":
        task = Task(id=task_id, workspace_id=workspace_id, project=InTaskProject(project_id), title=title, state=TaskReady)
        project_record: Project = Project(id=project_id, workspace_id=workspace_id, label="")
        projects = [project_record]
    run: Run = Run(id=run_id, task_id=task_id, state=RunPending)
    catalog: Catalog = Catalog(workspaces=[workspace], projects=projects, tasks=[task], runs=[run])
    match TaskDomain.validate_catalog(catalog):
        case CatalogInvalid(reason):
            return DurablePlanRefused(reason)
        case CatalogValid():
            return init_rows(rows, workspace_id, project_id, task_id, run_id, title)

def init_rows(rows: list[RecordRow], workspace_id: str, project_id: str, task_id: str, run_id: str, title: str) -> DurablePlan:
    workspace: RecordRow = get(rows, 0)
    task: RecordRow = get(rows, 1)
    run: RecordRow = get(rows, 2)
    if task.exists or run.exists:
        return DurablePlanRefused("already_exists")
    task = make_row(task.key, ["id", "workspace", "project", "title", "state"], [task_id, workspace_id, project_id, title, "ready"])
    run = make_row(run.key, ["id", "task", "state", "resource", "progress"], [run_id, task_id, "pending", "", ""])
    puts: list[RecordRow] = [task, run]
    if workspace.exists:
        if not TaskRecordCodec.fields_are(workspace, ["id"]):
            return DurablePlanRefused("invalid_workspace")
        if TaskRecordCodec.field(workspace, "id") != workspace_id:
            return DurablePlanRefused("invalid_workspace")
    else:
        workspace = make_row(workspace.key, ["id"], [workspace_id])
        puts = push(puts, workspace)
    if project_id != "":
        project: RecordRow = get(rows, 3)
        if project.exists:
            if not TaskRecordCodec.fields_are(project, ["id", "workspace"]):
                return DurablePlanRefused("invalid_project")
            if TaskRecordCodec.field(project, "id") != project_id:
                return DurablePlanRefused("invalid_project")
            if TaskRecordCodec.field(project, "workspace") != workspace_id:
                return DurablePlanRefused("cross_workspace_project")
        else:
            project = make_row(project.key, ["id", "workspace"], [project_id, workspace_id])
            puts = push(puts, project)
    return DurablePlanReady(TaskRecordCodec.batch(rows, puts))

@effects("io")
async def initialize(workspace_id: str, project_id: str, task_id: str, run_id: str, title: str, request_id: str) -> DurableOutcome:
    keys: list[str] = ["workspace:" + workspace_id, "task:" + task_id, "run:" + run_id]
    if project_id != "":
        keys = push(keys, "project:" + project_id)
    response: Result[str] = await records_read(TaskRecordCodec.read_request(keys))
    match response:
        case Err(reason):
            return failure(reason)
        case Ok(body):
            match TaskRecordCodec.decode(body, keys):
                case RecordsMalformed():
                    return DurableInvalid("invalid_records_response")
                case RecordsDecoded(rows):
                    match init_plan(rows, workspace_id, project_id, task_id, run_id, title):
                        case DurablePlanRefused(reason):
                            return DurableInvalid(reason)
                        case DurablePlanReady(batch):
                            result: Result[str] = await records_commit(request_id, batch)
                            match result:
                                case Err(reason):
                                    return failure(reason)
                                case Ok(committed_body):
                                    if TaskRecordCodec.receipt_matches(committed_body, request_id):
                                        return DurableCommitted(committed_body)
                                    return DurableOutcomeUnknown

def linked_rows(task: RecordRow, run: RecordRow, task_id: str, run_id: str) -> bool:
    if not TaskRecordCodec.fields_are(task, ["id", "workspace", "project", "title", "state"]):
        return False
    if not TaskRecordCodec.fields_are(run, ["id", "task", "state", "resource", "progress"]):
        return False
    if TaskRecordCodec.field(task, "id") != task_id:
        return False
    if TaskRecordCodec.field(task, "workspace") == "":
        return False
    if TaskRecordCodec.field(run, "id") != run_id:
        return False
    return TaskRecordCodec.field(run, "task") == task_id

def action_plan(rows: list[RecordRow], task_id: str, run_id: str, resource_id: str, action: str, progress: str) -> DurablePlan:
    task: RecordRow = get(rows, 0)
    run: RecordRow = get(rows, 1)
    claim: RecordRow = get(rows, 2)
    if not linked_rows(task, run, task_id, run_id):
        return DurablePlanRefused("invalid_task_run_link")
    if resource_id == "":
        return DurablePlanRefused("empty_resource_id")
    if claim.exists:
        if not TaskRecordCodec.fields_are(claim, ["resource", "task", "run"]):
            return DurablePlanRefused("invalid_claim")
        if TaskRecordCodec.field(claim, "resource") != resource_id:
            return DurablePlanRefused("invalid_claim")
    if action == "acquire":
        return acquire_plan(rows, task, run, claim, task_id, run_id, resource_id)
    if not claim.exists:
        return DurablePlanRefused("not_claim_holder")
    if TaskRecordCodec.field(claim, "task") != task_id:
        return DurablePlanRefused("not_claim_holder")
    if TaskRecordCodec.field(claim, "run") != run_id:
        return DurablePlanRefused("not_claim_holder")
    if TaskRecordCodec.field(run, "resource") != resource_id:
        return DurablePlanRefused("not_claim_holder")
    return held_plan(rows, task, run, claim, action, progress)

def acquire_plan(rows: list[RecordRow], task: RecordRow, run: RecordRow, claim: RecordRow, task_id: str, run_id: str, resource_id: str) -> DurablePlan:
    if TaskRecordCodec.field(task, "state") != "ready":
        return DurablePlanRefused("invalid_task_transition")
    if TaskRecordCodec.field(run, "state") != "pending":
        return DurablePlanRefused("invalid_run_transition")
    if TaskRecordCodec.field(run, "resource") != "":
        return DurablePlanRefused("run_already_has_resource")
    if claim.exists:
        holder_task: str = TaskRecordCodec.field(claim, "task")
        holder_run: str = TaskRecordCodec.field(claim, "run")
        if holder_task != "" or holder_run != "":
            return DurablePlanRefused("resource_busy")
    match TaskDomain.task_next_state(TaskReady, TaskStart):
        case TaskAccepted(next_task):
            match next_task:
                case TaskActive():
                    task = TaskRecordCodec.replaced(task, "state", "active")
                case _:
                    return DurablePlanRefused("invalid_task_transition")
        case _:
            return DurablePlanRefused("invalid_task_transition")
    match TaskDomain.run_next_state(RunPending, RunStart):
        case RunAccepted(next_run):
            match next_run:
                case RunRunning():
                    run = TaskRecordCodec.replaced(run, "state", "running")
                case _:
                    return DurablePlanRefused("invalid_run_transition")
        case _:
            return DurablePlanRefused("invalid_run_transition")
    run = TaskRecordCodec.replaced(run, "resource", resource_id)
    claim = make_row(claim.key, ["resource", "task", "run"], [resource_id, task_id, run_id])
    puts: list[RecordRow] = [task, run, claim]
    return DurablePlanReady(TaskRecordCodec.batch(rows, puts))

def held_plan(rows: list[RecordRow], task: RecordRow, run: RecordRow, claim: RecordRow, action: str, progress: str) -> DurablePlan:
    task_state: str = TaskRecordCodec.field(task, "state")
    run_state: str = TaskRecordCodec.field(run, "state")
    if action == "checkpoint":
        if task_state != "pause_requested" or run_state != "pause_pending":
            return DurablePlanRefused("pause_not_requested")
        match TaskDomain.task_next_state(TaskPauseRequested, TaskSafeBoundary):
            case TaskAccepted(paused_task):
                match paused_task:
                    case TaskPaused():
                        task = TaskRecordCodec.replaced(task, "state", "paused")
                    case _:
                        return DurablePlanRefused("invalid_task_transition")
            case _:
                return DurablePlanRefused("invalid_task_transition")
        match TaskDomain.run_next_state(RunPausePending, RunSafeBoundary):
            case RunAccepted(paused_run):
                match paused_run:
                    case RunSuspended():
                        run = TaskRecordCodec.replaced(run, "state", "suspended")
                    case _:
                        return DurablePlanRefused("invalid_run_transition")
            case _:
                return DurablePlanRefused("invalid_run_transition")
        run = TaskRecordCodec.replaced(run, "resource", "")
        claim = TaskRecordCodec.replaced(claim, "task", "")
        claim = TaskRecordCodec.replaced(claim, "run", "")
    elif action == "pause":
        if task_state != "active" or run_state != "running":
            return DurablePlanRefused("invalid_pause_transition")
        match TaskDomain.task_next_state(TaskActive, TaskRequestPause):
            case TaskAccepted(requested_task):
                match requested_task:
                    case TaskPauseRequested():
                        task = TaskRecordCodec.replaced(task, "state", "pause_requested")
                    case _:
                        return DurablePlanRefused("invalid_task_transition")
            case _:
                return DurablePlanRefused("invalid_task_transition")
        match TaskDomain.run_next_state(RunRunning, RunRequestPause):
            case RunAccepted(requested_run):
                match requested_run:
                    case RunPausePending():
                        run = TaskRecordCodec.replaced(run, "state", "pause_pending")
                    case _:
                        return DurablePlanRefused("invalid_run_transition")
            case _:
                return DurablePlanRefused("invalid_run_transition")
    elif action == "progress":
        if task_state != "active" or run_state != "running":
            return DurablePlanRefused("not_running")
        run = TaskRecordCodec.replaced(run, "progress", progress)
    else:
        return DurablePlanRefused("unknown_action")
    puts: list[RecordRow] = [task, run, claim]
    return DurablePlanReady(TaskRecordCodec.batch(rows, puts))

def scope_keys(rows: list[RecordRow]) -> list[str]:
    task: RecordRow = get(rows, 0)
    keys: list[str] = ["workspace:" + TaskRecordCodec.field(task, "workspace")]
    project_id: str = TaskRecordCodec.field(task, "project")
    if project_id != "":
        keys = push(keys, "project:" + project_id)
    return keys

def scoped_plan(rows: list[RecordRow], scopes: list[RecordRow], task_id: str, run_id: str, resource_id: str, action: str, progress: str) -> DurablePlan:
    task: RecordRow = get(rows, 0)
    workspace: RecordRow = get(scopes, 0)
    if not TaskRecordCodec.fields_are(workspace, ["id"]):
        return DurablePlanRefused("invalid_workspace")
    if TaskRecordCodec.field(workspace, "id") != TaskRecordCodec.field(task, "workspace"):
        return DurablePlanRefused("invalid_workspace")
    project_id: str = TaskRecordCodec.field(task, "project")
    if project_id != "":
        project: RecordRow = get(scopes, 1)
        if not TaskRecordCodec.fields_are(project, ["id", "workspace"]):
            return DurablePlanRefused("invalid_project")
        if TaskRecordCodec.field(project, "id") != project_id:
            return DurablePlanRefused("invalid_project")
        if TaskRecordCodec.field(project, "workspace") != TaskRecordCodec.field(task, "workspace"):
            return DurablePlanRefused("cross_workspace_project")
    conditions: list[RecordRow] = rows
    for scope_row in scopes:
        conditions = push(conditions, scope_row)
    return action_plan(conditions, task_id, run_id, resource_id, action, progress)

@effects("io")
async def act(task_id: str, run_id: str, resource_id: str, action: str, progress: str, request_id: str) -> DurableOutcome:
    keys: list[str] = ["task:" + task_id, "run:" + run_id, "claim:" + resource_id]
    response: Result[str] = await records_read(TaskRecordCodec.read_request(keys))
    match response:
        case Err(reason):
            return failure(reason)
        case Ok(body):
            match TaskRecordCodec.decode(body, keys):
                case RecordsMalformed():
                    return DurableInvalid("invalid_records_response")
                case RecordsDecoded(rows):
                    scopes_keys: list[str] = scope_keys(rows)
                    scope_response: Result[str] = await records_read(TaskRecordCodec.read_request(scopes_keys))
                    match scope_response:
                        case Err(reason):
                            return failure(reason)
                        case Ok(scope_body):
                            match TaskRecordCodec.decode(scope_body, scopes_keys):
                                case RecordsMalformed():
                                    return DurableInvalid("invalid_scope_response")
                                case RecordsDecoded(scopes):
                                    match scoped_plan(rows, scopes, task_id, run_id, resource_id, action, progress):
                                        case DurablePlanRefused(reason):
                                            return DurableInvalid(reason)
                                        case DurablePlanReady(batch):
                                            result: Result[str] = await records_commit(request_id, batch)
                                            match result:
                                                case Err(reason):
                                                    return failure(reason)
                                                case Ok(committed_body):
                                                    if TaskRecordCodec.receipt_matches(committed_body, request_id):
                                                        return DurableCommitted(committed_body)
                                                    return DurableOutcomeUnknown

@effects("io")
async def reconcile(request_id: str) -> DurableOutcome:
    result: Result[str] = await records_receipt(request_id)
    match result:
        case Err(reason):
            return failure(reason)
        case Ok(body):
            if not TaskRecordCodec.format_valid(body):
                return DurableUnavailable
            if json_kind(body, "receipt") == "null":
                return DurableAbsent
            if TaskRecordCodec.receipt_matches(body, request_id):
                return DurableReceipt(body)
            return DurableUnavailable
