# Python-authored task domain — #120 / #76

The owner authorized implementation through the **Python frontend**, with generic
runtime primitives where needed. The premature dormant Rust prototype is removed
from active source by a forward change; its history is retained. This checkpoint
is a pure executable domain module, not an installed task service.

## Owner-agreed behavior

- Track substantial requests automatically; quick interactions stay conversational.
  Follow-ups reuse tasks and growing requests can be promoted without ceremony.
- Workspaces separate personal/professional information and specifically shared
  groups. Projects are optional organization within one workspace. Repositories,
  services and documents are resources, not identities or access grants; tags
  likewise grant no authority. Group help/moderation does not require a project.
- A stable persistent task is distinct from its attempts/runs. Tasks and runs may
  proceed concurrently with isolated authority/context; conflicting resources
  queue, independent worktrees may write concurrently, no global project lock.
- Paused is distinct from blocked, awaiting review, failed runs and cancellation.
  Pause must reach a safe boundary and affect delegated work without stopping
  unrelated tasks. Project autonomy has task-specific overrides; it never grants
  new authority or silently crosses destructive/release/deployment boundaries.
- GitHub compatibility must preserve shared backlog ownership, source permissions
  and output audience; local completion must not automatically close issues.

These are goals, not claims that this pure checkpoint implements execution,
tracking, visibility, moderation, autonomy enforcement or persistence.

## Provisional internal module representation

`examples/tasks/domain.py` is authored Adapsis Python-subset code, not CPython.
`examples/` follows the existing executable Python examples; there was no existing
`modules/` source tree. It has no startup, route, IO, shared state or tool exposure.

| Record | Fields |
|---|---|
| Workspace | id, label |
| Project | id, workspace_id, label |
| Task | id, workspace_id, optional-project union, title, state |
| Run | id, task_id, state |
| Catalog | typed lists of the four records |

IDs are **supplied strings**, nonempty and unique within each record kind. The
same string may occur in different kinds; references select their record kind.
No ID generator, UUID contract, secrecy promise, authentication or database exists
here. Labels/reordering do not identify records. Run workspace is derived via its
task. `validate_catalog` rejects duplicate/empty IDs, missing references and
cross-workspace project links with `CatalogValid | CatalogInvalid(reason)`.
This is whole-value pure validation, **not a durable transaction**. Records remain
ordinary values; validation does not turn them into unforgeable authorized handles.

The principal APIs return explicit application unions, not builtin Result:

- `validate_catalog(Catalog) -> CatalogValidation`
- `run_workspace(Catalog, run_id) -> RunScope`
- `task_next_state(TaskState, TaskEvent) -> TaskTransition`
- `run_next_state(RunState, RunEvent) -> RunTransition`
- `resources_conflict(left_id, left_write, right_id, right_write) -> ResourceDecision`

Task states: ready, active, pause-requested, paused, blocked, awaiting-review,
completed (reserved), cancel-requested, cancelled. Run states: pending, running,
pause-pending, suspended, succeeded, failed, cancel-pending, cancelled. These
spellings and request bookkeeping are **provisional implementation choices**, not
a public API or previously agreed lifecycle schema.

Pause requests do not return paused: a separate safe-boundary event is required.
That event is only a caller claim here, not evidence of executor acknowledgment or
delegation propagation. Resume returns ready/pending for later admission. Pending
pause/cancel may race a truthful run success/failure; cancellation is terminal
only after confirmation. Terminal states reject all changes. Run failure does not
complete its task. Every task acceptance event is refused with
`acceptance_integration_required`; **no event enters completed** until evidence
and child-run integration exist. Task and run status are not coupled by a hidden
executor. Nonterminal helper functions are implementation decomposition, not
authorized external operations. All refusal reasons are categorical.

Resource comparisons are advisory: nonempty exact opaque IDs, read/read allowed,
any write conflicts only for the same resource. No path canonicalization or alias
resolution is claimed: `a/../b` and `b` are different supplied IDs. The function
does not acquire a lock, reserve resources, queue work or prevent a race. A future
trusted resource registry must resolve aliases before atomic claim admission.

## Verification and remaining evidence

`scenarios.py` contains test-only constructors/drivers, appended to the domain in
one isolated `TaskDomain` namespace. `test_scenarios.py` keeps supported scalar
test inputs/expectations; no test-data policy relaxation. The Rust cfg(test) harness
uses fixtures with explicitly typed record locals before list construction: direct
record literals inside lists lose the printer's struct context (existing G3).
No runtime/frontend relaxation is made. The harness
only loads the actual Python sources, checks exact full-module AST roundtrips
(ignoring typed NodeIds only), second-print stability, and evaluates all 47 cases
before and after printing through the real frontend/Adapsis test runtime:

```sh
cargo test --release task_domain
```

Tests cover optional projects, same-label workspaces, each invalid ID/link kind,
cross-workspace rejection, run-derived scope, rename/reorder/retry identity,
two-phase pause/cancel, terminal immutability, all 81 task state/event pairs
refusing entry into completed, run outcome races, and resource conflicts.
They do not prove live concurrency, authorization isolation, delegated pause,
restart safety or durable side-effect handling. No full-application/tier-parity
claim follows from the pure test runtime. The standalone CLI can also run a
temporary concatenation of domain.py and scenarios.py with:
`adapsis test COMBINED.py --language python --module TaskDomain --tests examples/tasks/test_scenarios.py`.

## Next reusable mechanisms and integration gaps

Checkpoint 1 implements the dormant general record/CAS mechanism
described in [record-store.md](record-store.md), with explicit test-only grants and
schema initialization. [Checkpoint 2](task-durable.md) adds fail-closed generic IO
and Python durable workflows tested against isolated databases, with production
bindings still absent. The remaining integration gaps below are not a task-specific
Rust manager authorization.

1. **Durable typed storage transactions / compare-and-write.** Proposed generic
   shape: read a scoped collection with version; atomically commit a bounded typed
   write set only if expected versions/absence checks still hold. Needed for task,
   run and resource-claim updates together. Include crash recovery, conflict result
   and idempotency receipts. No untyped JSON blob or file overwrite is a substitute.
   ID creation needs a generic opaque-ID primitive independently of labels.
2. **Bound identity and storage scope.** Server-observed principal/origin and an
   authorized namespace handle, not a caller-selected principal or an inferred
   permission from workspace grouping. Scope selection/membership and capability
   policy need separate approval before exposing any operation.
3. **Generic execution handles/control.** Isolated spawn, progress/outcome reporting,
   cooperative checkpoint and acknowledged pause/cancel with child execution
   tracking. Task transitions and conflict policy remain module logic.
4. **Durable timers and delivery.** Stable timer identity/due time, explicit
   skip/coalesce/catch-up policy; transactional outbox/idempotency acknowledgment
   before claiming restart-safe external effects. Sleep is not durable scheduling.

Storage technology and exact primitive APIs need separate reviewed slices.

## Migration and integration boundaries

Legacy `SessionMeta.plan`, `roadmap`, `agent_log`, Coroutine `TaskId` and
`AgentStatus` remain untouched. No automatic conversion/default workspace or
conversation merge. Later migration requires explicit authorized destinations,
source/item provenance and durable idempotent mappings; unknown scope is not
silently exposed. #70 identity work and #89 native parity remain separate gates.

GitHub references must remain distinct from task IDs and repository visibility
must not imply workspace membership. Tracker authority, explicit issue/PR links,
sync conflict/loop policy and audience mapping remain to be settled: no adapter,
automatic sync/closure or competing backlog is implemented. Automatic creation
heuristics, precise scope selection, memberships/roles, run grants/approvals,
moderation powers and cross-workspace sharing are likewise not filled in by this
representation.
