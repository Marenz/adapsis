# Record IO / Python durable task bridge — #120 checkpoint 2

This checkpoint connects real Python-frontend execution to the existing generic
Ladybug record backend. **Production transport binding remains disabled.** Its
executable fixtures attach test-only caller grants; ordinary conversation, CLI,
startup and route handles have no record binding and fail closed. No live module
installation, native task tool, workspace membership rule or automatic task tracking
is introduced. Task policy is Python, not a Rust task manager. The later
[operator provisioning checkpoint](record-grants.md) can prepare generic schema
and registry entries, but production caller attestation is still absent; it does
not activate this module or grant live turns access.

## Generic IO ABI

All three operations are direct IO builtins with exact String arguments and a
catchable `Result<String>` result. They take no namespace, principal or path:

| Operation | Arguments |
|---|---|
| `records_read` | strict v1 read request String |
| `records_commit` | request ID String, strict v1 commit batch String |
| `records_receipt` | request ID String |

See [record-store.md](record-store.md) for backend bounds and CAS/receipt semantics.
Success responses are canonical runtime-generated JSON:

```json
{"format":"records_v1","records":[null]}
{"format":"records_v1","records":[{"key":"a","revision":"2","fields":[{"name":"value","value":"text"}]}]}
{"format":"records_v1","receipt":null}
{"format":"records_v1","receipt":{"version":1,"request_id":"id","revisions":[{"key":"a","revision":"2"}]}}
```

Null is explicit absence, not an error or a successful commit. Commit never
returns an absent receipt. Error text is closed categorical backend text:
`record_access_denied`, `invalid_record_request`, `record_version_conflict`,
`record_request_id_mismatch`, `record_store_unavailable`,
`record_commit_outcome_unknown`. No raw DB error text, keys or data is logged.
Request/access Debug output is redacted; status uses the existing payload-free
Running category. Write-side channel loss, worker panic or response-serialization
failure is conservatively outcome-unknown, not a rollback claim. There are no
implicit retries or fresh request IDs.

`CoroutineHandle` optionally carries private store/binding access, installed only
by a cfg(test) helper. Clones preserve that exact access; changing the bound turn
does not create compatible rights. Both coroutine admission and the backend check
identity and write rights. Direct `IoRequest` dispatch still hits backend checks.
Receipt lookup requires write authority, not only read. Guest admission remains
at the existing outer capability ceiling. IO mocks cannot fake a record grant.
Database calls use `spawn_blocking`, launched off the runtime's IO event loop.

## Authored modules and bounded workflow

- `examples/tasks/domain.py`: existing pure TaskDomain states/links/policy.
- `examples/tasks/codec.py`: TaskRecordCodec, bounded trusted-envelope parsing and
  escaped field-map/batch encoding.
- `examples/tasks/durable.py`: TaskDurable, initialization, exclusive write-claim
  acquisition, progress, pause request, explicit safe-checkpoint acknowledgment,
  and receipt reconciliation.

`initialize(workspace_id, project_id, task_id, run_id, title, request_id)` creates
a ready task and pending run, plus missing workspace/project metadata rows, in
one batch. Empty project ID means no project. Existing workspace/project records
must match their schema and scope. Supplied IDs are stable strings, not generated
UUIDs or authorities. This slice initializes one run with a new task; adding
further attempts to an existing task is not yet a workflow.

`act(task_id, run_id, resource_id, action, progress, request_id)` supports:

- `acquire`: ready/pending -> active/running with an exclusive write claim;
- `progress`: update the running run's String progress marker;
- `pause`: active/running -> pause-requested/pause-pending;
- `checkpoint`: only those pending states -> paused/suspended, releasing the
  exact matching task/run/resource claim in the same batch.

TaskDomain transition functions decide start/pause/checkpoint states. A checkpoint
is a caller's explicit safe-boundary claim, **not proof that arbitrary external
work or delegated agents stopped**. A failed/stale/wrong holder cannot release
another task/run's resource through this module. All record revisions, including
referenced workspace/project metadata, are preconditions of the same commit.
Scope metadata is reread and validated on actions; cross-workspace project links
are rejected. Namespace binding is the authorization boundary, not these IDs.
Read-only resource claims, execution scheduling, new attempts, cancellation
orchestration, completion/evidence acceptance and external effects are deferred.

Direct builtin awaits retain Result, then async callers return application unions:
DurableCommitted, DurableReceipt, DurableAbsent, DurableConflict, DurableDenied,
DurableInvalid, DurableUnavailable, DurableOutcomeUnknown, DurableRequestMismatch.
There are no awaited Result-returning user functions and no relaxation of #82.
Only actual IO consumers produce successful commit/receipt reports. Pure planning
and codec functions do not establish authority or prove a database commit.
Application outcomes themselves are data, never unforgeable authorization tokens.

An exact primitive request replay returns its stored receipt without changing any
row, even after another holder acquires the resource. A stale batch with a new ID
conflicts. Re-running the high-level `act` method is not an automatic replay: it
re-reads current state and may refuse a now-inapplicable transition. Use
`reconcile(request_id)` for uncertainty; never assume a new ID is safe.

## Trusted codec boundary and frontend limitations

The decoder consumes canonical runtime-generated `records_v1` responses. It checks
the required format marker, keys, types/counts, positive revision spelling, bounded
field names/values and field-array uniqueness. Domain consumers check exact expected
field-name sets. Every `json_get` follows `json_kind`: only JSON Strings and arrays
are extracted (arrays have the documented serialized-String result), never numeric
or Bool scalars. Encoding quotes every supplied String using `json_escape`.

Current JSON primitives cannot enumerate object members or detect duplicate JSON
object members after serde parsing. Extra/duplicate envelope object keys are not
independently rejected by this pure decoder. It is **not an arbitrary external
JSON schema validator**, and a forged success-shaped object can pass the pure shape
check without changing storage or acquiring authority. Tests prove the backend
still refuses forged/malformed commit input and unbound callers. A future reusable
JSON introspection/strict-object primitive is a separate capability gap.

Two implementation pitfalls found by executable gates:

1. `regex_match` is **pattern first, text second**. Reversed arguments accidentally
   accepted revision `"1"` but rejected `"2"`, incorrectly classifying real commits
   as uncertain. Fixed in the module, not the runtime. Regression checks cover
   revisions 1/2/3/20/128, invalid spellings and exact revision-2 receipt replay.
2. Whole-module Python printing can reclassify an explicit local annotation naming
   another module's union when its internal fresh validator lacks that declaration
   (#86/#95 class). Repro: ChoiceOwner declares Selection with EmptySelection and
   NamedSelection(str); ChoiceUser.example declares
   `choice: Selection = EmptySelection` and returns a String. Function printing
   succeeds; module printing refuses an AST change. The test characterizes this
   limitation. TaskDurable instead constructs its typed Task directly with the
   project union in the field. This is new authored source with exact AST gates,
   not a claim of equivalence to a frozen original. No printer gate was relaxed.

## Executable evidence

Focused gates: `cargo test --release task_durable`, `records_bridge`,
`memory_graph::records`, `task_domain`, plus existing native/scoped guards.
Tests read the authored `.py` fixture files from this checkout, lower all three
modules through the real frontend, require exact full-program AST equality and
second-print byte identity, and use normal `eval_async_function` with real
CoroutineHandles/Runtime::handle_io against disk-backed temporary Ladybug stores.

Concurrent tests delay two real read replies only **after** database reads finish
and release locks; no DB responses are fabricated. They check independent progress,
one winner/no partial loser rows for a shared claim, pause A while B progresses,
wrong/stale holder refusal, old-release replay after a new holder, scope tampering,
escaped IDs/text, and close/reopen state+receipt recovery. Lost-ACK tests execute the
real commit then drop its reply: the module reports Unknown and reconciliation
finds the receipt without replaying the transition.

No subprocess crash/power-loss test or external-effect exactly-once/replay safety
claim follows. No live bot, live database, provider, deployment, scheduler, group
permission change or GitHub synchronization participates.
