# Generic record backend — #120 checkpoints 1–2

The implementation in `src/memory_graph/records/` reuses the existing
`MemoryGraph` Ladybug database owner and write lock. It contains **no task states,
workspace policy, resource policy, Python workflows or queue**. Those belong in
Python-frontend Adapsis modules. No new database/dependency, native task tool or live
startup/schema activation is added. Only `cfg(test)` code can initialize the two
tables or mint a binding/store. Checkpoint 2 adds a fail-closed generic coroutine
bridge; see [task-durable.md](task-durable.md). Production provisioning stays absent.

## Record/envelope contract (provisional, not an exposed API)

The backend Rust methods are `read(binding, turn, request)`,
`commit(binding, turn, request_id, request)` and
`receipt(binding, turn, request_id)`. They return typed Rust records/receipts or
categorical `StoreError`s. The generic IO adapter preserves these categories as
direct builtin `Result<String>` values; no storage access exists without a binding.

Read request: `{"version":1,"keys":["record-id"]}`. Returns one optional row per
requested key in request order, from one read-only transaction. Absence is explicit.
Keys are exact opaque strings, not paths; there is no prefix scan or raw query.

Commit request example:

```json
{
  "version": 1,
  "conditions": [
    {"key": "record-a", "expected": {"kind": "revision", "revision": "1"}},
    {"key": "record-b", "expected": {"kind": "absent"}}
  ],
  "puts": [
    {"key": "record-a", "fields": [{"name": "label", "value": "updated"}]},
    {"key": "record-b", "fields": [{"name": "label", "value": "new"}]}
  ]
}
```

Each record is a bounded unique field-name/String-value map represented by a field
array. It is validated before storage; this is not a whole opaque application JSON
blob or a task-schema validator. Python codecs must validate domain schemas.
Duplicate keys/fields, unknown fields (including authority arguments), unsupported
version, non-String values, empty keys/field names and malformed revisions fail.
Every put requires exactly one matching condition. Check-only conditions are
allowed. Conditions and puts are nonempty. There is no delete/reset operation.

Rows start at revision `"1"`; each successful write increments the signed-i64
revision, failing before commit on overflow. Tokens are canonical positive decimal
strings, compared exactly. No deletion/recreation means no revision-reset ABA in
this surface. Any failed condition aborts the entire batch before writes.

### Bounds enforced before acquiring the database lock

| Input | Maximum |
|---|---:|
| Encoded read/commit request | 128 KiB |
| Read keys / conditions | 32 |
| Puts | 16 |
| Key / request ID | 128 UTF-8 bytes, nonempty |
| Fields per row | 32, nonempty |
| Field name | 64 UTF-8 bytes, nonempty |
| Field value | 4096 UTF-8 bytes (empty allowed) |
| Canonical encoded row field array | 16 KiB |

Validation/canonical hashing runs outside the lock. Each row is a separate keyed
database node; JSON is only the bounded field-array encoding. Reads address keys
by the database primary key, not an unbounded scan. At most 32 × 16 KiB of stored
field data is decoded per read, plus bounded record metadata.

## Bound authority and receipt isolation

`RecordBinding` has private, non-serializable fields and only a test provisioning
constructor. It binds the exact store instance, namespace, principal, context,
grant ID and read/write permission. `TurnIdentity` is server-supplied, never read
from request JSON. Missing binding/turn, different store/principal/context or any
guest profile is denied. Writes **and receipt lookups** additionally require both
binding write permission and `turn.may_write`. Read permission alone cannot inspect
or replay a commit receipt. No workspace/context string automatically creates a
grant. Authorization executes before request parsing or database access.

Receipt identity is SHA-256 of length-prefixed namespace, principal and request
ID; stored namespace/principal/context/grant ID must match on every lookup/replay.
A trusted grant to another namespace sees only that namespace. Another principal
with a separately provisioned grant cannot inspect/replay the first principal's
receipt. Changed context/grant generation is denied, even for the same principal.
Reopening requires trusted reprovisioning; bindings are not deserialized authority.
Revocation/provisioning for real callers is a later reviewed integration.

## Atomicity, idempotency and failure boundaries

One existing DB write lock and one Ladybug transaction cover receipt check, all
conditions, all row writes and the success receipt. The lock is never held over
model calls, external tools or application callbacks. This serializes database
commits, not task execution or entire projects. Independent keys do not create
version conflicts; simultaneous writers to one expected version have one winner.

Canonical hashing includes schema version, every condition and all puts/fields;
unordered keys/fields sort before hashing. Exact semantic retry with the same
request ID returns the original receipt without changing revisions, even if its
original conditions are now stale. Changed batch under the same ID returns
`IdempotencyMismatch`. Conflicts/invalid requests create no successful receipt and
are not successful commits. A new attempt after a known conflict must reread,
revalidate policy and use a new request ID. Receipts have no automatic GC here.

Errors: `Denied`, `InvalidRequest`, `Conflict`, `IdempotencyMismatch`, `Unavailable`,
`OutcomeUnknown`. Display text is categorical; raw query/record/error text is not
returned or logged. Precommit errors roll back; rollback failure is conservatively
unknown. Once COMMIT is attempted, COMMIT or CHECKPOINT failure is
`OutcomeUnknown`, not a declaration that writes did not happen. Reconcile through
the same receipt or exact request retry; never generate a new ID automatically.
The existing memory lifecycle helper is unchanged.

Lock acquisition uses a 5-second try-lock wait budget; transactions have a
5-second between-query work budget and each query a 2-second Ladybug query timeout.
COMMIT/CHECKPOINT and rollback still run to finish/reconcile the transaction.
These are waiting/query/work budgets, **not a hard OS I/O deadline**, and timing
out does not prove a commit could not occur. They are not task runtime limits.

## Measured checkpoint evidence and limits

`cargo test --release memory_graph::records` exercises real disk-backed temporary
Ladybug databases with checksums and throw-on-WAL-replay-failure enabled (matching
production's safety flags), bounded 64 MiB buffer pools and two DB threads:

- Strict envelope/canonical hashing and size/operation bounds; invalid requests
  reject before attempting a deliberately held database lock.
- Missing/mismatched/guest/read-only authority; namespace/principal/grant isolation.
- Concurrent attempts synchronized after reads: disjoint rows both commit;
  competing resource-like keys have one winner and no partial losing rows/receipt.
- Stale/check-only conditions; injected failure after row/receipt writes but before
  COMMIT rolls the entire transaction back.
- Close/reopen preserves rows/revisions and receipts; duplicate retry does not
  increment versions. A post-COMMIT injected lost acknowledgment returns unknown
  and is reconciled by receipt/exact retry. This is **not a process-crash test**.
- Maximum-count puts with maximum-size values complete and release the write lock
  within a practical 15-second test assertion, not a production latency guarantee.

No kill-after-COMMIT subprocess gate, power-loss/fsync certification, multiprocess
writer test, live user database test, effect replay safety or task pause/executor
integration is claimed by these backend tests. The separate IO/Python bridge tests
are documented in task-durable.md; no schedule, roles or GitHub integration is bundled.
