# Explicit record grants and the remaining ingress boundary

This checkpoint adds a **general operator provisioning CLI and durable grant
registry**, not a Rust task manager. No task/index/workspace rows are created by
provisioning. Python remains responsible for all task domain policy.

**Record access is still unavailable to ordinary live turns.** A grant file,
`TurnIdentity`, `may_write`, a model/profile name or Adapsis `source_metadata` is
not proof of caller identity. The binding factory additionally requires a sealed
`TrustedCaller`, whose only issuer in this checkpoint is `cfg(test)`. There is no
production issuer, new authenticated transport, service `--record-grants` activation,
or live handle attachment. Existing no-record turn construction is unchanged.

## Operator-only provisioning

```sh
adapsis records-provision --database /explicit/existing/memory.lbug --grants /explicit/grants.toml --dry-run
adapsis records-provision --database /explicit/existing/memory.lbug --grants /explicit/grants.toml
```

There is no default database path or environment fallback, and a nonexistent
database is refused without creating anything. Dry-run opens Ladybug read-only,
not through `MemoryGraph::open` (which initializes memory schema). It validates
the entire proposed change and reports counts/migration need without DDL, COMMIT,
CHECKPOINT or grant writes. Real CLI subprocess tests compare all fixture-directory
file bytes and verify the registry schema remains absent.

Actual provisioning expects the database owner to be stopped: it respects Ladybug
file locking rather than bypassing the running process. Before applying to a real
node, the operator should stop/checkpoint it and retain a consistent backup of its
database and associated files plus the grant config. Run dry-run against the exact
target, then apply. Provisioning adds the generic record/receipt tables and the
version-1 `RecordGrantRegistry` table, with one bounded typed registry row. Existing
Memory nodes are not changed; a disk-backed sentinel test verifies this. All grant
changes are validated before writing, and schema/registry installation is one DB
transaction. COMMIT/CHECKPOINT ambiguity remains outcome-unknown.

On failure, do not assume rollback after a commit attempt. Reconcile using a fresh
read-only dry-run/inspection before retrying. Restore a backup only as an explicit
whole-database recovery decision (it also rolls back memory data); do not casually
restore old grant floors while retaining newer runtime/data state. Disabling the
record feature or reverting the binary requires no deletion of record/grant data.

The JSON report's `active_grants` means enabled registry entries, **not working
live caller bindings**. Operator provisioning alone cannot cross the missing
authenticated-ingress boundary.

## Strict configuration

```toml
version = 1
[[grants]]
id = "personal-store-grant"
generation = 1
namespace = "opaque-namespace-id"
principal = "server-principal-id"
context = "exact-private-context-id"
permission_group = "personal_tasks"
rights = ["read", "write"]
```

Names are exact strings, never wildcards or caller-selected scope. No default or
`main` context is permitted. Initial intended deployment is one explicitly
provisioned owner's personal context; this generic schema does not invent group
membership or automatically expose a professional/shared namespace. Two grants
for the same principal/context are ambiguous and rejected.

Limits: regular UTF-8 config file ≤64 KiB, ≤32 active grants, ≤256 historical grant
IDs including revocation tombstones, identity/group strings ≤128 bytes, positive
signed-i64 generation. Unknown fields, duplicate/invalid rights and malformed
config are rejected. Rights must include read, optionally write. The registry
stores a canonical semantic SHA-256 digest, not an mtime-only cache. Harmless TOML
format/order changes do not change the semantic configuration.

Grant ID is stable. Namespace/principal/context identity is immutable under that
ID; changing identity needs a new ID. Rights/group changes require a higher
generation. Same-generation changed rights, decreasing generations, and restoring
a revoked generation are refused. Missing grant entries leave durable tombstones;
there is no automatic history GC. Restoring access requires a higher generation.

At binding and backend read/commit/receipt admission, the full configured file is
read and validated under the existing DB write lock. Invalid/missing files fail
closed and revoke previously enabled registry entries when observed by a trusted
access attempt. Repair then needs fresh generations; use atomic file replacement
to avoid a partially written configuration being observed. This is access-time
refresh, not a background watcher or retroactive cancellation of completed reads.

Refresh and record transactions use the same lock. A commit already admitted
under that lock may complete before a queued revocation; after revocation, old
epoch handles/clones fail. Tests pause a real commit after registry authorization,
queue revocation, and verify precisely this ordering. No retroactive undo is claimed.

Receipts remain bound to stable grant ID, namespace, principal and context; their
identity does not include generation. A currently authorized writer under a newer
generation can reconcile its own old receipt. Revoked old handles cannot, and a
read-only grant cannot inspect receipts. Changing identity/grant ID does not
transfer historical receipt rights.

## Permission ceiling is separate from authentication

`RecordCeiling` snapshots the actual generating model plus context and principal
profiles and the process cap, using existing `PermissionConfig::resolve_group`.
The named permission group must exist. Guest profiles, missing actual model and
untrusted turn absence deny. A sealed caller's original ceiling is intersected
again with the current executor ceiling, so mid-turn model/profile changes cannot
widen it. Tests cover both actual-model-deny/context-allow and the converse,
principal narrowing, read/write/none, process caps and forged owner metadata.

The API is deliberately not wired from `AppConfig.turn`: today's takeover
`source_metadata` is supplied as an Adapsis-callable String argument in
`coroutine/mod.rs`, and `api/llm_handlers.rs` derives its speaker from that metadata.
The temporary takeover config also uses an effective profile instead of necessarily
the actual generating model. Neither is promoted into an identity attestation by
this change. Existing legacy behavior is preserved rather than silently rewritten.

## Required next trusted path (not implemented)

The smallest acceptable future bridge is a sealed in-process caller value minted
at a genuinely authenticated transport boundary, before any Adapsis invocation:
for example, #70's authenticated session resolves a principal and exact server
conversation, then captures the actual model/permission snapshot and forwards that
value through the invocation. Application arguments cannot supply or replace it.
An operator-controlled capability is another possible boundary but needs a separately
reviewed issuer/audience/revocation design; a caller `trusted=true` field is not one.

Current limitations remain explicit: #108's unauthenticated Telegram webhook is
not verified ingress; #70's transport/account authentication is separate work;
#79's trusted module/transitive-call policy does not make a module name or arbitrary
metadata an authenticated principal. No Telegram rewrite or new auth scheme is
bundled here. The personal ledger and production handle attachment remain blocked
until one genuine trusted path is reviewed.

Focused gates: `cargo test --release record_grants` (registry/authority tests plus
real CLI dry-run subprocess tests), existing `memory_graph::records`,
`records_bridge`, `task_durable`, and native/scoped guards. All fixtures use explicit
temporary databases; none opens the user's live database or installs a service.
