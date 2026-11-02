# Task identity/scope foundation

This dormant, crate-private model is a structural checkpoint for #76, not a
task service, durable storage, authorization system or #89 protocol capability.

## Owner-agreed concepts

- Substantial requests should eventually be tracked automatically; quick
  interactions stay lightweight and conversational. Creation heuristics are unset.
- A persistent task is distinct from an agent attempt/run.
- Workspaces separate personal/professional information and support shared groups
  with specific people. Every task belongs to exactly one workspace.
- Projects are optional grouping within one workspace, not synonyms for repos.
  Group help/moderation remains conversational and needs no project.
- Repository resources do not imply access; organizational tags grant no authority.

## Provisional implementation representation

`src/task_foundation` holds version-1 standalone serialization: workspace
`{id,label}`, project `{id,workspace_id,label}`, task
`{id,workspace_id,project_id,title}`, run `{id,task_id}`. These are deliberately
incomplete records, not a settled owner/participant/provenance/lifecycle schema.
Personal/professional/group categories and memberships are not encoded yet.

IDs use canonical lowercase hyphenated RFC4122 UUIDv4 strings prefixed with
`ws_`, `project_`, `task_` or `run_`. Tags prevent cross-kind decoding; IDs encode
no person, path, title or authority. Rename/reorder/roundtrip preserve identity.
No missing ID is generated during decoding. Multiple runs retain one task ID.

An untrusted `FoundationSnapshot` becomes an immutable `FoundationCatalog` only
after whole-candidate validation. Unsupported versions, duplicate same-kind IDs,
dangling references and cross-workspace task/project links reject the candidate.
Unknown fields are rejected. A run's workspace is derived solely from its task.
Validation errors contain categories and repair guidance, not record titles.
Raw serializer errors are not a public API/logging contract.

Structural validity is **not access authorization**. No runtime consumer,
principal lookup, permission grant, storage loader/writer, API, native tool or
automation is provided. There is no action/admission path while policy remains
undecided; callers must not treat a future catalog lookup as an access check.

## Legacy migration boundary

`SessionMeta.plan`, `roadmap`, and `agent_log` remain untouched global vectors.
They do not establish reliable workspace/owner provenance. There is no default
personal or shared workspace, automatic conversion, conversation merge or
exposure of their contents through this model. Coroutine `TaskId` and existing
`AgentStatus` keep their separate meanings.

A future migration needs explicit reviewed destinations, authorized scope,
stable source-snapshot/item provenance and a durable mapping for idempotency.
Conflicts must be reported before import, not resolved by guessing membership.
Storage choice and migration implementation are deferred. #70 conversation/auth
work, #39 execution policies and #89 native parity remain separate gates.

## Remaining owner decisions and verification

Scope selection, automatic creation heuristics, roles/powers, memberships,
run grants/approvals, origin/owner provenance, lifecycle/delivery, moderation
actions and cross-workspace sharing are unsettled. None is implied by these fields.

Focused crate tests: `cargo test --release task_foundation`. Tests cover identity
roundtrip, rename/reorder, optional projects, multiple runs, duplicate and dangling
IDs, cross-workspace rejection, version/required-field checks and atomic validation.
The initial disk-efficient standalone harness compiles these actual source files
with repository-locked dependencies; it is not a full Adapsis integration build.
