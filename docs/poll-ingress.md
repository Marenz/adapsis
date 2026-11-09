# Finite host-poll ingress: fixture-only checkpoint

`src/ingress/` is a small authenticated-receive substrate plus a Telegram protocol
adapter. **It is not activated in the bot.** Source configuration, synthetic
credentials and the narrowly scoped service capability can only be constructed
under `cfg(test)`. There is no service flag, credential-file reader, model builtin,
automatic poll loop, webhook change, caller attestation issuer or live cutover.

Bot/task policy stays in Python. This checkpoint stops at durable intake and one
handler admission. It does **not** propagate origin through Spawn/LlmTakeover or
issue `TrustedCaller`/user record grants. A real Python probe handler executes via
the normal frontend/runtime, receives the original JSON unchanged, and remains
unable to read records. That is a boundary test, not end-to-end live task access.

## Host receive contract

- A private source has an opaque ID, expected bot account, credential bytes supplied
  only by the fixture constructor, handler target, explicit cursor seed and private
  route mappings. Production credential-reference resolution is deferred.
- Non-test endpoint code is fixed to `https://api.telegram.org`; no caller URL is
  accepted. Only cfg(test) substitutes a local synthetic HTTP server. Requests use
  no ambient proxy, no redirects, a 10-second request deadline and bounded bodies.
- Every network receive first authenticates the configured bot request through
  `getMe`, checking exact signed-integer bot ID and `is_bot=true`, then calls
  `getUpdates?offset=<durable cursor>&limit=8&timeout=0`. Ordinary `http_get` results
  and application JSON never become evidence. No actual Telegram endpoint or live
  credential is used by the tests.
- `ok` must be a JSON Bool true. The result page is bounded and update IDs must be
  represented nonnegative signed integers below i64::MAX. Reordered IDs and altered
  duplicates within a page reject before acceptance. Identity-bearing object trees
  reject duplicate JSON members rather than silently adopting the last value.
- `serde_json`'s `raw_value` feature retains the **original JSON value bytes** of
  each update, including internal whitespace, escaping and key order; whitespace
  between array elements is not part of the value. No event reserialization is
  claimed to preserve original bytes. SHA-256 is over those exact retained bytes.
  This adds a feature to the existing locked dependency, not a version/new DB.

Private-text evidence requires a configured sender route, `chat.type="private"`,
matching positive sender/chat IDs, a non-bot sender, a positive message ID,
nonnegative date and nonempty bounded text. Current media fields suppress evidence.
This first transport supports explicit corresponding legacy private aliases
`telegram:<sender>` or `telegram:user:<sender>`, not group/main contexts or implicit
account merging. Handler/callback names are dispatch targets, not identity evidence.
Other events remain raw, unattested data for the eventual Python dispatcher; their
business behavior is not reimplemented here or certified by this checkpoint.

Evidence holds the observed principal, exact context/text/reply handler/destination,
source, event ID, digest and incarnation. It has no Deserialize/native representation
or application constructor, and Debug is redacted. Payload alteration or mixing a
delivery from another source instance cannot admit work. The data-only `agrees`
predicate is not authorization and cannot create a record grant.

## Existing Ladybug records, not a new file store

The private `IntakeStore` capability accesses only the explicitly assigned intake
namespace as a host service actor. It is not a user record binding, and neither
the capability nor credentials are included in a delivered event. Existing
RecordStore CAS/receipt transactions provide persistence and concurrency.

One transaction includes all UTF-8-safe payload chunks, event digest/status,
index entry and cursor advance. Chunk fields are ≤2048 bytes, at most three per
row, fitting existing per-field/row/batch bounds. The small source record indexes
retained IDs directly; no unbounded scan or task-specific table is introduced.
Protocol/size/order validation covers the page before admission, but each event
has its own transaction. A later capacity/storage failure can leave earlier events
accepted and recoverable; it is not a whole-page rollback.

Accepted receipt idempotency is **not** handler admission: accepted replay with the
same digest does nothing, whereas each competing handler admission uses a distinct
attempt ID with an expected event revision. Exactly one can change accepted to
started. A deterministic test barrier after both reads (outside any DB lock)
proves that generic successful receipt replay cannot create two handler winners.

Opening a source advances a bounded incarnation counter and installs a fresh host
nonce in a CAS transaction. Concurrent opens cannot share ownership. Fingerprints
bind source/bot/credential digest/handler/routes/seed, refusing implicit reuse under
changed identity. Successful reopening fences old receive/admission handles.
Credential or route changes need a separately reviewed migration, not an automatic
reinterpretation of historical intake. No plaintext credential is persisted.

Accepted entries are redispatchable after close/reopen, using exact protected
payload bytes and digest. Previously started entries become explicit **unknown**;
they are never automatically rerun or called delivered. Already admitted handler
computation is not retroactively stopped by fencing. This is clean reopen evidence,
not a process-kill/power-loss or external exactly-once claim. Takeover admission and
delivery acknowledgment remain separate future stages.

## Deliberately finite retention

| Bound, per configured fixture source | Limit |
|---|---:|
| HTTP response body | 512 KiB |
| Updates per response/request limit | 8 |
| Original event JSON value | 32 KiB |
| Retained event count | 16 |
| Total retained original payload | 256 KiB |
| Source incarnations | 32 |

No event, unknown record or idempotency evidence is silently pruned. Capacity
exhaustion returns a per-source error without advancing over an unaccepted event;
other sources can progress. The source cannot run indefinitely under these fixture
limits. There are at most 32 successful boot receipts and three successful receipts
per event (accept/start/unknown), hence at most 80. Failed admission races add no
success receipts. Production retention/archival and global source-count policy
require separate review before activation.

Ladybug work runs off the async IO executor via `spawn_blocking`; network waits
hold no DB lock. The receive mutex is per source, not a bot/project-wide global lock.

## Evidence and next integration boundary

`cargo test --release --locked ingress_poll` uses synthetic getMe/getUpdates HTTP
responses, actual Ladybug record transactions and the authored
`tests/fixtures/ingress/probe.py` through the real normal async evaluator. The probe
also passes module print/second-print checks. Fixtures cover raw-byte retention and
large Unicode chunks, accepted recovery/started-unknown, single-winner admission,
old-incarnation/cross-source fencing, wrong credentials/account, oversized/invalid
pages without cursor movement, independent sources, duplicate/altered replay,
unattested foreign/group/media updates, caller-forged metadata denial and finite
receipt/page/incarnation capacity.

Origin propagation to records is described in the next section; it is still
fixture-only and production `main` supplies no record store.

The existing Python poller is unchanged and still owns live intake. Before any
future cutover, reviewed authoritative `.py` guards must disable its poll loop and
poll_once in host-poll mode; stop/drain it, capture the final effective cursor,
explicitly seed the host source, and enforce one receiver per bot. Rollback must
stop the host source first and reconcile its cursor/pending/unknown records. No
mode guard, source activation or live cursor capture was performed here. #121
Python source authority remains compatible; no evidence is persisted as module
source/shared variables. #70 login/session work and #108 webhook authentication
remain unchanged separate workstreams.

## Origin propagation checkpoint (fixture-only)

An admitted private-text event yields a sealed `InvocationOrigin` (one `Arc`,
redacted Debug, no Serialize/Deserialize, no Adapsis or native constructor). It
travels only out-of-band:

- `CoroutineHandle.origin` defaults to `None` in every constructor; only the host
  admission path sets it. `+spawn` copies it into `IoRequest::Spawn` (shared
  `eval/spawn.rs`), and the shared `spawn_dispatch` (also used by production
  `main`) installs it on the spawned handle. `llm_takeover` copies it into
  `IoRequest::LlmTakeover`. Ordinary HTTP/route/source/timer/CLI/inject paths and
  direct handler calls carry `None`. A handler or callback name is never trust.
- `api/takeover_dispatch.rs` is the single production entry into the existing
  `handle_llm_takeover` loop (no second loop). For an origin-bearing request it
  first checks the exact original text, context, `speaker_id` metadata, reply
  handler and reply argument (attachments refused), then durably CAS-claims the
  event `started -> takeover_started` with a unique attempt ID. Mismatch and lost
  races fail with a categorical message before the conversation, permission
  profile, history or provider are touched. Cloned origins in one process also
  share an in-memory one-shot latch. Origin-free requests keep legacy behaviour,
  including the pre-loop permission-profile update.
- Only a `ClaimedOrigin` reaches the new `TrustedCaller::from_ingress` issuer. The
  record ceiling is captured from the ACTUAL generating model plus the context
  permission profile, principal profile and process cap; guests/missing model
  deny. The per-turn `RecordTurn` adjunct stays separate from `TurnIdentity`.
- `AppConfig::coroutine_handle` is the one handle factory for executor evaluation.
  It attaches record access only when the adjunct matches the turn's principal and
  context, recaptures the CURRENT ceiling (pin ∩ current; never widens), and binds
  through the grant registry. The backend rechecks generation/revocation per
  operation. Everything else returns the legacy unbound handle.
- **Revocable, per-handle ceiling.** The `TrustedCaller` carries a liveness flag
  shared with the turn's `RecordTurn`. `takeover_dispatch` clears it on every exit
  (finished, unknown, error, or the future being dropped), and the backend checks it
  on every read/commit/receipt as well as at bind time. A record-bearing handle that
  escapes its turn is therefore denied afterwards. Ceilings are evaluated per handle
  (pin ∩ current at mint time); this is acceptable only because revocation exists.
- **The turn's own profile is pinned.** The pin uses the dispatch's own
  `permission_model`, not the shared conversation field (which a concurrent
  origin-less dispatch to the same context can rewrite before this turn takes the
  context lock). `coroutine_handle` intersects both the live field and that pinned
  profile; either can only narrow.
- **Origins are single-use.** The first takeover attempt (successful or not) takes
  the intake store out of the lease. A spent origin is not forwarded by `+spawn` or
  `llm_takeover`, so later calls and descendants run as origin-free legacy work with
  no record access, and long-lived tasks do not retain the intake store.
  **Burn-on-mismatch:** a refused (mismatched) first attempt also spends the origin.
  A correctly bound retry from the same handler then runs as legacy work without
  record access; the event was never claimed, stays `started`, and becomes `unknown`
  on the next source reopen. A handler must bind correctly the first time.
- After the loop: an accepted finish (native) or accepted `!done` (legacy) marks
  the event `turn_finished`; provider/other failure marks `unknown`. Neither is a
  delivery or external-effect certificate, and `unknown` is never auto-retried.

Production wiring is deliberately inert: the IO loop passes `record_store: None`,
no service flag/config/credential reader exists, and source construction remains
`cfg(test)`. The intake service capability is never placed on a coroutine handle.

Gate `cargo test --release --locked ingress_pipeline` (9): synthetic getMe/getUpdates
-> admitted event -> authored `tests/fixtures/ingress/pipeline.py` entry -> real
`+spawn` through `spawn_dispatch` -> worker `llm_takeover` through
`takeover_dispatch` -> existing native loop with a scripted localhost provider ->
`evaluate Flow.write` -> `records_commit` on a temp Ladybug with operator-provisioned
grants. Positive: exactly one row + receipt, event `turn_finished`, duplicate spawned
worker refused before any provider call (2 provider rounds total), no credential or
origin text in history/provider requests. Negatives: identical JSON without origin
writes nothing although the write was attempted (`record_access_denied` observed in
the tool result); actual-model `None` and context-profile `Read` each block the
write; context mismatch is refused before claim/conversation/provider; provider
failure leaves the event `unknown`. A handle minted by the verified turn and used
after `turn_finished` is denied (no row); a descendant spawned after consumption runs
legacy to another context with its write denied; a verified turn pinned to an
Execute profile stays denied after a concurrent admin dispatch resets the shared
field before it acquires the lock. The last two regressions were checked to fail
with revocation or pinning disabled.

`eval::Value` embeds `CoroutineHandle` by value, so its size bounds tree-walker
recursion before the depth guard fires. Record access and origin are therefore
behind single pointers, and `value_stays_small_enough_for_deep_tree_recursion`
pins `size_of::<Value>() <= 184`. (Before this, the full suite aborted with a stack
overflow in `python_frontend_logical_call_operand_short_circuits_tree_and_vm`.)

Not delivered: live activation, host poll loop/cutover, Python receiver mode
guards, media/group attestation, external exactly-once delivery, #70 or #108.

The `__` identifier prefix is reserved (`src/reserved_names.rs`): user-written
reads, let/set/await/call/spawn bindings, parameters, loop bindings and match
captures named `__*` are refused at validation, and the evaluator refuses to read
`__*` identifiers. Only parser-generated `__pat_<n>` nested-pattern placeholders
are allowed. This removes the in-language way to copy `__coroutine_handle`; it also
protects the live `TurnIdentity` handle independently of records.
