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

Next checkpoint must carry an admitted sealed origin out-of-band through the
actual Python intake → Spawn → takeover path, revalidate source/admission freshness
and exact model-context/reply binding, and only then connect the record caller
attestation issuer. No service capability may be forwarded to application handles.
This checkpoint does not claim that chain exists yet.

The existing Python poller is unchanged and still owns live intake. Before any
future cutover, reviewed authoritative `.py` guards must disable its poll loop and
poll_once in host-poll mode; stop/drain it, capture the final effective cursor,
explicitly seed the host source, and enforce one receiver per bot. Rollback must
stop the host source first and reconcile its cursor/pending/unknown records. No
mode guard, source activation or live cursor capture was performed here. #121
Python source authority remains compatible; no evidence is persisted as module
source/shared variables. #70 login/session work and #108 webhook authentication
remain unchanged separate workstreams.
