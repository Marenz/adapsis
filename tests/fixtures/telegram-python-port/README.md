# TelegramBot Python port — partial corrected-origin program

This is a separate port work area, not a replacement for the frozen
`tests/fixtures/python-parity` corpus. The original corpus and manifest are
unchanged. `manifest.json` records their exact Git blob hashes, baseline commit,
per-function transformation ledger, implemented and pending inventories.

## Present subset

- **TelegramBot: 31 of 42 original functions, plus one new partial entry.**
  Sixteen literal-paired pure helpers, one trusted-Result pure helper,
  two shared-state authorization helpers and twelve
  IO functions are separately covered. `process_text_update` is NEW and is not
  counted among the original 42; original `process_update` remains pending.
  A new application outcome union and the original sanitized empty token shared
  declaration support this bounded send-chain port.
- **TgFormat: all 18 functions**, including Markdown/code-block conversion,
  splitting, preview and sample helpers. This is function coverage, not full
  original-module parity or service-limit-safe delivery.
- Unrepaired functions permit only approved local-call qualification: **37 call
  sites in TgFormat, 5 in TelegramBot**, counted per function in the manifest.
  The test-only transform proves all other function fields, including docs,
  unchanged. Thirteen separately ledgered semantic repairs deliberately change return
  types/effects/control flow/docs; these are not qualification-only equivalence.
  Corrected origins and Python lowering compare exact structures
  ignoring only typed NodeIds. Printing/re-lowering and second-print identity
  are checked. Direct-VM pure functions retain paired assertions; shared-state,
  IO and new-entry functions have separately enforced named normal-dispatch tests.
- TgFormat's module doc is now included unchanged. TelegramBot's module doc,
  remaining shared state, lifecycle/routes and original tests remain **pending**, as do
  TgFormat's original stored tests.
  Paired Python tests are newly authored literal cases; redacted original test
  expectations are not reinterpreted. The selected TelegramBot declaration order
  is intentionally different and recorded. There are no placeholder functions.
- The #91 voice repair is explicitly authored in this partial fixture; the
  malformed frozen function is not treated as an equivalent source program.
  Original `process_update` integration remains pending.

## Measured gates

```sh
cargo test --release telegram_port -- --nocapture
```

This focused command currently has **38 passing tests**, including the original
presence gate (#105) without changed expectations. No failure allowlist or
fallback execution counts as a direct-VM pass. The separate frozen-corpus parity
test still fails in the full suite; this partial port does not replace it.

| Subset | Tree per origin | Direct VM per origin |
|---|---:|---:|
| TelegramBot pure helpers (51 literal cases) | 51 pass | 51 pass |
| TgFormat (38 literal cases) | 38 pass | 38 pass |

Both corrected `.ax` and Python-lowered origins have the same results. The first
slice's VM failures (#98), preserved in historical manifest metadata, were cases
1, 3 and 4: nonempty numeric chat/sender IDs. The narrow runtime repair now makes
the original assertions pass unchanged. `json_get` returns
an Int for JSON integers; VM mixed equality now matches the tree for those cases. The
declared String annotation and frontend catalog are not proof of a String value.
Contract fixtures explicitly cover actual `json_get` String/Int/Float/Bool,
null, object, array and missing-key variants. No coercion or runtime fix is hidden
in these port files. Current behavioral cases include HTML escaping, inline and
fenced code, Unicode, boundary splitting, parsed ID lists and stable principals
derived from `message.from.id` rather than display names or spoofed text prefixes.

## Inherited quirks and unresolved behavior

- **#99: splitting is not service-limit safe.** At configured limit 3800,
  generated 5000-character ASCII plain text stays one 5000-character part;
  a pre block containing the same content stays one 5011-character HTML part.
  Even excluding HTML tags, content length 5000 exceeds Telegram's documented
  4096 cap. This is a pure-function observation, not a live delivery experiment.
  The preserved original module doc's “length-safe” wording is not a new claim.
- String `len` uses UTF-8 bytes here. The Unicode boundary tests preserve that
  behavior; they do not establish Telegram character-count compatibility.
- Missing message ID plus a valid date produces a JSON message_id value containing
  two double-quote characters: `to_string` formats the missing field's empty
  String using Value Display. Missing date fails with
  `to_int: cannot parse '""' as integer`. Both origins and tiers preserve these
  source quirks; they are **not desirable API contracts** or fixes to input handling.

## Production IO contracts, without external operations

`telegram_port_tests/io_contracts.rs` invokes the production CoroutineHandle
dispatch, intercepts the exact IoRequest and supplies an in-memory reply. It
never starts the Runtime IO loop. No file access, actual sleep, network,
provider call or credentials are used by those IO fixtures. Small attachments
remain memory-backed. Request/result waits are bounded.

| Operation | Request checks | Actual return contract |
|---|---|---|
| read_file | exact path | bare String, outer error on failure |
| file_exists | exact path | Bool, outer error on failure |
| sleep | exact milliseconds | Int(0), outer error on failure |
| http_get | exact URL | Ok(String) / Err(String) Value |
| http_post | URL/body/default or explicit content type | Ok(String) / Err(String) Value |
| http_get_binary | HttpPostBinary variant, URL, empty body, octet-stream | bare Attachment / outer error |
| http_upload | path or bytes, name/MIME, field and extra pairs | Ok(String) / Err(String) Value |
| llm_takeover | context, message, reply binding, profile, metadata overload/attachment | bare String / outer error |

These valid-call contract fixtures are **not** a complete frontend IO signature
or arity checker. Bare IO call-site validation checks effects/await mode, not all
signatures; many result types remain Unknown. No new permissive typing was added.

## Frozen wrapper boundary characterization (#82)

`wrapper_contract.rs` selects the original `send_message_part`,
`send_message_parts`, `send_message_body`, `telegram_response_ok` and token shared
declaration. A separate in-memory variant qualifies exactly those four call
targets. An inverse AST check proves **only call qualification** changed; neither
variant rewrites wrapper handling or overwrites the frozen fixture.

- `send_message_part` (`String [io,async]`) performs one synthetic HttpPost and
  returns a Value::Ok(String) or Value::Err despite its String declaration.
- The original normal-dispatch checkpoint failed at user await with zero requests,
  for both bare and qualified targets. The dispatch repair checkpoint `e5337b0`
  now preflights reachable user awaits and chooses tree before execution. Normal
  dispatch therefore reaches the same one-request behavior described below.
  Qualification alone was not the fix; replay flags were not reset.
- An explicitly labeled direct-tree diagnostic route reaches the consumer:
  successful user await unwraps Ok to String, succeeds after one request;
  transport Err stays wrapped, reaches `index_of`, and fails after one request.
  It fails before the retry branch. Automatically retrying that uncertain
  transport outcome could duplicate delivery and is not the proposed policy.
  Both name variants agree.
  Direct-tree success is not a normal-dispatch success claim.

These independent blockers require separately reviewed runtime/source decisions.
This slice does not change IO dispatch, reset replay guards, unwrap failures,
relax permissions, or resolve #82/#94/#97.

Current voice-slice release gate: **1817 passed, 1 failed, 2 ignored; CLI 5/5**.
The remaining failure is the frozen-corpus parity gate. Targeted `telegram_port`:
**38 passed**. This remains partial routing evidence, not deployment certification.
The earlier first-slice RED checkpoint remains in the manifest's explicitly
historical field. Neither checkpoint is full functional-port/deployment acceptance.

## Remaining full port gate

The manifest explicitly lists **11 TelegramBot pending original functions**; TgFormat has
no pending functions but its original stored-test surface remains pending.
Still required: full corrected-origin provenance/review and whole-module paired
structural/idempotence gates, integration of the pure metadata/principal helpers
into raw-update admin/guest/group routing, takeover bindings, photo/video/voice
paths, integration of splitting with delivery/retry and
attachments, startup/poll/shared/spawn behavior, plus representative error paths.
The partial tests here do not close #86, certify deployment, or imply the original
frozen TelegramBot fixture passed its separate parity gate.

## Corrected send-chain policy — implemented in this partial port

The corrected send functions return `TgSendAttempt`, an application union with
`TgSent(String)`, `TgRejected(String)` and `TgUncertain(String)`, and declare
`[io,fail,async]`. Their original parameters are retained, but return type,
effects, control flow and docs deliberately differ from the frozen originals.
`semantic_repairs` records that separately from local-call qualification.
The unchanged frozen-wrapper characterization above remains as historical evidence.

The leaf explicitly matches the **builtin** `http_post` Result. Only a parsed
Boolean true constructs Sent; only Boolean false constructs Rejected. Transport
Err and nonboolean/missing status construct Uncertain. Malformed JSON raises a
terminal parse error rather than a uniform union outcome. No retry occurs in the
leaf. The chain attempts HTML first, permits one plain fallback only on Rejected,
then stops remaining chunks on any non-Sent outcome. Earlier failure is not
masked by later success. Empty/out-of-range input returns Uncertain with zero
requests; the legacy `last` argument cannot manufacture success.

Both `.ax` and Python origins execute through production `eval_async_function`
with intercepted IoRequests, not only a direct-tree diagnostic route. Tests cover
real Bool versus strings/numbers/null/objects/arrays/missing/malformed status,
whitespace, multiple chunks, exact HTML/plain ordering, second rejection,
transport errors (including text containing `ok:true`), outer await rebinding,
empty wildcard-arm execution and zero-request invalid indices. Raw return values
must be the application Union, not an unwrapped Result or String.

The token default is empty and identical to the sanitized original declaration.
Tests install only `fixture-token` in isolated runtime state and intercept every
request. No live credential read, actual network operation or remote-delivery
experiment occurs. Local attempt counts **do not establish exactly-once remote
delivery**; transport uncertainty may follow an already delivered request.

`to_string(json_get(body,"ok"))` is a tested dynamic discriminator, **not sound
static JSON typing**: the existing catalog mismatch remains unresolved. Its
runtime type-path tests ensure JSON strings `"true"`/`"false"` cannot impersonate
Booleans. No frontend/runtime permissive changes or global user-Result-await
relaxation were made. Full original intake and outbound attachment integration remain
pending, and #99 is deliberately unfixed: retry fixtures use explicit short chunks
and do not certify service-limit-safe splitting.

## Media acquisition and attachment-handler checkpoint

The five added originals are `telegram_file_path`, `telegram_file_error_message`,
`handle_group_attachment`, `handle_admin_dm_attachment` and
`handle_user_dm_attachment`. The error-message helper is unchanged and has two
additional paired pure assertions. File lookup retains its String/error-String
contract; it now explicitly awaits and matches the builtin Result. An inverse
AST check proves only that ledgered transformation and helper qualification.
Malformed JSON remains a terminal error. The attachment handlers retain the
original takeover prefix and propagate TgTextTurn/nested send outcomes without
an additional retry, matching the previously repaired text handlers.

`media.rs` first checks real Attachment annotations through the existing frontend
and canonical roundtrip. `media/runtime.rs` uses a NEW test-only
`PortMediaDriver.run`, excluded from the original-function inventory and source
parity claims. It joins file lookup, binary download, authorization and attachment
handlers through normal async dispatch on both origins. Synthetic photo/video
payloads cover private/group/supergroup contexts and admin/non-admin speakers.
Assertions check request ordering, structured principal/profile/callback/metadata,
three-byte memory-backed Attachment content/name/MIME, no immediate reply, first
success, plain-fallback success, double rejection and transport uncertainty.
Lookup errors/missing paths, malformed JSON, binary failure and takeover failure
stop later requests. No runtime IO loop, network, credentials or live chat is used.

This media checkpoint is not original `process_update` parity, voice/outbound
attachment support, remote delivery proof or callback acknowledgement (#104).
The subsequent voice helper slice below does not resolve #99 splitting limits.
Media checkpoint full-gate output:
`tool_18ad87056001s1tyPm13lmCcNP`.

## Voice helper slice — explicit #91 repair

`transcribe_voice` now explicitly awaits getFile's builtin Result and assigns
its transport error to the outer Err arm, preserving the exact getFile-failed
placeholder. This is an intentional authored decision for the malformed original
nested match (duplicate Err/out-of-scope e), not inferred AST equivalence.
The function preserves its parameter, String return, URL construction, ASR URL,
multipart field `file` and empty extras; it adds `[fail]`. File path must be a
nonempty JSON String or the existing no-file placeholder is returned.

The binary download still returns a bare Attachment or an outer error. Its error
terminates execution before upload. Upload's actual builtin Result is passed intact
to `transcription_upload_response`; there is no fabricated Ok wrapping or retry.
The helper preserves its signature, doc and original branches, adding only a
`json_kind` String guard before extraction (proved by inverse AST comparison).
Missing, wrong-kind or empty transcript returns the existing transcription-failed
placeholder. Malformed JSON is terminal. Legitimate Strings `null`, `{}`, `[]`,
Unicode and newlines are preserved.

`telegram_port_voice_helper_trusted_results` runs 17 Rust-constructed Result inputs
through both tree and direct VM on both origins; no Python literal restriction is
relaxed. `telegram_port_voice_normal_async_contracts` runs 34 cases per origin
through normal async dispatch, checking exact ordered requests and outcomes:

| Outcome | Requests |
|---|---|
| getFile transport error → getFile-failed placeholder | getFile (1) |
| path absent/wrong-kind/empty → no-file placeholder | getFile (1) |
| malformed getFile JSON → terminal error | getFile (1) |
| binary failure → terminal outer error | getFile, binary (2) |
| valid transcript → unchanged String | getFile, binary, upload (3) |
| upload transport error or absent/wrong-kind/empty text → failure placeholder | getFile, binary, upload (3) |
| malformed ASR JSON → terminal error | getFile, binary, upload (3) |

Tests assert three-byte memory-backed audio/ogg `voice.ogg` passes unchanged to
HttpUploadBytes, along with URL/field/extras and no extra request. They run no real
ASR, network, filesystem IO, takeover or message delivery.

The strict global corpus/manifest are unchanged and still report invalid original
source. Only the port's original-AST equivalence subset omits malformed
`transcribe_voice`; its separately named authored-repair source/signature/ledger
test replaces that impossible equivalence claim. It remains explicitly in the
original-function inventory and the complete corrected-origin/Python AST and
idempotence gate. Manifest categories require named actual helper/IO tests.
This does not integrate original `process_update`, caller metadata/context routing,
or callback acknowledgement. Full-gate log: `tool_18ae802c4001WxBSoBgqxiBBzD`.

## Higher callable reply closure and callback blocker #104

`send_message` preserves the existing TgFormat conversion and 3800 split limit,
then explicitly awaits and returns the chain outcome unchanged. `send_reply`
preserves `to_int` destination parsing, explicitly awaits `send_message`, and
returns the same outcome. Neither adds a retry. Combined-module signatures and
exact AST/idempotence gates include TgFormat; the Python execution side lowers
both modules from Python. Both-origin normal-async tests check exact HTML and
HTML/plain ordering, parsed destinations (negative IDs and leading zeroes),
unchanged Sent/Rejected/Uncertain/error propagation, and zero requests for empty
text or an invalid destination. Inputs are short, not a #99 safety certification.

**This does not certify callback-delivery acknowledgement (#104).** Read-only
inspection of `src/main.rs` found the spawned callback paths (near lines 1373 and
1783) classify every evaluator `Ok(val)` as `WaitReason::Completed(format!(val))`.
`await_spawned_callback_with_timeout` (50–76) treats every Completed string as
success; `dispatch_optional_conversation_callback` (129–131) maps it to
`"notified"`. A returned TgRejected/TgUncertain would therefore be misclassified.
No dispatcher or routing code is changed here. A later transport-independent,
legacy-compatible acknowledgement contract must not parse formatted strings or
hardcode Telegram union tags in the generic runtime. The callable result tests
above make no claim that uncertainty means delivered/notified.

## Partial text routing — #105 presence gate repaired, not full process_update

The three text handlers preserve original context/profile/callback/metadata
construction (an AST prefix check against frozen source proves it, allowing only
the existing format_sender qualification). They now return TgTextTurn:
NoImmediateReply for empty model text, or DeliveryAttempt containing the exact
TgSendAttempt. Failure is not discarded and no extra retry is added. The named
payload reference retains the canonical parser's Struct spelling while resolving
to the declared TgSendAttempt union; runtime tests inspect the actual nested union.

The NEW `process_text_update` is an intentionally narrower text-only fixture.
It now calls `json_kind` before conversion/authorization/IO. Chat/from/date must
be represented signed-i64 `int`, with the previous nonzero-chat/positive-sender/
nonnegative-date bounds; explicitly provided date 0 remains valid. Chat type and
text require `string`. Bool true, integral Float1.0 and String"1" cannot become
administrator Int1. Missing or wrong-kind fields are ignored/deferred before
conversion; represented-u64 overflow or malformed JSON aborts before IO. Existing
authorization rules and handler request construction are unchanged.

The old List.contains raw/candidate guards and to_string/startswith text heuristics
are removed ONLY from this NEW entry. Legitimate Strings `null`, `{}`, `[]` and
prefix-leading text are now admitted; their non-String JSON counterparts defer.
The enum membership check for supported String chat types remains. This deliberate
adoption is ledgered separately, not described as original process_update parity
or a general correction of json_get's still-polymorphic catalog entry.

**The original #105 gate now passes unchanged:** absent media admits the text
control, while present-empty media defers with zero IO for all **11 keys × both
origins**. json_get still returns identical empty Strings in that diagnostic, but
the entry uses json_kind's preserved presence. Every present supported media kind,
including null/false/empty arrays/empty Strings, defers; primitive parse/range
errors terminate without IO. There are no exemptions, regex scans or new heuristics.
This does not implement attachment handling or complete original raw-update routing.

Other normal-dispatch tests on both origins prove private/group/supergroup routing,
sender-versus-chat distinction, exact structured principal/profile/reply arguments,
name/text spoof controls, immediate Sent/Rejected/Uncertain propagation, and
isolated fake shared lists restored between cases. Authorization helpers explicitly
fail direct VM compilation (shared variables unsupported) and pass normal fallback
value tests; this is not direct-VM parity. Takeover responses are synthetic: no
model, upstream Telegram authentication, remote delivery or callback execution is
certified. #104 remains a separate deployment blocker.

The primitive was independently accepted in `3e58e30`; this separate port adoption
does not change it, json_get, runtime dispatch, aliases or frontend signatures.
Tests retain original handlers/helpers, both-origin exact AST/idempotence and the
unchanged pure gates; additional collision, huge-number and i64-edge controls run
through the actual normal async entry. #91, #99 and #104 remain separate blockers.
