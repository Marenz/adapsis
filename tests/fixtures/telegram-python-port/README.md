# TelegramBot Python port — partial, pure dependency closure

This is a separate port work area, not a replacement for the frozen
`tests/fixtures/python-parity` corpus. The original corpus and manifest are
unchanged. `manifest.json` records their exact Git blob hashes, baseline commit,
per-function transformation ledger, implemented and pending inventories.

## Present subset

- **TelegramBot: 14 of 42 functions.** The original pure subset plus
  `parse_id_list`, `extract_sender_name`, `speaker_principal`, `source_metadata`.
- **TgFormat: all 18 functions**, including Markdown/code-block conversion,
  splitting, preview and sample helpers. This is function coverage, not full
  original-module parity or service-limit-safe delivery.
- Only approved local-call qualification changes frozen function ASTs: **37 call
  sites in TgFormat, 5 in TelegramBot**, counted per function in the manifest.
  The test-only transform proves all other function fields, including docs,
  unchanged. Corrected origins and Python lowering compare exact structures
  ignoring only typed NodeIds. Printing/re-lowering and second-print identity
  are checked. Every implemented function must have a paired assertion.
- TgFormat's module doc is now included unchanged. TelegramBot's module doc,
  shared state, lifecycle/routes and original tests remain **pending**, as do
  TgFormat's original stored tests.
  Paired Python tests are newly authored literal cases; redacted original test
  expectations are not reinterpreted. The selected TelegramBot declaration order
  is intentionally different and recorded. There are no placeholder functions.
- The #91 voice repair has **not** been authored. It needs explicit control-flow
  review before inclusion. Qualification/async-await transformations of the full
  corrected origin are likewise pending.

## Measured gates

```sh
cargo test --release telegram_port -- --nocapture
```

This focused command currently passes **15 tests**. No failure allowlist or
fallback execution counts as a direct-VM pass. The separate frozen-corpus parity
test still fails in the full suite; this partial port does not replace it.

| Subset | Tree per origin | Direct VM per origin |
|---|---:|---:|
| TelegramBot (46 literal cases) | 46 pass | 46 pass |
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

Current pure-expansion release gate: **1787 passed, strict original-corpus parity
alone failed, 2 ignored; CLI 5/5**. Targeted `telegram_port`: **15 passed**.
The earlier first-slice RED checkpoint remains in the manifest's explicitly
historical field. Neither checkpoint is full functional-port/deployment acceptance.

## Remaining full port gate

The manifest explicitly lists **28 TelegramBot pending functions**; TgFormat has
no pending functions but its original stored-test surface remains pending.
Still required: full corrected-origin provenance/review and whole-module paired
structural/idempotence gates, integration of the pure metadata/principal helpers
into raw-update admin/guest/group routing, takeover bindings, photo/video/voice
paths, integration of splitting with delivery/retry and
attachments, startup/poll/shared/spawn behavior, plus representative error paths.
The partial tests here do not close #86, certify deployment, or imply the original
frozen TelegramBot fixture passed its separate parity gate.

## Send-error repair — design only, not implemented

The proposed corrected-origin repair explicitly matches the builtin HTTP Result
and uses trusted application outcome tags. Only an actual parsed Boolean
`ok:false` may trigger one plain-text fallback; actual Boolean true means success.
Transport uncertainty, malformed/missing/nonboolean status, or rejection of the
plain fallback must stop remaining chunks. Never infer retry permission from an
untrusted text sentinel, or infer exactly-once remote delivery from mock counts.
The current `json_get` catalog mismatch remains explicit: any proposed
`to_string(json_get(...))` discriminator requires actual type-path tests rather
than a static-typing claim. This policy changes corrected-origin behavior and
needs separate ledger/request-order review. No send-path or runtime changes are
included in this pure-function expansion.
