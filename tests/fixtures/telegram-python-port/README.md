# TelegramBot Python port — partial, runtime gate RED

This is a separate port work area, not a replacement for the frozen
`tests/fixtures/python-parity` corpus. The original corpus and manifest are
unchanged. `manifest.json` records their exact Git blob hashes, baseline commit,
per-function transformation ledger, implemented and pending inventories.

## Present subset

- **TelegramBot: 10 of 42 functions.** Extract IDs/type/text/media references,
  format sender, match ID-list boundaries, inspect response success/file path.
- **TgFormat: 3 of 18 functions.** HTML escaping, inline formatting, tag building.
- Selected function ASTs, including docs, match the frozen originals. Partial
  origins and Python lowering compare exact structures ignoring only typed
  NodeIds. Printing/re-lowering and second-print identity are checked.
- Module docs, shared state, lifecycle/routes and original tests are **pending**.
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

The command is intentionally **nonzero** until the direct-VM port regression is
fixed. No failure allowlist or fallback execution counts as a direct-VM pass.

| Subset | Tree per origin | Direct VM per origin |
|---|---:|---:|
| TelegramBot (28 literal cases) | 28 pass | 25 pass, 3 fail |
| TgFormat (7 literal cases) | 7 pass | 7 pass |

Both `.ax` and Python-lowered origins have the same results. TelegramBot VM
failures (#98) are cases 1, 3 and 4: nonempty numeric chat/sender IDs. `json_get` returns
an Int for JSON integers; `val == ""` works in the tree evaluator but VM mixed
comparison errors (`comparison: unsupported types 123 and ""`, or 42). The
declared String annotation and frontend catalog are not proof of a String value.
Contract fixtures explicitly cover actual `json_get` String/Int/Float/Bool,
null, object, array and missing-key variants. No coercion or runtime fix is hidden
in these port files.

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
- Normal `eval_async_function` dispatch of `send_message_parts` fails at the
  user await with **zero** requests, for both bare and qualified targets. The VM
  treats the user await as a builtin AwaitIo; the attempt sets `io_performed`
  before builtin dispatch rejects it. **Qualification alone does not fix this.**
- An explicitly labeled direct-tree diagnostic route reaches the consumer:
  successful user await unwraps Ok to String, succeeds after one request;
  transport Err stays wrapped, reaches `index_of`, and fails after one request.
  It does not perform the intended plain-text retry. Both name variants agree.
  Direct-tree success is not a normal-dispatch success claim.

These independent blockers require separately reviewed runtime/source decisions.
This slice does not change IO dispatch, reset replay guards, unwrap failures,
relax permissions, or resolve #82/#94/#97.

First-slice final release gate: **1768 passed, 2 failed, 2 ignored; CLI 5/5**.
The failures are the strict original-corpus parity gate and this explicit
direct-VM port gate. Targeted `telegram_port`: **12 passed, 1 failed**. This is
an evidence checkpoint for review, not functional port acceptance.

## Remaining full port gate

The manifest explicitly lists 32 TelegramBot and 15 TgFormat pending functions.
Still required: full corrected-origin provenance/review, paired exact structural
and idempotence gates, metadata/principal helpers, raw-update admin/guest/group
routing, takeover bindings, photo/video/voice paths, reply splitting/retry and
attachments, startup/poll/shared/spawn behavior, plus representative error paths.
The partial tests here do not close #86, certify deployment, or imply the original
frozen TelegramBot fixture passed its separate parity gate.
