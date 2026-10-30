# AdapsisOS Roadmap

## Recovery — Python parity #86 (current: 2026-10-28, red)

### Current pending slice — two voice helpers, baseline `3c10387`

- Fixtures/tests/docs only;31/42originals+1NEWtextentry,11originalspending.
  transcribe_voice is an intentional #91 authored control-flow repair, not inferred
  equivalence to malformed duplicate-Err/out-of-scope-e source. Explicit getFile
  Result binding/outer error placeholder, [fail], actual upload Result passed intact
  to transcription_upload_response. Both file_path and text require nonempty
  JSON Strings; wrongkind/missing returns existing placeholders, legitimate
  null/{} /[] Strings preserved. JSON parse and binary outer errors are terminal.
- Helper inverse-guard proof preserves all remaining original AST/signature/doc.
  Strict global corpus unchanged/still invalid; only the port original-equivalence
  subset handles malformed transcribe_voice through a separate authored-repair
  gate. It remains in original inventory and full corrected/Python AST/idempotence.
  Named coverage enforced: trusted Result helper17cases/tree+VM/bothorigins;
  normal async34cases/origin checking exact1/2/3requests and no fourth/retry.
  Lookup failures stop1, download failure stops2, upload outcomes finish3; no
  live ASR/network/files, caller/process_update/context/takeover integration.
- Targeted38/38; single fullgate1817passed/strictparity1failed/2ignored;CLI5/5.
  Log `tool_18ae802c4001WxBSoBgqxiBBzD`. #99/#104 remain open. Current authored
  manifest/origin/Python blobs:
  `73812273798591049536755bfb9be391c9b77432`,
  `2597eb267b35aac54c11f074f8aecbfda7db02ff`,
  `7a1a205585e04bb4fcdd2a571722bc379381a00a`.
  Frozen pending independent review; manager checkpoints, no writer commit/deploy.

### Accepted checkpoint `3c10387` — media acquisition/handlers

- Fixtures/tests/docs only: adds telegram_file_path, telegram_file_error_message
  and three attachment handlers. **29/42 originals plus1NEW text entry**, 13
  originals pending; TgFormat remains18/18. NEW test-only PortMediaDriver is
  excluded from inventory/source parity; original process_update remains pending.
- Actual Attachment annotation/roundtrip admission passed without frontend changes.
  File lookup explicitly awaits/matches builtin Result while retaining original
  String/error-String contract (inverse AST proof); malformed JSON is terminal.
  Attachment-handler takeover prefixes are preserved; nested TgTextTurn delivery
  outcomes propagate without extra retry. Semantic repair ledger now11 functions.
- Both-origin normal async fixtures intercept getFile/binary/takeover/reply requests,
  checking private/group/supergroup × admin/non-admin × photo/video, metadata,
  principal/profile/callback and tiny memory-backed Attachment bytes/name/MIME.
  No-reply, HTML success, plain fallback success, rejection and uncertainty covered;
  acquisition/parse/binary/takeover errors stop later requests. No live IO.
- Targeted port **35/35**; pure cases51Telegram+38TgFormat per origin/tier;
  exact combined AST/idempotence and source-fidelity checks pass. One full gate:
  **1814 passed, strict corpus alone failed, 2 ignored; CLI5/5**, log
  `tool_18ad87056001s1tyPm13lmCcNP`. Frozen corpus/runtime/old wrapper unchanged
  against3f81a9e. No #91 voice, #99 splitting or #104 callback ACK resolution.
- Current authored manifest/origin/Python blobs (not historical proofs):
  `832b2641358270dbad9e58963348481d0cd1cadc`,
  `f04b14733900e2a9fabc70125e945840a14139a7`,
  `ad6ab3c68f1f5bf83f8587091ce7e7bae10561cd`.
  Accepted and checkpointed at3c10387; no deployment.

### Accepted checkpoint `3f81a9e` — #105 NEW-entry adoption

- Fixtures/tests/docs only. The introduced process_text_update entry now checks
  exact json_kind labels before conversion/auth/IO: int chat/from/date with prior
  value bounds, string chat.type/text, and every present media kind deferred.
  Missing/wrong-kind IDs now return Ignored before to_int; primitive parse/u64
  errors remain terminal/no IO. Integer date0 is accepted. No original function,
  handler, runtime, catalog, primitive or permission rule changed.
- Removed only the NEW-entry List.contains identity guard and text rendering/
  prefix heuristics. Strings null/{} /[] /prefix-leading now reach the correct
  structured principal/profile routing; non-String collision partners defer0IO.
  Chat-type wrong kinds ignore0IO. Huge represented-u64 errors and serde-rounded
  Float IDs ignore/error0IO, while signed-i64 edge IDs pass valid bounds.
- **Original #105 presence gate GREEN with unchanged assertions:** all11media
  keys ×2origins present-empty defer with zero IO; absent controls remain admitted.
  The diagnostic still proves json_get alone erases absence/present-empty, so no
  json_get semantic change is hidden. No exclusions, regexes or raw-JSON scans.
- Targeted port **32/32**; exact combined AST/idempotence and existing routing,
  profile, outcome/state-isolation and unchanged pure gates pass. One full gate
  **1811 passed, strict corpus alone failed, 2 ignored; CLI5/5**. Log
  `tool_18a63a9f0001dV4TpnBuPZYcrt`. Corpus54/573 remains **116/58/399**,
  other buckets0. No original process_update parity or deployed/remote ACK claim.
- Count remains **24/42 originals plus1NEW entry**, 18originalspending, TgFormat
  18/18functions with other module surfaces pending. #91/#99/#104 remain separate.
  Authored manifest/origin/Python blobs recorded now:
  `bb40ae70239db2601b802c4de96e0209b2b2ea1d`,
  `6feb6273874b8974a080bb8686293ba42422b9c9`,
  `c8c6b1356686b8ea11b1445744403910e5793d54`.
  Adoption is explicitly ledgered against3e58e30; primitive/frozen corpus/wrapper
  code is unchanged. Accepted and checkpointed at3f81a9e.

### Accepted checkpoint `3e58e30` — #105 json_kind primitive

- Pure `json_kind(json,path)->String`, exact two String arguments, no aliases/IO:
  closed missing/null/bool/int/float/string/array/object labels. Extracted
  `eval/json.rs` shares one unchanged resolver with json_get, preserving its
  scalar conversions, missing/null/aggregate outputs and error text. Oversized
  eval/mod.rs and builtins.rs shrink; JSON registry constants live in a leaf.
- Exact legacy path grammar retained: empty dot components ignored/root, usize
  components (including +0/00) index arrays even on objects; other/overflowing
  numeric components are literal keys. No new bracket/escape syntax or malformed
  String-path error. Wrong container/out-of-bounds is missing. Argument type/arity
  and malformed JSON are actionable evaluator errors.
- `int` means represented signed-i64. Represented u64 above i64 returns Err;
  finite serde floating representation is float, including 1.0/-0.0 and oversized
  integer lexemes rounded into floats. Parser overflow errors; no lexical-precision
  guarantee, dependency or serde-feature change. json_get behavior is unchanged.
- New accurate frontend `(str,str)->str` signature, registry metadata, legacy and
  native prompt guidance and Python example docs. Tests cover type/presence
  collision pairs, root/path/index/key quirks, min/max i64/u64 overflow/rounded
  floats/exponent overflow, invalid JSON/arguments, unchanged get outputs, no IO,
  Python exact roundtrip and native mutate/test/inspect/evaluate with hidden-state
  canary and authorization denial. No port heuristic adoption or fixture edits.
- Native protocol limitation explicitly pinned: evaluator Err is rendered in
  `eval_results` as `= Err(json_kind...)` while `has_errors=false`; this is NOT a
  correctly flagged error status or a successful kind String. Program snapshots
  remain unchanged. The legacy mutation_results vector also carries eval reports,
  so emptiness of that vector is not a mutation test. Generic protocol untouched.
- Targeted json_kind **7/7**, json_get regressions **5/5**, frontend catalog **7/7**,
  registry **42/42**. Single full gate **1808 passed, 2 failed, 2 ignored; CLI5/5**.
  Failures are unchanged corpus and deliberately retained #105 port presence gate.
  Log `tool_18a4a8d3c001x6vHUIJ7ODvyoG`. Corpus54/573 remains **116/58/399**,
  other buckets zero, no causal113→116 attribution. No deployment/port-completion
  claim; adoption is a separate subsequent scope after primitive review/checkpoint.
- New untracked review files: eval/json.rs +json/tests.rs, builtins/json.rs,
  python_frontend/json_kind_tests.rs, api/native_tools/json_kind_tests.rs.
  Frozen/port fixtures and their tests are byte-unchanged from a731d12. Manager
  owns checkpoint after independent review; no writer commit/push/liveIO/deploy.

### Accepted checkpoint `a731d12` — partial text-routing RED evidence

- Five more original functions: is_admin/is_allowed and the three text handlers.
  **24/42 originals** now have fixtures; **18 originals remain pending**.
  `process_text_update` is a separately ledgered NEW partial entry, not counted
  among 42 and not a replacement/stub for still-pending process_update. No runtime,
  frontend, permission, callback dispatcher, original-corpus or wrapper changes.
- Handlers return TgTextTurn: NoImmediateReply is not delivery success; nested
  DeliveryAttempt preserves the exact TgSendAttempt. Exact AST prefix comparison
  preserves original takeover context/profile/callback/metadata construction
  (only format_sender qualification differs). Corrected-source/Python combined
  AST and idempotence pass. Named union payload references retain their canonical
  Struct spelling but resolve to the declared union; actual nested values pass.
- Coverage is explicit: unchanged 49-case direct-VM pure gate, separate normal
  shared-state contracts (with direct VM refusal controls and restored fake state),
  original IO contracts and NEW-entry tests. Both-origin intercepted real
  IoRequest::LlmTakeover tests verify private/group/supergroup, admin/allow-list
  policy, sender != chat, structured from.id principal, forged name/embedded-prefix
  controls, profiles/callback arguments, immediate outcome propagation and empty
  reply behavior. No model, remote or callback execution/ACK is certified (#104).
- NEW-entry dynamic guards use List.contains/Value::matches to prevent Bool true,
  integral Float or numeric String becoming an administrator Int. Nonzero chat,
  positive sender and explicit nonnegative integer date precede authorization.
  Huge/nonnumeric/null/missing/object/array inputs fail/ignore before IO without
  panic; date 0 is admitted. Stub positive controls prove guards precede auth
  helpers. This depends on runtime identity semantics, not sound JSON static typing.
- Text null/object/array values collapse into Strings in json_get. The explicitly
  partial entry therefore conservatively defers text `null` or starting `{`/`[`,
  including legitimate same-looking Strings. No equivalent change is made to
  original extract_text/handlers. This heuristic is not a complete raw-text policy.
- **[#105](../../issues/105) gate intentionally RED:** absent media and present
  malformed `""` both yield identical json_get String(""). For **11 media keys ×
  both origins**, present-empty currently emits **1 takeover versus required 0**.
  The absent control is admitted. No exemption, regex workaround or narrower
  success claim hides the failure; new entry must not be deployed.
- Targeted port **29 passed / 1 failed**. One full gate **1801 passed, 2 failed,
  2 ignored; CLI 5/5**. Failures: strict corpus and #105 presence gate. Log
  `tool_18a249871001Lpi5gnJ40TGwiq`. Corpus unchanged 54/573, **116/58/399**,
  other buckets zero. New review files: `telegram_port_tests/routing.rs` and
  `routing/tests.rs`. Existing source/wrapper/pure expected values remain intact.
- json_kind is **design accepted only**, not implemented: pure closed type labels,
  same resolver/path semantics as json_get, absent vs null, actual signed-i64
  integer vs Float, explicit malformed/path/out-of-range error behavior. A separate
  reviewed primitive should replace NEW-entry identity/text heuristics and turn
  the RED presence gate green; no generalized coercion or parallel resolver.
- Authored manifest/origin/Python blobs recorded now:
  `d265ace303facc296d39a27f10a1c902620bdbe9`,
  `0b9503e9ed39bea2444bda76504f973dbb68842e`,
  `bc0f301bed9b2ed23d29c2f60e1017f3afc6c649`.
  #91/#99/#104 and full port/deployment remain separate. Await evidence review;
  manager owns checkpointing. No live IO, credentials read, writer commit/push/deploy.

### Accepted checkpoint `d9b8a5f` — two-function reply closure

- Fixtures/tests/docs only: add send_message and send_reply, bringing TelegramBot
  to **19/42 original functions** (15 pure, 4 IO), with **23 functions pending**.
  Both preserve parameters but deliberately change return to TgSendAttempt and
  effects to io/fail/async, explicitly await their qualified callee and propagate
  the outcome unchanged. Changes are separately ledgered semantic repairs; no
  additional retry or runtime/frontend/permission change is introduced.
- send_message retains the exact existing TgFormat formatting and 3800 split
  limit; #99 is not fixed or certified. send_reply preserves destination parsing.
  Combined-module production signatures, exact whole-program AST equality and
  print/re-lower/idempotence pass; Python runtime fixtures lower both TelegramBot
  and TgFormat from Python, rather than substituting the `.ax` dependency.
- Both-origin normal eval_async_function fixtures assert exact HTML/body/headers,
  negative and leading-zero destinations, Sent/Rejected/Uncertain propagation,
  confirmed-rejection fallback order, no caller retries, malformed terminal errors
  and zero requests for empty text/invalid destination. Only short inputs and
  fake token/intercepted requests are used. Existing frozen wrapper fixture and
  frozen corpus are unchanged; no exactly-once remote delivery claim follows.
- **[#104](../../issues/104) remains an integration blocker.** Read-only source
  inspection: main.rs spawned-task paths near 1373/1783 turn every evaluator
  Ok(val) into Completed(format!(val)); await_spawned_callback_with_timeout50–76
  accepts all Completed values; optional callback dispatch129–131 returns
  `notified`. Tagged Rejected/Uncertain would be misclassified as success there.
  This slice certifies callable outcome propagation only, **not callback delivery
  acknowledgement**. Dispatcher/routing edits are out of scope; later design must
  be transport-independent/legacy-compatible, not parse formatted strings or
  hardcode Telegram tags in the generic runtime.
- Targeted port **22/22**. One full gate **1794 passed, strict corpus alone failed,
  2 ignored; CLI 5/5**. Log `tool_189f1b88b001aPvQjLSfFWbeRq`. Corpus remains
  **116/58/399** over 54 files/573 entries, other buckets zero; no 113→116
  attribution. New review file:
  `telegram_port_tests/send_chain/runtime/reply_closure.rs`.
- Authored current manifest/origin/Python blobs respectively:
  `9825305b5c72096fec8ea65bf9fab5f85cab795c`,
  `352e9a45e95e969da5479baef7340e5af3215bac`,
  `a4cf09e55a087f3d98c93f8073b7c5b976ffb631`.
  Routing/media/lifecycle, remaining shared declarations and original stored-test
  surfaces remain pending; #91 repair is separate. Manager owns checkpoint after
  independent review. No live IO, writer commit, push or deployment.

### Accepted checkpoint `b3cf08e` — corrected send-chain policy

- Fixtures/tests/docs only. TelegramBot now has **17/42 original functions**:
  15 pure helpers plus repaired send_message_part/send_message_parts. New
  `TgSendAttempt` is an application union with Sent/Rejected/Uncertain variants,
  each carrying String; it is not Result and does not change runtime unwrapping.
  The original sanitized empty telegram_token shared declaration is copied
  exactly; tests inject only `fixture-token` in isolated state and intercept IO.
- The two repaired functions deliberately change return type (String→union),
  effects (`io,fail,async`), control flow and docs, preserving input parameters.
  `semantic_repairs` records these separately from qualification-only changes.
  Corrected-origin/Python whole-program AST equality, print/re-lower/second-print
  identity, union schema and runtime-argument validation all pass with the existing
  frontend. No runtime/permission/typing permissiveness or user-Result-await
  relaxation was introduced. Earlier frozen-wrapper tests are untouched.
- Builtin http_post Result is explicitly matched. Only actual parsed Bool true
  yields Sent; Bool false yields Rejected and authorizes **one** plain fallback.
  Any non-Sent after that optional fallback terminates remaining chunks; failure
  cannot be masked by later success. Transport/missing/nonboolean status yields
  Uncertain. Malformed JSON raises a terminal parse error (not a uniform tag).
  Empty/invalid index yields Uncertain with zero requests; forged legacy `last`
  data cannot manufacture Sent. Outer await rebinding and empty wildcard-arm
  execution are proven by actual normal-route tests, not just syntax inspection.
- Tests invoke production `eval_async_function` on **both origins**, assert raw
  Union results and exact URL/fake-token/body/header/order/count. Cases include
  whitespace, Bool versus strings/numbers/null/objects/arrays/missing status,
  malformed JSON on first/plain attempt, transport error containing `ok:true`,
  second rejection/uncertainty, multiple chunks and early termination. These are
  local synthetic attempts, **not exactly-once remote-delivery evidence**. All
  chunks are explicit and short; #99 remains unfixed and no service-limit-safe
  splitting is claimed. json_get's wrong static String catalog remains explicit;
  the rendering discriminator is supported by runtime type-path tests only.
- Targeted port **20/20**, including 5 new send-contract tests. Pure paired cases
  are now **49 TelegramBot / 38 TgFormat**, passing both origins and tree/direct
  VM; the two effectful functions have separately required normal IO coverage.
  Single full gate **1792 passed, strict corpus alone failed, 2 ignored; CLI 5/5**.
  Log `tool_183bdc377001I5xt3mp0AYgNsT`. Frozen corpus remains **116/58/399** over
  54 files/573 entries, other buckets zero, no 113→116 causal attribution.
- Remaining: **25 TelegramBot functions**, eight original shared declarations,
  module doc/startup/route/original stored tests and higher send/reply/intake/media
  integration; TgFormat original stored tests remain pending. #91 repair and full
  port/deployment acceptance remain open. New untracked review inputs are
  `telegram_port_tests/send_chain.rs` and `send_chain/runtime.rs`.
- Current authored blobs (recorded now): manifest
  `b2c45fada813ac94c8fb68c2dfc96f4f99cbf9d1`; Telegram origin
  `f04115ffb50dd85be2f4b4562ca4a66c14b37d80`; Python
  `a77449e0fbae33da057f765ba30c26d3522ad538`; paired tests
  `fae6cf3542ac0092b6e813d4725f44040e711690`. Frozen originals/provenance untouched.
  This bounded application-policy slice awaits independent review; manager owns
  checkpointing. No live IO, credentials read, push or deployment.

### Accepted checkpoint `862ccc2` — pure port closure expansion

- Fixtures/tests/docs only. **14/42 TelegramBot functions** and **18/18 TgFormat
  functions** are now in the separate corrected-origin/Python pairs. Telegram's
  added pure helpers are parse_id_list, extract_sender_name, speaker_principal,
  source_metadata. TgFormat's module doc is included unchanged. No frontend or
  runtime semantic edits, no send-path repair, no stubs and no frozen-corpus edits.
- New test-only `telegram_port_tests/qualification.rs` applies only the approved
  local-call qualification to cloned frozen function ASTs, then checks exact
  equality ignoring typed NodeIds. Per-function site counts are ledgered and
  verified: **37 TgFormat sites, 5 TelegramBot sites**. Corrected-origin↔Python
  AST equality, print/re-lower and second-print byte identity pass. Qualification
  intentionally changes source spelling/resolution; this is not frozen-source
  parity or a production global name-normalization change.
- Every implemented function has a paired assertion. **46 TelegramBot cases**
  and **38 TgFormat cases** pass on both `.ax`/Python origins and tree/direct VM,
  including code fences/HTML escaping, Unicode, split boundaries, ID-list loops,
  media metadata and principal-from-from.id despite spoofed names/text prefixes.
  VM result checks now use the existing Value comparator instead of potentially
  ambiguous display-string equality. No existing accepted expectation was changed.
- Two new characterizations preserve inherited quirks, not desirable contracts:
  missing message_id becomes quoted-empty content because to_string uses Value
  Display; missing date errors exactly `to_int: cannot parse '""' as integer`.
  **[#99](../../issues/99)**: 5000 ASCII content characters remain one part at
  limit 3800, plain (5000 raw characters) or pre-wrapped (5011 raw HTML characters).
  Content alone exceeds the documented Telegram 4096 cap. Splitting is **not
  service-limit safe**; no live Telegram rejection was measured or sourcefix made.
- Targeted port **15/15**. Single final full gate **1787 passed, strict corpus
  alone failed, 2 ignored; CLI 5/5**. Log `tool_18380c2f10017KYq3nwwlNt5B6`.
  Frozen corpus remains 54 files/573 entries, **116/58/399**, other buckets zero;
  no causal/per-entry 113→116 attribution. Full parity and deployment remain open.
- Remaining: **28 TelegramBot functions**, its module doc/shared/startup/route
  surfaces, original stored tests for both modules, full intake/takeover/media/
  lifecycle/delivery integration. #91 voice repair still unimplemented. TgFormat
  function coverage is not complete original-module surface coverage.
- Send-error repair remains **design only**: explicit builtin Result match and
  trusted app outcome tagging; success only actual parsed Bool true, one plain
  fallback only actual Bool false. Uncertain transport/malformed/nonboolean status
  or rejected plain fallback stops remaining chunks. This is a deliberate future
  corrected-origin policy change. `to_string(json_get(...))` is not proof of sound
  static typing; actual Bool/string/number/null/missing/malformed type-path tests
  are required. No global user-Result-await loosening or exactly-once remote
  delivery claim. Current send wrappers remain unchanged and still error on Err.
- Authored now (not historical baseline proof): manifest blob
  `b345d790bf9507a8de31972c4b3f460c2eea0fa3`;
  Telegram origin/Python/tests blobs respectively
  `1934b33dc355019d12d2470a396e70bde6e6f507`,
  `99426da7d97dc7a5cbf589c506c1a77cfbabfcb0`,
  `df9c957e03d4370e54e17821204c33be0622445f`;
  TgFormat origin/Python/tests blobs respectively
  `29894ad367edb5650f6b37f737c87e370200868e`,
  `6125a46151eed1471f5b79ca373dd718149040f2`,
  `149b15b438270d085e9c2b7fa667dddd59100b64`.
  Frozen original blobs remain pinned/verified in the manifest. This slice awaits
  independent review; manager owns the next checkpoint. No live IO/push/deploy.

### Accepted checkpoint `e5337b0` — #82 dispatch admission, baseline `a2551b7`

- Manager checkpoint `a2551b7` contains the previously accepted compiler, partial
  port evidence and narrow #98 repair. This new slice changes production only in
  `vm/await_admission.rs` plus its declaration and first-line check in
  `vm::compile_function`. The check runs before Compiler construction, argument
  slot allocation or root bytecode emission; no VM instruction has executed.
- An iterative worklist exhaustively visits executable StatementKind/Expr slots:
  call arguments (including nested calls), let/set/return/yield values, call/
  spawn/await targets, checks, branch conditions/both bodies, while/each
  conditions/iterators/bodies, match subjects/all arms, record values, field bases,
  unary/binary operands, source timer expressions and emitted event payloads.
  Source handler names are future-registration metadata, not immediate calls.
  A reachable Await resolving to a user function produces a categorical compile
  error selecting the existing tree path before execution. Unknown targets and
  builtin-only graphs retain their existing behavior.
- Function identity is a pointer used solely as a visited-set key within the
  borrowed immutable snapshot: no dereference, persistent cache or cross-revision
  reuse. The supplied root is scanned directly even if owned outside/unregistered
  in Program; callees resolve through the original spelling and `get_function`.
  Same-named functions in different modules are distinct identities. SCCs terminate;
  caller-module-first rewriting is forbidden. False-positive graph edges for
  builtin/variant collisions may conservatively choose tree, never redirect calls.
- `eval_async_function`/CoroutineHandle/`io_performed`/continuation and replay
  guards are untouched. Whole-root admission prevents an earlier builtin effect
  being executed before discovering a transitive user await. Tests cover exact
  request order/count before/after IO and in nested arguments, child failure with
  no later effect/replay, original mocks/stubs, guest denial before mocks/requests,
  and bound turn identity. These do not certify every historical permission issue.
- Normal frozen-wrapper dispatch now matches the existing explicit tree diagnostic
  for both bare and qualified targets: one request then bare String on success;
  one request then `index_of expects (String, String)` for transport Err. Only
  normal-route Rust test expectations changed; source fixture/provenance bytes and
  direct-tree diagnostics are preserved. Wrapper unboxing/error handling, #97,
  and the Python Result-returning-user-await restriction are **not** changed.
- Performance tradeoff: any syntactically reachable user Await, even in an untaken
  branch, makes the invocation use tree (including pure async user calls). A
  synthetic 128-function recursive SCC with repeated edges visits **1025 work
  items**; **1000 scans took 23971 µs** in the final author targeted release test
  (~24 µs/scan, with test counters). This is only synthetic scan cost, **not** a
  production latency benchmark. Lazy compiles may repeat scans; no global cache.
- Author targeted: user-await 13/13, port 13/13, existing VM 41/41, native call-site
  8/8, scoped 20/20 and unchanged no-user-await post-IO division/no-replay 1/1.
  Single final full gate: **1785 passed, strict corpus alone failed, 2 ignored;
  CLI 5/5**. Log `tool_1834efc94001YdvI1XigmzXSvF`. Corpus remains 54/573 with
  **116/58/399**, other buckets zero. This slice awaits independent review.
- Review inputs: new `vm/await_admission.rs`, `vm/await_admission/tests.rs`,
  `vm/await_admission/tests/dispatch.rs`; two integration lines in `vm.rs` and
  the normal-route wrapper assertions. Current wrapper Rust test blob (recorded
  now after this change): `6ada414dd8737af07be8ccb60b9b951f415a0a43`; the older
  authored-hash table below is the prior checkpoint, not a claim of current bytes.
  No writer commit, push, deployment, live IO or frontend policy relaxation.
- Correction to an earlier binder report: `eval::input_to_vm_args` is
  **pub(crate)**, not private, but accepts named fields rather than lower_tests'
  multi-argument `_index` carriers. The richer canonical tree binder is private.
  The existing tested port adapter is unchanged in this dispatch slice.

### Accepted checkpoint `a2551b7` — narrow #98 repair

- `vm/equality.rs` routes only differing Value variants, excluding Int↔Float
  numeric pairs, through the tree's shared `Value::matches` and its negation.
  `matches` is minimally `pub(crate)`. VM same-family/mixed-numeric `binary_cmp`
  and all ordered opcodes are unchanged. No empty-string special case, json_get
  coercion, frontend typing change or JIT change was introduced.
- New both-order heterogeneous Eq/Neq matrix covers Int/String (empty and numeric
  text), Float/String, Bool/String, None/String, Ok/Err/Union versus String and
  Bool/Int. Same-type scalar controls and ordered mixed-type errors remain.
  Existing residuals are deliberately characterized separately: VM Int↔Float
  equality coercion differs from tree; NaN/identical-infinity Neq remains unequal
  between tiers; same-family Result/Union/None equality remains unsupported in VM.
  These #90/#86 residuals prevent a claim of full equality parity.
- **Unmodified port assertions now pass:** TelegramBot **28/28 on both origins
  and both tree/direct VM**, TgFormat 7/7 likewise. The first-slice RED gate was
  not exempted or rewritten. Frozen originals, port `.ax`/Python/test files and
  manifest remain byte-identical. Their README/observed-status metadata preserve
  the earlier RED-evidence checkpoint; this recovery header records the current
  pending runtime correction. Frontend json_get's String catalog mismatch remains
  unresolved; #82 user-await/wrapper behavior is unchanged.
- Targeted equality 3/3, port 13/13, existing VM 41/41. Final author
  `cargo test --release --no-fail-fast`: **1772 passed, strict corpus alone failed,
  2 ignored; CLI 5/5**. Full log `tool_18326f8420017vJBiGyyjCQ0pJ`.
  Corpus remains **116/58/399** across 54 files/573 entries; other buckets zero.
  No attribution for 113→116. `git diff --check` clean.
- New untracked `src/vm/equality.rs` must accompany the cumulative untracked
  compiler/port evidence files at review. No commit/push/deployment. Narrow #98
  repair received independent bounded ACCEPT (VM 47/47 and port 13/13); the preceding port slice was accepted
  only as RED evidence, not as a complete functional port. #82 dispatch design
  is next after this review, based on the preserved wrapper diagnostics.

### Authored port evidence hashes — recorded now, 2026-10-28

These are Git blob hashes computed from the **current authored working-tree
files**, after the bounded #98 review. They are not a claim that these new files
existed in baseline `05d85cc` or any earlier historical checkpoint. The source
provenance hashes in the port manifest separately identify the frozen originals.
Paths below are relative to `tests/fixtures/telegram-python-port/` unless noted:

| Authored file | Git blob |
|---|---|
| `manifest.json` | `9346a60cdf9ffa3407e8216b1aed501b80b839d6` |
| `origin/TelegramBot.ax` | `617280c450eb209ab3b38d8405ef4c7427bee2f3` |
| `origin/TgFormat.ax` | `a95ac4d862f69822d7617ef7300f01a0e78bf33e` |
| `python/TelegramBot.py` | `4281f8934c8957e5da14455602e3996c4f704c47` |
| `python/TgFormat.py` | `00e5b1e58312c235885d278e8a9cba17a1baf1b7` |
| `python/test_telegram_bot.py` | `0fd159ab44d1444b064a679af0d9a4e4eb2cf9c4` |
| `python/test_tg_format.py` | `61aa42fc491640f34cd2c05e914002f6b5b91068` |
| `src/python_frontend/telegram_port_tests.rs` (repo-relative) | `8b43c5142f8169bb6b763216f4f3820e68b01e28` |
| `src/python_frontend/telegram_port_tests/io_contracts.rs` (repo-relative) | `2db7a4837269ce27460184078bedb9098669ddee` |
| `src/python_frontend/telegram_port_tests/wrapper_contract.rs` (repo-relative) | `87313e911ca403c1a0b3d5594837d8b9b4beabcb` |

### Accepted earlier checkpoint — first partial port, RED evidence gate

- Tests/fixtures/docs plus minimal test-module wiring only; the independently
  accepted compiler slice below was preserved. Separate files under
  `tests/fixtures/telegram-python-port/` contain **10/42 TelegramBot functions**
  and **3/18 TgFormat functions**, `.ax`/Python pairs, literal paired tests,
  README and manifest. Exact source Git blob hashes and baseline are pinned;
  implemented/pending sets account for every original function, with a
  per-function transformation ledger. Frozen corpus files/manifest are untouched.
- Selected bodies/docs exactly match their frozen function ASTs. Corrected partial
  origin→Python AST equality and print/re-lower/second-print identity pass. Source
  declaration selection/order and new test-data provenance are disclosed. No
  stubs, #91 voice repair or full module/shared/lifecycle/route port is hidden.
- **[#98](../../issues/98): keep direct-VM port gate RED.** All 28 TelegramBot
  tree cases pass on each origin; direct VM passes 25 and fails cases 1/3/4:
  numeric chat/sender IDs compared to empty String. Actual `json_get` scalar
  variants are String/Int/Float/Bool (null/object/array have String encodings),
  unlike the String catalog declaration. Both source origins fail identically;
  no coercion, ignore list or source rewrite hides this runtime discrepancy.
  TgFormat's 7 cases pass both origins and both tested tiers.
- Six contract tests cover production CoroutineHandle requests/replies and actual
  JSON variants using in-memory replies, with no real file/sleep/network/provider
  work. This is valid-call runtime contract evidence, **not complete frontend IO
  signature/arity support**. Missing IO table entries remain a #86 gap.
- **[#82](../../issues/82): wrapper diagnostic corrected and pinned.** Normal
  `eval_async_function(send_message_parts)` emits zero requests and fails unknown
  user-await operation for both bare and qualified targets. It is not a
  qualification-only bug. Qualification-only test-copy changes are inverse-AST
  verified. Explicit direct-tree diagnostic reaches the boundary: successful
  `send_message_part` returns Ok(String), user await unwraps it, caller succeeds
  after one request; transport Err remains wrapped and caller fails at `index_of`
  after one request, before fallback send. Normal-dispatch success is not claimed.
- Final author gate: **1768 passed, 2 failed, 2 ignored; CLI 5/5**. Failures are
  the unchanged strict corpus and explicit direct-VM port gate. Targeted port
  tests **12 passed / 1 failed**. Full log: `tool_1831c9387001ynVtLFOgkJ2K6P`.
  Strict corpus remains 54/573, **116 verified / 58 invalid / 399 unsupported**,
  other status buckets zero, still no causal/per-entry 113→116 attribution.
- New untracked review inputs: `src/python_frontend/telegram_port_tests.rs`, its
  `io_contracts.rs` and `wrapper_contract.rs` children, and the whole separate
  fixture directory. Include earlier untracked `compiler/truthiness.rs` and
  `python_frontend/condition_tier_tests.rs` when reviewing the cumulative cut.
  No commit/push/deployment/live IO. First port evidence awaits independent review;
  full corrected-origin port, #86/#90/#96/#97/#98/#82 remain open.

### Accepted compiler checkpoint — immutable normalization (uncommitted)

- Built on the accepted NOT checkpoint below. All JIT Boolean consumers now use
  one representation-aware normalization: AND/OR, NOT, Check, Branch and While.
  Input evaluation is preserved even for always-truthy Float; branch results are
  canonical I8. Identity-safe homogeneous scalar comparisons are admitted for
  conditions; NOT retains its narrower binary/argument-call refusal contract.
- Review correction: **Float Eq/Neq now refuse condition compilation**, including
  nested conditions, rather than emitting IEEE equality for the runtime's epsilon
  comparison. Near-equal finite values, signed zero, infinities from scalar calls
  and NaN AST cases have explicit refusal coverage; finite epsilon fallback is
  separately verified through the normal evaluator with `used_jit=false`.
  Float ordered comparisons and Bool/Int Eq/Neq remain direct-JIT controls.
  **OPEN #90 residual:** VM Neq uses `abs(a-b)>=EPS` while tree uses
  `!(abs(a-b)<EPS)`; NaN/identical infinities therefore disagree. JIT refusal
  does not fix that VM fallback discrepancy. Global float equality is unchanged.
- Manager-approved exception preserves the two existing direct-JIT
  `runtime_semantics_tests.rs` division-program gates: condition calls may take
  exact Bool/Int/Float arguments, emitted recursively from proven values in
  authored order. No String/unknown conversions or IO/async callees. Explicit
  `[fail]` callees in the new argument-call path, including nested arguments,
  refuse compilation until [#97](../../issues/97) is resolved. Zero-argument
  fallible calls retain their existing **tier-specific trap characterization**;
  a scalar signature alone does not prove the runtime returned wrapper/value,
  and caller admission never certifies all callee internals.
- Local/rebound/unknown conditions refuse compilation before execution, including
  mutable-counter loops. The normal evaluator falls back to VM/tree; tests assert
  the result and `used_jit=false`, not fake direct-JIT evidence. No mutable-binding
  machinery or general compiler namespace refactor was added. Coarse admission
  still accepts more shapes than final compilation.
- Author final `cargo test --release --no-fail-fast`: **1756 passed, strict parity
  alone failed, 2 ignored; CLI 5/5**. Corpus remains **116/58/399** (54 files,
  573 entries; other status buckets zero), no 113→116 attribution. Full log
  `tool_182f4ee0c0012k4GVf4G1AEFwc`. Targeted: condition tests 6/6, compiler 41/41,
  VM 41/41, runtime semantics 7/7, logical 17/17, native no-replay 1/1. Final
  `git diff --check` clean. Independent Astra ACCEPT followed, including release
  build, condition 6/6 and direct refusal/ordered controls; no whole #90/#86 or
  deployment acceptance follows from that bounded review.
- New untracked files to include in review/durable cut: `src/compiler/truthiness.rs`
  and `src/python_frontend/condition_tier_tests.rs`. #96 remains open; direct
  nested-call tests use current-executable children with deadlines, not fallback.
- **[#97](../../issues/97) author-observed bug, not intended semantics:**
  `fail_bool()->Bool [fail]` fails a Check; caller `probe()->Int [io,async]`
  then attempts `file_exists`. Normal `eval::eval_async_function` runs the VM:
  caller performs **one subsequent effect**, whereas direct tree evaluation
  aborts with **zero**. VM Check currently emits Err+Return from its callee frame;
  the caller continues. `trusted_ax_failed_check_caller_effect_characterization`
  pins this discrepancy separately from normalization correctness. It must not
  be cited as acceptable failure propagation or all-tier parity. A separate
  trusted `.ax` fixture verifies that an actual VM division error after one IO
  request does not replay that request. Neither is evidence of JIT/Python IO
  admission. No #82/#94 guards were loosened. #90 and #86 both remain open.

### Current accepted-but-uncommitted checkpoint

- Durable baseline: `05d85cc`. The pending frontend Boolean-admission/warning
  cleanup plus narrow representation-safe JIT NOT slice has **independent Astra
  ACCEPT**, including the subsequent String and scalar scratch-collision repairs.
  Runtime code is frozen pending the next approved design; no commit, push or
  deployment has been authorized here. Acceptance is **not** whole #90/#86 or
  deployment acceptance. Full sanitized Python TelegramBot port is still absent.
- Author final `cargo test --release --no-fail-fast`: **1750 passed, 1 strict
  parity failure, 2 ignored; CLI 5/5**. Targeted logical suite **17/17**.
  Final author log: `tool_182cd0c46001A7J44EJkArxih6` under the local OpenCode
  tool-output directory. Reviewer independently passed release build and 17
  logical tests, and direct CLI String/scalar collision cases returned **1/0**.
- Corpus: **54 files / 573 entries; 116 verified, 58 invalid, 399 unsupported**;
  input errors / AST mismatches / non-idempotent entries all zero. There is **no
  verified per-entry or causal attribution for 113→116**. Zero AST mismatches
  among admitted entries is not full feature coverage.
- Current sessions/models (all explicitly confirmed `openai/gpt-6-astra`):
  manager `ses_e8d6d9d1bffe8G7KEI4yUw8DPi`;
  author `ses_e8b17a23affedgH4V52JdwxNDL`;
  independent reviewer `ses_e82647931ffewdBFqa9bGCmPIz`.
  Earlier provider/reviewer history is historical, not current routing.
- **Include new untracked `src/compiler/truthiness.rs`** in any later authorized
  durable cut; ordinary tracked-file diff summaries omit it. It holds NOT's
  proven-operand emission and mechanically extracted, still-unfixed AND/OR.
- **#90 remains open:** AND/OR and Check/Branch/While normalization is the next
  design-only bounded step. The NOT slice deliberately refuses binary operands,
  argument-bearing calls and unproven/rebound values before execution, retaining
  tree/VM fallback; zero-argument scalar callees are not globally certified.
- **#96 remains open:** parallel-suite JIT relocation panic, documented below.
  Child isolation of the regression avoids the observed interference; it does
  not fix production relocation or establish its root cause.

### Authorized next scopes — compiler slice, then TelegramBot port

- Next #90 implementation is shared normalization for AND/OR and
  Check/Branch/While using **representation-proven immutable** operands. Mutable
  condition dependencies must refuse JIT compilation before execution and use
  existing VM/tree fallback; no tracked mutable-binding write-path extension is
  authorized. Inventory found no direct-JIT loop execution test: compiler.rs
  `compilable_body_with_while` tests only coarse admission, and VM loop tests are
  bytecode evidence. This distinction must remain explicit. Existing direct-JIT
  short-circuit user-call tests in `runtime_semantics_tests.rs` must be retained.
- Immediately after that bounded slice is frozen/handed to independent review,
  the sole writer's priority becomes the TelegramBot port, ahead of optional JIT
  work. [#86](../../issues/86) authorizes a **separate reviewed corrected sanitized
  `.ax` origin plus complete paired Python port**, with provenance and an explicit
  transformation ledger. The frozen parity corpus stays untouched; exact AST and
  idempotence for the corrected origin form a separate port gate, never a claim
  that the original fixture or whole parity gate passed. No runtime async or
  permission guard loosening, new builtin, live Telegram/network/credentials or
  deployment is authorized.
- Concrete port transformations/blockers (line numbers reference the frozen
  `tests/fixtures/python-parity/library/TelegramBot.ax`, not a live library):
  1. Bare local calls throughout (`is_admin:64`, `process_update:234`, startup
     `spawn poll_loop:21`, route `handle_update:25`) have no exact Python spelling
     under `python_frontend/printer.rs`'s bare-call refusal. Qualify calls in the
     **new corrected origin**, recording the deliberate dispatch/AST change.
  2. Repair [#91](../../issues/91)'s malformed `transcribe_voice:410–425` nested
     match/duplicate Err/out-of-scope `e` in that new origin. Original fixture
     remains invalid; repair is not evidence its original AST was representable.
  3. `send_message:442–445` and `send_message_parts:605–613` return async calls
     directly. Rewrite to annotated await-bind-then-return in the new origin;
     `python_frontend/call_sites.rs`'s async gate remains intact.
  4. [#94](../../issues/94): `send_reply_with_attachment:457–466` directly matches
     `http_upload` after awaiting `send_message`; `transcribe_voice:419` does so
     after binary download. Use direct-builtin await-bind-then-match in the new
     origin and test no replay after prior IO. Address #82 only if a concrete
     port call awaits a Result-returning **user** function, not as a blanket
     prerequisite for fallible IO builtins.
  5. Verify actual IO arity/wrappers for startup `read_file`/`file_exists:13`,
     `poll_once`/`http_get:84`, `process_update`/binary Attachment downloads:234,
     takeover metadata/attachment overloads in `handle_*:323–375`, and
     `send_message_part`/`http_post:596`. Pure `builtin_signatures.rs` inventory
     does not establish these. Use `runtime_arguments.rs`, `call_sites.rs` and
     production coroutine dispatch with trusted in-memory IoRequest responses.
     Current IO typing is **not complete**: `expressions.rs` special-cases
     `read_file -> String` and `http_get -> Result<String>`, while many required
     IO results remain Unknown. `runtime_arguments.rs` checks exact user-call
     arity but treats bare IO as builtin/effect checks, not a complete signature
     table. Do not add broad Unknown acceptance or call this strict IO coverage.
     Actual coroutine `http_post` returns Result<String>; frozen
     `send_message_part:598` binds it as String, then its user-function result is
     consumed by `send_message_parts:609–610`/`telegram_response_ok`. The first
     port scope must test the actual direct-builtin wrapper **and intervening
     user-await boundary** before diagnosing or changing that binding. It is not
     an ignored result; do not silently unbox it or count a correction as original
     AST parity. `llm_takeover` metadata overloads likewise require exact request
     field assertions, not only an await-mode acceptance test.
  6. Build missing complete Python source and paired tests: start with a port
     skeleton, pure `extract_*`, `source_metadata:202`, `speaker_principal:191`,
     `format_sender:140`, `telegram_response_ok:587` and TgFormat dependencies.
     Then raw-update admin/guest/group routing through `process_update:234`,
     takeover/callback binding, photo/video/voice flows, split/retry order
     `send_message_parts:605`, and bounded startup/poll/shared/spawn coverage
     `13,68–108`. Existing `bundled_modules_tests.rs` and `api/takeover_native`
     are harness entrypoints, not substitutes for executing the Python port.
- These port tasks can progress on tree/VM with trusted mocks; optional mutable
  JIT support and open #96 relocation are **not** port prerequisites. Port code
  must not be interleaved into the current compiler slice.

### Historical recovery ledger — superseded snapshots, not current status

The entries below retain intermediate measurements, rejected claims and later
corrections. Words such as “current”, “awaiting review” or “not implemented” in
this ledger refer to that earlier checkpoint; the header above is authoritative.

- Full feature equality is **not achieved**. Last pre-foundational-fix gate:
  54 fixture files / 573 inventory entries; 114 verified, 58 invalid-program
  entries (4 roots plus descendants), 329 unsupported, 72 exact AST mismatches.
  Release tests: 1662 passed, parity gate failed, 2 ignored; CLI 5/5 passed.
  These are historical measured counts, not current-fix verification or file coverage.
- Foundational steps 1–2 now reject non-exact bare-call inspection, share full
  structural comparison after typed NodeId clearing, and restore known operand
  checks. Latest `cargo test --release python_`: **130 passed, 2 failed** — the
  corpus gate and the separate new owner-colliding-type native regression.
  Corpus remains 54 files / 573 entries: 90 verified, 58 invalid, 425 unsupported;
  zero AST mismatches now means earlier refusal, not fixed representation gaps.
  Full release suite has not been rerun for this foundational delta. Target slot
  yielded to TUI after the focused run; no main builds until manager handoff.
- After the next manager handoff, bounded owner-aware catalog work fixed the
  duplicate-type regression. Current release gate: **1672 passed, corpus parity
  alone failed, 2 ignored; CLI 5/5 passed**. Corpus counts remain 90 verified /
  58 invalid / 425 unsupported. Internal type identities now retain declaration
  and signature owners without changing persisted AST names. Both-order native
  struct/recursive-union inspect/resubmit and hidden-competitor tests pass.
  This slice awaits independent review before the native-reference pass.
- Review found a hidden-schema risk in filtered unique fallback after owner
  erasure. The bounded pre-executor binding guard now checks full-state metadata
  (owner/kind/order only), including global union reclassification, and refuses
  colliding external fallback before schema validation. Native canaries and safe
  local/top-level/no-collision controls pass. Latest release gate: **1678 passed,
  strict corpus parity alone failed, 2 ignored; CLI 5/5 passed**. Corpus counts
  remain unchanged. Independent bounded re-review accepted this correction;
  whole #86 remains red and no release/deployment approval follows.
- Slice 2 was subsequently authorized and implemented (server-derived target/
  same-submission union-constructor roles, sequential lexical field scopes,
  neutral collision refusal, owner-qualified type closure): final combined gate
  **1705 passed, strict parity alone failed, 2 ignored; CLI 5/5**. Frozen for
  independent review. External bare Variants remain explicitly unsupported and
  visible. #94 after-earlier-IO direct-match limitation is documented; slice 3
  (shared defaults) is not implemented.
- #95 fixed and slice 3 implemented as the next authorized pair. #95: native
  Python mutation refuses a declared function whose spelling collides with a
  variant in any module (state becomes unreachable), while the runtime recheck is
  now scoped to the reachable call graph so legacy collisions keep unrelated
  functions usable. Slice 3: module shared declarations are structurally
  preflighted, shared-only submissions are valid, and defaults are lowered by a
  data-only walker and type-checked against the complete local catalog. Final
  combined gate: **1712 passed, strict parity alone failed, 2 ignored; CLI 5/5**.
  Frozen for independent review. Record defaults remain outside the native
  literal policy by existing design (frontend accepts, native tool refuses).
- Reviewer REJECT follow-up fixed: #95 reachable-graph walkers now recurse into
  call arguments (nested-argument consumers no longer escape the check); the
  declaration guard uses the POST-SUBMISSION catalog and checks BOTH directions
  (function-vs-variant and variant-vs-function), so single-mutate and reverse-order
  collisions are refused. Corrected a factual misreport: typed RECORD shared
  defaults are natively admitted end-to-end; only VARIANT constructor defaults are
  refused by the native literal gate while the frontend accepts them — a known
  frontend/native policy inconsistency, deliberately not widened. Qualified-call
  over-refusal is recorded as a known safe-direction limitation.
- Pending #90 frontend follow-through admits statically Bool call operands and
  comparison expressions in `and`/`or`; the prior totality restriction was removed.
  Python-source tests exercise recursion/division skipping and Bool calls in NOT
  and conditions in tree/VM. The request-count fixture is **legacy `.ax` only**:
  implicit async calls in Boolean operands are not supported Python evidence.
  No async guard was relaxed. The previous 113→116 count observation has no
  verified per-entry attribution to `05d85cc`; do not attribute all three to calls.
- Pending narrow JIT NOT correction: Bool/Int zero-test; Float always truthy
  (including ±0 and NaN), with operand evaluation preserved. Strings use length,
  never pointer-nullness. Proven forms are literals and untouched, unambiguous
  String ABI parameters; local/rebound strings and string-returning calls refuse
  compilation before execution because their length tracking is not certified.
  Direct compiled **Int-return wrappers** cover empty/nonempty/Unicode String,
  Float zero/negative/nonzero/NaN and Bool/Int controls. A separately executed,
  core-disabled child proves a Float-returning call's Check trap is not skipped.
  NOT helpers live in `compiler/truthiness.rs`; the existing AND/OR IR was moved
  there mechanically to keep compiler.rs under 2000 lines.
  Review found a real scratch collision: String parameter `_last_str` had its
  `_last_str_len` lookup overwritten by a subsequent literal. The correction
  captures the immutable entry-block ABI length Value before parameter binding,
  independently of the mutable name map; source rebinding still invalidates the
  proof for the whole function. Both collision directions have exact compiled
  Int-wrapper assertions. The older pure AND/OR test also now uses Int wrappers.
  A second review reproduced the scalar equivalent (`_last_str_len:Int` shadowed
  by literal scratch). The cohesive correction captures immutable scalar entry
  Values too and emits admitted operands recursively without generic mutable-name
  lookup: literals, untouched parameters, nested NOT and numeric negation.
  Binary operands and argument-bearing calls now refuse JIT compilation, including
  when nested; tree/VM can execute them through the existing pre-execution fallback.
  Zero-argument scalar user calls resolve the actual signature and emit once;
  their internals are not certified generally namespace-safe by this fix. No
  generic compiler namespace refactor or new name blacklist was introduced.
  Both scalar collision directions plus nested NOT/negation and diagnostic-bearing
  safe refusal controls pass; original String/Float/error-evaluation tests remain.
  Freeze checks after correction: logical suite **17/17**; full release **1750 passed, 1 strict
  parity failure, 2 ignored**; CLI **5/5**. Current corpus: 54 files, 573 entries,
  **116 verified / 58 invalid / 399 unsupported**, zero input errors, AST
  mismatches or non-idempotent entries. Aggregate counts do not establish the
  identity/cause of the three-entry increase. Test compilation still emits
  existing warnings; the earlier zero-warning statement concerned non-test build.
- **OPEN JIT relocation/concurrency evidence:** the first full parallel run of
  the new scratch-collision nested-call regression panicked at
  `cranelift-jit-0.119.1/src/compiled_blob.rs:56:80`, `Result::unwrap()` on
  `TryFromIntError(())`, while its targeted run passed. That full run had 1747
  passes, this failure and strict parity failure. The fixture now runs both exact
  compiled results in the current test executable's isolated child, with a
  60-second kill deadline; panic/nonzero/timeout fails the parent, and a completion
  marker proves both directions ran. Isolation avoids this observed concurrent
  allocation/relocation failure in the final run; it **does not fix or close**
  the underlying JIT relocation issue. No production relocation code changed.
- **OPEN #90 residual — next bounded priority after NOT acceptance:** JIT
  AND/OR still branch on raw values, use XOR-one for the AND left condition, and
  store an unnormalized RHS in an I8 result variable
  (`compiler/truthiness.rs::compile_logical`). `compiler.rs::compile_statement`
  Check/Branch/While likewise feed `compile_expr(..., I8)` straight to `brif`;
  expected type does not convert a literal/identifier's actual representation.
  Proposed minimal direct-JIT Int-return-wrapper repros: `if 2 AND true`,
  `if true AND 2`, `if ""`, `if 0.0`, `check required 0.0`, and a bounded
  `while ""` that immediately returns 1 if entered. Compare against tree/VM:
  Int 2 is truthy, empty String false, **Float 0.0 true**. Run potentially invalid
  IR/trap cases in isolated children. Shared normalization is not implemented in
  this freeze, and all-tier truthiness/parity is **not closed**. String-return ABI
  propagation, arbitrary local bindings and parser-expression evaluation also
  require separate evidence; do not infer those from the narrow NOT tests.
- #90 tree/VM semantics aligned for the covered operations. Truthiness is now a
  single public `Value::is_truthy`; the VM's `pop_bool` delegates to it so
  `AND`/`OR`/`NOT`/`BranchIf`/`BranchIfNot` share one definition (String/None/Err
  rows are `.ax`-only since the Python frontend emits Bool). VM `AND`/`OR` compile
  to short-circuit branches via a new `ToBool` opcode plus the existing
  Jump/BranchIf/BranchIfNot patching; the JIT emits branch + result variable for
  both ops (compiler.rs `compile_expr`, the And/Or arm; it previously used
  `band`/`bor` with both operands eagerly compiled). Differential tests assert
  tree/VM agreement on skipped errors, recursion and mocked IO counts; direct JIT
  coverage is narrower. The old claim that these establish general three-tier
  parity was incorrect. Earlier eager-RHS characterization tests were inverted,
  but the OPEN residual above still prevents closure.
- Builtin signature completeness (#86): new `python_frontend/builtin_signatures.rs`
  gives every pure BUILTINS name and alias a checked arity/result signature
  (grounded in the runtime's own `expects N argument` guards and return values).
  Audit tests assert catalog-to-registry agreement, full alias reachability, and
  no uncatalogued reachable pure builtin. Gaps found and fixed: `home_dir` was
  missing entirely; `map`/`set`/`concat`/`Ok`/`Err`/`Some` are special-cased
  outside the catalog by design. `str` is an Adapsis alias for `to_string` but
  stays rejected in Python source as a coercion (documented, not silently
  rebound). IO builtins are outside this catalog: they are await-validated by
  `call_sites.rs`. Reviewer follow-up (a) was initially reported as done but was
  comment-only: `check_runtime_variant`'s fallthrough call arm accepted ANY
  non-list call, so `check_runtime` was permissive despite the strict contract in
  its comment. Now that arm matches only the literal `list` builtin, so an empty
  or non-matching `variants` set makes every non-list call fall through to
  `literal()`. Unreachable in production (parse-time `authored::literal`
  pre-gates those positions) but the ordered hardening is real, with unit tests.
- Next authorized pair implemented. (1) Native literal gate widened ONLY for
  shared-variable defaults whose declared annotation is a union and whose tree
  shape is data-only variant construction (literal payloads, `list(...)`; a
  zero-payload variant uses the bare constructor spelling). Spawn bindings have no
  defaults and are not part of this widening. The slice3 walker still owns
  arity/type checking, nested calls stay refused, and the #95 function/variant
  declaration guards are untouched, so a variant default cannot smuggle a
  same-named function. Records were already natively admitted. The validator now
  union-reclassifies `+shared` annotations like function signatures, which also
  corrects `.ax` written through the same path (previously a union-typed shared
  var persisted as `Struct(name)`). Inherited caveat: the shared arm resolves the
  annotation by bare name, so a pre-existing bare union-name ambiguity is treated
  the same way as elsewhere in the frontend rather than being resolved here. (2) #92:
  `i64::MIN / -1` wraps to `i64::MIN` and `i64::MIN % -1` is `0` in tree, VM and
  JIT via shared `checked_div`/`checked_rem` (JIT routes through runtime helpers);
  zero divisors keep their previous error/trap behaviour. Pending combined gate
  and re-freeze.
- Earlier design note (superseded for slices 1-2 by the authorizations above): (1) exhaustive
  direct-call admission visitor and builtin-only IO-match exception; (2) scoped
  variant/function/value-reference classification, preserving authority-first
  lookup and bare-user-call refusal; (3) structural shared-declaration preflight
  plus typed recursively data-only defaults. Each slice needs actual native
  refusal/no-mutation regressions and independent review before broadening.
  Builtin signatures, runtime #90/#92 and the full TelegramBot port remain
  separately sequenced; current source/tool descriptions must stay truthful.
- Slice 1 was subsequently authorized and implemented (explicit root call modes,
  exhaustive executable traversal, native/direct-validator regressions). It is
  **not accepted yet**: full release run found 1690 passing / 2 failing / 2 ignored
  (strict parity plus the new native synchronous-Result-match positive test).
  Bounded differential diagnosis proves identical `.ax`/Python ASTs: tree returns
  5 for Ok(5), 0 for Err; VM and normal native routing return None because VM
  MatchVariant only handles Value::Union, not dedicated Ok/Err values. No await/IO
  is involved; this is distinct from #82. The positive assertion stays red;
  runtime correction awaits separate authorization. Slices 2–3 remain unstarted.
- #93 was separately authorized: a cohesive VM match helper now recognizes
  dedicated Ok/Err values and supplies the existing payload/bool stack protocol.
  Targeted Result-match gates pass, including unchanged native expected 5,
  original `.ax`/Python AST equality, nested match statements, Union controls
  and no IO replay. Final combined #93 + slice1 release gate: **1697 passed,
  strict parity alone failed, 2 ignored; CLI 5/5 passed**. Both deltas are frozen
  for independent review; neither acceptance nor deployment is implied. Option,
  nested payload patterns, non-exhaustive matches and JIT remain separate.
- Sole main writer: `ses_e8b17a23affedgH4V52JdwxNDL` (Astra). Manager/release:
  `ses_e8d6d9d1bffe8G7KEI4yUw8DPi`; P0 reviewer:
  `ses_e888436abffenlBGkG1Ek8bZHF`. Independent #87/#88/#89 and TUI worktrees
  remain separate. Shared main target builds are manager-coordinated.
- Approved current slice: exact typed-NodeId-only production comparison,
  rejection of unsafe call qualification/Unknown admission, native regressions,
  truthful prompt and module-lowering extraction. Owner-aware catalogs and other
  parity/authority gaps need separate bounded plans. #90 runtime Boolean behavior
  and #91 original TelegramBot source correction are independent gates.
- Required acceptance: a runnable checked-in full sanitized TelegramBot Python
  port with paired tests, exact AST/idempotence and mocked raw-update behavior
  across admin/guest/group routing, text/media/replies and lifecycle/shared/spawn.
  Diagnose the invalid original visibly; use a separately reviewed #91 correction,
  never silently repair the fixture. No live Telegram/network or permission changes.
- Complementary owner requirement: native conversation capability parity (#89),
  including a permission-checked `!opencode` equivalent, has its own implementation
  owner. The #86 TelegramBot port must test protocol/takeover interactions but
  does not absorb agent/plan/roadmap/model/mock/stub/remove tool implementation.
  Native-default deployment remains deferred until both acceptance gates pass.
- No commit, push, integration deployment or service restart authorized.

## Current execution — native tools for takeover conversations (#85)

Owner request 2026-10-25: "switch everything" to native/Python. Issue:
https://github.com/Marenz/adapsis/issues/85. Phase: **implemented offline, all
gates green; awaiting independent review, then manager commit/deploy + Telegram
smoke.** Not committed, installed, deployed or restarted.

- With `--native-tools`, write-capable non-guest `llm_takeover` turns (Telegram
  admin DMs, groups while the admin speaks) use the shared native driver
  `NativeLoop::drive` (`src/api/native_turn.rs`); `src/api/takeover_native.rs`
  only decides eligibility and renders the reply. Guests, execute-only/read-only
  speakers and processes without the flag keep the legacy protocol unchanged.
- Explicitly unavailable natively (stated in the prompt): plan, roadmap,
  `!agent`, `!opencode`, model switching, mocks/stubs, removal. The speaker block
  reports agents/opencode as unavailable for native turns.
- Fixed live bug: native requests sent `temperature: 0.7`, which
  `anthropic/claude-opus-5` rejects with 400; now omitted (regression in the
  HTTP/stream lifecycle and takeover fixtures).
- Gates: `cargo test --release` 1589 passed + 5 CLI, 2 ignored (6 new
  `takeover_native` tests); release build; `memory-eval` passed; 28 `.ax`
  examples; Python CLI examples 2/2 and 7/7.
- Remaining: independent review, deploy here, live Telegram smoke on the dev-bot
  chain (opus-5 primary). Guest native schemas are a separate later slice.

## Completed — working Python frontend (#81)

**User-approved stop gate (2026-10-25):** "your stop gate is the working python impl".
Native HTTP tools #78 are committed as `8e10876` on `work/native-tools`; the
checkout was clean before this design-stage recovery update. Build a working
Python-subset frontend through the guarded inspect → mutate → test → evaluate →
finish workflow, with the existing AST/runtime authoritative and `.ax` retained.
A standalone syntax demo or CPython `eval`/`exec` is not the stop gate.

- Issue: https://github.com/Marenz/adapsis/issues/81.
- Current phase: **#81 stage complete; stop gate met.** Code is committed locally
  as `61b3911` on `work/native-tools` (on top of native tools `8e10876`); the
  checkout was clean after that commit. All final gates passed and both
  independent reviews accepted. The release binary was built and tested locally
  only: **not installed, deployed, pushed or restarted as a service**, and no
  live-provider trials were run. Gateway fixes are separately committed as
  `0f7d7ea` in `~/Projects/llm-gateway` and likewise not installed.
- Historical progression: initial offline gates passed (1569 Rust unit tests plus
  4 CLI subprocess tests, 2 ignored; Python-focused 38+4, native 70, parser helper
  5) before review corrections; final corrected counts are recorded below.
- Sole Adapsis integration/docs writer: `ses_e8d51b0b2ffe2YU2n1nfJe0WbS`.
  Manager/release authority: `ses_e8d6d9d1bffe8G7KEI4yUw8DPi`.
- Isolated frontend writer: `ses_e8d0359d6ffe9vFSOf5LDkQyNI`, worktree
  `/home/marenz/Projects/adapsis-python-frontend`, owning only new
  `src/python_frontend/**` (lowering, canonical serialization and Python printer).
  Its seven finalized files were imported; no duplicate target/cache was created.
- Python reviewers (read-only): Astra critical integration/authority/CLI/
  persistence `ses_e8d27edc9ffeFZJshutWc7PbXM`; DeepSeek Pro language semantics,
  roundtrip and parser `ses_e8d279e0bffe6hI60QWbBO7jmg`. All corrections remain
  under main-checkout sole-writer ownership. Both reviewers accepted all final
  corrections, including the last printer/IO-registry/Result-await changes,
  conditional on the final complete offline gates.
- Parallel gateway compatibility writer: `ses_e8d51200cffe40osdeEbEWe0JC`, in
  `~/Projects/llm-gateway`, branch `work/native-tool-compat`, baseline `98d7240`.
  Gateway #2 and Python #81 are distinct workstreams; no shared source ownership.
  Gateway fixes are now committed locally as `0f7d7ea` on that branch; 70 offline
  tests, check and release build passed, with independent DeepSeek acceptance.
  Its checkout is clean. It has not been installed/restarted or live-certified;
  gateway #2 remains open and no gateway build is running.
- Preserve #78's pinned authority, strict caller-authored-reference validation,
  literal test/input policy, immutable turn completion baseline, CoW write-back,
  complete tool history and explicit completion reporting. Unsupported Python
  semantics must fail with source locations, never silently approximate behavior.
- No deployment, service changes or live model-effectiveness claims. Existing
  main target was reused; builds were coordinated with the gateway writer, with
  no new target trees. Last disk state: `/home` 31 GiB free, `/tmp` 15 GiB free.

### Python stage status
- [x] Parser/lowering and subset contract approved
- [x] Initial frontend implementation and main native integration
- [x] Initial typed/control-flow/collection examples and guarded end-to-end tests
- [x] Trusted mocked read_file acceptance (test-only IO, no live worker)
- [x] Initial AST/source/session/library roundtrips and unsupported-AST diagnostics
- [x] Review corrections: Python argument binding, async call fidelity and helper shadowing
- [x] Single-function inspect→resubmit with permission-filtered runtime signatures
- [x] Independent reviews and complete offline release gates
- [x] Code committed locally (`61b3911`); local binary tested; not deployed

### Final Python acceptance evidence
- `cargo test --release python_`: **52 unit + 5 CLI subprocess tests passed**.
- `cargo test --release native_`: **84 passed**.
- `cargo test --release`: **1583 unit tests passed, 2 ignored**, plus **5 CLI
  integration tests passed**. These gates were rerun after the final mechanical
  admission-helper extraction; parse → authorize → finalize → reauthorize →
  runtime-check ordering and error messages stayed unchanged.
- `cargo build --release` passed. Package version remains 0.1.0. The current
  executable's `test --help` documents language/module/paired-test flags.
- `cargo run --release -- memory-eval` passed: 6 cases, recall 1.0, zero duplicate,
  precedence, contradiction, ACL, citation, stale and omission violations.
- All **28 existing `.ax` example files passed**; `node tools/test_native_events.mjs`
  passed. Actual provider-free Python commands passed:
  ```sh
  ./target/release/adapsis test examples/python/basic.py --language python --module PyBasic --tests examples/python/test_basic.py
  # TEST SUMMARY: 2 passed, 0 failed
  ./target/release/adapsis test examples/python/flow.py --language python --module PyFlow --tests examples/python/test_flow.py
  # TEST SUMMARY: 7 passed, 0 failed
  ```
- Both independent source reviews accepted all corrections; Astra separately
  confirmed the final admission-helper extraction. No remaining review blocker.
  Runtime follow-ups #82 (user Result-await semantics) and #83 (narrower helper
  collision usability) remain explicitly deferred, with safe frontend refusal.
- `Cargo.lock` adds 28 parser-related packages without upgrading existing locked
  dependencies. No CPython, origin metadata, deployment, service restart, live
  provider trial, or claim of improved model effectiveness is part of this gate.
- Three release warnings remain: the existing structural-only native validator
  API and reserved frontend source-map fields are unused by production callers.
  Diagnostics themselves use actual source spans. Existing main target only;
  final free space remains about 31 GiB on `/home`, 15 GiB on `/tmp`.

### Review corrections (accepted; retained as design record)
- Keep Python positional-vs-keyword call semantics: exact parameter count/names
  are checked at dispatch and in the isolated CLI test path. `_0`, `_1`, …
  parameter/keyword names are reserved to avoid ambiguity with legacy positional
  carriers. Newly submitted direct calls use same-source plus current-program
  signatures; never re-analyze pre-existing trusted bodies transitively (#79).
- Direct user-call async effects must match await usage; synchronous functions
  cannot be awaited and async functions cannot be silently called without await.
  Runtime-provided asynchronous IO builtins remain supported.
- Recheck helper collisions at Python test/evaluate, not only mutation: later
  `.ax` edits must not intercept a previously installed Python list/concat/len
  helper. Without origin metadata, Python execution can conservatively reject
  runtime builtin-name collisions in the current program, including after reload.
  This includes `BUILTINS` and `IO_BUILTINS` and all aliases. Python inspection
  runs the same neutral helper safety check separately from its filtered
  signature catalog. Narrower helper-specific usability is tracked in #83.
- Approved two-phase mutation admission: complete parser-only supported-syntax,
  declaration/effect/annotation and direct-reference preflight with no existence
  checks; authorize module Write and direct-reference Execute using the pinned
  turn permissions; then per-call final lowering with only authorized runtime
  signatures plus same-source declarations. The resulting canonical operations
  still pass all existing native checks and executor gates. This enables an
  inspected single function with nested calls/arithmetic/loop conditions to be
  resubmitted without requiring its entire module. No hidden signature may
  influence diagnostics before authorization. Batches remain sequential, not
  atomic; later calls see earlier successfully applied mutations.
- Result-returning **user-function awaits** are rejected in Python lowering,
  direct runtime-call checks and Python inspection until the inherited runtime
  unwrapping policy is reconciled (#82). No interpreter semantics were changed.
  Direct IO builtin Result awaits remain supported: an in-memory IoRequest
  responder verifies the actual Ok wrapper through native evaluation, without
  any HTTP/network request. Trusted mocked read_file remains the runnable IO test.
- The approved quote-aware parser correction is isolated in
  `src/parser/test_values.rs`: quoted/escaped parentheses/braces and test `->`
  separators remain literal data. i64::MIN remains explicitly unsupported.
- Last disk/build state: existing main target only; 31 GiB free on `/home`,
  15 GiB free on `/tmp`. No deployment, service changes or live-provider trials.

### Approved Python v1 contract
- Exact `rustpython-parser = "=0.4.0"`, default location/bigint features. The
  crate was published 2024-08-06; the repository remains maintained. Its MSRV
  1.72.1 works with current Rust 1.93.1. Ruff's current published parser needs a
  newer toolchain and is not selected. Parse using `ast::Suite::parse`; convert
  UTF-8 byte spans to one-based source line/column diagnostics using the source
  text (including character-aware columns).
- Lower into typed `parser::Operation` declarations/tests, then a trusted
  canonical `.ax` emitter and structural roundtrip validation before the existing
  guarded native executor. This is node serialization, never text substitution
  or Python execution. Keep the same authorization and immutable completion gate.
- Native mutate/test add `language: "adapsis" | "python"` (default Adapsis);
  Python requires explicit `module`, Adapsis rejects the new module field.
  Evaluate adds the same optional language for literal input conversion, keeping
  the qualified `function` name. Inspect source adds `format: "adapsis" | "python"`
  (default Adapsis); other inspection kinds reject format. Finish is unchanged.
- Python modules contain an optional docstring plus annotated `def`/`async def`.
  Parameters, return values and new locals require annotations; reassignments
  lower to Set. Types: int, float, bool, str, list[T], Result[T]. Regular functions
  cannot return None (the runtime has no Unit type). Integer literals range from
  -9223372036854775807 through 9223372036854775807; i64::MIN is rejected because
  canonical `.ax` cannot parse its positive magnitude before unary minus.
  Numeric storage/overflow remains explicitly the existing typed
  runtime model, not a claim of arbitrary-precision CPython semantics.
- Support if/elif/else, while without else, and for over statically typed lists.
  A for target is fresh and confined to its body: reject prior bindings, reuse
  and reads outside the loop instead of silently changing Python's target
  leakage. New block locals cannot escape their block; existing variables can
  be reassigned. No break/continue or loop else.
- Collections are read-only values: list literals, iteration and len(list). Mutable
  aliases, append, item/slice assignment, += and subscripts are deferred. Numeric
  +, -, * and single comparisons are supported; string + maps to concat only
  when both operands are statically strings. Reject /, //, %, **, chained
  comparisons and operand-returning truthiness. Boolean and/or require Bool and
  total call-free operands unless exact short-circuit lowering is provided.
- Bare local calls lower to explicit metadata-module qualification; cross-module
  calls use Module.fn directly. Builtin-name definitions, callable variables,
  dynamic attributes, dunders, eval/exec/open/getattr and imports are rejected.
  No classes, lambdas, comprehensions, f-strings, ternaries, exceptions or nested
  defs. `@effects("io", "fail", "mut")` is a closed inert declaration; async def
  supplies async. Await is supported only in an annotated direct-call binding.
- Python tests are zero-argument `def test_*` wrappers containing one equality
  assertion of a direct function call against a literal, optionally with a
  docstring. They lower to TestMutation, never runtime function definitions or
  arbitrary test execution. Inputs/expectations accept scalar/read-only list
  literals; native Result matchers/global after assertions remain deferred.
- Python evaluate parses only literal arguments (including True/False and list
  syntax), not general expressions. The literal policy may admit canonical
  `list(literals...)` only with collision checks proving no user function/union
  variant can intercept construction; arbitrary nested calls remain forbidden.
  Python mutation lowering also checks synthetic helper collisions (including
  concat for string addition and builtin calls), even against otherwise
  authorized user functions; Python syntax must retain the promised operation.
- Mocked IO acceptance uses trusted harness-seeded session IO mocks and the
  existing mock-only test runtime, with a String-returning read_file example.
  Python cannot install mocks and no new mock schema/global state is introduced.
- AST is the only persisted truth: no frontend-origin metadata, sidecars or `.ax`
  header markers. Explicit Python inspection prints supported AST after reload;
  unsupported legacy forms return precise target/construct errors. Canonical
  Python print→lower→print is idempotent with normalized semantic AST equivalence;
  formatting/comments are not retained. Submitted functions replace existing
  functions as `.ax` does; unsubmitted functions keep CoW identity. No inferred
  deletes or whole-file diff system.
- Release cases: typed scalar, local calls, if, while, list/for, list arguments,
  explicit cross-module calls, bools, reserved-name/authority rejection,
  canonical source roundtrip and session/library reload. HTTP and SSE fixtures
  must demonstrate Python mutate → premature finish refusal → Python test →
  Python-literal evaluate → finish. Include location negatives, caller-authored
  nested references, list-constructor collisions and trusted mocked IO. Run
  Python/native/full Rust, example, memory and UI gates plus independent reviews.
- Direct executable demonstration uses the existing isolated CLI test runtime:
  `adapsis test examples/python/basic.py --language python --module PyBasic
  --tests examples/python/test_basic.py`. Production lowerers supply canonical
  operations to the existing test path; Python module/tests flags are required
  together, with no zero-test success. Legacy `.ax` defaults remain unchanged.
  No CPython, provider, live library state or new mock-authoring API is involved.

## Completed foundation — native tools first (#78)

**Approved direction (2026-10-25).** Implement native JSON-schema tool calls over
the existing permission-checked runtime before any Python frontend. The first
slice must complete one inspect → mutate → test → evaluate → finish task without
bypassing runtime authorization, test obligations, CoW write-back, session
persistence or completion validation. Related: #78 (this work) and #76 (manager
cutover, **not** satisfied by this slice). Preserve the legacy `<code>`
mutation/`!done` protocol throughout migration.

### Scope and explicit non-goals
- In: fixed named tool schemas; strict server-side argument validation; exposure
  filtered by permissions (exposure is not the execution boundary); streaming and
  non-streaming tool calls with correlated results; backward-compatible persisted
  history; shared execution/completion semantics for HTTP (`ask`, `ask_stream`);
  offline mock-transport and negative authorization/protocol tests. Takeover stays
  on the legacy protocol in this first slice.
- Out (first slice): deployment, generic MCP client, Python frontend, model-quality
  claims, changing `!opencode`/agent authority, replacing the AST/runtime.
- Later, separately measured: Python-subset frontend lowering to the existing AST.

### Execution phases
1. Research and agree the shared API (tool schema shape, assistant `tool_calls`
   envelope, tool-result correlation, persisted-history representation).
2. Isolated implementation in worktrees (schemas/dispatcher, transport, fixtures).
3. Single-writer integration in `src/llm.rs`, `src/api/{llm_handlers,execute}.rs`,
   `src/context_prompt.rs`, `src/permissions.rs`, `src/session.rs`, `src/builtins.rs`.
4. Full release tests, offline evaluation gate, independent diff review.

### Evaluation gate
- A fixed offline fixture suite certifies protocol shape and execution semantics
  with a mock provider. It does **not** certify model tool-selection quality.
- Real-model effectiveness is a separate opt-in measurement using the same tasks
  and result schema (model id/revision, repeats, success, attempts, tokens,
  latency); it is not a CI gate and no improvement claim is made without it.
- Mock results record token/latency provenance explicitly so harness overhead is
  never presented as model cost or speed.
- Report contract for future measured runs (documentation only; no generic
  benchmark CLI yet): `schema_version`, `fixture_id`, `transport` (`http`/`sse`),
  `mode` (`offline-scripted`/`live-provider`), `model_id`, `model_revision`,
  `repeat_index`, `attempts`, `completed`, `encountered_errors`, `assertions_passed`,
  `prompt_tokens`, `completion_tokens`, `token_source`, `latency_ms`, and
  `latency_source`. Offline provider fixtures use model `native-fixture`, fixed
  synthetic usage (10 input/5 output per scripted response),
  `token_source=fixture`, and `latency_source=local-harness`; never infer model
  performance from those values. Live runs require exact model/revision and
  provider-reported usage, retaining missing counters as null rather than zero.
- Fixture cases live in `src/api/native_turn_tests.rs`,
  `src/api/native_tools/tests.rs`, `src/llm/native.rs`, and
  `src/session/native_history.rs`: full lifecycle with early finish rejection,
  no-finish exhaustion, closed schemas, permission narrowing, guest exclusion,
  duplicate ids, same arguments/new ids, truncated/error streams, no legacy
  double-execution, immutable baseline, dependent batches and restored/bounded
  protocol pairs. A successful `finish` is task completion; encountered and
  subsequently corrected errors remain separately observable.

### Recovery / baseline
- Branch `work/native-tools`, baseline `90883a2`; upstream `origin/master`
  `6667139`. Do not merge pre-sanitization history or archived feature branches.
- Local-only recovery refs: `refs/archive/2026-10-25/*` (see AGENTS.md). Inspect,
  do not publish. No deployment or commit is authorized by the roadmap.
- Main integration writer: `ses_e8d51b0b2ffe2YU2n1nfJe0WbS` (main checkout).
  Transport: `ses_e8d53de21ffeTkWfOEP3G1FCYy` (`adapsis-native-transport`).
  Dispatcher: `ses_e8d5328a9ffepJ6zzxG43iUzuC` (`adapsis-native-dispatch`).
  Manager `ses_e8d6d9d1bffe8G7KEI4yUw8DPi` imports reviewed final diffs and
  retains architecture/release authority. Workers never commit or deploy.
- Approved surface: `inspect` (symbols/source/deps/tasks/library), `mutate`
  (explicit module/function/type/doc/shared declarations), `test` (qualified
  function test blocks), `evaluate` (qualified named functions), `finish`.
  Opt-in `--native-tools`, write-capable non-guest HTTP turns only. Plans,
  roadmap, agents, guest native tools and Python are deferred.
- Current phase: finalized transport/dispatcher imported; integrated offline gates
  pass and critical integration review accepted. Native
  completion uses one immutable turn baseline and explicit successful finish;
  provider stop/prose never certifies completion, and no legacy fallback runs.
- Model policy: preserve Claude quota above 90%; integration uses no Claude.
  Offline mock results only; real-model effectiveness remains unmeasured.
- Disk guard: serialize builds and recheck `df` before heavy phases. Reuse the
  existing main target; no new `/home` target trees. Cache inventory worker
  `ses_e8d384e9affeGaFjnegdZpoKtz` is read-only until the manager approves an
  exact cleanup list. Main target and both native worker trees/targets are
  protected; do not delete unrelated caches or files.
- Approved cleanup completed: six inactive cache/target paths reclaimed 25.0 GiB;
  `/home` free space rose from 7 to 32 GiB. Active main/native targets were
  untouched. Continue serial builds with a fresh space check before verification.
- Independent reviewers: Astra critical integration
  `ses_e8d27edc9ffeFZJshutWc7PbXM`; DeepSeek Pro transport/history
  `ses_e8d279e0bffe6hI60QWbBO7jmg`. Both accepted reviewed corrections,
  including DeepSeek's fresh acceptance of the final whitespace id/name
  validation tightening. All source reviews are clear with final offline gates
  green. Read-only reviewers; integration remains sole writer.
- Integrated verification after review fixes (2026-10-25):
  `cargo test --release native_` 57 passed; `cargo test --release llm::` 41 passed;
  completion suite 9 passed; conversation-view suite 7 passed;
  full `cargo test --release` 1529 passed,
  2 ignored; release build passed; `memory-eval` passed (6 cases, recall 1.0,
  zero ACL/citation/stale/duplicate violations); all 28 example files passed.
  No live provider calls, deployment, service changes, commits or pushes.
- UI gates: `node tools/test_native_events.mjs` passed; independent browser
  validation by `ses_e8d5eb051ffe2vVYWvhJwWh71j` passed at 1280×800 and 390×844
  against isolated mock SSE using the actual `web/adapsis.html`. Incomplete
  turns, corrected completion-with-errors, legacy done events and thinking
  toggles were verified, with no overflow or JavaScript errors. Screenshots:
  `/tmp/opencode/native-ui-verify/{desktop,mobile}-*.png`. The pre-existing
  mobile program-panel width issue is deferred as #80; the mock server stopped.
- Final build state: package version remains 0.1.0; `os --help` documents the
  HTTP-only flag. One release dead-code warning remains for the structural-only
  `validate_native_tool` API exercised by tests; production uses its authorized
  counterpart. At gate time new files were intent-to-add only; the manager later
  committed the #78 slice as `8e10876`. Disk was 32 GiB free on `/home`, 9.5 GiB
  on `/tmp` after those sequential gates.
- Post-gate cleanup removed only the two disposable `/tmp` native worktree
  target caches after builds stopped, reclaiming 5.36 GiB of tmpfs. Current free
  space is 15 GiB on `/tmp` and 32 GiB on `/home`; the main target, screenshots
  and logs remain available. No functional changes followed the accepted gates.
- Live gateway usability remains blocked/unverified by llm-gateway #2; legacy
  permission follow-up is tracked in #79. Neither is certified by offline gates.
- Review tightened the initial surface: source/deps require qualified members;
  evaluate/test inputs, exact expectations and shared defaults are literal-only;
  contains/starts_with matchers remain, Result matchers and global +after are
  deferred. New user-function calls must be qualified, including local calls.
  Direct authored module references are checked; task rows are Read-filtered
  and contain only categorical states. Unknown qualified eval/test targets get
  neutral diagnostics rather than legacy cross-module suggestions. Browser/TUI
  event handling distinguishes successful finish from incomplete stream end.

### Status
- [x] Baseline cleanup and direction recorded (2026-10-25)
- [x] Shared tool-call API and persisted-history compatibility
- [x] Tool dispatcher over existing guarded operations
- [x] Offline fixture suite + documented results contract
- [x] Offline streaming/non-streaming transport certification
- [x] Negative protocol + authorization suite
- [x] Independent review acceptance
- [ ] Live gateway/provider certification (llm-gateway #2)
- [x] Working Python-subset frontend (completed in #81 above; model-effectiveness trials remain separate)

---

## Historical sections (pre-#78)

Everything below this line predates the 2026-10-25 native-tools decision and is
retained for context. It is not the current execution plan; in particular
"Current Priority: Telegram Bot" and the "Next Targets" list are historical.

## The Three Modes

AdapsisOS operates in three distinct modes, each building on the previous:

### Agentic Mode (current focus)

AdapsisOS as a **live, autonomous AI programming environment**. The AI has a session, a
program state, a plan, and works toward goals — both user-directed and self-initiated.

- The AI receives user requests and builds Adapsis programs incrementally
- It can spawn sub-agents with scoped access for parallel work
- It proactively identifies missing capabilities and extends itself:
  - **Adapsis-level**: write new functions, types, modules in Adapsis
  - **Rust-level**: use `!opencode` to add builtins, fix runtime bugs, add IO operations
- It maintains a plan and works through it step by step
- Agents communicate via `!msg` / `?inbox`
- All mutations are logged, rewindable, inspectable
- `--autonomous` flag: inject a goal and let the AI work without user input

This is the default `adapsis os` mode.

### Adaptive Mode (planned)

AdapsisOS loads a **pre-built Adapsis program** and the AI can modify/extend it at runtime
via the mutation protocol. Like a live-patching server.

- Start with: `adapsis os --load program.ax --adaptive`
- The program runs (e.g. an HTTP server, a data pipeline, a game)
- The AI monitors it via `!watch` and `?tasks`
- When the user or an event triggers a change, the AI hot-patches the running program
- New functions can be added, existing ones replaced, types extended
- The program never stops — mutations apply live

Use cases: self-adapting servers, AI-driven game NPCs, live data pipeline reconfiguration.

### Compiled Mode (planned)

Freeze the current program state into a **standalone native binary** via AOT compilation.
No runtime, no mutation protocol, no LLM — just a compiled executable.

- Build with: `adapsis compile --output binary session.json`
- Takes the current `Program` AST and compiles everything through Cranelift
- Produces a statically-linked binary with no Adapsis runtime overhead
- Useful for deploying what the AI built as a production artifact

The flow: Agentic mode → build and iterate → Compiled mode → deploy.

## Current Priority: Telegram Bot

The first autonomous goal. The AI should build a Telegram bot entirely in Adapsis:

- Use the async IO system (TCP, HTTP requests via shell/builtins)
- Connect to the Telegram Bot API (HTTP long polling or webhooks)
- Handle messages, respond with AI-generated content
- Persist state across restarts via session
- If missing builtins are needed (HTTP client, JSON parsing), use `!opencode` to add them

This exercises the full stack: async IO, LLM calls, self-extension, error handling,
and proves AdapsisOS can build real-world applications autonomously.

## Self-Enhancement Roadmap

The AI inside AdapsisOS should be working toward making itself better. The loop:

1. **Identify a limitation** — missing builtin, awkward syntax, slow eval, missing IO op
2. **Try to solve it in Adapsis first** — write a function, use existing builtins
3. **If Adapsis can't do it, use `!opencode`** — request a Rust-level change
4. **After rebuild, verify the new capability works** — write a test, eval it
5. **Document what changed** — update the prompt, add to the registry

### Done
- [x] Full language: types, functions, pattern matching, effects, modules
- [x] Async IO: TCP, files, shell, stdin, LLM calls
- [x] Coroutine runtime with task registry and wait-reason tracking
- [x] Multi-agent with scoped branches
- [x] Agent-to-agent messaging (`!msg` / `?inbox`)
- [x] Self-extension via `!opencode` → OpenCode → cargo build → exec restart
- [x] Cranelift JIT for Int/Float/Bool/String/Struct
- [x] HTTP API + SSE streaming + web UI
- [x] Plan management with auto-numbering, plan pinned to chat
- [x] AI feedback loop: full eval/test/mutation results fed back to LLM
- [x] `+spawn` returns task handle, tasks trackable via `?tasks`
- [x] AdapsisOS identity prompt: AI knows it's a self-improving system
- [x] Three modes documented
- [x] 118 unit tests, 12 e2e tests

### In Progress
- [ ] **Autonomous mode** — `--autonomous` flag, goal injection, structured logging
- [ ] **Telegram bot** — first real-world autonomous build target

### Next Targets (priority order)
1. **HTTP client builtin** — needed for Telegram API, currently only TCP raw sockets
2. **JSON parsing builtin** — needed for any API interaction
3. **`!config` command** — per-task model selection, temperature
4. **AOT compilation** — freeze program to standalone binary (Compiled mode)
5. **Adaptive mode** — `--load` flag, hot-patching running programs
6. **Provenance tracking** — who wrote what, when, why (design doc Phase 7)
7. **Grammar constraints** — GBNF for guaranteed syntactically valid output
8. **LoRA fine-tuning** — train a model specifically on Adapsis
9. **Self-hosting** — Adapsis parser/evaluator written in Adapsis (design doc Phase 8)

## Post-Self-Hosting Goal: AI-in-the-Loop

Once Adapsis can parse and evaluate itself, the next milestone is the AI running inside AdapsisOS:

1. The AI operates through the AdapsisOS HTTP API — emitting mutations, running evals, querying state
2. The AI modifies the Adapsis runtime FROM INSIDE Adapsis — writing parser improvements that the parser then uses
3. A self-modifying AI in its own environment

The loop: AI writes Adapsis code → AdapsisOS validates and runs it → results feed back → AI
improves the code → the improved code improves AdapsisOS itself.

This is the design doc's Phase 8 vision made concrete.
