# AdapsisOS Roadmap

## Recovery — Python parity #86 (2026-10-26, red)

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
