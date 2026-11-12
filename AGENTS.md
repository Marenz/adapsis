# Adapsis — Adaptive, Self-Modifying AI Programming Environment

## Overview
Adapsis is an AI-native programming language and live agentic environment. Programs are built incrementally through validated mutations. The AI writes Adapsis code, gets immediate feedback, and iterates. The system can modify its own runtime via `!opencode`.

## Design authority
[DESIGN.md](DESIGN.md) is the authoritative, owner-confirmed decision record (2026-11-09).
Narrowing or removing a recorded capability requires an explicit owner decision recorded
there. Workers cite the DESIGN.md rows their change touches; reviewers check for
narrowing. Where this file disagrees with DESIGN.md, DESIGN.md wins.

## Development baseline (2026-10-25 cleanup)
- **Current here deployment (2026-11-12, switch #128): commit `b7bfa64` on
  `work/python-source-library`**, binary sha256
  `fa04e3677b4adf70aafc65196c7ff6facaece2c155f3f9b36de272359c7ce3e8`. The module
  library on this node is manifest-managed Python: `~/.config/adapsis/modules/`
  holds `library-sources.json` and `Chronica.py`, `Peer.py`, `TelegramBot.py`,
  `TgFormat.py`, `Watchdog.py` (dependencies: `TelegramBot → [TgFormat]`, others
  none; 275 stored cases, all re-run at start). There are no `.ax` files there any
  more; modules are created and edited as Python, the writer saves them, and
  `library_reload`, `!undo` and `/api/rewind` are refused (see "Issue #121 opt-in
  source library" below and `docs/python-source-library.md`).
  - Unit: `~/.config/systemd/user/adapsis.service.d/10-source-refusal.conf`
    (`RestartPreventExitStatus=78`, `OnFailure=adapsis-start-failed.service`) and
    the helper `~/.config/systemd/user/adapsis-start-failed.service`. A refused
    start (source policy or unreadable session) exits 78 and is NOT restarted.
  - Rollback bundle: `~/.local/share/adapsis/switch-128/pre-switch-bundle/`
    (session at revision 244, the five `.ax`, old binary `11487f02…`, unit state);
    old binary also at `~/.local/bin/adapsis.bak-11487f02`. Rollback restores the
    bundle as ONE unit (`~/.local/share/adapsis/rehearsal-128/REPORT.md`, step 8):
    the old binary cannot read a session that holds a Python mutation record.
    Switch notes: `~/.local/share/adapsis/switch-128/notes/RESULT.txt`. The bundle
    and older backups contain the bot token (`rehearsal-128/token-copies.paths`).
  - The `telegram_token` default in `TelegramBot.py` is blank; the token comes
    from the token file at startup and is no longer in the source or the session.
  - **sleek and edox are unchanged**: older binaries, legacy `.ax` libraries, no
    manifest. Nothing in this entry applies to them; a switch there needs its own
    conversion set, review and rehearsal.
  - Gate and tools for a repeat: `rehearsal-128/rehearse.sh`, `selftest.sh`,
    `bin/preflight.py` (requests a save and proves the session file is final).
- Historical (superseded 2026-11-12): here Telegram deployment (2026-11-02): runtime binary `88b4275`;
  TelegramBot/TgFormat are now production-Python-frontend-authored modules
  (48/18 functions, 221 stored cases), prepared with reviewed tooling `9ffd3fa`.
  Private provenance/rollback: `~/.local/share/adapsis/telegram-python-migration-20261102/`
  (`backup`, `capture`, `python-authored`); durable `.py` is authorship provenance,
  while the runtime library stores lowered canonical `.ax`. Preserve effective shared
  settings and final poll cursor; session snapshots alone omit these values.
  Library persistence changes source bytes: compare parser/full module AST and
  grouped tests against the authored artifact before treating a hash change as drift.
  Post-install exact text reply `PYTHONLIVE7429` was observed by manager; underscore
  stripping is an end-to-end residual of unestablished cause. No live repair-task/attachment or failure-ACK
  certification follows. ROADMAP.md's newest entry supersedes deployment status below.
- Release worktree integration (2026-11-02): `release/2026-11-01` combines the
  deployed `1ed6424` line (#87/#88, TUI, native opt-in) with exact main checkpoint
  `108e4dc` (G2/#114/G1/#112/#116). See ROADMAP.md's top release section for merge
  state, worker ownership and measured gates; historical branch guidance below
  does not authorize switching this release worktree or touching the main checkout.
- Historical (superseded 2026-11-12): here deployment (2026-11-02): manager installed release `e424fc9`;
  `/proc` binary SHA-256 `7a91463243d68a2c12bdd40d0906e1c99dcaca5fd95cc51ebae4c241cca341f3`
  verified. Here's runtime and service default model are `chatgpt/gpt-6-astra`
  (no explicit reasoning-effort variant), replacing the previous deepseek alias
  selection on this node only. Permissions/context files unchanged, native still
  opt-in. ROADMAP.md records rollback paths, worker IDs, full offline gates and
  live status/pure-eval/provider smoke. Strict parity remains red (324/58/191);
  no live Telegram reply or complete native-provider/task-quality certification yet.
- Started on `work/issue-42` at `22d93d6`; the consolidated, sanitized upstream
  baseline is `origin/master` at `6667139`. Continue on `work/native-tools`.
- Old feature branches already have their functionality integrated upstream
  (including memory lifecycle, creative tools, context caching and generic HTTP).
  Do not merge pre-sanitization history back into the current branch.
- Local recovery refs live under `refs/archive/2026-10-25/`: `heads/<old-branch>`,
  `stashes/0`, `stashes/1`, `legacy-runtime-wip`, and `tui-fix`.
  Inspect with `git for-each-ref refs/archive/2026-10-25/`; these are local-only,
  not branches or tags to publish.
- Retired experiments: Stratum prompt injection (superseded by Ladybug), legacy
  HTTP/auth convenience variants (use guarded `http_request`), duplicate struct
  propagation and old eval-comment edits. Recovery refs retain their originals.
- Personal `contexts/` files remain ignored/local-only, including the latest
  language-profile edit from the old issue-57 branch.
- Native JSON-schema tools over the existing permission-checked runtime landed
  in `8e10876`; the Python-subset frontend now lowers to the same AST and guarded
  workflow. Live gateway certification and measured model effectiveness remain
  separate gates. Preserve tests and explicit completion validation: native
  schemas or familiar syntax alone do not guarantee correct calls or completion.

## Language Design Goals
- **Optimized for LLM-driven construction**, not human typing
- **Superseded by DESIGN.md (L1, L2):** `.ax` and `<code>` are being removed and Python is
  the only language; the `+`/`!`/`?` syntax bullets below describe the legacy surface.
- Every operation starts with `+`, `!`, or `?` — clear code vs prose distinction
  - `+` prefix = scoped declarations (belong to a module): `+module`, `+fn`, `+test`, `+doc`, `+shared`, `+route`, `+startup`, `+shutdown`, `+source`
  - `!` prefix = global runtime commands: `!eval`, `!agent`, `!plan`, `!roadmap`, `!done`, `!opencode`, `!reload`, `!msg`, `!stub`, `!unmock`
  - `?` prefix = queries: `?source`, `?library`, `?symbols`
- Explicit types everywhere — no inference
- Effect system: `[io]`, `[async]`, `[fail]`, `[mut]`
- `+end` closes all blocks (`+fn`, `+if`, `+while`, `+match`, `+each`)
- `+module Name` as state change (no `+end` needed for modules)
- Pattern matching over method chains — `+match`/`+case` is primary
- Auto-propagation for errors — `[fail]` + `+call val:T = func(x)` propagates errors
- No `is_ok`/`unwrap` builtins — use `+match` on Result directly:
  ```
  +match fetch_data(url)
  +case Ok(data)
    +return data
  +case Err(e)
    +return concat("error: ", e)
  +end
  ```
- Test enforcement — functions >2 statements must have passing `+test` before `!eval`
- Tests persist across sessions and auto-rerun when functions change
- `!mock` for IO testing — intercepts `+await` calls with fake responses
- `+doc "description"` — documents a module (after `+module Name`) or function (after `+end`)

## IMPORTANT: Keep the prompt updated
When modifying the runtime (adding builtins, IO operations, commands, syntax changes):
1. Update `src/prompt.rs` with documentation and examples
2. Register new builtins in `src/builtins.rs`
3. Update this file if the change affects architecture or design

## Code Quality Rules
- **No file over 2000 lines.** If a file approaches this, refactor by extracting modules/functions.
- **No function over 100 lines.** Extract helpers. If a function does multiple distinct things, split it.
- **No duplicated logic.** If two code paths do the same thing (e.g. ask() and llm_takeover() both execute code), extract a shared function.
- **Every new IO builtin needs a test** in the coroutine test module.
- **Every new parser feature needs a test** in the parser test module.
- **Error messages must be actionable.** Include what went wrong, what was expected, and where (function name, module name, line if available).
- **Prefer returning Result over panicking.** Route handlers, IO handlers, and eval paths must never panic — always return errors gracefully.
- **Training data matters.** Code execution loops that interact with the LLM should produce training data entries.
- **Known tech debt:** eval.rs (8900+), parser.rs (6500+), api.rs (5200+), coroutine.rs (4300+) all exceed the 2000 line limit. Refactor these when touching them — don't make them bigger.

## Build & Test
```bash
cargo build --release
cargo test --release                    # Rust unit tests
# Deterministic offline Ladybug quality/ACL gate (JSON stdout, summary stderr)
cargo run --release -- memory-eval
# Adapsis tests: `adapsis test` exits non-zero on failure and prints "TEST SUMMARY: N passed, M failed"
for f in examples/*.ax; do ./target/release/adapsis test "$f" >/dev/null 2>&1 || echo "FAILED: $f"; done
# Provider-free Python frontend examples (production lowerer + isolated runtime)
./target/release/adapsis test examples/python/basic.py --language python --module PyBasic --tests examples/python/test_basic.py
./target/release/adapsis test examples/python/flow.py --language python --module PyFlow --tests examples/python/test_flow.py
cargo test --release python_           # frontend/native/reload/CLI regressions
```

**Formatting pitfall:** standalone `rustfmt src/main.rs` recursively formats every
module declared by the crate root, creating repository-wide churn even when one
filename was supplied. Never use it for scoped work. Format only leaf modules and
verify `git diff --stat` immediately after formatting.

**Delegated worktree scope:** create writer sessions with their session directory
anchored to the actual worktree they will edit. A session scoped to the main checkout
but instructed to write a sibling worktree triggers external-path permission prompts,
and a manager's permission observer can miss those prompts. Put disposable model
calibration and caches under `/tmp/opencode`; do not let tools create implicit caches
in either checkout.

## Run AdapsisOS
```bash
# Interactive with browser UI
adapsis os --port 3002 --url http://127.0.0.1:4000 --model chatgpt/gpt-5.4 --session my-session.json --log-file my.log

# Autonomous mode (works through roadmap)
adapsis os --port 3002 --url http://127.0.0.1:4000 --model chatgpt/gpt-5.4 --session my-session.json --log-file my.log --max-iterations 100 --autonomous "Check !roadmap and continue"

# Inject messages to running instance (no parallel streams)
curl -X POST http://localhost:3002/api/inject -H 'Content-Type: application/json' -d '{"message": "your message"}'
```

## HTTP API
```
POST /api/mutate      — apply Adapsis code mutations
POST /api/eval        — evaluate a function
POST /api/query       — semantic queries (?symbols, ?source, ?tasks, ?deps)
GET  /api/status      — program state, plan, roadmap
POST /api/inject      — queue message for autonomous loop
POST /api/drain-queue — drain queued messages
POST /api/ask-stream  — SSE streaming AI conversation
GET  /api/events      — SSE broadcast of all activity (for web UI)
GET  /api/tasks       — list spawned async tasks
GET  /api/log         — recent log entries
GET  /api/sessions          — list all session IDs
POST /api/sessions          — create session: {"session_id":"name"}
DELETE /api/sessions/:id    — delete a session
POST /api/sessions/:id/eval — eval in a specific session
POST /api/sessions/:id/mutate — mutate in a specific session
```

**Task status is redacted by construction (#112).** `WaitReason`'s `Display`
(`coroutine/task_status.rs`) is the only rendering used by `/api/tasks`, `?tasks`,
`query_tasks`, `?inspect task N` and the snapshot `wait_reason`. HTTP operations show
`scheme://host[:port]` only (`<invalid url>` when unparsable or non-http), file/shell/takeover
operations show the operation name, and `Completed`/`Failed` show `done`/`failed` with no
payload. The stored values stay intact (callback acknowledgement in `main.rs` reads them), so
never print a `WaitReason` payload directly. Out of scope and tracked separately: task locals
in `?inspect task N` (#115) is out of scope and tracked separately.

**HTTP builtin errors never contain the request URL (#116).** reqwest appends
` for url (<url>)` to its `Display`, which carries bot tokens, userinfo and query secrets, so
every HTTP builtin (`http_get`, `http_post`, `http_post_binary`/`http_get_binary`,
`http_upload`, `http_upload_bytes`, the upload response stream) renders transport failures
through `task_status::transport_error`: kind text, a class from reqwest's predicates
(`timed out` / `connection failed` / `response decode failed` / `redirect failed`) and the
host-only target, e.g. `error sending request: connection failed (http://127.0.0.1:34567)`;
`{:#}` appends the URL-free source chain. `http_request` (`http_policy.rs`) names the host only
too, and no longer echoes an invalid or non-http URL (a scheme-less `bot123:TOKEN` parses with
the scheme `bot123`, so even the scheme is not printed). **Observable change:** the old
`error sending request for url (...)` text and `http_request GET <url> failed` are gone; nothing
in the repository, fixtures or library modules matched on them. Not changed: `http_post_binary`
falls back to the last URL path segment (query included) as the attachment name when the
response has no Content-Disposition header.

## Architecture

- Issue #121 opt-in source library: `library-sources.json` explicitly maps module
  names to `python`/transitional `legacy_ax` plus a closed acyclic dependency set.
  Active manifests whitelist source files: unlisted files are diagnosed without
  reading/applying them, detached unlisted snapshot saves refuse, and cached
  top-level members refuse startup with recovery preserved. No-manifest legacy
  behavior is unchanged; unreadable policy is never treated as absent.
  **Run-time modules (#128), `library/source_enrol.rs`:** the manifest follows the
  live Program. `flush_live_sources` enrols every unlisted live module as `python`
  and keeps every listed module's dependency set equal to what its code references
  (`python_frontend::module_dependencies`); equal sets are never rewritten. Write
  order: source, manifest (commit point), session. Every manifest commit must
  load with the files on disk at that moment: dependencies are only ADDED before
  a module's source save (`grow`) and only REMOVED after it (`shrink`), and
  `source_io::would_load` decodes the on-disk sources against the new manifest
  before each commit (a change that fails stays pending). One save runs
  grow/save/shrink at most twice. Do not write the manifest from anywhere else.
  The source files of one save are separate renames too: `source_flush::save_managed`
  prints everything first, then writes a changed file only when the set on disk
  with that one file replaced still loads (dependencies first, the rest retried
  after every write). A module whose new text needs another module that could not
  be saved stays pending and the message names it. Remaining gap: functions moved
  between two modules in BOTH directions have no safe order; those files are
  written anyway and logged (a crash between the renames refuses the next start,
  exit 78). A save that changes no file decodes nothing. `RunToolScratch` and `__` names
  are never enrolled. Unprintable modules and cross-module cycles are not enrolled
  and are reported on every save (the cycle text names both modules and how to
  break it, and the closing mutation's acknowledgement carries it as an error).
  At startup a session-cached unlisted module is adopted if it prints, otherwise
  dropped into `<library>/recovery/<Module>-<ms>.json` (one journal entry + one log
  line each); leftover unlisted files are moved aside as `X.py.orphan-<ms>`, never
  deleted. Under a manifest a module is created by a Python mutation only:
  `library::refuse_unsupported_operations` (in `Session::apply_operations_async`,
  `AgentBranch::apply_async`, and the `mutate`/`module_create`/`move_symbols`/
  `module_remove` builtins) refuses Adapsis-source creation and whole-module
  removal (#126) before any state change. A new mutation entry point that can
  create or remove a module must call it. A refused start exits with code 78
  (`library::EXIT_SOURCE_REFUSED`); the unit needs `RestartPreventExitStatus=78`.
  An unreadable session file at `adapsis os` startup is refused the same way
  (exit 78, file named and untouched), with or without a manifest. "No session
  yet" requires `symlink_metadata` to answer NotFound: never use `Path::exists()`
  for that decision (a dangling link would start an empty session).
  `adopt_cached` order is plan (`grow_as(.., Step::Plan)`, no writes), recovery
  files, commit: a recovery file that cannot be written refuses the start before
  any source or manifest write. Keep fallible preparation ahead of the commit.
  Known residual: `!sandbox discard` after an enrolment leaves a listed module
  that is not live (same state as #126).
  Retained manifest presence includes empty sets: deletion/dangling links never
  restore legacy reload/writes in-process. Source-root create/inspection/enumeration
  failures block startup before cached code or initialization can execute.
  `library/source.rs` owns manifest/conflict/atomic-file policy;
  `library/source_io.rs` adapts the checked whole-module Python serializer.
  Never infer authority from mutation language, choose between colliding `.py`
  and `.ax`, or consult stale session modules while lowering managed source.
  Failed managed startup aborts before initialization; ALL managed hot reload
  requires restart, preserving live memory, baselines and queued edits.
  Cases parsed from source start untested; `library/source_verify.rs` re-runs them
  once at managed startup with the mock-only test handle (no real IO possible) and
  marks only passing cases passed. Failing cases stay untested, never refuse startup,
  and are reported by function name/count (log) and one journal entry per module.
  `flush_live_sources` is the sole managed writer: a blocking worker holds live
  Program read -> source transaction -> metadata locks through checked printing
  and rename, with bounded inputs and no model/eval/IO awaits. Queued requests
  carry names, never detached ASTs. Snapshot persistence APIs refuse managed
  modules; mutation acknowledgements mean pending source save, not durability.
  Failures retain pending names and live edits for recovery. Manifest is durable policy,
  LibraryState owns runtime source-byte hashes, session AST is recovery cache.
  Source/session writes are not cross-file atomic; final hash-check/rename has a
  documented external-process race. See `docs/python-source-library.md`.
  Persisted `@stored_test` data uses `tests_lower::Data::Trusted` (pure calls with exact
  callee spelling, Ok/Err/Some/None) checked by `module_source::check_stored_test_purity`
  at save AND load; `ir_surface::validate_stored_tests` is its structural guard. Native
  test/evaluate and CLI stay on the unchanged `Data::Literal` policy.
  Persistence printing qualifies a bare local call ONLY when `get_function` provably
  resolves it to the caller module's own declaration (catalog AND whole live Program);
  `python_frontend/qualification.rs` applies the same ledgered rewrite to the original
  so the round-trip comparison stays strict. Order-dependent names (declared by several
  modules) are frozen to today's binding and logged; never add caller-module-first
  runtime lookup. Runtime-defined named types live in `builtins::RUNTIME_TYPES` (Attachment).
  Focused isolated gates: `cargo test --release managed_` and
  `cargo test --release --test python_source_startup`. Without a manifest the
  legacy `.ax` behaviour is still the default of the binary. On here the manifest
  is active since 2026-11-12 (Development baseline); sleek and edox have none. A
  migration of another node requires its own conversion set, review and rehearsal.
  **Pitfall, observed live 2026-11-12, not changed (the owner is deciding how
  saving should work):** the mutation acknowledgement says "accepted in memory;
  authoritative source save is pending ... (not yet durable)". The bot repeated
  that to the user as "accepted in memory only, the file may not have changed
  yet", although the writer had written the file within the same second. The
  wording is true at the moment it is produced (the save is debounced and runs
  after the turn's write-back) but there is no later message that the save
  succeeded, so the model doubts a save that worked. Failures are reported
  (`library_errors()`, journal); success is only in the `[save]` log lines and in
  `?library` (no "Source save pending" line). Do not paper over this in a prompt.

- #120 task-domain logic lives in Python-frontend Adapsis modules, initially
  `examples/tasks/domain.py`; the premature Rust foundation is removed from active
  source, history retained. Pure scoped-record validation, transition proposals
  and advisory resource conflicts grant no authority and provide no persistence or
  executor. Task completion remains unreachable pending evidence/run integration.
  `scenarios.py` is test-only; literal-only assertions run through the real frontend
  and runtime with exact AST/reprint gates: `cargo test --release task_domain`.
  Legacy vectors remain unchanged. See `docs/task-foundation.md` for provisional
  states, supplied-string ID limitations and reusable primitive gaps.
  Fixture records inside list literals need explicitly typed locals first: the
  current Python printer lacks enclosing struct context for anonymous list-element
  record ASTs (G3). Preserve exact roundtrip gates; do not relax them for fixtures.

```
src/
  main.rs        — CLI, autonomous loop, session management
  api/           — HTTP API: mod.rs (routes), execute.rs (execute_code pipeline),
                   llm_handlers.rs (ask/ask_stream, llm_takeover), tests.rs
  ast.rs         — Core AST (Program, Module, Function, Statement, Expr, Type, Effect)
  parser.rs      — Line-oriented parser (+, !, ? prefixes, +end blocks)
  python_frontend/ — RustPython-backed restricted Python parsing, checked lowering
                   to canonical validated operations, and AST-based Python printing
  python_cli.rs  — paired Python source/test files through the existing CLI test path
  validator.rs   — Parser→AST, mutations, !replace, !remove, !module merging
  eval/          — Tree-walking interpreter, builtins, test runner, +match on Result
  vm.rs          — Stack-based bytecode VM (tier 2, see Execution Tiers)
  intern.rs      — String interner (InternedId=u32) for variable/field/variant names
  compiler.rs    — Cranelift JIT (Int, Float, Bool, String, Struct, While, Match)
  coroutine/     — Async IO runtime (TCP, HTTP, files, shell, LLM, mocks, task registry)
  llm.rs         — LLM client (OpenAI-compatible, streaming, thinking mode, retries)
  session.rs     — Session persistence (program AST, tests, roadmap, plan, mocks, chat)
  library.rs     — ~/.config/adapsis/modules/ auto-load/persist
  prompt.rs      — System prompt with language spec, builtins, examples
  context_prompt.rs — Per-context identity composition (core + identity file +
                   speaker authority) and the instruction-proposal loop
  builtins.rs    — Single source of truth for all builtins/commands/queries
  typeck.rs      — Type checker, symbol table, ?source reconstruction, semantic queries
  telegram.rs    — Telegram bot integration
web/
  adapsis.html   — Browser UI (SSE broadcast, collapsible thinking)
```

- Shared module vars live in `RuntimeState.shared_vars` under `Module.name` keys; eval/test paths must install both shared runtime and shared program snapshots so missing shared slots can be materialized from `+shared` defaults.
- **A `+shared` read resolves through `FN_NAME_STACK`, so the entry point's name must be qualified.** `Env::current_module_name()` builds the `Module.var` lookup key purely from the `.`-prefix of the stack top; a bare entry name yields `None` and every `+shared` read in that function fails with `undefined variable`. Entry points are reached with names typed by a human or read out of a `+route`/`+source` declaration, i.e. routinely bare — `!eval get_node_types` reported a healthy `Stratum` as broken while `!eval Stratum.get_node_types` worked, because only the async tier lacked the qualification the sync tier had. The invariant now lives in **`eval_function_body_named`** (`eval/mod.rs`), which every entry point pushes through; do not move qualification back out to the call sites, where it drifted before. It is *not* an instance of #37 — that is two `Program` Arcs; `shared_runtime` is genuinely one Arc and was never the problem.
- Module library synchronization is first-writer-wins. `LibraryState` hashes every successfully loaded/written `.ax` file; persistence refuses to overwrite an external edit, reconstructed source must parse before rename, and a failed load keeps the last-known-good in-memory module. Reconcile conflicts with `library_reload("Name")` and retry rather than bypassing the check. **The one deliberate exception: a file this process never loaded *and* that does not parse is overwritten (loudly) by the in-memory module** — refusing there would make corruption permanent, since the reload we would tell the model to run cannot parse it either. Load errors are keyed per module and save errors are deduplicated + capped, because both feed strings the LLM sees on every turn. **These legacy rules apply only without a source manifest.** Under an active manifest (`docs/python-source-library.md`) hot reload requires restart, the live writer persists managed modules, and `!undo` / `/api/rewind` refuse before any state change: rewind rebuilds from mutation history only, which has no startup baseline for disk-loaded modules, so it would drop or misstate them (`library::refuse_managed_rewind`, enforced in `session::rewind_to`). A rejected replacement (failing existing tests) restores the complete prior `FunctionDecl` Arc, signature included, so the writer never persists a new signature over an old body.

## Concurrency / Three-Tier State (issue #9)
- **Tier 1 Program** = `Arc<RwLock<Program>>`; **Tier 2 RuntimeState** = `Arc<RwLock<RuntimeState>>` (`SharedRuntime`); **Tier 3 SessionMeta** = `Arc<Mutex<SessionMeta>>` (`SharedMeta`). No god-lock — a handler locks only what it touches, and **never** holds a lock across an LLM call, eval, test, or IO. `ask_stream` clones state → drops locks → calls the LLM → re-locks briefly per operation.
- **Per-function CoW write-back — DO NOT regress to `*program.write() = mutated`.** Mutations run on a clone; write-back uses `Program::merge_changed_from(base, new)` (`ast.rs`), which diffs the `base→new` function set via `Arc::ptr_eq` and applies only that delta onto the live program. This is why the validator must keep allocating a fresh `Arc<FunctionDecl>` *only* for the function it rewrites (untouched fns keep pointer identity). A whole-program overwrite silently drops a concurrent writer's change to a *different* function. All 8 write-back sites (mutate, WorkingSet, llm_takeover, agent-branch merge, self-trigger, inline-IO eval, async-fn eval, route handler) capture a `base_program` clone for the merge. The only intentional full reset is `rewind_to`.
- **`SharedVars` (`session.rs`) = per-key locks.** `shared_vars: HashMap<String, Arc<RwLock<Value>>>` — the outer lock guards map structure only; an existing var's value is read/written under its own inner lock (a structure *read* guard is enough to insert-if-absent or update an existing slot). `Clone` **deep-copies** (fresh inner locks) so RuntimeState forks stay isolated — never assume a cloned RuntimeState aliases the live shared vars. Write-back into the live runtime uses `SharedVars::replace_from(snapshot)` (reuses per-key locks so live readers see new values); **don't** assign the whole `SharedVars` (that swaps the Arc out from under live readers). `Value` is not `Serialize`, so the field stays `#[serde(skip)]`.

## Execution Tiers
Function evaluation goes through up to three tiers (fastest first):

**Interpreter Result contract (#82/#97 C-core):** genuine declared calls use
`eval/invocation.rs` with the actual FunctionDecl, including selected stubs and
parser/named/async/test/task/source/route paths. `complete_invocation` wraps a successful
plain T once for Result<T>, preserves explicit Ok/Err, and catches only typed language
Check/propagation failures. Operational/internal errors stay Rust Err. Ordinary calls and
user await preserve Result. `+call T` unwraps one Ok or propagates explicit Err; `+call
Result<T>` captures only typed language failure, with argument evaluation outside its catch
scope. Nested Result requires explicit outer wrappers, not recursive flattening. No general
payload type inference is added. Python now admits Result user awaits and exact-known-inner
returns (G7); this does not lift production VM/JIT Result quarantine.
Raw lifecycle/blocks keep `eval_function_body_named` without signature lookup. CLI
`run-async` retains its legacy unframed entry setup through
`eval_declared_function_unframed`; unlike framed entry it cannot resolve root +shared
reads, a separately characterized exception to the named-frame invariant above.
`statement_execution.rs` runs the same statements for ordinary evaluation and tracing;
`tracing.rs` observes root statements only, executes control flow instead of skipping it,
and appends the normalized invocation outcome. No nested private-body events or extra IO
authority are inherited. Test/trace observers may assert/report an operational error, but
never resume evaluated code or create a recoverable in-language error value.
**Compatibility:** named/sync VM operational errors now propagate structurally; API/native
success flags reflect failure instead of successful `Err(...)` text. Application Value::Err
and ordinary error-looking Strings remain data. CLI `eval` exits nonzero on API success=false;
`run-async` exits nonzero on Rust Err, but application Err values still exit successfully.
Generic callback ACK/app-outcome interpretation (#104) is unchanged. Other live module
signatures/catch consumers must be audited separately before deployment.

1. **Cranelift JIT** (`compiler.rs`) — numeric/string functions only
   (`is_compilable_function` gate; bails on each/await/spawn/yield/List/Map/Union).
   - `compiler/failure_admission.rs` (#97/#119 safety checkpoint) refuses reachable
     Check, `[fail]`, Result-bearing signatures/bindings/named types and Result
     constructors before native code emission. Direct `compile(program)` checks every
     emitted top-level function; cached dispatch additionally checks the current root's
     reachable graph, using the shared iterative executable-AST walker and per-snapshot
     function identity. The local `is_compilable_function` hint alone is not graph proof.
   - `eval/jit_dispatch.rs` requires `cached_revision == revision` before invocation,
     discards stale modules before rebuilding, and never calls old code after a failed
     rebuild (#119). Warm same-revision safe numeric controls remain JIT-backed.
     This JIT checkpoint does NOT itself repair Result invocation normalization. The
     subsequent scalar VM safety rules are below. Native Check traps are unreachable through
     supported compile paths; other JIT limitations (e.g. integer zero-divisor abort,
     #92, and relocation #96) are not fixed by this admission rule.
2. **Bytecode VM** (`vm.rs`) — near-full language coverage.
   - Sync path: `eval_compiled_or_interpreted_cached` → `try_vm_execute` (eval/mod.rs).
   - Async path: `eval_async_function` (eval/mod.rs) — IO suspensions serviced
     via `CoroutineHandle::execute_await`, so mocks/in-process ops work.
   - The VM **bails to the tree-walker** on: `+shared` variable access
     (unresolved identifiers), `+spawn`, `+yield`, `+source`/`+event`.
     Bare identifiers only compile to variant constructors if they're declared
     union variants — anything else is a compile error (fallback signal).
  - Async fallback rule: only safe BEFORE any IO has been performed.
    `eval_async_function` tracks this; never re-run a function after IO.
  - **Scalar failure checkpoint (#97):** `vm/execution.rs::VmAttempt` distinguishes
    `NotAdmitted(error)` (no instruction/effect ran) from `Executed(Result<Value>)`.
    Only the former permits tree fallback; executed operational OR language failure
    never retries, even before the first IO. Both synchronous and async dispatch now
    propagate structural execution errors without calling the tree again.
    Check emits `RaiseCheck`, yielding typed `LanguageFailure::Check` and discarding
    the whole scalar VM state/frames, never returning an Err value to a scalar caller.
  - `vm/scalar_admission.rs` temporarily quarantines reachable Result signatures,
    binding/capture types and constructors until a separate value-capable VM migration.
    These graphs use the interpreter's normalized, typed-only catching contract.
    Non-Await builtin IO also selects tree before execution: CallBuiltin has no coroutine
    handle. Dynamic Result values from builtin Await retain existing value semantics;
    this is not general runtime type validation or complete #82 normalization.
  - Preparation emits each unique reachable callee once, resolves the original spelling
    through `get_function`, shares one growing interner and fills the call cache BEFORE
    the first instruction. Aliases share bytecode by snapshot-local function identity;
    recursive graphs terminate. No lazy compile/fallback during execution. Unsupported
    nested callees (e.g. shared access) select tree before earlier IO, preserving those
    programs when execution-error fallback is removed. Sync preparation rejects reachable
    AwaitIo before execution. A missing prepared cache entry is an internal execution error.
  - `vm/await_admission.rs` preflights the supplied root and its statically
    reachable calls before bytecode emission. Any user-function Await forces
    pre-execution tree fallback, including a transitive await after earlier IO.
    Iterative executable-AST traversal and per-snapshot function identity prevent
    cycles/name collisions; do not replace this with caller-module-first lookup
    or a cached decision across Program revisions. Unknown/builtin-only awaits
    retain existing behavior. `io_performed` and post-effect replay guards stay
    unchanged; never reset them to recover from a partially executed user await.
  - `vm/field_admission.rs` (#114) refuses, BEFORE execution, any function whose
    reachable graph contains a `FieldAccess` whose base is not statically proven to
    be a `Value::Struct`. The VM's `GetField` reads structs only, while the tree also
    gives `.len`/`.is_empty` (String/List), `.is_ok`/`.is_err`/`.unwrap`/`.error`/
    `.unwrap_err` (Result) and transparent `Ok(struct).field` meaning; after IO the
    async path cannot fall back, so this has to be decided at compile time. Same
    iterative traversal and tree `get_function` lookup as `await_admission` (shares its
    `Work`/`statement_work`/`expression_work`). Proof = declared `Struct` parameters
    and `+each` bindings, `+let` of a declared Struct whose VALUE is proven, struct
    literals, plain user calls returning a declared non-Result Struct, and chained
    fields declared Struct with a UNIQUE declaration. NEVER proven: any `+call`/
    `+spawn`/`+await` binding (VM keeps `Ok` where the tree unwraps, #82 viaok),
    match captures (they poison same-named bindings), shared vars, builtin calls.
    Parameters are only sound because (a) every in-graph call site must pass a proven
    struct to a Struct-typed parameter and (b) `check_arguments`
    shape-checks the ROOT call's entry arguments in `vm/execution.rs` preparation
    (`!eval f p=Ok({..})` reaches a Struct
    parameter otherwise). Conservative by design: it may send a struct-only function to
    the tree, never admits an unproven base. VM `GetField` and the JIT are unchanged.
    #114 blast radius on `examples/*.ax` (99 functions): VM-compilable 95 -> 52 (43 more go
    to the tree; `vm/blast_tests.rs` prints counts and asserts no function with only
    ordinary field names is refused by field admission); parity library unchanged
    (161 functions, 107). Subsequent Result/direct-IO quarantine changes emission
    eligibility to 49/99 examples and 99/161 parity-library functions; eager preparation
    can additionally refuse nested unsupported callees before execution.
   - Dedicated `Value::Ok`/`Value::Err` matching is handled by
     `vm/result_match.rs` (#93): push the inner value/error String before the
     match Bool so the existing reverse-order binding stores consume it correctly.
     User Union payload ordering remains separate. Nested match statements are
     covered; this does not certify nested payload patterns, Option matching or
     non-exhaustive-match behavior across tiers.
     Production Result graphs are now quarantined; literal Result opcode evidence uses
     the cfg(test)-only `compile_result_opcode_fixture`, restricted to pure graphs without
     Check or effects. Tests separately assert production refusal/normal tree fallback;
     this fixture is never evidence of normal VM Result invocation support.
3. **Tree-walker** (`eval/mod.rs`) — universal fallback; the only tier with
   `Env`-based shared vars, function stubs, spawned-task snapshot tracking.

Tests (`+test`) always run the tree-walker with mock-only coroutine handles.
Spawned tasks (task_id set) always use the tree-walker (snapshot tracking).

## Pure JSON presence/type helper (#105)

`json_kind(json:String, path:String) -> String` is pure, exact arity two, no aliases
or await. Closed labels: missing/null/bool/int/float/string/array/object.
`eval/json.rs` owns the ONE path resolver used by json_kind and the extracted
json_get implementation. Preserve legacy rules: empty dot components ignored
(empty/all-dot path = root), usize-like segments index arrays including +0/00,
other segments are object keys, overflowing numeric path segments fall back to
keys, no bracket/escape syntax. Wrong container/out-of-bounds means missing;
there is no newly invented malformed-String-path rejection.

`int` requires represented signed-i64. Represented u64 > i64 errors; finite serde
floating representation is float, including 1.0/-0.0 and oversized integer lexemes
that serde rounded into floats. Parser overflow/malformed JSON/type/arity error
without panic. No lexical-number precision claim or serde feature change.
json_get's polymorphic output/sentinels remain unchanged; its old frontend String
catalog entry is not made sound by this new helper. The new helper's catalog is
accurate `(str,str)->str`; JSON registry metadata lives in `builtins/json.rs`.

Native evaluate now flags operational json_kind failures with has_errors=true, using
structural execution failure rather than inspecting error text. Valid-kind Strings and
ordinary false values remain successes; tests also pin unchanged Program snapshots.
Primitive and port adoption are separate checkpoints. The NEW text-entry fixture
now uses json_kind before conversion/auth/IO; the original #105 presence assertions
pass without exemptions. This is not a change to original process_update or a
general fix of json_get's static catalog.

`json_valid(json:String) -> Bool` (separate checkpoint) is pure, exact arity one, no
aliases or await, in `eval/json.rs` + `builtins/json.rs`. Contract: true iff the shared
JSON parse step used by json_get/json_kind/json_array_len succeeds; it is not strict
RFC 8259. serde_json decrements a 128 budget per container, so nesting deeper than 127
containers is false (a clean `Err`, never a stack overflow; nothing in this tree disables
the limit) and non-finite numbers such as 1e400 are false. It never errors for any
String (malformed, empty, truncated, BOM, NUL, HTML, trailing garbage); arity/non-String
arguments error like json_kind. **Validity is not representability**: a represented u64
above i64::MAX is valid while json_kind errors, so never route integer-ID validation
through it. The JIT only compiles top-level functions and has no json_* lowering
(`undefined function` refusal, same as json_kind); VM/tree share eval_builtin_or_user.
Adoption in the Telegram port (poll_once guard) is a later checkpoint.

## Key Adapsis Commands
```
+module Name          — switch module context (all +fn/+type after go here)
+doc "description"    — document module (after +module Name) or function (after +end)
+test Module.fn       — run tests (blocks !eval if untested)
!plan set / done N    — task planning
!roadmap add/done/show — persistent long-term roadmap
+await roadmap_list/roadmap_add/roadmap_done — programmatic roadmap IO builtins
!mock op "pattern" -> "response"  — IO mocking for tests (returns String, for IO builtins)
!stub Module.func "pattern" -> expr — function stub (returns typed value, for user functions)
!unmock               — clear IO mocks
!unstub               — clear function stubs
!eval Module.fn       — evaluate function
!remove Module.fn     — remove function/type/module
!done                 — signal task completion
!opencode <desc>      — request Rust-level runtime change (use sparingly)
?symbols / ?source / ?tasks / ?deps — queries
```

## Autonomous Mode
- `--autonomous` injects a goal, the loop runs indefinitely
- After `!done`: checks roadmap for next undone item, continues automatically
- `/api/inject` queues messages picked up at next iteration (no parallel streams)
- Session survives restarts (program AST + tests + roadmap + plan serialized)
- `!opencode` triggers rebuild + exec restart, session preserved

## !opencode Rules
- Use for runtime bugs, missing builtins, or elegant language enhancements
- Do NOT use for application logic — write that in Adapsis
- Each call appends: "update src/prompt.rs and src/builtins.rs with new features"
- Uses `--fork` to avoid stale session issues
- 60 min timeout, 5 min idle timeout, process group kill
- Sequential lock — one at a time

## LLM Backends
- **Cloud** (port 4000): claude-sonnet-4-6, claude-opus-4-6, gpt-5.4, mimo-v2-pro
- **Local 9B** (port 8081): `systemctl --user start llama-server` — Qwen3.5-9B
- **Local A3** (port 8082): `systemctl --user start llama-server-a3` — Qwen3.5-35B-A3B
- **Local Nemotron** (port 8083): `systemctl --user start llama-server-nemotron` — Nemotron Cascade 2 30B-A3B (Mamba2 hybrid, ~178 tok/s, 1M context, no KV cache scaling)
- Only one local model at a time (GPU conflicts). Nemotron, A3, and 9B are mutually exclusive.
- Qwen3.5 family works from system prompt. Nemotron Cascade 2 handles Adapsis syntax reasonably well out-of-the-box.
- LLM retries honor server-provided rate-limit hints when present (`Retry-After`, `x-ratelimit-reset`, JSON `retry_after[_ms]`) before falling back to exponential backoff.

### LLM Gateway & `--model` (per-node)
All AdapsisOS instances point `--url` at a **local llm-gateway** on `:4000`
(`~/.config/llm-gateway/config.json`). Same gateway setup on every node; the
**only difference is the `--model` name in each systemd `ExecStart`**, which the
gateway resolves via `model_aliases` or `virtual_models`.
- **`virtual_models`** are failover chains — the gateway tries each target in
  order until one returns 200. Defined per gateway config, so a new virtual model
  must be added to **every** node's gateway that uses it (here and sleek each run
  their own gateway).
- Current virtual models:
  - `family-bot` (edox) → `chatgpt/gpt-5.6-sol` → `chatgpt/gpt-5.5`
    (2026-10-08: both verified against edox's live gateway. Removed broken
    Anthropic/DeepInfra/DeepSeek backups after a transient primary failure
    surfaced the last provider's `402 Insufficient Balance` to FamilyUser.
    Anthropic currently returns 401, so restoring an Anthropic backup needs
    working authentication as well as a valid model ID. These two working
    models share ChatGPT authentication/quota; this is model-level fallback,
    not independent-provider redundancy. Other nodes were not changed.)
  - `dev-bot` (here, sleek) → `anthropic/claude-opus-5` → `chatgpt/gpt-5.6-sol`
    → `chatgpt/gpt-5.5` → `deepseek-v4-flash` → `deepinfra/zai-org/GLM-5.2`
    (2026-07-27: opus-5 promoted to primary, opus-4-8 dropped; both verified 200
    against the live gateway. Adapsis-fluent models lead, deepseek/GLM are
    last-resort only.)
- **Pick an Adapsis-aware model for dev nodes.** Raw `deepseek/*` (incl. the
  misleading `deepseek-reasoner` alias → `deepseek-v4-flash`) does NOT know
  Adapsis syntax — it emits "Plan set" prose + fake `async {}`/`:=` code and
  hallucinated builtins. Opus/sonnet/gpt-5.5 produce clean Adapsis. That's why
  here/sleek were switched off DeepSeek onto `dev-bot`.
- **Gotcha:** model IDs are picky and the provider `models` list in config.json
  can contain IDs the upstream doesn't actually serve. Tested against the live
  gateway: `anthropic/claude-opus-4-8` works (200); `claude-opus-4-6-20260205`,
  `claude-opus-4-8-20260205`, and `...-4-8-latest` all **404**; an unprefixed
  `claude-opus-4-8` errors (needs the `anthropic/` provider prefix). Always test
  a new model ID directly (`POST /v1/chat/completions` with
  `{"model":"...","messages":[...],"max_tokens":10}`) before putting it first in
  a chain — the gateway silently advances past a 404 target to the next one.
- **`deepinfra/zai-org/GLM-5.2` returned 402** on every request (no credit) as of
  2026-07-27. It used to be `dev-bot`'s *first* target, so every call burned a
  round trip before falling through — now demoted to the tail. DeepInfra credit
  is intermittent: it worked on 2026-11-01 evening and returned 402 ("positive
  balance required") again by 2026-11-02 ~02:30. The here-gateway alias
  `deepseek-v4-flash` was pointed at `deepinfra/deepseek-ai/DeepSeek-V4-Flash` and
  then rolled back to `deepseek/deepseek-v4-flash` (DeepSeek's own balance is also
  low). **A provider balance running out silently breaks a bot whose `--model` is an
  alias**: re-test the alias target (`POST /v1/chat/completions`, `max_tokens` ~50)
  whenever replies stop. The `openai` provider also
  401s on `/models` (`OPENAI_API_KEY` stale), but that's cosmetic: `chatgpt/*`
  routes through the Codex OAuth provider, not `openai`.
- **A hung gateway is usually a stale `*-auth.json.lock`, not broken auth.**
  Symptom (seen 2026-07-27): every `chatgpt/*` request times out with **no
  gateway log line at all**, `GET /v1/models` hangs, and `llm-gateway login
  chatgpt` hangs *after* you paste the callback URL. Cause: llm-gateway's
  `TokenStore` guards the token file with a `create_new` lock file
  (`~/.config/llm-gateway/<provider>-auth.json.lock`) that only `Drop` removes —
  so a killed/crashed process (e.g. `systemctl restart` during a token refresh)
  orphans it, and *every* later token read spun at 50 ms **forever**. Diagnose
  with `ls -l ~/.config/llm-gateway/*.lock` plus `fuser` on it; if nobody holds
  it, `rm` the lock and the blocked processes resume instantly. Fixed upstream in
  `~/Projects/llm-gateway` (`src/oauth/token_store.rs`): locks older than 30 s
  are reclaimed, acquisition times out after 15 s with an actionable error, and
  the owning PID is written into the lock file.
- After editing a gateway config: `systemctl --user restart llm-gateway.service`
  (on each affected node). After changing a node's `--model`: edit its
  `adapsis.service`/`adapsis-bot.service` `ExecStart`, `daemon-reload`, restart.

## Test Infrastructure

- #121/#122 typed persistence: new Python test submissions store optional versioned
  `ast::TestCase.typed` payloads; input/expectation strings remain display/legacy
  import data. Present invalid typed data errors, never falls back. Legacy absent
  payloads retain their importer. Re-test/rollback is shared in `session/retest.rs`.
  A Python module loader must attach verified cases with `store_test_payload(...,
  true)`; `store_test` deliberately remains the legacy adapter.
- `session/replay.rs` records original Python source, explicit module/version,
  requested operation IR and the ACTUAL affected-module after-state, including
  stored tests/NodeIds. The outcome is accepted/rejected/partial: rejected requested
  bodies are never reapplied (#122). No unrelated module/global authority is copied
  into the checkpoint. This duplicates the affected module per mutation; it is
  recovery data, not maintained Python source or a history-retention policy.
  Old `sources: Vec<String>` entries decode and serialize as strings. Mutation
  entries default to Adapsis origin; new Python entries retain exact submitted
  text and explicit origin via one recording helper.
- Rewind builds a fresh Program and swaps only on success; it never runs historical
  tests, initializers, startup, eval or IO. New checkpoints restore their actual
  outcomes and rebuild indexes. Legacy failed/missing/misaligned outcome records
  refuse categorically rather than invent accepted state. Full rewind still resets
  runtime; standalone historical test events are not newly reconstructed.
  Editing after an EARLIER rewind is intentionally unsupported for new Python
  declarations: `revision != sources.len()` refuses before submission effects,
  preserving future history. Reads/evaluation/tests remain available. No branch
  lineage or future-history deletion is invented; #122 retains this residual.
- Python replay source is capped at 1 MiB before copying it into a record. Typed
  persistence work checks use 2048 JSON levels (`replay/limits.rs::MAX_JSON_DEPTH`),
  200,000 JSON-value nodes and 16 MiB text;
  prospective mutation checks budget both old/candidate module states before
  application (retest may combine them). These are structural-work budgets after
  deserialization, not a cap on whole-session file allocation or hostile-file
  hardening. Unknown versions,
  malformed payloads and scope mismatches fail closed without private payload text.
- **One nesting limit (2026-11-12).** Whatever the submission guard accepts (100
  source levels) must be storable, replayable, readable again and durable as
  managed source; no layer behind the guard may have a lower limit of its own.
  MEASURED: a declaration at 100 source levels is up to 408 JSON levels deep
  (nested blocks and `elif` chains: 4 JSON levels per source level; operators 211,
  calls 311) and its replay record 409. The old depth budget of 96 refused Python
  mutations above 21 `elif` branches, and serde_json's default recursion limit
  (127) made the SESSION FILE unreadable above 28 (written fine, next start
  failed; also reachable through `.ax` edits and managed files).
  - **Session reader, `session/deep_read.rs`:** `Session::load` (also used by
    `memory-migrate`) scans the file's container depth with a loop, refuses more
    than 4096 levels with a clean error, and deserializes with the recursion limit
    disabled on a dedicated 64 MiB thread (MEASURED at most 619 bytes of stack per
    JSON level; the cap needs about 2.5 MiB). Needs serde_json's `unbounded_depth`
    feature. **Every other deserializer keeps serde_json's default recursion
    limit** (`json_valid`, `json_get`, `json_kind`, `json_array_len`, HTTP bodies,
    provider responses, typed payloads, the manifest): the default exists for
    untrusted JSON, the session file is this process's own output. Do not call
    `disable_recursion_limit` anywhere else, and do not read a session with plain
    `serde_json::from_str`.
  - Not changed: `Session::save` writes pretty JSON, whose indentation grows with
    depth. One 98-branch `elif` function is 64 KB compact and 3.1 MB pretty, and
    every Python replay checkpoint of that module repeats it (session-size issue).
  - Tests: `cargo test --release storage_depth` (six shapes at the limit through
    native mutate/test/evaluate/inspect, session save, read in a fresh process,
    rewind, managed save, startup load; the deepest managed file per shape on
    every evaluation path; all on 2 MiB stacks), `session_reader`,
    `every_other_json_reader`, `typed_record_depth_budget`, `stored_source`.

- Python test submissions group cases by exact lowered target (#118) before
  canonical emission. Targets execute in first-seen order (interleaved targets
  regroup); cases retain per-target source order. Each def keeps its SourceSpan,
  mapping to the grouped operation. All cases must pass to store the replacement
  set; later submissions replace rather than append. Session/executor storage
  semantics are unchanged. Strict parity compares the entire ordered case set;
  per-def fixture inventories use source spans to consume each grouped case once.
- Tests persist in `session.stored_tests` (HashMap<fn_name, Vec<StoredTestCase>>)
- Auto-rerun when functions change via `invalidate_and_retest()`
- `!mock` intercepts `+await` calls during `!test` with fake IO responses
- Functions >2 statements blocked from `!eval` until tested
- `!done` rejected if untested functions exist
- Test expectations should be literal values, not function calls
- Pure function calls allowed in test inputs (e.g. `+with config=default_config()`)
- **Test matchers**: `contains("substr")`, `starts_with("prefix")`, bare `Ok`/`Err`, `Err("msg")`
- **+after assertions**: `+after routes contains "/chat"`, `+after modules contains "Name"` — check side effects after test execution
- **`cargo test` must never touch `~/.config/adapsis/modules/`.** The suite drives
  `execute_code`/`mutate` with fixture modules and library persistence is real, so
  `TestIO.ax`, `Math.ax`, `Live.ax` … used to be written into the developer's live
  library — which the running daemon auto-loads at startup, i.e. running the tests
  injected fixtures into production, and two tests defining the same module raced on
  the same `.tmp` rename. `library::library_dir()` now returns a per-process temp dir
  under `cfg(test)`; `ADAPSIS_MODULES_DIR` overrides it outside tests (useful for a
  second instance without also relocating `$HOME`). A fixture module still needs a
  name unique to its test.

## Smoke-testing an IO module against the running instance

An API client is `[io,async]` by construction, so the useful test is the live call:
mocking an HTTP call verifies the mock, not the endpoint. Everything below runs in
the **daemon's** process, on the daemon's program state, through its coroutine
runtime — which is what makes it evidence that the daemon has the code, unlike
copying the module to `/tmp` and running `run-async` (a different process).

```
adapsis eval --api http://127.0.0.1:3002 'Chronica.health()'   # real HTTP call
adapsis eval --api http://127.0.0.1:3002 'Chronica.health'     # zero-arg, no parens
!eval Chronica.health()                                        # same, from a <code> block
```

- **One predicate decides**: `eval::expr_needs_io_runtime` = an IO builtin **or** a
  user function declaring `[io]`/`[async]`. `/api/eval` and
  `/api/sessions/:id/eval` used to check only the builtin half, so
  `{"expression": "Chronica.health()"}` was refused as a *test expression* by the
  very process that answered `{"function": "Chronica.health"}` with a live call.
  Add a new inline-eval entry point and it must call that predicate, not re-derive it.
- **A dotted name is resolved before the `result.name` hack.** `Chronica.health`
  parses as `FieldAccess`, which the standalone evaluator turns into
  `Value::Err("Chronica.health")` so test expectations can write `result.name` —
  reaching it from `!eval` reported a plausible `Err(Chronica.health)` with
  `success: true`. `named_zero_arg_function` resolves the name first; nothing is
  declared as `result.name`, so test expectations are untouched.
- **A dotted name that resolves to nothing is an error in inline eval.** Standalone's
  last resort is "treat an unknown call as a union variant constructor", so
  `Chronica.helth()` answered `= Chronica.helth`, successfully. Variants are bare
  identifiers, so a dotted name is decidable — but only the `InlineEval` context
  rejects it; test expressions keep the lenient fallback.
- **`PureEvalContext` exists for the diagnostic, not the rule.** The effect guard is
  identical either way. `TestExpression` says `+with`/`expect` must be pure;
  `InlineEval` says there is no coroutine runtime here and names the live path. It
  must not tell an `!eval` caller to mock — that was the original wrong advice.
- There is deliberately **no `--live` flag**: `adapsis eval` has always dispatched to
  the running instance over `/api/eval`. The default `--api` port is 3001 while the
  daemon here runs on **3002**, and a bare reqwest failure reads as "this does not
  work" — so a connect/timeout failure now names the flag and the cause chain.

## Untrusted input must not abort the process (#133, #134)

- **Python nesting guard (`python_frontend/nesting.rs`).** `Context::parse` is the only
  place Python source reaches the rustpython parser, and it runs `nesting::check` first.
  Lowering, emission, the `.ax` reparse, validation and evaluation all recurse on nesting,
  and a stack overflow aborts the process. MEASURED 2026-11-10 on a 2 MiB stack (tokio
  worker / blocking thread): lowering overflows at ≈730 nested calls or blocks and ≈1,270
  chained operators, evaluating an `elif` chain at ≈660 branches, dropping the parser's
  tree at ≈15,000–31,000 levels. The guard is a linear lexical scan (no parse, no
  recursion) that keeps an upper bound of the depth: open brackets + indentation levels +
  `elif` branches of the open chains + depth-forming tokens since the last separator
  (operators, `.`, `not and or if else in is for await yield from`; `lambda` is not reset
  by commas; a closed bracket group counts 1 + the deepest point inside it). Limit
  `MAX_NESTING` = 100 in total; f-string `{}` fields are bounded at 2,000 bytes per
  literal (f-strings are never lowered; this protects the parser only). Refusals carry
  line and column. Repository sources reach 13 levels; `tools/cad_prepare.py` (ordinary
  Python) 49. Do NOT add a second parse entry point, and do not raise the limit without
  re-measuring on a 2 MiB stack. Tests: `cargo test --release nesting`.
  **Stored source has its own limit, `MAX_STORED_NESTING` = 250**, in force only
  inside a `nesting::StoredSource` scope (thread-local, restored on drop and on
  unwinding): the whole-module loader/printer round trip (`lower_python_module`:
  managed files and the text the checked printer just produced) and the printers
  behind source inspection. Reason: the printer parenthesizes every binary
  operation and `not` operand and qualifies calls, so the printed form of a
  function accepted at 100 scans at up to 199 (MEASURED); with one limit such a
  function could not be saved or inspected. Every submission path keeps 100. A
  start prints each module it loads, so a hand-edited managed file is accepted
  only if its PRINTED form fits 250: blocks and `elif` chains up to real depth
  250, expression shapes about 125. Those depths run on every evaluation path on
  2 MiB stacks (`api/storage_depth_tests.rs`). Never enter the scope for text a
  model or a client supplied. Bring the limit down when the printer emits minimal
  parentheses (#140). The legacy `.ax`
  parser has no such guard (it is being removed, DESIGN.md L1/L2). Native tool arguments
  are capped anyway (`source` 32,000 bytes, arguments 64,000).
- **String builtins use byte offsets and never split a character.** `substring` moves an
  offset inside a multi-byte character DOWN to that character's start (both ends, so
  `substring(s,0,k)` + `substring(s,k,len(s))` is always `s`); `char_at` returns the whole
  character containing the byte. `eval/strings.rs::floor_char_boundary` is the helper; use
  it in any new builtin that slices a String by an Int.
- **Transport operations check their argument count** (`coroutine::transport_min_args`):
  the arms of `execute_transport_op` index `args` directly. Add new arms to that table.
- **A panic message never reaches the journal.** Rust slice/index panics quote the value
  they failed on. `error_journal::panic_hook` (installed in `main`) journals `panic at
  file:line:column (message of N bytes not recorded)` with one fingerprint per source
  location; the message goes only to the content-gated diagnostic log. A panicked spawned
  task is described by `error_journal::describe_join_error`: never format a tokio
  `JoinError` with `{}` into the journal or the task registry (tokio ≥ 1.4x quotes the
  panic message there).

## Runtime error journal & Watchdog

- `src/error_journal.rs` is an in-process ring (500 entries) plus per-fingerprint
  counters (2000). Failure sites feed it with ONE line, `error_journal::record(category,
  source, message)`, and keep their existing logging/behaviour: library load/save/reload
  (`LibraryState::record_error`; boot and reload failures go ONLY through
  `journal_load_failure` — module name + line number, never parser text, one entry),
  route handler errors and missing handlers (`adapsis_route_dispatch` via
  `record_route_error`: structural — request body (raw or escaped) replaced by
  `<request body>`, then route + first line only, so it does not rely on handler hygiene), spawned-task failures/panics, takeover/native/agent
  provider errors, session save failures, startup errors, the panic hook (`panic_hook` →
  `record_panic(location, payload length)`, never blocks, never the message) and refusals (shell policy reason only, guest IO boundary). New failure
  points should call the same API rather than a new channel.
- Messages are redacted on entry: URLs host-only (`coroutine::host_only`), Telegram tokens
  and long digit-bearing secret runs masked, 1 KiB cap. Never pass request bodies or chat
  text. Provider error text is kept (it is the diagnosis) and is still untrusted data.
- Fingerprint = category + source + message with digit/hex runs collapsed.
- `runtime_errors(after_seq)` (IO builtin, `coroutine/runtime_errors.rs`) returns
  `{snapshot_id,now,last_seq,dropped,entries,fingerprints}` JSON (snapshot_id is a fresh
  uuid per call); refused in guest/execute-only turns.
- `Watchdog.ax.work` (repo root) polls it each minute and triages new fingerprints (or 10x
  growth) with a tool-less `llm_call`, max 6/hour and once per fingerprint per 24 h, then
  `conversation_notify`s `admin_context` (empty in the repo; the installed copy sets the
  admin DM key `telegram:<admin id>`). It has no write path and never triages sources
  starting with `Watchdog.`. Its busy latch is owned by the snapshot_id token: a busy call
  returns without advancing last_seq, a latch older than 600 s is taken over, and only
  the current owner can release it. Its ledger is runtime-only, so a restart re-reports.
  Tests: `cargo test --release journal` and `cargo test --release watchdog`.
- #113: library `+route` registration outcomes log as `library-route`, not an error.
  The old `[library-error] detail_bytes=101` at every start was the benign
  `updated route POST /webhook/telegram …` message for the session-persisted route.

## Training Data
- JSONL training log at `--training-log` path (default: `training.jsonl`)
- Each iteration: model, context, thinking, code, outcome, tests passed/failed
- Accumulates in `~/.config/adapsis/training/`
- **`--training-log` deliberately stores full content (prompt/thinking/code/outcome).**
  It is a separate fine-tuning artifact, enabled by default by `adapsis os` at
  `training.jsonl`, not the journal — the #88 metadata-only gate does not apply
  to it. Select its path with the expectation that it holds model/user content.
- `tools/generate_training_data.py` — generates examples using Opus, validates through parser
- `tools/module_training_examples.py` — hand-crafted per-module examples
- `tools/merge_training_data.py` — deduplicates and merges all JSONL files
- `tools/finetune_gemma4.py` — QLoRA fine-tuning with unsloth

## Attachment System
- `Value::Attachment` — binary data type (audio, images, files)
- `AttachmentData::Memory` (≤10MB in RAM) / `AttachmentData::File` (>10MB on disk)
- `http_post_binary(url, body)` → returns `Attachment` from HTTP response
- `http_upload(url, attachment, field, extra)` — multipart upload from Attachment;
  returns `Result<String>`, so callers must `+match` Ok/Err
- `conversation_notify(context, message, attachment)` — delivers to conversation with file
- `http_request(method, url, headers, body)` is the raw coroutine HTTP escape hatch for
  GET/POST/PUT/DELETE/PATCH, including local/mesh services. Its method/header/URL policy
  is enforced in the runtime IO loop, not only by the evaluator. It returns catchable
  `Result<String>`, rejects non-HTTP(S) URLs and non-2xx responses, caps successful
  bodies at 2 MiB and error bodies at 64 KiB, and has a 30-second timeout.

## Per-Context Identity (issue #41)

**The prompt keys on the conversation; the permissions key on the speaker.**

The system prompt used to branch on `permission_model.is_some()` — i.e. on *who
spoke last*. A shared group therefore had no stable identity: German family
assistant when a non-admin spoke, English Adapsis programmer when the admin did,
with `set_primary_system` rewriting `messages[0]` each turn, so the persisted
history held assistant turns authored under two personas. `src/context_prompt.rs`
replaces that branch with one composer.

- **Composition order** (`context_prompt::compose`), each layer unable to undo the
  one before it:
  1. `technical_core` — `<code>` mechanics, `!done`, `<iteration_budget>`,
     `?source`, the `!agent`-cannot-do-IO rule. Fixed text; a context file cannot
     override it. **The `<iteration_budget>` instruction lives here and nowhere
     else** — it used to be duplicated in both branches and the copies drifted.
  2. `mesh_topology()` when a `mesh.md` exists.
  3. Adapsis language reference + `adapsis_identity()` — **only when the speaker
     may write.** Handing the language spec to an execute-only conversation
     invites mutations the permission layer refuses, which reads as model
     incompetence.
  4. The context's identity file. These first four layers form the stable system
     prefix for a given context and write-authority class.
  5. Permission-filtered program summary and current model, carried in the
     user-role runtime tail so provider adapters cannot hoist it ahead of history.
  6. `speaker_section` in that same runtime tail. States what THIS speaker may do.
- **Identity files:** `~/.config/adapsis/contexts/<stem>.md`, tracked originals in
  `contexts/`. `<stem>` maps every character outside `[A-Za-z0-9_-]` to `_`
  (`telegram:group:-513…` → `telegram_group_-513….md`). Total sanitizing, not a
  `:`-only substitution: keys arrive from Telegram payloads and must not be able
  to name a file outside the directory. `ADAPSIS_CONTEXTS_DIR` overrides the
  directory. Read fresh every turn — no rebuild to edit one.
- **Exact match only, and no fallback.** A context without a file gets the core
  and is *told* it has no persona. It does NOT inherit `persona.md`. An
  unconfigured conversation wearing someone else's persona is the bug.
- **`prompt::persona()`, `persona_from_paths()` and `default_persona()` are
  deleted.** They assembled one GLOBAL identity that every sandboxed conversation
  fell back to — the mechanism by which a game-dev group answered as a family
  assistant. The built-in German persona now lives in
  `contexts/templates/family-assistant.md`, to be copied to a real context key on
  a node that wants it. Nothing under `contexts/templates/` is ever loaded as a
  context.
- **`persona-notes.md` is opt-in per context.** `persona_notes_path()` is still
  the single definition of where the notes live (`Wolfi.remember` derives the same
  path via `home_dir()`), but the reader moved: a context file pulls them in with
  `{{persona_notes}}` (`context_prompt::PERSONA_NOTES_MARKER`). Without the marker
  the notes never reach that conversation. An unresolved marker is stripped rather
  than shown. #42 replaces the file with per-context Ladybug memories.
- **`Conversation::system_prompt` is gone.** It was initialized `None`, never
  written, and read through `unwrap_or_else` — populating it would have *frozen*
  the prompt, defeating the refresh-after-upgrade behaviour the read site exists
  for. Old sessions carrying the field still deserialize.
- **Turn binding (`coroutine::TurnIdentity`).** `AppConfig.turn` carries the
  context, principal and write-capability of the conversational turn into
  `CoroutineHandle::with_turn`. Identity-scoped builtins read it instead of
  trusting an argument, and **fail closed** outside a conversation (route
  handler, `+startup`, CLI) rather than defaulting.
  - `memory_cypher(query)` takes **one** argument now. The ACL principal used to
    be argument zero, and `authorized_cypher` faithfully computes the readable
    set of whatever principal it is handed — so reading another conversation's
    memories was a matter of typing a different string. The two-argument form is
    rejected loudly; accepting it silently would leave callers believing they
    still choose the principal.
  - `context_propose(text)` writes `<stem>.proposed.md` for the **turn's** context
    (not an argument), so a conversation can only propose for itself.
    `context_proposals()` / `context_approve(key)` / `context_reject(key)` are
    gated on the speaker's write capability. Proposals are disk-backed (survive
    restart) and one file per context (concurrent proposals cannot collide).
    Set `ADAPSIS_ADMIN_CONTEXT` to have the diff pushed to an admin conversation;
    delivery is best-effort because the proposal is already durable.
- **Sub-agents inherit the spawning context's composed prompt**, not a
  freshly-built `system_prompt()`. `!agent --model X` chooses who *generates*; the
  agent's authority is `spawner.narrowed_to(model)` — narrowing at spawn is fine,
  widening is not.

## Conversation System (llm_takeover)
- **Native tools remain opt-in pending parity (#89).** `--native-tools` enables
  the native protocol for eligible write-capable, non-guest `/api/ask`,
  `/api/ask-stream` and `llm_takeover` turns. With no flag, legacy remains the
  default. `--legacy-tools` explicitly forces legacy and wins if both flags are
  supplied, regardless of order. Guests/execute-only speakers retain whole-turn
  legacy routing. Native turns do not have per-feature fallback for unavailable
  operations. The owner's superseding gate requires executable conversation
  capability parity and independent review before changing the default. Preserve
  legacy code and the guarded external `!opencode` integration (process/model
  gates and runner boundary); native schemas do not replace that development
  environment. Memory/logging/TUI fixes can ship independently under opt-in.
  `api/native_turn.rs` owns the shared HTTP loop; `api/native_tools.rs` validates
  arguments and permissions over the existing executor. The initial surface is
  inspect (symbols/source/deps/tasks/library), mutate (explicit module/function/
  type/doc/shared declarations), test (qualified function test blocks), evaluate
  (qualified named functions), finish. Plans and agents are deferred. The native
  prompt recommends the explicit Python frontend below; legacy non-native
  conversations retain their Adapsis prompt.
  Source/deps inspection requires `Module.member` (bare module/function/type
  lookups are rejected); use symbols for module listings. Evaluate/test inputs,
  exact expectations and shared defaults are literal-only, including checked
  recursive list constructors. List construction is refused if any user function
  or union variant shadows `list`, even if authorized. Only contains and
  starts_with test matchers are supported; Result matchers and global +after
  assertions are deferred because the legacy expected-expression/diagnostic
  paths can resolve or reveal hidden modules. Canonical/Adapsis mutation bodies
  must qualify user-function calls, including local calls; Python local calls
  are qualified by the frontend. Bare calls are limited to
  builtins, and direct unauthorized module references are rejected before apply.
  Pre-existing trusted function internals are not reinterpreted as a new
  transitive capability policy (#79). Task inspection lists only readable,
  attributable functions with categorical states; raw results/paths/commands
  are intentionally omitted.
  Native mode never executes `<code>` text or falls back after provider errors.
  Successful finish uses one immutable turn-start baseline; stop/prose is not
  completion. Every complete stream is validated before any tool executes.
  Calls execute sequentially with a fresh WorkingSet and CoW write-back per call;
  a batch is not an atomic transaction. Duplicate ids are rejected without replay;
  identical arguments with a new id intentionally run again.
  Native HTTP responses add `completed` (and streaming done events carry it):
  only successful finish makes it true. `has_errors` retains legacy aggregate
  semantics, so a corrected error can coexist with `completed: true`.
  The main conversation's permission profile intersects the active model per
  group in a pinned turn snapshot; an override can only narrow execution.
- **Native takeover turns (#85).** With the native protocol opted in (see above),
  `handle_llm_takeover`
  (Telegram DMs/groups) uses the native protocol when the speaker's pinned snapshot
  `native_turn::permission_snapshot_for(tmp_config, context)` (active/effective
  model intersected with the context `permission_model` and `[principals]`
  profile, never widened) is write-capable and non-guest
  (`takeover_native::native_snapshot`). Everything around the model loop is
  unchanged and shared with legacy: per-context lock, Ladybug user/attachment
  ingest, ranked + canonical memory, image recovery, episodes, activity snapshot,
  reply_fn/reply_arg storage, the user turn in session history. The branch
  changes only the stable prefix (`context_prompt::native_takeover_prefix_keyed`
  = native prompt + `prompt::native_takeover_mechanics` + mesh + identity file;
  the speaker/runtime tail is the same, with agents/opencode reported as
  unavailable) and then calls `takeover_native::run`.
  - **One loop.** `NativeLoop::drive` serves HTTP and takeover. `RoundBudget`
    distinguishes them: `Fixed(max_iterations)` for HTTP, and for takeover the
    legacy semantics from `api/takeover_budget.rs`: `<iteration_budget>` estimate
    (default 10), 50-generation ceiling, corrections extend an exhausted estimate,
    one empty-response retry, one final status round. The final round is sent
    without tool schemas (`tool_choice: none`); any tool calls it still returns
    are reported as an error and never executed, and the turn stays unconfirmed.
    The incomplete-turn note is reply-only for takeover (HTTP still persists it
    in `main`). Takeover batches go to the
    context's own conversation; no conversation is trimmed any more (#68), the
    checkpointed view + compaction bound what the model sees. Every non-empty assistant text is also written
    to Ladybug as `agent:kronk`. A Ladybug write failure aborts the turn (`Err`),
    as legacy does. `NativeLoop.generator` keeps the generating model separate
    from the pinned permission model.
  - **Reply = latest visible prose** (`visible_takeover_prose`), never tool JSON.
    If finish was not accepted, the reply says so truthfully: exhaustion note,
    50-generation note, the German provider-error apology, or `EMPTY_RESPONSE_ERROR`.
    An accepted finish with no prose at all returns `""`, as legacy `!done` does,
    so the Telegram module sends nothing.
  - Not available natively: `!agent` (so no agent completion callbacks),
    plan/roadmap, `!opencode`, model switching, mocks. Other builtins are reachable
    only through existing module functions via evaluate.
  - **Turn-bound memory tools (#87).** `memory_inspect`, `memory_remember`
    (scope `context`|`global`), `memory_correct` and `memory_forget` are native
    tools that dispatch the *same* turn-bound builtins as `!eval memory_*`
    (`api/native_tools.rs` → `CoroutineHandle::with_turn` → `execute_await`), so
    ACL/scope/write-capability rules and caps are identical. They take no
    context/principal arguments, are advertised only when the turn has a
    `TurnIdentity` (fail closed otherwise), and are reachable only for
    write-capable non-guest turns. The native prompt points memory at these tools
    (Ladybug) instead of module functions, so models stop wrapping them in
    throwaway library modules.
    Broad inventory questions require authorized `memory_inspect` first;
    retrieved rows (including inspection itself) are bounded and never establish
    completeness. This is not permission to search other private chats. Native
    forget retains the builtin's all-governing-groups contribution ACL, which
    can authorize non-guest cross-context changes; no narrower rule is invented.
  - Native assistant text has no `!done`, so compaction's "end chunk at the last
    `!done`" boundary does not apply to native segments (it compacts the whole
    bounded chunk).
  - Native requests omit `temperature` (claude-opus-5 400s on any explicit value).
  - Tests: `cargo test --release takeover_native` (mock provider, in-memory
    Ladybug, `MemoryEmbedder::deterministic()`, which exists only under
    `cfg(test)`). Under `cfg(test)`, `contexts_dir()` and `mesh_topology_path()`
    resolve to per-process temp paths unless the env override is set, so tests
    never read live identity/mesh files.
- Native session history stores complete assistant/correlated-result batches;
  `session/native_history.rs` drops malformed/orphan batches on conversion and
  trims complete groups. Bounded views budget argument text and never truncate a
  native batch into a partial protocol. Hidden reasoning is not replayed as text.
  Offline fixture gates: `cargo test --release native_`, plus existing completion
  and conversation-view tests. These measure protocol/runtime behavior, not model
  quality. **Live gateway compatibility is blocked/unverified**: llm-gateway #2
  tracks tool-dropping virtual routes, call-id and terminal-status translation.
  Do not advertise live-provider usability until that separate gate passes.
- **Known safe-direction limitation (#95 related):** native Python mutation may
  refuse a qualified call whose spelling is also a variant name in a reachable
  module, even when the fully-qualified call would resolve unambiguously. That is
  a conservative over-refusal, kept deliberately: it never admits a program whose
  behaviour could differ between execution tiers. Do not "fix" it by relaxing the
  declaration or reachable-graph checks without an owner/proven representation.
- **Python-subset frontend (#81, #86 parity work in progress):** exact
  `rustpython-parser = "=0.4.0"` parses
  syntax without a Python interpreter. `src/python_frontend/` lowers a bounded
  typed subset directly into `parser::Operation`. #121's independent
  `python_frontend/ir_surface.rs` structural guard checks supported IR
  placement, names/types/effects, direct call shapes, pattern/binding coherence,
  test-data/matcher shapes and inert source/event fragments. It rejects casts
  (runtime conversion otherwise erases them). The guard uses existing identifier,
  effect, type-fragment parsing and conversion helpers; it is NOT type checking,
  authority, or runtime admission. Current errors identify an IR operation path
  at submission line 1; existing SourceSpans are unchanged and are not a nested
  node-location map. All native guards remain. `LoweredSource.operations` and
  `source_map` are execution data; `legacy_export()` is a fallible, explicit
  emit/reparse oracle/export, never an execution fallback. Native Python and CLI
  test paths use typed operations; legacy string entry points parse then delegate
  to the same executor/session application. Python evaluate has typed literal
  inputs, without constructing an `!eval` string. Library format activation is
  separate work: retained legacy library hooks still write `.ax` until the
  reviewed Python backend is integrated. No maintained `.py` authority follows
  merely from this execution change. #86 adds forms toward `.ax` parity:
  `@dataclass`/`@union` types, `match`/`case`, `+shared`, `+call`/`+check`/`+spawn`
  via `propagate`/`check`/`spawn`, `div`/`rem` helpers, all eight effects, and
  source/event/route/lifecycle declarations. `src/python_frontend/parity.rs`
  (test `python_parity`) is a strict test-only inventory/roundtrip gate over
  `tests/fixtures/python-parity/manifest.json` (examples + a sanitized library copy).
  The gate is currently RED. It reports invalid source separately, retains
  unsupported top-level declarations/commands/tests and module-only failures,
  compares exact ASTs ignoring only typed NodeId fields, and checks second-print
  byte identity. Do not count a fixture as covered merely because a function
  printer passed; unqualified call normalization is an AST difference.
  Stored module `+test` blocks are compared as `parser::TestCase` values (the validator
  never stores tests in the Program) through the harness-only `tests_print.rs` inverse of
  `tests_lower`: print, lower, Debug-equal per case, second print byte-identical, with the
  session's bare-name target rule (session.rs:636-641) required to name the same function
  as the module-qualified lowering. The Python test-data policy is unchanged (no records,
  variants, calls, Result inputs), top-level blocks use the module-less surface below, and
  `success()` is untouched. Module-less TYPES (A1) have a
  `#[cfg(test)]`-only surface with the real empty owner (`top_level.rs`,
  `print_toplevel_type`): unwrapped `Operation::Type`s, type-vs-module and type-vs-variant
  collisions refused, shared/lifecycle/route/source not part of it. A2 adds module-less
  functions to the same surface (`lower_toplevel`, `print_toplevel_function`): BARE
  signature keys for top-level functions only, so a bare call resolves exactly as the
  runtime's top-level-first `Program::get_function`; module functions stay reachable only
  qualified and a bare module callee is refused. `expressions.rs::call` classifies a bare
  name with a signature as a user call (`local_user`) and builds the resolved name with
  `types::owned_name`; do not return to `resolved.contains('.')`, which misclassifies owner
  "". The printer prints a bare user call only at owner "" for a real `program.functions`
  entry. It has no authority model and must never be exposed to native tools. Current
  strict counts after G7: 334 verified / 58 invalid / 181 unsupported (54 files, 573 entries). G2
  allows `input` as a parameter/local/capture/value/field-base name ONLY
  (`forbidden_binding`); as a callee, function, type, member, shared, annotation or attribute
  name it stays forbidden. A3
  verifies whole programs (top-level types/functions printed as one submission, rebuilt
  between types and modules, `equality::program`) and B2 module-less stored tests (the
  lowered target must equal the runtime string `get_function` resolves; module blocks keep
  the #110 session rule). The test printer refuses negative numeric CONSTANT nodes (#111,
  keyword shorthand only; `Unary Neg` spellings are exact). The
  remaining top-level function blockers are G3 anonymous struct literals, G4 json_get
  typing and G5 shared `Some` (owner decisions). G7 now permits a plain exact-known
  inner T return under Result[T], preserving the Return AST; `result_returns.rs` owns
  this return-only rule. Unknown-containing types do not qualify for this new rule,
  and nested Result/outer Option require explicit wrappers (Some shares Value::Ok).
  List[Option[T]] retains its distinct outer representation and is admitted.
  Native Python test/evaluate checks current reachable user-await outer Result agreement
  in both directions before execution. The iterative snapshot-local walker reuses tree
  lookup and function identity; categorical errors disclose no hidden callee metadata.
  This is not full payload checking or transitive authorization. General compatibility, arguments,
  assignments and explicit-constructor payload admission are unchanged. G1 is done
  (`pseudo_fields.rs`): `.len`/`.is_empty` on str/list and
  `.is_ok`/`.is_err`/`.unwrap` on a Result lower as PROPERTIES to the same `FieldAccess`
  AST, only for a non-struct base (a struct field of the same name wins); Map/Set `.len`,
  Result `.len`, `error`/`unwrap_err`/`is_some`/`is_none` stay refused. The text is not
  runnable CPython. `vm/field_admission.rs` (#114) sends these functions to the tree.
  #90 remains open: VM Boolean consumption uses `Value::is_truthy`; the reviewed
  JIT immutable normalization slice shares one emitter across AND/OR, NOT and
  Check/Branch/While (Check is now pre-execution-refused by failure admission).
  Local/rebound condition dependencies conservatively refuse
  compilation, including mutable loops; normal eval then uses VM/tree fallback.
  This does not establish general tier parity. Narrow JIT NOT
  uses proven semantic representations: Float (including zero/NaN) is always
  truthy, String tests length rather than pointer-nullness, and unproven string
  locals/call results refuse compilation. Direct JIT tests must use Int-return
  wrappers, not `call_i64` on I8 Bool returns. Do not claim general tier parity
  from these narrow tests or relax async admission to manufacture IO evidence.
  NOT operand emission captures immutable scalar/String-length entry SSA Values,
  not mutable compiler scratch names, and recursively handles literals, untouched
  parameters, NOT and numeric negation. Binary operands and argument-bearing
  calls conservatively refuse JIT compilation before execution (tree/VM fallback).
  Zero-argument scalar calls execute once; this does not certify their internals.
  Condition-only policy additionally admits homogeneous scalar comparisons and
  exact Bool/Int/Float user-call arguments through identity-safe recursive
  emission (no generic scratch-name lookup). It refuses IO/async callees and
  explicit `[fail]` argument callees, including nested arguments, pending #97.
  #97 scalar Check now unwinds with a typed language failure instead of returning Err
  as data to a caller that could continue effects. Result/capture graphs select the
  normalized interpreter path; lifting VM quarantine remains a separate acceptance gate.
  JIT Check/fallible graphs refuse before compilation rather than executing the historical
  native trap. #96 JIT relocation remains open too.
  Float Eq/Neq condition operands deliberately refuse JIT compilation: tree/VM
  finite equality uses epsilon rather than IEEE equality. Ordered Float
  comparisons remain admitted. VM unordered Neq (NaN/identical infinities)
  disagrees with tree already, so fallback is not a fix for that #90 residual.
  A separate partial TelegramBot port lives in `tests/fixtures/telegram-python-port`
  (now 42/42 original TelegramBot functions plus a separate partial text entry,
  three separately inventoried pure intake validators, a pure upload classifier
  and a pure batch update-id validator,
  and 18/18 TgFormat functions; module/stored-test and
  intake/delivery surfaces still incomplete). Polling slice A ports extract_update_id,
  init, handle_update and dispatch_updates: spawn is fire-and-forget (at-most-once, no
  outcome, same-chat order within a batch unguaranteed), invalid update ids and
  non-array results are skipped/"no result" instead of terminal. Slice B ports
  poll_once: explicit builtin Result, json_valid guard (invalid body), dispatch
  "no result" and "dispatched 0, skipped " outcomes each back off 3s instead of
  hot-looping or aborting; valid/empty/mixed pages stay immediate ok; dispatch_updates
  itself still treats a malformed top-level body as terminal. Slice C ports poll_loop
  (42/42): bounded to 1000 cycles, then the latch is cleared BEFORE one zero-arg
  successor spawn ("poll loop restarted", #107: the tree while cap is 10000); stop
  requests spawn nothing; an outer error from poll_once keeps the latch set and cannot be
  caught in-language (#109); the Bool latch is not an identity (stop + immediate restart
  during an in-flight getUpdates can run two loops). Non-function surfaces (module doc,
  token_path, allowed_path, startup, route) are ported in slice D1: exact module doc,
  startup (qualification + effect spelling only, no semantic repair; the token only as
  BYTE length), and the webhook route verified through a dispatch replica AND the real
  HTTP layer (200 "{}" + spawn; no header/secret access, #108). The 61 original
  TelegramBot stored cases each have one manifest disposition (D2): 41 carried verbatim
  (frozen/corrected .ax/Python Debug-equal), 2 carried in the .ax only (Result literals;
  Python literal policy unchanged), 1 covered by a named default-state test, 14 redacted
  and unrecoverable (never recovered from non-fixture sources), 3 obsolete by ledgered
  repair. TgFormat's 57 stored cases (D3) are dispositioned the same way: 39 carried,
  17 carried in the .ax only (calls in test data), 1 redacted and unrecoverable. No port
  inventory remains pending and #86 is not complete.
  Media fixtures add file-path/error
  helpers and three attachment handlers, using actual Attachment annotations.
  The NEW test-only PortMediaDriver is excluded from original-function inventory
  and source parity. Both-origin normal async tests intercept acquisition,
  takeover and reply requests with tiny memory-backed Attachments; no Runtime IO
  loop or live service is used. Lookup keeps String/error-String results through
  an explicit builtin Result binding; attachment handlers preserve takeover
  prefixes and propagate TgTextTurn without added retries. Callable outbound
  attachment closure now returns TgSendAttempt: nonempty text must be Sent before
  one upload; empty text permits upload-only; actual builtin Result is classified
  by the new pure telegram_upload_attempt with exact Bool JSON status. Uploads
  never retry. Test-only PortOutboundDriver acquires tiny memory bytes before the
  normal async caller, so invalid destination means one acquisition/zero delivery,
  not total0IO. Path upload tests intercept an inert path before any Runtime IO loop.
  send_audio_file rejects caption '&' before IO: raw multipart extras cannot escape
  it (#106); '=' Unicode/newline/% are preserved, percent encoding is not a fix.
  General arbitrary-caption fidelity remains unresolved. extract_reply preserves
  original AST/doc and inherited json_get polymorphism; it is not the classifier.
  #99/#104 remain open. Original-name process_update now integrates
  text/photo/video/voice through normal async both-origin raw-payload tests without
  a driver. It deliberately adds conservative consumed-field preflight, including
  represented int IDs/date and date<=9223372036854775, before authorization/IO;
  optional message_id remains missing-compatible, latitude allows int/Float, and
  unconsumed harmless metadata is ignored. Malformed lower-priority consumed media
  also defers0IO; empty photo arrays retain fallthrough. Existing helper bodies and
  NEW text entry stay unchanged. TgTextTurn propagates handler outcomes; getFile
  guidance returns a delivery attempt, not media-processing success. Awaited-call
  arguments retain original spelling modulo qualification/folded duplicate group
  dispatch. describe_update_kind preserves exact original AST. Poll/webhook/lifecycle
  remain pending; no deployed or callback ACK certification follows. The two-function
  voice helper slice is an explicit #91 authored repair, not equivalence to the
  malformed frozen transcribe_voice. Its getFile Result error is scoped to the
  outer arm; path and transcript require nonempty JSON Strings, malformed JSON
  and binary outer errors terminate, and upload Result passes intact to the pure
  helper. Trusted Result helper tests use tree/direct VM on both origins without
  weakening Python literal policy; normal async tests assert 1/2/3 ordered requests
  and memory-backed audio bytes/name/MIME. Strict global corpus is unchanged; only the port's
  original-equivalence subset handles the malformed function through a separately
  named repair gate, while original inventory and corrected full AST gates include it.
  Its source hashes,
  per-function ledger and explicit pending inventory must stay distinct from the
  frozen parity corpus. Its originally RED direct-VM gate for #98 now passes
  with the pending narrow `vm/equality.rs` repair: differing Value variants except
  Int/Float numeric pairs share tree `Value::matches` and negation. Port fixture
  bytes/expectations were unchanged; their observed-status metadata preserves the
  earlier RED checkpoint. Do not add exemptions or coerce the fixture. Same-family
  and mixed-numeric comparison behavior is unchanged; numeric coercion, unordered
  Float Neq and unsupported aggregate equality remain #90/#86 residuals. The
  frontend json_get String signature remains an unresolved catalog mismatch.
  #82's original wrapper checkpoint pinned normal user-await dispatch failing
  with zero requests (bare AND qualified). Admission selects tree before execution.
  C-core now preserves user-await Result values: the frozen misannotated String wrapper
  returns builtin Result and BOTH its success/failure cases hit the String consumer error
  after exactly one request. Its characterization was deliberately updated; frozen bytes
  were not. This is not the corrected/live app-union port, whose behavior remains tested.
  The later frontend acceptance permits Result user awaits with Result bindings;
  the frozen misannotated String wrapper remains a distinct characterization.
  Direct-tree success alone is not normal-dispatch evidence. IO contract fixtures use in-memory IoRequest
  replies, not the real Runtime IO loop or live services.
  Pure port expansion permits only ledgered local-call qualification, proved by
  a test-only AST transform against frozen originals. Preserve original corpus
  bytes and explicit pending inventory. #99 remains unresolved: plain/pre content
  of 5000 characters stays one part at limit 3800, exceeding Telegram's documented
  4096 cap. Do not describe that splitter as service-limit safe. Missing-ID/date
  quoted-empty metadata behavior is likewise inherited, not a desired API contract.
  The corrected send-chain fixture separately ledgers two intentional signature/
  control-flow repairs: explicit builtin HTTP Result matching and an application
  TgSendAttempt union. Only parsed Bool false allows one plain fallback; transport
  uncertainty or rejected fallback stops later chunks. Malformed JSON is terminal,
  and empty/out-of-range inputs cannot manufacture Sent from a prior response.
  Both-origin normal-async request fixtures prove local ordering/count, not
  exactly-once remote delivery. Frozen wrapper characterization remains unchanged;
  that send-chain checkpoint did not relax Result user awaits or the frontend catalog.
  The higher corrected send_message/send_reply callers propagate the app outcome
  unchanged with no additional retry; combined-module tests include TgFormat.
  This is not callback acknowledgement certification: #104 tracks generic spawned
  callbacks treating any evaluator Ok (including app Rejected/Uncertain unions)
  as Completed and then notified. Do not parse formatted strings or hardcode
  Telegram tags in generic runtime code to hide that integration gap.
  The partial process_text_update fixture is NEW, not original process_update
  parity. #105's original RED gate is preserved and now passes after adopting
  json_kind only in that NEW entry. IDs/date require int; chat.type/text require
  string before conversion/auth/IO. Present media defers even when empty/null/false;
  legitimate Strings null/{} /[] /prefix-leading are accepted as text. List identity
  and prefix/rendering type heuristics were removed, not widened. Keep the original
  helpers/handlers and the full 11-key presence regression intact; no exemptions.
  Shared authorization helpers use tested normal tree fallback, not direct-VM
  parity; fake state must be restored. Neither routing fixtures nor returned
  NoImmediateReply/DeliveryAttempt certify callback acknowledgement (#104).
  Frontend type catalog keys and resolved Named types retain declaration owners
  internally (`Owner.Type`); parser/Python emission preserves the original bare
  AST name. Function signature types carry the callee's owner, never the caller's.
  Local declarations win, then top-level types, then only a unique cross-module
  fallback; ambiguous fallback fails without listing candidate owners. Feed only
  the authority-filtered catalog to native lowering/inspection. Do not flatten
  it back to bare-name keys: same-named struct/union declarations otherwise overwrite.
  A filtered catalog's unique fallback is NOT sufficient for native mutation:
  stripping its owner can make full-program validation select a denied type.
  `python_frontend/type_bindings.rs` runs before canonical execution and verifies
  owner/kind/declaration-order invariants against metadata-only full-state lookup,
  including global union-name reclassification. Colliding external fallback is
  refused categorically before hidden schema validation, never by sanitizing its
  errors afterward. Keep this guard after filtered lowering and before execution.
  Never interpolate Python text into commands
  or add Python eval/exec/import execution. Native normalization lives in
  `api/native_tools/frontend.rs`: Python mutation admission first checks the
  complete supported syntax and collects direct references without looking up
  runtime signatures. Only after pinned module Write/reference Execute checks
  does dispatch supply an authority-filtered signature catalog to strict final
  lowering. Earlier batch calls may create those functions. The canonical output
  then traverses the existing native checks, executor, tests and CoW write-back.
  This split keeps single-function Python inspect→resubmit repairs usable without
  consulting hidden signatures or weakening the type rules.
  Mutate/test require `language:"python"` plus one explicit `module`; evaluate
  accepts `language:"python"` for literal argument syntax; source inspection
  accepts `format:"python"`. Defaults remain Adapsis. Python inspection prints
  supported AST regardless of origin using an authority-filtered signature
  catalog; unauthorized signatures cannot change printer diagnostics. No origin
  metadata, sidecars or persistence schema is added. Submitted functions replace
  old Arcs as `.ax` does; unsubmitted functions retain identity. No inferred deletes.
  `python_frontend/runtime_arguments.rs` enforces exact Python argument counts/
  keyword sets and direct submitted call async/await semantics against same-source
  and actual runtime signatures. `_0`, `_1`, … parameter/keyword names are reserved
  to avoid the legacy positional-carrier ambiguity; `.ax` binding stays unchanged.
  Python test/evaluate/inspection conservatively rejects current builtin-name
  collisions in both `BUILTINS` and `IO_BUILTINS` (including aliases and user
  union variants), so later `.ax` edits cannot intercept installed helpers.
  #83 has ONE narrow exception in `api/native_tools/scalar_entry.rs`: effect-free
  scalar signatures, straight-line fresh scalar lets followed by return, and only
  scalar literals/bound parameters/earlier locals/unary/binary operators. Python
  execution additionally requires scalar literal argument carriers/expectations,
  no matchers/after-checks, and a call-free scalar proof for ALL potentially
  initialized shared defaults (empty binding set, full current Program). Runtime
  eagerly evaluates unrelated defaults, so checking only the target body is unsafe.
  Inspection uses the same body proof with authorized metadata; it executes no
  defaults. Root named tests/evaluation bypass function-call stubs, and admitted
  bodies/data contain no call site that could invoke one. Recheck each snapshot;
  preserve unconditional reachable-constructor checks and existing literal guards.
  Calls, fields, constructors, unknown/shared identifiers, control flow, effects,
  list data, or unproved defaults retain conservative admission. This is not full
  helper-specific resolution or general #83 completion. Tests remain data-only.
  Supported syntax/exclusions and runnable commands: `examples/python/README.md`.
  New bindings/parameters/returns are annotated; imports, mutable collections,
  dynamic calls and ambiguous numeric/truthiness behavior fail with source
  locations. For targets cannot escape/reuse; `len`/`length` accept a statically known
  list, str, Map or Set (str length is BYTES). Pure async calls
  need await; synchronous calls cannot be awaited. IO await also needs the inert
  io effect declaration. Awaiting Result-returning user functions is admitted
  only when bound as Result, preserving the wrapper (#82); direct IO builtin
  Result awaits remain supported and their actual wrapper is covered by an
  in-memory IoRequest reply fixture. Trusted offline fixtures introduce no live
  IO or Python mock installation. Numeric literals exclude i64::MIN because canonical
  `.ax` parses its positive magnitude before unary minus; fail explicitly rather
  than approximating it. `parser/test_values.rs` preserves quoted/escaped delimiters
  and `->` inside literal test data, including nested list/record arguments.
- `/api/ask` also requires explicit `<code>!done</code>` for
  conversational completion. Prose/empty replies receive an execute-or-finish
  correction within the existing iteration budget; exhaustion is reported as
  incomplete (`has_errors`), including after successful operations. Completed
  operations and their results stay in context and are never replayed by the
  correction. Behavioral HTTP/mock-provider tests: `cargo test --release
  ask_completion_tests`.
- HTTP and streaming turns capture a single immutable program baseline at turn
  start. Completion requires tests for new/replaced functions, not unchanged
  pre-existing test debt (#66). `execute_code_with_baseline` checks CoW function
  identity; never refresh that baseline between rounds or changes could escape
  the gate. Standalone and mixed-operation `!done` use the same executor. Direct
  mutation/evaluation callers without a turn baseline retain the global gate.
- Per-context conversation history: `ConversationManager` in `SessionMeta`
- `llm_takeover(context, message, reply_fn, reply_arg[, permission_model])` — conversational LLM with history
- Iterative loop: call LLM → execute code inline → feed results back (max 10 rounds)
- `!agent` breaks the loop for background work with completion callback
- Reply callbacks: `reply_fn(reply_arg, text)` for text, `reply_fn_with_attachment(...)` for files
- Conversations persist across restarts via session serialization
- Optional `permission_model` parameter overrides which model's permissions are used for the
  program summary shown to the LLM. Used for non-admin Telegram users (e.g. `"gemma4s"` = execute-only).
  The override only restricts visibility, not actual execution permissions.
- A takeover turn ends only on explicit `<code>!done</code>` (or after executing a
  background `!agent`). Prose without code receives corrective feedback and another
  iteration. This prevents promises such as "I'll inspect that next" from ending the
  turn without doing the work; final prose and clarification questions must include
  the hidden completion command.
- The model estimates each takeover's action-round budget in its first non-empty response
  with `<iteration_budget>N</iteration_budget>`. Missing estimates default to 10; estimates
  are clamped to the hard safety ceiling of 50 total generations, including a final summary
  and any empty-response replacement. Protocol-correction rounds may extend an exhausted
  estimate only while that shared generation budget remains.
- A completely empty model response is retried once, with one replacement round when
  needed and the hard ceiling permits. The correction explicitly resumes from the
  latest tool result and forbids
  repeating completed or irreversible actions; two consecutive empty responses still
  terminate the turn with the existing user-visible error.
- **Provider-facing context is prefix-structured (`context_log::ContextLog`).** All three
  conversation builders (`ask`, `ask_stream`, `llm_takeover`) send one stable system
  prefix, append-only history, then one user-role multipart envelope containing the volatile
  runtime tail and current speaker input in distinct content blocks. The tail must not use
  `system`: the gateway's Anthropic and ChatGPT adapters hoist every system message ahead of
  history. The first envelope block is runtime-injected; trusted runtime blocks end at the
  final text block, which the runtime prefixes as untrusted speaker input. A speaker can copy
  either marker only inside that final block and therefore cannot manufacture an authority
  block even when Anthropic merges consecutive roles. `ContextLog` marks the stable prefix
  and history end as explicit cache boundaries and moves the changing bounded-history
  omission notice into the envelope.
- `/api/ask` and `/api/ask-stream` serialize the `main` conversation, persist the user before
  the provider call and one complete assistant turn afterward, and never hold a state lock
  across generation. The next request must see the prior `user, assistant` pair exactly once.
- Explicit breakpoints require llm-gateway's top-level `message.cache_control` passthrough
  to Anthropic content blocks; older deployed gateways ignore that field and retain only
  their last-three-message heuristic. Cache token counters are logged raw per context with
  latency. Do not add cache reads to provider totals: ChatGPT counts reads inside
  `prompt_tokens`, while the current Anthropic translation reports them separately.

## Ladybug Episodic Memory

- LadybugDB is the sole authoritative long-term memory database. The default path is
  `~/.config/adapsis/memory.lbug`; override it with `ADAPSIS_MEMORY_DB`. This claim was
  false until #42: `persona-notes.md` was a second store — one global markdown file, no
  scope, no provenance, written by `Wolfi.remember` and read into the persona under a
  hardcoded German heading naming one person. #41 made the read opt-in per context; #42
  replaces the file with canonical memories.

### Canonical vs extracted memories (#42)

Two kinds of `Memory` node, distinguished by the `canonical` flag that existed in the
schema since day one and was never set:

| | extracted | canonical |
|---|---|---|
| written by | compaction (`llm_handlers.rs`) | `memory_remember` builtin |
| provenance | `DERIVED_FROM` messages, `EXTRACTED_BY` a run | `ASSERTED_BY` a principal |
| confidence | model estimate | fixed 1.0 |
| injected | up to 5 deduplicated representatives ranked against the current message | **all distinct assertions within hard caps, every turn, unranked** |

- **The split is observation vs instruction.** Ranking is right for "the mobile header
  overflows" and wrong for "I prefer large text" — the latter must hold on a turn that
  mentions neither text nor size. So canonical memories bypass ranking entirely and are
  bounded by count (`CANONICAL_MEMORY_LIMIT` = 40) and characters
  (`ADAPSIS_CANONICAL_MEMORY_CHARS`, default 8000) instead. Equivalent active records
  with identical governance collapse to the newest representative before the limits apply.
  Truncation drops the **oldest**, because a new assertion is usually a correction of an
  old one; selected representatives are then rendered oldest-first.
- **`recall_authorized*` filters `canonical = false`.** Otherwise a canonical memory would
  appear twice in the prompt — once injected verbatim, once occupying a ranked recall slot.
- **`create_memory_node` is the only `CREATE (:Memory …)`.** `canonical` and `confidence`
  used to be string literals inside the extraction path's Cypher, which is exactly why the
  flag sat unused for so long. Extraction passes `false` structurally, so an inference
  cannot become an assertion.
- **Scope is governance, not a new concept.** Recall requires membership in *every* group
  governing a memory, so a memory governed only by `access:global` — which every principal
  joins on first message — is readable everywhere. `MemoryScope::{Context,Global}`. There is
  no `global` column.
- **`memory_remember` / `memory_forget` are builtins, not module functions.** `remember`
  was a method on the `Wolfi` family-persona module, so it had to be *handed* a scope and
  could be handed the wrong one: a feature request about an unrelated codebase was filed as
  something learned about FamilyUser. A builtin reads context + principal off
  `coroutine::TurnIdentity` and fails closed outside a conversation. **Do not add an
  argument that names a context or a principal to either one.**
- **`memory_inspect([query])` is the user-facing, turn-bound inspector.** Its structured
  core is `memory_graph/inspect.rs`; conversational rendering is separate so another
  transport can serialize `Inspection` later without accepting identity arguments.
  Zero/empty query shows newest-preserved canonical assertions oldest-first plus recent
  extracted records. A non-empty query resolves an exact id first, then topic search;
  embedding and similarity-query failures have distinct truthful lexical-only labels. A
  global memory's private `ORIGINATED_IN` context is redacted to `global` in the structured
  result, not merely hidden by the renderer. The shared context-authorized predicate
  requires both membership in every governing group and governance by
  `access:<current-context>` or `access:global`, even for the administrator. Do not
  replace it with a broader principal-only recall query or expose paths, embeddings or
  Cypher in the renderer. Output is capped at 12k characters, content at 2k, canonical
  rows at 40, extracted rows at 10, citations at 3 and lexical candidates at 200.
- **Memory correction/lifecycle (#57):** schema version 2 adds optional
  `TemporalValidity` nodes plus `HAS_VALIDITY`; omitted bounds are absent, never
  inferred. `Memory.status` is `active`, `superseded`, `disputed` or `forgotten`.
  Only active records participate in canonical injection, ordinary inspection and
  ranked recall; exact authorized inspection may show every state and ACL-filtered
  `SUPERSEDES`/`CONTRADICTS` links. `memory_correct(old_id, replacement[,
  valid_from_ms[, valid_until_ms]])` is turn-bound and inherits the target's exact
  single-group governance. Context participants need Contribute membership; global
  correction additionally requires a write-capable turn. Replacement IDs hash the
  actor, inherited scope key (the context id for context records, `access:global` for
  global records), target, whitespace-normalized content and validity, so exact retries
  survive restart and a global retry from another authorized context cannot create a
  false dispute. Competing successors are all disputed and
  therefore unrecalled; correcting a disputed branch creates one active resolution
  that supersedes the authorized contradiction cluster. A rollback is another new
  correction, never status rewinding or deletion. Keep correction creation, edges and
  status transitions in one Ladybug transaction under `MemoryGraph.write_lock`.
- **Forget is a status change**, not a delete — provenance survives. "Not yours" and "does
  not exist" return the same message on purpose; distinguishing them turns `memory_forget`
  into a probe for other conversations' memory ids.
- **`--admin-id`** replaces the hardcoded `telegram:user:100000001` in the ingest path (and
  the hardcoded `"Marenz"` display name). Process-level via
  `memory_graph::set_admin_principal`, set once at startup — it does not vary per request,
  and two ingest paths disagreeing about who the admin is would silently mis-ACL every new
  group. Defaults to the old literal so an un-flagged process is unchanged.
- **Migration:** `adapsis memory-import-notes --context <ctx> [--principal P] [--global]
  [--dry-run] [notes.md]` turns a legacy `persona-notes.md` into canonical memories.
  `--context` is required — a global file has no context of its own, and guessing one is
  how these notes ended up filed under the wrong person. **Run this on any node whose
  `persona-notes.md` is non-empty before the read path is deleted.** On `here` the file is
  0 bytes (the #38 path bug sent every write to `/home/adapsis/…`); on **edox** `$HOME` *is*
  `/home/adapsis`, so writer and reader agreed there and the file may hold real notes.
- **Still to do (blocked on edox being reachable):** delete `Wolfi.remember` /
  `Wolfi.forget_all`, drop `persona_notes_path` and the `{{persona_notes}}` marker.
- Every incoming message is written before inference and linked to its `Context`,
  `Principal`, dynamic `AccessGroup`, original platform message ID, and timestamp.
  Telegram participants receive Read+Contribute membership when first observed;
  `telegram:user:100000001` also receives Manage. Membership grants the full context history.
- **The principal comes from `source_metadata.speaker_id`, never from the message
  text.** An intake module fills that field (`TelegramBot.speaker_principal`);
  `takeover_principal` reads only it, falling back to a context principal. The
  `[user:<id> <name>] ` prefix a group module prepends is *prose for the model* —
  `strip_sender_prefix` removes it from stored content and the recall query and
  nothing else. This split exists because identity used to be parsed back out of
  that prefix: when #38 item 4 added the display name, `[user:100000103 Kata] `
  silently became the principal `telegram:user:100000103 Kata`, a second identity
  for the same person that would re-mint on every Telegram rename. Adding a field
  to the prefix must never be able to change who someone is. Guarded by
  `telegram_payload_resolves_a_stable_principal` (bundled_modules_tests.rs), which
  drives a real payload through the module into the runtime — both sides had
  their own tests and neither caught it.
- A context with no per-speaker metadata resolves to a context principal
  (`telegram:group:<id>`, `kind = 'group'`), not a fabricated user. Stripping
  `telegram:` off a group context used to yield `telegram:user:group:<id>`.
- Memories link explicitly to every origin context/access group and source message.
  Cross-context recall requires membership in every governing group. `DENIED_TO` provides
  a future per-memory/per-user override.
- Attachments are SHA-256-addressed under `~/.local/share/adapsis/attachments/`; Ladybug
  stores the hash, MIME type, platform reference, storage path, and source-message edge.
- `fastembed` runs `intfloat/multilingual-e5-small` in-process for German/English recall.
  Its cache is `~/.local/share/adapsis/.fastembed_cache`.
- Automatic recall semantically ranks a bounded ACL-filtered candidate set, collapses
  equivalent records only within the same channel and exact governance set, and injects
  compact source citations. Polarity and changed literals prevent false merges; uncertain
  pairs remain separate. Lexical search remains an inspection fallback, not a prompt-ranking
  input.
- Compaction is non-destructive: raw messages remain forever. The active conversation
  model creates immutable `Episode` summaries and source-backed index `Memory` nodes.
  Failed compaction leaves raw history pending and uses prior episodes plus recent turns.
- **Per-model limits (#68), `src/model_limits.rs`:** context window, output limit, request
  and chunk-gap timeouts per model, from `ADAPSIS_MODELS_FILE` /
  `~/.config/adapsis/models.toml`, then sourced built-ins (exact ids; `provider/` prefixes
  are supported): anthropic claude-opus-5 / opus-5-5 (output 128k MEASURED on opus-5; 1M
  context from Anthropic /v1/models metadata + context-window docs), sonnet-5 / sonnet-5-5
  (registry), openai/gpt-6-astra 1.05M/128k (registry). Every built-in needs a source comment (measured error with date,
  provider metadata/docs or registry); never a guessed value. ChatGPT-route models have no entry
  except chatgpt/gpt-6-astra, gpt-6-sol, gpt-6-luna and gpt-6.1-sol (872k: Codex metadata
  max_context_window; 470,030 prompt tokens measured accepted on astra); others use the
  unknown default `ADAPSIS_CONTEXT_TOKENS` (128k).
- **Token projection is calibrated per conversation and model (#68 follow-up).** There is
  no fixed chars-per-token constant. `Conversation.token_calibration` (persisted) holds the
  last answered request as `(serialized text bytes, provider input tokens, model)`.
  `request_meter::SentRequest::measure(model, messages, extra)` takes the bytes before the
  call (messages + native tool schemas, image base64 excluded) and `.calibrate` stores them
  with `LlmUsage::accounted_input_tokens()` after it (ask, ask_stream, legacy and native
  takeover, `!watch` self-trigger). The view checkpoint AND the compaction trigger convert
  with `conv.bytes_per_token(model)`: the measurement counts only for the model that made
  it (another tokenizer), otherwise `DEFAULT_BYTES_PER_TOKEN` = 2.5 (MEASURED 2026-11-10:
  the admin DM's ≈1.46 MB view was counted as 593,447 tokens; the old 4 chars/token would
  have let it grow to ≈1.13M real tokens). Clamped to 1.5–6.0; ignored when input tokens
  are zero/missing, under 2,000 text tokens, or the client had to shrink the request.
  Images cost a fixed `IMAGE_TOKENS` (1,600) each and are subtracted before the ratio. The
  ratio moves only the checkpoint position, never the bytes of the visible history. New
  request paths must call `SentRequest::measure`/`calibrate`, or their conversation stays on
  the default and its cache evidence is never recorded.
- **Units:** budgets are serialized bytes (`model_limits::serialized_len`, a JSON string)
  and tokens derived from them. `ModelLimits::max_message_bytes` (1/8 window at the fixed
  default ratio, and 1/8 of the route limit) is UTF-8 bytes of message text;
  `MemoryGraph::pending_context_messages(context, max_bytes)` and the compaction thresholds
  count serialized bytes. Do not introduce character counts for budgets.
- **Measured runtime tail:** `Conversation.tail_bytes` (persisted) is the first request of
  the previous turn minus that turn's view: memories, episodes, activity, authority text
  and tool schemas together. `ViewPolicy::for_model` uses it (plus a quarter of margin, at
  most 1/4 of the route limit) instead of the fixed 16k-token / 64 KiB allowance, which
  remains the no-measurement default.
- **A capped message says so.** `conversation_view::truncate_middle` keeps head and tail
  (cut on character boundaries, at most `max_message_bytes`) and inserts `[... N characters
  omitted here by the runtime to fit the model context; the stored message is complete
  ...]` with the exact count (short form for small caps). The cut depends only on the
  message and the cap; persisted history is never changed. A native batch containing a
  message over the cap is still dropped from the view whole.
- **Cache boundaries sit only on assistant messages (2026-11-10 fix).** The gateway merges
  consecutive same-role messages (a tool result is a user message for Anthropic), puts an
  explicit `cache_control` marker on the final block of the MERGED message, and rejects two
  markers in one merged message with HTTP 400 ("distinct cache boundaries that collapse
  into one Anthropic message"). dd8cea4 marked the last non-empty history message; after a
  native turn that is the `finish` tool result, so (a) the marker landed after the volatile
  runtime tail and (b) one failed turn later every request of that context was rejected
  until restart. `context_log::is_cache_boundary` now allows a boundary only on the last
  message of an assistant run with text or tool calls. `context_log::gateway_boundary_errors`
  (test-only) is the oracle for the gateway's rules; use it on recorded wire requests in
  any new request-building test. Do not mark user, tool or envelope messages.
- **Cache lifetime per route (`model_limits::BUILT_IN_CACHE`, `src/cache_lifetime.rs`).**
  Each route has a requested marker lifetime (`cache_ttl`: "5m"/"1h", sent as
  `cache_control.ttl` on EVERY boundary via `ContextLog::set_cache_ttl`, set in
  `report_prompt_epoch`) and a default lifetime in seconds for the cold-start rule.
  Built-ins need a source comment (MEASURED / DOCS / UNVERIFIED; UNKNOWN is a valid entry):
  anthropic/* requests 1h = 3600 s (owner decision; DOCS + MEASURED), openai/* 1800 s
  (DOCS), deepseek/* 3600 s (UNVERIFIED), moonshot/* 300 s (DOCS), chatgpt/* 1800 s (DOCS +
  MEASURED, probabilistic, see below), deepinfra/* and xiaomi-mimo/* UNKNOWN. `models.toml`: `cache_ttl`, `cache_ttl_secs` (0 = never cold-start),
  `cold_start_tokens` (default 100,000, env `ADAPSIS_COLD_START_TOKENS`).
- **`prompt_cache_key` (ChatGPT/OpenAI routes).** Every conversation request carries a
  stable key, `cache_lifetime::prompt_cache_key("conv", context, model)` =
  `adapsis-conv-<32 hex>` (hash of purpose, context id and model; never message content, so
  it survives checkpoints and cold starts; the context id itself does not leave the host).
  `http_context::conversation_client` builds the client for ask, ask_stream, legacy and
  native takeover and the `!watch` self-trigger; compaction uses `"compact"`. `llm_call`,
  agents, agent summaries and CLI commands send NO key (the gateway derives one from model,
  instructions and first input item). The gateway (llm-gateway #4, c4f0d36) validates the
  key (1–64 printable ASCII), sends it upstream on chatgpt (body + `session-id` header) and
  openai, and drops it for every other route; older gateways ignore the unknown field.
- **Probabilistic routes (`model_limits::PROBABILISTIC_CACHE`, `cache_probabilistic`).**
  chatgpt/* has a lifetime of 1800 s (DOCS minimum; MEASURED readable at 3900 s) and
  PROBABILISTIC hits (MEASURED ≈60% at every gap from 12 s to 65 min). On such a route a
  miss is not evidence: `classify_for` reports `miss-not-evidence`, it is never recorded,
  stored misses are ignored, and only hits can raise the lifetime. One uncached answer
  never means "cold"; the cold start is decided by the pause alone. Its usage reports
  reads inside `prompt_tokens` and no writes.
- **Learned lifetime.** `ConversationManager.cache_lifetimes` (persisted with the session,
  per model id) records, for every answered request where a hit could be expected, the
  pause since the conversation's previous request (START to START: the provider counts the
  lifetime from the start of the request that wrote or read the entry) and hit/miss
  (cache read ≥ 50% / ≤ 10% of the expected prefix). A hit at or past the default raises
  the lifetime to that pause + 1 s. One miss changes nothing; the second-shortest miss is
  a confirmed upper bound that lowers the lifetime below the default, or makes an unknown
  route known. Misses expire after 14 days; values are clamped to 60 s – 7 days. Not
  recorded: first request after a restart/checkpoint/prefix change, prefixes under 8,000
  tokens, routes that report no cache reads, shrunk requests, REFUSED turns (MEASURED: a
  refused request reports a cache write the next request cannot read; legacy
  `LlmOutput.refused`, native refusals are provider errors), and the turn after a turn whose
  first request failed or was refused. `TurnProbe` (in memory) carries this per turn.
- **Cold start (`request_meter::turn_view`).** At the START of a turn only: if the pause
  since the conversation's last answered request is at least the route's effective
  lifetime, `view_start` moves once so the history is at most `cold_start_tokens` (and
  never more than a budget checkpoint would leave: target minus overhead), same mechanics
  as a checkpoint (user-turn boundary preferred, batches whole, newest turn kept). It is
  reported as a checkpoint, so the cache epoch resets. Unknown lifetime or unknown pause
  (session from before this field): no cold start. Rounds inside a turn never call
  `turn_view`; `calibrate` never moves the view. Log lines: `[llm:<ctx>] cold start:
  pause_s=… lifetime_s=… (learned|default) history_tokens=A->B view_start=X->Y` and per
  request `[llm:<ctx>] cache: pause_s=… expected_tokens=… cached_read=… outcome=…
  lifetime_s=… source=…` (metadata only). `checkpointed_view` must be called exactly once
  per turn, before the turn's first request.
- **Turn accounting belongs to lock-holding turns.** `Conversation.turn_probe` is one per
  conversation. `ask`, `ask_stream` and `llm_takeover` hold `takeover_context_lock(context)`
  for the whole turn, so two turns never share it. The `!watch` self-trigger holds no lock
  and is not a user turn: it builds a plain view (`self_trigger_messages`, no cold start)
  and calls only `SentRequest::calibrate_tokens_only`. It is not the conversation's last
  request for the pause, measures no tail and records no cache evidence. A new request
  path that is not under the turn lock must do the same.
- **Request body limit:** `max_request_bytes` per model (default `ADAPSIS_MAX_REQUEST_BYTES`
  or 60 MiB, under the gateway's 64 MiB limit from llm-gateway 332a3b0, deployed
  2026-11-10; the old gateway rejected bodies above 2 MiB). The view
  checkpoints on bytes as well as tokens (image base64 counted; images limited to ~30% of
  the route), recovered tail images are fitted to the remaining headroom, and
  `OpenAiBackend::send_request` refuses an oversized body before sending
  (`LlmError::RequestTooLarge`). A provider 413 is not retried and says so.
- **Self-heal (#68):** a too-large refusal (pre-send, 413, or a provider context-length
  400) is retried ONCE with `llm::shrink_request` (system prefix + current turn from the last
  runtime envelope + the newest whole history units that fit half the size), and the caller
  persists a forced view checkpoint (`http_context::after_llm_call`: only the newest turn
  stays visible), so a context can never stay stuck. Known tail parts (runtime/authority
  text, native tool schemas) are measured into the projection; 64 KiB covers the rest.
- **Timeouts (#68):** streaming requests bound the wait for response headers by the
  per-model chunk-gap timeout (120 s; `send_streaming_request`, not reqwest's per-request
  timeout, which would also cut long streams), then the same gap between chunks, plus the
  derived total; non-streaming requests (fallback, native `complete_turn`, which
  native takeover uses) cap at 20 min; any timeout is retried at most once. Compaction
  has its own 5-minute bound.
- Secondary `system` messages (runtime notifications, agent results) are sent as user-role
  `[Runtime notice …]` messages; providers hoist system messages ahead of history, which
  broke the cached prefix. The `!watch` self-trigger stores its event the same way and sends
  the checkpointed view of `main`, never the full history.
- **Checkpointed view (#68), `session/conversation_view.rs`:** the model history is
  append-only and byte-stable between checkpoints. `Conversation.view_start` moves only when
  the projected request (system + 16k tail allowance + output reserve + history) exceeds
  85% of the generating model's window, in ONE step down to 50%, preferring a user-turn
  boundary, never splitting a native batch, never cutting the newest turn. Per-message cap
  = 1/8 window (deterministic). Images: newest 3 / 20 MB from `image_start`, which jumps to
  the newest image message on overflow (one break per ~3 images). The omission notice stays
  in the volatile tail. Persisted history is never trimmed (the old `trim(51/101)` and the
  64-message / 60k-char view caps are gone). `ContextLog` marks a third cache breakpoint
  at the previous request's history end (from the epoch tracker), so long turns stay
  within the provider's breakpoint lookback.
- Compaction uses the same rule: it starts when uncompacted history passes 85% of the
  window minus reserved output, folds the oldest part toward 50%, one episode of at most
  1/8 window per turn, timed out at the model's request timeout, and retries at most once
  per context every 15 minutes. Compaction changes only the volatile tail, so it is no
  longer an epoch reset; only a view checkpoint is.
- `memory_cypher(query)` exposes ACL-filtered, read-only Cypher anchored at
  `MATCH (memory:Memory {id: $memory_id})`. It deliberately rejects query shapes that can
  escape the authorized memory anchor. The principal is **bound to the turn**, not passed
  in — see "Per-Context Identity" above.
- Operations: `adapsis memory-migrate <session.json> [--database PATH]` is idempotent;
  `adapsis memory-stats [--database PATH] [--context CONTEXT]` reports graph/checkpoint state.
- Quality gate: `adapsis memory-eval` loads only the versioned sanitized fixtures in
  `tests/fixtures/memory-quality/`, builds an in-memory graph, emits stable JSON plus
  a human summary, and fails on explicit thresholds (ACL leaks have zero tolerance).
  Extraction uses a frozen provider snapshot and is scored separately from retrieval.
  The baseline is manual-only; see the fixture README before adding a real regression.
- On openSUSE, LibreSSL development symlinks can shadow OpenSSL 3. `build.rs` links
  Ladybug's prebuilt archive to exact `libssl.so.3`/`libcrypto.so.3` SONAMEs.

## Telegram Bot (TelegramBot.ax)

- Native `cad_prepare(title, design_brief[, PLA|PETG[, use_recent_images]])` is a turn-bound
  prepare-only guest capability. `src/capabilities/cad.rs` launches the trusted
  runner with a design brief. `tools/cad_astra.py` pins CAD authoring/revisions
  to `chatgpt/gpt-6-astra` with high reasoning, independently of the chat model.
  No model fallback: unavailable/wrong-model/truncated replies fail preparation.
  Each job records cad-author.json and report.cad_model. Live Astra generation,
  geometry validation and slicing have been verified end-to-end.
  Geometry preparation allows up to three isolated checked attempts: actual
  sandbox/mesh failures and the previous source go back to the same pinned
  Astra author. Attempts keep private source/logs/failure.txt; only validated
  geometry is promoted for slicing/publication. Author-service failures fail
  immediately rather than being mislabeled as geometry problems. Job request.json
  preserves the brief for operator recovery. Never weaken mesh checks to make a
  failed generation appear successful.
  The brief is sent off-host to the external cloud author. Reference images are
  explicit opt-in only: the final Bool defaults false and consent must name that
  up to N images will also be sent to `chatgpt/gpt-6-astra` off-host.
  When true, `Request::Cad` carries the server-bound turn principal/context and
  dispatch queries Ladybug for at most three recent readable user images from
  that exact context, with a 20MB total budget. Callers never supply paths,
  attachment ids, principals or contexts. Rust sends bounded bytes only through
  trusted runner stdin; private job metadata records hash/MIME/size, and neither
  sandbox output nor public artifacts contain source references. Astra receives
  the same image data URLs on initial and geometry-repair turns. Their private
  bytes are deleted when authoring succeeds or fails; request metadata retains
  only SHA-256/MIME/size. Start/result notifications state the external author and
  reference count and must not imply use when it is zero.
  `src/capabilities/cad.rs` launches the trusted
  `tools/cad_prepare.py` runner; generated code runs under bubblewrap with no
  network/home/credentials, read-only CAD environment and isolated job writes.
  Mesh checks, official A1 slicing and allowlisted public previews follow. Results
  return to the bound chat; admin gets a print-approval request. No printer
  upload/start is implemented by this capability. Do not replace this boundary
  with guest shell/admin permissions. Python backend tests:
  `~/Projects/piercing-gauge/.venv/bin/python -m unittest discover -s tools -p test_cad_prepare.py`.
  Requires bwrap/prlimit, existing CAD venv and extracted official Bambu profiles.
  Profiles are stored durably at ~/.local/share/cad-pipeline/squashfs-root/resources/profiles/BBL;
  override ADAPSIS_CAD_PROFILES if needed. Sandbox files live in each job's
  sandbox/ subdirectory; trusted slicing/manifest/log outputs stay outside its
  writable bind. Symlink outputs/parents are rejected before host processing.
  Job folders ~/Projects/kronk-designs; public-only assets /srv/www/cat-bowl/kronk-designs.
- **The CAD sandbox is process-capped by a transient user systemd service (#50).**
  `tools/cad_prepare.py` `sandbox_command()` wraps the `prlimit`+`bwrap` guest in
  `systemd-run --user --wait --collect --pipe` with a unique
  `adapsis-cad-sandbox-<uuid>` unit, `TasksMax=64` and
  `KillMode=control-group`. Do NOT replace this with `prlimit --nproc`:
  `RLIMIT_NPROC` is counted per user, so it would starve the adapsis service's own
  forks. A service (not scope) gets a private cgroup, so the cap cannot touch the
  shared user slice. Only the bwrap guest is inside it — font-cache warmup and
  trusted slicing stay outside. `RuntimeMaxSec` is a hard backstop just above the
  300 s Python timeout; the Python timeout fires first and `_teardown_unit()`
  SIGKILLs the whole control group, so a `TimeoutExpired`/launch failure never
  leaves a forking guest running. A user-manager probe runs before paid Astra
  generation. Missing managers plus later systemd connection/start failures are
  `SandboxUnavailable` (fail-fast, not retried as geometry failures) —
  never silently run uncapped. `env -i PATH=...` inside the unit is load-bearing:
  systemd starts the service with the **user manager's** environment (API keys
  included), so the old PATH-only client env no longer protects `/proc/1/environ`;
  only the `env -i` layer does, and systemd-run's own status lines are stripped as
  runner noise so a signal death still reads `terminated by signal N`.
- **Assembled review data contract (#51).** A multipart job may also emit
  `/job/output/assembly.json` as pure JSON data so the review page can show the
  assembled object, not only separated print parts. Trusted `validate_assembly`
  (`tools/cad_prepare.py`) strict-validates it before it can reach a report:
  whitelisted top-level fields (`units`=`"mm"`, `envelope_mm`, `instances`), each
  instance exactly `{part, transform}`, `part` must exactly name an exported STL,
  1–`ASSEMBLY_MAX_INSTANCES` (64) instances, `envelope_mm` three positive finite
  dims ≤ `ASSEMBLY_MAX_DIM_MM` (1200), and each `transform` a finite row-major 4x4
  affine matrix whose 3x3 part is orthonormal with |det|=1 (rotation/reflection
  only — no scale/shear). The assembled bounding box computed from the real
  meshes must fit the declared envelope within `ASSEMBLY_TOLERANCE_MM`. Rejected
  values are never echoed back; only numbers and already-checked filenames are
  stored in the report/`manifest.json` — never guest SVG/HTML/JS. Absent or invalid
  data is **not** a geometry failure and never triggers an Astra revision: it
  degrades to print-parts-only with a visible reason. `tools/cad_preview.html`
  offers Assembled and Print parts views built from the same hash-checked STLs
  offered for download; `publish()` still copies only allowlisted part files,
  optional STEP, the 3MF, `manifest.json` and the trusted `index.html`.
- **Native guest tools (creative profile):** `[principals]` maps server-observed
  speaker IDs to permission profiles, taking precedence over the intake module's
  generic non-admin profile. `[guest_profiles]` explicitly lists native tools;
  `config/guest-creative.toml` is a mergeable deployment example. Guest blocks
  are validated in full before execution and admit only `!done` or direct
  `!eval tool(literals)` calls. Module calls, nested calls, tests, mutations,
  shell/raw HTTP/file IO and agents are not guest tools. The coroutine IO boundary
  independently enforces the same list. Identity text never grants a capability.
- **Creative deployment gate:** before restarting a node that grants GuestUser the
  creative profile, verify both deployment inputs match the reviewed tree:
  `python3 tools/check_guest_creative_config.py config/guest-creative.toml
  ~/.config/adapsis/permissions.toml` and
  `cmp contexts/telegram_user_100000101.md ~/.config/adapsis/contexts/telegram_user_100000101.md`.
  The TOML checker requires the reviewed fragment's `principals`,
  `guest_profiles` and `model.creative` entries to match while deliberately
  preserving unrelated live tables and keys. A mismatch blocks deployment: the
  permission file decides which native tools execute, while the context file
  carries the consent and privacy instructions.
  After restart, confirm the service's `--permissions-file` points at that exact
  permissions file before sending a live test message.
- `src/coroutine/scoped.rs` implements turn-bound argument checks and requests;
  `src/capabilities{,/music,/web}.rs` handles host IO. `music_generate` always
  delivers to the bound turn context, accepts 10–120 seconds, and allows one
  native job at a time. Failure also produces a notification. The existing
  `MusicGen.generate(context,...)` module remains an admin tool for explicit
  cross-chat delivery, not a guest capability.
- `conversation_history(context[, limit, before_ms])` reads original messages
  with read-membership checks. It is restricted to the configured administrator's
  private Telegram DM, not group turns. Pages are 1–100 messages (default 30),
  at most 30k content characters, chronological within each page; `before_ms` is
  exclusive (equal-timestamp rows may straddle pages). Assistant rows establish
  generation only, not successful delivery. Guests cannot invoke this tool.
- Guest `memory_remember` accepts context scope only. Guest `memory_correct` is
  turn-bound and can change only an authorized single-context record; its scoped IO
  request independently preserves the bound principal/context and non-write authority.
  Guest `memory_forget` additionally requires an assertion by that principal governed
  solely by that context, so global notes and other conversations cannot be changed.
- Public web tools use Bing RSS (`roxmltree`) and static HTML extraction
  (`scraper`), returning source URLs. Fetching verifies public IPs, pins DNS for
  the request, rechecks each redirect, disables ambient proxies, and caps response
  size/time. No authenticated pages, cookies or JavaScript execution. Test with
  `cargo test --release capabilities` and `cargo test --release scoped`.
- Immediate inbox awareness: `memory_graph/activity.rs` reads latest incoming-message
  timestamps per speaker/context directly from Message nodes, before compaction.
  `llm_takeover` injects a fresh metadata-only snapshot (30 latest pairs) exclusively
  into the configured `--admin-id` principal's private Telegram DM. Both speaker
  and destination must match; admin turns in groups never receive this snapshot.
  Read membership is checked, message contents are omitted, and the snapshot is
  transient rather than appended to persisted conversation history. Missing rows
  do not prove absence of older messages; receipt does not prove reply delivery.
- **Photo input**: `process_update` accepts Telegram's largest `message.photo`
  variant or an image MIME `message.document`, downloads it as an `Attachment`,
  and passes it as the optional sixth `llm_takeover` argument. Adapsis emits
  OpenAI `image_url` content using a base64 data URL; the gateway translates it
  for the selected provider.
  Regression fixed: bounded conversation views previously dropped captionless
  user messages AND cleared all attachments. `session/conversation_view.rs` now
  retains up to3 recent images/20MB independently of the text budget. If a
  restored conversation has no image parts, takeover reloads up to3 authorized
  references from the Ladybug attachment store (read ACL and exact context).
  Never infer "user supplied no photo" merely from empty text in session JSON.
  Telegram may serve JPEGs as application/octet-stream. Attachment::image_mime
  centralizes MIME/extension/magic-header recognition for budgets, recovery and
  outbound LLM image parts; MIME-only filters silently discard real photos.
  Tests: `cargo test --release conversation_view` and
  `cargo test --release saved_images_are_recovered_only_for_authorized_context`.
- **Video input and stabilization (#52):** `process_update` also accepts
  `message.video` and video-MIME documents, downloads the original as an
  Attachment, and passes it plus Telegram file id/name/MIME/duration/size
  metadata through `llm_takeover`. `video_stabilize()` is a zero-argument,
  turn-bound native guest tool: Rust resolves the latest user video through a
  Ladybug read-membership check for the exact principal/context, so the caller
  cannot name a host path, source conversation or destination. The trusted
  pipeline in `src/capabilities/video.rs` permits one job at a time; defaults to
  absolute `/usr/bin/ffprobe` and `/usr/bin/ffmpeg` (absolute overrides:
  `ADAPSIS_FFPROBE`/`ADAPSIS_FFMPEG`); and limits input to Telegram's 20 MB bot
  download cap/120 s, output to 49 MB below Telegram's 50 MB delivery cap, two
  encoder threads and 15 minutes. It runs libvidstab detect then
  transform without a shell, probes ffmpeg with fixed arguments to prefer
  `libopenh264` and falls back to built-in `mpeg4`, derives video bitrate from
  duration/output budget, retains audio as AAC when present, checks the
  output with ffprobe, cleans isolated failure/job files, and copies success to
  `~/.local/share/adapsis/video-output/` before asynchronous bound delivery.
  This is a conventional transform of original frames, never generative frame
  replacement. `ADAPSIS_VIDEO_OUTPUT_DIR` is a test/operator override; durable
  successful outputs remain until the platform callback completes. If attachment
  delivery fails, the capability makes one bound text-only fallback attempt,
  then removes the output regardless of either result; it never recursively
  retries. `conversation_notify` enqueueing alone is not delivery success: it
  observes the callback task's terminal registry state with a 135-second bound
  (above the 120-second upload timeout). Spawned-task terminal entries are
  load-bearing for that acknowledgement and must not be pruned without replacing
  the channel. A context with no callback still keeps the appended message and
  returns a truthful stored-but-not-delivered result for startup/Wolfi callers.
- **Multi-admin**: `admin_user_ids` shared var, comma-separated (e.g. `"100000001,456789"`)
- **Group chats**: messages in group/supergroup → `telegram:group:<chat_id>` context (shared by all senders)
- **Admin DMs**: `telegram:<user_id>` context with full LLM access
- **Non-admin DMs**: `telegram:user:<user_id>` context with sandboxed `permission_model` (default: `gemma4s`)
- **Non-admin in groups**: same group context but with restricted `permission_model`
- Context routing extracts `message.from.id` (sender) and `message.chat.type` (private/group/supergroup)

## Permission System
Layered access control: Process level → Model level → Context level. Each layer can only restrict.

### Process level (`--access-level` CLI flag)
- `full` — everything allowed including `!opencode`
- `adapsis-only` — can modify any module, no `!opencode` **(default)**
- `user-only` — can only modify non-core modules, no `!opencode`
- `execute-only` — cannot modify anything, can only `!eval`

**Default is `adapsis-only`** so `!opencode` (which rebuilds and re-execs the
runtime) is opt-in. The self-improving dev loop must pass `--access-level full`
explicitly (see `start.sh`). Deployments that should never rewrite their own
runtime (e.g. a family member's machine) should keep the default or go lower.

### Shell policy (`ADAPSIS_SHELL_POLICY` env var)
`shell_exec`/`exec` is gated independently of the permission system, enforced at
the IO loop (`coroutine/shell_policy.rs`):
- unset / `unrestricted` — any command (legacy default)
- `deny` — all shell execution refused
- `allow:git,ls,systemctl` — only listed programs (first token) may run

Unknown/empty specs fail safe to `deny`. For a locked-down box, set
`ADAPSIS_SHELL_POLICY=deny` or an explicit allowlist. This is the OS-capability
gate the per-module permissions do **not** provide.

**Destructive-command guard (applies under EVERY policy, incl. `unrestricted`).**
The design philosophy: a family/assistant bot needs *broad* shell reach to be
useful (read logs, try fixes, install drivers, restart services), so an
allowlist is the wrong tool — it cripples debugging. The real risk isn't
"running commands" but an *irreversible* mistake. So `check()` enforces an
absolute denylist **before** consulting the policy mode (`destructive_reason()`
in `shell_policy.rs`): `rm -rf` of a critical root (`/`, `/etc`, `/home`,
`/usr`, …), `dd` to a `/dev/*` block device, `mkfs*`/`wipefs`/`blkdiscard`/
`shred` on a device, redirecting onto a raw disk, and fork bombs. Refusal
message: `refused as destructive/irreversible: <reason>`. The guard is
deliberately narrow (device-wipers + root-tree deletes only) to keep false
positives near zero — `rm -rf /tmp/build`, `dd` to a regular file, `mkfs` on a
loopback image, etc. all still run. edox runs `unrestricted` **plus** this guard:
the bot can freely debug, but can't wipe FamilyUser's laptop from a misread message.

### Model level (`--permissions-file` → `permissions.toml`)
```toml
[groups]
core = ["TelegramBot", "MusicGen"]
data = ["Stratum", "Memory"]
infra = ["GithubSync", "IssueReader"]
# Modules not in any group belong to "user"

[model.gemma4s]
core = "execute"    # can call functions only
data = "execute"
user = "execute"
opencode = false

[model."chatgpt/gpt-5.4"]
core = "read"       # can call + view source
data = "write"      # can modify
user = "write"
opencode = false

[model."anthropic/claude-opus-4-6"]
core = "write"
opencode = true
```

Permission levels per group: `none` < `execute` < `read` < `write`
- `none` — module invisible
- `execute` — can `!eval` functions
- `read` — execute + `?source`, visible in program summary with docs
- `write` — read + can `+module` to modify

### Context level override
Each conversation can set `permission_model` to use a different model's permissions.
Only restricts — never expands beyond the active model.

### Program summary filtering
The system prompt only shows modules the model can at least Read. Execute-level modules show function signatures without docs.

## Model Management
- `llm_set_model(name)` — switch LLM at runtime (validates before switching)
- `llm_get_model()` — returns current model name
- `!agent --model gemma4-31b task` — per-agent model override
- Music generation runs in its own daemon and leaves the conversational model unchanged.

## Music Generation (ace-step-rs)
- HTTP endpoint: `POST http://127.0.0.1:8092/generate` → encoded audio bytes.
  **8091 is occupied by Chronica**; the old endpoint returned HTTP 405. Native
  tools honor `ADAPSIS_MUSIC_URL` (base URL; default `http://127.0.0.1:8092`).
- Backend build: `cargo build --release --bin generation-daemon --features audio-all,http`
  in `~/Projects/ace-step-rs`, with that project's CUDA build environment. Both
  MP3 and HTTP features are required by the deployed callers. The user service
  must point at the actual built executable and use `--http-port 8092`.
- On this 24GB node the int8 Parakeet ASR service may coexist with CPU-offloaded
  ACE-Step. Remove `ace-step-gen.service` from Parakeet's `Conflicts=` list;
  otherwise starting music silently stops voice transcription. Keep the heavy
  LLM/image-service conflicts. Verify `/health` on 8090 after enabling music.
- CPU offload: text encoder on CPU, DiT+VAE on GPU (13GB → 10GB VRAM)
- Non-blocking: `MusicGen.generate()` spawns background task, returns immediately
- Delivery: `conversation_notify` with Attachment → `send_reply_with_attachment`
- Generation uses the dedicated daemon; the deployed MusicGen module and native
  music tool do not switch the conversational LLM model.

## Infrastructure
- Caddy HTTPS on port 443 (Let's Encrypt), only `/webhook/telegram` exposed
- Systemd services: `adapsis.service`, `llama-server.service`, `ace-step-gen.service`, `parakeet-server.service`, `caddy.service`
- llama-server: TurboQuant build with turbo3 KV cache, Gemma 4 fixes
- Save-on-change (debounced), not periodic autosave. **Conversation turns count
  as changes** (2026-07-06): `handle_llm_takeover` fires `save_notify` after the
  user message and at loop end, so Telegram chats survive restarts — previously
  only code mutations saved, and the bot "resumed" from a stale snapshot (the
  "will shutdown now!" greeting bug on edox). `Session::save` is atomic
  (temp file + rename). `llm_takeover` also writes `--log-file` entries now
  (user/ai-text/llm-error), and an LLM failure returns a short apology to the
  reply callback instead of silently dropping the message.
- Panic hook + exit logging for crash debugging
- **Metadata-only logging (#88).** `src/logging.rs` gates model output, prompts,
  tool arguments, eval results, extracted memories, self-trigger previews and
  user/ai-text journal entries behind `ADAPSIS_LOG_CONTENT=1`. Default stdout/
  stderr/`--log-file` carry only context, model, latency, token usage, end state
  and error class — never reasoning, completion text or user data (which escape
  Ladybug ACLs). Set `ADAPSIS_LOG_CONTENT=1` to restore full content logging for
  debugging. Existing journal entries persist until rotation
  (`journalctl --user --vacuum-time=`, operator decision, not automatic).
  Error text is content too (serde/regex/provider errors can echo inputs).
  `logging::diagnostic` and `logging::event` render categories and byte counts
  at the sink; event tags never authorize their detail strings. Caller/UI errors
  retain details. Tests use child processes with immutable logging policy, never
  shared flag overrides across async waits.

## Voice Transcription (ASR)
> **Voice messages required FOUR stacked bug fixes (2026-06-21).** Each masked
> the next, so symptoms looked like "bot ignores voice". Validate ASR by
> actually sending a real voice note (MCP `send_voice` to the bot's chat) and
> watching for `llm_takeover ... done` + a delivered reply — NOT by sending a
> `.wav` document and calling `transcribe_voice` directly (that skips the real
> `message.voice` poll path and gives false confidence).
>
> 1. **`http_get` returns a `Result`, not a bare String.** `transcribe_voice`
>    did `+await file_info:String = http_get(url)` then `json_get(file_info,…)`.
>    The `:String` binding stored the `Ok("…")` wrapper → `json_get` got a
>    non-String arg0 → `json_get expects (String, String)`. Text replies were
>    unaffected (their `http_post` result is ignored, never fed to `json_get`).
>    Fix: unwrap with `+match http_get(url)` / `+case Ok` / `+case Err`. **Rule:
>    never bind a fallible IO builtin straight to a typed var then feed it to
>    another builtin — `+match` it first.** (module fix, disk-persisted)
> 2. **Module re-saved BROKEN by the rewriter.** `+test` `+with` inputs with
>    literal newlines (multi-line string values) were serialized back to disk as
>    raw newlines (`library.rs`/`session.rs`), producing a module that **failed
>    to parse on next start** → `+startup`/`poll_loop` never ran → bot silently
>    stopped polling. Fix: `ast::escape_test_input_linebreaks` escapes
>    `\n`/`\r`/`\t` in serialized test inputs. (**runtime fix — needs rebuild**)
> 3. **Tree-walker scope bug — the transcript was lost.** `+await text:String =
>    transcribe_voice(…)` lived inside a nested `+if`. The `Await` handler used
>    `env.set` (innermost scope) instead of `env.set_existing`, so it **shadowed**
>    the outer `text` and the value vanished when the `+if` block ended →
>    `handle_admin_dm` got an empty message → empty LLM call. Fix: `Await` uses
>    `env.set_existing` (walk scopes, reassign the existing binding). Spawned
>    tasks always use the tree-walker, so this only bit the real voice path.
>    (**runtime fix — needs rebuild**; regression test
>    `test_await_reassign_in_nested_if_escapes_block`)
> 4. **llm-gateway: `cache_control` on empty text blocks.** Even with a valid
>    transcript, an empty message anywhere in the last 3 turns made the gateway
>    attach `cache_control` to an empty block → Anthropic 400
>    `cache_control cannot be set for empty text blocks`. Fix in
>    `~/Projects/llm-gateway` (`apply_cache_breakpoints`): skip the breakpoint
>    when the block text is blank. (**gateway rebuild + `systemctl --user restart
>    llm-gateway.service`**)
>
> Deploy note: rebuilds (#2,#3) ship in the adapsis binary. The module library now
> detects edits made while a process is running and refuses to clobber them; use
> `library_reload("Name")`, reconcile, and retry. Binary is glibc-2.39-linked;
> edox (Mint 22.3, glibc 2.39) runs it.

`TelegramBot.transcribe_voice` downloads a Telegram voice note and POSTs it to
`http://127.0.0.1:8090/transcribe` (multipart field `file`) expecting JSON
`{"text": ...}`. The ASR backend is **server-agnostic** — adapsis just needs
something on 8090 honoring that contract. (Was whisper; now Parakeet.)

- **Backend: `parakeet-server.service`** runs `tools/parakeet_server.py` — a
  FastAPI wrapper around NVIDIA `nemo-parakeet-tdt-0.6b-v3` via the `onnx-asr`
  package (int8 ONNX, ~0.7 GB). Auto-detects 25 EU languages (incl. de/en),
  emits punctuation + capitalization. Started via `uv run --with onnx-asr[...]`
  (no venv to manage). Telegram sends OGG/Opus → ffmpeg transcodes to 16 kHz
  mono WAV before inference (onnx-asr reads WAV only).
- **Endpoints:** `POST /transcribe` (adapsis contract) and
  `POST /v1/audio/transcriptions` (OpenAI-compatible alias), plus `GET /health`.
- **Per-node variant (the only difference is GPU vs CPU):**
  - **here/Kronk (RTX 3090):** `onnx-asr[gpu,hub]`, `CUDA_VISIBLE_DEVICES=0`,
    user service. Has `Conflicts=` GPU mutex (llama-server,
    comfyui, lucebox-dflash) AND `Conflicts=whisper-server.service` (both bind
    8090). whisper-server is now disabled here.
  - **edox/Moonwolf (no usable GPU — GTX 970M Maxwell left on CPU):**
    `onnx-asr[cpu,hub]`, no CUDA env, no GPU mutex. **System** service
    (`User=adapsis`, `/etc/systemd/system/`). 8 CPU cores handle short voice
    notes well above real-time. Driver deliberately NOT installed (Optimus risk
    on a family laptop, Maxwell too old for modern CUDA onnxruntime).
- **Long audio / VAD:** Parakeet caps a single forward pass at ~20–30 s.
  `parakeet_server.py` loads **Silero VAD** (`onnx-asr.load_vad("silero")` →
  `model.with_vad(vad)`) and routes by clip length: clips ≤
  `PARAKEET_VAD_THRESHOLD_SECS` (default 20 s, from the WAV header) take the
  cheap single pass; longer clips are split into speech segments and the segment
  texts joined. `GET /health` reports `vad: true/false`. Env knobs:
  `PARAKEET_VAD=0` to disable, `PARAKEET_VAD_MODEL`, `PARAKEET_VAD_THRESHOLD_SECS`.
  VAD load failures degrade gracefully (ASR still works, long audio truncates).
- **Swapping backends = zero adapsis change.** Keep something on 8090 with the
  `/transcribe` + `{"text":...}` contract. This is infra, not a language change
  — do NOT use `!opencode` for it.

## Multi-Host Bind (`--host`)
`adapsis os --host` accepts **multiple** bind addresses so one instance can listen
on several interfaces at once:
- Repeat the flag: `--host 127.0.0.1 --host 10.0.0.4`
- Or comma-separate: `--host 127.0.0.1,10.0.0.4` (`value_delimiter = ','`)
- Default stays `127.0.0.1` (loopback only — the code-executing API is not exposed
  unless you opt in).

Implementation (`src/main.rs`, `Command::Os`): addresses are trimmed/de-duped
(order preserved), one `TcpListener` is bound per host (binding fails fast on a
typo or busy port), and the same axum `Router` is served on all of them
concurrently via `futures::future::try_join_all` (the Router is cloned per
listener). Each interface prints its own API/Browser URL on startup.

**Security note:** binding a `10.0.0.x` / `0.0.0.0` interface exposes
`/api/eval`, `/api/mutate`, `/api/opencode` etc. to anyone who can reach that
address. Only do it on a trusted network (the WireGuard mesh below). Prefer
`127.0.0.1 + <own VPN IP>` over `0.0.0.0` so you don't also expose the LAN
(`192.168.1.x`) interface.

## AdapsisOS VPN Mesh (10.0.0.0/24)
Three AdapsisOS nodes share a WireGuard VPN. Each binds `127.0.0.1` **and** its
own `10.0.0.x` on port **3002**, and each has a `mesh.md` (loaded at runtime by
`prompt::mesh_topology()`, no rebuild needed) that names itself + its peers and
lists the exposed endpoints so the bots can talk to each other via
`http_get`/`http_post`/`http_request`. Who each node *is* now comes from its
per-context files (`~/.config/adapsis/contexts/`), not from a `persona.md`.

| VPN IP   | Host  | Persona  | Runs as / service                          | Binary path                |
|----------|-------|----------|--------------------------------------------|----------------------------|
| 10.0.0.1 | here  | Kronk    | marenz, **user** `adapsis.service`         | `~/.local/bin/adapsis`     |
| 10.0.0.2 | sleek | Hobbes   | marenz, **user** `adapsis.service`         | `~/.local/bin/adapsis`     |
| 10.0.0.4 | edox  | Moonwolf | `adapsis` user, **system** `adapsis-bot.service` (model `family-bot`, FamilyUser's machine) | `/home/adapsis/bin/adapsis` |

- **Identity vs capability module:** identity/tone lives in the per-context file;
  the *capabilities* live in modules. On edox the family-bot's capability module
  is still `Wolfi.ax` (don't rename — `Wolfi.remember(...)` is a real function),
  but the bot's **identity** is "Moonwolf". `~/.config/adapsis/persona.md` is no
  longer read at all; migrate its content to a context file (start from
  `contexts/templates/family-assistant.md` for a family node).
- **Mesh topology = `mesh.md`, NOT an identity file.** The VPN/peer table +
  exposed endpoints live in `~/.config/adapsis/mesh.md` (override path via
  `ADAPSIS_MESH_FILE`), loaded by `prompt::mesh_topology()` at runtime (no
  rebuild to edit content). Keep them separate: mesh = facts (shared across
  conversations), context file = voice (one conversation).
- **Prompt builders — there were THREE and they drifted; there is now ONE
  composer.** `handle_llm_takeover` (Telegram & agent), `ask` (`POST /api/ask`)
  and `ask_stream` (`POST /api/ask-stream`) each used to assemble their own
  system prompt, so a fragment added to one did not appear in the others — that
  is how `mesh_topology()` once reached only the non-admin takeover branch and
  **admin DMs and `/api/ask` claimed to know nothing about peers**. All three now
  call `context_prompt::compose` (`ask`/`ask_stream` through the shared
  `main_context_prompt` helper). Add fragments there, not in a handler. Each
  builder still refreshes the first system message on every request while
  preserving conversation history, so prompt, mesh, identity-file and
  visible-capability changes reach existing persisted conversations on their next
  turn.
- **Topology = hub-and-spoke, NOT full mesh.** `here` (10.0.0.1) is the always-on
  WireGuard **hub** (`wg0`, one `/32` peer per spoke). Spokes (sleek, edox) only
  peer with the hub; their `AllowedIPs` toward the hub is `10.0.0.0/24` so all
  intra-subnet traffic is crypto-routed to the hub. **Spoke↔spoke works only by
  the hub relaying it** (and only while both spokes are online — only the hub is
  always available).
- **Hub relay persistence (survives reboot):**
  - `/etc/sysctl.d/99-wg-hub-forward.conf` → `net.ipv4.ip_forward = 1`
    (the `99-` prefix beats `70-yast.conf` which sets it to `0`).
  - firewalld: hub `wg0` is in the `trusted` zone (allows forwarding); on sleek
    `wg1` is also in the `trusted` zone so its `:3002` is reachable on the VPN.
  - edox's WireGuard is managed by **netplan→NetworkManager** (NM keyfile is
    ephemeral in `/run`; edit `/etc/netplan/90-NM-*.yaml`, not the `/run` file).
    sleek/here use `wg-quick@.service` with `/etc/wireguard/*.conf`.
- **Deploy = no auto-restart by default.** Replacing the binary (`install -m755`,
  `-f` to overwrite; running process keeps its old inode) + editing the unit's
  `ExecStart` + `daemon-reload` does NOT pick up changes until the service is
  restarted. Restart commands: `systemctl --user restart adapsis.service` (here/
  sleek), `sudo systemctl restart adapsis-bot.service` (edox).
- **sudo:** edox runs the bot as the `adapsis` system user; for ops there use
  `sudo` as `marenz` (in the `sudo` group). On `here`, use `sudo -A` (GUI askpass)
  per the global AGENTS.md.

## FamilyUser Desktop Helper (edox only, 2026-07-06)
The `adapsis` service user cannot touch FamilyUser's desktop session (correct OS
behavior), so edox runs **`family_user-agent`** — a stdlib-only Python HTTP daemon
as user `family_user` on `127.0.0.1:3010` (systemd **user** unit
`~family_user/.config/systemd/user/family_user-agent.service`, script at
`~family_user/.local/bin/family_user-agent.py`, linger enabled). Endpoints: `/notify`
(notify-send), `/play` (paplay, named sounds only), `/open-url` (http/https
only), `/dialog` (zenity yes/no, 2-min timeout), `GET /health`. The adapsis
side is the **`FamilyUserAgent.ax`** module (in the `assist` permission group, so
FamilyUser's sandboxed conversations may call it): `notify`, `play_sound`,
`open_url`, `ask_yes_no` — all ≤2-statement io fns (exempt from the test gate)
delegating JSON building to tested pure helpers. Use `ask_yes_no` for consent
before invasive actions.
