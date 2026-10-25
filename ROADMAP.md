# AdapsisOS Roadmap

## Current execution — native tools first (#78)

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
  counterpart. Only intended new files are intent-to-add for full diff visibility;
  no staged content or commits. Disk remains 32 GiB free on `/home`, 9.5 GiB on
  `/tmp` after sequential gates.
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
- [ ] Measured Python-subset frontend (later)

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
