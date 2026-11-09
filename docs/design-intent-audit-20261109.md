# Adapsis Design-Intent Audit — 2026-11-09

Read-only audit of original design intents vs. what the four current
conversation/authoring surfaces actually support. Goal frame: "a bot that can
replace OpenCode for programming / agent / task work."

Sources of original intent, oldest/most authoritative first:

- `adapsis-design-doc.md` (language design v0.1, ~Mar 2026) — the full vision:
  mutation protocol, effect system, provenance tracker, grammar constraint,
  FFI, structured concurrency, evaluation/simulation/test modes, browser,
  self-hosting.
- Early `src/prompt.rs` (e.g. `272a320`, `!eval` commit) — the spec the model
  was actually taught (8 effects, `!replace`/`!trace`/`?symbols`/`?callers`/
  `?effects`/`?type`).
- `README.md` — `!eval <expr>`, `!trace`, top-level statements (`x = 10+5`),
  semantic queries, `!replace`.
- `AGENTS.md` "Language Design Goals", "Key Adapsis Commands", "Autonomous
  Mode", "!opencode Rules", "Conversation System".
- `git log --reverse` + `git log -p --follow src/prompt.rs` (693 commits).
- `refs/archive/2026-10-25/*` (read-only `git show`).
- GitHub issues `gh issue list --state all` (127 issues) — esp. #22, #26, #33,
  #76, #84, #86, #89, #120, #121.

Current surfaces compared:

- **legacy** — `<code>` text protocol (`src/api/execute.rs`,
  `src/api/llm_handlers.rs`, `src/parser.rs`, `src/validator.rs`).
- **native** — JSON-schema tools `inspect|mutate|test|evaluate|finish`
  (`src/api/native_tools*.rs`, `src/api/native_turn.rs`, `src/prompt/native.rs`).
- **Python** — typed Python-subset frontend (`src/python_frontend/`,
  `examples/python/README.md`).
- **guest** — turn-bound allowlisted tools (`src/coroutine/scoped.rs`).

Status key: **KEPT** / **NARROWED** / **LOST** / **NEVER-BUILT**.

---

## Master table

### 1. Ad-hoc evaluation & top-level statements

- **Intent:** `!` commands, `!eval` of arbitrary expressions, and top-level
  statements execute immediately — Adapsis is a *surface*, not just a
  define-then-call function store.
- **Stated at:** design doc §12.3 ("Expression-Level Evaluation",
  `!eval input.age>=0 … +with …`); README `!eval fizzbuzz 15`; README
  self-hosting (`x = 10+5` top-level assignment); issue #22 ("`!eval` should
  accept inline expressions, not just function names"); `validator.rs:187`
  ("top-level statement (execute immediately)").

| surface | status | evidence |
|---|---|---|
| legacy | **KEPT** | `eval_inline_expr` / `eval_inline_expr_with_io` (`src/api/execute.rs:479-516`); top-level `Let/Set/Call/Check/If/Each/While/Match/Await/Spawn` (`validator.rs:167-192`) |
| native | **NARROWED** | `evaluate` tool = qualified function name + literal args only (`native.rs:63-64`, `native_tools.rs:195-202`); no expressions, no top-level statements; "native inputs … must be literals" (`docs/native-compare-2026-10-25.md:78`) |
| Python | **LOST** | no ad-hoc form; Python is `def`-based module source only (`native.rs:40-66`, `python_frontend/lower.rs`); module-less surface is `#[cfg(test)]`-only (`python_frontend/mod.rs:88-94`) |
| guest | **NARROWED** | guest blocks admit only `!done` or direct `!eval tool(literals)` for a fixed allowlist (`AGENTS.md` "Native guest tools", `scoped.rs:8-11`) |

- **Deliberate?** **SILENT.** The native/Python surfaces were built as
  "inspect → mutate → test → evaluate(fn) → finish" with no ad-hoc equivalent
  and no ticket records "drop ad-hoc eval". This is the owner's already-known
  loss.

---

### 2. `!agent` sub-agents & background work

- **Intent:** spawn background agents with scopes (`read-only`, `new-only`,
  `module X`, `full`) and completion callbacks; parallel work without blocking.
- **Stated at:** `builtins.rs:1497-1501`; "Conversation System" (`!agent` breaks
  loop for background work); issue #39 (sub-agent execution model).

| legacy | native | Python | guest |
|---|---|---|---|
| **KEPT** | **LOST** | **LOST** | **LOST** |

- Evidence: native `native.rs:183-184` "background agents (`!agent`) …
  unavailable"; "No agent … tool exists" (`native.rs:160`). **Deliberate &
  deferred** — sequenced in #89 item 2, #120.

---

### 3. `!plan` / `!roadmap` & autonomous mode

- **Intent:** persistent task plan + long-term roadmap; autonomous loop
  continues after `!done`; `/api/inject` queues messages.
- **Stated at:** `builtins.rs:1478-1490`; "Autonomous Mode" (AGENTS.md).

| legacy | native | Python | guest |
|---|---|---|---|
| **KEPT** | **LOST** | **LOST** | **LOST** |

- Evidence: `native.rs:160` "No agent, plan or roadmap tool exists";
  `native.rs:183-184` "plans (`!plan`), roadmap (`!roadmap`) … unavailable".
  **Deliberate & deferred** — #89 item 2, #120.

---

### 4. `!opencode` self-modification

- **Intent:** request Rust-level runtime change, rebuild + exec-restart; the
  "self-improving dev loop" (`--access-level full`).
- **Stated at:** `builtins.rs:1536-1540`; "!opencode Rules" (AGENTS.md);
  commit `b3fb593`/`c962f73` (full self-extension loop verified).

| legacy | native | Python | guest |
|---|---|---|---|
| **KEPT** | **LOST** | **LOST** | **LOST** |

- Evidence: `native.rs:184` "`!opencode` runtime changes … unavailable".
  **Deliberate & deferred** — #89 item 2 ("opencode only when process+model
  allow").

---

### 5. `!mock` / `!stub` testing

- **Intent:** mock IO (`!mock`) and stub user functions (`!stub`) for effectful
  tests; `!unmock`/`!unstub`.
- **Stated at:** `builtins.rs:1504-1528`; design doc §12.5 ("scripted IO");
  commit `f3d2464` (mock/stub pattern matching).

| legacy | native | Python | guest |
|---|---|---|---|
| **KEPT** | **LOST** | **LOST** | **LOST** |

- Evidence: `native.rs:184-185` "IO mocks/stubs … unavailable"; Python
  "cannot install mocks" (`native.rs:118`), only "trusted runtime-provided IO
  mocks". **Deliberate** (documented in native prompt), but leaves IO-function
  testing unreachable in the new surfaces.

---

### 6. `!watch`, `!trace`, `?` queries

- **Intent:** background observation loop (`!watch`), step-by-step tracing
  (`!trace`), semantic queries (`?symbols/?source/?callers/?callees/?deps/
  ?deps-all/?deps-modules/?effects/?type/?agents/?plan/?inbox/?tasks/?library`).
- **Stated at:** early `prompt.rs` (`?symbols/?callers/?effects/?type`);
  `builtins.rs:1310-1389` (QUERIES), `builtins.rs:1426-1432` (`!trace`),
  `1492-1496` (`!watch`).

| legacy | native | Python | guest |
|---|---|---|---|
| **KEPT** | **NARROWED** | **LOST** | **LOST** |

- Evidence: native `inspect` covers symbols/source/deps/tasks/library only
  (`native_tools.rs:159-166`, `native.rs:158`); `!trace`/`!watch` absent;
  Python has no query surface. **Partly deliberate** (inspect whitelist),
  partly **silent** (trace/watch just absent; watchpoints re-filed as #127).

---

### 7. `!msg` / inbox

- **Intent:** agent-to-agent and agent↔main messaging via `!msg` / `?inbox`.
- **Stated at:** `builtins.rs:1530-1534`, `?inbox` (`builtins.rs:1377-1382`);
  commit `131cbe5` (agent-to-agent messaging).

| legacy | native | Python | guest |
|---|---|---|---|
| **KEPT** | **LOST** | **LOST** | **LOST** |

- Evidence: `native.rs:184` "every other legacy `<code>` command" unavailable.
  **Deliberate & deferred** (#89).

---

### 8. `!sandbox`, `!undo` / rewind

- **Intent:** isolated experiment sandbox (`!sandbox enter/merge/discard`) and
  mutation rollback (`!undo`), plus session rewind-to-revision.
- **Stated at:** `builtins.rs:1462-1477`; README "rewind to any revision";
  `session.rs` mutation log.

| legacy | native | Python | guest |
|---|---|---|---|
| **KEPT** | **LOST** | **LOST** | **LOST** |

- Evidence: `native.rs:185` "removal commands and every other legacy `<code>`
  command" unavailable. **Deliberate & deferred**; rewind correctness itself is
  buggy and still open (#122 "history stores requested source, not accepted
  state", #123 "rewind needs startup baseline", #126 "module removal
  resurrected").

---

### 9. `+source` / `+event` / `+route` / `+startup`(+`+shutdown`) declarations

- **Intent:** service system — module lifecycle (`+startup`/`+shutdown`), source
  handlers (`+source`), events, HTTP routes (`+route`).
- **Stated at:** issue #33; commits `26a6f24`→`88654a0` (service system);
  `+route`/`+startup` in AGENTS.md "Key Adapsis Commands".

| legacy | native | Python | guest |
|---|---|---|---|
| **KEPT** | **LOST** | **LOST** (pending) | **LOST** |

- Evidence: legacy `SourceAdd/Remove/Replace/List`, `EventRegister/Emit`,
  `Startup/Shutdown` operations (`validator.rs:167-205`); native
  `native.rs:104-106` "Routes, source/event declarations … and
  startup/shutdown lifecycle blocks are NOT available through native mutation";
  Python frontend has partial `+source`/`+event`/route/lifecycle forms but the
  strict gate still marks them unsupported (issue #86 acceptance item 2, #33
  still OPEN). **Deliberate & tracked** (#86/#33), but blocks a service-building
  bot from authoring its own runtime surface.

---

### 10. `!move`, `!replace`, `!remove`

- **Intent:** refactor/move with auto call-site updates; targeted statement
  replace; remove function/type/module (auto-drops routes).
- **Stated at:** `builtins.rs:1433-1461`; design doc §4.6 "multi-scale
  mutations"; commit `1737c1e` (`!move`), `!replace` (`2f4b538` era).

| legacy | native | Python | guest |
|---|---|---|---|
| **KEPT** | **NARROWED** | **NARROWED** | **LOST** |

- Evidence: native `mutate` replaces functions by redeclaration
  (`native.rs:44-45`) — no `!remove`/`!move`/statement-replace; `native.rs:185`
  "removal commands … unavailable". **Deliberate** (declaration-based native
  mutation), but loses fine-grained statement-level editing.

---

### 11. Model switching

- **Intent:** `llm_set_model` / `llm_get_model` at runtime (validated).
- **Stated at:** `builtins.rs`; `prompt.rs:996-1009`; commit `14b5bfb`.

| legacy | native | Python | guest |
|---|---|---|---|
| **KEPT** | **LOST** | **LOST** | **LOST** |

- Evidence: `native.rs:184` "model switching … unavailable". **Deliberate &
  deferred** (#89 item 2).

---

### 12. Test enforcement & auto-retest

- **Intent:** functions >2 statements blocked from `!eval` until tested; tests
  persist and auto-rerun when functions change; `!done` rejected with untested
  functions; matchers `contains/starts_with/Ok/Err`; `+after` side-effect
  assertions.
- **Stated at:** AGENTS.md "Test enforcement"; `builtins.rs:1406-1413`;
  commits `4f71490`/`cdf87c5` (enforcement).

| legacy | native | Python | guest |
|---|---|---|---|
| **KEPT** | **KEPT** | **KEPT** | **KEPT** |

- Evidence: native `finish` checks tests for changed functions (#66 baseline,
  `native.rs:10-12`); Python test gate (`native.rs:65-66`). Auto-retest-on-change
  (`invalidate_and_retest`) is legacy-only (**NARROWED** elsewhere) but the
  gate is preserved. This is the best-preserved intent.

---

### 13. Training data logging

- **Intent:** JSONL per-iteration record (model, context, thinking, code,
  outcome, tests passed/failed); accumulates in `~/.config/adapsis/training/`.
- **Stated at:** AGENTS.md "Training Data".

| legacy | native | Python | guest |
|---|---|---|---|
| **KEPT** | **KEPT** | **KEPT** | **KEPT** |

- Evidence: `log_training_data` (`llm_handlers.rs:506`) also called from
  `native_turn.rs:506` and `takeover_native.rs:69-80`. Fine-tuning pipeline
  (#13 LoRA, #12 DPO) remains **NEVER-BUILT**.

---

### 14. The effect system

- **Intent:** 8 effects — `io, mut, fail, async, rand, yield, parallel, unsafe`;
  every function declares them; effect-specific test modes
  (`io:scripted`, `io:virtual_fs`, `rand:seed(N)`, `time:fixed`, `yield:collect`,
  `parallel:sequential`).
- **Stated at:** design doc §4.1/§12.6; early `prompt.rs` (8 effects listed).

| legacy | native | Python | guest |
|---|---|---|---|
| **NARROWED** | **NARROWED** | **NARROWED** | **LOST** |

- Evidence: `Effect` enum still has all 8 variants (`ast.rs:1006-1014`) but the
  taught/documented set is only `[io],[async],[fail],[mut]` (AGENTS.md
  "Effect system", `prompt.rs`); `rand/parallel/unsafe` parse but have no
  statements behind them; `+parallel` structured parallelism, `+extern` FFI,
  `[unsafe]` blocks, and ALL six effect-specific test modes are **NEVER-BUILT**
  (only `!mock`/`!stub` exist). Python `@effects` admits the same four + `rand/
  yield/parallel/unsafe` (`native.rs:110-112`) but `parallel`/`unsafe` lower to
  nothing usable. **SILENT** (no ticket records the 8→4 doc narrowing or the
  dropped test modes).

---

### 15. Error-propagation conventions

- **Intent:** `[fail]` + `+call T` auto-propagation (Rust-`?` style); no
  `is_ok`/`unwrap` builtins — `+match` on `Result` directly.
- **Stated at:** AGENTS.md "Language Design Goals"; design doc §4.5; README.

| legacy | native | Python | guest |
|---|---|---|---|
| **KEPT** | **KEPT** | **KEPT** | **KEPT** |

- Evidence: recently *hardened* (#82/#97 Result contract, `eval/result_contract.rs`);
  Python admits Result user awaits and exact-known-inner returns (`native.rs:112-117`).
  Best-preserved core semantic; the only intent actively strengthened.

---

### 16. Multi-session API

- **Intent:** multiple named sessions; per-session eval/mutate; session
  persistence of program AST + tests + roadmap + plan + mocks + chat.
- **Stated at:** AGENTS.md "HTTP API" (sessions endpoints); commit `0a8d426`
  (session management).

| legacy | native | Python | guest |
|---|---|---|---|
| **KEPT** | **NARROWED** | **NARROWED** | **LOST** |

- Evidence: `/api/sessions*` endpoints remain (AGENTS.md), but native turns run
  the main conversation loop with an immutable turn baseline; no native
  per-session tool routing is exposed. **Silent** narrowing (not tracked).

---

### 17. Web UI

- **Intent:** live multi-representation browser — mutation feed, pseudocode,
  flowchart, test dashboard, provenance view, thinking panel, human steering
  (design doc §13).
- **Stated at:** design doc §13; README "Browser UI".

| legacy | native | Python | guest |
|---|---|---|---|
| **NARROWED** | **NARROWED** | **LOST** | **LOST** |

- Evidence: shipped `web/adapsis.html` is a basic editor + program state +
  SSE feed (`web/adapsis.html`); last touched `8e10876` (2026-10-25). The rich
  flowchart/provenance/steering views were **NEVER-BUILT**; no native-turn
  rendering. **Silent** (aspirational §13 never delivered; no ticket).

---

### 18. Other design-doc intents (never built)

All verified absent from `src/` (grep for `!patch|!propagate|!simulate|?similar|
?provenance|+stmt|+parallel|+extern` returns nothing beyond memory-provenance):

- **Grammar constraint (GBNF)** — design doc §7, "Layer 3" — **NEVER-BUILT**
  (deliberate, documented as final layer).
- **Provenance tracker & `!propagate`/`?similar`/`!fix`** — design doc §6 — the
  flagship "abstraction via tooling" concept — **NEVER-BUILT**; the word
  "provenance" was silently re-purposed to mean Ladybug memory provenance
  (`memory_graph.rs`, `builtins.rs:698`). **SILENT.**
- **Symbolic simulation `!simulate`** — design doc §12.4 — **NEVER-BUILT.**
- **Structured parallelism / channels / `+spawn`-join / `[parallel]`** — design
  doc §11 — **NEVER-BUILT** (only fire-and-forget `+spawn`).
- **FFI / `+extern rust|C`** — design doc §10 — **NEVER-BUILT.**
- **Self-hosting of runtime/compiler** — design doc §14 — **NEVER-BUILT** (only
  a toy self-hosted REPL demo).
- **Architect mode** — README "architect mode" — present early
  (`orchestrator.rs`) but vestigial; not in any current surface — **SILENT.**
- **REPL as the surface** — README/`repl.rs` — legacy CLI only; no REPL in
  native/Python — **NARROWED (silent).**

---

## Top 10 silent losses vs. "replace OpenCode"

Ranked by how much each hurts a bot doing real programming/agent/task work
(only items that were *built and then silently narrowed*, or whose absence is
unticketed).

1. **Ad-hoc evaluation & top-level statements.** The single most OpenCode-like
   capability — run an arbitrary expression or statement right now — exists only
   in legacy `!eval`/top-level and is absent from native (fn+literals only) and
   Python (nothing). *Restore:* add an `evaluate` mode accepting an arbitrary
   expression/statement (not just a qualified call) to native + Python, reusing
   `eval_inline_expr`.

2. **`!agent` sub-agents / background work.** Replacing OpenCode means spawning
   scoped parallel agents with completion callbacks; LOST in every non-legacy
   surface. *Restore:* native `agent` tool + Python-authored task/run model
   (#120) with preserved callbacks.

3. **`!plan`/`!roadmap` + autonomous mode.** A task bot needs durable plan state
   and self-continuation; LOST outside legacy. *Restore:* native plan/roadmap
   tools (#89 item 2) before any default flip.

4. **`!opencode` self-modification.** The "self-improving" loop is the project's
   identity but is legacy-only. *Restore:* gated native `opencode` tool
   (process+model `write`).

5. **`!mock`/`!stub` + effect-specific test modes.** Effectful code (IO) cannot
   be tested from the Python/native surfaces at all ("Python cannot install
   mocks"), so the test gate only really protects pure code. *Restore:* expose
   mock/stub as native test fixtures or Python test decorators; implement
   `time:fixed`/`rand:seed` (cheap, high value).

6. **`!trace` / `!watch` debugging.** No step-through or value-watch in
   native/Python; a programming bot that can't debug is half a bot. *Restore:*
   native `trace` tool + watchpoints (#127).

7. **`+source`/`+event`/`+route`/`+startup` lifecycle.** A bot that builds bots
   (TelegramBot, HTTP servers) cannot author routes/lifecycle/sources in
   native or Python. *Restore:* admit lifecycle/route/source declarations in the
   Python frontend (#86) and native mutate.

8. **Effect-system narrowing (8 → 4 documented; test modes never built).** The
   effect model is the safety/legibility core; half of it silently dropped from
   docs and the sophisticated test modes never shipped. *Restore:* re-document
   `rand/yield/parallel/unsafe` or formally retire them; add the high-value test
   modes.

9. **`!undo`/`!sandbox`/rewind.** Safe experimentation and rollback are broken
   (#122/#123/#126) and absent from new surfaces; a self-driving bot needs a
   safe "try and revert". *Restore:* fix rewind to store accepted state; native
   `sandbox` tool.

10. **Provenance tracker / `?similar` / `!propagate`.** The flagship "abstraction
    via tooling" idea was never built and the word was silently re-used for
    memory provenance, so the original concept no longer even has a name.
    *Restore:* at minimum rename to avoid the collision and file the concept
    (e.g. a "duplicate/DRY detector" issue) rather than letting it vanish.

### Honorable mentions (deliberate/deferred, tracked — still gaps, not silent)

- `!msg`/inbox, `!move`/`!remove`, model switching, multi-session API,
  fine-grained statement editing, web-UI richness, architect mode, REPL-as-
  surface, training-data *pipeline* (#12/#13), grammar constraint (§7).

### Key cross-cutting observation

The narrowings fall into two buckets: (a) items explicitly deferred and ticketed
(#89, #86, #33, #120) — these are recoverable but must be delivered *before* any
native-default cutover (#89) or the legacy path is deleted; and (b) silent
narrowings with no ticket (ad-hoc eval, effect doc narrowing, provenance rename,
test-mode absence, trace/watch, rewind bugs) — these are the ones that will keep
surprising the owner until they are written down as explicit gaps.

---

*Audit method: read-only; evidence from `git log --reverse`, `git show` on
archived refs, `src/prompt.rs` history, `adapsis-design-doc.md`, `README.md`,
`AGENTS.md`, `ROADMAP.md`, and `gh issue` views. No files, services, or commits
changed. Report written to this file only.*
