# Adapsis — Adaptive, Self-Modifying AI Programming Environment

## Overview
Adapsis is an AI-native programming language and live agentic environment. Programs are built incrementally through validated mutations. The AI writes Adapsis code, gets immediate feedback, and iterates. The system can modify its own runtime via `!opencode`.

## Language Design Goals
- **Optimized for LLM-driven construction**, not human typing
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
# Adapsis tests: `adapsis test` exits non-zero on failure and prints "TEST SUMMARY: N passed, M failed"
for f in examples/*.ax; do ./target/release/adapsis test "$f" >/dev/null 2>&1 || echo "FAILED: $f"; done
```

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

## Architecture
```
src/
  main.rs        — CLI, autonomous loop, session management
  api/           — HTTP API: mod.rs (routes), execute.rs (execute_code pipeline),
                   llm_handlers.rs (ask/ask_stream, llm_takeover), tests.rs
  ast.rs         — Core AST (Program, Module, Function, Statement, Expr, Type, Effect)
  parser.rs      — Line-oriented parser (+, !, ? prefixes, +end blocks)
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
  builtins.rs    — Single source of truth for all builtins/commands/queries
  typeck.rs      — Type checker, symbol table, ?source reconstruction, semantic queries
  telegram.rs    — Telegram bot integration
web/
  adapsis.html   — Browser UI (SSE broadcast, collapsible thinking)
```

- Shared module vars live in `RuntimeState.shared_vars` under `Module.name` keys; eval/test paths must install both shared runtime and shared program snapshots so missing shared slots can be materialized from `+shared` defaults.
- Module library synchronization is first-writer-wins. `LibraryState` hashes every successfully loaded/written `.ax` file; persistence refuses to overwrite an external edit, reconstructed source must parse before rename, and a failed load keeps the last-known-good in-memory module. Reconcile conflicts with `library_reload("Name")` and retry rather than bypassing the check. **The one deliberate exception: a file this process never loaded *and* that does not parse is overwritten (loudly) by the in-memory module** — refusing there would make corruption permanent, since the reload we would tell the model to run cannot parse it either. Load errors are keyed per module and save errors are deduplicated + capped, because both feed strings the LLM sees on every turn.

## Concurrency / Three-Tier State (issue #9)
- **Tier 1 Program** = `Arc<RwLock<Program>>`; **Tier 2 RuntimeState** = `Arc<RwLock<RuntimeState>>` (`SharedRuntime`); **Tier 3 SessionMeta** = `Arc<Mutex<SessionMeta>>` (`SharedMeta`). No god-lock — a handler locks only what it touches, and **never** holds a lock across an LLM call, eval, test, or IO. `ask_stream` clones state → drops locks → calls the LLM → re-locks briefly per operation.
- **Per-function CoW write-back — DO NOT regress to `*program.write() = mutated`.** Mutations run on a clone; write-back uses `Program::merge_changed_from(base, new)` (`ast.rs`), which diffs the `base→new` function set via `Arc::ptr_eq` and applies only that delta onto the live program. This is why the validator must keep allocating a fresh `Arc<FunctionDecl>` *only* for the function it rewrites (untouched fns keep pointer identity). A whole-program overwrite silently drops a concurrent writer's change to a *different* function. All 8 write-back sites (mutate, WorkingSet, llm_takeover, agent-branch merge, self-trigger, inline-IO eval, async-fn eval, route handler) capture a `base_program` clone for the merge. The only intentional full reset is `rewind_to`.
- **`SharedVars` (`session.rs`) = per-key locks.** `shared_vars: HashMap<String, Arc<RwLock<Value>>>` — the outer lock guards map structure only; an existing var's value is read/written under its own inner lock (a structure *read* guard is enough to insert-if-absent or update an existing slot). `Clone` **deep-copies** (fresh inner locks) so RuntimeState forks stay isolated — never assume a cloned RuntimeState aliases the live shared vars. Write-back into the live runtime uses `SharedVars::replace_from(snapshot)` (reuses per-key locks so live readers see new values); **don't** assign the whole `SharedVars` (that swaps the Arc out from under live readers). `Value` is not `Serialize`, so the field stays `#[serde(skip)]`.

## Execution Tiers
Function evaluation goes through up to three tiers (fastest first):

1. **Cranelift JIT** (`compiler.rs`) — numeric/string functions only
   (`is_compilable_function` gate; bails on each/await/spawn/yield/List/Map/Union).
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
3. **Tree-walker** (`eval/mod.rs`) — universal fallback; the only tier with
   `Env`-based shared vars, function stubs, spawned-task snapshot tracking.

Tests (`+test`) always run the tree-walker with mock-only coroutine handles.
Spawned tasks (task_id set) always use the tree-walker (snapshot tracking).

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
  - `family-bot` (edox) → `deepseek-v4-flash` → `claude-sonnet-4-5` → `chatgpt/gpt-5.5`
    (DeepSeek made primary 2026-07-06 to spare OpenAI quota — temporary, revert
    when limits reset. Same reorder applied to the local gateway config on here.)
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
- **`deepinfra/zai-org/GLM-5.2` returns 402** on every request (no credit). It
  used to be `dev-bot`'s *first* target, so every call burned a round trip
  before falling through — now demoted to the tail. The `openai` provider also
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
- Tests persist in `session.stored_tests` (HashMap<fn_name, Vec<StoredTestCase>>)
- Auto-rerun when functions change via `invalidate_and_retest()`
- `!mock` intercepts `+await` calls during `!test` with fake IO responses
- Functions >2 statements blocked from `!eval` until tested
- `!done` rejected if untested functions exist
- Test expectations should be literal values, not function calls
- Pure function calls allowed in test inputs (e.g. `+with config=default_config()`)
- **Test matchers**: `contains("substr")`, `starts_with("prefix")`, bare `Ok`/`Err`, `Err("msg")`
- **+after assertions**: `+after routes contains "/chat"`, `+after modules contains "Name"` — check side effects after test execution

## Training Data
- JSONL training log at `--training-log` path (default: `training.jsonl`)
- Each iteration: model, context, thinking, code, outcome, tests passed/failed
- Accumulates in `~/.config/adapsis/training/`
- `tools/generate_training_data.py` — generates examples using Opus, validates through parser
- `tools/module_training_examples.py` — hand-crafted per-module examples
- `tools/merge_training_data.py` — deduplicates and merges all JSONL files
- `tools/finetune_gemma4.py` — QLoRA fine-tuning with unsloth

## Attachment System
- `Value::Attachment` — binary data type (audio, images, files)
- `AttachmentData::Memory` (≤10MB in RAM) / `AttachmentData::File` (>10MB on disk)
- `http_post_binary(url, body)` → returns `Attachment` from HTTP response
- `http_upload(url, attachment, field, extra)` — multipart upload from Attachment
- `conversation_notify(context, message, attachment)` — delivers to conversation with file

## Conversation System (llm_takeover)
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
- The model estimates each takeover's action-round budget in its first response with
  `<iteration_budget>N</iteration_budget>`. Missing estimates default to 10; estimates
  are clamped to the hard safety ceiling of 50. Protocol-correction rounds may extend
  an exhausted estimate, but never beyond that ceiling.

## Ladybug Episodic Memory

- LadybugDB is the sole authoritative long-term memory database. The default path is
  `~/.config/adapsis/memory.lbug`; override it with `ADAPSIS_MEMORY_DB`.
- Every incoming message is written before inference and linked to its `Context`,
  `Principal`, dynamic `AccessGroup`, original platform message ID, and timestamp.
  Telegram participants receive Read+Contribute membership when first observed;
  `telegram:user:1815217` also receives Manage. Membership grants the full context history.
- Memories link explicitly to every origin context/access group and source message.
  Cross-context recall requires membership in every governing group. `DENIED_TO` provides
  a future per-memory/per-user override.
- Attachments are SHA-256-addressed under `~/.local/share/adapsis/attachments/`; Ladybug
  stores the hash, MIME type, platform reference, storage path, and source-message edge.
- `fastembed` runs `intfloat/multilingual-e5-small` in-process for German/English recall.
  Its cache is `~/.local/share/adapsis/.fastembed_cache`.
- Automatic recall merges semantic vector ranking with lexical term matches and injects
  compact source citations only after ACL filtering.
- Compaction is non-destructive: raw messages remain forever. The active conversation
  model creates immutable `Episode` summaries and source-backed index `Memory` nodes.
  Failed compaction leaves raw history pending and uses prior episodes plus recent turns.
- Checkpoint generation times out after 30 seconds and retries at most once per context every
  15 minutes, so a slow provider cannot block every conversational turn.
- The transient takeover view keeps up to 60k characters of recent conversation independently
  of the mandatory system prompt. Subtracting the system-prompt size from this budget can omit
  the newest user turn when the system prompt alone exceeds the limit.
- Compaction defaults: 128k model context, 70% trigger, 32k output/tool reserve, and
  120k-character checkpoint chunks. Override with `ADAPSIS_CONTEXT_TOKENS`,
  `ADAPSIS_COMPACTION_PERCENT`, and `ADAPSIS_OUTPUT_RESERVE_TOKENS`.
- `memory_cypher(principal_id, query)` exposes ACL-filtered, read-only Cypher anchored at
  `MATCH (memory:Memory {id: $memory_id})`. It deliberately rejects query shapes that can
  escape the authorized memory anchor.
- Operations: `adapsis memory-migrate <session.json> [--database PATH]` is idempotent;
  `adapsis memory-stats [--database PATH] [--context CONTEXT]` reports graph/checkpoint state.
- On openSUSE, LibreSSL development symlinks can shadow OpenSSL 3. `build.rs` links
  Ladybug's prebuilt archive to exact `libssl.so.3`/`libcrypto.so.3` SONAMEs.

## Telegram Bot (TelegramBot.ax)
- **Photo input**: `process_update` accepts Telegram's largest `message.photo`
  variant or an image MIME `message.document`, downloads it as an `Attachment`,
  and passes it as the optional sixth `llm_takeover` argument. Adapsis emits
  OpenAI `image_url` content using a base64 data URL; the gateway translates it
  for the selected provider.
- **Multi-admin**: `admin_user_ids` shared var, comma-separated (e.g. `"1815217,456789"`)
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
the bot can freely debug, but can't wipe Renate's laptop from a misread message.

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
- MusicGen auto-switches to `gemma4s` during generation to free VRAM

## Music Generation (ace-step-rs)
- HTTP endpoint: `POST http://127.0.0.1:8091/generate` → raw OGG bytes
- CPU offload: text encoder on CPU, DiT+VAE on GPU (13GB → 10GB VRAM)
- Non-blocking: `MusicGen.generate()` spawns background task, returns immediately
- Delivery: `conversation_notify` with Attachment → `send_reply_with_attachment`
- Auto model switch: saves current model, switches to gemma4s, generates, switches back

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
    user service. Has `Conflicts=` GPU mutex (llama-server, ace-step-gen,
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
own `10.0.0.x` on port **3002**, and each has a `persona.md` (loaded at runtime
by `prompt.rs::persona()` from `~/.config/adapsis/persona.md`, no rebuild needed)
that names itself + its peers and lists the exposed endpoints so the bots can
talk to each other via `http_get`/`http_post`/`http_request`.

| VPN IP   | Host  | Persona  | Runs as / service                          | Binary path                |
|----------|-------|----------|--------------------------------------------|----------------------------|
| 10.0.0.1 | here  | Kronk    | marenz, **user** `adapsis.service`         | `~/.local/bin/adapsis`     |
| 10.0.0.2 | sleek | Hobbes   | marenz, **user** `adapsis.service`         | `~/.local/bin/adapsis`     |
| 10.0.0.4 | edox  | Moonwolf | `adapsis` user, **system** `adapsis-bot.service` (model `family-bot`, Renate's machine) | `/home/adapsis/bin/adapsis` |

- **Persona vs capability module:** the persona (identity/tone) lives in
  `persona.md`; the *capabilities* live in modules. On edox the family-bot's
  capability module is still `Wolfi.ax` (don't rename — `Wolfi.remember(...)` is
  a real function), but the bot's **identity** is "Moonwolf".
- **Mesh topology = `mesh.md`, NOT `persona.md`.** The VPN/peer table + exposed
  endpoints live in `~/.config/adapsis/mesh.md` (override path via
  `ADAPSIS_MESH_FILE`), loaded by `prompt::mesh_topology()` at runtime (no
  rebuild to edit content). `persona.md` is purely character/tone. Keep them
  separate: mesh = facts (shared across conversations), persona = voice.
- **Prompt builders — there are THREE, keep them in sync.** The system prompt is
  assembled in three independent places; a fragment added to one does NOT appear
  in the others:
  1. `handle_llm_takeover` (`api/llm_handlers.rs` ~796) — **Telegram & agent**
     path. Has two branches: sandboxed/non-admin (uses `persona()`) and
     admin (uses `system_prompt()` + `adapsis_identity()`). Mesh is injected
     into **both**.
  2. `ask` (`api/llm_handlers.rs` ~197) — `POST /api/ask` (CLI/HTTP).
  3. `ask_stream` (`api/llm_handlers.rs` ~404) — `POST /api/ask-stream` (SSE).
  All three now append `mesh_topology()`. The original bug: the VPN info only
  reached the non-admin llm_takeover branch, so **admin DMs and `/api/ask`
  claimed to know nothing about peers**. Each builder refreshes the first system
  message on every request while preserving the remaining conversation history,
  so runtime prompt, persona, mesh and visible-capability changes reach existing
  persisted conversations after the updated process handles its next message.
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

## Renate Desktop Helper (edox only, 2026-07-06)
The `adapsis` service user cannot touch Renate's desktop session (correct OS
behavior), so edox runs **`renate-agent`** — a stdlib-only Python HTTP daemon
as user `renate` on `127.0.0.1:3010` (systemd **user** unit
`~renate/.config/systemd/user/renate-agent.service`, script at
`~renate/.local/bin/renate-agent.py`, linger enabled). Endpoints: `/notify`
(notify-send), `/play` (paplay, named sounds only), `/open-url` (http/https
only), `/dialog` (zenity yes/no, 2-min timeout), `GET /health`. The adapsis
side is the **`RenateAgent.ax`** module (in the `assist` permission group, so
Renate's sandboxed conversations may call it): `notify`, `play_sound`,
`open_url`, `ask_yes_no` — all ≤2-statement io fns (exempt from the test gate)
delegating JSON building to tested pure helpers. Use `ask_yes_no` for consent
before invasive actions.
