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
  - `family-bot` (edox) → `chatgpt/gpt-5.5` → `claude-sonnet-4-5` → `deepseek-v4-flash`
  - `dev-bot` (here, sleek) → `anthropic/claude-opus-4-8` → `chatgpt/gpt-5.5` → `deepseek-v4-flash`
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

## Telegram Bot (TelegramBot.ax)
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
- Systemd services: `adapsis.service`, `llama-server.service`, `ace-step-gen.service`, `whisper-server.service`, `caddy.service`
- llama-server: TurboQuant build with turbo3 KV cache, Gemma 4 fixes
- Save-on-change (debounced), not periodic autosave
- Panic hook + exit logging for crash debugging

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
  claimed to know nothing about peers**. Also: each builder caches its system
  prompt at conversation creation (`if conv.messages.is_empty()`), so a
  persona/mesh edit only takes effect for **new** conversations or after a
  restart (which clears in-memory contexts when nothing is persisted).
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
