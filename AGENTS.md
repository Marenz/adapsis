# Adapsis — Adaptive, Self-Modifying AI Programming Environment

## Overview
Adapsis is an AI-native programming language and live agentic environment. Programs are built incrementally through validated mutations. The AI writes Adapsis code, gets immediate feedback, and iterates. The system can modify its own runtime via `!opencode`.

## Language Design Goals
- **Optimized for LLM-driven construction**, not human typing
- Every operation starts with `+`, `!`, or `?` — clear code vs prose distinction
- Explicit types everywhere — no inference
- Effect system: `[io]`, `[async]`, `[fail]`, `[mut]`
- `+end` closes all blocks (`+fn`, `+if`, `+while`, `+match`, `+each`)
- `!module Name` as state change (no `+end` needed for modules)
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
- Test enforcement — functions >2 statements must have passing `!test` before `!eval`
- Tests persist across sessions and auto-rerun when functions change
- `!mock` for IO testing — intercepts `+await` calls with fake responses

## IMPORTANT: Keep the prompt updated
When modifying the runtime (adding builtins, IO operations, commands, syntax changes):
1. Update `src/prompt.rs` with documentation and examples
2. Register new builtins in `src/builtins.rs`
3. Update this file if the change affects architecture or design

## Build & Test
```bash
cargo build --release
cargo test --release                    # Rust unit tests
for f in examples/*.ax; do cargo run -q --release -- test "$f" 2>&1 | grep -c PASS; done | paste -sd+ | bc  # Adapsis tests
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
```

## Architecture
```
src/
  main.rs        — CLI, autonomous loop, session management
  api.rs         — HTTP API, ask/ask_stream handlers, !opencode orchestration
  ast.rs         — Core AST (Program, Module, Function, Statement, Expr, Type, Effect)
  parser.rs      — Line-oriented parser (+, !, ? prefixes, +end blocks)
  validator.rs   — Parser→AST, mutations, !replace, !remove, !module merging
  eval.rs        — Tree-walking interpreter, builtins, test runner, +match on Result
  compiler.rs    — Cranelift JIT (Int, Float, Bool, String, Struct, While, Match)
  coroutine.rs   — Async IO runtime (TCP, HTTP, files, shell, LLM, mocks, task registry)
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

## Key Adapsis Commands
```
!module Name          — switch module context (all +fn/+type after go here)
!plan set / done N    — task planning
!roadmap add/done/show — persistent long-term roadmap
!mock op "pattern" -> "response"  — IO mocking for tests
!unmock               — clear mocks
!test Module.fn       — run tests (blocks !eval if untested)
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
- Qwen3.5 family works from system prompt. Non-Qwen models fail on Adapsis syntax.

## Test Infrastructure
- Tests persist in `session.stored_tests` (HashMap<fn_name, Vec<StoredTestCase>>)
- Auto-rerun when functions change via `invalidate_and_retest()`
- `!mock` intercepts `+await` calls during `!test` with fake IO responses
- Functions >2 statements blocked from `!eval` until tested
- `!done` rejected if untested functions exist
- Test expectations should be literal values, not function calls
- Pure function calls allowed in test inputs (e.g. `+with config=default_config()`)

## Training Data
- JSONL training log at `--training-log` path (default: `training.jsonl`)
- Each iteration: model, context, thinking, code, outcome, tests passed/failed
- Accumulates in `~/.config/adapsis/training/`
