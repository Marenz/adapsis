# Adapsis — AI-First Programming Language (AdapsisOS)

## Overview
Adapsis is an AI-native program representation and mutation protocol with a constrained surface language. Programs are built incrementally through validated mutations. The AST is the source of truth, changes happen through small mutations with immediate feedback, and every modification is logged in an append-only revision history.

See `adapsis-design-doc.md` for the full design document.

## Build & Run

```bash
cargo build

# AdapsisOS — the full environment (HTTP API + browser UI + persistent session)
cargo run -- os --port 3000 --session project.json

# Interactive REPL
cargo run -- repl --session project.json

# LLM feedback loop (one-shot task)
cargo run -- run --task "description" -m 10

# Architect mode (design first, implement per-function)
cargo run -- architect --task "description" --port 3000

# Browser UI for one-shot task
cargo run -- serve --task "description" --port 3000

# Parse and validate
cargo run -- check examples/fizzbuzz.ax

# Run tests
cargo run -- test examples/fizzbuzz.ax

# Compile to native code
cargo run -- compile examples/compile_test.ax -f add -a "3,4"
```

## HTTP API (AdapsisOS mode)

```
POST /api/mutate   — {"source": "+fn add..."} → revision + results
POST /api/eval     — {"function": "add", "input": "a=3 b=4"} → result
POST /api/test     — {"source": "!test add\n  +with..."} → pass/fail
POST /api/query    — {"query": "?symbols"} → response
GET  /api/status   — program state, revision, function/type list
POST /api/history  — {"limit": 20} → mutation log + working history
POST /api/rewind   — {"revision": 5} → rewinds program state
```

## Architecture

```
src/
  main.rs          — CLI entry point (clap)
  api.rs           — HTTP REST API for AdapsisOS
  ast.rs           — Core AST types (Program, Module, Function, Statement, Expr, Type, Effect)
  parser.rs        — Line-oriented parser for mutation syntax
  validator.rs     — Parser → AST conversion, validation, mutation application
  typeck.rs        — Type checker, symbol table, semantic queries
  eval.rs          — Tree-walking interpreter for tests, evals, traces
  compiler.rs      — Cranelift JIT compiler (Int, Float, Bool → native x86-64)
  llm.rs           — LLM client (OpenAI-compatible, streaming SSE, Qwen thinking mode)
  orchestrator.rs  — Feedback loop + architect mode
  session.rs       — Mutation log, working history, save/load, rewind
  library.rs       — Persistent module library (~/.config/adapsis/modules/), auto-load/persist
  repl.rs          — Interactive REPL with /commands
  events.rs        — Event streaming (WebSocket)
  server.rs        — Browser UI server
  prompt.rs        — System prompt and feedback messages
web/
  index.html       — Browser UI (vanilla JS, Tokyo Night theme)
examples/
  validate.ax   — Validation with checks
  fizzbuzz.ax   — If/elif/else with modulo
  if_else.ax    — Age classification
  modulo.ax     — Modulo operator
  result_chain.ax — Error propagation (auto + explicit)
  simple_syntax.ax — key=value test syntax
  compile_test.ax  — Compiler test (5 functions)
training/
  generate_corpus.py     — Manual training examples (35)
  auto_generate_corpus.py — Auto-generation via LLM (42 tasks)
  train_lora.py          — QLoRA training script (unsloth)
  benchmark.sh           — Multi-model benchmark
```

## Key Design Decisions

- **Append-only mutation log** — every change gets a revision ID. Program state is reconstructable by replaying mutations 0..N. Rewind to any revision.
- **Two error handling patterns** — auto-propagation (`+call val:T = func(x)` with `[fail]`) and explicit handling (`+call res:Result<T> = func(x)` with `.is_ok`/`.unwrap`/`.error`).
- **key=value test syntax** — `+with a=3 b=4 -> expect 7` instead of JSON-like struct literals. Cleaner for LLMs.
- **Parser has its own types** separate from AST. Validator converts between them.
- **Each LLM response in `run` mode is a fresh program.** In `architect` and `repl` modes, program state persists across iterations.
- **Session persistence** — JSON file with mutation sources for replay. Auto-saves every 30s in `os` mode.

## LLM Backend

Multiple LLM backends available:

### Cloud (via llm-gateway on port 4000)
- `anthropic/claude-sonnet-4-6`, `anthropic/claude-opus-4-6`, etc.
- `chatgpt/gpt-5.4`, `chatgpt/gpt-5.4-mini`
- URL: `http://127.0.0.1:4000`

### Local (via llama.cpp in podman, GPU-accelerated)
- **Qwen3.5-9B** (Q4_K_M) — systemd service `llama-server`, port 8081
  - `systemctl --user start llama-server`
  - Model: `/home/marenz/models/Qwen3.5-9B-Q4_K_M.gguf`
- **Qwen3.5-35B-A3B** (Q4_K_XL, MoE active 3B) — systemd service `llama-server-a3`, port 8082
  - `systemctl --user start llama-server-a3`
  - Model: `/home/marenz/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`
  - 20GB model, needs full 3090 (can't run alongside 9B)
- Both use podman with `llama-server-cuda` image, NVIDIA GPU passthrough
- Context: 32-64k tokens, flash attention, Q8 KV cache

### Local (via Ollama on port 11434)
- `glm-4.7-flash`, `gpt-oss:20b`
- URL: `http://127.0.0.1:11434`

### Available local models
```
~/models/Qwen3.5-9B-Q4_K_M.gguf
~/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
~/models/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled.i1-Q4_K_M.gguf
~/models/Qwen3.5-4B-Forge-LoRA-BF16.gguf
~/models/Qwen3.5-9B-Forge-LoRA-v2-Q4_K_M.gguf
~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
~/models/Phi-4-mini-instruct-Q4_K_M.gguf
~/models/gemma-3-4b-it-Q4_K_M.gguf
```

### Notes
- Qwen3.5 family learns Adapsis syntax from system prompt alone
- Non-Qwen models (Gemma, Phi, Llama) produce zero valid Adapsis
- Handles Qwen's thinking mode (`reasoning_content` field)

## AdapsisOS !opencode Integration

- `!opencode` uses `opencode run --attach http://localhost:4096` (connects to running server)
- Sessions visible in `opencode sessions list`
- Each `!opencode` gets its own git worktree for isolation
- Sequential lock: only one `!opencode` at a time
- Config flags: `--opencode-git-dir`, `--opencode-worktree-dir`

## Test Commands

```bash
# All examples
for f in examples/*.ax; do echo "$f:"; cargo run -q -- test "$f" | grep -c PASS; done

# Compiler test
cargo run -- compile examples/compile_test.ax -f fizz_type -a "15"
```

## Dependencies

- `reqwest` — HTTP client for LLM API
- `tokio` — async runtime
- `serde`/`serde_json` — serialization
- `clap` — CLI argument parsing
- `anyhow` — error handling
- `tracing` — logging
- `axum` — HTTP/WebSocket server
- `tower-http` — CORS middleware
- `futures` — async utilities
- `cranelift` + `cranelift-jit` + `cranelift-module` + `cranelift-native` — JIT compiler
