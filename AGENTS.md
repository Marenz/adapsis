# Forge — AI-First Programming Language

## Overview
Forge is an AI-native program representation and mutation protocol with a constrained surface language. The core insight: incremental, feedback-driven program construction produces more correct software than one-shot code generation.

See `forge-design-doc.md` for the full design document.

## Build & Run

```bash
cargo build
cargo run -- check examples/validate.forge    # Parse and validate a .forge file
cargo run -- test examples/validate.forge     # Run tests in a .forge file
cargo run -- run --task "description" -m 10   # Run the LLM feedback loop
```

## Architecture

```
src/
  main.rs          — CLI entry point (clap)
  ast.rs           — Core AST types (Program, Module, Function, Statement, Expr, Type, Effect)
  parser.rs        — Line-oriented parser for Forge mutation syntax (+fn, +let, +call, etc.)
  validator.rs     — Converts parser operations → AST, validates, applies mutations
  eval.rs          — Basic interpreter for !test blocks (evaluates functions with concrete inputs)
  llm.rs           — LLM client (OpenAI-compatible HTTP API, streaming SSE, backend trait)
  orchestrator.rs  — The feedback loop: prompt → LLM → parse → validate → test → feedback → repeat
  prompt.rs        — System prompt and feedback message construction
```

## Key Design Decisions

- **Parser has its own types** (`parser::Operation`, `parser::Expr`, etc.) separate from `ast.rs`. The validator converts between them. This is intentional — parser types are close to surface syntax, AST types are the validated program state.
- **Each LLM response is a complete program** (when it contains type/function definitions). The orchestrator resets program state before applying. This avoids duplicate definition errors.
- **Stray `end` tokens after test blocks are silently ignored** — LLMs frequently add them.
- **`Ok(None)` in test expectations is a wildcard** — matches any `Ok(...)` value.
- **Result-returning functions auto-wrap** — if a function's return type is `Result<T>`, successful returns are wrapped in `Ok()`.

## LLM Backend

- Uses llama.cpp server via OpenAI-compatible API at `http://127.0.0.1:8081`
- Model: Qwen3.5-35B-A3B (MoE, 3B active params) or Qwen3.5-9B
- Systemd user service: `llama-server.service`
- Handles Qwen's thinking mode (separate `reasoning_content` field in API response)

## Test Commands

```bash
cargo run -- test examples/validate.forge     # All 4 tests should pass
```

## Dependencies

- `reqwest` — HTTP client for LLM API
- `tokio` — async runtime
- `serde`/`serde_json` — serialization
- `clap` — CLI argument parsing
- `anyhow` — error handling
- `tracing` — logging
