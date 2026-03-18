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
  typeck.rs        — Type checker: symbol table, type inference, semantic queries
  eval.rs          — Interpreter for !test and !trace (evaluates functions with concrete inputs)
  llm.rs           — LLM client (OpenAI-compatible HTTP API, streaming SSE, backend trait)
  orchestrator.rs  — The feedback loop: prompt → LLM → parse → validate → typecheck → test → feedback → repeat
  prompt.rs        — System prompt and feedback message construction
```

## Key Design Decisions

- **Parser has its own types** (`parser::Operation`, `parser::Expr`, etc.) separate from `ast.rs`. The validator converts between them. This is intentional — parser types are close to surface syntax, AST types are the validated program state.
- **Each LLM response is a complete program** (when it contains type/function definitions). The orchestrator resets program state before applying. This avoids duplicate definition errors.
- **Stray `end` tokens after test blocks are silently ignored** — LLMs frequently add them.
- **`Ok(None)` in test expectations is a wildcard** — matches any `Ok(...)` value.
- **Result-returning functions auto-wrap** — if a function's return type is `Result<T>`, successful returns are wrapped in `Ok()`.
- **+branch is for tagged unions only** — for string/value-based dispatch, use multiple +check statements with early returns.

## LLM Backend

- Uses llama.cpp server via OpenAI-compatible API at `http://127.0.0.1:8081`
- Model: Qwen3.5-35B-A3B (MoE, 3B active params) or Qwen3.5-9B (backup at ~/models/)
- Systemd user service: `llama-server.service`
- Handles Qwen's thinking mode (separate `reasoning_content` field in API response)
- To switch models, edit `~/.config/systemd/user/llama-server.service` and `systemctl --user daemon-reload && systemctl --user restart llama-server`

## Test Commands

```bash
cargo run -- test examples/validate.forge     # All 4 tests should pass
cargo run -- test examples/trace_test.forge   # Tests trace and query features
```

## Known Limitations

- **+branch syntax** is too rigid — only supports identifier patterns, not string literals. Use check+return for conditional dispatch.
- **Type checking** is lenient — struct types match by name only, no field-level checking yet.
- **Evaluator** doesn't support all expression types in test case inputs (e.g., nested calls).
- **No incremental compilation** — each LLM response resets the program if it contains definitions.

## Dependencies

- `reqwest` — HTTP client for LLM API
- `tokio` — async runtime
- `serde`/`serde_json` — serialization
- `clap` — CLI argument parsing
- `anyhow` — error handling
- `tracing` — logging
