# Adapsis

An AI-native programming language and live programming environment.

Adapsis is a system for AI-assisted software construction where programs are built incrementally through validated mutations, not generated in one shot. The LLM emits small changes, the runtime validates each one immediately, and errors feed back into the next generation step.

## What works today

- **Feedback loop** — LLM generates Adapsis code, runtime validates and runs tests, errors feed back, model self-corrects
- **Architect mode** — model designs types + function signatures first, then implements one function at a time
- **REPL** — interactive session with mutation log, working history, save/load, rewind to any revision
- **Browser UI** — WebSocket-based live view of mutations, program state, and test results
- **Cranelift JIT compiler** — Adapsis programs compile to native x86-64 machine code
- **Type checker** — symbol table, type inference, semantic queries (`?symbols`, `?callers`, `?effects`)
- **Statement tracing** — step-by-step execution with `!trace`
- **Result chaining** — auto-propagation (like Rust's `?`) and explicit handling

## Quick start

```bash
# Build
cargo build

# Run the LLM feedback loop (needs llama.cpp server on port 8081)
cargo run -- run --task "Write a fizzbuzz function" --max-iterations 5

# Architect mode: design first, implement per-function
cargo run -- architect --task "Build a user management system" --port 3000

# Interactive REPL with persistent session
cargo run -- repl --session my-project.json

# Browser UI
cargo run -- serve --task "Build a calculator" --port 3000

# Parse and validate a .forge file
cargo run -- check examples/fizzbuzz.forge

# Run tests
cargo run -- test examples/fizzbuzz.forge

# Compile to native code and execute
cargo run -- compile examples/compile_test.forge -f add -a "3,4"
```

## Adapsis syntax

```
// Types
+type User = {name:String, age:Int, email:String}
+type Color = Red | Green | Blue

// Functions with effects
+fn validate (input:User)->Result<User> [fail]
  +check name input.name.len>0 ~err_empty_name
  +check age input.age>=0 ~err_negative_age
  +return input

// Conditionals
+fn fizzbuzz (n:Int)->String
  +let mod3:Int = n % 3
  +let mod5:Int = n % 5
  +if mod3 == 0 AND mod5 == 0
    +return "fizzbuzz"
  +elif mod3 == 0
    +return "fizz"
  +elif mod5 == 0
    +return "buzz"
  +else
    +return "other"

// Error auto-propagation (like Rust's ?)
+fn process (input:User)->Result<Output> [fail]
  +call validated:User = validate(input)
  +return {greeting: concat("Hello, ", validated.name)}

// Tests with key=value syntax
!test fizzbuzz
  +with 15 -> expect "fizzbuzz"
  +with 3 -> expect "fizz"
  +with 7 -> expect "other"

!test validate
  +with name="alice" age=25 email="a@b.com" -> expect Ok
  +with name="" age=25 email="a@b.com" -> expect Err(err_empty_name)

// Eval and trace
!eval fizzbuzz 15
!trace validate name="alice" age=25 email="a@b.com"

// Semantic queries
?symbols
?callers validate
?effects process
```

## Architecture

```
src/
  ast.rs         — AST types (Program, Function, Statement, Expr, Type, Effect)
  parser.rs      — Line-oriented parser for mutation syntax
  validator.rs   — Parser → AST conversion, validation, mutation application
  typeck.rs      — Type checker, symbol table, semantic queries
  eval.rs        — Tree-walking interpreter for tests and evals
  compiler.rs    — Cranelift JIT compiler (Int, Float, Bool, function calls, if/else)
  llm.rs         — LLM client (OpenAI-compatible, streaming, Qwen thinking mode)
  orchestrator.rs — Feedback loop and architect mode
  session.rs     — Mutation log, working history, save/load, rewind
  repl.rs        — Interactive REPL
  server.rs      — Browser UI (axum + WebSocket)
  events.rs      — Event streaming for browser
  prompt.rs      — System prompt and feedback messages
```

## LLM compatibility

Tested with Qwen3.5 family (9B, 35B-A3B MoE). The model learns Adapsis syntax from the system prompt alone — no fine-tuning required for basic tasks. Other model families (Gemma, Phi, Llama) failed to produce valid Adapsis from prompt alone.

## Design document

See [forge-design-doc.md](forge-design-doc.md) for the full language design, including the mutation protocol, effect system, provenance tracking, and self-hosting roadmap.

## Status

This is an active research project. The core loop works — LLMs can write, test, and iterate on Adapsis programs. Compilation to native code works for numeric types. Strings and structs in the compiler are next.

## License

TBD

## Self-Hosting

Adapsis can host itself — the REPL is written in Adapsis:

```bash
# Start the self-hosted REPL (with AI-in-the-loop)
cargo run -- run-async examples/self_hosted_repl.forge

# Features:
#   x = 10+5           — variable assignment
#   y = x * 2          — expressions with variables
#   +fn double(n) = n*2 — function definitions
#   !eval double(5)     — function calls
#   triple(y)           — inline function calls
#   /history            — mutation log
#   /save               — save session to disk
#   /quit               — save and exit
#   natural language     — asks the AI via HTTP
```

Session persists to disk: variables, functions, and history survive restarts.
