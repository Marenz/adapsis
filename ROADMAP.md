# ForgeOS Roadmap

## The Three Modes

ForgeOS operates in three distinct modes, each building on the previous:

### Agentic Mode (current focus)

ForgeOS as a **live, autonomous AI programming environment**. The AI has a session, a
program state, a plan, and works toward goals — both user-directed and self-initiated.

- The AI receives user requests and builds Forge programs incrementally
- It can spawn sub-agents with scoped access for parallel work
- It proactively identifies missing capabilities and extends itself:
  - **Forge-level**: write new functions, types, modules in Forge
  - **Rust-level**: use `!opencode` to add builtins, fix runtime bugs, add IO operations
- It maintains a plan and works through it step by step
- Agents communicate via `!msg` / `?inbox`
- All mutations are logged, rewindable, inspectable

This is the default `forge os` mode.

### Adaptive Mode (planned)

ForgeOS loads a **pre-built Forge program** and the AI can modify/extend it at runtime
via the mutation protocol. Like a live-patching server.

- Start with: `forge os --load program.forge --adaptive`
- The program runs (e.g. an HTTP server, a data pipeline, a game)
- The AI monitors it via `!watch` and `?tasks`
- When the user or an event triggers a change, the AI hot-patches the running program
- New functions can be added, existing ones replaced, types extended
- The program never stops — mutations apply live

Use cases: self-adapting servers, AI-driven game NPCs, live data pipeline reconfiguration.

### Compiled Mode (planned)

Freeze the current program state into a **standalone native binary** via AOT compilation.
No runtime, no mutation protocol, no LLM — just a compiled executable.

- Build with: `forge compile --output binary session.json`
- Takes the current `Program` AST and compiles everything through Cranelift
- Produces a statically-linked binary with no Forge runtime overhead
- Useful for deploying what the AI built as a production artifact

The flow: Agentic mode → build and iterate → Compiled mode → deploy.

## Self-Enhancement Roadmap

The AI inside ForgeOS should be working toward making itself better. The loop:

1. **Identify a limitation** — missing builtin, awkward syntax, slow eval, missing IO op
2. **Try to solve it in Forge first** — write a function, use existing builtins
3. **If Forge can't do it, use `!opencode`** — request a Rust-level change
4. **After rebuild, verify the new capability works** — write a test, eval it
5. **Document what changed** — update the prompt, add to the registry

### Current capabilities (what works)
- Full language: types, functions, pattern matching, effects, modules
- Async IO: TCP, files, shell, stdin, LLM calls
- Coroutine runtime with task registry and wait-reason tracking
- Multi-agent with scoped branches, message bus
- Self-extension via `!opencode` → OpenCode → cargo build → exec restart
- Cranelift JIT for Int/Float/Bool/String/Struct
- 118 test cases, 12 e2e tests
- HTTP API + SSE streaming + web UI

### Missing / next targets
- **AOT compilation** — freeze program to standalone binary (Compiled mode)
- **Adaptive mode** — `--load` flag, hot-patching running programs
- **Provenance tracking** — who wrote what, when, why (design doc Phase 7)
- **Grammar constraints** — GBNF for guaranteed syntactically valid output
- **LoRA fine-tuning** — train a model specifically on Forge
- **Self-hosting** — Forge parser/evaluator written in Forge (design doc Phase 8)
- **Agent-to-agent direct communication** — beyond message bus, shared state
- **`!config` command** — per-task model selection, temperature, etc.

## Post-Self-Hosting Goal: AI-in-the-Loop

Once Forge can parse and evaluate itself, the next milestone is the AI running inside ForgeOS:

1. The AI operates through the ForgeOS HTTP API — emitting mutations, running evals, querying state
2. The AI modifies the Forge runtime FROM INSIDE Forge — writing parser improvements that the parser then uses
3. A self-modifying AI in its own environment

The loop: AI writes Forge code → ForgeOS validates and runs it → results feed back → AI
improves the code → the improved code improves ForgeOS itself.

This is the design doc's Phase 8 vision made concrete.
