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
- `--autonomous` flag: inject a goal and let the AI work without user input

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

## Current Priority: Telegram Bot

The first autonomous goal. The AI should build a Telegram bot entirely in Forge:

- Use the async IO system (TCP, HTTP requests via shell/builtins)
- Connect to the Telegram Bot API (HTTP long polling or webhooks)
- Handle messages, respond with AI-generated content
- Persist state across restarts via session
- If missing builtins are needed (HTTP client, JSON parsing), use `!opencode` to add them

This exercises the full stack: async IO, LLM calls, self-extension, error handling,
and proves ForgeOS can build real-world applications autonomously.

## Self-Enhancement Roadmap

The AI inside ForgeOS should be working toward making itself better. The loop:

1. **Identify a limitation** — missing builtin, awkward syntax, slow eval, missing IO op
2. **Try to solve it in Forge first** — write a function, use existing builtins
3. **If Forge can't do it, use `!opencode`** — request a Rust-level change
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
- [x] ForgeOS identity prompt: AI knows it's a self-improving system
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
8. **LoRA fine-tuning** — train a model specifically on Forge
9. **Self-hosting** — Forge parser/evaluator written in Forge (design doc Phase 8)

## Post-Self-Hosting Goal: AI-in-the-Loop

Once Forge can parse and evaluate itself, the next milestone is the AI running inside ForgeOS:

1. The AI operates through the ForgeOS HTTP API — emitting mutations, running evals, querying state
2. The AI modifies the Forge runtime FROM INSIDE Forge — writing parser improvements that the parser then uses
3. A self-modifying AI in its own environment

The loop: AI writes Forge code → ForgeOS validates and runs it → results feed back → AI
improves the code → the improved code improves ForgeOS itself.

This is the design doc's Phase 8 vision made concrete.
