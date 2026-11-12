# Adapsis design decisions

**This file is authoritative.** It records the owner-confirmed design (interview of
2026-11-09). Narrowing or removing a recorded capability requires an explicit owner
decision, recorded here. Workers cite the rows their change touches; reviewers check
every change for narrowing against these rows. Until the task system's own decision
records exist, this file is their importable source.

Evidence: [docs/design-intent-audit-20261109.md](docs/design-intent-audit-20261109.md)
(design-intent audit that preceded the interview).

## Goal

| ID | Area | Decision | Rationale/notes | Status | Issue |
|---|---|---|---|---|---|
| G1 | Goal | Adapsis is the owner's system for overview, management, tracking and delegation on a well-defined task system, and it does the coding itself with its own agents. | | active | #120 |
| G2 | Goal | It proves itself first on Adapsis itself and admin work (here, edox), then on Frequenz (sleek), then on Chronica. | Rollout order of use cases. | active | |
| G3 | Goal | Success = 3 weeks of working through Kronk/Hobbes, with OpenCode only as an emergency exit. | | active | |

## Interfaces

| ID | Area | Decision | Rationale/notes | Status | Issue |
|---|---|---|---|---|---|
| I1 | Interfaces | Web is primary, plus TUI and Telegram. One backend (the Adapsis axum server) serves view models and an SSE event stream to all of them. | | active | |
| I2 | Interfaces | Web UI is server-rendered htmx. Templates and view models live in Python modules, so the bot can extend the UI. | | active | |
| I3 | Interfaces | TUI: `adapsis tui` (ratatui). | | active | |
| I4 | Interfaces | Telegram is an easy UI: inline buttons (approve / merge / pause / cancel / details), `/tasks`, and a reply to a task message routes to that task. | | active | |
| I5 | Interfaces | Voice and discussion interfaces. | Later. | deferred | |

## Nodes & security

| ID | Area | Decision | Rationale/notes | Status | Issue |
|---|---|---|---|---|---|
| N1 | Nodes | One manager on sleek (professional/Frequenz) and one here (personal). edox is an executor for Renate's PC, with desktop consent prompts. | | active | |
| N2 | Nodes | The web UI combines views. here and sleek see each other; both see edox. edox communicates (messages, notifications, requests) but sees no foreign task views. Frequenz titles may show here. | | active | |
| N3 | Security | Node identity = WireGuard IP plus a per-node permission profile. | Signed node keys later (N7). | active | |
| N4 | Security | Web UI is reachable on our WireGuard VPN only, with no login. It is NEVER bound to the Frequenz company network; bind explicitly to the WireGuard interface. | Login later (N8). | active | |
| N5 | Security | No extra execution isolation; tasks run as the owner's user. | Owner clarification 2026-11-10: ordinary owner-authority workspace bash; no bwrap filesystem secret masking requirement. Cgroups bound command lifetime, not filesystem authority. Accidental secret accessibility through bash is accepted; deliberately discovering/reading/dumping credentials must be discouraged in tool instructions. See clarification below. | active | #129 |
| N6 | Security | Secrets are named capabilities; tokens never enter model context or logs through those capabilities. | Owner clarification 2026-11-10: secrets should live in a secret store, not scattered plaintext. Named capabilities remain planned; concrete store/provider and migration are not implemented or decided. This is not a claim that owner-authority bash cannot read secrets, nor that active avoidance is verified. Existing logging/redaction gates remain required. | active; implementation pending | #129 |
| N7 | Security | Signed node keys. | Later. | deferred | |
| N8 | Security | Web login. | Later. | deferred | |

**Owner clarification (2026-11-10, relayed through the delegation manager):**
"I think our secrets should generally be in a \"secret store\" and if there are
any found through bash then so be it. they should be properly set anyway and of
course the bot should actively try to avoid reading any in .env files and such,
but it does that anyway nowadays, no?"

This resolves the workspace secret-masking question in favor of ordinary
owner-authority bash, without mandatory filesystem masking. Tool instructions
must discourage deliberate discovery, reading or dumping of `.env`, credential
and token files or process environments; that guidance is behavioral, not an
isolation guarantee. The wiring slice must add it explicitly. A concrete secret
store, its provider and migration remain follow-up decisions/implementation;
neither an implemented store nor verified active avoidance follows from this
clarification. It does not remove named secret capabilities or existing
content-logging/redaction requirements.

## Language & tools

| ID | Area | Decision | Rationale/notes | Status | Issue |
|---|---|---|---|---|---|
| L1 | Language | `.ax` and the legacy `<code>` mode are removed; Python is the only language. | Supersedes the `+`/`!`/`?` syntax goals in AGENTS.md. | active | #86 #89 |
| L2 | Tools | Interaction = direct JSON tools (read / search / edit / run / git) plus `run` with ad-hoc Python, which replaces the role of `!`. | Workspace bash follows N5/N6's 2026-11-10 clarification: ordinary owner authority, behavioral secret-avoidance instructions, named capabilities still planned. | active | #89 #129 |
| L3 | Tools | Permission profiles map to JSON tool lists and are never widened. Every list has `finish` and a restricted `run`: a block may call only what its speaker can already call, so `run` adds composition, never authority. Guests: their profile's guest tools as direct tools; in a block, pure builtins plus exactly those tools. Execute-only speakers (e.g. Renate): inspect / evaluate / the turn-bound memory tools; in a block, pure builtins plus functions of modules they have Execute on. Neither gets a `module` scope, `+shared` access, program-changing builtins, spawn, or introspection beyond what they have today. Pure builtins are the non-IO registry builtins minus those that reveal host information (`home_dir`). A turn that cannot be classified with certainty is a guest without tools. | Owner decision of 2026-11-10; replaces "guests get their guest tools" only. Restricted blocks are metered inside the interpreter (steps, value size, IO calls per block, costly tools once per block, wall clock). New limits: 24 turn-bound tool calls per turn for restricted speakers, direct calls included (legacy `<code>` turns did not bound them), and 180 seconds of block run time per turn (60 per block). Not carried over from the legacy path: its unchecked operations for execute-only speakers (evaluating any function or inline expression without a per-module check, unfiltered `?` queries, other legacy commands). Guests on HTTP stay on the legacy protocol (no turn identity to bind tools to). Behaviour change on every path, legacy included: `memory_remember` with scope `global` is refused for every turn that is not write-capable, because a global memory is injected into every conversation including the owner's. | active | #129 |
| L4 | Language | Restore IO testing via pytest / `unittest.mock`-style test-local mocks plus scripted IO sequences. | | active | |
| L5 | Language | Restore route / startup / event-source declarations in Python and in native mode. | | active | |
| L6 | Language | Restore structured parallelism (`asyncio.gather` / TaskGroup style), with `parallel` as an enforced effect. | | active | |
| L7 | Language | Effects: io, async, fail, mut (+ parallel once enforced). rand, yield and unsafe are removed. | | active | |
| L8 | Language | Provenance tracker. | Idea kept for later. | deferred | |

## Task system

| ID | Area | Decision | Rationale/notes | Status | Issue |
|---|---|---|---|---|---|
| T1 | Tasks | Tasks + runs, with dependencies, blocker reason, parent/child, gate reached + gate ceiling, priority, and progress as plan steps. | | active | #120 |
| T2 | Tasks | Sub-agents: child tasks for real work plus non-writing helper runs inside a task. The task system replaces `!agent`, `!plan` and `!roadmap` completely. | | active | #120 |
| T3 | Tasks | Task sources: GitHub, Slack, owner requests. Automatic tasks start on their own within their gate. | | active | #120 |
| T4 | Tasks | One writer per worktree. Task worktrees are `git worktree`s of the shared repository (no clone per task). | | active | #120 |
| T5 | Tasks | Start limits: depth 3, 4 concurrent writers per task tree, a cost cap. On budget exhaustion, finish cheaply and report. | Revisit with data. | active | #120 |
| T6 | Tasks | Everything is resumable across restarts. | | active | #120 |
| T7 | Tasks | Design decisions are durable records; the bot runs design sessions; a narrowing gate prevents silent capability loss. | This file is their source until those records exist. | active | #120 |

## Autonomy

| ID | Area | Decision | Rationale/notes | Status | Issue |
|---|---|---|---|---|---|
| A1 | Autonomy | Gate ladder: Plan → Implement → Tests → Review → Draft PR → Ready → Merge → Deploy. The stop point is configurable per project/task. | | active | |
| A2 | Autonomy | Default stop points: Adapsis through deploy; Chronica through merge; admin work through, with Renate's consent for invasive steps on edox; Frequenz draft PR, with full autonomy per task possible (e.g. following requested reviews). | | active | |
| A3 | Autonomy | Always, at any autonomy level: tests green; independent review (different provider where possible); no force-push or history rewrite on shared branches (own task/PR branches may be rebased and force-pushed with lease); no secrets in commits; budget caps; the destructive-command guard; resumability. | | active | |
| A4 | Autonomy | Branch + PR everywhere, including own repos. | | active | |
| A5 | Autonomy | Self-deploy is always done by the bot, including edox, at any time, with release gate + health check + automatic rollback + notification. | | active | |
| A6 | Autonomy | Model routing by task fit, with budget overrides. Automatic provider failover with notification. Frequenz uses only OpenAI/Anthropic. | | active | |

## Versioning & self-modification (option E)

| ID | Area | Decision | Rationale/notes | Status | Issue |
|---|---|---|---|---|---|
| V1 | Versioning | The module library is a local git repository. Link records: task/run → repository + branch + PR; commits are read live from git; the merge/squash commit is recorded. Uniform for all repos. | | active | |
| V2 | Self-modification | Self-modification follows the project gate setting. Start rule: an independent review before merging every self-modification (cheap capable model, different provider). | Reduce later from cost data. | active | |
| V3 | Self-modification | Automatic rollback of module code to last-known-good on regression. | | active | |
| V4 | Versioning | Rewind keeps runtime state by default; each `+shared` value is type-checked on reload, with reset to default or a migration function; an explicit "with state" option restores a snapshot. | Replaces the built-in replay rewind. | active | #122 #123 #126 |
| V5 | Versioning | External module repos: pinned by tag/sha; added by the bot only inside an owner-started task; no permissions by default (explicit grant the first times); source-prefixed names; updates only via a reviewed pin change. | | active | |

## Integrations & Frequenz

| ID | Area | Decision | Rationale/notes | Status | Issue |
|---|---|---|---|---|---|
| F1 | Integrations | GitHub and Slack are polled. | | active | |
| F2 | Frequenz | Frequenz rules (`~/frequenz/AGENTS.md` on sleek plus the OpenCode manager/orchestrator/liaison agent files) are imported ONCE into Adapsis's structure. | | active | |
| F3 | Frequenz | Slack posts as the owner. Simple review requests and trivial posts on its own; requested reviews posted directly; everything addressed to people only on explicit request, usually as a draft he checks. | | active | |
| F4 | Frequenz | Hard switch on sleek, importing the open work; OpenCode stays installed as a fallback. | | active | |

## Operations

| ID | Area | Decision | Rationale/notes | Status | Issue |
|---|---|---|---|---|---|
| O1 | Operations | Runtime error watchdog is live. | Deployed in 1bc06e8. | active | #113 |
| O2 | Operations | Error snapshots with variables + watchpoints. | Next. | active | #127 |
| O3 | Operations | Nightly backups to the other nodes, unencrypted for now. Frequenz data only goes sleek → here, never to edox. | | active | |
| O4 | Operations | Independent supervisor. | | deferred | |
| O5 | Operations | Encrypted backups. | Later (node keys N7, web login N8 likewise). | deferred | |

## Build order

| ID | Area | Decision | Rationale/notes | Status | Issue |
|---|---|---|---|---|---|
| B1 | Build order | 1. Finish the `.py` migration, remove `<code>`/`.ax`, JSON tool profiles for all speakers (Kronk first, edox later via self-deploy). | | active | #86 #89 |
| B2 | Build order | 2. `run` + workspace tools + secret capabilities. | | active | |
| B3 | Build order | 3. Task MVP here + web UI. | | active | #120 |
| B4 | Build order | 4. TUI + Telegram views. | | active | |
| B5 | Build order | 5. sleek manager (hard switch). | | active | |
| B6 | Build order | 6. edox. | | active | |
