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
| I6 | Conversation feedback | Normal conversation answers directly without a runtime receipt/start message. Only a conversation-lock wait lasting 10 seconds may produce one honest queue notice; a paired start requires that notice to have actually launched. | Owner UX correction 2026-11-13: confidence without chat noise. This does not supply substantive task-progress reporting or typing indicators. | active | #147 |
| I7 | Interactive completion | Native interactive finish requires a nonempty visible reply or a server-proved visible outcome; one bounded correction is allowed before a visible incomplete result. Tool success, hidden reasoning and queue notices are not outcome evidence. | Owner direction 2026-11-14: enforce this in the system. Finish ends a turn, not a physical job or durable task. Unproved native callback silence fails closed until typed delivery evidence exists (#104). | active | #151 |

## Nodes & security

| ID | Area | Decision | Rationale/notes | Status | Issue |
|---|---|---|---|---|---|
| N1 | Nodes | One manager on sleek (professional/Frequenz) and one here (personal). edox is an executor for Renate's PC, with desktop consent prompts. | | active | |
| N2 | Nodes | The web UI combines views. here and sleek see each other; both see edox. edox communicates (messages, notifications, requests) but sees no foreign task views. Frequenz titles may show here. | | active | |
| N3 | Security | Node identity = WireGuard IP plus a per-node permission profile. | Signed node keys later (N7). | active | |
| N4 | Security | Web UI is reachable on our WireGuard VPN only, with no login. It is NEVER bound to the Frequenz company network; bind explicitly to the WireGuard interface. | Login later (N8). | active | |
| N5 | Security | No extra execution isolation; tasks run as the owner's user. | | active | |
| N6 | Security | Secrets are named capabilities; tokens never enter model context or logs. | | active | |
| N7 | Security | Signed node keys. | Later. | deferred | |
| N8 | Security | Web login. | Later. | deferred | |

## Language & tools

| ID | Area | Decision | Rationale/notes | Status | Issue |
|---|---|---|---|---|---|
| L1 | Language | `.ax` and the legacy `<code>` mode are removed; Python is the only language. | Supersedes the `+`/`!`/`?` syntax goals in AGENTS.md. | active | #86 #89 |
| L2 | Tools | Interaction = direct JSON tools (read / search / edit / run / git) plus `run` with ad-hoc Python, which replaces the role of `!`. | | active | #89 |
| L3 | Tools | Permission profiles map to JSON tool lists and are never widened. Execute-only speakers (e.g. Renate) get evaluate / inspect / finish; guests get their guest tools. | | active | |
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

## Context & cost

| ID | Area | Decision | Rationale/notes | Status | Issue |
|---|---|---|---|---|---|
| C1 | Context | Every conversation uses the generating model's full context window. The history is append-only and is cut only at a checkpoint: when the projected request passes 85% of the window, in one step down to 50%. There is no smaller per-chat budget; cost is controlled by task and turn budgets. Persisted history is never trimmed. | A stable prefix is what the provider cache needs. Implemented in dd8cea4. | active | #68 |
| C2 | Context | Model limits (context window, output limit, cache lifetime) must carry a source: a measured provider answer with date, provider documentation or metadata, or a registry. A model without a sourced value uses the unknown default; unknown is a valid entry. | Guessed limits caused both truncation and over-long requests. Tests enforce the source comment. dd8cea4. | active | #68 |
| C3 | Context | Tokens are projected from serialized request bytes with a bytes-per-token ratio measured from the provider's usage report, per conversation and per generating model; the conservative default before a measurement is 2.5. The same estimate drives the compaction trigger. | Measured 2026-11-10: 2.46 bytes/token on the admin DM, where 4 chars/token had been assumed. 4733b2f; bound to the model in a66a839. | active | #68 |
| C4 | Context | One message is capped at 1/8 of the window in the model view. A capped message tells the model that it was cut and by how many characters; the stored message stays complete. | 4733b2f. The cap is in UTF-8 bytes. | active | #68 |
| C5 | Context | A request must fit the route's body limit. A request refused as too large is retried once in reduced form and the conversation's view is checkpointed, so a context can never stay stuck. | dd8cea4. | active | #68 |
| C6 | Cost | Cache-aware cold start: when a new user turn starts after a pause longer than the provider cache lifetime of the generating model's route, the history in the view is cut once to a compact size (default 100,000 tokens) and then grows append-only again. Never between the rounds of a running turn. The lifetime is a sourced default per route, overridden by what real usage shows (longest pause that still hit, confirmed misses); an unknown lifetime means no cold start. | After the cache has expired the whole history would be re-written at full price; episodes and memories cover the older part. Implemented in a66a839. | active | #68 |
| C7 | Cost | Anthropic routes request the 1-hour cache lifetime on every cache boundary. | With the 5-minute default almost every chat turn is cold. Write price 2x instead of 1.25x, on newly appended tokens only. Measured 2026-11-10: hit at 1200 s, miss at 3900 s. Implemented in a66a839. | active | #68 |

## Build order

| ID | Area | Decision | Rationale/notes | Status | Issue |
|---|---|---|---|---|---|
| B1 | Build order | 1. Finish the `.py` migration, remove `<code>`/`.ax`, JSON tool profiles for all speakers (Kronk first, edox later via self-deploy). | | active | #86 #89 |
| B2 | Build order | 2. `run` + workspace tools + secret capabilities. | | active | |
| B3 | Build order | 3. Task MVP here + web UI. | | active | #120 |
| B4 | Build order | 4. TUI + Telegram views. | | active | |
| B5 | Build order | 5. sleek manager (hard switch). | | active | |
| B6 | Build order | 6. edox. | | active | |
