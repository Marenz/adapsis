# Per-context identity files

One markdown file per conversation. The runtime composes every turn's system
prompt as:

1. **Technical core** — `<code>` mechanics, `!done`, `<iteration_budget>`,
   `?source`. Fixed. Nothing in these files can override it.
2. **This file** — who the runtime *is* in that conversation.
3. **Speaker addendum** — what the person who sent *this* message may have done.

The prompt follows the conversation; the permissions follow the speaker
(issue #41). A shared group therefore keeps one identity no matter who speaks.

## Naming

`<context key with every character outside [A-Za-z0-9_-] replaced by _>.md`

| context key | file |
|---|---|
| `main` | `main.md` |
| `telegram:1815217` | `telegram_1815217.md` |
| `telegram:user:7179396338` | `telegram_user_7179396338.md` |
| `telegram:group:-5134158198` | `telegram_group_-5134158198.md` |

Exact match only. There is no wildcard or prefix fallback: a context with no
file gets the technical core and **no identity at all**, and is told so. An
unconfigured conversation inheriting someone else's persona is the bug this
directory exists to fix.

## Deploying

These are the tracked originals. The runtime reads
`~/.config/adapsis/contexts/` (override with `ADAPSIS_CONTEXTS_DIR`):

```bash
mkdir -p ~/.config/adapsis/contexts
cp contexts/*.md ~/.config/adapsis/contexts/
```

No rebuild is needed to change one — the file is read fresh on every turn.

## Self-maintained notes

A module's `remember` function writes to `~/.config/adapsis/persona-notes.md`.
That file is **global**, so it is opt-in per context: put

```
{{persona_notes}}
```

where you want it in a context file and its contents are substituted in. Leave
the marker out and the notes never reach that conversation — which is the point:
they used to ride inside the global persona and therefore reached every one.
Issue #42 replaces the file with per-context Ladybug memories; the marker goes
with it.

## templates/

Starting points to copy, not identities. Nothing under `templates/` is ever
loaded as a context — `family-assistant.md` is the German family-assistant
persona that used to be compiled into the binary as the global fallback. Copy it
to `<the context key>.md` on a node that wants it.

## Proposals

A conversation may propose a rewrite of its **own** file with
`context_propose("<full replacement text>")`. That writes
`<stem>.proposed.md` and, if `ADAPSIS_ADMIN_CONTEXT` is set, pushes the diff to
the administrator. Nothing takes effect until an administrator runs
`context_approve("<context key>")` (or discards it with `context_reject`).
Review pending proposals with `context_proposals()`.
