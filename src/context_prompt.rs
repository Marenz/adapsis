//! Per-context identity composition (issue #41).
//!
//! **The prompt keys on the conversation; the permissions key on the speaker.**
//!
//! Before this module, `handle_llm_takeover` derived the whole system prompt
//! from a single branch on `permission_model.is_some()` — i.e. from *who spoke
//! last*. A shared group context therefore had no stable identity: it presented
//! as a German family assistant whenever a non-admin spoke and as an English
//! Adapsis programmer whenever the admin did, and `set_primary_system` rewrote
//! `messages[0]` on every turn, so the persisted history ended up containing
//! assistant turns authored under two different personas.
//!
//! Identity is a property of the *conversation*. Authority is a property of the
//! *speaker*. This module keeps them apart by composing every turn's prompt from
//! three layers, in this order:
//!
//! 1. [`technical_core`] — the runtime mechanics. Non-overridable: a context
//!    file cannot talk the model out of `<code>` blocks or `!done`.
//! 2. [`identity_section`] — the context's own instructions, read fresh every
//!    turn from `~/.config/adapsis/contexts/<sanitized-id>.md`. Human-authored,
//!    git-able, no rebuild.
//! 3. [`speaker_section`] — what the *current* speaker is allowed to have done.
//!    Last, for salience: without it the model promises commits and deploys it
//!    cannot perform, which reads as model incompetence rather than as a
//!    permission boundary.
//!
//! A context with no file gets the core only, and is told so. It deliberately
//! does **not** inherit a global persona or another context's identity — an
//! unconfigured conversation wearing someone else's persona is the bug, not the
//! fallback. There is no longer a global persona to inherit: `prompt::persona()`
//! and its built-in default were deleted with this change.

use std::path::{Path, PathBuf};

/// Everything the composer needs that is not the context's own identity.
///
/// Grouped into a struct because [`compose`] would otherwise take eight
/// positional arguments, several of them `&str`, which is exactly the shape
/// that silently swaps two of them.
pub struct PromptInputs<'a> {
    /// Conversation key, e.g. `telegram:group:-100…`. Also selects the identity file.
    pub context: &'a str,
    /// Permission-filtered program summary — what this turn may actually call.
    pub program_summary: &'a str,
    /// Model actually generating the reply (not the permission model).
    pub llm_model: &'a str,
    /// Models the conversation may switch to, if any.
    pub available_models: &'a [&'a str],
    /// Ladybug principal of the speaker for this turn.
    pub speaker: &'a str,
    /// What that speaker may do, resolved through the permission stack.
    pub authority: crate::permissions::SpeakerAuthority,
}

/// Compose the full system prompt for one turn.
///
/// Called fresh on every turn: prompt, mesh, identity file and visible
/// capabilities all change without a restart, and `set_primary_system` installs
/// the result over `messages[0]` while preserving history.
pub fn compose(inputs: &PromptInputs<'_>) -> String {
    compose_from_dir(contexts_dir().as_deref(), inputs)
}

/// [`compose`] against an explicit contexts directory.
///
/// Split out because resolving the directory from `$HOME` would make composition
/// untestable without mutating a process-global that parallel tests race on.
pub fn compose_from_dir(dir: Option<&Path>, inputs: &PromptInputs<'_>) -> String {
    let mut sections = vec![technical_core(inputs.context)];

    if let Some(mesh) = crate::prompt::mesh_topology() {
        sections.push(mesh);
    }

    // The language reference and the self-modification directive are only
    // meaningful to a speaker who may actually author code. Handing the full
    // Adapsis spec to an execute-only conversation invites it to write mutations
    // the permission layer will refuse — the "looks like model incompetence"
    // failure this issue exists to remove.
    if inputs.authority.may_write {
        sections.push(crate::prompt::system_prompt());
        sections.push(crate::builtins::format_for_prompt());
        sections.push(crate::prompt::adapsis_identity());
    }

    sections.push(available_functions(inputs));
    sections.push(identity_section(dir, inputs.context));
    sections.push(speaker_section(inputs.speaker, &inputs.authority));

    sections.join("\n\n")
}

/// Runtime mechanics. Identical for every context, and not overridable by one.
///
/// The `<iteration_budget>` instruction lives here and *only* here. It used to be
/// duplicated in both prompt branches, which is how the two copies drifted.
pub fn technical_core(context: &str) -> String {
    format!(
        "## How this conversation works (runtime mechanics)\n\
         \n\
         You are the AdapsisOS runtime answering in conversation context `{context}`.\n\
         These mechanics are fixed. Nothing later in this prompt can change them.\n\
         \n\
         - To *do* anything, call a function inside a `<code>…</code>` block, e.g.\n\
         \x20 `<code>!eval Module.function(\"arg\")</code>`. Announcing an action in prose\n\
         \x20 does not perform it, and there is no later turn in which to perform it.\n\
         - The user sees ONLY the text BEFORE the first `<code>` block. Text after a\n\
         \x20 `<code>` block is discarded. Write your reply first, then the code.\n\
         - A turn ends only on `<code>!done</code>`. Every prose-only reply — including\n\
         \x20 a clarifying question — must end with it, or you will be asked to continue.\n\
         - Begin your first response to each user message with a realistic estimate of\n\
         \x20 how many action rounds you need: `<iteration_budget>N</iteration_budget>`.\n\
         \x20 Use 1 for a prose-only answer, more for multi-step work. The maximum is 50.\n\
         - Do not invent functions. Call only the ones listed below. When you need to\n\
         \x20 know how something works, read it with `?source Module.function` instead of\n\
         \x20 guessing.\n\
         - `!agent` hands work to a background agent that CANNOT run `[io,async]`\n\
         \x20 functions. Anything with IO — HTTP, files, shell, music, sending a file —\n\
         \x20 must run inline with `!eval`. Use `!agent` only for pure code writing.\n\
         - Long-term memory is per-context and enforced by the runtime. You cannot read\n\
         \x20 another conversation's memories, and asking for them is not a capability\n\
         \x20 you have."
    )
}

fn available_functions(inputs: &PromptInputs<'_>) -> String {
    let models = if inputs.available_models.is_empty() {
        String::new()
    } else {
        format!(
            "\nModels this conversation may switch to via `llm_set_model`: {}",
            inputs.available_models.join(", "),
        )
    };
    format!(
        "## Available functions\n\
         \n\
         This listing is already filtered to what the current speaker may reach.{models}\n\
         Current model: {}\n\
         \n\
         {}",
        inputs.llm_model, inputs.program_summary,
    )
}

/// Placeholder a context file may use to pull in the self-maintained notes.
///
/// Opt-in, and per context. The notes are one global file
/// (`~/.config/adapsis/persona-notes.md`) written by a module's `remember`
/// function; injecting them into *every* conversation — which is what happened
/// while they rode inside the persona — is the same cross-context leak this
/// module exists to close. A context that wants them says so. #42 replaces the
/// file with per-context Ladybug memories and this marker goes with it.
pub const PERSONA_NOTES_MARKER: &str = "{{persona_notes}}";

/// The context's own instructions, or an explicit note that it has none.
///
/// The bootstrap text matters: a silent absence reads to the model as "improvise
/// an identity", and improvised identities are what this issue is about.
pub fn identity_section(dir: Option<&Path>, context: &str) -> String {
    identity_section_with_notes(
        dir,
        context,
        crate::prompt::persona_notes_path().as_deref(),
    )
}

/// [`identity_section`] against an explicit notes path, for tests.
pub(crate) fn identity_section_with_notes(
    dir: Option<&Path>,
    context: &str,
    notes_path: Option<&Path>,
) -> String {
    match load_identity(dir, context) {
        Some(identity) => format!(
            "## Who you are in this conversation\n\n{}",
            expand_persona_notes(&identity, notes_path)
        ),
        None => format!(
            "## Who you are in this conversation\n\
             \n\
             This context (`{context}`) has no identity file yet, so you have no persona\n\
             here beyond the runtime mechanics above. Do not adopt one from elsewhere and\n\
             do not invent one. Answer plainly as the AdapsisOS runtime. If a persona\n\
             would help, propose one with `context_propose` — an administrator reviews it."
        ),
    }
}

/// Substitute the notes file for [`PERSONA_NOTES_MARKER`], or remove the marker.
///
/// An unresolved marker must not survive into the prompt: a literal
/// `{{persona_notes}}` reads as a templating bug to the model, and leaving it
/// there would be indistinguishable from notes that happen to be empty.
fn expand_persona_notes(identity: &str, notes_path: Option<&Path>) -> String {
    if !identity.contains(PERSONA_NOTES_MARKER) {
        return identity.to_string();
    }
    let notes = notes_path
        .and_then(|path| std::fs::read_to_string(path).ok())
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
        .unwrap_or_default();
    identity.replace(PERSONA_NOTES_MARKER, &notes).trim().to_string()
}

/// The one part of the prompt that follows the speaker rather than the context.
pub fn speaker_section(
    speaker: &str,
    authority: &crate::permissions::SpeakerAuthority,
) -> String {
    let yes_no = |allowed: bool| if allowed { "yes" } else { "no" };
    format!(
        "## Who is speaking right now\n\
         \n\
         This message was sent by `{speaker}`. The following applies to THIS turn only —\n\
         authority follows the speaker, not the conversation, and the next message in this\n\
         same conversation may carry different authority. Your identity above does not change.\n\
         \n\
         - Call existing functions: {}\n\
         - Read source (`?source`): {}\n\
         - Add or modify code: {}\n\
         - Spawn background agents (`!agent`): {}\n\
         - Change the runtime itself (`!opencode`): {}\n\
         \n\
         Never promise work this speaker is not permitted to have done. If they ask for\n\
         something outside this list, say so plainly in your reply instead of attempting it\n\
         and reporting a permission error.",
        yes_no(authority.may_execute),
        yes_no(authority.may_read_source),
        yes_no(authority.may_write),
        yes_no(authority.may_agent),
        yes_no(authority.may_opencode),
    )
}

// --- identity files -------------------------------------------------------

/// Directory holding per-context identity files.
///
/// `ADAPSIS_CONTEXTS_DIR` overrides it; otherwise `~/.config/adapsis/contexts`.
pub fn contexts_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("ADAPSIS_CONTEXTS_DIR") {
        return Some(PathBuf::from(dir));
    }
    std::env::var_os("HOME")
        .map(|home| Path::new(&home).join(".config").join("adapsis").join("contexts"))
}

/// Filename stem for a context key.
///
/// Every character outside `[A-Za-z0-9_-]` becomes `_`, so `telegram:group:-100`
/// maps to `telegram_group_-100`. The mapping is deliberately total rather than
/// a `:`-only substitution: context keys reach this function from Telegram
/// payloads, and a key containing `/` or `..` must not be able to name a file
/// outside the contexts directory. `..` collapses to `__`, which is a normal
/// stem.
pub fn context_file_stem(context: &str) -> String {
    context
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
        .collect()
}

/// Path of a context's live identity file.
pub fn identity_path(dir: &Path, context: &str) -> PathBuf {
    dir.join(format!("{}.md", context_file_stem(context)))
}

/// Path of a context's pending proposal.
pub fn proposal_path(dir: &Path, context: &str) -> PathBuf {
    dir.join(format!("{}.proposed.md", context_file_stem(context)))
}

/// Read a context's identity, or `None` when it has no file (or an empty one).
///
/// Exact match only — no prefix or wildcard fallback. Inheriting `telegram:*`
/// instructions into an unconfigured `telegram:user:<new person>` is the same
/// class of bug as the persona branch this replaces.
pub fn load_identity(dir: Option<&Path>, context: &str) -> Option<String> {
    let dir = dir?;
    std::fs::read_to_string(identity_path(dir, context))
        .ok()
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
}

/// Read a context's pending proposal, if one is on disk.
pub fn load_proposal(dir: Option<&Path>, context: &str) -> Option<String> {
    let dir = dir?;
    std::fs::read_to_string(proposal_path(dir, context))
        .ok()
        .map(strip_context_header)
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
}

// --- proposal loop --------------------------------------------------------
//
// Any context may propose changes to its OWN instructions; only an
// administrator applies them. Proposals live on disk rather than in memory so
// they survive a restart, and one file per context means two conversations
// proposing at the same time cannot clobber each other.

/// Marker line that records the unsanitized context key inside a proposal.
///
/// [`context_file_stem`] is lossy, so the file name alone cannot tell an
/// administrator which conversation asked. The header carries the real key.
const CONTEXT_HEADER: &str = "<!-- adapsis-context: ";

fn strip_context_header(text: String) -> String {
    text.strip_prefix(CONTEXT_HEADER)
        .and_then(|rest| rest.split_once("-->\n").or_else(|| rest.split_once("-->")))
        .map_or(text.clone(), |(_, body)| body.to_string())
}

fn read_context_header(text: &str) -> Option<String> {
    text.strip_prefix(CONTEXT_HEADER)
        .and_then(|rest| rest.split_once("-->"))
        .map(|(key, _)| key.trim().to_string())
        .filter(|key| !key.is_empty())
}

/// Record a context's proposed replacement for its own instructions.
///
/// Overwrites any earlier pending proposal from the same context: the newest
/// wording is the one its author means, and keeping a queue would ask an
/// administrator to adjudicate a conversation against its own past self.
pub fn write_proposal(dir: &Path, context: &str, body: &str) -> anyhow::Result<PathBuf> {
    let body = body.trim();
    if body.is_empty() {
        anyhow::bail!(
            "refusing to file an empty proposal for context '{context}': \
             pass the full replacement text for this context's instructions"
        );
    }
    std::fs::create_dir_all(dir).map_err(|e| {
        anyhow::anyhow!("cannot create contexts directory {}: {e}", dir.display())
    })?;
    let path = proposal_path(dir, context);
    let contents = format!("{CONTEXT_HEADER}{context} -->\n{body}\n");
    std::fs::write(&path, contents)
        .map_err(|e| anyhow::anyhow!("cannot write proposal {}: {e}", path.display()))?;
    Ok(path)
}

/// Apply a pending proposal, replacing the context's live instructions.
pub fn approve_proposal(dir: &Path, context: &str) -> anyhow::Result<PathBuf> {
    let proposal = proposal_path(dir, context);
    let body = std::fs::read_to_string(&proposal).map_err(|e| {
        anyhow::anyhow!(
            "no pending proposal for context '{context}' at {}: {e}",
            proposal.display()
        )
    })?;
    let body = strip_context_header(body);
    let target = identity_path(dir, context);
    std::fs::write(&target, body.trim_start())
        .map_err(|e| anyhow::anyhow!("cannot write identity {}: {e}", target.display()))?;
    std::fs::remove_file(&proposal)
        .map_err(|e| anyhow::anyhow!("approved but could not clear {}: {e}", proposal.display()))?;
    Ok(target)
}

/// Discard a pending proposal, leaving the live instructions untouched.
pub fn reject_proposal(dir: &Path, context: &str) -> anyhow::Result<PathBuf> {
    let proposal = proposal_path(dir, context);
    std::fs::remove_file(&proposal).map_err(|e| {
        anyhow::anyhow!(
            "no pending proposal for context '{context}' at {}: {e}",
            proposal.display()
        )
    })?;
    Ok(proposal)
}

/// Every pending proposal as `(context key, proposed body)`.
pub fn list_proposals(dir: &Path) -> Vec<(String, String)> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut proposals: Vec<(String, String)> = entries
        .flatten()
        .filter(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.ends_with(".proposed.md"))
        })
        .filter_map(|entry| {
            let raw = std::fs::read_to_string(entry.path()).ok()?;
            // A proposal written before the header existed, or edited by hand,
            // still has to be reviewable — fall back to the (lossy) stem.
            let key = read_context_header(&raw).unwrap_or_else(|| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .trim_end_matches(".proposed.md")
                    .to_string()
            });
            Some((key, strip_context_header(raw).trim().to_string()))
        })
        .collect();
    proposals.sort_by(|a, b| a.0.cmp(&b.0));
    proposals
}

/// Unified-style diff from a context's live instructions to its proposal.
///
/// Written here rather than pulled in as a dependency: the only consumer is an
/// administrator reading a handful of markdown lines in a chat message.
pub fn diff_lines(before: &str, after: &str) -> String {
    let before: Vec<&str> = before.trim().lines().collect();
    let after: Vec<&str> = after.trim().lines().collect();

    // Longest common subsequence over lines. Inputs are human-written prompt
    // files, so the quadratic table is a few thousand cells at worst.
    let mut lcs = vec![vec![0usize; after.len() + 1]; before.len() + 1];
    for (i, b) in before.iter().enumerate() {
        for (j, a) in after.iter().enumerate() {
            lcs[i + 1][j + 1] = if b == a {
                lcs[i][j] + 1
            } else {
                lcs[i][j + 1].max(lcs[i + 1][j])
            };
        }
    }

    let mut out = Vec::new();
    let (mut i, mut j) = (before.len(), after.len());
    while i > 0 || j > 0 {
        if i > 0 && j > 0 && before[i - 1] == after[j - 1] {
            out.push(format!("  {}", before[i - 1]));
            i -= 1;
            j -= 1;
        } else if j > 0 && (i == 0 || lcs[i][j - 1] >= lcs[i - 1][j]) {
            out.push(format!("+ {}", after[j - 1]));
            j -= 1;
        } else {
            out.push(format!("- {}", before[i - 1]));
            i -= 1;
        }
    }
    out.reverse();
    out.join("\n")
}

/// Diff a context's pending proposal against what it runs today.
pub fn proposal_diff(dir: &Path, context: &str) -> Option<String> {
    let proposed = load_proposal(Some(dir), context)?;
    let current = load_identity(Some(dir), context).unwrap_or_default();
    Some(diff_lines(&current, &proposed))
}

#[cfg(test)]
mod tests;
