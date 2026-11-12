//! Post-deserialization structural work budgets, not a hostile-file memory cap.
//! Session JSON also retains serde_json's default recursion limit. Never disable it.
use anyhow::{Result, bail};
use serde_json::Value;

/// Deepest JSON value of a typed record. MEASURED 2026-11-12: a declaration at
/// the Python submission limit (100 source levels) reaches 409 levels here
/// (nested blocks and `elif` chains: 4 JSON levels per source level, plus the
/// record's wrappers). A hand-edited managed file may carry 250 source levels
/// (`python_frontend::MAX_STORED_NESTING`), about 1,010 here. The budget leaves
/// room for 8 JSON levels per stored source level. It must never be lower than
/// what the nesting guards admit, and it stays below the session reader's cap
/// (`session::deep_read::MAX_DEPTH`, 4096). The check below is a loop over an
/// explicit work list: its own stack use does not grow with the depth.
pub(crate) const MAX_JSON_DEPTH: usize = 2048;

#[cfg(test)]
thread_local! {
    /// Deepest value this thread's checks have seen (evidence for the budget).
    pub(crate) static DEEPEST_SEEN: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(super) fn check(root: &Value) -> Result<()> {
    let mut pending = vec![(root, 0usize)];
    let mut nodes = 0usize;
    let mut text_bytes = 0usize;
    while let Some((value, depth)) = pending.pop() {
        nodes += 1;
        #[cfg(test)]
        DEEPEST_SEEN.with(|seen| seen.set(seen.get().max(depth)));
        if nodes > 200_000 || depth > MAX_JSON_DEPTH {
            bail!("typed persistence structural budget exceeded")
        }
        match value {
            Value::Array(values) => pending.extend(values.iter().map(|v| (v, depth + 1))),
            Value::Object(values) => {
                text_bytes += values.keys().map(String::len).sum::<usize>();
                pending.extend(values.values().map(|v| (v, depth + 1)));
            }
            Value::String(text) => text_bytes += text.len(),
            _ => {}
        }
        if text_bytes > 16 * 1024 * 1024 {
            bail!("typed persistence text budget exceeded")
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn check_for_tests(root: &Value) -> Result<()> {
    check(root)
}
