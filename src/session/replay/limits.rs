//! Post-deserialization structural work budgets, not a hostile-file memory cap.
//! Session JSON also retains serde_json's default recursion limit. Never disable it.
use anyhow::{Result, bail};
use serde_json::Value;

pub(super) fn check(root: &Value) -> Result<()> {
    let mut pending = vec![(root, 0usize)];
    let mut nodes = 0usize;
    let mut text_bytes = 0usize;
    while let Some((value, depth)) = pending.pop() {
        nodes += 1;
        if nodes > 200_000 || depth > 96 {
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
