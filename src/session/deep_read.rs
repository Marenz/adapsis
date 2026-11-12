//! The session file is read without serde_json's recursion limit.
//!
//! A stored declaration nests far deeper as JSON than as source: MEASURED
//! 2026-11-12, a function at the Python nesting limit (100 levels) is up to 408
//! JSON levels deep (nested blocks and `elif` chains: 4 JSON levels per source
//! level), and serde_json refuses to read more than 127. Such a session was
//! written and could never be read again. This file is the process's own
//! output, so its reader (and ONLY its reader) runs with the limit disabled,
//! on a dedicated thread whose stack is sized for the depth cap below. Every
//! other deserializer in the process keeps serde_json's default limit.
//!
//! Sizing (MEASURED 2026-11-12, release build): the derived `Session` reader
//! uses at most 619 bytes of stack per JSON level plus about 15 KiB fixed, so
//! `MAX_DEPTH` levels need about 2.5 MiB. The thread has 64 MiB (25 times
//! that; 250 times what a declaration at the Python nesting limit needs). The
//! depth is checked by a linear scan BEFORE deserializing, so a deeper file is
//! a clean error and never a stack overflow.
use anyhow::{anyhow, bail, Context, Result};

/// Deepest session JSON the reader accepts: 10 times the deepest declaration
/// the Python nesting guard admits (408), which leaves room for declarations
/// from the unguarded legacy `.ax` path (about 2,000 chained operators).
pub(crate) const MAX_DEPTH: usize = 4096;
const STACK_BYTES: usize = 64 * 1024 * 1024;

/// Deepest point of `json` in containers. Linear, no recursion. Malformed
/// text gets some number here and its real diagnosis from the deserializer.
pub(crate) fn json_depth(json: &str) -> usize {
    let (mut depth, mut deepest, mut in_string, mut escaped) = (0usize, 0usize, false, false);
    for byte in json.bytes() {
        if in_string {
            match byte {
                _ if escaped => escaped = false,
                b'\\' => escaped = true,
                b'"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' | b'[' => {
                depth += 1;
                deepest = deepest.max(depth);
            }
            b'}' | b']' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    deepest
}

/// Runs `work` on the reader thread and waits for it. Everything that walks
/// the freshly read session recursively (validation, index rebuild) belongs
/// inside, not on the caller's stack.
pub(super) fn on_reader_stack<T: Send>(work: impl FnOnce() -> Result<T> + Send) -> Result<T> {
    std::thread::scope(|scope| {
        std::thread::Builder::new()
            .name("session-reader".into())
            .stack_size(STACK_BYTES)
            .spawn_scoped(scope, work)
            .context("could not start the session reader thread")?
            .join()
            .map_err(|_| anyhow!("the session reader stopped unexpectedly; the session file is unchanged"))?
    })
}

/// Must be called on the reader stack (`on_reader_stack`).
pub(super) fn parse<T: serde::de::DeserializeOwned>(json: &str) -> Result<T> {
    let depth = json_depth(json);
    if depth > MAX_DEPTH {
        bail!("session file is nested {depth} levels deep; the reader accepts {MAX_DEPTH}. The file is unchanged; a declaration this deep has to be removed from it by hand");
    }
    let mut deserializer = serde_json::Deserializer::from_str(json);
    deserializer.disable_recursion_limit();
    let value = T::deserialize(&mut deserializer)
        .and_then(|value| deserializer.end().map(|()| value))
        .map_err(|error| anyhow!("invalid session encoding at line {}, column {}", error.line(), error.column()))?;
    Ok(value)
}

#[cfg(test)]
#[path = "deep_read_tests.rs"]
mod tests;
