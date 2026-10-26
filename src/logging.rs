//! Content-logging gate (#88).
//!
//! Model output, prompts, tool arguments, eval results and user messages are
//! metadata-only by default: the journal and `--log-file` carry context, model,
//! latency, token usage, end state and error class, never reasoning, completion
//! text or user data. Those can escape Ladybug ACLs, so they must be opt-in.
//!
//! Set `ADAPSIS_LOG_CONTENT=1` to restore full content logging for debugging.

use std::sync::OnceLock;

/// Whether model/user content may be written to stdout, stderr or `--log-file`.
pub fn content() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ADAPSIS_LOG_CONTENT").as_deref() == Ok("1"))
}

/// Render at the sink, not where an error originates: serde, regex, database,
/// provider and task errors may all embed arbitrary input in their Display.
/// The category is a static call-site label, never a provider's error string.
pub fn diagnostic(category: &'static str, detail: impl std::fmt::Display) {
    let detail = detail.to_string();
    eprintln!("{}", render(category, &detail));
}

pub fn render(category: &'static str, detail: &str) -> String {
    if content() {
        format!("[{category}] {detail}")
    } else {
        format!("[{category}] detail_bytes={}", detail.len())
    }
}

/// Event tags do not make their detail safe. Even 'llm-error' or 'done-rejected'
/// can contain a provider body or evaluated value. Unknown tags are data too.
pub fn event(event: &str, detail: &str) -> String {
    let category = match event {
        "iter" => "iteration",
        "done" => "done",
        "done-rejected" => "done-rejected",
        "llm-error" => "llm-error",
        "memory-compaction-error" => "memory-compaction-error",
        "user" => "user",
        "ai-text" => "ai-text",
        "code" => "code",
        "think" => "think",
        "feedback" => "feedback",
        "inject" => "inject",
        _ => "activity",
    };
    render(category, detail)
}

// Each capture runs the actual test fixture in a fresh process with an immutable
// policy. No global flag overrides or environment changes across async awaits.
#[cfg(test)]
pub mod tests {
    pub fn child() -> bool {
        std::env::var_os("ADAPSIS_LOG_TEST_CHILD").is_some()
    }

    pub fn capture(test: &str, enabled: bool) -> String {
        capture_with_env(test, enabled, &[])
    }

    pub fn capture_with_env(test: &str, enabled: bool, env: &[(&str, &str)]) -> String {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--nocapture"])
            .env("ADAPSIS_LOG_TEST_CHILD", "1")
            .env("ADAPSIS_LOG_CONTENT", if enabled { "1" } else { "0" })
            .envs(env.iter().copied())
            .output().unwrap();
        let captured = format!("{}{}", String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr));
        assert!(output.status.success(), "child failed: {captured}");
        assert!(captured.contains("1 passed"), "fixture did not run: {captured}");
        captured
    }
}
