//! Policy gate for `shell_exec` / `exec`.
//!
//! The IO loop is the single choke point that actually spawns `sh -c`, so the
//! policy is enforced here regardless of which Adapsis module or permission
//! level requested the command. This is defense-in-depth: even if a non-admin
//! conversation manages to invoke a function that calls `shell_exec`, the
//! policy decides whether the command is allowed to run at all.
//!
//! Configured via the `ADAPSIS_SHELL_POLICY` environment variable:
//!   - unset / "unrestricted"  -> any command runs (legacy dev behavior)
//!   - "deny"                  -> all shell execution refused
//!   - "allow:git,ls,systemctl"-> only commands whose program (first token)
//!                                is in the comma-separated list may run
//!
//! For a locked-down deployment (e.g. a family member's machine), set
//! `ADAPSIS_SHELL_POLICY=deny` or an explicit allowlist.

/// How shell command execution is gated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShellPolicy {
    /// Any command may run (legacy behavior). Logs a warning at construction.
    Unrestricted,
    /// No command may run.
    Denied,
    /// Only commands whose program (first whitespace-delimited token) is in
    /// this set may run.
    Allowlist(Vec<String>),
}

impl Default for ShellPolicy {
    fn default() -> Self {
        ShellPolicy::Unrestricted
    }
}

impl ShellPolicy {
    /// Build a policy from the `ADAPSIS_SHELL_POLICY` environment variable.
    pub fn from_env() -> Self {
        match std::env::var("ADAPSIS_SHELL_POLICY") {
            Ok(v) => Self::parse(&v),
            Err(_) => ShellPolicy::Unrestricted,
        }
    }

    /// Parse a policy spec string. Unknown specs fall back to `Denied` — a
    /// typo must fail safe, never silently grant unrestricted shell access.
    pub fn parse(spec: &str) -> Self {
        let spec = spec.trim();
        let lower = spec.to_ascii_lowercase();
        if lower == "unrestricted" || lower == "any" || lower == "all" {
            return ShellPolicy::Unrestricted;
        }
        if lower == "deny" || lower == "denied" || lower == "none" || lower.is_empty() {
            return ShellPolicy::Denied;
        }
        if let Some(rest) = lower.strip_prefix("allow:") {
            let cmds: Vec<String> = rest
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if cmds.is_empty() {
                // "allow:" with nothing useful -> deny, don't fall open.
                return ShellPolicy::Denied;
            }
            return ShellPolicy::Allowlist(cmds);
        }
        // Unknown spec: fail safe.
        ShellPolicy::Denied
    }

    /// Extract the program (first token) from a shell command line.
    /// Handles a leading `VAR=val` assignment by skipping it, and an absolute
    /// path by taking the final path component (so `/usr/bin/git` -> `git`).
    fn program_of(command: &str) -> Option<String> {
        for token in command.split_whitespace() {
            // Skip leading environment assignments like FOO=bar.
            if token.contains('=') && !token.contains('/') {
                continue;
            }
            let base = token.rsplit('/').next().unwrap_or(token);
            if base.is_empty() {
                continue;
            }
            return Some(base.to_string());
        }
        None
    }

    /// Decide whether a command may run. `Ok(())` to allow, `Err(reason)` to
    /// refuse with a human-readable message.
    pub fn check(&self, command: &str) -> Result<(), String> {
        match self {
            ShellPolicy::Unrestricted => Ok(()),
            ShellPolicy::Denied => Err(
                "shell execution is disabled by policy (ADAPSIS_SHELL_POLICY=deny)".to_string(),
            ),
            ShellPolicy::Allowlist(allowed) => {
                let prog = Self::program_of(command).ok_or_else(|| {
                    "shell command is empty or has no program to run".to_string()
                })?;
                if allowed.iter().any(|a| a == &prog) {
                    Ok(())
                } else {
                    Err(format!(
                        "shell command `{prog}` is not in the allowlist ({})",
                        allowed.join(", ")
                    ))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unrestricted_allows_everything() {
        let p = ShellPolicy::Unrestricted;
        assert!(p.check("rm -rf /").is_ok());
        assert!(p.check("git status").is_ok());
    }

    #[test]
    fn denied_refuses_everything() {
        let p = ShellPolicy::Denied;
        assert!(p.check("ls").is_err());
        assert!(p.check("git status").is_err());
    }

    #[test]
    fn allowlist_permits_listed_program_only() {
        let p = ShellPolicy::Allowlist(vec!["git".into(), "ls".into()]);
        assert!(p.check("git status").is_ok());
        assert!(p.check("ls -la /tmp").is_ok());
        assert!(p.check("rm -rf /").is_err());
        assert!(p.check("curl http://evil").is_err());
    }

    #[test]
    fn allowlist_matches_absolute_path_by_basename() {
        let p = ShellPolicy::Allowlist(vec!["git".into()]);
        assert!(p.check("/usr/bin/git status").is_ok());
    }

    #[test]
    fn allowlist_skips_leading_env_assignment() {
        let p = ShellPolicy::Allowlist(vec!["git".into()]);
        assert!(p.check("GIT_PAGER=cat git log").is_ok());
        let p2 = ShellPolicy::Allowlist(vec!["cat".into()]);
        // The program is still git here, not the env var, so cat-only denies it.
        assert!(p2.check("GIT_PAGER=cat git log").is_err());
    }

    #[test]
    fn parse_known_specs() {
        assert_eq!(ShellPolicy::parse("unrestricted"), ShellPolicy::Unrestricted);
        assert_eq!(ShellPolicy::parse("deny"), ShellPolicy::Denied);
        assert_eq!(
            ShellPolicy::parse("allow:git, ls ,systemctl"),
            ShellPolicy::Allowlist(vec!["git".into(), "ls".into(), "systemctl".into()])
        );
    }

    #[test]
    fn parse_unknown_or_empty_fails_safe_to_denied() {
        assert_eq!(ShellPolicy::parse("garbage"), ShellPolicy::Denied);
        assert_eq!(ShellPolicy::parse(""), ShellPolicy::Denied);
        assert_eq!(ShellPolicy::parse("allow:"), ShellPolicy::Denied);
        assert_eq!(ShellPolicy::parse("allow: , "), ShellPolicy::Denied);
    }

    #[test]
    fn empty_command_under_allowlist_is_refused() {
        let p = ShellPolicy::Allowlist(vec!["git".into()]);
        assert!(p.check("   ").is_err());
    }
}
