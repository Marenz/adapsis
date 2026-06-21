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
//!
//! ## Destructive-command guard (applies under EVERY policy)
//! A family/assistant bot needs broad shell reach to actually be useful —
//! reading logs, trying fixes, installing drivers, restarting services. The
//! danger is not "running commands" but a misunderstanding causing
//! *irreversible* harm (wiping a disk, `rm -rf /`, reformatting). So instead of
//! a restrictive allowlist (which would cripple debugging), an absolute
//! denylist of catastrophic, unrecoverable operations is enforced *before* the
//! policy mode is consulted. Even `Unrestricted` refuses these. The guard is
//! intentionally narrow: it targets device-wipers and root-tree deletions, not
//! merely "risky" commands, to keep false positives near zero.

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

    /// Normalize a command for destructive-pattern matching: lowercase and
    /// collapse all runs of whitespace to single spaces so `rm   -rf  /` and
    /// `rm -rf /` look identical.
    fn normalized(command: &str) -> String {
        command.split_whitespace().collect::<Vec<_>>().join(" ").to_ascii_lowercase()
    }

    /// Detect catastrophic, irreversible operations that must never run no
    /// matter the policy mode. Returns `Some(reason)` if the command is judged
    /// destructive. Deliberately narrow — only true "cliff edge" operations,
    /// to avoid blocking legitimate debugging/maintenance.
    pub fn destructive_reason(command: &str) -> Option<String> {
        let n = Self::normalized(command);

        // Fork bomb: `:(){ :|:& };:` and common spacing variants.
        let nospace: String = n.chars().filter(|c| !c.is_whitespace()).collect();
        if nospace.contains(":(){:|:&};:") || nospace.contains(":(){:|:&}:") {
            return Some("fork bomb".to_string());
        }

        // `dd` writing to a raw block device.
        if n.starts_with("dd ") || n.contains(" dd ") || n.contains("|dd ") || n.contains("| dd ") {
            if n.contains("of=/dev/sd")
                || n.contains("of=/dev/nvme")
                || n.contains("of=/dev/vd")
                || n.contains("of=/dev/mmcblk")
                || n.contains("of=/dev/disk")
                || n.contains("of=/dev/hd")
            {
                return Some("dd writing directly to a block device".to_string());
            }
        }

        // Filesystem creation / wipe on a whole device. `mkfs` also matches the
        // `mkfs.ext4` / `mkfs.xfs` family by prefix.
        let touches_device = n.contains("/dev/sd")
            || n.contains("/dev/nvme")
            || n.contains("/dev/vd")
            || n.contains("/dev/mmcblk")
            || n.contains("/dev/disk")
            || n.contains("/dev/hd");
        if touches_device {
            for prog in ["mkfs", "wipefs", "blkdiscard", "shred"] {
                if Self::has_program_token_prefix(&n, prog) {
                    return Some(format!("{prog} on a block device (would destroy data)"));
                }
            }
        }

        // Redirecting output straight onto a raw disk device.
        if n.contains("> /dev/sd")
            || n.contains(">/dev/sd")
            || n.contains("> /dev/nvme")
            || n.contains(">/dev/nvme")
            || n.contains("> /dev/mmcblk")
            || n.contains(">/dev/mmcblk")
        {
            return Some("redirecting output onto a raw disk device".to_string());
        }

        // Recursive+forced removal of a critical system root.
        if Self::is_rm_recursive_force(&n) {
            if let Some(reason) = Self::rm_hits_critical_root(&n) {
                return Some(reason);
            }
        }

        None
    }

    /// True if `n` contains `prog` as a standalone program token (at the start,
    /// or following a shell separator / pipe), not as a substring of a path.
    fn has_program_token(n: &str, prog: &str) -> bool {
        Self::program_tokens(n).any(|base| base == prog)
    }

    /// Like `has_program_token`, but also matches `prog.suffix` families such as
    /// `mkfs.ext4`, `mkfs.xfs` for `prog == "mkfs"`.
    fn has_program_token_prefix(n: &str, prog: &str) -> bool {
        Self::program_tokens(n).any(|base| {
            base == prog || base.strip_prefix(prog).map_or(false, |r| r.starts_with('.'))
        })
    }

    /// Yield the basename of the first token of each pipe/`;`/`&`-separated
    /// segment (i.e. the program being invoked in each stage).
    fn program_tokens(n: &str) -> impl Iterator<Item = &str> {
        n.split(|c| c == '|' || c == ';' || c == '&').filter_map(|seg| {
            seg.trim()
                .split_whitespace()
                .next()
                .map(|first| first.rsplit('/').next().unwrap_or(first))
        })
    }

    /// Whether the command is an `rm` with both recursive and force semantics
    /// (combined `-rf`/`-fr` or separate `-r`/`-R`/`--recursive` + `-f`/`--force`).
    fn is_rm_recursive_force(n: &str) -> bool {
        if !Self::has_program_token(n, "rm") {
            return false;
        }
        let mut recursive = false;
        let mut force = false;
        for tok in n.split_whitespace() {
            if tok == "--recursive" {
                recursive = true;
            }
            if tok == "--force" {
                force = true;
            }
            if tok.starts_with('-') && !tok.starts_with("--") {
                if tok.contains('r') || tok.contains('R') {
                    recursive = true;
                }
                if tok.contains('f') {
                    force = true;
                }
            }
        }
        recursive && force
    }

    /// If an `rm -rf` targets `/` or a critical top-level system directory,
    /// return the refusal reason. Allows deletions safely nested deeper.
    fn rm_hits_critical_root(n: &str) -> Option<String> {
        // Critical paths that must never be recursively force-removed wholesale.
        const CRITICAL: &[&str] = &[
            "/", "/*", "/etc", "/etc/*", "/boot", "/boot/*", "/usr", "/usr/*",
            "/bin", "/sbin", "/lib", "/lib64", "/var", "/var/*", "/home",
            "/home/*", "/root", "/dev", "/proc", "/sys", "/run", "/opt",
        ];
        for tok in n.split_whitespace() {
            // Skip options.
            if tok.starts_with('-') {
                continue;
            }
            // Strip a trailing slash for comparison ("/etc/" == "/etc"), but keep
            // bare "/" intact.
            let stripped = if tok.len() > 1 {
                tok.trim_end_matches('/')
            } else {
                tok
            };
            if CRITICAL.contains(&tok) || CRITICAL.contains(&stripped) {
                return Some(format!("recursive force-remove of critical path `{tok}`"));
            }
        }
        None
    }

    /// Decide whether a command may run. `Ok(())` to allow, `Err(reason)` to
    /// refuse with a human-readable message.
    pub fn check(&self, command: &str) -> Result<(), String> {
        // Absolute backstop: catastrophic, irreversible operations are refused
        // under EVERY policy mode, including Unrestricted.
        if let Some(reason) = Self::destructive_reason(command) {
            return Err(format!(
                "refused as destructive/irreversible: {reason}. This guard \
                 applies under all shell policies."
            ));
        }
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
    fn unrestricted_allows_normal_commands() {
        let p = ShellPolicy::Unrestricted;
        assert!(p.check("git status").is_ok());
        assert!(p.check("apt-get install nvidia-driver").is_ok());
        assert!(p.check("journalctl -u NetworkManager --no-pager").is_ok());
        assert!(p.check("dmesg | tail -50").is_ok());
        assert!(p.check("rm -rf /tmp/build").is_ok());
        assert!(p.check("rm -rf /home/adapsis/.cache/foo").is_ok());
    }

    #[test]
    fn destructive_guard_overrides_unrestricted() {
        // The whole point: even unrestricted refuses the cliff edges.
        let p = ShellPolicy::Unrestricted;
        assert!(p.check("rm -rf /").is_err());
        assert!(p.check("rm -rf /*").is_err());
        assert!(p.check("rm -rf /etc").is_err());
        assert!(p.check("rm -rf /home").is_err());
        assert!(p.check("rm -fr /usr/").is_err());
        assert!(p.check("dd if=/dev/zero of=/dev/sda bs=1M").is_err());
        assert!(p.check("mkfs.ext4 /dev/sdb1").is_err());
        assert!(p.check("wipefs -a /dev/nvme0n1").is_err());
        assert!(p.check("blkdiscard /dev/sda").is_err());
        assert!(p.check("echo x > /dev/sda").is_err());
        assert!(p.check(":(){ :|:& };:").is_err());
    }

    #[test]
    fn destructive_guard_allows_legit_lookalikes() {
        let p = ShellPolicy::Unrestricted;
        // dd to a regular file is fine.
        assert!(p.check("dd if=/dev/zero of=/tmp/img.bin bs=1M count=10").is_ok());
        // reading from a device is fine.
        assert!(p.check("dd if=/dev/sda of=/tmp/backup.img bs=4M").is_ok());
        // mkfs on a loopback file path that isn't /dev — still fine here.
        assert!(p.check("mkfs.ext4 /tmp/disk.img").is_ok());
        // rm -rf of a deep path is fine.
        assert!(p.check("rm -rf /var/log/myapp/old").is_ok());
        // rm without force, or without recursive, isn't the catastrophic combo.
        assert!(p.check("rm -r /etc").is_ok());
        assert!(p.check("rm /etc/hosts").is_ok());
        // a file literally named /devsomething shouldn't trip the device check.
        assert!(p.check("cat /home/x/devnotes").is_ok());
    }

    #[test]
    fn destructive_reason_is_descriptive() {
        assert!(ShellPolicy::destructive_reason("rm -rf /")
            .unwrap()
            .contains("critical path"));
        assert!(ShellPolicy::destructive_reason("dd if=/dev/zero of=/dev/sda")
            .unwrap()
            .contains("block device"));
        assert!(ShellPolicy::destructive_reason("git status").is_none());
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
