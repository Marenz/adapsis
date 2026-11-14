//! A guard against accidental pushes to shared branches, not a shell sandbox.
//! It understands ordinary quoted git argv, -C/-c and explicit refspecs. Shell
//! aliases, generated commands and arbitrary programs can bypass it (DESIGN N5).
use super::error::{Error, Result};

const HINT: &str = "push was not run: use an explicit remote and a nonempty source:refs/heads/<destination>, for example `git push origin HEAD:refs/heads/adapsis/<name>`; shared base/main/master/release branches, implicit/short/bulk destinations and any plain --force/-f or +refspec are refused, even with a lease flag. Use --force-with-lease alone for an allowed explicit destination. This is a mistake guard, not a shell sandbox";

pub fn check(command: &str, base: &str) -> Result<()> {
    let commands = words(command).map_err(|_| Error::CommandRefused(HINT.into()))?;
    for argv in commands {
        let mut at = 0;
        while argv
            .get(at)
            .is_some_and(|word| word.contains('=') || matches!(word.as_str(), "command" | "exec"))
        {
            at += 1;
        }
        if !argv
            .get(at)
            .is_some_and(|word| word.rsplit('/').next() == Some("git"))
        {
            continue;
        }
        at += 1;
        while let Some(option) = argv.get(at) {
            if matches!(
                option.as_str(),
                "-C" | "-c" | "--git-dir" | "--work-tree" | "--namespace"
            ) {
                at += 2;
            } else if option.starts_with('-') && option != "--" {
                at += 1;
            } else {
                break;
            }
        }
        if argv.get(at).map(String::as_str) == Some("--") {
            at += 1;
        }
        if argv.get(at).map(String::as_str) == Some("push") {
            push(&argv[at + 1..], base).map_err(|_| Error::CommandRefused(HINT.into()))?;
        }
    }
    Ok(())
}

fn branch(name: &str) -> &str {
    if let Some(local) = name.strip_prefix("refs/heads/") {
        return local;
    }
    if let Some(remote) = name.strip_prefix("refs/remotes/") {
        return remote.split_once('/').map_or(remote, |(_, branch)| branch);
    }
    name.strip_prefix("origin/").unwrap_or(name)
}

fn protected(destination: &str, base: &str) -> bool {
    let destination = destination
        .strip_prefix("refs/heads/")
        .unwrap_or(destination);
    destination == branch(base)
        || matches!(destination, "main" | "master")
        || destination.starts_with("release/")
}

fn push(args: &[String], base: &str) -> std::result::Result<(), ()> {
    let (mut positional, mut options) = (Vec::new(), true);
    for arg in args {
        if options && arg == "--" {
            options = false;
            continue;
        }
        if options && arg.starts_with('-') {
            if matches!(
                arg.as_str(),
                "--all" | "--mirror" | "--tags" | "--follow-tags" | "--prune"
            ) {
                return Err(());
            }
            if arg == "--force-with-lease" || arg.starts_with("--force-with-lease=") {
                continue;
            }
            if arg == "--force" || (!arg.starts_with("--") && arg[1..].contains('f')) {
                // Plain force overrides Git's stale-lease check: merely seeing
                // a lease flag elsewhere must never make this combination safe.
                return Err(());
            }
            // Options without operands; unrecognised/operand-taking options need
            // a simpler explicit push rather than guessing the destination.
            if !matches!(
                arg.as_str(),
                "--atomic"
                    | "--verbose"
                    | "--quiet"
                    | "--dry-run"
                    | "--set-upstream"
                    | "--delete"
                    | "--porcelain"
                    | "--no-verify"
                    | "--force-if-includes"
                    | "-u"
                    | "-v"
                    | "-q"
                    | "-n"
                    | "-d"
            ) {
                return Err(());
            }
        } else {
            positional.push(arg.as_str());
        }
    }
    if positional.len() < 2 || positional[0].is_empty() {
        return Err(());
    }
    for spec in &positional[1..] {
        if spec.starts_with('+') || spec.contains(['*', '?', '[', ']']) {
            return Err(());
        }
        // Source-only refspecs can be redirected by remote.<name>.push config.
        // A full explicit destination overrides that mapping; never resolve or
        // guess repository push configuration ourselves.
        let (source, target) = spec.split_once(':').ok_or(())?;
        if source.is_empty()
            || !target.starts_with("refs/heads/")
            || target == "refs/heads/"
            || target.contains(':')
            || protected(target, base)
        {
            return Err(());
        }
    }
    Ok(())
}

/// Minimal shell-word lexer: quotations/backslash escapes and command
/// separators. Deliberately no expansion, alias resolution or execution.
fn words(source: &str) -> std::result::Result<Vec<Vec<String>>, ()> {
    let (mut commands, mut argv, mut word) = (Vec::new(), Vec::new(), String::new());
    let (mut quote, mut escaped, mut started, mut comment) = (None, false, false, false);
    for c in source.chars() {
        if comment {
            if c == '\n' {
                comment = false;
            } else {
                continue;
            }
        }
        if escaped {
            word.push(c);
            escaped = false;
            started = true;
            continue;
        }
        if c == '\\' && quote != Some('\'') {
            escaped = true;
            started = true;
            continue;
        }
        if let Some(q) = quote {
            if c == q {
                quote = None;
            } else {
                word.push(c);
            }
            continue;
        }
        if c == '#' && !started {
            comment = true;
            continue;
        }
        if matches!(c, '\'' | '"') {
            quote = Some(c);
            started = true;
            continue;
        }
        if c.is_whitespace() || matches!(c, ';' | '&' | '|' | '(' | ')' | '`') {
            if started {
                argv.push(std::mem::take(&mut word));
                started = false;
            }
            if c == '\n' || matches!(c, ';' | '&' | '|' | '(' | ')' | '`') {
                if !argv.is_empty() {
                    commands.push(std::mem::take(&mut argv));
                }
            }
        } else {
            word.push(c);
            started = true;
        }
    }
    if quote.is_some() || escaped {
        return Err(());
    }
    if started {
        argv.push(word);
    }
    if !argv.is_empty() {
        commands.push(argv);
    }
    Ok(commands)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn push_mistakes_are_refused_without_claiming_shell_isolation() {
        for command in [
            "git push",
            "git push origin",
            "git push origin main",
            "git -C '/path with spaces' -c core.foo=bar push origin HEAD:refs/heads/main",
            "git push origin HEAD:master",
            "git push origin release/v1",
            "git push origin custom-base",
            "git push origin +HEAD:adapsis/x",
            "git push --all origin",
            "git push --mirror origin",
            "git push origin 'refs/heads/*:refs/heads/*'",
            "git push -f origin adapsis/x",
            "git push --force origin adapsis/x",
            "git push origin HEAD",
            "true; git push origin main",
            "git push origin HEAD:refs/heads/release/foo",
            "git push origin adapsis/task",
            "git push origin HEAD:adapsis/task",
            "git push origin :refs/heads/adapsis/task",
            "git push origin HEAD:",
            "git push origin HEAD:refs/heads/",
            "git push origin HEAD:refs/tags/task",
            "git push origin ''",
            "git push '' HEAD:refs/heads/adapsis/task",
            "git push --force --force-with-lease origin HEAD:refs/heads/adapsis/task",
            "git push --force-with-lease --force origin HEAD:refs/heads/adapsis/task",
            "git push -f --force-with-lease origin HEAD:refs/heads/adapsis/task",
            "git push --force-with-lease -uf origin HEAD:refs/heads/adapsis/task",
            "git push -vf --force-with-lease origin HEAD:refs/heads/adapsis/task",
            "git push origin HEAD:refs/heads/adapsis/task HEAD:main",
            "git push origin HEAD:refs/heads/custom-base",
        ] {
            assert!(check(command, "origin/custom-base").is_err(), "{command}");
        }
        for base in [
            "origin/release/foo",
            "refs/remotes/origin/release/foo",
            "refs/heads/release/foo",
        ] {
            assert!(check("git push origin HEAD:refs/heads/release/foo", base).is_err());
        }
        for command in [
            "git status",
            "git ls-remote origin",
            "git push origin HEAD:refs/heads/adapsis/task",
            "git -C '/some path' -c 'core.foo=bar' push --force-with-lease origin HEAD:refs/heads/adapsis/task",
            "git push -u origin 'refs/heads/adapsis/task:refs/heads/adapsis/task'",
            "git push origin HEAD:refs/heads/adapsis/task HEAD:refs/heads/adapsis/other",
            "git push --force-with-lease=refs/heads/adapsis/task:abc123 origin HEAD:refs/heads/adapsis/task",
            "echo 'git push origin main'",
        ] {
            assert!(check(command, "refs/heads/main").is_ok(), "{command}");
        }
        assert!(
            check(
                "git push origin HEAD:refs/heads/custom-base",
                "refs/remotes/upstream/custom-base"
            )
            .is_err()
        );
        assert!(
            check(
                "git push origin HEAD:refs/heads/origin/task",
                "refs/heads/main"
            )
            .is_ok(),
            "origin/ inside a local branch is literal"
        );
        assert!(
            check(
                "git push origin HEAD:refs/heads/origin/task",
                "refs/heads/origin/task"
            )
            .is_err()
        );
    }
}

#[cfg(test)]
#[path = "push_guard_git_tests.rs"]
mod git_tests;
