//! Mock/stub pattern matching, extracted from the coroutine dispatcher.

/// Match a mock/stub `pattern` against an argument string (issue #7).
///
/// Backward-compatible matching modes, chosen by the pattern's own syntax so
/// existing substring mocks keep working:
///   - **Anchored / exact**: a pattern starting with `^` and/or ending with `$`
///     anchors that side. `^foo$` = exact equality, `^foo` = prefix, `foo$` =
///     suffix. (A literal `^`/`$` can still be matched via glob, below.)
///   - **Glob**: a pattern containing `*` or `?` (and not anchored) is treated
///     as a glob — `*` matches any run (including empty), `?` matches exactly
///     one char. The glob must match the *whole* argument.
///   - **Substring** (default, legacy): plain `contains` check.
pub fn pattern_matches(pattern: &str, arg: &str) -> bool {
    let anchored_start = pattern.starts_with('^');
    let anchored_end = pattern.ends_with('$') && !pattern.ends_with("\\$");
    if anchored_start || anchored_end {
        let inner = &pattern[anchored_start as usize..pattern.len() - anchored_end as usize];
        return match (anchored_start, anchored_end) {
            (true, true) => glob_match(inner, arg),
            (true, false) => glob_match(&format!("{inner}*"), arg),
            (false, true) => glob_match(&format!("*{inner}"), arg),
            (false, false) => unreachable!(),
        };
    }
    if pattern.contains('*') || pattern.contains('?') {
        return glob_match(pattern, arg);
    }
    arg.contains(pattern)
}

/// Whole-string glob match: `*` = any run (incl. empty), `?` = exactly one char.
/// All other chars match literally. Linear-time backtracking (patterns are tiny).
fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star, mut star_ti): (Option<usize>, usize) = (None, 0);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            star_ti = ti;
            pi += 1;
        } else if let Some(sp) = star {
            pi = sp + 1;
            star_ti += 1;
            ti = star_ti;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}
