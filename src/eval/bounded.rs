//! Builtins whose cost is not bounded by the size of their arguments (#129, review
//! findings F1 and F3): regular expressions and JSON parsing. Outside a `run` block every
//! function here behaves exactly as the plain implementation it replaced; inside one, the
//! output is bounded before it is built and, for restricted speakers, the work of a regex
//! call is bounded before it starts.
use super::{meter, Value};
use anyhow::{bail, Result};
use std::time::Instant;

/// A compiled pattern and the size class it compiled under (`None` outside the limits).
struct Compiled {
    regex: regex::Regex,
    class: Option<usize>,
}

enum Pattern {
    Ready(Compiled),
    /// Not a valid pattern (or larger than the regex crate's own default limit): the
    /// caller keeps its existing behaviour for invalid patterns.
    Invalid(regex::Error),
}

/// Compile for a text of `text_len` bytes. With regex limits (restricted speakers) the
/// pattern is compiled under the smallest power-of-two program size that admits it, and
/// refused when that size times the text length exceeds the work limit: the search time
/// of the engine's fallback is proportional to that product (see `run_limits.rs`).
fn compile(builtin: &str, pattern: &str, text_len: usize) -> Result<Pattern> {
    let Some(limits) = meter::regex_limits() else {
        return Ok(match regex::Regex::new(pattern) {
            Ok(regex) => Pattern::Ready(Compiled { regex, class: None }),
            Err(error) => Pattern::Invalid(error),
        });
    };
    if pattern.len() > limits.max_pattern {
        bail!(
            "{builtin}: a pattern of {} bytes is too long for this conversation (limit {} bytes)",
            pattern.len(),
            limits.max_pattern
        );
    }
    let allowed = (limits.work / text_len.max(1) as u64).min(limits.max_program as u64) as usize;
    let mut class = 1024usize;
    loop {
        let attempt = regex::RegexBuilder::new(pattern)
            .size_limit(class)
            .dfa_size_limit(limits.dfa_cache)
            .build();
        match attempt {
            Ok(regex) => {
                // Worst-case seconds of this call, charged to the step budget up front.
                let work = class as u64 * text_len.max(1) as u64;
                let steps = (work as u128 * limits.steps_per_second as u128 / limits.work as u128) as u64;
                meter::charge(steps)?;
                return Ok(Pattern::Ready(Compiled { regex, class: Some(class) }));
            }
            Err(regex::Error::CompiledTooBig(_)) if class < allowed => class = (class * 2).min(allowed.max(class + 1)),
            Err(regex::Error::CompiledTooBig(_)) => bail!(
                "{builtin}: this pattern is too large for a text of {text_len} bytes in this conversation \
                 (compiled pattern size times text length is limited to keep one call near a second). \
                 Use a simpler pattern (ASCII classes such as [A-Za-z0-9_] instead of \\w or \\d) or a shorter text."
            ),
            Err(error) => return Ok(Pattern::Invalid(error)),
        }
    }
}

/// Top up the step charge when a call took longer than its worst-case model (the
/// quadratic case of iterating matches), so repeating it drains the budget too.
fn settle(compiled: &Compiled, text_len: usize, started: Instant) -> Result<()> {
    let (Some(limits), Some(class)) = (meter::regex_limits(), compiled.class) else { return Ok(()) };
    let modelled = (class as u128 * text_len.max(1) as u128 * limits.steps_per_second as u128 / limits.work as u128) as u64;
    let measured = (started.elapsed().as_secs_f64() * limits.steps_per_second as f64) as u64;
    meter::charge(measured.saturating_sub(modelled))
}

/// Every regex search can select capture-sized engine state, including boolean
/// is_match and offset-only find_iter. Admit scratch and weighted work BEFORE
/// the first search, regardless of the public operation or replacement syntax.
fn admit_search(builtin: &str, compiled: &Compiled, text_len: usize) -> Result<()> {
    if let (Some(limits), Some(class)) = (meter::regex_limits(), compiled.class) {
        let groups = compiled.regex.captures_len().saturating_sub(1);
        let slots = compiled.regex.captures_len().saturating_mul(2);
        if groups > limits.capture_groups || class.saturating_mul(slots) > limits.capture_state_bytes
            || (class as u64).saturating_mul(slots as u64).saturating_mul(text_len as u64) > limits.work
        {
            bail!("{builtin}: capture tracking is too expensive for this pattern and text in this conversation; use fewer capturing groups, noncapturing groups `(?:...)`, a simpler pattern or a shorter text");
        }
        meter::charge(((class as u128 * text_len as u128 * slots.saturating_sub(1) as u128 * limits.steps_per_second as u128) / limits.work as u128) as u64)?;
    }
    Ok(())
}

pub(super) fn regex_match(args: &[Value]) -> Result<Value> {
    if args.len() != 2 {
        bail!("regex_match(pattern, text) expects 2 arguments");
    }
    let (Value::String(pattern), Value::String(text)) = (&args[0], &args[1]) else {
        bail!("regex_match expects (String, String)")
    };
    match compile("regex_match", pattern, text.len())? {
        Pattern::Ready(compiled) => {
            admit_search("regex_match", &compiled, text.len())?;
            let started = Instant::now();
            let found = compiled.regex.is_match(text);
            settle(&compiled, text.len(), started)?;
            Ok(Value::Bool(found))
        }
        Pattern::Invalid(_) => Ok(Value::Bool(false)),
    }
}

pub(super) fn regex_replace(args: &[Value]) -> Result<Value> {
    if args.len() != 3 {
        bail!("regex_replace(pattern, replacement, text) expects 3 arguments");
    }
    let (Value::String(pattern), Value::String(replacement), Value::String(text)) = (&args[0], &args[1], &args[2]) else {
        bail!("regex_replace expects (String, String, String)")
    };
    let compiled = match compile("regex_replace", pattern, text.len())? {
        Pattern::Ready(compiled) => compiled,
        Pattern::Invalid(error) => bail!("regex_replace: invalid pattern '{}': {}", pattern, error),
    };
    if !meter::active() {
        return Ok(Value::string(compiled.regex.replace_all(text, replacement.as_str()).into_owned()));
    }
    admit_search("regex_replace", &compiled, text.len())?;
    let started = Instant::now();
    // A literal replacement needs match offsets only: do not ask PikeVM to track
    // captures (its scratch grows as capture slots x active program states).
    let literal = literal_replacement(replacement);
    let replaced = match literal {
        Some(literal) => replace_literal(&compiled, &literal, text, started)?,
        None => replace_bounded(&compiled, replacement, text, started)?,
    };
    settle(&compiled, text.len(), started)?;
    Ok(Value::string(replaced))
}

/// `$$` expands to one literal dollar. Any other dollar conservatively selects
/// the capture path, preserving the regex crate's exact numbered/named/trailing
/// dollar expansion rules there instead of approximating them here.
fn literal_replacement(replacement: &str) -> Option<String> {
    let mut chars = replacement.chars();
    let mut literal = String::new();
    while let Some(ch) = chars.next() {
        if ch == '$' && chars.next() != Some('$') { return None; }
        literal.push(ch);
    }
    Some(literal)
}

fn replace_literal(compiled: &Compiled, replacement: &str, text: &str, started: Instant) -> Result<String> {
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for (index, whole) in compiled.regex.find_iter(text).enumerate() {
        meter::reserve(out.len().saturating_add(whole.start() - last).saturating_add(replacement.len()))?;
        out.push_str(&text[last..whole.start()]);
        out.push_str(replacement);
        last = whole.end();
        if index % 16 == 15 {
            meter::alive()?;
            if meter::regex_limits().is_some_and(|limits| started.elapsed() > limits.call_time) {
                bail!("regex_replace: stopped after 1 second(s) in this conversation: this pattern rescans the text for every match. Anchor it or make it more specific.");
            }
        }
    }
    meter::reserve(out.len().saturating_add(text.len() - last))?;
    out.push_str(&text[last..]);
    Ok(out)
}

/// `Regex::replace_all` with the output bounded BEFORE every append. One expansion can be
/// far larger than its match (`$0$0$0...`), so each is bounded by the replacement text
/// plus one whole match per `$`; that over-estimates references to short groups and never
/// under-estimates. The result is identical to `replace_all` whenever it is admitted.
fn replace_bounded(compiled: &Compiled, replacement: &str, text: &str, started: Instant) -> Result<String> {
    let cap = meter::value_cap().unwrap_or(usize::MAX);
    let references = replacement.bytes().filter(|byte| *byte == b'$').count();
    let call_time = meter::regex_limits().map(|limits| limits.call_time);
    let mut out = String::with_capacity(text.len().min(cap));
    let mut last = 0;
    for (index, captures) in compiled.regex.captures_iter(text).enumerate() {
        let whole = captures.get(0).expect("group 0 always participates");
        let expansion = replacement.len().saturating_add(references.saturating_mul(whole.len()));
        meter::reserve(out.len().saturating_add(whole.start() - last).saturating_add(expansion))?;
        out.push_str(&text[last..whole.start()]);
        captures.expand(replacement, &mut out);
        last = whole.end();
        if index % 16 == 15 {
            meter::alive()?;
            if call_time.is_some_and(|limit| started.elapsed() > limit) {
                bail!(
                    "regex_replace: stopped after {} second(s) in this conversation: this pattern rescans the text for every match. \
                     Anchor it or make it more specific.",
                    call_time.unwrap_or_default().as_secs()
                );
            }
        }
    }
    meter::reserve(out.len().saturating_add(text.len() - last))?;
    out.push_str(&text[last..]);
    Ok(out)
}

/// Conservative reservation for parsing `text` (retained tree AND parser scratch), from a
/// single pass over the bytes. MEASURED peak allocation per construct (release build):
/// an array element 32 bytes plus vector growth, a nested array 160, an object entry
/// about 110 to 250, an object 690 (one B-tree node even for one entry); adversarial
/// texts reach 99 times their length, ordinary JSON 3 to 10 times. Counting structural
/// bytes inside strings too only raises the bound. Three text lengths additionally
/// cover decoded string storage and the parser's growable unescape scratch. These
/// coefficients are tested against measured allocations, not a serde implementation
/// contract; the separate per-builtin allocation gate remains <=3 times the value cap.
pub(super) fn json_tree_bound(text: &str) -> usize {
    let (mut objects, mut arrays, mut commas, mut colons) = (0usize, 0usize, 0usize, 0usize);
    for byte in text.bytes() {
        match byte {
            b'{' => objects += 1,
            b'[' => arrays += 1,
            b',' => commas += 1,
            b':' => colons += 1,
            _ => {}
        }
    }
    // Escaped strings/keys also need the parser's growable unescape scratch,
    // simultaneously with the retained decoded string. Reserve three text lengths
    // (measured escaped-root peak is 3x text) in addition to structural storage.
    3 * text.len() + 64 + 700 * objects + 160 * arrays + 96 * commas + 160 * colons
}

/// Parse JSON; inside a block the tree's memory bound is reserved first.
pub(super) fn parse_json(text: &str) -> Result<serde_json::Result<serde_json::Value>> {
    if meter::active() {
        meter::reserve(json_tree_bound(text))?;
    }
    Ok(serde_json::from_str(text))
}
