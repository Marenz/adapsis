//! The pure builtins a restricted `run` block (guests, execute-only speakers) may call
//! (#129, DESIGN L3): every non-IO registry builtin except those that reveal something
//! about the host. Pure here means: the result depends only on the arguments, and the
//! call reads or changes nothing outside them (no clock, randomness, environment, files).

/// Non-IO builtins excluded from restricted blocks, with the reason.
pub const HOST_REVEALING: &[(&str, &str)] = &[(
    "home_dir",
    "returns the service user's home directory path (host layout, user name)",
)];

fn canonical(name: &str) -> Option<&'static str> {
    super::BUILTINS
        .iter()
        .find(|builtin| builtin.name == name || builtin.aliases.contains(&name))
        .map(|builtin| builtin.name)
}

/// True for a spelling (name or alias) a restricted block may call without any grant.
pub fn allowed_in_restricted_block(name: &str) -> bool {
    canonical(name).is_some_and(|canonical| !HOST_REVEALING.iter().any(|(excluded, _)| *excluded == canonical))
}

/// Every allowed registry spelling (names and aliases), for tests and sweeps.
#[cfg(test)]
pub fn restricted_block_builtins() -> Vec<&'static str> {
    super::BUILTINS
        .iter()
        .flat_map(|builtin| std::iter::once(builtin.name).chain(builtin.aliases.iter().copied()))
        .filter(|name| allowed_in_restricted_block(name))
        .collect()
}

/// What bounds the memory and time of ONE call of a pure builtin inside a `run` block
/// (review findings F1 and F3). Every pure builtin has exactly one entry in [`COSTS`];
/// a test fails for a builtin without one, and `eval/builtin_alloc_tests.rs` measures the
/// real allocations of every builtin against hostile arguments at the argument cap.
///
/// In every class the arguments are admitted first (each within the single-value cap,
/// together at most twice the cap) and each call is charged one step plus one step per
/// 4 KiB of arguments, so time over a whole block is bounded by the step budget and, for
/// restricted speakers, by the wall clock between calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cost {
    /// Unique-set construction compares each input with all earlier inputs.
    Quadratic,
    /// Reads fixed-size values or returns a shared reference: constant memory and time.
    Constant,
    /// Scans or copies its arguments: output at most the input (plus one element), time
    /// linear in the argument bytes (`sort`: n log n comparisons).
    Linear,
    /// Output can exceed the input: its exact size (or a bound) is reserved against the
    /// value cap BEFORE anything is allocated; time linear in input plus output.
    Reserved,
    /// Renders values to text through a writer that stops at the value cap.
    Rendered,
    /// Regular expression: for restricted speakers, pattern length, compiled size and
    /// (compiled size x text length) are limited so one call stays near a second, and
    /// the modelled work is charged to the step budget; the replacement output is
    /// bounded before every append and the match loop is stopped after a second.
    Regex,
    /// JSON: an upper bound of the parsed tree, computed in one pass over the text, is
    /// reserved before parsing; nesting is limited to 127 levels by the parser.
    Json,
}

pub const COSTS: &[(&str, Cost)] = &[
    ("concat", Cost::Rendered), ("char_at", Cost::Constant), ("substring", Cost::Linear),
    ("starts_with", Cost::Linear), ("ends_with", Cost::Linear), ("contains", Cost::Linear),
    ("index_of", Cost::Linear), ("split", Cost::Reserved), ("trim", Cost::Linear),
    ("len", Cost::Constant), ("error_suggest", Cost::Rendered),
    ("to_string", Cost::Rendered), ("to_int", Cost::Linear), ("digit_value", Cost::Constant),
    ("is_digit_char", Cost::Constant), ("char_code", Cost::Constant), ("from_char_code", Cost::Constant),
    ("to_hex", Cost::Constant), ("u32_wrap", Cost::Constant),
    ("abs", Cost::Constant), ("sqrt", Cost::Constant), ("pow", Cost::Constant), ("div", Cost::Constant),
    ("rem", Cost::Constant), ("floor", Cost::Constant), ("min", Cost::Constant), ("max", Cost::Constant),
    ("bit_and", Cost::Constant), ("bit_or", Cost::Constant), ("bit_xor", Cost::Constant), ("bit_not", Cost::Constant),
    ("shl", Cost::Constant), ("shr", Cost::Constant), ("left_rotate", Cost::Constant),
    ("list", Cost::Linear), ("push", Cost::Linear), ("get", Cost::Constant), ("join", Cost::Reserved),
    ("pop", Cost::Linear), ("remove", Cost::Linear), ("insert", Cost::Linear), ("reverse", Cost::Linear),
    ("sort", Cost::Linear), ("slice", Cost::Linear),
    ("map", Cost::Linear), ("map_set", Cost::Linear), ("map_get", Cost::Linear), ("map_has", Cost::Linear),
    ("map_remove", Cost::Linear), ("map_keys", Cost::Linear), ("map_values", Cost::Linear), ("map_entries", Cost::Reserved),
    ("set", Cost::Quadratic), ("set_add", Cost::Linear), ("set_remove", Cost::Linear), ("set_has", Cost::Linear),
    ("set_to_list", Cost::Constant),
    ("Ok", Cost::Constant), ("Err", Cost::Rendered), ("Some", Cost::Constant),
    ("regex_match", Cost::Regex), ("regex_replace", Cost::Regex), ("base64_encode", Cost::Reserved),
    ("json_get", Cost::Json), ("json_kind", Cost::Json), ("json_valid", Cost::Json),
    ("json_array_len", Cost::Json), ("json_escape", Cost::Reserved),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The audit, pinned: adding a non-IO builtin must be a decision about this list.
    #[test]
    fn the_pure_set_is_the_non_io_registry_minus_host_revealing_builtins() {
        let canonical: Vec<&str> = crate::builtins::BUILTINS
            .iter()
            .map(|builtin| builtin.name)
            .filter(|name| allowed_in_restricted_block(name))
            .collect();
        assert_eq!(
            canonical,
            [
                "concat", "char_at", "substring", "starts_with", "ends_with", "contains", "index_of", "split", "trim",
                "len", "error_suggest",
                "to_string", "to_int", "digit_value", "is_digit_char", "char_code", "from_char_code", "to_hex", "u32_wrap",
                "abs", "sqrt", "pow", "div", "rem", "floor", "min", "max",
                "bit_and", "bit_or", "bit_xor", "bit_not", "shl", "shr", "left_rotate",
                "list", "push", "get", "join", "pop", "remove", "insert", "reverse", "sort", "slice",
                "map", "map_set", "map_get", "map_has", "map_remove", "map_keys", "map_values", "map_entries",
                "set", "set_add", "set_remove", "set_has", "set_to_list",
                "Ok", "Err", "Some",
                "regex_match", "regex_replace", "base64_encode",
                "json_get", "json_kind", "json_valid", "json_array_len", "json_escape",
            ],
            "the registry's non-IO builtins changed: decide for each new one whether a restricted block may call it"
        );
        assert_eq!(crate::builtins::BUILTINS.len(), canonical.len() + HOST_REVEALING.len());
    }

    /// A new pure builtin needs a decision about what bounds one call of it.
    #[test]
    fn every_pure_builtin_has_exactly_one_cost_class() {
        let classified: Vec<&str> = COSTS.iter().map(|(name, _)| *name).collect();
        let pure: Vec<&str> = crate::builtins::BUILTINS
            .iter()
            .map(|builtin| builtin.name)
            .filter(|name| allowed_in_restricted_block(name))
            .collect();
        assert_eq!(classified, pure, "classify every pure builtin in COSTS (same order as the registry): say what bounds its output and time, and make it so in the implementation");
        let count = |cost: Cost| COSTS.iter().filter(|(_, class)| *class == cost).count();
        assert_eq!(count(Cost::Quadratic), 1);
        assert_eq!(
            (count(Cost::Constant), count(Cost::Linear), count(Cost::Reserved), count(Cost::Rendered), count(Cost::Regex), count(Cost::Json)),
            (27, 25, 5, 4, 2, 4)
        );
    }

    #[test]
    fn io_builtins_host_revealing_builtins_and_unknown_names_are_not_pure() {
        for (name, _) in HOST_REVEALING {
            assert!(crate::builtins::is_builtin(name) && !allowed_in_restricted_block(name));
        }
        for builtin in crate::builtins::IO_BUILTINS {
            for name in std::iter::once(builtin.name).chain(builtin.aliases.iter().copied()) {
                assert!(!allowed_in_restricted_block(name), "IO builtin `{name}` must never be pure");
            }
        }
        for name in ["", "Module.function", "file_read", "eval", "None", "spawn"] {
            assert!(!allowed_in_restricted_block(name), "{name}");
        }
        for name in ["str", "length", "substr", "ord", "chr", "rotl", "Ok", "Err", "Some"] {
            assert!(allowed_in_restricted_block(name), "{name}");
        }
        assert!(restricted_block_builtins().len() > 70);
    }
}
