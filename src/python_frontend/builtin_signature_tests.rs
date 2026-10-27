//! Signature-catalog audit: every catalogued builtin must exist in the runtime
//! registry (with its aliases), every runtime name the frontend can reach must be
//! catalogued or explicitly excluded, and bogus arities must be refused.
use super::builtin_signatures::{catalog, lookup};
use super::*;
use crate::{ast, parser, validator};

/// IO builtins are awaited, not called as values, and are validated by
/// `call_sites`, so they are deliberately outside this catalog.
fn is_io(name: &str) -> bool {
    crate::builtins::is_io_builtin(name)
}

/// Commands/queries are not callable expressions in the frontend.
fn is_command(name: &str) -> bool {
    name.starts_with('!') || name.starts_with('?') || name.starts_with('+')
}

#[test]
fn python_builtin_catalog_names_and_aliases_match_the_runtime_registry() {
    for entry in catalog() {
        let registry = crate::builtins::BUILTINS
            .iter()
            .find(|b| b.name == entry.name)
            .unwrap_or_else(|| panic!("catalog name {} missing from BUILTINS", entry.name));
        // Every alias of the registry entry must resolve to this same catalog row.
        for alias in registry.aliases {
            let found = lookup(alias).unwrap_or_else(|| panic!("alias {alias} has no catalog row"));
            assert_eq!(
                found.name, entry.name,
                "alias {alias} resolved to {}",
                found.name
            );
        }
    }
}

#[test]
fn python_builtin_catalog_alias_map_covers_every_registry_aliases() {
    // Special-cased spellings are handled outside the catalog by design.
    let special = ["concat", "map", "set", "list", "Ok", "Err", "Some"];
    for builtin in crate::builtins::BUILTINS {
        for spelling in std::iter::once(builtin.name).chain(builtin.aliases.iter().copied()) {
            if special.contains(&spelling) {
                continue;
            }
            let resolved = lookup(spelling);
            assert!(
                resolved.is_some(),
                "BUILTINS spelling `{spelling}` (alias of {}) is unreachable in the catalog",
                builtin.name
            );
        }
    }
}

/// Reachability audit: a pure builtin the frontend can dispatch must either be
/// catalogued or be a constructor/IO/command. Anything else would silently fall
/// through to `Ty::Unknown`.
#[test]
fn python_builtin_catalog_is_complete_for_reachable_pure_builtins() {
    // Handled outside the catalog: type-ish constructors and the string helpers
    // whose typing is special (concat is variadic, list is a literal form).
    // `concat` is variadic, `list`/`map`/`set` are collection literals, and
    // Ok/Err/Some are constructors; all are handled outside the catalog.
    let special = ["Ok", "Err", "Some", "list", "concat", "map", "set"];
    let mut missing = Vec::new();
    for builtin in crate::builtins::BUILTINS {
        for spelling in std::iter::once(builtin.name).chain(builtin.aliases.iter().copied()) {
            if is_io(spelling) || is_command(spelling) || special.contains(&spelling) {
                continue;
            }
            if lookup(spelling).is_none() {
                missing.push(spelling);
            }
        }
    }
    assert!(
        missing.is_empty(),
        "uncatalogued reachable pure builtins: {missing:?}"
    );
}

fn program(source: &str, module: &str) -> ast::Program {
    let lowered = lower_module(source, module).unwrap_or_else(|e| panic!("{e}"));
    let mut program = ast::Program::default();
    for op in lowered.operations {
        validator::apply_and_validate(&mut program, &op).unwrap();
    }
    program
}

fn reject(source: &str, fragment: &str) {
    let error = lower_module(source, "SigProbe").unwrap_err();
    assert!(
        error.message.contains(fragment),
        "expected {fragment:?}, got {error}"
    );
}

#[test]
fn python_builtin_catalog_rejects_wrong_arity_and_types() {
    // Wrong arity for several families.
    reject(
        "def f(s: str) -> int:\n    return len(s, s)\n",
        "len expects one argument",
    );
    reject(
        "def f(s: str) -> str:\n    return substring(s, 1)\n",
        "expects exactly 3 argument",
    );
    reject(
        "def f(a: int) -> int:\n    return bit_and(a)\n",
        "expects exactly 2 argument",
    );
    reject(
        "def f(m: Map[str, int]) -> list[str]:\n    return map_keys()\n",
        "expects exactly 1 argument",
    );
    // Wrong element type where the catalog knows one.
    reject(
        "def f(n: int) -> int:\n    return substring(n, 0, 1)\n",
        "argument expects Str, got Int",
    );
    reject(
        "def f(s: str) -> int:\n    return contains(s, 1)\n",
        "argument expects Str, got Int",
    );
    reject(
        "def f(n: int) -> int:\n    return is_digit_char(n)\n",
        "argument expects Str, got Int",
    );
    // `len` keeps the known-collection requirement even though its catalog param
    // is unconstrained.
    reject(
        "def f(x: int) -> int:\n    return len(x)\n",
        "statically known list, str, Map, or Set",
    );
}

#[test]
fn python_builtin_catalog_returns_accurate_result_types() {
    // Result types must flow so downstream typed operations are checked rather
    // than silently Unknown.
    let source = "def f(n: int) -> int:\n    return substring(to_string(n), 0, 1) + 1\n";
    let error = lower_module(source, "SigProbe").unwrap_err();
    assert!(
        error.message.contains("matching known int/float"),
        "string result must not be silently Unknown: {error}"
    );
    // split yields list[str], so `len` accepts it and iteration yields str.
    let ok = "def f(s: str) -> int:\n    parts: list[str] = split(s, ',')\n    return len(parts)\n";
    let program = program(ok, "SigProbe");
    assert!(program.get_function("SigProbe.f").is_some());
    // A string-returning builtin cannot satisfy an int return.
    reject(
        "def f(s: str) -> int:\n    return trim(s)\n",
        "type mismatch",
    );
    // A bool-returning builtin cannot satisfy an int return either.
    reject(
        "def f(s: str) -> int:\n    return contains(s, s)\n",
        "type mismatch",
    );
}

#[test]
fn python_builtin_catalog_reaches_every_alias_through_lowering() {
    // Each alias must type-check identically to its canonical spelling.
    // `str` is intentionally absent: inside Python source it is a coercion, so
    // the frontend rejects it instead of rebinding to the Adapsis alias.
    for (alias, canonical) in [
        ("length", "len"),
        ("substr", "substring"),
        ("parse_int", "to_int"),
        ("ord", "char_code"),
        ("chr", "from_char_code"),
        ("bit_shl", "shl"),
        ("rotl", "left_rotate"),
    ] {
        let alias_src = format!("def f(x: int) -> int:\n    return 1\n");
        let _ = alias_src;
        let a = lower_module(
            &format!("def f(v: int) -> int:\n    return {alias}(v)\n"),
            "AliasProbe",
        );
        let c = lower_module(
            &format!("def f(v: int) -> int:\n    return {canonical}(v)\n"),
            "AliasProbe",
        );
        assert_eq!(a.is_ok(), c.is_ok(), "alias {alias} vs {canonical}");
        if let (Some(ae), Some(ce)) = (a.as_ref().err(), c.as_ref().err()) {
            // Same diagnostic except for the spelling the caller typed.
            let normalized = ae.message.replace(alias, canonical);
            assert_eq!(
                normalized, ce.message,
                "alias {alias} must behave like {canonical}"
            );
        }
    }
}
