//! Python-frontend signature catalog for the runtime builtins a typed program
//! may call. Ground truth is the runtime implementation in `eval/`: each entry's
//! arity comes from the `expects N argument(s)` guards and its result type from
//! the concrete `Value` variant that branch returns.
//!
//! Only DIRECT builtin calls consult this catalog. User functions keep their own
//! declared signatures; IO builtins additionally require await (call_sites.rs).
//! Every name here must exist in BUILTINS/IO_BUILTINS (asserted by tests), and
//! every entry's alias set must match the registry.
use super::expressions::Ty;

pub(super) struct BuiltinSig {
    pub name: &'static str,
    pub params: Vec<Ty>,
    pub result: Ty,
}

fn s() -> Ty {
    Ty::Str
}
fn i() -> Ty {
    Ty::Int
}
fn b() -> Ty {
    Ty::Bool
}
fn f() -> Ty {
    Ty::Float
}
fn any() -> Ty {
    Ty::Unknown
}
fn list_of(inner: Ty) -> Ty {
    Ty::List(Box::new(inner))
}

fn entry(name: &'static str, params: Vec<Ty>, result: Ty) -> BuiltinSig {
    BuiltinSig {
        name,
        params,
        result,
    }
}

/// Built lazily because `Ty` owns `Box`es and cannot live in a `static`.
pub(super) fn catalog() -> &'static [BuiltinSig] {
    static CATALOG: std::sync::OnceLock<Vec<BuiltinSig>> = std::sync::OnceLock::new();
    CATALOG.get_or_init(|| {
        vec![
            // ── string ───────────────────────────────────────────────────────
            entry("len", vec![any()], i()),
            entry("error_suggest", vec![s()], s()),
            entry("to_string", vec![any()], s()),
            entry("char_at", vec![s(), i()], s()),
            entry("substring", vec![s(), i(), i()], s()),
            entry("starts_with", vec![s(), s()], b()),
            entry("ends_with", vec![s(), s()], b()),
            entry("contains", vec![s(), s()], b()),
            entry("regex_match", vec![s(), s()], b()),
            entry("regex_replace", vec![s(), s(), s()], s()),
            entry("index_of", vec![s(), s()], i()),
            entry("split", vec![s(), s()], list_of(s())),
            entry("trim", vec![s()], s()),
            entry("json_get", vec![s(), s()], s()),
            entry("json_array_len", vec![s()], i()),
            entry("json_escape", vec![s()], s()),
            entry("base64_encode", vec![s()], s()),
            entry("home_dir", vec![], s()),
            // ── list ─────────────────────────────────────────────────────────
            entry("push", vec![any(), any()], any()),
            entry("get", vec![any(), i()], any()),
            entry("join", vec![any(), s()], s()),
            entry("pop", vec![any()], any()),
            entry("remove", vec![any(), i()], any()),
            entry("insert", vec![any(), i(), any()], any()),
            entry("reverse", vec![any()], any()),
            entry("sort", vec![any()], any()),
            entry("slice", vec![any(), i(), i()], any()),
            // ── map ──────────────────────────────────────────────────────────
            entry("map_set", vec![any(), any(), any()], any()),
            entry("map_get", vec![any(), any()], any()),
            entry("map_has", vec![any(), any()], b()),
            entry("map_remove", vec![any(), any()], any()),
            entry("map_keys", vec![any()], any()),
            entry("map_values", vec![any()], any()),
            entry("map_entries", vec![any()], any()),
            // ── set ──────────────────────────────────────────────────────────
            entry("set_add", vec![any(), any()], any()),
            entry("set_remove", vec![any(), any()], any()),
            entry("set_has", vec![any(), any()], b()),
            entry("set_to_list", vec![any()], any()),
            // ── math ─────────────────────────────────────────────────────────
            entry("abs", vec![any()], any()),
            entry("sqrt", vec![any()], f()),
            entry("pow", vec![any(), any()], any()),
            entry("div", vec![any(), any()], any()),
            entry("rem", vec![any(), any()], any()),
            entry("floor", vec![any()], i()),
            entry("to_int", vec![any()], i()),
            entry("digit_value", vec![s()], i()),
            entry("is_digit_char", vec![s()], b()),
            // ── bitwise / numeric ────────────────────────────────────────────
            entry("bit_and", vec![i(), i()], i()),
            entry("bit_or", vec![i(), i()], i()),
            entry("bit_xor", vec![i(), i()], i()),
            entry("bit_not", vec![i()], i()),
            entry("shl", vec![i(), i()], i()),
            entry("shr", vec![i(), i()], i()),
            entry("left_rotate", vec![i(), i()], i()),
            entry("to_hex", vec![i()], s()),
            entry("char_code", vec![s()], i()),
            entry("from_char_code", vec![i()], s()),
            entry("u32_wrap", vec![i()], i()),
            entry("max", vec![any(), any()], any()),
            entry("min", vec![any(), any()], any()),
        ]
    })
}

/// Resolve a spelling (name OR alias) to its catalog entry. Aliases share the
/// canonical entry, mirroring the runtime's `"a" | "b" => ...` dispatch arms.
pub(super) fn lookup(name: &str) -> Option<&'static BuiltinSig> {
    let canonical = canonical_name(name)?;
    catalog().iter().find(|entry| entry.name == canonical)
}

/// Map any registry spelling to the canonical catalog name, or `None` when the
/// spelling is not a catalogued builtin (IO builtins, commands, user functions).
fn canonical_name(name: &str) -> Option<&'static str> {
    let entry = crate::builtins::BUILTINS
        .iter()
        .find(|b| b.name == name || b.aliases.contains(&name))?;
    Some(entry.name)
}
