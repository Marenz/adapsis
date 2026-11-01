use super::{Builtin, BuiltinCategory};

pub(super) const GET: Builtin = Builtin {
    name: "json_get", aliases: &[], category: BuiltinCategory::String,
    short: "extract value from JSON string by key path",
    long: "Extracts a value using dot-separated keys/array indices. Missing lookup returns empty String; invalid JSON fails. JSON strings/bools/numbers become runtime scalar values; null becomes String null, objects/arrays become serialized Strings. Use json_kind first when presence or JSON type matters.",
};

pub(super) const KIND: Builtin = Builtin {
    name: "json_kind", aliases: &[], category: BuiltinCategory::String,
    short: "pure JSON presence/type query: json_kind(json:String, path:String) -> String",
    long: "Returns exactly missing/null/bool/int/float/string/array/object without IO. Shares json_get path rules: empty/dot components ignored, usize-like components index arrays (including +0/00); other components are object keys, with no bracket/escape syntax. Wrong container/out-of-bounds is missing. int requires represented signed-i64; represented u64 outside i64 errors. Finite serde floating representation is float (including 1.0 and huge integer lexemes rounded by serde); parser overflow and malformed JSON error. This is not lexical number precision. Non-String arguments or wrong arity error.",
};

pub(super) const VALID: Builtin = Builtin {
    name: "json_valid", aliases: &[], category: BuiltinCategory::String,
    short: "pure JSON parse check: json_valid(json:String) -> Bool",
    long: "Returns true iff the shared JSON parse step used by json_get/json_kind/json_array_len succeeds; it is not strict RFC 8259. Nesting deeper than 127 containers (serde_json's recursion limit of 128 including the root budget) and non-finite numbers like 1e400 are false. Never errors for any String: malformed, empty, truncated, BOM-prefixed or trailing-garbage text returns false. Validity is not representability: a u64 above i64::MAX is valid here but json_kind still rejects it. Non-String arguments or wrong arity error. No IO; do not await.",
};

pub(super) const ARRAY_LEN: Builtin = Builtin {
    name: "json_array_len", aliases: &[], category: BuiltinCategory::String,
    short: "get length of JSON array",
    long: "Parses a JSON String that must be a top-level array and returns its length as Int. Invalid JSON or non-array input fails.",
};

pub(super) const ESCAPE: Builtin = Builtin {
    name: "json_escape", aliases: &[], category: BuiltinCategory::String,
    short: "escape string for use in JSON value",
    long: "Escapes backslashes, quotes, and common control characters for safe insertion into JSON string values. It returns only the escaped contents, not surrounding quotes.",
};
