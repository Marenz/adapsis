use super::*;

fn args(text: &str, path: &str) -> Vec<Value> {
    vec![Value::string(text), Value::string(path)]
}
fn label(text: &str, path: &str) -> String {
    match kind(&args(text, path)).unwrap() {
        Value::String(value) => value.as_ref().clone(),
        other => panic!("not String: {other}"),
    }
}

#[test]
fn json_kind_presence_types_and_root_collision_pairs() {
    for (text, expected) in [
        ("null", "null"),
        ("true", "bool"),
        ("false", "bool"),
        ("1", "int"),
        ("1.0", "float"),
        ("-0.0", "float"),
        (r#""null""#, "string"),
        (r#""{}""#, "string"),
        (r#""[]""#, "string"),
        (r#""""#, "string"),
        ("{}", "object"),
        ("[]", "array"),
    ] {
        assert_eq!(label(text, ""), expected);
        assert_eq!(label(text, "..."), expected);
    }
    assert_eq!(label("{}", "x"), "missing");
    assert_eq!(label(r#"{"x":""}"#, "x"), "string");
    assert_eq!(label(r#"{"x":null}"#, "x"), "null");
    assert_eq!(label(r#"{"x":"null"}"#, "x"), "string");
    assert!(get(&args("{}", "x"))
        .unwrap()
        .matches(&get(&args(r#"{"x":""}"#, "x")).unwrap()));
    assert!(get(&args(r#"{"x":null}"#, "x"))
        .unwrap()
        .matches(&get(&args(r#"{"x":"null"}"#, "x")).unwrap()));
}

#[test]
fn json_kind_and_get_share_legacy_path_resolution() {
    for path in ["items.0", ".items..0.", "items.+0", "items.00"] {
        assert_eq!(label(r#"{"items":[true]}"#, path), "bool");
        assert!(matches!(
            get(&args(r#"{"items":[true]}"#, path)).unwrap(),
            Value::Bool(true)
        ));
    }
    for (text, path) in [
        ("[]", "0"),
        ("null", "x"),
        (r#"{"a":null}"#, "a.b"),
        ("[true]", "[0]"),
        ("[true]", "-1"),
        (r#"{"a.b":true}"#, "a.b"),
        (r#"{"0":true,"00":true,"+0":true}"#, "0"),
        (r#"{"+0":true}"#, "+0"),
    ] {
        assert_eq!(label(text, path), "missing");
        assert!(matches!(get(&args(text,path)).unwrap(),Value::String(ref s) if s.is_empty()));
    }
    for key in [
        "-1",
        "[0]",
        " 0",
        "999999999999999999999999999999999999999999",
    ] {
        let text = serde_json::json!({key:true}).to_string();
        assert_eq!(label(&text, key), "bool");
        assert!(matches!(get(&args(&text, key)).unwrap(), Value::Bool(true)));
    }
}

#[test]
fn json_kind_number_representation_limits_are_explicit() {
    for text in ["9223372036854775807", "-9223372036854775808", "0"] {
        assert_eq!(label(text, ""), "int");
    }
    for text in ["9223372036854775808", "18446744073709551615"] {
        assert!(kind(&args(text, ""))
            .unwrap_err()
            .to_string()
            .contains("integer outside signed 64-bit range"));
        assert!(
            matches!(get(&args(text, "")).unwrap(), Value::Float(_)),
            "legacy json_get still converts these"
        );
    }
    // serde rounds these integer lexemes into floating representation. Never
    // claim json_kind recovers lexical precision that the parser discarded.
    for text in [
        "-9223372036854775809",
        "18446744073709551616",
        "1e0",
        "1.0",
        "-0.0",
    ] {
        assert_eq!(label(text, ""), "float");
    }
    for text in ["1e400", "-1e400"] {
        assert!(kind(&args(text, ""))
            .unwrap_err()
            .to_string()
            .starts_with("json_kind: invalid JSON:"));
        assert!(get(&args(text, ""))
            .unwrap_err()
            .to_string()
            .starts_with("json_get: invalid JSON:"));
    }
}

#[test]
fn json_kind_invalid_arguments_and_json_get_regressions() {
    for bad in [
        vec![],
        vec![Value::string("{}")],
        vec![Value::string("{}"), Value::string("x"), Value::Int(1)],
    ] {
        assert!(kind(&bad)
            .unwrap_err()
            .to_string()
            .contains("expects 2 arguments"));
        assert_eq!(
            get(&bad).unwrap_err().to_string(),
            "json_get(json, key_path) expects 2 arguments"
        );
    }
    for bad in [
        vec![Value::Int(1), Value::string("")],
        vec![Value::string("{}"), Value::Int(0)],
    ] {
        assert_eq!(
            kind(&bad).unwrap_err().to_string(),
            "json_kind expects (String, String)"
        );
        assert_eq!(
            get(&bad).unwrap_err().to_string(),
            "json_get expects (String, String)"
        );
    }
    assert!(kind(&args("not JSON", ""))
        .unwrap_err()
        .to_string()
        .starts_with("json_kind: invalid JSON:"));
    for (text, expected) in [
        (r#""hello""#, "hello"),
        ("null", "null"),
        ("[1,true]", "[1,true]"),
        (r#"{"x":1}"#, r#"{"x":1}"#),
    ] {
        assert!(
            matches!(get(&args(text,"")).unwrap(),Value::String(ref s) if s.as_str()==expected)
        );
    }
    assert!(matches!(get(&args("123", "")).unwrap(), Value::Int(123)));
    assert!(matches!(get(&args("1.25","")).unwrap(),Value::Float(n) if n==1.25));
    assert!(matches!(
        get(&args("false", "")).unwrap(),
        Value::Bool(false)
    ));
}
