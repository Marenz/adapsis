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

fn valid_of(text: &str) -> bool {
    match valid(&[Value::string(text)]).expect("json_valid is total on String") {
        Value::Bool(value) => value,
        other => panic!("not Bool: {other}"),
    }
}

fn nested(depth: usize) -> String {
    format!("{}{}", "[".repeat(depth), "]".repeat(depth))
}

#[test]
fn json_valid_accepts_what_the_shared_parser_accepts() {
    for text in [
        "{}",
        "[]",
        "0",
        "-0",
        "1.5",
        "1e5",
        "true",
        "null",
        r#""x""#,
        r#""""#,
        r#"{"a":[1,{"b":null}]}"#,
        " \n\t\r{} \n",
        r#"{"a":1,"a":2}"#,
        r#"{"ä":"😀 \ud83d\ude00 \u00e4"}"#,
        "123456789012345678901234567890",
        "18446744073709551615",
        "-9223372036854775809",
        "1e-400",
    ] {
        assert!(valid_of(text), "{text}");
    }
    // serde_json decrements a 128 budget per container and fails at zero, so 127
    // nested containers is the deepest accepted document.
    assert!(valid_of(&nested(127)));
}

#[test]
fn json_valid_rejects_malformed_text_without_erroring() {
    let nul = "{\0}";
    let bom = "\u{feff}{}";
    for text in [
        "",
        "   ",
        "\n\t",
        "{",
        "[1,",
        r#"{"a":1,}"#,
        "[1,]",
        "{'a':1}",
        "01",
        "+1",
        "NaN",
        "Infinity",
        "-Infinity",
        ".5",
        "1.",
        "tru",
        "{} x",
        "1 2",
        "// c\n{}",
        "/* c */ {}",
        "<html>bad gateway</html>",
        r#"{"a":"unterminated"#,
        r#"{"a":"bad \"#,
        "\"raw\nnewline\"",
        r#""\ud800""#,
        r#""\u12""#,
        bom,
        nul,
        "1e400",
        "-1e400",
    ] {
        assert!(!valid_of(text), "{text:?}");
    }
    // Beyond serde_json's 128 limit the parser reports an error; it must not
    // recurse, overflow the (test thread) stack or panic.
    assert!(!valid_of(&nested(128)));
    assert!(!valid_of(&nested(129)));
    assert!(!valid_of(&"[".repeat(100_000)));
    assert!(!valid_of(&format!(
        "{}1{}",
        "{\"a\":".repeat(100_000),
        "}".repeat(100_000)
    )));
}

#[test]
fn json_valid_is_validity_not_representability_and_matches_the_shared_parse_step() {
    // Valid JSON the representation-checking primitives still reject.
    assert!(valid_of("18446744073709551615"));
    assert!(kind(&args("18446744073709551615", ""))
        .unwrap_err()
        .to_string()
        .contains("outside signed 64-bit range"));
    // For every String: json_valid == false  <=>  all three parse-step users report
    // "invalid JSON"; json_valid == true  =>  none of them fails at the parse step.
    let nul = "{\0}".to_string();
    let corpus = [
        "",
        "{",
        "[1,",
        "{} x",
        "1e400",
        "18446744073709551615",
        "{}",
        "[]",
        "null",
        "-0",
        "\u{feff}{}",
        nul.as_str(),
        "<html></html>",
        r#"{"a":[1]}"#,
        "1e-400",
    ];
    let deep = nested(129);
    for text in corpus.into_iter().chain([deep.as_str()]) {
        let parse_failed = |result: anyhow::Result<Value>| {
            result.is_err_and(|e| e.to_string().contains("invalid JSON"))
        };
        let get_failed = parse_failed(get(&args(text, "")));
        let kind_failed = parse_failed(kind(&args(text, "")));
        let len_failed = parse_failed(crate::eval::eval_builtin_string(
            "json_array_len",
            vec![Value::string(text)],
        ));
        let valid = valid_of(text);
        assert_eq!(get_failed, kind_failed, "{text:?}");
        assert_eq!(valid, !get_failed, "{text:?}");
        // json_array_len also parse-fails exactly when json_valid is false (a valid
        // non-array fails with a different, non-parse message).
        assert_eq!(valid, !len_failed, "{text:?}");
    }
}

#[test]
fn json_valid_arity_and_argument_types_error_but_strings_never_do() {
    for bad in [vec![], vec![Value::string("{}"), Value::string("")]] {
        assert_eq!(
            valid(&bad).unwrap_err().to_string(),
            "json_valid(json) expects 1 argument"
        );
    }
    for bad in [
        Value::Int(1),
        Value::Bool(true),
        Value::None,
        Value::Float(1.0),
    ] {
        assert_eq!(
            valid(&[bad]).unwrap_err().to_string(),
            "json_valid expects (String)"
        );
    }
    for text in ["", "{", "ä", "\0", "1e400", &"[".repeat(1000)] {
        assert!(valid(&[Value::string(text)]).is_ok());
    }
}
