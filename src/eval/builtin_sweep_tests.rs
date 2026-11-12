//! Every pure builtin against hostile arguments (#129): the author of a restricted `run`
//! block is untrusted, so a pure builtin must answer any input with a value or an error,
//! never a panic. The sweep calls each registry spelling with every combination of a
//! pool of awkward values, without a meter (the path module code of the live bot takes).
use super::*;

fn pool() -> Vec<Value> {
    let text = |value: &str| Value::string(value);
    let mut fields = HashMap::new();
    fields.insert("x".to_string(), Value::Int(1));
    vec![
        Value::Int(0),
        Value::Int(1),
        Value::Int(-1),
        Value::Int(2),
        Value::Int(64),
        Value::Int(i64::MIN),
        Value::Int(i64::MAX),
        Value::Int(1 << 40),
        Value::Float(0.5),
        Value::Float(-0.0),
        Value::Float(f64::NAN),
        Value::Float(f64::INFINITY),
        Value::Float(-1e308),
        Value::Bool(true),
        text(""),
        text("a"),
        text("héllo wörld ✓ 日本"),
        text("-9223372036854775808"),
        text("99999999999999999999"),
        text("(unclosed[a-"),
        text("{\"a\":[1,{\"b\":null}],\"n\":18446744073709551615}"),
        text(".0.a..b"),
        Value::list(vec![]),
        Value::list(vec![Value::Int(3), Value::Int(1), Value::Int(2)]),
        Value::list(vec![Value::Float(f64::NAN), Value::Float(1.0), text("b"), Value::Int(1)]),
        Value::list(vec![Value::list(vec![text("k"), Value::Int(1)]), Value::None]),
        Value::Map(Arc::new(vec![(text("k"), Value::Int(1)), (Value::Float(f64::NAN), Value::None)])),
        Value::Set(Arc::new(vec![Value::Int(1), text("a")])),
        Value::strct("Point", fields),
        Value::Ok(Box::new(Value::Int(1))),
        Value::Err("boom".into()),
        Value::None,
    ]
}

fn spellings() -> Vec<&'static str> {
    let names = crate::builtins::pure::restricted_block_builtins();
    assert!(names.len() > 70, "{names:?}");
    names
}

fn call(name: &str, args: Vec<Value>) -> std::thread::Result<Result<Value>> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let program = ast::Program::default();
        let mut env = Env::new_with_shared_interner(&program.shared_interner);
        eval_builtin_or_user(&program, name, args, &mut env)
    }))
}

fn describe(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|text| text.to_string()))
        .unwrap_or_else(|| "non-text panic payload".into())
}

#[test]
fn pure_builtins_answer_hostile_arguments_with_a_value_or_an_error_never_a_panic() {
    let pool = pool();
    let mut panics: std::collections::BTreeMap<&str, (usize, String)> = Default::default();
    let mut calls = 0usize;
    let mut record = |name: &'static str, args: Vec<Value>| {
        calls += 1;
        let shown = format!("{name}({})", args.iter().map(|arg| arg.to_string()).collect::<Vec<_>>().join(", "));
        if let Err(payload) = call(name, args) {
            let entry = panics.entry(name).or_insert((0, format!("{shown}: {}", describe(payload))));
            entry.0 += 1;
        }
    };
    for name in spellings() {
        record(name, vec![]);
        for a in &pool {
            record(name, vec![a.clone()]);
            for b in &pool {
                record(name, vec![a.clone(), b.clone()]);
                for c in &pool {
                    record(name, vec![a.clone(), b.clone(), c.clone()]);
                }
            }
        }
        // Arity four and five on a diagonal: every value in every position at least once.
        for (index, a) in pool.iter().enumerate() {
            let at = |offset: usize| pool[(index * 7 + offset * 11) % pool.len()].clone();
            record(name, vec![a.clone(), at(1), at(2), at(3)]);
            record(name, vec![at(4), a.clone(), at(5), at(6), at(7)]);
        }
    }
    assert!(calls > 2_000_000, "{calls}");
    let unexpected: Vec<String> = panics
        .iter()
        .map(|(name, (count, first))| format!("{name}: {count} panicking argument sets, first: {first}"))
        .collect();
    assert!(unexpected.is_empty(), "pure builtins panicked:\n{}", unexpected.join("\n"));
}

/// The same sweep inside a restricted block's meter, on the combining builtins: whatever
/// comes back is within the single-value cap.
#[test]
fn metered_builtin_results_stay_within_the_value_cap() {
    let big = Value::string("x".repeat(800_000));
    let list = Value::list(vec![big.clone()]);
    let context = || meter::RunContext {
        limits: crate::run_limits::RunLimits::restricted(),
        policy: None,
        turn: meter::TurnBudget::new(None),
    };
    let program = ast::Program::default();
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    for (name, args) in [
        ("concat", vec![big.clone(), big.clone()]),
        ("list", vec![big.clone(), big.clone()]),
        ("push", vec![list.clone(), big.clone()]),
        ("insert", vec![list.clone(), Value::Int(0), big.clone()]),
        ("map", vec![Value::string("a"), big.clone(), Value::string("b"), big.clone()]),
        ("map_set", vec![Value::Map(Arc::new(vec![(Value::string("a"), big.clone())])), Value::string("b"), big.clone()]),
        ("set_add", vec![Value::Set(Arc::new(vec![big.clone()])), Value::string("y".repeat(800_000))]),
        ("to_string", vec![Value::list(vec![list.clone(), list.clone()])]),
        ("join", vec![Value::list(vec![big.clone(), big.clone()]), Value::string(",")]),
        ("regex_replace", vec![Value::string("x"), Value::string("xx"), big.clone()]),
        ("split", vec![big.clone(), Value::string("")]),
        ("Ok", vec![Value::list(vec![big.clone(), big.clone()])]),
        ("base64_encode", vec![big.clone()]),
        ("json_escape", vec![Value::string("\"".repeat(700_000))]),
    ] {
        // A tripped limit is sticky, so each case gets its own block.
        let _meter = meter::install(context());
        let error = eval_builtin_or_user(&program, name, args, &mut env).expect_err(name).to_string();
        assert!(error.contains("a value exceeds 1 MiB"), "{name}: {error}");
    }
    // And calls that fit.
    let _meter = meter::install(context());
    assert_eq!(eval_builtin_or_user(&program, "len", vec![big.clone()], &mut env).unwrap().to_string(), "800000");
    assert!(eval_builtin_or_user(&program, "starts_with", vec![big.clone(), big], &mut env).is_ok());
}

/// The fixes this sweep led to, pinned: an error or the wrapped value where there used to
/// be a panic, and unchanged results everywhere else.
#[test]
fn fixed_builtin_edge_cases() {
    let eval = |name: &str, args: Vec<Value>| call(name, args).expect("no panic").map(|value| value.to_string()).map_err(|error| error.to_string());
    let (min, text) = (Value::Int(i64::MIN), Value::string("héllo"));
    // Like the `/` and `%` operators (`checked_div`/`checked_rem`).
    assert_eq!(eval("div", vec![min.clone(), Value::Int(-1)]), Ok(i64::MIN.to_string()));
    assert_eq!(eval("rem", vec![min.clone(), Value::Int(-1)]), Ok("0".into()));
    assert_eq!(eval("div", vec![Value::Int(7), Value::Int(2)]), Ok("3".into()));
    assert_eq!(eval("rem", vec![Value::Int(-7), Value::Int(2)]), Ok("-1".into()));
    assert!(eval("div", vec![Value::Int(1), Value::Int(0)]).unwrap_err().contains("division by zero"));
    // `char_at` and `substring` are the release line's #134 implementation
    // (`eval/strings.rs`, with its own tests): byte offsets as before, the whole
    // character CONTAINING the offset, and cuts moved down to a character boundary.
    assert_eq!(eval("char_at", vec![text.clone(), Value::Int(0)]), Ok("\"h\"".into()));
    assert_eq!(eval("char_at", vec![text.clone(), Value::Int(1)]), Ok("\"é\"".into()));
    assert_eq!(eval("char_at", vec![text.clone(), Value::Int(2)]), Ok("\"é\"".into()));
    assert_eq!(eval("char_at", vec![text.clone(), Value::Int(3)]), Ok("\"l\"".into()));
    assert!(eval("char_at", vec![text.clone(), Value::Int(6)]).unwrap_err().contains("out of bounds"));
    assert_eq!(eval("substring", vec![text.clone(), Value::Int(2), Value::Int(4)]), Ok("\"él\"".into()));
    assert_eq!(eval("substr", vec![text, Value::Int(0), Value::Int(2)]), Ok("\"h\"".into()));
    // Wrapping made explicit; the release results are unchanged.
    assert_eq!(eval("abs", vec![min]), Ok(i64::MIN.to_string()));
    assert_eq!(eval("abs", vec![Value::Int(-5)]), Ok("5".into()));
    assert_eq!(eval("pow", vec![Value::Int(2), Value::Int(10)]), Ok("1024".into()));
    assert_eq!(eval("pow", vec![Value::Int(2), Value::Int(64)]), Ok("0".into()));
}
