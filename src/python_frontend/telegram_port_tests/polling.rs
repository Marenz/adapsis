//! Polling slice A: extract_update_id, init, handle_update, dispatch_updates and the
//! NEW batch validator. `+spawn` is intercepted as an IoRequest, so process_update
//! never runs here and spawn gives no outcome (#104). No network, files or sleeps.
use crate::{
    ast,
    coroutine::{CoroutineHandle, IoRequest},
    eval::{self, Value},
    parser, shared_state,
};
use serde_json::{json, Value as Json};
pub(super) mod cycle;
pub(super) mod looping;

pub(super) fn verify_repair(original: &ast::FunctionDecl, corrected: &ast::FunctionDecl) {
    use ast::StatementKind as S;
    assert_eq!(original.params, corrected.params);
    assert_eq!(original.return_type, corrected.return_type);
    // Python spells async last ([io,mut,async]); the effect SET is what must match.
    let set = |effects: &[ast::Effect]| {
        let mut names: Vec<_> = effects.iter().map(|e| format!("{e:?}")).collect();
        names.sort();
        names
    };
    assert_eq!(set(&original.effects), set(&corrected.effects));
    assert_eq!(corrected.effects.len(), 3);
    let S::While {
        condition: old_condition,
        body: old_loop,
    } = &original.body[4].kind
    else {
        panic!("original loop moved")
    };
    let S::While {
        condition,
        body: new_loop,
    } = &corrected.body[5].kind
    else {
        panic!("corrected loop moved")
    };
    assert_eq!(old_condition, condition);
    let S::Branch {
        then_body,
        else_body,
        ..
    } = &new_loop[1].kind
    else {
        panic!("validity branch missing")
    };
    assert_eq!(else_body.len(), 1);
    // `let update` intentionally reuses the validated index instead of re-formatting i.
    assert!(
        matches!(&then_body[0].kind, S::Let { value: ast::Expr::Call(call), .. }
        if call.callee == "json_get"
            && call.args == [ast::Expr::Identifier("arr".into()), ast::Expr::Identifier("idx".into())])
    );
    // Everything else survives modulo local-call qualification, including the
    // offset rule and the ADVANCE-BEFORE-SPAWN order.
    let mut old = original.clone();
    old.effects = corrected.effects.clone(); // order-normalized; set equality asserted above
    old.body = vec![
        original.body[0].clone(),
        original.body[2].clone(),
        original.body[3].clone(),
        old_loop[1].clone(),
        old_loop[2].clone(),
        old_loop[3].clone(),
        old_loop[4].clone(),
    ];
    let mut new = corrected.clone();
    new.body = vec![
        corrected.body[1].clone(),
        corrected.body[2].clone(),
        corrected.body[3].clone(),
        then_body[1].clone(),
        then_body[2].clone(),
        then_body[3].clone(),
        new_loop[2].clone(),
    ];
    new.doc = old.doc.clone();
    let locals = ["extract_update_id", "process_update"]
        .map(String::from)
        .into_iter()
        .collect();
    assert_eq!(
        super::qualification::verify(&old, &new, "TelegramBot", &locals),
        2
    );
    let manifest: Json = serde_json::from_str(include_str!(
        "../../../tests/fixtures/telegram-python-port/manifest.json"
    ))
    .unwrap();
    let module = &manifest["modules"][0];
    assert_eq!(
        module["contract_tests"]["telegram_port_polling_dispatch_updates_batches"],
        json!([
            "dispatch_updates",
            "extract_update_id",
            "batch_update_id_valid"
        ])
    );
    assert_eq!(
        module["contract_tests"]["telegram_port_polling_dispatch_updates_skips_and_non_arrays"],
        json!(["dispatch_updates", "batch_update_id_valid"])
    );
    assert_eq!(
        module["introduced_polling_pure_functions"],
        json!(["batch_update_id_valid"])
    );
}

struct Outcome {
    result: Result<Json, String>,
    spawned: Vec<(String, Vec<String>)>,
    state: Vec<(&'static str, Option<Value>)>,
}

const WATCHED: [&str; 4] = [
    "poll_offset",
    "telegram_token",
    "admin_user_ids",
    "allowed_user_ids",
];

fn collect(mut receiver: tokio::sync::mpsc::Receiver<IoRequest>) -> Vec<(String, Vec<String>)> {
    let mut spawned = Vec::new();
    while let Some(request) = receiver.blocking_recv() {
        let IoRequest::Spawn {
            function_name,
            args,
            reply,
            ..
        } = request
        else {
            panic!("slice A permits only intercepted +spawn; no network, file or sleep IO")
        };
        // No binding means the evaluator never reads the reply; nothing runs.
        drop(reply);
        let args = args
            .into_iter()
            .map(|value| match value {
                Value::String(text) => text.to_string(),
                other => panic!("spawn argument must be a String, got {other:?}"),
            })
            .collect();
        spawned.push((function_name, args));
    }
    spawned
}

fn run(program: &ast::Program, target: &str, input: &str, seeds: &[(&str, Value)]) -> Outcome {
    let _guard = super::routing::install(program, "1", "7");
    let runtime = shared_state::get_shared_runtime().unwrap();
    // Typed offset is always present so unrelated functions can be checked for non-interference.
    runtime
        .read()
        .unwrap()
        .shared_vars
        .insert("TelegramBot.poll_offset".into(), Value::Int(0));
    for (name, value) in seeds {
        runtime
            .read()
            .unwrap()
            .shared_vars
            .insert(format!("TelegramBot.{name}"), value.clone());
    }
    let (sender, receiver) = tokio::sync::mpsc::channel(256);
    let worker = std::thread::spawn(move || collect(receiver));
    let result = eval::eval_async_function(
        program,
        target,
        &parser::parse_single_expr(input).unwrap(),
        CoroutineHandle::new(sender),
    )
    .map(super::routing::observed)
    .map_err(|error| error.to_string());
    let spawned = worker.join().unwrap();
    let state = WATCHED
        .iter()
        .map(|name| {
            let value = runtime
                .read()
                .unwrap()
                .shared_vars
                .get(&format!("TelegramBot.{name}"));
            (*name, value)
        })
        .collect();
    Outcome {
        result,
        spawned,
        state,
    }
}

impl Outcome {
    fn ok(&self) -> Json {
        self.result.clone().expect("expected Ok result")
    }
    fn err(&self) -> String {
        self.result.clone().expect_err("expected terminal error")
    }
    fn get(&self, name: &str) -> &Value {
        self.state
            .iter()
            .find(|(key, _)| *key == name)
            .and_then(|(_, value)| value.as_ref())
            .unwrap_or_else(|| panic!("{name} was not materialized"))
    }
    fn offset(&self) -> i64 {
        match self.get("poll_offset") {
            Value::Int(value) => *value,
            other => panic!("poll_offset must stay Int, got {other:?}"),
        }
    }
    fn text(&self, name: &str) -> String {
        match self.get(name) {
            Value::String(value) => value.to_string(),
            other => panic!("{name} must stay String, got {other:?}"),
        }
    }
}

fn string_input(name: &str, value: &str) -> String {
    format!("{{{name}:{}}}", serde_json::to_string(value).unwrap())
}

fn batch(updates: &[Json]) -> String {
    json!({"ok": true, "result": updates}).to_string()
}

fn dispatch(program: &ast::Program, body: &str, offset: i64) -> Outcome {
    run(
        program,
        "TelegramBot.dispatch_updates",
        &string_input("body", body),
        &[("poll_offset", Value::Int(offset))],
    )
}

fn assert_spawned(outcome: &Outcome, expected: &[String]) {
    let actual: Vec<_> = outcome
        .spawned
        .iter()
        .map(|(name, args)| {
            assert_eq!(name, "TelegramBot.process_update", "unqualified spawn");
            assert_eq!(args.len(), 1);
            args[0].clone()
        })
        .collect();
    assert_eq!(actual, expected, "spawn order/arguments");
}

#[test]
fn telegram_port_polling_init_writes_shared_state() {
    for program in super::send_chain::programs() {
        for name in ["init", "handle_update", "dispatch_updates"] {
            assert!(
                crate::vm::compile_function(
                    program
                        .get_function(&format!("TelegramBot.{name}"))
                        .unwrap(),
                    &program
                )
                .is_err(),
                "{name}: shared access is not direct-VM coverage"
            );
        }
        let outcome = run(
            &program,
            "TelegramBot.init",
            "{token:\"tok-ä\",admins:\"1,2\"}",
            &[],
        );
        assert_eq!(outcome.ok(), json!("ok"));
        assert!(outcome.spawned.is_empty());
        assert_eq!(outcome.text("telegram_token"), "tok-ä");
        assert_eq!(outcome.text("admin_user_ids"), "1,2");
        // Unrelated state, and the typed offset, are untouched.
        assert_eq!(outcome.text("allowed_user_ids"), "7");
        assert_eq!(outcome.offset(), 0);
        let empty = run(
            &program,
            "TelegramBot.init",
            "{token:\"\",admins:\"\"}",
            &[],
        );
        assert_eq!(empty.ok(), json!("ok"));
        assert_eq!(empty.text("telegram_token"), "");
        assert_eq!(empty.text("admin_user_ids"), "");
    }
}

#[test]
fn telegram_port_polling_handle_update_spawn_and_mode() {
    for program in super::send_chain::programs() {
        let payload = "{\"update_id\":1,\"message\":{\"text\":\"ä😀\\n\"}}";
        let call = |seeds: &[(&str, Value)], payload: &str| {
            run(
                &program,
                "TelegramBot.handle_update",
                &string_input("payload", payload),
                seeds,
            )
        };
        let webhook = [("intake_mode", Value::string("webhook"))];
        let outcome = call(&webhook, payload);
        assert_eq!(outcome.ok(), json!("{}"));
        assert_spawned(&outcome, &[payload.to_owned()]);
        assert_eq!(
            outcome.offset(),
            0,
            "webhook intake never touches the offset"
        );
        // Ingress performs no validation: malformed or empty bodies are still
        // acknowledged and handed to the (isolated) spawned task.
        for raw in ["not json", ""] {
            let outcome = call(&webhook, raw);
            assert_eq!(outcome.ok(), json!("{}"));
            assert_spawned(&outcome, &[raw.to_owned()]);
        }
        // Default (declared "poll") and every non-exact mode acknowledge without spawning.
        let mut modes = vec![None];
        modes.extend(["poll", "", "Webhook", "webhook ", "webhooks"].map(Some));
        for mode in modes {
            let seeds: Vec<(&str, Value)> = mode
                .map(|text| ("intake_mode", Value::string(text)))
                .into_iter()
                .collect();
            let outcome = call(&seeds, payload);
            assert_eq!(outcome.ok(), json!("{}"), "mode {mode:?}");
            assert!(outcome.spawned.is_empty(), "mode {mode:?}");
        }
    }
}

#[test]
fn telegram_port_polling_dispatch_updates_batches() {
    for program in super::send_chain::programs() {
        let update = |id: i64| json!({"update_id": id, "message": {"text": format!("t{id} ä😀")}});
        let expected = |updates: &[Json]| updates.iter().map(Json::to_string).collect::<Vec<_>>();

        let two = [update(5), update(6)];
        let outcome = dispatch(&program, &batch(&two), 0);
        assert_eq!(outcome.ok(), json!("dispatched 2"));
        assert_spawned(&outcome, &expected(&two));
        assert_eq!(outcome.offset(), 7);

        // Offset never moves backwards, yet every listed update is still handed
        // over (Telegram-side offset is the only dedup, as in the original).
        let outcome = dispatch(&program, &batch(&two), 10);
        assert_eq!(outcome.ok(), json!("dispatched 2"));
        assert_spawned(&outcome, &expected(&two));
        assert_eq!(outcome.offset(), 10);

        let unordered = [update(9), update(7), update(8)];
        let outcome = dispatch(&program, &batch(&unordered), 0);
        assert_eq!(outcome.ok(), json!("dispatched 3"));
        assert_spawned(&outcome, &expected(&unordered));
        assert_eq!(
            outcome.offset(),
            10,
            "maximum id + 1, array order for spawns"
        );

        let repeated = [update(3), update(3)];
        let outcome = dispatch(&program, &batch(&repeated), 0);
        assert_spawned(&outcome, &expected(&repeated));
        assert_eq!(outcome.offset(), 4);

        let zero = [update(0)];
        let outcome = dispatch(&program, &batch(&zero), 0);
        assert_eq!(outcome.ok(), json!("dispatched 1"));
        assert_eq!(outcome.offset(), 1);

        let empty = dispatch(&program, &batch(&[]), 4);
        assert_eq!(empty.ok(), json!("dispatched 0"));
        assert!(empty.spawned.is_empty());
        assert_eq!(empty.offset(), 4);

        let top = [update(i64::MAX - 1)];
        let outcome = dispatch(&program, &batch(&top), 0);
        assert_eq!(outcome.ok(), json!("dispatched 1"));
        assert_eq!(outcome.offset(), i64::MAX, "uid+1 stays representable");

        // A full Telegram page (limit 100) fits the intercepted channel and
        // preserves array order.
        let page: Vec<Json> = (1000..1100).map(update).collect();
        let outcome = dispatch(&program, &batch(&page), 0);
        assert_eq!(outcome.ok(), json!("dispatched 100"));
        assert_spawned(&outcome, &expected(&page));
        assert_eq!(outcome.offset(), 1100);
    }
}

#[test]
fn telegram_port_polling_dispatch_updates_skips_and_non_arrays() {
    for program in super::send_chain::programs() {
        for body in [
            "{}",
            r#"{"ok":false,"error_code":401,"description":"Unauthorized"}"#,
            r#"{"result":null}"#,
            r#"{"result":{}}"#,
            r#"{"result":"[]"}"#,
            r#"{"result":""}"#,
            r#"{"result":5}"#,
            r#"{"result":true}"#,
        ] {
            let outcome = dispatch(&program, body, 3);
            assert_eq!(outcome.ok(), json!("no result"), "{body}");
            assert!(outcome.spawned.is_empty(), "{body}");
            assert_eq!(outcome.offset(), 3, "{body}");
        }
        let mixed = vec![
            json!({"update_id": 10}),
            json!({}),
            json!({"update_id": "11"}),
            json!({"update_id": 12.0}),
            json!({"update_id": null}),
            json!({"update_id": -3}),
            json!({"update_id": i64::MAX}),
            json!({"update_id": 5}),
            json!(5),
            json!("x"),
            json!([1]),
        ];
        let outcome = dispatch(&program, &batch(&mixed), 0);
        assert_eq!(outcome.ok(), json!("dispatched 2, skipped 9"));
        assert_spawned(&outcome, &[mixed[0].to_string(), mixed[7].to_string()]);
        assert_eq!(outcome.offset(), 11, "only valid ids advance the offset");

        let invalid_first = [json!({"message": {}}), json!({"update_id": 21})];
        let outcome = dispatch(&program, &batch(&invalid_first), 0);
        assert_eq!(outcome.ok(), json!("dispatched 1, skipped 1"));
        assert_spawned(&outcome, &[invalid_first[1].to_string()]);
        assert_eq!(outcome.offset(), 22, "the batch continues after a skip");

        let all_invalid = dispatch(&program, &batch(&[json!({})]), 6);
        assert_eq!(all_invalid.ok(), json!("dispatched 0, skipped 1"));
        assert!(all_invalid.spawned.is_empty());
        assert_eq!(all_invalid.offset(), 6);

        // Characterized limits, not fixes: malformed top-level JSON is terminal
        // (pending json_valid), as is an id beyond i64 -- after earlier updates
        // were already advanced and spawned (at-most-once, no rollback).
        let malformed = dispatch(&program, "<html>bad gateway</html>", 3);
        assert!(malformed.err().starts_with("json_kind: invalid JSON:"));
        assert!(malformed.spawned.is_empty());
        assert_eq!(malformed.offset(), 3);

        let huge = r#"{"result":[{"update_id":4},{"update_id":18446744073709551615}]}"#;
        let outcome = dispatch(&program, huge, 0);
        assert!(outcome
            .err()
            .contains("integer outside signed 64-bit range"));
        assert_spawned(&outcome, &[json!({"update_id": 4}).to_string()]);
        assert_eq!(outcome.offset(), 5);
    }
}
