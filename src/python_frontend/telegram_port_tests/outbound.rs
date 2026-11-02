//! Outbound fixtures: app outcomes, not callback acknowledgements (#104).
use crate::{ast, eval};
use serde_json::{json, Value as Json};
mod runtime;
mod tests;

pub(super) fn verify_repair(original: &ast::FunctionDecl, corrected: &ast::FunctionDecl) {
    use ast::StatementKind as S;
    assert_eq!(original.params, corrected.params);
    assert_eq!(
        corrected.effects,
        [ast::Effect::Io, ast::Effect::Fail, ast::Effect::Async]
    );
    let (old_indices, new_indices, old_upload, new_upload) = if original.name == "send_audio_file" {
        (vec![0, 1], vec![1, 2], 2, 3)
    } else {
        (vec![0, 2, 3], vec![0, 2, 3], 4, 4)
    };
    let mut old = original.clone();
    old.body = old_indices
        .iter()
        .map(|i| original.body[*i].clone())
        .collect();
    let mut restored = old.clone();
    restored.body = new_indices
        .iter()
        .map(|i| corrected.body[*i].clone())
        .collect();
    assert!(super::super::ast_equality::function(&old, &restored));
    let S::Match {
        expr: ast::Expr::Call(old_call),
        ..
    } = &original.body[old_upload].kind
    else {
        panic!()
    };
    let S::Await { call, .. } = &corrected.body[new_upload].kind else {
        panic!()
    };
    assert_eq!(
        old_call, call,
        "original upload URL/source/field/extras arguments"
    );
    let ledger: Json = serde_json::from_str(include_str!(
        "../../../tests/fixtures/telegram-python-port/manifest.json"
    ))
    .unwrap();
    assert_eq!(
        ledger["modules"][0]["introduced_trusted_result_contract_functions"],
        json!(["telegram_upload_attempt"])
    );
    let test = if original.name == "send_audio_file" {
        "telegram_port_outbound_path_normal_async"
    } else {
        "telegram_port_outbound_attachment_normal_async"
    };
    assert_eq!(
        ledger["modules"][0]["contract_tests"][test],
        json!([original.name])
    );
}

pub(super) fn outcome(tag: &str, body: &str) -> Json {
    json!({"tag":tag,"payload":[body]})
}

pub(super) fn upload_cases() -> Vec<(
    Result<&'static str, &'static str>,
    Result<Json, &'static str>,
)> {
    let mut cases = vec![
        (
            Ok(r#"{"ok":true}"#),
            Ok(outcome("TgSent", r#"{"ok":true}"#)),
        ),
        (
            Ok(r#"{"ok":false}"#),
            Ok(outcome("TgRejected", r#"{"ok":false}"#)),
        ),
        (
            Err("transport lost"),
            Ok(outcome("TgUncertain", "transport lost")),
        ),
        (Ok("not json"), Err("json_kind: invalid JSON:")),
    ];
    for body in [
        "{}",
        r#"{"error":"failed"}"#,
        r#"{"ok":"true"}"#,
        r#"{"ok":"false"}"#,
        r#"{"ok":null}"#,
        r#"{"ok":0}"#,
        r#"{"ok":1}"#,
        r#"{"ok":1.0}"#,
        r#"{"ok":[]}"#,
        r#"{"ok":{}}"#,
    ] {
        cases.push((Ok(body), Ok(outcome("TgUncertain", body))));
    }
    cases
}

pub(super) fn check(actual: Result<Json, String>, expected: &Result<Json, &str>) {
    match expected {
        Ok(value) => assert_eq!(actual.unwrap(), *value),
        Err(prefix) => assert!(actual.unwrap_err().starts_with(prefix)),
    }
}

#[test]
fn telegram_port_outbound_upload_classifier_and_extract_reply() {
    for program in super::send_chain::programs() {
        for (response, expected) in upload_cases() {
            let input = match response {
                Ok(body) => eval::Value::Ok(Box::new(eval::Value::string(body))),
                Err(e) => eval::Value::Err(e.into()),
            };
            for result in [
                super::super::logical_tier_tests::tree(
                    &program,
                    "TelegramBot.telegram_upload_attempt",
                    vec![input.clone()],
                ),
                super::super::logical_tier_tests::vm_result_opcode_fixture_run(
                    &program,
                    "TelegramBot.telegram_upload_attempt",
                    vec![input],
                ),
            ] {
                check(
                    result
                        .map(super::routing::observed)
                        .map_err(|e| e.to_string()),
                    &expected,
                );
            }
        }
        for (body, expected) in [
            (r#"{"reply":7}"#, eval::Value::Int(7)),
            (r#"{"reply":1.5}"#, eval::Value::Float(1.5)),
            (r#"{"reply":false}"#, eval::Value::Bool(false)),
            (r#"{"reply":null}"#, eval::Value::string("null")),
            (r#"{"reply":{}}"#, eval::Value::string("{}")),
            (r#"{"reply":[]}"#, eval::Value::string("[]")),
        ] {
            for result in [
                super::super::logical_tier_tests::tree(
                    &program,
                    "TelegramBot.extract_reply",
                    vec![eval::Value::string(body)],
                ),
                super::super::logical_tier_tests::vm_run(
                    &program,
                    "TelegramBot.extract_reply",
                    vec![eval::Value::string(body)],
                ),
            ] {
                assert!(result.unwrap().matches(&expected));
            }
        }
        for result in [
            super::super::logical_tier_tests::tree(
                &program,
                "TelegramBot.extract_reply",
                vec![eval::Value::string("not json")],
            ),
            super::super::logical_tier_tests::vm_run(
                &program,
                "TelegramBot.extract_reply",
                vec![eval::Value::string("not json")],
            ),
        ] {
            assert!(result
                .unwrap_err()
                .to_string()
                .starts_with("json_get: invalid JSON:"));
        }
    }
}
