//! Original-name intake integration; no synthetic driver and no live IO.
use crate::ast;
mod runtime;
mod tests;
pub(super) const VALIDATORS: &[&str] = &[
    "intake_validate",
    "intake_validate_fields",
    "intake_validate_media",
];

pub(super) fn verify_repair(original: &ast::FunctionDecl, corrected: &ast::FunctionDecl) {
    assert_eq!(original.params, corrected.params);
    assert_eq!(original.doc, corrected.doc);
    assert_eq!(original.return_type, ast::Type::String);
    assert_eq!(
        corrected.effects,
        [ast::Effect::Io, ast::Effect::Fail, ast::Effect::Async]
    );
    // Only duplicate group/supergroup branches were folded. Every original
    // awaited call and its arguments must remain, modulo local qualification.
    assert_eq!(await_calls(&original.body), await_calls(&corrected.body));
    let ledger: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/telegram-python-port/manifest.json"
    ))
    .unwrap();
    assert_eq!(
        ledger["modules"][0]["introduced_pure_paired_functions"],
        serde_json::json!(VALIDATORS)
    );
    assert_eq!(
        ledger["modules"][0]["contract_tests"]
            ["telegram_port_intake_raw_media_text_voice_normal_dispatch"],
        serde_json::json!(["process_update"])
    );
}

fn await_calls(body: &[ast::Statement]) -> std::collections::BTreeSet<String> {
    let mut calls = std::collections::BTreeSet::new();
    let mut todo = vec![body];
    while let Some(body) = todo.pop() {
        for statement in body {
            match &statement.kind {
                ast::StatementKind::Await { call, .. } => {
                    let mut call = call.clone();
                    normalize_call(&mut call);
                    calls.insert(serde_json::to_string(&call).unwrap());
                }
                ast::StatementKind::Branch {
                    then_body,
                    else_body,
                    ..
                } => {
                    todo.push(then_body);
                    todo.push(else_body);
                }
                ast::StatementKind::Let { .. }
                | ast::StatementKind::Set { .. }
                | ast::StatementKind::Return { .. } => {}
                _ => panic!("intake repair acquired a new control-flow form; review ledger"),
            }
        }
    }
    calls
}

fn normalize_call(call: &mut ast::CallExpr) {
    call.callee = call
        .callee
        .strip_prefix("TelegramBot.")
        .unwrap_or(&call.callee)
        .to_owned();
    for arg in &mut call.args {
        if let ast::Expr::Call(nested) = arg {
            normalize_call(nested);
        }
    }
}
