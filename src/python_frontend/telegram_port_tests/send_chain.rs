//! Corrected-source send policy, separate from the untouched frozen wrapper
//! characterization. All execution uses normal async dispatch with intercepted IO.
use crate::{ast, eval, parser};
mod runtime;

pub(super) fn verify_repair(
    original: &ast::FunctionDecl,
    corrected: &ast::FunctionDecl,
    ledger: &serde_json::Value,
) {
    let kind = match original.name.as_str() {
        "send_message_part" => "typed_send_attempt_v1",
        "send_message_parts" => "typed_send_chain_v1",
        "send_message" => "typed_send_message_v1",
        "send_reply" => "typed_send_reply_v1",
        _ => panic!("new semantic repair requires explicit review"),
    };
    assert_eq!(ledger["kind"], kind);
    assert_eq!(ledger["parameters"], "preserved");
    assert_eq!(ledger["original_return"], "String");
    assert_eq!(ledger["corrected_return"], "TgSendAttempt");
    assert_eq!(
        ledger["corrected_effects"],
        serde_json::json!(["io", "fail", "async"])
    );
    assert_eq!(original.params, corrected.params);
    assert_eq!(original.return_type, ast::Type::String);
    assert!(
        matches!(&corrected.return_type, ast::Type::TaggedUnion(name) if name == "TgSendAttempt")
    );
    assert_eq!(original.effects, [ast::Effect::Io, ast::Effect::Async]);
    assert_eq!(
        corrected.effects,
        [ast::Effect::Io, ast::Effect::Fail, ast::Effect::Async]
    );
    assert_ne!(original.doc, corrected.doc);
    // Full corrected bodies are checked against Python in the parent gate,
    // while the production request matrix below checks the deliberate policy.
    assert!(!super::super::ast_equality::function(original, corrected));
}

pub(super) fn programs() -> [ast::Program; 2] {
    let mut origin = super::program(&parser::parse(super::ORIGIN).unwrap());
    let lowered = super::lower_telegram(super::PYTHON);
    let mut python = super::program(&lowered.operations);
    super::attach_format(&mut origin, false);
    super::attach_format(&mut python, true);
    super::super::validate_runtime_arguments(&lowered.operations, &python).unwrap();
    [origin, python]
}

#[test]
fn telegram_port_send_union_signatures_and_shared_fixture_are_explicit() {
    for program in programs() {
        for name in [
            "send_message_part",
            "send_message_parts",
            "send_message",
            "send_reply",
        ] {
            let function = program
                .get_function(&format!("TelegramBot.{name}"))
                .unwrap();
            assert!(
                matches!(&function.return_type, ast::Type::TaggedUnion(name) if name == "TgSendAttempt")
            );
            assert!(function.effects.contains(&ast::Effect::Async));
        }
        let module = &program.modules[0];
        let outcome = module
            .types
            .iter()
            .find_map(|ty| match ty {
                ast::TypeDecl::TaggedUnion(value) if value.name == "TgSendAttempt" => Some(value),
                _ => None,
            })
            .unwrap();
        assert_eq!(outcome.name, "TgSendAttempt");
        assert_eq!(
            outcome
                .variants
                .iter()
                .map(|v| v.name.as_str())
                .collect::<Vec<_>>(),
            ["TgSent", "TgRejected", "TgUncertain"]
        );
        assert!(outcome
            .variants
            .iter()
            .all(|v| v.payload == [ast::Type::String]));
        assert_eq!(module.shared_vars.len(), 4);
        assert_eq!(module.shared_vars[0].name, "telegram_token");
        assert!(
            matches!(eval::eval_expr_standalone(&program, &module.shared_vars[0].default).unwrap(), eval::Value::String(ref value) if value.is_empty())
        );
    }
    let manifest: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/telegram-python-port/manifest.json"
    ))
    .unwrap();
    let entry = &manifest["modules"][0];
    assert_eq!(
        entry["normal_io_contract_functions"],
        serde_json::json!([
            "send_message_part",
            "send_message_parts",
            "send_message",
            "send_reply",
            "handle_group",
            "handle_admin_dm",
            "handle_user_dm",
            "telegram_file_path",
            "handle_group_attachment",
            "handle_admin_dm_attachment",
            "handle_user_dm_attachment",
            "transcribe_voice",
        ])
    );
    assert_eq!(entry["semantic_repairs"].as_object().unwrap().len(), 13);
    assert_eq!(
        entry["introduced_types"]["TgSendAttempt"],
        serde_json::json!([
            "TgSent(String)",
            "TgRejected(String)",
            "TgUncertain(String)"
        ])
    );
}
