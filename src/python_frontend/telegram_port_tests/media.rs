//! Attachment annotation admission must pass without inventing a data class.
use crate::parser;
mod runtime;

pub(super) fn verify_file_path(
    original: &crate::ast::FunctionDecl,
    corrected: &crate::ast::FunctionDecl,
) {
    use crate::ast::{Effect, Expr, StatementKind as S, Type};
    assert_eq!(original.params, corrected.params);
    assert_eq!(original.return_type, Type::String);
    assert_eq!(corrected.return_type, Type::String);
    assert_eq!(corrected.effects, [Effect::Io, Effect::Fail, Effect::Async]);
    let S::Match {
        expr: Expr::Call(old_call),
        ..
    } = &original.body[1].kind
    else {
        panic!()
    };
    let S::Await { name, ty, call } = &corrected.body[1].kind else {
        panic!()
    };
    assert_eq!(name, "response");
    assert_eq!(*ty, Type::Result(Box::new(Type::String)));
    assert_eq!(call, old_call);
    let mut restored = corrected.clone();
    restored.body.remove(1);
    let S::Match { expr, .. } = &mut restored.body[1].kind else {
        panic!()
    };
    assert_eq!(*expr, Expr::Identifier("response".into()));
    *expr = Expr::Call(old_call.clone());
    restored.effects = original.effects.clone();
    restored.doc = original.doc.clone();
    assert_eq!(
        super::qualification::verify(
            original,
            &restored,
            "TelegramBot",
            &["telegram_file_path_response".to_owned()]
                .into_iter()
                .collect()
        ),
        1
    );
}

#[test]
fn telegram_port_attachment_annotation_uses_existing_frontend_contract() {
    let ax="+module AttachmentProbe\n+fn probe (file:Attachment)->String [io,async]\n  +await reply:String = llm_takeover(\"fixture\", \"message\", \"Callback.reply\", \"7\", \"\", file, \"{}\")\n  +return reply\n+end\n";
    let py="@effects('io')\nasync def probe(file: Attachment) -> str:\n    reply: str = await llm_takeover('fixture', 'message', 'Callback.reply', '7', '', file, '{}')\n    return reply\n";
    let origin = super::program(&parser::parse(ax).unwrap());
    let lowered = super::super::lower_module(py, "AttachmentProbe").unwrap();
    let python = super::program(&lowered.operations);
    assert!(super::super::ast_equality::program(&origin, &python));
    let printed = super::super::print_module(&python, "AttachmentProbe").unwrap();
    let second = super::program(
        &super::super::lower_module(&printed, "AttachmentProbe")
            .unwrap()
            .operations,
    );
    assert!(super::super::ast_equality::program(&python, &second));
    assert_eq!(
        super::super::print_module(&second, "AttachmentProbe").unwrap(),
        printed
    );
    super::super::validate_runtime_arguments(&lowered.operations, &python).unwrap();
    let caller="@effects('io')\nasync def caller() -> str:\n    file: Attachment = await http_get_binary('fixture://image')\n    result: str = await AttachmentProbe.probe(file)\n    return result\n";
    let lowered =
        super::super::lower_module_with_program(caller, "AttachmentCaller", &python).unwrap();
    super::super::validate_runtime_arguments(&lowered.operations, &python).unwrap();
}
