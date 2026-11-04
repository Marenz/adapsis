use super::*;
use crate::session::{Session, rewind_to};
use serde_json::json;

const GOOD: &str = "# Original Python provenance\ndef total(price: int, quantity: int, fee: int) -> int:\n    product: int = price * quantity\n    subtotal: int = product + fee\n    return subtotal\n";
const CASES: &str = "def test_regular():\n    assert total(7, 3, 2) == 23\ndef test_zero():\n    assert total(0, 3, 2) == 2\n";

async fn mutation(session: &mut Session, module: &str, source: &str) -> Vec<(String, bool)> {
    let lowered =
        crate::python_frontend::lower_module_with_program(source, module, &session.program)
            .unwrap();
    session
        .apply_operations_async(
            &lowered.operations,
            &Submission::python(module, source),
            None,
        )
        .await
        .unwrap()
}

async fn tested(module: &str) -> Session {
    let mut session = Session::new();
    assert!(
        mutation(&mut session, module, GOOD)
            .await
            .iter()
            .all(|(_, ok)| *ok)
    );
    let lowered = crate::python_frontend::lower_tests(CASES, module).unwrap();
    let results = session
        .apply_operations_async(
            &lowered.operations,
            &Submission::python(module, CASES),
            None,
        )
        .await
        .unwrap();
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|(_, ok)| *ok));
    session
}

fn rewind(session: &mut Session, revision: usize) -> Result<()> {
    rewind_to(
        &mut session.program,
        &mut session.runtime,
        &mut session.meta,
        &mut session.sandbox,
        revision,
    )
}

fn value(session: &Session, module: &str) -> String {
    crate::eval::eval_call_with_input(
        &session.program,
        &format!("{module}.total"),
        &crate::python_frontend::lower_input_expr("7, 3, 2").unwrap(),
    )
    .unwrap()
}

#[tokio::test]
async fn typed_replay_122_rejected_replacement_keeps_actual_body_and_two_tests() {
    let module = "TypedRejected";
    let mut session = tested(module).await;
    let source = GOOD.replace("return subtotal", "return fee");
    let result = mutation(&mut session, module, &source).await;
    assert!(result.iter().any(|(_, ok)| !ok));
    assert_eq!(value(&session, module), "23");
    let recorded = serde_json::to_value(session.meta.sources.last().unwrap()).unwrap();
    assert_eq!(recorded["outcome"], "rejected");
    assert_eq!(recorded["source"], source);
    assert_eq!(session.meta.mutations.last().unwrap().source, source);
    // Actual old algorithm: replay requested operations without auto-retest.
    let mut wrong = crate::ast::Program::default();
    for record in &session.meta.sources {
        for op in record.operations().unwrap() {
            let _ = crate::validator::apply_and_validate(&mut wrong, &op);
        }
    }
    let input = crate::python_frontend::lower_input_expr("7, 3, 2").unwrap();
    assert_eq!(
        crate::eval::eval_call_with_input(&wrong, &format!("{module}.total"), &input).unwrap(),
        "2"
    );
    let revision = session.meta.revision;
    rewind(&mut session, revision).unwrap();
    assert_eq!(value(&session, module), "23");
    let tests = &session
        .program
        .get_function(&format!("{module}.total"))
        .unwrap()
        .tests;
    assert_eq!(tests.len(), 2);
    assert!(tests.iter().all(|test| test.passed && test.typed.is_some()));
}

#[tokio::test]
async fn typed_replay_122_partial_module_retains_only_actual_accepted_changes() {
    let module = "TypedPartial";
    let mut session = tested(module).await;
    let source = format!(
        "{}\ndef added() -> int:\n    return 42\n",
        GOOD.replace("return subtotal", "return fee")
    );
    assert!(
        mutation(&mut session, module, &source)
            .await
            .iter()
            .any(|(_, ok)| !ok)
    );
    assert_eq!(
        serde_json::to_value(session.meta.sources.last().unwrap()).unwrap()["outcome"],
        "partial"
    );
    let before = serde_json::to_value(&session.program.modules).unwrap();
    let revision = session.meta.revision;
    rewind(&mut session, revision).unwrap();
    assert_eq!(
        before,
        serde_json::to_value(&session.program.modules).unwrap()
    );
    assert_eq!(value(&session, module), "23");
    assert!(
        session
            .program
            .get_function(&format!("{module}.added"))
            .is_some()
    );
    assert!(
        mutation(
            &mut session,
            module,
            "def later(x: int) -> int:\n    return x + 1\n"
        )
        .await
        .iter()
        .all(|(_, ok)| *ok)
    );
    let module = session
        .program
        .modules
        .iter()
        .find(|m| m.name == module)
        .unwrap();
    let ids: std::collections::HashSet<_> = module.functions.iter().map(|f| &f.id).collect();
    assert_eq!(ids.len(), module.functions.len());
    assert!(module.functions.iter().all(|f| !f.id.is_empty()));
}

#[tokio::test]
async fn typed_replay_old_strings_remain_strings_and_ambiguous_failure_is_atomic() {
    let legacy = "+module LegacyTypedReplay\n+fn value ()->Int\n +return 1\n+end";
    let mut session = Session::new();
    session.apply_async(legacy, None).await.unwrap();
    mutation(&mut session, "MixedTypedReplay", GOOD).await;
    let json = serde_json::to_value(&session).unwrap();
    assert_eq!(json["sources"][0], legacy);
    assert_eq!(json["sources"][1]["format"], "python");
    let mut restored: Session = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(
        serde_json::to_value(&restored).unwrap()["sources"],
        json["sources"]
    );
    rewind(&mut restored, 2).unwrap();
    let before = serde_json::to_value(&restored.program).unwrap();
    restored.meta.mutations[0].success = false;
    let error = rewind(&mut restored, 2).unwrap_err().to_string();
    assert_eq!(before, serde_json::to_value(&restored.program).unwrap());
    assert!(error.contains("revision 1 (adapsis)"));
    assert!(!error.contains("LegacyTypedReplay"));
}

#[tokio::test]
async fn typed_replay_corrupt_record_fails_closed_without_replacing_live_state() {
    let module = "TypedCorrupt";
    let mut session = tested(module).await;
    let record = serde_json::to_value(&session.meta.sources[0]).unwrap();
    let before = serde_json::to_value(&session.program).unwrap();
    let mut bads = Vec::new();
    let mut version = record.clone();
    version["version"] = json!(999);
    bads.push(version);
    let mut owner = record.clone();
    owner["checkpoint"]["name"] = json!("PRIVATE_SENTINEL");
    bads.push(owner);
    let mut action = record.clone();
    action["operations"] = json!(["Done"]);
    bads.push(action);
    let mut future_test = record.clone();
    future_test["checkpoint"]["functions"][0]["tests"] = json!([{
        "input":"PRIVATE_PAYLOAD_SENTINEL", "expected":"1", "passed":true,
        "typed":{"version":999,"case":{"input":{"Int":1},"expected":{"Int":1},"matcher":null,"after_checks":[]}}
    }]);
    bads.push(future_test);
    for bad in bads {
        session.meta.sources[0] = serde_json::from_value(bad).unwrap();
        let error = rewind(&mut session, 1).unwrap_err().to_string();
        assert_eq!(before, serde_json::to_value(&session.program).unwrap());
        assert_eq!(error, "replay revision 1 (python): invalid source record");
    }
    assert!(
        serde_json::from_str::<ReplayEntry>(&format!("{}0{}", "[".repeat(200), "]".repeat(200)))
            .is_err()
    );
}

#[tokio::test]
async fn typed_stored_tests_survive_real_save_load_and_reject_later_regression() {
    let module = "TypedSaveLoad";
    let session = tested(module).await;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("session.json");
    session.save(&path).unwrap();
    let mut loaded = Session::load(&path).unwrap();
    let source = GOOD.replace("return subtotal", "return fee");
    assert!(
        mutation(&mut loaded, module, &source)
            .await
            .iter()
            .any(|(_, ok)| !ok)
    );
    assert_eq!(value(&loaded, module), "23");
    assert_eq!(
        loaded
            .program
            .get_function(&format!("{module}.total"))
            .unwrap()
            .tests
            .len(),
        2
    );
}

#[tokio::test]
async fn typed_replay_122_earlier_rewind_refuses_edits_without_erasing_future_history() {
    let mut session = Session::new();
    mutation(&mut session, "TypedBranchRoot", GOOD).await;
    mutation(
        &mut session,
        "TypedBranchFuture",
        "def future() -> int:\n    return 9\n",
    )
    .await;
    rewind(&mut session, 1).unwrap();
    assert!(
        session
            .program
            .get_function("TypedBranchFuture.future")
            .is_none()
    );
    let source = GOOD.replace("return subtotal", "return fee");
    let lowered = crate::python_frontend::lower_module(&source, "TypedBranchRoot").unwrap();
    for revision in [1, 99] {
        session.meta.revision = revision;
        let before = serde_json::to_value(&session).unwrap();
        let error = session
            .apply_operations_async(
                &lowered.operations,
                &Submission::python("TypedBranchRoot", &source),
                None,
            )
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("history branch policy"));
        assert_eq!(before, serde_json::to_value(&session).unwrap());
    }
    session.meta.revision = 1;
    let tests = crate::python_frontend::lower_tests(CASES, "TypedBranchRoot").unwrap();
    assert!(
        session
            .apply_operations_async(
                &tests.operations,
                &Submission::python("TypedBranchRoot", CASES),
                None
            )
            .await
            .unwrap()
            .iter()
            .all(|(_, ok)| *ok)
    );
    assert_eq!(value(&session, "TypedBranchRoot"), "23");
    assert_eq!(session.meta.sources.len(), 2);
    assert!(
        session
            .program
            .get_function("TypedBranchFuture.future")
            .is_none()
    );
}

#[tokio::test]
async fn typed_replay_budget_refusal_precedes_application_and_record_copy() {
    let mut session = Session::new();
    let lowered = crate::python_frontend::lower_module(GOOD, "TypedBudget").unwrap();
    let submission = Submission::python("TypedBudget", &"#".repeat(1024 * 1024 + 1));
    let before = serde_json::to_value(&session).unwrap();
    let error = session
        .apply_operations_async(&lowered.operations, &submission, None)
        .await
        .unwrap_err()
        .to_string();
    assert_eq!(error, "Python replay source exceeds 1 MiB");
    assert_eq!(before, serde_json::to_value(&session).unwrap());
}

#[tokio::test]
async fn typed_stored_payload_present_invalid_never_falls_back_to_legacy_strings() {
    let session = tested("TypedPayloadVersion").await;
    let original = serde_json::to_value(&session).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("session.json");
    for payload in [
        json!({"version":999,"case":{
            "input":{"Int":7},"expected":{"Int":7},"matcher":null,"after_checks":[]
        }}),
        json!({"version":1,"case":"PRIVATE_PAYLOAD_SENTINEL"}),
    ] {
        let mut corrupted = original.clone();
        corrupted["program"]["modules"][0]["functions"][0]["tests"][0]["typed"] = payload;
        std::fs::write(&path, serde_json::to_vec(&corrupted).unwrap()).unwrap();
        let error = Session::load(&path).unwrap_err().to_string();
        assert!(!error.contains("PRIVATE_PAYLOAD_SENTINEL"));
        assert!(error.contains("invalid"));
    }
}
