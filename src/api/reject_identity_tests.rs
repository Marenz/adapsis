//! Reviewer residual (a): a replacement rejected by reject-on-fail must leave
//! the pre-turn Arc in place, so CoW write-back sees no delta and a concurrent
//! live change to the same function survives.
use super::*;

fn slot(program: &crate::ast::Program) -> std::sync::Arc<crate::ast::FunctionDecl> {
    program
        .modules
        .iter()
        .find(|m| m.name == "RjM")
        .unwrap()
        .functions
        .iter()
        .find(|f| f.name == "f")
        .unwrap()
        .clone()
}

#[tokio::test]
async fn rejected_replacement_keeps_pre_turn_arc_and_concurrent_live_change() {
    let config = test_config();
    let mut setup = config.snapshot_working_set().await;
    let res = execute::execute_code(
        "+module RjM\n+fn f (x:Int)->Int\n  +return x + 1\n+end\n+test f\n  +with x=1 -> expect 2\n",
        &config,
        &mut setup,
        None,
    )
    .await;
    assert!(!res.has_errors, "{:?}", res.mutation_results);
    config.write_back_working_set(&setup).await;

    // Turn snapshot, then a concurrent writer changes the same live function.
    let mut turn = config.snapshot_working_set().await;
    let pre_turn = slot(&turn.program);
    {
        let mut live = config.program.write().await;
        live.get_function_mut("RjM.f").unwrap().doc = Some("concurrent change".into());
    }

    // Signature + body replacement that fails the stored test.
    let res = execute::execute_code(
        "+module RjM\n+fn f (x:Int, y:Int)->Int\n  +return 0\n+end\n",
        &config,
        &mut turn,
        None,
    )
    .await;
    assert!(
        res.mutation_results
            .iter()
            .any(|r| !r.success && r.message.contains("REJECTED")),
        "{:?}",
        res.mutation_results
    );
    assert!(
        std::sync::Arc::ptr_eq(&slot(&turn.program), &pre_turn),
        "rejected replacement must restore the exact pre-turn Arc"
    );
    let live = config.program.read().await;
    let current = live.get_function("RjM.f").unwrap();
    assert_eq!(current.doc.as_deref(), Some("concurrent change"));
    assert_eq!(current.params.len(), 1);
    assert_eq!(current.body, pre_turn.body);
}
