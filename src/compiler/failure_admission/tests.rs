use super::*;
use crate::compiler::admission_measurement_tests::program;

fn refused(source: &str) {
    let program = program(source);
    let error = match crate::compiler::compile(&program) {
        Ok(_) => panic!("unsafe direct compile was admitted"),
        Err(error) => error,
    };
    assert!(error.to_string().contains(REFUSAL), "{error}");
}

#[test]
fn direct_compile_refuses_check_fail_and_result_contracts_before_emission() {
    for source in [
        "+fn f ()->Int [fail]\n  +check ready false ~not_ready\n  +return 1\n+end\n",
        "+fn f ()->Int [fail]\n  +return 1\n+end\n",
        "+fn f ()->Result<Int>\n  +return 1\n+end\n",
        "+fn f (r:Result<Int>)->Int\n  +return 1\n+end\n",
        "+fn f ()->Int\n  +let r:Result<Int> = Ok(1)\n  +return 1\n+end\n",
        "+fn leaf ()->Int\n  +return 1\n+end\n+fn f ()->Int\n  +call r:Result<Int> = leaf()\n  +return 1\n+end\n",
        "+fn f ()->Int\n  +return Err(\"not scalar\")\n+end\n",
        "+fn f (r:List<Result<Int>>)->Int\n  +return 1\n+end\n",
        "+type Box = r:Result<Int>\n+fn f (b:Box)->Int\n  +return 1\n+end\n",
    ] {
        refused(source);
    }
}

#[test]
fn check_cannot_escape_admission_by_omitting_the_fail_annotation() {
    let mut program =
        program("+fn f ()->Int [fail]\n  +check ready true ~not_ready\n  +return 1\n+end\n");
    program.get_function_mut("f").unwrap().effects.clear();
    assert!(check_root(program.get_function("f").unwrap(), &program).is_err());
    assert!(!crate::compiler::is_compilable_function(
        program.get_function("f").unwrap()
    ));
    assert!(crate::compiler::compile(&program).is_err());
}

#[test]
fn reachable_failure_in_conditions_tail_calls_and_nested_arguments_is_refused() {
    let leaf =
        "+fn leaf ()->Int [fail]\n  +return 7\n+end\n+fn id (n:Int)->Int\n  +return n\n+end\n";
    for body in [
        "+return leaf()",
        "+return id(leaf())",
        "+return 1 + leaf()",
        "+return leaf() + 1",
        "+if leaf() > 0\n    +return 1\n  +end\n  +return 0",
        "+while leaf() > 0\n    +return 1\n  +end\n  +return 0",
    ] {
        let program = program(&format!("{leaf}+fn probe ()->Int\n  {body}\n+end\n"));
        assert!(crate::compiler::is_compilable_function(
            program.get_function("probe").unwrap()
        ));
        assert!(
            check_root(program.get_function("probe").unwrap(), &program).is_err(),
            "{body}"
        );
        assert!(crate::compiler::compile(&program).is_err());
    }
}

#[test]
fn graph_identity_cycles_and_original_callee_spelling_are_preserved() {
    let clean = program("+fn a ()->Int\n  +return b()\n+end\n+fn b ()->Int\n  +return a()\n+end\n");
    assert!(check_root(clean.get_function("a").unwrap(), &clean).is_ok());
    // Only inspect the cycle; executing it would not terminate.
    let dirty = program("+fn a ()->Int\n  +return b()\n+end\n+fn b ()->Int\n  +return a() + bad()\n+end\n+fn bad ()->Int [fail]\n  +return 1\n+end\n");
    assert!(check_root(dirty.get_function("a").unwrap(), &dirty).is_err());
    let modules = program("+module A\n+fn child ()->Int [fail]\n  +return 1\n+end\n+module B\n+fn child ()->Int\n  +return 2\n+end\n+fn bare ()->Int\n  +return child()\n+end\n+fn qualified ()->Int\n  +return B.child()\n+end\n");
    assert!(check_root(modules.get_function("B.bare").unwrap(), &modules).is_err());
    assert!(check_root(modules.get_function("B.qualified").unwrap(), &modules).is_ok());
}

#[test]
fn compile_checks_unsafe_siblings_but_root_admission_is_reachability_scoped() {
    let program = program(
        "+fn safe ()->Int\n  +return 1\n+end\n+fn unused ()->Int [fail]\n  +return 2\n+end\n",
    );
    assert!(check_root(program.get_function("safe").unwrap(), &program).is_ok());
    assert!(
        check_program(&program).is_err(),
        "compile emits the unsafe sibling too"
    );
    assert!(crate::compiler::compile(&program).is_err());
}

#[test]
fn scalar_numeric_control_still_compiles_and_executes() {
    let program = program("+fn probe (n:Int)->Int\n  +if n > 0\n    +return n * 3\n  +else\n    +return 0\n  +end\n+end\n");
    assert!(check_root(program.get_function("probe").unwrap(), &program).is_ok());
    let mut jit = crate::compiler::compile(&program).unwrap();
    assert_eq!(jit.call_i64("probe", &[4]).unwrap(), 12);
    assert_eq!(jit.call_i64("probe", &[-4]).unwrap(), 0);
}

#[test]
fn admission_is_snapshot_local_after_a_callee_changes() {
    let mut program =
        program("+fn leaf ()->Int\n  +return 1\n+end\n+fn probe ()->Int\n  +return leaf()\n+end\n");
    assert!(check_root(program.get_function("probe").unwrap(), &program).is_ok());
    program
        .get_function_mut("leaf")
        .unwrap()
        .effects
        .push(ast::Effect::Fail);
    assert!(check_root(program.get_function("probe").unwrap(), &program).is_err());
    program.get_function_mut("leaf").unwrap().effects.clear();
    assert!(check_root(program.get_function("probe").unwrap(), &program).is_ok());
}

#[test]
fn result_hidden_in_named_or_recursive_types_is_still_refused() {
    for source in [
        "+type Payload = value:Result<Int>\n+type Envelope = payload:Payload\n+fn root (e:Envelope)->Int\n  +return 1\n+end\n",
        "+type Chain = Next(Chain) | End(Result<Int>)\n+fn root (c:Chain)->Int\n  +return 1\n+end\n",
        "+fn root (m:Map<String, List<Result<Int>>>)->Int\n  +return 1\n+end\n",
    ] {
        let program = program(source);
        assert!(check_root(program.get_function("root").unwrap(), &program).is_err());
        assert!(check_program(&program).is_err());
    }
    // Recursive type without Result terminates; ordinary compiler restrictions are
    // separate from this failure-contract safety check.
    let clean = program(
        "+type Chain = Next(Chain) | End(Int)\n+fn root (c:Chain)->Int\n  +return 1\n+end\n",
    );
    assert!(check_root(clean.get_function("root").unwrap(), &clean).is_ok());
}
