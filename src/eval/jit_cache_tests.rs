use super::*;

fn program(source: &str) -> ast::Program {
    let mut program = ast::Program::default();
    for operation in parser::parse(source).unwrap() {
        crate::validator::apply_and_validate(&mut program, &operation).unwrap();
    }
    program.rebuild_function_index();
    program
}

#[test]
fn jit_failed_revision_rebuild_never_executes_old_code_119() {
    let old = program("+fn probe ()->Int\n  +return 11\n+end\n");
    let new = program("+fn probe ()->Int\n  +return 22\n+end\n+fn uncompiled ()->List<Int>\n  +return list(1)\n+end\n");
    assert!(crate::compiler::is_compilable_function(
        new.get_function("probe").unwrap()
    ));
    assert!(crate::compiler::compile(&new).is_err());
    let input = parser::parse_single_expr("{}").unwrap();
    let cache = new_jit_cache();
    assert_eq!(
        eval_compiled_or_interpreted_cached(&old, "probe", &input, Some(&cache), 1).unwrap(),
        ("11".into(), true)
    );
    let actual =
        eval_compiled_or_interpreted_cached(&new, "probe", &input, Some(&cache), 2).unwrap();
    assert_eq!(
        actual,
        ("22".into(), false),
        "must execute current VM/tree code, never cached revision 1"
    );
    assert!(
        cache.lock().unwrap().is_none(),
        "failed rebuild discards old module"
    );
}

#[test]
fn jit_warm_same_revision_control_remains_enabled() {
    let program = program("+fn probe ()->Int\n  +return 11\n+end\n");
    let input = parser::parse_single_expr("{}").unwrap();
    let cache = new_jit_cache();
    for _ in 0..3 {
        assert_eq!(
            eval_compiled_or_interpreted_cached(&program, "probe", &input, Some(&cache), 1)
                .unwrap(),
            ("11".into(), true)
        );
        assert!(matches!(&*cache.lock().unwrap(), Some((1, _))));
    }
}

#[test]
fn jit_current_graph_admission_is_checked_even_before_an_exact_revision_cache_hit() {
    let old = program("+fn probe ()->Int\n  +return 11\n+end\n");
    let new = program("+fn probe ()->Int\n  +return child()\n+end\n+fn child ()->Int [fail]\n  +return 22\n+end\n");
    let input = parser::parse_single_expr("{}").unwrap();
    let cache = new_jit_cache();
    assert_eq!(
        eval_compiled_or_interpreted_cached(&old, "probe", &input, Some(&cache), 1).unwrap(),
        ("11".into(), true)
    );
    // Deliberately supply the same revision to isolate current-graph checking from the
    // revision-mismatch guard. The root itself is locally compilable, its child is not.
    assert!(crate::compiler::is_compilable_function(
        new.get_function("probe").unwrap()
    ));
    assert_eq!(
        eval_compiled_or_interpreted_cached(&new, "probe", &input, Some(&cache), 1).unwrap(),
        ("22".into(), false)
    );
}

#[test]
fn jit_later_successful_revision_replaces_cache_after_failed_rebuild() {
    let input = parser::parse_single_expr("{}").unwrap();
    let cache = new_jit_cache();
    for (revision, value) in [(1, 11), (2, 22), (3, 33)] {
        let extra = if revision == 2 {
            "+fn nope ()->List<Int>\n  +return list()\n+end\n"
        } else {
            ""
        };
        let program = program(&format!(
            "+fn probe ()->Int\n  +return {value}\n+end\n{extra}"
        ));
        assert_eq!(
            eval_compiled_or_interpreted_cached(&program, "probe", &input, Some(&cache), revision)
                .unwrap(),
            (value.to_string(), revision != 2)
        );
    }
    assert!(matches!(&*cache.lock().unwrap(), Some((3, _))));
}

#[test]
fn jit_string_cache_failed_revision_also_returns_the_current_value() {
    let old = program("+fn probe ()->String\n  +return \"old\"\n+end\n");
    let new = program("+fn probe ()->String\n  +return \"new\"\n+end\n+fn nope ()->List<Int>\n  +return list()\n+end\n");
    let input = parser::parse_single_expr("{}").unwrap();
    let cache = new_jit_cache();
    assert_eq!(
        eval_compiled_or_interpreted_cached(&old, "probe", &input, Some(&cache), 1).unwrap(),
        ("\"old\"".into(), true)
    );
    assert_eq!(
        eval_compiled_or_interpreted_cached(&new, "probe", &input, Some(&cache), 2).unwrap(),
        ("\"new\"".into(), false)
    );
}
