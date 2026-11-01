use super::*;

fn corpus(source: &str) -> tempfile::TempDir {
    let temp = tempfile::tempdir().unwrap();
    std::fs::create_dir(temp.path().join("examples")).unwrap();
    std::fs::create_dir(temp.path().join("library")).unwrap();
    std::fs::write(
        temp.path().join("manifest.json"),
        r#"{"files":["examples/probe.ax"]}"#,
    )
    .unwrap();
    std::fs::write(temp.path().join("examples/probe.ax"), source).unwrap();
    temp
}

fn program(source: &str) -> ast::Program {
    let mut program = ast::Program::default();
    build_into(&mut program, &parser::parse(source).unwrap()).unwrap();
    program
}

#[test]
fn python_parity_inventory_fails_closed_on_missing_unlisted_empty_and_invalid_inputs() {
    let absent = tempfile::tempdir().unwrap();
    assert!(run(absent.path()).count(Status::InputError) > 0);
    for action in ["missing", "unlisted", "directory", "empty", "syntax"] {
        let fixture = corpus("+fn f ()->Int\n  +return 1\n+end\n");
        let path = fixture.path();
        match action {
            "missing" => std::fs::remove_file(path.join("examples/probe.ax")).unwrap(),
            "unlisted" => std::fs::write(path.join("examples/extra.ax"), "!done").unwrap(),
            "directory" => std::fs::remove_dir(path.join("library")).unwrap(),
            "empty" => std::fs::write(path.join("examples/probe.ax"), "").unwrap(),
            "syntax" => std::fs::write(path.join("examples/probe.ax"), "+fn ???").unwrap(),
            _ => unreachable!(),
        }
        let report = run(path);
        assert!(!report.success(), "{action}");
        assert!(report.count(Status::InputError) > 0, "{action}: {report:?}");
    }
}

#[test]
fn python_parity_inventories_top_level_functions_types_tests_and_commands() {
    let fixture = corpus(
        "+type Record = id:Int\n+fn f ()->Int\n  +return 1\n+end\n+test f\n  +with -> expect 1\n!eval f\n",
    );
    let report = run(fixture.path());
    // Module-less TYPES (A1), FUNCTIONS (A2), stored TESTS and whole PROGRAMS (A3/B2)
    // have a real (owner "") surface; only runtime commands stay open.
    for kind in ["type", "function", "Test"] {
        assert!(
            report
                .entries
                .iter()
                .any(|entry| entry.kind == kind && entry.status == Status::Verified),
            "{kind}: {report:?}"
        );
    }
    let program = report
        .entries
        .iter()
        .find(|entry| entry.kind == "program" && entry.file == "examples/probe.ax")
        .unwrap();
    assert_eq!(program.status, Status::Verified, "{report:?}");
    assert!(
        report
            .entries
            .iter()
            .any(|entry| entry.kind == "Eval" && entry.status != Status::Verified),
        "{report:?}"
    );
    assert!(!report.success());
}

#[test]
fn python_parity_keeps_module_only_loss_when_members_roundtrip() {
    let fixture = corpus(
        "+module Routes\n+fn f ()->Int\n  +return 1\n+end\n+route GET \"/a\" -> f\n+route GET \"/b\" -> f\n",
    );
    let report = run(fixture.path());
    assert!(
        report
            .entries
            .iter()
            .any(|e| e.kind == "function" && e.status == Status::Verified),
        "{report:?}"
    );
    assert!(
        report
            .entries
            .iter()
            .any(|e| e.kind == "module" && e.status != Status::Verified),
        "{report:?}"
    );
    assert!(
        report
            .entries
            .iter()
            .any(|e| e.kind == "Route" && e.status != Status::Verified),
        "{report:?}"
    );
}

#[test]
fn python_parity_catches_unqualified_call_normalization() {
    let fixture = corpus(
        "+module Calls\n+fn leaf ()->Int\n  +return 1\n+end\n+fn caller ()->Int\n  +return leaf()\n+end\n",
    );
    let report = run(fixture.path());
    assert!(
        report.entries.iter().any(|e| e.target == "Calls.caller"
            && e.status == Status::Unsupported
            && e.detail.contains("exact Python roundtrip")),
        "{report:?}"
    );
}

#[test]
fn python_parity_invalid_program_is_visible_never_counted_as_verified() {
    let fixture = corpus("+module Bad\n+fn f ()->Int\n  +return missing\n+end\n");
    let report = run(fixture.path());
    assert!(
        report
            .entries
            .iter()
            .any(|e| e.target == "Bad.f" && e.status == Status::InvalidProgram),
        "{report:?}"
    );
    assert_eq!(report.count(Status::Verified), 0);
}

#[test]
fn python_parity_typed_id_erasure_preserves_user_data_order_effects_and_metadata() {
    let source = "+module Exact\n+type Record = id:Int\n+shared slot:Int = 1\n+fn f (n:Int)->Record [io,async]\n  +let first:Int = 1\n  +let second:Int = 2\n  +return {id:n}\n+end\n+startup [io,async]\n  +await _:Int = sleep(1)\n+end\n";
    let original = program(source);
    let same = program(source);
    assert!(equality::program(&original, &same));
    for change in [
        "callee",
        "effects",
        "data-id",
        "order",
        "shared",
        "lifecycle",
        "docs",
    ] {
        let mut changed = same.clone();
        let f = changed.get_function_mut("Exact.f").unwrap();
        match change {
            "callee" => {
                f.body[2].kind = ast::StatementKind::Return {
                    value: ast::Expr::Call(ast::CallExpr {
                        callee: "Exact.other".into(),
                        args: vec![],
                    }),
                }
            }
            "effects" => f.effects.reverse(),
            "data-id" => {
                let ast::StatementKind::Return {
                    value: ast::Expr::StructInit { fields, .. },
                } = &mut f.body[2].kind
                else {
                    panic!()
                };
                fields[0].name = "changed_id".into();
            }
            "order" => f.body.swap(0, 1),
            "shared" => {
                changed.modules[0].shared_vars[0].default = ast::Expr::Literal(ast::Literal::Int(2))
            }
            "lifecycle" => changed.modules[0].startup.as_mut().unwrap().body.clear(),
            "docs" => f.doc = Some("changed".into()),
            _ => unreachable!(),
        }
        assert!(!equality::program(&original, &changed), "{change}");
    }
}

#[test]
fn python_parity_checks_second_print_bytes() {
    let program = program("+module Stable\n+fn f ()->Int\n  +return 1\n+end\n");
    assert!(check_function(&program, "Stable.f").is_ok());
    let first = super::super::print_function(&program, "Stable.f").unwrap();
    // An AST-preserving whitespace change is still non-idempotent output.
    assert_eq!(
        same_text(&first, &format!("{first}\n")).unwrap_err().0,
        Status::NonIdempotent
    );
}

#[test]
fn python_parity_does_not_certify_overwritten_declaration_from_final_state() {
    let fixture = corpus(
        "+module Repeat\n+fn f ()->Int\n  +return 1\n+end\n+fn f ()->Int\n  +return 2\n+end\n",
    );
    let report = run(fixture.path());
    let functions: Vec<_> = report
        .entries
        .iter()
        .filter(|e| e.target == "Repeat.f")
        .collect();
    assert_eq!(functions.len(), 2);
    assert!(
        functions.iter().all(|e| e.status != Status::Verified),
        "{report:?}"
    );
}

#[test]
fn python_parity_equality_retains_source_id_configuration_module_docs_and_routes() {
    let mut original = program("+module Metadata\n+fn f ()->Int\n +return 1\n+end\n");
    original.modules[0].doc = Some("original doc".into());
    original.modules[0].routes.push(ast::HttpRoute {
        method: "GET".into(),
        path: "/original".into(),
        handler_fn: "Metadata.f".into(),
    });
    original.modules[0].sources.push(ast::SourceDecl {
        name: "clock".into(),
        source_type: "timer".into(),
        config: vec![("id".into(), "original-id-value".into())],
        handler: "f".into(),
    });
    for change in ["source-id", "module-doc", "route"] {
        let mut altered = original.clone();
        match change {
            "source-id" => altered.modules[0].sources[0].config[0].1 = "changed-id-value".into(),
            "module-doc" => altered.modules[0].doc = Some("changed doc".into()),
            "route" => altered.modules[0].routes[0].path = "/changed".into(),
            _ => unreachable!(),
        }
        assert!(
            !equality::module(&original.modules[0], &altered.modules[0]),
            "{change}"
        );
        assert!(!equality::program(&original, &altered), "{change}");
    }
    let mut ids_only = original.clone();
    ids_only.modules[0].id = "new-actual-node-id".into();
    assert!(equality::program(&original, &ids_only));
}

fn test_entries(source: &str) -> Vec<(Status, String)> {
    let fixture = corpus(source);
    run(fixture.path())
        .entries
        .into_iter()
        .filter(|entry| entry.kind == "Test")
        .map(|entry| (entry.status, entry.detail))
        .collect()
}

const MODULE_F: &str = "+module T\n+fn f (n:Int)->Int\n  +return n\n+end\n";

#[test]
fn python_parity_module_test_blocks_are_compared_as_cases_not_skipped() {
    // Literal, matcher and keyword forms roundtrip; one entry per block.
    let entries = test_entries(&format!(
        "{MODULE_F}+test f\n  +with 1 -> expect 1\n  +with n=2 -> expect 2\n  +with 3 -> expect contains(\"3\")\n"
    ));
    assert_eq!(entries, [(Status::Verified, String::new())]);
    // One inexpressible case keeps the WHOLE block unsupported and names the case.
    let entries = test_entries(&format!(
        "{MODULE_F}+test f\n  +with 1 -> expect 1\n  +with Some(1) -> expect 1\n"
    ));
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0, Status::Unsupported);
    assert!(
        entries[0].1.starts_with("case 2: call to Some"),
        "{entries:?}"
    );
    // Module-less blocks use the owner-"" surface (B2).
    let entries =
        test_entries("+fn f (n:Int)->Int\n  +return n\n+end\n+test f\n  +with 1 -> expect 1\n");
    assert_eq!(entries, [(Status::Verified, String::new())]);
}

#[test]
fn python_parity_test_block_target_rules_follow_the_session() {
    // A test for a function no scope declares is a source defect.
    let entries = test_entries(&format!("{MODULE_F}+test missing\n  +with 1 -> expect 1\n"));
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0, Status::InvalidProgram);
    assert!(
        entries[0].1.contains("undeclared function `missing`"),
        "{entries:?}"
    );
    // session.rs runs a bare module test against ANY function of that name (the global
    // first match), so a later module's same-named function is not what the
    // module-qualified Python lowering targets.
    let entries = test_entries(&format!(
        "{MODULE_F}+module U\n+fn f (n:Int)->Int\n  +return n\n+end\n+test f\n  +with 1 -> expect 1\n"
    ));
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].0, Status::Unsupported);
    assert!(
        entries[0]
            .1
            .contains("runs f, but the module-qualified Python lowering targets U.f"),
        "{entries:?}"
    );
    // A top-level function of the same name captures the bare test name the same way.
    let entries = test_entries(&format!(
        "+fn f (n:Int)->Int\n  +return n\n+end\n{MODULE_F}+test f\n  +with 1 -> expect 1\n"
    ));
    assert!(
        entries
            .iter()
            .any(|(status, detail)| *status == Status::Unsupported
                && detail.contains("runs f, but the module-qualified Python lowering targets T.f")),
        "{entries:?}"
    );
    // A qualified target is exact.
    let entries = test_entries(&format!("{MODULE_F}+test T.f\n  +with 1 -> expect 1\n"));
    assert_eq!(entries, [(Status::Verified, String::new())]);
}

fn type_entries(source: &str) -> Vec<(String, Status, String)> {
    let fixture = corpus(source);
    run(fixture.path())
        .entries
        .into_iter()
        .filter(|entry| entry.kind == "type")
        .map(|entry| (entry.target, entry.status, entry.detail))
        .collect()
}

#[test]
fn python_parity_top_level_types_use_the_empty_owner_surface() {
    // Struct, union, recursive union and a cross-reference all roundtrip.
    let entries = type_entries(
        "+type Point = x:Int, y:Int\n+type Line = start:Point, end:Point\n+type IntList = Cons(Int, IntList) | Nil\n+type Option = Some(Int) | None\n",
    );
    assert_eq!(entries.len(), 4);
    assert!(
        entries
            .iter()
            .all(|(_, status, _)| *status == Status::Verified),
        "{entries:?}"
    );
    // Unprintable member names stay Unsupported with the frontend's own reason.
    let entries = type_entries("+type Env = vars:Int, other:Int\n");
    assert_eq!(entries[0].1, Status::Unsupported);
    assert!(
        entries[0].2.contains("unsupported member name \"vars\""),
        "{entries:?}"
    );
    // A top-level type colliding with a module is reported, never approximated. The
    // harness's own repeated-target rule fires first (module and type share the
    // target spelling); the lowering-level rule is covered in top_level_tests.
    let entries =
        type_entries("+type Geo = x:Int\n+module Geo\n+fn f ()->Int\n  +return 1\n+end\n");
    let geo = entries.iter().find(|(target, ..)| target == "Geo").unwrap();
    assert_eq!(geo.1, Status::Unsupported);
    assert!(geo.2.contains("repeated declaration"), "{entries:?}");
    // Type/variant collisions across top-level types are reported on BOTH sides.
    let entries = type_entries("+type Shape = Circle(Int) | Dot\n+type Circle = r:Int\n");
    assert!(
        entries
            .iter()
            .all(|(_, status, detail)| *status == Status::Unsupported
                && detail.contains("collides with a")),
        "{entries:?}"
    );
    // Module types are untouched by the top-level path.
    let entries = type_entries("+module M\n+type P = x:Int\n+fn f ()->Int\n  +return 1\n+end\n");
    assert_eq!(entries[0].0, "M.P");
}

fn kind_entries(source: &str, kind: &str) -> Vec<(String, Status, String)> {
    let fixture = corpus(source);
    run(fixture.path())
        .entries
        .into_iter()
        .filter(|entry| entry.kind == kind && entry.file == "examples/probe.ax")
        .map(|entry| (entry.target, entry.status, entry.detail))
        .collect()
}

const TOP_F: &str = "+fn f (n:Int)->Int\n  +return n\n+end\n";

#[test]
fn python_parity_top_level_test_blocks_keep_the_runtime_target_string() {
    // Literal, keyword, matcher and Err forms; one entry per block.
    let source = format!(
        "{TOP_F}+test f\n  +with 1 -> expect 1\n  +with n=2 -> expect 2\n  +with {{n: -3}} -> expect -3\n  +with 4 -> expect contains(\"4\")\n"
    );
    assert_eq!(kind_entries(&source, "Test").len(), 1);
    assert_eq!(kind_entries(&source, "Test")[0].1, Status::Verified);
    // One inexpressible case keeps the whole block unsupported and names the case.
    let entries = kind_entries(
        &format!("{TOP_F}+test f\n  +with Some(1) -> expect 1\n"),
        "Test",
    );
    assert_eq!(entries[0].1, Status::Unsupported);
    assert!(
        entries[0].2.starts_with("case 1: call to Some"),
        "{entries:?}"
    );
    // A test for a function nobody declares is a source defect.
    let entries = kind_entries(
        &format!("{TOP_F}+test missing\n  +with 1 -> expect 1\n"),
        "Test",
    );
    assert_eq!(entries[0].1, Status::InvalidProgram);
    assert!(
        entries[0].2.contains("undeclared function `missing`"),
        "{entries:?}"
    );
    // A dotted target stays dotted; a bare target that only a MODULE defines is the
    // string the runtime resolves through Program::get_function, so it is exact too.
    let module = "+module T\n+fn g (n:Int)->Int\n  +return n\n+end\n";
    for target in ["T.g", "g"] {
        let entries = kind_entries(
            &format!("{module}+test {target}\n  +with 1 -> expect 1\n"),
            "Test",
        );
        assert_eq!(entries[0].1, Status::Verified, "{target}: {entries:?}");
    }
    // The session-rule refusal (#110) is MODULE-scoped and unchanged: a module block
    // `+test g` runs a same-named TOP-LEVEL `g`, while the module lowering targets T.g.
    let source =
        format!("+fn g (n:Int)->Int\n  +return n\n+end\n{module}+test g\n  +with 1 -> expect 1\n");
    let entries = kind_entries(&source, "Test");
    assert_eq!(entries[0].1, Status::Unsupported, "{entries:?}");
    assert!(entries[0].2.contains("(#110)"), "{entries:?}");
}

#[test]
fn python_parity_keyword_shorthand_negatives_are_unsupported_never_mismatched() {
    // #111: `name=-1` parses to Int(-1); Python's `-1` lowers to Unary Neg.
    for case in ["n=-1", "n=-1.5", "n=-0.0"] {
        let source = format!("{TOP_F}+test f\n  +with {case} -> expect 1\n");
        let entries = kind_entries(&source, "Test");
        assert_eq!(entries[0].1, Status::Unsupported, "{case}: {entries:?}");
        assert!(entries[0].2.contains("(#111)"), "{case}: {entries:?}");
        // Module-scoped blocks share the printer, so they are refused the same way.
        let source = format!("{MODULE_F}+test f\n  +with {case} -> expect 1\n");
        let entries = kind_entries(&source, "Test");
        assert_eq!(entries[0].1, Status::Unsupported, "{case}: {entries:?}");
    }
    // Every other spelling of a negative stays exact.
    for case in [
        "{n: -1}",
        "-1",
        "{_0: -1, _1: 2}",
        "n=list(-1)",
        "{n: -1.5}",
    ] {
        let source = format!("{TOP_F}+test f\n  +with {case} -> expect -2\n");
        let entries = kind_entries(&source, "Test");
        assert_eq!(entries[0].1, Status::Verified, "{case}: {entries:?}");
    }
}

#[test]
fn python_parity_whole_programs_roundtrip_top_level_and_module_declarations() {
    // Types + mutually recursive functions + a module using a top-level type.
    let source = "+type Point = x:Int, y:Int\n+fn is_even (n:Int)->Bool\n  +if n == 0\n    +return true\n  +end\n  +return is_odd(n - 1)\n+end\n+fn is_odd (n:Int)->Bool\n  +if n == 0\n    +return false\n  +end\n  +return is_even(n - 1)\n+end\n+module Geo\n+fn origin ()->Point\n  +return {x: 0, y: 0}\n+end\n";
    let entries = kind_entries(source, "program");
    assert_eq!(
        entries,
        [(String::new(), Status::Verified, String::new())],
        "{entries:?}"
    );
    // Top-level only, and module only (unchanged behavior), also verify.
    let entries = kind_entries(&format!("{TOP_F}"), "program");
    assert_eq!(entries[0].1, Status::Verified, "{entries:?}");
    let entries = kind_entries(MODULE_F, "program");
    assert_eq!(entries[0].1, Status::Verified, "{entries:?}");
}

#[test]
fn python_parity_whole_program_reports_the_first_unprintable_member() {
    // An unprintable top-level member keeps the program Unsupported with ITS reason.
    let entries = kind_entries("+fn f (eval:Int)->Int\n  +return eval\n+end\n", "program");
    assert_eq!(entries[0].1, Status::Unsupported);
    assert!(entries[0].2.contains("top-level function f"), "{entries:?}");
    let entries = kind_entries("+type Env = vars:Int, other:Int\n", "program");
    assert_eq!(entries[0].1, Status::Unsupported);
    assert!(entries[0].2.contains("top-level type Env"), "{entries:?}");
    // A module that calls a top-level function bare is still unprintable: only owner ""
    // may print a bare user call.
    let entries = kind_entries(
        "+fn leaf (n:Int)->Int\n  +return n\n+end\n+module M\n+fn g (n:Int)->Int\n  +return leaf(n)\n+end\n",
        "program",
    );
    assert_eq!(entries[0].1, Status::Unsupported);
    assert!(
        entries[0].2.contains("unqualified user call leaf"),
        "{entries:?}"
    );
}

#[test]
fn python_parity_lowered_test_target_must_equal_the_runtime_string() {
    let text = "def test_case_1() -> None:\n    assert f(1) == 1\n";
    assert!(test_blocks::lower(text, "", "f").is_ok());
    assert!(test_blocks::lower(text, "M", "M.f").is_ok());
    for (scope, wrong) in [("", "g"), ("", "M.f"), ("M", "f"), ("M", "N.f")] {
        let err = test_blocks::lower(text, scope, wrong).unwrap_err();
        assert_eq!(err.0, Status::AstMismatch, "{scope}/{wrong}");
        assert!(err.1.contains("differs from"), "{err:?}");
    }
}

#[test]
fn python_parity_whole_program_text_is_stable_across_a_rebuild() {
    let source = "+type Point = x:Int, y:Int\n+fn a (n:Int)->Int\n  +return b(n)\n+end\n+fn b (n:Int)->Int\n  +return n\n+end\n+module M\n+fn g (n:Int)->Int\n  +return n\n+end\n";
    let original = program(source);
    let (types, functions, text) = top_level::lower_all(&original).unwrap();
    assert!(text.contains("class Point") && text.contains("def a(") && text.contains("def b("));
    assert!(text.find("class Point").unwrap() < text.find("def a(").unwrap());
    let mut rebuilt = ast::Program::default();
    build_into(&mut rebuilt, &types).unwrap();
    for module in &original.modules {
        let printed = super::super::print_module(&original, &module.name).unwrap();
        let lowered =
            super::super::lower_module_with_program(&printed, &module.name, &original).unwrap();
        build_into(&mut rebuilt, &lowered.operations).unwrap();
    }
    build_into(&mut rebuilt, &functions).unwrap();
    assert!(equality::program(&original, &rebuilt));
    assert_eq!(text, top_level::print_all(&rebuilt).unwrap());
    // The empty surface prints and lowers to nothing (module-only programs).
    let (types, functions, text) = top_level::lower_all(&program(MODULE_F)).unwrap();
    assert!(types.is_empty() && functions.is_empty() && text.is_empty());
}

#[test]
fn python_parity_input_parameter_functions_verify_at_both_scopes() {
    // G2: `input` is a legal binding name, so these roundtrip exactly.
    let top = "+fn size (input:String)->Int\n  +let n:Int = len(input)\n  +return n\n+end\n";
    let entries = kind_entries(top, "function");
    assert_eq!(entries[0].1, Status::Verified, "{entries:?}");
    let module = format!("+module M\n{top}");
    let entries = kind_entries(&module, "function");
    assert_eq!(entries[0].1, Status::Verified, "{entries:?}");
    let entries = kind_entries(&module, "program");
    assert_eq!(entries[0].1, Status::Verified, "{entries:?}");
}
