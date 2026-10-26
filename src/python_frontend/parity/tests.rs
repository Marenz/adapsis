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
    for kind in ["type", "function", "Test", "Eval", "program"] {
        assert!(
            report
                .entries
                .iter()
                .any(|entry| entry.kind == kind && entry.status != Status::Verified),
            "{kind}: {report:?}"
        );
    }
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
