use super::*;
use crate::{parser, validator, vm};

mod dispatch;

fn program(source: &str) -> ast::Program {
    let mut program = ast::Program::default();
    for operation in parser::parse(source).unwrap() {
        validator::apply_and_validate(&mut program, &operation).unwrap();
    }
    program.rebuild_function_index();
    program
}

fn statement(kind: ast::StatementKind) -> ast::Statement {
    ast::Statement {
        id: "admission-fixture".into(),
        kind,
    }
}

fn danger() -> ast::Expr {
    ast::Expr::Call(ast::CallExpr {
        callee: "danger".into(),
        args: vec![],
    })
}

fn call(callee: &str) -> ast::CallExpr {
    ast::CallExpr {
        callee: callee.into(),
        args: vec![danger()],
    }
}

fn statement_cases() -> Vec<ast::StatementKind> {
    use ast::{Expr as E, StatementKind as S};
    let body = || vec![statement(S::Return { value: danger() })];
    let binding = || ast::Binding {
        name: "value".into(),
        ty: ast::Type::Int,
    };
    // Some constructed containers intentionally aren't fully typed programs:
    // assert the admission marker specifically, before another compiler refusal
    // could hide an omitted executable slot in the visitor.
    vec![
        S::Let {
            name: "x".into(),
            ty: ast::Type::Int,
            value: danger(),
        },
        S::Set {
            name: "x".into(),
            value: danger(),
        },
        S::Return { value: danger() },
        S::Yield { value: danger() },
        S::Call {
            binding: Some(binding()),
            call: call("outer"),
        },
        S::Spawn {
            binding: None,
            call: call("outer"),
        },
        S::Await {
            name: "x".into(),
            ty: ast::Type::Int,
            call: call("read_file"),
        },
        S::Check {
            label: "c".into(),
            condition: danger(),
            on_fail: "failed".into(),
        },
        S::Branch {
            condition: danger(),
            then_body: vec![],
            else_body: vec![],
        },
        S::Branch {
            condition: E::Literal(ast::Literal::Bool(false)),
            then_body: body(),
            else_body: vec![],
        },
        S::Branch {
            condition: E::Literal(ast::Literal::Bool(true)),
            then_body: vec![],
            else_body: body(),
        },
        S::While {
            condition: danger(),
            body: vec![],
        },
        S::While {
            condition: E::Literal(ast::Literal::Bool(false)),
            body: body(),
        },
        S::Each {
            iterator: danger(),
            binding: binding(),
            body: vec![],
        },
        S::Each {
            iterator: E::Literal(ast::Literal::Int(0)),
            binding: binding(),
            body: body(),
        },
        S::Match {
            expr: danger(),
            arms: vec![],
        },
        S::Match {
            expr: E::Literal(ast::Literal::Int(0)),
            arms: vec![ast::MatchArm {
                variant: "_".into(),
                bindings: vec![],
                patterns: None,
                body: body(),
            }],
        },
    ]
}

fn registration_cases() -> Vec<ast::StatementKind> {
    use ast::StatementKind as S;
    vec![
        S::Source(ast::SourceOp::Add {
            source_type: ast::SourceType::Timer(Box::new(danger())),
            alias: "s".into(),
            handler: "later".into(),
        }),
        S::Source(ast::SourceOp::Replace {
            source_type: ast::SourceType::Timer(Box::new(danger())),
            alias: "s".into(),
            handler: "later".into(),
        }),
        S::Event(ast::EventOp::Emit {
            name: "e".into(),
            value: Box::new(danger()),
        }),
    ]
}

#[test]
fn vm_user_await_exhaustive_executable_containers_preflight() {
    use ast::{Expr as E, StatementKind as S};
    let base = program("+fn leaf ()->Int [async]\n  +return 1\n+end\n+fn danger ()->Int [async]\n  +await value:Int = leaf()\n  +return value\n+end\n+fn root ()->Int [io,async]\n  +return 0\n+end\n");
    for (i, kind) in statement_cases()
        .into_iter()
        .chain(registration_cases())
        .enumerate()
    {
        let mut program = base.clone();
        program.get_function_mut("root").unwrap().body = vec![statement(kind)];
        let error =
            vm::compile_function(program.get_function("root").unwrap(), &program).unwrap_err();
        assert!(
            error.to_string().contains(REQUIRES_TREE),
            "container {i}: {error}"
        );
    }
    for expression in [
        E::Binary {
            left: Box::new(danger()),
            op: ast::BinaryOp::Add,
            right: Box::new(E::Literal(ast::Literal::Int(1))),
        },
        E::Binary {
            left: Box::new(E::Literal(ast::Literal::Int(1))),
            op: ast::BinaryOp::Add,
            right: Box::new(danger()),
        },
        E::Unary {
            op: ast::UnaryOp::Not,
            expr: Box::new(danger()),
        },
        E::FieldAccess {
            base: Box::new(danger()),
            field: "field".into(),
        },
        E::StructInit {
            ty: "Record".into(),
            fields: vec![ast::StructFieldValue {
                name: "field".into(),
                value: danger(),
            }],
        },
        E::Call(call("outer")),
    ] {
        let mut program = base.clone();
        program.get_function_mut("root").unwrap().body =
            vec![statement(S::Return { value: expression })];
        assert!(analyze(program.get_function("root").unwrap(), &program).requires_tree);
    }
}

#[test]
fn vm_user_await_unrelated_graph_unknown_and_builtin_controls() {
    let program = program("+fn leaf ()->Int [async]\n  +return 1\n+end\n+fn unrelated ()->Int [async]\n  +await value:Int = leaf()\n  +return value\n+end\n+fn root ()->String [io,async]\n  +await value:String = read_file(\"fixture\")\n  +return value\n+end\n+fn unknown ()->Int [async]\n  +await value:Int = unknown_target()\n  +return value\n+end\n");
    for name in ["root", "unknown"] {
        let result = analyze(program.get_function(name).unwrap(), &program);
        assert!(!result.requires_tree);
        assert_eq!(result.functions, 1);
        assert!(vm::compile_function(program.get_function(name).unwrap(), &program).is_ok());
    }
}

#[test]
fn vm_user_await_root_owned_outside_program_is_still_scanned() {
    let catalog = program("+fn leaf ()->Int [async]\n  +return 1\n+end\n");
    let owner = program("+fn unregistered ()->Int [async]\n  +await value:Int = leaf()\n  +return value\n+end\n+fn builtin_only ()->String [io,async]\n  +await value:String = read_file(\"fixture\")\n  +return value\n+end\n");
    assert!(catalog.get_function("unregistered").is_none());
    let root = owner.get_function("unregistered").unwrap();
    assert!(analyze(root, &catalog).requires_tree);
    assert!(vm::compile_function(root, &catalog)
        .unwrap_err()
        .to_string()
        .contains(REQUIRES_TREE));
    assert!(vm::compile_function(owner.get_function("builtin_only").unwrap(), &catalog).is_ok());
}

#[test]
fn vm_user_await_scc_scan_measurement_no_cross_revision_cache() {
    let mut source = String::new();
    for i in 0..128 {
        source.push_str(&format!(
            "+fn f{i} ()->Int\n  +let x:Int = f{}()\n  +return f{}()\n+end\n",
            (i + 1) % 128,
            (i + 1) % 128
        ));
    }
    let mut program = program(&source);
    let first = analyze(program.get_function("f0").unwrap(), &program);
    assert!(!first.requires_tree);
    assert_eq!(first.functions, 128);
    let started = std::time::Instant::now();
    for _ in 0..1000 {
        std::hint::black_box(analyze(program.get_function("f0").unwrap(), &program));
    }
    println!(
        "AWAIT_ADMISSION_MEASURE functions={} work_items={} scans=1000 elapsed_us={}",
        first.functions,
        first.nodes,
        started.elapsed().as_micros()
    );
    // A new snapshot with a reachable user await must not reuse the old decision.
    let mut changed = program.clone();
    changed
        .get_function_mut("f64")
        .unwrap()
        .body
        .push(statement(ast::StatementKind::Await {
            name: "later".into(),
            ty: ast::Type::Int,
            call: ast::CallExpr {
                callee: "f65".into(),
                args: vec![],
            },
        }));
    assert!(analyze(changed.get_function("f0").unwrap(), &changed).requires_tree);
    assert!(!analyze(program.get_function("f0").unwrap(), &program).requires_tree);
    program.get_function_mut("f64").unwrap().body =
        changed.get_function("f64").unwrap().body.clone();
    assert!(analyze(program.get_function("f0").unwrap(), &program).requires_tree);
}
