//! Proven local-call qualification and runtime types in persisted Python source.
use std::collections::BTreeSet;

use super::{lower_python_module, print_python_module_with_ledger};
use crate::{ast, eval, parser, validator};

fn program(source: &str) -> ast::Program {
    let mut program = ast::Program::default();
    for operation in parser::parse(source).unwrap() {
        validator::apply_and_validate(&mut program, &operation).unwrap();
    }
    program.pending_routes.clear();
    program.rebuild_function_index();
    program
}

const LOCAL: &str = "+module QualM\n+fn g (x:Int)->Int\n  +return x + 10\n+end\n+fn f (x:Int)->Int\n  +let y:Int = g(x)\n  +return g(y)\n+end\n";

fn reload(text: &str, module: &str, catalog: &ast::Program) -> ast::Program {
    let mut without = catalog.clone();
    without.modules.retain(|m| m.name != module);
    without.rebuild_function_index();
    let lowered = lower_python_module(text, module, &without).unwrap();
    for operation in &lowered.declarations.operations {
        validator::apply_and_validate(&mut without, operation).unwrap();
    }
    without.rebuild_function_index();
    without
}

fn run(program: &ast::Program, function: &str, input: i64) -> eval::Value {
    let mut env = eval::Env::new_with_shared_interner(&program.shared_interner);
    eval::eval_call_inner_with_args(program, function, vec![eval::Value::Int(input)], &mut env)
        .unwrap()
}

#[test]
fn qualifies_a_unique_local_call_with_a_ledger() {
    let p = program(LOCAL);
    let (text, ledger) = print_python_module_with_ledger(&p, "QualM").unwrap();
    assert_eq!(ledger.names, BTreeSet::from(["g".to_string()]));
    assert_eq!(ledger.call_sites, 2);
    assert_eq!(ledger.catalog_unique, ledger.names);
    assert!(
        ledger.catalog_order_dependent.is_empty()
            && ledger.catalog_order_dependent_sites.is_empty()
    );
    assert!(text.contains("QualM.g("), "{text}");
}

const ORDERED: &str = "+module First
+fn api_base ()->String
  +return \"first\"
+end
+fn rooms ()->String
  +return api_base()
+end
+module Second
+fn api_base ()->String
  +return \"second\"
+end
+fn later ()->String
  +return api_base()
+end
";

#[test]
fn earlier_module_self_call_qualifies_and_is_classified_order_dependent() {
    let p = program(ORDERED);
    assert_eq!(p.modules[0].name, "First");
    let (text, ledger) = print_python_module_with_ledger(&p, "First").unwrap();
    assert!(text.contains("First.api_base()"), "{text}");
    assert!(ledger.catalog_unique.is_empty());
    assert_eq!(
        ledger.catalog_order_dependent,
        BTreeSet::from(["api_base".to_string()])
    );
    assert_eq!(
        ledger.catalog_order_dependent_sites,
        vec![("rooms".to_string(), "api_base".to_string())]
    );
    // Reload freezes today's observed binding: same value as before.
    let reloaded = reload(&text, "First", &p);
    let mut env = eval::Env::new_with_shared_interner(&reloaded.shared_interner);
    let value =
        eval::eval_call_inner_with_args(&reloaded, "First.rooms", vec![], &mut env).unwrap();
    assert!(matches!(value, eval::Value::String(ref s) if s.as_str() == "first"));
}

#[test]
fn later_module_call_bound_to_an_earlier_module_is_refused() {
    let p = program(ORDERED);
    // `Second.later` calls bare api_base, which today runs First.api_base;
    // qualifying it as Second.api_base would change behaviour.
    let error = print_python_module_with_ledger(&p, "Second").unwrap_err();
    assert!(
        error.message.contains("unqualified user call api_base"),
        "{}",
        error.message
    );
}

#[test]
fn reloaded_qualified_call_resolves_to_the_same_function() {
    let p = program(LOCAL);
    let (text, _) = print_python_module_with_ledger(&p, "QualM").unwrap();
    let reloaded = reload(&text, "QualM", &p);
    let f = reloaded.get_function("QualM.f").unwrap();
    let callees: Vec<_> = f
        .body
        .iter()
        .filter_map(|s| match &s.kind {
            ast::StatementKind::Let {
                value: ast::Expr::Call(c),
                ..
            } => Some(c.callee.clone()),
            ast::StatementKind::Return {
                value: ast::Expr::Call(c),
            } => Some(c.callee.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(callees, ["QualM.g", "QualM.g"]);
    assert_eq!(
        reloaded.get_function("QualM.g").unwrap(),
        p.get_function("QualM.g").unwrap()
    );
    assert!(matches!(run(&p, "QualM.f", 1), eval::Value::Int(21)));
    assert!(matches!(run(&reloaded, "QualM.f", 1), eval::Value::Int(21)));
}

#[test]
fn refuses_when_a_top_level_function_of_the_same_name_exists() {
    let p = program(&format!("+fn g (x:Int)->Int\n  +return x\n+end\n{LOCAL}"));
    let module = p.modules.iter().find(|m| m.name == "QualM").unwrap();
    let error = super::printer::module_body_qualifying(&p, module).unwrap_err();
    assert!(error.contains("unqualified user call g"), "{error}");
}

#[test]
fn refuses_when_an_earlier_module_defines_the_name() {
    let p = program(&format!(
        "+module Earlier\n+fn g (x:Int)->Int\n  +return x\n+end\n{LOCAL}"
    ));
    assert_eq!(p.modules[0].name, "Earlier");
    let error = print_python_module_with_ledger(&p, "QualM").unwrap_err();
    assert!(
        error.message.contains("unqualified user call g"),
        "{}",
        error.message
    );
}

#[test]
fn equality_is_strict_beyond_ledgered_qualification() {
    let p = program(LOCAL);
    let original = p.modules.iter().find(|m| m.name == "QualM").unwrap();
    let (text, ledger) = print_python_module_with_ledger(&p, "QualM").unwrap();
    let reloaded = reload(&text, "QualM", &p);
    let rebuilt = reloaded.modules.iter().find(|m| m.name == "QualM").unwrap();
    // Unnormalized original differs; ledgered normalization is exactly equal.
    assert!(!super::ast_equality::module(original, rebuilt));
    let (normalized, sites) = super::qualification::qualify_module(original, &ledger.names);
    assert_eq!(sites.len(), 2);
    assert!(super::ast_equality::module(&normalized, rebuilt));
    // A name outside the ledger is not rewritten, so the comparison fails.
    let (unledgered, none) = super::qualification::qualify_module(original, &BTreeSet::new());
    assert!(none.is_empty());
    assert!(!super::ast_equality::module(&unledgered, rebuilt));
    // Any other difference (here a literal) still fails the strict comparison.
    let mut tampered = normalized.clone();
    let g = std::sync::Arc::make_mut(
        tampered
            .functions
            .iter_mut()
            .find(|f| f.name == "g")
            .unwrap(),
    );
    let ast::StatementKind::Return {
        value: ast::Expr::Binary { right, .. },
    } = &mut g.body[0].kind
    else {
        panic!("fixture shape changed")
    };
    **right = ast::Expr::Literal(ast::Literal::Int(11));
    assert!(!super::ast_equality::module(&tampered, rebuilt));
}

#[test]
fn runtime_attachment_type_prints_and_roundtrips() {
    let p = program("+module AttM\n+fn keep (file:Attachment)->Attachment\n  +return file\n+end\n+fn many (files:List<Attachment>)->Int\n  +return len(files)\n+end\n");
    let (text, ledger) = print_python_module_with_ledger(&p, "AttM").unwrap();
    assert_eq!(ledger.call_sites, 0);
    assert!(text.contains("file: Attachment) -> Attachment"), "{text}");
    let reloaded = reload(&text, "AttM", &p);
    assert!(super::ast_equality::module(
        p.modules.iter().find(|m| m.name == "AttM").unwrap(),
        reloaded.modules.iter().find(|m| m.name == "AttM").unwrap(),
    ));
}

#[test]
fn undeclared_non_runtime_type_is_still_refused() {
    let p = program("+module AttM\n+fn keep (file:Gadget)->Gadget\n  +return file\n+end\n");
    let error = print_python_module_with_ledger(&p, "AttM").unwrap_err();
    assert!(
        error.message.contains("type Gadget is not declared"),
        "{}",
        error.message
    );
}
