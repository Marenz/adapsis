//! The stored-source nesting limit (250) applies only inside its scope: text
//! the runtime printed and managed files it reads back. Submissions keep 100.
use super::nesting::{current_limit, StoredSource};
use super::{MAX_NESTING, MAX_STORED_NESTING};
use crate::ast::Program;

fn chain(operators: usize) -> String {
    format!("def f(x: int) -> int:\n    return x{}\n", " + 1".repeat(operators))
}

fn refused_as_submission(source: &str) {
    for refusal in [
        super::lower_module(source, "Sub").map(|_| ()).unwrap_err(),
        super::preflight_module(source, "Sub").map(|_| ()).unwrap_err(),
        super::lower_module_with_program(source, "Sub", &Program::default()).map(|_| ()).unwrap_err(),
        super::lower_toplevel(source, &Program::default()).map(|_| ()).unwrap_err(),
    ] {
        assert!(refusal.message.contains("nesting too deep: more than 100 levels"), "{}", refusal.message);
    }
}

#[test]
fn stored_source_limit_is_scoped_and_submissions_keep_the_submission_limit() {
    assert_eq!((MAX_NESTING, MAX_STORED_NESTING), (100, 250));
    assert_eq!(current_limit(), MAX_NESTING);
    // 149 operators: 150 levels. Stored source loads; the same text is refused
    // as a submission before and after, on the same thread.
    let deep = chain(149);
    refused_as_submission(&deep);
    let loaded = super::lower_python_module(&deep, "Stored", &Program::default()).unwrap();
    assert_eq!(loaded.declarations.operations.len(), 1);
    assert_eq!(current_limit(), MAX_NESTING, "the scope ended with the call");
    refused_as_submission(&deep);
    refused_as_submission(&chain(100));
    assert!(super::lower_module(&chain(99), "Sub").is_ok());
    // Stored source has its own limit, with its own number in the message.
    let refusal = super::lower_python_module(&chain(250), "Stored", &Program::default()).unwrap_err();
    assert!(refusal.message.contains("nesting too deep: more than 250 levels"), "{}", refusal.message);
    assert!(super::lower_python_module(&chain(249), "Stored", &Program::default()).is_ok());
    // A refused stored load restores the limit too.
    assert_eq!(current_limit(), MAX_NESTING);
    refused_as_submission(&deep);
}

#[test]
fn stored_source_scope_nests_and_is_restored_when_unwinding() {
    assert_eq!(current_limit(), MAX_NESTING);
    {
        let _outer = StoredSource::enter();
        assert_eq!(current_limit(), MAX_STORED_NESTING);
        {
            let _inner = StoredSource::enter();
            assert_eq!(current_limit(), MAX_STORED_NESTING);
        }
        assert_eq!(current_limit(), MAX_STORED_NESTING, "the inner scope must not end the outer one");
    }
    assert_eq!(current_limit(), MAX_NESTING);
    let panicked = std::panic::catch_unwind(|| {
        let _scope = StoredSource::enter();
        panic!("inside the stored-source scope");
    });
    assert!(panicked.is_err());
    assert_eq!(current_limit(), MAX_NESTING, "unwinding restored the submission limit");
    refused_as_submission(&chain(100));
}

#[test]
fn stored_source_scope_is_per_thread() {
    let _scope = StoredSource::enter();
    let other = std::thread::spawn(current_limit).join().unwrap();
    assert_eq!(other, MAX_NESTING);
    assert_eq!(current_limit(), MAX_STORED_NESTING);
}

/// What the stored limit is derived from: the printed form of a function at
/// the submission limit, per shape.
#[test]
fn stored_source_limit_covers_the_printed_form_of_every_accepted_shape() {
    let n = MAX_NESTING;
    let mut blocks = String::from("def f(x: bool) -> int:\n");
    for level in 0..n - 1 {
        blocks.push_str(&format!("{}if x:\n", " ".repeat(level + 1)));
    }
    blocks.push_str(&format!("{}return 7\n return 0\n", " ".repeat(n)));
    let mut elif = String::from("def f(x: int) -> int:\n    if x == 0:\n        return 0\n");
    for branch in 1..=n - 2 {
        elif.push_str(&format!("    elif x == {branch}:\n        return {branch}\n"));
    }
    elif.push_str("    return -1\n");
    let mixed = format!("def g(x: int) -> int:\n    return x + 1\n\ndef f(x: int) -> int:\n    return {}x{}\n", "g(".repeat(48), " + 1)".repeat(48));
    let shapes = [
        ("operators", chain(n - 1)),
        ("calls", format!("def g(x: int) -> int:\n    return x + 1\n\ndef f(x: int) -> int:\n    return {}x{}\n", "g(".repeat(n - 1), ")".repeat(n - 1))),
        ("calls and operators", mixed),
        ("parentheses", format!("def f(x: int) -> int:\n    return {}x{}\n", "(".repeat(n - 1), ")".repeat(n - 1))),
        ("not", format!("def f(x: bool) -> bool:\n    return {}x\n", "not ".repeat(2 * ((n - 1) / 2)))),
        ("negation", format!("def f(x: int) -> int:\n    return {}x\n", "-".repeat(n - 1))),
        ("field access", format!("@dataclass\nclass Node:\n    next: Node\n    value: int\n\ndef f(x: Node) -> int:\n    return x{}.value\n", ".next".repeat(n - 2))),
        ("blocks", blocks),
        ("elif", elif),
    ];
    let mut worst = 0;
    for (name, source) in shapes {
        let lowered = super::lower_module(&source, "Printed").unwrap_or_else(|error| panic!("{name}: {error}"));
        let mut program = Program::default();
        for operation in &lowered.operations {
            crate::validator::apply_and_validate(&mut program, operation).unwrap();
        }
        let printed = super::printed_nesting_for_tests(&program, &program.modules[0]).unwrap();
        eprintln!("STORED-LIMIT printed form of `{name}` at the submission limit scans at {printed}");
        worst = worst.max(printed);
        // And the checked print (print, lower again, compare) accepts it.
        super::print_python_module(&program, "Printed").unwrap_or_else(|error| panic!("{name}: {error}"));
    }
    assert!(worst > MAX_NESTING, "the printed form is deeper than the submission ({worst})");
    assert!(worst * 5 <= MAX_STORED_NESTING * 4, "{worst} needs at least a fifth of headroom below {MAX_STORED_NESTING}");
}
