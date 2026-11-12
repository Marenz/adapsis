use super::*;
use crate::compiler::admission_measurement_tests::program;
use crate::{eval, parser};

/// Function bodies that consume the length of a string expression that is not
/// a literal, a `concat(...)` or a String-returning user call.
const REFUSED: &[(&str, &str)] = &[
    // The reviewer's witness: returned "" before the rule.
    ("struct field returned", "+type Named = {label:String, n:Int}\n+fn f (p:Named)->String\n  +return p.label\n+end\n"),
    ("struct field as concat operand", "+type Named = {label:String, n:Int}\n+fn f (p:Named)->String\n  +return concat(p.label, \"!\")\n+end\n"),
    ("struct field compared with a literal", "+type Named = {label:String, n:Int}\n+fn f (p:Named)->Int\n  +return p.label == \"a\"\n+end\n"),
    ("two string calls compared (pointer comparison)", "+fn g ()->String\n  +return \"a\"\n+end\n+fn f ()->Int\n  +return g() == g()\n+end\n"),
    ("concat compared with a string call", "+fn g ()->String\n  +return \"a\"\n+end\n+fn f ()->Int\n  +return concat(\"a\", \"b\") != g()\n+end\n"),
    ("comparison of comparisons containing literals", "+fn f (n:Int)->Int\n  +return (n == 1) == (\"a\" == \"b\")\n+end\n"),
    ("variable passed to a String parameter", "+fn g (s:String)->Int\n  +return 1\n+end\n+fn f (s:String)->Int\n  +return g(s)\n+end\n"),
    ("struct field passed to a String parameter", "+type Named = {label:String, n:Int}\n+fn g (s:String)->Int\n  +return 1\n+end\n+fn f (p:Named)->Int\n  +return g(p.label)\n+end\n"),
    ("integer expression returned as String", "+fn f (n:Int)->String\n  +return n\n+end\n"),
    ("parameter returned", "+fn f (s:String)->String\n  +return s\n+end\n"),
    ("parameter as concat operand", "+fn f (name:String)->String\n  +return concat(\"hello \", name)\n+end\n"),
    ("parameters as both concat operands", "+fn f (a:String, b:String)->String\n  +return concat(a, b)\n+end\n"),
    ("parameter compared with a literal", "+fn f (s:String)->Int\n  +return s == \"a\"\n+end\n"),
    ("two parameters compared", "+fn f (a:String, b:String)->Int\n  +return a != b\n+end\n"),
    ("binding returned", "+fn f ()->String\n  +let s:String = \"text\"\n  +return s\n+end\n"),
    ("binding returned right after concat", "+fn f ()->String\n  +let s:String = concat(\"a\", \"b\")\n  +return s\n+end\n"),
    ("earlier binding returned", "+fn f ()->String\n  +let a:String = \"x\"\n  +let b:String = \"yyyy\"\n  +return a\n+end\n"),
    ("binding as concat operand", "+fn f ()->String\n  +let a:String = \"x\"\n  +return concat(a, \"yyyy\")\n+end\n"),
    ("+call binding returned", "+fn g ()->String\n  +return \"text\"\n+end\n+fn f ()->String\n  +call s:String = g()\n  +return s\n+end\n"),
    ("concat in a +call statement", "+fn f (a:String)->Int\n  +call s:String = concat(a, \"b\")\n  +return 1\n+end\n"),
    ("inside a branch", "+fn f (s:String, n:Int)->String\n  +if n > 0\n    +return s\n  +end\n  +return \"none\"\n+end\n"),
    ("inside a loop body", "+fn f (s:String, n:Int)->String\n  +while n > 0\n    +return concat(s, \"!\")\n  +end\n  +return \"none\"\n+end\n"),
    ("inside a match arm", "+fn f (s:String, n:Int)->String\n  +match n\n  +case 1\n    +return s\n  +case _\n    +return \"other\"\n  +end\n+end\n"),
    ("nested in an argument", "+fn f (s:String)->String\n  +return concat(concat(\"a\", s), \"c\")\n+end\n"),
    ("alias of a parameter", "+fn f (s:String)->String\n  +let t:String = s\n  +return t\n+end\n"),
];

#[test]
fn jit_string_variable_length_reads_are_refused_before_emission() {
    for (label, source) in REFUSED {
        let program = program(source);
        let function = program.get_function("f").unwrap();
        assert!(consumes_unknown_string_length(function, &program), "{label}: not detected");
        let error = match crate::compiler::compile(&program) {
            Ok(_) => panic!("{label}: compiled"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains(REFUSAL), "{label}: refused for another reason: {error}");
        assert!(crate::compiler::failure_admission::check_root(function, &program).is_err(), "{label}: admitted as a cached root");
    }
}

#[test]
fn jit_string_shapes_with_proven_lengths_stay_compiled() {
    for (label, source, call, expected) in [
        ("literal returned", "+fn f ()->String\n  +return \"text\"\n+end\n", "f", "text"),
        ("concat of literals", "+fn f ()->String\n  +return concat(\"a\", concat(\"b\", \"c\"))\n+end\n", "f", "abc"),
        ("string call returned", "+fn g ()->String\n  +return \"text\"\n+end\n+fn f ()->String\n  +return g()\n+end\n", "f", "text"),
        ("string call as concat operand", "+fn g ()->String\n  +return \"te\"\n+end\n+fn f ()->String\n  +return concat(g(), \"xt\")\n+end\n", "f", "text"),
    ] {
        let program = program(source);
        assert!(!consumes_unknown_string_length(program.get_function("f").unwrap(), &program), "{label}");
        let mut compiled = crate::compiler::compile(&program).unwrap_or_else(|error| panic!("{label}: {error}"));
        assert_eq!(compiled.call_string(call, &[]).unwrap(), expected, "{label}");
    }
    // A String parameter used only where its length is proven: `NOT s`, and
    // an Int result. Callers may pass a literal.
    let source = "+fn blank (s:String)->Int\n  +if NOT s\n    +return 1\n  +end\n  +return 0\n+end\n\
                  +fn full ()->Int\n  +return blank(\"text\")\n+end\n+fn empty ()->Int\n  +return blank(\"\")\n+end\n";
    let program = program(source);
    let mut compiled = crate::compiler::compile(&program).unwrap();
    assert_eq!((compiled.call_i64("full", &[]).unwrap(), compiled.call_i64("empty", &[]).unwrap()), (0, 1));
    // String comparisons the generator does by content: a literal against an
    // allowlisted operand. Results equal the VM's.
    let source = "+fn g ()->String\n  +return \"a\"\n+end\n\
                  +fn same ()->Int\n  +return g() == \"a\"\n+end\n+fn differ ()->Int\n  +return \"b\" == g()\n+end\n\
                  +fn joined ()->Int\n  +return concat(\"a\", \"b\") != \"ab\"\n+end\n+fn literals ()->Int\n  +return \"x\" != \"y\"\n+end\n";
    let comparisons = crate::compiler::admission_measurement_tests::program(source);
    let mut compiled = crate::compiler::compile(&comparisons).unwrap();
    for (function, expected) in [("same", 1), ("differ", 0), ("joined", 0), ("literals", 1)] {
        assert_eq!(compiled.call_i64(function, &[]).unwrap(), expected, "{function}");
    }
    // Numeric code is untouched by the rule: Int variables, Int struct
    // fields, `.len` of a String parameter, comparisons without any string.
    for source in [
        "+fn f (n:Int)->Int\n  +return n\n+end\n",
        "+fn f (a:Int, b:Int)->Int\n  +if a == b\n    +return 1\n  +end\n  +return a != 3\n+end\n",
        "+type Point = {x:Int, y:Int}\n+fn f (p:Point)->Int\n  +if p.x == p.y\n    +return p.x\n  +end\n  +return p.y\n+end\n",
        "+fn f (s:String)->Int\n  +return s.len == 3\n+end\n",
        "+fn g (n:Int)->Int\n  +return n\n+end\n+fn f (n:Int)->Int\n  +return g(n) == g(1)\n+end\n",
    ] {
        let program = crate::compiler::admission_measurement_tests::program(source);
        assert!(!consumes_unknown_string_length(program.get_function("f").unwrap(), &program), "{source}");
    }
}

/// The shape that used to return garbage with `[compiled]`: a zero-argument
/// root (the only kind the dispatcher can call with strings involved) that
/// passes a literal to a function reading its String parameter by name.
#[test]
fn jit_zero_argument_root_over_a_string_parameter_runs_on_the_vm_with_the_right_result() {
    let program = program(
        "+fn greet (name:String)->String\n  +return concat(\"hello \", name)\n+end\n+fn shout ()->String\n  +return greet(\"world\")\n+end\n\
         +fn pick ()->String\n  +let a:String = \"x\"\n  +let b:String = \"yyyy\"\n  +return a\n+end\n",
    );
    let cache = eval::new_jit_cache();
    let input = parser::parse_single_expr("{}").unwrap();
    for (function, expected) in [("shout", "\"hello world\""), ("pick", "\"x\"")] {
        for _ in 0..2 {
            let (value, compiled) = eval::eval_compiled_or_interpreted_cached(&program, function, &input, Some(&cache), 1).unwrap();
            assert_eq!((value.as_str(), compiled), (expected, false), "{function}");
        }
    }
    assert!(cache.lock().unwrap().is_none(), "nothing was compiled for the refused program");
}

/// The allowlist lets a call to something that is not a user function pass
/// as a string operand, on the ground that the code generator has no lowering
/// for it: `concat` is the only builtin it compiles. This test pins that
/// ground. Every builtin name (and alias) other than `concat`, called from a
/// String function, must make the compile fail with the generator's own
/// "undefined function" error (or be refused by the admission walk first).
/// If a builtin gets a lowering, this fails: decide then whether its result
/// carries a length and extend `provides_length` accordingly.
#[test]
fn jit_concat_is_the_only_builtin_the_code_generator_lowers() {
    let template = program("+fn f ()->String\n  +return concat(\"a\", \"b\")\n+end\n");
    assert!(crate::compiler::compile(&template).is_ok(), "the template itself compiles");
    let mut names: Vec<&str> = crate::builtins::all_builtin_names();
    names.extend(crate::builtins::IO_BUILTINS.iter().flat_map(|builtin| std::iter::once(builtin.name).chain(builtin.aliases.iter().copied())));
    names.sort();
    names.dedup();
    assert!(names.len() > 50 && names.contains(&"concat") && names.contains(&"json_get"), "{}", names.len());
    for name in names.into_iter().filter(|name| *name != "concat") {
        for arguments in 0..=3 {
            let mut edited = template.clone();
            let function = edited.get_function_mut("f").unwrap();
            let ast::StatementKind::Return { value: ast::Expr::Call(call) } = &mut function.body[0].kind else { panic!("template shape") };
            call.callee = name.to_string();
            call.args = vec![ast::Expr::Literal(ast::Literal::String("x".into())); arguments];
            edited.rebuild_function_index();
            let error = match crate::compiler::compile(&edited) {
                Ok(_) => panic!("`{name}` with {arguments} argument(s) compiles: the JIT lowers a builtin other than concat"),
                Err(error) => error.to_string(),
            };
            assert!(
                error.contains(&format!("undefined function `{name}`")) || error.starts_with("jit: "),
                "`{name}` with {arguments} argument(s) fails for another reason, as if it had a lowering: {error}"
            );
        }
    }
}
