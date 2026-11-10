use super::*;

fn body(expression: &str) -> String {
    format!("def f(x: int) -> int:\n    return {expression}\n")
}

fn offset(source: &str) -> Option<usize> {
    check(source).err().map(|deep| deep.offset)
}

/// Depth available to an expression in `body`: one indentation level is used.
const ROOM: usize = MAX_NESTING - 1;

#[test]
fn nesting_guard_accepts_ordinary_source() {
    for source in [
        "",
        "\n\n# only a comment\n",
        "def f(x: int) -> int:\n    return x + 1\n",
        "def f(a: list[list[int]]) -> dict[str, list[int]]:\n    y: int = g(h(a[0][1]), k=[1, 2, {3: (4, 5)}])\n    return y\n",
        "x = 1.5e-3 + 0x1F + 1_000 + .5 + 1e+10 + 5j\n",
        "s = 'a + b + c' + \"((((\" + '''[[[\n{{{''' + r'\\' + b'\\'' # ((((((((\n",
        "if a:\n    pass\nelif b:\n    pass\nelse:\n    pass\n",
        "x = 1 if a else 2\nfor i in y:\n    from_value = not i\n",
    ] {
        assert_eq!(check(source), Ok(()), "{source}");
    }
    // Width is not depth: thousands of siblings at one level.
    let wide = format!("x = [{}]\n", (0..5_000).map(|i| format!("a.b + {i}")).collect::<Vec<_>>().join(", "));
    assert_eq!(check(&wide), Ok(()));
    let calls = format!("x = f({})\n", (0..5_000).map(|i| format!("k{i}=g(h({i}))")).collect::<Vec<_>>().join(", "));
    assert_eq!(check(&calls), Ok(()));
    let statements = "x = a + b\n".repeat(20_000);
    assert_eq!(check(&statements), Ok(()));
    let dict = format!("x = {{{}}}\n", (0..3_000).map(|i| format!("'k{i}': a + b * c")).collect::<Vec<_>>().join(",\n    "));
    assert_eq!(check(&dict), Ok(()));
    // Strings, comments and continuation lines hide nothing and count nothing.
    let quiet = format!("x = \"{}\"  # {}\ny = '''{}'''\nz = 1 + \\\n    2\n", "(".repeat(500), "[".repeat(500), "- ".repeat(500));
    assert_eq!(check(&quiet), Ok(()));
}

/// Every depth-forming construct is accepted at the limit and refused one
/// level beyond it, at the position of the token that exceeds it.
#[test]
fn nesting_guard_limit_is_exact_for_each_construct() {
    type Make = fn(usize) -> String;
    let constructs: &[(&str, Make)] = &[
        ("operators", |n| body(&format!("x{}", " + 1".repeat(n)))),
        ("power", |n| body(&format!("x{}", " ** 2".repeat(n)))),
        ("unary minus", |n| body(&format!("{}x", "-".repeat(n)))),
        ("not", |n| body(&format!("{}x", "not ".repeat(n)))),
        ("and", |n| body(&format!("x{}", " and x".repeat(n)))),
        ("attributes", |n| body(&format!("x{}", ".a".repeat(n)))),
        ("parentheses", |n| body(&format!("{}x{}", "(".repeat(n), ")".repeat(n)))),
        ("lists", |n| body(&format!("{}x{}", "[".repeat(n), "]".repeat(n)))),
        ("dicts", |n| body(&format!("{}1{}", "{1: ".repeat(n), "}".repeat(n)))),
        ("call chain", |n| body(&format!("f{}", "()".repeat(n)))),
        ("subscript chain", |n| body(&format!("x{}", "[0]".repeat(n)))),
        ("lambda", |n| body(&format!("{}x", "lambda a, b: ".repeat(n)))),
        ("await", |n| body(&format!("{}x", "await ".repeat(n)))),
        ("annotation", |n| format!("def f(x: {}int{}) -> int:\n    return 1\n", "list[".repeat(n), "]".repeat(n))),
    ];
    for (name, make) in constructs {
        assert_eq!(check(&make(ROOM)), Ok(()), "{name} at the limit");
        let deep = check(&make(ROOM + 1)).expect_err(name);
        assert!(deep.message.starts_with("nesting too deep: more than 100 levels of nested brackets, chained operators, indented blocks and elif branches at this point (limit 100); split the expression"), "{name}: {}", deep.message);
        for far in [1_000, 100_000] {
            assert!(check(&make(far)).is_err(), "{name} at {far}");
        }
    }
    // Nested calls cost one level each, like any bracket.
    assert_eq!(check(&body(&format!("{}x{}", "g(".repeat(ROOM), ")".repeat(ROOM)))), Ok(()));
    assert!(check(&body(&format!("{}x{}", "g(".repeat(ROOM + 1), ")".repeat(ROOM + 1)))).is_err());
    // A ternary uses two tokens per level (`if` + `else`): refused earlier, never later.
    assert!(check(&body(&format!("{}x", "x if x else ".repeat(ROOM / 2 + 1)))).is_err());
    // The position is the token that exceeds the limit.
    let source = body(&format!("x{}", " + 1".repeat(ROOM + 1)));
    let at = offset(&source).unwrap();
    assert_eq!(&source[at..at + 3], "+ 1");
    assert_eq!(source[..at].matches('+').count(), ROOM, "the hundredth level is the first refused");
    let brackets = body(&format!("{}x{}", "(".repeat(ROOM + 1), ")".repeat(ROOM + 1)));
    assert_eq!(offset(&brackets), Some(brackets.find("return (").unwrap() + "return ".len() + ROOM));
}

#[test]
fn nesting_guard_adds_brackets_chains_and_blocks_along_one_path() {
    // 40 blocks + 30 brackets + 29 operators = 99 levels, plus the function body.
    let mut source = String::from("def f(x: int) -> int:\n");
    for level in 0..40 {
        source.push_str(&format!("{}if x > 0:\n", " ".repeat(level + 1)));
    }
    let line = |operators: usize| format!("{}return {}x{}{}\n", " ".repeat(41), "(".repeat(30), " + 1".repeat(operators), ")".repeat(30));
    // (41 indentation levels with the function body: 41 + 30 + 29 = 100.)
    assert_eq!(check(&format!("{source}{}", line(29))), Ok(()));
    assert!(check(&format!("{source}{}", line(30))).is_err());
    // A comma starts a sibling: its chain does not add to the previous one...
    assert_eq!(check(&body(&format!("g(x{}, x{})", " + 1".repeat(ROOM - 1), " + 1".repeat(ROOM - 1)))), Ok(()));
    // ...but what follows a closed group is built on top of what is inside it:
    // (49 operators) then 49 more is 99 levels with the group; one more is refused.
    let after = |inside: usize, outside: usize| body(&format!("(x{}){}", " + 1".repeat(inside), " + 1".repeat(outside)));
    assert_eq!(check(&after(49, 49)), Ok(()));
    assert!(check(&after(49, 50)).is_err());
    assert!(check(&after(50, 49)).is_err());
    assert!(check(&body(&format!("g(h(x{})).a.b{}", " + 1".repeat(60), "[0]".repeat(40)))).is_err());
    // Siblings inside the group contribute their deepest one, once.
    assert_eq!(check(&body(&format!("g(x{}, x{}){}", " + 1".repeat(40), " + 1".repeat(40), " + 1".repeat(50)))), Ok(()));
    // Blocks: dedenting gives the depth back; brackets ignore line breaks.
    let mut blocks = String::from("def f(x: int) -> int:\n");
    for _ in 0..500 {
        for level in 0..50 {
            blocks.push_str(&format!("{}if x > 0:\n", " ".repeat(level + 1)));
        }
        blocks.push_str(&format!("{}x = (\n1 +\n2)\n", " ".repeat(51)));
    }
    assert_eq!(check(&blocks), Ok(()));
    // `blocks` nested ifs inside the function body: the innermost statement is
    // at level blocks + 1.
    let nested = |blocks: usize| {
        let mut source = String::from("def f(x: bool) -> int:\n");
        for level in 0..blocks {
            source.push_str(&format!("{}if x:\n", " ".repeat(level + 1)));
        }
        source.push_str(&format!("{}return 1\n    return 0\n", " ".repeat(blocks + 1)));
        source
    };
    assert_eq!(check(&nested(MAX_NESTING - 1)), Ok(()));
    let deep = nested(MAX_NESTING);
    let refused = check(&deep).unwrap_err();
    assert_eq!(&deep[refused.offset..refused.offset + 8], "return 1", "refused at the statement that is too deep");
    for far in [500, 2_000] {
        assert!(check(&nested(far)).is_err());
    }
    // A separator starts a sibling at statement level too.
    let annotated = format!("x: {}int{} = {}1{}\n", "list[".repeat(99), "]".repeat(99), "[".repeat(99), "]".repeat(99));
    assert_eq!(check(&annotated), Ok(()));
    let signature = format!("def f(x: {a}int{b}, y: {a}int{b} = None) -> {a}int{b}:\n    return x\n", a = "list[".repeat(98), b = "]".repeat(98));
    assert_eq!(check(&signature), Ok(()));
    assert_eq!(check(&format!("d = {{'k': x{}, 'l': x{}}}\n", " + 1".repeat(98), " + 1".repeat(98))), Ok(()));
    // Lambda parameters and bodies are not siblings of the lambda.
    assert!(check(&body(&format!("{}x", "lambda a=1, b=2: ".repeat(ROOM + 1)))).is_err());
}

#[test]
fn nesting_guard_counts_elif_branches_and_bounds_fstring_fields() {
    // An `elif` nests in the branch before it: `count` branches at the top level,
    // each with a statement one block deeper.
    let chain = |count: usize, indent: usize| {
        let pad = " ".repeat(indent);
        let mut source = format!("{pad}if x == 0:\n{pad}    y = 0\n");
        for i in 0..count {
            source.push_str(&format!("{pad}elif x == {i}:\n{pad}    y = 1\n"));
        }
        source
    };
    // 99 branches + `==` on the elif line (or the block of its body) = 100.
    assert_eq!(check(&chain(99, 0)), Ok(()));
    let long = chain(100, 0);
    let refused = check(&long).unwrap_err();
    assert!(refused.message.contains("elif branches at this point (limit 100)"), "{}", refused.message);
    assert_eq!(&long[refused.offset..refused.offset + 2], "==", "refused in the hundredth branch");
    assert_eq!(long[..refused.offset].matches("elif").count(), 100);
    for far in [700, 5_000, 100_000] {
        assert!(check(&chain(far, 0)).is_err(), "{far}");
    }
    // A new `if` starts a new chain: chains one after the other do not add up...
    assert_eq!(check(&chain(99, 0).repeat(50)), Ok(()));
    // ...a chain inside the last branch of another does...
    assert!(check(&format!("{}{}", chain(60, 0), chain(60, 4))).is_err());
    assert_eq!(check(&format!("{}{}", chain(40, 0), chain(40, 4))), Ok(()));
    // ...and leaving that branch gives its depth back.
    assert_eq!(check(&format!("{}{}z = 1\n{}", chain(40, 0), chain(40, 4), chain(90, 0))), Ok(()));
    // f-strings: prose is free, expression text in the fields is bounded.
    let prose = format!("x = f\"{} {{{{literal braces}}}} {{name}}\"\n", "a - b. ".repeat(2_000));
    assert_eq!(check(&prose), Ok(()));
    let field = |n: usize| format!("x = f\"{{x{}}}\"\n", " + 1".repeat(n));
    assert_eq!(check(&field(400)), Ok(()));
    for prefix in ["f", "F", "rf", "fR"] {
        let source = format!("x = {prefix}\"{{x{}}}\"\n", " + 1".repeat(600));
        let refused = check(&source).unwrap_err();
        assert!(refused.message.starts_with("f-string replacement fields too large"), "{prefix}: {}", refused.message);
        assert_eq!(refused.offset, 4 + prefix.len(), "{prefix}: at the string");
    }
    // The same text in a plain string is data.
    assert_eq!(check(&format!("x = \"{{x{}}}\"\n", " + 1".repeat(50_000))), Ok(()));
}

/// The bound is on what the parser builds: with the guard in front, the tree of
/// any accepted source is shallow enough to parse, walk and drop on a small
/// stack, and refused sources never reach the parser.
#[test]
fn nesting_guard_keeps_the_parser_tree_shallow() {
    use rustpython_parser::Parse;
    let accepted = [
        body(&format!("x{}", " + 1".repeat(ROOM))),
        body(&format!("{}x{}", "g(".repeat(ROOM), ")".repeat(ROOM))),
        format!("x = f\"{{x{}}}\"\n", " + 1".repeat(400)),
        format!("if x == 0:\n    y = 0\n{}", "elif x == 1:\n    y = 1\n".repeat(99)),
    ];
    std::thread::Builder::new()
        .stack_size(512 * 1024)
        .spawn(move || {
            for source in &accepted {
                assert_eq!(check(source), Ok(()));
                assert!(rustpython_parser::ast::Suite::parse(source, "M").is_ok());
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

/// Headroom: every Python source shipped in the repository (examples, the
/// task-domain modules, the Telegram port, test fixtures) stays far below the
/// limit, so the guard cannot refuse code of the kind that is actually written.
#[test]
fn nesting_guard_leaves_wide_headroom_for_the_repository_sources() {
    fn collect(dir: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect(&path, files);
            } else if path.extension().is_some_and(|extension| extension == "py") {
                files.push(path);
            }
        }
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    for dir in ["examples", "tests/fixtures"] {
        collect(&root.join(dir), &mut files);
    }
    assert!(files.len() >= 10, "found {} Python sources", files.len());
    let mut worst = (0, String::new());
    for file in &files {
        let source = std::fs::read_to_string(file).unwrap();
        let depth = deepest(&source);
        if depth > worst.0 {
            worst = (depth, file.strip_prefix(root).unwrap().display().to_string());
        }
    }
    eprintln!("NESTING deepest repository source: {} levels in {} ({} files)", worst.0, worst.1, files.len());
    assert!(worst.0 * 3 <= MAX_NESTING, "{} levels in {}", worst.0, worst.1);
}

/// Every frontend entry point refuses sources far deeper than a native tool call
/// can carry (the CLI test path and library tooling have no size cap), on a
/// worker-sized stack, with the position of the first token that is too deep.
#[test]
fn nesting_guard_protects_every_frontend_entry_point_at_extreme_depth() {
    std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(|| {
            let program = crate::ast::Program::default();
            for depth in [40_000, 250_000] {
                let sources = [
                    body(&format!("x{}", " + 1".repeat(depth))),
                    body(&format!("{}x{}", "g(".repeat(depth), ")".repeat(depth))),
                    body(&format!("{}x{}", "[".repeat(depth), "]".repeat(depth))),
                    body(&format!("{}x", "not ".repeat(depth))),
                    body(&format!("x{}", ".a".repeat(depth))),
                    body(&format!("f\"{{x{}}}\"", " + 1".repeat(depth))),
                    format!("def f(x: int) -> int:\n    if x == 0:\n        return 0\n{}    return 1\n", "    elif x == 1:\n        return 1\n".repeat(depth)),
                ];
                for source in &sources {
                    let refusals = [
                        super::super::lower_module(source, "Deep").map(|_| ()).unwrap_err(),
                        super::super::preflight_module(source, "Deep").map(|_| ()).unwrap_err(),
                        super::super::lower_module_with_program(source, "Deep", &program).map(|_| ()).unwrap_err(),
                        super::super::lower_tests(source, "Deep").map(|_| ()).unwrap_err(),
                        super::super::lower_toplevel(source, &program).map(|_| ()).unwrap_err(),
                    ];
                    for refusal in refusals {
                        assert!(refusal.message.contains("nesting too deep") || refusal.message.contains("f-string replacement fields too large"), "{}", refusal.message);
                        // Line and column of the offending token, not 1:1.
                        assert!(refusal.line >= 2 && refusal.column > 1, "{refusal}");
                        assert!(refusal.to_string().starts_with(&format!("{}:{}:{}: ", refusal.module, refusal.line, refusal.column)));
                    }
                }
                // Evaluate inputs: one expression list.
                for input in [format!("{}1", "-".repeat(depth)), format!("{}1{}", "(".repeat(depth), ")".repeat(depth)), format!("x={}1{}", "[".repeat(depth), "]".repeat(depth))] {
                    let refusal = super::super::lower_input(&input).unwrap_err();
                    assert!(refusal.message.contains("nesting too deep"), "{}", refusal.message);
                    assert_eq!(refusal.line, 1);
                }
            }
        })
        .unwrap()
        .join()
        .expect("no entry point overflowed");
}
