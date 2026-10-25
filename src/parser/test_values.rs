//! Quoted delimiters are data, not boundaries in a test input value.
use anyhow::{Result, bail};

pub(super) fn balanced_end(line: usize, input: &str, open: char, close: char) -> Result<usize> {
    let mut depth = 0usize;
    let mut quoted = false;
    let mut escaped = false;
    for (offset, character) in input.char_indices() {
        if quoted {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                quoted = false;
            }
            continue;
        }
        if character == '"' {
            quoted = true;
        } else if character == open {
            depth += 1;
        } else if character == close {
            if depth == 0 {
                bail!("line {line}: unmatched {close} in test value");
            }
            depth -= 1;
            if depth == 0 {
                return Ok(offset + character.len_utf8());
            }
        }
    }
    bail!("line {line}: unterminated quoted string or unmatched {open}{close} in test value")
}

pub(super) fn split_case(line: usize, input: &str) -> Result<(&str, &str)> {
    let mut quoted = false;
    let mut escaped = false;
    let mut depth = 0usize;
    for (offset, character) in input.char_indices() {
        if quoted {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                quoted = false;
            }
            continue;
        }
        match character {
            '"' => quoted = true,
            '(' | '{' | '[' => depth += 1,
            ')' | '}' | ']' => depth = depth.saturating_sub(1),
            '-' if depth == 0 && input[offset..].starts_with("->") => {
                return Ok((&input[..offset], &input[offset + 2..]));
            }
            _ => {}
        }
    }
    bail!("line {line}: expected -> separator outside quoted or nested test values")
}

#[cfg(test)]
mod tests {
    use crate::parser::{Expr, parse_test_input};

    #[test]
    fn python_canonical_named_list_ignores_parentheses_in_strings() {
        let parsed = parse_test_input(7, r#"xs=list(")", "(", "é\\\")") count=3"#).unwrap();
        let Expr::StructLiteral(fields) = parsed else {
            panic!("expected named inputs")
        };
        assert_eq!(fields.len(), 2);
        let Expr::Call { args, .. } = &fields[0].value else {
            panic!("expected list constructor")
        };
        assert!(matches!(&args[0], Expr::String(value) if value == ")"));
        assert!(matches!(&args[1], Expr::String(value) if value == "("));
        assert!(matches!(&args[2], Expr::String(value) if value == r#"é\")"#));
        assert!(matches!(&fields[1].value, Expr::Int(3)));
    }

    #[test]
    fn python_canonical_named_struct_ignores_braces_in_strings() {
        let parsed = parse_test_input(8, r#"item={text:"}", other:"{"} flag=true"#).unwrap();
        let Expr::StructLiteral(fields) = parsed else {
            panic!("expected named inputs")
        };
        assert_eq!(fields.len(), 2);
        let Expr::StructLiteral(nested) = &fields[0].value else {
            panic!("expected nested struct")
        };
        assert!(matches!(&nested[0].value, Expr::String(value) if value == "}"));
        assert!(matches!(&nested[1].value, Expr::String(value) if value == "{"));
        assert!(matches!(&fields[1].value, Expr::Bool(true)));
    }

    #[test]
    fn malformed_nested_test_input_still_reports_its_line() {
        let error = parse_test_input(9, r#"xs=list(")" count=3"#).unwrap_err();
        assert!(error.to_string().contains("line 9"));
    }

    #[test]
    fn python_canonical_test_arrow_inside_escaped_string_is_data() {
        let ops = crate::parser::parse(
            r#"+test Demo.echo
 +with "quoted \" -> backslash \\ }" -> expect "quoted \" -> backslash \\ }"
"#,
        )
        .unwrap();
        let crate::parser::Operation::Test(test) = &ops[0] else {
            panic!("expected test")
        };
        let expected = r#"quoted " -> backslash \ }"#;
        assert!(matches!(&test.cases[0].input, Expr::String(value) if value == expected));
        assert!(matches!(&test.cases[0].expected, Expr::String(value) if value == expected));
    }

    #[test]
    fn nested_arrow_literal_and_plain_case_preserve_values() {
        let ops = crate::parser::parse("+test Demo.echo\n +with list(\"->\", \")\") -> expect list(\"->\", \")\")\n +with 1 -> expect 2\n").unwrap();
        let crate::parser::Operation::Test(test) = &ops[0] else {
            panic!("expected test")
        };
        for expr in [&test.cases[0].input, &test.cases[0].expected] {
            let Expr::Call { args, .. } = expr else {
                panic!("expected literal list")
            };
            assert!(matches!(&args[0], Expr::String(value) if value == "->"));
            assert!(matches!(&args[1], Expr::String(value) if value == ")"));
        }
        assert!(matches!(test.cases[1].input, Expr::Int(1)));
        assert!(matches!(test.cases[1].expected, Expr::Int(2)));
    }
}
