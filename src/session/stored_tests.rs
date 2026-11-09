//! Typed payloads avoid exporting new Python tests merely to run them again.
use super::*;

pub(super) fn operations(name: &str, cases: &[ast::TestCase]) -> Result<Vec<parser::Operation>> {
    if cases.iter().all(|case| case.typed.is_some()) {
        let cases = cases
            .iter()
            .map(|case| case.typed.as_ref().unwrap().case())
            .collect::<Result<Vec<_>>>()?;
        return Ok(vec![parser::Operation::Test(parser::TestMutation {
            function_name: name.into(),
            cases,
        })]);
    }
    // Legacy import only. Preserve its historical bare-target behavior; no new
    // Python submission goes through this textual reconstruction.
    let bare = name.rsplit('.').next().unwrap_or(name);
    let mut parsed = Vec::new();
    for case in cases {
        if let Some(typed) = &case.typed {
            parsed.push(typed.case()?);
            continue;
        }
        let expected = reconstruct_expect(&case.expected, case.matcher.as_deref());
        let mut source = format!(
            "+test {bare}\n  +with {} -> expect {}\n",
            ast::escape_test_input_linebreaks(&case.input),
            ast::escape_test_input_linebreaks(&expected)
        );
        for check in &case.after_checks {
            source.push_str(&format!(
                "  +after {} {} \"{}\"\n",
                check.target, check.matcher, check.value
            ));
        }
        let operations = parser::parse(&source)?;
        let [parser::Operation::Test(test)] = operations.as_slice() else {
            anyhow::bail!("invalid legacy stored test");
        };
        parsed.extend(test.cases.clone());
    }
    Ok(vec![parser::Operation::Test(parser::TestMutation {
        function_name: bare.into(),
        cases: parsed,
    })])
}
