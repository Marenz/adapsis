use super::*;
use crate::{parser, session::IoMock, validator};

/// Helper: parse Adapsis source and build a program from it.
fn build_program(source: &str) -> ast::Program {
    let ops = parser::parse(source).expect("parse failed");
    let mut program = ast::Program::default();
    for op in &ops {
        match op {
            parser::Operation::Test(_) | parser::Operation::Eval(_) => {}
            _ => {
                validator::apply_and_validate(&mut program, op).expect("validation failed");
            }
        }
    }
    program.rebuild_function_index();
    program
}

/// Helper: extract test cases from parsed source.
fn extract_test_cases(source: &str) -> Vec<(String, parser::TestCase)> {
    let ops = parser::parse(source).expect("parse failed");
    let mut cases = Vec::new();
    for op in &ops {
        if let parser::Operation::Test(test) = op {
            for case in &test.cases {
                cases.push((test.function_name.clone(), case.clone()));
            }
        }
        // Also extract tests embedded inside module bodies
        if let parser::Operation::Module(m) = op {
            for body_op in &m.body {
                if let parser::Operation::Test(test) = body_op {
                    for case in &test.cases {
                        cases.push((test.function_name.clone(), case.clone()));
                    }
                }
            }
        }
    }
    cases
}

/// Helper: extract mocks from parsed source.
fn extract_mocks(source: &str) -> Vec<IoMock> {
    let ops = parser::parse(source).expect("parse failed");
    let mut mocks = Vec::new();
    for op in ops {
        if let parser::Operation::Mock {
            operation,
            patterns,
            response,
        } = op
        {
            mocks.push(IoMock {
                operation,
                patterns,
                response,
            });
        }
    }
    mocks
}

// ── Sync function tests ───────────────────────────────────────────

#[test]
fn test_sync_function_passes() {
    let source = "\
+fn add (a:Int, b:Int)->Int
  +let sum:Int = a + b
  +return sum

+test add
  +with a=2 b=3 -> expect 5
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    assert_eq!(cases.len(), 1);

    let (fn_name, case) = &cases[0];
    let result = eval_test_case(&program, fn_name, case);
    assert!(result.is_ok(), "sync test should pass: {:?}", result);
    assert!(result.unwrap().contains("expected 5"));
}

#[test]
fn test_sync_function_fails_on_wrong_expected() {
    let source = "\
+fn add (a:Int, b:Int)->Int
  +let sum:Int = a + b
  +return sum

+test add
  +with a=2 b=3 -> expect 99
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    let result = eval_test_case(&program, fn_name, case);
    assert!(result.is_err(), "should fail when expected is wrong");
}

// ── Async function tests (mock-only path) ────────────────────────

#[test]
fn test_async_function_with_mock_http_get() {
    let source = "\
+fn fetch_data (url:String)->String [io,async]
  +await resp:String = http_get(url)
  +return resp
";
    let program = build_program(source);

    let test_source = "\
+test fetch_data
  +with url=\"https://example.com\" -> expect \"hello\"
";
    let cases = extract_test_cases(test_source);
    let (fn_name, case) = &cases[0];

    let mocks = vec![IoMock {
        operation: "http_get".to_string(),
        patterns: vec!["example.com".to_string()],
        response: "hello".to_string(),
    }];
    let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
    assert!(
        result.is_ok(),
        "async test with mock should pass: {:?}",
        result
    );
}

#[test]
fn test_async_function_without_mock_errors_on_unmocked_io() {
    // Tests always use mock-only handles to prevent deadlocks from
    // self-referential HTTP calls.  Unmocked IO should fail, not execute.
    let source = "\
+fn delayed_value ()->String [io,async]
  +await _:String = sleep(1)
  +return \"done\"
";
    let program = build_program(source);

    let test_source = "\
+test delayed_value
  +with -> expect \"done\"
";
    let cases = extract_test_cases(test_source);
    let (fn_name, case) = &cases[0];

    // No mocks — should fail with "no mock for sleep" instead of running real IO
    let result = eval_test_case(&program, fn_name, case);
    assert!(
        result.is_err(),
        "async test without mocks should error: {:?}",
        result
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("no mock"),
        "error should mention missing mock: {err}"
    );
}

#[test]
fn test_async_function_with_io_effect_gets_handle() {
    let source = "\
+fn fetch_data (url:String)->String [io,async]
  +await resp:String = http_get(url)
  +return resp
";
    let program = build_program(source);

    let test_source = "\
+test fetch_data
  +with url=\"https://example.com\" -> expect \"world\"
";
    let cases = extract_test_cases(test_source);
    let (fn_name, case) = &cases[0];

    let mocks = vec![IoMock {
        operation: "http_get".to_string(),
        patterns: vec!["example.com".to_string()],
        response: "world".to_string(),
    }];
    let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
    assert!(
        result.is_ok(),
        "io test with mock should pass: {:?}",
        result
    );
}

#[test]
fn test_async_function_await_sleep_with_mock() {
    // +await on `sleep` is a builtin IO op — should be intercepted by mock
    let source = "\
+fn delayed_value (ms:Int)->String [io,async]
  +await _:String = sleep(ms)
  +return \"done\"
";
    let program = build_program(source);

    let test_source = "\
+test delayed_value
  +with ms=1000 -> expect \"done\"
";
    let cases = extract_test_cases(test_source);
    let (fn_name, case) = &cases[0];

    // Mock sleep so it returns immediately without real delay
    let mocks = vec![IoMock {
        operation: "sleep".to_string(),
        patterns: vec!["1000".to_string()],
        response: "".to_string(),
    }];
    let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
    assert!(result.is_ok(), "sleep mock test should pass: {:?}", result);
}

#[test]
fn test_async_function_with_mock_inbox_read() {
    let source = "\
+fn drain_inbox ()->String [io,async]
  +await resp:String = inbox_read()
  +return resp
";
    let program = build_program(source);

    let test_source = "\
+test drain_inbox
  +with -> expect \"[\\\"mocked\\\"]\"
";
    let cases = extract_test_cases(test_source);
    let (fn_name, case) = &cases[0];

    let mocks = vec![IoMock {
        operation: "inbox_read".to_string(),
        patterns: vec!["".to_string()],
        response: "[\"mocked\"]".to_string(),
    }];
    let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
    assert!(
        result.is_ok(),
        "inbox_read mock test should pass: {:?}",
        result
    );
}

#[test]
fn test_async_function_nested_await_propagates_handle() {
    // An async function that calls another user-defined async function
    // which itself does +await on a builtin — handle must propagate
    let source = "\
+fn inner_fetch (url:String)->String [io,async]
  +await resp:String = http_get(url)
  +return resp

+fn outer_fetch (url:String)->String [io,async]
  +await data:String = inner_fetch(url)
  +return data
";
    let program = build_program(source);

    let test_source = "\
+test outer_fetch
  +with url=\"https://api.test.com\" -> expect \"nested_ok\"
";
    let cases = extract_test_cases(test_source);
    let (fn_name, case) = &cases[0];

    let mocks = vec![IoMock {
        operation: "http_get".to_string(),
        patterns: vec!["api.test.com".to_string()],
        response: "nested_ok".to_string(),
    }];
    let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
    assert!(
        result.is_ok(),
        "nested async call should pass: {:?}",
        result
    );
}

// ── Mock JSON response tests ─────────────────────────────────────

#[test]
fn test_mock_http_get_json_consumed_by_json_get() {
    // Mock returns JSON, function uses json_get to extract a field
    let source = "\
+fn get_name (url:String)->String [io,async]
  +await body:String = http_get(url)
  +let name:String = json_get(body, \"name\")
  +return name
";
    let program = build_program(source);

    let test_source = "\
+test get_name
  +with url=\"https://api.example.com/user\" -> expect \"alice\"
";
    let cases = extract_test_cases(test_source);
    let (fn_name, case) = &cases[0];

    // The mock response is valid JSON — note: IoMock.response is the
    // *decoded* string (no backslash escapes).
    let mocks = vec![IoMock {
        operation: "http_get".to_string(),
        patterns: vec!["api.example.com".to_string()],
        response: r#"{"name":"alice","age":30}"#.to_string(),
    }];
    let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
    assert!(
        result.is_ok(),
        "json_get on mock JSON should pass: {:?}",
        result
    );
}

#[test]
fn test_mock_http_get_json_consumed_by_json_array_len() {
    // Mock returns a JSON array, function uses json_array_len
    let source = "\
+fn count_items (url:String)->Int [io,async]
  +await body:String = http_get(url)
  +let count:Int = json_array_len(body)
  +return count
";
    let program = build_program(source);

    let test_source = "\
+test count_items
  +with url=\"https://api.example.com/items\" -> expect 3
";
    let cases = extract_test_cases(test_source);
    let (fn_name, case) = &cases[0];

    let mocks = vec![IoMock {
        operation: "http_get".to_string(),
        patterns: vec!["api.example.com".to_string()],
        response: r#"[1,2,3]"#.to_string(),
    }];
    let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
    assert!(
        result.is_ok(),
        "json_array_len on mock JSON should pass: {:?}",
        result
    );
}

#[test]
fn test_mock_escape_decoding_via_parser() {
    // Verify that !mock strings are properly unescaped by the parser.
    // The Adapsis source text: !mock http_get "api.test" -> "{\"ok\":true,\"items\":[1,2]}"
    let source = "!mock http_get \"api.test\" -> \"{\\\"ok\\\":true,\\\"items\\\":[1,2]}\"";
    let mocks = extract_mocks(source);
    assert_eq!(mocks.len(), 1);
    assert_eq!(mocks[0].patterns, vec!["api.test"]);
    // After unescape, response should be valid JSON without backslashes
    assert_eq!(mocks[0].response, r#"{"ok":true,"items":[1,2]}"#);
    // Verify it's valid JSON
    let parsed: serde_json::Value =
        serde_json::from_str(&mocks[0].response).expect("mock response should be valid JSON");
    assert_eq!(parsed["ok"], true);
}

#[test]
fn test_mock_json_end_to_end_with_parser_escaping() {
    // End-to-end: !mock with escaped JSON → function uses json_get
    let fn_source = "\
+fn check_status (url:String)->String [io,async]
  +await body:String = http_get(url)
  +let status:String = json_get(body, \"status\")
  +return status
";
    let program = build_program(fn_source);

    // This is how it would appear in a .ax file — escaped quotes
    // Adapsis source: !mock http_get "api.svc" -> "{\"status\":\"healthy\",\"uptime\":99}"
    let mock_source =
        "!mock http_get \"api.svc\" -> \"{\\\"status\\\":\\\"healthy\\\",\\\"uptime\\\":99}\"";
    let mocks = extract_mocks(mock_source);

    let test_source = "\
+test check_status
  +with url=\"https://api.svc.local/health\" -> expect \"healthy\"
";
    let cases = extract_test_cases(test_source);
    let (fn_name, case) = &cases[0];

    let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
    assert!(
        result.is_ok(),
        "end-to-end mock JSON test should pass: {:?}",
        result
    );
}

// ── Orchestrator-style integration tests ────────────────────────
// These simulate the exact flow: parse source with +fn, !mock, !test;
// collect mocks; build program; run eval_test_case_with_mocks.

/// Helper: simulate orchestrator flow — parse full source, collect mocks,
/// build program, run each test case with mocks.
fn run_orchestrator_style(source: &str) -> Vec<Result<String>> {
    let ops = parser::parse(source).expect("parse failed");
    let mut program = ast::Program::default();
    let mut test_ops = Vec::new();
    let mut io_mocks: Vec<IoMock> = Vec::new();

    for op in &ops {
        match op {
            parser::Operation::Test(test) => test_ops.push(test.clone()),
            parser::Operation::Module(m) => {
                // Extract tests embedded inside module bodies
                for body_op in &m.body {
                    if let parser::Operation::Test(test) = body_op {
                        test_ops.push(test.clone());
                    }
                }
                let _ = validator::apply_and_validate(&mut program, op);
            }
            parser::Operation::Mock {
                operation,
                patterns,
                response,
            } => {
                io_mocks.push(IoMock {
                    operation: operation.clone(),
                    patterns: patterns.clone(),
                    response: response.clone(),
                });
            }
            parser::Operation::Unmock => {
                io_mocks.clear();
            }
            parser::Operation::Eval(_) => {}
            _ => {
                let _ = validator::apply_and_validate(&mut program, op);
            }
        }
    }
    program.rebuild_function_index();

    let mut results = Vec::new();
    for test in &test_ops {
        for case in &test.cases {
            results.push(eval_test_case_with_mocks(
                &program,
                &test.function_name,
                case,
                &io_mocks,
                &[],
            ));
        }
    }
    results
}

#[test]
fn test_orchestrator_async_http_get_with_mock() {
    // Simulates LLM output containing +fn [async], !mock, !test
    let source = "\
+fn fetch_data (url:String)->String [io,async]
  +await resp:String = http_get(url)
  +return resp

!mock http_get \"example.com\" -> \"hello world\"

+test fetch_data
  +with url=\"https://example.com/api\" -> expect \"hello world\"
";
    let results = run_orchestrator_style(source);
    assert_eq!(results.len(), 1);
    assert!(
        results[0].is_ok(),
        "orchestrator async http_get test should pass: {:?}",
        results[0]
    );
}

#[test]
fn test_orchestrator_async_sleep_with_mock() {
    let source = "\
+fn delayed (ms:Int)->String [io,async]
  +await _:String = sleep(ms)
  +return \"done\"

!mock sleep \"500\" -> \"\"

+test delayed
  +with ms=500 -> expect \"done\"
";
    let results = run_orchestrator_style(source);
    assert_eq!(results.len(), 1);
    assert!(
        results[0].is_ok(),
        "orchestrator async sleep test should pass: {:?}",
        results[0]
    );
}

#[test]
fn test_orchestrator_nested_async_with_mock() {
    // wrapper -> inner_fetch -> http_get (all async, handle must propagate)
    let source = "\
+fn inner_fetch (url:String)->String [io,async]
  +await resp:String = http_get(url)
  +return resp

+fn wrapper (url:String)->String [io,async]
  +await data:String = inner_fetch(url)
  +return data

!mock http_get \"api.test\" -> \"nested result\"

+test wrapper
  +with url=\"https://api.test/v1\" -> expect \"nested result\"
";
    let results = run_orchestrator_style(source);
    assert_eq!(results.len(), 1);
    assert!(
        results[0].is_ok(),
        "orchestrator nested async test should pass: {:?}",
        results[0]
    );
}

#[test]
fn test_orchestrator_mock_json_escape_json_get_json_array_len() {
    // Proves: !mock with escaped JSON → parser decodes → json_get + json_array_len work
    // Adapsis source: !mock http_get "x" -> "{\"ok\":true,\"result\":[]}"
    let fn_source = "\
+fn check (url:String)->Int [io,async]
  +await body:String = http_get(url)
  +let arr:String = json_get(body, \"result\")
  +let count:Int = json_array_len(arr)
  +return count
";
    let mock_source = "!mock http_get \"x\" -> \"{\\\"ok\\\":true,\\\"result\\\":[]}\"";
    let test_source = "\
+test check
  +with url=\"x\" -> expect 0
";
    let full_source = format!("{fn_source}\n{mock_source}\n\n{test_source}");
    let results = run_orchestrator_style(&full_source);
    assert_eq!(results.len(), 1);
    assert!(
        results[0].is_ok(),
        "mock JSON escape + json_get + json_array_len should pass: {:?}",
        results[0]
    );
}

// ── Async eval_test_case_async tests ─────────────────────────────

#[tokio::test]
async fn test_async_eval_test_case_with_mocks() {
    let source = "\
+fn fetch_data (url:String)->String [io,async]
  +await resp:String = http_get(url)
  +return resp
";
    let program = build_program(source);

    let test_source = "\
+test fetch_data
  +with url=\"https://example.com\" -> expect \"async_hello\"
";
    let cases = extract_test_cases(test_source);
    let (fn_name, case) = &cases[0];

    let mocks = vec![IoMock {
        operation: "http_get".to_string(),
        patterns: vec!["example.com".to_string()],
        response: "async_hello".to_string(),
    }];

    let (tx, _rx) = tokio::sync::mpsc::channel(1);
    let result = eval_test_case_async(&program, fn_name, case, &mocks, tx, &[]).await;
    assert!(result.is_ok(), "async test case should pass: {:?}", result);
}

#[tokio::test]
async fn test_session_apply_async_runs_async_tests_with_mocks() {
    // Simulate the full session flow: define an async function,
    // register mocks, then run +test — all through apply_async.
    let mut session = crate::session::Session::new();

    // Define async function
    let define_source = "\
+fn fetch_data (url:String)->String [io,async]
  +await resp:String = http_get(url)
  +return resp
";
    let results = session.apply_async(define_source, None).await;
    assert!(results.is_ok(), "define should succeed: {:?}", results);

    // Register mock
    let mock_source = "!mock http_get \"example.com\" -> \"mocked_response\"";
    let results = session.apply_async(mock_source, None).await;
    assert!(results.is_ok(), "mock should succeed: {:?}", results);

    // Run test — async function with mock, no io_sender needed (mock-only)
    let test_source = "\
+test fetch_data
  +with url=\"https://example.com/api\" -> expect \"mocked_response\"
";
    let results = session.apply_async(test_source, None).await.unwrap();
    assert_eq!(results.len(), 1);
    assert!(
        results[0].1,
        "async test with mock should pass: {:?}",
        results[0]
    );
    assert!(
        results[0].0.contains("PASS"),
        "result should be PASS: {:?}",
        results[0]
    );
}

#[tokio::test]
async fn test_session_apply_async_nested_async_with_mocks() {
    // Nested async: wrapper -> inner_fetch -> http_get
    let mut session = crate::session::Session::new();

    let source = "\
+fn inner_fetch (url:String)->String [io,async]
  +await resp:String = http_get(url)
  +return resp

+fn wrapper (url:String)->String [io,async]
  +call data:String = inner_fetch(url)
  +return data
";
    let _ = session.apply_async(source, None).await;

    let mock_source = "!mock http_get \"api.test\" -> \"nested_result\"";
    let _ = session.apply_async(mock_source, None).await;

    let test_source = "\
+test wrapper
  +with url=\"https://api.test/v1\" -> expect \"nested_result\"
";
    let results = session.apply_async(test_source, None).await.unwrap();
    assert_eq!(results.len(), 1);
    assert!(
        results[0].1,
        "nested async test should pass: {:?}",
        results[0]
    );
}

#[tokio::test]
async fn test_async_eval_delegates_sync_to_sync_path() {
    let source = "\
+fn add (a:Int, b:Int)->Int
  +let sum:Int = a + b
  +return sum
";
    let program = build_program(source);

    let test_source = "\
+test add
  +with a=10 b=20 -> expect 30
";
    let cases = extract_test_cases(test_source);
    let (fn_name, case) = &cases[0];

    let (tx, _rx) = tokio::sync::mpsc::channel(1);
    let result = eval_test_case_async(&program, fn_name, case, &[], tx, &[]).await;
    assert!(
        result.is_ok(),
        "sync function via async path should pass: {:?}",
        result
    );
}

// ── UTF-8 regression tests ───────────────────────────────────────

#[test]
fn test_json_get_preserves_utf8() {
    // Verify json_get returns multi-byte UTF-8 characters intact
    // Use a wrapper that builds the JSON internally to avoid parser quoting issues
    let source = r#"
+fn get_cafe ()->String
  +let json:String = "{\"name\":\"café\"}"
  +return json_get(json, "name")
"#;
    let program = build_program(source);
    let input = parser::parse("!eval get_cafe")
        .unwrap()
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .unwrap();
    let result = eval_call_with_input(&program, &input.function_name, &input.input).unwrap();
    assert_eq!(result, r#""café""#, "json_get should preserve UTF-8 chars");
}

#[test]
fn test_json_escape_preserves_utf8() {
    // Verify json_escape passes multi-byte UTF-8 through unchanged
    let source = r#"
+fn escape_it (s:String)->String
  +return json_escape(s)
"#;
    let program = build_program(source);
    let input = parser::parse(r#"!eval escape_it s="café élève naïve""#)
        .unwrap()
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .unwrap();
    let result = eval_call_with_input(&program, &input.function_name, &input.input).unwrap();
    assert_eq!(
        result, r#""café élève naïve""#,
        "json_escape should preserve UTF-8 chars"
    );
}

#[test]
fn test_value_display_utf8_string() {
    // Verify Value::String Display preserves multi-byte UTF-8
    let val = Value::string("café ☕ 日本語");
    let displayed = format!("{val}");
    assert_eq!(
        displayed, r#""café ☕ 日本語""#,
        "Value display should preserve UTF-8"
    );
}

#[test]
fn test_concat_preserves_utf8() {
    let source = r#"
+fn greet (name:String)->String
  +return concat("Bonjour, ", name)
"#;
    let program = build_program(source);
    let input = parser::parse(r#"!eval greet name="André""#)
        .unwrap()
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .unwrap();
    let result = eval_call_with_input(&program, &input.function_name, &input.input).unwrap();
    assert_eq!(
        result, r#""Bonjour, André""#,
        "concat should preserve UTF-8"
    );
}

#[test]
fn test_unicode_string_literal_roundtrip() {
    // Full Unicode string literal: parse → eval → display must preserve
    // all multi-byte characters exactly.
    let source = "
+fn probe ()->String
  +return \"hé — 你好 ✓ ★\"
";
    let program = build_program(source);
    let input = parser::parse("!eval probe")
        .unwrap()
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .unwrap();
    let result = eval_call_with_input(&program, &input.function_name, &input.input).unwrap();
    assert_eq!(
        result, "\"hé — 你好 ✓ ★\"",
        "Unicode string literal must survive parse/eval/display without mojibake"
    );
    // Verify actual byte representation
    let inner = &result[1..result.len() - 1]; // strip quotes
    assert_eq!(
        inner.as_bytes(),
        "hé — 你好 ✓ ★".as_bytes(),
        "UTF-8 byte representation must match"
    );
}

#[test]
fn test_unicode_json_serialization_roundtrip() {
    // Build JSON containing Unicode via json_escape, then extract via json_get.
    // This simulates what send_message_body does with Unicode text.
    let source = r#"
+fn build_json (text:String)->String
  +let escaped:String = json_escape(text)
  +let body:String = concat("{\"text\":\"", concat(escaped, "\"}"))
  +return body

+fn extract_text (json:String)->String
  +return json_get(json, "text")

+fn roundtrip (text:String)->String
  +let json:String = build_json(text)
  +return extract_text(json)
"#;
    let program = build_program(source);
    let input = parser::parse("!eval roundtrip text=\"café — 你好 ✓ ★\"")
        .unwrap()
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .unwrap();
    let result = eval_call_with_input(&program, &input.function_name, &input.input).unwrap();
    assert_eq!(
        result, "\"café — 你好 ✓ ★\"",
        "Unicode text must survive json_escape → JSON embedding → json_get roundtrip"
    );
}

#[tokio::test]
async fn test_mocked_http_utf8_roundtrip() {
    // Simulate an http_get returning UTF-8 text via mock, then extracting it.
    let mut session = crate::session::Session::new();

    let source = "
+fn fetch_text (url:String)->String [io,async]
  +await resp:String = http_get(url)
  +return json_get(resp, \"text\")
";
    let _ = session.apply_async(source, None).await;

    // Mock returns JSON with Unicode content
    let mock_source = "!mock http_get \"unicode-test\" -> \"{\\\"text\\\":\\\"café — 你好 ✓\\\"}\"";
    let _ = session.apply_async(mock_source, None).await;

    let test_source = "
+test fetch_text
  +with url=\"https://unicode-test.example.com\" -> expect \"café — 你好 ✓\"
";
    let results = session.apply_async(test_source, None).await.unwrap();
    assert_eq!(results.len(), 1);
    assert!(
        results[0].1,
        "mocked HTTP UTF-8 test should pass: {:?}",
        results[0]
    );
    assert!(
        results[0].0.contains("PASS"),
        "mocked HTTP returning UTF-8 JSON should round-trip: {:?}",
        results[0]
    );
}

#[tokio::test]
async fn test_mocked_llm_utf8_roundtrip() {
    // Simulate llm_call returning UTF-8 text via mock.
    let mut session = crate::session::Session::new();

    let source = "
+fn ask_llm (prompt:String)->String [io,async]
  +await reply:String = llm_call(prompt, \"echo\")
  +return reply
";
    let _ = session.apply_async(source, None).await;

    // Mock llm_call to return Unicode text
    let mock_source = "!mock llm_call \"test\" -> \"café — 你好 ✓ ★\"";
    let _ = session.apply_async(mock_source, None).await;

    let test_source = "
+test ask_llm
  +with prompt=\"test prompt\" -> expect \"café — 你好 ✓ ★\"
";
    let results = session.apply_async(test_source, None).await.unwrap();
    assert_eq!(results.len(), 1);
    assert!(
        results[0].1,
        "mocked LLM UTF-8 test should pass: {:?}",
        results[0]
    );
}

// ── Pure function calls in test parameters ────────────────────────

#[test]
fn test_zero_arg_function_as_test_param_value() {
    // A function call with () used as a test parameter value
    // should call the function and use its return value
    let source = "\
+type Config = {host:String, port:Int}

+fn make_default ()->Config
  +let c:Config = {host: \"localhost\", port: 8080}
  +return c

+fn get_host (c:Config)->String
  +return c.host

+test get_host
  +with c=make_default() -> expect \"localhost\"
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    assert_eq!(cases.len(), 1);
    let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
    assert!(
        result.is_ok(),
        "zero-arg function call as param value should work: {:?}",
        result
    );
    assert!(result.unwrap().contains("expected \"localhost\""));
}

#[test]
fn test_bare_function_name_as_test_param_value() {
    // A bare function name (no parens) for a zero-arg function should
    // be called, not turned into Err("make_default")
    let source = "\
+type Config = {host:String, port:Int}

+fn make_default ()->Config
  +let c:Config = {host: \"localhost\", port: 8080}
  +return c

+fn get_host (c:Config)->String
  +return c.host

+test get_host
  +with c=make_default -> expect \"localhost\"
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    assert_eq!(cases.len(), 1);
    let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
    assert!(
        result.is_ok(),
        "bare function name as param value should call function: {:?}",
        result
    );
    assert!(result.unwrap().contains("expected \"localhost\""));
}

#[test]
fn test_function_call_with_args_in_test_param() {
    // A function call with arguments in a test parameter should work
    let source = "\
+type Config = {host:String, port:Int}

+fn make_config (h:String, p:Int)->Config
  +let c:Config = {host: h, port: p}
  +return c

+fn get_port (c:Config)->Int
  +return c.port

+test get_port
  +with c=make_config(\"example.com\", 3000) -> expect 3000
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    assert_eq!(cases.len(), 1);
    let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
    assert!(
        result.is_ok(),
        "function call with args as param value: {:?}",
        result
    );
    assert!(result.unwrap().contains("expected 3000"));
}

#[test]
fn test_function_call_in_expected_value() {
    // Function calls should also work on the expected side of ->
    let source = "\
+type Config = {host:String, port:Int}

+fn make_default ()->Config
  +let c:Config = {host: \"localhost\", port: 8080}
  +return c

+fn identity (c:Config)->Config
  +return c

+test identity
  +with c=make_default() -> expect make_default()
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    assert_eq!(cases.len(), 1);
    let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
    assert!(
        result.is_ok(),
        "function call in expected value: {:?}",
        result
    );
    assert!(result.unwrap().contains("expected"));
}

// ── Positional (space-separated) args in !eval and +with ──────────

#[test]
fn test_eval_positional_multiple_strings() {
    let source = "\
+fn concat_two (a:String, b:String)->String
  +let result:String = concat(a, b)
  +return result
";
    let program = build_program(source);
    let eval_source = r#"!eval concat_two "hello" "world""#;
    let ops = parser::parse(eval_source).expect("parse should succeed");
    let ev = ops
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .expect("should have eval op");
    let result = eval_compiled_or_interpreted(&program, &ev.function_name, &ev.input);
    assert!(
        result.is_ok(),
        "positional string args should work: {:?}",
        result
    );
    assert_eq!(result.unwrap().0, "\"helloworld\"");
}

#[test]
fn test_eval_positional_mixed_types() {
    let source = "\
+fn show (a:String, b:Int)->String
  +let result:String = concat(a, to_string(b))
  +return result
";
    let program = build_program(source);
    let eval_source = r#"!eval show "count:" 42"#;
    let ops = parser::parse(eval_source).expect("parse should succeed");
    let ev = ops
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .expect("should have eval op");
    let result = eval_compiled_or_interpreted(&program, &ev.function_name, &ev.input);
    assert!(result.is_ok(), "mixed positional args: {:?}", result);
    assert_eq!(result.unwrap().0, "\"count:42\"");
}

#[test]
fn test_eval_positional_ints() {
    let source = "\
+fn add (a:Int, b:Int)->Int
  +let result:Int = a + b
  +return result
";
    let program = build_program(source);
    let eval_source = "!eval add 3 4";
    let ops = parser::parse(eval_source).expect("parse should succeed");
    let ev = ops
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .expect("should have eval op");
    let result = eval_compiled_or_interpreted(&program, &ev.function_name, &ev.input);
    assert!(result.is_ok(), "positional int args: {:?}", result);
    assert_eq!(result.unwrap().0, "7");
}

#[test]
fn test_with_positional_strings() {
    let source = "\
+fn concat_two (a:String, b:String)->String
  +let result:String = concat(a, b)
  +return result

+test concat_two
  +with \"hello\" \"world\" -> expect \"helloworld\"
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    assert_eq!(cases.len(), 1);
    let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
    assert!(result.is_ok(), "+with positional strings: {:?}", result);
    assert!(result.unwrap().contains("expected \"helloworld\""));
}

#[test]
fn test_eval_builtin_positional_strings() {
    // Test that positional args also work for builtins
    let program = ast::Program::default();
    let eval_source = r#"!eval concat "foo" "bar""#;
    let ops = parser::parse(eval_source).expect("parse should succeed");
    let ev = ops
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .expect("should have eval op");
    let result = eval_compiled_or_interpreted(&program, &ev.function_name, &ev.input);
    assert!(result.is_ok(), "builtin positional strings: {:?}", result);
    assert_eq!(result.unwrap().0, "\"foobar\"");
}

// ── Inline expression eval (!eval <expr>) ─────────────────────────

#[test]
fn test_eval_inline_arithmetic() {
    let program = ast::Program::default();
    let source = "!eval 1 + 2";
    let ops = parser::parse(source).expect("parse should succeed");
    let ev = ops
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .expect("should have eval op");
    assert!(ev.inline_expr.is_some(), "should be inline expression");
    let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
    assert!(result.is_ok(), "inline 1+2: {:?}", result);
    assert_eq!(format!("{}", result.unwrap()), "3");
}

#[test]
fn test_eval_inline_multiply_add() {
    let program = ast::Program::default();
    let source = "!eval 3 * 4 + 1";
    let ops = parser::parse(source).expect("parse should succeed");
    let ev = ops
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .expect("should have eval op");
    assert!(ev.inline_expr.is_some());
    let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
    assert!(result.is_ok(), "inline 3*4+1: {:?}", result);
    assert_eq!(format!("{}", result.unwrap()), "13");
}

#[test]
fn test_eval_inline_string_literal() {
    let program = ast::Program::default();
    let source = r#"!eval "hello""#;
    let ops = parser::parse(source).expect("parse should succeed");
    let ev = ops
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .expect("should have eval op");
    assert!(ev.inline_expr.is_some());
    let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
    assert!(result.is_ok(), "inline string: {:?}", result);
    assert_eq!(format!("{}", result.unwrap()), "\"hello\"");
}

#[test]
fn test_eval_inline_numeric_literal() {
    let program = ast::Program::default();
    let source = "!eval 42";
    let ops = parser::parse(source).expect("parse should succeed");
    let ev = ops
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .expect("should have eval op");
    assert!(ev.inline_expr.is_some());
    let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
    assert!(result.is_ok(), "inline 42: {:?}", result);
    assert_eq!(format!("{}", result.unwrap()), "42");
}

#[test]
fn test_eval_inline_boolean() {
    let program = ast::Program::default();
    let source = "!eval true";
    let ops = parser::parse(source).expect("parse should succeed");
    let ev = ops
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .expect("should have eval op");
    assert!(ev.inline_expr.is_some());
    let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
    assert!(result.is_ok(), "inline true: {:?}", result);
    assert_eq!(format!("{}", result.unwrap()), "true");
}

#[test]
fn test_eval_inline_comparison() {
    let program = ast::Program::default();
    let source = "!eval 3 > 2";
    let ops = parser::parse(source).expect("parse should succeed");
    let ev = ops
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .expect("should have eval op");
    assert!(ev.inline_expr.is_some());
    let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
    assert!(result.is_ok(), "inline 3>2: {:?}", result);
    assert_eq!(format!("{}", result.unwrap()), "true");
}

#[test]
fn test_eval_inline_concat_call() {
    let program = ast::Program::default();
    let source = r#"!eval concat("hello", " ", "world")"#;
    let ops = parser::parse(source).expect("parse should succeed");
    let ev = ops
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .expect("should have eval op");
    assert!(ev.inline_expr.is_some());
    let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
    assert!(result.is_ok(), "inline concat: {:?}", result);
    assert_eq!(format!("{}", result.unwrap()), "\"hello world\"");
}

#[test]
fn test_eval_inline_len_call() {
    let program = ast::Program::default();
    let source = r#"!eval len("hello")"#;
    let ops = parser::parse(source).expect("parse should succeed");
    let ev = ops
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .expect("should have eval op");
    assert!(ev.inline_expr.is_some());
    let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
    assert!(result.is_ok(), "inline len: {:?}", result);
    assert_eq!(format!("{}", result.unwrap()), "5");
}

#[test]
fn test_error_suggest_matches_known_pattern() {
    let program = ast::Program::default();
    let mut env = Env::new();
    let result = eval_builtin_or_user(
        &program,
        "error_suggest",
        vec![Value::string("undefined variable `user_id`")],
        &mut env,
    )
    .unwrap();
    assert_eq!(
        format!("{result}"),
        r#""Check variable spelling. Variables must be declared with +let or +call before use. Function parameters use the exact names from the signature.""#
    );
}

#[test]
fn test_error_suggest_unknown_pattern_returns_empty_string() {
    let program = ast::Program::default();
    let mut env = Env::new();
    let result = eval_builtin_or_user(
        &program,
        "error_suggest",
        vec![Value::string("weird custom failure")],
        &mut env,
    )
    .unwrap();
    assert_eq!(format!("{result}"), "\"\"");
}

#[test]
fn test_eval_inline_nested_calls() {
    let program = ast::Program::default();
    let source = r#"!eval len(concat("a", "b"))"#;
    let ops = parser::parse(source).expect("parse should succeed");
    let ev = ops
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .expect("should have eval op");
    assert!(ev.inline_expr.is_some());
    let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
    assert!(result.is_ok(), "inline nested calls: {:?}", result);
    assert_eq!(format!("{}", result.unwrap()), "2");
}

#[test]
fn test_eval_inline_struct_literal() {
    let program = ast::Program::default();
    let source = r#"!eval {name: "alice", age: 25}"#;
    let ops = parser::parse(source).expect("parse should succeed");
    let ev = ops
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .expect("should have eval op");
    assert!(ev.inline_expr.is_some());
    let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
    assert!(result.is_ok(), "inline struct: {:?}", result);
    let val = result.unwrap();
    // Struct should contain the fields
    match val {
        Value::Struct(_, fields) => {
            let name_id = intern::intern_display("name");
            let age_id = intern::intern_display("age");
            assert!(
                matches!(fields.get(&name_id), Some(Value::String(s)) if s.as_str() == "alice"),
                "expected name=alice"
            );
            assert!(
                matches!(fields.get(&age_id), Some(Value::Int(25))),
                "expected age=25"
            );
        }
        _ => panic!("expected struct, got {val}"),
    }
}

#[test]
fn test_eval_inline_list_creation() {
    let program = ast::Program::default();
    let source = "!eval list(1, 2, 3)";
    let ops = parser::parse(source).expect("parse should succeed");
    let ev = ops
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .expect("should have eval op");
    assert!(ev.inline_expr.is_some());
    let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
    assert!(result.is_ok(), "inline list: {:?}", result);
    assert_eq!(format!("{}", result.unwrap()), "[1, 2, 3]");
}

#[test]
fn test_eval_inline_user_function_call() {
    // Inline expression calling a user-defined function
    let source = "\
+fn double (x:Int)->Int
  +let result:Int = x * 2
  +return result
";
    let program = build_program(source);
    let eval_source = "!eval double(5)";
    let ops = parser::parse(eval_source).expect("parse should succeed");
    let ev = ops
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .expect("should have eval op");
    assert!(ev.inline_expr.is_some());
    let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
    assert!(result.is_ok(), "inline user fn call: {:?}", result);
    assert_eq!(format!("{}", result.unwrap()), "10");
}

#[test]
fn test_eval_func_name_still_works() {
    // Existing !eval func_name syntax should still work
    let source = "\
+fn greet ()->String
  +return \"hello\"
";
    let program = build_program(source);
    let eval_source = "!eval greet";
    let ops = parser::parse(eval_source).expect("parse should succeed");
    let ev = ops
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .expect("should have eval op");
    assert!(
        ev.inline_expr.is_none(),
        "bare function name should not be inline"
    );
    let result = eval_compiled_or_interpreted(&program, &ev.function_name, &ev.input);
    assert!(result.is_ok(), "bare function name: {:?}", result);
    assert_eq!(result.unwrap().0, "\"hello\"");
}

#[test]
fn test_eval_func_with_args_still_works() {
    // Existing !eval func_name arg1 arg2 syntax should still work
    let source = "\
+fn add (a:Int, b:Int)->Int
  +let result:Int = a + b
  +return result
";
    let program = build_program(source);
    let eval_source = "!eval add 3 4";
    let ops = parser::parse(eval_source).expect("parse should succeed");
    let ev = ops
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .expect("should have eval op");
    assert!(ev.inline_expr.is_none(), "func + args should not be inline");
    let result = eval_compiled_or_interpreted(&program, &ev.function_name, &ev.input);
    assert!(result.is_ok(), "func with args: {:?}", result);
    assert_eq!(result.unwrap().0, "7");
}

// ── expr_contains_io_builtin detection ─────────────────────────────

#[test]
fn test_expr_contains_io_builtin_detects_direct_call() {
    // shell_exec("echo hello") should be detected as IO
    let source = r#"!eval shell_exec("echo hello")"#;
    let ops = parser::parse(source).expect("parse should succeed");
    let ev = ops
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .expect("should have eval op");
    assert!(ev.inline_expr.is_some());
    assert!(
        expr_contains_io_builtin(ev.inline_expr.as_ref().unwrap()),
        "shell_exec should be detected as IO builtin"
    );
}

#[test]
fn test_expr_contains_io_builtin_detects_nested_call() {
    // concat("result: ", shell_exec("echo hi")) — IO in args
    let source = r#"!eval concat("result: ", shell_exec("echo hi"))"#;
    let ops = parser::parse(source).expect("parse should succeed");
    let ev = ops
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .expect("should have eval op");
    assert!(ev.inline_expr.is_some());
    assert!(
        expr_contains_io_builtin(ev.inline_expr.as_ref().unwrap()),
        "nested shell_exec in concat args should be detected"
    );
}

#[test]
fn test_expr_contains_io_builtin_false_for_sync() {
    // concat("a", "b") is NOT an IO builtin
    let source = r#"!eval concat("a", "b")"#;
    let ops = parser::parse(source).expect("parse should succeed");
    let ev = ops
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .expect("should have eval op");
    assert!(ev.inline_expr.is_some());
    assert!(
        !expr_contains_io_builtin(ev.inline_expr.as_ref().unwrap()),
        "concat should NOT be detected as IO builtin"
    );
}

#[test]
fn test_expr_contains_io_builtin_false_for_arithmetic() {
    let source = "!eval 1 + 2";
    let ops = parser::parse(source).expect("parse should succeed");
    let ev = ops
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .expect("should have eval op");
    assert!(ev.inline_expr.is_some());
    assert!(
        !expr_contains_io_builtin(ev.inline_expr.as_ref().unwrap()),
        "arithmetic should NOT be detected as IO builtin"
    );
}

#[test]
fn test_expr_contains_io_builtin_detects_http_get() {
    let source = r#"!eval http_get("http://example.com")"#;
    let ops = parser::parse(source).expect("parse should succeed");
    let ev = ops
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .expect("should have eval op");
    assert!(ev.inline_expr.is_some());
    assert!(
        expr_contains_io_builtin(ev.inline_expr.as_ref().unwrap()),
        "http_get should be detected as IO builtin"
    );
}

#[test]
fn test_eval_inline_io_without_handle_still_errors() {
    // When no coroutine handle is available, IO builtins should still error
    let program = ast::Program::default();
    let source = r#"!eval shell_exec("echo hello")"#;
    let ops = parser::parse(source).expect("parse should succeed");
    let ev = ops
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .expect("should have eval op");
    let result = eval_inline_expr(&program, ev.inline_expr.as_ref().unwrap());
    assert!(result.is_err(), "IO builtin without handle should error");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("async IO operation"),
        "error should mention async IO: {err}"
    );
}

#[test]
fn test_eval_inline_io_with_coroutine_handle_via_mock() {
    // When a coroutine handle IS available (via eval_inline_expr_with_io),
    // IO builtins should execute through it.
    // We use a full tokio runtime + coroutine Runtime to test this end-to-end.
    let program = ast::Program::default();
    let source = r#"!eval println("test message")"#;
    let ops = parser::parse(source).expect("parse should succeed");
    let ev = ops
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .expect("should have eval op");
    let expr = ev.inline_expr.unwrap();

    // Spin up a real tokio runtime with coroutine IO loop
    let result = std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async {
            let (runtime, mut io_rx) = crate::coroutine::Runtime::new();
            let runtime = std::sync::Arc::new(runtime);
            let io_sender = runtime.io_sender();

            // Spawn IO loop to handle requests (same pattern as eval_test_case_with_mocks)
            let rt_handle = runtime.clone();
            let io_loop = tokio::spawn(async move {
                while let Some(request) = io_rx.recv().await {
                    let rt = rt_handle.clone();
                    tokio::spawn(async move {
                        rt.handle_io(request).await;
                    });
                }
            });

            let eval_result = tokio::task::spawn_blocking(move || {
                eval_inline_expr_with_io(&program, &expr, io_sender)
            })
            .await
            .unwrap();

            // Shut down the IO loop
            io_loop.abort();

            eval_result
        })
    })
    .join()
    .unwrap();

    // println returns "" (empty string) on success
    assert!(
        result.is_ok(),
        "println via IO handle should succeed: {:?}",
        result
    );
}

#[test]
fn test_eval_inline_io_builtin_in_user_function_context() {
    // eval_builtin_or_user should execute IO builtins when __coroutine_handle
    // is present in the env, rather than rejecting them
    let program = ast::Program::default();
    let mut env = Env::new();

    // Create a mock coroutine handle (will error on actual IO, but won't
    // give the "is an async IO operation" rejection)
    let handle = crate::coroutine::CoroutineHandle::new_mock(vec![crate::session::IoMock {
        operation: "println".to_string(),
        patterns: vec![],
        response: "".to_string(),
    }]);
    env.set("__coroutine_handle", Value::CoroutineHandle(handle));

    let result = eval_builtin_or_user(&program, "println", vec![Value::string("hello")], &mut env);
    // With mock, println should succeed (mocked response)
    assert!(
        result.is_ok(),
        "IO builtin with coroutine handle should not reject: {:?}",
        result
    );
}

// ── Side-effect checks for function calls in test params ──────────

#[test]
fn test_impure_function_rejected_in_test_param_bare() {
    // A function with [io,async] effects used as a bare name in a test
    // param should produce a clear error, not a confusing runtime failure
    let source = "\
+type State = {count:Int, name:String}

+fn fetch_state ()->State [io,async]
  +await data:String = http_get(\"http://example.com\")
  +let s:State = {count: 0, name: data}
  +return s

+fn get_name (state:State)->String
  +return state.name

+test get_name
  +with state=fetch_state -> expect \"default\"
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    assert_eq!(cases.len(), 1);
    let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
    assert!(
        result.is_err(),
        "impure function should be rejected: {:?}",
        result
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("side effects"),
        "error should mention side effects: {err}"
    );
    assert!(
        err.contains("fetch_state"),
        "error should name the function: {err}"
    );
}

#[test]
fn test_impure_function_rejected_in_test_param_call() {
    // Same but with parens: state=fetch_state()
    let source = "\
+type State = {count:Int, name:String}

+fn fetch_state ()->State [io,async]
  +await data:String = http_get(\"http://example.com\")
  +let s:State = {count: 0, name: data}
  +return s

+fn get_name (state:State)->String
  +return state.name

+test get_name
  +with state=fetch_state() -> expect \"default\"
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    assert_eq!(cases.len(), 1);
    let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
    assert!(
        result.is_err(),
        "impure function call should be rejected: {:?}",
        result
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("side effects"),
        "error should mention side effects: {err}"
    );
}

#[test]
fn test_fail_effect_allowed_in_test_param() {
    // [fail] is not a side effect — it should be allowed
    let source = "\
+type Config = {host:String, port:Int}

+fn validated_config (host:String, port:Int)->Config [fail]
  +check valid_port port > 0 ~err_invalid_port
  +let c:Config = {host: host, port: port}
  +return c

+fn get_host (cfg:Config)->String
  +return cfg.host

+test get_host
  +with cfg=validated_config(\"localhost\", 8080) -> expect \"localhost\"
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    assert_eq!(cases.len(), 1);
    let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
    assert!(
        result.is_ok(),
        "[fail] function should be allowed in test params: {:?}",
        result
    );
    assert!(result.unwrap().contains("expected \"localhost\""));
}

// ── Escaped quotes in test value strings ──────────────────────────

#[test]
fn test_escaped_quotes_in_key_value_string() {
    // Strings with escaped quotes in key=value test params should work
    let source = r#"
+fn identity (s:String)->String
  +return s

+test identity
  +with s="hello\"world" -> expect "hello\"world"
"#;
    let program = build_program(source);
    let cases = extract_test_cases(source);
    assert_eq!(cases.len(), 1);
    let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
    assert!(
        result.is_ok(),
        "escaped quotes in key=value string: {:?}",
        result
    );
}

#[test]
fn test_escaped_quotes_multiple_string_args() {
    // Multiple string args with escaped quotes should parse correctly
    let source = r#"
+fn concat_two (a:String, b:String)->String
  +let result:String = concat(a, b)
  +return result

+test concat_two
  +with a="he\"llo" b="wo\"rld" -> expect "he\"llowo\"rld"
"#;
    let program = build_program(source);
    let cases = extract_test_cases(source);
    assert_eq!(cases.len(), 1);
    let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
    assert!(
        result.is_ok(),
        "escaped quotes in multiple args: {:?}",
        result
    );
}

#[test]
fn test_newline_and_tab_escapes_in_test_value() {
    let source = r#"
+fn identity (s:String)->String
  +return s

+test identity
  +with s="line1\nline2" -> expect "line1\nline2"
"#;
    let program = build_program(source);
    let cases = extract_test_cases(source);
    assert_eq!(cases.len(), 1);
    let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
    assert!(result.is_ok(), "newline escape in test value: {:?}", result);
}

// ── IO builtin error message tests ──────────────────────────────

#[test]
fn test_io_builtin_without_await_gives_helpful_error() {
    // Calling an IO builtin like http_get without +await should give a
    // specific error, not "undefined function"
    let source = "\
+fn broken (url:String)->String
  +let resp:String = http_get(url)
  +return resp
";
    let program = build_program(source);
    let test_source = "\
+test broken
  +with url=\"http://example.com\" -> expect \"\"
";
    let cases = extract_test_cases(test_source);
    let (fn_name, case) = &cases[0];
    let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
    assert!(
        result.is_err(),
        "should fail when IO builtin used without +await"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("async IO operation"),
        "error should mention async IO: {err}"
    );
    assert!(err.contains("+await"), "error should suggest +await: {err}");
    assert!(
        !err.contains("undefined function"),
        "should NOT say undefined function: {err}"
    );
}

// ── +each scoping tests ──────────────────────────────────────────

#[test]
fn test_each_return_propagation() {
    // +return inside +each should propagate to the enclosing function
    let source = "\
+type Message = {role:String, content:String}

+fn find_role (messages:List<Message>)->String
  +each messages msg:Message
    +return msg.role
  +end
  +return \"none\"

+test find_role
  +with messages=list({role: \"user\", content: \"hi\"}) -> expect \"user\"
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    assert_eq!(cases.len(), 1);
    let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
    assert!(
        result.is_ok(),
        "+return inside +each should propagate: {:?}",
        result
    );
}

#[test]
fn test_each_field_access_and_let() {
    // +let with field access on the loop variable should work
    let source = "\
+type Message = {role:String, content:String}

+fn build_text (messages:List<Message>)->String
  +let result:String = \"\"
  +each messages msg:Message
    +let role:String = msg.role
    +set result = concat(result, role, \" \")
  +end
  +return result

+test build_text
  +with messages=list({role: \"user\", content: \"hi\"}, {role: \"bot\", content: \"hello\"}) -> expect \"user bot \"
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    assert_eq!(cases.len(), 1);
    let result = eval_test_case_with_mocks(&program, &cases[0].0, &cases[0].1, &[], &[]);
    assert!(
        result.is_ok(),
        "+let with field access in +each: {:?}",
        result
    );
}

// ── format_expr round-trip tests ──────────────────────────────────

#[test]
fn test_format_expr_struct_with_list_roundtrip() {
    // Bug: format_expr produced `messages=list()last_id=0` (missing separator)
    // for struct fields containing function calls, causing reparse failure.

    let source = "\
+type State = {messages:List<String>, last_id:Int}

+fn process (s:State)->Int
  +return s.last_id
";
    let program = build_program(source);

    // Simulate a stored test with struct input containing list()
    let test_source = "\
+test process
  +with s={messages: list(), last_id: 0} -> expect 0
";
    let cases = extract_test_cases(test_source);
    assert_eq!(cases.len(), 1);
    let (fn_name, case) = &cases[0];

    // First, verify the test passes directly
    let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
    assert!(result.is_ok(), "direct test should pass: {:?}", result);

    // Now simulate the store/reparse cycle that invalidate_and_retest does
    let input_str = crate::session::format_expr_pub(&case.input);
    let expected_str = crate::session::format_expr_pub(&case.expected);

    // Reconstruct test source (same as invalidate_and_retest)
    let reconstructed = format!(
        "+test process\n  +with {} -> expect {}\n",
        input_str, expected_str
    );

    // The reconstructed source must parse successfully
    let reparse = parser::parse(&reconstructed);
    assert!(
        reparse.is_ok(),
        "reconstructed test should parse: {:?} from source: {}",
        reparse.err(),
        reconstructed
    );

    // And the reparsed test should pass
    let reparsed_cases = extract_test_cases(&reconstructed);
    assert_eq!(reparsed_cases.len(), 1);
    let result = eval_test_case_with_mocks(
        &program,
        &reparsed_cases[0].0,
        &reparsed_cases[0].1,
        &[],
        &[],
    );
    assert!(result.is_ok(), "reparsed test should pass: {:?}", result);
}

// ── Test matchers ────────────────────────────────────────────────

#[test]
fn test_contains_matcher() {
    let source = "\
+fn greet (name:String)->String
  +return concat(\"hello \", name, \"!\")

+test greet
  +with name=\"world\" -> expect contains(\"hello\")
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    assert_eq!(cases.len(), 1);
    let (fn_name, case) = &cases[0];
    assert!(case.matcher.is_some(), "should have a matcher");
    let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
    assert!(result.is_ok(), "contains matcher should pass: {:?}", result);
}

#[test]
fn test_contains_matcher_fails_when_not_present() {
    let source = "\
+fn greet (name:String)->String
  +return concat(\"hello \", name, \"!\")

+test greet
  +with name=\"world\" -> expect contains(\"goodbye\")
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
    assert!(
        result.is_err(),
        "contains matcher should fail when substring absent"
    );
}

#[test]
fn test_starts_with_matcher() {
    let source = "\
+fn greet (name:String)->String
  +return concat(\"hello \", name)

+test greet
  +with name=\"world\" -> expect starts_with(\"hello\")
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
    assert!(
        result.is_ok(),
        "starts_with matcher should pass: {:?}",
        result
    );
}

#[test]
fn test_starts_with_matcher_fails() {
    let source = "\
+fn greet (name:String)->String
  +return concat(\"hello \", name)

+test greet
  +with name=\"world\" -> expect starts_with(\"goodbye\")
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
    assert!(
        result.is_err(),
        "starts_with matcher should fail on wrong prefix"
    );
}

#[test]
fn test_any_ok_matcher() {
    let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x

+test validate
  +with x=5 -> expect Ok
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    assert!(matches!(case.matcher, Some(parser::TestMatcher::AnyOk)));
    let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
    assert!(
        result.is_ok(),
        "AnyOk matcher should pass for Ok result: {:?}",
        result
    );
}

#[test]
fn test_any_ok_matcher_fails_on_err() {
    let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x

+test validate
  +with x=-1 -> expect Ok
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
    assert!(result.is_err(), "AnyOk matcher should fail on Err result");
}

#[test]
fn test_any_err_matcher() {
    let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x

+test validate
  +with x=-1 -> expect Err
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    assert!(matches!(case.matcher, Some(parser::TestMatcher::AnyErr)));
    let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
    assert!(
        result.is_ok(),
        "AnyErr matcher should pass for Err result: {:?}",
        result
    );
}

#[test]
fn test_any_err_matcher_fails_on_ok() {
    let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x

+test validate
  +with x=5 -> expect Err
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
    assert!(result.is_err(), "AnyErr matcher should fail on Ok result");
}

#[test]
fn test_err_containing_matcher() {
    let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x

+test validate
  +with x=-1 -> expect Err(\"err_negative\")
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    assert!(matches!(
        case.matcher,
        Some(parser::TestMatcher::ErrContaining(_))
    ));
    let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
    assert!(
        result.is_ok(),
        "ErrContaining matcher should pass: {:?}",
        result
    );
}

#[test]
fn test_err_containing_matcher_fails_on_wrong_msg() {
    let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x

+test validate
  +with x=-1 -> expect Err(\"err_something_else\")
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
    assert!(
        result.is_err(),
        "ErrContaining should fail on wrong message"
    );
}

// ── +after checks ────────────────────────────────────────────────

#[test]
fn test_after_routes_contains_pass() {
    // Build a program with a route, then test with +after routes contains
    let source = "\
+fn handler (body:String)->String
  +return \"ok\"

+route POST \"/chat\" -> handler

+test handler
  +with body=\"hello\" -> expect \"ok\"
  +after routes contains \"/chat\"
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    assert_eq!(cases.len(), 1);
    let (fn_name, case) = &cases[0];
    assert_eq!(case.after_checks.len(), 1);
    assert_eq!(case.after_checks[0].target, "routes");
    assert_eq!(case.after_checks[0].value, "/chat");

    // Routes are now in RuntimeState, pass them explicitly
    let routes = vec![crate::ast::HttpRoute {
        method: "POST".to_string(),
        path: "/chat".to_string(),
        handler_fn: "handler".to_string(),
    }];
    let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &routes);
    assert!(result.is_ok(), "+after routes should pass: {:?}", result);
}

#[test]
fn test_after_routes_contains_fail() {
    // Route doesn't exist — +after should fail
    let source = "\
+fn handler (body:String)->String
  +return \"ok\"

+test handler
  +with body=\"hello\" -> expect \"ok\"
  +after routes contains \"/nonexistent\"
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
    assert!(
        result.is_err(),
        "+after routes should fail when route missing"
    );
}

#[test]
fn test_after_modules_contains_pass() {
    let source = "\
+module TestMod

+fn helper ()->Int
  +return 42

+test helper
  +with -> expect 42
  +after modules contains \"TestMod\"
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
    assert!(result.is_ok(), "+after modules should pass: {:?}", result);
}

#[test]
fn test_after_modules_contains_fail() {
    let source = "\
+fn helper ()->Int
  +return 42

+test helper
  +with -> expect 42
  +after modules contains \"NonExistent\"
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
    assert!(
        result.is_err(),
        "+after modules should fail when module missing"
    );
}

#[test]
fn test_matchers_combined_with_after() {
    // Combine a matcher with +after checks
    let source = "\
+fn handler (body:String)->String
  +return concat(\"processed: \", body)

+route GET \"/api\" -> handler

+test handler
  +with body=\"test\" -> expect contains(\"processed\")
  +after routes contains \"/api\"
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    assert!(case.matcher.is_some());
    assert_eq!(case.after_checks.len(), 1);
    let routes = vec![crate::ast::HttpRoute {
        method: "GET".to_string(),
        path: "/api".to_string(),
        handler_fn: "handler".to_string(),
    }];
    let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &routes);
    assert!(result.is_ok(), "matcher + after should pass: {:?}", result);
}

#[test]
fn test_multiple_after_checks() {
    let source = "\
+module MyMod

+fn handler (body:String)->String
  +return \"ok\"

+route POST \"/webhook\" -> handler

+test handler
  +with body=\"x\" -> expect \"ok\"
  +after routes contains \"/webhook\"
  +after modules contains \"MyMod\"
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    assert_eq!(case.after_checks.len(), 2);
    let routes = vec![crate::ast::HttpRoute {
        method: "POST".to_string(),
        path: "/webhook".to_string(),
        handler_fn: "handler".to_string(),
    }];
    let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &routes);
    assert!(
        result.is_ok(),
        "multiple +after checks should pass: {:?}",
        result
    );
}

#[test]
fn test_exact_err_still_works_without_matcher() {
    // Err(err_label) without quotes should still work as exact match (no matcher)
    let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x

+test validate
  +with x=-1 -> expect Err(err_negative)
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    // No matcher — this is the old exact match behavior
    assert!(
        case.matcher.is_none(),
        "Err(bare_ident) should not create a matcher"
    );
    let result = eval_test_case_with_mocks(&program, fn_name, case, &[], &[]);
    assert!(
        result.is_ok(),
        "exact Err match should still work: {:?}",
        result
    );
}

// ═════════════════════════════════════════════════════════════════════
// Inline expression helpers for builtin/arithmetic tests
// ═════════════════════════════════════════════════════════════════════

/// Evaluate an inline expression string and return the formatted result.
fn eval_expr_str(program: &ast::Program, expr_text: &str) -> String {
    let expr = parser::parse_expr_pub(0, expr_text)
        .unwrap_or_else(|e| panic!("parse failed for `{expr_text}`: {e}"));
    let val = eval_inline_expr(program, &expr)
        .unwrap_or_else(|e| panic!("eval failed for `{expr_text}`: {e}"));
    format!("{val}")
}

/// Evaluate and return the raw Value (for type-specific assertions).
fn eval_expr_val(program: &ast::Program, expr_text: &str) -> Value {
    let expr = parser::parse_expr_pub(0, expr_text)
        .unwrap_or_else(|e| panic!("parse failed for `{expr_text}`: {e}"));
    eval_inline_expr(program, &expr)
        .unwrap_or_else(|e| panic!("eval failed for `{expr_text}`: {e}"))
}

/// Evaluate a full program and return the result of eval'ing a function.
/// Uses interpreter-only path to avoid JIT crashes in test context.
fn eval_fn_result(source: &str, fn_name: &str, input: &str) -> String {
    let program = build_program(source);
    let input_expr = if input.is_empty() {
        parser::Expr::StructLiteral(vec![])
    } else {
        parser::parse_test_input(0, input).expect("parse input failed")
    };
    eval_call_with_input(&program, fn_name, &input_expr)
        .unwrap_or_else(|e| panic!("eval `{fn_name}` failed: {e}"))
}

// ═════════════════════════════════════════════════════════════════════
// Arithmetic operations (+, -, *, /, %)
// ═════════════════════════════════════════════════════════════════════

#[test]
fn arith_subtraction() {
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, "10 - 3"), "7");
}

#[test]
fn arith_division() {
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, "10 / 3"), "3");
}

#[test]
fn arith_modulo() {
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, "10 % 3"), "1");
}

#[test]
fn arith_negative_result() {
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, "3 - 10"), "-7");
}

#[test]
fn arith_compound_precedence() {
    let p = ast::Program::default();
    // (2 * 3) + (10 / 5) - 1 = 6 + 2 - 1 = 7
    assert_eq!(eval_expr_str(&p, "2 * 3 + 10 / 5 - 1"), "7");
}

#[test]
fn arith_float_basic() {
    let p = ast::Program::default();
    let val = eval_expr_val(&p, "3.5 + 1.5");
    match val {
        Value::Float(f) => assert!((f - 5.0).abs() < 1e-10),
        _ => panic!("expected Float, got {val}"),
    }
}

// ═════════════════════════════════════════════════════════════════════
// Comparison and boolean logic
// ═════════════════════════════════════════════════════════════════════

#[test]
fn cmp_less_than() {
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, "2 < 5"), "true");
    assert_eq!(eval_expr_str(&p, "5 < 2"), "false");
}

#[test]
fn cmp_gte_lte() {
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, "5 >= 5"), "true");
    assert_eq!(eval_expr_str(&p, "5 <= 5"), "true");
    assert_eq!(eval_expr_str(&p, "4 >= 5"), "false");
    assert_eq!(eval_expr_str(&p, "6 <= 5"), "false");
}

#[test]
fn cmp_eq_neq() {
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, "42 == 42"), "true");
    assert_eq!(eval_expr_str(&p, "42 != 42"), "false");
    assert_eq!(eval_expr_str(&p, "42 != 43"), "true");
}

#[test]
fn bool_and_or_not() {
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, "true AND true"), "true");
    assert_eq!(eval_expr_str(&p, "true AND false"), "false");
    assert_eq!(eval_expr_str(&p, "false OR true"), "true");
    assert_eq!(eval_expr_str(&p, "false OR false"), "false");
    assert_eq!(eval_expr_str(&p, "NOT true"), "false");
    assert_eq!(eval_expr_str(&p, "NOT false"), "true");
}

// ═════════════════════════════════════════════════════════════════════
// Builtin function calls
// ═════════════════════════════════════════════════════════════════════

#[test]
fn builtin_split() {
    let p = ast::Program::default();
    assert_eq!(
        eval_expr_str(&p, r#"split("a,b,c", ",")"#),
        r#"["a", "b", "c"]"#
    );
}

#[test]
fn builtin_join() {
    let p = ast::Program::default();
    // join uses Display format for list items — strings include quotes
    assert_eq!(
        eval_expr_str(&p, "join(list(1, 2, 3), \"-\")"),
        r#""1-2-3""#
    );
}

#[test]
fn builtin_trim() {
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, r#"trim("  hello  ")"#), r#""hello""#);
}

#[test]
fn builtin_to_string() {
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, "to_string(42)"), r#""42""#);
    assert_eq!(eval_expr_str(&p, "to_string(true)"), r#""true""#);
}

#[test]
fn builtin_to_int() {
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, r#"to_int("42")"#), "42");
    assert_eq!(eval_expr_str(&p, "to_int(3.7)"), "3");
    assert_eq!(eval_expr_str(&p, "to_int(true)"), "1");
}

#[test]
fn builtin_push_and_get() {
    let p = ast::Program::default();
    // push returns a new list
    assert_eq!(eval_expr_str(&p, "push(list(1, 2), 3)"), "[1, 2, 3]");
    // get returns element at index
    assert_eq!(eval_expr_str(&p, "get(list(10, 20, 30), 1)"), "20");
}

#[test]
fn builtin_char_at() {
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, r#"char_at("hello", 0)"#), r#""h""#);
    assert_eq!(eval_expr_str(&p, r#"char_at("hello", 4)"#), r#""o""#);
}

#[test]
fn builtin_substring() {
    let p = ast::Program::default();
    assert_eq!(
        eval_expr_str(&p, r#"substring("hello world", 0, 5)"#),
        r#""hello""#
    );
    assert_eq!(
        eval_expr_str(&p, r#"substring("hello world", 6, 11)"#),
        r#""world""#
    );
}

#[test]
fn builtin_starts_with_ends_with() {
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, r#"starts_with("hello", "hel")"#), "true");
    assert_eq!(eval_expr_str(&p, r#"starts_with("hello", "xyz")"#), "false");
    assert_eq!(eval_expr_str(&p, r#"ends_with("hello", "llo")"#), "true");
    assert_eq!(eval_expr_str(&p, r#"ends_with("hello", "xyz")"#), "false");
}

#[test]
fn builtin_contains() {
    let p = ast::Program::default();
    assert_eq!(
        eval_expr_str(&p, r#"contains("hello world", "world")"#),
        "true"
    );
    assert_eq!(
        eval_expr_str(&p, r#"contains("hello world", "xyz")"#),
        "false"
    );
}

#[test]
fn builtin_index_of() {
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, r#"index_of("hello", "ll")"#), "2");
    assert_eq!(eval_expr_str(&p, r#"index_of("hello", "xyz")"#), "-1");
}

#[test]
fn builtin_abs() {
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, "abs(5)"), "5");
}

#[test]
fn builtin_min_max() {
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, "min(3, 7)"), "3");
    assert_eq!(eval_expr_str(&p, "max(3, 7)"), "7");
}

#[test]
fn builtin_floor() {
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, "floor(3.7)"), "3");
    assert_eq!(eval_expr_str(&p, "floor(3.2)"), "3");
}

#[test]
fn builtin_pow() {
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, "pow(2, 10)"), "1024");
}

#[test]
fn builtin_len_on_list() {
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, "len(list(1, 2, 3))"), "3");
    assert_eq!(eval_expr_str(&p, "len(list())"), "0");
}

#[test]
fn builtin_base64_encode() {
    let p = ast::Program::default();
    assert_eq!(
        eval_expr_str(&p, r#"base64_encode("hello")"#),
        r#""aGVsbG8=""#
    );
}

#[test]
fn builtin_digit_value() {
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, r#"digit_value("5")"#), "5");
    assert_eq!(eval_expr_str(&p, r#"digit_value("a")"#), "-1");
}

#[test]
fn builtin_is_digit_char() {
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, r#"is_digit_char("7")"#), "true");
    assert_eq!(eval_expr_str(&p, r#"is_digit_char("x")"#), "false");
}

#[test]
fn builtin_char_code_and_from_char_code() {
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, r#"char_code("A")"#), "65");
    assert_eq!(eval_expr_str(&p, "from_char_code(65)"), r#""A""#);
}

#[test]
fn builtin_bitwise_ops() {
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, "bit_and(12, 10)"), "8"); // 1100 & 1010 = 1000
    assert_eq!(eval_expr_str(&p, "bit_or(12, 10)"), "14"); // 1100 | 1010 = 1110
    assert_eq!(eval_expr_str(&p, "bit_xor(12, 10)"), "6"); // 1100 ^ 1010 = 0110
    assert_eq!(eval_expr_str(&p, "bit_shl(1, 3)"), "8"); // 1 << 3 = 8
    assert_eq!(eval_expr_str(&p, "bit_shr(8, 2)"), "2"); // 8 >> 2 = 2
}

// ═════════════════════════════════════════════════════════════════════
// Let bindings and variable lookup
// ═════════════════════════════════════════════════════════════════════

#[test]
fn let_binding_and_return() {
    let result = eval_fn_result(
        "\
+fn double (x:Int)->Int
  +let result:Int = x * 2
  +return result
",
        "double",
        "5",
    );
    assert_eq!(result, "10");
}

#[test]
fn let_multiple_bindings() {
    let result = eval_fn_result(
        "\
+fn compute (a:Int, b:Int)->Int
  +let sum:Int = a + b
  +let product:Int = a * b
  +let result:Int = sum + product
  +return result
",
        "compute",
        "a=3 b=4",
    );
    // sum=7, product=12, result=19
    assert_eq!(result, "19");
}

#[test]
fn set_mutation() {
    let result = eval_fn_result(
        "\
+fn count_up ()->Int
  +let i:Int = 0
  +set i = i + 1
  +set i = i + 1
  +set i = i + 1
  +return i
",
        "count_up",
        "",
    );
    assert_eq!(result, "3");
}

// ═════════════════════════════════════════════════════════════════════
// If/elif/else control flow
// ═════════════════════════════════════════════════════════════════════

#[test]
fn if_then_branch() {
    let source = "\
+fn classify (x:Int)->String
  +if x > 0
    +return \"positive\"
  +else
    +return \"non-positive\"
  +end
";
    assert_eq!(eval_fn_result(source, "classify", "5"), "\"positive\"");
    assert_eq!(eval_fn_result(source, "classify", "0"), "\"non-positive\"");
}

#[test]
fn if_elif_else() {
    let source = "\
+fn describe (x:Int)->String
  +if x > 0
    +return \"positive\"
  +elif x == 0
    +return \"zero\"
  +else
    +return \"negative\"
  +end
";
    assert_eq!(eval_fn_result(source, "describe", "5"), "\"positive\"");
    assert_eq!(eval_fn_result(source, "describe", "0"), "\"zero\"");
    assert_eq!(eval_fn_result(source, "describe", "-3"), "\"negative\"");
}

#[test]
fn nested_if() {
    let source = "\
+fn size (x:Int)->String
  +if x > 0
    +if x > 100
      +return \"big\"
    +else
      +return \"small\"
    +end
  +else
    +return \"non-positive\"
  +end
";
    assert_eq!(eval_fn_result(source, "size", "200"), "\"big\"");
    assert_eq!(eval_fn_result(source, "size", "50"), "\"small\"");
    assert_eq!(eval_fn_result(source, "size", "-1"), "\"non-positive\"");
}

// ═════════════════════════════════════════════════════════════════════
// Pattern matching on union types
// ═════════════════════════════════════════════════════════════════════

#[test]
fn match_on_union() {
    let source = "\
+type Shape = Circle(Float) | Rect(Float, Float) | Point

+fn area (s:Shape)->Float
  +match s
  +case Circle(r)
    +return r * r * 3.14
  +case Rect(w, h)
    +return w * h
  +case Point
    +return 0.0
  +end
";
    let program = build_program(source);
    // Test with Circle(5.0)
    let input = parser::parse_test_input(0, "Circle(5.0)").unwrap();
    let result = eval_call_with_input(&program, "area", &input).unwrap();
    let f: f64 = result.parse().unwrap();
    assert!((f - 78.5).abs() < 0.01, "Circle area: {result}");

    // Test with Rect(3.0, 4.0)
    let input = parser::parse_test_input(0, "Rect(3.0, 4.0)").unwrap();
    let result = eval_call_with_input(&program, "area", &input).unwrap();
    let f: f64 = result.parse().unwrap();
    assert!((f - 12.0).abs() < 0.01, "Rect area: {result}");
}

#[test]
fn match_wildcard() {
    let source = "\
+type Color = Red | Green | Blue

+fn is_red (c:Color)->Bool
  +match c
  +case Red
    +return true
  +case _
    +return false
  +end
";
    let program = build_program(source);
    let input = parser::parse_test_input(0, "Red").unwrap();
    let result = eval_call_with_input(&program, "is_red", &input).unwrap();
    assert_eq!(result, "true");

    let input = parser::parse_test_input(0, "Blue").unwrap();
    let result = eval_call_with_input(&program, "is_red", &input).unwrap();
    assert_eq!(result, "false");
}

#[test]
fn match_recursive_type() {
    let source = "\
+type Expr = Literal(Int) | Add(Expr, Expr)

+fn eval_expr (e:Expr)->Int
  +match e
  +case Literal(val)
    +return val
  +case Add(left, right)
    +let l:Int = eval_expr(left)
    +let r:Int = eval_expr(right)
    +return l + r
  +end
";
    let program = build_program(source);
    // Add(Literal(2), Literal(3)) should be 5
    let input = parser::parse_test_input(0, "Add(Literal(2), Literal(3))").unwrap();
    let result = eval_call_with_input(&program, "eval_expr", &input).unwrap();
    assert_eq!(result, "5");

    // Add(Literal(1), Add(Literal(2), Literal(3))) should be 6
    let input =
        parser::parse_test_input(0, "Add(Literal(1), Add(Literal(2), Literal(3)))").unwrap();
    let result = eval_call_with_input(&program, "eval_expr", &input).unwrap();
    assert_eq!(result, "6");
}

// ═════════════════════════════════════════════════════════════════════
// Check statements (success and failure)
// ═════════════════════════════════════════════════════════════════════

#[test]
fn check_passes() {
    let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +check small x < 100 ~err_too_large
  +return x
";
    // [fail] functions wrap successful returns in Ok
    assert_eq!(eval_fn_result(source, "validate", "50"), "Ok(50)");
}

#[test]
fn check_fails_first() {
    let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x
";
    let result = eval_fn_result(source, "validate", "-5");
    assert!(
        result.contains("Err") && result.contains("err_negative"),
        "expected Err(err_negative), got: {result}"
    );
}

#[test]
fn check_fails_second() {
    let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +check small x < 100 ~err_too_large
  +return x
";
    let result = eval_fn_result(source, "validate", "200");
    assert!(
        result.contains("Err") && result.contains("err_too_large"),
        "expected Err(err_too_large), got: {result}"
    );
}

// ═════════════════════════════════════════════════════════════════════
// Function calls with error propagation ([fail] effect)
// ═════════════════════════════════════════════════════════════════════

#[test]
fn function_call_chain() {
    let source = "\
+fn double (x:Int)->Int
  +let r:Int = x * 2
  +return r

+fn quadruple (x:Int)->Int
  +let d:Int = double(x)
  +let r:Int = double(d)
  +return r
";
    assert_eq!(eval_fn_result(source, "quadruple", "5"), "20");
}

#[test]
fn function_call_with_check_propagation() {
    // When a [fail] function calls another [fail] function that fails,
    // the error should propagate via the +call binding
    let source = "\
+fn validate_positive (x:Int)->Result<Int> [fail]
  +check pos x > 0 ~err_negative
  +return x

+fn process (x:Int)->Result<Int> [fail]
  +call val:Int = validate_positive(x)
  +let result:Int = val * 2
  +return result
";
    assert_eq!(eval_fn_result(source, "process", "5"), "Ok(10)");
    let result = eval_fn_result(source, "process", "-1");
    assert!(
        result.contains("Err"),
        "expected Err propagation, got: {result}"
    );
}

// ═════════════════════════════════════════════════════════════════════
// Struct construction and field access
// ═════════════════════════════════════════════════════════════════════

#[test]
fn struct_create_and_access() {
    let source = "\
+type Point = x:Int, y:Int

+fn get_x (p:Point)->Int
  +return p.x

+fn make_point ()->Point
  +let p:Point = {x: 10, y: 20}
  +return p
";
    let result = eval_fn_result(source, "make_point", "");
    // Field order in HashMap is not deterministic, check both fields present
    assert!(result.contains("x: 10"), "expected x: 10 in {result}");
    assert!(result.contains("y: 20"), "expected y: 20 in {result}");
    assert_eq!(eval_fn_result(source, "get_x", "x=5 y=10"), "5");
}

#[test]
fn struct_field_access_in_expression() {
    let source = "\
+type Rect = width:Int, height:Int

+fn area (r:Rect)->Int
  +let a:Int = r.width * r.height
  +return a
";
    assert_eq!(eval_fn_result(source, "area", "width=3 height=4"), "12");
}

#[test]
fn struct_nested_field_access() {
    let source = "\
+type Inner = value:Int
+type Outer = inner:Inner, label:String

+fn get_value (o:Outer)->Int
  +return o.inner.value
";
    assert_eq!(
        eval_fn_result(source, "get_value", r#"inner={value: 42} label="test""#),
        "42"
    );
}

// ═════════════════════════════════════════════════════════════════════
// While loops
// ═════════════════════════════════════════════════════════════════════

#[test]
fn while_loop_counter() {
    let source = "\
+fn count_to (n:Int)->Int
  +let i:Int = 0
  +while i < n
    +set i = i + 1
  +end
  +return i
";
    assert_eq!(eval_fn_result(source, "count_to", "5"), "5");
    assert_eq!(eval_fn_result(source, "count_to", "0"), "0");
}

#[test]
fn while_loop_accumulator() {
    let source = "\
+fn sum_to (n:Int)->Int
  +let total:Int = 0
  +let i:Int = 1
  +while i <= n
    +set total = total + i
    +set i = i + 1
  +end
  +return total
";
    assert_eq!(eval_fn_result(source, "sum_to", "10"), "55");
}

// ═════════════════════════════════════════════════════════════════════
// String concat with mixed types
// ═════════════════════════════════════════════════════════════════════

#[test]
fn concat_mixed_types() {
    let p = ast::Program::default();
    // concat coerces non-strings to string via Display
    assert_eq!(
        eval_expr_str(&p, r#"concat("count: ", 42)"#),
        r#""count: 42""#
    );
    assert_eq!(
        eval_expr_str(&p, r#"concat("flag: ", true)"#),
        r#""flag: true""#
    );
}

// ═════════════════════════════════════════════════════════════════════
// Error cases for builtins
// ═════════════════════════════════════════════════════════════════════

#[test]
fn builtin_get_out_of_bounds() {
    let p = ast::Program::default();
    let expr = parser::parse_expr_pub(0, "get(list(1, 2), 5)").unwrap();
    let result = eval_inline_expr(&p, &expr);
    assert!(result.is_err(), "expected error for out-of-bounds get");
}

#[test]
fn builtin_char_at_out_of_bounds() {
    let p = ast::Program::default();
    let expr = parser::parse_expr_pub(0, r#"char_at("hi", 10)"#).unwrap();
    let result = eval_inline_expr(&p, &expr);
    assert!(result.is_err(), "expected error for out-of-bounds char_at");
}

#[test]
fn builtin_to_int_invalid() {
    let p = ast::Program::default();
    let expr = parser::parse_expr_pub(0, r#"to_int("not_a_number")"#).unwrap();
    let result = eval_inline_expr(&p, &expr);
    assert!(result.is_err(), "expected error for invalid to_int");
}

// ═════════════════════════════════════════════════════════════════════
// Missing builtins: bit_not, left_rotate, to_hex, u32_wrap, sqrt
// ═════════════════════════════════════════════════════════════════════

#[test]
fn builtin_bit_not() {
    let p = ast::Program::default();
    // bit_not(0) = -1 (all bits flipped in two's complement)
    assert_eq!(eval_expr_str(&p, "bit_not(0)"), "-1");
    // bit_not(-1) = 0
    assert_eq!(eval_expr_str(&p, "bit_not(-1)"), "0");
}

#[test]
fn builtin_left_rotate() {
    let p = ast::Program::default();
    // 1 rotated left by 4 = 16 (32-bit rotation)
    assert_eq!(eval_expr_str(&p, "left_rotate(1, 4)"), "16");
    // rotl alias works too
    assert_eq!(eval_expr_str(&p, "rotl(1, 4)"), "16");
}

#[test]
fn builtin_to_hex() {
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, "to_hex(255)"), r#""000000ff""#);
    assert_eq!(eval_expr_str(&p, "to_hex(0)"), r#""00000000""#);
}

#[test]
fn builtin_u32_wrap() {
    let p = ast::Program::default();
    // Large value wraps to 32-bit unsigned
    assert_eq!(eval_expr_str(&p, "u32_wrap(256)"), "256");
    // Negative wraps around
    let val = eval_expr_val(&p, "u32_wrap(-1)");
    match val {
        Value::Int(n) => assert_eq!(n, 4294967295), // u32::MAX
        _ => panic!("expected Int, got {val}"),
    }
}

#[test]
fn builtin_sqrt() {
    let p = ast::Program::default();
    let val = eval_expr_val(&p, "sqrt(25)");
    match val {
        Value::Float(f) => assert!((f - 5.0).abs() < 1e-10),
        _ => panic!("expected Float, got {val}"),
    }
    let val = eval_expr_val(&p, "sqrt(2.0)");
    match val {
        Value::Float(f) => assert!((f - 1.41421356).abs() < 1e-5),
        _ => panic!("expected Float, got {val}"),
    }
}

// ═════════════════════════════════════════════════════════════════════
// Ok/Err/Some constructors
// ═════════════════════════════════════════════════════════════════════

#[test]
fn constructor_ok() {
    let p = ast::Program::default();
    let val = eval_expr_val(&p, "Ok(42)");
    match val {
        Value::Ok(inner) => {
            assert!(matches!(*inner, Value::Int(42)));
        }
        _ => panic!("expected Ok(42), got {val}"),
    }
}

#[test]
fn constructor_err() {
    let p = ast::Program::default();
    let val = eval_expr_val(&p, r#"Err("not found")"#);
    match val {
        Value::Err(msg) => assert_eq!(&*msg, "\"not found\""),
        _ => panic!("expected Err, got {val}"),
    }
}

#[test]
fn constructor_some() {
    let p = ast::Program::default();
    // Some(5) via parser expression path creates a Union variant
    let val = eval_expr_val(&p, "Some(5)");
    // The parser expr path treats Some as a generic union constructor
    let display = format!("{val}");
    assert!(
        display.contains("5"),
        "expected Some containing 5, got {display}"
    );
}

#[test]
fn constructor_ok_no_args() {
    let p = ast::Program::default();
    // Ok() with no args wraps None
    let val = eval_expr_val(&p, "Ok()");
    assert!(
        matches!(val, Value::Ok(ref inner) if matches!(**inner, Value::None)),
        "expected Ok(None), got {val}"
    );
}

// ═════════════════════════════════════════════════════════════════════
// regex_match and regex_replace
// ═════════════════════════════════════════════════════════════════════

#[test]
fn builtin_regex_match() {
    let p = ast::Program::default();
    assert_eq!(
        eval_expr_str(&p, r#"regex_match("^[0-9]+$", "12345")"#),
        "true"
    );
    assert_eq!(
        eval_expr_str(&p, r#"regex_match("^[0-9]+$", "abc")"#),
        "false"
    );
    assert_eq!(
        eval_expr_str(&p, r#"regex_match("[a-z]+", "Hello World")"#),
        "true"
    );
}

#[test]
fn builtin_regex_replace() {
    let p = ast::Program::default();
    assert_eq!(
        eval_expr_str(&p, r#"regex_replace("[0-9]+", "NUM", "foo123bar456")"#),
        r#""fooNUMbarNUM""#
    );
    assert_eq!(
        eval_expr_str(&p, r#"regex_replace("\\s+", "-", "hello   world")"#),
        r#""hello-world""#
    );
}

// ═════════════════════════════════════════════════════════════════════
// Result/Option pattern matching
// ═════════════════════════════════════════════════════════════════════

#[test]
fn match_on_result_ok() {
    let source = "\
+fn unwrap_or_default (r:Result<Int>)->Int
  +match r
  +case Ok(val)
    +return val
  +case Err(msg)
    +return 0
  +end
";
    // Pass Ok(42)
    assert_eq!(eval_fn_result(source, "unwrap_or_default", "Ok(42)"), "42");
}

#[test]
fn match_on_result_err() {
    let source = "\
+fn unwrap_or_default (r:Result<Int>)->Int
  +match r
  +case Ok(val)
    +return val
  +case Err(msg)
    +return 0
  +end
";
    // Pass Err("fail")
    assert_eq!(
        eval_fn_result(source, "unwrap_or_default", r#"Err("fail")"#),
        "0"
    );
}

#[test]
fn match_on_option_some_none() {
    let source = "\
+type Maybe = Just(Int) | Nothing

+fn get_or_zero (m:Maybe)->Int
  +match m
  +case Just(val)
    +return val
  +case Nothing
    +return 0
  +end
";
    assert_eq!(eval_fn_result(source, "get_or_zero", "Just(99)"), "99");
    assert_eq!(eval_fn_result(source, "get_or_zero", "Nothing"), "0");
}

// ═════════════════════════════════════════════════════════════════════
// Effect checking — pure function cannot call IO
// ═════════════════════════════════════════════════════════════════════

#[test]
fn pure_function_rejects_io_call() {
    // A pure function (no effects) calling an IO builtin produces an error
    // result (the function returns an Err value, not a hard crash)
    let result = eval_fn_result(
        "\
+fn bad ()->String
  +return http_get(\"http://example.com\")
",
        "bad",
        "",
    );
    // The eval catches the IO-without-await error and returns it
    assert!(
        result.contains("async IO operation") || result.contains("http_get"),
        "expected IO rejection message, got: {result}"
    );
}

#[test]
fn fail_effect_wraps_result() {
    // [fail] functions wrap their return in Ok on success, Err on check failure
    let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x
";
    let result = eval_fn_result(source, "validate", "10");
    assert!(
        result.starts_with("Ok("),
        "expected Ok wrapping, got: {result}"
    );
    let result = eval_fn_result(source, "validate", "-1");
    assert!(
        result.starts_with("Err("),
        "expected Err wrapping, got: {result}"
    );
}

// ═════════════════════════════════════════════════════════════════════
// Each loop over lists
// ═════════════════════════════════════════════════════════════════════

#[test]
fn each_loop_accumulate() {
    let source = "\
+fn sum_list (items:List<Int>)->Int
  +let total:Int = 0
  +each items item:Int
    +set total = total + item
  +end
  +return total
";
    assert_eq!(
        eval_fn_result(source, "sum_list", "list(1, 2, 3, 4, 5)"),
        "15"
    );
}

#[test]
fn each_loop_transform() {
    let source = "\
+fn double_all (items:List<Int>)->List<Int>
  +let result:List<Int> = list()
  +each items item:Int
    +set result = push(result, item * 2)
  +end
  +return result
";
    assert_eq!(
        eval_fn_result(source, "double_all", "list(1, 2, 3)"),
        "[2, 4, 6]"
    );
}

#[test]
fn each_loop_empty_list() {
    let source = "\
+fn sum_list (items:List<Int>)->Int
  +let total:Int = 0
  +each items item:Int
    +set total = total + item
  +end
  +return total
";
    assert_eq!(eval_fn_result(source, "sum_list", "list()"), "0");
}

// ═════════════════════════════════════════════════════════════════════
// While loop with early return
// ═════════════════════════════════════════════════════════════════════

#[test]
fn while_loop_factorial() {
    let source = "\
+fn factorial (n:Int)->Int
  +let result:Int = 1
  +let i:Int = 1
  +while i <= n
    +set result = result * i
    +set i = i + 1
  +end
  +return result
";
    assert_eq!(eval_fn_result(source, "factorial", "5"), "120");
    assert_eq!(eval_fn_result(source, "factorial", "0"), "1");
    assert_eq!(eval_fn_result(source, "factorial", "1"), "1");
}

// ═════════════════════════════════════════════════════════════════════
// If/elif/else chains (additional cases)
// ═════════════════════════════════════════════════════════════════════

#[test]
fn if_multiple_elif() {
    let source = "\
+fn grade (score:Int)->String
  +if score >= 90
    +return \"A\"
  +elif score >= 80
    +return \"B\"
  +elif score >= 70
    +return \"C\"
  +elif score >= 60
    +return \"D\"
  +else
    +return \"F\"
  +end
";
    assert_eq!(eval_fn_result(source, "grade", "95"), "\"A\"");
    assert_eq!(eval_fn_result(source, "grade", "85"), "\"B\"");
    assert_eq!(eval_fn_result(source, "grade", "75"), "\"C\"");
    assert_eq!(eval_fn_result(source, "grade", "65"), "\"D\"");
    assert_eq!(eval_fn_result(source, "grade", "50"), "\"F\"");
}

#[test]
fn if_with_complex_condition() {
    let source = "\
+fn in_range (x:Int)->Bool
  +if x >= 10 AND x <= 20
    +return true
  +else
    +return false
  +end
";
    assert_eq!(eval_fn_result(source, "in_range", "15"), "true");
    assert_eq!(eval_fn_result(source, "in_range", "5"), "false");
    assert_eq!(eval_fn_result(source, "in_range", "25"), "false");
}

// ═════════════════════════════════════════════════════════════════════
// Check statement (additional edge cases)
// ═════════════════════════════════════════════════════════════════════

#[test]
fn check_with_compound_condition() {
    let source = "\
+fn validate_range (x:Int)->Result<Int> [fail]
  +check in_range x >= 0 AND x <= 100 ~err_out_of_range
  +return x
";
    assert_eq!(eval_fn_result(source, "validate_range", "50"), "Ok(50)");
    let result = eval_fn_result(source, "validate_range", "-1");
    assert!(
        result.contains("err_out_of_range"),
        "expected err_out_of_range, got: {result}"
    );
    let result = eval_fn_result(source, "validate_range", "200");
    assert!(
        result.contains("err_out_of_range"),
        "expected err_out_of_range, got: {result}"
    );
}

#[test]
fn check_multiple_sequential() {
    // Multiple checks, each with different error labels
    let source = "\
+fn validate_user (name:String, age:Int)->Result<String> [fail]
  +check has_name len(name) > 0 ~err_empty_name
  +check valid_age age >= 0 AND age <= 150 ~err_bad_age
  +return concat(name, \" is valid\")
";
    assert_eq!(
        eval_fn_result(source, "validate_user", r#"name="alice" age=25"#),
        r#"Ok("alice is valid")"#
    );
    let result = eval_fn_result(source, "validate_user", r#"name="" age=25"#);
    assert!(result.contains("err_empty_name"), "got: {result}");
    let result = eval_fn_result(source, "validate_user", r#"name="bob" age=-5"#);
    assert!(result.contains("err_bad_age"), "got: {result}");
}

// ═════════════════════════════════════════════════════════════════════
// Match with nested variant bindings
// ═════════════════════════════════════════════════════════════════════

#[test]
fn match_nested_variant_bindings() {
    let source = "\
+type Tree = Leaf(Int) | Node(Tree, Tree)

+fn tree_sum (t:Tree)->Int
  +match t
  +case Leaf(val)
    +return val
  +case Node(left, right)
    +let l:Int = tree_sum(left)
    +let r:Int = tree_sum(right)
    +return l + r
  +end
";
    // Leaf(5) = 5
    assert_eq!(eval_fn_result(source, "tree_sum", "Leaf(5)"), "5");
    // Node(Leaf(3), Leaf(7)) = 10
    assert_eq!(
        eval_fn_result(source, "tree_sum", "Node(Leaf(3), Leaf(7))"),
        "10"
    );
    // Node(Node(Leaf(1), Leaf(2)), Leaf(3)) = 6
    assert_eq!(
        eval_fn_result(source, "tree_sum", "Node(Node(Leaf(1), Leaf(2)), Leaf(3))"),
        "6"
    );
}

// ═════════════════════════════════════════════════════════════════════
// Architecture: fork_runtime_for_test isolation
// ═════════════════════════════════════════════════════════════════════

#[test]
fn fork_runtime_creates_isolated_shared_vars() {
    let source = "\
+module Counter
+shared count:Int = 0
+fn get_count ()->Int
  +return count
";
    let program = build_program(source);
    let routes = vec![];
    let forked = fork_runtime_for_test(&program, &routes);
    assert!(forked.is_some());
    let rt = forked.unwrap();
    let state = rt.read().unwrap();
    // Should have Counter.count initialized to 0
    assert_eq!(
        state
            .shared_vars
            .get("Counter.count")
            .map(|v| format!("{v}")),
        Some("0".to_string()),
        "forked runtime should have Counter.count=0"
    );
}

#[test]
fn fork_runtime_mutation_does_not_affect_original() {
    let source = "\
+module State
+shared value:Int = 10
+fn get ()->Int
  +return value
";
    let program = build_program(source);
    let routes = vec![];

    let forked1 = fork_runtime_for_test(&program, &routes).unwrap();
    let forked2 = fork_runtime_for_test(&program, &routes).unwrap();

    // Mutate forked1
    {
        let mut state = forked1.write().unwrap();
        state
            .shared_vars
            .insert("State.value".to_string(), Value::Int(99));
    }

    // forked2 should still have original value
    {
        let state = forked2.read().unwrap();
        assert!(
            matches!(state.shared_vars.get("State.value"), Some(Value::Int(10))),
            "forked2 should be unaffected by forked1 mutation"
        );
    }
}

#[test]
fn fork_runtime_includes_http_routes() {
    let program = ast::Program::default();
    let routes = vec![ast::HttpRoute {
        method: "POST".to_string(),
        path: "/webhook".to_string(),
        handler_fn: "handle".to_string(),
    }];
    let forked = fork_runtime_for_test(&program, &routes).unwrap();
    let state = forked.read().unwrap();
    assert_eq!(state.http_routes.len(), 1);
    assert_eq!(state.http_routes[0].path, "/webhook");
}

#[test]
fn fork_runtime_empty_program_no_shared_vars() {
    let program = ast::Program::default();
    let forked = fork_runtime_for_test(&program, &[]).unwrap();
    let state = forked.read().unwrap();
    assert!(state.shared_vars.is_empty());
    assert!(state.http_routes.is_empty());
}

// ═════════════════════════════════════════════════════════════════════
// Architecture: +shared variable behavior
// ═════════════════════════════════════════════════════════════════════

#[test]
fn shared_var_populated_in_env() {
    let source = "\
+module Config
+shared debug:Bool = false
+shared max_retries:Int = 3
+fn get_retries ()->Int
  +return max_retries
";
    let program = build_program(source);
    let mut env = Env::new();
    env.populate_shared_from_program(&program);
    // Should have both shared vars in the cache
    assert!(matches!(
        env.shared_cache.get("Config.debug"),
        Some(Value::Bool(false))
    ));
    assert!(matches!(
        env.shared_cache.get("Config.max_retries"),
        Some(Value::Int(3))
    ));
}

#[test]
fn shared_var_accessible_via_qualified_name() {
    // Shared vars are stored as "Module.name" keys
    let source = "\
+module App
+shared counter:Int = 42
+fn get ()->Int
  +return 0
";
    let program = build_program(source);
    let mut env = Env::new();
    env.populate_shared_from_program(&program);
    // Direct cache lookup with qualified key
    assert!(matches!(
        env.shared_cache.get("App.counter"),
        Some(Value::Int(42))
    ));
}

#[test]
fn shared_var_multiple_modules() {
    let source = "\
+module A
+shared x:Int = 1
+fn get_x ()->Int
  +return x

+module B
+shared y:Int = 2
+fn get_y ()->Int
  +return y
";
    let program = build_program(source);
    let mut env = Env::new();
    env.populate_shared_from_program(&program);
    assert!(matches!(env.shared_cache.get("A.x"), Some(Value::Int(1))));
    assert!(matches!(env.shared_cache.get("B.y"), Some(Value::Int(2))));
}

#[test]
#[ignore] // Run with: cargo test --release bench_vm_vs_treewalker -- --ignored --nocapture
fn bench_vm_vs_treewalker() {
    use std::time::Instant;

    let source = r#"
+module Bench
+fn fib (n:Int)->Int
  +if n <= 1
    +return n
  +else
    +return fib(n - 1) + fib(n - 2)
  +end
+end

+fn sum_to (n:Int)->Int
  +let total:Int = 0
  +let i:Int = 0
  +while i <= n
    +set total = total + i
    +set i = i + 1
  +end
  +return total
+end

+fn collatz_steps (n:Int)->Int
  +let steps:Int = 0
  +let current:Int = n
  +while current != 1
    +if current % 2 == 0
      +set current = current / 2
    +else
      +set current = current * 3 + 1
    +end
    +set steps = steps + 1
  +end
  +return steps
+end

+fn nested_loops (n:Int)->Int
  +let total:Int = 0
  +let i:Int = 0
  +while i < n
    +let j:Int = 0
    +while j < n
      +set total = total + i * j
      +set j = j + 1
    +end
    +set i = i + 1
  +end
  +return total
+end
"#;

    let program = build_program(source);

    struct Bench {
        name: &'static str,
        input: &'static str,
        expected: i64,
    }

    let benches = vec![
        Bench {
            name: "Bench.fib",
            input: "25",
            expected: 75025,
        },
        Bench {
            name: "Bench.sum_to",
            input: "9000",
            expected: 40504500,
        },
        Bench {
            name: "Bench.collatz_steps",
            input: "27",
            expected: 111,
        },
        Bench {
            name: "Bench.nested_loops",
            input: "100",
            expected: 24502500,
        },
    ];

    println!("\n=== Adapsis VM vs Tree-Walker Benchmark ===\n");
    println!(
        "{:<25} {:>12} {:>12} {:>10}",
        "Function", "Tree-Walk", "VM", "Speedup"
    );
    println!("{}", "-".repeat(62));

    for b in &benches {
        let func = program.get_function(b.name).unwrap();
        let input_expr = crate::parser::parse_expr_pub(0, b.input).unwrap();

        // Tree-walker: warmup + measure
        let _ = eval_call_with_input(&program, b.name, &input_expr);
        let start = Instant::now();
        let tw_result = eval_call_with_input(&program, b.name, &input_expr).unwrap();
        let tw_time = start.elapsed();
        let tw_str = format!("{tw_result}");
        assert_eq!(
            tw_str,
            b.expected.to_string(),
            "tree-walk mismatch for {}",
            b.name
        );

        // VM: warmup + measure
        let vm_args_warmup = input_to_vm_args(&input_expr, func).unwrap();
        let compiled = crate::vm::compile_function(func, &program).unwrap();
        let _ = crate::vm::execute_with_io(&compiled, vm_args_warmup, &program, &|_, _| {
            anyhow::bail!("no IO")
        });
        let vm_args = input_to_vm_args(&input_expr, func).unwrap();
        let start = Instant::now();
        let vm_result = crate::vm::execute_with_io(&compiled, vm_args, &program, &|_, _| {
            anyhow::bail!("no IO")
        })
        .unwrap();
        let vm_time = start.elapsed();
        let vm_str = format!("{vm_result}");
        assert_eq!(vm_str, b.expected.to_string(), "VM mismatch for {}", b.name);

        let speedup = tw_time.as_secs_f64() / vm_time.as_secs_f64();
        println!(
            "{:<25} {:>10.2}ms {:>10.2}ms {:>9.1}x",
            b.name,
            tw_time.as_secs_f64() * 1000.0,
            vm_time.as_secs_f64() * 1000.0,
            speedup,
        );
    }
    println!();
}

// ═════════════════════════════════════════════════════════════════════
// make_shared_program_mut / read_back_program_mutations
// ═════════════════════════════════════════════════════════════════════

#[test]
fn make_shared_program_mut_creates_writable_clone() {
    let mut program = crate::ast::Program::default();
    // Add a function to verify the clone works
    let ops =
        crate::parser::parse("+fn hello ()->String\n  +return \"hi\"\n+end").expect("parse failed");
    for op in &ops {
        crate::validator::apply_and_validate(&mut program, op).unwrap();
    }
    program.rebuild_function_index();

    let shared = make_shared_program_mut(&program);

    // Should be able to read the program
    {
        let p = shared.read().unwrap();
        assert!(p.get_function("hello").is_some());
    }

    // Should be able to write to the program
    {
        let mut p = shared.write().unwrap();
        p.functions.clear();
        p.rebuild_function_index();
    }

    // Verify mutation took effect
    {
        let p = shared.read().unwrap();
        assert!(p.get_function("hello").is_none());
    }
}

#[test]
fn read_back_returns_mutated_state() {
    let mut program = crate::ast::Program::default();
    let ops =
        crate::parser::parse("+fn hello ()->String\n  +return \"hi\"\n+end").expect("parse failed");
    for op in &ops {
        crate::validator::apply_and_validate(&mut program, op).unwrap();
    }
    program.rebuild_function_index();

    let shared = make_shared_program_mut(&program);

    // Mutate via the lock
    {
        let mut p = shared.write().unwrap();
        let new_ops = crate::parser::parse("+fn goodbye ()->String\n  +return \"bye\"\n+end")
            .expect("parse failed");
        for op in &new_ops {
            crate::validator::apply_and_validate(&mut p, op).unwrap();
        }
        p.rebuild_function_index();
    }

    // Read back should see the new function
    let readback = read_back_program_mutations(&shared);
    assert!(readback.is_some());
    let p = readback.unwrap();
    assert!(
        p.get_function("hello").is_some(),
        "original function should still exist"
    );
    assert!(
        p.get_function("goodbye").is_some(),
        "new function should exist after mutation"
    );
}

#[test]
fn read_back_returns_original_when_no_mutation() {
    let program = crate::ast::Program::default();
    let shared = make_shared_program_mut(&program);

    // No mutations performed
    let readback = read_back_program_mutations(&shared);
    assert!(
        readback.is_some(),
        "read_back should succeed even without mutations"
    );
    assert!(
        readback.unwrap().functions.is_empty(),
        "should return empty program"
    );
}

#[test]
fn shared_program_mut_thread_local_roundtrip() {
    let mut program = crate::ast::Program::default();
    let ops =
        crate::parser::parse("+fn test_fn ()->Int\n  +return 42\n+end").expect("parse failed");
    for op in &ops {
        crate::validator::apply_and_validate(&mut program, op).unwrap();
    }
    program.rebuild_function_index();

    let shared = make_shared_program_mut(&program);
    set_shared_program_mut(Some(shared.clone()));

    // get_shared_program_mut should return the same Arc
    let retrieved = get_shared_program_mut();
    assert!(
        retrieved.is_some(),
        "should retrieve program_mut from thread-local"
    );

    // Mutate via the retrieved handle
    {
        let lock = retrieved.unwrap();
        let mut p = lock.write().unwrap();
        p.functions.clear();
        p.rebuild_function_index();
    }

    // read_back from the original shared handle should see the mutation
    let readback = read_back_program_mutations(&shared);
    assert!(readback.is_some());
    assert!(
        readback.unwrap().functions.is_empty(),
        "mutations should propagate through shared Arc"
    );

    // Clean up
    set_shared_program_mut(None);
}

#[test]
fn move_symbols_in_mutation_builtin_list() {
    // Verify that the is_mutation_builtin check in eval.rs includes move_symbols.
    // This is a structural test — we check the eval path by calling through
    // the actual code path (via the +await dispatch).
    //
    // We build a program with a function that does +await move_symbols(...),
    // set only the read-only snapshot (no set_shared_program_mut), and verify
    // that the eval.rs fallback creates the mutable wrapper automatically.
    let source = "+fn helper ()->String\n  +return \"hi\"\n+end";
    let ops = crate::parser::parse(source).expect("parse failed");
    let mut program = crate::ast::Program::default();
    for op in &ops {
        crate::validator::apply_and_validate(&mut program, op).unwrap();
    }
    program.rebuild_function_index();

    // Set only read-only snapshot — mutation builtins need the fallback
    set_shared_program(Some(std::sync::Arc::new(program.clone())));
    set_shared_program_mut(None);

    // Build a function that calls +await move_symbols("helper", "Utils")
    let fn_source = "+fn do_move ()->String [io,async]\n  +await result:String = move_symbols(\"helper\", \"Utils\")\n  +return result\n+end";
    let fn_ops = crate::parser::parse(fn_source).expect("parse fn failed");
    for op in &fn_ops {
        crate::validator::apply_and_validate(&mut program, op).unwrap();
    }
    program.rebuild_function_index();

    // Create a mock handle — move_symbols doesn't go through IO dispatch,
    // it's handled directly in execute_await, so a mock handle is fine
    let handle = crate::coroutine::CoroutineHandle::new_mock(vec![]);
    let mut env = Env::new();
    env.set("__coroutine_handle", Value::CoroutineHandle(handle));

    // Set the program again with the new function
    set_shared_program(Some(std::sync::Arc::new(program.clone())));
    // The is_mutation_builtin check in eval.rs should create the mutable wrapper
    // for move_symbols (it wouldn't before this fix)
    let result = eval_function_body_pub(
        &program,
        &program.get_function("do_move").unwrap().body,
        &mut env,
    );

    // Should succeed — if move_symbols wasn't in the is_mutation_builtin list,
    // it would fail with "program not available (no async context)"
    assert!(
        result.is_ok(),
        "move_symbols should work via eval.rs mutation builtin fallback: {:?}",
        result.err()
    );
    let val = result.unwrap();
    let val_str = format!("{val}");
    assert!(
        val_str.contains("moved") || val_str.contains("Utils"),
        "should confirm move: {val_str}"
    );

    // Clean up
    set_shared_program(None);
    set_shared_program_mut(None);
}

// ── String interning tests ────────────────────────────────────────

#[test]
fn test_env_interned_set_get() {
    // Basic: set a variable and get it back via the interned Env
    let mut env = Env::new();
    env.set("x", Value::Int(42));
    let val = env.get("x").unwrap();
    assert!(matches!(val, Value::Int(42)));
}

#[test]
fn test_env_interned_scope_shadowing() {
    // Inner scope shadows outer scope; after pop, outer is visible again
    let mut env = Env::new();
    env.set("x", Value::Int(1));
    env.push_scope();
    env.set("x", Value::Int(2));
    assert!(matches!(env.get("x").unwrap(), Value::Int(2)));
    env.pop_scope();
    assert!(matches!(env.get("x").unwrap(), Value::Int(1)));
}

#[test]
fn test_env_interned_undefined_variable() {
    // Looking up a variable that doesn't exist should return an error
    let mut env = Env::new();
    let result = env.get("nonexistent");
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("undefined variable"),
        "expected undefined variable error, got: {msg}"
    );
}

#[test]
fn test_env_interned_set_existing() {
    // set_existing should update the variable in the outer scope
    let mut env = Env::new();
    env.set("counter", Value::Int(0));
    env.push_scope();
    // set_existing walks scopes to find "counter" in the outer scope
    env.set_existing("counter", Value::Int(10));
    // Inner scope doesn't have "counter", so get() finds outer scope's updated value
    assert!(matches!(env.get("counter").unwrap(), Value::Int(10)));
    env.pop_scope();
    // After pop, the outer scope should have the updated value
    assert!(matches!(env.get("counter").unwrap(), Value::Int(10)));
}

#[test]
fn test_env_interned_get_raw() {
    // get_raw returns None for missing variables instead of an error
    let mut env = Env::new();
    assert!(env.get_raw("missing").is_none());
    env.set("present", Value::Bool(true));
    assert!(env.get_raw("present").is_some());
}

#[test]
fn test_env_interned_snapshot_bindings() {
    // snapshot_bindings should return name-value pairs, excluding __ prefixed
    let mut env = Env::new();
    env.set("x", Value::Int(1));
    env.set("__internal", Value::Int(999));
    env.set("y", Value::string("hello"));
    let bindings = env.snapshot_bindings();
    assert_eq!(bindings.len(), 2, "should exclude __internal");
    // Bindings are sorted by name
    assert_eq!(bindings[0].0, "x");
    assert_eq!(bindings[1].0, "y");
}

#[test]
fn test_env_interned_multiple_variables() {
    // Verify multiple variables with different types work correctly
    let mut env = Env::new();
    env.set("a", Value::Int(1));
    env.set("b", Value::Float(2.5));
    env.set("c", Value::Bool(true));
    env.set("d", Value::string("hello"));
    env.set("e", Value::None);

    assert!(matches!(env.get("a").unwrap(), Value::Int(1)));
    assert!(matches!(env.get("b").unwrap(), Value::Float(f) if (f - 2.5).abs() < f64::EPSILON));
    assert!(matches!(env.get("c").unwrap(), Value::Bool(true)));
    assert!(matches!(env.get("d").unwrap(), Value::String(s) if s.as_str() == "hello"));
    assert!(matches!(env.get("e").unwrap(), Value::None));
}

#[test]
fn test_interned_eval_function_with_variables() {
    // End-to-end: evaluate a function that uses local variables
    let source = "\
+fn compute (x:Int, y:Int)->Int
  +let a:Int = x + 1
  +let b:Int = y * 2
  +let result:Int = a + b
  +return result

+test compute
  +with x=3 y=4 -> expect 12
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    assert_eq!(cases.len(), 1);

    let (fn_name, case) = &cases[0];
    let result = eval_test_case(&program, fn_name, case);
    assert!(result.is_ok(), "interned eval should pass: {:?}", result);
}

#[test]
fn test_interned_eval_nested_scopes() {
    // Test that nested scopes (if/each/while blocks) work with interning
    let source = "\
+fn nested (x:Int)->Int
  +let result:Int = 0
  +if x > 0
    +let inner:Int = x + 10
    +set result = inner
  +end
  +return result

+test nested
  +with x=5 -> expect 15
  +with x=0 -> expect 0
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    assert_eq!(cases.len(), 2);

    for (fn_name, case) in &cases {
        let result = eval_test_case(&program, fn_name, case);
        assert!(
            result.is_ok(),
            "nested scope test should pass: {:?}",
            result
        );
    }
}

#[test]
fn test_intern_name_consistency() {
    // Verify that the intern_name helper returns consistent ids
    let mut env = Env::new();
    let id1 = env.intern_name("test_var");
    let id2 = env.intern_name("test_var");
    let id3 = env.intern_name("other_var");
    assert_eq!(id1, id2, "same string should get same id");
    assert_ne!(id1, id3, "different strings should get different ids");
}

#[test]
fn test_resolve_name_roundtrip() {
    // Verify that intern → resolve roundtrips correctly
    let env = Env::new();
    let id = env.intern_name("roundtrip_test");
    let resolved = env.resolve_name(id);
    assert_eq!(resolved, "roundtrip_test");
}

// ═════════════════════════════════════════════════════════════════════
// Program interner + Env::new_with_interner tests
// ═════════════════════════════════════════════════════════════════════

#[test]
fn test_program_intern_all_names() {
    // Verify that rebuild_function_index populates the program interner
    let source = "\
+fn add (a:Int, b:Int)->Int
  +let sum:Int = a + b
  +return sum
";
    let mut program = build_program(source);
    // rebuild_function_index is called by build_program, so the interner
    // should already contain all names.
    assert!(
        program.interner.get("add").is_some(),
        "function name should be interned"
    );
    assert!(
        program.interner.get("a").is_some(),
        "param 'a' should be interned"
    );
    assert!(
        program.interner.get("b").is_some(),
        "param 'b' should be interned"
    );
    assert!(
        program.interner.get("sum").is_some(),
        "local var 'sum' should be interned"
    );
    // Well-known names should also be interned
    assert!(program.interner.get("__coroutine_handle").is_some());
    assert!(program.interner.get("true").is_some());
    assert!(program.interner.get("false").is_some());
}

#[test]
fn test_env_new_with_interner_seeded() {
    // Env created with new_with_interner should have the same interned ids
    // as the program's interner
    let source = "\
+fn greet (name:String)->String
  +return name
";
    let program = build_program(source);
    let env = Env::new_with_shared_interner(&program.shared_interner);

    // The env's interner should know about the program's names
    let id_from_program = program.interner.get("name").unwrap();
    let id_from_env = env.intern_name("name");
    assert_eq!(
        id_from_program, id_from_env,
        "interned ids should match between program and env"
    );
}

#[test]
fn test_env_set_id_get_id() {
    // Verify that set_id and get_id work correctly as fast-path methods
    let mut env = Env::new();
    let id = env.intern_name("fast_var");
    env.set_id(id, Value::Int(99));
    let val = env.get_id(id);
    assert!(val.is_some(), "get_id should find the value");
    assert!(matches!(val.unwrap(), Value::Int(99)));
}

#[test]
fn test_env_set_id_scope_isolation() {
    // Values set via set_id in inner scope should not be visible after pop
    let mut env = Env::new();
    let id = env.intern_name("scoped_var");
    env.push_scope();
    env.set_id(id, Value::Int(42));
    assert!(env.get_id(id).is_some());
    env.pop_scope();
    assert!(
        env.get_id(id).is_none(),
        "value should not be visible after scope pop"
    );
}

#[test]
fn test_env_get_id_not_found() {
    // get_id should return None for unknown ids
    let env = Env::new();
    assert!(env.get_id(99999).is_none());
}

#[test]
fn test_program_interner_with_modules() {
    // Verify that module names, module function names, and shared vars are interned
    let source = "\
+module Math

+fn square (n:Int)->Int
  +return n * n
";
    let program = build_program(source);
    assert!(
        program.interner.get("Math").is_some(),
        "module name should be interned"
    );
    assert!(
        program.interner.get("square").is_some(),
        "module function name should be interned"
    );
    assert!(
        program.interner.get("n").is_some(),
        "param 'n' should be interned"
    );
}

#[test]
fn test_interner_eval_function_with_program_interner() {
    // End-to-end: evaluate a function using Env seeded from program's interner
    let source = "\
+fn double (x:Int)->Int
  +let result:Int = x * 2
  +return result

+test double
  +with x=5 -> expect 10
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    assert_eq!(cases.len(), 1);
    let (fn_name, case) = &cases[0];
    let result = eval_test_case(&program, fn_name, case);
    assert!(
        result.is_ok(),
        "eval with program interner should work: {:?}",
        result
    );
}

#[test]
fn test_interner_consistency_across_envs() {
    // Two Envs seeded from the same program interner should produce
    // the same interned ids, allowing values to be portable between them
    let source = "+fn identity (v:Int)->Int\n  +return v\n";
    let program = build_program(source);
    let env1 = Env::new_with_shared_interner(&program.shared_interner);
    let env2 = Env::new_with_shared_interner(&program.shared_interner);
    let id1 = env1.intern_name("v");
    let id2 = env2.intern_name("v");
    assert_eq!(id1, id2, "same interner seed should produce same ids");
}

// ═════════════════════════════════════════════════════════════════════
// SharedInterner + SmallVec scope chain optimization tests
// ═════════════════════════════════════════════════════════════════════

#[test]
fn env_shared_interner_o1_clone() {
    // Creating an Env from a SharedInterner should be O(1) — the Arc is
    // shared, not cloned. Verify that lookup still works correctly.
    let mut base = StringInterner::new();
    base.intern("x");
    base.intern("y");
    base.intern("z");
    let shared = base.shared();

    let mut env = Env::new_with_shared_interner(&shared);
    env.set("x", Value::Int(1));
    env.set("y", Value::Int(2));

    assert_eq!(format!("{}", env.get("x").unwrap()), "1");
    assert_eq!(format!("{}", env.get("y").unwrap()), "2");
    // z is interned but not set as a variable — get should return Err
    assert!(env.get("z").is_err(), "z was interned but not set");
    assert!(env.get("nonexistent").is_err());
}

#[test]
fn env_shared_interner_cow_on_new_name() {
    // When a truly new name is encountered, the SharedInterner should
    // copy-on-write: existing Envs sharing the same Arc are unaffected.
    let mut base = StringInterner::new();
    base.intern("known");
    let shared = base.shared();

    let mut env1 = Env::new_with_shared_interner(&shared);
    let mut env2 = Env::new_with_shared_interner(&shared);

    // Intern a new name in env1 (triggers copy-on-write)
    env1.set("brand_new", Value::Int(99));
    assert_eq!(format!("{}", env1.get("brand_new").unwrap()), "99");

    // env2 should NOT see "brand_new" because it has its own interner copy
    // (but env2 can still intern it independently)
    env2.set("brand_new", Value::Int(77));
    assert_eq!(format!("{}", env2.get("brand_new").unwrap()), "77");
}

#[test]
fn env_smallvec_scope_chain_basic() {
    // The scope chain uses SmallVec<[_; 4]> — verify push/pop works
    // correctly through nested scopes without heap allocation for
    // the typical case (≤4 scopes deep).
    let mut env = Env::new();
    env.set("outer", Value::Int(1));

    env.push_scope();
    env.set("inner1", Value::Int(2));
    assert_eq!(format!("{}", env.get("outer").unwrap()), "1");
    assert_eq!(format!("{}", env.get("inner1").unwrap()), "2");

    env.push_scope();
    env.set("inner2", Value::Int(3));
    assert_eq!(format!("{}", env.get("outer").unwrap()), "1");
    assert_eq!(format!("{}", env.get("inner2").unwrap()), "3");

    env.push_scope();
    env.set("inner3", Value::Int(4));
    // Still within SmallVec inline capacity (4 scopes: root + 3 nested)
    assert_eq!(format!("{}", env.get("inner3").unwrap()), "4");

    env.pop_scope();
    assert!(env.get("inner3").is_err()); // inner3 is gone

    env.pop_scope();
    assert!(env.get("inner2").is_err()); // inner2 is gone

    env.pop_scope();
    assert!(env.get("inner1").is_err()); // inner1 is gone
    assert_eq!(format!("{}", env.get("outer").unwrap()), "1"); // outer still there
}

#[test]
fn env_smallvec_spills_to_heap_gracefully() {
    // When scope depth exceeds SmallVec inline capacity (4), it should
    // spill to heap and continue working correctly.
    let mut env = Env::new();
    env.set("root", Value::Int(0));

    // Push 10 scopes (well beyond inline capacity of 4)
    for i in 1..=10 {
        env.push_scope();
        env.set(&format!("level_{i}"), Value::Int(i as i64));
    }

    // All variables should be accessible
    assert_eq!(format!("{}", env.get("root").unwrap()), "0");
    for i in 1..=10 {
        assert_eq!(
            format!("{}", env.get(&format!("level_{i}")).unwrap()),
            format!("{i}")
        );
    }

    // Pop all nested scopes
    for _ in 1..=10 {
        env.pop_scope();
    }
    assert_eq!(format!("{}", env.get("root").unwrap()), "0");
    assert!(env.get("level_1").is_err());
}

#[test]
fn env_scope_shadowing_with_smallvec() {
    // Variable shadowing should work correctly with SmallVec scopes
    let mut env = Env::new();
    env.set("x", Value::Int(1));

    env.push_scope();
    env.set("x", Value::Int(2)); // shadows outer x
    assert_eq!(format!("{}", env.get("x").unwrap()), "2");

    env.pop_scope();
    assert_eq!(format!("{}", env.get("x").unwrap()), "1"); // outer x restored
}

#[test]
fn program_shared_interner_rebuilt_on_mutation() {
    // After rebuilding the function index, the shared_interner should
    // contain all AST names and be usable for Env creation.
    let source = "\
+fn add (a:Int, b:Int)->Int
  +let sum:Int = a + b
  +return sum
";
    let program = build_program(source);

    // shared_interner should be populated
    assert!(program.shared_interner.get("add").is_some());
    assert!(program.shared_interner.get("a").is_some());
    assert!(program.shared_interner.get("b").is_some());
    assert!(program.shared_interner.get("sum").is_some());

    // Creating an Env from shared_interner should work
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    env.set("a", Value::Int(1));
    assert_eq!(format!("{}", env.get("a").unwrap()), "1");
}

#[test]
fn shared_interner_ids_match_base_interner() {
    // IDs from SharedInterner should match the original StringInterner
    let source = "\
+fn greet (name:String)->String
  +return concat(\"hello \", name)
";
    let program = build_program(source);

    let base_id = program.interner.get("name").unwrap();
    let shared_id = program.shared_interner.get("name").unwrap();
    assert_eq!(
        base_id, shared_id,
        "IDs should be identical between base and shared interner"
    );
}

// ═════════════════════════════════════════════════════════════════════
// Performance optimization tests
// ═════════════════════════════════════════════════════════════════════

#[test]
fn test_concat_prealloc_strings() {
    // concat with multiple string arguments should produce correct result
    let p = ast::Program::default();
    assert_eq!(
        eval_expr_str(&p, r#"concat("hello", " ", "world")"#),
        r#""hello world""#
    );
}

#[test]
fn test_concat_prealloc_mixed_types() {
    // concat with mixed types (string, int, bool) should format correctly
    let p = ast::Program::default();
    let result = eval_expr_str(&p, r#"concat("count: ", 42)"#);
    assert_eq!(result, r#""count: 42""#);
}

#[test]
fn test_concat_empty_args() {
    // concat with no arguments should return empty string
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, r#"concat()"#), r#""""#);
}

#[test]
fn test_concat_single_arg() {
    // concat with a single argument
    let p = ast::Program::default();
    assert_eq!(eval_expr_str(&p, r#"concat("solo")"#), r#""solo""#);
}

#[test]
fn test_push_returns_extended_list() {
    // push should return a new list with the item appended
    let source = "\
+fn test_push ()->Int
  +let xs:List<Int> = list(1, 2, 3)
  +let ys:List<Int> = push(xs, 4)
  +return len(ys)

+test test_push
  +with -> expect 4
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    assert_eq!(cases.len(), 1);
    let (fn_name, case) = &cases[0];
    let result = eval_test_case(&program, fn_name, case);
    assert!(result.is_ok(), "push should work: {:?}", result);
}

#[test]
fn test_push_error_wrong_args() {
    // push with wrong number of args should error
    let p = ast::Program::default();
    let mut env = Env::new();
    let result = eval_builtin_or_user(&p, "push", vec![Value::list(vec![])], &mut env);
    assert!(result.is_err(), "push with 1 arg should error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("expects 2 arguments"),
        "error should mention arg count: {msg}"
    );
}

#[test]
fn test_push_error_not_list() {
    // push on a non-list should error
    let p = ast::Program::default();
    let mut env = Env::new();
    let result = eval_builtin_or_user(&p, "push", vec![Value::Int(42), Value::Int(1)], &mut env);
    assert!(result.is_err(), "push on non-list should error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("push expects"),
        "error should mention push: {msg}"
    );
}

#[test]
fn test_push_preserves_original_list_semantics() {
    // In Adapsis, push returns a new list; verify the function semantics
    // work correctly in a loop accumulation pattern
    let source = "\
+fn accumulate (n:Int)->Int
  +let result:List<Int> = list()
  +let i:Int = 0
  +while i < n
    +set result = push(result, i)
    +set i = i + 1
  +end
  +return len(result)

+test accumulate
  +with n=4 -> expect 4
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    let result = eval_test_case(&program, fn_name, case);
    assert!(
        result.is_ok(),
        "accumulate with push should work: {:?}",
        result
    );
}

#[test]
fn test_module_function_lookup_with_index() {
    // Module-qualified function lookup should work after rebuild_function_index
    let source = "\
+module Math
+fn add (a:Int, b:Int)->Int
  +return a + b

+test Math.add
  +with a=3 b=4 -> expect 7
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    let result = eval_test_case(&program, fn_name, case);
    assert!(
        result.is_ok(),
        "module function lookup should work: {:?}",
        result
    );
}

#[test]
fn test_module_function_lookup_not_found() {
    // Looking up a non-existent module function should return None
    let mut program = ast::Program::default();
    program.rebuild_function_index();
    assert!(program.get_function("NonExistent.func").is_none());
}

// ═════════════════════════════════════════════════════════════════════
// Name interning optimization tests
// ═════════════════════════════════════════════════════════════════════

#[test]
fn env_new_with_interner_seeds_names() {
    // When Env is created with a pre-populated interner, variable lookups
    // for pre-interned names should use cache hits (no new allocations).
    let mut interner = StringInterner::new();
    let id_x = interner.intern("x");
    let id_y = interner.intern("y");

    let mut env = Env::new_with_interner(&interner);
    env.set("x", Value::Int(42));
    env.set("y", Value::Int(99));

    // Look up by name — should hit the pre-interned cache
    assert!(matches!(env.get("x"), Ok(Value::Int(42))));
    assert!(matches!(env.get("y"), Ok(Value::Int(99))));

    // Look up by pre-interned id — fast path
    assert!(matches!(env.get_id(id_x), Some(Value::Int(42))));
    assert!(matches!(env.get_id(id_y), Some(Value::Int(99))));
}

#[test]
fn env_new_with_interner_handles_unknown_names() {
    // Names not in the pre-seeded interner should still work (interned on demand)
    let interner = StringInterner::new(); // empty interner
    let mut env = Env::new_with_interner(&interner);
    env.set("dynamic_var", Value::string("hello"));

    assert!(matches!(env.get("dynamic_var"), Ok(Value::String(s)) if s.as_str() == "hello"));
}

#[test]
fn env_undefined_variable_returns_error() {
    let mut env = Env::new();
    let result = env.get("nonexistent");
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("undefined variable"),
        "error should mention 'undefined variable', got: {err_msg}"
    );
}

#[test]
fn env_set_id_and_get_id_roundtrip() {
    let mut interner = StringInterner::new();
    let id = interner.intern("counter");
    let mut env = Env::new_with_interner(&interner);

    env.set_id(id, Value::Int(0));
    assert!(matches!(env.get_id(id), Some(Value::Int(0))));

    // Update via set_id
    env.set_id(id, Value::Int(1));
    assert!(matches!(env.get_id(id), Some(Value::Int(1))));
}

#[test]
fn union_variant_hashset_used_in_eval() {
    // Verify that is_union_variant uses the HashSet-based lookup
    let source = "\
+type Color = Red | Green | Blue
+fn get_color () -> Color
  +return Red
";
    let program = build_program(source);
    // The HashSet should contain the variants after rebuild
    assert!(program.is_union_variant("Red"));
    assert!(program.is_union_variant("Green"));
    assert!(program.is_union_variant("Blue"));
    assert!(!program.is_union_variant("Yellow"));
}

#[test]
fn user_function_call_uses_interned_env() {
    // When a user function is called via eval_builtin_or_user, the child Env
    // should be seeded with the program's interner for fast param lookups.
    let source = "\
+fn double (n:Int) -> Int
  +return n + n

+test double
  +with n=5 -> expect 10
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    let result = eval_test_case(&program, fn_name, case);
    assert!(
        result.is_ok(),
        "user function call with interned env should work: {:?}",
        result
    );
}

// ═════════════════════════════════════════════════════════════════════
// Arc-based Value optimisation tests
// ═════════════════════════════════════════════════════════════════════

#[test]
fn value_string_clone_is_cheap_arc_bump() {
    // Cloning a Value::String should share the same Arc allocation
    let v1 = Value::string("hello world");
    let v2 = v1.clone();
    // Both should point to the same underlying str (same Arc)
    match (&v1, &v2) {
        (Value::String(a), Value::String(b)) => {
            assert!(
                Arc::ptr_eq(a, b),
                "clone should share the Arc, not allocate a new string"
            );
        }
        _ => panic!("expected String variants"),
    }
}

#[test]
fn value_err_clone_preserves_value() {
    let v1 = Value::Err("some error".to_string());
    let v2 = v1.clone();
    match (&v1, &v2) {
        (Value::Err(a), Value::Err(b)) => {
            assert_eq!(a, b, "Err clone should preserve the error message");
        }
        _ => panic!("expected Err variants"),
    }
}

#[test]
fn value_list_clone_shares_arc() {
    let items = vec![Value::Int(1), Value::Int(2), Value::Int(3)];
    let v1 = Value::list(items);
    let v2 = v1.clone();
    match (&v1, &v2) {
        (Value::List(a), Value::List(b)) => {
            assert!(
                Arc::ptr_eq(a, b),
                "List clone should share the Arc<Vec<Value>>"
            );
        }
        _ => panic!("expected List variants"),
    }
}

#[test]
fn value_struct_clone_shares_field_arc() {
    let mut fields = HashMap::new();
    fields.insert("x".to_string(), Value::Int(10));
    fields.insert("y".to_string(), Value::Int(20));
    let v1 = Value::strct("Point", fields);
    let v2 = v1.clone();
    match (&v1, &v2) {
        (Value::Struct(n1, f1), Value::Struct(n2, f2)) => {
            assert_eq!(n1, n2, "Struct name should be preserved");
            assert!(Arc::ptr_eq(f1, f2), "Struct fields Arc should be shared");
        }
        _ => panic!("expected Struct variants"),
    }
}

#[test]
fn value_union_variant_clone_preserves_value() {
    let v1 = Value::Union {
        variant: intern::intern_display("Some"),
        payload: vec![Value::Int(42)],
    };
    let v2 = v1.clone();
    match (&v1, &v2) {
        (Value::Union { variant: a, .. }, Value::Union { variant: b, .. }) => {
            assert_eq!(a, b, "Union variant name should be preserved");
        }
        _ => panic!("expected Union variants"),
    }
}

#[test]
fn value_string_display_correct() {
    let v = Value::string("hello");
    assert_eq!(format!("{v}"), r#""hello""#);
}

#[test]
fn value_err_display_correct() {
    let v = Value::Err("fail".to_string());
    assert_eq!(format!("{v}"), "Err(fail)");
}

#[test]
fn value_struct_field_lookup_with_arc_keys() {
    // Ensure struct field access works with Arc<str> keys
    let source = "\
+type Point = x:Int, y:Int

+fn get_x (p:Point) -> Int
  +return p.x

+test get_x
  +with x=10 y=20 -> expect 10
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    let result = eval_test_case(&program, fn_name, case);
    assert!(
        result.is_ok(),
        "struct field access with Arc keys should work: {:?}",
        result
    );
}

#[test]
fn value_list_operations_with_arc_wrapper() {
    // push, get, len should all work through Arc<Vec<Value>>
    let source = "\
+fn list_ops ()->Int
  +let items:List<Int> = list(1, 2, 3)
  +let items2:List<Int> = push(items, 4)
  +return len(items2)
+end

+test list_ops
  +with -> expect 4
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    let result = eval_test_case(&program, fn_name, case);
    assert!(
        result.is_ok(),
        "list operations through Arc<Vec> should work: {:?}",
        result
    );
}

#[test]
fn value_string_builtin_ops_through_arc() {
    // String builtins (concat, len, split, trim, etc.) should work with Arc<str>
    let source = "\
+fn string_ops (s:String) -> String
  +let trimmed:String = trim(s)
  +let upper:String = concat(trimmed, \"!\")
  +return upper

+test string_ops
  +with s=\"  hello  \" -> expect \"hello!\"
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    let result = eval_test_case(&program, fn_name, case);
    assert!(
        result.is_ok(),
        "string builtins through Arc<str> should work: {:?}",
        result
    );
}

#[test]
fn value_matches_equality_with_arc() {
    // Value::matches() should work correctly with Arc<str> internals
    let a = Value::string("hello");
    let b = Value::string("hello");
    let c = Value::string("world");
    assert!(a.matches(&b), "same content Arc<str> should match");
    assert!(!a.matches(&c), "different content should not match");

    // Err matching
    let ea = Value::Err("fail".to_string());
    let eb = Value::Err("fail".to_string());
    let ec = Value::Err("other".to_string());
    assert!(ea.matches(&eb));
    assert!(!ea.matches(&ec));

    // Struct matching
    let mut f1 = HashMap::new();
    f1.insert("x".to_string(), Value::Int(1));
    let mut f2 = HashMap::new();
    f2.insert("x".to_string(), Value::Int(1));
    let s1 = Value::strct("P", f1);
    let s2 = Value::strct("P", f2);
    assert!(s1.matches(&s2));

    // List matching
    let l1 = Value::list(vec![Value::Int(1), Value::Int(2)]);
    let l2 = Value::list(vec![Value::Int(1), Value::Int(2)]);
    let l3 = Value::list(vec![Value::Int(1), Value::Int(3)]);
    assert!(l1.matches(&l2));
    assert!(!l1.matches(&l3));
}

#[test]
fn value_is_truthy_with_arc_string() {
    assert!(Value::string("x").is_truthy());
    assert!(!Value::string("").is_truthy());
}

#[test]
fn value_match_pattern_with_arc_union() {
    // +match on union variants should work correctly with Arc<str> variant names
    let source = "\
+type Color = Red | Green | Blue

+fn color_name (c:Color)->String
  +match c
  +case Red
    +return \"red\"
  +case Green
    +return \"green\"
  +case Blue
    +return \"blue\"
  +end
+end

+test color_name
  +with c=Red -> expect \"red\"
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    let result = eval_test_case(&program, fn_name, case);
    assert!(
        result.is_ok(),
        "union match with Arc variant names should work: {:?}",
        result
    );
}

#[test]
fn value_each_loop_with_arc_list() {
    // +each over a List should work with Arc<Vec<Value>> wrapper
    let source = "\
+fn sum_list (items:List<Int>)->Int
  +let total:Int = 0
  +each items item:Int
    +set total = total + item
  +end
  +return total
+end

+test sum_list
  +with items=list(1, 2, 3) -> expect 6
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    let result = eval_test_case(&program, fn_name, case);
    assert!(
        result.is_ok(),
        "+each with Arc<Vec> should work: {:?}",
        result
    );
}

// ═════════════════════════════════════════════════════════════════════
// Function dispatch with interned fn_index / module_index
// ═════════════════════════════════════════════════════════════════════

#[test]
fn interned_fn_index_dispatch_top_level() {
    // End-to-end: calling a top-level user function should dispatch through
    // the interned fn_index, not string comparison.
    let source = "\
+fn triple (n:Int)->Int
  +return n * 3

+test triple
  +with n=7 -> expect 21
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    let result = eval_test_case(&program, fn_name, case);
    assert!(
        result.is_ok(),
        "top-level function dispatch via interned index: {:?}",
        result
    );
}

#[test]
fn interned_fn_index_dispatch_module_qualified() {
    // Module-qualified function call dispatches through interned module_index + fn_index.
    let source = "\
+module Calc

+fn square (n:Int)->Int
  +return n * n

+test Calc.square
  +with n=6 -> expect 36
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    let result = eval_test_case(&program, fn_name, case);
    assert!(
        result.is_ok(),
        "module function dispatch via interned index: {:?}",
        result
    );
}

#[test]
fn interned_fn_index_cross_function_call() {
    // Function A calls function B — both dispatched through interned indices.
    let source = "\
+fn helper (x:Int)->Int
  +return x + 1

+fn main_fn (x:Int)->Int
  +call y:Int = helper(x)
  +return y * 2

+test main_fn
  +with x=4 -> expect 10
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    let result = eval_test_case(&program, fn_name, case);
    assert!(
        result.is_ok(),
        "cross-function dispatch via interned index: {:?}",
        result
    );
}

#[test]
fn interned_union_variant_lookup_in_match() {
    // Union variant dispatch uses interned HashSet<InternedId> via is_union_variant.
    let source = "\
+type Shape = Circle(Int) | Square(Int)

+fn area (s:Shape)->Int
  +match s
  +case Circle(r)
    +return r * r * 3
  +case Square(side)
    +return side * side
  +end

+test area
  +with s=Circle(5) -> expect 75
  +with s=Square(4) -> expect 16
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    for (fn_name, case) in &cases {
        let result = eval_test_case(&program, fn_name, case);
        assert!(
            result.is_ok(),
            "union variant dispatch via interned set: {:?}",
            result
        );
    }
}

#[test]
fn interned_fn_dispatch_unknown_function_error() {
    // Calling a non-existent function should produce a clear error, not panic.
    let program = build_program("+fn noop ()->Int\n  +return 0\n");
    let mut env = Env::new_with_shared_interner(&program.shared_interner);
    let result = eval_builtin_or_user(&program, "nonexistent_fn", vec![], &mut env);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("undefined function"),
        "should say undefined: {msg}"
    );
}

// ═════════════════════════════════════════════════════════════════════
// Interned struct fields / union variant tests
// ═════════════════════════════════════════════════════════════════════

#[test]
fn interned_struct_field_keys_are_u32() {
    // Value::Struct should use InternedId (u32) keys, not String.
    let mut fields = HashMap::new();
    fields.insert("x".to_string(), Value::Int(10));
    fields.insert("y".to_string(), Value::Int(20));
    let val = Value::strct("Point", fields);
    match &val {
        Value::Struct(name_id, field_map) => {
            // Keys should be u32 (InternedId)
            assert_eq!(field_map.len(), 2);
            let x_id = intern::intern_display("x");
            let y_id = intern::intern_display("y");
            assert!(
                matches!(field_map.get(&x_id), Some(Value::Int(10))),
                "expected x=10"
            );
            assert!(
                matches!(field_map.get(&y_id), Some(Value::Int(20))),
                "expected y=20"
            );
            // Name should resolve back to "Point"
            assert_eq!(intern::resolve_display(*name_id), "Point");
        }
        _ => panic!("expected Struct"),
    }
}

#[test]
fn interned_struct_display_resolves_names() {
    // Value::Struct Display should render field names correctly via the display interner.
    let mut fields = HashMap::new();
    fields.insert("name".to_string(), Value::string("alice"));
    fields.insert("age".to_string(), Value::Int(30));
    let val = Value::strct("User", fields);
    let display = format!("{val}");
    assert!(
        display.contains("User{"),
        "display should show struct name: {display}"
    );
    assert!(
        display.contains("name:"),
        "display should show field 'name': {display}"
    );
    assert!(
        display.contains("age:"),
        "display should show field 'age': {display}"
    );
    assert!(
        display.contains("alice"),
        "display should show field value: {display}"
    );
    assert!(
        display.contains("30"),
        "display should show field value: {display}"
    );
}

#[test]
fn interned_union_variant_display() {
    // Union variant Display should resolve the interned variant name.
    let val = Value::Union {
        variant: intern::intern_display("Some"),
        payload: vec![Value::Int(42)],
    };
    let display = format!("{val}");
    assert_eq!(display, "Some(42)");
}

#[test]
fn interned_union_no_payload_display() {
    let val = Value::Union {
        variant: intern::intern_display("None"),
        payload: vec![],
    };
    let display = format!("{val}");
    assert_eq!(display, "None");
}

#[test]
fn interned_struct_get_field_by_string() {
    // Value::get_field should look up by string name via the display interner.
    let mut fields = HashMap::new();
    fields.insert("x".to_string(), Value::Int(42));
    let val = Value::strct("P", fields);
    assert!(
        matches!(val.get_field("x"), Some(Value::Int(42))),
        "should find field x"
    );
    assert!(val.get_field("y").is_none(), "should not find field y");
    assert!(
        val.get_field("nonexistent").is_none(),
        "should not find nonexistent field"
    );
}

#[test]
fn interned_struct_field_access_in_eval() {
    // Full eval roundtrip: struct init → field access using interned keys
    let source = "\
+type Config = host:String, port:Int

+fn get_port (c:Config) -> Int
  +return c.port

+test get_port
  +with host=\"localhost\" port=8080 -> expect 8080
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    let (fn_name, case) = &cases[0];
    let result = eval_test_case(&program, fn_name, case);
    assert!(result.is_ok(), "interned struct field access: {:?}", result);
    let result_str = result.unwrap();
    assert!(
        result_str.contains("8080"),
        "result should contain 8080: {result_str}"
    );
}

#[test]
fn interned_union_match_dispatch() {
    // Full eval roundtrip: union variant construction → match dispatch using interned IDs
    let source = "\
+type Result = Success(Int) | Failure(String)

+fn unwrap_result (r:Result) -> Int
  +match r
  +case Success(val)
    +return val
  +case Failure(msg)
    +return -1
  +end

+test unwrap_result
  +with r=Success(42) -> expect 42
  +with r=Failure(\"oops\") -> expect -1
";
    let program = build_program(source);
    let cases = extract_test_cases(source);
    for (fn_name, case) in &cases {
        let result = eval_test_case(&program, fn_name, case);
        assert!(
            result.is_ok(),
            "interned union match dispatch: {:?}",
            result
        );
    }
}

#[test]
fn interned_struct_empty_name_matches_any() {
    // Struct with empty name (anonymous) should match any named struct
    let mut f1 = HashMap::new();
    f1.insert("x".to_string(), Value::Int(1));
    let named = Value::strct("Point", f1);

    let mut f2: HashMap<InternedId, Value> = HashMap::new();
    f2.insert(intern::intern_display("x"), Value::Int(1));
    let anon = Value::strct_interned(intern::intern_display(""), f2);

    assert!(named.matches(&anon), "named struct should match anonymous");
    assert!(anon.matches(&named), "anonymous struct should match named");
}

#[test]
fn interned_struct_different_names_dont_match() {
    let mut f1 = HashMap::new();
    f1.insert("x".to_string(), Value::Int(1));
    let s1 = Value::strct("A", f1);

    let mut f2 = HashMap::new();
    f2.insert("x".to_string(), Value::Int(1));
    let s2 = Value::strct("B", f2);

    assert!(!s1.matches(&s2), "different named structs should not match");
}

#[test]
fn interned_union_variant_equality() {
    // Two unions with the same interned variant should be equal
    let v1 = Value::Union {
        variant: intern::intern_display("Ok"),
        payload: vec![Value::Int(1)],
    };
    let v2 = Value::Union {
        variant: intern::intern_display("Ok"),
        payload: vec![Value::Int(1)],
    };
    assert!(v1.matches(&v2));

    // Different variants should not match
    let v3 = Value::Union {
        variant: intern::intern_display("Err"),
        payload: vec![Value::Int(1)],
    };
    assert!(!v1.matches(&v3));
}

#[test]
fn interned_struct_strct_interned_roundtrip() {
    // strct_interned should produce identical values to strct
    let mut string_fields = HashMap::new();
    string_fields.insert("a".to_string(), Value::Int(1));
    string_fields.insert("b".to_string(), Value::Int(2));
    let via_strct = Value::strct("T", string_fields);

    let mut interned_fields: HashMap<InternedId, Value> = HashMap::new();
    interned_fields.insert(intern::intern_display("a"), Value::Int(1));
    interned_fields.insert(intern::intern_display("b"), Value::Int(2));
    let via_interned = Value::strct_interned(intern::intern_display("T"), interned_fields);

    assert!(
        via_strct.matches(&via_interned),
        "strct and strct_interned should produce matching values"
    );
}

#[test]
fn interned_display_interner_fallback_for_unknown_id() {
    // resolve_display should return a fallback for unknown IDs
    let result = intern::resolve_display(999_999);
    assert!(
        result.starts_with("<id:"),
        "unknown ID should get fallback: {result}"
    );
}

// ── Async multi-parameter tests (regression for display interner bug) ───

#[test]
fn test_async_multi_param_function_binds_params_via_runtime() {
    // Regression test: async functions with multiple parameters previously
    // failed with "undefined variable" because the display interner was
    // not installed on the worker thread spawned by
    // eval_test_case_with_runtime. bind_input_to_params uses
    // intern::resolve_display / intern::intern_display to match struct
    // field names to parameter names, which requires the thread-local
    // display interner to be populated.
    let source = "\
+fn fetch_issues (owner:String, repo:String)->String [io,async]
  +await resp:String = http_get(concat(\"https://api.github.com/repos/\", owner, \"/\", repo, \"/issues\"))
  +return resp
";
    let program = build_program(source);

    let test_source = "\
+test fetch_issues
  +with owner=\"torvalds\" repo=\"linux\" -> expect \"issues_json\"
";
    let cases = extract_test_cases(test_source);
    let (fn_name, case) = &cases[0];

    let mocks = vec![IoMock {
        operation: "http_get".to_string(),
        patterns: vec!["api.github.com".to_string()],
        response: "issues_json".to_string(),
    }];
    let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
    assert!(
        result.is_ok(),
        "async multi-param test should pass (display interner must be set on worker thread): {:?}",
        result
    );
}

#[tokio::test]
async fn test_async_multi_param_function_binds_params_via_spawn_blocking() {
    // Same regression test but through eval_test_case_async (the
    // spawn_blocking path). Ensures the display interner is installed
    // on the tokio blocking thread pool thread as well.
    let source = "\
+fn fetch_issues (owner:String, repo:String)->String [io,async]
  +await resp:String = http_get(concat(\"https://api.github.com/repos/\", owner, \"/\", repo, \"/issues\"))
  +return resp
";
    let program = build_program(source);

    let test_source = "\
+test fetch_issues
  +with owner=\"torvalds\" repo=\"linux\" -> expect \"issues_json\"
";
    let cases = extract_test_cases(test_source);
    let (fn_name, case) = &cases[0];

    let mocks = vec![IoMock {
        operation: "http_get".to_string(),
        patterns: vec!["api.github.com".to_string()],
        response: "issues_json".to_string(),
    }];

    let (tx, _rx) = tokio::sync::mpsc::channel(1);
    let result = eval_test_case_async(&program, fn_name, case, &mocks, tx, &[]).await;
    assert!(
        result.is_ok(),
        "async multi-param test via eval_test_case_async should pass: {:?}",
        result
    );
}

#[test]
fn test_async_multi_param_wrong_expected_fails() {
    // Error case: async multi-param function test with wrong expected
    // value should fail cleanly (not with "undefined variable").
    let source = "\
+fn fetch_issues (owner:String, repo:String)->String [io,async]
  +await resp:String = http_get(concat(\"https://api.github.com/repos/\", owner, \"/\", repo, \"/issues\"))
  +return resp
";
    let program = build_program(source);

    let test_source = "\
+test fetch_issues
  +with owner=\"torvalds\" repo=\"linux\" -> expect \"wrong_value\"
";
    let cases = extract_test_cases(test_source);
    let (fn_name, case) = &cases[0];

    let mocks = vec![IoMock {
        operation: "http_get".to_string(),
        patterns: vec!["api.github.com".to_string()],
        response: "actual_issues".to_string(),
    }];
    let result = eval_test_case_with_mocks(&program, fn_name, case, &mocks, &[]);
    assert!(
        result.is_err(),
        "async test with wrong expected should fail"
    );
    let err = result.unwrap_err().to_string();
    // The error should be a value mismatch, NOT "undefined variable"
    assert!(
        !err.contains("undefined variable"),
        "error should NOT be 'undefined variable' (params should bind correctly): {err}"
    );
}

#[tokio::test]
async fn test_async_session_multi_param_function() {
    // End-to-end test through the session flow: define an async function
    // with multiple parameters, add mocks, and run tests.
    let mut session = crate::session::Session::new();

    let define_source = "\
+fn fetch_issues (owner:String, repo:String)->String [io,async]
  +await resp:String = http_get(concat(\"https://api.github.com/repos/\", owner, \"/\", repo, \"/issues\"))
  +return resp
";
    let results = session.apply_async(define_source, None).await;
    assert!(results.is_ok(), "define should succeed: {:?}", results);

    let mock_source = "!mock http_get \"api.github.com\" -> \"session_issues\"";
    let results = session.apply_async(mock_source, None).await;
    assert!(results.is_ok(), "mock should succeed: {:?}", results);

    let test_source = "\
+test fetch_issues
  +with owner=\"torvalds\" repo=\"linux\" -> expect \"session_issues\"
";
    let results = session.apply_async(test_source, None).await.unwrap();
    assert_eq!(results.len(), 1);
    assert!(
        results[0].1,
        "async multi-param test via session should pass: {:?}",
        results[0]
    );
}

// ── Value accessor / constructor tests ──────────────────────────

#[test]
fn value_string_uses_arc_for_cheap_clone() {
    let v = Value::string("hello");
    let v2 = v.clone();
    // Both clones share the same Arc allocation
    if let (Value::String(a), Value::String(b)) = (&v, &v2) {
        assert!(Arc::ptr_eq(a, b), "clone should share Arc pointer");
    } else {
        panic!("expected String variants");
    }
}

#[test]
fn value_list_uses_arc_for_cheap_clone() {
    let v = Value::list(vec![Value::Int(1), Value::Int(2)]);
    let v2 = v.clone();
    if let (Value::List(a), Value::List(b)) = (&v, &v2) {
        assert!(Arc::ptr_eq(a, b), "clone should share Arc pointer");
    } else {
        panic!("expected List variants");
    }
}

#[test]
fn value_as_str_returns_inner_slice() {
    let v = Value::string("café");
    assert_eq!(v.as_str(), Some("café"));
    assert_eq!(Value::Int(42).as_str(), None);
    assert_eq!(Value::None.as_str(), None);
}

#[test]
fn value_as_list_returns_inner_slice() {
    let v = Value::list(vec![Value::Int(1), Value::string("x")]);
    let slice = v.as_list().unwrap();
    assert_eq!(slice.len(), 2);
    assert!(matches!(&slice[0], Value::Int(1)));
    assert!(Value::Int(0).as_list().is_none());
}

#[test]
fn value_as_list_mut_cow_semantics() {
    // When there's only one Arc reference, as_list_mut should not clone
    let mut v = Value::list(vec![Value::Int(10)]);
    {
        let inner = v.as_list_mut().unwrap();
        inner.push(Value::Int(20));
    }
    let slice = v.as_list().unwrap();
    assert_eq!(slice.len(), 2);
    assert!(matches!(&slice[1], Value::Int(20)));

    // When there are multiple Arc references, as_list_mut should CoW-clone
    let v2 = v.clone();
    {
        let inner = v.as_list_mut().unwrap();
        inner.push(Value::Int(30));
    }
    // v was mutated (now has 3 elements), v2 still has 2
    assert_eq!(v.as_list().unwrap().len(), 3);
    assert_eq!(v2.as_list().unwrap().len(), 2);
}

#[test]
fn value_as_list_mut_returns_none_for_non_list() {
    let mut v = Value::Int(5);
    assert!(v.as_list_mut().is_none());
}

// ── Shared variable resolution in nested function calls ───────────

#[test]
fn shared_var_accessible_in_nested_call_during_test() {
    // Function A calls function B, both in the same module.
    // B accesses a shared variable. When testing A, B's shared
    // variable access must still work.
    let source = "\
+module Counter
+shared count:Int = 10
+fn get_count ()->Int
  +return count
+end

+fn doubled_count ()->Int
  +return get_count() * 2
+end
";
    let program = build_program(source);
    let mut env = Env::new();
    env.populate_shared_from_program(&program);

    // Test via eval_function_body_named (mimics test runner)
    let func = program.get_function("Counter.doubled_count").unwrap();
    FN_NAME_STACK.with(|s| s.borrow_mut().push("Counter.doubled_count".to_string()));
    let result = eval_function_body(&program, &func.body, &mut env).unwrap();
    FN_NAME_STACK.with(|s| s.borrow_mut().pop());

    // get_count() is called unqualified from doubled_count —
    // the fix qualifies it to "Counter.get_count" on FN_NAME_STACK
    // so `count` resolves as "Counter.count" in the shared cache.
    assert!(
        matches!(result, Value::Int(20)),
        "expected 20, got {result}"
    );
}

#[test]
fn shared_var_set_persists_across_eval_calls() {
    let source = r#"
+module Bot
+shared bot_token:String = ""
+fn set_token (raw:String)->String [mut]
  +let trimmed:String = trim(raw)
  +set bot_token = trimmed
  +return bot_token
+end

+fn auth_header ()->String
  +return concat("Bearer ", bot_token)
+end
"#;
    let program = build_program(source);
    let runtime = std::sync::Arc::new(std::sync::RwLock::new(
        crate::session::RuntimeState::default(),
    ));
    set_shared_runtime(Some(runtime.clone()));
    set_shared_program(Some(std::sync::Arc::new(program.clone())));

    let set_result = eval_call_with_input(
        &program,
        "Bot.set_token",
        &parser::Expr::String("  abc123  ".to_string()),
    )
    .unwrap();
    assert_eq!(set_result, "\"abc123\"");

    let header_result = eval_call_with_input(
        &program,
        "Bot.auth_header",
        &parser::Expr::StructLiteral(vec![]),
    )
    .unwrap();
    assert_eq!(header_result, "\"Bearer abc123\"");

    let state = runtime.read().unwrap();
    assert!(
        matches!(state.shared_vars.get("Bot.bot_token"), Some(Value::String(s)) if s.as_str() == "abc123")
    );
}

#[test]
fn shared_var_set_and_read_work_in_test_context() {
    let source = r#"
+module Bot
+shared bot_token:String = ""
+fn configure_and_read (raw:String)->String [mut]
  +let trimmed:String = trim(raw)
  +set bot_token = trimmed
  +return concat("Bearer ", bot_token)
+end

+test Bot.configure_and_read
  +with "  xyz  " -> expect "Bearer xyz"
"#;
    let program = build_program(source);
    let case = extract_test_cases(source).remove(0).1;
    let result = eval_test_case(&program, "Bot.configure_and_read", &case).unwrap();
    assert!(
        result.contains("expected \"Bearer xyz\""),
        "unexpected test result: {result}"
    );
}

#[test]
fn undeclared_shared_var_still_errors() {
    let source = r#"
+module Bot
+fn auth_header ()->String
  +return concat("Bearer ", bot_token)
+end
"#;
    let program = build_program(source);
    set_shared_program(Some(std::sync::Arc::new(program.clone())));

    let result = eval_call_with_input(
        &program,
        "Bot.auth_header",
        &parser::Expr::StructLiteral(vec![]),
    )
    .unwrap();
    assert!(
        result.contains("undefined variable `bot_token`"),
        "expected undefined variable error, got: {result}"
    );
}

#[test]
fn shared_var_nested_call_different_module_no_cross_leak() {
    // Function in module A calls function in module B.
    // Module B's function should see module B's shared vars,
    // NOT module A's shared vars with the same name.
    let source = "\
+module Alpha
+shared val:Int = 100

+fn get_both ()->Int
  +return val + Beta.get_val()
+end

+module Beta
+shared val:Int = 5

+fn get_val ()->Int
  +return val
+end
";
    let program = build_program(source);
    let mut env = Env::new();
    env.populate_shared_from_program(&program);

    let func = program.get_function("Alpha.get_both").unwrap();
    FN_NAME_STACK.with(|s| s.borrow_mut().push("Alpha.get_both".to_string()));
    let result = eval_function_body(&program, &func.body, &mut env).unwrap();
    FN_NAME_STACK.with(|s| s.borrow_mut().pop());

    // Alpha.val=100 + Beta.val=5 = 105
    assert!(
        matches!(result, Value::Int(105)),
        "expected 105, got {result}"
    );
}

#[test]
fn shared_var_inaccessible_from_wrong_module() {
    // A function in module A should NOT be able to access module B's
    // shared variables via bare name.
    let source = "\
+module OnlyHere
+shared secret:Int = 42
+fn get_secret ()->Int
  +return secret
+end

+module Other
+fn try_access ()->Int
  +return secret
+end
";
    let program = build_program(source);
    let mut env = Env::new();
    env.populate_shared_from_program(&program);

    let func = program.get_function("Other.try_access").unwrap();
    FN_NAME_STACK.with(|s| s.borrow_mut().push("Other.try_access".to_string()));
    let result = eval_function_body(&program, &func.body, &mut env);
    FN_NAME_STACK.with(|s| s.borrow_mut().pop());

    // Should fail because "Other.secret" doesn't exist
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("undefined variable"),
        "Expected 'undefined variable' error, got: {err_msg}"
    );
}

#[test]
fn qualify_function_name_returns_qualified_for_module_fn() {
    let source = "\
+module MyMod
+fn helper ()->Int
  +return 1
+end
";
    let program = build_program(source);
    assert_eq!(program.qualify_function_name("helper"), "MyMod.helper");
}

#[test]
fn qualify_function_name_returns_bare_for_top_level_fn() {
    let source = "\
+fn top_level ()->Int
  +return 1
+end
";
    let program = build_program(source);
    assert_eq!(program.qualify_function_name("top_level"), "top_level");
}

#[test]
fn qualify_function_name_preserves_already_qualified() {
    let source = "\
+module Foo
+fn bar ()->Int
  +return 1
+end
";
    let program = build_program(source);
    assert_eq!(program.qualify_function_name("Foo.bar"), "Foo.bar");
}

#[test]
fn qualify_function_name_unknown_returns_as_is() {
    let program = build_program("");
    assert_eq!(program.qualify_function_name("nonexistent"), "nonexistent");
}

#[test]
fn shared_var_nested_call_via_eval_call_with_input() {
    // Test the !eval path: eval_call_with_input should also
    // support shared variables in nested function calls.
    let source = "\
+module Store
+shared price:Int = 50
+fn get_price ()->Int
  +return price
+end

+fn total (qty:Int)->Int
  +return get_price() * qty
+end

!eval Store.total qty=3
";
    let program = build_program(source);
    let input = parser::parse(source)
        .unwrap()
        .into_iter()
        .find_map(|op| {
            if let parser::Operation::Eval(ev) = op {
                Some(ev)
            } else {
                None
            }
        })
        .unwrap();
    let result = eval_call_with_input(&program, &input.function_name, &input.input).unwrap();
    assert_eq!(result, "150");
}

#[test]
fn shared_var_deeply_nested_calls() {
    // A calls B calls C, all accessing shared state
    let source = "\
+module Deep
+shared base:Int = 7
+fn c ()->Int
  +return base
+end

+fn b ()->Int
  +return c() + 1
+end

+fn a ()->Int
  +return b() + 2
+end
";
    let program = build_program(source);
    let mut env = Env::new();
    env.populate_shared_from_program(&program);

    let func = program.get_function("Deep.a").unwrap();
    FN_NAME_STACK.with(|s| s.borrow_mut().push("Deep.a".to_string()));
    let result = eval_function_body(&program, &func.body, &mut env).unwrap();
    FN_NAME_STACK.with(|s| s.borrow_mut().pop());

    // base=7, c()=7, b()=8, a()=10
    assert!(
        matches!(result, Value::Int(10)),
        "expected 10, got {result}"
    );
}

// ═════════════════════════════════════════════════════════════════════
// Service runtime: +source add dispatch
// ═════════════════════════════════════════════════════════════════════

#[test]
fn source_add_timer_in_sync_context_no_error() {
    // Without a coroutine handle, +source add should silently succeed
    let source = r#"
+module Svc
+fn setup ()->String [io,async]
  +source add timer(5000) as poll -> on_tick
  +return "ok"
+fn on_tick ()->String
  +return "tick"
"#;
    let program = build_program(source);
    let func = program
        .get_function("Svc.setup")
        .expect("Svc.setup not found");
    let mut env = Env::new();
    // No __coroutine_handle set — sync context
    let result = eval_function_body(&program, &func.body, &mut env);
    assert!(
        result.is_ok(),
        "source add without handle should succeed: {result:?}"
    );
    assert_eq!(result.unwrap().to_string(), "\"ok\"");
}

#[test]
fn source_add_channel_in_sync_context() {
    let source = r#"
+module Svc
+fn setup ()->String [io,async]
  +source add channel as inbox -> on_msg
  +return "ok"
+fn on_msg (m:String)->String
  +return m
"#;
    let program = build_program(source);
    let func = program
        .get_function("Svc.setup")
        .expect("Svc.setup not found");
    let mut env = Env::new();
    let result = eval_function_body(&program, &func.body, &mut env);
    assert!(
        result.is_ok(),
        "channel source add should succeed: {result:?}"
    );
}

#[test]
fn source_add_event_in_sync_context() {
    let source = r#"
+module Listener
+fn setup ()->String [io,async]
  +source add Chat.new_message as msgs -> handle
  +return "ok"
+fn handle (m:String)->String
  +return m
"#;
    let program = build_program(source);
    let func = program.get_function("Listener.setup").expect("not found");
    let mut env = Env::new();
    let result = eval_function_body(&program, &func.body, &mut env);
    assert!(
        result.is_ok(),
        "event source add should succeed: {result:?}"
    );
}

#[test]
fn source_add_timer_with_async_handle() {
    // Create a real tokio channel to receive the IoRequest
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<crate::coroutine::IoRequest>(16);
        let source = r#"
+module Svc
+fn setup ()->String [io,async]
  +source add timer(3000) as poll -> on_tick
  +return "ok"
+fn on_tick ()->String
  +return "tick"
"#;
        let program = build_program(source);
        let func = program
            .get_function("Svc.setup")
            .expect("not found")
            .clone();

        // Run eval in a blocking thread
        let prog = program.clone();
        let join = tokio::task::spawn_blocking(move || {
            let mut env = Env::new();
            let handle = crate::coroutine::CoroutineHandle::new(tx);
            env.set("__coroutine_handle", Value::CoroutineHandle(handle));
            eval_function_body(&prog, &func.body, &mut env)
        });

        // Receive the IoRequest and reply
        if let Some(crate::coroutine::IoRequest::SourceAdd {
            source_type,
            interval_ms,
            alias,
            handler,
            reply,
            ..
        }) = rx.recv().await
        {
            assert_eq!(source_type, "timer");
            assert_eq!(interval_ms, Some(3000));
            assert_eq!(alias, "poll");
            assert!(
                handler.contains("on_tick"),
                "handler should contain on_tick: {handler}"
            );
            let _ = reply.send(Ok("registered".to_string()));
        } else {
            panic!("expected SourceAdd request");
        }

        let result = join.await.unwrap();
        assert!(result.is_ok(), "should succeed: {result:?}");
        assert_eq!(result.unwrap().to_string(), "\"ok\"");
    });
}

#[test]
fn source_add_timer_non_int_expr_errors() {
    let source = r#"
+module Svc
+fn setup ()->String [io,async]
  +source add timer("not_a_number") as poll -> on_tick
  +return "ok"
+fn on_tick ()->String
  +return "tick"
"#;
    let program = build_program(source);
    let func = program.get_function("Svc.setup").expect("not found");
    let mut env = Env::new();
    let result = eval_function_body(&program, &func.body, &mut env);
    assert!(result.is_err(), "string timer interval should fail");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("timer interval must be Int"),
        "error message: {err}"
    );
}

#[test]
fn source_add_module_name_from_handler() {
    // When __module_name isn't set, module_name is extracted from handler
    let source = r#"
+module MyMod
+fn setup ()->String [io,async]
  +source add channel as inbox -> on_msg
  +return "ok"
+fn on_msg (m:String)->String
  +return m
"#;
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<crate::coroutine::IoRequest>(16);
        let program = build_program(source);
        let func = program
            .get_function("MyMod.setup")
            .expect("not found")
            .clone();
        let prog = program.clone();
        let join = tokio::task::spawn_blocking(move || {
            let mut env = Env::new();
            let handle = crate::coroutine::CoroutineHandle::new(tx);
            env.set("__coroutine_handle", Value::CoroutineHandle(handle));
            // No __module_name set — should extract from handler "on_msg" -> "unknown"
            eval_function_body(&prog, &func.body, &mut env)
        });

        if let Some(crate::coroutine::IoRequest::SourceAdd {
            module_name,
            handler,
            reply,
            ..
        }) = rx.recv().await
        {
            assert_eq!(module_name, "unknown");
            assert_eq!(handler, "unknown.on_msg");
            let _ = reply.send(Ok("ok".to_string()));
        }

        let result = join.await.unwrap();
        assert!(result.is_ok());
    });
}

#[test]
fn source_add_with_module_name_env() {
    // When __module_name IS set, it's used for the module_name and full handler
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<crate::coroutine::IoRequest>(16);
        let source = r#"
+module MyMod
+fn setup ()->String [io,async]
  +source add channel as inbox -> on_msg
  +return "ok"
+fn on_msg (m:String)->String
  +return m
"#;
        let program = build_program(source);
        let func = program
            .get_function("MyMod.setup")
            .expect("not found")
            .clone();
        let prog = program.clone();
        let join = tokio::task::spawn_blocking(move || {
            let mut env = Env::new();
            let handle = crate::coroutine::CoroutineHandle::new(tx);
            env.set("__coroutine_handle", Value::CoroutineHandle(handle));
            env.set(
                "__module_name",
                Value::String(std::sync::Arc::new("MyMod".to_string())),
            );
            eval_function_body(&prog, &func.body, &mut env)
        });

        if let Some(crate::coroutine::IoRequest::SourceAdd {
            module_name,
            handler,
            reply,
            ..
        }) = rx.recv().await
        {
            assert_eq!(module_name, "MyMod");
            assert_eq!(handler, "MyMod.on_msg");
            let _ = reply.send(Ok("ok".to_string()));
        }

        let result = join.await.unwrap();
        assert!(result.is_ok());
    });
}

// ── Collection builtins: List extended, Map, Set ─────────────────────────────

fn call(name: &str, args: Vec<Value>) -> anyhow::Result<Value> {
    let p = ast::Program::default();
    let mut env = Env::new();
    eval_builtin_or_user(&p, name, args, &mut env)
}

#[test]
fn test_list_extended() {
    // pop
    let r = call("pop", vec![Value::list(vec![Value::Int(1), Value::Int(2)])]);
    assert!(r.is_ok());
    let result = r.unwrap();
    if let Value::List(pair) = result {
        assert_eq!(pair.len(), 2);
        assert!(matches!(&pair[1], Value::Int(2)));
    } else {
        panic!("pop should return a list");
    }

    // pop empty list errors
    let r = call("pop", vec![Value::list(vec![])]);
    assert!(r.is_err());

    // remove
    let r = call(
        "remove",
        vec![
            Value::list(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
            Value::Int(1),
        ],
    );
    assert!(r.is_ok());
    if let Value::List(v) = r.unwrap() {
        assert_eq!(v.len(), 2);
        assert!(matches!(&v[0], Value::Int(1)));
        assert!(matches!(&v[1], Value::Int(3)));
    } else {
        panic!("remove should return a list");
    }

    // insert
    let r = call(
        "insert",
        vec![
            Value::list(vec![Value::Int(1), Value::Int(3)]),
            Value::Int(1),
            Value::Int(2),
        ],
    );
    assert!(r.is_ok());
    if let Value::List(v) = r.unwrap() {
        assert_eq!(v.len(), 3);
        assert!(matches!(&v[1], Value::Int(2)));
    }

    // reverse
    let r = call(
        "reverse",
        vec![Value::list(vec![
            Value::Int(1),
            Value::Int(2),
            Value::Int(3),
        ])],
    );
    assert!(r.is_ok());
    if let Value::List(v) = r.unwrap() {
        assert!(matches!(&v[0], Value::Int(3)));
        assert!(matches!(&v[2], Value::Int(1)));
    }

    // sort ints
    let r = call(
        "sort",
        vec![Value::list(vec![
            Value::Int(3),
            Value::Int(1),
            Value::Int(2),
        ])],
    );
    assert!(r.is_ok());
    if let Value::List(v) = r.unwrap() {
        assert!(matches!(&v[0], Value::Int(1)));
        assert!(matches!(&v[2], Value::Int(3)));
    }

    // sort strings
    let r = call(
        "sort",
        vec![Value::list(vec![
            Value::string("banana"),
            Value::string("apple"),
            Value::string("cherry"),
        ])],
    );
    assert!(r.is_ok());
    if let Value::List(v) = r.unwrap() {
        assert!(matches!(&v[0], Value::String(s) if s.as_str() == "apple"));
    }

    // slice
    let r = call(
        "slice",
        vec![
            Value::list(vec![
                Value::Int(0),
                Value::Int(1),
                Value::Int(2),
                Value::Int(3),
            ]),
            Value::Int(1),
            Value::Int(3),
        ],
    );
    assert!(r.is_ok());
    if let Value::List(v) = r.unwrap() {
        assert_eq!(v.len(), 2);
        assert!(matches!(&v[0], Value::Int(1)));
        assert!(matches!(&v[1], Value::Int(2)));
    }

    // length of list
    let r = call("len", vec![Value::list(vec![Value::Int(1), Value::Int(2)])]);
    assert!(r.is_ok());
    assert!(matches!(r.unwrap(), Value::Int(2)));

    // contains on list
    let r = call(
        "contains",
        vec![
            Value::list(vec![Value::Int(1), Value::Int(2), Value::Int(3)]),
            Value::Int(2),
        ],
    );
    assert!(r.is_ok());
    assert!(matches!(r.unwrap(), Value::Bool(true)));

    let r = call(
        "contains",
        vec![
            Value::list(vec![Value::Int(1), Value::Int(2)]),
            Value::Int(5),
        ],
    );
    assert!(r.is_ok());
    assert!(matches!(r.unwrap(), Value::Bool(false)));
}

#[test]
fn test_map_operations() {
    // empty map
    let r = call("map", vec![]);
    assert!(r.is_ok());
    assert!(matches!(r.unwrap(), Value::Map(m) if m.is_empty()));

    // map with pairs
    let r = call(
        "map",
        vec![
            Value::string("a"),
            Value::Int(1),
            Value::string("b"),
            Value::Int(2),
        ],
    );
    assert!(r.is_ok());
    let m = r.unwrap();
    assert!(matches!(&m, Value::Map(entries) if entries.len() == 2));

    // map_set new key
    let base = Value::map(vec![(Value::string("x"), Value::Int(10))]);
    let r = call("map_set", vec![base, Value::string("y"), Value::Int(20)]);
    assert!(r.is_ok());
    assert!(matches!(&r.unwrap(), Value::Map(m) if m.len() == 2));

    // map_set existing key replaces
    let base = Value::map(vec![(Value::string("x"), Value::Int(10))]);
    let r = call("map_set", vec![base, Value::string("x"), Value::Int(99)]);
    assert!(r.is_ok());
    if let Value::Map(entries) = r.unwrap() {
        assert_eq!(entries.len(), 1);
        assert!(matches!(&entries[0].1, Value::Int(99)));
    }

    // map_get found
    let m = Value::map(vec![(Value::string("k"), Value::Int(42))]);
    let r = call("map_get", vec![m.clone(), Value::string("k")]);
    assert!(r.is_ok());
    assert!(matches!(r.unwrap(), Value::Int(42)));

    // map_get not found errors
    let r = call("map_get", vec![m.clone(), Value::string("missing")]);
    assert!(r.is_err());

    // map_get with default
    let r = call(
        "map_get",
        vec![m.clone(), Value::string("missing"), Value::Int(0)],
    );
    assert!(r.is_ok());
    assert!(matches!(r.unwrap(), Value::Int(0)));

    // map_has
    let r = call("map_has", vec![m.clone(), Value::string("k")]);
    assert!(r.is_ok());
    assert!(matches!(r.unwrap(), Value::Bool(true)));

    let r = call("map_has", vec![m.clone(), Value::string("nope")]);
    assert!(r.is_ok());
    assert!(matches!(r.unwrap(), Value::Bool(false)));

    // map_remove
    let m2 = Value::map(vec![
        (Value::string("a"), Value::Int(1)),
        (Value::string("b"), Value::Int(2)),
    ]);
    let r = call("map_remove", vec![m2, Value::string("a")]);
    assert!(r.is_ok());
    assert!(matches!(&r.unwrap(), Value::Map(entries) if entries.len() == 1));

    // map_keys
    let m3 = Value::map(vec![
        (Value::string("a"), Value::Int(1)),
        (Value::string("b"), Value::Int(2)),
    ]);
    let r = call("map_keys", vec![m3.clone()]);
    assert!(r.is_ok());
    assert!(matches!(&r.unwrap(), Value::List(v) if v.len() == 2));

    // map_values
    let r = call("map_values", vec![m3.clone()]);
    assert!(r.is_ok());
    assert!(matches!(&r.unwrap(), Value::List(v) if v.len() == 2));

    // map_entries
    let r = call("map_entries", vec![m3]);
    assert!(r.is_ok());
    if let Value::List(entries) = r.unwrap() {
        assert_eq!(entries.len(), 2);
        assert!(matches!(&entries[0], Value::List(pair) if pair.len() == 2));
    }

    // length of map
    let m4 = Value::map(vec![
        (Value::string("x"), Value::Int(1)),
        (Value::string("y"), Value::Int(2)),
    ]);
    let r = call("len", vec![m4.clone()]);
    assert!(r.is_ok());
    assert!(matches!(r.unwrap(), Value::Int(2)));

    // contains on map checks keys
    let r = call("contains", vec![m4.clone(), Value::string("x")]);
    assert!(r.is_ok());
    assert!(matches!(r.unwrap(), Value::Bool(true)));

    let r = call("contains", vec![m4, Value::string("z")]);
    assert!(r.is_ok());
    assert!(matches!(r.unwrap(), Value::Bool(false)));
}

#[test]
fn test_set_operations() {
    // empty set
    let r = call("set", vec![]);
    assert!(r.is_ok());
    assert!(matches!(r.unwrap(), Value::Set(s) if s.is_empty()));

    // set with items (deduplicated)
    let r = call(
        "set",
        vec![Value::Int(1), Value::Int(2), Value::Int(1), Value::Int(3)],
    );
    assert!(r.is_ok());
    assert!(matches!(&r.unwrap(), Value::Set(s) if s.len() == 3));

    // set_add new item
    let s = Value::set(vec![Value::Int(1), Value::Int(2)]);
    let r = call("set_add", vec![s, Value::Int(3)]);
    assert!(r.is_ok());
    assert!(matches!(&r.unwrap(), Value::Set(s) if s.len() == 3));

    // set_add duplicate
    let s = Value::set(vec![Value::Int(1), Value::Int(2)]);
    let r = call("set_add", vec![s, Value::Int(1)]);
    assert!(r.is_ok());
    assert!(matches!(&r.unwrap(), Value::Set(s) if s.len() == 2));

    // set_remove existing
    let s = Value::set(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
    let r = call("set_remove", vec![s, Value::Int(2)]);
    assert!(r.is_ok());
    assert!(matches!(&r.unwrap(), Value::Set(s) if s.len() == 2));

    // set_has
    let s = Value::set(vec![Value::Int(1), Value::Int(2)]);
    let r = call("set_has", vec![s.clone(), Value::Int(1)]);
    assert!(r.is_ok());
    assert!(matches!(r.unwrap(), Value::Bool(true)));

    let r = call("set_has", vec![s.clone(), Value::Int(99)]);
    assert!(r.is_ok());
    assert!(matches!(r.unwrap(), Value::Bool(false)));

    // set_to_list
    let s = Value::set(vec![Value::Int(1), Value::Int(2)]);
    let r = call("set_to_list", vec![s]);
    assert!(r.is_ok());
    assert!(matches!(&r.unwrap(), Value::List(v) if v.len() == 2));

    // length of set
    let s = Value::set(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
    let r = call("len", vec![s.clone()]);
    assert!(r.is_ok());
    assert!(matches!(r.unwrap(), Value::Int(3)));

    // contains on set
    let r = call("contains", vec![s.clone(), Value::Int(2)]);
    assert!(r.is_ok());
    assert!(matches!(r.unwrap(), Value::Bool(true)));

    let r = call("contains", vec![s, Value::Int(99)]);
    assert!(r.is_ok());
    assert!(matches!(r.unwrap(), Value::Bool(false)));
}

// bytecode VM placeholder
