//! Structural C-core/IO matrix. Synthetic requests only. Result graphs stay off VM/JIT;
//! these tests distinguish execution errors from application Err values before rendering.
use super::*;
use crate::{
    coroutine::{CoroutineHandle, IoRequest},
    eval, parser, validator,
};

fn program(source: &str) -> ast::Program {
    let mut p = ast::Program::default();
    for op in parser::parse(source).unwrap() {
        validator::apply_and_validate(&mut p, &op).unwrap();
    }
    p.rebuild_function_index();
    crate::intern::set_display_interner(&p.shared_interner);
    p
}

#[derive(Clone, Copy, Debug)]
enum Entry {
    Declared,
    Async,
    Traced,
    Task,
}
const ENTRIES: [Entry; 4] = [Entry::Declared, Entry::Async, Entry::Traced, Entry::Task];

fn run(p: &ast::Program, entry: Entry) -> (Result<Value>, Vec<String>, Vec<eval::TraceStep>) {
    let (sender, mut receiver) = tokio::sync::mpsc::channel(8);
    let responder = std::thread::spawn(move || {
        let mut requests = Vec::new();
        while let Some(request) = receiver.blocking_recv() {
            match request {
                IoRequest::FileExists { path, reply } => {
                    requests.push(path);
                    reply.send(Ok(true)).unwrap();
                }
                IoRequest::FileRead { path, reply } => {
                    requests.push(path.clone());
                    reply
                        .send(if path == "operational" {
                            Err(anyhow::anyhow!("operational fixture"))
                        } else {
                            Ok("hello".into())
                        })
                        .unwrap();
                }
                IoRequest::FileWrite { path, data, reply } => {
                    assert_eq!(data, "payload");
                    requests.push(path);
                    reply.send(Ok(())).unwrap();
                }
                IoRequest::HttpGet { url, reply } => {
                    requests.push(url);
                    reply
                        .send(Err(anyhow::anyhow!("builtin transport fixture")))
                        .unwrap();
                }
                _ => panic!("unexpected request"),
            }
            assert!(requests.len() < 10, "replayed/unbounded effects");
        }
        requests
    });
    let handle = if matches!(entry, Entry::Task) {
        CoroutineHandle::new_with_task(sender, 7, Default::default(), Default::default())
    } else {
        CoroutineHandle::new(sender)
    };
    let mut steps = Vec::new();
    let result = if matches!(entry, Entry::Async | Entry::Task) {
        eval::eval_async_function(p, "root", &parser::parse_single_expr("{}").unwrap(), handle)
    } else {
        let mut env = Env::new_with_shared_interner(&p.shared_interner);
        env.set("__coroutine_handle", Value::CoroutineHandle(handle));
        let function = p.get_function("root").unwrap();
        if matches!(entry, Entry::Traced) {
            let observed = eval::tracing::observe_invocation(p, "root", function, &mut env);
            steps = observed.1;
            observed.0
        } else {
            eval_declared_function(p, "root", function, &mut env)
        }
    };
    (result, responder.join().unwrap(), steps)
}

#[test]
fn plain_success_explicit_results_and_nested_results_are_structural_across_entries() {
    for (ty, value, expected_depth) in [
        ("Result<Int>", "7", 1),
        ("Result<Int>", "Ok(7)", 1),
        ("Result<Result<Int>>", "Ok(Ok(7))", 2),
    ] {
        let p = program(&format!("+fn root ()->{ty}\n  +return {value}\n+end\n"));
        for entry in ENTRIES {
            let (result, requests, _) = run(&p, entry);
            let mut result = result.unwrap();
            for _ in 0..expected_depth {
                let Value::Ok(inner) = result else {
                    panic!("missing Result layer: {entry:?}")
                };
                result = *inner;
            }
            assert!(
                matches!(result, Value::Int(7)),
                "extra Result layer: {entry:?}"
            );
            assert!(requests.is_empty());
        }
    }
    let p = program("+fn root ()->Result<Int>\n  +return Err(\"domain\")\n+end\n");
    for entry in ENTRIES {
        assert!(matches!(run(&p, entry).0.unwrap(), Value::Err(message) if message == "domain"));
    }
}

#[test]
fn declared_result_catches_check_once_and_does_not_run_later_effects() {
    let p = program("+fn root ()->Result<Int> [io,async,fail]\n  +await before:Bool = file_exists(\"before\")\n  +check ready false ~domain\n  +await after:Bool = file_exists(\"after\")\n  +return 7\n+end\n");
    for entry in ENTRIES {
        let (result, requests, _) = run(&p, entry);
        assert!(matches!(result.unwrap(), Value::Err(message) if message == "domain"));
        assert_eq!(requests, ["before"]);
    }
}

#[test]
fn plain_call_propagates_explicit_err_and_never_runs_after() {
    let p = program("+fn reject ()->Result<Int>\n  +return Err(\"rejected\")\n+end\n+fn root ()->Int [io,async,fail]\n  +await before:Bool = file_exists(\"before\")\n  +call value:Int = reject()\n  +await after:Bool = file_exists(\"after\")\n  +return value\n+end\n");
    for entry in ENTRIES {
        let (result, requests, _) = run(&p, entry);
        let error = result.unwrap_err();
        assert_eq!(
            error.downcast_ref::<eval::result_contract::LanguageFailure>(),
            Some(&eval::result_contract::LanguageFailure::Propagation {
                message: "rejected".into()
            })
        );
        assert_eq!(requests, ["before"]);
    }
}

#[test]
fn validation_then_write_unwraps_success_once_and_never_writes_on_domain_failure() {
    for valid in [false, true] {
        let p = program(&format!("+fn validate ()->Result<Int> [fail]\n  +check valid {valid} ~invalid\n  +return 7\n+end\n+fn root ()->Int [io,async,fail]\n  +call value:Int = validate()\n  +await written:String = file_write(\"write\", \"payload\")\n  +return value\n+end\n"));
        for entry in ENTRIES {
            let (result, requests, _) = run(&p, entry);
            if valid {
                assert!(matches!(result.unwrap(), Value::Int(7)));
                assert_eq!(requests, ["write"]);
            } else {
                let error = result.unwrap_err();
                assert!(error
                    .downcast_ref::<eval::result_contract::LanguageFailure>()
                    .is_some());
                assert!(requests.is_empty());
            }
        }
    }
}

#[test]
fn operational_capture_migration_aborts_but_explicit_domain_check_recovers_once() {
    for (body, recover) in [
        ("+return 1 / 0", false),
        ("+check divisor false ~zero_divisor\n  +return 1", true),
    ] {
        let p = program(&format!("+fn divide ()->Int [fail]\n  {body}\n+end\n+fn root ()->Int [io,async]\n  +await before:Bool = file_exists(\"before\")\n  +call result:Result<Int> = divide()\n  +match result\n  +case Ok(value)\n    +return value\n  +case Err(error)\n    +await recovered:Bool = file_exists(\"recovery\")\n    +return 8\n  +end\n+end\n"));
        for entry in ENTRIES {
            let (result, requests, _) = run(&p, entry);
            if recover {
                assert!(matches!(result.unwrap(), Value::Int(8)));
                assert_eq!(requests, ["before", "recovery"]);
            } else {
                let error = result.unwrap_err();
                assert!(error
                    .downcast_ref::<eval::result_contract::LanguageFailure>()
                    .is_none());
                assert!(error.to_string().contains("division by zero"));
                assert_eq!(requests, ["before"], "no operational recovery");
            }
        }
    }
}

#[test]
fn argument_failures_are_outside_target_capture_and_suppress_later_arguments() {
    let p = program("+fn failed_arg ()->Int [fail]\n  +check ready false ~argument_failure\n  +return 1\n+end\n+fn target (a:Int,b:Int)->Int\n  +return a + b\n+end\n+fn root ()->Int [io,async,fail]\n  +await before:Bool = file_exists(\"before\")\n  +call result:Result<Int> = target(failed_arg(), 1 / 0)\n  +await recovered:Bool = file_exists(\"must_not_recover_argument\")\n  +return 8\n+end\n");
    for entry in ENTRIES {
        let (result, requests, _) = run(&p, entry);
        let error = result.unwrap_err();
        assert_eq!(
            error.downcast_ref::<eval::result_contract::LanguageFailure>(),
            Some(&eval::result_contract::LanguageFailure::Check {
                label: "argument_failure".into()
            })
        );
        assert_eq!(requests, ["before"]);
    }
}

#[test]
fn user_await_keeps_normalized_ok_and_builtin_err_without_replay() {
    for (operation, capture, expected, order) in [
        ("read_file(\"fetch\")", "String", 5, vec!["fetch", "reply"]),
        (
            "http_get(\"http-fixture\")",
            "Result<String>",
            0,
            vec!["http-fixture", "recovery"],
        ),
    ] {
        let p = program(&format!("+fn fetch ()->Result<String> [io,async]\n  +await response:{capture} = {operation}\n  +return response\n+end\n+fn root ()->Int [io,async]\n  +await result:Result<String> = fetch()\n  +match result\n  +case Ok(text)\n    +await sent:Bool = file_exists(\"reply\")\n    +return len(text)\n  +case Err(error)\n    +await recovered:Bool = file_exists(\"recovery\")\n    +return 0\n  +end\n+end\n"));
        for entry in ENTRIES {
            let (result, requests, _) = run(&p, entry);
            assert!(matches!(result.unwrap(), Value::Int(n) if n == expected));
            assert_eq!(requests, order);
        }
    }
    // Raw file IO failure is an execution error, unlike the HTTP builtin's explicit Err.
    let p = program("+fn fetch ()->Result<String> [io,async]\n  +await text:String = read_file(\"operational\")\n  +return text\n+end\n+fn root ()->Int [io,async]\n  +call result:Result<String> = fetch()\n  +await recovered:Bool = file_exists(\"must_not_recover\")\n  +return 0\n+end\n");
    for entry in ENTRIES {
        let (result, requests, _) = run(&p, entry);
        assert!(result.is_err());
        assert_eq!(requests, ["operational"]);
    }
}

#[test]
fn traced_control_flow_executes_once_and_exposes_only_root_statements() {
    let p = program("+module PrivateC\n+fn hidden ()->Int\n  +let private_sentinel:Int = 77\n  +return private_sentinel\n+end\n+fn root ()->Result<Int> [io,async]\n  +let n:Int = 0\n  +while n < 2\n    +set n = n + 1\n  +end\n  +each list(1,2) item:Int\n    +set n = n + item\n  +end\n  +if n == 5\n    +await once:Bool = file_exists(\"once\")\n    +return PrivateC.hidden()\n  +end\n  +return 0\n+end\n");
    for entry in [Entry::Declared, Entry::Traced] {
        let (result, requests, steps) = run(&p, entry);
        assert!(matches!(result.unwrap(), Value::Ok(inner) if matches!(*inner, Value::Int(77))));
        assert_eq!(requests, ["once"]);
        if matches!(entry, Entry::Traced) {
            let root = p.get_function("root").unwrap();
            assert!(steps
                .iter()
                .all(|step| step.stmt_id == root.id
                    || root.body.iter().any(|s| s.id == step.stmt_id)));
            assert!(!steps
                .iter()
                .any(|s| s.description.contains("private_sentinel")));
            assert_eq!(steps.last().unwrap().result, "Ok(77)");
        }
    }
}

#[test]
fn public_trace_does_not_gain_io_authority() {
    let p = program("+fn root ()->String [io,async]\n  +await text:String = read_file(\"must-not-read\")\n  +return text\n+end\n");
    let steps =
        eval::trace_function(&p, "root", &parser::parse_single_expr("{}").unwrap()).unwrap();
    let terminal = steps.last().unwrap();
    assert!(matches!(terminal.status, eval::TraceStatus::Fail));
    assert!(
        terminal.result.contains("requires async context"),
        "{}",
        terminal.result
    );
}
