use std::process::Command;

#[test]
fn cli_tests_observe_normalized_invocations_without_test_only_success_wrapping() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("contract.ax");
    std::fs::write(&path, "+fn plain ()->Result<Int>\n  +return 7\n+end\n+fn rejected ()->Result<Int> [fail]\n  +check valid false ~invalid\n  +return 7\n+end\n+fn ordinary ()->Bool\n  +match plain()\n  +case Ok(value)\n    +return value == 7\n  +case Err(error)\n    +return false\n  +end\n+end\n+fn operational ()->Result<Int>\n  +return 1 / 0\n+end\n+test plain\n  +with -> expect Ok\n+test rejected\n  +with -> expect Err(invalid)\n+test ordinary\n  +with -> expect true\n+test operational\n  +with -> expect Err\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_adapsis"))
        .arg("test")
        .arg(path)
        .env("ADAPSIS_MODULES_DIR", dir.path().join("modules"))
        .env("ADAPSIS_MEMORY_DB", dir.path().join("unused.lbug"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("TEST SUMMARY: 4 passed, 0 failed"));
}

#[test]
fn cli_run_async_uses_declared_result_contract_without_gaining_io() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("contract.ax");
    std::fs::write(&path, "+fn main ()->Result<Int>\n  +return 7\n+end\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_adapsis"))
        .arg("run-async")
        .arg(path)
        .env("ADAPSIS_MODULES_DIR", dir.path().join("modules"))
        .env("ADAPSIS_MEMORY_DB", dir.path().join("unused.lbug"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("Result: Ok(7)"));
}

#[test]
fn cli_run_async_operational_failure_is_nonzero_but_same_text_data_is_successful() {
    for (return_type, expression, success) in [
        ("Result<Int>", "1 / 0", false),
        ("Result<String>", "Err(\"division by zero\")", true),
        ("String", "\"division by zero\"", true),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("contract.ax");
        std::fs::write(
            &path,
            format!("+fn main ()->{return_type}\n  +return {expression}\n+end\n"),
        )
        .unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_adapsis"))
            .arg("run-async")
            .arg(path)
            .env("ADAPSIS_MODULES_DIR", dir.path().join("modules"))
            .env("ADAPSIS_MEMORY_DB", dir.path().join("unused.lbug"))
            .output()
            .unwrap();
        assert_eq!(output.status.success(), success);
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(text.contains("division by zero"), "{text}");
    }
}

#[test]
fn cli_eval_exit_status_follows_api_success_not_result_text() {
    use std::io::{Read, Write};
    for (success, result) in [(false, "division by zero"), (true, "Err(division by zero)")] {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buffer = [0_u8; 4096];
            stream.read(&mut buffer).unwrap();
            let body = serde_json::json!({"success":success,"result":result}).to_string();
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
        });
        let dir = tempfile::tempdir().unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_adapsis"))
            .args(["eval", "probe", "--api", &format!("http://{address}")])
            .env("ADAPSIS_MODULES_DIR", dir.path().join("modules"))
            .env("ADAPSIS_MEMORY_DB", dir.path().join("unused.lbug"))
            .output()
            .unwrap();
        server.join().unwrap();
        assert_eq!(output.status.success(), success);
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(text.contains(result), "{text}");
    }
}
