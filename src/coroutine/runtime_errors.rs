//! `runtime_errors(after_seq)` — read-only view of the in-process error journal.
//!
//! Available to module code (spawned tasks, `+startup`, routes, write-capable turns).
//! Refused for guest and execute-only conversation turns: the journal describes the
//! whole process, not the speaker's conversation. Guests are additionally refused at
//! the IO boundary because this is not a guest tool.
use super::{CoroutineHandle, Value};
use anyhow::{bail, Result};

impl CoroutineHandle {
    pub(super) fn runtime_errors(&self, args: &[Value]) -> Result<Value> {
        if let Some(turn) = &self.turn
            && (turn.guest_tools.is_some() || !turn.may_write)
        {
            bail!("permission denied: runtime_errors is not available in this conversation");
        }
        let after_seq = match args {
            [Value::Int(seq)] if *seq >= 0 => *seq as u64,
            _ => bail!(
                "runtime_errors(after_seq:Int) expects one non-negative Int: the last seq \
                 you have seen (0 for everything still in the journal)"
            ),
        };
        Ok(Value::string(crate::error_journal::query(after_seq)))
    }
}

#[cfg(test)]
mod tests {
    use super::super::TurnIdentity;
    use super::*;

    fn turn(may_write: bool, guest: bool) -> TurnIdentity {
        TurnIdentity {
            context: "telegram:test".into(),
            principal: "telegram:user:1".into(),
            may_write,
            guest_tools: guest.then(|| vec!["web_search".to_string()]),
        }
    }

    fn handle(turn_identity: Option<TurnIdentity>) -> CoroutineHandle {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        CoroutineHandle::new(tx).with_turn(turn_identity)
    }

    #[test]
    fn runtime_errors_returns_entries_after_seq_for_module_code() {
        let seq = crate::error_journal::record(
            crate::error_journal::Category::Task,
            "RuntimeErrorsTest.builtin_marker",
            "synthetic failure",
        );
        let value = handle(None)
            .execute_await("runtime_errors", &[Value::Int(seq as i64 - 1)])
            .unwrap();
        let Value::String(text) = value else { panic!("expected String") };
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        let entries = json["entries"].as_array().unwrap();
        assert!(entries.iter().any(|e| e["source"] == "RuntimeErrorsTest.builtin_marker"));
        assert!(entries.iter().all(|e| e["seq"].as_u64().unwrap() >= seq));
        assert!(json["last_seq"].as_u64().unwrap() >= seq);
        // Admin (write-capable) turns may read it too.
        assert!(handle(Some(turn(true, false))).execute_await("runtime_errors", &[Value::Int(0)]).is_ok());
    }

    #[test]
    fn runtime_errors_refuses_guest_and_execute_only_turns_and_bad_arguments() {
        for (may_write, guest) in [(false, false), (false, true), (true, true)] {
            let error = handle(Some(turn(may_write, guest)))
                .execute_await("runtime_errors", &[Value::Int(0)])
                .unwrap_err()
                .to_string();
            assert!(error.contains("permission denied"), "{error}");
        }
        for bad in [vec![], vec![Value::Int(-1)], vec![Value::string("0")], vec![Value::Int(0), Value::Int(1)]] {
            assert!(handle(None).execute_await("runtime_errors", &bad).is_err());
        }
    }
}

/// #134: transport operations that index their arguments must refuse a call with
/// too few arguments instead of panicking.
#[cfg(test)]
mod transport_arity_tests {
    use super::super::{CoroutineHandle, IoRequest, Value};

    #[test]
    fn transport_ops_with_missing_arguments_return_errors_without_io() {
        let (sender, mut receiver) = tokio::sync::mpsc::channel::<IoRequest>(4);
        let handle = CoroutineHandle::new(sender);
        let cases: &[(&str, Vec<Value>, usize)] = &[
            ("tcp_listen", vec![], 1),
            ("tcp_accept", vec![], 1),
            ("tcp_read", vec![], 1),
            ("tcp_close", vec![], 1),
            ("tcp_write", vec![], 2),
            ("tcp_write", vec![Value::Int(3)], 2),
            ("tcp_connect", vec![Value::string("localhost")], 2),
            ("print", vec![], 1),
            ("println", vec![], 1),
            ("sleep", vec![], 1),
            ("file_read", vec![], 1),
            ("read_file_binary", vec![], 1),
            ("file_write", vec![Value::string("/tmp/x")], 2),
            ("file_exists", vec![], 1),
            ("list_dir", vec![], 1),
            ("shell_exec", vec![], 1),
        ];
        for (op, args, required) in cases {
            let error = handle.execute_await(op, args).expect_err(op).to_string();
            let plural = if *required == 1 { "" } else { "s" };
            assert_eq!(error, format!("{op} expects at least {required} argument{plural}, got {}", args.len()));
        }
        // Wrong types were and stay errors.
        let error = handle.execute_await("tcp_listen", &[Value::string("80")]).unwrap_err().to_string();
        assert_eq!(error, "tcp_listen expects Int port");
        assert!(receiver.try_recv().is_err(), "nothing reached the IO loop");
    }
}
