//! Tests for the native tool dispatch layer (issue #78).
//!
//! Fixture module names are prefixed `Nt` and are unique to this module so the
//! per-process library temp dir shared by `cargo test` cannot collide with
//! other suites' fixture modules.

use super::*;
use crate::ast::Program;
use crate::coroutine::TurnIdentity;
use crate::permissions::{AccessLevel, PermissionConfig};
use crate::session::{RuntimeState, SessionMeta};
use std::sync::Arc;
#[path = "regressions.rs"]
mod regressions;

// ── Helpers ────────────────────────────────────────────────────────────────

fn perm_config() -> PermissionConfig {
    toml::from_str(
        r#"
        [groups]
        core = ["NtCore"]
        data = ["NtData"]

        [model.writer]
        core = "write"
        data = "write"
        user = "write"

        [model.reader]
        core = "read"
        data = "execute"
        user = "write"

        [model.coreexec]
        core = "execute"
        data = "write"
        user = "write"

        [model.executor]
        core = "execute"
        data = "execute"
        user = "execute"

        [model.hidden]
        core = "none"
        data = "none"
        user = "write"
        "#,
    )
    .unwrap()
}

async fn build_program(source: &str) -> Program {
    let mut program = Program::default();
    let mut runtime = RuntimeState::default();
    let mut meta = SessionMeta::new();
    let mut sandbox = None;
    crate::session::apply_to_tiers_async(
        &mut program,
        &mut runtime,
        &mut meta,
        &mut sandbox,
        source,
        None,
    )
    .await
    .unwrap();
    program
}

fn config_for(
    model: &str,
    access_level: AccessLevel,
    perm: PermissionConfig,
    program: Program,
) -> AppConfig {
    let (trigger_tx, _trigger_rx) = tokio::sync::mpsc::channel::<String>(1);
    AppConfig {
        program: Arc::new(tokio::sync::RwLock::new(program)),
        meta: Arc::new(std::sync::Mutex::new(SessionMeta::new())),
        llm_url: String::new(),
        llm_model: Arc::new(std::sync::RwLock::new(model.to_string())),
        llm_api_key: None,
        project_dir: ".".to_string(),
        io_sender: None,
        self_trigger: trigger_tx,
        task_registry: None,
        snapshot_registry: None,
        log_file: None,
        training_log: None,
        jit_cache: crate::eval::new_jit_cache(),
        event_broadcast: tokio::sync::broadcast::channel(16).0,
        opencode_git_dir: ".".to_string(),
        opencode_lock: Arc::new(tokio::sync::Mutex::new(())),
        opencode_attach: None,
        message_queue: Arc::new(tokio::sync::Mutex::new(Vec::new())),
        max_iterations: 1,
        native_tools: false,
        runtime: Arc::new(std::sync::RwLock::new(RuntimeState::default())),
        sessions: Arc::new(tokio::sync::Mutex::new(std::collections::HashMap::new())),
        save_notify: None,
        access_level,
        permission_config: Arc::new(perm),
        turn: None,
    }
}

async fn writer_program() -> Program {
    let mut p = build_program("+module NtCore\n+fn add (a:Int, b:Int)->Int\n  +return a + b\n+end\n").await;
    p.require_modules = true;
    p
}

async fn untested_program() -> Program {
    let mut p = build_program(
        "+module NtCore\n+fn big (a:Int)->Int\n  +let x:Int = a\n  +let y:Int = x + 1\n  +return y\n+end\n",
    )
    .await;
    p.require_modules = true;
    p
}

fn module_names(program: &Program) -> Vec<String> {
    let mut v: Vec<String> = program.modules.iter().map(|m| m.name.clone()).collect();
    v.sort();
    v
}

// ── Schema / closed-object validation ─────────────────────────────────────

#[test]
fn tool_definitions_are_complete_and_closed() {
    let cfg = config_for(
        "writer",
        AccessLevel::Full,
        perm_config(),
        Program::default(),
    );
    let defs = tool_definitions(&cfg);
    let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["inspect", "mutate", "test", "evaluate", "finish"]
    );
    for d in &defs {
        assert_eq!(d.parameters["additionalProperties"], false, "{:?}", d.name);
        assert!(d.parameters["type"] == "object");
    }
}

#[test]
fn validate_rejects_extra_fields() {
    assert!(validate_native_tool("finish", r#"{"x": 1}"#).is_err());
    assert!(validate_native_tool("evaluate", r#"{"function":"A.b","bogus":1}"#).is_err());
    assert!(validate_native_tool("inspect", r#"{"kind":"symbols","extra":""}"#).is_err());
}

#[test]
fn validate_rejects_missing_or_wrong_typed_fields() {
    assert!(validate_native_tool("mutate", r#"{}"#).is_err());
    assert!(validate_native_tool("test", r#"{"source":123}"#).is_err());
    assert!(validate_native_tool("inspect", r#"{"kind":123}"#).is_err());
    assert!(validate_native_tool("inspect", r#"{"kind":"source"}"#).is_err());
    assert!(validate_native_tool("evaluate", r#"{"function":""}"#).is_err());
}

#[test]
fn validate_rejects_non_object_and_unknown_tool() {
    assert!(validate_native_tool("finish", "[]").is_err());
    assert!(validate_native_tool("finish", "\"x\"").is_err());
    assert!(validate_native_tool("bogus_tool", "{}").is_err());
}

#[test]
fn validate_accepts_wellformed_calls() {
    assert!(validate_native_tool("finish", "{}").is_ok());
    assert!(validate_native_tool("evaluate", r#"{"function":"A.b"}"#).is_ok());
    assert!(validate_native_tool("evaluate", r#"{"function":"A.b","input":"x=1"}"#).is_ok());
    assert!(validate_native_tool("inspect", r#"{"kind":"symbols"}"#).is_ok());
    assert!(validate_native_tool("inspect", r#"{"kind":"symbols","target":"M"}"#).is_ok());
    assert!(validate_native_tool("inspect", r#"{"kind":"source","target":"M.f"}"#).is_ok());
    assert!(validate_native_tool("mutate", r#"{"source":"+module M\n+fn f()->Int\n  +return 1\n+end\n"}"#).is_ok());
    assert!(validate_native_tool("test", r#"{"source":"+test M.f\n  +with 1 -> expect 1\n"}"#).is_ok());
}

// ── Injection / op-kind rejection (before any side effect) ────────────────

#[test]
fn validate_rejects_evaluate_newline_injection() {
    let err = validate_native_tool(
        "evaluate",
        r#"{"function":"A.b","input":"1\n!done"}"#,
    )
    .unwrap_err();
    assert!(err.contains("newline"), "{err}");
}

#[test]
fn validate_rejects_evaluate_non_qualified_function() {
    for bad in ["foo(1)", "a b", "A.b()", "1+1", "A..b", "!done"] {
        let args = format!(r#"{{"function":"{bad}"}}"#);
        assert!(
            validate_native_tool("evaluate", &args).is_err(),
            "expected rejection for `{bad}`"
        );
    }
}

#[test]
fn validate_rejects_mutate_command_and_test_smuggling() {
    // top-level !done after a module
    assert!(validate_native_tool(
        "mutate",
        r#"{"source":"+module M\n+fn f()->Int\n  +return 1\n+end\n!done\n"}"#
    )
    .is_err());
    // +test inside a module body
    assert!(validate_native_tool(
        "mutate",
        r#"{"source":"+module M\n+fn f()->Int\n  +return 1\n+end\n+test M.f\n  +with 1 -> expect 1\n"}"#
    )
    .is_err());
    // top-level +fn (not allowed — only +module)
    assert!(validate_native_tool(
        "mutate",
        r#"{"source":"+fn f()->Int\n  +return 1\n+end\n"}"#
    )
    .is_err());
    // !eval as a module-body entry
    assert!(validate_native_tool(
        "mutate",
        r#"{"source":"+module M\n!eval M.f\n"}"#
    )
    .is_err());
}

#[test]
fn validate_rejects_test_non_qualified_or_module_smuggling() {
    assert!(validate_native_tool(
        "test",
        r#"{"source":"+test f\n  +with 1 -> expect 1\n"}"#
    )
    .is_err());
    assert!(validate_native_tool(
        "test",
        r#"{"source":"+module M\n+fn f()->Int\n  +return 1\n+end\n"}"#
    )
    .is_err());
}

#[test]
fn nested_module_smuggling_rejected_recursively() {
    let nested = parser::ModuleDecl {
        name: "Inner".into(),
        body: vec![parser::Operation::Done],
        doc: None,
    };
    let outer = parser::ModuleDecl {
        name: "Outer".into(),
        body: vec![parser::Operation::Module(nested)],
        doc: None,
    };
    let ops = vec![parser::Operation::Module(outer)];
    let err = validate_mutate_ops(&ops).unwrap_err();
    assert!(err.contains("!done"), "{err}");
}

// ── Eligibility ───────────────────────────────────────────────────────────

#[test]
fn eligibility_excludes_non_write_and_guest() {
    let cfg = config_for("executor", AccessLevel::Full, perm_config(), Program::default());
    assert!(!native_tools_eligible(&cfg));

    let cfg = config_for("writer", AccessLevel::Full, perm_config(), Program::default());
    assert!(native_tools_eligible(&cfg));

    // guest turn → not eligible, even for a write model
    let mut guest = config_for("writer", AccessLevel::Full, perm_config(), Program::default());
    guest.turn = Some(TurnIdentity {
        context: "c".into(),
        principal: "p".into(),
        may_write: false,
        guest_tools: Some(vec!["web_search".to_string()]),
    });
    assert!(!native_tools_eligible(&guest));
}

#[tokio::test]
async fn guest_and_nonwrite_dispatch_denied() {
    let program = writer_program().await;
    let mut guest = config_for(
        "writer",
        AccessLevel::Full,
        perm_config(),
        program.clone(),
    );
    guest.turn = Some(TurnIdentity {
        context: "c".into(),
        principal: "p".into(),
        may_write: false,
        guest_tools: Some(vec![]),
    });
    let mut session = guest.snapshot_working_set().await;
    let baseline = guest.program.read().await.clone();
    let res = execute_native_tool("finish", "{}", &guest, &mut session, &baseline).await;
    assert!(res.execution.has_errors);
    assert!(res.content.contains("guest"));

    let mut cfg = config_for(
        "executor",
        AccessLevel::Full,
        perm_config(),
        program.clone(),
    );
    let mut session = cfg.snapshot_working_set().await;
    let baseline = cfg.program.read().await.clone();
    let res = execute_native_tool("finish", "{}", &cfg, &mut session, &baseline).await;
    assert!(res.execution.has_errors);
    assert!(res.content.contains("not write-capable"));
}

// ── Inspection permissions + filtering ────────────────────────────────────

#[tokio::test]
async fn inspect_source_requires_read() {
    let program = writer_program().await;
    // coreexec model: write-capable overall, but core=execute → cannot read source
    let cfg = config_for("coreexec", AccessLevel::Full, perm_config(), program.clone());
    let mut session = cfg.snapshot_working_set().await;
    let baseline = cfg.program.read().await.clone();
    let res = execute_native_tool(
        "inspect",
        r#"{"kind":"source","target":"NtCore.add"}"#,
        &cfg,
        &mut session,
        &baseline,
    )
    .await;
    assert!(res.execution.has_errors, "execute < Read must deny source");
    assert!(res.content.contains("need Read"), "{}", res.content);

    // reader model: core=read → allowed
    let cfg = config_for("reader", AccessLevel::Full, perm_config(), program.clone());
    let mut session = cfg.snapshot_working_set().await;
    let baseline = cfg.program.read().await.clone();
    let res = execute_native_tool(
        "inspect",
        r#"{"kind":"source","target":"NtCore.add"}"#,
        &cfg,
        &mut session,
        &baseline,
    )
    .await;
    assert!(!res.execution.has_errors, "{}", res.content);
    assert!(res.content.contains("add"), "{}", res.content);
}

#[tokio::test]
async fn inspect_metadata_never_discloses_hidden_modules() {
    let program = build_program(
        "+module NtCore\n+fn corefn (a:Int)->Int\n  +return a\n+end\n+module NtData\n+fn datafn (a:Int)->Int\n  +return a\n+end\n+module NtUser\n+fn userfn (a:Int)->Int\n  +return a\n+end\n",
    )
    .await;
    let cfg = config_for("hidden", AccessLevel::Full, perm_config(), program.clone());
    let mut session = cfg.snapshot_working_set().await;
    let baseline = cfg.program.read().await.clone();

    let res = execute_native_tool(
        "inspect",
        r#"{"kind":"symbols"}"#,
        &cfg,
        &mut session,
        &baseline,
    )
    .await;
    assert!(!res.execution.has_errors);
    assert!(res.content.contains("NtUser"), "{}", res.content);
    assert!(!res.content.contains("NtCore"), "{}", res.content);
    assert!(!res.content.contains("NtData"), "{}", res.content);

    let res = execute_native_tool(
        "inspect",
        r#"{"kind":"library"}"#,
        &cfg,
        &mut session,
        &baseline,
    )
    .await;
    assert!(!res.content.contains("NtCore"), "{}", res.content);
    assert!(!res.content.contains("NtData"), "{}", res.content);
}

#[tokio::test]
async fn inspect_symbols_scoped_denies_hidden_module() {
    let program = writer_program().await;
    let cfg = config_for("hidden", AccessLevel::Full, perm_config(), program.clone());
    let mut session = cfg.snapshot_working_set().await;
    let baseline = cfg.program.read().await.clone();
    let res = execute_native_tool(
        "inspect",
        r#"{"kind":"symbols","target":"NtCore"}"#,
        &cfg,
        &mut session,
        &baseline,
    )
    .await;
    assert!(res.execution.has_errors, "hidden module must not be listed");
}

// ── Mutate permissions ────────────────────────────────────────────────────

#[tokio::test]
async fn mutate_requires_write_on_every_module() {
    let program = writer_program().await;
    // reader model: core=read → mutate NtCore denied
    let cfg = config_for("reader", AccessLevel::Full, perm_config(), program.clone());
    let mut session = cfg.snapshot_working_set().await;
    let baseline = cfg.program.read().await.clone();
    let before = module_names(&session.program);
    let res = execute_native_tool(
        "mutate",
        r#"{"source":"+module NtCore\n+fn x()->Int\n  +return 1\n+end\n"}"#,
        &cfg,
        &mut session,
        &baseline,
    )
    .await;
    assert!(res.execution.has_errors, "read < Write must deny mutate");
    assert!(res.content.contains("need Write"), "{}", res.content);
    assert_eq!(module_names(&session.program), before, "no side effects");
}

#[tokio::test]
async fn mutate_rejection_has_no_side_effects() {
    let program = writer_program().await;
    let cfg = config_for("writer", AccessLevel::Full, perm_config(), program.clone());
    let mut session = cfg.snapshot_working_set().await;
    let baseline = cfg.program.read().await.clone();
    let before = module_names(&session.program);

    // top-level !done smuggled after module
    let res = execute_native_tool(
        "mutate",
        r#"{"source":"+module NtReject\n+fn f()->Int\n  +return 1\n+end\n!done\n"}"#,
        &cfg,
        &mut session,
        &baseline,
    )
    .await;
    assert!(res.execution.has_errors);
    assert_eq!(module_names(&session.program), before, "smuggled command must not apply");
}

#[tokio::test]
async fn mutate_applies_valid_module() {
    let program = writer_program().await;
    let cfg = config_for("writer", AccessLevel::Full, perm_config(), program.clone());
    let mut session = cfg.snapshot_working_set().await;
    let baseline = cfg.program.read().await.clone();
    let res = execute_native_tool(
        "mutate",
        r#"{"source":"+module NtMutApply\n+fn f()->Int\n  +return 7\n+end\n"}"#,
        &cfg,
        &mut session,
        &baseline,
    )
    .await;
    assert!(!res.execution.has_errors, "{}", res.content);
    assert!(session.program.get_function("NtMutApply.f").is_some());
}

// ── Test tool ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_requires_write_target() {
    let program = writer_program().await;
    let cfg = config_for("reader", AccessLevel::Full, perm_config(), program.clone());
    let mut session = cfg.snapshot_working_set().await;
    let baseline = cfg.program.read().await.clone();
    let res = execute_native_tool(
        "test",
        r#"{"source":"+test NtCore.add\n  +with a=2 b=3 -> expect 5\n"}"#,
        &cfg,
        &mut session,
        &baseline,
    )
    .await;
    assert!(res.execution.has_errors, "read < Write must deny test");
    assert!(res.content.contains("needs Write"), "{}", res.content);
}

#[tokio::test]
async fn test_stores_passing_case_in_ast() {
    let program = writer_program().await;
    let cfg = config_for("writer", AccessLevel::Full, perm_config(), program.clone());
    let mut session = cfg.snapshot_working_set().await;
    let baseline = cfg.program.read().await.clone();
    let res = execute_native_tool(
        "test",
        r#"{"source":"+test NtCore.add\n  +with a=2 b=3 -> expect 5\n"}"#,
        &cfg,
        &mut session,
        &baseline,
    )
    .await;
    assert!(!res.execution.has_errors, "{}", res.content);
    let f = session.program.get_function("NtCore.add").unwrap();
    assert_eq!(f.tests.len(), 1, "passing test must be persisted on the AST");
    assert!(f.tests.iter().all(|t| t.passed));
}

// ── Evaluate ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn evaluate_requires_execute_target() {
    let program = writer_program().await;
    // hidden model: core=none → evaluate NtCore.add denied
    let cfg = config_for("hidden", AccessLevel::Full, perm_config(), program.clone());
    let mut session = cfg.snapshot_working_set().await;
    let baseline = cfg.program.read().await.clone();
    let res = execute_native_tool(
        "evaluate",
        r#"{"function":"NtCore.add"}"#,
        &cfg,
        &mut session,
        &baseline,
    )
    .await;
    assert!(res.execution.has_errors);
    assert!(res.content.contains("needs Execute"), "{}", res.content);
}

#[tokio::test]
async fn evaluate_runs_qualified_function() {
    let program = writer_program().await;
    let cfg = config_for("writer", AccessLevel::Full, perm_config(), program.clone());
    let mut session = cfg.snapshot_working_set().await;
    let baseline = cfg.program.read().await.clone();
    let res = execute_native_tool(
        "evaluate",
        r#"{"function":"NtCore.add","input":"a=2 b=3"}"#,
        &cfg,
        &mut session,
        &baseline,
    )
    .await;
    assert!(!res.execution.has_errors, "{}", res.content);
    assert!(
        res.content.contains("= 5"),
        "expected eval result 5, got {}",
        res.content
    );
}

#[tokio::test]
async fn evaluate_refuses_untested_multi_statement() {
    let program = untested_program().await;
    let cfg = config_for("writer", AccessLevel::Full, perm_config(), program.clone());
    let mut session = cfg.snapshot_working_set().await;
    let baseline = cfg.program.read().await.clone();
    let res = execute_native_tool(
        "evaluate",
        r#"{"function":"NtCore.big"}"#,
        &cfg,
        &mut session,
        &baseline,
    )
    .await;
    assert!(res.execution.has_errors, "untested >2-stmt fn must refuse eval");
    assert!(
        res.content.contains("no passing tests") || res.content.contains("test"),
        "{}",
        res.content
    );
}

// ── Finish / immutable baseline ───────────────────────────────────────────

#[tokio::test]
async fn finish_accepts_unchanged_old_debt() {
    // Pre-existing untested debt is excused when unchanged from the baseline.
    let program = untested_program().await;
    let cfg = config_for("writer", AccessLevel::Full, perm_config(), program.clone());
    let baseline = cfg.program.read().await.clone();
    let mut session = cfg.snapshot_working_set().await;
    let res = execute_native_tool("finish", "{}", &cfg, &mut session, &baseline).await;
    assert!(
        res.execution.done_accepted && !res.execution.has_errors,
        "old debt unchanged from baseline must be excused: {}",
        res.content
    );
}

#[tokio::test]
async fn finish_refuses_new_untested_function() {
    let program = untested_program().await;
    let cfg = config_for("writer", AccessLevel::Full, perm_config(), program.clone());
    let baseline = cfg.program.read().await.clone();
    let mut session = cfg.snapshot_working_set().await;

    // Introduce a brand-new untested >2-statement function.
    let res = execute_native_tool(
        "mutate",
        r#"{"source":"+module NtMutFinish\n+fn big2 (a:Int)->Int\n  +let x:Int = a\n  +let y:Int = x + 1\n  +return y\n+end\n"}"#,
        &cfg,
        &mut session,
        &baseline,
    )
    .await;
    assert!(!res.execution.has_errors, "{}", res.content);

    let res = execute_native_tool("finish", "{}", &cfg, &mut session, &baseline).await;
    assert!(
        res.execution.has_errors && !res.execution.done_accepted,
        "new untested function must block finish: {}",
        res.content
    );
}

// ── Authorized preflight (no existence) ───────────────────────────────────

#[test]
fn authorized_preflight_checks_permission_but_not_existence() {
    let cfg = config_for("reader", AccessLevel::Full, perm_config(), Program::default());
    // Permission denied (core=read < Write).
    assert!(validate_native_tool_authorized(
        "mutate",
        r#"{"source":"+module NtCore\n+fn f()->Int\n  +return 1\n+end\n"}"#,
        &cfg,
    )
    .is_err());
    // Non-existent target module is fine for preflight (existence checked later).
    assert!(validate_native_tool_authorized(
        "evaluate",
        r#"{"function":"DoesNotExist.yet"}"#,
        &cfg,
    )
    .is_ok());
    // Well-formed but wrong-schema still fails.
    assert!(validate_native_tool_authorized("finish", r#"{"x":1}"#, &cfg).is_err());
}
