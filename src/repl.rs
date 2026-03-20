//! Interactive REPL for Forge.
//!
//! Commands:
//!   +fn, +type, +let, etc.  — Forge mutations (applied to program)
//!   !test, !eval, !trace    — evaluation commands
//!   ?symbols, ?callers      — queries
//!   /save <path>            — save session
//!   /load <path>            — load session
//!   /history                — show recent history
//!   /rewind <N>             — rewind to revision N
//!   /status                 — show program state
//!   /ask <text>             — send natural language to the LLM
//!   /quit                   — exit

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use anyhow::Result;

use crate::eval;
use crate::llm::{ChatMessage, LlmBackend, LlmClient};
use crate::parser;
use crate::prompt;
use crate::session::Session;
use crate::typeck;
use crate::validator;

pub async fn run_repl<B: LlmBackend>(
    llm: LlmClient<B>,
    session_path: Option<PathBuf>,
) -> Result<()> {
    let mut session = if let Some(ref path) = session_path {
        if path.exists() {
            println!("Loading session from {}...", path.display());
            let s = Session::load(path)?;
            println!(
                "Loaded: revision {}, {} mutations, {} history entries",
                s.revision,
                s.mutations.len(),
                s.history.len()
            );
            s
        } else {
            println!("New session (will save to {})", path.display());
            Session::new()
        }
    } else {
        Session::new()
    };

    let mut llm_messages = vec![ChatMessage::system(prompt::system_prompt())];

    println!("Forge REPL — just talk, or type Forge code (+fn, !eval, ?symbols)");
    println!("revision: {}", session.revision);
    println!();

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("forge[{}]> ", session.revision);
        stdout.flush()?;

        // Read input — may be multi-line (collect until blank line or single-line command)
        let mut input = String::new();
        let mut reader = stdin.lock();

        let mut first_line = String::new();
        if reader.read_line(&mut first_line)? == 0 {
            break; // EOF
        }
        let first_line = first_line.trim_end_matches('\n').trim_end_matches('\r');

        if first_line.is_empty() {
            continue;
        }

        // Single-line commands
        if first_line.starts_with('/') {
            handle_slash_command(
                first_line,
                &mut session,
                &session_path,
                &llm,
                &mut llm_messages,
            )
            .await?;
            continue;
        }

        // Check if this is Forge code or natural language
        let is_forge = first_line.starts_with('+')
            || first_line.starts_with('!')
            || first_line.starts_with('?');

        if is_forge {
            // Collect multi-line input for Forge code
            input.push_str(first_line);
            input.push('\n');

            let needs_body = first_line.starts_with("+fn ")
                || first_line.starts_with("+module ")
                || first_line.starts_with("!test ")
                || first_line.starts_with("+if ")
                || first_line.starts_with("+each ")
                || first_line.starts_with("+while ");

            if needs_body {
                loop {
                    print!("  ... ");
                    stdout.flush()?;
                    let mut line = String::new();
                    if reader.read_line(&mut line)? == 0 {
                        break;
                    }
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        break;
                    }
                    input.push_str(&line);
                }
            }

            process_input(&input, &mut session).await;
        } else {
            // Natural language → send to LLM with feedback loop
            ask_and_apply(
                first_line,
                &mut session,
                &llm,
                &mut llm_messages,
            )
            .await;
        }

        // Auto-save if path is set
        if let Some(ref path) = session_path {
            if let Err(e) = session.save(path) {
                eprintln!("  (auto-save failed: {e})");
            }
        }
    }

    println!("Goodbye.");
    Ok(())
}

async fn process_input(input: &str, session: &mut Session) {
    // First, apply any mutations
    let operations = match session.parse_operations(input) {
        Ok(ops) => ops,
        Err(e) => {
            eprintln!("  Parse error: {e}");
            return;
        }
    };

    // Separate mutations from actions
    let has_mutations = operations.iter().any(|op| {
        !matches!(
            op,
            parser::Operation::Test(_)
                | parser::Operation::Trace(_)
                | parser::Operation::Eval(_)
                | parser::Operation::Query(_)
        )
    });

    // Handle !undo before apply (it modifies program state directly)
    let has_undo = operations.iter().any(|op| matches!(op, parser::Operation::Undo));
    if has_undo {
        if session.revision > 0 {
            let prev = session.revision - 1;
            match session.rewind_to(prev) {
                Ok(()) => println!("  Undone. Now at revision {prev}"),
                Err(e) => eprintln!("  Undo failed: {e}"),
            }
        } else {
            println!("  Nothing to undo");
        }
        return;
    }

    if has_mutations {
        match session.apply(input) {
            Ok(results) => {
                for (msg, ok) in &results {
                    if *ok {
                        println!("  OK: {msg}");
                    } else {
                        eprintln!("  ERROR: {msg}");
                    }
                }
            }
            Err(e) => {
                eprintln!("  Apply error: {e}");
                return;
            }
        }
    }

    // Handle actions
    for op in &operations {
        match op {
            parser::Operation::Undo => {
                if session.revision > 0 {
                    let prev = session.revision - 1;
                    match session.rewind_to(prev) {
                        Ok(()) => println!("  Undone. Now at revision {prev}"),
                        Err(e) => eprintln!("  Undo failed: {e}"),
                    }
                } else {
                    println!("  Nothing to undo");
                }
            }
            parser::Operation::Test(test) => {
                println!("  Testing {}:", test.function_name);
                let mut passed = 0;
                let mut failed = 0;
                let mut details = Vec::new();
                for (i, case) in test.cases.iter().enumerate() {
                    match eval::eval_test_case(&session.program, &test.function_name, case) {
                        Ok(msg) => {
                            println!("    PASS [{i}]: {msg}");
                            passed += 1;
                            details.push(format!("PASS: {msg}"));
                        }
                        Err(e) => {
                            eprintln!("    FAIL [{i}]: {e}");
                            failed += 1;
                            details.push(format!("FAIL: {e}"));
                        }
                    }
                }
                session.record_test(&test.function_name, passed, failed, details);
            }
            parser::Operation::Eval(ev) => {
                match eval::eval_compiled_or_interpreted(
                    &session.program,
                    &ev.function_name,
                    &ev.input,
                ) {
                    Ok((result, compiled)) => {
                        let tag = if compiled { " [compiled]" } else { "" };
                        println!("  = {result}{tag}");
                        session.record_eval(
                            &ev.function_name,
                            &format!("{:?}", ev.input),
                            &result,
                        );
                    }
                    Err(e) => eprintln!("  Eval error: {e}"),
                }
            }
            parser::Operation::Trace(trace) => {
                match eval::trace_function(
                    &session.program,
                    &trace.function_name,
                    &trace.input,
                ) {
                    Ok(steps) => {
                        for step in &steps {
                            println!("    > {step}");
                        }
                        session.record_trace(&trace.function_name, steps.len());
                    }
                    Err(e) => eprintln!("  Trace error: {e}"),
                }
            }
            parser::Operation::Query(query) => {
                let table = typeck::build_symbol_table(&session.program);
                let response = typeck::handle_query(&session.program, &table, query);
                println!("  {response}");
                session.record_query(query, &response);
            }
            _ => {} // mutations already handled
        }
    }
}

/// Send natural language to the LLM, apply generated code, retry on errors.
async fn ask_and_apply<B: LlmBackend>(
    user_input: &str,
    session: &mut Session,
    llm: &LlmClient<B>,
    llm_messages: &mut Vec<ChatMessage>,
) {
    let max_retries = 5;

    // Build context with program state and history
    let context = format!(
        "{}\n\n{}\n\nUser request: {user_input}",
        validator::program_summary(&session.program),
        session.format_recent_history(10)
    );
    llm_messages.push(ChatMessage::user(context));

    for attempt in 0..max_retries {
        if attempt > 0 {
            println!("  [retry {attempt}/{max_retries}...]");
        } else {
            println!("  [asking LLM...]");
        }

        let output = match llm.generate(llm_messages.clone()).await {
            Ok(o) => o,
            Err(e) => {
                eprintln!("  LLM error: {e}");
                return;
            }
        };
        println!();

        llm_messages.push(ChatMessage::assistant(&output.text));

        // Extract code from the response
        let code = if output.code.is_empty() {
            let mut lines = Vec::new();
            for line in output.text.lines() {
                let t = line.trim();
                if t.starts_with('+') || t.starts_with('!') || t.starts_with('?') || t == "end" {
                    lines.push(line.to_string());
                }
            }
            lines.join("\n")
        } else {
            output.code
        };

        if code.is_empty() || code.trim() == "DONE" {
            // LLM responded with text only, no code — that's fine
            return;
        }

        // Try to apply the code
        println!("  [applying code...]");
        let operations = match session.parse_operations(&code) {
            Ok(ops) => ops,
            Err(e) => {
                let error_msg = format!("Parse error: {e}");
                eprintln!("  {error_msg}");
                // Feed error back to LLM for retry
                llm_messages.push(ChatMessage::user(format!(
                    "Your code had an error:\n{error_msg}\n\nPlease fix and resend the complete code."
                )));
                continue;
            }
        };

        // Apply mutations
        let mut errors = Vec::new();
        let has_mutations = operations.iter().any(|op| {
            !matches!(
                op,
                parser::Operation::Test(_)
                    | parser::Operation::Trace(_)
                    | parser::Operation::Eval(_)
                    | parser::Operation::Query(_)
            )
        });

        if has_mutations {
            match session.apply(&code) {
                Ok(results) => {
                    for (msg, ok) in &results {
                        if *ok {
                            println!("  OK: {msg}");
                        } else {
                            eprintln!("  ERROR: {msg}");
                            errors.push(msg.clone());
                        }
                    }
                }
                Err(e) => {
                    let msg = format!("{e}");
                    eprintln!("  ERROR: {msg}");
                    errors.push(msg);
                }
            }
        }

        // Run tests and evals
        for op in &operations {
            match op {
                parser::Operation::Test(test) => {
                    println!("  Testing {}:", test.function_name);
                    let mut passed = 0;
                    let mut failed = 0;
                    for (i, case) in test.cases.iter().enumerate() {
                        match eval::eval_test_case(&session.program, &test.function_name, case) {
                            Ok(msg) => {
                                println!("    PASS [{i}]: {msg}");
                                passed += 1;
                            }
                            Err(e) => {
                                let msg = format!("{e}");
                                eprintln!("    FAIL [{i}]: {msg}");
                                failed += 1;
                                errors.push(format!("test fail: {msg}"));
                            }
                        }
                    }
                    session.record_test(&test.function_name, passed, failed, vec![]);
                }
                parser::Operation::Eval(ev) => {
                    match eval::eval_compiled_or_interpreted(
                        &session.program,
                        &ev.function_name,
                        &ev.input,
                    ) {
                        Ok((result, compiled)) => {
                            let tag = if compiled { " [compiled]" } else { "" };
                            println!("  = {result}{tag}");
                            session.record_eval(&ev.function_name, "", &result);
                        }
                        Err(e) => {
                            eprintln!("  Eval error: {e}");
                            errors.push(format!("eval error: {e}"));
                        }
                    }
                }
                parser::Operation::Query(query) => {
                    let table = typeck::build_symbol_table(&session.program);
                    let response = typeck::handle_query(&session.program, &table, query);
                    println!("  {response}");
                }
                _ => {} // mutations already applied
            }
        }

        if errors.is_empty() {
            // Everything succeeded
            return;
        }

        // Feed errors back to LLM for retry
        let error_summary = errors.join("\n");
        llm_messages.push(ChatMessage::user(format!(
            "There were errors:\n{error_summary}\n\n\
             Please fix the issues and resend the complete corrected code."
        )));
    }

    eprintln!("  Max retries reached. Some errors remain.");
}

async fn handle_slash_command<B: LlmBackend>(
    line: &str,
    session: &mut Session,
    session_path: &Option<PathBuf>,
    llm: &LlmClient<B>,
    llm_messages: &mut Vec<ChatMessage>,
) -> Result<()> {
    let parts: Vec<&str> = line.splitn(2, ' ').collect();
    let cmd = parts[0];
    let arg = parts.get(1).copied().unwrap_or("");

    match cmd {
        "/quit" | "/exit" | "/q" => {
            if let Some(path) = session_path {
                session.save(path)?;
                println!("Session saved to {}", path.display());
            }
            std::process::exit(0);
        }
        "/save" => {
            let path = if arg.is_empty() {
                session_path
                    .as_deref()
                    .unwrap_or(std::path::Path::new("session.json"))
            } else {
                std::path::Path::new(arg)
            };
            session.save(path)?;
            println!("  Saved to {}", path.display());
        }
        "/load" => {
            let path = if arg.is_empty() {
                session_path
                    .as_deref()
                    .unwrap_or(std::path::Path::new("session.json"))
            } else {
                std::path::Path::new(arg)
            };
            *session = Session::load(path)?;
            println!(
                "  Loaded: revision {}, {} mutations",
                session.revision,
                session.mutations.len()
            );
        }
        "/history" | "/hist" | "/h" => {
            let n = arg.parse::<usize>().unwrap_or(20);
            print!("{}", session.format_recent_history(n));
        }
        "/rewind" => {
            let rev: usize = arg
                .parse()
                .map_err(|_| anyhow::anyhow!("usage: /rewind <revision>"))?;
            session.rewind_to(rev)?;
            println!("  Rewound to revision {rev}");
        }
        "/status" | "/s" => {
            println!("  Revision: {}", session.revision);
            println!("  Mutations: {}", session.mutations.len());
            println!("  History entries: {}", session.history.len());
            println!("  {}", session.program);
            // Show which functions are compilable
            for func in &session.program.functions {
                let compilable = crate::compiler::is_compilable_function(func);
                let tag = if compilable { "compilable" } else { "interpreted" };
                println!("    {} — {tag}", func.name);
            }
        }
        "/compile" | "/c" => {
            if crate::compiler::is_fully_compilable(&session.program) {
                match crate::compiler::compile(&session.program) {
                    Ok(_) => {
                        println!("  Compiled {} function(s) to native code",
                            session.program.functions.len());
                    }
                    Err(e) => eprintln!("  Compile error: {e}"),
                }
            } else {
                println!("  Not fully compilable. Compilable functions:");
                for func in &session.program.functions {
                    let ok = crate::compiler::is_compilable_function(func);
                    println!("    {} — {}", func.name, if ok { "OK" } else { "needs interpreter" });
                }
            }
        }
        "/ask" => {
            if arg.is_empty() {
                println!("  Usage: /ask <question or task>");
                return Ok(());
            }
            // Send to LLM with program context
            let context = format!(
                "{}\n\n{}\n\nUser request: {arg}",
                validator::program_summary(&session.program),
                session.format_recent_history(10)
            );
            llm_messages.push(ChatMessage::user(context));

            println!("  [asking LLM...]");
            let output = llm.generate(llm_messages.clone()).await?;
            println!();

            llm_messages.push(ChatMessage::assistant(&output.text));

            // If the LLM generated code, apply it
            let code = if output.code.is_empty() {
                // Try to extract forge code from the response
                let mut lines = Vec::new();
                for line in output.text.lines() {
                    let t = line.trim();
                    if t.starts_with('+') || t.starts_with('!') || t.starts_with('?') || t == "end"
                    {
                        lines.push(line.to_string());
                    }
                }
                lines.join("\n")
            } else {
                output.code
            };

            if !code.is_empty() && code.trim() != "DONE" {
                println!("  [applying generated code...]");
                process_input(&code, session).await;
            }
        }
        "/help" => {
            println!("  Natural language is sent to the AI automatically.");
            println!("  The AI generates Forge code, errors auto-retry.");
            println!();
            println!("  Forge code (prefix with +, !, or ?):");
            println!("    +fn, +type, +let, ...  — mutations");
            println!("    !test func             — run tests");
            println!("    !eval func args        — evaluate a function");
            println!("    !trace func args       — trace execution");
            println!("    ?symbols               — list types and functions");
            println!();
            println!("  Commands:");
            println!("    /status                — show program state");
            println!("    /history [N]           — show recent history");
            println!("    /rewind <rev>          — rewind to revision");
            println!("    /save [path]           — save session");
            println!("    /load [path]           — load session");
            println!("    /quit                  — exit");
        }
        _ => {
            println!("  Unknown command: {cmd} (type /help for help)");
        }
    }

    Ok(())
}
