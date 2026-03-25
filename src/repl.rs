//! Interactive REPL — thin CLI client to AdapsisOS API.
//!
//! All logic runs server-side via /api/ask-stream.
//! The REPL just reads input, sends it, and displays streamed results.

use std::io::{self, BufRead, Write};

use anyhow::Result;

/// Run the REPL as a client to a running AdapsisOS instance.
pub async fn run_repl(api_url: &str) -> Result<()> {
    let client = reqwest::Client::new();

    // Check connection
    match client.get(format!("{api_url}/api/status")).send().await {
        Ok(resp) if resp.status().is_success() => {
            let status: serde_json::Value = resp.json().await?;
            let rev = status.get("revision").and_then(|r| r.as_u64()).unwrap_or(0);
            let fns = status.get("functions").and_then(|f| f.as_array()).map(|a| a.len()).unwrap_or(0);
            println!("Connected to AdapsisOS at {api_url}");
            println!("  revision: {rev}, functions: {fns}");
        }
        _ => {
            println!("Cannot connect to AdapsisOS at {api_url}");
            println!("Start it with: forge os --session project.json -p PORT");
            return Ok(());
        }
    }

    println!("Type naturally or use Forge code (+fn, !eval, ?symbols). /quit to exit.\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        // Get current revision for prompt
        let rev = client.get(format!("{api_url}/api/status"))
            .send().await.ok()
            .and_then(|r| futures::executor::block_on(r.json::<serde_json::Value>()).ok())
            .and_then(|v| v.get("revision").and_then(|r| r.as_u64()))
            .unwrap_or(0);

        print!("forge[{rev}]> ");
        stdout.flush()?;

        let mut input = String::new();
        let mut reader = stdin.lock();
        if reader.read_line(&mut input)? == 0 {
            break; // EOF
        }
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        // Local commands
        match input {
            "/quit" | "/exit" | "/q" => {
                println!("Goodbye.");
                break;
            }
            "/status" | "/s" => {
                match client.get(format!("{api_url}/api/status")).send().await {
                    Ok(resp) => {
                        let s: serde_json::Value = resp.json().await?;
                        println!("  Revision: {}", s.get("revision").unwrap_or(&serde_json::json!(0)));
                        if let Some(fns) = s.get("functions").and_then(|f| f.as_array()) {
                            println!("  Functions ({}): {}", fns.len(),
                                fns.iter().filter_map(|f| f.as_str()).collect::<Vec<_>>().join(", "));
                        }
                        if let Some(plan) = s.get("plan").and_then(|p| p.as_array()) {
                            if !plan.is_empty() {
                                println!("  Plan:");
                                for (i, step) in plan.iter().enumerate() {
                                    let status = step.get("status").and_then(|s| s.as_str()).unwrap_or("?");
                                    let desc = step.get("description").and_then(|d| d.as_str()).unwrap_or("?");
                                    let icon = match status { "done" => "✅", "failed" => "❌", "in_progress" => "🔄", _ => "⬜" };
                                    println!("    {icon} {}: {desc}", i + 1);
                                }
                            }
                        }
                    }
                    Err(e) => eprintln!("  Error: {e}"),
                }
                continue;
            }
            "/help" | "/h" => {
                println!("  Just type naturally — the AI handles everything.");
                println!("  Forge code: +fn, +type, !eval, !test, ?symbols, ?source");
                println!("  /status — show program state");
                println!("  /quit — exit");
                continue;
            }
            _ => {}
        }

        // Check if it's direct Forge code (starts with + ! ?)
        let is_forge = input.starts_with('+') || input.starts_with('!') || input.starts_with('?');

        if is_forge {
            // For direct Forge code, use /api/mutate or specific endpoints
            if input.starts_with("!eval ") {
                let rest = &input[6..].trim();
                // Try parsing as inline expression first
                let is_inline = if let Ok(parsed) = crate::parser::parse_expr_pub(0, rest) {
                    !matches!(parsed, crate::parser::Expr::Ident(_))
                } else {
                    false
                };
                let req_json = if is_inline {
                    serde_json::json!({"function": "", "expression": rest})
                } else {
                    let (func, args) = rest.split_once(' ').unwrap_or((rest, ""));
                    serde_json::json!({"function": func, "input": args})
                };
                match client.post(format!("{api_url}/api/eval"))
                    .json(&req_json)
                    .send().await
                {
                    Ok(resp) => {
                        let d: serde_json::Value = resp.json().await?;
                        let result = d.get("result").and_then(|r| r.as_str()).unwrap_or("(none)");
                        let ok = d.get("success").and_then(|s| s.as_bool()).unwrap_or(false);
                        if ok { println!("  = {result}"); } else { println!("  Error: {result}"); }
                    }
                    Err(e) => eprintln!("  Error: {e}"),
                }
            } else if input.starts_with('?') {
                match client.post(format!("{api_url}/api/query"))
                    .json(&serde_json::json!({"query": input}))
                    .send().await
                {
                    Ok(resp) => {
                        let d: serde_json::Value = resp.json().await?;
                        println!("  {}", d.get("response").and_then(|r| r.as_str()).unwrap_or("(none)"));
                    }
                    Err(e) => eprintln!("  Error: {e}"),
                }
            } else {
                // Mutations and other commands → /api/mutate
                match client.post(format!("{api_url}/api/mutate"))
                    .json(&serde_json::json!({"source": input}))
                    .send().await
                {
                    Ok(resp) => {
                        let d: serde_json::Value = resp.json().await?;
                        if let Some(results) = d.get("results").and_then(|r| r.as_array()) {
                            for r in results {
                                let ok = r.get("success").and_then(|s| s.as_bool()).unwrap_or(false);
                                let msg = r.get("message").and_then(|m| m.as_str()).unwrap_or("");
                                if ok { println!("  OK: {msg}"); } else { eprintln!("  ERR: {msg}"); }
                            }
                        }
                    }
                    Err(e) => eprintln!("  Error: {e}"),
                }
            }
        } else {
            // Natural language → /api/ask-stream (SSE)
            match client.post(format!("{api_url}/api/ask-stream"))
                .json(&serde_json::json!({"message": input}))
                .send().await
            {
                Ok(resp) => {
                    let mut stream = resp.bytes_stream();
                    let mut raw_buf: Vec<u8> = Vec::new();
                    let mut buffer = String::new();

                    use futures::StreamExt;
                    while let Some(chunk) = stream.next().await {
                        match chunk {
                            Ok(bytes) => {
                                // Buffer raw bytes so multi-byte UTF-8 chars
                                // split across chunks are not corrupted.
                                raw_buf.extend_from_slice(&bytes);
                                let valid_up_to = match std::str::from_utf8(&raw_buf) {
                                    Ok(_) => raw_buf.len(),
                                    Err(e) => e.valid_up_to(),
                                };
                                if valid_up_to > 0 {
                                    buffer.push_str(std::str::from_utf8(&raw_buf[..valid_up_to]).unwrap());
                                    raw_buf.drain(..valid_up_to);
                                }
                                while let Some(newline) = buffer.find('\n') {
                                    let line = buffer[..newline].to_string();
                                    buffer = buffer[newline + 1..].to_string();

                                    if !line.starts_with("data: ") { continue; }
                                    let data = &line[6..];
                                    if data.is_empty() { continue; }

                                    if let Ok(event) = serde_json::from_str::<serde_json::Value>(data) {
                                        match event.get("type").and_then(|t| t.as_str()) {
                                            Some("iteration") => {
                                                let n = event.get("n").and_then(|n| n.as_u64()).unwrap_or(0);
                                                if n > 1 { println!("\n  ⟳ Iteration {n}"); }
                                            }
                                            Some("thinking") => {
                                                let text = event.get("text").and_then(|t| t.as_str()).unwrap_or("");
                                                if !text.is_empty() {
                                                    println!("  \x1b[2m{}\x1b[0m", &text[..text.len().min(200)]);
                                                }
                                            }
                                            Some("text") => {
                                                let text = event.get("text").and_then(|t| t.as_str()).unwrap_or("");
                                                println!("  {text}");
                                            }
                                            Some("code") => {
                                                let code = event.get("code").and_then(|c| c.as_str()).unwrap_or("");
                                                println!("\x1b[36m{code}\x1b[0m");
                                            }
                                            Some("result") => {
                                                let msg = event.get("message").and_then(|m| m.as_str()).unwrap_or("");
                                                let ok = event.get("success").and_then(|s| s.as_bool()).unwrap_or(false);
                                                if ok { println!("\x1b[32m  OK: {msg}\x1b[0m"); }
                                                else { println!("\x1b[31m  ERR: {msg}\x1b[0m"); }
                                            }
                                            Some("test") => {
                                                let msg = event.get("message").and_then(|m| m.as_str()).unwrap_or("");
                                                let pass = event.get("pass").and_then(|p| p.as_bool()).unwrap_or(false);
                                                if pass { println!("\x1b[32m  PASS: {msg}\x1b[0m"); }
                                                else { println!("\x1b[31m  FAIL: {msg}\x1b[0m"); }
                                            }
                                            Some("eval") => {
                                                let result = event.get("result").and_then(|r| r.as_str()).unwrap_or("");
                                                println!("  = {result}");
                                            }
                                            Some("query") => {
                                                let query = event.get("query").and_then(|q| q.as_str()).unwrap_or("");
                                                let response = event.get("response").and_then(|r| r.as_str()).unwrap_or("");
                                                println!("  \x1b[2m{query}:\n{response}\x1b[0m");
                                            }
                                            Some("feedback") => {
                                                let msg = event.get("message").and_then(|m| m.as_str()).unwrap_or("");
                                                println!("  \x1b[33m{msg}\x1b[0m");
                                            }
                                            Some("error") => {
                                                let msg = event.get("message").and_then(|m| m.as_str()).unwrap_or("");
                                                eprintln!("  \x1b[31mError: {msg}\x1b[0m");
                                            }
                                            Some("done") => {
                                                println!("  \x1b[32m✓ Done\x1b[0m");
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("  Stream error: {e}");
                                break;
                            }
                        }
                    }
                }
                Err(e) => eprintln!("  Error: {e}"),
            }
        }

        println!();
    }

    Ok(())
}
