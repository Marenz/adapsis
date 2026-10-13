//! Background CAD preparation. Generated Python runs only in the runner's
//! no-network filesystem sandbox; this capability never starts a printer.
use anyhow::{Context, Result, ensure};
use base64::Engine;
use crate::{attachment::Attachment, coroutine::IoRequest};
use std::{process::Stdio, sync::{Arc, OnceLock}};
use tokio::{io::AsyncWriteExt, sync::{Semaphore, mpsc, oneshot}};

static SLOT: OnceLock<Arc<Semaphore>> = OnceLock::new();
const MAX_REFERENCE_IMAGES: usize = 3;
const MAX_REFERENCE_BYTES: usize = 20 * 1024 * 1024;

/// Keep the tail of `s` (the end holds a traceback's final exception line), so a
/// long failure detail never buries the actionable exception behind the head.
fn tail_chars(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max { s.to_string() } else { chars[chars.len() - max..].iter().collect() }
}

/// The guest supplies a *brief*, not source, and cannot fix a gateway outage or an
/// Astra timeout — only a real model-script failure carries the "correct the model
/// source" wording.
fn failure_hint(detail: &str) -> &'static str {
    if detail.contains("Model script failed") || detail.contains("Model script exceeded") {
        "Correct the model source and retry."
    } else if detail.contains("CAD author (Astra)") {
        "The CAD authoring service did not produce a design; retry the design."
    } else if detail.contains("Slicing failed") || detail.contains("Slicer reported")
        || detail.contains("No successful sliced plates") {
        "Slicing failed; an operator can inspect the job logs."
    } else {
        "Retry the design; if it persists, an operator can inspect the job logs."
    }
}

pub(super) fn start(context: String, title: String, source: String, material: String,
                    images: Vec<Attachment>, sender: mpsc::Sender<IoRequest>) -> Result<String> {
    let permit = SLOT.get_or_init(|| Arc::new(Semaphore::new(1))).clone()
        .try_acquire_owned().context("CAD preparation is busy; wait for the current result")?;
    let references = encode_references(images);
    let reference_count = references.len();
    tokio::spawn(async move {
        let _permit = permit;
        match prepare(&title, &source, &material, references).await {
            Ok(report) => {
                let owner = crate::memory_graph::admin_principal().strip_prefix("telegram:user:")
                    .map(|id| format!("telegram:{id}"));
                let references = report["reference_images"].as_u64().unwrap_or(0);
                let reference_note = if references == 0 {
                    "No reference images were sent off-host to the CAD author.".to_string()
                } else {
                    format!("The external cloud CAD author chatgpt/gpt-6-astra received {references} reference image(s) from this conversation.")
                };
                let message = format!("Prepared design: {title}\n{}\n{reference_note}\nModel checks passed. {} minutes / {} g ({material}), across {} plate(s). Physical fit/anatomical accuracy are not certified. No print was started; Marenz must approve the actual print.",
                    report["url"].as_str().unwrap_or(""), report["minutes"], report["grams"], report["plates"]);
                notify(&sender, &context, message.clone()).await;
                if let Some(owner) = owner.filter(|c| c != &context) {
                    notify(&sender, &owner, format!("Print approval requested after design preparation from {context}.\n{message}\nJob folder: {}\nReview the model/slicer output and confirm filament and clear bed before printing. This is not authorization to start automatically.", report["job_dir"].as_str().unwrap_or(""))).await;
                }
            }
            Err(error) => {
                let detail = format!("{error:#}");
                eprintln!("[cad:{context}] preparation failed: {}", tail_chars(&detail, 4000));
                let message = format!("CAD preparation failed; no print was started. {} Details: {}",
                    failure_hint(&detail), tail_chars(&detail, 2500));
                notify(&sender, &context, message).await;
            }
        }
    });
    Ok(format!("CAD preparation started in the background. The design brief is being sent off-host to the external cloud CAD author chatgpt/gpt-6-astra; {reference_count} explicitly approved reference image(s) will also be sent. A checked preview/files or failure will arrive in THIS chat. The owner receives a print-approval request only after preparation succeeds. No printing is authorized or started. Do not submit duplicate jobs."))
}

fn encode_references(images: Vec<Attachment>) -> Vec<serde_json::Value> {
    let mut references = Vec::new();
    let mut remaining = MAX_REFERENCE_BYTES;
    for image in images {
        if references.len() == MAX_REFERENCE_IMAGES { break; }
        let Some(mime) = image.image_mime().map(str::to_string) else { continue };
        let Ok(bytes) = image.bytes() else { continue };
        if bytes.is_empty() || bytes.len() > remaining { continue; }
        remaining -= bytes.len();
        references.push(serde_json::json!({
            "mime": mime,
            "data_base64": base64::engine::general_purpose::STANDARD.encode(bytes),
        }));
    }
    references
}

async fn notify(sender: &mpsc::Sender<IoRequest>, context: &str, message: String) {
    let (tx,rx) = oneshot::channel();
    if sender.send(IoRequest::ConversationNotify { context: context.into(), message,
        attachment: None, reply: tx }).await.is_ok() {
        if !matches!(rx.await, Ok(Ok(_))) { eprintln!("[cad:{context}] notification failed"); }
    }
}

async fn prepare(title: &str, source: &str, material: &str,
                 references: Vec<serde_json::Value>) -> Result<serde_json::Value> {
    let home = std::env::var("HOME")?;
    let runner = std::env::var("ADAPSIS_CAD_RUNNER")
        .unwrap_or_else(|_| format!("{home}/Projects/adapsis/tools/cad_prepare.py"));
    let python = std::env::var("ADAPSIS_CAD_PYTHON")
        .unwrap_or_else(|_| format!("{home}/Projects/piercing-gauge/.venv/bin/python"));
    let mut child = tokio::process::Command::new(python).arg(runner)
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .kill_on_drop(true).spawn().context("launch configured CAD preparation runner")?;
    let input = serde_json::to_vec(&serde_json::json!({
        "title":title,"brief":source,"material":material,"references":references
    }))?;
    let mut stdin = child.stdin.take().context("CAD runner stdin")?;
    stdin.write_all(&input).await?; drop(stdin);
    let output = tokio::time::timeout(std::time::Duration::from_secs(2700), child.wait_with_output())
        .await.context("CAD preparation exceeded 45 minutes")??;
    ensure!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).context("CAD runner report")?;
    ensure!(report["passed"] == true && report["url"].is_string(), "CAD runner did not return checked files");
    ensure!(report["reference_images"].as_u64().is_some_and(|count| count <= MAX_REFERENCE_IMAGES as u64),
        "CAD runner did not report a valid reference image count");
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_chars_keeps_tail_and_does_not_panic_at_boundary() {
        assert_eq!(tail_chars("short", 100), "short");
        assert_eq!(tail_chars("abcdef", 6), "abcdef");
        assert_eq!(tail_chars("abcdef", 3), "def");
        assert_eq!(tail_chars("abcdef", 0), "");
        let detail = "Model script failed: Traceback (most recent call last):\n...\nAttributeError: boom";
        assert!(tail_chars(detail, 40).ends_with("AttributeError: boom"));
    }

    #[test]
    fn failure_hint_selects_each_branch() {
        assert_eq!(failure_hint("Model script failed: Traceback..."),
            "Correct the model source and retry.");
        assert_eq!(failure_hint("Model script exceeded the 300s sandbox limit without finishing"),
            "Correct the model source and retry.");
        assert_eq!(failure_hint("CAD author (Astra) did not respond within 600s after 2 attempts"),
            "The CAD authoring service did not produce a design; retry the design.");
        assert_eq!(failure_hint("Slicing failed; administrator can inspect slicer.log"),
            "Slicing failed; an operator can inspect the job logs.");
        assert_eq!(failure_hint("some other error"),
            "Retry the design; if it persists, an operator can inspect the job logs.");
    }
}
