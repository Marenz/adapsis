use anyhow::{Context, Result, ensure};
use crate::coroutine::IoRequest;
use std::sync::Arc;
use tokio::sync::{Semaphore, mpsc, oneshot};

static MUSIC_SLOT: std::sync::OnceLock<Arc<Semaphore>> = std::sync::OnceLock::new();

pub(super) fn start(context: String, caption: String, duration: i64, lyrics: String, sender: mpsc::Sender<IoRequest>) -> Result<String> {
    let permit = MUSIC_SLOT.get_or_init(|| Arc::new(Semaphore::new(1))).clone().try_acquire_owned()
        .context("Music generation is busy. Please try again after the current song finishes")?;
    tokio::spawn(async move {
        let _permit = permit;
        let result = generate(&caption, duration, &lyrics).await;
        let (message, attachment) = match result {
            Ok(audio) => ("Your generated song is ready.".to_string(), Some(audio)),
            Err(error) => {
                eprintln!("[music:{context}] generation failed: {error:#}");
                ("Music generation failed; no song was produced. Please try again later.".into(), None)
            }
        };
        let (tx, rx) = oneshot::channel();
        if sender.send(IoRequest::ConversationNotify { context: context.clone(), message, attachment, reply: tx }).await.is_ok() {
            match rx.await {
                Ok(Ok(_)) => eprintln!("[music:{context}] notification dispatched"),
                error => eprintln!("[music:{context}] notification failed: {error:?}"),
            }
        }
    });
    Ok("Music generation started in the background. The result or an error will be sent to THIS chat; do not start a duplicate job. Delivery is not yet confirmed.".into())
}

async fn generate(caption: &str, duration: i64, lyrics: &str) -> Result<crate::attachment::Attachment> {
    let base = std::env::var("ADAPSIS_MUSIC_URL").unwrap_or_else(|_| "http://127.0.0.1:8092".into());
    let client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(600)).build()?;
    let mut response = client.post(format!("{}/generate", base.trim_end_matches('/')))
        .json(&serde_json::json!({"caption": caption, "duration_s": duration, "lyrics": lyrics,
            "metas": "", "output": format!("adapsis-{}.mp3", uuid::Uuid::new_v4())}))
        .send().await?.error_for_status()?;
    let mime = response.headers().get(reqwest::header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    ensure!(mime.starts_with("audio/"), "music backend returned {mime}, expected audio");
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        ensure!(bytes.len() + chunk.len() <= 30_000_000, "music response exceeds 30 MB");
        bytes.extend_from_slice(&chunk);
    }
    ensure!(!bytes.is_empty(), "music backend returned empty audio");
    Ok(crate::attachment::Attachment::from_bytes(bytes, mime, "song.mp3")?)
}
