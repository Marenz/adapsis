//! Turn-bound, non-generative stabilization of the latest authorized video.
//! The caller supplies no path or destination: Ladybug selects the source after
//! ACL filtering, and delivery is fixed to the current conversation.

use crate::{
    attachment::{Attachment, AttachmentData},
    coroutine::IoRequest,
};
use anyhow::{Context, Result, bail, ensure};
use std::{
    path::{Path, PathBuf},
    process::Stdio,
    sync::{Arc, OnceLock},
    time::Duration,
};
use tokio::sync::{Semaphore, mpsc, oneshot};

const MAX_INPUT_BYTES: u64 = 20 * 1024 * 1024;
// Keep muxing overhead and Telegram's exact 50 MiB Bot API ceiling out of the
// delivery path. The post-encode check remains authoritative if ffmpeg exceeds
// `-fs` while closing the MP4 container.
const MAX_OUTPUT_BYTES: u64 = 49 * 1024 * 1024;
const MAX_DURATION_SECONDS: f64 = 120.0;
const MAX_WALL_SECONDS: u64 = 15 * 60;
const TOOL_SECONDS: u64 = 10 * 60;
const AUDIO_BITRATE_KBPS: u64 = 192;
const MIN_VIDEO_BITRATE_KBPS: u64 = 500;
const MAX_VIDEO_BITRATE_KBPS: u64 = 8_000;

static SLOT: OnceLock<Arc<Semaphore>> = OnceLock::new();

#[derive(Debug, Clone, PartialEq)]
struct Probe {
    duration_seconds: f64,
    width: u64,
    height: u64,
    has_audio: bool,
    video_codec: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VideoEncoder {
    OpenH264,
    Mpeg4,
}

impl VideoEncoder {
    fn ffmpeg_name(self) -> &'static str {
        match self {
            Self::OpenH264 => "libopenh264",
            Self::Mpeg4 => "mpeg4",
        }
    }
}

struct JobDir(PathBuf);

impl JobDir {
    fn create() -> Result<Self> {
        Self::create_in(&std::env::temp_dir())
    }

    fn create_in(parent: &Path) -> Result<Self> {
        let path = parent.join(format!("adapsis-video-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).context("create isolated video job directory")?;
        Ok(Self(path))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for JobDir {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.0)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            eprintln!(
                "[video] failed to remove job directory {}: {error}",
                self.0.display()
            );
        }
    }
}

pub(super) fn start(
    principal: String,
    context: String,
    graph: Arc<crate::memory_graph::MemoryGraph>,
    sender: mpsc::Sender<IoRequest>,
) -> Result<String> {
    let permit = SLOT
        .get_or_init(|| Arc::new(Semaphore::new(1)))
        .clone()
        .try_acquire_owned()
        .context("Video stabilization is busy; wait for the current result")?;
    tokio::spawn(async move {
        let _permit = permit;
        let result = tokio::time::timeout(
            Duration::from_secs(MAX_WALL_SECONDS),
            stabilize_latest(graph, &principal, &context),
        )
        .await;
        match result {
            Ok(Ok((attachment, summary))) => {
                if let Err(error) = deliver_stabilized(&sender, &context, summary, attachment).await {
                    eprintln!("[video:{context}] stabilized output delivery failed: {error:#}");
                }
            }
            Ok(Err(error)) => {
                eprintln!("[video:{context}] stabilization failed: {error:#}");
                let _ = notify(
                    &sender,
                    &context,
                    format!("Video stabilization failed; the original was not changed. {error:#}"),
                    None,
                )
                .await;
            }
            Err(_) => {
                eprintln!("[video:{context}] stabilization exceeded {MAX_WALL_SECONDS}s");
                let _ = notify(&sender, &context,
                    "Video stabilization stopped after the 15-minute processing limit. The original was not changed; try a shorter or lower-resolution clip.".into(), None).await;
            }
        }
    });
    Ok("Video stabilization started in the background using the latest authorized video in THIS chat. The stabilized original or an actionable failure will arrive here; no generative frame replacement is used. Do not submit a duplicate job.".into())
}

async fn stabilize_latest(
    graph: Arc<crate::memory_graph::MemoryGraph>,
    principal: &str,
    context: &str,
) -> Result<(Attachment, String)> {
    let graph_for_read = graph.clone();
    let principal = principal.to_string();
    let context_for_read = context.to_string();
    let source = tokio::task::spawn_blocking(move ||
        graph_for_read.latest_authorized_video(&principal, &context_for_read)
    ).await.context("join authorized video lookup")??
        .context("No authorized video from the last 7 days was found in this conversation. Send a video or a document with a video MIME type, then ask again")?;
    let (source, source_created_at_ms) = source;

    let source_size = source.len() as u64;
    ensure!(source_size > 0, "The latest video is empty; send it again");
    ensure!(
        source_size <= MAX_INPUT_BYTES,
        "The latest video is {:.1} MB; Telegram bots can download at most 20 MB",
        source_size as f64 / 1_048_576.0
    );
    ensure!(
        source.video_mime().is_some(),
        "The latest attachment is not a supported video container (MP4/MOV/WebM/MKV/AVI)"
    );

    let job = JobDir::create()?;
    let input = job.path().join("input-video");
    write_attachment(&source, &input).await?;
    let output = job.path().join("stabilized.mp4");
    let input_probe = probe_video(&input).await.context("inspect source video")?;
    validate_input(&input_probe, source_size)?;
    stabilize_file(
        &input,
        &output,
        job.path(),
        input_probe.duration_seconds,
    )
    .await?;
    let output_size = tokio::fs::metadata(&output)
        .await
        .context("read stabilized output metadata")?
        .len();
    ensure!(output_size > 0, "ffmpeg produced an empty stabilized video");
    validate_output_size(output_size)?;
    let output_probe = probe_video(&output)
        .await
        .context("verify stabilized output")?;
    ensure!(
        output_probe.duration_seconds >= input_probe.duration_seconds * 0.95,
        "Stabilized output is truncated ({:.1}s of {:.1}s)",
        output_probe.duration_seconds,
        input_probe.duration_seconds
    );
    ensure!(
        !input_probe.has_audio || output_probe.has_audio,
        "Stabilized output lost the source audio stream"
    );

    let attachment = persist_output(&output).await?;
    let audio = if input_probe.has_audio {
        "Audio was retained and re-encoded as AAC for MP4 compatibility."
    } else {
        "The source had no audio stream, so the output has none."
    };
    let summary = format!(
        "Stabilized video ready. Source: {} (received {}, timestamp {} ms). Resolution: {}x{} -> {}x{}. Crop/zoom: libvidstab automatic optimal zoom with black-border fallback. Video codec: {} -> {}. {} No generative frame replacement was used.",
        source.name,
        source_age(source_created_at_ms),
        source_created_at_ms,
        input_probe.width,
        input_probe.height,
        output_probe.width,
        output_probe.height,
        input_probe.video_codec,
        output_probe.video_codec,
        audio,
    );
    Ok((attachment, summary))
}

fn source_age(created_at_ms: i64) -> String {
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64;
    let seconds = now_ms.saturating_sub(created_at_ms).max(0) / 1000;
    match seconds {
        0..=59 => format!("{seconds} seconds ago"),
        60..=3599 => format!("{} minutes ago", seconds / 60),
        3600..=86_399 => format!("{} hours ago", seconds / 3600),
        _ => format!("{} days ago", seconds / 86_400),
    }
}

async fn write_attachment(source: &Attachment, destination: &Path) -> Result<()> {
    match &source.data {
        AttachmentData::Memory(bytes) => tokio::fs::write(destination, bytes.as_slice()).await?,
        AttachmentData::File(path) => {
            tokio::fs::copy(path, destination)
                .await
                .with_context(|| format!("copy source video from {}", path.display()))?;
        }
    }
    Ok(())
}

fn configured_binary(variable: &str, default: &str) -> Result<PathBuf> {
    let path = PathBuf::from(std::env::var(variable).unwrap_or_else(|_| default.into()));
    ensure!(
        path.is_absolute(),
        "{variable} must be an absolute executable path"
    );
    Ok(path)
}

async fn probe_video(path: &Path) -> Result<Probe> {
    let binary = configured_binary("ADAPSIS_FFPROBE", "/usr/bin/ffprobe")?;
    let args = vec![
        "-v".into(),
        "error".into(),
        "-show_entries".into(),
        "format=duration:stream=codec_type,codec_name,width,height".into(),
        "-of".into(),
        "json".into(),
        path.display().to_string(),
    ];
    let output = run_tool(&binary, &args).await.context("run ffprobe")?;
    parse_probe(&output.stdout)
}

fn parse_probe(stdout: &[u8]) -> Result<Probe> {
    let document: serde_json::Value =
        serde_json::from_slice(stdout).context("ffprobe returned invalid JSON")?;
    let streams = document["streams"]
        .as_array()
        .context("ffprobe JSON has no streams array")?;
    let video = streams
        .iter()
        .find(|stream| stream["codec_type"] == "video")
        .context("input contains no video stream")?;
    let duration = document["format"]["duration"]
        .as_str()
        .and_then(|value| value.parse::<f64>().ok())
        .or_else(|| document["format"]["duration"].as_f64())
        .context("ffprobe did not report a numeric duration")?;
    ensure!(
        duration.is_finite() && duration > 0.0,
        "video duration must be positive and finite"
    );
    let width = video["width"]
        .as_u64()
        .context("ffprobe did not report video width")?;
    let height = video["height"]
        .as_u64()
        .context("ffprobe did not report video height")?;
    ensure!(width > 0 && height > 0, "video dimensions must be positive");
    Ok(Probe {
        duration_seconds: duration,
        width,
        height,
        has_audio: streams.iter().any(|stream| stream["codec_type"] == "audio"),
        video_codec: video["codec_name"]
            .as_str()
            .unwrap_or("unknown")
            .to_string(),
    })
}

fn validate_input(probe: &Probe, size: u64) -> Result<()> {
    ensure!(
        size <= MAX_INPUT_BYTES,
        "video exceeds Telegram's 20 MB bot-download limit ({:.1} MB)",
        size as f64 / 1_048_576.0
    );
    ensure!(
        probe.duration_seconds <= MAX_DURATION_SECONDS,
        "video is {:.1} seconds long; the stabilization limit is 120 seconds",
        probe.duration_seconds
    );
    Ok(())
}

fn validate_output_size(size: u64) -> Result<()> {
    ensure!(
        size <= MAX_OUTPUT_BYTES,
        "Stabilized output is {:.1} MB; Telegram bot delivery is capped at 50 MB, so Adapsis keeps outputs at or below 49 MB. Try a shorter or lower-resolution clip",
        size as f64 / 1_048_576.0
    );
    Ok(())
}

fn detect_args(input: &Path, transforms: &Path) -> Vec<String> {
    vec![
        "-nostdin".into(),
        "-hide_banner".into(),
        "-loglevel".into(),
        "error".into(),
        "-y".into(),
        "-threads".into(),
        "2".into(),
        "-i".into(),
        input.display().to_string(),
        "-map".into(),
        "0:v:0".into(),
        "-an".into(),
        "-vf".into(),
        format!(
            "vidstabdetect=shakiness=5:accuracy=15:result={}",
            transforms.display()
        ),
        "-f".into(),
        "null".into(),
        "/dev/null".into(),
    ]
}

fn encoder_probe_args() -> Vec<String> {
    vec!["-hide_banner".into(), "-encoders".into()]
}

fn encoder_from_listing(stdout: &[u8]) -> VideoEncoder {
    if String::from_utf8_lossy(stdout).lines().any(|line| {
        line.split_whitespace().nth(1) == Some(VideoEncoder::OpenH264.ffmpeg_name())
    }) {
        VideoEncoder::OpenH264
    } else {
        VideoEncoder::Mpeg4
    }
}

async fn select_video_encoder(binary: &Path) -> VideoEncoder {
    match run_tool(binary, &encoder_probe_args()).await {
        Ok(output) => encoder_from_listing(&output.stdout),
        Err(error) => {
            eprintln!("[video] ffmpeg encoder probe failed; falling back to mpeg4: {error:#}");
            VideoEncoder::Mpeg4
        }
    }
}

fn video_bitrate_kbps(duration_seconds: f64) -> u64 {
    // Reserve 10% for MP4/container variance plus the AAC stream. `-fs` and the
    // post-encode byte check remain the hard cap if an encoder overshoots.
    let usable_total_kbps = (MAX_OUTPUT_BYTES as f64 * 8.0 * 0.90
        / duration_seconds.max(1.0)
        / 1000.0) as u64;
    usable_total_kbps
        .saturating_sub(AUDIO_BITRATE_KBPS)
        .clamp(MIN_VIDEO_BITRATE_KBPS, MAX_VIDEO_BITRATE_KBPS)
}

fn transform_args(
    input: &Path,
    transforms: &Path,
    output: &Path,
    encoder: VideoEncoder,
    duration_seconds: f64,
) -> Vec<String> {
    let bitrate = video_bitrate_kbps(duration_seconds);
    let mut args = vec![
        "-nostdin".into(),
        "-hide_banner".into(),
        "-loglevel".into(),
        "error".into(),
        "-y".into(),
        "-threads".into(),
        "2".into(),
        "-i".into(),
        input.display().to_string(),
        "-map".into(),
        "0:v:0".into(),
        "-map".into(),
        "0:a?".into(),
        "-vf".into(),
        format!(
            "vidstabtransform=input={}:smoothing=30:optzoom=1:crop=black",
            transforms.display()
        ),
        "-c:v".into(),
        encoder.ffmpeg_name().into(),
        "-b:v".into(),
        format!("{bitrate}k"),
    ];
    if encoder == VideoEncoder::OpenH264 {
        args.extend([
            "-maxrate".into(),
            format!("{bitrate}k"),
            "-bufsize".into(),
            format!("{}k", bitrate * 2),
        ]);
    }
    args.extend([
        "-pix_fmt".into(),
        "yuv420p".into(),
        "-c:a".into(),
        "aac".into(),
        "-b:a".into(),
        "192k".into(),
        "-movflags".into(),
        "+faststart".into(),
        "-fs".into(),
        MAX_OUTPUT_BYTES.to_string(),
        output.display().to_string(),
    ]);
    args
}

async fn stabilize_file(
    input: &Path,
    output: &Path,
    work_dir: &Path,
    duration_seconds: f64,
) -> Result<()> {
    let binary = configured_binary("ADAPSIS_FFMPEG", "/usr/bin/ffmpeg")?;
    let transforms = work_dir.join("transforms.trf");
    let detect = detect_args(input, &transforms);
    run_tool(&binary, &detect)
        .await
        .context("ffmpeg stabilization analysis pass")?;
    ensure!(
        transforms.is_file(),
        "ffmpeg analysis produced no stabilization transforms"
    );
    let encoder = select_video_encoder(&binary).await;
    let transform = transform_args(input, &transforms, output, encoder, duration_seconds);
    run_tool(&binary, &transform)
        .await
        .context("ffmpeg stabilization transform pass")?;
    Ok(())
}

async fn run_tool(binary: &Path, args: &[String]) -> Result<std::process::Output> {
    let mut command = tokio::process::Command::new(binary);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs(TOOL_SECONDS), command.output())
        .await
        .with_context(|| format!("{} exceeded the 10-minute tool limit", binary.display()))?
        .with_context(|| format!("launch {}", binary.display()))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: String = stderr
            .chars()
            .rev()
            .take(3000)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        bail!(
            "{} exited with {}: {}",
            binary.display(),
            output.status,
            tail.trim()
        );
    }
    Ok(output)
}

async fn persist_output(source: &Path) -> Result<Attachment> {
    let directory = std::env::var_os("ADAPSIS_VIDEO_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            dirs::data_dir()
                .unwrap_or_else(std::env::temp_dir)
                .join("adapsis/video-output")
        });
    persist_output_to(source, &directory).await
}

async fn persist_output_to(source: &Path, directory: &Path) -> Result<Attachment> {
    tokio::fs::create_dir_all(&directory)
        .await
        .context("create durable video output directory")?;
    let name = format!("stabilized-{}.mp4", uuid::Uuid::new_v4());
    let path = directory.join(&name);
    if let Err(error) = tokio::fs::copy(source, &path).await {
        let _ = tokio::fs::remove_file(&path).await;
        return Err(error).context("persist stabilized video for delivery");
    }
    Ok(Attachment {
        data: AttachmentData::File(path),
        mime_type: "video/mp4".into(),
        name,
    })
}

async fn remove_persisted_output(attachment: &Attachment) {
    if let AttachmentData::File(path) = &attachment.data
        && let Err(error) = tokio::fs::remove_file(path).await
        && error.kind() != std::io::ErrorKind::NotFound
    {
        eprintln!(
            "[video] failed to remove delivered output {}: {error}",
            path.display()
        );
    }
}

async fn deliver_stabilized(
    sender: &mpsc::Sender<IoRequest>,
    context: &str,
    summary: String,
    attachment: Attachment,
) -> Result<()> {
    let delivery = notify(sender, context, summary, Some(attachment.clone())).await;
    if let Err(error) = &delivery {
        eprintln!("[video:{context}] attachment callback failed; trying text fallback: {error:#}");
        let fallback = "The stabilized video was created, but the file could not be delivered to this chat. The temporary output has been cleaned up; please try again with a shorter or lower-resolution clip.";
        if let Err(fallback_error) = notify(sender, context, fallback.into(), None).await {
            eprintln!("[video:{context}] text-only delivery fallback also failed: {fallback_error:#}");
        }
    }
    remove_persisted_output(&attachment).await;
    delivery
}

async fn notify(
    sender: &mpsc::Sender<IoRequest>,
    context: &str,
    message: String,
    attachment: Option<Attachment>,
) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    sender
        .send(IoRequest::ConversationNotify {
            context: context.into(),
            message,
            attachment,
            reply: tx,
        })
        .await
        .context("queue video notification")?;
    rx.await.context("video notification reply was cancelled")??;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_probe_and_enforces_limits() -> Result<()> {
        let probe = parse_probe(br#"{"streams":[{"codec_type":"video","codec_name":"h264","width":1920,"height":1080},{"codec_type":"audio","codec_name":"aac"}],"format":{"duration":"12.500000"}}"#)?;
        assert_eq!(
            probe,
            Probe {
                duration_seconds: 12.5,
                width: 1920,
                height: 1080,
                has_audio: true,
                video_codec: "h264".into(),
            }
        );
        assert!(validate_input(&probe, MAX_INPUT_BYTES).is_ok());
        assert!(
            validate_input(
                &Probe {
                    duration_seconds: 120.1,
                    ..probe.clone()
                },
                1
            )
            .unwrap_err()
            .to_string()
            .contains("120 seconds")
        );
        assert!(
            validate_input(&probe, MAX_INPUT_BYTES + 1)
                .unwrap_err()
                .to_string()
                .contains("20 MB")
        );
        assert!(parse_probe(br#"{"streams":[],"format":{"duration":"1"}}"#).is_err());
        assert!(validate_output_size(MAX_OUTPUT_BYTES).is_ok());
        let error = validate_output_size(MAX_OUTPUT_BYTES + 1)
            .unwrap_err()
            .to_string();
        assert!(error.contains("50 MB"), "{error}");
        assert!(error.contains("49 MB"), "{error}");
        Ok(())
    }

    #[test]
    fn ffmpeg_commands_are_argument_vectors_not_shell_programs() {
        let input = Path::new("/tmp/in;touch-pwned.mp4");
        let transforms = Path::new("/tmp/job/transforms.trf");
        let output = Path::new("/tmp/out.mp4");
        let detect = detect_args(input, transforms);
        let transform = transform_args(input, transforms, output, VideoEncoder::OpenH264, 120.0);
        assert!(detect.contains(&input.display().to_string()));
        assert!(detect.iter().any(|arg| arg.starts_with("vidstabdetect=")));
        assert!(
            transform
                .iter()
                .any(|arg| arg.starts_with("vidstabtransform="))
        );
        assert!(transform.windows(2).any(|pair| pair == ["-map", "0:a?"]));
        assert!(transform.windows(2).any(|pair| pair == ["-c:v", "libopenh264"]));
        assert!(transform.windows(2).any(|pair| pair[0] == "-b:v" && pair[1].ends_with('k')));
        assert!(transform.windows(2).any(|pair| {
            pair[0] == "-fs" && pair[1] == MAX_OUTPUT_BYTES.to_string()
        }));
        assert!(!transform.iter().any(|arg| arg == "-q:v"));
        assert!(!detect.iter().any(|arg| matches!(arg.as_str(), "sh" | "-c")));
        assert!(
            !transform
                .iter()
                .any(|arg| matches!(arg.as_str(), "sh" | "-c"))
        );
    }

    #[test]
    fn encoder_probe_prefers_openh264_and_has_bounded_mpeg4_fallback() {
        assert_eq!(
            encoder_probe_args(),
            vec!["-hide_banner".to_string(), "-encoders".to_string()]
        );
        assert_eq!(
            encoder_from_listing(b" V....D libopenh264 OpenH264 H.264 encoder\n V.S..D mpeg4 MPEG-4"),
            VideoEncoder::OpenH264
        );
        assert_eq!(
            encoder_from_listing(b" V.S..D mpeg4 MPEG-4"),
            VideoEncoder::Mpeg4
        );
        let fallback = transform_args(
            Path::new("input"),
            Path::new("transforms"),
            Path::new("output"),
            VideoEncoder::Mpeg4,
            120.0,
        );
        assert!(fallback.windows(2).any(|pair| pair == ["-c:v", "mpeg4"]));
        assert!(fallback.windows(2).any(|pair| pair[0] == "-b:v" && pair[1].ends_with('k')));
        assert!(!fallback.iter().any(|arg| matches!(arg.as_str(), "-maxrate" | "-q:v")));
        assert!(video_bitrate_kbps(120.0) < video_bitrate_kbps(10.0));
        assert!(video_bitrate_kbps(10.0) <= MAX_VIDEO_BITRATE_KBPS);
    }

    #[test]
    fn source_age_is_human_readable_and_never_exposes_a_path() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        assert!(source_age(now - 120_000).contains("2 minutes ago"));
        assert!(!source_age(now - 120_000).contains('/'));
    }

    #[test]
    fn job_directory_is_removed_on_drop() -> Result<()> {
        let parent = tempfile::tempdir()?;
        let path;
        {
            let job = JobDir::create_in(parent.path())?;
            path = job.path().to_path_buf();
            std::fs::write(job.path().join("partial-output"), b"partial")?;
            assert!(path.exists());
        }
        assert!(!path.exists());
        Ok(())
    }

    #[tokio::test]
    async fn successful_output_outlives_isolated_job_directory() -> Result<()> {
        let parent = tempfile::tempdir()?;
        let durable = parent.path().join("durable");
        let attachment;
        let job_path;
        {
            let job = JobDir::create_in(parent.path())?;
            job_path = job.path().to_path_buf();
            let output = job.path().join("output.mp4");
            tokio::fs::write(&output, b"playable-after-job-cleanup").await?;
            attachment = persist_output_to(&output, &durable).await?;
        }
        assert!(!job_path.exists());
        assert_eq!(attachment.bytes()?, b"playable-after-job-cleanup");
        assert!(matches!(attachment.data, AttachmentData::File(_)));
        Ok(())
    }

    #[tokio::test]
    async fn persisted_output_is_removed_after_delivery_finishes() -> Result<()> {
        let parent = tempfile::tempdir()?;
        let source = parent.path().join("source.mp4");
        tokio::fs::write(&source, b"video").await?;
        let attachment = persist_output_to(&source, &parent.path().join("durable")).await?;
        let path = match &attachment.data {
            AttachmentData::File(path) => path.clone(),
            AttachmentData::Memory(_) => unreachable!(),
        };
        assert!(path.is_file());
        remove_persisted_output(&attachment).await;
        assert!(!path.exists());
        Ok(())
    }

    #[tokio::test]
    async fn failed_attachment_delivery_gets_one_text_fallback_before_cleanup() -> Result<()> {
        let parent = tempfile::tempdir()?;
        let source = parent.path().join("source.mp4");
        tokio::fs::write(&source, b"video").await?;
        let attachment = persist_output_to(&source, &parent.path().join("durable")).await?;
        let path = match &attachment.data {
            AttachmentData::File(path) => path.clone(),
            AttachmentData::Memory(_) => unreachable!(),
        };
        let (sender, mut receiver) = mpsc::channel(2);
        let observed_path = path.clone();
        let observer = tokio::spawn(async move {
            let first = receiver.recv().await.unwrap();
            match first {
                IoRequest::ConversationNotify { attachment: Some(_), reply, .. } => {
                    reply.send(Err(anyhow::anyhow!("HTTP 413 Payload Too Large"))).unwrap();
                }
                _ => panic!("expected attachment notification"),
            }
            let second = receiver.recv().await.unwrap();
            match second {
                IoRequest::ConversationNotify { message, attachment: None, reply, .. } => {
                    assert!(message.contains("could not be delivered"), "{message}");
                    assert!(observed_path.is_file(), "output was removed before fallback");
                    reply.send(Ok("fallback delivered".into())).unwrap();
                }
                _ => panic!("expected one text-only fallback"),
            }
            assert!(receiver.try_recv().is_err(), "fallback recursed");
        });
        let error = deliver_stabilized(&sender, "ctx", "ready".into(), attachment)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("413 Payload Too Large"), "{error}");
        observer.await?;
        assert!(!path.exists(), "durable output was not cleaned after fallback");
        Ok(())
    }

    #[tokio::test]
    async fn real_short_fixture_stabilizes_to_a_playable_video() -> Result<()> {
        let ffmpeg = Path::new("/usr/bin/ffmpeg");
        let ffprobe = Path::new("/usr/bin/ffprobe");
        if !ffmpeg.is_file() || !ffprobe.is_file() {
            return Ok(());
        }
        let filters = std::process::Command::new(ffmpeg)
            .args(["-hide_banner", "-filters"])
            .output()?;
        if !String::from_utf8_lossy(&filters.stdout).contains("vidstabdetect") {
            return Ok(());
        }
        let parent = tempfile::tempdir()?;
        let input = parent.path().join("fixture.mp4");
        let generated = std::process::Command::new(ffmpeg)
            .args([
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=160x120:rate=12:duration=1",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1",
                "-shortest",
                "-c:v",
                "mpeg4",
                "-q:v",
                "3",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                input.to_str().unwrap(),
            ])
            .status()?;
        ensure!(generated.success(), "failed to generate real video fixture");
        let work = JobDir::create_in(parent.path())?;
        let output = work.path().join("output.mp4");
        stabilize_file(&input, &output, work.path(), 1.0).await?;
        let probe = probe_video(&output).await?;
        assert_eq!((probe.width, probe.height), (160, 120));
        assert!(probe.duration_seconds > 0.8);
        assert!(probe.has_audio);
        Ok(())
    }
}
