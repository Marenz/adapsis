//! Attachment type for binary data in the Adapsis value system.
//!
//! Attachments represent binary blobs (audio files, images, etc.) that flow
//! through function calls without being serialized to text. Small attachments
//! (≤ 10MB) stay in memory; larger ones are written to a temp file.

use std::path::PathBuf;
use std::sync::Arc;

/// Threshold for in-memory vs file-backed storage.
const MEMORY_THRESHOLD: usize = 10 * 1024 * 1024; // 10 MB

/// Binary attachment with metadata.
#[derive(Debug, Clone)]
pub struct Attachment {
    pub data: AttachmentData,
    pub mime_type: String,
    pub name: String,
}

/// Storage for attachment data — in memory or on disk.
#[derive(Debug, Clone)]
pub enum AttachmentData {
    /// Small data kept in memory (≤ 10MB).
    Memory(Arc<Vec<u8>>),
    /// Large data stored in a temp file.
    File(PathBuf),
}

impl Attachment {
    fn header(&self) -> Option<([u8; 12], usize)> {
        let mut header = [0_u8; 12];
        let size = match &self.data {
            AttachmentData::Memory(bytes) => {
                let size = bytes.len().min(header.len());
                header[..size].copy_from_slice(&bytes[..size]);
                size
            }
            AttachmentData::File(path) => {
                use std::io::Read;

                let mut file = std::fs::File::open(path).ok()?;
                let size = file.metadata().ok()?.len().min(header.len() as u64) as usize;
                file.read_exact(&mut header[..size]).ok()?;
                size
            }
        };
        Some((header, size))
    }

    /// Recognize a video by MIME type, extension, or magic header.
    ///
    /// Telegram serves videos as `video/*`, or sometimes as
    /// `application/octet-stream` with no usable filename, so — exactly like
    /// [`Self::image_mime`] — the MIME and extension are only a first pass and
    /// the container header is the authoritative fallback. The one MIME filter
    /// that would silently discard a real video is the bare `video/*` prefix
    /// test; keep magic-header recognition so a `message.document` that lost its
    /// MIME still reaches the stabilization pipeline.
    pub fn video_mime(&self) -> Option<&str> {
        let mime = self.mime_type.split(';').next().unwrap_or("").trim();
        if mime.starts_with("video/") {
            return Some(mime);
        }
        match self.name.rsplit('.').next().unwrap_or("").to_ascii_lowercase().as_str() {
            "mp4" => return Some("video/mp4"),
            "m4v" => return Some("video/x-m4v"),
            "mov" => return Some("video/quicktime"),
            "webm" => return Some("video/webm"),
            "mkv" => return Some("video/x-matroska"),
            "avi" => return Some("video/x-msvideo"),
            "mpeg" | "mpg" => return Some("video/mpeg"),
            "3gp" | "3g2" => return Some("video/3gpp"),
            _ => {}
        }
        let (header, size) = self.header()?;
        let header = &header[..size];
        // ISO Base Media File Format (mp4/mov/m4v/3gp): `....ftyp` at bytes 4..8.
        if size >= 8 && &header[4..8] == b"ftyp" {
            Some("video/mp4")
        // EBML/Matroska/WebM magic.
        } else if header.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
            Some("video/webm")
        // RIFF AVI.
        } else if size == 12 && &header[..4] == b"RIFF" && &header[8..] == b"AVI " {
            Some("video/x-msvideo")
        } else {
            None
        }
    }

    /// Telegram file downloads may be application/octet-stream. Use the same
    /// image recognition for context budgets, durable recovery and LLM payloads.
    pub fn image_mime(&self) -> Option<&str> {
        let mime=self.mime_type.split(';').next().unwrap_or("").trim();
        if mime.starts_with("image/") {return Some(mime);}
        match self.name.rsplit('.').next().unwrap_or("").to_ascii_lowercase().as_str() {
            "jpg"|"jpeg" => return Some("image/jpeg"),
            "png" => return Some("image/png"),
            "gif" => return Some("image/gif"),
            "webp" => return Some("image/webp"),
            _ => {},
        }
        let (header,size)=self.header()?;
        let header=&header[..size];
        if header.starts_with(b"\xff\xd8\xff") {Some("image/jpeg")}
        else if header.starts_with(b"\x89PNG\r\n\x1a\n") {Some("image/png")}
        else if header.starts_with(b"GIF87a")||header.starts_with(b"GIF89a") {Some("image/gif")}
        else if size==12 && &header[..4]==b"RIFF" && &header[8..]==b"WEBP" {Some("image/webp")}
        else {None}
    }

    /// Create an attachment from raw bytes.
    /// If the data exceeds 10MB, writes to a temp file under
    /// `~/.local/share/adapsis/tmp/`.
    pub fn from_bytes(
        data: Vec<u8>,
        mime_type: impl Into<String>,
        name: impl Into<String>,
    ) -> std::io::Result<Self> {
        let mime_type = mime_type.into();
        let name = name.into();
        let data = if data.len() > MEMORY_THRESHOLD {
            let dir = dirs::data_dir()
                .unwrap_or_else(|| PathBuf::from("/tmp"))
                .join("adapsis/tmp");
            std::fs::create_dir_all(&dir)?;
            let path = dir.join(format!("attachment-{}-{}", std::process::id(), &name));
            std::fs::write(&path, &data)?;
            AttachmentData::File(path)
        } else {
            AttachmentData::Memory(Arc::new(data))
        };
        Ok(Self {
            data,
            mime_type,
            name,
        })
    }

    /// Create an attachment from an existing file on disk.
    #[cfg(test)]
    pub fn from_file(
        path: std::path::PathBuf,
        mime_type: impl Into<String>,
        name: impl Into<String>,
    ) -> Self {
        Self {
            data: AttachmentData::File(path),
            mime_type: mime_type.into(),
            name: name.into(),
        }
    }

    /// Get the raw bytes. Reads from file if necessary.
    pub fn bytes(&self) -> std::io::Result<Vec<u8>> {
        match &self.data {
            AttachmentData::Memory(data) => Ok(data.as_ref().clone()),
            AttachmentData::File(path) => std::fs::read(path),
        }
    }

    /// Size in bytes.
    pub fn len(&self) -> usize {
        match &self.data {
            AttachmentData::Memory(data) => data.len(),
            AttachmentData::File(path) => std::fs::metadata(path)
                .map(|m| m.len() as usize)
                .unwrap_or(0),
        }
    }

    /// Guess MIME type from file extension.
    #[cfg(test)]
    pub fn mime_from_extension(name: &str) -> &'static str {
        match name.rsplit('.').next() {
            Some("ogg") => "audio/ogg",
            Some("wav") => "audio/wav",
            Some("mp3") => "audio/mpeg",
            Some("flac") => "audio/flac",
            Some("jpg") | Some("jpeg") => "image/jpeg",
            Some("png") => "image/png",
            Some("gif") => "image/gif",
            Some("pdf") => "application/pdf",
            Some("json") => "application/json",
            Some("mp4") => "video/mp4",
            Some("mov") => "video/quicktime",
            Some("webm") => "video/webm",
            Some("mkv") => "video/x-matroska",
            Some("avi") => "video/x-msvideo",
            _ => "application/octet-stream",
        }
    }
}

impl std::fmt::Display for Attachment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let size = self.len();
        let unit = if size > 1_048_576 {
            format!("{:.1}MB", size as f64 / 1_048_576.0)
        } else if size > 1024 {
            format!("{:.1}KB", size as f64 / 1024.0)
        } else {
            format!("{size}B")
        };
        write!(f, "<attachment: {} {} {}>", self.name, self.mime_type, unit)
    }
}

impl PartialEq for Attachment {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.mime_type == other.mime_type && self.len() == other.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_attachment_stays_in_memory() {
        let data = vec![1, 2, 3, 4, 5];
        let att = Attachment::from_bytes(data.clone(), "audio/ogg", "test.ogg").unwrap();
        assert!(matches!(att.data, AttachmentData::Memory(_)));
        assert_eq!(att.bytes().unwrap(), data);
        assert_eq!(att.len(), 5);
        assert_eq!(att.mime_type, "audio/ogg");
        assert_eq!(att.name, "test.ogg");
    }

    #[test]
    fn display_format() {
        let att = Attachment::from_bytes(vec![0; 512_000], "audio/ogg", "song.ogg").unwrap();
        let display = format!("{att}");
        assert!(display.contains("song.ogg"));
        assert!(display.contains("audio/ogg"));
        assert!(display.contains("500.0KB"));
    }

    #[test]
    fn mime_from_extension() {
        assert_eq!(Attachment::mime_from_extension("song.ogg"), "audio/ogg");
        assert_eq!(Attachment::mime_from_extension("photo.png"), "image/png");
        assert_eq!(
            Attachment::mime_from_extension("data.bin"),
            "application/octet-stream"
        );
    }

    #[test]
    fn image_mime_recognizes_generic_octet_stream_by_magic() {
        let jpeg = Attachment::from_bytes(
            vec![0xff, 0xd8, 0xff, 0xe0, 0x00], "application/octet-stream", "telegram-file"
        ).unwrap();
        assert_eq!(jpeg.image_mime(), Some("image/jpeg"));
        let not_image = Attachment::from_bytes(
            b"plain text".to_vec(), "application/octet-stream", "telegram-file"
        ).unwrap();
        assert_eq!(not_image.image_mime(), None);
    }

    #[test]
    fn from_file() {
        let att = Attachment::from_file(PathBuf::from("/tmp/test.wav"), "audio/wav", "test.wav");
        assert!(matches!(att.data, AttachmentData::File(_)));
        assert_eq!(att.mime_type, "audio/wav");
    }

    #[test]
    fn video_mime_recognizes_mime_extension_and_magic() {
        // MIME prefix wins.
        assert_eq!(Attachment::from_bytes(vec![0; 4], "video/mp4", "x").unwrap().video_mime(), Some("video/mp4"));
        // Extension on an octet-stream download.
        assert_eq!(Attachment::from_bytes(vec![0; 4], "application/octet-stream", "clip.mov").unwrap().video_mime(), Some("video/quicktime"));
        assert_eq!(Attachment::from_bytes(vec![0; 4], "application/octet-stream", "clip.webm").unwrap().video_mime(), Some("video/webm"));
        // Magic header with no MIME and a useless name.
        let ftyp = Attachment::from_bytes(b"\x00\x00\x00\x18ftypisom".to_vec(), "application/octet-stream", "opaque-hash").unwrap();
        assert_eq!(ftyp.video_mime(), Some("video/mp4"));
        let webm = Attachment::from_bytes(vec![0x1a, 0x45, 0xdf, 0xa3, 0x00, 0x00, 0x00, 0x00], "application/octet-stream", "opaque-hash").unwrap();
        assert_eq!(webm.video_mime(), Some("video/webm"));
        let avi = Attachment::from_bytes(b"RIFF\x00\x00\x00\x00AVI ".to_vec(), "application/octet-stream", "opaque-hash").unwrap();
        assert_eq!(avi.video_mime(), Some("video/x-msvideo"));
        // Not a video.
        assert_eq!(Attachment::from_bytes(vec![0xff, 0xd8, 0xff], "application/octet-stream", "x").unwrap().video_mime(), None);
        assert_eq!(Attachment::from_bytes(vec![1, 2, 3], "application/octet-stream", "x").unwrap().video_mime(), None);
    }

    #[test]
    fn file_backed_magic_headers_are_read_completely() {
        let png = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(png.path(), b"\x89PNG\r\n\x1a\n").unwrap();
        let image = Attachment::from_file(
            png.path().to_path_buf(),
            "application/octet-stream",
            "opaque",
        );
        assert_eq!(image.image_mime(), Some("image/png"));

        let avi = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(avi.path(), b"RIFF\x00\x00\x00\x00AVI ").unwrap();
        let video = Attachment::from_file(
            avi.path().to_path_buf(),
            "application/octet-stream",
            "opaque",
        );
        assert_eq!(video.video_mime(), Some("video/x-msvideo"));
    }
}
