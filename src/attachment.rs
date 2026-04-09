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
    pub fn from_file(path: PathBuf, mime_type: impl Into<String>, name: impl Into<String>) -> Self {
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

    /// Whether the attachment has no data.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Guess MIME type from file extension.
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
    fn from_file() {
        let att = Attachment::from_file(PathBuf::from("/tmp/test.wav"), "audio/wav", "test.wav");
        assert!(matches!(att.data, AttachmentData::File(_)));
        assert_eq!(att.mime_type, "audio/wav");
    }
}
