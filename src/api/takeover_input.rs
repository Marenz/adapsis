//! Intake presentation/provenance helpers; identity never comes from prose.
pub(crate) fn strip_sender_prefix(message: &str) -> &str {
    message
        .strip_prefix("[user:")
        .and_then(|rest| rest.split_once("] "))
        .map_or(message, |(_, content)| content)
}

pub(crate) fn takeover_principal(context: &str, source_metadata: Option<&str>) -> String {
    source_metadata
        .and_then(|metadata| serde_json::from_str::<serde_json::Value>(metadata).ok())
        .and_then(|metadata| {
            metadata
                .get("speaker_id")
                .and_then(|value| value.as_str())
                .map(str::to_string)
        })
        .filter(|speaker| !speaker.is_empty())
        .unwrap_or_else(|| context_principal(context))
}

pub(crate) fn attachment_platform_ref(
    source_metadata: Option<&str>,
    attachment: &crate::attachment::Attachment,
) -> String {
    let metadata = source_metadata
        .and_then(|value| serde_json::from_str::<serde_json::Value>(value).ok())
        .unwrap_or_default();
    let text = |key: &str| {
        metadata
            .get(key)
            .and_then(|value| value.as_str())
            .unwrap_or_default()
    };
    let name = match text("attachment_name") {
        "" => attachment.name.as_str(),
        value => value,
    };
    let mime = match text("attachment_mime") {
        "" => attachment.mime_type.as_str(),
        value => value,
    };
    let size = text("attachment_size_bytes")
        .parse::<u64>()
        .unwrap_or(attachment.len() as u64);
    serde_json::json!({"platform_ref":text("attachment_ref"),"name":name,"mime_type":mime,
        "duration_seconds":text("attachment_duration_seconds"),"size_bytes":size})
    .to_string()
}

fn context_principal(context: &str) -> String {
    if context.starts_with("telegram:group:") {
        return context.to_string();
    }
    context
        .strip_prefix("telegram:user:")
        .or_else(|| context.strip_prefix("telegram:"))
        .map_or_else(
            || "user:unknown".to_string(),
            |id| format!("telegram:user:{id}"),
        )
}
