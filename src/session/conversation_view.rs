//! Bounded text history with a separate budget for recent image attachments.
use super::{ChatMessage, Conversation, truncate_middle};

const IMAGE_COUNT: usize = 3;
const IMAGE_BYTES: usize = 20 * 1024 * 1024;

pub(super) fn build(
    conv: &Conversation,
    max_messages: usize,
    max_chars: usize,
    max_message_chars: usize,
) -> Vec<crate::llm::ChatMessage> {
    if conv.messages.is_empty() || max_messages == 0 {
        return Vec::new();
    }
    let system = conv.messages.first().filter(|m| m.role == "system");
    let mut remaining_chars = max_chars;
    let mut image_count = IMAGE_COUNT;
    let mut image_bytes = IMAGE_BYTES;
    let limit = max_messages.saturating_sub(usize::from(system.is_some()));
    let mut recent = Vec::new();
    for group in super::native_history::complete_groups(&conv.messages)
        .into_iter()
        .rev()
    {
        if group.len() > 1 {
            let costs: Vec<usize> = group
                .iter()
                .map(|m| {
                    m.content.chars().count()
                        + m.tool_calls.as_ref().map_or(0, |calls| {
                            calls
                                .iter()
                                .map(|c| c.arguments.chars().count() + c.name.len() + c.id.len())
                                .sum::<usize>()
                        })
                })
                .collect();
            let cost = costs.iter().sum::<usize>();
            if recent.len() + group.len() <= limit
                && cost <= remaining_chars
                && costs.iter().all(|cost| *cost <= max_message_chars)
            {
                recent.extend(group.iter().rev().cloned());
                remaining_chars -= cost;
            }
            continue;
        }
        let message = &group[0];
        if system.is_some_and(|s| std::ptr::eq(s, message)) || recent.len() >= limit {
            continue;
        }
        let content = truncate_middle(&message.content, max_message_chars.min(remaining_chars));
        let content_chars = content.chars().count();
        let mut attachments = Vec::new();
        // Text and image budgets are independent: a photo without a caption is
        // still a user turn, even when the text history budget has been spent.
        if message.role == "user" {
            for attachment in &message.attachments {
                let bytes = attachment.len();
                if attachment.image_mime().is_some()
                    && image_count > 0
                    && bytes > 0
                    && bytes <= image_bytes
                {
                    attachments.push(attachment.clone());
                    image_count -= 1;
                    image_bytes -= bytes;
                }
            }
        }
        if content_chars == 0 && attachments.is_empty() {
            continue;
        }
        recent.push(ChatMessage {
            role: message.role.clone(),
            content,
            attachments,
            ..Default::default()
        });
        remaining_chars -= content_chars;
    }
    recent.reverse();
    let omitted = conv
        .messages
        .len()
        .saturating_sub(recent.len() + usize::from(system.is_some()));
    let mut result = Vec::new();
    if let Some(system) = system {
        result.push(system.clone());
    }
    if omitted > 0 {
        result.push(ChatMessage { role: "user".into(),
            content: format!("[System: {omitted} older message(s) are omitted from this model context; the raw history is retained.]"), ..Default::default() });
    }
    result.extend(recent);
    result.iter().map(super::native_history::to_llm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn image(n: u8) -> crate::attachment::Attachment {
        crate::attachment::Attachment::from_bytes(vec![n; 3], "image/png", "reference.png").unwrap()
    }
    #[test]
    fn photo_only_turn_survives_even_zero_text_budget() {
        let mut conv = Conversation::new();
        conv.push_system("system");
        conv.push_user_with_attachment("", image(1));
        let json = serde_json::to_string(&build(&conv, 4, 0, 0)).unwrap();
        assert!(json.contains("data:image/png;base64,AQEB"));
        assert_eq!(conv.messages[1].attachments.len(), 1);
    }
    #[test]
    fn captioned_image_is_sent_with_its_caption() {
        let mut conv = Conversation::new();
        conv.push_user_with_attachment("Use this reference", image(2));
        let json = serde_json::to_string(&build(&conv, 4, 100, 100)).unwrap();
        assert!(json.contains("Use this reference"));
        assert!(json.contains("data:image/png;base64,AgIC"));
    }
    #[test]
    fn telegram_octet_stream_photos_survive_and_use_image_mime() {
        let mut conv = Conversation::new();
        let attachment = crate::attachment::Attachment::from_bytes(
            vec![0xff, 0xd8, 0xff],
            "application/octet-stream",
            "opaque-hash",
        )
        .unwrap();
        conv.push_user_with_attachment("", attachment);
        let json = serde_json::to_string(&build(&conv, 4, 100, 100)).unwrap();
        assert!(json.contains("data:image/jpeg;base64,/9j/"));
    }
    #[test]
    fn newest_three_images_retained_without_changing_history() {
        let mut conv = Conversation::new();
        for n in 1..=5 {
            conv.push_user_with_attachment("", image(n));
        }
        let json = serde_json::to_string(&build(&conv, 10, 100, 100)).unwrap();
        assert!(!json.contains("base64,AQEB"));
        assert!(!json.contains("base64,AgIC"));
        for data in ["AwMD", "BAQE", "BQUF"] {
            assert!(json.contains(data));
        }
        assert_eq!(
            conv.messages
                .iter()
                .map(|m| m.attachments.len())
                .sum::<usize>(),
            5
        );
    }
}
