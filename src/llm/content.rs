//! Provider-facing multipart content, including images replayed from old history.
use super::{ChatContent, ChatContentPart};
use serde::ser::{Serialize, SerializeSeq, Serializer};

impl Serialize for ChatContent {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Text(text) => serializer.serialize_str(text),
            Self::Parts(parts) => {
                // Captionless photos used to carry {type:"text", text:""}.
                // Anthropic rejects that block, including when a later turn
                // replays the photo. Normalize at the shared wire boundary so
                // old/deserialized parts and current/recovered images agree.
                // Never trim nonempty text or change text-only/tool messages.
                let has_image = parts.iter().any(|part| {
                    matches!(part, ChatContentPart::ImageUrl { image_url } if !image_url.url.is_empty())
                });
                let mut sequence = serializer.serialize_seq(None)?;
                for part in parts {
                    if has_image
                        && matches!(part, ChatContentPart::Text { text } if text.is_empty())
                    {
                        continue;
                    }
                    sequence.serialize_element(part)?;
                }
                sequence.end()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ChatMessage, ToolCall};
    use serde_json::{json, Value};

    #[test]
    fn image_content_normalizes_old_parts_without_changing_images_or_nonempty_text() {
        let image =
            json!({"type":"image_url","image_url":{"url":"data:image/png;base64,aGVsbG8="}});
        let wire = json!({"role":"user","content":[
            {"type":"text","text":""}, image,
            {"type":"text","text":"caption"}, {"type":"text","text":""},
            image, {"type":"text","text":" "}
        ]});
        let restored: ChatMessage = serde_json::from_value(wire).unwrap();
        assert_eq!(
            serde_json::to_value(&restored).unwrap(),
            json!({"role":"user","content":[
                image, {"type":"text","text":"caption"}, image, {"type":"text","text":" "}
            ]})
        );
        // Rendering does not rewrite the stored/in-memory content.
        assert!(matches!(&restored.content, ChatContent::Parts(parts) if parts.len() == 6));
    }

    #[test]
    fn image_content_preserves_captioned_and_text_only_wire_shapes() {
        for wire in [
            json!({"role":"user","content":""}),
            json!({"role":"user","content":"text"}),
            json!({"role":"user","content":[{"type":"text","text":""}]}),
            json!({"role":"user","content":[{"type":"text","text":"caption"},
                {"type":"image_url","image_url":{"url":"https://example.invalid/photo.png"}}]}),
            json!({"role":"user","content":[{"type":"text","text":""},
                {"type":"image_url","image_url":{"url":""}}]}),
        ] {
            let message: ChatMessage = serde_json::from_value(wire.clone()).unwrap();
            assert_eq!(serde_json::to_value(message).unwrap(), wire);
        }
    }

    #[test]
    fn image_content_keeps_empty_assistant_tool_calls_and_correlated_results() {
        let calls = vec![ToolCall {
            id: "call-photo".into(),
            name: "inspect".into(),
            arguments: "{}".into(),
        }];
        let messages = vec![
            ChatMessage::assistant_with_tool_calls("", calls),
            ChatMessage::tool_result("call-photo", ""),
        ];
        let wire: Value = serde_json::to_value(messages).unwrap();
        assert_eq!(
            wire,
            json!([
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"call-photo","type":"function","function":{"name":"inspect","arguments":"{}"}}
                ]},
                {"role":"tool","content":"","tool_call_id":"call-photo"}
            ])
        );
    }
}
