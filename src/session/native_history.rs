//! Native assistant/results are indivisible history units, including on restore.
use super::{ChatMessage, Conversation};

pub(super) fn complete_groups(messages: &[ChatMessage]) -> Vec<&[ChatMessage]> {
    let mut groups = Vec::new();
    let mut index = 0;
    while index < messages.len() {
        let message = &messages[index];
        if let Some(calls) = &message.tool_calls {
            let end = index + 1 + calls.len();
            let unique: std::collections::HashSet<_> = calls.iter().map(|c| &c.id).collect();
            let valid = message.role == "assistant"
                && !calls.is_empty()
                && unique.len() == calls.len()
                && calls.iter().all(|c| !c.id.trim().is_empty())
                && end <= messages.len()
                && calls
                    .iter()
                    .zip(&messages[index + 1..end.min(messages.len())])
                    .all(|(call, result)| {
                        result.role == "tool"
                            && result.tool_call_id.as_ref() == Some(&call.id)
                            && result.tool_calls.is_none()
                    });
            if valid {
                groups.push(&messages[index..end]);
                index = end;
                continue;
            }
        } else if message.tool_call_id.is_none() {
            groups.push(&messages[index..index + 1]);
        }
        index += 1;
    }
    groups
}

pub(super) fn to_llm(message: &ChatMessage) -> crate::llm::ChatMessage {
    match message.role.as_str() {
        "system" => crate::llm::ChatMessage::system(message.content.clone()),
        "assistant" => match &message.tool_calls {
            Some(calls) => crate::llm::ChatMessage::assistant_with_tool_calls(
                message.content.clone(),
                calls.clone(),
            ),
            None => crate::llm::ChatMessage::assistant(&message.content),
        },
        "tool" if message.tool_call_id.is_some() => crate::llm::ChatMessage::tool_result(
            message.tool_call_id.clone().unwrap_or_default(),
            message.content.clone(),
        ),
        _ => crate::llm::ChatMessage::user_with_attachments(
            message.content.clone(),
            &message.attachments,
        ),
    }
}

impl Conversation {
    /// Append one fully correlated batch under the caller's metadata lock. Never
    /// save an assistant envelope before its complete result set is available.
    pub fn push_native_batch(
        &mut self,
        content: String,
        calls: Vec<crate::llm::ToolCall>,
        results: Vec<String>,
    ) {
        if calls.is_empty() || calls.len() != results.len() {
            return;
        }
        let mut batch = vec![ChatMessage {
            role: "assistant".into(),
            content,
            tool_calls: Some(calls.clone()),
            ..Default::default()
        }];
        batch.extend(
            calls
                .into_iter()
                .zip(results)
                .map(|(call, content)| ChatMessage {
                    role: "tool".into(),
                    content,
                    tool_call_id: Some(call.id),
                    ..Default::default()
                }),
        );
        if complete_groups(&batch)
            .first()
            .is_some_and(|group| group.len() == batch.len())
        {
            self.messages.extend(batch);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conversation() -> Conversation {
        let mut conv = Conversation::new();
        conv.push_system("stable");
        conv.push_user("task");
        conv.push_native_batch(
            String::new(),
            vec![crate::llm::ToolCall {
                id: "call1".into(),
                name: "inspect".into(),
                arguments: r#"{"kind":"symbols"}"#.into(),
            }],
            vec!["result".into()],
        );
        conv.push_user("next");
        conv
    }

    #[test]
    fn native_pairs_roundtrip_and_call_only_assistants_are_meaningful() {
        let conv = conversation();
        let restored: Conversation =
            serde_json::from_str(&serde_json::to_string(&conv).unwrap()).unwrap();
        let messages = restored.to_llm_messages();
        assert_eq!(messages.len(), 5);
        assert_eq!(messages[2].tool_calls[0].id, "call1");
        assert_eq!(messages[3].role, crate::llm::ChatRole::Tool);
        assert_eq!(messages[3].tool_call_id.as_deref(), Some("call1"));
        let mut restored = restored;
        let bounded = restored.model_view(&super::super::conversation_view::ViewPolicy::tokens_only(10_000, 5_000, 0, 50)).messages;
        assert_eq!(bounded.len(), 5);
        assert_eq!(bounded[2].tool_calls.len(), 1);
    }

    #[test]
    fn checkpoints_and_size_caps_omit_whole_pairs_without_truncating_arguments() {
        use super::super::conversation_view::ViewPolicy;
        // A checkpoint past the batch drops the whole pair; an oversized batch is
        // omitted whole rather than truncated into an invalid protocol.
        for policy in [
            ViewPolicy::tokens_only(1, 0, 0, 1_000),
            ViewPolicy::tokens_only(10_000, 5_000, 0, 5),
        ] {
            let mut conv = conversation();
            let messages = conv.model_view(&policy).messages;
            assert!(messages.iter().all(|m| m.tool_calls.is_empty() && m.tool_call_id.is_none()));
            assert_eq!(conv.messages.len(), 5, "persisted history untouched");
        }
    }

    #[test]
    fn orphan_and_partial_batches_are_not_replayed_after_restart() {
        let mut conv = conversation();
        conv.messages.remove(3);
        assert!(
            conv.to_llm_messages()
                .iter()
                .all(|m| m.tool_calls.is_empty() && m.tool_call_id.is_none())
        );
        conv.messages.push(ChatMessage {
            role: "tool".into(),
            tool_call_id: Some("orphan".into()),
            content: "no request".into(),
            ..Default::default()
        });
        assert!(
            conv.to_llm_messages()
                .iter()
                .all(|m| m.role != crate::llm::ChatRole::Tool)
        );
    }

    #[test]
    fn old_session_messages_need_no_tool_fields() {
        let old: ChatMessage =
            serde_json::from_str(r#"{"role":"assistant","content":"hello"}"#).unwrap();
        assert!(old.tool_calls.is_none() && old.tool_call_id.is_none());
        assert_eq!(
            serde_json::to_value(old).unwrap(),
            serde_json::json!({"role":"assistant","content":"hello"})
        );
    }
}
