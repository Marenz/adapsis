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

pub(super) fn trim(conv: &mut Conversation, max: usize) {
    if max < 2 || conv.messages.len() <= max {
        return;
    }
    let system = conv
        .messages
        .first()
        .filter(|m| m.role == "system")
        .cloned();
    let mut remaining = max - usize::from(system.is_some());
    let mut selected = Vec::new();
    for group in complete_groups(&conv.messages).into_iter().rev() {
        if system.is_some() && std::ptr::eq(&group[0], &conv.messages[0]) {
            continue;
        }
        if group.len() > remaining {
            break;
        }
        selected.push(group.to_vec());
        remaining -= group.len();
    }
    conv.messages = system
        .into_iter()
        .chain(selected.into_iter().rev().flatten())
        .collect();
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
        let bounded = restored.to_llm_messages_bounded(20, 100, 50);
        assert_eq!(bounded.len(), 5);
        assert_eq!(bounded[2].tool_calls.len(), 1);
    }

    #[test]
    fn budgets_omit_whole_pairs_without_truncating_arguments() {
        let conv = conversation();
        for messages in [
            conv.to_llm_messages_bounded(3, 100, 100),
            conv.to_llm_messages_bounded(20, 10, 100),
        ] {
            assert!(
                messages
                    .iter()
                    .all(|m| m.tool_calls.is_empty() && m.tool_call_id.is_none())
            );
        }
        let mut trimmed = conv;
        trimmed.trim(3);
        assert!(
            trimmed
                .to_llm_messages()
                .iter()
                .all(|m| m.tool_calls.is_empty() && m.tool_call_id.is_none())
        );
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

    #[test]
    fn native_trim_keeps_recent_secondary_system_notifications() {
        let mut conv = Conversation::new();
        conv.push_system("primary");
        for _ in 0..60 {
            conv.push_user("old");
        }
        conv.push_system("job result");
        conv.push_user("latest");
        conv.trim(51);
        assert_eq!(conv.messages[0].content, "primary");
        assert!(
            conv.messages
                .iter()
                .any(|m| m.role == "system" && m.content == "job result")
        );
        assert_eq!(
            conv.messages
                .iter()
                .filter(|m| m.content == "primary")
                .count(),
            1
        );
    }
}
