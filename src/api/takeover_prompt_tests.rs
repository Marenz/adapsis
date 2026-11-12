use super::*;
use crate::llm::{ChatContent, ChatContentPart, ChatMessage, ChatRole};
use crate::memory_graph::{PromptMemorySelection, PromptSelectionStats, RecalledMemory};

fn selection(canonical: bool, content: &str) -> PromptMemorySelection {
    PromptMemorySelection {
        memories: vec![RecalledMemory {
            id: format!("memory:{content}"),
            content: content.into(),
            memory_type: "fixture".into(),
            canonical,
            confidence: if canonical { 1.0 } else { 0.9 },
            citation: if canonical {
                String::new()
            } else {
                "ctx; speaker; 1; source".into()
            },
        }],
        stats: PromptSelectionStats {
            active_authorized_records: 1,
            candidate_records_scanned: 1,
            selected_representatives: 1,
            selected_row_chars: 1,
            max_count: 40,
            max_row_chars: 8_000,
            ..Default::default()
        },
    }
}

#[test]
fn memory_blocks_stay_in_the_runtime_envelope_before_the_newest_user_turn() {
    let messages = vec![
        ChatMessage::system("primary"),
        ChatMessage::user("older"),
        ChatMessage::user("newest"),
    ];
    let mut context_log = crate::context_log::ContextLog::from_conversation(messages);
    push_memory_prompt_blocks(
        &mut context_log,
        &selection(false, "ranked"),
        &selection(true, "canonical"),
    );
    let messages = context_log.into_messages();
    assert_eq!(messages.last().unwrap().role, ChatRole::User);
    let ChatContent::Parts(parts) = &messages.last().unwrap().content else {
        panic!("runtime envelope must be multipart");
    };
    let text = parts
        .iter()
        .filter_map(|part| match part {
            ChatContentPart::Text { text } => Some(text.as_str()),
            ChatContentPart::ImageUrl { .. } => None,
        })
        .collect::<Vec<_>>();
    assert!(text[1].contains("canonical"));
    assert!(text[2].contains("ranked"));
    assert!(text.last().unwrap().ends_with("newest"));
}
