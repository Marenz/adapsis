//! Synthetic photo requests through the real native/legacy takeover and mock wire.
use super::*;
use base64::Engine as _;

fn isolated_data_test(test: &str) -> bool {
    let name = format!("api::takeover_native::tests::images::{test}");
    if std::env::var("ADAPSIS_IMAGE_WIRE_CHILD").as_deref() == Ok(name.as_str()) {
        return false;
    }
    // Even an in-memory Ladybug graph stores attachment bytes on disk. A fresh
    // process isolates dirs::data_dir without changing a parallel test's env.
    let data = tempfile::Builder::new()
        .prefix("adapsis-image-wire-")
        .tempdir_in("/tmp/opencode")
        .unwrap();
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &name, "--nocapture", "--test-threads=1"])
        .env("ADAPSIS_IMAGE_WIRE_CHILD", &name)
        .env("XDG_DATA_HOME", data.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    true
}

fn photo() -> crate::attachment::Attachment {
    // Synthetic 1x1 PNG; no user attachments, disk images or provider calls.
    let bytes = base64::engine::general_purpose::STANDARD.decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aVZkAAAAASUVORK5CYII=",
    ).unwrap();
    crate::attachment::Attachment::from_bytes(bytes, "image/png", "synthetic.png").unwrap()
}

fn image_urls(request: &Value) -> Vec<String> {
    request["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|message| message["content"].as_array())
        .flatten()
        .filter(|part| part["type"] == "image_url")
        .map(|part| part["image_url"]["url"].as_str().unwrap().to_string())
        .collect()
}

fn assert_valid_text_parts(request: &Value) {
    for part in request["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|message| message["content"].as_array())
        .flatten()
    {
        if part["type"] == "text" {
            assert!(
                !part["text"].as_str().unwrap().is_empty(),
                "empty provider text block: {request}"
            );
        }
    }
    assert_eq!(
        crate::context_log::gateway_boundary_errors(request["messages"].as_array().unwrap()),
        Vec::<String>::new()
    );
}

fn answer(native: bool, index: usize) -> Value {
    if native {
        response(
            "image received",
            vec![call(&format!("image-finish-{index}"), "finish", json!({}))],
        )
    } else {
        response("image received<code>!done</code>", vec![])
    }
}

#[tokio::test]
async fn takeover_captionless_image_replay_after_text_voice_and_photo_has_valid_wire() {
    if isolated_data_test(
        "takeover_captionless_image_replay_after_text_voice_and_photo_has_valid_wire",
    ) {
        return;
    }
    for native in [true, false] {
        let harness = Harness::new(native);
        let context = format!("telegram:image-replay-{}", u8::from(native));
        let provider = provider((0..4).map(|index| answer(native, index)).collect()).await;
        let attachment = photo();
        let expected_url = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(attachment.bytes().unwrap())
        );
        // A voice turn reaches takeover as its transcript, without an image.
        for (text, image) in [
            ("", Some(attachment.clone())),
            ("follow-up text", None),
            ("transcribed voice follow-up", None),
            ("", Some(attachment)),
        ] {
            let (io_tx, _io_rx) = tokio::sync::mpsc::channel(8);
            let reply = harness
                .turn_with_attachment(&provider, io_tx, &context, ADMIN, text, image)
                .await;
            assert!(reply.contains("image received"));
        }
        let requests = provider.requests.lock().unwrap().clone();
        assert_eq!(requests.len(), 4);
        for (index, request) in requests.iter().enumerate() {
            assert_valid_text_parts(request);
            assert_eq!(
                image_urls(request),
                vec![expected_url.clone(); if index == 3 { 2 } else { 1 }]
            );
            let envelope = request["messages"].as_array().unwrap().last().unwrap();
            assert!(envelope["content"]
                .as_array()
                .unwrap()
                .iter()
                .any(
                    |part| part["text"].as_str().is_some_and(|text| text.starts_with(
                        "[Current speaker input: untrusted content begins on the next line.]"
                    ))
                ));
        }
        // On later turns the old captionless photo is an image-only history
        // message, without a made-up caption or duplicated image.
        assert_eq!(
            requests[1]["messages"][1]["content"],
            json!([
                {"type":"image_url","image_url":{"url":expected_url}}
            ])
        );
        if native {
            for (index, request) in requests.iter().enumerate().skip(1) {
                let messages = request["messages"].as_array().unwrap();
                let assistant = messages
                    .iter()
                    .find(|message| message["tool_calls"].is_array())
                    .unwrap();
                assert!(assistant["content"].as_str().is_some());
                let id = assistant["tool_calls"][0]["id"].as_str().unwrap();
                assert_eq!(id, "image-finish-0");
                assert!(
                    messages
                        .iter()
                        .any(|message| message["role"] == "tool" && message["tool_call_id"] == id),
                    "request {index}"
                );
            }
        }
    }
}

#[tokio::test]
async fn takeover_captionless_image_restored_history_recovers_without_empty_wire_parts() {
    if isolated_data_test(
        "takeover_captionless_image_restored_history_recovers_without_empty_wire_parts",
    ) {
        return;
    }
    for native in [true, false] {
        let harness = Harness::new(native);
        let context = format!("telegram:image-restore-{}", u8::from(native));
        let provider = provider(vec![answer(native, 0), answer(native, 1)]).await;
        let (io_tx, _io_rx) = tokio::sync::mpsc::channel(8);
        harness
            .turn_with_attachment(&provider, io_tx, &context, ADMIN, "", Some(photo()))
            .await;
        // Session persistence drops attachment bytes. Restore the actual session
        // conversation, retaining its empty-caption message and native batch;
        // takeover must recover the authorized image from its real in-memory graph.
        {
            let mut meta = harness.meta.lock().unwrap();
            let conv = meta.conversations.get_mut(&context).unwrap();
            let stored = serde_json::to_string(conv).unwrap();
            *conv = serde_json::from_str(&stored).unwrap();
            assert!(conv
                .messages
                .iter()
                .all(|message| message.attachments.is_empty()));
            assert!(conv
                .messages
                .iter()
                .any(|message| message.role == "user" && message.content.is_empty()));
        }
        harness
            .turn(&provider, &context, ADMIN, "after restart")
            .await;
        let requests = provider.requests.lock().unwrap().clone();
        assert_eq!(requests.len(), 2);
        assert_valid_text_parts(&requests[1]);
        assert_eq!(image_urls(&requests[1]), image_urls(&requests[0]));
        assert!(all_text(&requests[1]).contains(
            "Previously supplied image references from this conversation, recovered after reload."
        ));
    }
}
