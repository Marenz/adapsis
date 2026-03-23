//! Telegram bot integration for AdapsisOS.
//!
//! Routes admin messages through `/api/ask` (full session context)
//! and non-admin messages through a lightweight single LLM call.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

// ── Routing classification ───────────────────────────────────────────

/// How a Telegram message should be handled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TelegramRoute {
    /// Admin user — route through AdapsisOS `/api/ask` for full session context.
    AdminAsk,
    /// Regular user — lightweight single LLM call, no session mutation.
    UserLlmCall,
}

/// Classify a Telegram message based on the sender's chat ID.
pub fn classify_request(chat_id: i64, admin_chat_id: i64) -> TelegramRoute {
    if chat_id == admin_chat_id {
        TelegramRoute::AdminAsk
    } else {
        TelegramRoute::UserLlmCall
    }
}

// ── Telegram API types ───────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct TelegramUpdate {
    pub update_id: i64,
    #[serde(default)]
    pub message: Option<TelegramMessage>,
}

#[derive(Debug, Deserialize)]
pub struct TelegramMessage {
    pub message_id: i64,
    pub chat: TelegramChat,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub from: Option<TelegramUser>,
}

#[derive(Debug, Deserialize)]
pub struct TelegramChat {
    pub id: i64,
}

#[derive(Debug, Deserialize)]
pub struct TelegramUser {
    pub id: i64,
    #[serde(default)]
    pub first_name: Option<String>,
    #[serde(default)]
    pub username: Option<String>,
}

#[derive(Debug, Serialize)]
struct SendMessageRequest {
    chat_id: i64,
    text: String,
}

#[derive(Debug, Deserialize)]
struct TelegramResponse {
    ok: bool,
    #[serde(default)]
    result: serde_json::Value,
}

// ── Request builders (testable) ──────────────────────────────────────

/// Build the JSON body for an admin `/api/ask` request.
pub fn build_admin_ask_body(text: &str) -> serde_json::Value {
    serde_json::json!({ "message": text })
}

/// Build the `/api/ask` URL for the local AdapsisOS instance.
pub fn build_admin_ask_url(port: u16) -> String {
    format!("http://127.0.0.1:{port}/api/ask")
}

// ── Bot implementation ───────────────────────────────────────────────

pub struct TelegramBot {
    token: String,
    admin_chat_id: i64,
    api_port: u16,
    llm_url: String,
    llm_model: String,
    llm_api_key: Option<String>,
    http: reqwest::Client,
}

impl TelegramBot {
    pub fn new(
        token: String,
        admin_chat_id: i64,
        api_port: u16,
        llm_url: String,
        llm_model: String,
        llm_api_key: Option<String>,
    ) -> Self {
        Self {
            token,
            admin_chat_id,
            api_port,
            llm_url,
            llm_model,
            llm_api_key,
            http: reqwest::Client::new(),
        }
    }

    /// Run the long-polling loop. This never returns under normal operation.
    pub async fn run(&self) -> Result<()> {
        let mut offset: Option<i64> = None;
        eprintln!("[telegram] bot started, admin_chat_id={}", self.admin_chat_id);

        loop {
            match self.get_updates(offset).await {
                Ok(updates) => {
                    for update in updates {
                        if let Some(new_offset) = offset {
                            if update.update_id < new_offset {
                                continue;
                            }
                        }
                        offset = Some(update.update_id + 1);

                        if let Some(msg) = &update.message {
                            if let Some(text) = &msg.text {
                                self.handle_message(msg.chat.id, text).await;
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("[telegram] getUpdates error: {e}");
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                }
            }
        }
    }

    async fn get_updates(&self, offset: Option<i64>) -> Result<Vec<TelegramUpdate>> {
        let mut url = format!(
            "https://api.telegram.org/bot{}/getUpdates?timeout=30",
            self.token
        );
        if let Some(off) = offset {
            url.push_str(&format!("&offset={off}"));
        }

        let resp = self.http.get(&url)
            .timeout(std::time::Duration::from_secs(35))
            .send()
            .await?;

        let body: TelegramResponse = resp.json().await?;
        if !body.ok {
            return Err(anyhow!("Telegram API error: {:?}", body.result));
        }

        let updates: Vec<TelegramUpdate> = serde_json::from_value(body.result)?;
        Ok(updates)
    }

    async fn handle_message(&self, chat_id: i64, text: &str) {
        let route = classify_request(chat_id, self.admin_chat_id);
        eprintln!(
            "[telegram] chat_id={chat_id} route={route:?} text={}",
            text.chars().take(80).collect::<String>()
        );

        let reply = match route {
            TelegramRoute::AdminAsk => self.handle_admin(text).await,
            TelegramRoute::UserLlmCall => self.handle_user(text).await,
        };

        match reply {
            Ok(response) => {
                if let Err(e) = self.send_message(chat_id, &response).await {
                    eprintln!("[telegram] sendMessage error: {e}");
                }
            }
            Err(e) => {
                eprintln!("[telegram] handler error: {e}");
                let _ = self.send_message(chat_id, &format!("Error: {e}")).await;
            }
        }
    }

    /// Admin path: POST to local `/api/ask` for full session context.
    async fn handle_admin(&self, text: &str) -> Result<String> {
        let url = build_admin_ask_url(self.api_port);
        let body = build_admin_ask_body(text);

        let resp = self.http.post(&url)
            .json(&body)
            .timeout(std::time::Duration::from_secs(120))
            .send()
            .await
            .map_err(|e| anyhow!("Failed to reach AdapsisOS /api/ask: {e}"))?;

        let ask_resp: crate::api::AskResponse = resp.json().await
            .map_err(|e| anyhow!("Failed to parse /api/ask response: {e}"))?;

        // Build a readable reply from the AskResponse
        let mut parts = Vec::new();

        if !ask_resp.reply.is_empty() {
            parts.push(ask_resp.reply.clone());
        }

        if !ask_resp.code.is_empty() && ask_resp.code.trim() != "DONE" {
            parts.push(format!("```\n{}\n```", ask_resp.code));
        }

        let errors: Vec<&str> = ask_resp.results.iter()
            .filter(|r| !r.success)
            .map(|r| r.message.as_str())
            .collect();
        if !errors.is_empty() {
            parts.push(format!("Errors:\n{}", errors.join("\n")));
        }

        let test_summary: Vec<String> = ask_resp.test_results.iter()
            .map(|t| {
                let icon = if t.pass { "PASS" } else { "FAIL" };
                format!("{icon}: {}", t.message)
            })
            .collect();
        if !test_summary.is_empty() {
            parts.push(test_summary.join("\n"));
        }

        if parts.is_empty() {
            Ok("(no response)".to_string())
        } else {
            // Telegram has a 4096-char message limit
            let full = parts.join("\n\n");
            Ok(truncate_for_telegram(&full))
        }
    }

    /// Non-admin path: single LLM call, no session mutation.
    async fn handle_user(&self, text: &str) -> Result<String> {
        let llm = crate::llm::LlmClient::new_with_model_and_key(
            &self.llm_url,
            &self.llm_model,
            self.llm_api_key.clone(),
        );

        let messages = vec![
            crate::llm::ChatMessage::system(
                "You are a helpful AI assistant powered by AdapsisOS. \
                 Answer the user's question concisely."
            ),
            crate::llm::ChatMessage::user(text),
        ];

        let output = llm.generate(messages).await
            .map_err(|e| anyhow!("LLM call failed: {e}"))?;

        // Strip thinking tags and code blocks — just return the prose
        let mut clean = output.text.clone();
        while let Some(s) = clean.find("<think>") {
            if let Some(e) = clean.find("</think>") {
                clean.replace_range(s..e + 8, "");
            } else {
                break;
            }
        }
        while let Some(s) = clean.find("<code>") {
            if let Some(e) = clean.find("</code>") {
                clean.replace_range(s..e + 7, "");
            } else {
                break;
            }
        }
        let clean = clean.trim().to_string();

        if clean.is_empty() {
            Ok("(no response)".to_string())
        } else {
            Ok(truncate_for_telegram(&clean))
        }
    }

    async fn send_message(&self, chat_id: i64, text: &str) -> Result<()> {
        let url = format!(
            "https://api.telegram.org/bot{}/sendMessage",
            self.token
        );

        let body = SendMessageRequest {
            chat_id,
            text: text.to_string(),
        };

        let resp = self.http.post(&url)
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow!("sendMessage failed ({}): {}", status, body));
        }

        Ok(())
    }
}

/// Truncate text to fit Telegram's 4096-character message limit.
fn truncate_for_telegram(text: &str) -> String {
    const MAX_LEN: usize = 4000; // Leave some margin
    if text.len() <= MAX_LEN {
        text.to_string()
    } else {
        format!("{}...\n\n(truncated)", &text[..MAX_LEN])
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const ADMIN_CHAT_ID: i64 = 1815217;

    #[test]
    fn admin_routes_to_ask() {
        let route = classify_request(ADMIN_CHAT_ID, ADMIN_CHAT_ID);
        assert_eq!(route, TelegramRoute::AdminAsk);
    }

    #[test]
    fn non_admin_routes_to_llm_call() {
        let route = classify_request(9999999, ADMIN_CHAT_ID);
        assert_eq!(route, TelegramRoute::UserLlmCall);
    }

    #[test]
    fn different_non_admin_ids_all_route_to_llm_call() {
        for id in [1, 0, -1, 42, 1815218, 1815216, i64::MAX, i64::MIN] {
            let route = classify_request(id, ADMIN_CHAT_ID);
            assert_eq!(
                route,
                TelegramRoute::UserLlmCall,
                "chat_id {id} should route to UserLlmCall"
            );
        }
    }

    #[test]
    fn admin_ask_body_is_correct() {
        let body = build_admin_ask_body("hello world");
        assert_eq!(body["message"], "hello world");
        // Should be exactly one key
        let obj = body.as_object().unwrap();
        assert_eq!(obj.len(), 1);
    }

    #[test]
    fn admin_ask_url_uses_port() {
        let url = build_admin_ask_url(3001);
        assert_eq!(url, "http://127.0.0.1:3001/api/ask");
    }

    #[test]
    fn admin_ask_url_different_port() {
        let url = build_admin_ask_url(4096);
        assert_eq!(url, "http://127.0.0.1:4096/api/ask");
    }

    #[test]
    fn truncate_short_text_unchanged() {
        let text = "Hello, world!";
        assert_eq!(truncate_for_telegram(text), text);
    }

    #[test]
    fn truncate_long_text_gets_cut() {
        let text = "x".repeat(5000);
        let result = truncate_for_telegram(&text);
        assert!(result.len() < 4100);
        assert!(result.ends_with("(truncated)"));
    }

    #[test]
    fn classify_with_custom_admin_id() {
        // Verify the function works with any admin ID, not just hardcoded
        let custom_admin = 12345;
        assert_eq!(classify_request(12345, custom_admin), TelegramRoute::AdminAsk);
        assert_eq!(classify_request(12346, custom_admin), TelegramRoute::UserLlmCall);
    }

    #[test]
    fn admin_ask_body_preserves_special_chars() {
        let text = "hello\nworld\t\"quoted\" 日本語";
        let body = build_admin_ask_body(text);
        assert_eq!(body["message"].as_str().unwrap(), text);
    }
}
