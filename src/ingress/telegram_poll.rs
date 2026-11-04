use super::*;
use serde::Deserialize;
use serde_json::{Value, value::RawValue};
use sha2::{Digest, Sha256};

pub(super) struct Event {
    pub id: i64,
    pub raw: String,
    pub digest: String,
}

#[derive(Deserialize)]
struct Response {
    ok: bool,
    result: Box<RawValue>,
}

pub(super) fn digest(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

pub(super) fn source_identity(config: &SourceConfig) -> Result<String, IngressError> {
    if config.id.is_empty()
        || config.id.len() > 64
        || !qualified(&config.handler)
        || config.bot_id <= 0
        || config.seed < 0
        || config.credential.is_empty()
        || config.credential.len() > 256
        || config.routes.len() > 8
    {
        return Err(IngressError::Protocol);
    }
    let mut seen = std::collections::HashSet::new();
    for route in &config.routes {
        if route.sender_id <= 0
            || route.context.is_empty()
            || route.context.len() > 128
            || ![
                format!("telegram:{}", route.sender_id),
                format!("telegram:user:{}", route.sender_id),
            ]
            .contains(&route.context)
            || !qualified(&route.reply_handler)
            || route.reply_handler.len() > 128
            || !seen.insert(route.sender_id)
        {
            return Err(IngressError::Protocol);
        }
    }
    // Credential identity is fenced without storing credential bytes in the ledger.
    let mut routes = config.routes.clone();
    routes.sort_by_key(|route| route.sender_id);
    let value = serde_json::to_string(&(
        &config.id,
        &config.handler,
        config.bot_id,
        &routes,
        config.seed,
        digest(&config.credential),
    ))
    .map_err(|_| IngressError::Protocol)?;
    Ok(digest(&value))
}

fn qualified(name: &str) -> bool {
    let Some((module, member)) = name.split_once('.') else {
        return false;
    };
    [module, member].iter().all(|part| {
        let mut chars = part.chars();
        chars
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
    }) && name.len() <= 128
}

fn endpoint(config: &SourceConfig, method: &str) -> Result<reqwest::Url, IngressError> {
    #[cfg(not(test))]
    let mut url =
        reqwest::Url::parse("https://api.telegram.org").map_err(|_| IngressError::Protocol)?;
    #[cfg(test)]
    let mut url = config.endpoint.clone();
    url.path_segments_mut()
        .map_err(|_| IngressError::Protocol)?
        .push(&format!("bot{}", config.credential))
        .push(method);
    Ok(url)
}

async fn request(client: &reqwest::Client, url: reqwest::Url) -> Result<Response, IngressError> {
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|_| IngressError::Transport)?;
    if !response.status().is_success() {
        return Err(IngressError::Protocol);
    }
    use futures::StreamExt;
    let mut stream = response.bytes_stream();
    let mut bytes = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_| IngressError::Transport)?;
        if bytes.len() + chunk.len() > 512 * 1024 {
            return Err(IngressError::Oversized);
        }
        bytes.extend_from_slice(&chunk);
    }
    let decoded: Response = serde_json::from_slice(&bytes).map_err(|_| IngressError::Protocol)?;
    if !decoded.ok {
        return Err(IngressError::Protocol);
    }
    Ok(decoded)
}

pub(super) async fn fetch(config: &SourceConfig, cursor: i64) -> Result<Vec<Event>, IngressError> {
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|_| IngressError::Transport)?;
    let me = request(&client, endpoint(config, "getMe")?).await?;
    let bot = strict_json::parse(me.result.get())?;
    if bot.get("id").and_then(Value::as_i64) != Some(config.bot_id)
        || bot.get("is_bot").and_then(Value::as_bool) != Some(true)
    {
        return Err(IngressError::WrongBot);
    }
    let mut url = endpoint(config, "getUpdates")?;
    url.query_pairs_mut()
        .append_pair("offset", &cursor.to_string())
        .append_pair("limit", &PAGE_LIMIT.to_string())
        .append_pair("timeout", "0");
    let updates = request(&client, url).await?;
    let raw: Vec<Box<RawValue>> =
        serde_json::from_str(updates.result.get()).map_err(|_| IngressError::Protocol)?;
    if raw.len() > PAGE_LIMIT {
        return Err(IngressError::Oversized);
    }
    let events = raw
        .into_iter()
        .map(|raw| parse_event(raw.get()))
        .collect::<Result<Vec<_>, _>>()?;
    for pair in events.windows(2) {
        if pair[1].id < pair[0].id {
            return Err(IngressError::Protocol);
        }
        if pair[1].id == pair[0].id && pair[1].digest != pair[0].digest {
            return Err(IngressError::AlteredReplay);
        }
    }
    Ok(events)
}

pub(super) fn parse_event(raw: &str) -> Result<Event, IngressError> {
    if raw.len() > EVENT_BYTES {
        return Err(IngressError::Oversized);
    }
    let value = strict_json::parse(raw)?;
    let id = value
        .get("update_id")
        .and_then(Value::as_i64)
        .filter(|id| *id >= 0 && *id < i64::MAX)
        .ok_or(IngressError::Protocol)?;
    Ok(Event {
        id,
        raw: raw.to_string(),
        digest: digest(raw),
    })
}

pub(super) fn private_text(
    config: &SourceConfig,
    event: &Event,
    epoch: i64,
) -> Result<Option<VerifiedIngress>, IngressError> {
    let value: Value = serde_json::from_str(&event.raw).map_err(|_| IngressError::Protocol)?;
    let message = &value["message"];
    if message["chat"]["type"].as_str() != Some("private") {
        return Ok(None);
    }
    let Some(sender) = message["from"]["id"].as_i64().filter(|id| *id > 0) else {
        return Ok(None);
    };
    if message["chat"]["id"].as_i64() != Some(sender) {
        return Ok(None);
    }
    let Some(text) = message["text"].as_str() else {
        return Ok(None);
    };
    if text.is_empty()
        || text.chars().count() > 4096
        || message["from"]["is_bot"].as_bool() != Some(false)
        || message["message_id"].as_i64().is_none_or(|id| id <= 0)
        || message["date"].as_i64().is_none_or(|date| date < 0)
    {
        return Ok(None);
    }
    if [
        "photo",
        "video",
        "voice",
        "audio",
        "document",
        "animation",
        "sticker",
        "location",
        "contact",
    ]
    .iter()
    .any(|key| message.get(key).is_some())
    {
        return Ok(None);
    }
    let Some(route) = config.routes.iter().find(|route| route.sender_id == sender) else {
        return Ok(None);
    };
    Ok(Some(VerifiedIngress {
        source: config.id.clone(),
        update_id: event.id,
        epoch,
        digest: event.digest.clone(),
        principal: format!("telegram:user:{sender}"),
        context: route.context.clone(),
        text: text.into(),
        reply_handler: route.reply_handler.clone(),
        reply_arg: sender.to_string(),
    }))
}
