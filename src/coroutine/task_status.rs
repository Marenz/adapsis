//! #112: the one rendering of a task's wait reason that every status surface uses
//! (`GET /api/tasks`, `?tasks`, `query_tasks`, `?inspect task N`, snapshot wait reason).
//!
//! A wait reason carries the raw operation argument so the runtime can act on it, but its
//! `Display` must never print one: URLs hold bot tokens (`/bot<id>:<token>/…`), userinfo and
//! query secrets; paths and shell commands routinely embed credentials; takeover contexts
//! name private conversations. Rendering is therefore structural, not pattern-based:
//! HTTP operations show scheme and host only, everything else shows the operation name.
//! Stored values stay intact (callback acknowledgement reads them); only rendering is
//! redacted. Raw access, if ever needed, must be an explicitly named accessor.
use reqwest::Url;

/// `scheme://host[:port]` of an http(s) URL, or `<invalid url>`. Userinfo, path, query and
/// fragment are never reproduced, and unparsable input is never echoed back. (`Url::parse`
/// itself strips surrounding whitespace; the `None` host arm is defensive, since http(s)
/// URLs always parse with a host.)
pub(super) fn host_only(raw: &str) -> String {
    match Url::parse(raw) {
        Ok(url) if matches!(url.scheme(), "http" | "https") => match url.host_str() {
            Some(host) => match url.port() {
                Some(port) => format!("{}://{host}:{port}", url.scheme()),
                None => format!("{}://{host}", url.scheme()),
            },
            None => "<invalid url>".to_string(),
        },
        _ => "<invalid url>".to_string(),
    }
}
