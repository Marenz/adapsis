use super::task_status::{host_only, strip_and_describe};
use anyhow::{bail, ensure, Context, Result};

const MAX_ERROR_BYTES: usize = 64 * 1024;
const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;
const MAX_HEADERS_BYTES: usize = 64 * 1024;
const MAX_URL_BYTES: usize = 8 * 1024;

fn request_method(method: &str) -> Result<reqwest::Method> {
    match method.trim().to_ascii_uppercase().as_str() {
        "GET" => Ok(reqwest::Method::GET),
        "POST" => Ok(reqwest::Method::POST),
        "PUT" => Ok(reqwest::Method::PUT),
        "DELETE" => Ok(reqwest::Method::DELETE),
        "PATCH" => Ok(reqwest::Method::PATCH),
        other => bail!(
            "http_request: unsupported method `{other}`; expected GET, POST, PUT, DELETE, or PATCH"
        ),
    }
}

pub(super) async fn perform(
    method: String,
    url: String,
    headers: String,
    body: String,
) -> Result<String> {
    ensure!(
        url.len() <= MAX_URL_BYTES,
        "http_request: URL exceeds the {MAX_URL_BYTES}-byte limit"
    );
    ensure!(
        headers.len() <= MAX_HEADERS_BYTES,
        "http_request: headers exceed the {MAX_HEADERS_BYTES}-byte limit"
    );
    // #116: errors name the host only; the URL (bot tokens, userinfo, query secrets) is
    // never echoed, not even in the invalid-URL and wrong-scheme messages (a scheme-less
    // `bot123:TOKEN` parses with scheme `bot123`, so the scheme is not echoed either).
    let target = host_only(&url);
    let parsed_url = reqwest::Url::parse(&url).context("http_request: invalid URL")?;
    ensure!(
        matches!(parsed_url.scheme(), "http" | "https"),
        "http_request: URL must use http or https"
    );
    let method = request_method(&method)?;
    let method_name = method.as_str().to_string();
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .context("http_request: could not build HTTP client")?;
    let mut request = client.request(method.clone(), parsed_url);
    for (index, line) in headers.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (name, value) = line.split_once(':').ok_or_else(|| {
            anyhow::anyhow!(
                "http_request: malformed header on line {}; expected `Name: Value`",
                index + 1
            )
        })?;
        let name =
            reqwest::header::HeaderName::from_bytes(name.trim().as_bytes()).map_err(|_| {
                anyhow::anyhow!("http_request: invalid header name on line {}", index + 1)
            })?;
        let value = value
            .trim()
            .parse::<reqwest::header::HeaderValue>()
            .map_err(|_| {
                anyhow::anyhow!("http_request: invalid header value on line {}", index + 1)
            })?;
        request = request.header(name, value);
    }
    if method != reqwest::Method::GET && !body.is_empty() {
        request = request.body(body);
    }
    let mut response = request.send().await.map_err(|error| {
        anyhow::anyhow!(
            "http_request {method_name} {target} failed: {}",
            strip_and_describe(error).1
        )
    })?;
    let status = response.status();
    let limit = if status.is_success() {
        MAX_RESPONSE_BYTES
    } else {
        MAX_ERROR_BYTES
    };
    let mut bytes = Vec::new();
    let mut truncated = false;
    while let Some(chunk) = response.chunk().await.map_err(|error| {
        anyhow::anyhow!(
            "http_request {method_name} {target} could not read the response: {}",
            strip_and_describe(error).1
        )
    })? {
        let remaining = limit.saturating_sub(bytes.len());
        if chunk.len() > remaining {
            bytes.extend_from_slice(&chunk[..remaining]);
            truncated = true;
            break;
        }
        bytes.extend_from_slice(&chunk);
    }
    let mut response_body = String::from_utf8_lossy(&bytes).into_owned();
    if !status.is_success() {
        if truncated {
            response_body.push_str(" [response body truncated]");
        }
        if response_body.is_empty() {
            bail!("http_request {method_name} {target} returned HTTP {status}");
        }
        bail!("http_request {method_name} {target} returned HTTP {status}: {response_body}");
    }
    ensure!(
        !truncated,
        "http_request {method_name} {target} response exceeded the {MAX_RESPONSE_BYTES}-byte limit"
    );
    Ok(response_body)
}
