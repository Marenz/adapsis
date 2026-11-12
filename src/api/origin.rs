//! Where an HTTP request came from.
//!
//! The HTTP API has no authentication: whoever can reach a bound address can
//! use it. [`Origin`] is the one fact the server itself knows about a request,
//! the address of its TCP peer, reduced to "this node" or "somewhere else".
//!
//! DESIGN N3 makes node identity the WireGuard address plus a per-node
//! profile. That profile does not exist yet, so `Remote` carries the address
//! and nothing maps it to authority: a request from another node has no
//! authority we can name. The per-node profile is the place that will map
//! `Remote(ip)` to a profile; until then handlers that need to know may only
//! ask "is it local".
use axum::extract::{ConnectInfo, FromRequestParts};
use axum::http::request::Parts;
use std::net::{IpAddr, SocketAddr};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// The TCP peer is a loopback address and the request was not forwarded.
    Local,
    /// Anything else. The address is the TCP peer's (unspecified if unknown).
    Remote(IpAddr),
}

/// Headers a reverse proxy adds. A request that carries one did not originate
/// where its TCP peer is, so it is never local: a proxy on this node (Caddy
/// forwards to 127.0.0.1) would otherwise make every proxied request local.
/// The header's VALUE is never believed: it cannot make a request local.
const FORWARDING_HEADERS: &[&str] = &["forwarded", "x-forwarded-for", "x-real-ip"];

impl Origin {
    /// Classify a request by its TCP peer and its headers.
    pub fn of(peer: Option<IpAddr>, headers: &axum::http::HeaderMap) -> Self {
        let Some(peer) = peer else { return Self::Remote(IpAddr::from([0, 0, 0, 0])) };
        // `::ffff:127.0.0.1` is the IPv4 loopback seen through an IPv6 socket.
        let canonical = match peer {
            IpAddr::V6(v6) => v6.to_ipv4_mapped().map_or(peer, IpAddr::V4),
            v4 => v4,
        };
        let forwarded = FORWARDING_HEADERS.iter().any(|name| headers.contains_key(*name));
        if canonical.is_loopback() && !forwarded { Self::Local } else { Self::Remote(canonical) }
    }
}

/// Extractor. It never rejects: without connection information (a router
/// served without `into_make_service_with_connect_info`) the origin is remote.
impl<S: Send + Sync> FromRequestParts<S> for Origin {
    type Rejection = std::convert::Infallible;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        let peer = parts.extensions.get::<ConnectInfo<SocketAddr>>().map(|info| info.0.ip());
        Ok(Self::of(peer, &parts.headers))
    }
}

/// `/api/ask` with the request's origin recorded in the turn's configuration.
pub async fn ask(
    origin: Origin,
    axum::extract::State(mut config): axum::extract::State<super::AppConfig>,
    request: axum::Json<super::AskRequest>,
) -> axum::Json<super::AskResponse> {
    config.origin = Some(origin);
    super::llm_handlers::ask(axum::extract::State(config), request).await
}

/// `/api/ask-stream` with the request's origin recorded.
pub async fn ask_stream(
    origin: Origin,
    axum::extract::State(mut config): axum::extract::State<super::AppConfig>,
    request: axum::Json<super::AskRequest>,
) -> impl axum::response::IntoResponse {
    config.origin = Some(origin);
    super::llm_handlers::ask_stream(axum::extract::State(config), request).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{HeaderMap, HeaderValue};

    fn ip(text: &str) -> Option<IpAddr> {
        Some(text.parse().unwrap())
    }

    #[test]
    fn only_an_unforwarded_loopback_peer_is_local() {
        let none = HeaderMap::new();
        for local in ["127.0.0.1", "127.8.9.10", "::1", "::ffff:127.0.0.1", "::ffff:127.255.0.3"] {
            assert_eq!(Origin::of(ip(local), &none), Origin::Local, "{local}");
        }
        for remote in ["10.0.0.4", "192.168.1.201", "0.0.0.0", "::", "fd00::4", "::ffff:10.0.0.4", "128.0.0.1", "::2"] {
            assert!(matches!(Origin::of(ip(remote), &none), Origin::Remote(_)), "{remote}");
        }
        assert_eq!(Origin::of(ip("::ffff:10.0.0.4"), &none), Origin::Remote("10.0.0.4".parse().unwrap()));
        assert!(matches!(Origin::of(None, &none), Origin::Remote(_)), "an unknown peer is not local");
        // A forwarding header never widens and always narrows.
        for header in ["forwarded", "x-forwarded-for", "x-real-ip", "X-Forwarded-For"] {
            for value in ["127.0.0.1", "for=127.0.0.1", "::1", "10.0.0.4", ""] {
                let mut headers = HeaderMap::new();
                headers.insert(axum::http::HeaderName::from_bytes(header.as_bytes()).unwrap(), HeaderValue::from_str(value).unwrap());
                // The Caddy case: a loopback peer that forwards someone's request.
                assert_eq!(Origin::of(ip("127.0.0.1"), &headers), Origin::Remote("127.0.0.1".parse().unwrap()), "{header}: {value}");
                // A remote peer claiming to be local.
                assert_eq!(Origin::of(ip("10.0.0.4"), &headers), Origin::Remote("10.0.0.4".parse().unwrap()), "{header}: {value}");
            }
        }
    }
}
