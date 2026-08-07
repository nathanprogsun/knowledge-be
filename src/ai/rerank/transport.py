"""SSRF-safe HTTP plumbing for rerank clients.

Mirrors the upstream transport contract. ``validate_rerank_base_url``
runs the SSRF gate on the configured base URL (empty is allowed for
omitted/local endpoints). ``new_rerank_http_client`` builds an ``httpx``
client with redirect following disabled so every hop is re-validated
manually by :func:`post_json_with_ssrf_safety`: each ``Location`` target
re-runs the URL guard, the hop count is capped, and credential headers
are stripped when a redirect crosses to another origin.
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Mapping

import httpx

from src.common.exception import ExternalServiceError, ValidationError
from src.common.json import JsonValue
from src.common.oidc_client import validate_ssrf_safe_url

# Redirect budget, matching the upstream SSRF-safe client.
_MAX_REDIRECTS = 10

# Credential-bearing headers that must not follow a cross-origin redirect.
_REDIRECT_STRIPPED_HEADERS: frozenset[str] = frozenset(
    {
        "authorization",
        "cookie",
        "x-auth-token",
        "x-api-key",
        "api-key",
    }
)

# Default per-request timeout for rerank clients.
_DEFAULT_TIMEOUT_SECONDS = 30.0


async def validate_rerank_base_url(base_url: str) -> None:
    """Reject ``base_url`` that fails the SSRF guard.

    An empty value is allowed (the provider default applies); every
    non-empty value must pass the shared URL guard. Raises
    ``ValidationError`` with ``rerank.base_url_ssrf_blocked`` otherwise.
    """
    if not base_url:
        return
    try:
        await validate_ssrf_safe_url(base_url)
    except ValidationError as exc:
        raise ValidationError(
            code="rerank.base_url_ssrf_blocked",
            message="base URL SSRF check failed",
            details={"reason": exc.message},
        ) from exc


def new_rerank_http_client(
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> httpx.AsyncClient:
    """Build a rerank HTTP client without automatic redirect following.

    Redirects are re-validated by :func:`post_json_with_ssrf_safety`
    instead, so no hop bypasses the URL guard. Tests inject an
    ``httpx.MockTransport`` via ``transport=`` exactly as with the other
    HTTP adapters in this package.
    """
    if transport is not None:
        return httpx.AsyncClient(transport=transport, timeout=timeout, follow_redirects=False)
    return httpx.AsyncClient(timeout=timeout, follow_redirects=False)


def _same_http_origin(a: httpx.URL, b: httpx.URL) -> bool:
    """True when ``a`` and ``b`` share scheme and host (port-aware)."""
    return a.scheme.lower() == b.scheme.lower() and a.host.lower() == b.host.lower()


def _strip_redirect_sensitive_headers(headers: httpx.Headers) -> None:
    """Drop credential headers so they cannot leak on a cross-host hop."""
    for name in _REDIRECT_STRIPPED_HEADERS:
        if name in headers:
            del headers[name]


async def post_json_with_ssrf_safety(
    client: httpx.AsyncClient,
    url: str,
    *,
    json_body: dict[str, JsonValue] | None,
    headers: Mapping[str, str] | None,
) -> httpx.Response:
    """POST ``json_body`` to ``url``, re-validating every redirect hop.

    Mirrors the upstream redirect policy: cap the hop count, validate
    every redirect target against the SSRF guard (fail-closed), and
    strip credential headers when the redirect crosses origins.
    """
    request = httpx.Request("POST", url, json=json_body, headers=dict(headers or {}))
    for hop in range(_MAX_REDIRECTS + 1):
        await validate_rerank_base_url(str(request.url))
        try:
            response = await client.send(request)
        except httpx.HTTPError as exc:
            raise ExternalServiceError(
                code="rerank.request_failed",
                message=f"rerank request failed: {exc}",
            ) from exc
        if not response.is_redirect:
            return response
        if hop >= _MAX_REDIRECTS:
            raise ExternalServiceError(
                code="rerank.redirect_blocked",
                message=f"stopped after {_MAX_REDIRECTS} redirects",
            )
        location = response.headers.get("Location")
        if not location:
            raise ExternalServiceError(
                code="rerank.redirect_blocked",
                message="redirect response missing Location header",
            )
        next_url = urllib.parse.urljoin(str(request.url), location)
        next_headers = httpx.Headers(request.headers)
        if not _same_http_origin(request.url, httpx.URL(next_url)):
            _strip_redirect_sensitive_headers(next_headers)
        request = httpx.Request("POST", next_url, json=json_body, headers=next_headers)
    raise ExternalServiceError(
        code="rerank.redirect_blocked",
        message=f"stopped after {_MAX_REDIRECTS} redirects",
    )


__all__ = [
    "new_rerank_http_client",
    "post_json_with_ssrf_safety",
    "validate_rerank_base_url",
]
