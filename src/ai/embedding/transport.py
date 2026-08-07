"""SSRF-safe HTTP transport for embedding providers (transport.go).

``validate_embedding_base_url`` rejects a resolved embedding API base URL
that is not safe for outbound requests (empty URLs are allowed — callers
apply provider defaults). ``apply_custom_headers`` attaches user-supplied
custom request headers while skipping the reserved ones (``Authorization``,
``Content-Type``, ...) so custom values cannot break auth.
``post_embedding_with_retry`` posts a JSON body retrying transport errors
with exponential backoff — the shared body of every provider's upstream
``doRequestWithRetry`` loop.

The upstream transport pools a single SSRF-safe connection transport
process-wide; here each client owns its ``httpx.AsyncClient`` and, unlike
the upstream redirect-following transport, redirects are not followed so
a 3xx response surfaces as an error instead of an unchecked hop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import httpx

from src.common.exception import AIProviderError, ValidationError
from src.common.json import JsonValue
from src.common.oidc_client import validate_ssrf_safe_url

_MAX_BACKOFF_SECONDS = 10.0

# Reserved request headers that user custom headers must never override
# (the upstream ``ApplyCustomHeaders`` skip set). These are controlled by
# the provider's auth / signing / SSE flow.
_RESERVED_HEADERS: frozenset[str] = frozenset(
    {
        "authorization",
        "api-key",
        "x-api-key",
        "x-goog-api-key",
        "content-type",
        "content-length",
        "accept-encoding",
        "host",
        "connection",
        "transfer-encoding",
    }
)


def is_reserved_header(key: str) -> bool:
    """True when ``key`` is a reserved header that custom headers skip."""
    return key.strip().lower() in _RESERVED_HEADERS


def apply_custom_headers(
    headers: dict[str, str],
    custom: Mapping[str, str] | None,
) -> None:
    """Merge ``custom`` into ``headers``, skipping reserved names.

    Mirrors the upstream ``ApplyCustomHeaders``: reserved headers are
    skipped to avoid breaking auth / signing; any other name overrides
    the same-name default.
    """
    if not custom:
        return
    for key, value in custom.items():
        name = key.strip()
        if not name or is_reserved_header(name):
            continue
        headers[name] = value


async def post_embedding_with_retry(
    http_client: httpx.AsyncClient,
    url: str,
    payload: Mapping[str, JsonValue],
    headers: Mapping[str, str],
    max_retries: int,
) -> httpx.Response:
    """POST ``payload`` to ``url`` retrying transport errors with backoff.

    Mirrors the upstream ``doRequestWithRetry`` loops: only transport
    errors (``httpx.HTTPError``) are retried, with exponential backoff
    capped at ``_MAX_BACKOFF_SECONDS``; a non-2xx response is returned to
    the caller so each provider surfaces its own error body. When every
    attempt fails with a transport error, ``AIProviderError`` is raised.
    """
    last_error: httpx.HTTPError | None = None
    for attempt in range(max_retries + 1):
        if attempt > 0:
            await asyncio.sleep(min(2 ** (attempt - 1), _MAX_BACKOFF_SECONDS))
        try:
            return await http_client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise AIProviderError(
            f"send request: {last_error}",
            code="embedding.request_failed",
        ) from last_error
    raise AIProviderError(
        "send request: unknown error",
        code="embedding.request_failed",
    )


async def validate_embedding_base_url(base_url: str) -> None:
    """Reject a base URL that fails the SSRF safety check.

    Empty URLs are allowed (providers apply their own default endpoint).
    Raises ``ValidationError`` when the URL is unsafe.
    """
    if not base_url:
        return
    try:
        await validate_ssrf_safe_url(base_url)
    except ValidationError as exc:
        raise ValidationError(
            code="embedding.base_url_ssrf_blocked",
            message="base URL SSRF check failed",
            details={"reason": exc.message},
        ) from exc


def new_embedding_http_client(
    *,
    timeout: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """Return an outbound client for one embedding provider.

    ``transport`` is injectable so tests can supply an
    ``httpx.MockTransport`` without touching the network. Redirects are
    not followed (see the module docstring).
    """
    return httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=False,
        transport=transport,
    )


__all__ = [
    "apply_custom_headers",
    "is_reserved_header",
    "new_embedding_http_client",
    "post_embedding_with_retry",
    "validate_embedding_base_url",
]
