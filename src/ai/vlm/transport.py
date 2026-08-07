"""Shared HTTP transport for the vision-language clients.

Maps the upstream transport helpers: base-URL SSRF validation and the
HTTP client factory used by every OpenAI-compatible backend. Redirects
are disabled at the client level - the base URL is validated at
construction, and a redirect response fails the request instead of
following an unvalidated host (fail-closed).
"""

from __future__ import annotations

import os
from typing import Final

import httpx

from src.common.exception import ValidationError
from src.common.oidc_client import validate_ssrf_safe_url

# Dense scanned-PDF OCR (full-page text + layout extraction) can take well
# over a minute on slow endpoints, so the default timeout is intentionally
# generous and can be raised further via VLM_HTTP_TIMEOUT_SECONDS.
DEFAULT_TIMEOUT_SECONDS: Final = 180.0
VLM_HTTP_TIMEOUT_ENV: Final = "VLM_HTTP_TIMEOUT_SECONDS"


def vlm_http_timeout() -> float:
    """Return the HTTP client timeout for VLM requests.

    Reads ``VLM_HTTP_TIMEOUT_SECONDS`` when set (and positive), falling
    back to :data:`DEFAULT_TIMEOUT_SECONDS`.
    """
    raw = os.environ.get(VLM_HTTP_TIMEOUT_ENV, "").strip()
    if raw:
        try:
            seconds = int(raw)
        except ValueError:
            seconds = 0
        if seconds > 0:
            return float(seconds)
    return DEFAULT_TIMEOUT_SECONDS


async def validate_vlm_base_url(base_url: str) -> None:
    """Reject a base URL that is not SSRF-safe; empty URLs pass.

    Mirrors the upstream guard: an empty base URL is allowed (callers
    validate presence separately) while a non-empty URL must survive the
    shared SSRF check.
    """
    if not base_url:
        return
    try:
        await validate_ssrf_safe_url(base_url)
    except ValidationError as exc:
        raise ValidationError(
            f"base URL SSRF check failed: {exc.message}",
            code="vlm.base_url_ssrf_blocked",
        ) from exc


def new_vlm_http_client(
    timeout: float,
    *,
    custom_headers: dict[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """Create the HTTP client shared by all VLM backends.

    ``custom_headers`` are attached to every request (extra_headers
    semantics); ``transport`` is injected by tests.
    """
    headers: dict[str, str] | None = dict(custom_headers) if custom_headers else None
    if transport is not None:
        return httpx.AsyncClient(
            transport=transport,
            timeout=timeout,
            headers=headers,
            follow_redirects=False,
        )
    return httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=False)


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "VLM_HTTP_TIMEOUT_ENV",
    "new_vlm_http_client",
    "validate_vlm_base_url",
    "vlm_http_timeout",
]
