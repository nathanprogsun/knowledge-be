"""Webhook callback plumbing for embed channels.

Mirrors the upstream embed-webhook contract: every channel may declare
an outbound ``webhook_url`` plus an optional ``webhook_secret``. When
an embed session event fires (the web view layer is responsible for
the trigger), the service signs the JSON body with HMAC-SHA256 using
the channel's secret and POSTs it to the configured URL with the
``X-Embed-Webhook-Signature`` header.

The upstream dispatcher is fire-and-forget; this module preserves that
shape via :meth:`EmbedWebhookDispatcher.dispatch`, which schedules an
``asyncio`` task and returns it so callers (and tests) can ``await`` the
in-flight POST.

Three pure helpers back the dispatcher and the inbound callback handler
(``verify_embed_webhook_signature``): URL validation, body signing, and
constant-time signature verification. SSRF safety is delegated to the
factory-injected ``url_safety_check`` so unit tests can pin the gate
without DNS resolution; production wires the shared
:func:`src.common.oidc_client.validate_ssrf_safe_url` hook.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Final
from urllib.parse import urlparse

import httpx

from src.common.exception import ValidationError
from src.common.json import JsonValue
from src.db.models.embed_channel import EmbedChannel

#: HMAC-SHA256 hex digest; the on-wire header value is
#: ``"sha256=<hex>"`` to mirror the upstream signature envelope while
#: staying free of the upstream brand literal in code.
SIGNATURE_HEADER_NAME: Final[str] = "X-Embed-Webhook-Signature"
SIGNATURE_PREFIX: Final[str] = "sha256="

#: Per-call HTTP timeout. Matches the upstream ``embedWebhookTimeout``.
DEFAULT_WEBHOOK_TIMEOUT: Final[float] = 5.0


# ── Signature helpers ────────────────────────────────────────────────


def sign_embed_webhook_body(secret: str, raw: bytes) -> str:
    """Return the hex HMAC-SHA256 of ``raw`` keyed by ``secret``.

    An empty ``secret`` returns an empty digest (the dispatcher omits
    the signature header in that case, mirroring the upstream guard).
    """
    if not secret:
        return ""
    return hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()


def _normalize_signature(signature: str) -> str:
    """Strip the ``sha256=`` envelope so a bare hex digest is accepted too."""
    cleaned = signature.strip()
    if cleaned.startswith(SIGNATURE_PREFIX):
        return cleaned[len(SIGNATURE_PREFIX):]
    return cleaned


def verify_embed_webhook_signature(
    secret: str, raw: bytes, signature: str
) -> bool:
    """Constant-time check that ``signature`` matches the HMAC of ``raw``.

    Accepts either the on-wire form (``"sha256=<hex>"``) or a bare hex
    digest. Returns ``False`` for an empty secret / signature.
    """
    if not secret or not signature:
        return False
    expected = sign_embed_webhook_body(secret, raw)
    if not expected:
        return False
    return hmac.compare_digest(expected, _normalize_signature(signature))


# ── URL validation ───────────────────────────────────────────────────


async def validate_embed_webhook_url(
    raw: str,
    *,
    url_safety_check: Callable[[str], Awaitable[None]] | None = None,
) -> None:
    """Reject malformed embed webhook URLs.

    An empty URL is allowed (the channel simply never fires events).
    Non-empty values must be ``http`` or ``https`` with a non-empty
    host. When ``url_safety_check`` is provided, it is awaited with the
    trimmed URL and is expected to raise ``ValidationError`` on unsafe
    hosts — production wires the shared SSRF guard here.
    """
    cleaned = raw.strip()
    if not cleaned:
        return
    parsed = urlparse(cleaned)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValidationError(
            code="embed.webhook_url_invalid",
            message="webhook URL must use http or https",
        )
    if not parsed.hostname:
        raise ValidationError(
            code="embed.webhook_url_invalid",
            message="webhook URL must include a host",
        )
    if url_safety_check is not None:
        await url_safety_check(cleaned)


# ── Dispatcher ──────────────────────────────────────────────────────


class EmbedWebhookDispatcher:
    """Best-effort async webhook dispatcher with HMAC signing.

    Constructed per request with an injected ``http_client`` (the
    factory wires the shared ``httpx.AsyncClient``). ``dispatch``
    validates the channel URL, serialises the event, signs the body,
    and schedules the POST as a fire-and-forget task — mirroring the
    upstream goroutine dispatch.

    The task is returned so tests (and the small set of callers that
    care about completion) can ``await`` it; production callers
    typically ignore the return value.
    """

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient | None = None,
        url_safety_check: Callable[[str], Awaitable[None]] | None = None,
        timeout: float = DEFAULT_WEBHOOK_TIMEOUT,
    ) -> None:
        self._http_client = http_client
        self._owns_http_client = http_client is None
        self._client_to_close: httpx.AsyncClient | None = None
        if self._owns_http_client:
            self._client_to_close = httpx.AsyncClient(timeout=timeout)
        self._url_safety_check = url_safety_check

    async def aclose(self) -> None:
        """Close the dispatcher-owned HTTP client (no-op when injected)."""
        client = self._client_to_close
        if client is not None:
            await client.aclose()
            self._client_to_close = None

    async def validate_url(self, raw: str) -> None:
        """Expose :func:`validate_embed_webhook_url` with the injected SSRF hook."""
        await validate_embed_webhook_url(
            raw, url_safety_check=self._url_safety_check
        )

    def dispatch(
        self,
        channel: EmbedChannel,
        *,
        event_type: str,
        session_id: str,
        payload: Mapping[str, JsonValue],
    ) -> asyncio.Task[None]:
        """Schedule a signed POST to ``channel.webhook_url``.

        Returns the scheduled task. Empty webhook URLs short-circuit
        to a completed no-op task so the caller never needs a None
        branch. Dispatch errors are swallowed inside the task — the
        upstream behaviour is best-effort with a warning log.
        """
        url = channel.webhook_url.strip()
        if not url:
            return asyncio.create_task(_noop())

        body_bytes = _build_event_body(
            channel=channel,
            event_type=event_type,
            session_id=session_id,
            payload=payload,
        )
        secret = channel.webhook_secret.strip()
        return asyncio.create_task(
            self._send(
                url=url,
                body=body_bytes,
                secret=secret,
            ),
            name=f"embed-webhook-{channel.id}-{event_type}",
        )

    async def _send(
        self, *, url: str, body: bytes, secret: str
    ) -> None:
        """Sign and POST the webhook body; surface no error to the caller."""
        client = self._http_client or self._client_to_close
        if client is None:
            return
        try:
            await validate_embed_webhook_url(
                url, url_safety_check=self._url_safety_check
            )
            headers = {"Content-Type": "application/json"}
            if secret:
                headers[SIGNATURE_HEADER_NAME] = (
                    SIGNATURE_PREFIX + sign_embed_webhook_body(secret, body)
                )
            await client.post(url, content=body, headers=headers)
        except Exception:
            # Best-effort: the upstream logs and swallows every dispatch
            # failure. The task simply resolves to ``None`` so callers
            # can ``await`` it without an exception handler.
            return


async def _noop() -> None:
    """Stand-in for the no-dispatch path so the returned task is awaitable."""


def _build_event_body(
    *,
    channel: EmbedChannel,
    event_type: str,
    session_id: str,
    payload: Mapping[str, JsonValue],
) -> bytes:
    """Serialise the webhook envelope as a compact UTF-8 JSON body."""
    body: dict[str, JsonValue] = {
        "type": event_type,
        "channel_id": channel.id,
        "session_id": session_id,
        "timestamp": _utc_now_iso(),
    }
    for key, value in payload.items():
        body[key] = value
    return json.dumps(body, separators=(",", ":"), sort_keys=True).encode()


def _utc_now_iso() -> str:
    """RFC3339 UTC timestamp without fractional seconds (matches the upstream format)."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "DEFAULT_WEBHOOK_TIMEOUT",
    "SIGNATURE_HEADER_NAME",
    "SIGNATURE_PREFIX",
    "EmbedWebhookDispatcher",
    "sign_embed_webhook_body",
    "validate_embed_webhook_url",
    "verify_embed_webhook_signature",
]
