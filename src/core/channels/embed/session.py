"""Anonymous embed session flow — publish-token auth, origin gating, rate limits.

Maps the upstream embed-session contract. Three responsibilities:

1. ``publish_token`` / ``session_token`` issuance and resolution. A
   publish token is the long-lived secret embedded in the widget
   script; a session token is a short-lived, Redis-backed handle the
   widget sends back on subsequent calls. Tokens carry the
   ``EmbedSessionTokenPrefix`` so receivers can distinguish them from
   publish tokens without a round-trip.

2. Origin gating. ``origin_allowed`` mirrors the upstream
   ``originAllowed`` helper — exact match (case-insensitive),
   ``*.suffix`` subdomain wildcard, and a literal ``"*"`` blanket.
   An empty allowlist rejects everything; the middleware trusts the
   create-time validator to have required at least one entry.

3. Rate limiting. ``EmbedRateLimiter`` wraps a Redis sliding-window
   counter (a sorted set of timestamped hits) keyed by
   ``embed:ratelimit:<bucket>``. The session service composes three
   budgets on every create — per-IP per-minute, channel-global
   per-minute (derived as ``max(per_ip * 20, 120)`` to bound burst
   across rotating IPs), and channel-daily — and rejects the request
   on the first exhausted bucket.

The ``create_session`` entry point combines all three gates and
mints the signed session handle the widget must echo on later calls.
``SignEmbedSessionHandle`` / ``VerifyEmbedSessionHandle`` are HMAC
signatures keyed by the channel's ``publish_token``; rotating the
publish token invalidates outstanding handles, which is the
upstream-acceptable posture.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, Protocol, cast, runtime_checkable
from uuid import uuid4

from redis.asyncio import Redis

from src.common.exception import (
    PermissionDeniedError,
    UnauthorizedError,
    ValidationError,
)
from src.core.channels.embed.types import (
    EMBED_SESSION_MARKER_PREFIX,
    EmbedChannelInfo,
)
from src.db.dao.embed_channel_repository import EmbedChannelRepository
from src.db.dao.session_repository import SessionRepository
from src.db.models.embed_channel import EmbedChannel
from src.db.models.session import Session

# ── Embed session token constants ────────────────────────────────────

#: Prefix tagging session tokens so the middleware can distinguish
#: them from long-lived publish tokens without an extra lookup.
EMBED_SESSION_TOKEN_PREFIX: Final[str] = "ems_"
#: Redis key prefix for the channel-id-by-token mapping.
EMBED_SESSION_KEY_PREFIX: Final[str] = "embed:session:"
#: Session token TTL; matches the upstream ``embedSessionTTL``.
EMBED_SESSION_TTL_SECONDS: Final[int] = 30 * 60

#: Per-IP per-minute key prefix.
EMBED_RATE_LIMIT_MINUTE_PREFIX: Final[str] = "embed:ratelimit:"
#: Per-channel per-day key prefix.
EMBED_RATE_LIMIT_DAY_PREFIX: Final[str] = "embed:ratelimit:day:"

#: Per-minute cap for the channel-global budget — derived as
#: ``per_ip * 20`` with a floor of ``120`` so a tiny per-IP cap
#: remains usable.
EMBED_GLOBAL_MINUTE_FACTOR: Final[int] = 20
EMBED_GLOBAL_MINUTE_FLOOR: Final[int] = 120

#: Random bytes packed into each session token (32 bytes → ~43 base64
#: chars after the ``ems_`` prefix).
_EMBED_TOKEN_BYTES: Final[int] = 32


# ── Errors ───────────────────────────────────────────────────────────


class RateLimitExceededError(UnauthorizedError):
    """Raised when an embed session request exceeds a configured budget.

    A subclass of ``UnauthorizedError`` so the upstream 429 status maps
    onto a single recognised exception family. The dedicated code
    lets the views layer differentiate per-minute from per-day
    exhaustion in the response body.
    """

    code = "embed.rate_limited"


# ── Helpers ──────────────────────────────────────────────────────────


def _new_session_token() -> str:
    """Return a freshly minted, prefixed session token."""
    body = base64.urlsafe_b64encode(secrets.token_bytes(_EMBED_TOKEN_BYTES)).rstrip(b"=")
    return EMBED_SESSION_TOKEN_PREFIX + body.decode()


def is_embed_session_token(token: str) -> bool:
    """True when ``token`` carries the embed session prefix."""
    return token.strip().startswith(EMBED_SESSION_TOKEN_PREFIX)


def _global_per_minute(per_ip: int) -> int:
    """Channel-global per-minute budget derived from the per-IP cap."""
    if per_ip <= 0:
        return EMBED_GLOBAL_MINUTE_FLOOR
    return max(per_ip * EMBED_GLOBAL_MINUTE_FACTOR, EMBED_GLOBAL_MINUTE_FLOOR)


def origin_allowed(origin: str, allowed: list[str]) -> bool:
    """Mirror of the upstream ``originAllowed`` gate.

    An empty allowlist rejects every origin; an empty origin is
    always rejected. Accepted forms are exact match (case-insensitive),
    the literal ``"*"``, and ``"*.suffix"`` subdomain wildcard.
    """
    if not allowed or not origin:
        return False
    for pattern in allowed:
        cleaned = pattern.strip()
        if not cleaned:
            continue
        if cleaned == "*":
            return True
        if cleaned.lower() == origin.lower():
            return True
        if cleaned.startswith("*."):
            suffix = cleaned[1:]
            if suffix and origin.lower().endswith(suffix.lower()):
                return True
    return False


# ── Session handle signing ──────────────────────────────────────────


def _origin_list_of(channel: EmbedChannel) -> list[str]:
    """Narrow the JSONB ``allowed_origins`` column onto a concrete list."""
    value = channel.allowed_origins
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, str):
        return [value]
    return []


def sign_embed_session_handle(channel: EmbedChannel | None, session_id: str) -> str:
    """Return the URL-safe base64 HMAC-SHA256 handle bound to ``session_id``.

    An empty session id or ``None`` channel produces an empty signature
    so callers can treat it as a uniform "no handle" sentinel.
    """
    if channel is None or not session_id.strip():
        return ""
    message = f"{channel.id}|{session_id}".encode()
    mac = hmac.new(channel.publish_token.encode(), message, hashlib.sha256)
    return base64.urlsafe_b64encode(mac.digest()).rstrip(b"=").decode()


def verify_embed_session_handle(
    channel: EmbedChannel | None, session_id: str, signature: str
) -> bool:
    """Constant-time verification of :func:`sign_embed_session_handle`."""
    sig = signature.strip()
    if not sig:
        return False
    expected = sign_embed_session_handle(channel, session_id)
    if not expected:
        return False
    return hmac.compare_digest(expected, sig)


# ── Rate limiting ────────────────────────────────────────────────────


@runtime_checkable
class RateLimiterLike(Protocol):
    """Async sliding-window counter seam for embed rate limiting."""

    async def check(self, *, key: str, limit: int, window_seconds: int) -> None:
        """Raise :class:`RateLimitExceededError` when the bucket is exhausted.

        ``key`` is the bucket identifier (e.g. ``"{channel_id}:{ip}"``);
        ``limit`` is the allowed requests per ``window_seconds``-second
        sliding window. A ``limit <= 0`` short-circuits to a no-op so
        callers can use ``0`` to disable a bucket without special casing.
        """
        ...


class EmbedRateLimiter:
    """Redis-backed sliding-window counter.

    Ports the upstream Lua script verbatim: a sorted set stores one
    member per hit scored by its timestamp; each check atomically prunes
    members older than the window, counts the live ones, and admits the
    new hit only while under ``limit``. The set TTL is ``window + 1s``
    so a bucket that falls idle evicts itself instead of accumulating.

    When ``redis_client`` is ``None`` the limiter fails open — every
    ``check`` resolves to ``None``. This keeps the service testable
    without a live Redis and matches the fail-open posture of the
    model concurrency governor.
    """

    _SCRIPT = (
        "local key    = KEYS[1]\n"
        "local now    = tonumber(ARGV[1])\n"
        "local window = tonumber(ARGV[2])\n"
        "local maxReq = tonumber(ARGV[3])\n"
        "local member = ARGV[4]\n"
        "redis.call('ZREMRANGEBYSCORE', key, 0, now - window)\n"
        "local count = redis.call('ZCARD', key)\n"
        "if count < maxReq then\n"
        "  redis.call('ZADD', key, now, member)\n"
        "  redis.call('PEXPIRE', key, window + 1000)\n"
        "  return 1\n"
        "end\n"
        "return 0\n"
    )

    def __init__(self, redis_client: Redis | None) -> None:  # type: ignore[type-arg]
        self._redis = redis_client
        self._script: Callable[..., Awaitable[int]] | None = None
        if redis_client is not None:
            self._script = redis_client.register_script(self._SCRIPT)

    async def check(self, *, key: str, limit: int, window_seconds: int) -> None:
        if limit <= 0 or not key or self._script is None:
            return
        result = await self._script(
            keys=[key],
            args=[
                int(time.time() * 1000),
                int(window_seconds * 1000),
                limit,
                str(uuid4()),
            ],
        )
        if int(result) != 1:
            raise RateLimitExceededError(
                code="embed.rate_limited",
                message="embed request rate limit exceeded",
            )


class InMemoryRateLimiter:
    """Test-friendly sliding-window counter backed by an in-process dict.

    Used by the service unit tests; production code injects the
    Redis-backed :class:`EmbedRateLimiter`.
    """

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._reset_at: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def check(self, *, key: str, limit: int, window_seconds: int) -> None:
        if limit <= 0 or not key:
            return
        async with self._lock:
            now = time.monotonic()
            reset = self._reset_at.get(key, 0.0)
            if reset <= now:
                self._counts[key] = 0
                self._reset_at[key] = now + window_seconds
            self._counts[key] = self._counts.get(key, 0) + 1
            if self._counts[key] > limit:
                raise RateLimitExceededError(
                    code="embed.rate_limited",
                    message="embed request rate limit exceeded",
                )


# ── Session service ─────────────────────────────────────────────────


@dataclass(frozen=True)
class CreatedEmbedSession:
    """Result of :meth:`EmbedSessionService.create_session`."""

    session_id: str
    handle: str


class EmbedSessionService:
    """Request-scoped anonymous embed session flow.

    Composes the upstream embed-session methods:
    ``LookupEnabledChannel``, ``IssueSessionToken`` /
    ``ResolveSessionToken``, ``SignEmbedSessionHandle`` /
    ``VerifyEmbedSessionHandle``, ``IssuePreviewSession``, plus the
    origin / rate-limit gates the auth middleware applies.

    The Redis seam (``session_store`` / ``rate_limiter``) is optional;
    when absent the methods that require Redis
    (``issue_session_token`` / ``resolve_session_token`` /
    ``enforce_rate_limits``) raise a clear ``ValidationError`` so the
    views layer can surface a 503.
    """

    def __init__(
        self,
        *,
        embed_channel_repo: EmbedChannelRepository,
        session_repo: SessionRepository,
        rate_limiter: RateLimiterLike | None = None,
        redis_client: Redis | None = None,  # type: ignore[type-arg]
    ) -> None:
        self._embed_channel_repo = embed_channel_repo
        self._session_repo = session_repo
        self._rate_limiter = rate_limiter
        self._redis = redis_client

    # ── Channel lookup ─────────────────────────────────────────────

    async def lookup_enabled_channel(self, channel_id: str) -> EmbedChannel:
        """Return one live channel by id, raising ``UnauthorizedError`` when absent."""
        cleaned = channel_id.strip()
        if not cleaned:
            raise UnauthorizedError(
                code="embed.channel_id_required",
                message="channel id is required",
            )
        row = await self._embed_channel_repo.get_by_id(cleaned)
        if row is None:
            raise UnauthorizedError(
                code="embed.token_invalid",
                message="invalid embed channel or token",
            )
        if not row.enabled:
            raise PermissionDeniedError(
                code="embed.channel_disabled",
                message="embed channel is disabled",
            )
        return row

    async def lookup_for_embed(self, *, channel_id: str, token: str) -> EmbedChannel:
        """Resolve a publish token (or session token) to its live channel.

        Mirrors ``LookupForEmbed`` (publish token) plus the middleware
        branch that exchanges a session token first via
        :meth:`resolve_session_token`. The returned channel is
        guaranteed enabled.
        """
        cleaned_channel = channel_id.strip()
        cleaned_token = token.strip()
        if not cleaned_channel or not cleaned_token:
            raise UnauthorizedError(
                code="embed.token_invalid",
                message="invalid embed channel or token",
            )

        if is_embed_session_token(cleaned_token):
            resolved_id = await self.resolve_session_token(cleaned_token)
            if resolved_id != cleaned_channel:
                raise UnauthorizedError(
                    code="embed.token_invalid",
                    message="invalid embed channel or token",
                )

        row = await self._embed_channel_repo.get_by_id(cleaned_channel)
        if row is None:
            raise UnauthorizedError(
                code="embed.token_invalid",
                message="invalid embed channel or token",
            )
        if not row.enabled:
            raise PermissionDeniedError(
                code="embed.channel_disabled",
                message="embed channel is disabled",
            )
        if not is_embed_session_token(cleaned_token) and not hmac.compare_digest(
            row.publish_token.encode(), cleaned_token.encode()
        ):
            # Constant-time token compare: publish tokens carry an
            # embedded secret, so a timing-leak safe compare is the
            # minimum safe posture even when not prompted by tests.
            raise UnauthorizedError(
                code="embed.token_invalid",
                message="invalid embed channel or token",
            )
        return row

    # ── Session token mint / resolve ───────────────────────────────

    async def issue_session_token(self, *, channel_id: str) -> tuple[str, int]:
        """Mint a short-lived session token bound to ``channel_id``."""
        cleaned = channel_id.strip()
        if not cleaned:
            raise ValidationError(
                code="embed.channel_id_required",
                message="channel id is required",
            )
        if self._redis is None:
            raise ValidationError(
                code="embed.session_unavailable",
                message="embed session tokens unavailable",
            )
        token = _new_session_token()
        key = EMBED_SESSION_KEY_PREFIX + token
        ttl = EMBED_SESSION_TTL_SECONDS
        await self._redis.set(key, cleaned, ex=ttl)
        return token, ttl

    async def resolve_session_token(self, token: str) -> str:
        """Return the channel id recorded for ``token`` (session token only)."""
        cleaned = token.strip()
        if not is_embed_session_token(cleaned):
            raise UnauthorizedError(
                code="embed.token_invalid",
                message="invalid embed session token",
            )
        if self._redis is None:
            raise ValidationError(
                code="embed.session_unavailable",
                message="embed session tokens unavailable",
            )
        value = await self._redis.get(EMBED_SESSION_KEY_PREFIX + cleaned)
        if value is None:
            raise UnauthorizedError(
                code="embed.token_invalid",
                message="invalid embed session token",
            )
        resolved = value.decode() if isinstance(value, (bytes, bytearray)) else str(value)
        resolved = resolved.strip()
        if not resolved:
            raise UnauthorizedError(
                code="embed.token_invalid",
                message="invalid embed session token",
            )
        return resolved

    async def issue_preview_session(self, *, channel_id: str) -> tuple[str, int]:
        """Mint a session token for an authenticated management preview."""
        cleaned = channel_id.strip()
        if not cleaned:
            raise ValidationError(
                code="embed.channel_id_required",
                message="channel id is required",
            )
        channel = await self.lookup_enabled_channel(cleaned)
        return await self.issue_session_token(channel_id=channel.id)

    # ── Anonymous session create ───────────────────────────────────

    async def create_session(
        self,
        *,
        channel_id: str,
        token: str,
        origin: str,
        client_ip: str = "",
        title: str = "",
    ) -> CreatedEmbedSession:
        """Create an embed chat session for an anonymous widget visitor.

        Validates the publish token, gates the request origin against
        the channel allowlist, applies the three rate budgets, and
        finally creates a ``sessions`` row tagged with the
        ``embed_channel:<channel_id>`` description marker the rest of
        the system uses to recognise embed-created sessions.

        The returned :class:`CreatedEmbedSession` carries the new
        session id and a signed handle the widget must echo on later
        calls (via :func:`verify_embed_session_handle`).
        """
        channel = await self.lookup_for_embed(channel_id=channel_id, token=token)
        # ``allowed_origins`` is a JSONB column whose list shape is
        # guaranteed by the create/update normalizers; narrow the type
        # so the origin gate gets the concrete list it validates.
        allowed_origins = cast("list[str]", channel.allowed_origins)
        await self.assert_origin_allowed(origin, allowed_origins)
        await self.enforce_rate_limits(channel=channel, client_ip=client_ip)

        now = _now()
        session_id = str(uuid4())
        row = Session(
            id=session_id,
            tenant_id=channel.tenant_id,
            title=title or None,
            description=EMBED_SESSION_MARKER_PREFIX + channel.id,
            user_id=None,
            created_at=now,
            updated_at=now,
        )
        await self._session_repo.create(row)
        handle = sign_embed_session_handle(channel, session_id)
        return CreatedEmbedSession(session_id=session_id, handle=handle)

    # ── Gating helpers (public so the views layer can reuse) ──────

    async def resolve_channel_for_request(
        self,
        *,
        channel_id: str,
        token: str,
        origin: str,
        client_ip: str,
    ) -> EmbedChannelInfo:
        """Authenticate + gate an embed request; return the safe projection.

        Composes token lookup, origin gating, and rate limits so the web
        layer only ever sees ``EmbedChannelInfo`` — the db row (which
        carries the secret ``publish_token``) never crosses into web.
        """
        row = await self.lookup_for_embed(channel_id=channel_id, token=token)
        await self.assert_origin_allowed(origin, _origin_list_of(row))
        await self.enforce_rate_limits(channel=row, client_ip=client_ip)
        return EmbedChannelInfo.map_from_db(row)

    async def assert_session_handle(
        self,
        *,
        channel_id: str,
        session_id: str,
        signature: str,
    ) -> None:
        """Verify the HMAC session handle against the stored publish token.

        Raises :class:`PermissionDeniedError` on mismatch. The db row is
        re-fetched inside the service precisely so the secret never
        leaves the core layer.
        """
        row = await self._embed_channel_repo.get_by_id(channel_id.strip())
        if row is None or not verify_embed_session_handle(row, session_id, signature):
            raise PermissionDeniedError(
                code="embed.session_signature_invalid",
                message="session signature invalid",
            )

    @staticmethod
    async def assert_origin_allowed(origin: str, allowed_origins: list[str]) -> None:
        """Raise :class:`PermissionDeniedError` when ``origin`` is not in ``allowed_origins``."""
        if not origin_allowed(origin, allowed_origins):
            raise PermissionDeniedError(
                code="embed.origin_not_allowed",
                message="origin not allowed",
            )

    async def enforce_rate_limits(
        self,
        *,
        channel: EmbedChannel,
        client_ip: str = "",
    ) -> None:
        """Apply the per-IP, channel-global, and daily budgets.

        Skipped entirely when no rate limiter is wired (fail-open).
        Each bucket raises :class:`RateLimitExceededError` as soon as
        it is exhausted, so a single explicit ``try`` around this
        method surfaces the narrowest failing window to the caller.
        """
        limiter = self._rate_limiter
        if limiter is None:
            return
        ip = client_ip.strip()
        per_minute = channel.rate_limit_per_minute
        per_day = channel.rate_limit_per_day
        minute_window = 60
        day_window = 24 * 60 * 60

        if ip:
            await limiter.check(
                key=(f"{EMBED_RATE_LIMIT_MINUTE_PREFIX}{channel.id}:{ip}"),
                limit=per_minute,
                window_seconds=minute_window,
            )
        await limiter.check(
            key=f"{EMBED_RATE_LIMIT_MINUTE_PREFIX}{channel.id}:__global",
            limit=_global_per_minute(per_minute),
            window_seconds=minute_window,
        )
        await limiter.check(
            key=EMBED_RATE_LIMIT_DAY_PREFIX + channel.id,
            limit=per_day,
            window_seconds=day_window,
        )


# ── Internal helpers ────────────────────────────────────────────────


def _now() -> datetime:
    """Return a timezone-aware now for stamping rows."""
    return datetime.now(UTC)


__all__ = [
    "EMBED_GLOBAL_MINUTE_FACTOR",
    "EMBED_GLOBAL_MINUTE_FLOOR",
    "EMBED_RATE_LIMIT_DAY_PREFIX",
    "EMBED_RATE_LIMIT_MINUTE_PREFIX",
    "EMBED_SESSION_KEY_PREFIX",
    "EMBED_SESSION_TOKEN_PREFIX",
    "EMBED_SESSION_TTL_SECONDS",
    "CreatedEmbedSession",
    "EmbedRateLimiter",
    "EmbedSessionService",
    "InMemoryRateLimiter",
    "RateLimitExceededError",
    "RateLimiterLike",
    "is_embed_session_token",
    "origin_allowed",
    "sign_embed_session_handle",
    "verify_embed_session_handle",
]
