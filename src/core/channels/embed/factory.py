"""Embed-channel domain request-scoped service factory.

Follows the ``src.core.organizations.factory`` pattern: repositories and
services are built per request on the shared ``AsyncSession``; ``web``
never imports ``db``. The embed CRUD service, the anonymous embed
session flow, and the webhook dispatcher are wired in one place so a
request's reads and writes share a single transactional unit of work.

Redis-backed seams (session-token store and rate limiter) are optional:
pass ``redis_client`` to enable short-lived session tokens and
fixed-window rate limiting, or ``None`` for the fail-open / unavailable
path.
"""

from __future__ import annotations

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.oidc_client import validate_ssrf_safe_url
from src.core.agents.service.custom_agent_service import CustomAgentService
from src.core.agents.service.factory import build_custom_agent_service
from src.core.channels.embed.service.embed_channel_service import (
    AgentOwnershipLike,
    EmbedChannelService,
    _CustomAgentAdapter,
)
from src.core.channels.embed.session import (
    EmbedRateLimiter,
    EmbedSessionService,
)
from src.core.channels.embed.webhook import EmbedWebhookDispatcher
from src.db.dao.embed_channel_repository import EmbedChannelRepository
from src.db.dao.session_repository import SessionRepository


def build_embed_channel_service(
    session: AsyncSession,
) -> EmbedChannelService:
    """Build a per-request embed CRUD service with fresh repositories."""
    return EmbedChannelService(
        repo=EmbedChannelRepository(session),
        agent_ownership=_agent_ownership(session),
    )


def build_embed_session_service(
    session: AsyncSession,
    *,
    redis_client: Redis | None = None,  # type: ignore[type-arg]
) -> EmbedSessionService:
    """Build the per-request anonymous embed session flow.

    ``redis_client`` is the app-scope ``redis.asyncio.Redis`` instance
    (or ``None`` to run in the unavailable / fail-open mode). The rate
    limiter wraps the same client; an in-memory limiter would not
    enforce cross-process budgets, so production always passes a real
    client.
    """
    return EmbedSessionService(
        embed_channel_repo=EmbedChannelRepository(session),
        session_repo=SessionRepository(session),
        rate_limiter=(EmbedRateLimiter(redis_client) if redis_client is not None else None),
        redis_client=redis_client,
    )


def build_embed_webhook_dispatcher() -> EmbedWebhookDispatcher:
    """Build the webhook dispatcher (holds no DB / session state).

    Production wires the shared SSRF guard so every dispatched webhook
    URL is re-validated before the POST fires.
    """
    return EmbedWebhookDispatcher(url_safety_check=validate_ssrf_safe_url)


def _agent_ownership(session: AsyncSession) -> AgentOwnershipLike:
    """Wrap the per-request :class:`CustomAgentService` as the seam."""
    custom_agent_service: CustomAgentService = build_custom_agent_service(session)
    return _CustomAgentAdapter(custom_agent_service)


__all__ = [
    "build_embed_channel_service",
    "build_embed_session_service",
    "build_embed_webhook_dispatcher",
]
