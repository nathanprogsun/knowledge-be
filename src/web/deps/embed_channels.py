"""Embed-channel FastAPI dependency factories.

Forwarders to ``src.core.channels.embed.factory``: the per-request CRUD
service, the anonymous embed session flow, and the webhook dispatcher
are assembled on the shared ``AsyncSession`` (``web`` never imports
``db``).

The Redis-backed seams (short-lived session-token store and the
sliding-window rate limiter) are not wired at the app level yet, so the
session service is built in the fail-open / unavailable mode: rate
limiting no-ops and session-token issuance raises
``embed.session_unavailable`` (mapped to a 5xx). Tests replace these
services with fakes via ``app.dependency_overrides``.

``get_embed_channel`` / ``require_embed_session`` are the embed-auth
dependencies the public widget surface uses instead of the user-auth
``AuthDep``:

- ``get_embed_channel`` resolves the publish (or session) token from the
  ``Authorization: Embed <token>`` header, gates the request ``Origin``
  against the channel allowlist, applies the channel rate budgets, and
  stashes the resolved row on ``request.state.embed_channel``.
- ``require_embed_session`` re-verifies the per-visitor signed handle
  (``X-Embed-Session`` header) minted at session creation — the
  authorization secret that binds a caller to one embed session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Header, Request

from src.app_context import request_context
from src.common.exception import (
    PermissionDeniedError,
    ValidationError,
)
from src.core.channels.embed.factory import (
    build_embed_channel_service,
    build_embed_session_service,
    build_embed_webhook_dispatcher,
)
from src.core.channels.embed.service.embed_channel_service import (
    EmbedChannelService,
)
from src.core.channels.embed.session import (
    EmbedSessionService,
    verify_embed_session_handle,
)
from src.core.channels.embed.webhook import EmbedWebhookDispatcher
from src.core.chat.factory import build_chat_service
from src.core.chat.messages.factory import build_message_service
from src.core.chat.messages.service.message_service import MessageServiceImpl
from src.core.chat.pipeline.types import Context
from src.core.chat.service import ChatService
from src.core.knowledge.chunks.factory import build_chunk_service
from src.core.knowledge.chunks.service.chunk_service import ChunkService
from src.db.models.embed_channel import EmbedChannel
from src.web.deps.session import SessionDep


def get_embed_channel_service(session: SessionDep) -> EmbedChannelService:
    """Build a per-request ``EmbedChannelService`` on the shared session."""
    return build_embed_channel_service(session)


def get_embed_session_service(session: SessionDep) -> EmbedSessionService:
    """Build the per-request anonymous embed session flow.

    The Redis-backed seams are deferred: no app-scope Redis client is
    wired yet, so the service runs in the fail-open / unavailable mode
    (rate limits no-op; session-token issuance raises
    ``embed.session_unavailable``).
    """
    return build_embed_session_service(session)


def get_embed_webhook_dispatcher() -> EmbedWebhookDispatcher:
    """Build the embed webhook dispatcher (holds no DB / session state)."""
    return build_embed_webhook_dispatcher()


def get_embed_chat_service(
    request: Request,
    session: SessionDep,
) -> ChatService:
    """Build the chat pipeline for an anonymous embed visitor.

    The caller's tenant and channel come from the resolved embed channel
    (``request.state.embed_channel``, populated by ``get_embed_channel``);
    the visitor id is the synthetic embed-session principal so message /
    session scoping behaves like the upstream embed flow. ``user_id``
    falls back to the channel-scoped principal when the session gate has
    not run (chat routes always declare the session gate first).
    """
    channel = _require_state_channel(request)
    session_id = _state_session_id(request)
    principal = f"embed_session:{channel.tenant_id}:{channel.id}"
    if session_id:
        principal = f"{principal}:{session_id}"
    return build_chat_service(
        session,
        tenant_id=channel.tenant_id,
        user_id=principal,
        request_id=request_context.get_request_id() or "",
    )


def get_embed_message_service(session: SessionDep) -> MessageServiceImpl:
    """Build the per-request message service on the shared session."""
    return build_message_service(session)


@dataclass(frozen=True, slots=True)
class _EmbedMessageContext:
    """Minimal pipeline ``Context`` carrying the embed channel's tenant.

    The message service reads ``tenant_id`` for its session-existence
    checks; the other pipeline fields are protocol-narrow placeholders
    the chat-history seams never look at.
    """

    tenant_id: int


def get_embed_message_context(request: Request) -> Context:
    """Return the pipeline ``Context`` scoped to the embed channel's tenant."""
    channel = _require_state_channel(request)
    return _EmbedMessageContext(tenant_id=channel.tenant_id)


def get_embed_chunk_service(session: SessionDep) -> ChunkService:
    """Build the per-request chunk service on the shared session."""
    return build_chunk_service(session)


def _require_state_channel(request: Request) -> EmbedChannel:
    """Read the resolved embed channel from ``request.state``, or fail."""
    channel = getattr(request.state, "embed_channel", None)
    if not isinstance(channel, EmbedChannel):
        raise ValidationError(
            code="embed.channel_context_missing",
            message="embed channel context missing",
        )
    return channel


def _state_session_id(request: Request) -> str:
    """Return the embed session id stashed by ``require_embed_session``."""
    return str(getattr(request.state, "embed_session_id", "") or "")


def extract_embed_token(authorization: str | None) -> str:
    """Return the token from an ``Authorization: Embed <token>`` header.

    An absent, malformed, or non-``Embed`` header yields an empty string
    so callers can distinguish "no token" from a valid one.
    """
    if not authorization:
        return ""
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].strip() != "Embed":
        return ""
    return parts[1].strip()


def _origin_list(channel: EmbedChannel) -> list[str]:
    """Narrow the JSONB ``allowed_origins`` column onto a concrete list."""
    value = channel.allowed_origins
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, str):
        return [value]
    return []


async def get_embed_channel(
    request: Request,
    channel_id: str,
    session_service: EmbedSessionServiceDep,
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> EmbedChannel:
    """Resolve the embed channel for a publish-token-authenticated request.

    Mirrors the upstream embed-auth middleware: token lookup (accepting
    either a long-lived publish token or a short-lived session token),
    origin gating, and the per-IP / channel / daily rate budgets. The
    resolved row is also stashed on ``request.state`` so downstream
    dependencies (chat / message services) can read the channel's
    tenant without re-resolving it.
    """
    channel = await session_service.lookup_for_embed(
        channel_id=channel_id,
        token=extract_embed_token(authorization),
    )
    origin = request.headers.get("origin", "")
    await session_service.assert_origin_allowed(origin, _origin_list(channel))
    client_ip = request.client.host if request.client is not None else ""
    await session_service.enforce_rate_limits(
        channel=channel,
        client_ip=client_ip,
    )
    request.state.embed_channel = channel
    request.state.embed_tenant_id = str(channel.tenant_id)
    return channel


async def require_embed_session(
    request: Request,
    session_id: str,
    channel: EmbedChannelDep,
    x_embed_session: str | None = Header(default=None, alias="X-Embed-Session"),
) -> EmbedChannel:
    """Gate a session-scoped embed route on the signed visitor handle.

    ``X-Embed-Session`` is the HMAC handle minted at session creation
    (keyed by the channel publish token). Knowing the session id alone —
    e.g. from a leaked access log — is insufficient without the matching
    signature, so a bad handle is rejected with 403.
    """
    cleaned = session_id.strip()
    if not cleaned:
        raise ValidationError(
            code="embed.session_id_required",
            message="session_id is required",
        )
    signature = (x_embed_session or "").strip()
    if not verify_embed_session_handle(channel, cleaned, signature):
        raise PermissionDeniedError(
            code="embed.session_signature_invalid",
            message="session signature invalid",
        )
    request.state.embed_session_id = cleaned
    return channel


EmbedChannelServiceDep = Annotated[
    EmbedChannelService, Depends(get_embed_channel_service)
]
EmbedSessionServiceDep = Annotated[
    EmbedSessionService, Depends(get_embed_session_service)
]
EmbedWebhookDispatcherDep = Annotated[
    EmbedWebhookDispatcher, Depends(get_embed_webhook_dispatcher)
]
EmbedChannelDep = Annotated[EmbedChannel, Depends(get_embed_channel)]
EmbedSessionDep = Annotated[EmbedChannel, Depends(require_embed_session)]
EmbedChatServiceDep = Annotated[ChatService, Depends(get_embed_chat_service)]
EmbedMessageServiceDep = Annotated[
    MessageServiceImpl, Depends(get_embed_message_service)
]
EmbedMessageContextDep = Annotated[Context, Depends(get_embed_message_context)]
EmbedChunkServiceDep = Annotated[ChunkService, Depends(get_embed_chunk_service)]

__all__ = [
    "EmbedChannelDep",
    "EmbedChannelServiceDep",
    "EmbedChatServiceDep",
    "EmbedChunkServiceDep",
    "EmbedMessageContextDep",
    "EmbedMessageServiceDep",
    "EmbedSessionDep",
    "EmbedSessionServiceDep",
    "EmbedWebhookDispatcherDep",
    "extract_embed_token",
    "get_embed_channel",
    "get_embed_channel_service",
    "get_embed_chat_service",
    "get_embed_chunk_service",
    "get_embed_message_context",
    "get_embed_message_service",
    "get_embed_session_service",
    "get_embed_webhook_dispatcher",
    "require_embed_session",
]
