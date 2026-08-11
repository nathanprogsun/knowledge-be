"""Embed-channel HTTP endpoints - admin CRUD and the anonymous embed surface.

Registered by the app factory.

Three routers mirror the upstream route split:

- ``agents_router``  - ``/agents/{agent_id}/embed-channels`` CRUD (create / list)
- ``router``         - ``/embed-channels`` management (list-all / get / update /
  delete / rotate-token / preview-session / stats)
- ``public_router``  - ``/embed/{channel_id}/...`` anonymous widget surface

The admin endpoints require user auth (``AuthDep``) plus the upstream
role floor (Admin for mutations, Viewer for reads). The public endpoints
are authenticated by the embed publish token (``EmbedChannelDep``) and
deliberately take **no** user-auth dependency - they must work for
anonymous widget visitors. Session-scoped public routes additionally
require the signed visitor handle (``EmbedSessionDep``).

Capabilities the upstream handler delegates to not-yet-wired surfaces
(session stop, MCP OAuth, tool approvals, file serving) keep their
routes and return a clear ``embed.capability_unavailable`` error so the
wire surface stays faithful while the execution seams land in later
PRs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated, cast

from fastapi import (
    APIRouter,
    Depends,
    Header,
    Query,
    Request,
    Response,
)
from fastapi.responses import StreamingResponse

from src.common.exception import (
    ExternalServiceError,
    NotFoundError,
    PermissionDeniedError,
    UnauthorizedError,
    ValidationError,
)
from src.common.json import JsonValue
from src.core.channels.embed.session import is_embed_session_token
from src.core.chat.bus import Event
from src.core.chat.service import (
    AgentQARequestLike,
    ChatService,
    KnowledgeQARequestLike,
)
from src.core.chat.sessions.service.session_service import SessionListQuery
from src.core.contracts.sessions import (
    EnsureSuggestionsRequest,
    LoadMessagesQuery,
    SuggestionEventRequest,
)
from src.web.api.channels.embed.views import (
    EmbedChannelEnvelope,
    EmbedChannelListEnvelope,
    EmbedChannelRequest,
    EmbedChunkEnvelope,
    EmbedConfigEnvelope,
    EmbedSessionCreateData,
    EmbedSessionCreateEnvelope,
    EmbedSessionTokenData,
    EmbedSessionTokenEnvelope,
    EmbedStatsData,
    EmbedStatsEnvelope,
    EmbedSuggestedQuestionsData,
    EmbedSuggestedQuestionsEnvelope,
    EmbedSuggestionSuppressedData,
    EmbedSuggestionSuppressedEnvelope,
    EmbedWebhookAckResponse,
    EmbedWebhookEventRequest,
    SimpleSuccessResponse,
    clamp_suggestion_limit,
    embed_channel_record,
    embed_channel_record_from_row,
    embed_public_config,
    patch_embed_chat_payload,
    to_create_request,
    to_update_request,
    validate_allowed_origins,
)
from src.web.api.chat.messages.views import (
    MessageLoadEnvelope,
    SuggestionEnvelope,
    message_load_envelope,
    suggestion_envelope,
)
from src.web.api.chat.views import (
    CreateKnowledgeQARequest,
    StreamResponse,
    format_sse_frame,
    to_stream_response,
)
from src.web.deps import (
    AuthDep,
    MessageSuggestionServiceDep,
    RoleAdminDep,
    RoleViewerDep,
    SessionServiceDep,
)
from src.web.deps.context import get_tenant_id_dep
from src.web.deps.embed_channels import (
    EmbedChannelDep,
    EmbedChannelServiceDep,
    EmbedChatServiceDep,
    EmbedChunkServiceDep,
    EmbedMessageContextDep,
    EmbedMessageServiceDep,
    EmbedSessionDep,
    EmbedSessionServiceDep,
    EmbedWebhookDispatcherDep,
    extract_embed_token,
)

# Function-arg-style principal dep: the authenticated workspace id.
_PrincipalTenant = Annotated[int, Depends(get_tenant_id_dep)]

_DEFAULT_LOAD_LIMIT = 20
_SSE_MEDIA_TYPE = "text/event-stream"
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _require_tenant(tenant_id: int) -> int:
    """Return the active workspace id, or fail closed.

    Channel management is workspace-scoped; without a workspace context
    there is no safe default, so this rejects rather than guessing.
    """
    if tenant_id == 0:
        raise UnauthorizedError(
            code="embed.tenant_context_missing",
            message="unauthorized: workspace context missing",
        )
    return tenant_id


def _clamp_load_limit(limit: int) -> int:
    """Coerce a message-load limit onto ``[1, inf)`` (fallback 20)."""
    return limit if limit >= 1 else _DEFAULT_LOAD_LIMIT


def _parse_before_time(raw: str) -> datetime:
    """Parse an RFC3339 / RFC3339Nano cursor (upstream ``parseMessageBeforeTime``)."""
    stripped = raw.strip()
    try:
        return datetime.fromisoformat(stripped)
    except ValueError as exc:
        raise ValidationError(
            code="embed.invalid_before_time",
            message="Invalid time format, please use RFC3339 or RFC3339Nano format",
        ) from exc


def _capability_unavailable(message: str) -> ExternalServiceError:
    """Return the standard error for a not-yet-wired embed capability.

    The upstream handler returns a 5xx when the delegated handler is nil;
    this build maps the same gap to the closest server-side error family.
    """
    return ExternalServiceError(
        code="embed.capability_unavailable",
        message=message,
    )


# ── Admin: agent-scoped embed channels ────────────────────────────────


agents_router = APIRouter(prefix="/agents", tags=["embed-channels"])


@agents_router.post(
    "/{agent_id}/embed-channels",
    response_model=EmbedChannelEnvelope,
    status_code=201,
)
async def create_embed_channel(
    _auth: AuthDep,
    _role: RoleAdminDep,
    agent_id: str,
    body: EmbedChannelRequest,
    service: EmbedChannelServiceDep,
    tenant_id: _PrincipalTenant,
) -> EmbedChannelEnvelope:
    """Create an embed channel for an agent; admin only.

    The origin allowlist must contain at least one well-formed entry (a
    public widget with no allowlist is rejected, matching the upstream
    validator).
    """
    tid = _require_tenant(tenant_id)
    validate_allowed_origins(body.allowed_origins)
    info, token = await service.create_channel(
        tenant_id=tid,
        agent_id=agent_id,
        request=to_create_request(body),
    )
    return EmbedChannelEnvelope(
        success=True,
        data=embed_channel_record(info, publish_token=token),
    )


@agents_router.get(
    "/{agent_id}/embed-channels",
    response_model=EmbedChannelListEnvelope,
)
async def list_embed_channels(
    _auth: AuthDep,
    _role: RoleViewerDep,
    agent_id: str,
    service: EmbedChannelServiceDep,
    tenant_id: _PrincipalTenant,
) -> EmbedChannelListEnvelope:
    """List every live embed channel of one agent; viewer or above."""
    tid = _require_tenant(tenant_id)
    infos = await service.list_channels_by_agent(tenant_id=tid, agent_id=agent_id)
    return EmbedChannelListEnvelope(
        success=True,
        data=[embed_channel_record(info) for info in infos],
    )


# ── Admin: tenant-wide embed channels ─────────────────────────────────


router = APIRouter(prefix="/embed-channels", tags=["embed-channels"])


@router.get("", response_model=EmbedChannelListEnvelope)
async def list_all_embed_channels(
    _auth: AuthDep,
    _role: RoleViewerDep,
    service: EmbedChannelServiceDep,
    tenant_id: _PrincipalTenant,
) -> EmbedChannelListEnvelope:
    """List every embed channel of the workspace, across agents.

    Publish tokens are never included (sidebar session grouping surface).
    """
    tid = _require_tenant(tenant_id)
    infos = await service.list_channels_by_tenant(tenant_id=tid)
    return EmbedChannelListEnvelope(
        success=True,
        data=[embed_channel_record(info) for info in infos],
    )


@router.get("/{channel_id}", response_model=EmbedChannelEnvelope)
async def get_embed_channel(
    _auth: AuthDep,
    _role: RoleViewerDep,
    channel_id: str,
    service: EmbedChannelServiceDep,
    tenant_id: _PrincipalTenant,
) -> EmbedChannelEnvelope:
    """Return one channel, including its publish token for deploy snippets."""
    tid = _require_tenant(tenant_id)
    row = await service.get_owned_channel(tenant_id=tid, channel_id=channel_id)
    return EmbedChannelEnvelope(
        success=True,
        data=embed_channel_record_from_row(row),
    )


@router.put("/{channel_id}", response_model=EmbedChannelEnvelope)
async def update_embed_channel(
    _auth: AuthDep,
    _role: RoleAdminDep,
    channel_id: str,
    body: EmbedChannelRequest,
    service: EmbedChannelServiceDep,
    tenant_id: _PrincipalTenant,
) -> EmbedChannelEnvelope:
    """Update a channel's mutable fields; admin only.

    ``None`` fields mean "leave unchanged". The origin allowlist is
    re-validated only when the caller intends to change it; the webhook
    URL is validated by the service.
    """
    tid = _require_tenant(tenant_id)
    if body.allowed_origins is not None:
        validate_allowed_origins(body.allowed_origins)
    info = await service.update_channel(
        tenant_id=tid,
        channel_id=channel_id,
        request=to_update_request(body),
    )
    return EmbedChannelEnvelope(success=True, data=embed_channel_record(info))


@router.delete("/{channel_id}", response_model=SimpleSuccessResponse)
async def delete_embed_channel(
    _auth: AuthDep,
    _role: RoleAdminDep,
    channel_id: str,
    service: EmbedChannelServiceDep,
    tenant_id: _PrincipalTenant,
) -> SimpleSuccessResponse:
    """Soft-delete a channel; admin only."""
    tid = _require_tenant(tenant_id)
    await service.delete_channel(tenant_id=tid, channel_id=channel_id)
    return SimpleSuccessResponse()


@router.post("/{channel_id}/rotate-token", response_model=EmbedChannelEnvelope)
async def rotate_embed_token(
    _auth: AuthDep,
    _role: RoleAdminDep,
    channel_id: str,
    service: EmbedChannelServiceDep,
    tenant_id: _PrincipalTenant,
) -> EmbedChannelEnvelope:
    """Mint a fresh publish token; outstanding visitor handles invalidate."""
    tid = _require_tenant(tenant_id)
    info, token = await service.rotate_token(tenant_id=tid, channel_id=channel_id)
    return EmbedChannelEnvelope(
        success=True,
        data=embed_channel_record(info, publish_token=token),
    )


@router.post(
    "/{channel_id}/preview-session",
    response_model=EmbedSessionTokenEnvelope,
)
async def issue_preview_session(
    _auth: AuthDep,
    _role: RoleViewerDep,
    channel_id: str,
    session_service: EmbedSessionServiceDep,
    tenant_id: _PrincipalTenant,
) -> EmbedSessionTokenEnvelope:
    """Mint a short-lived session token for an authenticated management preview."""
    _require_tenant(tenant_id)
    session_token, expires_in = await session_service.issue_preview_session(
        channel_id=channel_id,
    )
    return EmbedSessionTokenEnvelope(
        success=True,
        data=EmbedSessionTokenData(
            session_token=session_token,
            expires_in=expires_in,
        ),
    )


@router.get("/{channel_id}/stats", response_model=EmbedStatsEnvelope)
async def get_embed_channel_stats(
    _auth: AuthDep,
    _role: RoleViewerDep,
    channel_id: str,
    service: EmbedChannelServiceDep,
    session_service: SessionServiceDep,
    tenant_id: _PrincipalTenant,
) -> EmbedStatsEnvelope:
    """Return lightweight usage stats for one channel.

    The count is the result of the tenant-wide source query the upstream
    stats handler issues; the repository-level source filter is a
    deferred seam, so today the total reflects the workspace session
    page rather than the channel-only subset.
    """
    tid = _require_tenant(tenant_id)
    await service.get_owned_channel(tenant_id=tid, channel_id=channel_id)
    result = await session_service.list_with_filters(
        SessionListQuery(
            source=f"embed:{channel_id}",
            page=1,
            page_size=1,
        )
    )
    return EmbedStatsEnvelope(
        success=True,
        data=EmbedStatsData(session_count=result.total),
    )


# ── Public: anonymous embed widget surface ────────────────────────────


public_router = APIRouter(prefix="/embed", tags=["embed"])


@public_router.post("/{channel_id}/exchange", response_model=EmbedSessionTokenEnvelope)
async def exchange_embed_session(
    channel: EmbedChannelDep,
    session_service: EmbedSessionServiceDep,
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> EmbedSessionTokenEnvelope:
    """Exchange a publish token for a short-lived session token.

    Only the long-lived publish token may mint session tokens; accepting
    a session token here would let a holder renew it indefinitely
    without re-presenting the publish token.
    """
    token = extract_embed_token(authorization)
    if not token or is_embed_session_token(token):
        raise PermissionDeniedError(
            code="embed.publish_token_required",
            message="publish token required",
        )
    try:
        session_token, expires_in = await session_service.issue_session_token(
            channel_id=channel.id,
        )
    except ValidationError as exc:
        if exc.code == "embed.session_unavailable":
            raise ExternalServiceError(
                code="embed.session_unavailable",
                message="session tokens unavailable",
            ) from exc
        raise
    return EmbedSessionTokenEnvelope(
        success=True,
        data=EmbedSessionTokenData(
            session_token=session_token,
            expires_in=expires_in,
        ),
    )


@public_router.get("/{channel_id}/config", response_model=EmbedConfigEnvelope)
async def get_embed_config(
    channel: EmbedChannelDep,
) -> EmbedConfigEnvelope:
    """Return the public widget config (no secrets)."""
    return EmbedConfigEnvelope(success=True, data=embed_public_config(channel))


@public_router.get(
    "/{channel_id}/suggested-questions",
    response_model=EmbedSuggestedQuestionsEnvelope,
)
async def get_embed_suggested_questions(
    channel: EmbedChannelDep,
    limit: int = Query(default=0, description="Cap 12; 0 means channel default"),
) -> EmbedSuggestedQuestionsEnvelope:
    """Return channel-level starter questions for the widget.

    A channel with suggestions disabled yields an empty list. The limit
    is clamped to the embed cap of 12; question generation itself is a
    deferred seam in this layer, so the enabled branch currently returns
    an empty list.
    """
    if not channel.show_suggested_questions:
        return EmbedSuggestedQuestionsEnvelope(
            success=True,
            data=EmbedSuggestedQuestionsData(questions=[]),
        )
    # The limit is honoured (capped at 12) even though question generation
    # is a deferred seam in this layer; the enabled branch is empty today.
    _ = clamp_suggestion_limit(limit)
    return EmbedSuggestedQuestionsEnvelope(
        success=True,
        data=EmbedSuggestedQuestionsData(questions=[]),
    )


@public_router.get(
    "/{channel_id}/chunks/{chunk_id}",
    response_model=EmbedChunkEnvelope,
)
async def get_embed_chunk(
    channel: EmbedChannelDep,
    chunk_id: str,
    chunk_service: EmbedChunkServiceDep,
) -> EmbedChunkEnvelope:
    """Return one chunk referenced by an embed reply.

    Cross-workspace chunk ids are rejected: the chunk must belong to the
    channel's tenant even though the lookup itself is id-only.
    """
    cleaned = chunk_id.strip()
    if not cleaned:
        raise ValidationError(
            code="embed.chunk_id_required",
            message="chunk_id is required",
        )
    try:
        chunk = await chunk_service.get_chunk_by_id_only(id=cleaned)
    except NotFoundError as exc:
        raise NotFoundError(
            code="embed.chunk_not_found",
            message="chunk not found",
        ) from exc
    if chunk.tenant_id != channel.tenant_id:
        raise PermissionDeniedError(
            code="embed.chunk_forbidden",
            message="chunk not accessible",
        )
    return EmbedChunkEnvelope(success=True, data=chunk.model_dump(mode="json"))


@public_router.post(
    "/{channel_id}/sessions",
    response_model=EmbedSessionCreateEnvelope,
    status_code=201,
)
async def create_embed_session(
    request: Request,
    channel_id: str,
    session_service: EmbedSessionServiceDep,
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> EmbedSessionCreateEnvelope:
    """Create an anonymous embed chat session.

    Validates the publish token, gates the request origin, applies the
    channel rate budgets, and hands back a signed handle the widget must
    echo (``X-Embed-Session``) on subsequent calls.
    """
    origin = request.headers.get("origin", "")
    client_ip = request.client.host if request.client is not None else ""
    created = await session_service.create_session(
        channel_id=channel_id,
        token=extract_embed_token(authorization),
        origin=origin,
        client_ip=client_ip,
    )
    return EmbedSessionCreateEnvelope(
        success=True,
        data=EmbedSessionCreateData(
            id=created.session_id,
            sig=created.handle,
        ),
    )


@public_router.post("/{channel_id}/knowledge-chat/{session_id}")
async def embed_knowledge_chat(
    request: Request,
    session_id: str,
    channel: EmbedChannelDep,
    _session: EmbedSessionDep,
    chat: EmbedChatServiceDep,
) -> StreamingResponse:
    """Stream a knowledge-QA turn for an embed session (SSE)."""
    return await _embed_chat_response(
        chat=chat,
        session_id=session_id,
        payload=patch_embed_chat_payload(
            await request.body(),
            channel,
            agent_mode=False,
        ),
        agent_mode=False,
        request_id=chat.request_id,
    )


@public_router.post("/{channel_id}/agent-chat/{session_id}")
async def embed_agent_chat(
    request: Request,
    session_id: str,
    channel: EmbedChannelDep,
    _session: EmbedSessionDep,
    chat: EmbedChatServiceDep,
) -> StreamingResponse:
    """Stream an agent-chat (ReAct) turn for an embed session (SSE)."""
    return await _embed_chat_response(
        chat=chat,
        session_id=session_id,
        payload=patch_embed_chat_payload(
            await request.body(),
            channel,
            agent_mode=True,
        ),
        agent_mode=True,
        request_id=chat.request_id,
    )


@public_router.get(
    "/{channel_id}/messages/{session_id}/load",
    response_model=MessageLoadEnvelope,
)
async def embed_load_messages(
    session_id: str,
    channel: EmbedChannelDep,
    _session: EmbedSessionDep,
    ctx: EmbedMessageContextDep,
    message_service: EmbedMessageServiceDep,
    query: Annotated[LoadMessagesQuery, Query()],
) -> MessageLoadEnvelope:
    """Load an embed session's message history, newest-window first."""
    sid = session_id.strip()
    limit = _clamp_load_limit(query.limit)
    if not query.before_time:
        rows = await message_service.get_recent_messages_by_session(
            ctx,
            sid,
            limit,
        )
    else:
        rows = await message_service.list_messages_by_session_before_time(
            ctx,
            sid,
            _parse_before_time(query.before_time),
            limit,
        )
    return message_load_envelope(rows)


@public_router.post("/{channel_id}/sessions/{session_id}/stop")
async def embed_stop_session(
    session_id: str,
    channel: EmbedChannelDep,
    _session: EmbedSessionDep,
) -> None:
    """Stop an in-flight embed generation for a session.

    The stream-stop registry is not wired into this layer yet; the route
    stays registered so the widget surface is faithful while the
    execution seam lands later.
    """
    raise _capability_unavailable("embed session stop is not yet wired")


@public_router.get(
    "/{channel_id}/sessions/{session_id}/messages/{message_id}/suggestions",
)
async def embed_get_message_suggestions(
    session_id: str,
    message_id: str,
    channel: EmbedChannelDep,
    _session: EmbedSessionDep,
    suggestion_service: MessageSuggestionServiceDep,
) -> EmbedSuggestionSuppressedEnvelope | SuggestionEnvelope:
    """Return cached follow-up suggestions for an embed assistant message."""
    if not channel.show_suggested_questions:
        return _suppressed_suggestions()
    suggestion_set = await suggestion_service.get_follow_ups(
        session_id=session_id.strip(),
        assistant_message_id=message_id.strip(),
    )
    return suggestion_envelope(suggestion_set)


@public_router.post(
    "/{channel_id}/sessions/{session_id}/messages/{message_id}/suggestions",
)
async def embed_ensure_message_suggestions(
    session_id: str,
    message_id: str,
    channel: EmbedChannelDep,
    _session: EmbedSessionDep,
    suggestion_service: MessageSuggestionServiceDep,
    body: EnsureSuggestionsRequest | None = None,
) -> EmbedSuggestionSuppressedEnvelope | SuggestionEnvelope:
    """Generate (or return cached) follow-up suggestions for an embed reply."""
    if not channel.show_suggested_questions:
        return _suppressed_suggestions()
    regenerate = body.regenerate if body is not None else False
    suggestion_set = await suggestion_service.ensure_follow_ups(
        session_id=session_id.strip(),
        assistant_message_id=message_id.strip(),
        regenerate=regenerate,
    )
    return suggestion_envelope(suggestion_set)


@public_router.post(
    "/{channel_id}/sessions/{session_id}/suggestion-events",
    status_code=204,
)
async def embed_record_suggestion_event(
    session_id: str,
    channel: EmbedChannelDep,
    _session: EmbedSessionDep,
    body: SuggestionEventRequest,
    suggestion_service: MessageSuggestionServiceDep,
) -> Response:
    """Record an exposure / click / dismiss event for an embed suggestion."""
    await suggestion_service.record_event(
        session_id=session_id.strip(),
        suggestion_set_id=body.suggestion_set_id.strip(),
        question_id=body.question_id.strip(),
        event_type=body.event_type.strip(),
    )
    return Response(status_code=204)


@public_router.post(
    "/{channel_id}/sessions/{session_id}/events",
    response_model=EmbedWebhookAckResponse,
)
async def embed_relay_webhook_event(
    session_id: str,
    channel: EmbedChannelDep,
    _session: EmbedSessionDep,
    body: EmbedWebhookEventRequest,
    dispatcher: EmbedWebhookDispatcherDep,
) -> EmbedWebhookAckResponse:
    """Forward a visitor chat event to the channel's outbound webhook."""
    event_type = body.type.strip()
    if event_type not in {"message_sent", "message_received"}:
        raise ValidationError(
            code="embed.unsupported_event_type",
            message="unsupported event type",
        )
    payload: dict[str, str] = {}
    if body.query.strip():
        payload["query"] = body.query.strip()
    if body.content.strip():
        payload["content"] = body.content.strip()
    target_session = body.session_id.strip() or session_id
    dispatcher.dispatch(
        channel,
        event_type=event_type,
        session_id=target_session,
        payload=payload,
    )
    return EmbedWebhookAckResponse()


@public_router.post(
    "/{channel_id}/sessions/{session_id}/mcp-oauth-resolutions/{pending_id}",
)
async def embed_resolve_mcp_oauth(
    session_id: str,
    pending_id: str,
    channel: EmbedChannelDep,
    _session: EmbedSessionDep,
) -> None:
    """Resolve a pending MCP OAuth flow for an embed session."""
    raise _capability_unavailable("embed MCP OAuth resolution is not yet wired")


@public_router.post(
    "/{channel_id}/sessions/{session_id}/mcp-oauth-resolutions/{pending_id}/cancel",
)
async def embed_cancel_mcp_oauth(
    session_id: str,
    pending_id: str,
    channel: EmbedChannelDep,
    _session: EmbedSessionDep,
) -> None:
    """Cancel a pending MCP OAuth flow for an embed session."""
    raise _capability_unavailable("embed MCP OAuth cancellation is not yet wired")


@public_router.post(
    "/{channel_id}/sessions/{session_id}/mcp-services/{service_id}/oauth/authorize-url",
)
async def embed_mcp_oauth_authorize_url(
    session_id: str,
    service_id: str,
    channel: EmbedChannelDep,
    _session: EmbedSessionDep,
) -> None:
    """Return an MCP OAuth authorize URL for an embed session."""
    raise _capability_unavailable("embed MCP OAuth authorize-url is not yet wired")


@public_router.get(
    "/{channel_id}/sessions/{session_id}/mcp-services/{service_id}/oauth/status",
)
async def embed_mcp_oauth_status(
    session_id: str,
    service_id: str,
    channel: EmbedChannelDep,
    _session: EmbedSessionDep,
) -> None:
    """Return the MCP OAuth status for an embed session."""
    raise _capability_unavailable("embed MCP OAuth status is not yet wired")


@public_router.post(
    "/{channel_id}/sessions/{session_id}/tool-approvals/{pending_id}",
)
async def embed_resolve_tool_approval(
    session_id: str,
    pending_id: str,
    channel: EmbedChannelDep,
    _session: EmbedSessionDep,
) -> None:
    """Resolve a pending tool-approval for an embed session."""
    raise _capability_unavailable("embed tool approval is not yet wired")


@public_router.get("/{channel_id}/files")
async def embed_files(
    channel: EmbedChannelDep,
) -> None:
    """Serve images embedded in embed bot replies."""
    raise _capability_unavailable("embed file serving is not yet wired")


# ── Internal helpers ──────────────────────────────────────────────────


def _suppressed_suggestions() -> EmbedSuggestionSuppressedEnvelope:
    """Return the channel-disabled suggestions suppression envelope."""
    return EmbedSuggestionSuppressedEnvelope(
        success=True,
        data=EmbedSuggestionSuppressedData(),
    )


async def _embed_chat_response(
    *,
    chat: ChatService,
    session_id: str,
    payload: dict[str, JsonValue],
    agent_mode: bool,
    request_id: str,
) -> StreamingResponse:
    """Stream an embed QA turn over the SSE ``event: message`` dialect."""
    sid = session_id.strip()
    if not sid:
        raise ValidationError(
            code="embed.session_id_required",
            message="session_id is required",
        )
    body = CreateKnowledgeQARequest.model_validate(payload)
    if agent_mode:
        events = await chat.stream_agent_qa(
            session_id=sid,
            request=cast(AgentQARequestLike, body),
        )
    else:
        events = await chat.stream_knowledge_qa(
            session_id=sid,
            request=cast(KnowledgeQARequestLike, body),
        )
    return _sse_response(_embed_frames(events, request_id))


async def _embed_frames(
    events: AsyncIterator[Event],
    request_id: str,
) -> AsyncIterator[StreamResponse]:
    """Map the service's domain events onto the wire frame shape."""
    async for event in events:
        frame = to_stream_response(event, request_id=request_id)
        if frame is not None:
            yield frame


def _sse_response(stream: AsyncIterator[StreamResponse]) -> StreamingResponse:
    """Wrap a frame stream in an SSE ``StreamingResponse``."""

    async def _render() -> AsyncIterator[str]:
        async for frame in stream:
            yield format_sse_frame(frame)

    return StreamingResponse(
        _render(),
        media_type=_SSE_MEDIA_TYPE,
        headers=_SSE_HEADERS,
    )


__all__ = [
    "agents_router",
    "public_router",
    "router",
]
