"""Message and suggestion HTTP endpoints.

Maps the message route family onto the request-scoped
``MessageServiceImpl`` and the suggestion route family onto
``MessageSuggestionService``:

====================  ============================================================  =============
Method                Path                                                            Handler
====================  ============================================================  =============
``GET    ``          ``/messages/{session_id}/load``                                  load history
``DELETE ``          ``/messages/{session_id}/{message_id}``                          delete message
``POST   ``          ``/messages/search``                                             search history
``GET    ``          ``/messages/chat-history-stats``                                 KB stats
``GET    ``          ``/sessions/{session_id}/messages/{message_id}/suggestions``     get suggestions
``POST   ``          ``/sessions/{session_id}/messages/{message_id}/suggestions``     ensure suggestions
``POST   ``          ``/sessions/{session_id}/suggestion-events``                     record event
====================  ============================================================  =============

The message endpoints are tenant-wide chat-history surfaces (Viewer+
RBAC, mirroring the upstream). The suggestion endpoints hang off the
``/sessions`` tree (their own prefixless router) because the handler
owns the session/message wildcards; they are driven by
``MessageSuggestionService``.

The static ``/messages/search`` and ``/messages/chat-history-stats``
paths are declared before the ``{session_id}``-shaped routes so a
literal segment is never captured as a session id.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Query, Response

from src.common.exception import NotFoundError, ValidationError
from src.core.chat.messages import MessageSearchMode, MessageSearchParams
from src.core.chat.messages.suggestion_service import SUGGESTION_STATUS_GENERATING
from src.core.contracts.sessions import (
    EnsureSuggestionsRequest,
    LoadMessagesQuery,
    SearchMessagesRequest,
    SuggestionEventRequest,
)
from src.web.api.chat.messages.views import (
    ChatHistoryStatsEnvelope,
    DeleteMessageResponse,
    MessageLoadEnvelope,
    SearchMessagesEnvelope,
    SuggestionEnvelope,
    chat_history_stats_envelope,
    delete_message_response,
    message_load_envelope,
    search_messages_envelope,
    suggestion_envelope,
)
from src.web.deps import AuthDep, RoleViewerDep
from src.web.deps.chat_sessions import (
    MessageContextDep,
    MessageServiceDep,
    MessageSuggestionServiceDep,
)

router = APIRouter(prefix="/messages", tags=["messages"])
suggestion_router = APIRouter(tags=["suggestions"])

_DEFAULT_LIMIT = 20

_DELETE_MESSAGE = "Message deleted successfully"


def _require_session_id(session_id: str) -> str:
    """Reject an empty session id (upstream rejects with a 400)."""
    if not session_id or not session_id.strip():
        raise ValidationError(
            code="message.session_required",
            message="Session ID is empty",
        )
    return session_id.strip()


def _clamp_limit(limit: int) -> int:
    """Coerce a limit onto ``[1, inf)`` (upstream falls back to 20)."""
    return limit if limit >= 1 else _DEFAULT_LIMIT


def _parse_before_time(raw: str) -> datetime:
    """Parse an RFC3339 / RFC3339Nano cursor (upstream ``parseMessageBeforeTime``)."""
    stripped = raw.strip()
    try:
        return datetime.fromisoformat(stripped)
    except ValueError as exc:
        raise ValidationError(
            code="message.invalid_before_time",
            message="Invalid time format, please use RFC3339 or RFC3339Nano format",
        ) from exc


def _resolve_search_mode(raw: str | None) -> MessageSearchMode:
    """Coerce the request mode string onto the enum (default ``hybrid``)."""
    try:
        return MessageSearchMode(raw or "hybrid")
    except ValueError as exc:
        raise ValidationError(
            code="message.invalid_search_mode",
            message=f"invalid search mode: {raw}",
        ) from exc


@router.get("/chat-history-stats", response_model=ChatHistoryStatsEnvelope)
async def chat_history_stats(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    ctx: MessageContextDep,
    message_service: MessageServiceDep,
) -> ChatHistoryStatsEnvelope:
    """Return chat-history knowledge-base stats for the workspace."""
    stats = await message_service.get_chat_history_kb_stats(ctx)
    return chat_history_stats_envelope(stats)


@router.post("/search", response_model=SearchMessagesEnvelope)
async def search_messages(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    body: SearchMessagesRequest,
    ctx: MessageContextDep,
    message_service: MessageServiceDep,
) -> SearchMessagesEnvelope:
    """Search chat history by keyword, vector, or hybrid fusion."""
    result = await message_service.search_messages(
        ctx,
        MessageSearchParams(
            query=body.query,
            mode=_resolve_search_mode(body.mode),
            limit=_clamp_limit(body.limit),
            session_ids=tuple(body.session_ids or ()),
        ),
    )
    return search_messages_envelope(result)


@router.get("/{session_id}/load", response_model=MessageLoadEnvelope)
async def load_messages(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    session_id: str,
    query: Annotated[LoadMessagesQuery, Query()],
    ctx: MessageContextDep,
    message_service: MessageServiceDep,
) -> MessageLoadEnvelope:
    """Load a session's message history, newest-window first.

    ``limit`` defaults to 20. When ``before_time`` is absent the most
    recent messages are returned; otherwise the window is the messages
    created strictly before the RFC3339 / RFC3339Nano cursor.
    """
    session = _require_session_id(session_id)
    limit = _clamp_limit(query.limit)
    if not query.before_time:
        rows = await message_service.get_recent_messages_by_session(
            ctx,
            session,
            limit,
        )
    else:
        rows = await message_service.list_messages_by_session_before_time(
            ctx,
            session,
            _parse_before_time(query.before_time),
            limit,
        )
    return message_load_envelope(rows)


@router.delete("/{session_id}/{message_id}", response_model=DeleteMessageResponse)
async def delete_message(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    session_id: str,
    message_id: str,
    ctx: MessageContextDep,
    message_service: MessageServiceDep,
) -> DeleteMessageResponse:
    """Soft-delete one message of a session."""
    session = _require_session_id(session_id)
    deleted = await message_service.delete_message(ctx, session, message_id)
    if not deleted:
        raise NotFoundError(
            code="message.not_found",
            message=f"message {message_id} not found in session {session}",
        )
    return delete_message_response(_DELETE_MESSAGE)


# ── Suggestion endpoints (sessions tree, message domain) ──────────────


@suggestion_router.get(
    "/sessions/{session_id}/messages/{message_id}/suggestions",
    response_model=SuggestionEnvelope,
)
async def get_suggestions(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    session_id: str,
    message_id: str,
    suggestion_service: MessageSuggestionServiceDep,
) -> SuggestionEnvelope:
    """Return the cached follow-up suggestions for an assistant message."""
    suggestion_set = await suggestion_service.get_follow_ups(
        session_id=session_id,
        assistant_message_id=message_id,
    )
    return suggestion_envelope(suggestion_set)


@suggestion_router.post(
    "/sessions/{session_id}/messages/{message_id}/suggestions",
    response_model=SuggestionEnvelope,
)
async def ensure_suggestions(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    session_id: str,
    message_id: str,
    response: Response,
    suggestion_service: MessageSuggestionServiceDep,
    body: EnsureSuggestionsRequest | None = None,
) -> SuggestionEnvelope:
    """Generate (or return the cached) follow-up suggestions.

    Mirrors the upstream handler: the body is optional; when the
    generated set is still ``generating`` the response status is 202
    Accepted instead of 200.
    """
    regenerate = body.regenerate if body is not None else False
    suggestion_set = await suggestion_service.ensure_follow_ups(
        session_id=session_id,
        assistant_message_id=message_id,
        regenerate=regenerate,
    )
    if suggestion_set is not None and suggestion_set.status == SUGGESTION_STATUS_GENERATING:
        response.status_code = 202
    return suggestion_envelope(suggestion_set)


@suggestion_router.post(
    "/sessions/{session_id}/suggestion-events",
    status_code=204,
)
async def record_suggestion_event(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    session_id: str,
    body: SuggestionEventRequest,
    suggestion_service: MessageSuggestionServiceDep,
) -> Response:
    """Record an exposure / click / dismiss event for a suggestion."""
    await suggestion_service.record_event(
        session_id=session_id,
        suggestion_set_id=body.suggestion_set_id.strip(),
        question_id=body.question_id.strip(),
        event_type=body.event_type.strip(),
    )
    return Response(status_code=204)


__all__ = ["router", "suggestion_router"]
