"""Chat HTTP endpoints — knowledge QA, agent chat, knowledge search.

Maps the chat route family onto the request-scoped ``ChatService``:

| Method | Path                           | Handler          |
| ------ | ------------------------------ | ---------------- |
| POST   | /knowledge-chat/{session_id}   | knowledge QA     |
| POST   | /agent-chat/{session_id}       | agent chat       |
| POST   | /knowledge-search              | knowledge search |

The two QA endpoints stream their result over Server-Sent Events using
the upstream ``event: message`` dialect; the search endpoint answers a
plain JSON envelope. All three are Viewer+ surfaces; per-session and
per-agent authorization is enforced inside the service layer.

The pipeline / agent loop is assembled per request by the service
factory; the views only validate the request surface and stream the
events produced by the service.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import cast

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.common.exception import ValidationError
from src.core.chat.bus import Event
from src.core.chat.service import (
    AgentQARequestLike,
    KnowledgeQARequestLike,
)
from src.web.api.chat.views import (
    CreateKnowledgeQARequest,
    SearchKnowledgeEnvelope,
    SearchKnowledgeRequest,
    StreamResponse,
    format_sse_frame,
    to_stream_response,
)
from src.web.deps.chat import ChatServiceDep
from src.web.deps.rbac import RoleViewerDep
from src.web.middleware.auth import AuthDep

router = APIRouter(tags=["chat"])

_SSE_MEDIA_TYPE = "text/event-stream"
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _require_session_id(session_id: str) -> str:
    """Reject an empty session id (upstream rejects with a 400)."""
    if not session_id or not session_id.strip():
        raise ValidationError(
            code="chat.session_required",
            message="Session ID is empty",
        )
    return session_id.strip()


def _require_query(query: str) -> str:
    """Reject an empty query (upstream rejects with a 400)."""
    if not query or not query.strip():
        raise ValidationError(
            code="chat.query_required",
            message="Query content cannot be empty",
        )
    return query.strip()


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


@router.post("/knowledge-chat/{session_id}")
async def knowledge_qa(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    session_id: str,
    body: CreateKnowledgeQARequest,
    chat: ChatServiceDep,
) -> StreamingResponse:
    """Run knowledge QA (RAG / pure chat) over an SSE stream."""
    _require_session_id(session_id)
    _require_query(body.query)
    events = await chat.stream_knowledge_qa(
        session_id=session_id,
        # The Pydantic body satisfies ``KnowledgeQARequestLike``
        # structurally; mypy cannot see the attribute overlap across the
        # Pydantic ``list`` fields, so the cross-layer contract is cast.
        request=cast(KnowledgeQARequestLike, body),
    )
    return _sse_response(_to_frames(events, chat.request_id))


@router.post("/agent-chat/{session_id}")
async def agent_qa(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    session_id: str,
    body: CreateKnowledgeQARequest,
    chat: ChatServiceDep,
) -> StreamingResponse:
    """Run agent chat (ReAct loop) over an SSE stream."""
    _require_session_id(session_id)
    _require_query(body.query)
    events = await chat.stream_agent_qa(
        session_id=session_id,
        request=cast(AgentQARequestLike, body),
    )
    return _sse_response(_to_frames(events, chat.request_id))


async def _to_frames(
    events: AsyncIterator[Event],
    request_id: str,
) -> AsyncIterator[StreamResponse]:
    """Map the service's domain events onto the wire frame shape."""
    async for event in events:
        frame = to_stream_response(event, request_id=request_id)
        if frame is not None:
            yield frame


@router.post("/knowledge-search", response_model=SearchKnowledgeEnvelope)
async def search_knowledge(
    _auth: AuthDep,
    _viewer: RoleViewerDep,
    body: SearchKnowledgeRequest,
    chat: ChatServiceDep,
) -> SearchKnowledgeEnvelope:
    """Run a retrieval-only knowledge search (no LLM summarization)."""
    _require_query(body.query)
    results = await chat.search_knowledge(
        query=body.query,
        knowledge_base_id=body.knowledge_base_id,
        knowledge_base_ids=body.knowledge_base_ids,
        knowledge_ids=body.knowledge_ids,
        tag_ids=body.tag_ids,
        mentioned_items=body.mentioned_items,
    )
    return SearchKnowledgeEnvelope(data=results)


__all__ = ["router"]
