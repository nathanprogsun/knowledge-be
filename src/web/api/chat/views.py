"""Wire-shape models and conversions for the chat endpoints.

Defines the request bodies (mirroring the upstream request types
field-for-field) and the SSE ``StreamResponse`` frame shape the QA
endpoints emit. ``to_stream_response`` projects a chat-domain ``Event``
onto the wire frame: the event's ``content`` / ``done`` payload is lifted
to the top level and the remaining data map travels as the frame's
``data`` slot, matching the upstream stream-response contract.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject, JsonValue
from src.core.chat.bus import Event
from src.core.chat.pipeline.types import SearchResult
from src.core.chat.service import WIRE_RESPONSE_TYPE

# ── Request bodies (upstream request shape) ───────────────────────────


class MentionedItemRequest(BaseModel):
    """One ``@mentioned`` item (kb / file / tag / mcp / skill)."""

    model_config = ConfigDict(frozen=True)

    id: str = ""
    name: str = ""
    type: str = ""
    kb_type: str = ""
    kb_id: str | None = None
    kb_name: str = ""
    service_id: str = ""
    skill_name: str = ""


class ImageAttachment(BaseModel):
    """An attached image (base64 data sent by the client)."""

    model_config = ConfigDict(frozen=True)

    data: str = ""
    url: str = ""
    caption: str = ""


class AttachmentUpload(BaseModel):
    """A base64-encoded file attachment."""

    model_config = ConfigDict(frozen=True)

    data: str
    file_name: str
    file_size: int = 0


class SuggestionAttribution(BaseModel):
    """Attribution record for a suggestion-sourced question."""

    model_config = ConfigDict(frozen=True)

    suggestion_set_id: str
    question_id: str


class CreateKnowledgeQARequest(BaseModel):
    """Body shared by the knowledge-QA and agent-chat endpoints."""

    model_config = ConfigDict(frozen=True)

    query: str
    knowledge_base_ids: list[str] | None = Field(default=None)
    knowledge_ids: list[str] | None = Field(default=None)
    agent_enabled: bool = False
    agent_id: str | None = Field(default=None)
    agent_source_tenant_id: int = 0
    web_search_enabled: bool = False
    summary_model_id: str | None = Field(default=None)
    mcp_service_ids: list[str] | None = Field(default=None)
    skill_names: list[str] | None = Field(default=None)
    tag_ids: list[str] | None = Field(default=None)
    mentioned_items: list[MentionedItemRequest] | None = Field(default=None)
    disable_title: bool = False
    images: list[ImageAttachment] | None = Field(default=None)
    attachment_uploads: list[AttachmentUpload] | None = Field(default=None)
    attachment_ids: list[str] | None = Field(default=None)
    channel: str | None = Field(default=None)
    suggestion_attribution: SuggestionAttribution | None = Field(default=None)


class SearchKnowledgeRequest(BaseModel):
    """Body of the retrieval-only knowledge-search endpoint."""

    model_config = ConfigDict(frozen=True)

    query: str
    knowledge_base_id: str | None = Field(default=None)
    knowledge_base_ids: list[str] | None = Field(default=None)
    knowledge_ids: list[str] | None = Field(default=None)
    tag_ids: list[str] | None = Field(default=None)
    mentioned_items: list[MentionedItemRequest] | None = Field(default=None)


# ── Response envelopes ────────────────────────────────────────────────


class SearchKnowledgeEnvelope(BaseModel):
    """``{"success": true, "data": [...]}`` — knowledge-search response."""

    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: list[SearchResult]


class StreamResponse(BaseModel):
    """One SSE frame payload (upstream stream-response shape).

    ``id`` is the request id so the client can correlate every frame of a
    turn; ``data`` carries the event-specific metadata. ``content`` and
    ``done`` are the per-chunk payload lifted from the domain event.
    """

    model_config = ConfigDict(frozen=True)

    id: str = ""
    response_type: str
    content: str = ""
    done: bool = False
    knowledge_references: list[SearchResult] | None = Field(default=None)
    session_id: str | None = Field(default=None)
    assistant_message_id: str | None = Field(default=None)
    tool_calls: list[JsonObject] | None = Field(default=None)
    data: JsonObject | None = Field(default=None)
    usage: JsonObject | None = Field(default=None)
    finish_reason: str | None = Field(default=None)


# ── Event → wire projection ───────────────────────────────────────────


def to_stream_response(event: Event, request_id: str) -> StreamResponse | None:
    """Project a chat-domain event onto the SSE frame shape.

    Returns ``None`` for internal events that are not forwarded to the
    wire (they stay inside the execution layer). The event's ``content``
    and ``done`` payload keys are lifted to the frame top level; the
    remaining data map becomes the frame's ``data`` slot.
    """
    response_type = WIRE_RESPONSE_TYPE.get(event.type)
    if response_type is None:
        return None

    data = event.data if isinstance(event.data, dict) else None
    content = str(data.get("content", "")) if data else ""
    done = bool(data.get("done", False)) if data else False
    metadata = (
        {key: value for key, value in data.items() if key not in {"content", "done"}}
        if data
        else None
    )

    knowledge_references: list[SearchResult] | None = None
    assistant_message_id: str | None = None
    tool_calls: list[JsonObject] | None = None
    usage: JsonObject | None = None
    finish_reason: str | None = None

    if response_type == "agent_query" and data is not None:
        assistant_id = str(data.get("assistant_message_id", "") or "")
        assistant_message_id = assistant_id or None
    elif response_type == "references" and data is not None:
        refs = data.get("references")
        if isinstance(refs, list):
            coerced: list[SearchResult] = []
            for ref in refs:
                candidate = _coerce_reference(ref)
                if candidate is not None:
                    coerced.append(candidate)
            knowledge_references = coerced
    elif response_type == "tool_call" and data is not None:
        calls = data.get("tool_calls")
        if isinstance(calls, list):
            tool_calls = [call for call in calls if isinstance(call, dict)]

    raw_usage = data.get("usage") if data is not None else None
    if isinstance(raw_usage, dict):
        usage = raw_usage
    if data is not None and data.get("finish_reason"):
        finish_reason = str(data["finish_reason"])

    return StreamResponse(
        id=request_id,
        response_type=response_type,
        content=content,
        done=done,
        knowledge_references=knowledge_references,
        session_id=event.session_id,
        assistant_message_id=assistant_message_id,
        tool_calls=tool_calls,
        data=metadata or None,
        usage=usage,
        finish_reason=finish_reason,
    )


def _coerce_reference(value: JsonValue) -> SearchResult | None:
    """Validate a reference payload onto the pipeline ``SearchResult``."""
    if isinstance(value, SearchResult):
        return value
    if isinstance(value, dict):
        try:
            return SearchResult.model_validate(value)
        except Exception:
            return None
    return None


def format_sse_frame(response: StreamResponse) -> str:
    """Serialize one frame in the SSE ``event: message`` dialect."""
    payload = response.model_dump(mode="json", exclude_none=True)
    return f"event: message\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


__all__ = [
    "AttachmentUpload",
    "CreateKnowledgeQARequest",
    "ImageAttachment",
    "MentionedItemRequest",
    "SearchKnowledgeEnvelope",
    "SearchKnowledgeRequest",
    "StreamResponse",
    "SuggestionAttribution",
    "format_sse_frame",
    "to_stream_response",
]
