"""Wire-shape conversion for the message and suggestion endpoints.

Projects the message-domain rows (``db.models.message.Message``) and
service DTOs (``MessageSearchResult`` / ``ChatHistoryKBStats`` /
``MessageSuggestionSet``) onto the frozen contract shapes in
``src/core/contracts/sessions.py`` and wraps them in the success
envelope.

The JSONB columns the storage layer exposes as raw JSON
(``knowledge_references``, ``agent_steps``, ``mentioned_items``,
``images``) are coerced leniently: malformed entries are dropped
rather than failing the whole response, mirroring the agent / KB
view conversions.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, ValidationError

from src.common.json import JsonObject, JsonValue
from src.core.chat.messages import (
    ChatHistoryKBStats,
    MessageSearchGroupItem,
    MessageSearchResult,
)
from src.core.chat.messages.suggestion_service import MessageSuggestionSet
from src.core.contracts.sessions import (
    ChatHistoryStats,
    KnowledgeReference,
    Message,
    MessageSearchHit,
    MessageSearchResponse,
    SuggestionQuestion,
    SuggestionSet,
)
from src.db.models.message import Message as MessageRow


class MessageLoadEnvelope(BaseModel):
    """``{"success": true, "data": [...]}`` - message list response."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: list[Message]


class DeleteMessageResponse(BaseModel):
    """``{"success": true, "message": "..."}`` - delete acknowledgement."""

    model_config = ConfigDict(frozen=True)

    success: bool
    message: str


class SearchMessagesEnvelope(BaseModel):
    """``{"success": true, "data": {"items": [...], "total": N}}``."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: MessageSearchResponse


class ChatHistoryStatsEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` - chat-history KB stats."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: ChatHistoryStats


class SuggestionEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` - suggestion-set response."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: SuggestionSet | None


def message_to_contract(row: MessageRow) -> Message:
    """Project a message row onto the frozen wire contract.

    The contract omits the storage-only columns (``rendered_content``,
    ``agent_id``, ``execution_context``, ...), so they are dropped here.
    """
    return Message(
        id=row.id,
        session_id=row.session_id,
        request_id=row.request_id or None,
        role=row.role,
        content=row.content,
        knowledge_references=_coerce_references(row.knowledge_references),
        agent_steps=_json_objects(row.agent_steps),
        is_completed=row.is_completed,
        is_fallback=row.is_fallback,
        agent_duration_ms=row.agent_duration_ms or None,
        mentioned_items=_json_objects(row.mentioned_items),
        images=_json_objects(row.images),
        channel=row.channel or None,
        created_at=row.created_at,
        updated_at=row.updated_at,
        deleted_at=row.deleted_at,
    )


def message_load_envelope(rows: list[MessageRow]) -> MessageLoadEnvelope:
    """Wrap a message page in the success envelope."""
    return MessageLoadEnvelope(
        success=True,
        data=[message_to_contract(row) for row in rows],
    )


def delete_message_response(message: str) -> DeleteMessageResponse:
    """Wrap a delete acknowledgement."""
    return DeleteMessageResponse(success=True, message=message)


def search_messages_envelope(
    result: MessageSearchResult,
) -> SearchMessagesEnvelope:
    """Wrap a chat-history search result in the success envelope."""
    return SearchMessagesEnvelope(
        success=True,
        data=MessageSearchResponse(
            items=[_hit_to_contract(item) for item in result.items],
            total=result.total,
        ),
    )


def chat_history_stats_envelope(stats: ChatHistoryKBStats) -> ChatHistoryStatsEnvelope:
    """Wrap the chat-history KB stats in the success envelope."""
    return ChatHistoryStatsEnvelope(
        success=True,
        data=ChatHistoryStats(
            enabled=stats.enabled,
            embedding_model_id=stats.embedding_model_id or None,
            knowledge_base_id=stats.knowledge_base_id or None,
            knowledge_base_name=stats.knowledge_base_name or None,
            indexed_message_count=stats.indexed_message_count or None,
            has_indexed_messages=stats.has_indexed_messages,
        ),
    )


def suggestion_envelope(
    suggestion_set: MessageSuggestionSet | None,
) -> SuggestionEnvelope:
    """Wrap a suggestion set in the success envelope (``data`` nullable)."""
    return SuggestionEnvelope(
        success=True,
        data=_suggestion_to_contract(suggestion_set),
    )


# ── Coercion helpers ──────────────────────────────────────────────────


def _json_objects(value: JsonValue | None) -> list[JsonObject] | None:
    """Coerce a JSONB list onto ``list[JsonObject]``, dropping non-dicts."""
    if not isinstance(value, list):
        return None
    objects = [item for item in value if isinstance(item, dict)]
    return objects or None


def _coerce_references(value: JsonValue | None) -> list[KnowledgeReference] | None:
    """Parse a JSONB reference list onto the typed contract, leniently."""
    if not isinstance(value, list):
        return None
    parsed: list[KnowledgeReference] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            parsed.append(KnowledgeReference.model_validate(item))
        except ValidationError:
            continue
    return parsed or None


def _hit_to_contract(item: MessageSearchGroupItem) -> MessageSearchHit:
    """Project one search group onto the wire hit shape."""
    return MessageSearchHit(
        request_id=item.request_id,
        session_id=item.session_id,
        session_title=item.session_title or None,
        query_content=item.query_content,
        answer_content=item.answer_content,
        score=item.score,
        match_type=item.match_type or None,
        created_at=item.created_at,
    )


def _suggestion_to_contract(
    suggestion_set: MessageSuggestionSet | None,
) -> SuggestionSet | None:
    """Project a suggestion-set row onto the wire contract.

    The row carries the tenant / agent bookkeeping columns the wire
    shape omits; only the display fields are projected.
    """
    if suggestion_set is None:
        return None
    questions = suggestion_set.questions
    parsed_questions = None
    if isinstance(questions, list):
        parsed: list[SuggestionQuestion] = []
        for item in questions:
            if not isinstance(item, dict):
                continue
            raw_category = item.get("category")
            parsed.append(
                SuggestionQuestion(
                    id=str(item.get("id", "")),
                    text=str(item.get("text", "")),
                    category=raw_category if isinstance(raw_category, str) else None,
                )
            )
        parsed_questions = parsed or None
    return SuggestionSet(
        id=suggestion_set.id,
        session_id=suggestion_set.session_id,
        assistant_message_id=suggestion_set.assistant_message_id,
        position=suggestion_set.placement,
        status=suggestion_set.status,
        language=suggestion_set.locale or None,
        questions=parsed_questions,
        created_at=suggestion_set.created_at,
        updated_at=suggestion_set.updated_at,
    )


__all__ = [
    "ChatHistoryStatsEnvelope",
    "DeleteMessageResponse",
    "MessageLoadEnvelope",
    "SearchMessagesEnvelope",
    "SuggestionEnvelope",
    "chat_history_stats_envelope",
    "delete_message_response",
    "message_load_envelope",
    "message_to_contract",
    "search_messages_envelope",
    "suggestion_envelope",
]
