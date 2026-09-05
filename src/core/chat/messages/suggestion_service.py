"""Follow-up suggestion generation for a completed assistant turn.

The SPA posts after every answer. The turn text may arrive in the
request because the chat stream and this call are different units of
work. Stored messages are the fallback once the stream has committed.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from src.core.chat.messages.follow_up_generator import (
    ChatClientFactory,
    resolve_follow_up_questions,
)
from src.core.chat.messages.types import ROLE_USER
from src.db.models.message import Message
from src.db.models.message_suggestion import (
    SUGGESTION_EVENT_CLICK,
    SUGGESTION_EVENT_DISMISS,
    SUGGESTION_EVENT_IMPRESSION,
    SUGGESTION_EVENT_REGENERATE,
    SUGGESTION_PLACEMENT_AFTER_ANSWER,
    SUGGESTION_STATUS_FAILED,
    SUGGESTION_STATUS_GENERATING,
    SUGGESTION_STATUS_READY,
    SUGGESTION_STATUS_SUPPRESSED,
    MessageSuggestionSet,
)

_CONFIG_HASH: str = "default"
_LOCALE: str = "zh-CN"


@runtime_checkable
class SuggestionStore(Protocol):
    """Cache-key lookup, lease claim, and save for suggestion sets."""

    async def get_by_cache_key(
        self,
        *,
        tenant_id: int,
        assistant_message_id: str,
        placement: str,
        config_hash: str,
        locale: str,
    ) -> MessageSuggestionSet | None: ...

    async def acquire_generation(
        self,
        candidate: MessageSuggestionSet,
        *,
        regenerate: bool,
        now: datetime,
    ) -> tuple[MessageSuggestionSet | None, bool]: ...

    async def save(self, row: MessageSuggestionSet) -> MessageSuggestionSet: ...


@runtime_checkable
class MessageStore(Protocol):
    """Reads the completed turn when the request body omitted it."""

    async def get_by_id_and_session(
        self, *, session_id: str, message_id: str
    ) -> Message | None: ...

    async def list_recent_by_session(self, session_id: str, *, limit: int) -> list[Message]: ...


class MessageSuggestionService:
    """Generate, cache, and no-op analytics for after-answer chips."""

    def __init__(
        self,
        *,
        tenant_id: int = 0,
        messages: MessageStore | None = None,
        suggestions: SuggestionStore | None = None,
        chat_models: ChatClientFactory | None = None,
    ) -> None:
        self._tenant_id = tenant_id
        self._messages = messages
        self._suggestions = suggestions
        self._chat_models = chat_models

    async def ensure_follow_ups(
        self,
        *,
        session_id: str,
        assistant_message_id: str,
        regenerate: bool,
        query: str | None = None,
        answer: str | None = None,
    ) -> MessageSuggestionSet:
        """Return a cached ready set, or generate one from the turn text."""
        now = datetime.now(UTC)
        claimed = await self._claim(
            session_id=session_id,
            assistant_message_id=assistant_message_id,
            regenerate=regenerate,
            now=now,
        )
        if claimed is not None and not claimed[1]:
            return claimed[0]
        query_text, answer_text = await self._resolve_turn(
            session_id=session_id,
            assistant_message_id=assistant_message_id,
            query=query,
            answer=answer,
        )
        row = await self._build_set(
            session_id=session_id,
            assistant_message_id=assistant_message_id,
            query=query_text,
            answer=answer_text,
            now=now,
            existing=claimed[0] if claimed is not None else None,
        )
        if self._suggestions is not None:
            return await self._suggestions.save(row)
        return row

    async def get_follow_ups(
        self,
        *,
        session_id: str,
        assistant_message_id: str,
    ) -> MessageSuggestionSet | None:
        """Return the cached set for this assistant message, if any."""
        del session_id
        if self._suggestions is None:
            return None
        return await self._suggestions.get_by_cache_key(
            tenant_id=self._tenant_id,
            assistant_message_id=assistant_message_id,
            placement=SUGGESTION_PLACEMENT_AFTER_ANSWER,
            config_hash=_CONFIG_HASH,
            locale=_LOCALE,
        )

    async def record_event(
        self,
        *,
        session_id: str,
        suggestion_set_id: str,
        question_id: str,
        event_type: str,
    ) -> None:
        """Analytics persist in a later seam. The toolbar must not 501."""
        del session_id, suggestion_set_id, question_id, event_type

    async def validate_attribution(
        self,
        *,
        session_id: str,
        query: str,
        suggestion_set_id: str,
        question_id: str,
    ) -> None:
        """Attribution checks stay open until suggestion analytics land."""
        del session_id, query, suggestion_set_id, question_id

    async def _claim(
        self,
        *,
        session_id: str,
        assistant_message_id: str,
        regenerate: bool,
        now: datetime,
    ) -> tuple[MessageSuggestionSet, bool] | None:
        if self._suggestions is None:
            return None
        candidate = _empty_set(
            tenant_id=self._tenant_id,
            session_id=session_id,
            assistant_message_id=assistant_message_id,
            now=now,
        )
        row, acquired = await self._suggestions.acquire_generation(
            candidate, regenerate=regenerate, now=now
        )
        if row is None:
            return None
        return row, acquired

    async def _resolve_turn(
        self,
        *,
        session_id: str,
        assistant_message_id: str,
        query: str | None,
        answer: str | None,
    ) -> tuple[str, str]:
        query_text = (query or "").strip()
        answer_text = (answer or "").strip()
        if query_text and answer_text:
            return query_text, answer_text
        stored_query, stored_answer = await self._load_turn(
            session_id=session_id,
            assistant_message_id=assistant_message_id,
        )
        return query_text or stored_query, answer_text or stored_answer

    async def _load_turn(
        self,
        *,
        session_id: str,
        assistant_message_id: str,
    ) -> tuple[str, str]:
        if self._messages is None:
            return "", ""
        assistant = await self._messages.get_by_id_and_session(
            session_id=session_id,
            message_id=assistant_message_id,
        )
        if assistant is None:
            return "", ""
        recent = await self._messages.list_recent_by_session(session_id, limit=20)
        return _previous_user_query(recent, assistant_message_id), assistant.content.strip()

    async def _build_set(
        self,
        *,
        session_id: str,
        assistant_message_id: str,
        query: str,
        answer: str,
        now: datetime,
        existing: MessageSuggestionSet | None,
    ) -> MessageSuggestionSet:
        if not query and not answer:
            return _finish_set(
                existing,
                tenant_id=self._tenant_id,
                session_id=session_id,
                assistant_message_id=assistant_message_id,
                now=now,
                status=SUGGESTION_STATUS_SUPPRESSED,
                reason="empty_context",
                questions=[],
            )
        texts = await resolve_follow_up_questions(
            tenant_id=self._tenant_id,
            chat_models=self._chat_models,
            query=query,
            answer=answer,
        )
        return _finish_set(
            existing,
            tenant_id=self._tenant_id,
            session_id=session_id,
            assistant_message_id=assistant_message_id,
            now=now,
            status=SUGGESTION_STATUS_READY,
            reason="",
            questions=_question_payload(texts),
        )


def _previous_user_query(rows: list[Message], assistant_message_id: str) -> str:
    prior = ""
    for row in rows:
        if row.id == assistant_message_id:
            return prior
        if row.role == ROLE_USER:
            prior = row.content.strip()
    return prior


def _question_payload(texts: list[str]) -> list[dict[str, str]]:
    return [{"id": str(uuid.uuid4()), "text": text} for text in texts]


def _empty_set(
    *,
    tenant_id: int,
    session_id: str,
    assistant_message_id: str,
    now: datetime,
) -> MessageSuggestionSet:
    return MessageSuggestionSet(
        id=str(uuid.uuid4()),
        tenant_id=tenant_id,
        session_id=session_id,
        assistant_message_id=assistant_message_id,
        placement=SUGGESTION_PLACEMENT_AFTER_ANSWER,
        config_hash=_CONFIG_HASH,
        locale=_LOCALE,
        status=SUGGESTION_STATUS_GENERATING,
        allow_regenerate=True,
        questions=[],
        created_at=now,
        updated_at=now,
    )


def _finish_set(
    existing: MessageSuggestionSet | None,
    *,
    tenant_id: int,
    session_id: str,
    assistant_message_id: str,
    now: datetime,
    status: str,
    reason: str,
    questions: list[dict[str, str]],
) -> MessageSuggestionSet:
    base = existing or _empty_set(
        tenant_id=tenant_id,
        session_id=session_id,
        assistant_message_id=assistant_message_id,
        now=now,
    )
    return base.model_copy(
        update={
            "status": status,
            "allow_regenerate": True,
            "suppression_reason": reason,
            "questions": questions,
            "lease_until": None,
            "generated_at": now,
            "updated_at": now,
        }
    )


__all__ = [
    "SUGGESTION_EVENT_CLICK",
    "SUGGESTION_EVENT_DISMISS",
    "SUGGESTION_EVENT_IMPRESSION",
    "SUGGESTION_EVENT_REGENERATE",
    "SUGGESTION_PLACEMENT_AFTER_ANSWER",
    "SUGGESTION_STATUS_FAILED",
    "SUGGESTION_STATUS_GENERATING",
    "SUGGESTION_STATUS_READY",
    "SUGGESTION_STATUS_SUPPRESSED",
    "MessageStore",
    "MessageSuggestionService",
    "MessageSuggestionSet",
    "SuggestionStore",
]
