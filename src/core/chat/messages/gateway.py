"""Persists the user and assistant shells of one QA turn.

Chat used to mint ids and drop the rows. History load and follow-up
generation both read ``messages``, so the gateway writes the same
session the stream already named.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from src.core.agents.types import CustomAgentInfo
from src.core.chat.messages.types import ROLE_ASSISTANT, ROLE_USER
from src.core.chat.service import AssistantMessage
from src.db.dao.message_repository import MessageRepository
from src.db.models.message import Message


def _now() -> datetime:
    return datetime.now(UTC)


class PersistentMessageGateway:
    """Request-scoped message writer used by ``ChatService``."""

    def __init__(self, messages: MessageRepository) -> None:
        self._messages = messages
        self._sessions: dict[str, str] = {}

    async def create_user_message(self, *, session_id: str, query: str) -> str:
        """Insert the user turn and return its id."""
        message_id = str(uuid.uuid4())
        now = _now()
        await self._messages.create(
            Message(
                id=message_id,
                session_id=session_id,
                role=ROLE_USER,
                content=query,
                is_completed=True,
                created_at=now,
                updated_at=now,
            )
        )
        return message_id

    async def create_assistant_message(
        self,
        *,
        session_id: str,
        request_id: str,
        agent: CustomAgentInfo | None,
        model_id: str,
    ) -> AssistantMessage:
        """Insert an incomplete assistant shell and remember its session."""
        message_id = str(uuid.uuid4())
        now = _now()
        await self._messages.create(
            Message(
                id=message_id,
                request_id=request_id,
                session_id=session_id,
                role=ROLE_ASSISTANT,
                content="",
                is_completed=False,
                agent_id=agent.id if agent is not None else "",
                agent_tenant_id=agent.tenant_id if agent is not None else 0,
                model_id=model_id,
                created_at=now,
                updated_at=now,
            )
        )
        self._sessions[message_id] = session_id
        return AssistantMessage(id=message_id, session_id=session_id)

    async def complete_assistant_message(
        self,
        *,
        assistant_message_id: str,
        content: str,
        is_fallback: bool = False,
    ) -> None:
        """Write the streamed answer onto the assistant row."""
        session_id = self._sessions.get(assistant_message_id)
        if session_id is None:
            return
        await self._messages.update(
            session_id=session_id,
            message_id=assistant_message_id,
            column_to_update={
                "content": content,
                "is_completed": True,
                "is_fallback": is_fallback,
                "updated_at": _now(),
            },
        )


__all__ = ["PersistentMessageGateway"]
