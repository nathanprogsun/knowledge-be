"""Unit tests for the persistent chat message gateway."""

from __future__ import annotations

from datetime import UTC, datetime

from src.core.agents.types import CustomAgentInfo
from src.core.chat.messages.gateway import PersistentMessageGateway
from src.core.chat.messages.types import ROLE_ASSISTANT, ROLE_USER
from src.db.models.message import Message

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


class _Repo:
    def __init__(self) -> None:
        self.created: list[Message] = []
        self.updates: list[dict[str, object]] = []

    async def create(self, row: Message) -> Message:
        self.created.append(row)
        return row

    async def update(
        self,
        *,
        session_id: str,
        message_id: str,
        column_to_update: dict[str, object],
    ) -> Message | None:
        self.updates.append(
            {
                "session_id": session_id,
                "message_id": message_id,
                "column_to_update": column_to_update,
            }
        )
        return None


def _agent() -> CustomAgentInfo:
    return CustomAgentInfo(
        id="agent-1",
        name="Test",
        tenant_id=1,
        config={},
        created_at=_NOW,
        updated_at=_NOW,
    )


async def test_gateway_writes_user_and_completes_assistant() -> None:
    repo = _Repo()
    gateway = PersistentMessageGateway(repo)  # type: ignore[arg-type]

    user_id = await gateway.create_user_message(session_id="s1", query="基金呢")
    assistant = await gateway.create_assistant_message(
        session_id="s1",
        request_id="r1",
        agent=_agent(),
        model_id="m1",
    )
    await gateway.complete_assistant_message(
        assistant_message_id=assistant.id,
        content="月报已出。",
        is_fallback=False,
    )

    assert repo.created[0].role == ROLE_USER
    assert repo.created[0].id == user_id
    assert repo.created[1].role == ROLE_ASSISTANT
    assert repo.created[1].is_completed is False
    assert repo.updates[0]["message_id"] == assistant.id
    update = repo.updates[0]["column_to_update"]
    assert isinstance(update, dict)
    assert update["content"] == "月报已出。"
    assert update["is_completed"] is True
