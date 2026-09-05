"""Unit tests for the chat composition root."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.core.chat.factory import build_chat_service
from src.core.chat.messages.gateway import PersistentMessageGateway
from src.core.chat.sessions.keyword_searcher import KeywordKnowledgeSearcher
from src.core.chat.sessions.knowledge_qa_runner import KnowledgeQARunner


def test_build_chat_service_wires_knowledge_qa_runner() -> None:
    service = build_chat_service(
        MagicMock(),
        tenant_id=1,
        user_id="u",
        request_id="r",
    )
    assert isinstance(service._knowledge_runner, KnowledgeQARunner)
    assert service._agent_runner is service._knowledge_runner
    assert isinstance(service._searcher, KeywordKnowledgeSearcher)
    assert isinstance(service._message_gateway, PersistentMessageGateway)
