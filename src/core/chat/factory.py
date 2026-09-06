"""Chat-domain request-scoped service factory.

Assembles a per-request ``ChatService`` on the shared ``AsyncSession``,
following the ``src.core.tenants.factory`` pattern: repositories and
services are built fresh for every request; ``web`` never imports ``db``.

The heavy execution seams are wired here at the composition root:

- the custom-agent resolver uses the real per-request agent service;
- the message gateway writes user and assistant rows so history load
  and follow-up generation can read the same turn;
- knowledge QA loads stored text chunks and streams through the
  existing pipeline steps;
- knowledge search ranks those same stored chunks by keyword;
- agent-chat reuses the knowledge-QA runner until the ReAct engine
  factory is composed. The ReAct loop is a later seam.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.common.exception import NotFoundError
from src.core.agents.service.custom_agent_service import CustomAgentService
from src.core.agents.service.factory import build_custom_agent_service
from src.core.agents.types import CustomAgentInfo
from src.core.chat.messages.gateway import PersistentMessageGateway
from src.core.chat.service import (
    AgentResolver,
    ChatService,
)
from src.core.chat.sessions.keyword_searcher import KeywordKnowledgeSearcher
from src.core.chat.sessions.knowledge_qa_runner import KnowledgeQARunner
from src.core.chat.stream.manager import StreamManager
from src.core.infra.models.factory import build_chat_model_service
from src.core.knowledge.chunks.factory import build_chunk_service
from src.core.knowledge.documents.factory import build_knowledge_service
from src.core.knowledge.knowledge_bases.factory import build_kb_service
from src.db.dao.message_repository import MessageRepository


class _AgentResolverImpl(AgentResolver):
    """Resolves agents through the request-scoped custom-agent service."""

    def __init__(self, service: CustomAgentService) -> None:
        self._service = service

    async def resolve(
        self,
        *,
        tenant_id: int,
        agent_id: str,
    ) -> CustomAgentInfo | None:
        try:
            return await self._service.get_agent_by_id(
                tenant_id=tenant_id,
                agent_id=agent_id,
            )
        except NotFoundError:
            return None


def build_chat_service(
    session: AsyncSession,
    *,
    tenant_id: int,
    user_id: str,
    request_id: str,
    stream_manager: StreamManager,
) -> ChatService:
    """Assemble a per-request ``ChatService`` on the shared session."""
    documents = build_knowledge_service(session)
    chunks = build_chunk_service(session)
    knowledge_runner = KnowledgeQARunner(
        chat_models=build_chat_model_service(session),
        documents=documents,
        chunks=chunks,
        knowledge_bases=build_kb_service(session),
    )
    return ChatService(
        tenant_id=tenant_id,
        user_id=user_id,
        request_id=request_id,
        agent_resolver=_AgentResolverImpl(build_custom_agent_service(session)),
        searcher=KeywordKnowledgeSearcher(documents=documents, chunks=chunks),
        knowledge_runner=knowledge_runner,
        agent_runner=knowledge_runner,
        message_gateway=PersistentMessageGateway(MessageRepository(session)),
        stream_manager=stream_manager,
    )


__all__ = ["build_chat_service"]
