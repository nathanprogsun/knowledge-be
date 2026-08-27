"""Chat-domain request-scoped service factory.

Assembles a per-request ``ChatService`` on the shared ``AsyncSession``,
following the ``src.core.tenants.factory`` pattern: repositories and
services are built fresh for every request; ``web`` never imports ``db``.

The heavy execution seams are wired here at the composition root:

- the custom-agent resolver uses the real per-request agent service;
- the message gateway is a local (non-persistent) implementation that
  mints stable message ids — message persistence is a deferred seam;
- the knowledge searcher and the QA runners are explicit not-wired seams
  that raise a clear error when invoked. The retrieval engine and the
  chat-model / agent-loop wiring land in a later change; the endpoint
  surface and orchestration already exist and are exercised by tests
  through the dependency-override seam.
"""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from src.common.exception import NotFoundError, NotImplementedFeatureError
from src.core.agents.service.custom_agent_service import CustomAgentService
from src.core.agents.service.factory import build_custom_agent_service
from src.core.agents.types import CustomAgentInfo
from src.core.chat.bus import EventBus
from src.core.chat.pipeline.types import Context, SearchResult
from src.core.chat.service import (
    AgentResolver,
    AssistantMessage,
    ChatService,
    KnowledgeQARequestLike,
    KnowledgeSearcher,
    MessageGateway,
    QARunner,
    TagScope,
)

#: Code for the not-wired execution seams (deferred infrastructure).
_SEARCH_NOT_WIRED = "chat.search_not_wired"
_KNOWLEDGE_QA_NOT_WIRED = "chat.knowledge_qa_not_wired"
_AGENT_QA_NOT_WIRED = "chat.agent_qa_not_wired"


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


class _LocalMessageGateway(MessageGateway):
    """Message-shell gateway that mints ids without persisting rows.

    Deferred seam: once message persistence lands, this implementation is
    replaced by the real repository-backed gateway. The ids it produces
    are stable for the duration of one turn so the SSE ``agent_query``
    event carries a non-empty ``assistant_message_id``.
    """

    async def create_user_message(self, *, session_id: str, query: str) -> str:
        return str(uuid.uuid4())

    async def create_assistant_message(
        self,
        *,
        session_id: str,
        request_id: str,
        agent: CustomAgentInfo | None,
        model_id: str,
    ) -> AssistantMessage:
        return AssistantMessage(id=str(uuid.uuid4()), session_id=session_id)

    async def complete_assistant_message(
        self,
        *,
        assistant_message_id: str,
        content: str,
        is_fallback: bool = False,
    ) -> None:
        return None


class _NotWiredSearcher(KnowledgeSearcher):
    """Search seam that has not been connected to a retrieval engine."""

    async def search(
        self,
        *,
        tenant_id: int,
        query: str,
        knowledge_base_ids: list[str],
        knowledge_ids: list[str],
        tag_scopes: list[TagScope],
    ) -> list[SearchResult]:
        raise NotImplementedFeatureError(
            code=_SEARCH_NOT_WIRED,
            message=(
                "Knowledge search execution is not yet wired; "
                "the retrieval engine is a deferred seam"
            ),
        )


class _NotWiredRunner(QARunner):
    """QA runner seam that has not been connected to the execution layer."""

    def __init__(self, *, code: str, stage: str) -> None:
        self._code = code
        self._stage = stage

    async def run(
        self,
        *,
        ctx: Context,
        session_id: str,
        request: KnowledgeQARequestLike,
        agent: CustomAgentInfo | None,
        event_bus: EventBus,
    ) -> None:
        raise NotImplementedFeatureError(
            code=self._code,
            message=(
                f"{self._stage} pipeline execution is not yet wired; "
                "the model / store seams are deferred"
            ),
        )


def build_chat_service(
    session: AsyncSession,
    *,
    tenant_id: int,
    user_id: str,
    request_id: str,
) -> ChatService:
    """Assemble a per-request ``ChatService`` on the shared session."""
    return ChatService(
        tenant_id=tenant_id,
        user_id=user_id,
        request_id=request_id,
        agent_resolver=_AgentResolverImpl(build_custom_agent_service(session)),
        searcher=_NotWiredSearcher(),
        knowledge_runner=_NotWiredRunner(
            code=_KNOWLEDGE_QA_NOT_WIRED,
            stage="knowledge QA",
        ),
        agent_runner=_NotWiredRunner(
            code=_AGENT_QA_NOT_WIRED,
            stage="agent QA",
        ),
        message_gateway=_LocalMessageGateway(),
    )


__all__ = ["build_chat_service"]
