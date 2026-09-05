"""Run document summary generation on the regenerate-summary path.

``process_summary`` already writes ``description`` and ``summary_status``.
The HTTP handler used to flip the row to ``pending`` and stop. This
refresher resolves a KnowledgeQA client and calls that function so the
drawer can leave the spinner.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.ai.llm.types import Chat
from src.common.exception import NotFoundError, ValidationError
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.documents.summary import process_summary
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_repository import KnowledgeRepository

_NOT_FOUND_CODE = "knowledge.not_found"
_SUMMARY_PROMPT = "Write a concise summary of the document. Respond in {{language}}."


@runtime_checkable
class ChatClientFactory(Protocol):
    """Resolves a live chat client without importing the infra service type."""

    async def get_chat_model(self, *, tenant_id: int, model_id: str) -> Chat: ...

    async def first_knowledge_qa_id(self, *, tenant_id: int) -> str | None: ...


class DocumentSummaryRefresher:
    """Runs ``process_summary`` with the KB model, else the first KnowledgeQA."""

    def __init__(
        self,
        *,
        knowledge_repo: KnowledgeRepository,
        chunk_repo: ChunkRepository,
        kb_service: KBService,
        chat_models: ChatClientFactory,
    ) -> None:
        self._knowledge_repo = knowledge_repo
        self._chunk_repo = chunk_repo
        self._kb_service = kb_service
        self._chat_models = chat_models

    async def refresh(self, *, tenant_id: int, knowledge_id: str) -> Knowledge:
        """Generate the summary and return the updated document."""
        row = await self._knowledge_repo.get_by_id(tenant_id, knowledge_id)
        if row is None:
            raise NotFoundError(
                code=_NOT_FOUND_CODE,
                message="knowledge not found",
            )
        kb = await self._kb_service.get_knowledge_base_by_id_and_tenant(
            tenant_id=tenant_id,
            knowledge_base_id=row.knowledge_base_id,
        )
        model_id = (kb.summary_model_id or "").strip()
        if not model_id:
            fallback = await self._chat_models.first_knowledge_qa_id(tenant_id=tenant_id)
            model_id = fallback or ""
        if not model_id:
            raise ValidationError(
                code="knowledge.summary_model_not_configured",
                message="summary model is not configured",
            )
        chat = await self._chat_models.get_chat_model(
            tenant_id=tenant_id,
            model_id=model_id,
        )
        result = await process_summary(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            chat=chat,
            knowledge_repo=self._knowledge_repo,
            chunk_repo=self._chunk_repo,
            kb_service=self._kb_service,
            prompt=_SUMMARY_PROMPT,
        )
        return result.knowledge


__all__ = ["ChatClientFactory", "DocumentSummaryRefresher"]
