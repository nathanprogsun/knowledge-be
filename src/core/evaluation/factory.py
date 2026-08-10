"""Evaluation-domain request-scoped service factory.

Repositories are built per request on the shared ``AsyncSession``;
``web`` never imports ``db``. Mirrors the pattern in
``src.core.infra.models.factory`` and the chat / knowledge factories.

The heavy seams (the knowledge-base adapter, the knowledge factory, and
the QA runner) are left at their defaults so a request-scoped
:class:`EvaluationService` can be constructed cheaply; the web layer
injects the live seams where the evaluation endpoints are wired.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.evaluation.dataset import DatasetService
from src.core.evaluation.service.evaluation_service import EvaluationService
from src.core.evaluation.service.seams import KnowledgeBaseInfoLike
from src.core.infra.models.factory import build_model_service
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.db.dao.evaluation_repository import (
    EvaluationDatasetRepository,
    EvaluationMetricRepository,
    EvaluationRepository,
    EvaluationRunRepository,
)


class _KBEvaluationAdapter:
    """Adapts :class:`KBService` to the evaluation flow's seams.

    The evaluation service needs only a narrow slice of the KB surface
    (create / read / delete a temporary evaluation KB); this adapter
    forwards those calls and translates the service's ``KnowledgeBaseInfo``
    view into the model-bindings projection the evaluation flow reads.
    """

    def __init__(self, kb_service: KBService, *, tenant_id: int) -> None:
        self._kb_service = kb_service
        self._tenant_id = tenant_id

    async def create_knowledge_base(
        self,
        *,
        name: str,
        description: str,
        embedding_model_id: str,
        summary_model_id: str,
    ) -> str:
        info = await self._kb_service.create_knowledge_base(
            tenant_id=self._tenant_id,
            name=name,
            description=description,
            embedding_model_id=embedding_model_id,
            summary_model_id=summary_model_id,
            is_temporary=True,
        )
        return info.id

    async def delete_knowledge_base(self, *, knowledge_base_id: str) -> bool:
        return await self._kb_service.delete_knowledge_base(
            knowledge_base_id=knowledge_base_id,
        )

    async def get_knowledge_base_by_id(
        self,
        *,
        knowledge_base_id: str,
    ) -> KnowledgeBaseInfoLike:
        return await self._kb_service.get_knowledge_base_by_id(
            knowledge_base_id=knowledge_base_id,
        )


def build_evaluation_service(
    session: AsyncSession,
    *,
    tenant_id: int,
    kb_service: KBService | None = None,
) -> EvaluationService:
    """Per-request :class:`EvaluationService` on ``session``.

    The dataset loader resolves the built-in sample dataset; the model
    service and evaluation repositories are built fresh on the shared
    session. When ``kb_service`` is supplied (as it is by the web layer)
    the KB adapter is wired so an evaluation may create a temporary
    knowledge base; otherwise those seams stay ``None`` and the caller
    must pass ``knowledge_base_id`` explicitly.
    """
    kb_adapter = None
    if kb_service is not None:
        kb_adapter = _KBEvaluationAdapter(kb_service, tenant_id=tenant_id)
    return EvaluationService(
        tenant_id=tenant_id,
        evaluation_repo=EvaluationRepository(session),
        dataset_repo=EvaluationDatasetRepository(session),
        run_repo=EvaluationRunRepository(session),
        metric_repo=EvaluationMetricRepository(session),
        model_service=build_model_service(session),
        dataset_service=DatasetService(),
        kb_creator=kb_adapter,
        kb_reader=kb_adapter,
        kb_deleter=kb_adapter,
    )


__all__ = ["build_evaluation_service"]
