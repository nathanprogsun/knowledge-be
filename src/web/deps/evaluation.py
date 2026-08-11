"""Evaluation-domain FastAPI dependency factory.

One-line forwarder to ``src.core.evaluation.factory``: repositories are
assembled in ``core`` on the request-scoped ``AsyncSession`` so the
request's reads and writes share one transactional unit of work. The
knowledge-base service is wired so the evaluation flow may create a
temporary evaluation knowledge base; ``web`` never imports ``db``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.evaluation.factory import build_evaluation_service
from src.core.evaluation.service.evaluation_service import EvaluationService
from src.web.deps.context import get_tenant_id_dep
from src.web.deps.knowledge_bases import KBServiceDep
from src.web.deps.session import SessionDep


def get_evaluation_service(
    session: SessionDep,
    tenant_id: Annotated[int, Depends(get_tenant_id_dep)],
    kb_service: KBServiceDep,
) -> EvaluationService:
    """Build a per-request ``EvaluationService`` on the shared session.

    The tenant id is read from the auth context so the service is scoped
    to the caller's active workspace; the KB service backs the temporary
    evaluation-knowledge-base seams.
    """
    return build_evaluation_service(
        session,
        tenant_id=tenant_id,
        kb_service=kb_service,
    )


EvaluationServiceDep = Annotated[EvaluationService, Depends(get_evaluation_service)]


__all__ = ["EvaluationServiceDep", "get_evaluation_service"]
