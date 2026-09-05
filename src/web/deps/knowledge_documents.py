"""Document-domain FastAPI dependency factories.

Forwards the merged ``KnowledgeService`` (CRUD) and the
``KnowledgeDocumentsOrchestrator`` (upload / reparse / cancel / clone /
move / delete) to the request-scoped factories in ``core`` so a
mutation and its audit row share one transactional unit of work.
``web`` never imports ``db``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from src.app_context.registry import get_lifespan_service
from src.core.infra.storage_backends.factory import build_storage_backend_service
from src.core.knowledge.documents.arq_enqueue import ArqDocumentEnqueuer
from src.core.knowledge.documents.backend_file_service import BackendFileServiceResolver
from src.core.knowledge.documents.documents_orchestrator import (
    KnowledgeDocumentsOrchestrator,
)
from src.core.knowledge.documents.factory import (
    build_documents_orchestrator,
    build_knowledge_service,
    build_span_tracker,
)
from src.core.knowledge.documents.service.knowledge_service import KnowledgeService
from src.core.knowledge.documents.span_tracker import SpanTracker
from src.core.knowledge.knowledge_bases.factory import build_kb_service
from src.web.deps.session import SessionDep


def get_knowledge_service(session: SessionDep) -> KnowledgeService:
    """Build a per-request ``KnowledgeService`` on the shared session."""
    return build_knowledge_service(session)


KnowledgeServiceDep = Annotated[KnowledgeService, Depends(get_knowledge_service)]


def get_span_tracker(session: SessionDep) -> SpanTracker:
    """Build a per-request ``SpanTracker`` on the shared session."""
    return build_span_tracker(session)


SpanTrackerDep = Annotated[SpanTracker, Depends(get_span_tracker)]


def _request_document_enqueuer(request: Request) -> ArqDocumentEnqueuer | None:
    """Return the APP-scope ARQ enqueuer when the Redis pool is live."""
    if not hasattr(request.app.state, "lifespan_service"):
        return None
    lifespan = get_lifespan_service(request.app)
    if lifespan.arq_redis is None:
        return None
    return ArqDocumentEnqueuer(
        lifespan.arq_redis,
        queue_name=lifespan.arq_queue_name,
    )


def get_documents_orchestrator(
    session: SessionDep,
    request: Request,
) -> KnowledgeDocumentsOrchestrator:
    """Build a per-request ``KnowledgeDocumentsOrchestrator``.

    The storage resolver is wired from the knowledge-base and
    storage-backend services so a file upload resolves the configured
    storage engine. The ARQ enqueuer is taken from the APP-scope pool
    when Redis connected at startup.
    """
    enqueuer = _request_document_enqueuer(request)
    kb_service = build_kb_service(session)
    storage_backend_service = build_storage_backend_service(session)
    return build_documents_orchestrator(
        session,
        storage_resolver=BackendFileServiceResolver(
            kb_service=kb_service,
            storage_backend_service=storage_backend_service,
        ),
        dispatcher=enqueuer,
        enqueuer=enqueuer,
    )


KnowledgeDocumentsDep = Annotated[
    KnowledgeDocumentsOrchestrator,
    Depends(get_documents_orchestrator),
]


__all__ = [
    "KnowledgeDocumentsDep",
    "KnowledgeServiceDep",
    "SpanTrackerDep",
    "get_documents_orchestrator",
    "get_knowledge_service",
    "get_span_tracker",
]
