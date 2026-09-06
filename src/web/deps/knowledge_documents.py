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

from src.ai.storage.base import FileService
from src.ai.storage.factory import new_file_service_from_storage_config
from src.app_context.registry import get_lifespan_service
from src.core.infra.storage_backends.factory import build_storage_backend_service
from src.core.infra.storage_backends.service.storage_backend_service import (
    StorageBackendService,
)
from src.core.infra.storage_backends.types import StorageBackendConfigInfo
from src.core.knowledge.documents.arq_enqueue import ArqDocumentEnqueuer
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
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.web.deps.session import SessionDep


class _ResolvableBackendConfig(StorageBackendConfigInfo):
    """Storage-config view with the provider fallback the factory reads.

    The file-service factory falls back to ``config.default_provider``
    when no provider is given; the stored backend config carries no such
    field, so this adds the blank default the factory tolerates.
    """

    default_provider: str = ""


class _BackendFileServiceResolver:
    """Resolve the storage file service for a knowledge base and tenant.

    Implements the ``StorageResolver`` seam consumed by the file-upload
    orchestration: the knowledge base names its storage backend, the
    registry resolves that backend to a provider + config, and the
    storage factory builds the concrete file service. Returns ``None``
    when no backend is configured, which the orchestration turns into
    the storage-configured error.
    """

    def __init__(
        self,
        *,
        kb_service: KBService,
        storage_backend_service: StorageBackendService,
    ) -> None:
        self._kb_service = kb_service
        self._storage_backend_service = storage_backend_service

    async def resolve_file_service(
        self,
        *,
        knowledge_base_id: str,
        tenant_id: int,
    ) -> FileService | None:
        """Return the file service for the knowledge base, or ``None``."""
        kb = await self._kb_service.get_knowledge_base_by_id(knowledge_base_id=knowledge_base_id)
        backend_id = (kb.storage_backend_id or "").strip()
        info = await self._storage_backend_service.resolve_backend(
            tenant_id=tenant_id,
            backend_id=backend_id,
        )
        if info is None:
            return None
        config = _ResolvableBackendConfig(**info.config.model_dump())
        return new_file_service_from_storage_config(info.provider, config)[0]


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
        storage_resolver=_BackendFileServiceResolver(
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
