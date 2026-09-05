"""Documents-domain request-scoped service factory.

See ``src.core.auth.factory`` for the pattern: repos are built per
request on the shared ``AsyncSession``; ``web`` never imports ``db``.
The web layer consumes this via a ``Depends`` forwarder.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.ai.storage.base import FileService
from src.core.infra.models.factory import build_chat_model_service
from src.core.infra.storage_backends.factory import build_storage_backend_service
from src.core.knowledge.chunks.factory import build_chunk_service
from src.core.knowledge.documents.backend_file_service import BackendFileServiceResolver
from src.core.knowledge.documents.cancel import ParseTaskInspector
from src.core.knowledge.documents.clone import ObjectCopier, VectorIndexReplicator
from src.core.knowledge.documents.create_file import StorageResolver
from src.core.knowledge.documents.documents_orchestrator import (
    KnowledgeDocumentsOrchestrator,
)
from src.core.knowledge.documents.move import ReparseTrigger
from src.core.knowledge.documents.parse_pipeline import DocumentReader
from src.core.knowledge.documents.process_document import DocumentProcessPipeline
from src.core.knowledge.documents.reparse import ReparseEnqueuer
from src.core.knowledge.documents.service.knowledge_service import KnowledgeService
from src.core.knowledge.documents.span_tracker import SpanTracker
from src.core.knowledge.documents.summary_refresh import DocumentSummaryRefresher
from src.core.knowledge.documents.temporary_document import TemporaryDocumentService
from src.core.knowledge.documents.upload_pipeline import DocumentTaskDispatcher
from src.core.knowledge.knowledge_bases.factory import build_kb_service
from src.core.knowledge.tags.factory import build_tag_service
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.dao.knowledge_span_repository import KnowledgeSpanRepository
from src.db.dao.knowledge_tag_repository import TagRepository
from src.db.dao.temporary_document_repository import TemporaryDocumentRepository


def build_temporary_document_service(session: AsyncSession) -> TemporaryDocumentService:
    """Per-request ``TemporaryDocumentService`` with a fresh repository."""
    return TemporaryDocumentService(repo=TemporaryDocumentRepository(session))


def build_knowledge_service(session: AsyncSession) -> KnowledgeService:
    """Per-request ``KnowledgeService`` with a fresh repository."""
    knowledge_repo = KnowledgeRepository(session)
    return KnowledgeService(
        knowledge_repo=knowledge_repo,
        tag_repo=TagRepository(session),
        summary_refresher=DocumentSummaryRefresher(
            knowledge_repo=knowledge_repo,
            chunk_repo=ChunkRepository(session),
            kb_service=build_kb_service(session),
            chat_models=build_chat_model_service(session),
        ),
    )


def build_span_tracker(session: AsyncSession) -> SpanTracker:
    """Per-request ``SpanTracker`` over the processing-spans table."""
    return SpanTracker(span_store=KnowledgeSpanRepository(session))


def build_document_process_pipeline(
    session: AsyncSession,
    *,
    reader: DocumentReader | None = None,
) -> DocumentProcessPipeline:
    """Per-job ``DocumentProcessPipeline`` for the worker runtime.

    Wires the same file-service resolver the upload path uses so a
    ``file_url`` row can ``save_bytes`` before parse.
    """
    kb_service = build_kb_service(session)
    return DocumentProcessPipeline(
        knowledge_repo=KnowledgeRepository(session),
        kb_service=kb_service,
        chunk_repo=ChunkRepository(session),
        reader=reader,
        file_service_resolver=BackendFileServiceResolver(
            kb_service=kb_service,
            storage_backend_service=build_storage_backend_service(session),
        ),
    )


def build_documents_orchestrator(
    session: AsyncSession,
    *,
    storage_resolver: StorageResolver | None = None,
    file_service: FileService | None = None,
    dispatcher: DocumentTaskDispatcher | None = None,
    enqueuer: ReparseEnqueuer | None = None,
    task_inspector: ParseTaskInspector | None = None,
    object_copier: ObjectCopier | None = None,
    index_replicator: VectorIndexReplicator | None = None,
    reparse_trigger: ReparseTrigger | None = None,
) -> KnowledgeDocumentsOrchestrator:
    """Per-request ``KnowledgeDocumentsOrchestrator``.

    Composes the merged repositories and domain services on the shared
    session. The optional infrastructure seams (storage resolution,
    task dispatch, parse enqueue, task inspection, object copying,
    index replication) default to ``None`` and are injected by the web
    layer once those domains land.
    """
    return KnowledgeDocumentsOrchestrator(
        knowledge_repo=KnowledgeRepository(session),
        chunk_repo=ChunkRepository(session),
        tag_repo=TagRepository(session),
        kb_service=build_kb_service(session),
        chunk_service=build_chunk_service(session),
        tag_service=build_tag_service(session),
        storage_resolver=storage_resolver,
        file_service=file_service,
        dispatcher=dispatcher,
        enqueuer=enqueuer,
        task_inspector=task_inspector,
        object_copier=object_copier,
        index_replicator=index_replicator,
        reparse_trigger=reparse_trigger,
    )


__all__ = [
    "build_document_process_pipeline",
    "build_documents_orchestrator",
    "build_knowledge_service",
    "build_span_tracker",
    "build_temporary_document_service",
]
