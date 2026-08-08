"""Request-scoped document orchestration facade.

Wraps the standalone create / reparse / cancel / clone / move / delete
modules so the web layer composes the merged document domain per request
on the shared session without ever importing ``db``. The facade builds
its repositories and services once and exposes thin async methods; the
optional infrastructure seams (storage resolution, task dispatch, parse
enqueue, task inspection, object copying, index replication) are
constructor-injected and default to safe values until those domains
land.

``build_documents_orchestrator`` in ``factory.py`` wires the concrete
repositories; this module owns no repository construction.
"""

from __future__ import annotations

from src.ai.storage.base import FileService, FileUpload
from src.common.json import JsonObject
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.chunks.service.chunk_service import ChunkService
from src.core.knowledge.documents.cancel import (
    ParseTaskInspector,
    cancel_knowledge_parse,
)
from src.core.knowledge.documents.clone import (
    ObjectCopier,
    VectorIndexReplicator,
    clone_knowledge,
)
from src.core.knowledge.documents.create_file import (
    StorageResolver,
    TenantStorageInfo,
    create_knowledge_from_file,
)
from src.core.knowledge.documents.create_manual import create_knowledge_from_manual
from src.core.knowledge.documents.create_passage import create_knowledge_from_passage
from src.core.knowledge.documents.create_url import create_knowledge_from_url
from src.core.knowledge.documents.delete import delete_knowledge
from src.core.knowledge.documents.move import (
    ReparseTrigger,
    move_knowledge,
)
from src.core.knowledge.documents.reparse import (
    DocumentProcessPayload,
    ReparseEnqueuer,
    reparse_knowledge,
)
from src.core.knowledge.documents.upload_pipeline import DocumentTaskDispatcher
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.tags.service.tag_service import TagService
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.dao.knowledge_tag_repository import TagRepository


class _NoopReparseEnqueuer:
    """No-op parse-submission hook used until the worker domain lands.

    Keeps the reparse endpoint functional: the row is reset to a fresh
    pending attempt and the enqueuer accepts the payload without
    persisting a broker task. Swap for a broker-backed implementation
    in the worker wave.
    """

    async def enqueue_manual_process(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        content: str,
    ) -> None:
        """Accept a manual-content parse submission without persisting it."""

    async def enqueue_document_process(
        self,
        *,
        tenant_id: int,
        payload: DocumentProcessPayload,
    ) -> None:
        """Accept a document parse submission without persisting it."""


class KnowledgeDocumentsOrchestrator:
    """Per-request facade over the document orchestration functions.

    The CRUD surface (get / list / update) is intentionally not exposed
    here — the merged ``KnowledgeService`` owns that, and the web layer
    consumes it through its own dependency.
    """

    def __init__(
        self,
        *,
        knowledge_repo: KnowledgeRepository,
        chunk_repo: ChunkRepository,
        tag_repo: TagRepository,
        kb_service: KBService,
        chunk_service: ChunkService,
        tag_service: TagService,
        storage_resolver: StorageResolver | None = None,
        file_service: FileService | None = None,
        dispatcher: DocumentTaskDispatcher | None = None,
        enqueuer: ReparseEnqueuer | None = None,
        task_inspector: ParseTaskInspector | None = None,
        object_copier: ObjectCopier | None = None,
        index_replicator: VectorIndexReplicator | None = None,
        reparse_trigger: ReparseTrigger | None = None,
        tenant_storage: TenantStorageInfo | None = None,
    ) -> None:
        self._knowledge_repo = knowledge_repo
        self._chunk_repo = chunk_repo
        self._tag_repo = tag_repo
        self._kb_service = kb_service
        self._chunk_service = chunk_service
        self._tag_service = tag_service
        self._storage_resolver = storage_resolver
        self._file_service = file_service
        self._dispatcher = dispatcher
        self._enqueuer: ReparseEnqueuer = (
            enqueuer if enqueuer is not None else _NoopReparseEnqueuer()
        )
        self._task_inspector = task_inspector
        self._object_copier = object_copier
        self._index_replicator = index_replicator
        self._reparse_trigger = reparse_trigger
        self._tenant_storage = tenant_storage

    # ── Upload ──────────────────────────────────────────────────────

    async def create_from_file(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        file: FileUpload,
        metadata: JsonObject | None = None,
        enable_multimodel: bool | None = None,
        custom_file_name: str | None = None,
        tag_ids: list[str] | None = None,
        channel: str = "",
        process_overrides: JsonObject | None = None,
        language: str = "",
    ) -> Knowledge:
        """Create a document from an uploaded file."""
        return await create_knowledge_from_file(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            file=file,
            knowledge_repo=self._knowledge_repo,
            kb_service=self._kb_service,
            storage_resolver=self._storage_resolver,
            file_service=self._file_service,
            tag_service=self._tag_service,
            dispatcher=self._dispatcher,
            metadata=metadata,
            enable_multimodel=enable_multimodel,
            custom_file_name=custom_file_name,
            tag_ids=tag_ids,
            channel=channel,
            process_overrides=process_overrides,
            tenant_storage=self._tenant_storage,
            language=language,
        )

    async def create_from_url(
        self,
        *,
        tenant_id: int,
        kb_id: str,
        url: str,
        file_name: str | None = None,
        file_type: str | None = None,
        enable_multimodel: bool | None = None,
        title: str | None = None,
        tag_ids: list[str] | None = None,
        channel: str | None = None,
    ) -> Knowledge:
        """Create a document from a web URL or a downloadable file URL."""
        return await create_knowledge_from_url(
            tenant_id=tenant_id,
            kb_id=kb_id,
            url=url,
            file_name=file_name,
            file_type=file_type,
            enable_multimodel=enable_multimodel,
            title=title,
            tag_ids=tag_ids,
            channel=channel,
            knowledge_repo=self._knowledge_repo,
            kb_service=self._kb_service,
            tag_repo=self._tag_repo,
        )

    async def create_from_passage(
        self,
        *,
        tenant_id: int,
        kb_id: str,
        passages: list[str],
        channel: str | None = None,
        sync: bool = False,
    ) -> Knowledge:
        """Create a document from text passages."""
        return await create_knowledge_from_passage(
            tenant_id=tenant_id,
            kb_id=kb_id,
            passages=passages,
            channel=channel,
            sync=sync,
            knowledge_repo=self._knowledge_repo,
            kb_service=self._kb_service,
            chunk_repo=self._chunk_repo,
        )

    async def create_from_manual(
        self,
        *,
        tenant_id: int,
        kb_id: str,
        title: str,
        content: str,
        status: str | None = None,
        tag_ids: list[str] | None = None,
        channel: str | None = None,
        process_overrides: JsonObject | None = None,
    ) -> Knowledge:
        """Create a manual Markdown document."""
        return await create_knowledge_from_manual(
            tenant_id=tenant_id,
            kb_id=kb_id,
            title=title,
            content=content,
            status=status,
            tag_ids=tag_ids,
            channel=channel,
            process_overrides=process_overrides,
            knowledge_repo=self._knowledge_repo,
            kb_service=self._kb_service,
            tag_repo=self._tag_repo,
        )

    # ── Lifecycle ───────────────────────────────────────────────────

    async def reparse(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        process_overrides: JsonObject | None = None,
    ) -> Knowledge:
        """Reset a document for a fresh parse attempt and submit it."""
        return await reparse_knowledge(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            knowledge_repo=self._knowledge_repo,
            kb_service=self._kb_service,
            chunk_service=self._chunk_service,
            enqueuer=self._enqueuer,
            process_overrides=process_overrides,
        )

    async def cancel_parse(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
    ) -> Knowledge:
        """Cancel an in-flight parse of a document."""
        return await cancel_knowledge_parse(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            knowledge_repo=self._knowledge_repo,
            task_inspector=self._task_inspector,
        )

    async def clone(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        target_kb_id: str,
    ) -> Knowledge | None:
        """Clone a completed document into another knowledge base."""
        return await clone_knowledge(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            target_kb_id=target_kb_id,
            knowledge_repo=self._knowledge_repo,
            chunk_repo=self._chunk_repo,
            tag_repo=self._tag_repo,
            kb_service=self._kb_service,
            object_copier=self._object_copier,
            index_replicator=self._index_replicator,
        )

    async def move(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        source_kb_id: str,
        target_kb_id: str,
        mode: str,
    ) -> Knowledge:
        """Move a completed document into another knowledge base."""
        return await move_knowledge(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            source_kb_id=source_kb_id,
            target_kb_id=target_kb_id,
            mode=mode,
            knowledge_repo=self._knowledge_repo,
            chunk_repo=self._chunk_repo,
            tag_service=self._tag_service,
            kb_service=self._kb_service,
            index_replicator=self._index_replicator,
            reparse_trigger=self._reparse_trigger,
        )

    async def delete(self, *, tenant_id: int, id: str) -> bool:
        """Soft-delete a document and cascade its chunks."""
        return await delete_knowledge(
            tenant_id=tenant_id,
            id=id,
            knowledge_repo=self._knowledge_repo,
            chunk_repo=self._chunk_repo,
        )


__all__ = ["KnowledgeDocumentsOrchestrator"]
