"""Document processing pipeline orchestrator (parse → chunk → embed → index).

Runs the four pipeline stages for one knowledge item and owns the
document status transitions:

- entry guards short-circuit ``deleting`` / ``cancelled`` (abort) and
  ``completed`` (idempotent skip), while ``failed`` rows are retryable;
- the row is flipped to ``processing`` before any work happens;
- a ``file_url`` row with an empty ``file_path`` is downloaded and
  stored through ``FileService.save_bytes`` before parse; type ``url``
  is left without bytes;
- parse → chunk → persist → index runs in order; a failure in any stage
  marks the row ``failed`` with the error message;
- on success the row becomes queryable (``enable_status=enabled``,
  ``storage_size``, ``processed_at``). Rows that produced text chunks
  stay ``processing`` because the enrichment stages (summary / question /
  graph) still have work to fan out; rows with no text chunks complete
  immediately. Advancing ``processing`` → ``completed`` after enrichment
  is the job of the post-process stage, whose dispatch is a deferred seam
  (``PostProcessDispatcher``) the worker layer wires later.

All external dependencies are injected: the document reader (docreader),
the storage file reader, the file-service resolver, the embedding
resolver, the retrieval-index resolver, the post-process dispatcher and
the tenant storage accounting. Until those are composed by the worker
layer, the seams stay protocols.
"""

from __future__ import annotations

import contextlib
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, TypeVar, runtime_checkable

from src.ai.embedding import Context, Embedder, TaskContext
from src.app_logging import logger
from src.common.exception import ApplicationError, ValidationError
from src.common.json import JsonObject
from src.core.knowledge.documents.chunk_pipeline import (
    chunk_markdown,
)
from src.core.knowledge.documents.chunk_rows import build_chunk_rows
from src.core.knowledge.documents.create_file import StorageResolver
from src.core.knowledge.documents.file_url_store import (
    FileUrlDownloader,
    FileUrlStoreResult,
    store_file_url_bytes,
)
from src.core.knowledge.documents.index_pipeline import IndexEngine, build_index_infos
from src.core.knowledge.documents.parse_pipeline import (
    DocumentReader,
    FileReader,
    ParseResult,
    ReadRequest,
    parse_document,
)
from src.core.knowledge.documents.types import (
    PARSE_STATUS_CANCELLED,
    PARSE_STATUS_COMPLETED,
    PARSE_STATUS_DELETING,
    PARSE_STATUS_FAILED,
    PARSE_STATUS_PENDING,
    PARSE_STATUS_PROCESSING,
    SUMMARY_STATUS_NONE,
)
from src.core.knowledge.documents.upload_pipeline import (
    is_audio_type,
    is_image_type,
    is_video_type,
)
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.knowledge import Document

# ── Constants ──────────────────────────────────────────────────────────

# Rows become queryable once chunks and indexes are persisted.
_ENABLE_STATUS_ENABLED = "enabled"

# Message stamped when no document reader is configured.
_DOCREADER_NOT_CONFIGURED = "document parsing service is not configured"

# Bounds the error message written to the documents row.
_MAX_ERROR_MESSAGE_CHARS = 2048

# Result type of a best-effort operation (return value is discarded).
_IgnoreT = TypeVar("_IgnoreT")


def _asr_enabled(config: JsonObject | None) -> bool:
    """True when an ASR config enables transcription.

    Accepts either an explicit ``enabled`` flag or a configured model id
    / name, matching the shared media-prerequisite semantics.
    """
    if not isinstance(config, dict):
        return False
    if config.get("enabled") is True:
        return True
    return bool(config.get("model_id")) or bool(config.get("model_name"))


def _metadata_title(result: ParseResult) -> str:
    """Return the extracted page title from a parse result, or ""."""
    title = result.metadata.get("title")
    return title if isinstance(title, str) else ""


# ── Domain value types ────────────────────────────────────────────────


@dataclass(frozen=True)
class ProcessOutcome:
    """Result of one pipeline run for a knowledge item."""

    parse_status: str
    enable_status: str = ""
    summary_status: str = SUMMARY_STATUS_NONE
    storage_size: int = 0
    error_message: str | None = None
    text_chunk_count: int = 0
    skipped: bool = False


@dataclass(frozen=True)
class TenantStorageInfo:
    """Storage accounting used by the quota gate (0 disables the check)."""

    storage_quota: int = 0
    storage_used: int = 0


@dataclass(frozen=True)
class PostProcessPayload:
    """Payload of the post-process fan-out task (summary / question / graph)."""

    tenant_id: int
    knowledge_id: str
    knowledge_base_id: str
    language: str = ""


# ── Injectable seams ──────────────────────────────────────────────────


@runtime_checkable
class EmbeddingResolver(Protocol):
    """Resolve the embedder for a KB's configured embedding model."""

    async def resolve_embedder(self, *, embedding_model_id: str) -> Embedder | None:
        """Return the embedder, or ``None`` when the model is unavailable."""
        ...


@runtime_checkable
class IndexEngineResolver(Protocol):
    """Resolve the retrieval index engine for a KB."""

    async def resolve_engine(
        self,
        *,
        tenant_id: int,
        vector_store_id: str | None,
    ) -> IndexEngine | None:
        """Return the index engine, or ``None`` when unavailable."""
        ...


@runtime_checkable
class PostProcessDispatcher(Protocol):
    """Enqueue the enrichment fan-out task after a successful index."""

    async def dispatch(self, *, payload: PostProcessPayload) -> None:
        """Persist the task for later processing."""
        ...


@runtime_checkable
class TenantStorageResolver(Protocol):
    """Tenant storage accounting for the quota gate and usage adjustment."""

    async def get_storage(self, *, tenant_id: int) -> TenantStorageInfo:
        """Return the tenant's current storage accounting."""
        ...

    async def adjust_storage_used(self, *, tenant_id: int, delta: int) -> None:
        """Adjust the tenant's used-storage by ``delta`` (best-effort)."""
        ...


def kb_needs_embedding(indexing_strategy: JsonObject | None) -> bool:
    """True when vector or keyword indexing is enabled.

    An empty strategy resolves to the default (both enabled), mirroring
    the upstream default indexing strategy.
    """
    strategy = indexing_strategy if isinstance(indexing_strategy, dict) else {}
    if not strategy:
        return True
    return bool(strategy.get("vector_enabled")) or bool(strategy.get("keyword_enabled"))


def _validate_file_prerequisites(
    *,
    kb: KnowledgeBaseInfo,
    file_type: str,
    enable_multimodel: bool,
) -> str | None:
    """Return an error message when the file type cannot be processed.

    Video imports are rejected outright; image imports require multimodal
    processing to be enabled; audio imports require a configured ASR
    model. Returns ``None`` when the type is processable.
    """
    if is_video_type(file_type):
        return "video files are not supported"
    if is_image_type(file_type) and not enable_multimodel:
        return "image import requires multimodal processing to be enabled"
    if is_audio_type(file_type) and not _asr_enabled(kb.asr_config):
        return "audio import requires an ASR model to be configured"
    return None


# ── Pipeline ──────────────────────────────────────────────────────────


class DocumentProcessPipeline:
    """Stateless per-item processor over injected repositories and seams."""

    def __init__(
        self,
        *,
        knowledge_repo: KnowledgeRepository,
        kb_service: KBService,
        chunk_repo: ChunkRepository,
        reader: DocumentReader | None = None,
        file_reader: FileReader | None = None,
        embedding_resolver: EmbeddingResolver | None = None,
        index_engine_resolver: IndexEngineResolver | None = None,
        post_process_dispatcher: PostProcessDispatcher | None = None,
        storage_resolver: TenantStorageResolver | None = None,
        file_service_resolver: StorageResolver | None = None,
        file_url_downloader: FileUrlDownloader | None = None,
    ) -> None:
        self._knowledge_repo = knowledge_repo
        self._kb_service = kb_service
        self._chunk_repo = chunk_repo
        self._reader = reader
        self._file_reader = file_reader
        self._embedding_resolver = embedding_resolver
        self._index_engine_resolver = index_engine_resolver
        self._post_process_dispatcher = post_process_dispatcher
        self._storage_resolver = storage_resolver
        self._file_service_resolver = file_service_resolver
        self._file_url_downloader = file_url_downloader

    # ── Entry ────────────────────────────────────────────────────────

    async def run(
        self,
        *,
        ctx: Context,
        tenant_id: int,
        knowledge_id: str,
        knowledge_base_id: str,
        file_path: str = "",
        file_name: str = "",
        file_type: str = "",
        url: str = "",
        enable_multimodel: bool = False,
        language: str = "",
        request_id: str = "",
        now: datetime | None = None,
    ) -> ProcessOutcome:
        """Process one knowledge item end-to-end.

        Entry guards mirror the worker semantics: a row being deleted or
        cancelled aborts immediately, a completed row is an idempotent
        skip, and a failed row is retried.
        """
        now = now or datetime.now(UTC)
        row = await self._knowledge_repo.get_by_id(tenant_id, knowledge_id)
        if row is None:
            return self._skipped(PARSE_STATUS_PENDING)
        if row.parse_status in (PARSE_STATUS_DELETING, PARSE_STATUS_CANCELLED):
            return self._skipped(row.parse_status)
        if row.parse_status == PARSE_STATUS_COMPLETED:
            return self._skipped(row.parse_status)

        kb = await self._kb_service.get_knowledge_base_by_id(knowledge_base_id=knowledge_base_id)

        # Re-check abort before flipping to "processing" so a cancel
        # racing this worker cannot be overwritten.
        if await self._is_aborted(tenant_id=tenant_id, knowledge_id=knowledge_id):
            status = await self._current_status(tenant_id=tenant_id, knowledge_id=knowledge_id)
            return self._skipped(status or PARSE_STATUS_CANCELLED)

        processing = await self._mark_processing(row, now)

        reason = _validate_file_prerequisites(
            kb=kb,
            file_type=file_type,
            enable_multimodel=enable_multimodel,
        )
        if reason is not None:
            return await self._fail(processing, reason, now)

        embedder, engine = await self._resolve_ai(kb=kb, tenant_id=tenant_id)

        await self._cleanup_stale(
            ctx=ctx,
            row=processing,
            engine=engine,
            embedder=embedder,
            knowledge_type=row.type,
            now=now,
        )

        stored = await self._store_file_url_or_fail(
            processing,
            file_path=file_path,
            url=url,
            file_name=file_name,
            file_type=file_type,
            now=now,
        )
        if isinstance(stored, ProcessOutcome):
            return stored
        processing = stored.row
        parse_result = await self._parse_or_fail(
            processing,
            stored=stored,
            file_path=file_path,
            file_name=file_name,
            file_type=file_type,
            request_id=request_id,
            now=now,
        )
        if isinstance(parse_result, ProcessOutcome):
            return parse_result

        # URL imports adopt the extracted page title when no title is set.
        if url and (not processing.title or processing.title == url):
            extracted = _metadata_title(parse_result)
            if extracted:
                processing = processing.model_copy(update={"title": extracted, "updated_at": now})
                with contextlib.suppress(Exception):
                    await self._knowledge_repo.update(processing)

        # ── Chunk ──────────────────────────────────────────────────
        chunking = chunk_markdown(parse_result.markdown_content, kb.chunking_config)
        all_rows, text_rows = build_chunk_rows(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
            knowledge_base_id=knowledge_base_id,
            chunks=chunking.chunks,
            parent_chunks=chunking.parent_chunks,
            now=now,
        )
        try:
            await self._chunk_repo.create_many(all_rows)
        except Exception as exc:
            return await self._fail(processing, f"create chunks failed: {exc}", now)

        # ── Embed + index ──────────────────────────────────────────
        if await self._is_aborted(tenant_id=tenant_id, knowledge_id=knowledge_id):
            status = await self._current_status(tenant_id=tenant_id, knowledge_id=knowledge_id)
            return self._skipped(status or PARSE_STATUS_CANCELLED)

        storage_size = 0
        if embedder is not None and engine is not None and text_rows:
            index_infos = build_index_infos(
                chunks=text_rows,
                knowledge_id=knowledge_id,
                knowledge_base_id=knowledge_base_id,
                title=processing.title,
            )
            storage_size = engine.estimate_storage_size(ctx, embedder, index_infos)
            if await self._quota_exceeded(tenant_id, storage_size):
                return await self._fail(processing, "storage quota exceeded", now)
            try:
                await engine.batch_index(ctx, embedder, index_infos)
            except Exception as exc:
                await self._remove_chunks_and_index(
                    ctx=ctx,
                    row=processing,
                    engine=engine,
                    embedder=embedder,
                    knowledge_type=row.type,
                    now=now,
                )
                return await self._fail(processing, f"batch index failed: {exc}", now)

        # ── Finalize ───────────────────────────────────────────────
        if await self._is_aborted(tenant_id=tenant_id, knowledge_id=knowledge_id):
            status = await self._current_status(tenant_id=tenant_id, knowledge_id=knowledge_id)
            if (status or PARSE_STATUS_CANCELLED) == PARSE_STATUS_DELETING:
                await self._remove_chunks_and_index(
                    ctx=ctx,
                    row=processing,
                    engine=engine,
                    embedder=embedder,
                    knowledge_type=row.type,
                    now=now,
                )
            return self._skipped(status or PARSE_STATUS_CANCELLED)

        return await self._finalize(
            row=processing,
            text_chunk_count=len(text_rows),
            storage_size=storage_size,
            language=language,
            now=now,
        )

    # ── Stage helpers ──────────────────────────────────────────────

    async def _current_status(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
    ) -> str | None:
        row = await self._knowledge_repo.get_by_id(tenant_id, knowledge_id)
        if row is None:
            return None
        return row.parse_status

    async def _is_aborted(self, *, tenant_id: int, knowledge_id: str) -> bool:
        status = await self._current_status(tenant_id=tenant_id, knowledge_id=knowledge_id)
        return status in (PARSE_STATUS_DELETING, PARSE_STATUS_CANCELLED)

    async def _store_file_url_or_fail(
        self,
        processing: Document,
        *,
        file_path: str,
        url: str,
        file_name: str,
        file_type: str,
        now: datetime,
    ) -> FileUrlStoreResult | ProcessOutcome:
        """Persist ``file_url`` bytes, or mark the row failed."""
        try:
            return await store_file_url_bytes(
                row=processing,
                file_path=file_path,
                url=url,
                file_name=file_name,
                file_type=file_type,
                resolver=self._file_service_resolver,
                downloader=self._file_url_downloader,
                knowledge_repo=self._knowledge_repo,
                now=now,
            )
        except ApplicationError as exc:
            return await self._fail(processing, exc.message, now)

    async def _parse_or_fail(
        self,
        processing: Document,
        *,
        stored: FileUrlStoreResult,
        file_path: str,
        file_name: str,
        file_type: str,
        request_id: str,
        now: datetime,
    ) -> ParseResult | ProcessOutcome:
        """Parse from stored bytes/path, or mark the row failed."""
        if self._reader is None:
            return await self._fail(processing, _DOCREADER_NOT_CONFIGURED, now)
        try:
            return await parse_document(
                reader=self._reader,
                request=ReadRequest(
                    file_content=stored.file_content,
                    file_path=stored.file_path or file_path,
                    file_name=file_name,
                    file_type=file_type,
                    url=stored.url,
                    title=processing.title,
                    request_id=request_id,
                ),
                file_reader=self._file_reader,
            )
        except Exception as exc:
            return await self._fail(processing, f"document read failed: {exc}", now)

    async def _mark_processing(self, row: Document, now: datetime) -> Document:
        """Flip the row to ``processing`` and return the persisted row."""
        updated = row.model_copy(
            update={"parse_status": PARSE_STATUS_PROCESSING, "updated_at": now}
        )
        return await self._knowledge_repo.update(updated)

    async def _fail(self, row: Document, message: str, now: datetime) -> ProcessOutcome:
        """Mark the row failed with the error message and return the outcome."""
        message = message[:_MAX_ERROR_MESSAGE_CHARS]
        updated = row.model_copy(
            update={
                "parse_status": PARSE_STATUS_FAILED,
                "error_message": message,
                "updated_at": now,
            }
        )
        with contextlib.suppress(Exception):
            await self._knowledge_repo.update(updated)
        return ProcessOutcome(
            parse_status=PARSE_STATUS_FAILED,
            enable_status=row.enable_status,
            summary_status=row.summary_status,
            storage_size=row.storage_size,
            error_message=message,
        )

    async def _finalize(
        self,
        *,
        row: Document,
        text_chunk_count: int,
        storage_size: int,
        language: str,
        now: datetime,
    ) -> ProcessOutcome:
        """Settle the row after a successful index.

        Chunks stay ``processing`` only when a post-process dispatcher
        will still run; otherwise the row completes immediately.
        """
        awaiting_enrichment = text_chunk_count > 0 and self._post_process_dispatcher is not None
        parse_status = PARSE_STATUS_PROCESSING if awaiting_enrichment else PARSE_STATUS_COMPLETED
        updated = row.model_copy(
            update={
                "parse_status": parse_status,
                "enable_status": _ENABLE_STATUS_ENABLED,
                "summary_status": SUMMARY_STATUS_NONE,
                "storage_size": storage_size,
                "processed_at": now,
                "updated_at": now,
            }
        )
        with contextlib.suppress(Exception):
            await self._knowledge_repo.update(updated)

        if self._post_process_dispatcher is not None and text_chunk_count > 0:
            payload = PostProcessPayload(
                tenant_id=row.tenant_id,
                knowledge_id=row.id,
                knowledge_base_id=row.knowledge_base_id,
                language=language,
            )
            await self._ignore_errors(
                "post-process dispatch", self._post_process_dispatcher.dispatch(payload=payload)
            )
        if self._storage_resolver is not None:
            await self._ignore_errors(
                "storage accounting",
                self._storage_resolver.adjust_storage_used(
                    tenant_id=row.tenant_id, delta=storage_size
                ),
            )

        return ProcessOutcome(
            parse_status=parse_status,
            enable_status=_ENABLE_STATUS_ENABLED,
            summary_status=SUMMARY_STATUS_NONE,
            storage_size=storage_size,
            text_chunk_count=text_chunk_count,
        )

    async def _resolve_ai(
        self,
        *,
        kb: KnowledgeBaseInfo,
        tenant_id: int,
    ) -> tuple[Embedder | None, IndexEngine | None]:
        """Resolve the embedder and index engine when embedding is enabled.

        A KB that does not need an embedding model (neither vector nor
        keyword indexing) skips resolution entirely. Either seam returning
        ``None`` disables indexing for this run.
        """
        if not kb_needs_embedding(kb.indexing_strategy):
            return None, None
        embedder: Embedder | None = None
        if self._embedding_resolver is not None:
            embedder = await self._embedding_resolver.resolve_embedder(
                embedding_model_id=kb.embedding_model_id
            )
        engine: IndexEngine | None = None
        if self._index_engine_resolver is not None:
            engine = await self._index_engine_resolver.resolve_engine(
                tenant_id=tenant_id,
                vector_store_id=kb.vector_store_id,
            )
        return embedder, engine

    async def _quota_exceeded(self, tenant_id: int, storage_size: int) -> bool:
        """Return whether the tenant lacks storage room for ``storage_size``.

        Skipped when no storage accounting is wired or no quota is set.
        """
        if self._storage_resolver is None or storage_size <= 0:
            return False
        storage = await self._storage_resolver.get_storage(tenant_id=tenant_id)
        if storage.storage_quota <= 0:
            return False
        return storage.storage_used + storage_size > storage.storage_quota

    async def _cleanup_stale(
        self,
        *,
        ctx: Context,
        row: Document,
        engine: IndexEngine | None,
        embedder: Embedder | None,
        knowledge_type: str,
        now: datetime,
    ) -> None:
        """Best-effort removal of previously persisted chunks and index rows.

        Re-runs are idempotent: old chunks and their index entries are
        deleted before new ones are written.
        """
        await self._ignore_errors(
            "stale chunk cleanup",
            self._chunk_repo.delete_by_knowledge_id(
                tenant_id=row.tenant_id,
                knowledge_id=row.id,
                now=now,
            ),
        )
        if engine is not None and embedder is not None:
            await self._ignore_errors(
                "stale index cleanup",
                engine.delete_by_knowledge_id_list(
                    ctx,
                    [row.id],
                    embedder.get_dimensions(),
                    knowledge_type,
                ),
            )

    async def _remove_chunks_and_index(
        self,
        *,
        ctx: Context,
        row: Document,
        engine: IndexEngine | None,
        embedder: Embedder | None,
        knowledge_type: str,
        now: datetime,
    ) -> None:
        """Best-effort removal of the chunks and index just written.

        Used to roll back after an index failure and to drop rows written
        for a document that was being deleted mid-flight.
        """
        await self._ignore_errors(
            "chunk rollback",
            self._chunk_repo.delete_by_knowledge_id(
                tenant_id=row.tenant_id,
                knowledge_id=row.id,
                now=now,
            ),
        )
        if engine is not None and embedder is not None:
            await self._ignore_errors(
                "index rollback",
                engine.delete_by_knowledge_id_list(
                    ctx,
                    [row.id],
                    embedder.get_dimensions(),
                    knowledge_type,
                ),
            )

    @staticmethod
    async def _ignore_errors(label: str, operation: Awaitable[_IgnoreT]) -> None:
        """Await ``operation``, logging a warning instead of propagating."""
        try:
            await operation
        except Exception as exc:
            logger.warning("{}: {}", label, exc)

    @staticmethod
    def _skipped(status: str) -> ProcessOutcome:
        """A no-op outcome for an aborted or idempotent run."""
        return ProcessOutcome(parse_status=status, skipped=True)


# ── Worker entry point ─────────────────────────────────────────────────


async def process_document(
    *,
    tenant_id: int,
    knowledge_id: str,
    knowledge_base_id: str,
    file_path: str = "",
    file_name: str = "",
    file_type: str = "",
    url: str = "",
    enable_multimodel: bool = False,
    language: str = "",
    request_id: str = "",
    now: datetime | None = None,
    ctx: Context | None = None,
    pipeline: DocumentProcessPipeline | None = None,
) -> ProcessOutcome:
    """Worker-side entry point for one document-process run.

    Requires a composed :class:`DocumentProcessPipeline`. The worker
    wiring layer supplies knowledge repo, KB service, chunk repo,
    reader, file reader, embedding / index resolvers, post-process
    dispatcher, and storage resolver.

    ``ctx`` defaults to a background :class:`TaskContext` so background
    ingestion workers hit the provider governor's throttled path.
    """
    if pipeline is None:
        raise ValidationError(
            code="knowledge.pipeline_required",
            message="document process pipeline is not composed",
        )
    selected_ctx = ctx or TaskContext(is_background_task=True)
    return await pipeline.run(
        ctx=selected_ctx,
        tenant_id=tenant_id,
        knowledge_id=knowledge_id,
        knowledge_base_id=knowledge_base_id,
        file_path=file_path,
        file_name=file_name,
        file_type=file_type,
        url=url,
        enable_multimodel=enable_multimodel,
        language=language,
        request_id=request_id,
        now=now,
    )


__all__ = [
    "DocumentProcessPipeline",
    "EmbeddingResolver",
    "IndexEngineResolver",
    "PostProcessDispatcher",
    "PostProcessPayload",
    "ProcessOutcome",
    "TenantStorageInfo",
    "TenantStorageResolver",
    "build_chunk_rows",
    "kb_needs_embedding",
    "process_document",
]
