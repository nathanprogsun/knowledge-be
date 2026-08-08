"""Wiki ingest orchestrator.

The high-level entry points for the wiki ingest pipeline: enqueue an
ingest / retract operation onto the durable pending queue, and run one
batch (map -> taxonomy -> reduce -> settle). All storage and seam
dependencies arrive through an immutable :class:`WikiIngestDeps` bundle,
so the worker layer can wire the real document parser, embedding model,
retrieval composite, and synthesis seams without changing the pipeline.

This module also owns the default :class:`CompositeIndexWriter`, which
wraps the merged composite retrieval engine so a worker that has a
composite wired can index without any custom broker.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from src.ai.embedding import Context, Embedder, TaskContext
from src.ai.retrieval.composite import CompositeRetrieveEngine
from src.ai.retrieval.types import IndexInfo, SourceType
from src.core.knowledge.wiki.ingest_batch import process_wiki_ingest_batch
from src.core.knowledge.wiki.ingest_types import (
    WIKI_KNOWLEDGE_TYPE,
    WIKI_MAX_DOCS_PER_BATCH,
    WIKI_MAX_FAIL_RETRIES,
    WIKI_OP_INGEST,
    WIKI_OP_RETRACT,
    WikiBatchContext,
    WikiBatchOutcome,
    WikiIngestDeps,
    WikiIngestOp,
    chunk_embedding_text,
)
from src.db.models.chunk import Chunk

logger = logging.getLogger(__name__)


class CompositeIndexWriter:
    """Default index writer backed by the merged composite retrieval engine.

    Builds the ``IndexInfo`` payload for each chunk and fans the write out
    through :meth:`CompositeRetrieveEngine.batch_index`. The composite
    embeds internally using the supplied embedder; ``embeddings`` (already
    computed by the pipeline's embed stage) is carried on the protocol so a
    custom broker can reuse the vectors instead of re-embedding.
    """

    def __init__(
        self,
        *,
        composite: CompositeRetrieveEngine,
        knowledge_type: str = WIKI_KNOWLEDGE_TYPE,
    ) -> None:
        self._composite = composite
        self._knowledge_type = knowledge_type

    async def write_chunks(
        self,
        *,
        ctx: Context,
        tenant_id: int,
        knowledge_base_id: str,
        knowledge_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        embedder: Embedder,
    ) -> None:
        infos = [
            IndexInfo(
                id=chunk.id,
                content=chunk_embedding_text(chunk),
                source_id=chunk.id,
                source_type=SourceType.CHUNK,
                chunk_id=chunk.id,
                knowledge_id=knowledge_id,
                knowledge_base_id=knowledge_base_id,
                knowledge_type=self._knowledge_type,
                tag_id=chunk.tag_id or "",
                is_enabled=chunk.is_enabled,
            )
            for chunk in chunks
        ]
        await self._composite.batch_index(ctx, embedder, infos)

    async def delete_by_source_id_list(
        self,
        *,
        ctx: Context,
        tenant_id: int,
        knowledge_base_id: str,
        source_id_list: list[str],
        dimension: int,
    ) -> None:
        await self._composite.delete_by_source_id_list(
            ctx, source_id_list, dimension, self._knowledge_type
        )


class WikiIngestService:
    """Orchestrates the wiki ingest pipeline over a dependency bundle."""

    def __init__(self, *, deps: WikiIngestDeps) -> None:
        self._deps = deps

    @property
    def deps(self) -> WikiIngestDeps:
        """The immutable dependency bundle backing this service."""
        return self._deps

    async def enqueue_ingest(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        knowledge_id: str,
        language: str = "",
    ) -> bool:
        """Queue a document for wiki ingestion; return whether it was accepted."""
        op = WikiIngestOp(op=WIKI_OP_INGEST, knowledge_id=knowledge_id, language=language)
        return await self._deps.pending_store.enqueue(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            op=op,
        )

    async def enqueue_retract(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        knowledge_id: str,
        doc_title: str,
        doc_summary: str = "",
        page_slugs: Sequence[str] = (),
        folder_ids: Sequence[str] = (),
        language: str = "",
    ) -> bool:
        """Queue a wiki retraction for a deleted document; return acceptance."""
        op = WikiIngestOp(
            op=WIKI_OP_RETRACT,
            knowledge_id=knowledge_id,
            language=language,
            doc_title=doc_title,
            doc_summary=doc_summary,
            page_slugs=tuple(page_slugs),
            folder_ids=tuple(folder_ids),
        )
        return await self._deps.pending_store.enqueue(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            op=op,
        )

    async def process_batch(
        self,
        ctx: Context | None,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        language: str = "",
        batch_ctx: WikiBatchContext | None = None,
        max_docs: int = WIKI_MAX_DOCS_PER_BATCH,
        max_fail_retries: int = WIKI_MAX_FAIL_RETRIES,
    ) -> WikiBatchOutcome:
        """Run one ingest batch and settle the pending queue.

        ``ctx`` defaults to a background task context so the batch's
        embedding calls stay throttled by the per-model governor.
        """
        if ctx is None:
            ctx = TaskContext(is_background_task=True)
        return await process_wiki_ingest_batch(
            ctx,
            deps=self._deps,
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            language=language,
            batch_ctx=batch_ctx or WikiBatchContext(),
            max_docs=max_docs,
            max_fail_retries=max_fail_retries,
        )


__all__ = [
    "CompositeIndexWriter",
    "WikiIngestService",
]
