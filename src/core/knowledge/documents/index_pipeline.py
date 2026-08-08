"""Embed + index stage of the document-processing pipeline.

Builds retrieval index entries from persisted chunks and writes them
through an injectable retrieval-index seam. The composite retrieval
engine lands in the wiring layer; this module only fixes the seam
contract (``IndexEngine``) and the content builders the orchestrator
uses to shape each entry.

The document title is prepended to every entry's searchable content and
chunk context headers (heading breadcrumbs) carry section context,
mirroring the upstream index-content builder.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.ai.embedding import Context, Embedder
from src.ai.retrieval.types import IndexInfo, SourceType
from src.db.models.chunk import Chunk


@runtime_checkable
class IndexEngine(Protocol):
    """Retrieval-index seam (satisfied by the composite retrieve engine).

    Exposes only the write/estimate surface the pipeline needs; the
    retrieval query surface is used elsewhere and is out of scope here.
    """

    async def batch_index(
        self,
        ctx: Context,
        embedder: Embedder,
        index_info_list: list[IndexInfo],
    ) -> None:
        """Embed and index a batch of entries."""
        ...

    async def delete_by_knowledge_id_list(
        self,
        ctx: Context,
        knowledge_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        """Remove the index rows of one or more knowledge items."""
        ...

    def estimate_storage_size(
        self,
        ctx: Context,
        embedder: Embedder,
        index_info_list: list[IndexInfo],
    ) -> int:
        """Return the estimated storage bytes for the given entries."""
        ...


def build_knowledge_index_content(title: str, content: str) -> str:
    """Prepend the document title to searchable text.

    The title line sits outermost; custom metadata stays document-scoped
    (supplied once to the answer/summary models) and is not repeated per
    chunk, mirroring the upstream index-content rule.
    """
    title = title.strip()
    if title == "":
        return content
    return title + "\n" + content


def chunk_embedding_content(chunk: Chunk) -> str:
    """Chunk content with the context header prepended when set.

    Surrounding whitespace is trimmed so boundary slices do not dilute
    the embedded vector; the header keeps heading breadcrumbs separate
    from the literal content.
    """
    body = chunk.content.strip()
    if not chunk.context_header:
        return body
    return chunk.context_header + "\n\n" + body


def build_index_infos(
    *,
    chunks: list[Chunk],
    knowledge_id: str,
    knowledge_base_id: str,
    title: str,
    knowledge_type: str = "",
) -> list[IndexInfo]:
    """Build one index entry per chunk, in document order.

    Each entry references the chunk row as its source and chunk id and
    carries the title-prefixed embedding content.
    """
    return [
        IndexInfo(
            content=build_knowledge_index_content(title, chunk_embedding_content(chunk)),
            source_id=chunk.id,
            source_type=SourceType.CHUNK,
            chunk_id=chunk.id,
            knowledge_id=knowledge_id,
            knowledge_base_id=knowledge_base_id,
            knowledge_type=knowledge_type,
            is_enabled=True,
        )
        for chunk in chunks
    ]


__all__ = [
    "IndexEngine",
    "build_index_infos",
    "build_knowledge_index_content",
    "chunk_embedding_content",
]
