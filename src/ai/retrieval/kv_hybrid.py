"""Keywords-vector hybrid retrieve engine (upstream ``keywords_vector_hybrid_indexer.go``).

``KVHybridRetrieveEngine`` decorates a ``RetrieveEngineRepository`` with the
embedding step: ``index`` / ``batch_index`` embed the content through the
injected ``Embedder`` before delegating storage to the repository, and every
delegated method (retrieve, deletes, batch updates, copy) passes through
unmodified. ``sanitize_for_embedding`` scrubs inline base64 payloads and caps
input length so pathological content cannot blow up the embedding call.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping

from src.ai.embedding import Context, Embedder
from src.ai.retrieval.base import RetrieveEngineRepository
from src.ai.retrieval.types import (
    IndexInfo,
    IndexSaveParams,
    RetrieveParams,
    RetrieverEngineType,
    RetrieveResult,
    RetrieverType,
)
from src.app_logging import logger

#: Absolute upper bound for any single embedding input; beyond this we
#: truncate (with a warning) instead of blindly forwarding to the embedding
#: API. Set well above any current chunk-size budget.
_SAFETY_MAX_CHARS: int = 20000

#: Exponential backoff applied to ``BatchEmbedWithPool`` calls.
_EMBED_RETRY_ATTEMPTS: int = 5
_EMBED_RETRY_BASE_DELAY_SECONDS: float = 0.2

#: Batch size for vector-backed index saves and the concurrency cap shared
#: by both the vector and keyword-only paths (upstream constants).
_BATCH_SIZE: int = 40
_KEYWORD_BATCH_SIZE: int = 10
_MAX_CONCURRENCY: int = 5

_EMBEDDING_IMAGE_PAYLOAD_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?is)<img\b[^>]*\bsrc=[\"']\s*data:image/[a-z0-9.+-]+;base64,[^\"']+[\"'][^>]*>"),
    re.compile(r"(?is)!\[[^\]]*\]\(\s*data:image/[a-z0-9.+-]+;base64,[^)]+\)"),
    re.compile(r"(?i)data:image/[a-z0-9.+-]+;base64,[a-z0-9+/=]{200,}"),
    re.compile(r"(?i)data:[a-z0-9.+/-]+;base64,[a-z0-9+/=]{200,}"),
)


def sanitize_for_embedding(ctx: Context, content: str) -> str:
    """Cap content length and scrub inline base64 image payloads.

    The truncation point is code-point based, not token based, so it sits
    well above any realistic token limit. ``ctx`` is accepted for upstream
    signature parity (the upstream helper logs through it).
    """
    del ctx
    sanitized = content
    # Scrubbing only matters when an inline base64 payload is present; skip
    # the regex passes otherwise so the common (no-image) path stays cheap.
    if "base64," in content:
        for pattern in _EMBEDDING_IMAGE_PAYLOAD_PATTERNS:
            sanitized = pattern.sub("[image]", sanitized)
    if len(sanitized) <= _SAFETY_MAX_CHARS:
        return sanitized
    logger.warning(
        "embedding input truncated: {} chars -> {}",
        len(sanitized),
        _SAFETY_MAX_CHARS,
    )
    return sanitized[:_SAFETY_MAX_CHARS]


def _chunk_slice(items: list[IndexInfo], chunk_size: int) -> list[list[IndexInfo]]:
    """Split ``items`` into contiguous sub-lists of at most ``chunk_size``."""
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def _embedding_map(
    chunk: list[IndexInfo],
    embeddings: list[list[float]],
    batch_index: int,
    batch_size: int,
) -> dict[str, list[float]]:
    """Map each chunk's source id to its embedding at the batch offset."""
    return {
        item.source_id: embeddings[batch_index * batch_size + index]
        for index, item in enumerate(chunk)
    }


async def _batch_embed_with_backoff(
    ctx: Context,
    embedder: Embedder,
    content_list: list[str],
) -> list[list[float]]:
    """Call ``BatchEmbedWithPool`` with exponential backoff on failure.

    Returns the last embedding result on success, or raises the last error
    when every attempt failed.
    """
    delay = _EMBED_RETRY_BASE_DELAY_SECONDS
    last_error: Exception | None = None
    for attempt in range(_EMBED_RETRY_ATTEMPTS):
        try:
            return await embedder.batch_embed_with_pool(ctx, embedder, content_list)
        except Exception as exc:
            last_error = exc
            logger.error(
                "BatchEmbedWithPool attempt {}/{} failed: {}",
                attempt + 1,
                _EMBED_RETRY_ATTEMPTS,
                exc,
            )
            if attempt + 1 < _EMBED_RETRY_ATTEMPTS:
                await asyncio.sleep(delay)
                delay *= 2
    assert last_error is not None
    raise last_error


class KVHybridRetrieveEngine:
    """Hybrid retrieve engine supporting keyword and vector retrieval.

    Wraps a ``RetrieveEngineRepository``; ``index`` / ``batch_index`` add the
    embedding step and every other method delegates straight through.
    """

    def __init__(
        self,
        index_repository: RetrieveEngineRepository,
        engine_type: RetrieverEngineType,
    ) -> None:
        self._index_repository = index_repository
        self._engine_type = engine_type

    def engine_type(self) -> RetrieverEngineType:
        return self._engine_type

    async def retrieve(self, ctx: Context, params: RetrieveParams) -> list[RetrieveResult]:
        return await self._index_repository.retrieve(ctx, params)

    async def index(
        self,
        ctx: Context,
        embedder: Embedder,
        index_info: IndexInfo,
        retriever_types: list[RetrieverType],
    ) -> None:
        params: IndexSaveParams = {}
        embedding_map: dict[str, list[float]] = {}
        if RetrieverType.VECTOR in retriever_types:
            embedding = await embedder.embed(ctx, sanitize_for_embedding(ctx, index_info.content))
            embedding_map[index_info.source_id] = embedding
        params["embedding"] = embedding_map
        await self._index_repository.save(ctx, index_info, params)

    async def batch_index(
        self,
        ctx: Context,
        embedder: Embedder,
        index_info_list: list[IndexInfo],
        retriever_types: list[RetrieverType],
    ) -> None:
        if not index_info_list:
            return
        if RetrieverType.VECTOR in retriever_types:
            content_list = [sanitize_for_embedding(ctx, item.content) for item in index_info_list]
            embeddings = await _batch_embed_with_backoff(ctx, embedder, content_list)
            chunks = _chunk_slice(index_info_list, _BATCH_SIZE)
            if len(chunks) <= _MAX_CONCURRENCY:
                await self._concurrent_batch_save(ctx, chunks, embeddings, _BATCH_SIZE)
            else:
                await self._bounded_concurrent_batch_save(
                    ctx, chunks, embeddings, _BATCH_SIZE, _MAX_CONCURRENCY
                )
            return
        chunks = _chunk_slice(index_info_list, _KEYWORD_BATCH_SIZE)
        if len(chunks) <= _MAX_CONCURRENCY:
            await self._concurrent_batch_save_no_embedding(ctx, chunks)
        else:
            await self._bounded_concurrent_batch_save_no_embedding(ctx, chunks, _MAX_CONCURRENCY)

    def estimate_storage_size(
        self,
        ctx: Context,
        embedder: Embedder,
        index_info_list: list[IndexInfo],
        retriever_types: list[RetrieverType],
    ) -> int:
        params: IndexSaveParams = {}
        if RetrieverType.VECTOR in retriever_types:
            # Placeholder vectors sized by the embedder dimensions; only the
            # storage footprint matters here.
            params["embedding"] = {
                item.chunk_id: [0.0] * embedder.get_dimensions() for item in index_info_list
            }
        return self._index_repository.estimate_storage_size(ctx, index_info_list, params)

    def support(self) -> list[RetrieverType]:
        return self._index_repository.support()

    async def copy_indices(
        self,
        ctx: Context,
        source_knowledge_base_id: str,
        source_to_target_kb_id_map: Mapping[str, str],
        source_to_target_chunk_id_map: Mapping[str, str],
        target_knowledge_base_id: str,
        dimension: int,
        knowledge_type: str,
    ) -> None:
        await self._index_repository.copy_indices(
            ctx,
            source_knowledge_base_id,
            source_to_target_kb_id_map,
            source_to_target_chunk_id_map,
            target_knowledge_base_id,
            dimension,
            knowledge_type,
        )

    async def delete_by_chunk_id_list(
        self,
        ctx: Context,
        index_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        await self._index_repository.delete_by_chunk_id_list(
            ctx, index_id_list, dimension, knowledge_type
        )

    async def delete_by_source_id_list(
        self,
        ctx: Context,
        source_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        await self._index_repository.delete_by_source_id_list(
            ctx, source_id_list, dimension, knowledge_type
        )

    async def delete_by_knowledge_id_list(
        self,
        ctx: Context,
        knowledge_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        await self._index_repository.delete_by_knowledge_id_list(
            ctx, knowledge_id_list, dimension, knowledge_type
        )

    async def batch_update_chunk_enabled_status(
        self, ctx: Context, chunk_status_map: Mapping[str, bool]
    ) -> None:
        await self._index_repository.batch_update_chunk_enabled_status(ctx, chunk_status_map)

    async def batch_update_chunk_tag_id(
        self, ctx: Context, chunk_tag_map: Mapping[str, str]
    ) -> None:
        await self._index_repository.batch_update_chunk_tag_id(ctx, chunk_tag_map)

    # ── batch-save helpers ───────────────────────────────────────────

    async def _concurrent_batch_save(
        self,
        ctx: Context,
        chunks: list[list[IndexInfo]],
        embeddings: list[list[float]],
        batch_size: int,
    ) -> None:
        results = await asyncio.gather(
            *[
                self._index_repository.batch_save(
                    ctx,
                    chunk,
                    {"embedding": _embedding_map(chunk, embeddings, index, batch_size)},
                )
                for index, chunk in enumerate(chunks)
            ],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def _bounded_concurrent_batch_save(
        self,
        ctx: Context,
        chunks: list[list[IndexInfo]],
        embeddings: list[list[float]],
        batch_size: int,
        max_concurrency: int,
    ) -> None:
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _save(index: int, chunk: list[IndexInfo]) -> None:
            async with semaphore:
                await self._index_repository.batch_save(
                    ctx,
                    chunk,
                    {"embedding": _embedding_map(chunk, embeddings, index, batch_size)},
                )

        results = await asyncio.gather(
            *[_save(index, chunk) for index, chunk in enumerate(chunks)],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def _concurrent_batch_save_no_embedding(
        self,
        ctx: Context,
        chunks: list[list[IndexInfo]],
    ) -> None:
        results = await asyncio.gather(
            *[self._index_repository.batch_save(ctx, chunk, IndexSaveParams()) for chunk in chunks],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def _bounded_concurrent_batch_save_no_embedding(
        self,
        ctx: Context,
        chunks: list[list[IndexInfo]],
        max_concurrency: int,
    ) -> None:
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _save(chunk: list[IndexInfo]) -> None:
            async with semaphore:
                await self._index_repository.batch_save(ctx, chunk, IndexSaveParams())

        results = await asyncio.gather(
            *[_save(chunk) for chunk in chunks],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result


def new_kv_hybrid_retrieve_engine(
    index_repository: RetrieveEngineRepository,
    engine_type: RetrieverEngineType,
) -> KVHybridRetrieveEngine:
    """Create a keywords-vector hybrid retrieve engine (upstream ``NewKVHybridRetrieveEngine``)."""
    return KVHybridRetrieveEngine(index_repository, engine_type)


__all__ = [
    "KVHybridRetrieveEngine",
    "new_kv_hybrid_retrieve_engine",
    "sanitize_for_embedding",
]
