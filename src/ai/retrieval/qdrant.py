"""Qdrant retrieval engine repository.

Implements ``RetrieveEngineRepository`` against the Qdrant vector database.
Collection management creates dimension-suffixed collections with payload
indexes (keyword, bool, text) for efficient filtering. Vector retrieval uses
HNSW search; keyword retrieval uses payload text matching across all
dimension-suffixed collections.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import jieba
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchText,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    TextIndexParams,
    TextIndexType,
    TokenizerType,
    VectorParams,
)

from src.ai.embedding import Context
from src.ai.retrieval.types import (
    IndexConfig,
    IndexInfo,
    IndexSaveParams,
    IndexWithScore,
    MatchType,
    RetrieveParams,
    RetrieverEngineType,
    RetrieveResult,
    RetrieverType,
    SourceType,
)
from src.app_logging import logger
from src.common.exception import ValidationError

# ── Payload field names ───────────────────────────────────────────────

FIELD_CONTENT = "content"
FIELD_SOURCE_ID = "source_id"
FIELD_SOURCE_TYPE = "source_type"
FIELD_CHUNK_ID = "chunk_id"
FIELD_KNOWLEDGE_ID = "knowledge_id"
FIELD_KNOWLEDGE_BASE_ID = "knowledge_base_id"
FIELD_TAG_ID = "tag_id"
FIELD_IS_ENABLED = "is_enabled"

#: Key in ``IndexSaveParams`` that carries the embedding map.
EMBEDDING_KEY = "embedding"

# ── Collection resolution ─────────────────────────────────────────────

ENV_QDRANT_COLLECTION = "QDRANT_COLLECTION"
DEFAULT_COLLECTION_NAME = "kb_embeddings"

# ── Batch sizes ──────────────────────────────────────────────────────

_BATCH_SAVE_SIZE = 100
_COPY_BATCH_SIZE = 64

# ── Storage estimate constants ────────────────────────────────────────

_HNSW_M = 16
_HNSW_LINK_BYTES = 8
_ID_TRACKER_BYTES = 24
_SOURCE_TYPE_BYTES = 8


# ── Helpers ──────────────────────────────────────────────────────────


def _resolve_collection_name(index_config: IndexConfig | None) -> str:
    """Resolve the collection base name (upstream ``ResolveCollectionName``).

    Priority: ``collection_prefix`` > ``collection_name`` > env var
    ``QDRANT_COLLECTION`` > default ``kb_embeddings``.
    """
    if index_config is not None:
        if index_config.collection_prefix:
            return index_config.collection_prefix
        if index_config.collection_name:
            return index_config.collection_name
    env_val = os.getenv(ENV_QDRANT_COLLECTION, "")
    if env_val:
        return env_val
    return DEFAULT_COLLECTION_NAME


def _collection_name(base: str, dimension: int) -> str:
    """Return the dimension-suffixed collection name."""
    return f"{base}_{dimension}"


def _tokenize_query(query: str) -> list[str]:
    """Tokenize a query string for OR-based full-text search.

    Uses jieba search mode for Chinese word segmentation, then filters
    to unique tokens with at least 2 characters (matching the upstream
    ``tokenizeQuery`` logic).
    """
    query = query.strip()
    if not query:
        return []
    words = jieba.cut_for_search(query, HMM=True)
    seen: set[str] = set()
    result: list[str] = []
    for word in words:
        word = word.strip().lower()
        if len(word) < 2 or word in seen:
            continue
        seen.add(word)
        result.append(word)
    return result


@dataclass(frozen=True, slots=True)
class _EmbeddingData:
    """Payload + embedding extracted from an ``IndexInfo``."""

    content: str
    source_id: str
    source_type: int
    chunk_id: str
    knowledge_id: str
    knowledge_base_id: str
    tag_id: str
    is_enabled: bool
    embedding: list[float]


def _to_embedding_data(info: IndexInfo, params: IndexSaveParams) -> _EmbeddingData:
    """Convert ``IndexInfo`` + ``IndexSaveParams`` to ``_EmbeddingData``."""
    embedding: list[float] = []
    embedding_map = params.get(EMBEDDING_KEY, {})
    if isinstance(embedding_map, dict):
        raw = embedding_map.get(info.source_id, [])
        embedding = list(raw) if raw else []
    return _EmbeddingData(
        content=info.content,
        source_id=info.source_id,
        source_type=info.source_type,
        chunk_id=info.chunk_id,
        knowledge_id=info.knowledge_id,
        knowledge_base_id=info.knowledge_base_id,
        tag_id=info.tag_id,
        is_enabled=info.is_enabled,
        embedding=embedding,
    )


def _build_payload(data: _EmbeddingData) -> dict[str, object]:
    """Build the Qdrant payload dict from embedding data."""
    return {
        FIELD_CONTENT: data.content,
        FIELD_SOURCE_ID: data.source_id,
        FIELD_SOURCE_TYPE: data.source_type,
        FIELD_CHUNK_ID: data.chunk_id,
        FIELD_KNOWLEDGE_ID: data.knowledge_id,
        FIELD_KNOWLEDGE_BASE_ID: data.knowledge_base_id,
        FIELD_TAG_ID: data.tag_id,
        FIELD_IS_ENABLED: data.is_enabled,
    }


def _payload_to_index_with_score(
    point_id: str | int,
    payload: dict[str, object],
    score: float,
    match_type: MatchType,
) -> IndexWithScore:
    """Convert a Qdrant point payload to ``IndexWithScore``."""
    source_type_raw = payload.get(FIELD_SOURCE_TYPE, 0)
    source_type = (
        SourceType(source_type_raw) if isinstance(source_type_raw, int) else SourceType.CHUNK
    )
    content = str(payload.get(FIELD_CONTENT, ""))
    source_id = str(payload.get(FIELD_SOURCE_ID, ""))
    chunk_id = str(payload.get(FIELD_CHUNK_ID, ""))
    knowledge_id = str(payload.get(FIELD_KNOWLEDGE_ID, ""))
    knowledge_base_id = str(payload.get(FIELD_KNOWLEDGE_BASE_ID, ""))
    tag_id = str(payload.get(FIELD_TAG_ID, ""))
    return IndexWithScore(
        id=str(point_id),
        content=content,
        source_id=source_id,
        source_type=source_type,
        chunk_id=chunk_id,
        knowledge_id=knowledge_id,
        knowledge_base_id=knowledge_base_id,
        tag_id=tag_id,
        score=score,
        match_type=match_type,
        is_enabled=bool(payload.get(FIELD_IS_ENABLED, False)),
    )


def _build_retrieve_result(
    results: list[IndexWithScore], retriever_type: RetrieverType
) -> list[RetrieveResult]:
    """Wrap results in a single-element ``RetrieveResult`` list."""
    return [
        RetrieveResult(
            results=results,
            retriever_engine_type=RetrieverEngineType.QDRANT,
            retriever_type=retriever_type,
        )
    ]


def _calculate_storage_size(data: _EmbeddingData) -> int:
    """Estimate per-point storage (upstream ``calculateStorageSize``)."""
    payload_size = (
        len(data.content)
        + len(data.source_id)
        + len(data.chunk_id)
        + len(data.knowledge_id)
        + len(data.knowledge_base_id)
        + _SOURCE_TYPE_BYTES
    )
    vector_size = 0
    hnsw_index = 0
    if data.embedding:
        dimensions = len(data.embedding)
        vector_size = dimensions * 4
        hnsw_index = _HNSW_M * 2 * _HNSW_LINK_BYTES
    return payload_size + vector_size + hnsw_index + _ID_TRACKER_BYTES


def _filter(
    must: list[FieldCondition] | None = None,
    must_not: list[FieldCondition] | None = None,
) -> Filter:
    """Build a ``Filter`` from homogeneous ``FieldCondition`` lists."""
    return Filter(must=must, must_not=must_not)  # type: ignore[arg-type]


# ── Repository ───────────────────────────────────────────────────────


class QdrantRetrieveEngineRepository:
    """Qdrant-backed retrieve engine repository.

    Manages dimension-suffixed collections with lazy creation and payload
    indexes. Supports keyword and vector retrieval.
    """

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_base_name: str,
        shard_number: int,
        replication_factor: int,
    ) -> None:
        self._client = client
        self._collection_base_name = collection_base_name
        self._shard_number = shard_number
        self._replication_factor = replication_factor
        self._lock = asyncio.Lock()
        self._initialized: set[int] = set()

    def engine_type(self) -> RetrieverEngineType:
        return RetrieverEngineType.QDRANT

    def support(self) -> list[RetrieverType]:
        return [RetrieverType.KEYWORDS, RetrieverType.VECTOR]

    def _get_collection_name(self, dimension: int) -> str:
        return _collection_name(self._collection_base_name, dimension)

    async def _ensure_collection(self, ctx: Context, dimension: int) -> None:
        """Create the collection for ``dimension`` if it does not exist."""
        del ctx
        if dimension in self._initialized:
            return
        async with self._lock:
            if dimension in self._initialized:
                return
            collection_name = self._get_collection_name(dimension)
            exists = await self._client.collection_exists(collection_name)
            if not exists:
                await self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=dimension,
                        distance=Distance.COSINE,
                    ),
                    shard_number=self._shard_number or None,
                    replication_factor=self._replication_factor or None,
                )
                await self._create_payload_indexes(collection_name)
                logger.info(
                    "Qdrant collection {} created (dim={})",
                    collection_name,
                    dimension,
                )
            self._initialized.add(dimension)

    async def _create_payload_indexes(self, collection_name: str) -> None:
        """Create keyword, bool, and text payload indexes (best-effort)."""
        keyword_fields = [
            FIELD_CHUNK_ID,
            FIELD_KNOWLEDGE_ID,
            FIELD_KNOWLEDGE_BASE_ID,
            FIELD_SOURCE_ID,
        ]
        for field in keyword_fields:
            try:
                await self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:
                logger.warning("Qdrant index creation failed for {}: {}", field, exc)
        try:
            await self._client.create_payload_index(
                collection_name=collection_name,
                field_name=FIELD_IS_ENABLED,
                field_schema=PayloadSchemaType.BOOL,
            )
        except Exception as exc:
            logger.warning("Qdrant index creation failed for {}: {}", FIELD_IS_ENABLED, exc)
        try:
            await self._client.create_payload_index(
                collection_name=collection_name,
                field_name=FIELD_CONTENT,
                field_schema=TextIndexParams(
                    type=TextIndexType.TEXT,
                    tokenizer=TokenizerType.MULTILINGUAL,
                    lowercase=True,
                ),
            )
        except Exception as exc:
            logger.warning("Qdrant text index creation failed for {}: {}", FIELD_CONTENT, exc)

    def _get_base_filter(self, params: RetrieveParams) -> Filter:
        """Build the common filter (is_enabled + KB/KW/tag/exclude)."""
        must: list[FieldCondition] = [
            FieldCondition(
                key=FIELD_IS_ENABLED,
                match=MatchValue(value=True),
            )
        ]
        if params.knowledge_base_ids:
            must.append(
                FieldCondition(
                    key=FIELD_KNOWLEDGE_BASE_ID,
                    match=MatchAny(any=params.knowledge_base_ids),
                )
            )
        if params.knowledge_ids:
            must.append(
                FieldCondition(
                    key=FIELD_KNOWLEDGE_ID,
                    match=MatchAny(any=params.knowledge_ids),
                )
            )
        if params.tag_ids:
            must.append(
                FieldCondition(
                    key=FIELD_TAG_ID,
                    match=MatchAny(any=params.tag_ids),
                )
            )
        must_not: list[FieldCondition] = []
        if params.exclude_knowledge_ids:
            must_not.append(
                FieldCondition(
                    key=FIELD_KNOWLEDGE_ID,
                    match=MatchAny(any=params.exclude_knowledge_ids),
                )
            )
        if params.exclude_chunk_ids:
            must_not.append(
                FieldCondition(
                    key=FIELD_CHUNK_ID,
                    match=MatchAny(any=params.exclude_chunk_ids),
                )
            )
        return _filter(must, must_not)

    def _matches_base_name(self, collection_name: str) -> bool:
        """Check if ``collection_name`` belongs to this repository."""
        return collection_name.startswith(self._collection_base_name)

    # ── Save ──────────────────────────────────────────────────────────

    async def save(self, ctx: Context, index_info: IndexInfo, params: IndexSaveParams) -> None:
        """Store a single point in Qdrant."""
        data = _to_embedding_data(index_info, params)
        if not data.embedding:
            raise ValidationError(
                code="qdrant.empty_embedding",
                message=f"empty embedding vector for chunk ID: {index_info.chunk_id}",
            )
        dimension = len(data.embedding)
        await self._ensure_collection(ctx, dimension)
        collection_name = self._get_collection_name(dimension)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=data.embedding,
            payload=_build_payload(data),
        )
        await self._client.upsert(
            collection_name=collection_name,
            points=[point],
        )

    async def batch_save(
        self,
        ctx: Context,
        index_info_list: list[IndexInfo],
        params: IndexSaveParams,
    ) -> None:
        """Store multiple points grouped by dimension (batches of 100)."""
        if not index_info_list:
            return
        points_by_dim: dict[int, list[PointStruct]] = {}
        for info in index_info_list:
            data = _to_embedding_data(info, params)
            if not data.embedding:
                logger.warning("Qdrant: skipping empty embedding for chunk {}", info.chunk_id)
                continue
            dim = len(data.embedding)
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=data.embedding,
                payload=_build_payload(data),
            )
            points_by_dim.setdefault(dim, []).append(point)
        if not points_by_dim:
            return
        for dim, points in points_by_dim.items():
            await self._ensure_collection(ctx, dim)
            collection_name = self._get_collection_name(dim)
            for i in range(0, len(points), _BATCH_SAVE_SIZE):
                batch = points[i : i + _BATCH_SAVE_SIZE]
                await self._client.upsert(
                    collection_name=collection_name,
                    points=batch,
                )

    # ── Retrieve ─────────────────────────────────────────────────────

    async def retrieve(self, ctx: Context, params: RetrieveParams) -> list[RetrieveResult]:
        """Dispatch retrieval based on ``params.retriever_type``."""
        if params.retriever_type == RetrieverType.VECTOR:
            return await self._vector_retrieve(ctx, params)
        if params.retriever_type == RetrieverType.KEYWORDS:
            return await self._keywords_retrieve(ctx, params)
        raise ValidationError(
            code="qdrant.invalid_retriever_type",
            message=f"invalid retriever type: {params.retriever_type}",
        )

    async def _vector_retrieve(self, ctx: Context, params: RetrieveParams) -> list[RetrieveResult]:
        """Perform HNSW vector similarity search."""
        del ctx
        dimension = len(params.embedding)
        collection_name = self._get_collection_name(dimension)
        exists = await self._client.collection_exists(collection_name)
        if not exists:
            return _build_retrieve_result([], RetrieverType.VECTOR)
        query_filter = self._get_base_filter(params)
        score_threshold = params.threshold if params.threshold > 0.0 else None
        response = await self._client.query_points(
            collection_name=collection_name,
            query=params.embedding,
            query_filter=query_filter,
            limit=params.top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        results: list[IndexWithScore] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                _payload_to_index_with_score(
                    str(point.id), payload, point.score, MatchType.EMBEDDING
                )
            )
        return _build_retrieve_result(results, RetrieverType.VECTOR)

    async def _keywords_retrieve(
        self, ctx: Context, params: RetrieveParams
    ) -> list[RetrieveResult]:
        """Perform keyword text search across all matching collections."""
        del ctx
        collections_response = await self._client.get_collections()
        all_names = [c.name for c in collections_response.collections]
        matching = [n for n in all_names if self._matches_base_name(n)]
        query_filter = self._get_base_filter(params)
        tokens = _tokenize_query(params.query)
        limit = params.top_k
        all_results: list[IndexWithScore] = []
        for collection_name in matching:
            if tokens:
                should = [
                    FieldCondition(key=FIELD_CONTENT, match=MatchText(text=token))
                    for token in tokens
                ]
                scroll_filter = _build_keyword_filter(query_filter, should_conditions=should)
            else:
                must_conditions = list(query_filter.must or [])
                must_conditions.append(
                    FieldCondition(key=FIELD_CONTENT, match=MatchText(text=params.query))
                )
                scroll_filter = _build_keyword_filter_fallback(
                    query_filter, must_conditions=cast("list[FieldCondition]", must_conditions)
                )
            records, _next = await self._client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                with_payload=True,
            )
            for record in records:
                payload = record.payload or {}
                all_results.append(
                    _payload_to_index_with_score(str(record.id), payload, 1.0, MatchType.KEYWORDS)
                )
        if limit > 0 and len(all_results) > limit:
            all_results = all_results[:limit]
        return _build_retrieve_result(all_results, RetrieverType.KEYWORDS)

    # ── Delete ────────────────────────────────────────────────────────

    async def delete_by_chunk_id_list(
        self,
        ctx: Context,
        index_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        """Delete points by chunk ID filter."""
        del ctx, knowledge_type
        if not index_id_list:
            return
        collection_name = self._get_collection_name(dimension)
        await self._client.delete(
            collection_name=collection_name,
            points_selector=_filter(
                must=[
                    FieldCondition(
                        key=FIELD_CHUNK_ID,
                        match=MatchAny(any=index_id_list),
                    )
                ]
            ),
        )

    async def delete_by_source_id_list(
        self,
        ctx: Context,
        source_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        """Delete points by source ID filter."""
        del ctx, knowledge_type
        if not source_id_list:
            return
        collection_name = self._get_collection_name(dimension)
        await self._client.delete(
            collection_name=collection_name,
            points_selector=_filter(
                must=[
                    FieldCondition(
                        key=FIELD_SOURCE_ID,
                        match=MatchAny(any=source_id_list),
                    )
                ]
            ),
        )

    async def delete_by_knowledge_id_list(
        self,
        ctx: Context,
        knowledge_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        """Delete points by knowledge ID filter."""
        del ctx, knowledge_type
        if not knowledge_id_list:
            return
        collection_name = self._get_collection_name(dimension)
        await self._client.delete(
            collection_name=collection_name,
            points_selector=_filter(
                must=[
                    FieldCondition(
                        key=FIELD_KNOWLEDGE_ID,
                        match=MatchAny(any=knowledge_id_list),
                    )
                ]
            ),
        )

    # ── Batch update ──────────────────────────────────────────────────

    async def batch_update_chunk_enabled_status(
        self, ctx: Context, chunk_status_map: Mapping[str, bool]
    ) -> None:
        """Update ``is_enabled`` payload across all matching collections."""
        del ctx
        if not chunk_status_map:
            return
        collections_response = await self._client.get_collections()
        all_names = [c.name for c in collections_response.collections]
        matching = [n for n in all_names if self._matches_base_name(n)]
        enabled_ids = [cid for cid, en in chunk_status_map.items() if en]
        disabled_ids = [cid for cid, en in chunk_status_map.items() if not en]
        for collection_name in matching:
            if enabled_ids:
                await self._client.set_payload(
                    collection_name=collection_name,
                    payload={FIELD_IS_ENABLED: True},
                    points=_filter(
                        must=[
                            FieldCondition(
                                key=FIELD_CHUNK_ID,
                                match=MatchAny(any=enabled_ids),
                            )
                        ]
                    ),
                )
            if disabled_ids:
                await self._client.set_payload(
                    collection_name=collection_name,
                    payload={FIELD_IS_ENABLED: False},
                    points=_filter(
                        must=[
                            FieldCondition(
                                key=FIELD_CHUNK_ID,
                                match=MatchAny(any=disabled_ids),
                            )
                        ]
                    ),
                )

    async def batch_update_chunk_tag_id(
        self, ctx: Context, chunk_tag_map: Mapping[str, str]
    ) -> None:
        """Update ``tag_id`` payload across all matching collections."""
        del ctx
        if not chunk_tag_map:
            return
        collections_response = await self._client.get_collections()
        all_names = [c.name for c in collections_response.collections]
        matching = [n for n in all_names if self._matches_base_name(n)]
        tag_groups: dict[str, list[str]] = {}
        for chunk_id, tag_id in chunk_tag_map.items():
            tag_groups.setdefault(tag_id, []).append(chunk_id)
        for collection_name in matching:
            for tag_id, chunk_ids in tag_groups.items():
                await self._client.set_payload(
                    collection_name=collection_name,
                    payload={FIELD_TAG_ID: tag_id},
                    points=_filter(
                        must=[
                            FieldCondition(
                                key=FIELD_CHUNK_ID,
                                match=MatchAny(any=chunk_ids),
                            )
                        ]
                    ),
                )

    # ── Copy indices ──────────────────────────────────────────────────

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
        """Copy index data from a source KB to a target KB."""
        del knowledge_type
        if not source_to_target_chunk_id_map:
            return
        collection_name = self._get_collection_name(dimension)
        await self._ensure_collection(ctx, dimension)
        offset: int | str | None = None
        total_copied = 0
        while True:
            records, next_offset = await self._client.scroll(
                collection_name=collection_name,
                scroll_filter=_filter(
                    must=[
                        FieldCondition(
                            key=FIELD_KNOWLEDGE_BASE_ID,
                            match=MatchValue(value=source_knowledge_base_id),
                        )
                    ]
                ),
                limit=_COPY_BATCH_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if not records:
                break
            target_points: list[PointStruct] = []
            for record in records:
                payload = record.payload or {}
                source_chunk_id = str(payload.get(FIELD_CHUNK_ID, ""))
                source_knowledge_id = str(payload.get(FIELD_KNOWLEDGE_ID, ""))
                original_source_id = str(payload.get(FIELD_SOURCE_ID, ""))
                target_chunk_id = source_to_target_chunk_id_map.get(source_chunk_id)
                if target_chunk_id is None:
                    continue
                target_knowledge_id = source_to_target_kb_id_map.get(source_knowledge_id)
                if target_knowledge_id is None:
                    continue
                target_source_id = self._resolve_target_source_id(
                    original_source_id, source_chunk_id, target_chunk_id
                )
                is_enabled = bool(payload.get(FIELD_IS_ENABLED, True))
                new_payload: dict[str, object] = {
                    FIELD_CONTENT: str(payload.get(FIELD_CONTENT, "")),
                    FIELD_SOURCE_ID: target_source_id,
                    FIELD_SOURCE_TYPE: payload.get(FIELD_SOURCE_TYPE, 0),
                    FIELD_CHUNK_ID: target_chunk_id,
                    FIELD_KNOWLEDGE_ID: target_knowledge_id,
                    FIELD_KNOWLEDGE_BASE_ID: target_knowledge_base_id,
                    FIELD_TAG_ID: str(payload.get(FIELD_TAG_ID, "")),
                    FIELD_IS_ENABLED: is_enabled,
                }
                vectors = record.vector
                if not vectors:
                    continue
                target_points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=cast("list[float]", vectors),
                        payload=new_payload,
                    )
                )
            if target_points:
                await self._client.upsert(
                    collection_name=collection_name,
                    points=target_points,
                )
                total_copied += len(target_points)
            if next_offset is None:
                break
            offset = cast("int | str", next_offset)
            if len(records) < _COPY_BATCH_SIZE:
                break
        logger.info(
            "Qdrant: copied {} points from {} to {}",
            total_copied,
            source_knowledge_base_id,
            target_knowledge_base_id,
        )

    @staticmethod
    def _resolve_target_source_id(
        original_source_id: str, source_chunk_id: str, target_chunk_id: str
    ) -> str:
        """Resolve the target source ID, preserving question suffixes."""
        if original_source_id == source_chunk_id:
            return target_chunk_id
        prefix = f"{source_chunk_id}-"
        if original_source_id.startswith(prefix):
            question_id = original_source_id[len(prefix) :]
            return f"{target_chunk_id}-{question_id}"
        return str(uuid.uuid4())

    # ── Estimate storage size ─────────────────────────────────────────

    def estimate_storage_size(
        self,
        ctx: Context,
        index_info_list: list[IndexInfo],
        params: IndexSaveParams,
    ) -> int:
        """Estimate total storage size for the index info list."""
        del ctx
        total = 0
        for info in index_info_list:
            data = _to_embedding_data(info, params)
            total += _calculate_storage_size(data)
        return total


# ── Keyword filter helpers ──────────────────────────────────────────


def _build_keyword_filter(
    base: Filter,
    should_conditions: list[FieldCondition],
) -> Filter:
    """Build a keyword filter that OR-matches tokens via ``should``."""
    return Filter(must=base.must, must_not=base.must_not, should=should_conditions)  # type: ignore[arg-type]


def _build_keyword_filter_fallback(
    base: Filter,
    must_conditions: list[FieldCondition],
) -> Filter:
    """Build a keyword filter that AND-matches the full query via ``must``."""
    return Filter(must=must_conditions, must_not=base.must_not)  # type: ignore[arg-type]


# ── Factory ──────────────────────────────────────────────────────────


async def new_qdrant_retrieve_engine_repository(
    host: str,
    port: int,
    api_key: str,
    use_tls: bool,
    index_config: IndexConfig | None = None,
) -> QdrantRetrieveEngineRepository:
    """Create a Qdrant retrieve engine repository with a connected client."""
    client = AsyncQdrantClient(
        host=host,
        grpc_port=port,
        prefer_grpc=True,
        api_key=api_key if api_key else None,
        https=use_tls,
    )
    collection_base_name = _resolve_collection_name(index_config)
    shard_number = index_config.shard_number if index_config else 0
    replication_factor = index_config.replication_factor if index_config else 0
    return QdrantRetrieveEngineRepository(
        client=client,
        collection_base_name=collection_base_name,
        shard_number=shard_number,
        replication_factor=replication_factor,
    )


__all__ = [
    "FIELD_CHUNK_ID",
    "FIELD_CONTENT",
    "FIELD_IS_ENABLED",
    "FIELD_KNOWLEDGE_BASE_ID",
    "FIELD_KNOWLEDGE_ID",
    "FIELD_SOURCE_ID",
    "FIELD_SOURCE_TYPE",
    "FIELD_TAG_ID",
    "QdrantRetrieveEngineRepository",
    "new_qdrant_retrieve_engine_repository",
]
