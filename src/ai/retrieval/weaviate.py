"""Weaviate retrieval engine repository.

Implements ``RetrieveEngineRepository`` against the Weaviate vector database.
Collection management creates dimension-suffixed classes with HNSW vector
indexes (``self_provided`` vectorizer, cosine distance) and per-property
filterable indexes for the IDs that participate in retrieval filters. Vector
retrieval uses ``near_vector``; keyword retrieval fans out to BM25 across every
dimension-suffixed collection owned by this repository.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import Mapping
from typing import Any, Protocol

from weaviate import connect_to_custom
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    Tokenization,
    VectorDistances,
)
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery

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

#: Name of the single self-provided named vector.
VECTOR_NAME = "embedding"

# ── Collection resolution ─────────────────────────────────────────────

ENV_WEAVIATE_COLLECTION = "WEAVIATE_COLLECTION"
DEFAULT_COLLECTION_NAME = "weknora_embeddings"

# ── Storage estimate constants (mirrors the upstream storage formula) ──

_HNSW_M = 32
_HNSW_LINK_BYTES = 8
_ID_TRACKER_BYTES = 24
_SOURCE_TYPE_BYTES = 8

# ── Batch + copy batch sizes ──────────────────────────────────────────

_BATCH_INSERT_SIZE = 100
_COPY_BATCH_SIZE = 64


# ── Client + collection protocols (structural seams for tests) ────────


class _WeaviateCollections(Protocol):
    """Subset of ``WeaviateAsyncClient.collections`` used by the repo."""

    async def exists(self, name: str) -> bool: ...

    async def list_all(self, *args: Any, **kwargs: Any) -> Any: ...

    def get(self, name: str, *args: Any, **kwargs: Any) -> Any: ...

    async def create(self, *args: Any, **kwargs: Any) -> Any: ...


class _WeaviateClientLike(Protocol):
    """Subset of ``WeaviateAsyncClient`` used by the repo."""

    collections: _WeaviateCollections

    async def close(self) -> None: ...


# ── Helpers ──────────────────────────────────────────────────────────


def _resolve_collection_name(index_config: IndexConfig | None) -> str:
    """Resolve the collection base name (upstream ``ResolveCollectionName``).

    Priority: ``collection_prefix`` > ``collection_name`` > env var
    ``WEAVIATE_COLLECTION`` > default ``weknora_embeddings``.
    """
    if index_config is not None:
        if index_config.collection_prefix:
            return index_config.collection_prefix
        if index_config.collection_name:
            return index_config.collection_name
    env_val = os.getenv(ENV_WEAVIATE_COLLECTION, "")
    if env_val:
        return env_val
    return DEFAULT_COLLECTION_NAME


def _collection_name(base: str, dimension: int) -> str:
    """Return the dimension-suffixed collection name."""
    return f"{base}_{dimension}"


def _resolve_replication_factor(index_config: IndexConfig | None) -> int:
    if index_config is None:
        return 0
    return index_config.replication_factor


def _resolve_desired_shard_count(index_config: IndexConfig | None) -> int:
    if index_config is None:
        return 0
    return index_config.desired_shard_count


def _resolve_hnsw_ef_construction(index_config: IndexConfig | None) -> int:
    if index_config is None or index_config.hnsw_ef_construction <= 0:
        return 128
    return index_config.hnsw_ef_construction


def _resolve_hnsw_ef(index_config: IndexConfig | None) -> int:
    if index_config is None or index_config.hnsw_ef_search <= 0:
        return 64
    return index_config.hnsw_ef_search


def _resolve_hnsw_m(index_config: IndexConfig | None) -> int:
    if index_config is None or index_config.hnsw_m <= 0:
        return _HNSW_M
    return index_config.hnsw_m


def _to_embedding(info: IndexInfo, params: IndexSaveParams) -> list[float]:
    """Pull the per-source-id embedding from ``params`` (empty when absent)."""
    embedding_map = params.get(EMBEDDING_KEY, {})
    if isinstance(embedding_map, dict):
        raw = embedding_map.get(info.source_id, [])
        if raw:
            return [float(x) for x in raw]
    return []


def _build_properties(data: dict[str, Any]) -> dict[str, Any]:
    """Translate ``IndexInfo`` -> Weaviate payload dict."""
    return {
        FIELD_CONTENT: data[FIELD_CONTENT],
        FIELD_SOURCE_ID: data[FIELD_SOURCE_ID],
        FIELD_SOURCE_TYPE: data[FIELD_SOURCE_TYPE],
        FIELD_CHUNK_ID: data[FIELD_CHUNK_ID],
        FIELD_KNOWLEDGE_ID: data[FIELD_KNOWLEDGE_ID],
        FIELD_KNOWLEDGE_BASE_ID: data[FIELD_KNOWLEDGE_BASE_ID],
        FIELD_TAG_ID: data[FIELD_TAG_ID],
        FIELD_IS_ENABLED: data[FIELD_IS_ENABLED],
    }


def _payload_from_info(info: IndexInfo) -> dict[str, Any]:
    return _build_properties(
        {
            FIELD_CONTENT: info.content,
            FIELD_SOURCE_ID: info.source_id,
            FIELD_SOURCE_TYPE: int(info.source_type),
            FIELD_CHUNK_ID: info.chunk_id,
            FIELD_KNOWLEDGE_ID: info.knowledge_id,
            FIELD_KNOWLEDGE_BASE_ID: info.knowledge_base_id,
            FIELD_TAG_ID: info.tag_id,
            FIELD_IS_ENABLED: info.is_enabled,
        }
    )


def _payload_to_index_with_score(
    point_id: Any,
    payload: Mapping[str, Any] | None,
    score: float,
    match_type: MatchType,
) -> IndexWithScore:
    """Convert a Weaviate object payload to ``IndexWithScore``."""
    payload = payload or {}
    source_type_raw = payload.get(FIELD_SOURCE_TYPE, 0)
    if isinstance(source_type_raw, int):
        source_type = SourceType(source_type_raw)
    else:
        try:
            source_type = SourceType(int(source_type_raw))
        except (TypeError, ValueError):
            source_type = SourceType.CHUNK
    return IndexWithScore(
        id=str(point_id),
        content=str(payload.get(FIELD_CONTENT, "")),
        source_id=str(payload.get(FIELD_SOURCE_ID, "")),
        source_type=source_type,
        chunk_id=str(payload.get(FIELD_CHUNK_ID, "")),
        knowledge_id=str(payload.get(FIELD_KNOWLEDGE_ID, "")),
        knowledge_base_id=str(payload.get(FIELD_KNOWLEDGE_BASE_ID, "")),
        tag_id=str(payload.get(FIELD_TAG_ID, "")),
        score=float(score),
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
            retriever_engine_type=RetrieverEngineType.WEAVIATE,
            retriever_type=retriever_type,
        )
    ]


def _calculate_storage_size(
    content: str,
    source_id: str,
    chunk_id: str,
    knowledge_id: str,
    knowledge_base_id: str,
    embedding: list[float],
    *,
    hnsw_m: int = _HNSW_M,
) -> int:
    """Estimate per-point storage (upstream ``calculateStorageSize``)."""
    payload_size = (
        len(content)
        + len(source_id)
        + len(chunk_id)
        + len(knowledge_id)
        + len(knowledge_base_id)
        + _SOURCE_TYPE_BYTES
    )
    vector_size = 0
    hnsw_index = 0
    if embedding:
        dimensions = len(embedding)
        vector_size = dimensions * 4
        hnsw_index = hnsw_m * 2 * _HNSW_LINK_BYTES
    return payload_size + vector_size + hnsw_index + _ID_TRACKER_BYTES


def _matches_base_name(collection_name: str, base_name: str) -> bool:
    """Check whether ``collection_name`` belongs to this repository."""
    return collection_name.startswith(base_name)


def _resolve_target_source_id(
    original_source_id: str, source_chunk_id: str, target_chunk_id: str
) -> str:
    """Resolve the target source id, preserving question suffixes."""
    if original_source_id == source_chunk_id:
        return target_chunk_id
    prefix = f"{source_chunk_id}-"
    if original_source_id.startswith(prefix):
        question_id = original_source_id[len(prefix) :]
        return f"{target_chunk_id}-{question_id}"
    return str(uuid.uuid4())


# ── Repository ───────────────────────────────────────────────────────


class WeaviateRetrieveEngineRepository:
    """Weaviate-backed retrieve engine repository.

    Manages dimension-suffixed classes with lazy creation and per-property
    filterable indexes. Supports keyword and vector retrieval.
    """

    def __init__(
        self,
        client: _WeaviateClientLike,
        collection_base_name: str,
        replication_factor: int = 0,
        desired_shard_count: int = 0,
        hnsw_ef_construction: int = 128,
        hnsw_ef: int = 64,
        hnsw_m: int = _HNSW_M,
    ) -> None:
        self._client = client
        self._collection_base_name = collection_base_name
        self._replication_factor = replication_factor
        self._desired_shard_count = desired_shard_count
        self._hnsw_ef_construction = hnsw_ef_construction
        self._hnsw_ef = hnsw_ef
        self._hnsw_m = hnsw_m
        self._lock = asyncio.Lock()
        self._initialized: set[int] = set()

    # ── Engine metadata ──────────────────────────────────────────────

    def engine_type(self) -> RetrieverEngineType:
        return RetrieverEngineType.WEAVIATE

    def support(self) -> list[RetrieverType]:
        return [RetrieverType.KEYWORDS, RetrieverType.VECTOR]

    # ── Collection helpers ───────────────────────────────────────────

    def _get_collection_name(self, dimension: int) -> str:
        return _collection_name(self._collection_base_name, dimension)

    async def _ensure_collection(self, ctx: Context, dimension: int) -> None:
        """Create the Weaviate class for ``dimension`` if it does not exist."""
        del ctx
        if dimension in self._initialized:
            return
        async with self._lock:
            if dimension in self._initialized:
                return
            collection_name = self._get_collection_name(dimension)
            if not await self._client.collections.exists(collection_name):
                create_kwargs: dict[str, Any] = {
                    "name": collection_name,
                    "description": f"Embedding collection (dim={dimension})",
                    "properties": self._build_properties_schema(),
                    "vector_config": Configure.Vectors.self_provided(
                        name=VECTOR_NAME,
                        vector_index_config=Configure.VectorIndex.hnsw(
                            distance_metric=VectorDistances.COSINE,
                            ef_construction=self._hnsw_ef_construction,
                            ef=self._hnsw_ef,
                            max_connections=self._hnsw_m,
                        ),
                    ),
                }
                if self._replication_factor > 0:
                    create_kwargs["replication_config"] = Configure.replication(
                        factor=self._replication_factor,
                    )
                if self._desired_shard_count > 0:
                    create_kwargs["sharding_config"] = Configure.sharding(
                        desired_count=self._desired_shard_count,
                    )
                await self._client.collections.create(**create_kwargs)
                logger.info(
                    "Weaviate collection {} created (dim={})",
                    collection_name,
                    dimension,
                )
            self._initialized.add(dimension)

    @staticmethod
    def _build_properties_schema() -> list[Property]:
        """Build the per-property schema for the embedding class."""
        return [
            Property(
                name=FIELD_CONTENT,
                data_type=DataType.TEXT,
                tokenization=Tokenization.GSE,
                index_filterable=False,
                index_searchable=True,
            ),
            Property(
                name=FIELD_SOURCE_ID,
                data_type=DataType.TEXT,
                index_filterable=False,
                index_searchable=False,
            ),
            Property(
                name=FIELD_SOURCE_TYPE,
                data_type=DataType.INT,
                index_filterable=False,
                index_searchable=False,
            ),
            Property(
                name=FIELD_CHUNK_ID,
                data_type=DataType.TEXT,
                index_filterable=True,
                index_searchable=False,
            ),
            Property(
                name=FIELD_KNOWLEDGE_ID,
                data_type=DataType.TEXT,
                index_filterable=True,
                index_searchable=False,
            ),
            Property(
                name=FIELD_KNOWLEDGE_BASE_ID,
                data_type=DataType.TEXT,
                index_filterable=True,
                index_searchable=False,
            ),
            Property(
                name=FIELD_TAG_ID,
                data_type=DataType.TEXT,
                index_filterable=True,
                index_searchable=False,
            ),
            Property(
                name=FIELD_IS_ENABLED,
                data_type=DataType.BOOL,
                index_filterable=True,
                index_searchable=False,
            ),
        ]

    # ── Filter assembly ──────────────────────────────────────────────

    @staticmethod
    def _build_base_filter(params: RetrieveParams) -> Any:
        """Build the common ``where`` filter (enabled + KB / KW / tag / exclude)."""
        operands: list[Any] = [
            Filter.by_property(FIELD_IS_ENABLED).equal(True),
        ]
        if params.knowledge_base_ids:
            operands.append(
                Filter.by_property(FIELD_KNOWLEDGE_BASE_ID).contains_any(
                    params.knowledge_base_ids
                )
            )
        if params.knowledge_ids:
            operands.append(
                Filter.by_property(FIELD_KNOWLEDGE_ID).contains_any(
                    params.knowledge_ids
                )
            )
        if params.tag_ids:
            operands.append(
                Filter.by_property(FIELD_TAG_ID).contains_any(params.tag_ids)
            )
        if params.exclude_knowledge_ids:
            operands.append(
                Filter.all_of(
                    [
                        Filter.by_property(FIELD_KNOWLEDGE_ID).not_equal(
                            value
                        )
                        for value in params.exclude_knowledge_ids
                    ]
                )
            )
        if params.exclude_chunk_ids:
            operands.append(
                Filter.all_of(
                    [
                        Filter.by_property(FIELD_CHUNK_ID).not_equal(value)
                        for value in params.exclude_chunk_ids
                    ]
                )
            )
        if len(operands) == 1:
            return operands[0]
        return Filter.all_of(operands)

    # ── Save ──────────────────────────────────────────────────────────

    async def save(
        self, ctx: Context, index_info: IndexInfo, params: IndexSaveParams
    ) -> None:
        """Store a single object in Weaviate."""
        embedding = _to_embedding(index_info, params)
        if not embedding:
            raise ValidationError(
                code="weaviate.empty_embedding",
                message=f"empty embedding vector for chunk ID: {index_info.chunk_id}",
            )
        dimension = len(embedding)
        await self._ensure_collection(ctx, dimension)
        collection_name = self._get_collection_name(dimension)
        collection = self._client.collections.get(collection_name)
        object_id = index_info.chunk_id or str(uuid.uuid4())
        await collection.data.insert(
            uuid=object_id,
            properties=_payload_from_info(index_info),
            vector={VECTOR_NAME: embedding},
        )

    async def batch_save(
        self,
        ctx: Context,
        index_info_list: list[IndexInfo],
        params: IndexSaveParams,
    ) -> None:
        """Store multiple objects grouped by dimension (batches of 100)."""
        if not index_info_list:
            return
        pending_by_dim: dict[int, list[tuple[IndexInfo, list[float]]]] = {}
        for info in index_info_list:
            embedding = _to_embedding(info, params)
            if not embedding:
                logger.warning(
                    "Weaviate: skipping empty embedding for chunk {}",
                    info.chunk_id,
                )
                continue
            pending_by_dim.setdefault(len(embedding), []).append((info, embedding))
        if not pending_by_dim:
            return
        for dimension, pending in pending_by_dim.items():
            await self._ensure_collection(ctx, dimension)
            collection_name = self._get_collection_name(dimension)
            collection = self._client.collections.get(collection_name)
            for start in range(0, len(pending), _BATCH_INSERT_SIZE):
                batch = pending[start : start + _BATCH_INSERT_SIZE]
                objects = [
                    {
                        "uuid": info.chunk_id or str(uuid.uuid4()),
                        "properties": _payload_from_info(info),
                        "vector": {VECTOR_NAME: embedding},
                    }
                    for info, embedding in batch
                ]
                await collection.data.insert_many(objects=objects)

    # ── Retrieve ─────────────────────────────────────────────────────

    async def retrieve(
        self, ctx: Context, params: RetrieveParams
    ) -> list[RetrieveResult]:
        """Dispatch retrieval based on ``params.retriever_type``."""
        if params.retriever_type == RetrieverType.VECTOR:
            return await self._vector_retrieve(ctx, params)
        if params.retriever_type == RetrieverType.KEYWORDS:
            return await self._keywords_retrieve(ctx, params)
        raise ValidationError(
            code="weaviate.invalid_retriever_type",
            message=f"invalid retriever type: {params.retriever_type}",
        )

    async def _vector_retrieve(
        self, ctx: Context, params: RetrieveParams
    ) -> list[RetrieveResult]:
        """Perform cosine near-vector similarity search."""
        del ctx
        dimension = len(params.embedding)
        collection_name = self._get_collection_name(dimension)
        if not await self._client.collections.exists(collection_name):
            return _build_retrieve_result([], RetrieverType.VECTOR)
        collection = self._client.collections.get(collection_name)
        kwargs: dict[str, Any] = {
            "near_vector": {VECTOR_NAME: params.embedding},
            "limit": params.top_k,
            "filters": self._build_base_filter(params),
            "return_metadata": MetadataQuery(certainty=True),
        }
        if params.threshold > 0.0:
            kwargs["certainty"] = float(params.threshold)
        response = await collection.query.near_vector(**kwargs)
        results = [
            _payload_to_index_with_score(
                obj.uuid,
                obj.properties,
                float(getattr(obj.metadata, "certainty", 0.0) or 0.0),
                MatchType.EMBEDDING,
            )
            for obj in response.objects
        ]
        return _build_retrieve_result(results, RetrieverType.VECTOR)

    async def _keywords_retrieve(
        self, ctx: Context, params: RetrieveParams
    ) -> list[RetrieveResult]:
        """Perform BM25 keyword search across all matching collections."""
        del ctx
        all_names = await self._list_collection_names()
        matching = [
            name for name in all_names if _matches_base_name(name, self._collection_base_name)
        ]
        if not matching:
            return _build_retrieve_result([], RetrieverType.KEYWORDS)
        base_filter = self._build_base_filter(params)
        query = params.query or ""
        all_results: list[IndexWithScore] = []
        for collection_name in matching:
            collection = self._client.collections.get(collection_name)
            response = await collection.query.bm25(
                query=query,
                query_properties=[FIELD_CONTENT],
                limit=params.top_k,
                filters=base_filter,
                return_metadata=MetadataQuery(score=True),
            )
            for obj in response.objects:
                all_results.append(
                    _payload_to_index_with_score(
                        obj.uuid,
                        obj.properties,
                        1.0,
                        MatchType.KEYWORDS,
                    )
                )
        if params.top_k > 0 and len(all_results) > params.top_k:
            all_results = all_results[: params.top_k]
        return _build_retrieve_result(all_results, RetrieverType.KEYWORDS)

    async def _list_collection_names(self) -> list[str]:
        """List every collection class name from the live schema."""
        listed = await self._client.collections.list_all()
        names: list[str] = []
        for item in listed:
            name = getattr(item, "name", None) or getattr(item, "class_name", None)
            if name:
                names.append(str(name))
        return names

    # ── Delete ────────────────────────────────────────────────────────

    async def delete_by_chunk_id_list(
        self,
        ctx: Context,
        index_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        """Delete objects matching the chunk id list."""
        del ctx, knowledge_type
        if not index_id_list:
            return
        collection_name = self._get_collection_name(dimension)
        if not await self._client.collections.exists(collection_name):
            return
        collection = self._client.collections.get(collection_name)
        await collection.data.delete_many(
            where=Filter.by_property(FIELD_CHUNK_ID).contains_any(index_id_list),
        )

    async def delete_by_source_id_list(
        self,
        ctx: Context,
        source_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        """Delete objects matching the source id list."""
        del ctx, knowledge_type
        if not source_id_list:
            return
        collection_name = self._get_collection_name(dimension)
        if not await self._client.collections.exists(collection_name):
            return
        collection = self._client.collections.get(collection_name)
        await collection.data.delete_many(
            where=Filter.by_property(FIELD_SOURCE_ID).contains_any(source_id_list),
        )

    async def delete_by_knowledge_id_list(
        self,
        ctx: Context,
        knowledge_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        """Delete objects matching the knowledge id list."""
        del ctx, knowledge_type
        if not knowledge_id_list:
            return
        collection_name = self._get_collection_name(dimension)
        if not await self._client.collections.exists(collection_name):
            return
        collection = self._client.collections.get(collection_name)
        await collection.data.delete_many(
            where=Filter.by_property(FIELD_KNOWLEDGE_ID).contains_any(knowledge_id_list),
        )

    # ── Batch update ──────────────────────────────────────────────────

    async def batch_update_chunk_enabled_status(
        self, ctx: Context, chunk_status_map: Mapping[str, bool]
    ) -> None:
        """Update ``is_enabled`` across all matching collections (by chunk id)."""
        del ctx
        if not chunk_status_map:
            return
        all_names = await self._list_collection_names()
        for collection_name in all_names:
            if not _matches_base_name(collection_name, self._collection_base_name):
                continue
            collection = self._client.collections.get(collection_name)
            for chunk_id, enabled in chunk_status_map.items():
                try:
                    await collection.data.update(
                        uuid=chunk_id,
                        properties={FIELD_IS_ENABLED: bool(enabled)},
                    )
                except Exception as exc:
                    logger.warning(
                        "Weaviate update {} in {} failed: {}",
                        chunk_id,
                        collection_name,
                        exc,
                    )

    async def batch_update_chunk_tag_id(
        self, ctx: Context, chunk_tag_map: Mapping[str, str]
    ) -> None:
        """Update ``tag_id`` across all matching collections (by chunk id)."""
        del ctx
        if not chunk_tag_map:
            return
        all_names = await self._list_collection_names()
        for collection_name in all_names:
            if not _matches_base_name(collection_name, self._collection_base_name):
                continue
            collection = self._client.collections.get(collection_name)
            for chunk_id, tag_id in chunk_tag_map.items():
                try:
                    await collection.data.update(
                        uuid=chunk_id,
                        properties={FIELD_TAG_ID: tag_id},
                    )
                except Exception as exc:
                    logger.warning(
                        "Weaviate tag update {} in {} failed: {}",
                        chunk_id,
                        collection_name,
                        exc,
                    )

    # ── Copy indices ─────────────────────────────────────────────────

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
        collection = self._client.collections.get(collection_name)
        offset: str | None = None
        total_copied = 0
        while True:
            response = await collection.query.fetch_objects(
                limit=_COPY_BATCH_SIZE,
                filters=Filter.by_property(FIELD_KNOWLEDGE_BASE_ID).equal(
                    source_knowledge_base_id
                ),
                include_vector=[VECTOR_NAME],
                after=offset,
                return_metadata=MetadataQuery(),
            )
            objects = list(response.objects)
            if not objects:
                break
            target_objects: list[dict[str, Any]] = []
            for obj in objects:
                payload = obj.properties or {}
                source_chunk_id = str(payload.get(FIELD_CHUNK_ID, ""))
                source_knowledge_id = str(payload.get(FIELD_KNOWLEDGE_ID, ""))
                original_source_id = str(payload.get(FIELD_SOURCE_ID, ""))
                target_chunk_id = source_to_target_chunk_id_map.get(source_chunk_id)
                if target_chunk_id is None:
                    continue
                target_knowledge_id = source_to_target_kb_id_map.get(source_knowledge_id)
                if target_knowledge_id is None:
                    continue
                target_source_id = _resolve_target_source_id(
                    original_source_id, source_chunk_id, target_chunk_id
                )
                vector = obj.vector
                embedding = (
                    vector.get(VECTOR_NAME) if isinstance(vector, Mapping) else vector
                )
                if embedding is None:
                    continue
                new_properties = _build_properties(
                    {
                        FIELD_CONTENT: str(payload.get(FIELD_CONTENT, "")),
                        FIELD_SOURCE_ID: target_source_id,
                        FIELD_SOURCE_TYPE: payload.get(FIELD_SOURCE_TYPE, 0),
                        FIELD_CHUNK_ID: target_chunk_id,
                        FIELD_KNOWLEDGE_ID: target_knowledge_id,
                        FIELD_KNOWLEDGE_BASE_ID: target_knowledge_base_id,
                        FIELD_TAG_ID: str(payload.get(FIELD_TAG_ID, "")),
                        FIELD_IS_ENABLED: bool(payload.get(FIELD_IS_ENABLED, True)),
                    }
                )
                target_objects.append(
                    {
                        "uuid": str(uuid.uuid4()),
                        "properties": new_properties,
                        "vector": {VECTOR_NAME: list(embedding)},
                    }
                )
            if target_objects:
                await collection.data.insert_many(objects=target_objects)
                total_copied += len(target_objects)
            offset = str(objects[-1].uuid)
            if len(objects) < _COPY_BATCH_SIZE:
                break
        logger.info(
            "Weaviate: copied {} points from {} to {}",
            total_copied,
            source_knowledge_base_id,
            target_knowledge_base_id,
        )

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
            embedding = _to_embedding(info, params)
            total += _calculate_storage_size(
                info.content,
                info.source_id,
                info.chunk_id,
                info.knowledge_id,
                info.knowledge_base_id,
                embedding,
                hnsw_m=self._hnsw_m,
            )
        return total


# ── Factory ──────────────────────────────────────────────────────────


def _split_host(grpc_address: str) -> tuple[str, int]:
    """Split ``host:port`` (or ``host``) into ``(host, port)``."""
    if not grpc_address:
        return ("weaviate", 50051)
    last_colon = grpc_address.rfind(":")
    if last_colon <= 0:
        return (grpc_address, 50051)
    host = grpc_address[:last_colon]
    port_raw = grpc_address[last_colon + 1 :]
    try:
        port = int(port_raw)
    except ValueError:
        port = 50051
    return (host, port)


async def new_weaviate_retrieve_engine_repository(
    client: _WeaviateClientLike,
    index_config: IndexConfig | None = None,
) -> WeaviateRetrieveEngineRepository:
    """Build a repository around an already-constructed Weaviate client.

    Mirrors the upstream ``NewWeaviateRetrieveEngineRepository`` entry point:
    the caller is responsible for building the ``WeaviateAsyncClient`` (env or
    DB-store wiring), and this function only derives the per-instance
    settings (collection base name, replication, sharding, HNSW tuning).
    """
    collection_base_name = _resolve_collection_name(index_config)
    return WeaviateRetrieveEngineRepository(
        client=client,
        collection_base_name=collection_base_name,
        replication_factor=_resolve_replication_factor(index_config),
        desired_shard_count=_resolve_desired_shard_count(index_config),
        hnsw_ef_construction=_resolve_hnsw_ef_construction(index_config),
        hnsw_ef=_resolve_hnsw_ef(index_config),
        hnsw_m=_resolve_hnsw_m(index_config),
    )


def _build_weaviate_client(
    host: str,
    grpc_address: str,
    scheme: str,
    api_key: str,
    *,
    skip_init_checks: bool = True,
) -> Any:
    """Construct a ``WeaviateAsyncClient`` from the env driver wiring."""
    http_secure = scheme.lower() == "https"
    grpc_host, grpc_port = _split_host(grpc_address)
    auth = Auth.api_key(api_key) if api_key else None
    return connect_to_custom(
        http_host=host,
        http_port=80,
        http_secure=http_secure,
        grpc_host=grpc_host,
        grpc_port=grpc_port,
        grpc_secure=http_secure,
        auth_credentials=auth,
        skip_init_checks=skip_init_checks,
    )


async def new_weaviate_retrieve_engine_repository_from_env(
    host: str,
    grpc_address: str,
    scheme: str,
    api_key: str,
    index_config: IndexConfig | None = None,
) -> WeaviateRetrieveEngineRepository:
    """Connect to Weaviate and build the engine repository.

    The env driver (``RETRIEVE_DRIVER=weaviate``) wires this entry point so a
    single ``WeaviateAsyncClient`` is constructed per driver enable.
    """
    client = _build_weaviate_client(host, grpc_address, scheme, api_key)
    return await new_weaviate_retrieve_engine_repository(client, index_config)


__all__ = [
    "FIELD_CHUNK_ID",
    "FIELD_CONTENT",
    "FIELD_IS_ENABLED",
    "FIELD_KNOWLEDGE_BASE_ID",
    "FIELD_KNOWLEDGE_ID",
    "FIELD_SOURCE_ID",
    "FIELD_SOURCE_TYPE",
    "FIELD_TAG_ID",
    "VECTOR_NAME",
    "WeaviateRetrieveEngineRepository",
    "new_weaviate_retrieve_engine_repository",
    "new_weaviate_retrieve_engine_repository_from_env",
]
