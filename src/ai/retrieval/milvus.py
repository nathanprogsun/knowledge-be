"""Milvus retrieval engine repository.

Mirrors the upstream Milvus retriever: per-dimension collection management
(HNSW vector index, BM25 sparse index, scalar payload indexes), single and
batch upsert, vector / keywords retrieval, delete-by-field, copy-indices,
batch chunk status / tag updates, and storage-size estimation.

The collection schema pairs a float-vector ``embedding`` field with a
BM25-function-backed sparse ``content_sparse`` field so the same collection
serves both vector and full-text search. Scalar payload fields carry an
auto index for filtered retrieval.

pymilvus is the client SDK. The repository is constructed with an injected
client so tests can substitute a fake; the module-level constructor
``new_milvus_retrieve_engine_repository`` builds the client from a
``MilvusClientConfig``.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pymilvus import DataType, Function, FunctionType, MilvusClient
from pymilvus.milvus_client.index import IndexParams

from src.ai.embedding import Context
from src.ai.retrieval.types import (
    IndexConfig,
    IndexInfo,
    IndexSaveParams,
    IndexWithScore,
    MatchType,
    MilvusClientConfig,
    RetrieveParams,
    RetrieverEngineType,
    RetrieveResult,
    RetrieverType,
    SourceType,
)
from src.app_logging import logger
from src.common.exception import ValidationError

# ── Field constants (upstream const block) ──────────────────────────

FIELD_ID: str = "id"
FIELD_CONTENT: str = "content"
FIELD_SOURCE_ID: str = "source_id"
FIELD_SOURCE_TYPE: str = "source_type"
FIELD_CHUNK_ID: str = "chunk_id"
FIELD_KNOWLEDGE_ID: str = "knowledge_id"
FIELD_KNOWLEDGE_BASE_ID: str = "knowledge_base_id"
FIELD_TAG_ID: str = "tag_id"
FIELD_EMBEDDING: str = "embedding"
FIELD_IS_ENABLED: str = "is_enabled"
FIELD_CONTENT_SPARSE: str = "content_sparse"

#: Fields read back from query / search results (excludes the auto-generated
#: sparse vector). Matches the upstream ``allFields`` slice.
ALL_FIELDS: tuple[str, ...] = (
    FIELD_ID,
    FIELD_CONTENT,
    FIELD_SOURCE_ID,
    FIELD_SOURCE_TYPE,
    FIELD_CHUNK_ID,
    FIELD_KNOWLEDGE_ID,
    FIELD_KNOWLEDGE_BASE_ID,
    FIELD_TAG_ID,
    FIELD_IS_ENABLED,
    FIELD_EMBEDDING,
)

# ── Env / defaults ───────────────────────────────────────────────────

ENV_MILVUS_COLLECTION: str = "MILVUS_COLLECTION"
ENV_MILVUS_METRIC_TYPE: str = "MILVUS_METRIC_TYPE"
DEFAULT_COLLECTION_NAME: str = "weknora_embeddings"

#: Default metric type when ``MILVUS_METRIC_TYPE`` is unset or unrecognized.
DEFAULT_METRIC_TYPE: str = "IP"

#: Accepted metric type names (case-insensitive env value -> Milvus name).
_METRIC_TYPES: dict[str, str] = {"COSINE": "COSINE", "L2": "L2", "IP": "IP"}

# HNSW index parameters (upstream ``NewHNSWIndex(metricType, 16, 128)``).
HNSW_M: int = 16
HNSW_EF_CONSTRUCTION: int = 128

#: Batch size for paginated source reads in ``copy_indices``.
COPY_BATCH_SIZE: int = 64

#: BM25 function name wiring ``content`` -> ``content_sparse``.
BM25_FUNCTION_NAME: str = "text_bm25_emb"

#: Scalar payload fields that receive an auto index for filtered retrieval
#: (upstream ``indexFields``).
_SCALAR_INDEX_FIELDS: tuple[str, ...] = (
    FIELD_CHUNK_ID,
    FIELD_KNOWLEDGE_ID,
    FIELD_KNOWLEDGE_BASE_ID,
    FIELD_SOURCE_ID,
    FIELD_IS_ENABLED,
)


# ── Collection name / metric resolution (upstream vectorstore.go) ────


def _resolve_collection_name(
    index_config: IndexConfig | None, env_key: str, default_val: str
) -> str:
    """Resolve the collection base name.

    Priority: ``IndexConfig.collection_prefix`` > ``collection_name`` > env var
    > default. Mirrors the upstream ``ResolveCollectionName``.
    """
    if index_config is not None:
        if index_config.collection_prefix:
            return index_config.collection_prefix
        if index_config.collection_name:
            return index_config.collection_name
    env_val = os.getenv(env_key, "")
    if env_val:
        return env_val
    return default_val


def _resolve_metric_type() -> str:
    """Resolve the Milvus metric type from ``MILVUS_METRIC_TYPE``."""
    raw = os.getenv(ENV_MILVUS_METRIC_TYPE, "")
    if not raw:
        return DEFAULT_METRIC_TYPE
    key = raw.upper()
    mapped = _METRIC_TYPES.get(key)
    if mapped is None:
        logger.warning(
            "[Milvus] Unknown MILVUS_METRIC_TYPE '{}', using default {}",
            raw,
            DEFAULT_METRIC_TYPE,
        )
        return DEFAULT_METRIC_TYPE
    logger.info("[Milvus] Using metric type: {}", mapped)
    return mapped


def _get_shards_num(index_config: IndexConfig | None) -> int:
    """Return the configured shards_num, or 0 for the server default."""
    if index_config is not None and index_config.shards_num > 0:
        return index_config.shards_num
    return 0


def _get_replica_number(index_config: IndexConfig | None) -> int:
    """Return the configured replica_number, or 0 for the server default."""
    if index_config is not None and index_config.replica_number > 0:
        return index_config.replica_number
    return 0


# ── Filter expression builder (upstream filter.go) ───────────────────
#
# Builds Milvus template expressions (``field op {param}``) with a params
# dict so values are bound server-side rather than inlined. The converter is
# stateless; a per-conversion counter produces unique param names.


@dataclass(frozen=True, slots=True)
class UniversalFilterCondition:
    """One node of the universal filter tree.

    ``field`` / ``value`` are required for comparison and ``in`` operators;
    ``value`` holds the sub-condition list for logical operators.
    """

    field: str = ""
    operator: str = ""
    value: Any = None


_COMPARISON_OPERATORS: dict[str, str] = {
    "eq": "==",
    "ne": "!=",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
    "like": "like",
    "not like": "not like",
}

_LOGICAL_OPERATORS: frozenset[str] = frozenset({"and", "or"})
_IN_OPERATORS: frozenset[str] = frozenset({"in", "not in"})


@dataclass(frozen=True, slots=True)
class ConvertedFilter:
    """Result of converting one filter condition."""

    expr_str: str = ""
    params: dict[str, Any] = field(default_factory=dict)


class MilvusFilterConverter:
    """Convert ``UniversalFilterCondition`` trees to Milvus template expressions."""

    def convert(self, cond: UniversalFilterCondition | None) -> ConvertedFilter:
        counter = [0]
        return self._convert_condition(cond, counter)

    # ── internal helpers ───────────────────────────────────────────

    @staticmethod
    def _param_name(field: str, counter: list[int]) -> str:
        counter[0] += 1
        return f"{field.replace('.', '_')}_{counter[0]}"

    def _convert_comparison(
        self, cond: UniversalFilterCondition, counter: list[int]
    ) -> ConvertedFilter:
        if not cond.field or cond.value is None:
            raise ValidationError(
                code="milvus.filter_condition_nil",
                message="milvus filter condition is nil",
            )
        operator = _COMPARISON_OPERATORS.get(cond.operator)
        if operator is None:
            raise ValidationError(
                code="milvus.unsupported_comparison_operator",
                message=f"unsupported comparison operator: {cond.operator}",
            )
        name = self._param_name(cond.field, counter)
        return ConvertedFilter(
            expr_str=f"{cond.field} {operator} {{{name}}}",
            params={name: cond.value},
        )

    def _convert_logical(
        self, cond: UniversalFilterCondition, counter: list[int]
    ) -> ConvertedFilter:
        if cond.value is None:
            raise ValidationError(
                code="milvus.filter_condition_nil",
                message="milvus filter condition is nil",
            )
        children = cond.value
        if not isinstance(children, list):
            raise ValidationError(
                code="milvus.invalid_logical_condition_value",
                message="invalid logical condition value type",
            )
        expr = ""
        params: dict[str, Any] = {}
        for child in children:
            child_result = self._convert_condition(child, counter)
            if not child_result.expr_str:
                continue
            if not expr:
                expr = child_result.expr_str
                params = dict(child_result.params)
            else:
                expr = f"({expr}) {cond.operator} ({child_result.expr_str})"
                params.update(child_result.params)
        if not expr:
            raise ValidationError(
                code="milvus.empty_logical_condition",
                message="empty logical condition",
            )
        return ConvertedFilter(expr_str=expr, params=params)

    def _convert_in(
        self, cond: UniversalFilterCondition, counter: list[int]
    ) -> ConvertedFilter:
        if not cond.field or cond.value is None:
            raise ValidationError(
                code="milvus.filter_condition_nil",
                message="milvus filter condition is nil",
            )
        if not isinstance(cond.value, list) or len(cond.value) == 0:
            raise ValidationError(
                code="milvus.invalid_in_value",
                message="in operator value must be a slice with at least one value",
            )
        name = self._param_name(cond.field, counter)
        return ConvertedFilter(
            expr_str=f"{cond.field} {cond.operator} {{{name}}}",
            params={name: cond.value},
        )

    def _convert_between(
        self, cond: UniversalFilterCondition, counter: list[int]
    ) -> ConvertedFilter:
        if not cond.field or cond.value is None:
            raise ValidationError(
                code="milvus.filter_condition_nil",
                message="milvus filter condition is nil",
            )
        if not isinstance(cond.value, list) or len(cond.value) != 2:
            raise ValidationError(
                code="milvus.invalid_between_value",
                message="between operator value must be a slice with two elements",
            )
        base = self._param_name(cond.field, counter)
        name1 = f"{base}_0"
        name2 = f"{base}_1"
        return ConvertedFilter(
            expr_str=f"{cond.field} >= {{{name1}}} and {cond.field} <= {{{name2}}}",
            params={name1: cond.value[0], name2: cond.value[1]},
        )

    def _convert_condition(
        self, cond: UniversalFilterCondition | None, counter: list[int]
    ) -> ConvertedFilter:
        if cond is None:
            raise ValidationError(
                code="milvus.filter_condition_nil",
                message="milvus filter condition is nil",
            )
        if cond.operator in _COMPARISON_OPERATORS:
            return self._convert_comparison(cond, counter)
        if cond.operator in _LOGICAL_OPERATORS:
            return self._convert_logical(cond, counter)
        if cond.operator in _IN_OPERATORS:
            return self._convert_in(cond, counter)
        if cond.operator == "between":
            return self._convert_between(cond, counter)
        raise ValidationError(
            code="milvus.unsupported_operator",
            message=f"unsupported operator: {cond.operator}",
        )


# ── Embedding row (upstream structs.go) ─────────────────────────────


@dataclass
class MilvusVectorEmbedding:
    """One indexed row as persisted to Milvus."""

    id: str = ""
    content: str = ""
    source_id: str = ""
    source_type: int = 0
    chunk_id: str = ""
    knowledge_id: str = ""
    knowledge_base_id: str = ""
    tag_id: str = ""
    embedding: list[float] = field(default_factory=list)
    is_enabled: bool = False


@dataclass
class MilvusVectorEmbeddingWithScore:
    """A retrieved row with its match score."""

    embedding: MilvusVectorEmbedding
    score: float = 0.0


# ── Conversion helpers ───────────────────────────────────────────────


def _to_milvus_vector_embedding(
    index_info: IndexInfo, params: IndexSaveParams
) -> MilvusVectorEmbedding:
    """Convert ``IndexInfo`` to the Milvus row shape (upstream ``toMilvusVectorEmbedding``)."""
    vector = MilvusVectorEmbedding(
        content=index_info.content,
        source_id=index_info.source_id,
        source_type=int(index_info.source_type),
        chunk_id=index_info.chunk_id,
        knowledge_id=index_info.knowledge_id,
        knowledge_base_id=index_info.knowledge_base_id,
        tag_id=index_info.tag_id,
        is_enabled=index_info.is_enabled,
    )
    if params:
        embedding_map = params.get(FIELD_EMBEDDING)
        if embedding_map is not None:
            vector.embedding = list(embedding_map.get(index_info.source_id, []))
    return vector


def _from_milvus_vector_embedding(
    row_id: str, emb: MilvusVectorEmbeddingWithScore, match_type: MatchType
) -> IndexWithScore:
    """Convert a retrieved row to the domain ``IndexWithScore`` (upstream ``fromMilvusVectorEmbedding``)."""
    return IndexWithScore(
        id=row_id,
        content=emb.embedding.content,
        source_id=emb.embedding.source_id,
        source_type=SourceType(emb.embedding.source_type),
        chunk_id=emb.embedding.chunk_id,
        knowledge_id=emb.embedding.knowledge_id,
        knowledge_base_id=emb.embedding.knowledge_base_id,
        tag_id=emb.embedding.tag_id,
        score=emb.score,
        match_type=match_type,
        is_enabled=emb.embedding.is_enabled,
    )


def _to_upsert_row(emb: MilvusVectorEmbedding) -> dict[str, Any]:
    """Build one row dict for pymilvus ``upsert`` (upstream ``createUpsert``).

    The sparse vector is omitted: the BM25 function derives it from
    ``content`` server-side.
    """
    return {
        FIELD_ID: emb.id,
        FIELD_EMBEDDING: emb.embedding,
        FIELD_CONTENT: emb.content,
        FIELD_SOURCE_ID: emb.source_id,
        FIELD_SOURCE_TYPE: emb.source_type,
        FIELD_CHUNK_ID: emb.chunk_id,
        FIELD_KNOWLEDGE_ID: emb.knowledge_id,
        FIELD_KNOWLEDGE_BASE_ID: emb.knowledge_base_id,
        FIELD_TAG_ID: emb.tag_id,
        FIELD_IS_ENABLED: emb.is_enabled,
    }


def _convert_result_row(row: Mapping[str, Any]) -> MilvusVectorEmbeddingWithScore:
    """Convert one pymilvus result dict to a scored row.

    pymilvus returns rows as dicts keyed by field name; ``distance`` carries
    the similarity score for search results (absent for plain queries).
    """
    embedding = MilvusVectorEmbedding(
        id=str(row.get(FIELD_ID, "")),
        content=str(row.get(FIELD_CONTENT, "")),
        source_id=str(row.get(FIELD_SOURCE_ID, "")),
        source_type=int(row.get(FIELD_SOURCE_TYPE, 0)),
        chunk_id=str(row.get(FIELD_CHUNK_ID, "")),
        knowledge_id=str(row.get(FIELD_KNOWLEDGE_ID, "")),
        knowledge_base_id=str(row.get(FIELD_KNOWLEDGE_BASE_ID, "")),
        tag_id=str(row.get(FIELD_TAG_ID, "")),
        embedding=list(row.get(FIELD_EMBEDDING, [])),
        is_enabled=bool(row.get(FIELD_IS_ENABLED, False)),
    )
    return MilvusVectorEmbeddingWithScore(
        embedding=embedding, score=float(row.get("distance", 0.0))
    )


def _calculate_storage_size(emb: MilvusVectorEmbedding) -> int:
    """Estimate the per-row storage footprint (upstream ``calculateStorageSize``)."""
    payload = (
        len(emb.content)
        + len(emb.source_id)
        + len(emb.chunk_id)
        + len(emb.knowledge_id)
        + len(emb.knowledge_base_id)
        + 8  # source_type int64
    )
    vector_size = 0
    index_bytes = 0
    if emb.embedding:
        dimensions = len(emb.embedding)
        vector_size = dimensions * 4
        index_bytes = vector_size + 16
    metadata_bytes = 32
    return payload + vector_size + index_bytes + metadata_bytes


# ── Schema / index construction ─────────────────────────────────────


def _build_collection_schema(dimension: int, collection_name: str) -> Any:
    """Build the Milvus collection schema for a given dimension."""
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(FIELD_ID, DataType.VARCHAR, max_length=1024, is_primary=True)
    schema.add_field(FIELD_EMBEDDING, DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_field(
        FIELD_CONTENT,
        DataType.VARCHAR,
        max_length=65535,
        enable_analyzer=True,
        enable_match=True,
    )
    schema.add_field(FIELD_CONTENT_SPARSE, DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(FIELD_SOURCE_ID, DataType.VARCHAR, max_length=255)
    schema.add_field(FIELD_SOURCE_TYPE, DataType.INT64)
    schema.add_field(FIELD_CHUNK_ID, DataType.VARCHAR, max_length=255)
    schema.add_field(FIELD_KNOWLEDGE_ID, DataType.VARCHAR, max_length=255)
    schema.add_field(FIELD_KNOWLEDGE_BASE_ID, DataType.VARCHAR, max_length=255)
    schema.add_field(FIELD_TAG_ID, DataType.VARCHAR, max_length=255)
    schema.add_field(FIELD_IS_ENABLED, DataType.BOOL)
    schema.add_function(
        Function(
            name=BM25_FUNCTION_NAME,
            function_type=FunctionType.BM25,
            input_field_names=[FIELD_CONTENT],
            output_field_names=[FIELD_CONTENT_SPARSE],
        )
    )
    return schema


def _build_index_params(metric_type: str) -> IndexParams:
    """Build the index params: HNSW on the vector field, BM25 auto index on the
    sparse field, and auto indexes on scalar payload fields.
    """
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name=FIELD_EMBEDDING,
        index_type="HNSW",
        metric_type=metric_type,
        params={"M": HNSW_M, "efConstruction": HNSW_EF_CONSTRUCTION},
    )
    index_params.add_index(
        field_name=FIELD_CONTENT_SPARSE,
        index_type="AUTOINDEX",
        metric_type="BM25",
    )
    for field_name in _SCALAR_INDEX_FIELDS:
        index_params.add_index(
            field_name=field_name, index_type="AUTOINDEX", metric_type=DEFAULT_METRIC_TYPE
        )
    return index_params


# ── Repository ──────────────────────────────────────────────────────


class MilvusRetrieveEngineRepository:
    """Milvus-backed retrieve engine repository.

    Collections are dimension-scoped (``{base}_{dim}``) so embeddings of
    different dimensions never collide. The schema, indexes, and load state
    are created lazily on first use of a dimension.
    """

    def __init__(
        self,
        client: MilvusClient,
        index_config: IndexConfig | None = None,
    ) -> None:
        self._client = client
        self._collection_base_name = _resolve_collection_name(
            index_config, ENV_MILVUS_COLLECTION, DEFAULT_COLLECTION_NAME
        )
        self._metric_type = _resolve_metric_type()
        self._shards_num = _get_shards_num(index_config)
        self._replica_number = _get_replica_number(index_config)
        self._filter = MilvusFilterConverter()
        #: dimension -> True once the collection has been created and loaded.
        self._initialized_collections: dict[int, bool] = {}
        logger.info("[Milvus] Initializing Milvus retriever engine repository")
        logger.info("[Milvus] Using metric type: {}", self._metric_type)
        logger.info("[Milvus] Successfully initialized repository")

    # ── RetrieveEngine surface ─────────────────────────────────────

    def engine_type(self) -> RetrieverEngineType:
        return RetrieverEngineType.MILVUS

    def support(self) -> list[RetrieverType]:
        return [RetrieverType.KEYWORDS, RetrieverType.VECTOR]

    # ── Collection management ───────────────────────────────────────

    def _get_collection_name(self, dimension: int) -> str:
        return f"{self._collection_base_name}_{dimension}"

    def _ensure_collection(self, ctx: Context, dimension: int) -> None:
        """Create and load the dimension collection if not already initialized."""
        del ctx  # retained for upstream signature parity
        if self._initialized_collections.get(dimension):
            return
        collection_name = self._get_collection_name(dimension)
        has_collection = self._client.has_collection(collection_name=collection_name)
        if not has_collection:
            logger.info(
                "[Milvus] Creating collection {} with dimension {}",
                collection_name,
                dimension,
            )
            schema = _build_collection_schema(dimension, collection_name)
            index_params = _build_index_params(self._metric_type)
            create_kwargs: dict[str, Any] = {}
            if self._shards_num > 0:
                create_kwargs["shards_num"] = self._shards_num
            self._client.create_collection(
                collection_name=collection_name,
                schema=schema,
                **create_kwargs,
            )
            self._client.create_index(
                collection_name=collection_name, index_params=index_params
            )
            logger.info("[Milvus] Successfully created collection {}", collection_name)
        load_kwargs: dict[str, Any] = {}
        if self._replica_number > 0:
            load_kwargs["replica_number"] = self._replica_number
        self._client.load_collection(collection_name=collection_name, **load_kwargs)
        self._initialized_collections[dimension] = True

    # ── Save / BatchSave ───────────────────────────────────────────

    async def save(
        self, ctx: Context, index_info: IndexInfo, params: IndexSaveParams
    ) -> None:
        logger.debug("[Milvus] Saving index for chunk ID: {}", index_info.chunk_id)
        embedding_db = _to_milvus_vector_embedding(index_info, params)
        if not embedding_db.embedding:
            raise ValidationError(
                code="milvus.empty_embedding",
                message=f"empty embedding vector for chunk ID: {index_info.chunk_id}",
            )
        dimension = len(embedding_db.embedding)
        self._ensure_collection(ctx, dimension)
        collection_name = self._get_collection_name(dimension)
        embedding_db.id = str(uuid.uuid4())
        self._client.upsert(
            collection_name=collection_name,
            data=[_to_upsert_row(embedding_db)],
        )
        logger.info(
            "[Milvus] Successfully saved index for chunk ID: {}", index_info.chunk_id
        )

    async def batch_save(
        self,
        ctx: Context,
        index_info_list: list[IndexInfo],
        params: IndexSaveParams,
    ) -> None:
        if not index_info_list:
            logger.warning("[Milvus] Empty list provided to BatchSave, skipping")
            return
        logger.info("[Milvus] Batch saving {} indices", len(index_info_list))
        embeddings_by_dimension: dict[int, list[IndexInfo]] = {}
        for index_info in index_info_list:
            embedding_db = _to_milvus_vector_embedding(index_info, params)
            if not embedding_db.embedding:
                logger.warning(
                    "[Milvus] Skipping empty embedding for chunk ID: {}",
                    index_info.chunk_id,
                )
                continue
            dimension = len(embedding_db.embedding)
            embeddings_by_dimension.setdefault(dimension, []).append(index_info)
        if not embeddings_by_dimension:
            logger.warning("[Milvus] No valid points to save after filtering")
            return
        total_saved = 0
        for dimension, embeddings in embeddings_by_dimension.items():
            self._ensure_collection(ctx, dimension)
            collection_name = self._get_collection_name(dimension)
            rows = []
            for index_info in embeddings:
                embedding_db = _to_milvus_vector_embedding(index_info, params)
                embedding_db.id = str(uuid.uuid4())
                rows.append(_to_upsert_row(embedding_db))
            self._client.upsert(collection_name=collection_name, data=rows)
            total_saved += len(embeddings)
            logger.info(
                "[Milvus] Saved {} points to collection {}",
                len(embeddings),
                collection_name,
            )
        logger.info("[Milvus] Successfully batch saved {} indices", total_saved)

    # ── Retrieve ───────────────────────────────────────────────────

    async def retrieve(
        self, ctx: Context, params: RetrieveParams
    ) -> list[RetrieveResult]:
        logger.debug(
            "[Milvus] Processing retrieval request of type: {}", params.retriever_type
        )
        if params.retriever_type == RetrieverType.VECTOR:
            return await self.vector_retrieve(ctx, params)
        if params.retriever_type == RetrieverType.KEYWORDS:
            return await self.keywords_retrieve(ctx, params)
        raise ValidationError(
            code="milvus.invalid_retriever_type",
            message=f"invalid retriever type: {params.retriever_type}",
        )

    async def vector_retrieve(
        self, ctx: Context, params: RetrieveParams
    ) -> list[RetrieveResult]:
        del ctx
        dimension = len(params.embedding)
        logger.info(
            "[Milvus] Vector retrieval: dim={}, topK={}, threshold={:.4f}",
            dimension,
            params.top_k,
            params.threshold,
        )
        collection_name = self._get_collection_name(dimension)
        has_collection = self._client.has_collection(collection_name=collection_name)
        if not has_collection:
            logger.warning(
                "[Milvus] Collection {} does not exist, returning empty results",
                collection_name,
            )
            return self._build_retrieve_result([], RetrieverType.VECTOR)
        expr, expr_params = self._build_base_filter(params)
        search_params: dict[str, Any] = {}
        if params.threshold > 0:
            search_params["params"] = {"radius": params.threshold}
        results = self._client.search(
            collection_name=collection_name,
            data=[list(params.embedding)],
            filter=expr,
            limit=params.top_k,
            output_fields=list(ALL_FIELDS),
            anns_field=FIELD_EMBEDDING,
            search_params=search_params or None,
            filter_params=expr_params,
        )
        sets = self._convert_search_results(results)
        index_results = [
            _from_milvus_vector_embedding(s.embedding.id, s, MatchType.EMBEDDING)
            for s in sets
        ]
        if not index_results:
            logger.warning(
                "[Milvus] No vector matches found that meet threshold {:.4f}",
                params.threshold,
            )
        else:
            logger.info(
                "[Milvus] Vector retrieval found {} results", len(index_results)
            )
        return self._build_retrieve_result(index_results, RetrieverType.VECTOR)

    async def keywords_retrieve(
        self, ctx: Context, params: RetrieveParams
    ) -> list[RetrieveResult]:
        del ctx
        logger.info(
            "[Milvus] Performing keywords retrieval with query: {}, topK: {}",
            params.query,
            params.top_k,
        )
        collections = self._client.list_collections()
        all_results: list[IndexWithScore] = []
        for collection_name in collections:
            if not self._owns_collection(collection_name):
                continue
            expr, expr_params = self._build_base_filter(params)
            try:
                results = self._client.search(
                    collection_name=collection_name,
                    data=[params.query],
                    filter=expr,
                    limit=params.top_k,
                    output_fields=list(ALL_FIELDS),
                    anns_field=FIELD_CONTENT_SPARSE,
                    filter_params=expr_params,
                )
            except Exception as exc:
                logger.error("[Milvus] Keywords search failed: {}", exc)
                continue
            sets = self._convert_search_results(results)
            for s in sets:
                s.score = 1.0
                all_results.append(
                    _from_milvus_vector_embedding(s.embedding.id, s, MatchType.KEYWORDS)
                )
        if len(all_results) > params.top_k:
            all_results = all_results[: params.top_k]
        if not all_results:
            logger.warning(
                "[Milvus] No keyword matches found for query: {}", params.query
            )
        else:
            logger.info(
                "[Milvus] Keywords retrieval found {} results", len(all_results)
            )
        return self._build_retrieve_result(all_results, RetrieverType.KEYWORDS)

    # ── Delete by * ────────────────────────────────────────────────

    async def delete_by_chunk_id_list(
        self,
        ctx: Context,
        index_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        del ctx, knowledge_type
        await self._delete_by_field(FIELD_CHUNK_ID, index_id_list, dimension, "chunk IDs")

    async def delete_by_knowledge_id_list(
        self,
        ctx: Context,
        knowledge_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        del ctx, knowledge_type
        await self._delete_by_field(
            FIELD_KNOWLEDGE_ID, knowledge_id_list, dimension, "knowledge IDs"
        )

    async def delete_by_source_id_list(
        self,
        ctx: Context,
        source_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        del ctx, knowledge_type
        await self._delete_by_field(
            FIELD_SOURCE_ID, source_id_list, dimension, "source IDs"
        )

    async def _delete_by_field(
        self,
        field: str,
        ids: list[str],
        dimension: int,
        label: str,
    ) -> None:
        if not ids:
            logger.warning(
                "[Milvus] Empty {} list provided for deletion, skipping", label
            )
            return
        collection_name = self._get_collection_name(dimension)
        logger.info(
            "[Milvus] Deleting indices by {} from {}, count: {}",
            label,
            collection_name,
            len(ids),
        )
        converted = self._filter.convert(
            UniversalFilterCondition(field=field, operator="in", value=ids)
        )
        self._client.delete(
            collection_name=collection_name,
            filter=converted.expr_str,
            filter_params=converted.params,
        )
        logger.info("[Milvus] Successfully deleted documents by {}", label)

    # ── CopyIndices ─────────────────────────────────────────────────

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
        del knowledge_type
        logger.info(
            "[Milvus] Copying indices from source knowledge base {} to target {}, "
            "count: {}, dimension: {}",
            source_knowledge_base_id,
            target_knowledge_base_id,
            len(source_to_target_chunk_id_map),
            dimension,
        )
        if not source_to_target_chunk_id_map:
            logger.warning("[Milvus] Empty mapping, skipping copy")
            return
        collection_name = self._get_collection_name(dimension)
        self._ensure_collection(ctx, dimension)
        total_copied = 0
        offset = 0
        while True:
            source_rows, count = self._search_by_filter(
                ctx,
                collection_name,
                UniversalFilterCondition(
                    field=FIELD_KNOWLEDGE_BASE_ID,
                    operator="eq",
                    value=source_knowledge_base_id,
                ),
                limit=COPY_BATCH_SIZE,
                offset=offset,
            )
            if not source_rows:
                break
            target_rows: list[MilvusVectorEmbedding] = []
            for source_row in source_rows:
                source_chunk_id = source_row.embedding.chunk_id
                source_knowledge_id = source_row.embedding.knowledge_id
                original_source_id = source_row.embedding.source_id
                target_chunk_id = source_to_target_chunk_id_map.get(source_chunk_id)
                if target_chunk_id is None:
                    logger.warning(
                        "[Milvus] Source chunk {} not found in target mapping, skipping",
                        source_chunk_id,
                    )
                    continue
                target_knowledge_id = source_to_target_kb_id_map.get(
                    source_knowledge_id
                )
                if target_knowledge_id is None:
                    logger.warning(
                        "[Milvus] Source knowledge {} not found in target mapping, skipping",
                        source_knowledge_id,
                    )
                    continue
                target_source_id = self._resolve_target_source_id(
                    original_source_id, source_chunk_id, target_chunk_id
                )
                target_rows.append(
                    MilvusVectorEmbedding(
                        id=str(uuid.uuid4()),
                        content=source_row.embedding.content,
                        source_id=target_source_id,
                        source_type=source_row.embedding.source_type,
                        chunk_id=target_chunk_id,
                        knowledge_id=target_knowledge_id,
                        knowledge_base_id=target_knowledge_base_id,
                        tag_id=source_row.embedding.tag_id,
                        embedding=source_row.embedding.embedding,
                        is_enabled=source_row.embedding.is_enabled,
                    )
                )
            if target_rows:
                self._client.upsert(
                    collection_name=collection_name,
                    data=[_to_upsert_row(r) for r in target_rows],
                )
                total_copied += len(target_rows)
                logger.info(
                    "[Milvus] Successfully copied batch, batch size: {}, total copied: {}",
                    len(target_rows),
                    total_copied,
                )
            if count < COPY_BATCH_SIZE:
                break
            offset += count
        logger.info("[Milvus] Index copy completed, total copied: {}", total_copied)

    @staticmethod
    def _resolve_target_source_id(
        original_source_id: str, source_chunk_id: str, target_chunk_id: str
    ) -> str:
        """Map a source_id to its target counterpart (upstream CopyIndices logic)."""
        if original_source_id == source_chunk_id:
            return target_chunk_id
        prefix = f"{source_chunk_id}-"
        if original_source_id.startswith(prefix):
            question_id = original_source_id[len(prefix) :]
            return f"{target_chunk_id}-{question_id}"
        return str(uuid.uuid4())

    # ── BatchUpdateChunk* ──────────────────────────────────────────

    async def batch_update_chunk_enabled_status(
        self, ctx: Context, chunk_status_map: Mapping[str, bool]
    ) -> None:
        if not chunk_status_map:
            logger.warning("[Milvus] Empty chunk status map provided, skipping")
            return
        logger.info(
            "[Milvus] Batch updating chunk enabled status, count: {}",
            len(chunk_status_map),
        )
        collections = self._client.list_collections()
        enabled_ids = [cid for cid, enabled in chunk_status_map.items() if enabled]
        disabled_ids = [cid for cid, enabled in chunk_status_map.items() if not enabled]
        errors: list[Exception] = []
        for collection_name in collections:
            if not self._owns_collection(collection_name):
                continue
            for ids, enabled in ((enabled_ids, True), (disabled_ids, False)):
                try:
                    self._update_chunk_enabled_status_in_collection(
                        ctx, collection_name, ids, enabled
                    )
                except Exception as exc:
                    errors.append(exc)
                    logger.warning(
                        "[Milvus] Failed to update chunks in {}: {}",
                        collection_name,
                        exc,
                    )
        if errors:
            raise errors[0]
        logger.info("[Milvus] Batch update chunk enabled status completed")

    def _update_chunk_enabled_status_in_collection(
        self,
        ctx: Context,
        collection_name: str,
        chunk_ids: list[str],
        enabled: bool,
    ) -> None:
        if not chunk_ids:
            return
        embeddings, _ = self._search_by_filter(
            ctx,
            collection_name,
            UniversalFilterCondition(
                field=FIELD_CHUNK_ID, operator="in", value=chunk_ids
            ),
            limit=None,
            offset=None,
        )
        if not embeddings:
            return
        for row in embeddings:
            row.embedding.is_enabled = enabled
        self._client.upsert(
            collection_name=collection_name,
            data=[_to_upsert_row(r.embedding) for r in embeddings],
        )

    async def batch_update_chunk_tag_id(
        self, ctx: Context, chunk_tag_map: Mapping[str, str]
    ) -> None:
        if not chunk_tag_map:
            logger.warning("[Milvus] Empty chunk tag map provided, skipping")
            return
        logger.info("[Milvus] Batch updating chunk tag ID, count: {}", len(chunk_tag_map))
        collections = self._client.list_collections()
        tag_groups: dict[str, list[str]] = {}
        for chunk_id, tag_id in chunk_tag_map.items():
            tag_groups.setdefault(tag_id, []).append(chunk_id)
        for collection_name in collections:
            if not self._owns_collection(collection_name):
                continue
            for tag_id, chunk_ids in tag_groups.items():
                try:
                    embeddings, _ = self._search_by_filter(
                        ctx,
                        collection_name,
                        UniversalFilterCondition(
                            field=FIELD_CHUNK_ID, operator="in", value=chunk_ids
                        ),
                        limit=None,
                        offset=None,
                    )
                except Exception as exc:
                    logger.warning(
                        "[Milvus] Failed to search chunks in {}: {}",
                        collection_name,
                        exc,
                    )
                    continue
                if not embeddings:
                    continue
                for row in embeddings:
                    row.embedding.tag_id = tag_id
                try:
                    self._client.upsert(
                        collection_name=collection_name,
                        data=[_to_upsert_row(r.embedding) for r in embeddings],
                    )
                except Exception as exc:
                    logger.warning(
                        "[Milvus] Failed to update chunks in {}: {}",
                        collection_name,
                        exc,
                    )

    # ── EstimateStorageSize ─────────────────────────────────────────

    def estimate_storage_size(
        self,
        ctx: Context,
        index_info_list: list[IndexInfo],
        params: IndexSaveParams,
    ) -> int:
        del ctx
        total = 0
        for index_info in index_info_list:
            embedding_db = _to_milvus_vector_embedding(index_info, params)
            total += _calculate_storage_size(embedding_db)
        logger.info(
            "[Milvus] Storage size for {} indices: {} bytes",
            len(index_info_list),
            total,
        )
        return total

    # ── Shared helpers ─────────────────────────────────────────────

    def _owns_collection(self, collection_name: str) -> bool:
        """Report whether ``collection_name`` belongs to this repository's base."""
        base = self._collection_base_name
        if len(collection_name) <= len(base):
            return False
        return collection_name[: len(base)] == base

    def _build_base_filter(
        self, params: RetrieveParams
    ) -> tuple[str, dict[str, Any]]:
        """Build the common filter expression from retrieve params."""
        conditions: list[UniversalFilterCondition] = []
        if params.knowledge_base_ids:
            conditions.append(
                UniversalFilterCondition(
                    field=FIELD_KNOWLEDGE_BASE_ID, operator="in", value=list(params.knowledge_base_ids)
                )
            )
        if params.knowledge_ids:
            conditions.append(
                UniversalFilterCondition(
                    field=FIELD_KNOWLEDGE_ID, operator="in", value=list(params.knowledge_ids)
                )
            )
        if params.tag_ids:
            conditions.append(
                UniversalFilterCondition(
                    field=FIELD_TAG_ID, operator="in", value=list(params.tag_ids)
                )
            )
        if params.exclude_knowledge_ids:
            conditions.append(
                UniversalFilterCondition(
                    field=FIELD_KNOWLEDGE_ID,
                    operator="not in",
                    value=list(params.exclude_knowledge_ids),
                )
            )
        if params.exclude_chunk_ids:
            conditions.append(
                UniversalFilterCondition(
                    field=FIELD_CHUNK_ID,
                    operator="not in",
                    value=list(params.exclude_chunk_ids),
                )
            )
        conditions.append(
            UniversalFilterCondition(
                field=FIELD_IS_ENABLED, operator="eq", value=True
            )
        )
        if not conditions:
            return "", {}
        result = self._filter.convert(
            UniversalFilterCondition(operator="and", value=conditions)
        )
        return result.expr_str, result.params

    def _search_by_filter(
        self,
        ctx: Context,
        collection_name: str,
        cond: UniversalFilterCondition,
        limit: int | None,
        offset: int | None,
    ) -> tuple[list[MilvusVectorEmbeddingWithScore], int]:
        """Query rows matching ``cond`` with optional pagination."""
        del ctx
        converted = self._filter.convert(cond)
        query_kwargs: dict[str, Any] = {
            "collection_name": collection_name,
            "filter": converted.expr_str,
            "output_fields": list(ALL_FIELDS),
            "filter_params": converted.params,
        }
        if limit is not None:
            query_kwargs["limit"] = limit
        if offset is not None:
            query_kwargs["offset"] = offset
        rows = self._client.query(**query_kwargs)
        embeddings = [_convert_result_row(row) for row in rows]
        return embeddings, len(embeddings)

    @staticmethod
    def _convert_search_results(
        results: Any,
    ) -> list[MilvusVectorEmbeddingWithScore]:
        """Convert pymilvus search results (list of hit lists) to scored rows."""
        if not results:
            return []
        first = results[0]
        if not first:
            return []
        return [_convert_result_row(row) for row in first]

    @staticmethod
    def _build_retrieve_result(
        results: list[IndexWithScore], retriever_type: RetrieverType
    ) -> list[RetrieveResult]:
        return [
            RetrieveResult(
                results=results,
                retriever_engine_type=RetrieverEngineType.MILVUS,
                retriever_type=retriever_type,
                error=None,
            )
        ]


# ── Constructor ──────────────────────────────────────────────────────


async def new_milvus_retrieve_engine_repository(
    client_cfg: MilvusClientConfig,
    index_config: IndexConfig | None = None,
) -> MilvusRetrieveEngineRepository:
    """Build a ``MilvusRetrieveEngineRepository`` from a client config.

    Constructs the pymilvus ``MilvusClient`` and wraps it in the repository.
    ``index_config`` is optional - pass ``None`` for the env path.
    """
    client_kwargs: dict[str, Any] = {"uri": client_cfg.address}
    if client_cfg.username:
        client_kwargs["user"] = client_cfg.username
    if client_cfg.password:
        client_kwargs["password"] = client_cfg.password
    if client_cfg.db_name:
        client_kwargs["db_name"] = client_cfg.db_name
    client = MilvusClient(**client_kwargs)
    return MilvusRetrieveEngineRepository(client, index_config)


__all__ = [
    "ALL_FIELDS",
    "BM25_FUNCTION_NAME",
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_METRIC_TYPE",
    "ENV_MILVUS_COLLECTION",
    "ENV_MILVUS_METRIC_TYPE",
    "FIELD_CHUNK_ID",
    "FIELD_CONTENT",
    "FIELD_CONTENT_SPARSE",
    "FIELD_EMBEDDING",
    "FIELD_ID",
    "FIELD_IS_ENABLED",
    "FIELD_KNOWLEDGE_BASE_ID",
    "FIELD_KNOWLEDGE_ID",
    "FIELD_SOURCE_ID",
    "FIELD_SOURCE_TYPE",
    "FIELD_TAG_ID",
    "HNSW_EF_CONSTRUCTION",
    "HNSW_M",
    "MilvusClientConfig",
    "MilvusFilterConverter",
    "MilvusRetrieveEngineRepository",
    "MilvusVectorEmbedding",
    "MilvusVectorEmbeddingWithScore",
    "UniversalFilterCondition",
    "new_milvus_retrieve_engine_repository",
]
