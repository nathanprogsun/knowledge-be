"""Tencent VectorDB retrieve engine repository (upstream ``tencentvectordb/repository.go``).

Communicates with Tencent VectorDB over its RPC SDK (``tcvectordb``).
Collections are sharded by embedding dimension:
``<collectionBaseName>_<dim>`` when no explicit ``CollectionName`` is set,
or a single collection otherwise.

The SDK is synchronous, so every call is dispatched through
``asyncio.to_thread``. Keyword retrieval uses ``search_by_text`` (text-based
search) instead of the upstream ``FullTextSearch`` with BM25 sparse vectors,
because the Python SDK does not ship the BM25 text encoder
(``tcvdbtext.encoder``) that the Go path depends on.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from src.ai.embedding import Context
from src.ai.retrieval.types import (
    IndexConfig,
    IndexInfo,
    IndexSaveParams,
    RetrieveParams,
    RetrieverEngineType,
    RetrieveResult,
    RetrieverType,
)
from src.app_logging import logger

# ── Constants ────────────────────────────────────────────────────────

_ENV_TENCENT_VECTORDB_DATABASE = "TENCENT_VECTORDB_DATABASE"
_ENV_TENCENT_VECTORDB_COLLECTION = "TENCENT_VECTORDB_COLLECTION"
_ENV_TENCENT_VECTORDB_REPLICA_NUM = "TENCENT_VECTORDB_REPLICA_NUMBER"
_DEFAULT_DATABASE_NAME = "weknora"
_DEFAULT_COLLECTION_NAME = "weknora_embeddings"
_DEFAULT_REPLICA_NUMBER = 1
_COPY_INDICES_PAGE_SIZE = 500

_FIELD_ID = "id"
_FIELD_VECTOR = "vector"
_FIELD_SPARSE_VECTOR = "sparse_vector"
_FIELD_CONTENT = "content"
_FIELD_SOURCE_ID = "source_id"
_FIELD_SOURCE_TYPE = "source_type"
_FIELD_CHUNK_ID = "chunk_id"
_FIELD_KNOWLEDGE_ID = "knowledge_id"
_FIELD_KNOWLEDGE_BASE_ID = "knowledge_base_id"
_FIELD_TAG_ID = "tag_id"
_FIELD_IS_ENABLED = "is_enabled"

_OUTPUT_FIELDS = [
    _FIELD_ID, _FIELD_CONTENT, _FIELD_SOURCE_ID, _FIELD_SOURCE_TYPE,
    _FIELD_CHUNK_ID, _FIELD_KNOWLEDGE_ID, _FIELD_KNOWLEDGE_BASE_ID,
    _FIELD_TAG_ID, _FIELD_IS_ENABLED,
]


# ── Helpers ─────────────────────────────────────────────────────────


def _resolve_collection_name(index_cfg: IndexConfig | None, env_key: str, default_val: str) -> str:
    if index_cfg is not None:
        if index_cfg.collection_prefix != "":
            return index_cfg.collection_prefix
        if index_cfg.collection_name != "":
            return index_cfg.collection_name
    env_val = os.getenv(env_key, "")
    if env_val != "":
        return env_val
    return default_val


def _resolve_replica_number(index_cfg: IndexConfig | None) -> int:
    if index_cfg is not None and index_cfg.replica_number > 0:
        return index_cfg.replica_number
    raw = os.getenv(_ENV_TENCENT_VECTORDB_REPLICA_NUM, "").strip()
    if raw:
        try:
            replicas = int(raw)
            if replicas >= 0:
                return replicas
        except ValueError:
            pass
    return _DEFAULT_REPLICA_NUMBER


def _should_use_dimension_suffix(index_cfg: IndexConfig | None) -> bool:
    return index_cfg is None or index_cfg.collection_name == ""


def _default_if_zero(value: int, default: int) -> int:
    return value if value > 0 else default


def _clean_invalid_utf8(s: str) -> str:
    """Strip invalid UTF-8 sequences and NUL chars (upstream ``cleanInvalidUTF8``)."""
    cleaned: list[str] = []
    for ch in s:
        if ch == "\x00":
            continue
        if unicodedata.category(ch) != "Co":
            cleaned.append(ch)
    return "".join(cleaned)


def _bool_to_uint64(v: bool) -> int:
    return 1 if v else 0


def _translate_source_id(original: str, source_chunk_id: str, target_chunk_id: str) -> str:
    if original == source_chunk_id:
        return target_chunk_id
    if original.startswith(source_chunk_id + "-"):
        question_id = original.removeprefix(source_chunk_id + "-")
        return f"{target_chunk_id}-{question_id}"
    digest = hashlib.sha256(
        f"{target_chunk_id}\x00{source_chunk_id}\x00{original}".encode()
    ).hexdigest()[:16]
    return f"{target_chunk_id}-{digest}"


def _in_expr(field: str, values: list[str]) -> str:
    quoted = ", ".join(f'"{v}"' for v in values)
    return f'{field} in ({quoted})'


# ── Domain model ────────────────────────────────────────────────────


@dataclass
class VectorEmbedding:
    """One document in a Tencent VectorDB collection (upstream ``vectorEmbedding``)."""

    id: str = ""
    content: str = ""
    source_id: str = ""
    source_type: int = 0
    chunk_id: str = ""
    knowledge_id: str = ""
    knowledge_base_id: str = ""
    tag_id: str = ""
    embedding: list[float] = field(default_factory=list)
    sparse_vector: Any = None
    is_enabled: bool = False
    score: float = 0.0


def _to_vector_embedding(index_info: IndexInfo, params: IndexSaveParams) -> VectorEmbedding:
    emb: list[float] = []
    embedding_map = params.get("embedding") if params else None
    if isinstance(embedding_map, dict):
        raw = embedding_map.get(index_info.source_id)
        if raw is None:
            raw = embedding_map.get(index_info.chunk_id)
        if raw is not None:
            emb = [float(v) for v in raw]
    doc_id = index_info.id or index_info.source_id or index_info.chunk_id
    return VectorEmbedding(
        id=doc_id,
        content=_clean_invalid_utf8(index_info.content),
        source_id=index_info.source_id,
        source_type=int(index_info.source_type),
        chunk_id=index_info.chunk_id,
        knowledge_id=index_info.knowledge_id,
        knowledge_base_id=index_info.knowledge_base_id,
        tag_id=index_info.tag_id,
        embedding=emb,
        is_enabled=index_info.is_enabled,
    )


def _to_document(emb: VectorEmbedding) -> dict[str, Any]:
    """Build a dict document for the SDK's ``upsert``."""
    return {
        "id": emb.id,
        "vector": emb.embedding,
        _FIELD_CONTENT: emb.content,
        _FIELD_SOURCE_ID: emb.source_id,
        _FIELD_SOURCE_TYPE: _bool_to_uint64(bool(emb.source_type)),
        _FIELD_CHUNK_ID: emb.chunk_id,
        _FIELD_KNOWLEDGE_ID: emb.knowledge_id,
        _FIELD_KNOWLEDGE_BASE_ID: emb.knowledge_base_id,
        _FIELD_TAG_ID: emb.tag_id,
        _FIELD_IS_ENABLED: _bool_to_uint64(emb.is_enabled),
    }


def _from_document(doc: dict) -> VectorEmbedding:
    """Parse a search/query result dict into a VectorEmbedding."""
    fields = doc.get("fields", doc)

    def _str(name: str) -> str:
        v = fields.get(name, "")
        return str(v) if v is not None else ""

    def _int(name: str) -> int:
        v = fields.get(name, 0)
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    return VectorEmbedding(
        id=str(doc.get("id", "")),
        content=_str(_FIELD_CONTENT),
        source_id=_str(_FIELD_SOURCE_ID),
        source_type=_int(_FIELD_SOURCE_TYPE),
        chunk_id=_str(_FIELD_CHUNK_ID),
        knowledge_id=_str(_FIELD_KNOWLEDGE_ID),
        knowledge_base_id=_str(_FIELD_KNOWLEDGE_BASE_ID),
        tag_id=_str(_FIELD_TAG_ID),
        embedding=doc.get("vector", []) or [],
        is_enabled=_int(_FIELD_IS_ENABLED) == 1,
        score=float(doc.get("score", 0.0)),
    )


def _base_filter(params: RetrieveParams) -> str:
    conditions = [f"{_FIELD_IS_ENABLED}=1"]
    if params.knowledge_base_ids:
        conditions.append(_in_expr(_FIELD_KNOWLEDGE_BASE_ID, list(params.knowledge_base_ids)))
    if params.knowledge_ids:
        conditions.append(_in_expr(_FIELD_KNOWLEDGE_ID, list(params.knowledge_ids)))
    if params.tag_ids:
        conditions.append(_in_expr(_FIELD_TAG_ID, list(params.tag_ids)))
    if params.exclude_knowledge_ids:
        conditions.append(f"not ({_in_expr(_FIELD_KNOWLEDGE_ID, list(params.exclude_knowledge_ids))})")
    if params.exclude_chunk_ids:
        conditions.append(f"not ({_in_expr(_FIELD_CHUNK_ID, list(params.exclude_chunk_ids))})")
    return " and ".join(conditions)


def _retrieve_result(
    results: list, retriever_type: RetrieverType
) -> list[RetrieveResult]:
    return [RetrieveResult(
        results=results,
        retriever_engine_type=RetrieverEngineType.TENCENT_VECTORDB,
        retriever_type=retriever_type,
        error=None,
    )]


# ── Repository ──────────────────────────────────────────────────────


class TencentVectorDBRepository:
    """Tencent VectorDB retrieve engine repository (upstream ``repository``)."""

    def __init__(
        self,
        client: Any,
        database_name: str,
        index_cfg: IndexConfig | None,
    ) -> None:
        if not database_name:
            database_name = os.getenv(_ENV_TENCENT_VECTORDB_DATABASE, "")
        if not database_name:
            database_name = _DEFAULT_DATABASE_NAME
        self._client = client
        self._database_name = database_name
        self._collection_base_name = _resolve_collection_name(
            index_cfg, _ENV_TENCENT_VECTORDB_COLLECTION, _DEFAULT_COLLECTION_NAME
        )
        self._use_dimension_suffix = _should_use_dimension_suffix(index_cfg)
        shards_num = 1
        if index_cfg is not None:
            shards_num = index_cfg.shards_num
        self._shards_num = _default_if_zero(shards_num, 1)
        self._replicas_num = _resolve_replica_number(index_cfg)
        self._initialized: set[int] = set()

    # ── async SDK wrappers ──

    async def _to_thread(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(func, *args, **kwargs)

    def _db(self) -> Any:
        return self._client.Database(self._database_name)

    def _collection(self, dimension: int) -> Any:
        return self._db().Collection(self._collection_name(dimension))

    def _collection_name(self, dimension: int) -> str:
        if not self._use_dimension_suffix:
            return self._collection_base_name
        return f"{self._collection_base_name}_{dimension}"

    def _matches_collection(self, collection_name: str) -> bool:
        if not self._use_dimension_suffix:
            return collection_name == self._collection_base_name
        return collection_name.startswith(self._collection_base_name + "_")

    # ── RetrieveEngine ──

    def engine_type(self) -> RetrieverEngineType:
        return RetrieverEngineType.TENCENT_VECTORDB

    def support(self) -> list[RetrieverType]:
        return [RetrieverType.KEYWORDS, RetrieverType.VECTOR]

    async def retrieve(self, ctx: Context, params: RetrieveParams) -> list[RetrieveResult]:
        if params.retriever_type == RetrieverType.VECTOR:
            return await self._vector_retrieve(ctx, params)
        if params.retriever_type == RetrieverType.KEYWORDS:
            return await self._keywords_retrieve(ctx, params)
        raise ValueError(f"invalid retriever type: {params.retriever_type}")

    # ── save / batch_save ──

    async def save(self, ctx: Context, index_info: IndexInfo, params: IndexSaveParams) -> None:
        await self.batch_save(ctx, [index_info], params)

    async def batch_save(
        self, ctx: Context, index_info_list: list[IndexInfo], params: IndexSaveParams
    ) -> None:
        if not index_info_list:
            return
        groups: dict[int, list[VectorEmbedding]] = {}
        for index_info in index_info_list:
            embedding = _to_vector_embedding(index_info, params)
            if not embedding.embedding:
                logger.warning("skip empty embedding for chunk_id={}", index_info.chunk_id)
                continue
            dim = len(embedding.embedding)
            groups.setdefault(dim, []).append(embedding)
        if not groups:
            return
        for dim, embeddings in groups.items():
            await self._ensure_collection(ctx, dim)
            collection_name = self._collection_name(dim)
            docs = [_to_document(e) for e in embeddings]
            try:
                await self._to_thread(
                    self._collection(dim).upsert, docs
                )
            except Exception as exc:
                raise RuntimeError(f"tencent vectordb batch save {collection_name}: {exc}") from exc

    # ── estimate_storage_size ──

    def estimate_storage_size(
        self, ctx: Context, index_info_list: list[IndexInfo], params: IndexSaveParams
    ) -> int:
        total = 0
        for index_info in index_info_list:
            embedding = _to_vector_embedding(index_info, params)
            total += len(embedding.content)
            total += len(embedding.embedding) * 4
            total += len(embedding.content) * 2
            total += (
                len(embedding.source_id) + len(embedding.chunk_id)
                + len(embedding.knowledge_id) + len(embedding.knowledge_base_id) + 256
            )
        return total

    # ── delete_by_* ──

    async def delete_by_chunk_id_list(
        self, ctx: Context, chunk_id_list: list[str], dimension: int, knowledge_type: str
    ) -> None:
        await self._delete_by_filter(ctx, dimension, _in_expr(_FIELD_CHUNK_ID, chunk_id_list))

    async def delete_by_source_id_list(
        self, ctx: Context, source_id_list: list[str], dimension: int, knowledge_type: str
    ) -> None:
        await self._delete_by_filter(ctx, dimension, _in_expr(_FIELD_SOURCE_ID, source_id_list))

    async def delete_by_knowledge_id_list(
        self, ctx: Context, knowledge_id_list: list[str], dimension: int, knowledge_type: str
    ) -> None:
        await self._delete_by_filter(ctx, dimension, _in_expr(_FIELD_KNOWLEDGE_ID, knowledge_id_list))

    async def _delete_by_filter(self, ctx: Context, dimension: int, cond: str) -> None:
        if not cond:
            return
        collection_name = self._collection_name(dimension)
        try:
            await self._to_thread(
                self._collection(dimension).delete,
                filter=cond,
            )
        except Exception as exc:
            raise RuntimeError(f"tencent vectordb delete from {collection_name}: {exc}") from exc

    # ── copy_indices ──

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
        if not source_to_target_chunk_id_map:
            return
        chunk_ids = list(source_to_target_chunk_id_map.keys())
        all_embeddings: list[VectorEmbedding] = []
        offset = 0
        while True:
            conditions = [_in_expr(_FIELD_CHUNK_ID, chunk_ids)]
            if source_knowledge_base_id:
                conditions.insert(
                    0, _in_expr(_FIELD_KNOWLEDGE_BASE_ID, [source_knowledge_base_id])
                )
            filter_expr = " and ".join(conditions)
            try:
                docs = await self._to_thread(
                    self._collection(dimension).query,
                    filter=filter_expr,
                    retrieve_vector=True,
                    output_fields=_OUTPUT_FIELDS,
                    limit=_COPY_INDICES_PAGE_SIZE,
                    offset=offset,
                )
            except Exception as exc:
                raise RuntimeError(f"tencent vectordb query source indices: {exc}") from exc
            batch = docs if isinstance(docs, list) else docs.get("documents", []) if isinstance(docs, dict) else []
            for doc in batch:
                emb = _from_document(doc)
                target_chunk_id = source_to_target_chunk_id_map.get(emb.chunk_id, "")
                if target_chunk_id == "":
                    continue
                original_source_id = emb.source_id or emb.id
                target_source_id = _translate_source_id(original_source_id, emb.chunk_id, target_chunk_id)
                emb.id = target_source_id
                emb.source_id = target_source_id
                emb.chunk_id = target_chunk_id
                emb.knowledge_base_id = target_knowledge_base_id
                target_kid = source_to_target_kb_id_map.get(emb.knowledge_id, "")
                if target_kid:
                    emb.knowledge_id = target_kid
                all_embeddings.append(emb)
            if len(batch) < _COPY_INDICES_PAGE_SIZE:
                break
            offset += _COPY_INDICES_PAGE_SIZE
        if not all_embeddings:
            return
        docs = [_to_document(e) for e in all_embeddings]
        try:
            await self._to_thread(self._collection(dimension).upsert, docs)
        except Exception as exc:
            raise RuntimeError(f"tencent vectordb copy indices: {exc}") from exc

    # ── batch_update_chunk_* ──

    async def batch_update_chunk_enabled_status(
        self, ctx: Context, chunk_status_map: Mapping[str, bool]
    ) -> None:
        if not chunk_status_map:
            return
        grouped: dict[bool, list[str]] = {}
        for chunk_id, enabled in chunk_status_map.items():
            grouped.setdefault(enabled, []).append(chunk_id)
        for enabled, chunk_ids in grouped.items():
            await self._update_chunk_fields(
                ctx, chunk_ids, {_FIELD_IS_ENABLED: _bool_to_uint64(enabled)}
            )

    async def batch_update_chunk_tag_id(
        self, ctx: Context, chunk_tag_map: Mapping[str, str]
    ) -> None:
        if not chunk_tag_map:
            return
        grouped: dict[str, list[str]] = {}
        for chunk_id, tag_id in chunk_tag_map.items():
            grouped.setdefault(tag_id, []).append(chunk_id)
        for tag_id, chunk_ids in grouped.items():
            await self._update_chunk_fields(ctx, chunk_ids, {_FIELD_TAG_ID: tag_id})

    async def _update_chunk_fields(
        self, ctx: Context, chunk_ids: list[str], fields: dict
    ) -> None:
        try:
            collections = await self._to_thread(self._db().list_collections)
        except Exception as exc:
            raise RuntimeError(f"tencent vectordb list collections: {exc}") from exc
        collection_list = collections if isinstance(collections, list) else getattr(collections, "collections", [])
        for collection in collection_list:
            name = collection if isinstance(collection, str) else getattr(collection, "collection_name", "")
            if not self._matches_collection(name):
                continue
            try:
                await self._to_thread(
                    self._db().Collection(name).update,
                    filter=_in_expr(_FIELD_CHUNK_ID, chunk_ids),
                    update_fields=fields,
                )
            except Exception as exc:
                raise RuntimeError(f"tencent vectordb update chunks in {name}: {exc}") from exc

    # ── retrieve ──

    async def _vector_retrieve(self, ctx: Context, params: RetrieveParams) -> list[RetrieveResult]:
        from src.ai.retrieval.types import IndexWithScore, MatchType
        dimension = len(params.embedding)
        if dimension == 0:
            return _retrieve_result([], RetrieverType.VECTOR)
        collection_name = self._collection_name(dimension)
        try:
            exists = await self._to_thread(self._db().exists_collection, collection_name)
        except Exception as exc:
            raise RuntimeError(f"tencent vectordb check collection {collection_name}: {exc}") from exc
        if not exists:
            return _retrieve_result([], RetrieverType.VECTOR)
        limit = params.top_k if params.top_k > 0 else 10
        search_kwargs: dict[str, Any] = {
            "filter": _base_filter(params),
            "retrieve_vector": False,
            "output_fields": _OUTPUT_FIELDS,
            "limit": limit,
        }
        if params.threshold > 0:
            search_kwargs["radius"] = float(params.threshold)
        try:
            search = await self._to_thread(
                self._collection(dimension).search,
                [params.embedding],
                **search_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(f"tencent vectordb vector search {collection_name}: {exc}") from exc
        docs = search[0] if search and isinstance(search, list) and isinstance(search[0], list) else []
        results = [
            IndexWithScore(
                id=d.get("id", ""),
                content=d.get("fields", {}).get(_FIELD_CONTENT, ""),
                source_id=d.get("fields", {}).get(_FIELD_SOURCE_ID, ""),
                source_type=int(d.get("fields", {}).get(_FIELD_SOURCE_TYPE, 0)),
                chunk_id=d.get("fields", {}).get(_FIELD_CHUNK_ID, ""),
                knowledge_id=d.get("fields", {}).get(_FIELD_KNOWLEDGE_ID, ""),
                knowledge_base_id=d.get("fields", {}).get(_FIELD_KNOWLEDGE_BASE_ID, ""),
                tag_id=d.get("fields", {}).get(_FIELD_TAG_ID, ""),
                score=float(d.get("score", 0.0)),
                match_type=MatchType.EMBEDDING,
                is_enabled=int(d.get("fields", {}).get(_FIELD_IS_ENABLED, 0)) == 1,
            )
            for d in docs
        ]
        return _retrieve_result(results, RetrieverType.VECTOR)

    async def _keywords_retrieve(self, ctx: Context, params: RetrieveParams) -> list[RetrieveResult]:
        from src.ai.retrieval.types import IndexWithScore, MatchType
        query = params.query.strip()
        if query == "":
            return _retrieve_result([], RetrieverType.KEYWORDS)
        try:
            collections = await self._to_thread(self._db().list_collections)
        except Exception as exc:
            raise RuntimeError(f"tencent vectordb list collections: {exc}") from exc
        collection_list = collections if isinstance(collections, list) else getattr(collections, "collections", [])
        limit = params.top_k if params.top_k > 0 else 10
        results: list[IndexWithScore] = []
        matched = 0
        failed = 0
        for collection in collection_list:
            name = collection if isinstance(collection, str) else getattr(collection, "collection_name", "")
            if not self._matches_collection(name):
                continue
            matched += 1
            try:
                search_result = await self._to_thread(
                    self._db().Collection(name).search_by_text,
                    [query],
                    filter=_base_filter(params),
                    retrieve_vector=False,
                    output_fields=_OUTPUT_FIELDS,
                    limit=limit,
                )
            except Exception as exc:
                failed += 1
                logger.warning("keyword search failed in {}: {}", name, exc)
                continue
            docs = search_result if isinstance(search_result, list) else search_result.get("documents", []) if isinstance(search_result, dict) else []
            for d in docs:
                doc = d if isinstance(d, dict) else {}
                fields = doc.get("fields", doc)
                results.append(IndexWithScore(
                    id=str(doc.get("id", "")),
                    content=str(fields.get(_FIELD_CONTENT, "")),
                    source_id=str(fields.get(_FIELD_SOURCE_ID, "")),
                    source_type=int(fields.get(_FIELD_SOURCE_TYPE, 0)),
                    chunk_id=str(fields.get(_FIELD_CHUNK_ID, "")),
                    knowledge_id=str(fields.get(_FIELD_KNOWLEDGE_ID, "")),
                    knowledge_base_id=str(fields.get(_FIELD_KNOWLEDGE_BASE_ID, "")),
                    tag_id=str(fields.get(_FIELD_TAG_ID, "")),
                    score=float(doc.get("score", 0.0)),
                    match_type=MatchType.KEYWORDS,
                    is_enabled=int(fields.get(_FIELD_IS_ENABLED, 0)) == 1,
                ))
        if matched > 0 and failed == matched:
            raise RuntimeError(
                "tencent vectordb keyword search failed in all matched collections; "
                "ensure collections support text search"
            )
        results.sort(key=lambda r: r.score, reverse=True)
        if len(results) > limit:
            results = results[:limit]
        return _retrieve_result(results, RetrieverType.KEYWORDS)

    # ── ensure_collection ──

    async def _ensure_collection(self, ctx: Context, dimension: int) -> None:
        if dimension in self._initialized:
            return
        try:
            await self._to_thread(self._client.create_database_if_not_exists, self._database_name)
        except Exception as exc:
            raise RuntimeError(f"tencent vectordb ensure database {self._database_name}: {exc}") from exc
        collection_name = self._collection_name(dimension)
        try:
            exists = await self._to_thread(self._db().exists_collection, collection_name)
        except Exception as exc:
            raise RuntimeError(f"tencent vectordb check collection {collection_name}: {exc}") from exc
        if exists:
            self._initialized.add(dimension)
            return
        from tcvectordb.model.enum import FieldType, IndexType, MetricType  # type: ignore[import-untyped]
        from tcvectordb.model.index import (  # type: ignore[import-untyped]
            FilterIndex,
            HNSWParams,
            SparseIndex,
            VectorIndex,
        )
        indexes = [
            VectorIndex(
                name=_FIELD_VECTOR,
                dimension=dimension,
                index_type=IndexType.HNSW,
                metric_type=MetricType.COSINE,
                field_type=FieldType.Vector,
                params=HNSWParams(m=16, efconstruction=200),
            ),
            SparseIndex(name=_FIELD_SPARSE_VECTOR),
            FilterIndex(name=_FIELD_ID, field_type=FieldType.String, index_type=IndexType.PRIMARY_KEY),
            FilterIndex(name=_FIELD_CONTENT, field_type=FieldType.String, index_type=IndexType.FILTER),
            FilterIndex(name=_FIELD_SOURCE_ID, field_type=FieldType.String, index_type=IndexType.FILTER),
            FilterIndex(name=_FIELD_SOURCE_TYPE, field_type=FieldType.Uint64, index_type=IndexType.FILTER),
            FilterIndex(name=_FIELD_CHUNK_ID, field_type=FieldType.String, index_type=IndexType.FILTER),
            FilterIndex(name=_FIELD_KNOWLEDGE_ID, field_type=FieldType.String, index_type=IndexType.FILTER),
            FilterIndex(name=_FIELD_KNOWLEDGE_BASE_ID, field_type=FieldType.String, index_type=IndexType.FILTER),
            FilterIndex(name=_FIELD_TAG_ID, field_type=FieldType.String, index_type=IndexType.FILTER),
            FilterIndex(name=_FIELD_IS_ENABLED, field_type=FieldType.Uint64, index_type=IndexType.FILTER),
        ]
        try:
            await self._to_thread(
                self._db().create_collection,
                collection_name,
                self._shards_num,
                self._replicas_num,
                f"embeddings collection with dimension {dimension}",
                indexes=indexes,
            )
        except Exception as exc:
            msg = str(exc).lower()
            if "code: 15202" in msg or "already exist" in msg:
                logger.info("collection {} already exists, skip create", collection_name)
                self._initialized.add(dimension)
                return
            raise RuntimeError(f"tencent vectordb create collection {collection_name}: {exc}") from exc
        self._initialized.add(dimension)


def new_tencent_vectordb_retrieve_engine_repository(
    client: Any,
    database_name: str,
    index_cfg: IndexConfig | None,
) -> TencentVectorDBRepository:
    """Create a Tencent VectorDB retrieve engine repository.

    Upstream ``NewTencentVectorDBRetrieveEngineRepository``.
    """
    return TencentVectorDBRepository(client, database_name, index_cfg)


__all__ = [
    "TencentVectorDBRepository",
    "new_tencent_vectordb_retrieve_engine_repository",
]
