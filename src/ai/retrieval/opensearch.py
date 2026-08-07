"""OpenSearch k-NN retrieve engine repository.

Maps the upstream OpenSearch contract: per-dimension lazy index creation
(alias-backed ``<base>_<dim>_v1`` -> ``<base>_<dim>``), k-NN vector
retrieval, BM25 keyword retrieval, bulk save with size caps, delete-by-
query, copy-indices, batch update-by-query, and an audit sink for index
provisioning / reindex events.

Uses the ``opensearchpy`` sync client; SDK calls are wrapped in
``asyncio.to_thread`` so the async event loop is not blocked.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import uuid
from collections.abc import Mapping
from typing import Any, cast

from opensearchpy import OpenSearch
from opensearchpy.exceptions import (
    AuthenticationException,
    AuthorizationException,
    NotFoundError,
    TransportError,
)
from opensearchpy.exceptions import (
    ConnectionError as OSConnectionError,
)

from src.ai.retrieval.base import AuditSink, Context, RetrieveEngineRepository
from src.ai.retrieval.types import (
    ConnectionConfig,
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

_DEFAULT_BASE_INDEX: str = "weknora"
_ENV_INDEX_KEY: str = "OPENSEARCH_INDEX"
_COPY_BATCH_SIZE: int = 500
_BULK_MAX_DOCS: int = 1000
_BULK_MAX_BYTES: int = 10 * 1024 * 1024
_MAX_RESULT_WINDOW: int = 10000
_DIM_MAX: int = 16000
_DEFAULT_TOP_K: int = 10

_INDEX_NAME_RE: re.Pattern[str] = re.compile(r"^[a-z0-9][a-z0-9_-]{0,254}$")


# ── Sentinel errors ─────────────────────────────────────────────────


class OpenSearchEngineError(Exception):
    """Base error for OpenSearch engine failures."""


class IndexNotFoundError(OpenSearchEngineError):
    """Alias / underlying index missing."""


class DimensionMismatchError(OpenSearchEngineError):
    """Embedding dimension violates the per-dim invariant."""


class AuthError(OpenSearchEngineError):
    """Cluster returned 401 / 403."""


class TransportEngineError(OpenSearchEngineError):
    """Network / 5xx / opaque cluster error (transient)."""


class VersionUnsupportedError(OpenSearchEngineError):
    """Cluster is not OpenSearch or version is unsupported."""


class ConfigInvalidError(OpenSearchEngineError):
    """IndexConfig / storeID / index-name validation failed."""


class FeatureNotEnabledError(OpenSearchEngineError):
    """Method not yet implemented in this build."""


class BatchTooLargeError(OpenSearchEngineError):
    """Save / Delete batch exceeded the driver's sync cap."""


class CircuitBreakerError(OpenSearchEngineError):
    """k-NN circuit breaker returned 429 (transient)."""


def _is_transient(exc: Exception) -> bool:
    return isinstance(exc, (TransportEngineError, CircuitBreakerError))


def _wrap_transport(exc: Exception) -> Exception:
    """Classify a raw SDK error into a sentinel based on HTTP status."""
    if isinstance(exc, (AuthenticationException, AuthorizationException)):
        return AuthError("authentication failed")
    if isinstance(exc, NotFoundError):
        return exc
    if isinstance(exc, TransportError):
        status = getattr(exc, "status_code", 0)
        if status == 429:
            return CircuitBreakerError("circuit breaker open")
        return TransportEngineError("transport error")
    if isinstance(exc, OSConnectionError):
        return TransportEngineError("transport error")
    return TransportEngineError(str(exc))


def _is_not_found(exc: Exception) -> bool:
    return isinstance(exc, NotFoundError)


def _is_already_exists(exc: Exception) -> bool:
    if isinstance(exc, TransportError):
        status = getattr(exc, "status_code", 0)
        info = getattr(exc, "info", {})
        err_type = info.get("error", {}).get("type", "") if isinstance(info, dict) else ""
        return status == 400 and err_type == "resource_already_exists_exception"
    return False


# ── Internal config ────────────────────────────────────────────────


def _build_internal_cfg(c: IndexConfig | None) -> dict[str, Any]:
    """Project IndexConfig to driver defaults (mirrors the upstream builder)."""
    cfg: dict[str, Any] = {
        "shards": 4,
        "replicas": 1,
        "knn_engine": "lucene",
        "hnsw_m": 16,
        "hnsw_ef_construction": 100,
        "ef_search": 100,
    }
    if c is None:
        return cfg
    if c.number_of_shards > 0:
        cfg["shards"] = c.number_of_shards
    if c.number_of_replicas > 0:
        cfg["replicas"] = c.number_of_replicas
    if c.knn_engine:
        cfg["knn_engine"] = c.knn_engine
    if c.hnsw_m > 0:
        cfg["hnsw_m"] = c.hnsw_m
    if c.hnsw_ef_construction > 0:
        cfg["hnsw_ef_construction"] = c.hnsw_ef_construction
    if c.hnsw_ef_search > 0:
        cfg["ef_search"] = c.hnsw_ef_search
    return cfg


# ── Mapping / query builders (pure functions) ───────────────────────


def _build_index_mapping(cfg: dict[str, Any], dim: int) -> dict[str, Any]:
    return {
        "settings": {
            "index": {
                "knn": True,
                "number_of_shards": cfg.get("shards", 4),
                "number_of_replicas": cfg.get("replicas", 1),
                "refresh_interval": "1s",
                "knn.algo_param.ef_search": cfg.get("ef_search", 100),
            }
        },
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": cfg.get("knn_engine", "lucene"),
                        "parameters": {
                            "m": cfg.get("hnsw_m", 16),
                            "ef_construction": cfg.get("hnsw_ef_construction", 100),
                        },
                    },
                },
                "content": {"type": "text", "analyzer": "standard"},
                "chunk_id": {"type": "keyword"},
                "knowledge_id": {"type": "keyword"},
                "knowledge_base_id": {"type": "keyword"},
                "tag_id": {"type": "keyword"},
                "source_id": {"type": "keyword"},
                "source_type": {"type": "integer"},
                "is_enabled": {"type": "boolean"},
                "is_recommended": {"type": "boolean"},
            }
        },
    }


def _build_keywords_mapping(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "settings": {
            "index": {
                "number_of_shards": cfg.get("shards", 4),
                "number_of_replicas": cfg.get("replicas", 1),
                "refresh_interval": "1s",
            }
        },
        "mappings": {
            "properties": {
                "content": {"type": "text", "analyzer": "standard"},
                "chunk_id": {"type": "keyword"},
                "knowledge_id": {"type": "keyword"},
                "knowledge_base_id": {"type": "keyword"},
                "tag_id": {"type": "keyword"},
                "source_id": {"type": "keyword"},
                "source_type": {"type": "integer"},
                "is_enabled": {"type": "boolean"},
                "is_recommended": {"type": "boolean"},
            }
        },
    }


def _build_filter_must(params: RetrieveParams) -> list[dict[str, Any]]:
    must: list[dict[str, Any]] = []
    if params.knowledge_base_ids:
        must.append({"terms": {"knowledge_base_id": list(params.knowledge_base_ids)}})
    if params.knowledge_ids:
        must.append({"terms": {"knowledge_id": list(params.knowledge_ids)}})
    if params.tag_ids:
        must.append({"terms": {"tag_id": list(params.tag_ids)}})
    if params.exclude_chunk_ids:
        must.append({"bool": {"must_not": {"terms": {"chunk_id": list(params.exclude_chunk_ids)}}}})
    if params.exclude_knowledge_ids:
        must.append({"bool": {"must_not": {"terms": {"knowledge_id": list(params.exclude_knowledge_ids)}}}})
    must.append({"term": {"is_enabled": True}})
    return must


def _build_knn_query(
    embedding: list[float], top_k: int, threshold: float, must: list[dict[str, Any]]
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": list(embedding),
                    "k": top_k,
                    "filter": {"bool": {"must": must}},
                }
            }
        },
    }
    if threshold > 0:
        body["min_score"] = threshold
    return body


def _build_keyword_query(
    query_text: str, top_k: int, threshold: float, must: list[dict[str, Any]]
) -> dict[str, Any]:
    must = [*must, {"match": {"content": query_text}}]
    body: dict[str, Any] = {
        "size": top_k,
        "query": {"bool": {"must": must}},
    }
    if threshold > 0:
        body["min_score"] = threshold
    return body


# ── Audit ───────────────────────────────────────────────────────────


class _NopAuditSink(AuditSink):
    """Null-object audit sink (no-op when no sink is configured)."""

    async def emit_index_created(self, ctx: Context, alias: str, dim: int) -> None:
        pass

    async def emit_reindex_executed(
        self, ctx: Context, src_alias: str, dst_alias: str, docs: int
    ) -> None:
        pass


# ── Document helpers ────────────────────────────────────────────────


def _to_doc(info: IndexInfo, emb: list[float] | None, enabled: bool) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "chunk_id": info.chunk_id,
        "knowledge_id": info.knowledge_id,
        "knowledge_base_id": info.knowledge_base_id,
        "source_id": info.source_id,
        "source_type": int(info.source_type),
        "tag_id": info.tag_id,
        "content": info.content,
        "is_enabled": enabled,
        "is_recommended": info.is_recommended,
    }
    if emb:
        doc["embedding"] = list(emb)
    return doc


def _lookup_embedding(params: IndexSaveParams, source_id: str) -> list[float] | None:
    if not params:
        return None
    emb_map = params.get("embedding")
    if isinstance(emb_map, dict):
        vec = emb_map.get(source_id)
        if vec:
            return list(vec)
    return None


def _lookup_chunk_enabled(params: IndexSaveParams, chunk_id: str, default: bool) -> bool:
    if not params:
        return default
    en_map = params.get("chunk_enabled")
    if isinstance(en_map, dict) and chunk_id in en_map:
        return bool(en_map[chunk_id])
    return default


def _transform_source_id(source_id: str, chunk_id: str, target_chunk_id: str) -> str:
    if source_id == chunk_id:
        return target_chunk_id
    prefix = f"{chunk_id}-"
    if source_id.startswith(prefix):
        return f"{target_chunk_id}-{source_id[len(prefix):]}"
    return str(uuid.uuid4())


# ── Repository ──────────────────────────────────────────────────────


class OpenSearchRepository(RetrieveEngineRepository):
    """OpenSearch k-NN engine repository with per-dimension lazy index init."""

    def __init__(
        self,
        client: OpenSearch,
        base_index: str,
        cfg: dict[str, Any],
        audit_sink: AuditSink | None = None,
    ) -> None:
        self._client = client
        self._base_index = base_index
        self._cfg = cfg
        self._audit = audit_sink or _NopAuditSink()
        self._init_lock = asyncio.Lock()
        self._dims_ready: set[int] = set()
        self._dim_errs: dict[int, Exception | None] = {}
        self._keywords_ready = False
        self._keywords_err: Exception | None = None

    def _index_alias(self, dim: int) -> str:
        return f"{self._base_index}_{dim}"

    def _keywords_index(self) -> str:
        return f"{self._base_index}_keywords"

    # ── lazy init ────────────────────────────────────────────────────

    async def _ensure_ready(self, ctx: Context, dim: int) -> None:
        if dim <= 0 or dim > _DIM_MAX:
            raise DimensionMismatchError(f"dim {dim} out of range (1..{_DIM_MAX})")
        if dim in self._dims_ready:
            err = self._dim_errs.get(dim)
            if err is not None:
                raise err
            return
        async with self._init_lock:
            if dim in self._dims_ready:
                err = self._dim_errs.get(dim)
                if err is not None:
                    raise err
                return
            try:
                await self._create_index_and_alias(ctx, dim)
                self._dims_ready.add(dim)
                self._dim_errs[dim] = None
            except Exception as exc:
                if not _is_transient(exc):
                    self._dims_ready.add(dim)
                    self._dim_errs[dim] = exc
                raise

    async def _ensure_keywords_index(self, ctx: Context) -> None:
        if self._keywords_ready:
            if self._keywords_err:
                raise self._keywords_err
            return
        async with self._init_lock:
            if self._keywords_ready:
                if self._keywords_err:
                    raise self._keywords_err
                return
            try:
                name = self._keywords_index()
                exists = await self._alias_exists(ctx, name)
                if not exists:
                    body = _build_keywords_mapping(self._cfg)
                    await asyncio.to_thread(
                        self._client.indices.create, index=name, body=body
                    )
                self._keywords_ready = True
                self._keywords_err = None
                await self._audit.emit_index_created(ctx, name, 0)
            except Exception as exc:
                if not _is_transient(exc):
                    self._keywords_ready = True
                    self._keywords_err = exc
                raise

    async def _create_index_and_alias(self, ctx: Context, dim: int) -> None:
        alias = self._index_alias(dim)
        real_index = f"{alias}_v1"
        exists = await self._alias_exists(ctx, alias)
        if exists:
            return
        body = _build_index_mapping(self._cfg, dim)
        try:
            await asyncio.to_thread(
                self._client.indices.create, index=real_index, body=body
            )
            index_created = True
        except Exception as exc:
            if _is_already_exists(exc):
                index_created = False
            else:
                if isinstance(exc, OpenSearchEngineError):
                    raise exc
                raise _wrap_transport(exc) from exc
        else:
            index_created = True
        try:
            await asyncio.to_thread(
                self._client.indices.put_alias, index=real_index, name=alias
            )
        except Exception as exc:
            if index_created:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(self._client.indices.delete, index=real_index)
            if isinstance(exc, OpenSearchEngineError):
                raise exc
            raise _wrap_transport(exc) from exc
        if index_created:
            await self._audit.emit_index_created(ctx, alias, dim)

    async def _alias_exists(self, ctx: Context, alias: str) -> bool:
        del ctx
        try:
            await asyncio.to_thread(self._client.indices.exists_alias, name=alias)
            return True
        except NotFoundError:
            return False
        except Exception as exc:
            raise _wrap_transport(exc) from exc

    # ── protocol: engine_type / support ───────────────────────────────

    def engine_type(self) -> RetrieverEngineType:
        return RetrieverEngineType.OPENSEARCH

    def support(self) -> list[RetrieverType]:
        return [RetrieverType.KEYWORDS, RetrieverType.VECTOR]

    # ── protocol: estimate_storage_size ───────────────────────────────

    def estimate_storage_size(
        self,
        ctx: Context,
        index_info_list: list[IndexInfo],
        params: IndexSaveParams,
    ) -> int:
        del ctx, params
        if not index_info_list:
            return 0
        content_bytes = 1024
        emb_dim_guess = 768
        hnsw_overhead = 128
        return len(index_info_list) * (content_bytes + 4 * emb_dim_guess + hnsw_overhead)

    # ── protocol: save / batch_save ───────────────────────────────────

    async def save(
        self, ctx: Context, index_info: IndexInfo, params: IndexSaveParams
    ) -> None:
        emb = _lookup_embedding(params, index_info.source_id)
        enabled = _lookup_chunk_enabled(params, index_info.chunk_id, index_info.is_enabled)
        dim = len(emb) if emb else 0
        if dim > 0:
            await self._ensure_ready(ctx, dim)
            target = self._index_alias(dim)
        else:
            await self._ensure_keywords_index(ctx)
            target = self._keywords_index()
        doc = _to_doc(index_info, emb, enabled)
        await asyncio.to_thread(
            self._client.index, index=target, id=index_info.chunk_id, body=doc
        )

    async def batch_save(
        self,
        ctx: Context,
        index_info_list: list[IndexInfo],
        params: IndexSaveParams,
    ) -> None:
        if not index_info_list:
            return
        embs, dim = _extract_batch_embeddings(params, index_info_list)
        if dim > 0:
            estimated = len(index_info_list) * (100 + dim * 5 + 1024)
            if estimated > _BULK_MAX_BYTES:
                raise BatchTooLargeError(
                    f"estimated bulk body {estimated}B exceeds {_BULK_MAX_BYTES}B cap"
                )
        if len(index_info_list) > _BULK_MAX_DOCS:
            raise BatchTooLargeError(
                f"bulk n={len(index_info_list)} exceeds {_BULK_MAX_DOCS}-doc cap"
            )
        if dim > 0:
            await self._ensure_ready(ctx, dim)
            alias = self._index_alias(dim)
        else:
            await self._ensure_keywords_index(ctx)
            alias = self._keywords_index()
        body = _build_bulk_body(alias, index_info_list, embs, params)
        await asyncio.to_thread(self._client.bulk, body=body)

    # ── protocol: delete_by_* ─────────────────────────────────────────

    async def delete_by_chunk_id_list(
        self, ctx: Context, index_id_list: list[str], dimension: int, knowledge_type: str
    ) -> None:
        del knowledge_type
        await self._delete_by_list(ctx, index_id_list, dimension, "chunk_id")

    async def delete_by_source_id_list(
        self, ctx: Context, source_id_list: list[str], dimension: int, knowledge_type: str
    ) -> None:
        del knowledge_type
        await self._delete_by_list(ctx, source_id_list, dimension, "source_id")

    async def delete_by_knowledge_id_list(
        self, ctx: Context, knowledge_id_list: list[str], dimension: int, knowledge_type: str
    ) -> None:
        del knowledge_type
        await self._delete_by_list(ctx, knowledge_id_list, dimension, "knowledge_id")

    async def _delete_by_list(
        self, ctx: Context, ids: list[str], dim: int, field: str
    ) -> None:
        if not ids:
            return
        if len(ids) > _BULK_MAX_DOCS:
            raise BatchTooLargeError(f"{field}-delete batch {len(ids)} > {_BULK_MAX_DOCS} cap")
        if dim == 0:
            await self._ensure_keywords_index(ctx)
            index = self._keywords_index()
        else:
            await self._ensure_ready(ctx, dim)
            index = self._index_alias(dim)
        body = {"query": {"terms": {field: ids}}}
        await asyncio.to_thread(
            self._client.delete_by_query, index=index, body=body, refresh=True
        )

    # ── protocol: retrieve ───────────────────────────────────────────

    async def retrieve(
        self, ctx: Context, params: RetrieveParams
    ) -> list[RetrieveResult]:
        dim, multi_index = _resolve_dim(params)
        if params.retriever_type == RetrieverType.VECTOR:
            if dim == 0:
                raise DimensionMismatchError("vector retrieve requires embedding or dim")
            await self._ensure_ready(ctx, dim)
            return await self._vector_retrieve(ctx, params, dim)
        if params.retriever_type == RetrieverType.KEYWORDS:
            if multi_index:
                return await self._keywords_retrieve(ctx, params, f"{self._base_index}_*")
            await self._ensure_ready(ctx, dim) if dim > 0 else await self._ensure_keywords_index(ctx)
            index = self._index_alias(dim) if dim > 0 else self._keywords_index()
            return await self._keywords_retrieve(ctx, params, index)
        raise ValueError(f"unsupported retriever type: {params.retriever_type}")

    async def _vector_retrieve(
        self, ctx: Context, params: RetrieveParams, dim: int
    ) -> list[RetrieveResult]:
        del ctx
        top_k = _effective_top_k(params)
        must = _build_filter_must(params)
        body = _build_knn_query(list(params.embedding), top_k, params.threshold, must)
        hits = await self._search(self._index_alias(dim), body)
        return [_wrap_results(hits, RetrieverType.VECTOR, MatchType.EMBEDDING)]

    async def _keywords_retrieve(
        self, ctx: Context, params: RetrieveParams, index_pattern: str
    ) -> list[RetrieveResult]:
        del ctx
        top_k = _effective_top_k(params)
        must = _build_filter_must(params)
        body = _build_keyword_query(params.query, top_k, params.threshold, must)
        hits = await self._search(index_pattern, body)
        return [_wrap_results(hits, RetrieverType.KEYWORDS, MatchType.KEYWORDS)]

    async def _search(self, index_pattern: str, body: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            response = await asyncio.to_thread(
                self._client.search, index=index_pattern, body=body
            )
        except NotFoundError as exc:
            raise IndexNotFoundError(f"index {index_pattern} missing") from exc
        except Exception as exc:
            raise _wrap_transport(exc) from exc
        hits_obj = cast(dict[str, Any], response).get("hits", {})
        return cast(list[dict[str, Any]], hits_obj.get("hits", []))

    # ── protocol: copy_indices ───────────────────────────────────────

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
        if dimension <= 0:
            raise DimensionMismatchError(f"CopyIndices requires dim > 0, got {dimension}")
        await self._ensure_ready(ctx, dimension)
        alias = self._index_alias(dimension)
        total = 0
        for from_val in range(0, _MAX_RESULT_WINDOW, _COPY_BATCH_SIZE):
            docs = await self._copy_scan_batch(alias, source_knowledge_base_id, from_val, _COPY_BATCH_SIZE)
            if not docs:
                break
            infos, emb_map, enabled_map = _process_copy_batch(
                docs, source_to_target_kb_id_map, source_to_target_chunk_id_map,
                target_knowledge_base_id, knowledge_type,
            )
            if infos:
                params: IndexSaveParams = {"embedding": emb_map}
                params["chunk_enabled"] = enabled_map  # type: ignore[assignment]
                await self.batch_save(ctx, infos, params)
                total += len(infos)
            if len(docs) < _COPY_BATCH_SIZE:
                break
        await self._audit.emit_reindex_executed(ctx, alias, alias, total)

    async def _copy_scan_batch(
        self, index: str, source_kb: str, from_val: int, size: int
    ) -> list[dict[str, Any]]:
        body = {
            "from": from_val,
            "size": size,
            "query": {"bool": {"filter": [{"term": {"knowledge_base_id": source_kb}}]}},
        }
        try:
            response = await asyncio.to_thread(
                self._client.search, index=index, body=body
            )
        except NotFoundError as exc:
            raise IndexNotFoundError(f"index {index} missing") from exc
        except Exception as exc:
            raise _wrap_transport(exc) from exc
        hits_obj = response.get("hits", {})
        return [h.get("_source", {}) for h in hits_obj.get("hits", []) if isinstance(h.get("_source"), dict)]

    # ── protocol: batch_update_chunk_* ───────────────────────────────

    async def batch_update_chunk_enabled_status(
        self, ctx: Context, chunk_status_map: Mapping[str, bool]
    ) -> None:
        del ctx
        if not chunk_status_map:
            return
        groups: dict[bool, list[str]] = {}
        for chunk_id, enabled in chunk_status_map.items():
            groups.setdefault(enabled, []).append(chunk_id)
        for enabled in (False, True):
            ids = groups.get(enabled, [])
            if not ids:
                continue
            await self._update_by_query(
                sorted(ids),
                "ctx._source.is_enabled = params.v",
                {"v": enabled},
            )

    async def batch_update_chunk_tag_id(
        self, ctx: Context, chunk_tag_map: Mapping[str, str]
    ) -> None:
        del ctx
        if not chunk_tag_map:
            return
        tag_groups: dict[str, list[str]] = {}
        for chunk_id, tag_id in chunk_tag_map.items():
            tag_groups.setdefault(tag_id, []).append(chunk_id)
        for tag_id in sorted(tag_groups):
            await self._update_by_query(
                sorted(tag_groups[tag_id]),
                "ctx._source.tag_id = params.v",
                {"v": tag_id},
            )

    async def _update_by_query(
        self, chunk_ids: list[str], script_source: str, params: dict[str, Any]
    ) -> None:
        body: dict[str, Any] = {
            "query": {"terms": {"chunk_id": chunk_ids}},
            "script": {"lang": "painless", "source": script_source, "params": params},
        }
        await asyncio.to_thread(
            self._client.update_by_query,
            index=f"{self._base_index}_*", body=body, refresh=True,
        )


# ── Helpers ─────────────────────────────────────────────────────────


def _resolve_dim(params: RetrieveParams) -> tuple[int, bool]:
    """Return (dim, multi_index). multi_index=True means dim is unknown."""
    if params.additional_params:
        v = params.additional_params.get("dim")
        if isinstance(v, int) and v > 0:
            return v, False
    if params.embedding:
        return len(params.embedding), False
    return 0, True


def _effective_top_k(params: RetrieveParams) -> int:
    if params.top_k <= 0:
        return _DEFAULT_TOP_K
    return min(params.top_k, _MAX_RESULT_WINDOW)


def _wrap_results(
    hits: list[dict[str, Any]], rt: RetrieverType, mt: MatchType
) -> RetrieveResult:
    results: list[IndexWithScore] = []
    for hit in hits:
        doc_id = hit.get("_id", "")
        score = hit.get("_score", 0.0)
        source = hit.get("_source", {})
        if not isinstance(source, dict):
            continue
        results.append(IndexWithScore(
            id=doc_id,
            chunk_id=source.get("chunk_id", ""),
            knowledge_id=source.get("knowledge_id", ""),
            knowledge_base_id=source.get("knowledge_base_id", ""),
            source_id=source.get("source_id", ""),
            source_type=SourceType(source.get("source_type", 0)),
            tag_id=source.get("tag_id", ""),
            content=source.get("content", ""),
            score=score,
            match_type=mt,
            is_enabled=source.get("is_enabled", False),
        ))
    return RetrieveResult(
        results=results,
        retriever_engine_type=RetrieverEngineType.OPENSEARCH,
        retriever_type=rt,
    )


def _extract_batch_embeddings(
    params: IndexSaveParams, infos: list[IndexInfo]
) -> tuple[list[list[float] | None], int]:
    out: list[list[float] | None] = []
    dim = 0
    for info in infos:
        emb = _lookup_embedding(params, info.source_id)
        out.append(emb)
        if emb:
            if dim == 0:
                dim = len(emb)
            elif len(emb) != dim:
                raise DimensionMismatchError(
                    f"embedding[{info.source_id}] dim={len(emb)} != first non-empty dim={dim}"
                )
    return out, dim


def _build_bulk_body(
    alias: str,
    infos: list[IndexInfo],
    embs: list[list[float] | None],
    params: IndexSaveParams,
) -> list[dict[str, Any]]:
    body: list[dict[str, Any]] = []
    for i, info in enumerate(infos):
        emb = embs[i] if i < len(embs) else None
        enabled = _lookup_chunk_enabled(params, info.chunk_id, info.is_enabled)
        body.append({"index": {"_index": alias, "_id": info.chunk_id}})
        body.append(_to_doc(info, emb, enabled))
    return body


def _process_copy_batch(
    docs: list[dict[str, Any]],
    source_to_target_kb_id_map: Mapping[str, str],
    source_to_target_chunk_id_map: Mapping[str, str],
    target_knowledge_base_id: str,
    knowledge_type: str,
) -> tuple[list[IndexInfo], dict[str, list[float]], dict[str, bool]]:
    infos: list[IndexInfo] = []
    emb_map: dict[str, list[float]] = {}
    enabled_map: dict[str, bool] = {}
    for d in docs:
        source_chunk_id = d.get("chunk_id", "")
        target_chunk_id = source_to_target_chunk_id_map.get(source_chunk_id)
        if not target_chunk_id:
            continue
        source_knowledge_id = d.get("knowledge_id", "")
        target_knowledge_id = source_to_target_kb_id_map.get(source_knowledge_id)
        if not target_knowledge_id:
            continue
        target_source_id = _transform_source_id(
            d.get("source_id", ""), source_chunk_id, target_chunk_id
        )
        emb = d.get("embedding")
        if emb:
            emb_map[target_source_id] = list(emb)
        enabled_map[target_chunk_id] = d.get("is_enabled", True)
        infos.append(IndexInfo(
            content=d.get("content", ""),
            source_id=target_source_id,
            source_type=SourceType(d.get("source_type", 0)),
            chunk_id=target_chunk_id,
            knowledge_id=target_knowledge_id,
            knowledge_base_id=target_knowledge_base_id,
            knowledge_type=knowledge_type,
            tag_id=d.get("tag_id", ""),
            is_enabled=d.get("is_enabled", True),
            is_recommended=d.get("is_recommended", False),
        ))
    return infos, emb_map, enabled_map


# ── Client + repository construction ───────────────────────────────


def new_opensearch_client(cc: ConnectionConfig) -> OpenSearch:
    """Build an ``opensearchpy`` sync client from a connection config."""
    if not cc.addr:
        raise ConfigInvalidError("ConnectionConfig.Addr required")
    http_auth = (cc.username, cc.password) if cc.username else None
    return OpenSearch(
        hosts=[cc.addr],
        http_auth=http_auth,
        verify_certs=not cc.insecure_skip_verify,
        ssl_assert_hostname=not cc.insecure_skip_verify,
    )


def _resolve_base_index(store_id: str, index_config: IndexConfig | None) -> str:
    """Resolve the base index name, folding the store ID when present."""
    base = index_config.index_name if index_config and index_config.index_name else ""
    if not base:
        base = os.getenv(_ENV_INDEX_KEY, _DEFAULT_BASE_INDEX)
    if store_id:
        if len(store_id) < 16:
            raise ConfigInvalidError(
                f"storeID must be empty or >=16 chars, got {len(store_id)}"
            )
        base = f"{base}_{store_id[:12]}"
    if not _INDEX_NAME_RE.match(base):
        raise ConfigInvalidError(f"invalid index base name: {base!r}")
    return base


async def new_opensearch_repository(
    cc: ConnectionConfig,
    audit_sink: AuditSink | None,
    store_id: str,
    index_config: IndexConfig | None,
) -> OpenSearchRepository:
    """Construct an OpenSearch repository from connection params."""
    client = new_opensearch_client(cc)
    base = _resolve_base_index(store_id, index_config)
    cfg = _build_internal_cfg(index_config)
    return OpenSearchRepository(client, base, cfg, audit_sink)


__all__ = [
    "AuthError",
    "BatchTooLargeError",
    "CircuitBreakerError",
    "ConfigInvalidError",
    "DimensionMismatchError",
    "FeatureNotEnabledError",
    "IndexNotFoundError",
    "OpenSearchEngineError",
    "OpenSearchRepository",
    "TransportEngineError",
    "VersionUnsupportedError",
    "new_opensearch_client",
    "new_opensearch_repository",
]
