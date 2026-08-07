"""Doris retrieve engine repository (upstream ``doris/repository.go``).

Communicates with Apache Doris 4.1 over the MySQL protocol (pymysql) for
DDL / DML and over HTTP (Stream Load) for legacy partial updates. Tables
are sharded by embedding dimension: ``<tableBaseName>_<dim>``.

Two compatibility modes mirror the upstream contract:

* ``inner_product_duplicate`` (default): ``DUPLICATE KEY(id)`` tables with
  normalized inner-product ANN. Writes use delete + insert to preserve
  replace semantics.
* ``legacy``: ``UNIQUE KEY(id)`` tables with ``cosine_distance`` ANN.
  Writes use Stream Load partial updates.

The mode is resolved lazily on first use and cached for the process.
pymysql is synchronous, so every DB call is dispatched through
``asyncio.to_thread`` under a per-instance lock.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast
from urllib.parse import quote

import pymysql  # type: ignore[import-untyped]

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
)
from src.app_logging import logger
from src.common.exception import StorageBackendError, ValidationError

# ── Constants ────────────────────────────────────────────────────────

_DEFAULT_TABLE_BASE_NAME = "weknora_embeddings"
_ENV_DORIS_TABLE_PREFIX = "DORIS_TABLE_PREFIX"
_ENV_DORIS_COMPAT_MODE = "DORIS_COMPAT_MODE"

_DEFAULT_BUCKETS_NUM = 10
_DEFAULT_REPLICATION_NUM = 1

_FIELD_ID = "id"
_FIELD_CONTENT = "content"
_FIELD_SOURCE_ID = "source_id"
_FIELD_SOURCE_TYPE = "source_type"
_FIELD_CHUNK_ID = "chunk_id"
_FIELD_KNOWLEDGE_ID = "knowledge_id"
_FIELD_KNOWLEDGE_BASE_ID = "knowledge_base_id"
_FIELD_TAG_ID = "tag_id"
_FIELD_IS_ENABLED = "is_enabled"
_FIELD_EMBEDDING = "embedding"

_COLUMNS = [
    _FIELD_ID, _FIELD_CONTENT, _FIELD_SOURCE_ID, _FIELD_SOURCE_TYPE,
    _FIELD_CHUNK_ID, _FIELD_KNOWLEDGE_ID, _FIELD_KNOWLEDGE_BASE_ID,
    _FIELD_TAG_ID, _FIELD_IS_ENABLED, _FIELD_EMBEDDING,
]
_COLUMNS_FOR_RETRIEVE = [
    _FIELD_ID, _FIELD_CONTENT, _FIELD_SOURCE_ID, _FIELD_SOURCE_TYPE,
    _FIELD_CHUNK_ID, _FIELD_KNOWLEDGE_ID, _FIELD_KNOWLEDGE_BASE_ID,
    _FIELD_TAG_ID, _FIELD_IS_ENABLED,
]
_COLUMNS_FOR_COPY = [
    _FIELD_ID, _FIELD_CONTENT, _FIELD_SOURCE_ID, _FIELD_SOURCE_TYPE,
    _FIELD_CHUNK_ID, _FIELD_KNOWLEDGE_ID, _FIELD_KNOWLEDGE_BASE_ID,
    _FIELD_TAG_ID, _FIELD_IS_ENABLED, _FIELD_EMBEDDING,
]

CompatMode = Literal["auto", "legacy", "inner_product_duplicate"]

_COMPAT_MODE_AUTO: CompatMode = "auto"
_COMPAT_MODE_LEGACY: CompatMode = "legacy"
_COMPAT_MODE_INNER_PRODUCT_DUPLICATE: CompatMode = "inner_product_duplicate"


# ── Helpers ─────────────────────────────────────────────────────────


def _resolve_collection_name(index_cfg: IndexConfig | None, env_key: str, default_val: str) -> str:
    """Resolve the collection/table base name (upstream ``ResolveCollectionName``)."""
    if index_cfg is not None:
        if index_cfg.collection_prefix != "":
            return index_cfg.collection_prefix
        if index_cfg.collection_name != "":
            return index_cfg.collection_name
    env_val = os.getenv(env_key, "")
    if env_val != "":
        return env_val
    return default_val


def _get_buckets_num(cfg: IndexConfig | None, default: int) -> int:
    if cfg is not None and cfg.buckets_num > 0:
        return cfg.buckets_num
    return default


def _get_replication_num(cfg: IndexConfig | None, default: int) -> int:
    if cfg is not None and cfg.replication_num > 0:
        return cfg.replication_num
    return default


def _resolve_configured_compat_mode() -> tuple[str, str]:
    """Parse ``DORIS_COMPAT_MODE`` env var; returns (mode, invalid_raw)."""
    raw = os.getenv(_ENV_DORIS_COMPAT_MODE, "").strip()
    if raw == "":
        return _COMPAT_MODE_AUTO, ""
    lowered = raw.lower()
    if lowered == _COMPAT_MODE_AUTO:
        return _COMPAT_MODE_AUTO, ""
    if lowered == _COMPAT_MODE_LEGACY:
        return _COMPAT_MODE_LEGACY, ""
    if lowered in (
        _COMPAT_MODE_INNER_PRODUCT_DUPLICATE,
        "inner-product-duplicate",
        "inner_product",
        "inner-product",
    ):
        return _COMPAT_MODE_INNER_PRODUCT_DUPLICATE, ""
    return _COMPAT_MODE_AUTO, raw


def _normalize_embeddings(mode: CompatMode) -> bool:
    return mode != _COMPAT_MODE_LEGACY


def _uses_replace_write(mode: CompatMode) -> bool:
    return mode != _COMPAT_MODE_LEGACY


def _uses_rewrite_chunk_updates(mode: CompatMode) -> bool:
    return mode != _COMPAT_MODE_LEGACY


def _embedding_literal(vec: list[float]) -> str:
    """Serialize a float list to a Doris ``ARRAY<FLOAT>`` literal."""
    if not vec:
        return "[]"
    return "[" + ",".join(repr(float(v)) for v in vec) + "]"


def _parse_embedding_literal(raw: str | bytes) -> list[float]:
    """Parse a Doris ``ARRAY<FLOAT>`` literal (``"[1,2,3]"``)."""
    s = raw.decode() if isinstance(raw, bytes) else raw
    s = s.strip()
    if s == "":
        return []
    s = s.removeprefix("[").removesuffix("]")
    if s == "":
        return []
    return [float(p.strip()) for p in s.split(",") if p.strip() != ""]


def _validate_embedding(vec: list[float]) -> None:
    """Raise ``ValidationError`` if any element is NaN or infinite."""
    for i, v in enumerate(vec):
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            raise ValidationError(
                code="doris.embedding_not_finite",
                message=f"embedding[{i}] is not finite: {f}",
            )


def _normalize_embedding(vec: list[float]) -> list[float]:
    """Return a unit-length copy so inner-product ANN preserves cosine semantics."""
    if not vec:
        return []
    sum_squares = sum(float(v) * float(v) for v in vec)
    if sum_squares == 0:
        return list(vec)
    norm = math.sqrt(sum_squares)
    return [float(v) / norm for v in vec]


def _translate_source_id(original: str, source_chunk_id: str, target_chunk_id: str) -> str:
    """Translate a source ID to the target chunk (upstream ``translateSourceID``)."""
    if original == source_chunk_id:
        return target_chunk_id
    if original.startswith(source_chunk_id + "-"):
        question_id = original.removeprefix(source_chunk_id + "-")
        return f"{target_chunk_id}-{question_id}"
    return str(uuid.uuid4())


def _dedupe_rows_by_id(rows: list[DorisVectorEmbedding]) -> list[DorisVectorEmbedding]:
    """Last-wins dedup by ``id`` (upstream ``dedupeRowsByID``)."""
    if len(rows) < 2:
        return list(rows)
    out: list[DorisVectorEmbedding] = []
    positions: dict[str, int] = {}
    for row in rows:
        if row.id in positions:
            out[positions[row.id]] = row
            continue
        positions[row.id] = len(out)
        out.append(row)
    return out


# ── Domain model ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DorisVectorEmbedding:
    """One row in a Doris embedding table (upstream ``DorisVectorEmbedding``)."""

    id: str = ""
    content: str = ""
    source_id: str = ""
    source_type: int = 0
    chunk_id: str = ""
    knowledge_id: str = ""
    knowledge_base_id: str = ""
    tag_id: str = ""
    is_enabled: bool = False
    embedding: list[float] = field(default_factory=list)


def _to_doris_vector_embedding(
    info: IndexInfo,
    params: IndexSaveParams,
    compat_mode: CompatMode,
) -> DorisVectorEmbedding:
    """Build a Doris row from an IndexInfo + embedding map (upstream ``toDorisVectorEmbedding``)."""
    emb: list[float] = []
    embedding_map = params.get(_FIELD_EMBEDDING) if params else None
    if isinstance(embedding_map, dict):
        raw = embedding_map.get(info.source_id)
        if raw is not None:
            emb = [float(v) for v in raw]
            if _normalize_embeddings(compat_mode):
                emb = _normalize_embedding(emb)
    return DorisVectorEmbedding(
        id=info.id,
        content=info.content,
        source_id=info.source_id,
        source_type=int(info.source_type),
        chunk_id=info.chunk_id,
        knowledge_id=info.knowledge_id,
        knowledge_base_id=info.knowledge_base_id,
        tag_id=info.tag_id,
        is_enabled=info.is_enabled,
        embedding=emb,
    )


def _calculate_storage_size(emb: DorisVectorEmbedding) -> int:
    """Estimate one row's storage cost (upstream ``calculateStorageSize``)."""
    payload = (
        len(emb.content) + len(emb.source_id) + len(emb.chunk_id)
        + len(emb.knowledge_id) + len(emb.knowledge_base_id) + len(emb.tag_id) + 8
    )
    vec_bytes = len(emb.embedding) * 4 if emb.embedding else 0
    hnsw_bytes = 32 * 2 * 8 if emb.embedding else 0  # max_degree=32
    return payload + vec_bytes + hnsw_bytes + 24


# ── WHERE builder ───────────────────────────────────────────────────


@dataclass
class _WhereBuilder:
    conds: list[tuple[str, list[Any]]] = field(default_factory=list)

    def add_equal(self, field_name: str, value: Any) -> None:
        self.conds.append((f"{field_name} = %s", [value]))

    def add_in(self, field_name: str, values: list[str]) -> None:
        if not values:
            return
        placeholders = ", ".join(["%s"] * len(values))
        self.conds.append((f"{field_name} IN ({placeholders})", list(values)))

    def add_not_in(self, field_name: str, values: list[str]) -> None:
        if not values:
            return
        placeholders = ", ".join(["%s"] * len(values))
        self.conds.append((f"{field_name} NOT IN ({placeholders})", list(values)))

    def build(self) -> tuple[str, list[Any]]:
        if not self.conds:
            return "1 = 1", []
        parts: list[str] = []
        args: list[Any] = []
        for clause, clause_args in self.conds:
            parts.append(clause)
            args.extend(clause_args)
        return " AND ".join(parts), args


def _build_base_filter(params: RetrieveParams) -> _WhereBuilder:
    wb = _WhereBuilder()
    wb.add_equal(_FIELD_IS_ENABLED, True)
    if params.knowledge_base_ids:
        wb.add_in(_FIELD_KNOWLEDGE_BASE_ID, list(params.knowledge_base_ids))
    if params.knowledge_ids:
        wb.add_in(_FIELD_KNOWLEDGE_ID, list(params.knowledge_ids))
    if params.tag_ids:
        wb.add_in(_FIELD_TAG_ID, list(params.tag_ids))
    if params.exclude_knowledge_ids:
        wb.add_not_in(_FIELD_KNOWLEDGE_ID, list(params.exclude_knowledge_ids))
    if params.exclude_chunk_ids:
        wb.add_not_in(_FIELD_CHUNK_ID, list(params.exclude_chunk_ids))
    return wb


def _build_retrieve_result(
    results: list[Any], retriever_type: RetrieverType
) -> list[RetrieveResult]:
    return [RetrieveResult(
        results=results,
        retriever_engine_type=RetrieverEngineType.DORIS,
        retriever_type=retriever_type,
        error=None,
    )]


# ── DDL ─────────────────────────────────────────────────────────────


def _build_create_table_ddl(
    table_name: str,
    dimension: int,
    buckets: int,
    replication: int,
    compat_mode: CompatMode,
) -> str:
    metric_type = "inner_product"
    key_mode = "DUPLICATE KEY(id)"
    properties = f'\t"replication_num"="{replication}"'
    if compat_mode == _COMPAT_MODE_LEGACY:
        metric_type = "cosine_distance"
        key_mode = "UNIQUE KEY(id)"
        properties = (
            f'\t"replication_num"="{replication}",\n'
            f'\t"enable_unique_key_merge_on_write"="true"'
        )
    return (
        f"CREATE TABLE IF NOT EXISTS `{table_name}` (\n"
        f"    id                VARCHAR(64)  NOT NULL,\n"
        f"    chunk_id          VARCHAR(64),\n"
        f"    knowledge_id      VARCHAR(64),\n"
        f"    knowledge_base_id VARCHAR(64),\n"
        f"    source_id         VARCHAR(255),\n"
        f"    source_type       INT,\n"
        f"    tag_id            VARCHAR(64),\n"
        f"    is_enabled        BOOLEAN,\n"
        f"    content           TEXT,\n"
        f"    embedding         ARRAY<FLOAT> NOT NULL,\n"
        f"    INDEX idx_chunk    (chunk_id)          USING INVERTED,\n"
        f"    INDEX idx_kb       (knowledge_base_id) USING INVERTED,\n"
        f"    INDEX idx_kid      (knowledge_id)      USING INVERTED,\n"
        f"    INDEX idx_src      (source_id)         USING INVERTED,\n"
        f"    INDEX idx_tag      (tag_id)            USING INVERTED,\n"
        f"    INDEX idx_enabled  (is_enabled)        USING INVERTED,\n"
        f"    INDEX idx_content  (content)           USING INVERTED"
        f' PROPERTIES("parser"="chinese","support_phrase"="true"),\n'
        f"    INDEX idx_emb      (embedding)         USING ANN PROPERTIES(\n"
        f'        "index_type"="hnsw",\n'
        f'        "metric_type"="{metric_type}",\n'
        f'        "dim"="{dimension}",\n'
        f'        "max_degree"="32",\n'
        f'        "ef_construction"="200"\n'
        f"    )\n"
        f") ENGINE=OLAP\n"
        f"{key_mode}\n"
        f"DISTRIBUTED BY HASH(id) BUCKETS {buckets}\n"
        f"PROPERTIES(\n"
        f"{properties}\n"
        f");"
    )


# ── Repository ──────────────────────────────────────────────────────


class DorisRepository:
    """Apache Doris retrieve engine repository (upstream ``dorisRepository``)."""

    def __init__(
        self,
        db: Any,
        fe_http_base: str,
        username: str,
        password: str,
        database: str,
        index_cfg: IndexConfig | None,
    ) -> None:
        self._db = db
        self._fe_http_base = fe_http_base.rstrip("/")
        self._username = username
        self._password = password
        self._database = database
        self._table_base_name = _resolve_collection_name(
            index_cfg, _ENV_DORIS_TABLE_PREFIX, _DEFAULT_TABLE_BASE_NAME
        )
        self._buckets_num = _get_buckets_num(index_cfg, 0)
        self._replication_num = _get_replication_num(index_cfg, 0)
        configured, invalid = _resolve_configured_compat_mode()
        self._compat_mode_requested: str = configured
        if invalid:
            logger.warning(
                "Invalid {}={}, defaulting to {}",
                _ENV_DORIS_COMPAT_MODE, invalid, _COMPAT_MODE_AUTO,
            )
        self._compat_mode_resolved: CompatMode | None = None
        self._compat_resolve_error: Exception | None = None
        self._lock = asyncio.Lock()
        self._initialized_tables: set[int] = set()

    # ── async pymysql wrappers ──

    async def _execute(self, sql: str, args: list[Any] | None = None) -> int:
        async with self._lock:
            def _do() -> int:
                with self._db.cursor() as cur:
                    affected = cur.execute(sql, args or ())
                    self._db.commit()
                    return int(affected or 0)
            return await asyncio.to_thread(_do)

    async def _query(self, sql: str, args: list[Any] | None = None) -> list[tuple[Any, ...]]:
        async with self._lock:
            def _do() -> list[tuple[Any, ...]]:
                with self._db.cursor() as cur:
                    cur.execute(sql, args or ())
                    return list(cur.fetchall())
            return await asyncio.to_thread(_do)

    async def _query_row(self, sql: str, args: list[Any] | None = None) -> tuple[Any, ...] | None:
        async with self._lock:
            def _do() -> tuple[Any, ...] | None:
                with self._db.cursor() as cur:
                    cur.execute(sql, args or ())
                    row = cur.fetchone()
                    return cast("tuple[Any, ...] | None", row)
            return await asyncio.to_thread(_do)

    # ── compat mode resolution ──

    async def _resolve_compat_mode(self, ctx: Context) -> CompatMode:
        if self._compat_mode_resolved is not None:
            return self._compat_mode_resolved
        if self._compat_resolve_error is not None:
            raise StorageBackendError(str(self._compat_resolve_error))
        requested: str = self._compat_mode_requested
        if requested == "" or requested == _COMPAT_MODE_AUTO:
            if requested == "":
                self._compat_mode_resolved = _COMPAT_MODE_INNER_PRODUCT_DUPLICATE
                return self._compat_mode_resolved
            probe = await self._probe_compat_mode(ctx)
            if probe:
                self._compat_mode_resolved = _COMPAT_MODE_INNER_PRODUCT_DUPLICATE
            else:
                self._compat_mode_resolved = _COMPAT_MODE_LEGACY
            logger.warning(
                "Auto-selected compat mode {} for new tables", self._compat_mode_resolved
            )
            return self._compat_mode_resolved
        self._compat_mode_resolved = cast(CompatMode, requested)
        return self._compat_mode_resolved

    async def _probe_compat_mode(self, ctx: Context) -> bool:
        try:
            await self._query_row("SELECT inner_product_approximate([1.0],[1.0])")
            return True
        except Exception:
            return False

    # ── table management ──

    def _get_table_name(self, dimension: int) -> str:
        return f"{self._table_base_name}_{dimension}"

    async def _table_exists(self, table_name: str) -> bool:
        row = await self._query_row(
            "SELECT COUNT(1) FROM information_schema.tables "
            "WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s",
            [self._database, table_name],
        )
        return row is not None and int(row[0]) > 0

    async def _ensure_table(self, ctx: Context, dimension: int) -> None:
        if dimension in self._initialized_tables:
            return
        compat_mode = await self._resolve_compat_mode(ctx)
        table_name = self._get_table_name(dimension)
        if not await self._table_exists(table_name):
            buckets = self._buckets_num if self._buckets_num > 0 else _DEFAULT_BUCKETS_NUM
            replication = (
                self._replication_num if self._replication_num > 0 else _DEFAULT_REPLICATION_NUM
            )
            ddl = _build_create_table_ddl(table_name, dimension, buckets, replication, compat_mode)
            await self._execute(ddl)
        self._initialized_tables.add(dimension)

    async def _list_embedding_tables(self) -> list[str]:
        rows = await self._query(
            "SELECT TABLE_NAME FROM information_schema.tables "
            "WHERE TABLE_SCHEMA = %s AND TABLE_NAME LIKE %s",
            [self._database, self._table_base_name + "\\_%"],
        )
        return [r[0] for r in rows]

    # ── RetrieveEngine ──

    def engine_type(self) -> RetrieverEngineType:
        return RetrieverEngineType.DORIS

    def support(self) -> list[RetrieverType]:
        return [RetrieverType.KEYWORDS, RetrieverType.VECTOR]

    async def retrieve(self, ctx: Context, params: RetrieveParams) -> list[RetrieveResult]:
        if params.retriever_type == RetrieverType.VECTOR:
            return await self._vector_retrieve(ctx, params)
        if params.retriever_type == RetrieverType.KEYWORDS:
            return await self._keywords_retrieve(ctx, params)
        raise ValidationError(
            code="doris.invalid_retriever_type",
            message=f"invalid retriever type: {params.retriever_type}",
        )

    # ── save / batch_save ──

    async def save(self, ctx: Context, index_info: IndexInfo, params: IndexSaveParams) -> None:
        await self.batch_save(ctx, [index_info], params)

    async def batch_save(
        self, ctx: Context, index_info_list: list[IndexInfo], params: IndexSaveParams
    ) -> None:
        if not index_info_list:
            return
        compat_mode = await self._resolve_compat_mode(ctx)
        groups: dict[int, list[DorisVectorEmbedding]] = {}
        for info in index_info_list:
            emb = _to_doris_vector_embedding(info, params, compat_mode)
            if not emb.embedding:
                logger.warning("Skipping empty embedding for chunk {}", info.chunk_id)
                continue
            _validate_embedding(emb.embedding)
            row = emb
            if row.id == "":
                row = DorisVectorEmbedding(
                    id=emb.source_id or str(uuid.uuid4()),
                    content=emb.content, source_id=emb.source_id, source_type=emb.source_type,
                    chunk_id=emb.chunk_id, knowledge_id=emb.knowledge_id,
                    knowledge_base_id=emb.knowledge_base_id, tag_id=emb.tag_id,
                    is_enabled=emb.is_enabled, embedding=emb.embedding,
                )
            dim = len(emb.embedding)
            groups.setdefault(dim, []).append(row)

        for dim, rows in groups.items():
            await self._ensure_table(ctx, dim)
            table = self._get_table_name(dim)
            if _uses_replace_write(compat_mode):
                await self._replace_rows(ctx, table, rows)
            else:
                await self._insert_rows(ctx, table, rows)
            logger.info("Saved {} rows to {}", len(rows), table)

    async def _insert_rows(self, ctx: Context, table: str, rows: list[DorisVectorEmbedding]) -> None:
        if not rows:
            return
        parts: list[str] = []
        args: list[Any] = []
        for e in rows:
            parts.append("(%s, %s, %s, %s, %s, %s, %s, %s, %s, " + _embedding_literal(e.embedding) + ")")
            args.extend([
                e.id, e.content, e.source_id, e.source_type,
                e.chunk_id, e.knowledge_id, e.knowledge_base_id, e.tag_id,
                e.is_enabled,
            ])
        stmt = f"INSERT INTO `{table}` ({', '.join(_COLUMNS)}) VALUES {', '.join(parts)}"
        await self._execute(stmt, args)

    async def _replace_rows(self, ctx: Context, table: str, rows: list[DorisVectorEmbedding]) -> None:
        deduped = _dedupe_rows_by_id(rows)
        if not deduped:
            return
        ids = [r.id for r in deduped]
        await self._delete_rows_by_id(ctx, table, ids)
        await self._insert_rows(ctx, table, deduped)

    async def _delete_rows_by_id(self, ctx: Context, table: str, ids: list[str]) -> None:
        if not ids:
            return
        placeholders = ", ".join(["%s"] * len(ids))
        stmt = f"DELETE FROM `{table}` WHERE {_FIELD_ID} IN ({placeholders})"
        await self._execute(stmt, ids)

    # ── estimate_storage_size ──

    def estimate_storage_size(
        self, ctx: Context, index_info_list: list[IndexInfo], params: IndexSaveParams
    ) -> int:
        total = 0
        for info in index_info_list:
            emb = _to_doris_vector_embedding(info, params, _COMPAT_MODE_INNER_PRODUCT_DUPLICATE)
            total += _calculate_storage_size(emb)
        return total

    # ── delete_by_* ──

    async def delete_by_chunk_id_list(
        self, ctx: Context, index_id_list: list[str], dimension: int, knowledge_type: str
    ) -> None:
        await self._delete_by_field(ctx, _FIELD_CHUNK_ID, index_id_list, dimension)

    async def delete_by_source_id_list(
        self, ctx: Context, source_id_list: list[str], dimension: int, knowledge_type: str
    ) -> None:
        await self._delete_by_field(ctx, _FIELD_SOURCE_ID, source_id_list, dimension)

    async def delete_by_knowledge_id_list(
        self, ctx: Context, knowledge_id_list: list[str], dimension: int, knowledge_type: str
    ) -> None:
        await self._delete_by_field(ctx, _FIELD_KNOWLEDGE_ID, knowledge_id_list, dimension)

    async def _delete_by_field(
        self, ctx: Context, field_name: str, ids: list[str], dimension: int
    ) -> None:
        if not ids:
            return
        table = self._get_table_name(dimension)
        placeholders = ", ".join(["%s"] * len(ids))
        stmt = f"DELETE FROM `{table}` WHERE {field_name} IN ({placeholders})"
        await self._execute(stmt, ids)
        logger.info("Deleted {} rows from {} by {}", len(ids), table, field_name)

    # ── retrieve ──

    async def _vector_retrieve(self, ctx: Context, params: RetrieveParams) -> list[RetrieveResult]:
        _validate_embedding(params.embedding)
        compat_mode = await self._resolve_compat_mode(ctx)
        query_embedding = list(params.embedding)
        if _normalize_embeddings(compat_mode):
            query_embedding = _normalize_embedding(query_embedding)
        dim = len(params.embedding)
        table = self._get_table_name(dim)

        if not await self._table_exists(table):
            logger.warning("Table {} does not exist, returning empty results", table)
            return _build_retrieve_result([], RetrieverType.VECTOR)

        wb = _build_base_filter(params)
        where_clause, where_args = wb.build()
        score_expr = f"inner_product_approximate(`{_FIELD_EMBEDDING}`, {_embedding_literal(query_embedding)})"
        if compat_mode == _COMPAT_MODE_LEGACY:
            score_expr = f"(1 - cosine_distance_approximate(`{_FIELD_EMBEDDING}`, {_embedding_literal(query_embedding)}))"
        stmt = (
            f"SELECT {', '.join(_COLUMNS_FOR_RETRIEVE)}, {score_expr} AS score "
            f"FROM `{table}` WHERE {where_clause} "
            f"HAVING score >= %s "
            f"ORDER BY score DESC LIMIT {params.top_k}"
        )
        args = [*where_args, params.threshold]
        rows = await self._query(stmt, args)
        results = [
            IndexWithScore(
                id=r[0], content=r[1], source_id=r[2], source_type=r[3],
                chunk_id=r[4], knowledge_id=r[5], knowledge_base_id=r[6],
                tag_id=r[7], is_enabled=r[8], score=float(r[9]),
                match_type=MatchType.EMBEDDING,
            )
            for r in rows
        ]
        logger.info("Vector retrieval found {} results in {}", len(results), table)
        return _build_retrieve_result(results, RetrieverType.VECTOR)

    async def _keywords_retrieve(self, ctx: Context, params: RetrieveParams) -> list[RetrieveResult]:
        query = params.query.strip()
        if query == "":
            return _build_retrieve_result([], RetrieverType.KEYWORDS)
        tables = await self._list_embedding_tables()
        if not tables:
            return _build_retrieve_result([], RetrieverType.KEYWORDS)

        wb = _build_base_filter(params)
        where_clause, where_args = wb.build()
        all_results: list[IndexWithScore] = []
        for table in tables:
            stmt = (
                f"SELECT {', '.join(_COLUMNS_FOR_RETRIEVE)} "
                f"FROM `{table}` WHERE {where_clause} "
                f"AND {_FIELD_CONTENT} MATCH_ANY %s LIMIT {params.top_k}"
            )
            args = [*where_args, query]
            try:
                rows = await self._query(stmt, args)
            except Exception as exc:
                logger.warning("Keyword retrieve in {} failed: {}", table, exc)
                continue
            for r in rows:
                all_results.append(IndexWithScore(
                    id=r[0], content=r[1], source_id=r[2], source_type=r[3],
                    chunk_id=r[4], knowledge_id=r[5], knowledge_base_id=r[6],
                    tag_id=r[7], is_enabled=r[8], score=1.0,
                    match_type=MatchType.KEYWORDS,
                ))
        if len(all_results) > params.top_k:
            all_results = all_results[:params.top_k]
        logger.info("Keywords retrieval found {} results across {} tables", len(all_results), len(tables))
        return _build_retrieve_result(all_results, RetrieverType.KEYWORDS)

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
        await self._ensure_table(ctx, dimension)
        table = self._get_table_name(dimension)
        page_size = 64
        offset = 0
        total_copied = 0
        while True:
            stmt = (
                f"SELECT {', '.join(_COLUMNS_FOR_COPY)} "
                f"FROM `{table}` WHERE {_FIELD_KNOWLEDGE_BASE_ID} = %s "
                f"ORDER BY {_FIELD_ID} LIMIT {page_size} OFFSET {offset}"
            )
            rows = await self._query(stmt, [source_knowledge_base_id])
            if not rows:
                break
            targets: list[DorisVectorEmbedding] = []
            for r in rows:
                src = DorisVectorEmbedding(
                    id=r[0], content=r[1], source_id=r[2], source_type=r[3],
                    chunk_id=r[4], knowledge_id=r[5], knowledge_base_id=r[6],
                    tag_id=r[7], is_enabled=r[8],
                    embedding=_parse_embedding_literal(r[9]),
                )
                target_chunk_id = source_to_target_chunk_id_map.get(src.chunk_id, "")
                if target_chunk_id == "":
                    continue
                target_knowledge_id = source_to_target_kb_id_map.get(src.knowledge_id, "")
                if target_knowledge_id == "":
                    continue
                target_source_id = _translate_source_id(src.source_id, src.chunk_id, target_chunk_id)
                targets.append(DorisVectorEmbedding(
                    id=str(uuid.uuid4()),
                    content=src.content,
                    source_id=target_source_id,
                    source_type=src.source_type,
                    chunk_id=target_chunk_id,
                    knowledge_id=target_knowledge_id,
                    knowledge_base_id=target_knowledge_base_id,
                    tag_id=src.tag_id,
                    is_enabled=src.is_enabled,
                    embedding=src.embedding,
                ))
            if targets:
                await self._insert_rows(ctx, table, targets)
                total_copied += len(targets)
            if len(rows) < page_size:
                break
            offset += page_size
        logger.info("CopyIndices done, dim={}, copied={}", dimension, total_copied)

    # ── batch_update_chunk_* ──

    async def batch_update_chunk_enabled_status(
        self, ctx: Context, chunk_status_map: Mapping[str, bool]
    ) -> None:
        if not chunk_status_map:
            return
        compat_mode = await self._resolve_compat_mode(ctx)
        if not _uses_rewrite_chunk_updates(compat_mode):
            await self._batch_update_chunk_enabled_status_legacy(ctx, chunk_status_map)
            return
        chunk_ids = list(chunk_status_map.keys())

        def mutate(row: DorisVectorEmbedding) -> DorisVectorEmbedding | None:
            enabled = chunk_status_map.get(row.chunk_id)
            if enabled is None or row.is_enabled == enabled:
                return None
            return DorisVectorEmbedding(
                id=row.id, content=row.content, source_id=row.source_id,
                source_type=row.source_type, chunk_id=row.chunk_id,
                knowledge_id=row.knowledge_id, knowledge_base_id=row.knowledge_base_id,
                tag_id=row.tag_id, is_enabled=enabled, embedding=row.embedding,
            )
        await self._rewrite_chunk_rows(ctx, chunk_ids, mutate, "rewrite is_enabled")

    async def batch_update_chunk_tag_id(
        self, ctx: Context, chunk_tag_map: Mapping[str, str]
    ) -> None:
        if not chunk_tag_map:
            return
        compat_mode = await self._resolve_compat_mode(ctx)
        if not _uses_rewrite_chunk_updates(compat_mode):
            await self._batch_update_chunk_tag_id_legacy(ctx, chunk_tag_map)
            return
        chunk_ids = list(chunk_tag_map.keys())

        def mutate(row: DorisVectorEmbedding) -> DorisVectorEmbedding | None:
            tag_id = chunk_tag_map.get(row.chunk_id)
            if tag_id is None or row.tag_id == tag_id:
                return None
            return DorisVectorEmbedding(
                id=row.id, content=row.content, source_id=row.source_id,
                source_type=row.source_type, chunk_id=row.chunk_id,
                knowledge_id=row.knowledge_id, knowledge_base_id=row.knowledge_base_id,
                tag_id=tag_id, is_enabled=row.is_enabled, embedding=row.embedding,
            )
        await self._rewrite_chunk_rows(ctx, chunk_ids, mutate, "rewrite tag_id")

    async def _rewrite_chunk_rows(
        self,
        ctx: Context,
        chunk_ids: list[str],
        mutate: Any,
        action: str,
    ) -> None:
        if not chunk_ids:
            return
        tables = await self._list_embedding_tables()
        for table in tables:
            rows = await self._load_rows_by_chunk_ids(ctx, table, chunk_ids)
            updated: list[DorisVectorEmbedding] = []
            for row in rows:
                mutated = mutate(row)
                if mutated is not None:
                    updated.append(mutated)
            if not updated:
                continue
            await self._replace_rows(ctx, table, updated)

    async def _load_rows_by_chunk_ids(
        self, ctx: Context, table: str, chunk_ids: list[str]
    ) -> list[DorisVectorEmbedding]:
        if not chunk_ids:
            return []
        placeholders = ", ".join(["%s"] * len(chunk_ids))
        stmt = (
            f"SELECT {', '.join(_COLUMNS_FOR_COPY)} "
            f"FROM `{table}` WHERE {_FIELD_CHUNK_ID} IN ({placeholders})"
        )
        rows = await self._query(stmt, chunk_ids)
        return [
            DorisVectorEmbedding(
                id=r[0], content=r[1], source_id=r[2], source_type=r[3],
                chunk_id=r[4], knowledge_id=r[5], knowledge_base_id=r[6],
                tag_id=r[7], is_enabled=r[8],
                embedding=_parse_embedding_literal(r[9]),
            )
            for r in rows
        ]

    async def _batch_update_chunk_enabled_status_legacy(
        self, ctx: Context, chunk_status_map: Mapping[str, bool]
    ) -> None:
        mapping = await self._lookup_chunk_row_keys(ctx, list(chunk_status_map.keys()))
        by_table: dict[str, list[dict[str, Any]]] = {}
        for chunk_id, locations in mapping.items():
            enabled = chunk_status_map.get(chunk_id)
            if enabled is None:
                continue
            for loc in locations:
                by_table.setdefault(loc[0], []).append({
                    _FIELD_ID: loc[1], _FIELD_IS_ENABLED: enabled,
                })
        for table, rows in by_table.items():
            await self._partial_update_rows(ctx, table, [_FIELD_ID, _FIELD_IS_ENABLED], rows)

    async def _batch_update_chunk_tag_id_legacy(
        self, ctx: Context, chunk_tag_map: Mapping[str, str]
    ) -> None:
        mapping = await self._lookup_chunk_row_keys(ctx, list(chunk_tag_map.keys()))
        by_table: dict[str, list[dict[str, Any]]] = {}
        for chunk_id, locations in mapping.items():
            tag_id = chunk_tag_map.get(chunk_id)
            if tag_id is None:
                continue
            for loc in locations:
                by_table.setdefault(loc[0], []).append({
                    _FIELD_ID: loc[1], _FIELD_TAG_ID: tag_id,
                })
        for table, rows in by_table.items():
            await self._partial_update_rows(ctx, table, [_FIELD_ID, _FIELD_TAG_ID], rows)

    async def _lookup_chunk_row_keys(
        self, ctx: Context, chunk_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        if not chunk_ids:
            return {}
        tables = await self._list_embedding_tables()
        if not tables:
            return {}
        placeholders = ", ".join(["%s"] * len(chunk_ids))
        out: dict[str, list[tuple[str, str]]] = {}
        for table in tables:
            stmt = (
                f"SELECT {_FIELD_ID}, {_FIELD_CHUNK_ID} "
                f"FROM `{table}` WHERE {_FIELD_CHUNK_ID} IN ({placeholders})"
            )
            rows = await self._query(stmt, chunk_ids)
            for r in rows:
                out.setdefault(r[1], []).append((table, r[0]))
        return out

    async def _partial_update_rows(
        self, ctx: Context, table: str, columns: list[str], rows: list[dict[str, Any]]
    ) -> None:
        """Stream Load partial update (legacy compat mode only)."""
        if not rows:
            return

        url = f"{self._fe_http_base}/api/{quote(self._database)}/{quote(table)}/_stream_load"
        body = json.dumps(rows).encode()
        req = urllib.request.Request(url, data=body, method="PUT")
        auth = base64.b64encode(f"{self._username}:{self._password}".encode()).decode()
        req.add_header("Authorization", f"Basic {auth}")
        req.add_header("Expect", "100-continue")
        req.add_header("Content-Type", "application/json")
        req.add_header("format", "json")
        req.add_header("strip_outer_array", "true")
        req.add_header("partial_columns", "true")
        req.add_header("columns", ",".join(columns))
        req.add_header("merge_type", "APPEND")

        def _do() -> None:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_body = resp.read()
                if resp.status // 100 != 2:
                    raise StorageBackendError(
                        code="doris.stream_load_http_failed",
                        message=f"stream load HTTP {resp.status}: {resp_body}",
                    )
                result = json.loads(resp_body)
                status = result.get("Status", "")
                if status not in ("Success", "Publish Timeout"):
                    raise StorageBackendError(
                        code="doris.stream_load_failed",
                        message=f"stream load failed: status={status} msg={result.get('Message')}",
                    )

        async with self._lock:
            await asyncio.to_thread(_do)


def new_doris_retrieve_engine_repository(
    db: Any,
    fe_http_base: str,
    username: str,
    password: str,
    database: str,
    index_cfg: IndexConfig | None,
) -> DorisRepository:
    """Create a Doris retrieve engine repository (upstream ``NewDorisRetrieveEngineRepository``)."""
    logger.info("[Doris] Initializing Doris retriever engine repository")
    return DorisRepository(db, fe_http_base, username, password, database, index_cfg)


def _connect_doris(addr: str, username: str, password: str, database: str) -> pymysql.connections.Connection:
    """Create a pymysql connection to Doris FE (MySQL protocol)."""
    host = addr
    port = 9030
    if ":" in addr:
        host, _, port_str = addr.rpartition(":")
        port = int(port_str)
    return pymysql.connect(
        host=host,
        port=port,
        user=username,
        password=password,
        database=database,
        charset="utf8mb4",
        autocommit=False,
    )


__all__ = [
    "DorisRepository",
    "new_doris_retrieve_engine_repository",
]
