"""Postgres + pgvector retrieval engine repository.

Mirrors the upstream postgres repository contract: keyword search via
ParadeDB's ``|||`` operator and vector similarity search via pgvector's
``<=>`` cosine distance operator on half-precision vectors. Index
records are stored in the ``embeddings`` table.

The repository holds a session factory (an ``async_sessionmaker``) and
creates a fresh session per operation, matching the upstream pattern
where each method borrows a transaction from the shared DB handle.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Protocol, TypeAlias, cast

from sqlalchemy import bindparam, text
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.elements import BindParameter, TextClause

from src.ai.embedding import Context
from src.ai.retrieval.base import Database, RetrieveEngineRepository
from src.ai.retrieval.types import (
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

# ── Constants ────────────────────────────────────────────────────────

_TABLE: str = "embeddings"

# Storage-size estimation (mirrors the upstream calculateIndexStorageSize).
_METADATA_SIZE_BYTES: int = 200
_HNSW_INDEX_OVERHEAD_MULTIPLIER: int = 2
_HALF_PRECISION_BYTES_PER_DIM: int = 2

# Vector retrieve expanded-topK bounds (keeps HNSW efficient).
_EXPANDED_TOPK_MIN: int = 100
_EXPANDED_TOPK_MAX: int = 200
_HNSW_EF_SEARCH_MIN: int = 40

# Copy indices pagination batch size.
_COPY_BATCH_SIZE: int = 500

# Bind-param value union for INSERT/SELECT operations.
_DbParamValue: TypeAlias = str | int | bool | float | list[str]


# ── Structural protocols ─────────────────────────────────────────────


class _SessionFactory(Protocol):
    """Callable producing an ``AsyncSession`` (async context manager)."""

    def __call__(self) -> AsyncSession: ...


class _DatabaseHandle(Protocol):
    """Structural view of the DB engine exposing session creation."""

    session_factory: async_sessionmaker[AsyncSession]


# ── Row conversion helpers ───────────────────────────────────────────


def _format_halfvec(embedding: list[float]) -> str:
    """Format an embedding vector as a pgvector halfvec string literal."""
    return "[" + ",".join(str(x) for x in embedding) + "]"


def _to_db_row(index_info: IndexInfo, params: IndexSaveParams) -> dict[str, str | int | bool]:
    """Convert an ``IndexInfo`` to a column-value dict for INSERT.

    Mirrors the upstream ``toDBVectorEmbedding``: base fields come from
    ``index_info``; the embedding and dimension come from the
    ``"embedding"`` sub-map keyed by source id. When no embedding is
    present, dimension defaults to 0 and embedding to an empty halfvec.
    """
    row: dict[str, str | int | bool] = {
        "source_id": index_info.source_id,
        "source_type": int(index_info.source_type),
        "chunk_id": index_info.chunk_id,
        "knowledge_id": index_info.knowledge_id,
        "knowledge_base_id": index_info.knowledge_base_id,
        "tag_id": index_info.tag_id,
        "content": index_info.content,
        "is_enabled": index_info.is_enabled,
    }
    embedding_map = params.get("embedding", {})
    embedding = embedding_map.get(index_info.source_id)
    if embedding is not None:
        row["dimension"] = len(embedding)
        row["embedding"] = _format_halfvec(embedding)
    else:
        row["dimension"] = 0
        row["embedding"] = "[]"
    return row


def _calculate_index_storage_size(content: str, dimension: int) -> int:
    """Estimate the storage size for a single index entry in bytes."""
    content_size = len(content.encode("utf-8"))
    vector_size = dimension * _HALF_PRECISION_BYTES_PER_DIM if dimension > 0 else 0
    metadata_size = _METADATA_SIZE_BYTES
    index_overhead = vector_size * _HNSW_INDEX_OVERHEAD_MULTIPLIER
    return content_size + vector_size + metadata_size + index_overhead


def _from_row_with_score(row: RowMapping, match_type: MatchType) -> IndexWithScore:
    """Convert a DB row (with ``score`` column) to an ``IndexWithScore``."""
    id_val = row.get("id")
    return IndexWithScore(
        id=str(id_val) if id_val is not None else "",
        content=str(row.get("content", "")),
        source_id=str(row.get("source_id", "")),
        source_type=SourceType(int(row.get("source_type", 0))),
        chunk_id=str(row.get("chunk_id", "")),
        knowledge_id=str(row.get("knowledge_id", "")),
        knowledge_base_id=str(row.get("knowledge_base_id", "")),
        tag_id=str(row.get("tag_id", "")),
        score=float(row.get("score", 0.0)),
        match_type=match_type,
        is_enabled=bool(row.get("is_enabled", False)),
    )


def _build_insert_stmt_text(columns: tuple[str, ...], *, on_conflict_do_nothing: bool) -> str:
    """Build an ``INSERT INTO embeddings (...) VALUES (...)`` statement.

    The ``embedding`` column receives a ``CAST(:embedding AS halfvec)``
    so the string literal is coerced to the pgvector type at the DB side.
    """
    col_list = ", ".join(columns)
    value_parts: list[str] = []
    for col in columns:
        if col == "embedding":
            value_parts.append("CAST(:embedding AS halfvec)")
        else:
            value_parts.append(f":{col}")
    param_list = ", ".join(value_parts)
    conflict = " ON CONFLICT DO NOTHING" if on_conflict_do_nothing else ""
    return f"INSERT INTO {_TABLE} ({col_list}) VALUES ({param_list}){conflict}"


def _transform_source_id(src_source_id: str, src_chunk_id: str, target_chunk_id: str) -> str:
    """Transform a source ID during index copy.

    Regular chunks have ``source_id == chunk_id`` and adopt the target
    chunk id. Generated questions carry ``{chunkID}-{questionID}`` and
    preserve the question suffix. Other forms get a fresh UUID.
    """
    if src_source_id == src_chunk_id:
        return target_chunk_id
    if src_chunk_id and src_source_id.startswith(src_chunk_id + "-"):
        question_id = src_source_id[len(src_chunk_id) + 1 :]
        return f"{target_chunk_id}-{question_id}"
    return str(uuid.uuid4())


# ── Repository ───────────────────────────────────────────────────────


class PostgresRetrieveEngineRepository:
    """Postgres + pgvector retrieval engine repository.

    Implements ``RetrieveEngineRepository`` over the ``embeddings`` table.
    Each method creates a session from the injected factory, executes raw
    SQL via ``sqlalchemy.text()``, and commits write operations.
    """

    def __init__(self, session_factory: _SessionFactory) -> None:
        self._session_factory = session_factory

    def engine_type(self) -> RetrieverEngineType:
        return RetrieverEngineType.POSTGRES

    def support(self) -> list[RetrieverType]:
        return [RetrieverType.KEYWORDS, RetrieverType.VECTOR]

    # ── Storage size ────────────────────────────────────────────────

    def estimate_storage_size(
        self,
        ctx: Context,
        index_info_list: list[IndexInfo],
        params: IndexSaveParams,
    ) -> int:
        """Estimate total storage size for the index info list."""
        del ctx
        total = 0
        for index_info in index_info_list:
            row = _to_db_row(index_info, params)
            content = str(row["content"])
            dimension = int(row["dimension"])
            total += _calculate_index_storage_size(content, dimension)
        logger.info(
            "[Postgres] Estimated storage size for {} indices: {} bytes",
            len(index_info_list),
            total,
        )
        return total

    # ── Save / BatchSave ────────────────────────────────────────────

    async def save(self, ctx: Context, index_info: IndexInfo, params: IndexSaveParams) -> None:
        """Save a single index entry."""
        del ctx
        row = _to_db_row(index_info, params)
        columns = tuple(row.keys())
        stmt = text(_build_insert_stmt_text(columns, on_conflict_do_nothing=False))
        async with self._session_factory() as session:
            await session.execute(stmt, row)
            await session.commit()

    async def batch_save(
        self,
        ctx: Context,
        index_info_list: list[IndexInfo],
        params: IndexSaveParams,
    ) -> None:
        """Save a list of index entries in one batch (ON CONFLICT DO NOTHING)."""
        del ctx
        if not index_info_list:
            return
        rows = [_to_db_row(item, params) for item in index_info_list]
        columns = tuple(rows[0].keys())
        stmt = text(_build_insert_stmt_text(columns, on_conflict_do_nothing=True))
        async with self._session_factory() as session:
            await session.execute(stmt, rows)
            await session.commit()

    # ── Retrieve ────────────────────────────────────────────────────

    async def retrieve(self, ctx: Context, params: RetrieveParams) -> list[RetrieveResult]:
        """Route to keywords or vector retrieval based on retriever type."""
        if params.retriever_type == RetrieverType.KEYWORDS:
            return await self._keywords_retrieve(ctx, params)
        if params.retriever_type == RetrieverType.VECTOR:
            return await self._vector_retrieve(ctx, params)
        raise ValidationError(
            code="pgvector.invalid_retriever_type",
            message=f"invalid retriever type: {params.retriever_type}",
        )

    async def _keywords_retrieve(
        self, ctx: Context, params: RetrieveParams
    ) -> list[RetrieveResult]:
        """Keyword-based search using ParadeDB's ``|||`` operator."""
        del ctx
        where_parts: list[str] = []
        bind_params: dict[str, _DbParamValue] = {}
        expanding: list[BindParameter[str]] = []

        if params.knowledge_base_ids:
            where_parts.append("knowledge_base_id IN :kb_ids")
            expanding.append(bindparam("kb_ids", expanding=True))
            bind_params["kb_ids"] = params.knowledge_base_ids
        if params.knowledge_ids:
            where_parts.append("knowledge_id IN :k_ids")
            expanding.append(bindparam("k_ids", expanding=True))
            bind_params["k_ids"] = params.knowledge_ids
        if params.tag_ids:
            where_parts.append("tag_id IN :tag_ids")
            expanding.append(bindparam("tag_ids", expanding=True))
            bind_params["tag_ids"] = params.tag_ids

        where_parts.append("content ||| :query")
        bind_params["query"] = params.query

        where_parts.append("(is_enabled IS NULL OR is_enabled = :is_enabled)")
        bind_params["is_enabled"] = True

        where_clause = " AND ".join(where_parts)
        sql = (
            f"SELECT paradedb.score(id) as score, id, content, source_id, "
            f"source_type, chunk_id, knowledge_id, knowledge_base_id, tag_id "
            f"FROM {_TABLE} WHERE {where_clause} ORDER BY score DESC LIMIT :top_k"
        )
        bind_params["top_k"] = params.top_k

        stmt = text(sql).bindparams(*expanding)
        async with self._session_factory() as session:
            result = await session.execute(stmt, bind_params)
            rows = list(result.mappings().all())

        results = [_from_row_with_score(row, MatchType.KEYWORDS) for row in rows]
        return [
            RetrieveResult(
                results=results,
                retriever_engine_type=RetrieverEngineType.POSTGRES,
                retriever_type=RetrieverType.KEYWORDS,
                error=None,
            )
        ]

    async def _vector_retrieve(self, ctx: Context, params: RetrieveParams) -> list[RetrieveResult]:
        """Vector similarity search using pgvector's ``<=>`` cosine distance."""
        del ctx
        dimension = len(params.embedding)
        embedding_str = _format_halfvec(params.embedding)

        where_parts: list[str] = ["dimension = :dimension"]
        bind_params: dict[str, _DbParamValue] = {
            "embedding": embedding_str,
            "dimension": dimension,
        }
        expanding: list[BindParameter[str]] = []

        if params.knowledge_base_ids:
            where_parts.append("knowledge_base_id IN :kb_ids")
            expanding.append(bindparam("kb_ids", expanding=True))
            bind_params["kb_ids"] = params.knowledge_base_ids
        if params.knowledge_ids:
            where_parts.append("knowledge_id IN :k_ids")
            expanding.append(bindparam("k_ids", expanding=True))
            bind_params["k_ids"] = params.knowledge_ids
        if params.tag_ids:
            where_parts.append("tag_id IN :tag_ids")
            expanding.append(bindparam("tag_ids", expanding=True))
            bind_params["tag_ids"] = params.tag_ids

        where_parts.append("(is_enabled IS NULL OR is_enabled = :is_enabled)")
        bind_params["is_enabled"] = True

        where_clause = "WHERE " + " AND ".join(where_parts)

        expanded_topk = params.top_k * 2
        if expanded_topk < _EXPANDED_TOPK_MIN:
            expanded_topk = _EXPANDED_TOPK_MIN
        if expanded_topk > _EXPANDED_TOPK_MAX:
            expanded_topk = _EXPANDED_TOPK_MAX
        if expanded_topk < params.top_k:
            expanded_topk = params.top_k

        query = (
            f"SELECT id, content, source_id, source_type, chunk_id, "
            f"knowledge_id, knowledge_base_id, tag_id, (1 - distance) as score "
            f"FROM ("
            f"SELECT id, content, source_id, source_type, chunk_id, "
            f"knowledge_id, knowledge_base_id, tag_id, "
            f"embedding::halfvec({dimension}) <=> :embedding::halfvec({dimension}) "
            f"as distance FROM {_TABLE} {where_clause} "
            f"ORDER BY embedding::halfvec({dimension}) <=> "
            f":embedding::halfvec({dimension}) LIMIT :subquery_limit"
            f") AS candidates WHERE distance <= :distance_threshold "
            f"ORDER BY distance ASC LIMIT :final_limit"
        )

        bind_params["subquery_limit"] = expanded_topk
        bind_params["distance_threshold"] = 1.0 - params.threshold
        bind_params["final_limit"] = params.top_k

        stmt = text(query).bindparams(*expanding)
        ef_search = max(expanded_topk, _HNSW_EF_SEARCH_MIN)

        rows = await self._execute_vector_query(stmt, bind_params, ef_search)

        if len(rows) > params.top_k:
            rows = rows[: params.top_k]

        results = [_from_row_with_score(row, MatchType.EMBEDDING) for row in rows]
        return [
            RetrieveResult(
                results=results,
                retriever_engine_type=RetrieverEngineType.POSTGRES,
                retriever_type=RetrieverType.VECTOR,
                error=None,
            )
        ]

    async def _execute_vector_query(
        self,
        stmt: TextClause,
        bind_params: Mapping[str, _DbParamValue],
        ef_search: int,
    ) -> list[RowMapping]:
        """Execute the vector similarity query with HNSW GUC overrides.

        Wraps the query in a transaction so ``SET LOCAL`` can raise
        ``hnsw.ef_search`` / ``hnsw.iterative_scan`` for the session. If
        the GUCs are unavailable (older pgvector), the transaction
        aborts and the query is retried without the overrides.
        """
        try:
            async with self._session_factory() as session, session.begin():
                await session.execute(text(f"SET LOCAL hnsw.ef_search = {ef_search}"))
                await session.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
                result = await session.execute(stmt, dict(bind_params))
                return list(result.mappings().all())
        except Exception as exc:
            exc_str = str(exc)
            if "hnsw.ef_search" not in exc_str and "hnsw.iterative_scan" not in exc_str:
                raise
            logger.warning(
                "[Postgres] Retrying vector query without HNSW GUC overrides: {}",
                exc,
            )
            async with self._session_factory() as session:
                result = await session.execute(stmt, dict(bind_params))
                return list(result.mappings().all())

    # ── Deletes ─────────────────────────────────────────────────────

    async def delete_by_chunk_id_list(
        self,
        ctx: Context,
        index_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        """Delete index records by chunk id list."""
        del ctx, dimension, knowledge_type
        if not index_id_list:
            return
        stmt = text(f"DELETE FROM {_TABLE} WHERE chunk_id IN :ids").bindparams(
            bindparam("ids", expanding=True)
        )
        async with self._session_factory() as session:
            await session.execute(stmt, {"ids": index_id_list})
            await session.commit()

    async def delete_by_source_id_list(
        self,
        ctx: Context,
        source_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        """Delete index records by source id list."""
        del ctx, dimension, knowledge_type
        if not source_id_list:
            return
        stmt = text(f"DELETE FROM {_TABLE} WHERE source_id IN :ids").bindparams(
            bindparam("ids", expanding=True)
        )
        async with self._session_factory() as session:
            await session.execute(stmt, {"ids": source_id_list})
            await session.commit()

    async def delete_by_knowledge_id_list(
        self,
        ctx: Context,
        knowledge_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        """Delete index records by knowledge id list."""
        del ctx, dimension, knowledge_type
        if not knowledge_id_list:
            return
        stmt = text(f"DELETE FROM {_TABLE} WHERE knowledge_id IN :ids").bindparams(
            bindparam("ids", expanding=True)
        )
        async with self._session_factory() as session:
            await session.execute(stmt, {"ids": knowledge_id_list})
            await session.commit()

    # ── Copy indices ────────────────────────────────────────────────

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
        """Copy index data from a source KB to a target KB (paginated)."""
        del ctx, dimension, knowledge_type
        if not source_to_target_chunk_id_map:
            return
        offset = 0
        select_stmt = text(
            f"SELECT content, source_id, source_type, chunk_id, "
            f"knowledge_id, dimension, embedding FROM {_TABLE} "
            f"WHERE knowledge_base_id = :source_kb_id "
            f"LIMIT :limit OFFSET :offset"
        )
        insert_columns = (
            "content",
            "source_id",
            "source_type",
            "chunk_id",
            "knowledge_id",
            "knowledge_base_id",
            "dimension",
            "embedding",
        )
        insert_stmt = text(_build_insert_stmt_text(insert_columns, on_conflict_do_nothing=True))
        while True:
            async with self._session_factory() as session:
                result = await session.execute(
                    select_stmt,
                    {
                        "source_kb_id": source_knowledge_base_id,
                        "limit": _COPY_BATCH_SIZE,
                        "offset": offset,
                    },
                )
                source_rows = list(result.mappings().all())
            if not source_rows:
                break
            target_rows = self._build_copy_rows(
                source_rows,
                source_to_target_kb_id_map,
                source_to_target_chunk_id_map,
                target_knowledge_base_id,
            )
            if target_rows:
                async with self._session_factory() as session:
                    await session.execute(insert_stmt, target_rows)
                    await session.commit()
            offset += len(source_rows)
            if len(source_rows) < _COPY_BATCH_SIZE:
                break

    @staticmethod
    def _build_copy_rows(
        source_rows: Sequence[RowMapping],
        source_to_target_kb_id_map: Mapping[str, str],
        source_to_target_chunk_id_map: Mapping[str, str],
        target_knowledge_base_id: str,
    ) -> list[dict[str, str | int | bool]]:
        """Transform source rows into target INSERT rows."""
        target_rows: list[dict[str, str | int | bool]] = []
        for source_row in source_rows:
            src_chunk_id = str(source_row.get("chunk_id", ""))
            target_chunk_id = source_to_target_chunk_id_map.get(src_chunk_id)
            if target_chunk_id is None:
                continue
            src_knowledge_id = str(source_row.get("knowledge_id", ""))
            target_knowledge_id = source_to_target_kb_id_map.get(src_knowledge_id)
            if target_knowledge_id is None:
                continue
            src_source_id = str(source_row.get("source_id", ""))
            target_source_id = _transform_source_id(src_source_id, src_chunk_id, target_chunk_id)
            emb_val = source_row.get("embedding")
            target_rows.append(
                {
                    "content": str(source_row.get("content", "")),
                    "source_id": target_source_id,
                    "source_type": int(source_row.get("source_type", 0)),
                    "chunk_id": target_chunk_id,
                    "knowledge_id": target_knowledge_id,
                    "knowledge_base_id": target_knowledge_base_id,
                    "dimension": int(source_row.get("dimension", 0)),
                    "embedding": str(emb_val) if emb_val is not None else "[]",
                }
            )
        return target_rows

    # ── Batch updates ───────────────────────────────────────────────

    async def batch_update_chunk_enabled_status(
        self, ctx: Context, chunk_status_map: Mapping[str, bool]
    ) -> None:
        """Update the enabled status of chunks in batch."""
        del ctx
        if not chunk_status_map:
            return
        enabled_ids = [k for k, v in chunk_status_map.items() if v]
        disabled_ids = [k for k, v in chunk_status_map.items() if not v]
        async with self._session_factory() as session:
            if enabled_ids:
                stmt = text(
                    f"UPDATE {_TABLE} SET is_enabled = TRUE WHERE chunk_id IN :ids"
                ).bindparams(bindparam("ids", expanding=True))
                await session.execute(stmt, {"ids": enabled_ids})
            if disabled_ids:
                stmt = text(
                    f"UPDATE {_TABLE} SET is_enabled = FALSE WHERE chunk_id IN :ids"
                ).bindparams(bindparam("ids", expanding=True))
                await session.execute(stmt, {"ids": disabled_ids})
            await session.commit()

    async def batch_update_chunk_tag_id(
        self, ctx: Context, chunk_tag_map: Mapping[str, str]
    ) -> None:
        """Update the tag id of chunks in batch."""
        del ctx
        if not chunk_tag_map:
            return
        tag_groups: dict[str, list[str]] = {}
        for chunk_id, tag_id in chunk_tag_map.items():
            tag_groups.setdefault(tag_id, []).append(chunk_id)
        async with self._session_factory() as session:
            for tag_id, chunk_ids in tag_groups.items():
                stmt = text(
                    f"UPDATE {_TABLE} SET tag_id = :tag_id WHERE chunk_id IN :ids"
                ).bindparams(bindparam("ids", expanding=True))
                await session.execute(stmt, {"tag_id": tag_id, "ids": chunk_ids})
            await session.commit()


# ── Constructor ─────────────────────────────────────────────────────


def new_postgres_retrieve_engine_repository(db: Database) -> RetrieveEngineRepository:
    """Create a Postgres + pgvector retrieval engine repository.

    The ``db`` handle is the opaque ``Database`` protocol forwarded by
    the factory/registry layers. At runtime it is the async SQLAlchemy
    engine wrapper; this function extracts its ``session_factory`` via
    a structural cast.
    """
    handle = cast("_DatabaseHandle", db)
    logger.info("[Postgres] Initializing PostgreSQL retriever engine repository")
    return PostgresRetrieveEngineRepository(handle.session_factory)


__all__ = [
    "PostgresRetrieveEngineRepository",
    "new_postgres_retrieve_engine_repository",
]
