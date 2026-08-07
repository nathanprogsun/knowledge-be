"""SQLite-vec retrieve engine repository (upstream ``sqlite/repository.go``).

Stores embedding metadata in ``lite_embeddings`` (a regular table), full-text
data in ``lite_embeddings_fts`` (FTS5 contentless virtual table), and vectors
in per-dimension ``vec_embeddings_<dim>`` vec0 virtual tables loaded by the
``sqlite-vec`` extension.

The repository uses the project's async SQLAlchemy engine (the same one the
application uses). The ``sqlite-vec`` extension is registered on every raw
DBAPI connection via a SQLAlchemy ``connect`` event listener. All SQL is
issued through ``text()`` so the statements stay portable and testable.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

import sqlalchemy
from sqlalchemy.ext.asyncio import AsyncEngine

import sqlite_vec

from src.ai.embedding import Context
from src.ai.retrieval.types import (
    IndexInfo,
    IndexSaveParams,
    RetrieveParams,
    RetrieverEngineType,
    RetrieveResult,
    RetrieverType,
)
from src.app_logging import logger

# ── Constants ────────────────────────────────────────────────────────

_TABLE_LITE_EMBEDDINGS = "lite_embeddings"
_TABLE_FTS = "lite_embeddings_fts"

_FIELD_ID = "id"
_FIELD_SOURCE_ID = "source_id"
_FIELD_SOURCE_TYPE = "source_type"
_FIELD_CHUNK_ID = "chunk_id"
_FIELD_KNOWLEDGE_ID = "knowledge_id"
_FIELD_KNOWLEDGE_BASE_ID = "knowledge_base_id"
_FIELD_TAG_ID = "tag_id"
_FIELD_CONTENT = "content"
_FIELD_DIMENSION = "dimension"
_FIELD_IS_ENABLED = "is_enabled"
_FIELD_EMBEDDING = "embedding"

_DDL_LITE_EMBEDDINGS = f"""\
CREATE TABLE IF NOT EXISTS {_TABLE_LITE_EMBEDDINGS} (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
    source_id         TEXT NOT NULL,
    source_type       INTEGER NOT NULL,
    chunk_id          TEXT,
    knowledge_id      TEXT,
    knowledge_base_id TEXT,
    tag_id            TEXT,
    content           TEXT NOT NULL,
    dimension         INTEGER NOT NULL DEFAULT 0,
    is_enabled        BOOLEAN DEFAULT 1,
    UNIQUE(source_id, source_type)
)"""

_DDL_FTS = f"""\
CREATE VIRTUAL TABLE IF NOT EXISTS {_TABLE_FTS} USING fts5(
    content, source_id, chunk_id, knowledge_id, knowledge_base_id,
    content='',
    contentless_delete=1,
    tokenize='unicode61'
)"""


# ── Helpers ─────────────────────────────────────────────────────────


def _vec_table_name(dim: int) -> str:
    return f"vec_embeddings_{dim}"


def _vec_ddl(dim: int) -> str:
    return (
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {_vec_table_name(dim)} "
        f"USING vec0(embedding float[{dim}] distance_metric=cosine)"
    )


def _extract_embedding(params: IndexSaveParams | None, source_id: str) -> list[float]:
    if not params:
        return []
    embedding_map = params.get("embedding")
    if not isinstance(embedding_map, dict):
        return []
    raw = embedding_map.get(source_id)
    if raw is None:
        return []
    return [float(v) for v in raw]


def _clean_invalid_utf8(s: str) -> str:
    cleaned: list[str] = []
    for ch in s:
        if ch == "\x00":
            continue
        if unicodedata.category(ch) != "Co":
            cleaned.append(ch)
    return "".join(cleaned)


# ── CJK bigram tokenizer ────────────────────────────────────────────

_CJK_RANGES = (
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Extension A
    (0x20000, 0x2A6DF),  # CJK Extension B
    (0x2A700, 0x2B73F),  # CJK Extension C
    (0x2B740, 0x2B81F),  # CJK Extension D
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0x3040, 0x309F),    # Hiragana
    (0x30A0, 0x30FF),     # Katakana
    (0xAC00, 0xD7AF),    # Hangul Syllables
)


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    for lo, hi in _CJK_RANGES:
        if lo <= cp <= hi:
            return True
    return False


def _tokenize_cjk_bigram(text: str) -> str:
    """Split CJK sequences into overlapping bigrams; keep non-CJK words intact."""
    parts: list[str] = []
    current_cjk: list[str] = []
    current_non_cjk: list[str] = []

    def flush_cjk() -> None:
        if not current_cjk:
            return
        if len(current_cjk) == 1:
            parts.append(current_cjk[0])
        else:
            for i in range(len(current_cjk) - 1):
                parts.append(current_cjk[i] + current_cjk[i + 1])
        current_cjk.clear()

    def flush_non_cjk() -> None:
        if current_non_cjk:
            parts.append("".join(current_non_cjk))
            current_non_cjk.clear()

    for ch in text:
        if _is_cjk(ch):
            flush_non_cjk()
            current_cjk.append(ch)
        elif ch.isspace() or unicodedata.category(ch).startswith(("P", "S")):
            flush_cjk()
            flush_non_cjk()
        else:
            flush_cjk()
            current_non_cjk.append(ch)
    flush_cjk()
    flush_non_cjk()
    return " ".join(parts)


def _sanitize_fts5_query(query: str) -> str:
    """Build an FTS5 MATCH query from user input with bigram tokenization."""
    query = query.strip()
    if not query:
        return ""
    tokenized = _tokenize_cjk_bigram(query)
    fields = tokenized.split()
    if not fields:
        return ""
    return " OR ".join(f'"{f}"' for f in fields if f)


def _placeholders(n: int) -> str:
    return ",".join(["?"] * n)


def _build_filter_where(params: RetrieveParams, alias: str) -> list[tuple[str, list[Any]]]:
    parts: list[tuple[str, list[Any]]] = []
    if params.knowledge_base_ids:
        parts.append((
            f"{alias}.knowledge_base_id IN ({_placeholders(len(params.knowledge_base_ids))})",
            list(params.knowledge_base_ids),
        ))
    if params.knowledge_ids:
        parts.append((
            f"{alias}.knowledge_id IN ({_placeholders(len(params.knowledge_ids))})",
            list(params.knowledge_ids),
        ))
    if params.tag_ids:
        parts.append((
            f"{alias}.tag_id IN ({_placeholders(len(params.tag_ids))})",
            list(params.tag_ids),
        ))
    return parts


# ── Repository ──────────────────────────────────────────────────────


class SQLiteRepository:
    """SQLite-vec retrieve engine repository (upstream ``sqliteRepository``)."""

    def __init__(self, db: Any) -> None:
        self._db = db
        engine = getattr(db, "engine", None)
        if engine is None:
            raise ValueError("sqlite repository requires a db handle with an 'engine' attribute")
        self._engine: AsyncEngine = engine
        self._vec_tables: set[int] = set()
        self._schema_initialized = False
        self._register_sqlite_vec_extension(engine)

    def _register_sqlite_vec_extension(self, engine: AsyncEngine) -> None:
        """Load the sqlite-vec extension on every new DBAPI connection."""
        try:
            @sqlalchemy.event.listens_for(engine.sync_engine, "connect")
            def _on_connect(dbapi_conn, _connection_record):
                sqlite_vec.load(dbapi_conn)
        except Exception as exc:
            logger.warning("Failed to register sqlite-vec extension: {}", exc)

    async def _ensure_schema(self) -> None:
        """Create tables and FTS5 on first use (async)."""
        if self._schema_initialized:
            return
        async with self._engine.begin() as conn:
            await conn.execute(sqlalchemy.text(_DDL_LITE_EMBEDDINGS))
            await conn.execute(sqlalchemy.text(_DDL_FTS))
        await self._ensure_existing_vec_tables()
        self._schema_initialized = True

    async def _ensure_existing_vec_tables(self) -> None:
        async with self._engine.connect() as conn:
            result = await conn.execute(
                sqlalchemy.text(
                    f"SELECT DISTINCT dimension FROM {_TABLE_LITE_EMBEDDINGS} WHERE dimension > 0"
                )
            )
            for row in result:
                dim = int(row[0])
                await self._ensure_vec_table(dim)

    async def _ensure_vec_table(self, dim: int) -> None:
        if dim <= 0 or dim in self._vec_tables:
            return
        async with self._engine.begin() as conn:
            await conn.execute(sqlalchemy.text(_vec_ddl(dim)))
        self._vec_tables.add(dim)

    # ── RetrieveEngine ──

    def engine_type(self) -> RetrieverEngineType:
        return RetrieverEngineType.SQLITE

    def support(self) -> list[RetrieverType]:
        return [RetrieverType.KEYWORDS, RetrieverType.VECTOR]

    # ── save / batch_save ──

    async def save(self, ctx: Context, index_info: IndexInfo, params: IndexSaveParams) -> None:
        await self._ensure_schema()
        row_id = await self._insert_embedding(index_info)
        emb = _extract_embedding(params, index_info.source_id)
        if emb and row_id > 0:
            dim = len(emb)
            await self._ensure_vec_table(dim)
            await self._insert_vec(row_id, dim, emb)
        if row_id > 0:
            await self._sync_fts5_insert(row_id, index_info)

    async def batch_save(
        self, ctx: Context, index_info_list: list[IndexInfo], params: IndexSaveParams
    ) -> None:
        if not index_info_list:
            return
        await self._ensure_schema()
        for info in index_info_list:
            row_id = await self._insert_embedding(info)
            emb = _extract_embedding(params, info.source_id)
            if emb and row_id > 0:
                dim = len(emb)
                await self._ensure_vec_table(dim)
                await self._insert_vec(row_id, dim, emb)
            if row_id > 0:
                await self._sync_fts5_insert(row_id, info)

    async def _insert_embedding(self, info: IndexInfo) -> int:
        emb_dim = 0
        # dimension is set after we know the embedding; for the INSERT we use 0
        # and update it after the vec insert (matching the Go pattern).
        sql = sqlalchemy.text(
            f"INSERT OR IGNORE INTO {_TABLE_LITE_EMBEDDINGS} "
            f"(source_id, source_type, chunk_id, knowledge_id, knowledge_base_id, "
            f"tag_id, content, dimension, is_enabled) "
            f"VALUES (:source_id, :source_type, :chunk_id, :knowledge_id, "
            f":knowledge_base_id, :tag_id, :content, :dimension, :is_enabled)"
        )
        async with self._engine.begin() as conn:
            result = await conn.execute(sql, {
                "source_id": info.source_id,
                "source_type": int(info.source_type),
                "chunk_id": info.chunk_id,
                "knowledge_id": info.knowledge_id,
                "knowledge_base_id": info.knowledge_base_id,
                "tag_id": info.tag_id,
                "content": _clean_invalid_utf8(info.content),
                "dimension": emb_dim,
                "is_enabled": 1 if info.is_enabled else 0,
            })
            row_id = result.lastrowid or 0
            if row_id == 0:
                # INSERT OR IGNORE skipped; look up existing row by source_id
                lookup = await conn.execute(
                    sqlalchemy.text(
                        f"SELECT id FROM {_TABLE_LITE_EMBEDDINGS} "
                        f"WHERE source_id = :source_id AND source_type = :source_type"
                    ),
                    {"source_id": info.source_id, "source_type": int(info.source_type)},
                )
                row = lookup.fetchone()
                row_id = int(row[0]) if row else 0
        return row_id

    async def _insert_vec(self, row_id: int, dim: int, emb: list[float]) -> None:
        blob = sqlite_vec.serialize_float32(emb)
        table = _vec_table_name(dim)
        async with self._engine.begin() as conn:
            await conn.execute(
                sqlalchemy.text(f"INSERT OR REPLACE INTO {table} (rowid, embedding) VALUES (:rowid, :blob)"),
                {"rowid": row_id, "blob": blob},
            )
            await conn.execute(
                sqlalchemy.text(
                    f"UPDATE {_TABLE_LITE_EMBEDDINGS} SET dimension = :dim WHERE id = :rowid"
                ),
                {"dim": dim, "rowid": row_id},
            )

    async def _sync_fts5_insert(self, row_id: int, info: IndexInfo) -> None:
        tokenized = _tokenize_cjk_bigram(info.content)
        async with self._engine.begin() as conn:
            await conn.execute(
                sqlalchemy.text(
                    f"INSERT OR REPLACE INTO {_TABLE_FTS} "
                    f"(rowid, content, source_id, chunk_id, knowledge_id, knowledge_base_id) "
                    f"VALUES (:rowid, :content, :source_id, :chunk_id, :knowledge_id, :knowledge_base_id)"
                ),
                {
                    "rowid": row_id,
                    "content": tokenized,
                    "source_id": info.source_id,
                    "chunk_id": info.chunk_id,
                    "knowledge_id": info.knowledge_id,
                    "knowledge_base_id": info.knowledge_base_id,
                },
            )

    # ── estimate_storage_size ──

    def estimate_storage_size(
        self, ctx: Context, index_info_list: list[IndexInfo], params: IndexSaveParams
    ) -> int:
        total = 0
        for info in index_info_list:
            total += len(info.content) + 200
        return total

    # ── retrieve ──

    async def retrieve(self, ctx: Context, params: RetrieveParams) -> list[RetrieveResult]:
        await self._ensure_schema()
        results: list[RetrieveResult] = []
        if params.retriever_type in (RetrieverType.KEYWORDS, ""):
            res = await self._keywords_retrieve(ctx, params)
            if res:
                results.extend(res)
        if params.retriever_type in (RetrieverType.VECTOR, ""):
            res = await self._vector_retrieve(ctx, params)
            if res:
                results.extend(res)
        return results

    async def _keywords_retrieve(self, ctx: Context, params: RetrieveParams) -> list[RetrieveResult]:
        from src.ai.retrieval.types import IndexWithScore, MatchType
        if not params.query:
            return []
        fts_query = _sanitize_fts5_query(params.query)
        if not fts_query:
            return []
        sql = (
            f"SELECT e.id, e.source_id, e.source_type, e.chunk_id, "
            f"e.knowledge_id, e.knowledge_base_id, e.tag_id, e.content, "
            f"(bm25({_TABLE_FTS}) * -1000000.0) AS score "
            f"FROM {_TABLE_FTS} "
            f"JOIN {_TABLE_LITE_EMBEDDINGS} e ON e.id = {_TABLE_FTS}.rowid "
            f"WHERE {_TABLE_FTS} MATCH :fts_query "
            f"AND (e.is_enabled IS NULL OR e.is_enabled = 1)"
        )
        args: dict[str, Any] = {"fts_query": fts_query}
        for clause, clause_args in _build_filter_where(params, "e"):
            sql += f" AND {clause}"
            for i, val in enumerate(clause_args):
                args[f"f_{i}"] = val
        sql += " ORDER BY score DESC LIMIT :top_k"
        args["top_k"] = params.top_k if params.top_k > 0 else 10
        async with self._engine.connect() as conn:
            result = await conn.execute(sqlalchemy.text(sql), args)
            rows = result.fetchall()
        items = [
            IndexWithScore(
                id=str(r[0]),
                source_id=r[1],
                source_type=r[2],
                chunk_id=r[3],
                knowledge_id=r[4],
                knowledge_base_id=r[5],
                tag_id=r[6],
                content=r[7],
                score=float(r[8]),
                match_type=MatchType.KEYWORDS,
            )
            for r in rows
        ]
        return [RetrieveResult(
            results=items,
            retriever_engine_type=RetrieverEngineType.SQLITE,
            retriever_type=RetrieverType.KEYWORDS,
        )]

    async def _vector_retrieve(self, ctx: Context, params: RetrieveParams) -> list[RetrieveResult]:
        from src.ai.retrieval.types import IndexWithScore, MatchType
        if not params.embedding:
            return []
        dim = len(params.embedding)
        await self._ensure_vec_table(dim)
        query_blob = sqlite_vec.serialize_float32(params.embedding)
        table = _vec_table_name(dim)
        vec_sql = (
            f"SELECT v.rowid, v.distance, "
            f"e.source_id, e.source_type, e.chunk_id, "
            f"e.knowledge_id, e.knowledge_base_id, e.tag_id, e.content "
            f"FROM {table} v "
            f"JOIN {_TABLE_LITE_EMBEDDINGS} e ON e.id = v.rowid "
            f"WHERE v.embedding MATCH :query_blob "
            f"AND k = :top_k "
            f"AND v.rowid IN ("
            f"SELECT filtered.id FROM {_TABLE_LITE_EMBEDDINGS} filtered "
            f"WHERE (filtered.is_enabled IS NULL OR filtered.is_enabled = 1)"
        )
        args: dict[str, Any] = {"query_blob": query_blob, "top_k": params.top_k}
        for clause, clause_args in _build_filter_where(params, "filtered"):
            vec_sql += f" AND {clause}"
            for i, val in enumerate(clause_args):
                args[f"f_{i}"] = val
        vec_sql += ") ORDER BY v.distance ASC"
        async with self._engine.connect() as conn:
            result = await conn.execute(sqlalchemy.text(vec_sql), args)
            rows = result.fetchall()
        items: list[IndexWithScore] = []
        for r in rows:
            distance = float(r[1])
            score = 1.0 - distance
            if params.threshold > 0 and score < params.threshold:
                continue
            items.append(IndexWithScore(
                id=str(r[0]),
                source_id=r[2],
                source_type=r[3],
                chunk_id=r[4],
                knowledge_id=r[5],
                knowledge_base_id=r[6],
                tag_id=r[7],
                content=r[8],
                score=score,
                match_type=MatchType.EMBEDDING,
            ))
        return [RetrieveResult(
            results=items,
            retriever_engine_type=RetrieverEngineType.SQLITE,
            retriever_type=RetrieverType.VECTOR,
        )]

    # ── delete_by_* ──

    async def delete_by_chunk_id_list(
        self, ctx: Context, chunk_id_list: list[str], dimension: int, knowledge_type: str
    ) -> None:
        await self._delete_by_field(_FIELD_CHUNK_ID, chunk_id_list)

    async def delete_by_source_id_list(
        self, ctx: Context, source_id_list: list[str], dimension: int, knowledge_type: str
    ) -> None:
        await self._delete_by_field(_FIELD_SOURCE_ID, source_id_list)

    async def delete_by_knowledge_id_list(
        self, ctx: Context, knowledge_id_list: list[str], dimension: int, knowledge_type: str
    ) -> None:
        await self._delete_by_field(_FIELD_KNOWLEDGE_ID, knowledge_id_list)

    async def _delete_by_field(self, field_name: str, ids: list[str]) -> None:
        if not ids:
            return
        await self._ensure_schema()
        ph = _placeholders(len(ids))
        async with self._engine.begin() as conn:
            rows = await conn.execute(
                sqlalchemy.text(
                    f"SELECT id, dimension FROM {_TABLE_LITE_EMBEDDINGS} "
                    f"WHERE {field_name} IN ({ph})"
                ),
                ids,
            )
            dim_ids: dict[int, list[int]] = {}
            for row in rows:
                row_id = int(row[0])
                dim = int(row[1])
                if dim > 0:
                    dim_ids.setdefault(dim, []).append(row_id)
            for dim, row_ids in dim_ids.items():
                if dim not in self._vec_tables:
                    continue
                table = _vec_table_name(dim)
                id_ph = _placeholders(len(row_ids))
                await conn.execute(
                    sqlalchemy.text(f"DELETE FROM {table} WHERE rowid IN ({id_ph})"),
                    row_ids,
                )
            await conn.execute(
                sqlalchemy.text(f"DELETE FROM {_TABLE_FTS} WHERE rowid IN ("
                               f"SELECT id FROM {_TABLE_LITE_EMBEDDINGS} WHERE {field_name} IN ({ph}))"),
                ids,
            )
            await conn.execute(
                sqlalchemy.text(f"DELETE FROM {_TABLE_LITE_EMBEDDINGS} WHERE {field_name} IN ({ph})"),
                ids,
            )

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
        import uuid as _uuid
        await self._ensure_schema()
        for source_chunk_id, target_chunk_id in source_to_target_chunk_id_map.items():
            async with self._engine.connect() as conn:
                result = await conn.execute(
                    sqlalchemy.text(
                        f"SELECT source_id, source_type, chunk_id, knowledge_id, "
                        f"knowledge_base_id, tag_id, content, dimension, is_enabled "
                        f"FROM {_TABLE_LITE_EMBEDDINGS} WHERE chunk_id = :chunk_id LIMIT 1"
                    ),
                    {"chunk_id": source_chunk_id},
                )
                src = result.fetchone()
            if src is None:
                continue
            target_knowledge_id = source_to_target_kb_id_map.get(src[3], "")
            new_source_id = str(_uuid.uuid4())
            async with self._engine.begin() as conn:
                insert_result = await conn.execute(
                    sqlalchemy.text(
                        f"INSERT INTO {_TABLE_LITE_EMBEDDINGS} "
                        f"(source_id, source_type, chunk_id, knowledge_id, "
                        f"knowledge_base_id, tag_id, content, dimension, is_enabled) "
                        f"VALUES (:source_id, :source_type, :chunk_id, :knowledge_id, "
                        f":knowledge_base_id, :tag_id, :content, :dimension, :is_enabled)"
                    ),
                    {
                        "source_id": new_source_id,
                        "source_type": int(src[1]),
                        "chunk_id": target_chunk_id,
                        "knowledge_id": target_knowledge_id,
                        "knowledge_base_id": target_knowledge_base_id,
                        "tag_id": src[5],
                        "content": src[6],
                        "dimension": int(src[7]),
                        "is_enabled": int(src[8]),
                    },
                )
                new_row_id = insert_result.lastrowid or 0
                if new_row_id > 0 and int(src[7]) > 0:
                    await self._copy_vec(conn, int(src[0]), new_row_id, int(src[7]))
                    await self._sync_fts5_insert_row(
                        conn, new_row_id, src[6], new_source_id,
                        target_chunk_id, target_knowledge_id, target_knowledge_base_id,
                    )

    async def _copy_vec(self, conn, src_id: int, dst_id: int, dim: int) -> None:
        if dim not in self._vec_tables:
            return
        table = _vec_table_name(dim)
        await conn.execute(
            sqlalchemy.text(
                f"INSERT INTO {table} (rowid, embedding) "
                f"SELECT :dst_id, embedding FROM {table} WHERE rowid = :src_id"
            ),
            {"dst_id": dst_id, "src_id": src_id},
        )

    async def _sync_fts5_insert_row(
        self, conn, row_id: int, content: str, source_id: str,
        chunk_id: str, knowledge_id: str, knowledge_base_id: str,
    ) -> None:
        tokenized = _tokenize_cjk_bigram(content)
        await conn.execute(
            sqlalchemy.text(
                f"INSERT INTO {_TABLE_FTS} "
                f"(rowid, content, source_id, chunk_id, knowledge_id, knowledge_base_id) "
                f"VALUES (:rowid, :content, :source_id, :chunk_id, :knowledge_id, :knowledge_base_id)"
            ),
            {
                "rowid": row_id, "content": tokenized, "source_id": source_id,
                "chunk_id": chunk_id, "knowledge_id": knowledge_id,
                "knowledge_base_id": knowledge_base_id,
            },
        )

    # ── batch_update_chunk_* ──

    async def batch_update_chunk_enabled_status(
        self, ctx: Context, chunk_status_map: Mapping[str, bool]
    ) -> None:
        if not chunk_status_map:
            return
        await self._ensure_schema()
        async with self._engine.begin() as conn:
            for chunk_id, enabled in chunk_status_map.items():
                await conn.execute(
                    sqlalchemy.text(
                        f"UPDATE {_TABLE_LITE_EMBEDDINGS} "
                        f"SET is_enabled = :enabled WHERE chunk_id = :chunk_id"
                    ),
                    {"enabled": 1 if enabled else 0, "chunk_id": chunk_id},
                )

    async def batch_update_chunk_tag_id(
        self, ctx: Context, chunk_tag_map: Mapping[str, str]
    ) -> None:
        if not chunk_tag_map:
            return
        await self._ensure_schema()
        async with self._engine.begin() as conn:
            for chunk_id, tag_id in chunk_tag_map.items():
                await conn.execute(
                    sqlalchemy.text(
                        f"UPDATE {_TABLE_LITE_EMBEDDINGS} "
                        f"SET tag_id = :tag_id WHERE chunk_id = :chunk_id"
                    ),
                    {"tag_id": tag_id, "chunk_id": chunk_id},
                )


def new_sqlite_retrieve_engine_repository(db: Any) -> SQLiteRepository:
    """Create a SQLite-vec retrieve engine repository (upstream ``NewSQLiteRetrieveEngineRepository``)."""
    logger.info("[SQLite] Initializing SQLite retriever engine repository with sqlite-vec")
    return SQLiteRepository(db)


__all__ = [
    "SQLiteRepository",
    "new_sqlite_retrieve_engine_repository",
]
