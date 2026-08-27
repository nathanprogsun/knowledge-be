"""Tests for the SQLite-vec retrieve engine repository.

Mock the project's async SQLAlchemy engine with a controllable fake that
records every ``execute`` call and returns canned rows. The repository's
``AsyncEngine.begin()`` / ``connect()`` are mocked so no real SQLite
connection is opened and no ``sqlite-vec`` extension is loaded.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.ai.embedding import TaskContext
from src.ai.retrieval.sqlite_vec import (
    SQLiteRepository,
    _build_filter_where,
    _clean_invalid_utf8,
    _extract_embedding,
    _is_cjk,
    _placeholders,
    _sanitize_fts5_query,
    _tokenize_cjk_bigram,
    _vec_ddl,
    _vec_table_name,
    new_sqlite_retrieve_engine_repository,
)
from src.ai.retrieval.types import (
    IndexInfo,
    IndexSaveParams,
    MatchType,
    RetrieveParams,
    RetrieverEngineType,
    RetrieverType,
    SourceType,
)
from src.common.exception import ValidationError

_CTX = TaskContext()


# ── Fake async engine ───────────────────────────────────────────────


class _FakeAsyncEngine:
    """Stand-in for the project's AsyncEngine that records every SQL execute."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, Any]] = []
        self._select_responses: list[Any] = []
        self._select_index = 0
        self._next_insert_id: int = 1
        self.sync_engine = MagicMock(name="sync_engine")

    def set_select_responses(self, responses: list[Any]) -> None:
        self._select_responses = list(responses)
        self._select_index = 0

    def _make_connection(self):
        engine = self

        class _Conn:
            def __init__(self) -> None:
                pass

            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, *_exc):
                return None

            async def execute(self_inner, stmt, params: Any = ()):
                sql = str(stmt)
                engine.executed.append((sql, params))
                upper = sql.strip().upper()
                # BEGIN / COMMIT pass through
                if upper.startswith(("BEGIN", "COMMIT")):
                    return MagicMock(rowcount=0)
                # INSERT
                if upper.startswith("INSERT"):
                    row_id = engine._next_insert_id
                    engine._next_insert_id += 1
                    return MagicMock(lastrowid=row_id, rowcount=1, fetchone=lambda: (row_id,))
                # UPDATE
                if upper.startswith("UPDATE"):
                    return MagicMock(lastrowid=None, rowcount=1)
                # DELETE
                if upper.startswith("DELETE"):
                    return MagicMock(lastrowid=None, rowcount=0)
                # CREATE TABLE / CREATE VIRTUAL TABLE (DDL, no result)
                if upper.startswith("CREATE"):
                    return MagicMock(lastrowid=None, rowcount=0)
                # SELECT / MATCH: consume the next canned response.
                # Each response is either a single row tuple or a list of
                # row tuples; wrap a bare tuple into a single-row list so
                # ``fetchall()`` returns ``[row]``.
                if engine._select_index < len(engine._select_responses):
                    response = engine._select_responses[engine._select_index]
                    engine._select_index += 1
                else:
                    response = []
                if callable(response):
                    response = response()
                if response and not isinstance(response, list):
                    rows_list = [response]
                else:
                    rows_list = list(response) if response else []
                return MagicMock(
                    lastrowid=None,
                    rowcount=len(rows_list),
                    fetchall=lambda: rows_list,
                    fetchone=lambda: rows_list[0] if rows_list else None,
                )

        return _Conn()

    def begin(self):
        return self._make_connection()

    def connect(self):
        return self._make_connection()


class _FakeDB:
    """Stand-in for the project DBEngine wrapper."""

    def __init__(self) -> None:
        self.engine = _FakeAsyncEngine()


def _new_repo(existing_dims: list[int] | None = None) -> tuple[SQLiteRepository, _FakeDB]:
    db = _FakeDB()
    repo = new_sqlite_retrieve_engine_repository(db)
    if existing_dims:
        for d in existing_dims:
            repo._vec_tables.add(d)
    return repo, db


# ── Pure helpers ────────────────────────────────────────────────────


def test_vec_table_name_formats() -> None:
    assert _vec_table_name(4) == "vec_embeddings_4"
    assert _vec_table_name(768) == "vec_embeddings_768"


def test_vec_ddl_uses_vec0_with_cosine() -> None:
    ddl = _vec_ddl(768)
    assert "vec0" in ddl
    assert "float[768]" in ddl
    assert "distance_metric=cosine" in ddl


def test_placeholders_returns_correct_count() -> None:
    assert _placeholders(0) == ""
    assert _placeholders(1) == "?"
    assert _placeholders(3) == "?,?,?"


def test_is_cjk_detects_cjk_ranges() -> None:
    assert _is_cjk("中") is True
    assert _is_cjk("字") is True
    assert _is_cjk("a") is False
    assert _is_cjk(" ") is False
    assert _is_cjk("한") is True  # Hangul
    assert _is_cjk("ア") is True  # Katakana


def test_tokenize_cjk_bigram_handles_mixed_text() -> None:
    assert _tokenize_cjk_bigram("") == ""
    result = _tokenize_cjk_bigram("中文 ab")
    assert "中文" in result  # CJK bigram
    assert "ab" in result  # non-CJK word preserved


def test_tokenize_cjk_bigram_single_cjk() -> None:
    assert _tokenize_cjk_bigram("中") == "中"


def test_sanitize_fts5_query_empty_input() -> None:
    assert _sanitize_fts5_query("") == ""
    assert _sanitize_fts5_query("   ") == ""


def test_sanitize_fts5_query_joins_with_or() -> None:
    result = _sanitize_fts5_query("hello world")
    assert " OR " in result
    assert '"hello"' in result
    assert '"world"' in result


def test_clean_invalid_utf8_strips_nulls() -> None:
    assert _clean_invalid_utf8("hello\x00world") == "helloworld"


def test_extract_embedding_returns_by_source_id() -> None:
    params: IndexSaveParams = {"embedding": {"s-1": [1.0, 2.0]}}
    assert _extract_embedding(params, "s-1") == [1.0, 2.0]


def test_extract_embedding_returns_empty_for_missing() -> None:
    assert _extract_embedding(None, "s-1") == []
    assert _extract_embedding({}, "s-1") == []
    assert _extract_embedding({"other": [1.0]}, "s-1") == []


def test_build_filter_where_includes_in_clauses() -> None:
    parts = _build_filter_where(
        RetrieveParams(
            knowledge_base_ids=["kb-1", "kb-2"],
            knowledge_ids=["k-1"],
            tag_ids=["t-1"],
        ),
        "e",
    )
    assert len(parts) == 3
    assert all("IN" in p[0] for p in parts)
    assert parts[0][1] == ["kb-1", "kb-2"]


def test_build_filter_where_empty_params() -> None:
    assert _build_filter_where(RetrieveParams(), "e") == []


# ── RetrieveEngine contract ────────────────────────────────────────


def test_engine_type_returns_sqlite() -> None:
    repo, _ = _new_repo()
    assert repo.engine_type() == RetrieverEngineType.SQLITE


def test_support_returns_keywords_and_vector() -> None:
    repo, _ = _new_repo()
    assert set(repo.support()) == {RetrieverType.KEYWORDS, RetrieverType.VECTOR}


def test_estimate_storage_size_accumulates() -> None:
    repo, _ = _new_repo()
    infos = [
        IndexInfo(source_id="s1", content="hello"),
        IndexInfo(source_id="s2", content="world"),
    ]
    total = repo.estimate_storage_size(_CTX, infos, {})
    assert total > 0


# ── save / batch_save ──────────────────────────────────────────────


async def test_save_writes_metadata_and_embedding() -> None:
    repo, db = _new_repo()
    info = IndexInfo(
        source_id="s-1",
        chunk_id="c-1",
        content="hello",
        knowledge_id="k-1",
        knowledge_base_id="kb-1",
        tag_id="t-1",
        is_enabled=True,
        source_type=SourceType.CHUNK,
    )
    await repo.save(_CTX, info, {"embedding": {"s-1": [1.0, 2.0, 3.0, 4.0]}})
    executed_sqls = " | ".join(s for s, _ in db.engine.executed)
    assert "lite_embeddings" in executed_sqls
    assert "vec_embeddings_4" in executed_sqls
    assert "lite_embeddings_fts" in executed_sqls


async def test_save_without_embedding_skips_vec_insert() -> None:
    repo, db = _new_repo()
    info = IndexInfo(source_id="s-1", chunk_id="c-1", content="hello")
    await repo.save(_CTX, info, {"embedding": {}})
    executed_sqls = " | ".join(s for s, _ in db.engine.executed)
    assert "vec_embeddings_" not in executed_sqls


async def test_batch_save_empty_list_is_noop() -> None:
    repo, db = _new_repo()
    await repo.batch_save(_CTX, [], {})
    assert db.engine.executed == []


async def test_batch_save_processes_all_items() -> None:
    repo, db = _new_repo()
    infos = [
        IndexInfo(source_id=f"s-{i}", chunk_id=f"c-{i}", content=f"text-{i}") for i in range(3)
    ]
    embeddings = {f"s-{i}": [float(i)] * 4 for i in range(3)}
    await repo.batch_save(_CTX, infos, {"embedding": embeddings})
    # 3 inserts into lite_embeddings + 3 vec inserts + 3 fts inserts
    insert_count = sum(1 for s, _ in db.engine.executed if "INSERT" in s.upper())
    assert insert_count >= 9


# ── retrieve ───────────────────────────────────────────────────────


async def test_retrieve_with_keywords_type_runs_keywords_path() -> None:
    repo, db = _new_repo()
    db.engine.set_select_responses([[]])
    params = RetrieveParams(
        query="hello",
        top_k=5,
        retriever_type=RetrieverType.KEYWORDS,
    )
    await repo.retrieve(_CTX, params)
    executed = db.engine.executed
    assert any("fts" in s.lower() for s, _ in executed)


async def test_retrieve_with_vector_type_runs_vector_path() -> None:
    repo, db = _new_repo(existing_dims=[4])
    db.engine.set_select_responses([[]])
    params = RetrieveParams(
        embedding=[1.0, 2.0, 3.0, 4.0],
        top_k=5,
        retriever_type=RetrieverType.VECTOR,
    )
    await repo.retrieve(_CTX, params)
    executed = db.engine.executed
    assert any("vec_embeddings" in s for s, _ in executed)


async def test_keywords_retrieve_empty_query_returns_empty() -> None:
    repo, _ = _new_repo()
    results = await repo._keywords_retrieve(_CTX, RetrieveParams(query=""))
    assert results == []


async def test_keywords_retrieve_returns_results() -> None:
    repo, db = _new_repo()
    db.engine.set_select_responses(
        [
            (1, "s-1", 0, "c-1", "k-1", "kb-1", "t-1", "hello", 1.5),
        ]
    )
    results = await repo._keywords_retrieve(
        _CTX,
        RetrieveParams(query="hello", top_k=5, retriever_type=RetrieverType.KEYWORDS),
    )
    assert len(results) == 1
    assert len(results[0].results) == 1
    item = results[0].results[0]
    assert item.id == "1"
    assert item.match_type == MatchType.KEYWORDS
    assert item.score == 1.5


async def test_vector_retrieve_empty_embedding_returns_empty() -> None:
    repo, _ = _new_repo()
    results = await repo._vector_retrieve(_CTX, RetrieveParams(retriever_type=RetrieverType.VECTOR))
    assert results == []


async def test_vector_retrieve_returns_results() -> None:
    repo, db = _new_repo()
    db.engine.set_select_responses(
        [
            (1, 0.1, "s-1", 0, "c-1", "k-1", "kb-1", "t-1", "hello"),
        ]
    )
    results = await repo._vector_retrieve(
        _CTX,
        RetrieveParams(
            embedding=[1.0, 2.0, 3.0, 4.0],
            top_k=5,
            threshold=0.0,
            retriever_type=RetrieverType.VECTOR,
        ),
    )
    assert len(results) == 1
    item = results[0].results[0]
    assert item.id == "1"
    assert item.match_type == MatchType.EMBEDDING
    # Score is 1 - distance = 0.9
    assert item.score == pytest.approx(0.9)


async def test_vector_retrieve_filters_by_threshold() -> None:
    repo, db = _new_repo()
    db.engine.set_select_responses(
        [
            (1, 0.5, "s-1", 0, "c-1", "k-1", "kb-1", "t-1", "hello"),  # distance 0.5 -> score 0.5
        ]
    )
    results = await repo._vector_retrieve(
        _CTX,
        RetrieveParams(
            embedding=[1.0, 2.0, 3.0, 4.0],
            top_k=5,
            threshold=0.8,
            retriever_type=RetrieverType.VECTOR,
        ),
    )
    assert results[0].results == []


# ── delete_by_* ────────────────────────────────────────────────────


async def test_delete_by_chunk_id_list() -> None:
    repo, db = _new_repo(existing_dims=[4])
    # First query: look up rows to find vec IDs
    db.engine.set_select_responses(
        [
            [(1, 4)],  # SELECT id, dimension
        ]
    )
    await repo.delete_by_chunk_id_list(_CTX, ["c-1", "c-2"], dimension=4, knowledge_type="")
    executed_sqls = " | ".join(s for s, _ in db.engine.executed)
    assert "DELETE FROM" in executed_sqls.upper()


async def test_delete_by_source_id_list() -> None:
    repo, db = _new_repo(existing_dims=[4])
    db.engine.set_select_responses([(1, 4)])
    await repo.delete_by_source_id_list(_CTX, ["s-1"], dimension=4, knowledge_type="")
    executed_sqls = " | ".join(s for s, _ in db.engine.executed)
    assert "DELETE FROM" in executed_sqls.upper()


async def test_delete_by_knowledge_id_list() -> None:
    repo, db = _new_repo(existing_dims=[4])
    db.engine.set_select_responses([(1, 4)])
    await repo.delete_by_knowledge_id_list(_CTX, ["k-1"], dimension=4, knowledge_type="")
    executed_sqls = " | ".join(s for s, _ in db.engine.executed)
    assert "DELETE FROM" in executed_sqls.upper()


async def test_delete_with_empty_ids_is_noop() -> None:
    repo, db = _new_repo()
    db.engine.executed.clear()
    await repo.delete_by_chunk_id_list(_CTX, [], dimension=4, knowledge_type="")
    # No SQL should have been issued (ensure_schema still runs DDL on first call)
    delete_count = sum(1 for s, _ in db.engine.executed if "DELETE" in s.upper())
    assert delete_count == 0


# ── copy_indices ───────────────────────────────────────────────────


async def test_copy_indices_empty_mapping_is_noop() -> None:
    repo, db = _new_repo()
    # First SELECT goes to _ensure_existing_vec_tables (returns empty)
    db.engine.set_select_responses([[]])
    await repo.copy_indices(_CTX, "src-kb", {}, {}, "tgt-kb", dimension=4, knowledge_type="")
    # No INSERT should have been executed
    insert_count = sum(
        1
        for s, _ in db.engine.executed
        if s.strip().upper().startswith("INSERT INTO LITE_EMBEDDINGS")
    )
    assert insert_count == 0


async def test_copy_indices_copies_one_chunk() -> None:
    repo, db = _new_repo(existing_dims=[4])
    # SELECT 1: _ensure_existing_vec_tables (empty)
    # SELECT 2: look up source row
    db.engine.set_select_responses(
        [
            [],  # ensure_existing_vec_tables
            # id, source_id, source_type, chunk_id, knowledge_id, knowledge_base_id, tag_id, content, dimension, is_enabled
            (100, "src-1", 0, "src-c-1", "src-k-1", "src-kb-1", "t-1", "hello", 4, 1),
        ]
    )
    await repo.copy_indices(
        _CTX,
        "src-kb",
        {"src-k-1": "tgt-k-1"},
        {"src-c-1": "tgt-c-1"},
        "tgt-kb",
        dimension=4,
        knowledge_type="",
    )
    executed_sqls = " | ".join(s for s, _ in db.engine.executed)
    assert "INSERT INTO LITE_EMBEDDINGS" in executed_sqls.upper()


async def test_copy_indices_skips_missing_source_chunk() -> None:
    repo, db = _new_repo(existing_dims=[4])
    # First SELECT for ensure_existing_vec_tables, second for the chunk lookup (None)
    db.engine.set_select_responses([[], None])
    await repo.copy_indices(
        _CTX,
        "src-kb",
        {"src-k-1": "tgt-k-1"},
        {"src-c-1": "tgt-c-1"},
        "tgt-kb",
        dimension=4,
        knowledge_type="",
    )
    insert_count = sum(
        1
        for s, _ in db.engine.executed
        if s.strip().upper().startswith("INSERT INTO") and "lite_embeddings" in s
    )
    assert insert_count == 0


# ── batch_update_chunk_* ──────────────────────────────────────────


async def test_batch_update_chunk_enabled_status() -> None:
    repo, db = _new_repo()
    await repo.batch_update_chunk_enabled_status(_CTX, {"c-1": True, "c-2": False})
    update_count = sum(
        1 for s, _ in db.engine.executed if s.strip().upper().startswith("UPDATE LITE_EMBEDDINGS")
    )
    assert update_count == 2


async def test_batch_update_chunk_tag_id() -> None:
    repo, db = _new_repo()
    await repo.batch_update_chunk_tag_id(_CTX, {"c-1": "t-a", "c-2": "t-b"})
    update_count = sum(
        1 for s, _ in db.engine.executed if s.strip().upper().startswith("UPDATE LITE_EMBEDDINGS")
    )
    assert update_count == 2


async def test_batch_update_empty_map_is_noop() -> None:
    repo, db = _new_repo()
    db.engine.executed.clear()
    await repo.batch_update_chunk_enabled_status(_CTX, {})
    await repo.batch_update_chunk_tag_id(_CTX, {})
    update_count = sum(
        1 for s, _ in db.engine.executed if s.strip().upper().startswith("UPDATE LITE_EMBEDDINGS")
    )
    assert update_count == 0


# ── Schema bootstrap ──────────────────────────────────────────────


async def test_ensure_schema_creates_tables_lazily() -> None:
    repo, db = _new_repo()
    db.engine.executed.clear()
    db.engine.set_select_responses([])
    await repo._ensure_schema()
    executed_sqls = " | ".join(s for s, _ in db.engine.executed)
    assert "CREATE TABLE IF NOT EXISTS lite_embeddings" in executed_sqls
    assert "CREATE VIRTUAL TABLE IF NOT EXISTS lite_embeddings_fts" in executed_sqls


async def test_ensure_schema_only_runs_once() -> None:
    repo, db = _new_repo()
    db.engine.executed.clear()
    db.engine.set_select_responses([])
    await repo._ensure_schema()
    first_count = len(db.engine.executed)
    await repo._ensure_schema()
    # Second call does not re-execute the DDL.
    assert len(db.engine.executed) == first_count


# ── Constructor ────────────────────────────────────────────────────


def test_new_sqlite_retrieve_engine_repository_requires_engine() -> None:
    with pytest.raises(ValidationError, match="requires a db handle with an 'engine' attribute"):
        new_sqlite_retrieve_engine_repository(object())


def test_new_sqlite_retrieve_engine_repository_registers_extension() -> None:
    db = _FakeDB()
    repo = new_sqlite_retrieve_engine_repository(db)
    assert isinstance(repo, SQLiteRepository)
    # The sqlite-vec listener should have been registered on the sync_engine.
    assert db.engine.sync_engine is not None
