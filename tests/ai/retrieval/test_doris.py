"""Tests for the Doris retrieve engine repository.

Mock a pymysql connection (the ``db`` handle) with a cursor factory that
returns a controllable fake cursor. The fake cursor records the last SQL
and arguments and dispatches canned results so every branch can be pinned.
No real Doris / MySQL server is contacted.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.ai.embedding import TaskContext
from src.ai.retrieval.doris import (
    DorisRepository,
    _build_base_filter,
    _build_create_table_ddl,
    _build_retrieve_result,
    _calculate_storage_size,
    _dedupe_rows_by_id,
    _embedding_literal,
    _get_buckets_num,
    _get_replication_num,
    _normalize_embedding,
    _parse_embedding_literal,
    _resolve_collection_name,
    _resolve_configured_compat_mode,
    _to_doris_vector_embedding,
    _translate_source_id,
    _validate_embedding,
    new_doris_retrieve_engine_repository,
)
from src.ai.retrieval.types import (
    IndexConfig,
    IndexInfo,
    IndexSaveParams,
    MatchType,
    RetrieveParams,
    RetrieverEngineType,
    RetrieverType,
    SourceType,
)

_CTX = TaskContext()


# ── Fake pymysql DB ─────────────────────────────────────────────────


class _FakeCursor:
    """Controllable cursor: records SQL/args and dispatches canned results."""

    def __init__(self, script: list[tuple[str, Any]]) -> None:
        self._script = script
        self._index = 0
        self.executed: list[tuple[str, list[Any]]] = []
        self._fetchone_result: tuple[Any, ...] | None = None
        self._fetchall_result: list[tuple[Any, ...]] = []
        self.rowcount = 1

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *_exc: Any) -> None:
        return None

    def execute(self, sql: str, args: Any = ()) -> int:
        self.executed.append((sql, list(args) if args else []))
        if self._index < len(self._script):
            key, response = self._script[self._index]
            self._index += 1
            if key != "*" and sql.strip().upper().startswith(key.upper()):
                self._set_response(response)
                return self.rowcount
        self._set_response(None)
        return self.rowcount

    def _set_response(self, response: Any) -> None:
        if isinstance(response, BaseException):
            raise response
        if response is None:
            self._fetchone_result = None
            self._fetchall_result = []
            self.rowcount = 0
        elif isinstance(response, tuple):
            self._fetchone_result = response
            self._fetchall_result = []
            self.rowcount = 1
        elif isinstance(response, list):
            self._fetchone_result = response[0] if response else None
            self._fetchall_result = response
            self.rowcount = len(response)
        else:
            self._fetchone_result = None
            self._fetchall_result = []
            self.rowcount = 1

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._fetchone_result

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._fetchall_result)


class _FakeDB:
    """Stand-in for a pymysql connection with a single cursor factory."""

    def __init__(self, script: list[tuple[str, Any]]) -> None:
        self._cursor = _FakeCursor(script)
        self.committed = 0
        self.rolled_back = 0

    def cursor(self) -> _FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.committed += 1

    def rollback(self) -> None:
        self.rolled_back += 1


def _new_db(script: list[tuple[str, Any]]) -> _FakeDB:
    return _FakeDB(script)


def _new_repo(
    script: list[tuple[str, Any]] | None = None,
    index_config: IndexConfig | None = None,
    compat_mode_env: str = "",
    env: dict[str, str] | None = None,
) -> tuple[DorisRepository, _FakeDB]:
    """Build a DorisRepository with a mocked pymysql connection."""
    if env:
        for _k, _v in env.items():
            pass
    db = _new_db(script or [])
    repo = DorisRepository(
        db=db,
        fe_http_base="http://doris-fe:8030",
        username="root",
        password="",
        database="test_db",
        index_cfg=index_config,
    )
    # Pre-resolve compat mode to avoid lazy probing in tests
    repo._compat_mode_resolved = "inner_product_duplicate"
    return repo, db


# ── Pure helpers ────────────────────────────────────────────────────


def test_resolve_collection_name_priority() -> None:
    cfg = IndexConfig(collection_prefix="prefix_x", collection_name="name_y")
    assert _resolve_collection_name(cfg, "ENV_KEY", "default") == "prefix_x"
    cfg2 = IndexConfig(collection_name="name_y")
    assert _resolve_collection_name(cfg2, "ENV_KEY", "default") == "name_y"
    cfg3 = IndexConfig()
    # Falls through to env (cleared in test env or env_key not set)
    assert _resolve_collection_name(cfg3, "UNLIKELY_ENV_KEY_xyz", "default") == "default"


def test_get_buckets_num_default_when_zero() -> None:
    assert _get_buckets_num(None, 10) == 10
    assert _get_buckets_num(IndexConfig(buckets_num=0), 10) == 10
    assert _get_buckets_num(IndexConfig(buckets_num=42), 10) == 42


def test_get_replication_num_default_when_zero() -> None:
    assert _get_replication_num(None, 1) == 1
    assert _get_replication_num(IndexConfig(replication_num=0), 1) == 1
    assert _get_replication_num(IndexConfig(replication_num=3), 1) == 3


def test_resolve_configured_compat_mode_defaults_to_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DORIS_COMPAT_MODE", raising=False)
    mode, invalid = _resolve_configured_compat_mode()
    assert mode == "auto"
    assert invalid == ""


def test_resolve_configured_compat_mode_recognizes_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    for value, expected in (
        ("legacy", "legacy"),
        ("inner_product_duplicate", "inner_product_duplicate"),
        ("inner-product-duplicate", "inner_product_duplicate"),
        ("inner_product", "inner_product_duplicate"),
    ):
        monkeypatch.setenv("DORIS_COMPAT_MODE", value)
        mode, invalid = _resolve_configured_compat_mode()
        assert mode == expected, f"value={value}"
        assert invalid == ""


def test_resolve_configured_compat_mode_invalid_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DORIS_COMPAT_MODE", "garbage")
    mode, invalid = _resolve_configured_compat_mode()
    assert mode == "auto"
    assert invalid == "garbage"


def test_embedding_literal_empty_returns_empty_array() -> None:
    assert _embedding_literal([]) == "[]"


def test_embedding_literal_formats_floats() -> None:
    literal = _embedding_literal([0.1, 0.2, 1.5])
    assert literal.startswith("[") and literal.endswith("]")
    assert "," in literal


def test_parse_embedding_literal_roundtrip() -> None:
    raw = "[0.1,0.2,1.5]"
    parsed = _parse_embedding_literal(raw)
    assert pytest.approx(parsed, rel=1e-5) == [0.1, 0.2, 1.5]


def test_parse_embedding_literal_empty() -> None:
    assert _parse_embedding_literal("") == []
    assert _parse_embedding_literal("[]") == []
    assert _parse_embedding_literal(b"[1, 2]") == [1.0, 2.0]


def test_validate_embedding_rejects_nan_inf() -> None:
    _validate_embedding([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="not finite"):
        _validate_embedding([1.0, float("nan"), 2.0])
    with pytest.raises(ValueError, match="not finite"):
        _validate_embedding([1.0, float("inf")])
    with pytest.raises(ValueError, match="not finite"):
        _validate_embedding([float("-inf"), 2.0])


def test_normalize_embedding_unit_length() -> None:
    normalized = _normalize_embedding([3.0, 4.0])
    assert pytest.approx(sum(v * v for v in normalized), abs=1e-6) == 1.0


def test_normalize_embedding_zero_vector_passthrough() -> None:
    vec = [0.0, 0.0, 0.0]
    assert _normalize_embedding(vec) == vec


def test_translate_source_id_same_chunk_returns_target() -> None:
    assert _translate_source_id("chunk-1", "chunk-1", "chunk-2") == "chunk-2"


def test_translate_source_id_question_suffix() -> None:
    assert _translate_source_id("chunk-1-q1", "chunk-1", "chunk-2") == "chunk-2-q1"


def test_translate_source_id_other_returns_uuid() -> None:
    result = _translate_source_id("orphan-id", "chunk-1", "chunk-2")
    assert result != "orphan-id"
    assert len(result) == 36  # uuid4 length


def test_dedupe_rows_by_id_last_wins() -> None:
    from src.ai.retrieval.doris import DorisVectorEmbedding
    rows = [
        DorisVectorEmbedding(id="a", content="first"),
        DorisVectorEmbedding(id="b", content="middle"),
        DorisVectorEmbedding(id="a", content="second"),
    ]
    deduped = _dedupe_rows_by_id(rows)
    assert len(deduped) == 2
    by_id = {r.id: r for r in deduped}
    assert by_id["a"].content == "second"
    assert by_id["b"].content == "middle"


# ── DDL builders ────────────────────────────────────────────────────


def test_build_create_table_ddl_uses_duplicate_key_for_inner_product() -> None:
    ddl = _build_create_table_ddl("emb_4", 4, 5, 1, "inner_product_duplicate")
    assert "DUPLICATE KEY(id)" in ddl
    assert "inner_product" in ddl
    assert "ARRAY<FLOAT>" in ddl
    assert "USING ANN" in ddl


def test_build_create_table_ddl_uses_unique_key_for_legacy() -> None:
    ddl = _build_create_table_ddl("emb_4", 4, 5, 1, "legacy")
    assert "UNIQUE KEY(id)" in ddl
    assert "cosine_distance" in ddl
    assert "enable_unique_key_merge_on_write" in ddl


def test_build_create_table_ddl_includes_dimension() -> None:
    ddl = _build_create_table_ddl("emb_768", 768, 5, 1, "inner_product_duplicate")
    assert '"dim"="768"' in ddl


# ── Domain helpers ─────────────────────────────────────────────────


def test_to_doris_vector_embedding_extracts_embedding() -> None:
    info = IndexInfo(
        id="row-1", content="hello", source_id="src-1", source_type=SourceType.CHUNK,
        chunk_id="chunk-1", knowledge_id="kid-1", knowledge_base_id="kb-1",
        tag_id="t-1", is_enabled=True,
    )
    params: IndexSaveParams = {"embedding": {"src-1": [1.0, 2.0, 3.0]}}
    emb = _to_doris_vector_embedding(info, params, "inner_product_duplicate")
    assert emb.id == "row-1"
    assert emb.chunk_id == "chunk-1"
    # normalized in inner_product_duplicate mode
    sq = sum(v * v for v in emb.embedding)
    assert pytest.approx(sq, abs=1e-6) == 1.0


def test_to_doris_vector_embedding_no_embedding_returns_empty() -> None:
    info = IndexInfo(source_id="src-1", chunk_id="chunk-1")
    emb = _to_doris_vector_embedding(info, {}, "inner_product_duplicate")
    assert emb.embedding == []


def test_calculate_storage_size_includes_vec_and_payload() -> None:
    from src.ai.retrieval.doris import DorisVectorEmbedding
    with_vec = DorisVectorEmbedding(content="hello", embedding=[1.0, 2.0, 3.0])
    without_vec = DorisVectorEmbedding(content="hello", embedding=[])
    assert _calculate_storage_size(with_vec) > _calculate_storage_size(without_vec)


# ── RetrieveEngine contract ────────────────────────────────────────


def test_engine_type_returns_doris() -> None:
    repo, _ = _new_repo()
    assert repo.engine_type() == RetrieverEngineType.DORIS


def test_support_returns_keywords_and_vector() -> None:
    repo, _ = _new_repo()
    assert set(repo.support()) == {RetrieverType.KEYWORDS, RetrieverType.VECTOR}


def test_estimate_storage_size_accumulates() -> None:
    repo, _ = _new_repo()
    infos = [
        IndexInfo(source_id="s1", content="hello"),
        IndexInfo(source_id="s2", content="world"),
    ]
    total = repo.estimate_storage_size(_CTX, infos, {"embedding": {"s1": [1.0], "s2": [2.0]}})
    assert total > 0


# ── save / batch_save ──────────────────────────────────────────────


async def test_save_delegates_to_batch_save(monkeypatch: pytest.MonkeyPatch) -> None:
    repo, _db = _new_repo()
    bs = AsyncMock(return_value=None)
    monkeypatch.setattr(repo, "batch_save", bs)
    info = IndexInfo(source_id="s1", chunk_id="c1")
    await repo.save(_CTX, info, {"embedding": {"s1": [1.0, 2.0]}})
    bs.assert_awaited_once()


async def test_batch_save_with_empty_list_is_noop() -> None:
    repo, db = _new_repo()
    await repo.batch_save(_CTX, [], {})
    assert db._cursor.executed == []


async def test_batch_save_writes_to_dimension_table() -> None:
    # ensureTable → SHOW COUNT(1) returns 1 (table exists)
    # insertRows → multi-row INSERT
    repo, db = _new_repo(script=[
        ("SELECT COUNT", (1,)),  # table exists for dim=4
    ])
    info = IndexInfo(source_id="src-1", chunk_id="c1", content="hello")
    await repo.batch_save(_CTX, [info], {"embedding": {"src-1": [1.0, 2.0, 3.0, 4.0]}})
    executed = db._cursor.executed
    # First statement was the existence check
    assert any("INFORMATION_SCHEMA.TABLES" in s.upper() for s, _ in executed)
    # Second statement was the INSERT
    inserts = [s for s, _ in executed if s.strip().upper().startswith("INSERT INTO")]
    assert len(inserts) == 1
    assert "weknora_embeddings_4" in inserts[0]


async def test_batch_save_skips_empty_embedding() -> None:
    repo, db = _new_repo()
    info = IndexInfo(source_id="src-1", chunk_id="c1")
    await repo.batch_save(_CTX, [info], {"embedding": {}})
    assert db._cursor.executed == []


async def test_batch_save_validates_embedding() -> None:
    repo, _db = _new_repo()
    info = IndexInfo(source_id="src-1", chunk_id="c1")
    with pytest.raises(ValueError, match="not finite"):
        await repo.batch_save(_CTX, [info], {"embedding": {"src-1": [float("nan")]}})


# ── retrieve ───────────────────────────────────────────────────────


async def test_retrieve_dispatches_by_type() -> None:
    repo, _ = _new_repo()
    monkey_calls: list[str] = []

    async def _kw(*_a: Any, **_k: Any) -> list[Any]:
        monkey_calls.append("kw")
        return []

    async def _vc(*_a: Any, **_k: Any) -> list[Any]:
        monkey_calls.append("vc")
        return []

    repo._keywords_retrieve = _kw  # type: ignore[method-assign]
    repo._vector_retrieve = _vc  # type: ignore[method-assign]
    await repo.retrieve(_CTX, RetrieveParams(retriever_type=RetrieverType.KEYWORDS))
    await repo.retrieve(_CTX, RetrieveParams(retriever_type=RetrieverType.VECTOR))
    assert monkey_calls == ["kw", "vc"]


async def test_retrieve_invalid_type_raises() -> None:
    repo, _ = _new_repo()
    with pytest.raises(ValueError, match="invalid retriever type"):
        await repo.retrieve(_CTX, RetrieveParams(retriever_type=RetrieverType.WEB_SEARCH))


async def test_vector_retrieve_returns_empty_when_table_missing() -> None:
    repo, _db = _new_repo(script=[("SELECT COUNT", (0,))])
    params = RetrieveParams(
        embedding=[1.0, 2.0], retriever_type=RetrieverType.VECTOR, top_k=5, threshold=0.0
    )
    results = await repo._vector_retrieve(_CTX, params)
    assert results == _build_retrieve_result([], RetrieverType.VECTOR)


async def test_vector_retrieve_runs_inner_product_query() -> None:
    repo, _db = _new_repo(script=[
        ("SELECT COUNT", (1,)),  # table exists
        ("SELECT", [(  # retrieve rows
            "row-1", "hello", "src-1", 0, "c-1", "k-1", "kb-1", "t-1", True, 0.95,
        )]),
    ])
    params = RetrieveParams(
        embedding=[1.0, 2.0, 3.0, 4.0],
        retriever_type=RetrieverType.VECTOR,
        top_k=5,
        threshold=0.0,
    )
    results = await repo._vector_retrieve(_CTX, params)
    assert len(results) == 1
    assert len(results[0].results) == 1
    item = results[0].results[0]
    assert item.id == "row-1"
    assert item.match_type == MatchType.EMBEDDING


async def test_keywords_retrieve_empty_query_returns_empty() -> None:
    repo, _ = _new_repo()
    results = await repo._keywords_retrieve(_CTX, RetrieveParams(query=""))
    assert results == _build_retrieve_result([], RetrieverType.KEYWORDS)


async def test_keywords_retrieve_merges_across_tables() -> None:
    repo, _db = _new_repo(script=[
        ("SELECT TABLE_NAME", [("weknora_embeddings_4",), ("weknora_embeddings_768",)]),
        ("SELECT", [(  # table 4 results
            "row-1", "hello", "src-1", 0, "c-1", "k-1", "kb-1", "t-1", True,
        )]),
        ("SELECT", [  # table 768 results
            ("row-2", "world", "src-2", 0, "c-2", "k-2", "kb-2", "t-2", False),
        ]),
    ])
    params = RetrieveParams(query="hello world", top_k=10, retriever_type=RetrieverType.KEYWORDS)
    results = await repo._keywords_retrieve(_CTX, params)
    assert len(results) == 1
    assert len(results[0].results) == 2


# ── delete_by_* ────────────────────────────────────────────────────


async def test_delete_by_chunk_id_list() -> None:
    repo, db = _new_repo()
    await repo.delete_by_chunk_id_list(_CTX, ["c1", "c2"], dimension=4, knowledge_type="")
    executed = db._cursor.executed
    assert any("DELETE FROM" in s.upper() and "chunk_id" in s.lower() for s, _ in executed)


async def test_delete_by_source_id_list() -> None:
    repo, db = _new_repo()
    await repo.delete_by_source_id_list(_CTX, ["s1"], dimension=4, knowledge_type="")
    assert any("source_id" in s.lower() for s, _ in db._cursor.executed)


async def test_delete_by_knowledge_id_list() -> None:
    repo, db = _new_repo()
    await repo.delete_by_knowledge_id_list(_CTX, ["k1"], dimension=4, knowledge_type="")
    assert any("knowledge_id" in s.lower() for s, _ in db._cursor.executed)


async def test_delete_with_empty_ids_is_noop() -> None:
    repo, db = _new_repo()
    await repo.delete_by_chunk_id_list(_CTX, [], dimension=4, knowledge_type="")
    assert db._cursor.executed == []


# ── copy_indices ───────────────────────────────────────────────────


async def test_copy_indices_empty_mapping_is_noop() -> None:
    repo, db = _new_repo()
    await repo.copy_indices(
        _CTX, "src-kb", {}, {}, "tgt-kb", dimension=4, knowledge_type=""
    )
    assert db._cursor.executed == []


async def test_copy_indices_copies_with_translated_ids() -> None:
    repo, db = _new_repo(script=[
        ("SELECT COUNT", (1,)),  # ensureTable
        ("SELECT", [(  # page 1: source rows
            "row-1", "content", "src-1", 0, "src-c-1", "src-k-1",
            "src-kb-1", "tag-1", True, b"[1.0,2.0,3.0,4.0]",
        )]),
    ])
    await repo.copy_indices(
        _CTX,
        "src-kb",
        {"src-k-1": "tgt-k-1"},
        {"src-c-1": "tgt-c-1"},
        "tgt-kb",
        dimension=4,
        knowledge_type="",
    )
    executed = db._cursor.executed
    assert any("INSERT INTO" in s.upper() for s, _ in executed)


# ── batch_update_chunk_* (rewrite path, default compat mode) ──────


async def test_batch_update_chunk_enabled_status_rewrite() -> None:
    repo, db = _new_repo(script=[
        ("SELECT TABLE_NAME", [("weknora_embeddings_4",)]),
        ("SELECT", [(  # load rows
            "row-1", "content", "src-1", 0, "c-1", "k-1", "kb-1", "tag-1", False,
            b"[1.0,2.0]",
        )]),
        ("SELECT COUNT", (1,)),
        ("DELETE", (1,)),  # delete by id
        ("INSERT", (1,)),  # insert
    ])
    await repo.batch_update_chunk_enabled_status(_CTX, {"c-1": True})
    executed = db._cursor.executed
    assert any("SELECT" in s.upper() for s, _ in executed)


async def test_batch_update_chunk_tag_id_rewrite() -> None:
    repo, db = _new_repo(script=[
        ("SELECT TABLE_NAME", [("weknora_embeddings_4",)]),
        ("SELECT", [(
            "row-1", "content", "src-1", 0, "c-1", "k-1", "kb-1", "old-tag", True,
            b"[1.0,2.0]",
        )]),
        ("SELECT COUNT", (1,)),
        ("DELETE", (1,)),
        ("INSERT", (1,)),
    ])
    await repo.batch_update_chunk_tag_id(_CTX, {"c-1": "new-tag"})
    assert any("SELECT" in s.upper() for s, _ in db._cursor.executed)


# ── batch_update_chunk_* (legacy compat mode) ──────────────────────


async def test_batch_update_chunk_enabled_status_legacy_uses_stream_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, db = _new_repo()
    repo._compat_mode_resolved = "legacy"
    # lookupChunkRowKeys: listEmbeddingTables + SELECT per table
    db._cursor._script = [
        ("SELECT TABLE_NAME", [("weknora_embeddings_4",)]),
        ("SELECT", [("row-1", "c-1")]),  # lookup
    ]
    partial_update = AsyncMock(return_value=None)
    monkeypatch.setattr(repo, "_partial_update_rows", partial_update)
    await repo.batch_update_chunk_enabled_status(_CTX, {"c-1": True})
    partial_update.assert_awaited()


# ── Constructor ────────────────────────────────────────────────────


def test_new_doris_retrieve_engine_repository_returns_repo() -> None:
    repo = new_doris_retrieve_engine_repository(
        db=_FakeDB([]),
        fe_http_base="http://doris-fe:8030/",
        username="root",
        password="",
        database="test_db",
        index_cfg=None,
    )
    assert isinstance(repo, DorisRepository)
    # Trailing slash is trimmed.
    assert repo._fe_http_base == "http://doris-fe:8030"


def test_new_doris_retrieve_engine_repository_reads_index_config() -> None:
    repo = new_doris_retrieve_engine_repository(
        db=_FakeDB([]),
        fe_http_base="http://doris-fe:8030",
        username="root",
        password="",
        database="test_db",
        index_cfg=IndexConfig(collection_prefix="custom_base", buckets_num=20),
    )
    assert repo._table_base_name == "custom_base"
    assert repo._buckets_num == 20


# ── WHERE builder ──────────────────────────────────────────────────


def test_build_base_filter_includes_is_enabled() -> None:
    wb = _build_base_filter(RetrieveParams())
    where, args = wb.build()
    assert "is_enabled" in where.lower()
    assert 1 in args


def test_build_base_filter_includes_optional_filters() -> None:
    wb = _build_base_filter(RetrieveParams(
        knowledge_base_ids=["kb-1", "kb-2"],
        knowledge_ids=["k-1"],
        tag_ids=["t-1"],
        exclude_knowledge_ids=["k-ex"],
        exclude_chunk_ids=["c-ex"],
    ))
    where, args = wb.build()
    assert "knowledge_base_id" in where
    assert "tag_id" in where
    assert "NOT IN" in where
    assert "kb-1" in args
    assert "k-ex" in args
