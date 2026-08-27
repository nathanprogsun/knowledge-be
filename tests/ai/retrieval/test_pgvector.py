"""Tests for the Postgres + pgvector retrieval engine repository.

Mocks the session factory so no real Postgres connection is required. Each
method is exercised against a canned result chain to verify the SQL
shape, bind params, and the wire result transformation.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.ai.embedding import TaskContext
from src.ai.retrieval.base import Database
from src.ai.retrieval.pgvector import (
    PostgresRetrieveEngineRepository,
    new_postgres_retrieve_engine_repository,
)
from src.ai.retrieval.types import (
    IndexInfo,
    IndexSaveParams,
    MatchType,
    RetrieverEngineType,
    RetrieverType,
    SourceType,
)
from src.common.exception import ValidationError

_CTX = TaskContext()


# ── Mocks ──────────────────────────────────────────────────────────────


class _ChainResult:
    """Stand-in for ``AsyncResult.mappings().all()`` chain."""

    def __init__(self, rows: list[Mapping[str, Any]]) -> None:
        self._rows = rows

    def mappings(self) -> _ChainResult:
        return self

    def all(self) -> list[Mapping[str, Any]]:
        return list(self._rows)


class _FakeTransaction:
    """Async context manager returned by ``session.begin()``."""

    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0

    async def __aenter__(self) -> _FakeTransaction:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None


class _FakeSession:
    """AsyncSession stand-in recording execute calls and committing."""

    def __init__(self) -> None:
        self.executes: list[tuple[Any, Any]] = []
        self.commits = 0
        self.rollback_count = 0
        self._rows: list[Mapping[str, Any]] = []
        self._remaining_results: list[_ChainResult] = []
        self._begin_exc: Exception | None = None
        self.transaction = _FakeTransaction()

    def set_rows(self, rows: list[Mapping[str, Any]]) -> None:
        """Configure the single result returned by the next ``execute`` call."""
        self._rows = rows

    def queue_results(self, results: Sequence[Sequence[Mapping[str, Any]]]) -> None:
        """Queue a sequence of results, one per ``execute`` call."""
        self._remaining_results = [_ChainResult(list(r)) for r in results]

    def set_begin_exception(self, exc: Exception | None) -> None:
        """Force ``session.begin()`` to raise (simulates HNSW GUC failure)."""
        self._begin_exc = exc

    async def execute(self, stmt: Any, params: Any = None) -> _ChainResult:
        self.executes.append((stmt, params))
        if self._remaining_results:
            return self._remaining_results.pop(0)
        return _ChainResult(self._rows)

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        self.rollback_count += 1

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def begin(self) -> _FakeTransaction:
        if self._begin_exc is not None:
            raise self._begin_exc
        return self.transaction


def _make_factory(
    rows: list[Mapping[str, Any]] | None = None,
) -> tuple[async_sessionmaker[AsyncSession], _FakeSession]:
    """Build a mock session factory yielding a single shared fake session."""
    session = _FakeSession()
    if rows is not None:
        session.set_rows(rows)
    factory = cast(
        async_sessionmaker[AsyncSession],
        MagicMock(return_value=session),
    )
    return factory, session


def _make_repo(
    session_result_chain: list[list[Mapping[str, Any]]] | None = None,
) -> tuple[PostgresRetrieveEngineRepository, _FakeSession]:
    """Build a repository backed by a mocked session factory."""
    factory, session = _make_factory()
    if session_result_chain is not None:
        session.queue_results(session_result_chain)
    repo = PostgresRetrieveEngineRepository(factory)
    return repo, session


def _emb() -> IndexInfo:
    """Build a sample IndexInfo for save/estimate tests."""
    return IndexInfo(
        id="row-1",
        content="hello world",
        source_id="src-1",
        source_type=SourceType.CHUNK,
        chunk_id="chunk-1",
        knowledge_id="k-1",
        knowledge_base_id="kb-1",
        tag_id="",
        is_enabled=True,
        is_recommended=False,
    )


# ── Domain / structural ───────────────────────────────────────────────


def test_engine_type_returns_postgres() -> None:
    repo, _ = _make_repo()
    assert repo.engine_type() == RetrieverEngineType.POSTGRES


def test_support_returns_keywords_and_vector() -> None:
    repo, _ = _make_repo()
    assert repo.support() == [RetrieverType.KEYWORDS, RetrieverType.VECTOR]


def test_repository_satisfies_protocol() -> None:
    factory, _ = _make_factory()
    repo = PostgresRetrieveEngineRepository(factory)
    required_methods = [
        "engine_type",
        "retrieve",
        "support",
        "save",
        "batch_save",
        "estimate_storage_size",
        "delete_by_chunk_id_list",
        "delete_by_source_id_list",
        "delete_by_knowledge_id_list",
        "copy_indices",
        "batch_update_chunk_enabled_status",
        "batch_update_chunk_tag_id",
    ]
    for method in required_methods:
        assert hasattr(repo, method), f"missing method: {method}"
        assert callable(getattr(repo, method))


# ── Constructor ───────────────────────────────────────────────────────


def test_new_postgres_retrieve_engine_repository_uses_session_factory() -> None:
    session = _FakeSession()
    factory = cast(
        async_sessionmaker[AsyncSession],
        MagicMock(return_value=session),
    )

    class _Handle:
        session_factory: async_sessionmaker[AsyncSession] = factory

    handle = cast(Database, _Handle())
    repo = new_postgres_retrieve_engine_repository(handle)
    assert isinstance(repo, PostgresRetrieveEngineRepository)
    assert repo._session_factory is factory


# ── Storage size ──────────────────────────────────────────────────────


def test_estimate_storage_size_with_dim_zero() -> None:
    repo, _ = _make_repo()
    info = _emb()
    # No embedding map → dimension defaults to 0 → vector_size 0
    assert repo.estimate_storage_size(_CTX, [info], {}) == len(b"hello world") + 200


def test_estimate_storage_size_with_embedding_uses_byte_length() -> None:
    repo, _ = _make_repo()
    info = _emb()
    params: IndexSaveParams = {"embedding": {"src-1": [0.1, 0.2, 0.3]}}
    # dimension=3 → vector_size=6, index_overhead=12, content=11, metadata=200
    expected = 11 + 6 + 200 + 12
    assert repo.estimate_storage_size(_CTX, [info], params) == expected


def test_estimate_storage_size_sums_multiple_index_infos() -> None:
    repo, _ = _make_repo()
    a = IndexInfo(content="aaa", source_id="a")
    b = IndexInfo(content="bbbbb", source_id="b")
    # Each row: content_bytes + 200
    expected = 3 + 200 + 5 + 200
    assert repo.estimate_storage_size(_CTX, [a, b], {}) == expected


# ── Save / BatchSave ──────────────────────────────────────────────────


async def test_save_executes_insert_and_commits() -> None:
    repo, session = _make_repo()
    info = _emb()
    params: IndexSaveParams = {"embedding": {"src-1": [0.1, 0.2, 0.3]}}
    await repo.save(_CTX, info, params)
    assert session.commits == 1
    assert len(session.executes) == 1
    stmt, params_passed = session.executes[0]
    sql_text = str(stmt)
    assert "INSERT INTO embeddings" in sql_text
    assert "CAST(:embedding AS halfvec)" in sql_text
    assert "ON CONFLICT" not in sql_text
    assert params_passed["source_id"] == "src-1"
    assert params_passed["dimension"] == 3
    assert params_passed["embedding"] == "[0.1,0.2,0.3]"
    assert params_passed["is_enabled"] is True


async def test_save_without_embedding_uses_zero_dimension() -> None:
    repo, session = _make_repo()
    info = _emb()
    await repo.save(_CTX, info, {})
    _, params_passed = session.executes[0]
    assert params_passed["dimension"] == 0
    assert params_passed["embedding"] == "[]"


async def test_batch_save_executes_batch_insert_with_on_conflict() -> None:
    repo, session = _make_repo()
    items = [
        IndexInfo(content="a", source_id="a", chunk_id="a"),
        IndexInfo(content="b", source_id="b", chunk_id="b"),
    ]
    params: IndexSaveParams = {
        "embedding": {"a": [0.1], "b": [0.2]},
    }
    await repo.batch_save(_CTX, items, params)
    assert session.commits == 1
    assert len(session.executes) == 1
    stmt, params_passed = session.executes[0]
    sql_text = str(stmt)
    assert "INSERT INTO embeddings" in sql_text
    assert "ON CONFLICT DO NOTHING" in sql_text
    assert isinstance(params_passed, list)
    assert len(params_passed) == 2


async def test_batch_save_empty_list_is_noop() -> None:
    repo, session = _make_repo()
    await repo.batch_save(_CTX, [], {})
    assert session.commits == 0
    assert session.executes == []


# ── Retrieve dispatch ─────────────────────────────────────────────────


async def test_retrieve_invalid_type_raises_value_error() -> None:
    """A valid-but-unsupported retriever type (e.g. WEB_SEARCH) raises ValueError."""
    repo, _ = _make_repo()
    with pytest.raises(ValidationError, match="invalid retriever type"):
        await repo.retrieve(
            _CTX,
            _params(retriever_type=RetrieverType.WEB_SEARCH),
        )


def _params(
    *,
    embedding: list[float] | None = None,
    knowledge_base_ids: list[str] | None = None,
    knowledge_ids: list[str] | None = None,
    tag_ids: list[str] | None = None,
    threshold: float = 0.0,
    top_k: int = 5,
    retriever_type: RetrieverType = RetrieverType.VECTOR,
) -> Any:
    """Build a RetrieveParams with sensible defaults."""
    from src.ai.retrieval.types import RetrieveParams

    return RetrieveParams(
        query="",
        embedding=embedding if embedding is not None else [],
        knowledge_base_ids=knowledge_base_ids or [],
        knowledge_ids=knowledge_ids or [],
        tag_ids=tag_ids or [],
        exclude_knowledge_ids=[],
        exclude_chunk_ids=[],
        top_k=top_k,
        threshold=threshold,
        knowledge_type="",
        additional_params={},
        retriever_type=retriever_type,
    )


async def test_retrieve_keywords_routes_to_keywords_retrieve() -> None:
    sample = _ChainResult(
        [
            {
                "id": 7,
                "content": "hit",
                "source_id": "src-7",
                "source_type": 0,
                "chunk_id": "c-7",
                "knowledge_id": "k-7",
                "knowledge_base_id": "kb-7",
                "tag_id": "",
                "score": 0.9,
                "is_enabled": True,
            }
        ]
    )
    factory, session = _make_factory()
    session._remaining_results = [sample]
    repo = PostgresRetrieveEngineRepository(factory)
    result = await repo.retrieve(_CTX, _params(retriever_type=RetrieverType.KEYWORDS))
    assert len(result) == 1
    results = result[0].results
    assert len(results) == 1
    assert results[0].id == "7"
    assert results[0].chunk_id == "c-7"
    assert results[0].match_type == MatchType.KEYWORDS
    sql_text = str(session.executes[0][0])
    assert "paradedb.score" in sql_text
    assert "content ||| :query" in sql_text
    assert "ORDER BY score DESC" in sql_text


async def test_retrieve_vector_routes_to_vector_retrieve() -> None:
    rows = [
        {
            "id": 1,
            "content": "v",
            "source_id": "src-1",
            "source_type": 1,
            "chunk_id": "c-1",
            "knowledge_id": "k-1",
            "knowledge_base_id": "kb-1",
            "tag_id": "t-1",
            "score": 0.95,
            "is_enabled": True,
        }
    ]
    repo, session = _make_repo()
    # Three ``execute`` calls: two SET LOCAL + one query. Queue a result
    # for each so the query call returns the canned row.
    session.queue_results([[], [], rows])
    result = await repo.retrieve(
        _CTX,
        _params(
            embedding=[0.1, 0.2, 0.3],
            knowledge_base_ids=["kb-1"],
            knowledge_ids=["k-1"],
            tag_ids=["t-1"],
            threshold=0.5,
            top_k=3,
        ),
    )
    assert len(result) == 1
    results = result[0].results
    assert len(results) == 1
    assert results[0].match_type == MatchType.EMBEDDING
    assert results[0].score == 0.95
    assert results[0].source_type == SourceType.PASSAGE
    # Two SET LOCAL executes + the query execute
    assert len(session.executes) == 3
    set_local_calls = session.executes[:2]
    for stmt, _ in set_local_calls:
        sql_text = str(stmt)
        assert "SET LOCAL" in sql_text
    set_local_statements = [str(session.executes[i][0]) for i in range(2)]
    assert any("hnsw.ef_search" in s for s in set_local_statements)
    assert any("hnsw.iterative_scan" in s for s in set_local_statements)
    query_stmt, query_params = session.executes[2]
    query_sql = str(query_stmt)
    assert "embedding::halfvec(3)" in query_sql
    assert "<=>" in query_sql
    assert query_params["dimension"] == 3
    assert query_params["embedding"] == "[0.1,0.2,0.3]"
    assert query_params["is_enabled"] is True
    assert query_params["distance_threshold"] == 0.5  # 1 - 0.5
    assert query_params["final_limit"] == 3


async def test_vector_retrieve_expands_topk_below_minimum() -> None:
    """When top_k * 2 < 100, the subquery budget is floored to 100."""
    repo, session = _make_repo()
    session.queue_results([[], [], []])
    await repo.retrieve(
        _CTX,
        _params(embedding=[0.0, 0.0], top_k=10),
    )
    _, params_passed = session.executes[2]
    assert params_passed["subquery_limit"] == 100  # 10*2=20 < 100 → floored
    assert params_passed["final_limit"] == 10


async def test_vector_retrieve_caps_expanded_topk_above_maximum() -> None:
    """When top_k * 2 > 200, the subquery budget is capped at 200."""
    repo, session = _make_repo()
    session.queue_results([[], [], []])
    await repo.retrieve(
        _CTX,
        _params(embedding=[0.0, 0.0], top_k=200),
    )
    _, params_passed = session.executes[2]
    assert params_passed["subquery_limit"] == 200  # 200*2=400 > 200 → capped
    assert params_passed["final_limit"] == 200


async def test_vector_retrieve_falls_back_when_hnsw_guc_unknown() -> None:
    """The transaction-level ``SET LOCAL`` raises; the query is retried."""
    session = _FakeSession()
    session.set_begin_exception(RuntimeError("unrecognized configuration parameter hnsw.ef_search"))
    # Results queued for the retry path (post-fallback SELECT); the inner
    # transaction's execute was never reached.
    session.queue_results(
        [
            [
                {
                    "id": 1,
                    "content": "x",
                    "source_id": "s",
                    "source_type": 0,
                    "chunk_id": "c",
                    "knowledge_id": "k",
                    "knowledge_base_id": "kb",
                    "tag_id": "",
                    "score": 0.5,
                    "is_enabled": True,
                }
            ]
        ]
    )
    factory = cast(
        async_sessionmaker[AsyncSession],
        MagicMock(return_value=session),
    )
    repo = PostgresRetrieveEngineRepository(factory)
    await repo.retrieve(
        _CTX,
        _params(embedding=[0.0, 0.0], top_k=3),
    )
    # SET LOCAL never ran; the fallback executed the query once.
    assert len(session.executes) == 1
    sql_text = str(session.executes[0][0])
    assert "embedding::halfvec" in sql_text


async def test_vector_retrieve_propagates_non_hnsw_exception() -> None:
    """Exceptions unrelated to HNSW GUCs are re-raised."""
    session = _FakeSession()
    session.set_begin_exception(ValueError("connection failed"))
    factory = cast(
        async_sessionmaker[AsyncSession],
        MagicMock(return_value=session),
    )
    repo = PostgresRetrieveEngineRepository(factory)
    with pytest.raises(ValueError, match="connection failed"):
        await repo.retrieve(_CTX, _params(embedding=[0.0, 0.0]))


async def test_vector_retrieve_caps_results_to_top_k() -> None:
    """The outer LIMIT is enforced after the subquery returns."""
    rows = [
        {
            "id": i,
            "content": "x",
            "source_id": "s",
            "source_type": 0,
            "chunk_id": "c",
            "knowledge_id": "k",
            "knowledge_base_id": "kb",
            "tag_id": "",
            "score": 0.5,
            "is_enabled": True,
        }
        for i in range(3)
    ]
    repo, session = _make_repo()
    # Three execute calls per query: 2 SET LOCAL + 1 query. Queue the
    # actual rows at the third position so the query returns 3 rows.
    session.queue_results([[], [], rows])
    result = await repo.retrieve(
        _CTX,
        _params(embedding=[0.0, 0.0], top_k=2),
    )
    assert len(result) == 1
    assert len(result[0].results) == 2


# ── Deletes ───────────────────────────────────────────────────────────


async def test_delete_by_chunk_id_list() -> None:
    repo, session = _make_repo()
    await repo.delete_by_chunk_id_list(_CTX, ["c-1", "c-2"], 3, "doc")
    assert session.commits == 1
    stmt, params_passed = session.executes[0]
    sql_text = str(stmt)
    assert "DELETE FROM embeddings" in sql_text
    assert "chunk_id IN" in sql_text
    assert params_passed["ids"] == ["c-1", "c-2"]


async def test_delete_by_chunk_id_list_empty_is_noop() -> None:
    repo, session = _make_repo()
    await repo.delete_by_chunk_id_list(_CTX, [], 3, "doc")
    assert session.executes == []


async def test_delete_by_source_id_list() -> None:
    repo, session = _make_repo()
    await repo.delete_by_source_id_list(_CTX, ["s-1"], 3, "doc")
    stmt, params_passed = session.executes[0]
    assert "source_id IN" in str(stmt)
    assert params_passed["ids"] == ["s-1"]


async def test_delete_by_source_id_list_empty_is_noop() -> None:
    repo, session = _make_repo()
    await repo.delete_by_source_id_list(_CTX, [], 3, "doc")
    assert session.executes == []


async def test_delete_by_knowledge_id_list() -> None:
    repo, session = _make_repo()
    await repo.delete_by_knowledge_id_list(_CTX, ["k-1"], 3, "doc")
    stmt, params_passed = session.executes[0]
    assert "knowledge_id IN" in str(stmt)
    assert params_passed["ids"] == ["k-1"]


async def test_delete_by_knowledge_id_list_empty_is_noop() -> None:
    repo, session = _make_repo()
    await repo.delete_by_knowledge_id_list(_CTX, [], 3, "doc")
    assert session.executes == []


# ── Copy indices ──────────────────────────────────────────────────────


async def test_copy_indices_empty_is_noop() -> None:
    repo, session = _make_repo()
    await repo.copy_indices(
        _CTX,
        source_knowledge_base_id="src-kb",
        source_to_target_kb_id_map={"k-1": "k-2"},
        source_to_target_chunk_id_map={},
        target_knowledge_base_id="tgt-kb",
        dimension=3,
        knowledge_type="doc",
    )
    assert session.executes == []


async def test_copy_indices_paginates_and_inserts() -> None:
    """First batch returns rows; loop terminates when the next batch is empty."""
    source_rows = [
        {
            "content": "hi",
            "source_id": "c-1",
            "source_type": 0,
            "chunk_id": "c-1",
            "knowledge_id": "k-1",
            "dimension": 2,
            "embedding": "[0.1,0.2]",
        },
        {
            "content": "bye",
            "source_id": "c-2-q",
            "source_type": 0,
            "chunk_id": "c-2",
            "knowledge_id": "k-1",
            "dimension": 2,
            "embedding": "[0.3,0.4]",
        },
    ]
    empty: list[Mapping[str, Any]] = []
    repo, session = _make_repo()
    session.queue_results([source_rows, empty])
    chunk_map = {"c-1": "tc-1", "c-2": "tc-2"}
    kb_map = {"k-1": "tk-1"}
    await repo.copy_indices(
        _CTX,
        source_knowledge_base_id="src-kb",
        source_to_target_kb_id_map=kb_map,
        source_to_target_chunk_id_map=chunk_map,
        target_knowledge_base_id="tgt-kb",
        dimension=3,
        knowledge_type="doc",
    )
    # 1 SELECT + 1 INSERT batch
    assert len(session.executes) == 2
    select_stmt, select_params = session.executes[0]
    assert "SELECT content, source_id" in str(select_stmt)
    assert "knowledge_base_id = :source_kb_id" in str(select_stmt)
    assert select_params["source_kb_id"] == "src-kb"
    assert select_params["limit"] == 500
    assert select_params["offset"] == 0
    insert_stmt, insert_params = session.executes[1]
    assert "INSERT INTO embeddings" in str(insert_stmt)
    assert "ON CONFLICT DO NOTHING" in str(insert_stmt)
    assert isinstance(insert_params, list)
    # c-1 → source_id == chunk_id → target source_id = tc-1
    # c-2-q → prefix matches c-2 → target source_id = "tc-2-q"
    assert insert_params[0]["source_id"] == "tc-1"
    assert insert_params[0]["chunk_id"] == "tc-1"
    assert insert_params[0]["knowledge_id"] == "tk-1"
    assert insert_params[0]["knowledge_base_id"] == "tgt-kb"
    assert insert_params[0]["embedding"] == "[0.1,0.2]"
    assert insert_params[1]["source_id"] == "tc-2-q"
    assert insert_params[1]["chunk_id"] == "tc-2"


async def test_copy_indices_falls_back_to_uuid_for_unknown_source_id() -> None:
    """When the source_id doesn't match the chunk_id pattern, a UUID is generated."""
    source_rows = [
        {
            "content": "x",
            "source_id": "totally-different",
            "source_type": 0,
            "chunk_id": "c-1",
            "knowledge_id": "k-1",
            "dimension": 1,
            "embedding": "[0.0]",
        }
    ]
    repo, session = _make_repo()
    session.queue_results([source_rows, []])
    await repo.copy_indices(
        _CTX,
        "src-kb",
        {"k-1": "tk-1"},
        {"c-1": "tc-1"},
        "tgt-kb",
        1,
        "doc",
    )
    insert_params = session.executes[1][1]
    assert isinstance(insert_params, list)
    # UUID validation
    uuid.UUID(insert_params[0]["source_id"])


async def test_copy_indices_skips_rows_without_chunk_mapping() -> None:
    """Rows whose chunk_id is not in the mapping are dropped during transform."""
    source_rows = [
        {
            "content": "x",
            "source_id": "c-1",
            "source_type": 0,
            "chunk_id": "c-1",
            "knowledge_id": "k-1",
            "dimension": 1,
            "embedding": "[0.0]",
        },
        {
            "content": "y",
            "source_id": "unmapped",
            "source_type": 0,
            "chunk_id": "c-unknown",
            "knowledge_id": "k-1",
            "dimension": 1,
            "embedding": "[0.0]",
        },
    ]
    repo, session = _make_repo()
    session.queue_results([source_rows, []])
    await repo.copy_indices(
        _CTX,
        "src-kb",
        {"k-1": "tk-1"},
        {"c-1": "tc-1"},
        "tgt-kb",
        1,
        "doc",
    )
    insert_params = session.executes[1][1]
    assert isinstance(insert_params, list)
    assert len(insert_params) == 1
    assert insert_params[0]["chunk_id"] == "tc-1"


async def test_copy_indices_no_insert_when_no_chunk_matches() -> None:
    repo, session = _make_repo()
    session.queue_results(
        [
            [
                {
                    "content": "x",
                    "source_id": "c-1",
                    "source_type": 0,
                    "chunk_id": "c-1",
                    "knowledge_id": "k-1",
                    "dimension": 1,
                    "embedding": "[0.0]",
                }
            ],
            [],
        ]
    )
    await repo.copy_indices(
        _CTX,
        "src-kb",
        {"k-1": "tk-1"},
        {"c-other": "tc-other"},  # c-1 not in mapping
        "tgt-kb",
        1,
        "doc",
    )
    # Only the SELECT — no INSERT batch because no rows survived transform.
    assert len(session.executes) == 1


# ── Batch updates ─────────────────────────────────────────────────────


async def test_batch_update_chunk_enabled_status_groups_enabled_and_disabled() -> None:
    repo, session = _make_repo()
    await repo.batch_update_chunk_enabled_status(
        _CTX,
        {"c-1": True, "c-2": False, "c-3": True},
    )
    assert session.commits == 1
    assert len(session.executes) == 2
    enabled_stmt, enabled_params = session.executes[0]
    disabled_stmt, disabled_params = session.executes[1]
    assert "is_enabled = TRUE" in str(enabled_stmt)
    assert "chunk_id IN" in str(enabled_stmt)
    assert enabled_params["ids"] == ["c-1", "c-3"]
    assert "is_enabled = FALSE" in str(disabled_stmt)
    assert disabled_params["ids"] == ["c-2"]


async def test_batch_update_chunk_enabled_status_empty_is_noop() -> None:
    repo, session = _make_repo()
    await repo.batch_update_chunk_enabled_status(_CTX, {})
    assert session.commits == 0
    assert session.executes == []


async def test_batch_update_chunk_tag_id_groups_by_tag() -> None:
    repo, session = _make_repo()
    await repo.batch_update_chunk_tag_id(
        _CTX,
        {"c-1": "t-A", "c-2": "t-B", "c-3": "t-A"},
    )
    assert session.commits == 1
    assert len(session.executes) == 2
    tag_params = {p[1]["tag_id"]: p[1]["ids"] for p in session.executes}
    assert tag_params["t-A"] == ["c-1", "c-3"]
    assert tag_params["t-B"] == ["c-2"]


async def test_batch_update_chunk_tag_id_empty_is_noop() -> None:
    repo, session = _make_repo()
    await repo.batch_update_chunk_tag_id(_CTX, {})
    assert session.commits == 0
    assert session.executes == []


# ── Bind-param / SQL helpers ──────────────────────────────────────────


def test_build_insert_stmt_text_adds_on_conflict_when_requested() -> None:
    """Internal helper smoke test: verifies the SQL builder output shape."""
    from src.ai.retrieval.pgvector import _build_insert_stmt_text

    stmt = _build_insert_stmt_text(
        ("source_id", "embedding", "is_enabled"),
        on_conflict_do_nothing=True,
    )
    assert stmt.startswith("INSERT INTO embeddings")
    assert "source_id, embedding, is_enabled" in stmt
    assert "source_id" in stmt
    assert "CAST(:embedding AS halfvec)" in stmt
    assert "is_enabled" in stmt
    assert stmt.endswith(" ON CONFLICT DO NOTHING")


def test_build_insert_stmt_text_omits_on_conflict_by_default() -> None:
    from src.ai.retrieval.pgvector import _build_insert_stmt_text

    stmt = _build_insert_stmt_text(("a",), on_conflict_do_nothing=False)
    assert "ON CONFLICT" not in stmt


def test_format_halfvec_produces_bracketed_csv() -> None:
    from src.ai.retrieval.pgvector import _format_halfvec

    assert _format_halfvec([0.1, 0.2, 0.3]) == "[0.1,0.2,0.3]"


def test_transform_source_id_regular_chunk() -> None:
    from src.ai.retrieval.pgvector import _transform_source_id

    assert _transform_source_id("c-1", "c-1", "tc-1") == "tc-1"


def test_transform_source_id_generated_question() -> None:
    from src.ai.retrieval.pgvector import _transform_source_id

    assert _transform_source_id("c-1-qA", "c-1", "tc-1") == "tc-1-qA"


def test_transform_source_id_unrelated_uses_uuid() -> None:
    """When the source_id doesn't match any pattern, the helper returns a UUID."""
    from src.ai.retrieval.pgvector import _transform_source_id

    out = _transform_source_id("different", "c-1", "tc-1")
    uuid.UUID(out)
