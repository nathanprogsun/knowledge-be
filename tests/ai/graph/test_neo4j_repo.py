"""Unit tests for the Neo4j graph repository.

The neo4j driver is mocked throughout — no real database is contacted.
Covered: model shape, label computation, environment gating, the
connect-with-retry bootstrap, and the add/del/search write paths.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from neo4j import AsyncGraphDatabase

from src.ai.graph import neo4j_repo
from src.ai.graph.neo4j_repo import (
    _NODE_IMPORT_QUERY,
    _REL_IMPORT_QUERY,
    Neo4jConnectionError,
    Neo4jRepository,
    build_graph_repository,
    connect_driver,
    graph_enabled,
    remove_hyphen,
)
from src.ai.graph.types import (
    GraphData,
    GraphNode,
    GraphRelation,
    NameSpace,
    RetrieveGraphRepository,
)

# ── Model shape ────────────────────────────────────────────────────


def test_graph_data_serializes_with_upstream_json_names() -> None:
    """JSON field names match the upstream contract exactly."""
    graph = GraphData(
        text="t",
        node=[GraphNode(name="a", chunks=["c"], attributes=["x"])],
        relation=[GraphRelation(node1="a", node2="b", type="HAS")],
    )
    assert graph.model_dump() == {
        "text": "t",
        "node": [{"name": "a", "chunks": ["c"], "attributes": ["x"]}],
        "relation": [{"node1": "a", "node2": "b", "type": "HAS"}],
    }


def test_namespace_labels_uses_non_empty_segments() -> None:
    """``labels()`` mirrors the upstream scope semantics."""
    assert NameSpace().labels() == []
    assert NameSpace(knowledge_base="kb").labels() == ["kb"]
    assert NameSpace(knowledge_base="kb", knowledge="k").labels() == ["kb", "k"]


# ── Label helpers ──────────────────────────────────────────────────


def test_remove_hyphen_replaces_all_hyphens() -> None:
    assert remove_hyphen("kb-1") == "kb_1"
    assert remove_hyphen("a-b-c") == "a_b_c"
    assert remove_hyphen("plain") == "plain"


def test_repository_labels_prefix_and_normalize_hyphens() -> None:
    repo = Neo4jRepository(driver=None)
    namespace = NameSpace(knowledge_base="kb-1", knowledge="know-1")
    assert repo.labels(namespace) == ["ENTITYkb_1", "ENTITYknow_1"]
    assert repo._label_expr(namespace) == "ENTITYkb_1:ENTITYknow_1"


def test_repository_empty_namespace_yields_empty_label_expr() -> None:
    repo = Neo4jRepository(driver=None)
    assert repo.labels(NameSpace()) == []
    assert repo._label_expr(NameSpace()) == ""


# ── Environment gating ─────────────────────────────────────────────


def test_graph_enabled_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEO4J_ENABLE", raising=False)
    assert graph_enabled() is False
    monkeypatch.setenv("NEO4J_ENABLE", "true")
    assert graph_enabled() is True
    monkeypatch.setenv("NEO4J_ENABLE", "True")
    assert graph_enabled() is True
    monkeypatch.setenv("NEO4J_ENABLE", "yes")
    assert graph_enabled() is False


# ── Disabled repository (driver=None) ─────────────────────────────


async def test_disabled_repository_implements_protocol() -> None:
    assert isinstance(Neo4jRepository(driver=None), RetrieveGraphRepository)


async def test_add_graph_is_noop_when_disabled() -> None:
    repo = Neo4jRepository(driver=None)
    await repo.add_graph(NameSpace(knowledge="k"), [GraphData()])


async def test_del_graph_is_noop_when_disabled() -> None:
    repo = Neo4jRepository(driver=None)
    await repo.del_graph([NameSpace(knowledge="k")])


async def test_search_node_returns_none_when_disabled() -> None:
    repo = Neo4jRepository(driver=None)
    assert await repo.search_node(NameSpace(knowledge="k"), ["a"]) is None


async def test_close_is_noop_when_disabled() -> None:
    repo = Neo4jRepository(driver=None)
    await repo.close()


# ── Fakes ──────────────────────────────────────────────────────────


class _FakeResult:
    """AsyncIterable of records; also supports ``consume``."""

    def __init__(self, records: list[dict[str, object]] | None = None) -> None:
        self._records = records or []
        self.consumed = False

    async def consume(self) -> None:
        self.consumed = True

    def __aiter__(self) -> AsyncIterator[dict[str, object]]:
        async def _iterate() -> AsyncGenerator[dict[str, object], None]:
            for record in self._records:
                yield record

        return _iterate()


class _FakeTx:
    def __init__(self, result: _FakeResult | None = None) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self._result = result or _FakeResult()

    async def run(self, query: str, parameters: dict[str, object]) -> _FakeResult:
        self.calls.append((query, parameters))
        return self._result


class _FakeSession:
    def __init__(self, tx: _FakeTx) -> None:
        self._tx = tx
        self.closed = False

    async def execute_write(self, fn: Callable[..., Awaitable[object]], *args: object) -> object:
        return await fn(self._tx, *args)

    async def execute_read(self, fn: Callable[..., Awaitable[object]], *args: object) -> object:
        return await fn(self._tx, *args)

    async def close(self) -> None:
        self.closed = True


def _make_repo(tx: _FakeTx) -> tuple[Neo4jRepository, Mock]:
    """Repository backed by a mocked driver + fake session.

    ``driver.session()`` is synchronous in the real async driver, so the
    fake must be a plain ``Mock``; ``close`` stays async.
    """
    session = _FakeSession(tx)
    driver = AsyncMock()
    driver.session = Mock(return_value=session)
    repo = Neo4jRepository(driver=driver)
    return repo, driver


# ── add_graph ──────────────────────────────────────────────────────


async def test_add_graph_writes_nodes_and_rels() -> None:
    tx = _FakeTx()
    repo, driver = _make_repo(tx)
    namespace = NameSpace(knowledge_base="kb-1", knowledge="know-1")
    graph = GraphData(
        node=[
            GraphNode(name="a", chunks=["c1"], attributes=["x1"]),
            GraphNode(name="b"),
        ],
        relation=[GraphRelation(node1="a", node2="b", type="HAS")],
    )

    await repo.add_graph(namespace, [graph])

    assert driver.session.call_count == 1
    assert tx.calls[0][0] == _NODE_IMPORT_QUERY
    assert tx.calls[0][1] == {
        "data": [
            {
                "name": "a",
                "knowledge_id": "know-1",
                "props": {"attributes": ["x1"]},
                "chunks": ["c1"],
                "labels": ["ENTITYkb_1", "ENTITYknow_1"],
            },
            {
                "name": "b",
                "knowledge_id": "know-1",
                "props": {"attributes": []},
                "chunks": [],
                "labels": ["ENTITYkb_1", "ENTITYknow_1"],
            },
        ]
    }
    assert tx.calls[1][0] == _REL_IMPORT_QUERY
    assert tx.calls[1][1] == {
        "data": [
            {
                "source": "a",
                "target": "b",
                "knowledge_id": "know-1",
                "type": "HAS",
                "source_labels": ["ENTITYkb_1", "ENTITYknow_1"],
                "target_labels": ["ENTITYkb_1", "ENTITYknow_1"],
            }
        ]
    }


async def test_add_graph_opens_one_session_per_graph() -> None:
    tx = _FakeTx()
    repo, driver = _make_repo(tx)
    await repo.add_graph(NameSpace(knowledge="k"), [GraphData(), GraphData()])
    assert driver.session.call_count == 2


# ── del_graph ──────────────────────────────────────────────────────


async def test_del_graph_deletes_rels_then_nodes_per_namespace() -> None:
    tx = _FakeTx()
    repo, _driver = _make_repo(tx)

    await repo.del_graph(
        [NameSpace(knowledge_base="kb", knowledge="k1"), NameSpace(knowledge="k2")]
    )

    assert len(tx.calls) == 4
    rels_query, rels_params = tx.calls[0]
    nodes_query, nodes_params = tx.calls[1]
    assert '"DELETE r"' in rels_query
    assert '"DELETE n"' in nodes_query
    assert "ENTITYkb:ENTITYk1" in rels_query
    assert "ENTITYkb:ENTITYk1" in nodes_query
    assert rels_params == {"knowledge_id": "k1"}
    assert nodes_params == {"knowledge_id": "k1"}

    rels_query2, rels_params2 = tx.calls[2]
    assert "ENTITYk2" in rels_query2
    assert rels_params2 == {"knowledge_id": "k2"}


# ── search_node ────────────────────────────────────────────────────


def _record(name_a: str, name_b: str, rel_type: str) -> dict[str, object]:
    return {
        "n": {"name": name_a, "chunks": [f"c-{name_a}"], "attributes": [f"a-{name_a}"]},
        "r": SimpleNamespace(type=rel_type),
        "m": {"name": name_b, "chunks": [], "attributes": []},
    }


async def test_search_node_builds_graph_from_records() -> None:
    tx = _FakeTx(_FakeResult([_record("alice", "bob", "KNOWS")]))
    repo, _driver = _make_repo(tx)
    namespace = NameSpace(knowledge_base="kb", knowledge="k")

    graph = await repo.search_node(namespace, ["ali"])

    assert graph is not None
    assert graph.model_dump() == {
        "text": "",
        "node": [
            {"name": "alice", "chunks": ["c-alice"], "attributes": ["a-alice"]},
            {"name": "bob", "chunks": [], "attributes": []},
        ],
        "relation": [{"node1": "alice", "node2": "bob", "type": "KNOWS"}],
    }


async def test_search_node_dedupes_nodes_across_records() -> None:
    tx = _FakeTx(
        _FakeResult([_record("alice", "bob", "KNOWS"), _record("alice", "carol", "WORKS_WITH")])
    )
    repo, _driver = _make_repo(tx)

    graph = await repo.search_node(NameSpace(knowledge="k"), ["ali"])

    assert graph is not None
    assert [node.name for node in graph.node] == ["alice", "bob", "carol"]
    assert len(graph.relation) == 2


async def test_search_node_returns_empty_graph_when_no_records() -> None:
    tx = _FakeTx(_FakeResult())
    repo, _driver = _make_repo(tx)

    graph = await repo.search_node(NameSpace(knowledge="k"), ["nope"])

    assert graph is not None
    assert graph.node == []
    assert graph.relation == []


async def test_search_node_passes_nodes_param_and_query() -> None:
    tx = _FakeTx(_FakeResult())
    repo, _driver = _make_repo(tx)
    namespace = NameSpace(knowledge_base="kb", knowledge="k")

    await repo.search_node(namespace, ["ali", "bob"])

    query, params = tx.calls[0]
    assert "MATCH (n:ENTITYkb:ENTITYk)-[r]-(m:ENTITYkb:ENTITYk)" in query
    assert params == {"nodes": ["ali", "bob"]}


# ── connect_driver ─────────────────────────────────────────────────


async def test_connect_driver_connects_on_first_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = AsyncMock()
    driver.verify_authentication = AsyncMock()
    monkeypatch.setattr(AsyncGraphDatabase, "driver", Mock(return_value=driver))

    result = await connect_driver("bolt://localhost:7687", "neo4j", "secret")

    assert result is driver
    driver.verify_authentication.assert_awaited_once()


async def test_connect_driver_retries_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = AsyncMock()
    driver.verify_authentication = AsyncMock(side_effect=[RuntimeError("auth down"), None])
    sleep = AsyncMock()
    monkeypatch.setattr(AsyncGraphDatabase, "driver", Mock(return_value=driver))
    monkeypatch.setattr(asyncio, "sleep", sleep)

    result = await connect_driver(
        "bolt://localhost:7687", "neo4j", "secret", max_retries=3, retry_interval=0.0
    )

    assert result is driver
    assert driver.verify_authentication.await_count == 2
    driver.close.assert_awaited_once()
    sleep.assert_awaited_once()


async def test_connect_driver_raises_after_all_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = AsyncMock()
    driver.verify_authentication = AsyncMock(side_effect=RuntimeError("nope"))
    monkeypatch.setattr(AsyncGraphDatabase, "driver", Mock(return_value=driver))
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    with pytest.raises(Neo4jConnectionError):
        await connect_driver(
            "bolt://localhost:7687", "neo4j", "secret", max_retries=2, retry_interval=0.0
        )

    assert driver.verify_authentication.await_count == 2
    assert driver.close.await_count == 2


# ── build_graph_repository ─────────────────────────────────────────


async def test_build_graph_repository_disabled_when_not_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(neo4j_repo, "graph_enabled", Mock(return_value=False))
    connect = AsyncMock(side_effect=AssertionError("must not connect when disabled"))
    monkeypatch.setattr(neo4j_repo, "connect_driver", connect)

    repo = await build_graph_repository()

    assert repo._driver is None
    connect.assert_not_awaited()


async def test_build_graph_repository_connects_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(neo4j_repo, "graph_enabled", Mock(return_value=True))
    driver = AsyncMock()
    monkeypatch.setattr(neo4j_repo, "connect_driver", AsyncMock(return_value=driver))

    repo = await build_graph_repository()

    assert repo._driver is driver
