"""Tests for the composite retrieval engine.

Covers engine grouping and support validation during construction, the
per-retriever-type fan-out of ``retrieve`` with merged results, the
not-found / engine-error paths, ``support_retriever``, the write-operation
fan-out (indexing, deletes, batch updates, index copy) to every engine,
the batch-index source-id dedup, and the summed storage-size estimate. No
vector database is contacted — every engine is faked.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pytest

from src.ai.embedding import Context, Embedder, TaskContext
from src.ai.retrieval.base import RetrieveEngineService
from src.ai.retrieval.composite import new_composite_retrieve_engine
from src.ai.retrieval.registry import (
    RetrieveEngineRegistry,
    new_retrieve_engine_registry,
)
from src.ai.retrieval.types import (
    IndexInfo,
    IndexWithScore,
    RetrieveParams,
    RetrieverEngineParams,
    RetrieverEngineType,
    RetrieveResult,
    RetrieverType,
)
from src.common.exception import NotFoundError, ValidationError

_CTX = TaskContext()


class _FakeEmbedder:
    """Embedder stand-in; the composite never calls into it."""


class _FakeEngine:
    """Engine service recording every call for fan-out assertions."""

    def __init__(
        self,
        engine_type: RetrieverEngineType,
        support: list[RetrieverType],
        *,
        name: str = "",
        retrieve_results: list[RetrieveResult] | None = None,
        retrieve_error: Exception | None = None,
        index_error: Exception | None = None,
        estimate_size: int = 0,
    ) -> None:
        self._engine_type = engine_type
        self._support = support
        self.name = name
        self.calls: list[tuple[str, object]] = []
        self._retrieve_results = retrieve_results or []
        self._retrieve_error = retrieve_error
        self._index_error = index_error
        self._estimate_size = estimate_size

    def engine_type(self) -> RetrieverEngineType:
        return self._engine_type

    def support(self) -> list[RetrieverType]:
        return list(self._support)

    async def retrieve(self, _ctx: Context, params: RetrieveParams) -> list[RetrieveResult]:
        self.calls.append(("retrieve", params))
        if self._retrieve_error is not None:
            raise self._retrieve_error
        return self._retrieve_results

    async def index(
        self,
        _ctx: Context,
        _embedder: Embedder,
        index_info: IndexInfo,
        retriever_types: list[RetrieverType],
    ) -> None:
        self.calls.append(("index", (index_info, tuple(retriever_types))))
        if self._index_error is not None:
            raise self._index_error

    async def batch_index(
        self,
        _ctx: Context,
        _embedder: Embedder,
        index_info_list: list[IndexInfo],
        retriever_types: list[RetrieverType],
    ) -> None:
        self.calls.append(
            (
                "batch_index",
                (
                    tuple(item.source_id for item in index_info_list),
                    tuple(retriever_types),
                ),
            )
        )

    def estimate_storage_size(
        self,
        _ctx: Context,
        _embedder: Embedder,
        index_info_list: list[IndexInfo],
        _retriever_types: list[RetrieverType],
    ) -> int:
        self.calls.append(("estimate_storage_size", len(index_info_list)))
        return self._estimate_size

    async def delete_by_chunk_id_list(
        self,
        _ctx: Context,
        index_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        self.calls.append(("delete_by_chunk_id_list", (index_id_list, dimension, knowledge_type)))

    async def delete_by_source_id_list(
        self,
        _ctx: Context,
        source_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        self.calls.append(("delete_by_source_id_list", (source_id_list, dimension, knowledge_type)))

    async def copy_indices(
        self,
        _ctx: Context,
        source_knowledge_base_id: str,
        source_to_target_kb_id_map: Mapping[str, str],
        source_to_target_chunk_id_map: Mapping[str, str],
        target_knowledge_base_id: str,
        dimension: int,
        knowledge_type: str,
    ) -> None:
        self.calls.append(
            (
                "copy_indices",
                (
                    source_knowledge_base_id,
                    source_to_target_kb_id_map,
                    source_to_target_chunk_id_map,
                    target_knowledge_base_id,
                    dimension,
                    knowledge_type,
                ),
            )
        )

    async def delete_by_knowledge_id_list(
        self,
        _ctx: Context,
        knowledge_id_list: list[str],
        dimension: int,
        knowledge_type: str,
    ) -> None:
        self.calls.append(
            (
                "delete_by_knowledge_id_list",
                (knowledge_id_list, dimension, knowledge_type),
            )
        )

    async def batch_update_chunk_enabled_status(
        self, _ctx: Context, chunk_status_map: Mapping[str, bool]
    ) -> None:
        self.calls.append(("batch_update_chunk_enabled_status", chunk_status_map))

    async def batch_update_chunk_tag_id(
        self, _ctx: Context, chunk_tag_map: Mapping[str, str]
    ) -> None:
        self.calls.append(("batch_update_chunk_tag_id", chunk_tag_map))


def _svc(engine: _FakeEngine) -> RetrieveEngineService:
    return cast("RetrieveEngineService", engine)


def _embedder() -> Embedder:
    return cast("Embedder", _FakeEmbedder())


def _params(
    engine_type: RetrieverEngineType, retriever_type: RetrieverType
) -> RetrieverEngineParams:
    return RetrieverEngineParams(retriever_engine_type=engine_type, retriever_type=retriever_type)


def _registry(*engines: _FakeEngine) -> RetrieveEngineRegistry:
    registry = new_retrieve_engine_registry(None, None)
    for engine in engines:
        registry.register(_svc(engine))
    return registry


def _result(
    engine_type: RetrieverEngineType,
    retriever_type: RetrieverType,
    *,
    chunk_id: str = "c1",
    score: float = 1.0,
) -> RetrieveResult:
    return RetrieveResult(
        results=[IndexWithScore(id=chunk_id, score=score)],
        retriever_engine_type=engine_type,
        retriever_type=retriever_type,
    )


# ── construction ─────────────────────────────────────────────────────


def test_new_composite_groups_by_engine_type() -> None:
    engine = _FakeEngine(
        RetrieverEngineType.ELASTICSEARCH,
        [RetrieverType.KEYWORDS, RetrieverType.VECTOR],
    )
    composite = new_composite_retrieve_engine(
        _registry(engine),
        [
            _params(RetrieverEngineType.ELASTICSEARCH, RetrieverType.KEYWORDS),
            _params(RetrieverEngineType.ELASTICSEARCH, RetrieverType.VECTOR),
        ],
    )
    infos = composite._engine_infos
    assert len(infos) == 1
    assert infos[0].retrieve_engine is engine
    assert infos[0].retriever_types == (
        RetrieverType.KEYWORDS,
        RetrieverType.VECTOR,
    )


def test_new_composite_multiple_engines_preserved() -> None:
    keyword_engine = _FakeEngine(RetrieverEngineType.ELASTICSEARCH, [RetrieverType.KEYWORDS])
    vector_engine = _FakeEngine(RetrieverEngineType.QDRANT, [RetrieverType.VECTOR])
    composite = new_composite_retrieve_engine(
        _registry(keyword_engine, vector_engine),
        [
            _params(RetrieverEngineType.ELASTICSEARCH, RetrieverType.KEYWORDS),
            _params(RetrieverEngineType.QDRANT, RetrieverType.VECTOR),
        ],
    )
    infos = composite._engine_infos
    assert [info.retrieve_engine for info in infos] == [keyword_engine, vector_engine]


def test_new_composite_unsupported_retriever_type_raises() -> None:
    engine = _FakeEngine(RetrieverEngineType.ELASTICSEARCH, [RetrieverType.KEYWORDS])
    with pytest.raises(ValidationError, match="does not support"):
        new_composite_retrieve_engine(
            _registry(engine),
            [_params(RetrieverEngineType.ELASTICSEARCH, RetrieverType.VECTOR)],
        )


def test_new_composite_missing_engine_raises() -> None:
    with pytest.raises(NotFoundError, match="not found"):
        new_composite_retrieve_engine(
            _registry(),
            [_params(RetrieverEngineType.QDRANT, RetrieverType.VECTOR)],
        )


def test_support_retriever() -> None:
    keyword_engine = _FakeEngine(RetrieverEngineType.ELASTICSEARCH, [RetrieverType.KEYWORDS])
    composite = new_composite_retrieve_engine(
        _registry(keyword_engine),
        [_params(RetrieverEngineType.ELASTICSEARCH, RetrieverType.KEYWORDS)],
    )
    assert composite.support_retriever(RetrieverType.KEYWORDS) is True
    assert composite.support_retriever(RetrieverType.VECTOR) is False


# ── retrieve fan-out ─────────────────────────────────────────────────


async def test_retrieve_dispatches_per_retriever_type_and_merges() -> None:
    keyword_engine = _FakeEngine(
        RetrieverEngineType.ELASTICSEARCH,
        [RetrieverType.KEYWORDS],
        retrieve_results=[_result(RetrieverEngineType.ELASTICSEARCH, RetrieverType.KEYWORDS)],
    )
    vector_engine = _FakeEngine(
        RetrieverEngineType.QDRANT,
        [RetrieverType.VECTOR],
        retrieve_results=[_result(RetrieverEngineType.QDRANT, RetrieverType.VECTOR)],
    )
    composite = new_composite_retrieve_engine(
        _registry(keyword_engine, vector_engine),
        [
            _params(RetrieverEngineType.ELASTICSEARCH, RetrieverType.KEYWORDS),
            _params(RetrieverEngineType.QDRANT, RetrieverType.VECTOR),
        ],
    )
    keyword_params = RetrieveParams(query="q", retriever_type=RetrieverType.KEYWORDS)
    vector_params = RetrieveParams(query="q", retriever_type=RetrieverType.VECTOR)

    results = await composite.retrieve(_CTX, [keyword_params, vector_params])

    assert results == [
        _result(RetrieverEngineType.ELASTICSEARCH, RetrieverType.KEYWORDS),
        _result(RetrieverEngineType.QDRANT, RetrieverType.VECTOR),
    ]
    assert keyword_engine.calls == [("retrieve", keyword_params)]
    assert vector_engine.calls == [("retrieve", vector_params)]


async def test_retrieve_routes_to_first_matching_engine() -> None:
    # The first registered engine serves both retriever types, so a VECTOR
    # param must not spill onto the later VECTOR-only engine.
    hybrid_engine = _FakeEngine(
        RetrieverEngineType.ELASTICSEARCH,
        [RetrieverType.KEYWORDS, RetrieverType.VECTOR],
        retrieve_results=[_result(RetrieverEngineType.ELASTICSEARCH, RetrieverType.VECTOR)],
    )
    vector_engine = _FakeEngine(RetrieverEngineType.QDRANT, [RetrieverType.VECTOR])
    composite = new_composite_retrieve_engine(
        _registry(hybrid_engine, vector_engine),
        [
            _params(RetrieverEngineType.ELASTICSEARCH, RetrieverType.VECTOR),
            _params(RetrieverEngineType.QDRANT, RetrieverType.VECTOR),
        ],
    )
    params = RetrieveParams(query="q", retriever_type=RetrieverType.VECTOR)

    results = await composite.retrieve(_CTX, [params])

    assert results == [_result(RetrieverEngineType.ELASTICSEARCH, RetrieverType.VECTOR)]
    assert hybrid_engine.calls == [("retrieve", params)]
    assert vector_engine.calls == []


async def test_retrieve_empty_params_returns_empty() -> None:
    keyword_engine = _FakeEngine(RetrieverEngineType.ELASTICSEARCH, [RetrieverType.KEYWORDS])
    composite = new_composite_retrieve_engine(
        _registry(keyword_engine),
        [_params(RetrieverEngineType.ELASTICSEARCH, RetrieverType.KEYWORDS)],
    )
    assert await composite.retrieve(_CTX, []) == []
    assert keyword_engine.calls == []


async def test_retrieve_unsupported_retriever_type_raises_not_found() -> None:
    keyword_engine = _FakeEngine(RetrieverEngineType.ELASTICSEARCH, [RetrieverType.KEYWORDS])
    composite = new_composite_retrieve_engine(
        _registry(keyword_engine),
        [_params(RetrieverEngineType.ELASTICSEARCH, RetrieverType.KEYWORDS)],
    )
    params = RetrieveParams(query="q", retriever_type=RetrieverType.VECTOR)

    with pytest.raises(NotFoundError, match="retriever type vector not found"):
        await composite.retrieve(_CTX, [params])


async def test_retrieve_propagates_engine_error() -> None:
    vector_engine = _FakeEngine(
        RetrieverEngineType.QDRANT,
        [RetrieverType.VECTOR],
        retrieve_error=RuntimeError("backend down"),
    )
    composite = new_composite_retrieve_engine(
        _registry(vector_engine),
        [_params(RetrieverEngineType.QDRANT, RetrieverType.VECTOR)],
    )
    params = RetrieveParams(query="q", retriever_type=RetrieverType.VECTOR)

    with pytest.raises(RuntimeError, match="backend down"):
        await composite.retrieve(_CTX, [params])


# ── write fan-out ────────────────────────────────────────────────────


async def test_index_fans_out_to_every_engine_with_its_retriever_types() -> None:
    keyword_engine = _FakeEngine(RetrieverEngineType.ELASTICSEARCH, [RetrieverType.KEYWORDS])
    vector_engine = _FakeEngine(RetrieverEngineType.QDRANT, [RetrieverType.VECTOR])
    composite = new_composite_retrieve_engine(
        _registry(keyword_engine, vector_engine),
        [
            _params(RetrieverEngineType.ELASTICSEARCH, RetrieverType.KEYWORDS),
            _params(RetrieverEngineType.QDRANT, RetrieverType.VECTOR),
        ],
    )
    index_info = IndexInfo(id="c1", source_id="s1")

    await composite.index(_CTX, _embedder(), index_info)

    assert keyword_engine.calls == [("index", (index_info, (RetrieverType.KEYWORDS,)))]
    assert vector_engine.calls == [("index", (index_info, (RetrieverType.VECTOR,)))]


async def test_batch_index_deduplicates_by_source_id() -> None:
    engine = _FakeEngine(RetrieverEngineType.ELASTICSEARCH, [RetrieverType.KEYWORDS])
    composite = new_composite_retrieve_engine(
        _registry(engine),
        [_params(RetrieverEngineType.ELASTICSEARCH, RetrieverType.KEYWORDS)],
    )
    items = [
        IndexInfo(id="a", source_id="s1"),
        IndexInfo(id="b", source_id="s1"),
        IndexInfo(id="c", source_id="s2"),
    ]

    await composite.batch_index(_CTX, _embedder(), items)

    # Only the first occurrence of each source id is forwarded.
    assert engine.calls == [("batch_index", (("s1", "s2"), (RetrieverType.KEYWORDS,)))]


async def test_write_ops_fan_out_to_every_engine() -> None:
    keyword_engine = _FakeEngine(RetrieverEngineType.ELASTICSEARCH, [RetrieverType.KEYWORDS])
    vector_engine = _FakeEngine(RetrieverEngineType.QDRANT, [RetrieverType.VECTOR])
    composite = new_composite_retrieve_engine(
        _registry(keyword_engine, vector_engine),
        [
            _params(RetrieverEngineType.ELASTICSEARCH, RetrieverType.KEYWORDS),
            _params(RetrieverEngineType.QDRANT, RetrieverType.VECTOR),
        ],
    )
    await composite.delete_by_chunk_id_list(_CTX, ["c1"], 768, "doc")
    await composite.delete_by_source_id_list(_CTX, ["s1"], 768, "doc")
    await composite.copy_indices(
        _CTX, "kb-src", {"kb-src": "kb-dst"}, {"c1": "c2"}, "kb-dst", 768, "doc"
    )
    await composite.delete_by_knowledge_id_list(_CTX, ["k1"], 768, "doc")
    await composite.batch_update_chunk_enabled_status(_CTX, {"c1": False})
    await composite.batch_update_chunk_tag_id(_CTX, {"c1": "tag1"})

    expected = [
        ("delete_by_chunk_id_list", (["c1"], 768, "doc")),
        ("delete_by_source_id_list", (["s1"], 768, "doc")),
        (
            "copy_indices",
            ("kb-src", {"kb-src": "kb-dst"}, {"c1": "c2"}, "kb-dst", 768, "doc"),
        ),
        ("delete_by_knowledge_id_list", (["k1"], 768, "doc")),
        ("batch_update_chunk_enabled_status", {"c1": False}),
        ("batch_update_chunk_tag_id", {"c1": "tag1"}),
    ]
    assert keyword_engine.calls == expected
    assert vector_engine.calls == expected


async def test_write_op_propagates_engine_error() -> None:
    failing_engine = _FakeEngine(
        RetrieverEngineType.ELASTICSEARCH,
        [RetrieverType.KEYWORDS],
        index_error=RuntimeError("save failed"),
    )
    vector_engine = _FakeEngine(RetrieverEngineType.QDRANT, [RetrieverType.VECTOR])
    composite = new_composite_retrieve_engine(
        _registry(failing_engine, vector_engine),
        [
            _params(RetrieverEngineType.ELASTICSEARCH, RetrieverType.KEYWORDS),
            _params(RetrieverEngineType.QDRANT, RetrieverType.VECTOR),
        ],
    )

    with pytest.raises(RuntimeError, match="save failed"):
        await composite.index(_CTX, _embedder(), IndexInfo(id="c1"))


def test_estimate_storage_size_sums_across_engines() -> None:
    engine_a = _FakeEngine(
        RetrieverEngineType.ELASTICSEARCH,
        [RetrieverType.KEYWORDS],
        estimate_size=10,
    )
    engine_b = _FakeEngine(RetrieverEngineType.QDRANT, [RetrieverType.VECTOR], estimate_size=20)
    composite = new_composite_retrieve_engine(
        _registry(engine_a, engine_b),
        [
            _params(RetrieverEngineType.ELASTICSEARCH, RetrieverType.KEYWORDS),
            _params(RetrieverEngineType.QDRANT, RetrieverType.VECTOR),
        ],
    )
    items = [IndexInfo(id="a", source_id="s1")]

    assert composite.estimate_storage_size(_CTX, _embedder(), items) == 30
    assert engine_a.calls == [("estimate_storage_size", 1)]
    assert engine_b.calls == [("estimate_storage_size", 1)]
