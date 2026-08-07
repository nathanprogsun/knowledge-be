"""Tests for the score normalizer.

Covers keyword passthrough, the Milvus raw-cosine rescale, the [0, 1]
passthrough engines, the Elasticsearch / OpenSearch passthrough regression
guards, NaN/Inf clamping, the ``clamp01`` envelope, and concurrent use.
``normalize`` is pure and stateless, so concurrency safety is a property of
construction — the concurrent test guards against accidental shared state
being introduced later.
"""

from __future__ import annotations

import asyncio
import math

import pytest

from src.ai.embedding import TaskContext
from src.ai.retrieval.normalizer import EngineAwareNormalizer, clamp01
from src.ai.retrieval.types import RetrieverEngineType, RetrieverType

_CTX = TaskContext()


def _normalizer() -> EngineAwareNormalizer:
    return EngineAwareNormalizer()


#: Engines whose effective vector score at the normalizer's input is already
#: in [0, 1] and must pass through unchanged (see the module docstring for the
#: per-engine derivation).
_PASSTHROUGH_ENGINES: tuple[RetrieverEngineType, ...] = (
    RetrieverEngineType.ELASTICSEARCH,
    RetrieverEngineType.ELASTICFAISS,
    RetrieverEngineType.OPENSEARCH,
    RetrieverEngineType.WEAVIATE,
    RetrieverEngineType.POSTGRES,
    RetrieverEngineType.SQLITE,
    RetrieverEngineType.QDRANT,
    RetrieverEngineType.INFINITY,
    RetrieverEngineType.TENCENT_VECTORDB,
    RetrieverEngineType.DORIS,
)


# ── keyword passthrough ──────────────────────────────────────────────


@pytest.mark.parametrize("score", (-12.5, 0.0, 0.7, 1.0, 27.3, math.inf))
def test_keyword_scores_pass_through_unchanged(score: float) -> None:
    normalizer = _normalizer()
    for engine in _PASSTHROUGH_ENGINES:
        got = normalizer.normalize(
            _CTX, score, RetrieverType.KEYWORDS, engine
        )
        assert got == score


def test_opensearch_keyword_passes_through_unbounded() -> None:
    # BM25 keyword scores are unbounded positive values and MUST pass through
    # unchanged so rank-based fusion handles the scale-mixed combination.
    got = _normalizer().normalize(
        _CTX, 12.7, RetrieverType.KEYWORDS, RetrieverEngineType.OPENSEARCH
    )
    assert got == 12.7


# ── Milvus raw-cosine rescale ────────────────────────────────────────


@pytest.mark.parametrize(
    "score, expected",
    [
        (-1.0, 0.0),
        (-0.5, 0.25),
        (0.0, 0.5),
        (0.5, 0.75),
        (1.0, 1.0),
        # Drift beyond [-1, 1] is clamped.
        (-1.5, 0.0),
        (1.5, 1.0),
    ],
)
def test_milvus_raw_cosine_rescaled_to_unit_interval(
    score: float, expected: float
) -> None:
    got = _normalizer().normalize(
        _CTX, score, RetrieverType.VECTOR, RetrieverEngineType.MILVUS
    )
    assert got == pytest.approx(expected)


# ── [0, 1] passthrough engines ───────────────────────────────────────


@pytest.mark.parametrize("engine", _PASSTHROUGH_ENGINES)
@pytest.mark.parametrize(
    "score, expected",
    [
        (-0.1, 0.0),
        (0.0, 0.0),
        (0.25, 0.25),
        (0.999, 0.999),
        (1.0, 1.0),
        (1.5, 1.0),
    ],
)
def test_unit_interval_engines_pass_through(
    engine: RetrieverEngineType, score: float, expected: float
) -> None:
    got = _normalizer().normalize(
        _CTX, score, RetrieverType.VECTOR, engine
    )
    assert got == pytest.approx(expected)


@pytest.mark.parametrize(
    "score, expected",
    [
        # cos=0 orthogonal IR-clamped to 0.
        (0.0, 0.0),
        # cos=0.25 IR-typical low passes through.
        (0.25, 0.25),
        # cos=0.5 IR-typical mid passes through (not (0.5+1)/2 = 0.75).
        (0.5, 0.5),
        # cos=0.75 IR-typical high passes through.
        (0.75, 0.75),
        # cos=1 identical passes through as 1.
        (1.0, 1.0),
        # Negative defensive clamps to 0.
        (-0.5, 0.0),
        # Engine drift past 1 clamps to 1.
        (1.0001, 1.0),
        # Float-edge cases handled by clamp01.
        (math.inf, 1.0),
        (-math.inf, 0.0),
        (math.nan, 0.0),
    ],
)
def test_elasticsearch_cosine_passes_through(score: float, expected: float) -> None:
    # Elasticsearch's driver issues a script_score cosineSimilarity script;
    # Lucene rejects negative final scores, so the effective range observed
    # here is [0, 1] and cos=0.5 must map to 0.5, not 0.75.
    got = _normalizer().normalize(
        _CTX, score, RetrieverType.VECTOR, RetrieverEngineType.ELASTICSEARCH
    )
    assert got == pytest.approx(expected)


@pytest.mark.parametrize(
    "score, expected",
    [
        (0.0, 0.0),
        (0.5, 0.5),
        (0.75, 0.75),
        (1.0, 1.0),
        (1.0001, 1.0),
        (-0.5, 0.0),
        (math.inf, 1.0),
        (-math.inf, 0.0),
        (math.nan, 0.0),
    ],
)
def test_opensearch_cosine_passes_through(score: float, expected: float) -> None:
    # OpenSearch k-NN cosinesimil scores arrive pre-mapped to [0, 1] by the
    # plugin's score translation; the normalizer only guards engine drift.
    got = _normalizer().normalize(
        _CTX, score, RetrieverType.VECTOR, RetrieverEngineType.OPENSEARCH
    )
    assert got == pytest.approx(expected)


# ── unknown-engine default branch ────────────────────────────────────


def test_non_milvus_engines_clamp_to_unit_interval() -> None:
    # Every engine that is not Milvus takes the default clamp branch — a
    # hypothetical future engine cannot leak past the [0, 1] envelope.
    normalizer = _normalizer()
    assert (
        normalizer.normalize(
            _CTX, 0.42, RetrieverType.VECTOR, RetrieverEngineType.POSTGRES
        )
        == pytest.approx(0.42)
    )
    assert (
        normalizer.normalize(
            _CTX, 5.0, RetrieverType.VECTOR, RetrieverEngineType.POSTGRES
        )
        == pytest.approx(1.0)
    )


# ── clamp01 envelope ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "score, expected",
    [
        (-1.0, 0.0),
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (2.0, 1.0),
        (math.nan, 0.0),
        (math.inf, 1.0),
        (-math.inf, 0.0),
    ],
)
def test_clamp01(score: float, expected: float) -> None:
    assert clamp01(score) == pytest.approx(expected)


def test_nan_and_inf_handled_across_formulas() -> None:
    normalizer = _normalizer()
    # NaN -> 0 so a strict-weak-ordering comparator downstream never sees a
    # value that compares neither greater nor less than anything.
    assert (
        normalizer.normalize(
            _CTX, math.nan, RetrieverType.VECTOR, RetrieverEngineType.POSTGRES
        )
        == 0.0
    )
    assert (
        normalizer.normalize(
            _CTX, math.inf, RetrieverType.VECTOR, RetrieverEngineType.POSTGRES
        )
        == 1.0
    )
    assert (
        normalizer.normalize(
            _CTX, -math.inf, RetrieverType.VECTOR, RetrieverEngineType.POSTGRES
        )
        == 0.0
    )
    # NaN through the Milvus cosine formula too.
    assert (
        normalizer.normalize(
            _CTX, math.nan, RetrieverType.VECTOR, RetrieverEngineType.MILVUS
        )
        == 0.0
    )


# ── concurrency ──────────────────────────────────────────────────────


async def test_normalize_is_concurrent_safe() -> None:
    normalizer = _normalizer()
    scores = await asyncio.gather(
        *(
            asyncio.to_thread(
                normalizer.normalize,
                _CTX,
                float(index),
                RetrieverType.VECTOR,
                RetrieverEngineType.MILVUS,
            )
            for index in range(64)
        )
    )
    for index, score in enumerate(scores):
        expected = (index + 1.0) / 2.0 if index <= 1.0 else 1.0
        assert score == pytest.approx(expected)
