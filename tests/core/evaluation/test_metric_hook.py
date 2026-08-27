"""Unit tests for the evaluation metric hook and its calculators.

Exercises the retrieval metrics (precision / recall / NDCG@k / MRR /
MAP) and generation metrics (BLEU-1/2/4, ROUGE-1/2/L) with hand-computed
expected values, plus the content-matching path that maps retrieved
chunks back to ground-truth passage ids.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from src.core.evaluation.dataset import QAPair
from src.core.evaluation.metric_hook import (
    GenerationMetrics,
    HookMetric,
    MetricInput,
    MetricList,
    MetricResult,
    RetrievalMetrics,
    _compute_map,
    _compute_mrr,
    _compute_ndcg,
    _compute_precision,
    _compute_recall,
)


class _Hit:
    """Minimal search-hit stand-in carrying only ``content``."""

    def __init__(self, content: str) -> None:
        self.content = content


class _Chat:
    """Minimal chat-response stand-in carrying only ``content``."""

    def __init__(self, content: str) -> None:
        self.content = content


# ── MetricInput calculators (via HookMetric record_finish) ─────────────


def _run_hook(
    pair: QAPair,
    *,
    search: Sequence[_Hit] = (),
    rerank: Sequence[_Hit] = (),
    chat: _Chat | None = None,
) -> MetricResult:
    """Record one pair through the hook and return its bundle."""
    hook = HookMetric(capacity=1)
    hook.record_init(0)
    hook.record_qa_pair(0, pair)
    if search:
        hook.record_search_result(0, list(search))
    if rerank:
        hook.record_rerank_result(0, list(rerank))
    if chat is not None:
        hook.record_chat_response(0, chat)
    hook.record_finish(0)
    return hook.metric_result()


def _input(retrieval_ids: list[int], gt: list[list[int]] | None = None) -> MetricInput:
    """Build a raw metric input for the calculator-level tests."""
    return MetricInput(
        retrieval_gt=gt or [[1, 2]],
        retrieval_ids=retrieval_ids,
        generated_texts="",
        generated_gt="",
    )


def _pair(pids: list[int], passages: list[str], answer: str = "") -> QAPair:
    return QAPair(
        qid=1,
        question="question",
        pids=pids,
        passages=passages,
        aid=1,
        answer=answer,
    )


# ── Retrieval metric calculators ───────────────────────────────────────


class TestRetrievalMetrics:
    def test_precision_counts_hits_over_retrieved(self) -> None:
        assert _compute_precision(_input([1, 3])) == pytest.approx(0.5)

    def test_recall_counts_hits_over_ground_truth(self) -> None:
        assert _compute_recall(_input([1, 3])) == pytest.approx(0.5)

    def test_ndcg_at_3(self) -> None:
        assert _compute_ndcg(_input([1, 3]), 3) == pytest.approx(1.0 / (1.0 + 1.0 / math.log2(3)))

    def test_ndcg_at_10_equals_ndcg_at_3_for_short_lists(self) -> None:
        assert _compute_ndcg(_input([1, 3]), 10) == pytest.approx(_compute_ndcg(_input([1, 3]), 3))

    def test_mrr_is_reciprocal_of_first_hit_rank(self) -> None:
        assert _compute_mrr(_input([3, 1])) == pytest.approx(0.5)

    def test_map_is_one_when_every_hit_is_relevant(self) -> None:
        assert _compute_map(_input([1, 2])) == pytest.approx(1.0)

    def test_no_retrieval_returns_zero_scores(self) -> None:
        assert _compute_precision(_input([])) == 0.0
        assert _compute_recall(_input([])) == 0.0
        assert _compute_ndcg(_input([]), 3) == 0.0
        assert _compute_mrr(_input([])) == 0.0
        assert _compute_map(_input([])) == 0.0

    def test_rerank_results_preferred_over_search_results(self) -> None:
        pair = _pair(pids=[1, 2], passages=["a", "b"])
        # Search returns the correct hit; rerank is empty. The hook must
        # fall back to search (upstream ``recordFinish``).
        result = _run_hook(pair, search=[_Hit("a")], rerank=[])
        assert result.retrieval.precision == pytest.approx(1.0)

    def test_content_matching_dedupes_passages(self) -> None:
        pair = _pair(pids=[1, 2], passages=["alpha beta", "gamma delta"])
        # Two chunks map onto the same passage; only one id kept, so
        # recall is 1/2 rather than 1.0.
        result = _run_hook(pair, search=[_Hit("alpha"), _Hit("beta")])
        assert result.retrieval.precision == pytest.approx(1.0)
        assert result.retrieval.recall == pytest.approx(0.5)


# ── Generation metrics ────────────────────────────────────────────────


class TestGenerationMetrics:
    def test_bleu1_exact(self) -> None:
        pair = _pair(pids=[1], passages=["x"], answer="a b c")
        result = _run_hook(pair, chat=_Chat("a b c"))
        assert result.generation.bleu1 == pytest.approx(1.0)

    def test_bleu1_partial_overlap(self) -> None:
        pair = _pair(pids=[1], passages=["x"], answer="a b d")
        result = _run_hook(pair, chat=_Chat("a b c"))
        assert result.generation.bleu1 == pytest.approx(0.75)

    def test_bleu4_exact(self) -> None:
        pair = _pair(pids=[1], passages=["x"], answer="a b c d")
        result = _run_hook(pair, chat=_Chat("a b c d"))
        assert result.generation.bleu4 == pytest.approx(1.0)

    def test_bleu_missing_chat_response_is_zero(self) -> None:
        pair = _pair(pids=[1], passages=["x"], answer="a b c")
        result = _run_hook(pair)
        assert result.generation.bleu1 == 0.0

    def test_rouge1_f(self) -> None:
        pair = _pair(pids=[1], passages=["x"], answer="a b d")
        result = _run_hook(pair, chat=_Chat("a b c"))
        assert result.generation.rouge1 == pytest.approx(
            2.0 * (2.0 / 3.0) * (2.0 / 3.0) / (4.0 / 3.0 + 1e-8)
        )

    def test_rouge2_exact(self) -> None:
        pair = _pair(pids=[1], passages=["x"], answer="a b c")
        result = _run_hook(pair, chat=_Chat("a b c"))
        assert result.generation.rouge2 == pytest.approx(1.0)

    def test_rougel_f(self) -> None:
        pair = _pair(pids=[1], passages=["x"], answer="a b d")
        result = _run_hook(pair, chat=_Chat("a b c"))
        assert result.generation.rougel == pytest.approx(
            2.0 * (2.0 / 3.0) * (2.0 / 3.0) / (4.0 / 3.0 + 1e-8)
        )


# ── Aggregation ───────────────────────────────────────────────────────


class TestMetricList:
    def test_avg_over_empty_list_returns_zeros(self) -> None:
        avg = MetricList().avg()
        assert avg.retrieval.precision == 0.0
        assert avg.generation.bleu1 == 0.0

    def test_avg_averages_each_field(self) -> None:
        first = MetricResult(
            retrieval=RetrievalMetrics(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
            generation=GenerationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        second = MetricResult(
            retrieval=RetrievalMetrics(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            generation=GenerationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        lst = MetricList()
        lst.append(first)
        lst.append(second)
        avg = lst.avg()
        assert avg.retrieval.precision == pytest.approx(0.75)
        assert avg.retrieval.recall == pytest.approx(0.75)


# ── Hook orchestration ────────────────────────────────────────────────


class TestHookMetric:
    def test_record_finish_before_init_is_silent(self) -> None:
        hook = HookMetric(capacity=1)
        hook.record_finish(0)  # no init → no slot
        assert hook.metric_result() == MetricResult.zeros()

    def test_multiple_pairs_average_together(self) -> None:
        hook = HookMetric(capacity=2)
        first = _pair(pids=[1], passages=["a"], answer="a b c")
        second = _pair(pids=[2], passages=["b"], answer="x y z")
        for index, (pair, hit, chat) in enumerate(
            (
                (first, _Hit("a"), _Chat("a b c")),
                (second, _Hit("b"), _Chat("x y z")),
            )
        ):
            hook.record_init(index)
            hook.record_qa_pair(index, pair)
            hook.record_search_result(index, [hit])
            hook.record_chat_response(index, chat)
            hook.record_finish(index)
        result = hook.metric_result()
        assert result.retrieval.precision == pytest.approx(1.0)
        assert result.generation.bleu1 == pytest.approx(1.0)

    def test_metric_input_shapes(self) -> None:
        """The hook builds the expected retrieval ground truth."""
        pair = _pair(pids=[3, 4], passages=["c", "d"])
        hook = HookMetric(capacity=1)
        hook.record_init(0)
        hook.record_qa_pair(0, pair)
        hook.record_search_result(0, [_Hit("c")])
        hook.record_finish(0)
        result = hook.metric_result()
        assert result.retrieval.recall == pytest.approx(0.5)
