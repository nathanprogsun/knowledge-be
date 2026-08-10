"""Metric hook — collect retrieval + generation metrics during a run.

Mirrors the upstream ``HookMetric`` / ``MetricList`` pair and the metric
calculators they drive: precision / recall / NDCG@k / MRR / MAP for
retrieval, plus BLEU-1/2/4 and ROUGE-1/2/L for generation. The hook is
the single place the evaluation flow converts raw pipeline artefacts
(search hits, reranked hits, chat response) into averaged scores.

Scope of this module
--------------------

- ``HookMetric`` — per-QA-pair collector; ``record_*`` methods capture
  the artefacts, ``record_finish`` computes one pair's scores, and
  ``metric_result`` returns the running average.
- ``MetricList`` — aggregates per-pair ``MetricResult`` values.
- ``MetricInput`` / ``MetricResult`` / ``RetrievalMetrics`` /
  ``GenerationMetrics`` — value objects carrying one pair's scores.
- The individual calculators (``PrecisionMetric`` … ``RougeMetric``)
  are private; they are wired through ``_CALCULATORS``.

Threading
---------

``HookMetric`` is safe for concurrent use: ``record_finish`` serialises
on an internal lock (mirroring the upstream mutex) and ``metric_result``
takes a read lock, so a fan-out runner can append from many workers.
"""

from __future__ import annotations

import math
import re
import threading
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import jieba

from src.core.evaluation.dataset import QAPair

# ── Structural seams ───────────────────────────────────────────────────


@runtime_checkable
class SearchHitLike(Protocol):
    """A retrieved chunk carrying ``content`` text (structural).

    Both the pipeline's ``SearchResult`` and the pre-rerank search hits
    satisfy this protocol; the hook never depends on a concrete hit
    type.
    """

    content: str


@runtime_checkable
class ChatResponseLike(Protocol):
    """A generated answer carrying ``content`` text (structural).

    The pipeline's ``ChatResponse`` model satisfies this protocol.
    """

    content: str


# ── Value objects ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MetricInput:
    """Inputs for one QA pair's metric calculation.

    Mirrors the upstream ``types.MetricInput``: ``retrieval_gt`` is the
    list of ground-truth id sets (one entry per query), ``retrieval_ids``
    the retrieved passage ids in rank order, ``generated_texts`` the
    pipeline's chat answer, and ``generated_gt`` the reference answer.
    """

    retrieval_gt: list[list[int]]
    retrieval_ids: list[int]
    generated_texts: str
    generated_gt: str


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Retrieval quality scores for one pair (upstream ``RetrievalMetrics``)."""

    precision: float
    recall: float
    ndcg3: float
    ndcg10: float
    mrr: float
    map_: float


@dataclass(frozen=True, slots=True)
class GenerationMetrics:
    """Generation quality scores for one pair (upstream ``GenerationMetrics``)."""

    bleu1: float
    bleu2: float
    bleu4: float
    rouge1: float
    rouge2: float
    rougel: float


@dataclass(frozen=True, slots=True)
class MetricResult:
    """One pair's full metric bundle (upstream ``MetricResult``)."""

    retrieval: RetrievalMetrics
    generation: GenerationMetrics

    @classmethod
    def zeros(cls) -> MetricResult:
        """Return a fresh all-zero bundle."""
        retrieval = RetrievalMetrics(
            precision=0.0,
            recall=0.0,
            ndcg3=0.0,
            ndcg10=0.0,
            mrr=0.0,
            map_=0.0,
        )
        generation = GenerationMetrics(
            bleu1=0.0,
            bleu2=0.0,
            bleu4=0.0,
            rouge1=0.0,
            rouge2=0.0,
            rougel=0.0,
        )
        return cls(retrieval=retrieval, generation=generation)


# ── Aggregate list ────────────────────────────────────────────────────


class MetricList:
    """Stores and aggregates per-pair :class:`MetricResult` values."""

    def __init__(self) -> None:
        self._results: list[MetricResult] = []

    def append(self, result: MetricResult) -> None:
        """Record one pair's metric bundle."""
        self._results.append(result)

    def avg(self) -> MetricResult:
        """Average every recorded bundle; all-zero when none recorded."""
        if not self._results:
            return MetricResult.zeros()
        count = float(len(self._results))
        acc = MetricResult.zeros()
        for r in self._results:
            acc = _add(acc, r)
        return MetricResult(
            retrieval=RetrievalMetrics(
                precision=acc.retrieval.precision / count,
                recall=acc.retrieval.recall / count,
                ndcg3=acc.retrieval.ndcg3 / count,
                ndcg10=acc.retrieval.ndcg10 / count,
                mrr=acc.retrieval.mrr / count,
                map_=acc.retrieval.map_ / count,
            ),
            generation=GenerationMetrics(
                bleu1=acc.generation.bleu1 / count,
                bleu2=acc.generation.bleu2 / count,
                bleu4=acc.generation.bleu4 / count,
                rouge1=acc.generation.rouge1 / count,
                rouge2=acc.generation.rouge2 / count,
                rougel=acc.generation.rougel / count,
            ),
        )


# ── The hook ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _QaPairMetric:
    """Per-pair artefacts captured between ``record_init`` and ``record_finish``."""

    qa_pair: QAPair | None = None
    search_result: list[SearchHitLike] = field(default_factory=list)
    rerank_result: list[SearchHitLike] = field(default_factory=list)
    chat_response: ChatResponseLike | None = None


class HookMetric:
    """Tracks evaluation metrics for QA pairs.

    Mirrors the upstream ``HookMetric``: ``record_init`` allocates the
    per-index slot, the ``record_*`` methods fill it with pipeline
    artefacts, and ``record_finish`` maps the retrieved chunks back to
    ground-truth passage ids, computes the metric bundle, and appends it
    to the aggregate list.
    """

    def __init__(self, capacity: int) -> None:
        self._pairs: list[_QaPairMetric | None] = [None] * max(0, capacity)
        self._list = MetricList()
        self._lock = threading.RLock()

    def record_init(self, index: int) -> None:
        """Allocate the slot for pair ``index``."""
        self._pairs[index] = _QaPairMetric()

    def record_qa_pair(self, index: int, qa_pair: QAPair) -> None:
        """Capture the ground-truth pair."""
        slot = self._require_slot(index)
        self._pairs[index] = _QaPairMetric(
            qa_pair=qa_pair,
            search_result=slot.search_result,
            rerank_result=slot.rerank_result,
            chat_response=slot.chat_response,
        )

    def record_search_result(self, index: int, search_result: list[SearchHitLike]) -> None:
        """Capture the pre-rerank search hits."""
        slot = self._require_slot(index)
        self._pairs[index] = _QaPairMetric(
            qa_pair=slot.qa_pair,
            search_result=list(search_result),
            rerank_result=slot.rerank_result,
            chat_response=slot.chat_response,
        )

    def record_rerank_result(self, index: int, rerank_result: list[SearchHitLike]) -> None:
        """Capture the reranked hits."""
        slot = self._require_slot(index)
        self._pairs[index] = _QaPairMetric(
            qa_pair=slot.qa_pair,
            search_result=slot.search_result,
            rerank_result=list(rerank_result),
            chat_response=slot.chat_response,
        )

    def record_chat_response(self, index: int, chat_response: ChatResponseLike | None) -> None:
        """Capture the generated answer."""
        slot = self._require_slot(index)
        self._pairs[index] = _QaPairMetric(
            qa_pair=slot.qa_pair,
            search_result=slot.search_result,
            rerank_result=slot.rerank_result,
            chat_response=chat_response,
        )

    def record_finish(self, index: int) -> None:
        """Compute the pair's bundle and append it to the aggregate list."""
        slot = self._pairs[index]
        if slot is None or slot.qa_pair is None:
            return
        metric_input = _build_metric_input(slot)
        result = _compute_all(metric_input)
        with self._lock:
            self._list.append(result)

    def metric_result(self) -> MetricResult:
        """Return the averaged metric bundle (all-zero when nothing recorded)."""
        with self._lock:
            return self._list.avg()

    # ── Internal ─────────────────────────────────────────────────────

    def _require_slot(self, index: int) -> _QaPairMetric:
        slot = self._pairs[index]
        if slot is None:
            slot = _QaPairMetric()
            self._pairs[index] = slot
        return slot


# ── Metric input construction ──────────────────────────────────────────


def _build_metric_input(slot: _QaPairMetric) -> MetricInput:
    """Map captured artefacts onto a :class:`MetricInput`.

    Retrieval source prefers the reranked hits and falls back to the raw
    search hits (upstream ``recordFinish`` behaviour). Each retrieved
    chunk's content is matched against the ground-truth passages so the
    chunk's KB-relative index is translated back into the dataset's
    passage id.
    """
    qa_pair = slot.qa_pair
    assert qa_pair is not None
    retrieval_source = slot.rerank_result or slot.search_result

    retrieval_ids: list[int] = []
    seen: set[int] = set()
    for hit in retrieval_source:
        content = _content_of(hit)
        if not content:
            continue
        for pid, passage in zip(qa_pair.pids, qa_pair.passages, strict=False):
            if not passage:
                continue
            if content in passage or passage in content:
                if pid not in seen:
                    seen.add(pid)
                    retrieval_ids.append(pid)
                break

    generated_texts = _content_of(slot.chat_response)
    return MetricInput(
        retrieval_gt=[list(qa_pair.pids)],
        retrieval_ids=retrieval_ids,
        generated_texts=generated_texts,
        generated_gt=qa_pair.answer,
    )


def _content_of(value: SearchHitLike | ChatResponseLike | None) -> str:
    """Extract the text payload from a pipeline artefact."""
    if value is None:
        return ""
    content = getattr(value, "content", "")
    if isinstance(content, str):
        return content
    return ""


# ── Retrieval metric calculators ──────────────────────────────────────


def _compute_precision(metric_input: MetricInput) -> float:
    """Precision = average over ground-truth sets of |hit| / |retrieved|."""
    gts = metric_input.retrieval_gt
    ids = metric_input.retrieval_ids
    if not gts or not ids:
        return 0.0
    total = 0.0
    for gt in gts:
        gt_set = set(gt)
        hits = sum(1 for doc_id in ids if doc_id in gt_set)
        total += hits / len(ids)
    return total / len(gts)


def _compute_recall(metric_input: MetricInput) -> float:
    """Recall = average over ground-truth sets of |hit| / |gt|."""
    gts = metric_input.retrieval_gt
    ids = metric_input.retrieval_ids
    if not gts or not ids:
        return 0.0
    total = 0.0
    for gt in gts:
        gt_set = set(gt)
        if not gt_set:
            continue
        hits = sum(1 for doc_id in ids if doc_id in gt_set)
        total += hits / len(gt_set)
    return total / len(gts)


def _compute_ndcg(metric_input: MetricInput, k: int) -> float:
    """NDCG@k over the ranked retrieved ids."""
    gts = metric_input.retrieval_gt
    ids = metric_input.retrieval_ids[:k]
    relevant: set[int] = set()
    count_gt = 0
    for gt in gts:
        count_gt += len(gt)
        relevant.update(gt)

    dcg = 0.0
    for i, doc_id in enumerate(ids):
        gain = 1.0 if doc_id in relevant else 0.0
        dcg += (2**gain - 1) / math.log2(i + 2)

    ideal_len = min(count_gt, len(ids))
    idcg = 0.0
    for i in range(ideal_len):
        idcg += 1.0 / math.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def _compute_mrr(metric_input: MetricInput) -> float:
    """Mean reciprocal rank across ground-truth sets."""
    gts = metric_input.retrieval_gt
    ids = metric_input.retrieval_ids
    if not gts:
        return 0.0
    total = 0.0
    for gt in gts:
        gt_set = set(gt)
        for i, doc_id in enumerate(ids):
            if doc_id in gt_set:
                total += 1.0 / (i + 1)
                break
    return total / len(gts)


def _compute_map(metric_input: MetricInput) -> float:
    """Mean average precision across ground-truth sets."""
    gts = metric_input.retrieval_gt
    ids = metric_input.retrieval_ids
    if not gts:
        return 0.0
    total = 0.0
    for gt in gts:
        gt_set = set(gt)
        hit_count = 0
        ap = 0.0
        for i, doc_id in enumerate(ids):
            if doc_id in gt_set:
                hit_count += 1
                ap += hit_count / (i + 1)
        if hit_count > 0:
            ap /= hit_count
        total += ap
    return total / len(gts)


# ── Generation metric calculators ─────────────────────────────────────


def _compute_bleu(
    metric_input: MetricInput,
    weights: tuple[float, float, float, float],
) -> float:
    """BLEU with the upstream n-gram weights + length smoothing."""
    candidate = _tokenize(metric_input.generated_texts)
    references = [_tokenize(metric_input.generated_gt)]

    precisions = [
        _modified_precision(candidate, references, n, smoothing=True)
        for n in range(1, 5)
    ]

    overlap = 0
    score = 0.0
    for weight, pn in zip(weights, precisions, strict=True):
        if pn > 0.0:
            overlap += 1
            score += weight * math.log(pn)
    if overlap == 0:
        return 0.0

    bp = _brevity_penalty(candidate, references)
    return bp * math.exp(score)


def _compute_rouge(metric_input: MetricInput, name: str) -> float:
    """ROUGE-1 / ROUGE-2 / ROUGE-L F-measure."""
    hyp_sentences = _split_sentences(metric_input.generated_texts)
    ref_sentences = _split_sentences(metric_input.generated_gt)
    if name == "rouge-l":
        return _rouge_l_summary(hyp_sentences, ref_sentences)
    n = 1 if name == "rouge-1" else 2
    return _rouge_n(hyp_sentences, ref_sentences, n)


# ── Text preprocessing (upstream ``common.go``) ────────────────────────


#: Chinese full stop / English period — the sentence boundary used by
#: the upstream splitter.
_SENTENCE_BOUNDARY = "([。.])"

#: Matches a Chinese block or an English / alphanumeric run. Punctuation
#: is recovered from the gaps between matches via :func:`unicodedata`.
_WORD_RE = re.compile(r"([一-鿿]+)|([a-zA-Z0-9_.,!?]+)")


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` on the boundary, keeping non-empty sentences."""
    parts = re.split(_SENTENCE_BOUNDARY, text)
    sentences: list[str] = []
    current: list[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            current.append(part)
        else:
            sentence = "".join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []
    remaining = "".join(current).strip()
    if remaining:
        sentences.append(remaining)
    return sentences


def _split_into_words(sentences: list[str]) -> list[str]:
    """Tokenize sentences into words.

    Chinese blocks run through jieba; English blocks split on
    whitespace; each punctuation character becomes its own token.
    Mirrors the upstream ``splitIntoWords`` helper.
    """
    tokens: list[str] = []
    for text in sentences:
        pos = 0
        for match in _WORD_RE.finditer(text):
            tokens.extend(_punctuation_tokens(text[pos : match.start()]))
            chinese_block, english_block = match.groups()
            if chinese_block:
                tokens.extend(_cut_chinese(chinese_block))
            elif english_block:
                tokens.extend(english_block.split())
            pos = match.end()
        tokens.extend(_punctuation_tokens(text[pos:]))
    return tokens


def _punctuation_tokens(chunk: str) -> list[str]:
    """Yield each Unicode punctuation char of ``chunk`` as its own token."""
    tokens: list[str] = []
    for char in chunk:
        if char.isspace():
            continue
        if unicodedata.category(char).startswith("P"):
            tokens.append(char)
    return tokens


def _cut_chinese(block: str) -> list[str]:
    """Segment one Chinese block with jieba (HMM on, like the upstream)."""
    return list(jieba.cut(block, HMM=True))


def _tokenize(text: str) -> list[str]:
    """Full tokenization pipeline: sentences → words → lowercase."""
    return [word.lower() for word in _split_into_words(_split_sentences(text))]


# ── BLEU internals ─────────────────────────────────────────────────────


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """All contiguous n-grams of ``tokens``."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _modified_precision(
    candidate: list[str],
    references: list[list[str]],
    n: int,
    *,
    smoothing: bool,
) -> float:
    """Clipped n-gram precision against the reference set."""
    candidate_grams = _ngrams(candidate, n)
    if not candidate_grams:
        return 0.0
    counts: dict[tuple[str, ...], int] = {}
    for gram in candidate_grams:
        counts[gram] = counts.get(gram, 0) + 1

    max_counts: dict[tuple[str, ...], int] = {}
    for reference in references:
        ref_counts: dict[tuple[str, ...], int] = {}
        for gram in _ngrams(reference, n):
            ref_counts[gram] = ref_counts.get(gram, 0) + 1
        for gram in counts:
            max_counts[gram] = max(max_counts.get(gram, 0), ref_counts.get(gram, 0))

    clipped = sum(min(count, max_counts.get(gram, 0)) for gram, count in counts.items())
    total = sum(counts.values())
    smoothing_factor = 1.0 if smoothing else 0.0
    return (clipped + smoothing_factor) / (total + smoothing_factor)


def _brevity_penalty(candidate: list[str], references: list[list[str]]) -> float:
    """Length penalty: 1.0 when the candidate is long enough, else exp."""
    c = len(candidate)
    if c == 0:
        return 0.0
    ref_lens = [len(ref) for ref in references]
    closest = min(ref_lens, key=lambda length: (abs(length - c), length))
    if c > closest:
        return 1.0
    return math.exp(1.0 - closest / c)


# ── ROUGE internals ────────────────────────────────────────────────────


def _rouge_n(
    hyp_sentences: list[str],
    ref_sentences: list[str],
    n: int,
) -> float:
    """ROUGE-N F-measure over the token n-gram sets."""
    hyp = _split_into_words(hyp_sentences)
    ref = _split_into_words(ref_sentences)
    hyp_ngrams = set(_ngrams(hyp, n))
    ref_ngrams = set(_ngrams(ref, n))
    overlap = len(hyp_ngrams & ref_ngrams)
    p = overlap / len(hyp_ngrams) if hyp_ngrams else 0.0
    r = overlap / len(ref_ngrams) if ref_ngrams else 0.0
    return 2.0 * (p * r / (p + r + 1e-8))


def _rouge_l_summary(hyp_sentences: list[str], ref_sentences: list[str]) -> float:
    """ROUGE-L F-measure at summary level (union of per-sentence LCS)."""
    ref_words = _split_into_words(ref_sentences)
    m = len(ref_words)
    hyp_words = _split_into_words(hyp_sentences)
    n = len(hyp_words)

    llcs = 0
    union: set[str] = set()
    for ref_sentence in ref_sentences:
        ref_sentence_words = _split_into_words([ref_sentence])
        for hyp_sentence in hyp_sentences:
            hyp_sentence_words = _split_into_words([hyp_sentence])
            lcs_words = _lcs(ref_sentence_words, hyp_sentence_words)
            for word in lcs_words:
                if word not in union:
                    union.add(word)
                    llcs += 1

    r_lcs = llcs / m if m else 0.0
    p_lcs = llcs / n if n else 0.0
    return 2.0 * (p_lcs * r_lcs / (p_lcs + r_lcs + 1e-8))


def _lcs(left: list[str], right: list[str]) -> list[str]:
    """Longest common subsequence, reconstructed from the DP table."""
    rows, cols = len(left), len(right)
    table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if left[i - 1] == right[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])

    i, j = rows, cols
    sequence: list[str] = []
    while i > 0 and j > 0:
        if left[i - 1] == right[j - 1]:
            sequence.append(left[i - 1])
            i -= 1
            j -= 1
        elif table[i - 1][j] >= table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    sequence.reverse()
    return sequence


# ── Calculator wiring ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _CalculatorConfig:
    """One configured metric: its calculator and the result field it fills."""

    calc: Callable[[MetricInput], float]
    field: str


_CALCULATORS: tuple[_CalculatorConfig, ...] = (
    _CalculatorConfig(_compute_precision, "precision"),
    _CalculatorConfig(_compute_recall, "recall"),
    _CalculatorConfig(lambda inp: _compute_ndcg(inp, 3), "ndcg3"),
    _CalculatorConfig(lambda inp: _compute_ndcg(inp, 10), "ndcg10"),
    _CalculatorConfig(_compute_mrr, "mrr"),
    _CalculatorConfig(_compute_map, "map_"),
    _CalculatorConfig(lambda inp: _compute_bleu(inp, (1.0, 0.0, 0.0, 0.0)), "bleu1"),
    _CalculatorConfig(lambda inp: _compute_bleu(inp, (0.5, 0.5, 0.0, 0.0)), "bleu2"),
    _CalculatorConfig(lambda inp: _compute_bleu(inp, (0.25, 0.25, 0.25, 0.25)), "bleu4"),
    _CalculatorConfig(lambda inp: _compute_rouge(inp, "rouge-1"), "rouge1"),
    _CalculatorConfig(lambda inp: _compute_rouge(inp, "rouge-2"), "rouge2"),
    _CalculatorConfig(lambda inp: _compute_rouge(inp, "rouge-l"), "rougel"),
)


def _compute_all(metric_input: MetricInput) -> MetricResult:
    """Run every configured calculator over one pair's input."""
    scores: dict[str, float] = {}
    for config in _CALCULATORS:
        scores[config.field] = config.calc(metric_input)
    return MetricResult(
        retrieval=RetrievalMetrics(
            precision=scores["precision"],
            recall=scores["recall"],
            ndcg3=scores["ndcg3"],
            ndcg10=scores["ndcg10"],
            mrr=scores["mrr"],
            map_=scores["map_"],
        ),
        generation=GenerationMetrics(
            bleu1=scores["bleu1"],
            bleu2=scores["bleu2"],
            bleu4=scores["bleu4"],
            rouge1=scores["rouge1"],
            rouge2=scores["rouge2"],
            rougel=scores["rougel"],
        ),
    )


def _add(left: MetricResult, right: MetricResult) -> MetricResult:
    """Element-wise sum of two bundles (helper for ``MetricList.avg``)."""
    return MetricResult(
        retrieval=RetrievalMetrics(
            precision=left.retrieval.precision + right.retrieval.precision,
            recall=left.retrieval.recall + right.retrieval.recall,
            ndcg3=left.retrieval.ndcg3 + right.retrieval.ndcg3,
            ndcg10=left.retrieval.ndcg10 + right.retrieval.ndcg10,
            mrr=left.retrieval.mrr + right.retrieval.mrr,
            map_=left.retrieval.map_ + right.retrieval.map_,
        ),
        generation=GenerationMetrics(
            bleu1=left.generation.bleu1 + right.generation.bleu1,
            bleu2=left.generation.bleu2 + right.generation.bleu2,
            bleu4=left.generation.bleu4 + right.generation.bleu4,
            rouge1=left.generation.rouge1 + right.generation.rouge1,
            rouge2=left.generation.rouge2 + right.generation.rouge2,
            rougel=left.generation.rougel + right.generation.rougel,
        ),
    )


__all__ = [
    "ChatResponseLike",
    "GenerationMetrics",
    "HookMetric",
    "MetricInput",
    "MetricList",
    "MetricResult",
    "RetrievalMetrics",
    "SearchHitLike",
]
