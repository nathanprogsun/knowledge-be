"""Score normalization for cross-engine result comparison (upstream ``normalizer.go``).

``ScoreNormalizer`` maps raw retriever scores to a common [0, 1] scale so
that vector results produced by different engines can be compared in a
single ranked list. Implementations MUST be IO-free (``normalize`` is called
inside a hot loop and may not log or block) and safe for concurrent use.

Only vector scores are normalized. Keyword (BM25) scores have an unbounded
positive range; rescaling them would collapse the long tail. Downstream
rank-based (RRF) fusion is immune to scale, so keyword scores pass through
unchanged.

``EngineAwareNormalizer`` applies the documented per-engine cosine-score
formula. The caller enforces a same-embedding-model precondition, so
post-normalization values are semantically comparable across engines:

- Milvus (COSINE metric) surfaces the raw signed cosine range [-1, 1] and
  is the only engine in this codebase whose driver does so; its score is
  rescaled via ``(score + 1) / 2``.
- Every other engine already yields a score in [0, 1] at the normalizer's
  input: Lucene's ``script_score`` non-negative invariant (Elasticsearch),
  the k-NN plugin's cosinesimil pre-translation (OpenSearch), Weaviate's
  intrinsic certainty, and the L2-normalized positive-component IR
  embeddings that keep cosine inside [0, 1] for pgvector / sqlite-vec /
  Qdrant / TencentVectorDB / Doris. These pass through unchanged.
- Unknown engines clamp defensively to [0, 1]; the fan-out caller emits a
  single warning per request so ``normalize`` itself stays lock-free.
"""

from __future__ import annotations

import math
from typing import Protocol

from src.ai.embedding import Context
from src.ai.retrieval.types import RetrieverEngineType, RetrieverType


class ScoreNormalizer(Protocol):
    """Maps a raw retriever score to the common [0, 1] scale.

    ``ctx`` is accepted for signature parity; implementations must not
    read it (the call sits inside a hot loop).
    """

    def normalize(
        self,
        ctx: Context,
        score: float,
        retriever_type: RetrieverType,
        engine_type: RetrieverEngineType,
    ) -> float: ...


class EngineAwareNormalizer:
    """Per-engine cosine-score normalizer (see module docstring)."""

    def normalize(
        self,
        _ctx: Context,
        score: float,
        retriever_type: RetrieverType,
        engine_type: RetrieverEngineType,
    ) -> float:
        if retriever_type != RetrieverType.VECTOR:
            # BM25 and other non-vector retrievers: passthrough. Rank-based
            # fusion handles scale-mixed input correctly.
            return score
        if engine_type == RetrieverEngineType.MILVUS:
            # Raw cosine in [-1, 1] → [0, 1]; clamp once more so a
            # misbehaving engine cannot leak past the envelope (the caller
            # sorts by score afterwards).
            return clamp01((score + 1) / 2)
        # Already in [0, 1] when the value reaches us; clamp defensively.
        return clamp01(score)


def clamp01(score: float) -> float:
    """Map any float into [0, 1] safely, including NaN/Inf inputs.

    NaN would otherwise poison the strict-weak-ordering invariant of the
    downstream sort comparator (it compares neither greater nor less than
    anything).
    """
    if math.isnan(score):
        return 0.0
    if score <= 0 or (math.isinf(score) and score < 0):
        return 0.0
    if score >= 1 or (math.isinf(score) and score > 0):
        return 1.0
    return score


__all__ = [
    "EngineAwareNormalizer",
    "ScoreNormalizer",
    "clamp01",
]
