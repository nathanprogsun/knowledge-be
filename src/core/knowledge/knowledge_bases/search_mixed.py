"""Fusion of vector and keyword retrieval results.

Vector-only and keyword-only result sets are deduplicated by chunk id
keeping the best score; hybrid result sets are merged with weighted
Reciprocal Rank Fusion. The RRF parameters (smoothing constant ``k``,
vector weight, keyword weight) are supplied by the caller — typically
from the tenant retrieval config — with documented defaults.
"""

from __future__ import annotations

from src.ai.retrieval.types import IndexWithScore, RetrieveResult, RetrieverType

#: RRF defaults when the retrieval config is absent.
_DEFAULT_RRF_K = 60
_DEFAULT_RRF_VECTOR_WEIGHT = 0.7
_DEFAULT_RRF_KEYWORD_WEIGHT = 0.3


def classify_retrieval_results(
    results: list[RetrieveResult],
) -> tuple[list[IndexWithScore], list[IndexWithScore]]:
    """Split retrieval results into (vector, keyword) hit lists.

    Any non-vector retriever lands in the keyword bucket, mirroring the
    two-bucket model of the fusion stage.
    """
    vector: list[IndexWithScore] = []
    keyword: list[IndexWithScore] = []
    for result in results:
        if result.retriever_type == RetrieverType.VECTOR:
            vector.extend(result.results)
        else:
            keyword.extend(result.results)
    return vector, keyword


def fuse_or_deduplicate(
    vector_results: list[IndexWithScore],
    keyword_results: list[IndexWithScore],
    *,
    rrf_k: int = _DEFAULT_RRF_K,
    vector_weight: float = _DEFAULT_RRF_VECTOR_WEIGHT,
    keyword_weight: float = _DEFAULT_RRF_KEYWORD_WEIGHT,
) -> list[IndexWithScore]:
    """Fuse hybrid results via RRF, or deduplicate single-retriever results.

    A single empty retriever falls through to deduplication of the other
    one so its original scores are preserved (important for FAQ hits).
    """
    if not keyword_results:
        return deduplicate_by_score(vector_results)
    if not vector_results:
        return deduplicate_by_score(keyword_results)
    return fuse_with_rrf(
        vector_results,
        keyword_results,
        rrf_k=rrf_k,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )


def deduplicate_by_score(results: list[IndexWithScore]) -> list[IndexWithScore]:
    """Keep the highest-scoring hit per chunk id, sorted by score desc."""
    best: dict[str, IndexWithScore] = {}
    for hit in results:
        existing = best.get(hit.chunk_id)
        if existing is None or hit.score > existing.score:
            best[hit.chunk_id] = hit
    return sorted(best.values(), key=lambda hit: hit.score, reverse=True)


def fuse_with_rrf(
    vector_results: list[IndexWithScore],
    keyword_results: list[IndexWithScore],
    *,
    rrf_k: int = _DEFAULT_RRF_K,
    vector_weight: float = _DEFAULT_RRF_VECTOR_WEIGHT,
    keyword_weight: float = _DEFAULT_RRF_KEYWORD_WEIGHT,
) -> list[IndexWithScore]:
    """Merge two ranked lists via weighted Reciprocal Rank Fusion.

    ``RRF = vector_weight/(k+vector_rank) + keyword_weight/(k+keyword_rank)``
    with 1-indexed ranks. Chunk metadata prefers the vector hit (then the
    higher-scoring duplicate); fused results are sorted by RRF score desc.
    """
    vector_ranks = _first_ranks(vector_results)
    keyword_ranks = _first_ranks(keyword_results)

    by_chunk: dict[str, IndexWithScore] = {}
    for hit in vector_results:
        existing = by_chunk.get(hit.chunk_id)
        if existing is None or hit.score > existing.score:
            by_chunk[hit.chunk_id] = hit
    for hit in keyword_results:
        if hit.chunk_id not in by_chunk:
            by_chunk[hit.chunk_id] = hit

    fused: list[IndexWithScore] = []
    for chunk_id, info in by_chunk.items():
        score = 0.0
        rank = vector_ranks.get(chunk_id)
        if rank is not None:
            score += vector_weight / (rrf_k + rank)
        rank = keyword_ranks.get(chunk_id)
        if rank is not None:
            score += keyword_weight / (rrf_k + rank)
        fused.append(info.model_copy(update={"score": score}))
    fused.sort(key=lambda hit: hit.score, reverse=True)
    return fused


def _first_ranks(results: list[IndexWithScore]) -> dict[str, int]:
    """Map chunk id to its first (1-indexed) rank in ``results``."""
    ranks: dict[str, int] = {}
    for index, hit in enumerate(results):
        if hit.chunk_id not in ranks:
            ranks[hit.chunk_id] = index + 1
    return ranks


__all__ = [
    "classify_retrieval_results",
    "deduplicate_by_score",
    "fuse_or_deduplicate",
    "fuse_with_rrf",
]
