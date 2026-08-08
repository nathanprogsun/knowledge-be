"""Scope filtering and result trimming for hybrid search.

Two responsibilities:

- ``scope_retrieve_params`` copies a ``RetrieveParams`` and applies the
  search scope filters (knowledge ids, tag ids, excluded ids) so the
  index query itself is narrowed.
- ``filter_index_scores`` trims a ranked result list by the enabled flag,
  an optional minimum score, and the excluded-chunk set, preserving
  relative order.
"""

from __future__ import annotations

from collections.abc import Iterable

from src.ai.retrieval.types import IndexWithScore, RetrieveParams


def scope_retrieve_params(
    params: RetrieveParams,
    *,
    knowledge_ids: Iterable[str] = (),
    tag_ids: Iterable[str] = (),
    exclude_knowledge_ids: Iterable[str] = (),
    exclude_chunk_ids: Iterable[str] = (),
) -> RetrieveParams:
    """Return a copy of ``params`` narrowed to the given scope.

    Empty inputs leave the corresponding field untouched so callers can
    layer filters without clobbering engine defaults. The input is never
    mutated.
    """
    updates: dict[str, list[str]] = {}
    if knowledge := _dedupe(knowledge_ids):
        updates["knowledge_ids"] = knowledge
    if tags := _dedupe(tag_ids):
        updates["tag_ids"] = tags
    if exclude_knowledge := _dedupe(exclude_knowledge_ids):
        updates["exclude_knowledge_ids"] = exclude_knowledge
    if exclude_chunks := _dedupe(exclude_chunk_ids):
        updates["exclude_chunk_ids"] = exclude_chunks
    if not updates:
        return params
    return params.model_copy(update=updates)


def filter_index_scores(
    results: list[IndexWithScore],
    *,
    excluded_chunk_ids: Iterable[str] = (),
    threshold: float | None = None,
    enabled_only: bool = True,
) -> list[IndexWithScore]:
    """Keep enabled hits at or above ``threshold``, preserving order.

    A ``None`` threshold applies no score floor — keyword scores are
    unbounded and are never thresholded here. ``excluded_chunk_ids``
    drops hits whose chunk id is in the set (a defensive echo of the
    engine-side exclusion the scope filter already requested).
    """
    excluded = frozenset(excluded_chunk_ids)
    kept: list[IndexWithScore] = []
    for hit in results:
        if hit.chunk_id in excluded:
            continue
        if enabled_only and not hit.is_enabled:
            continue
        if threshold is not None and hit.score < threshold:
            continue
        kept.append(hit)
    return kept


def _dedupe(values: Iterable[str]) -> list[str]:
    """Deduplicate and order a string iterable, preserving first occurrence."""
    return list(dict.fromkeys(values))


__all__ = ["filter_index_scores", "scope_retrieve_params"]
