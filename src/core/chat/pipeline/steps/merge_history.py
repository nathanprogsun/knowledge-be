"""History-reference injection for the chunk-merge step.

Pulls the most recent conversation round that carries knowledge
references, filters them by Jaccard similarity to the cleaned-up query,
and discounts their scores so freshly retrieved hits rank above stale
references.
"""

from __future__ import annotations

from src.ai.retrieval.types import MatchType
from src.core.agents.tools.text_utils import jaccard, tokenize_simple
from src.core.chat.pipeline.common import pipeline_info
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.types import SearchResult

#: Minimum Jaccard similarity between the current query and a history
#: chunk's content for it to be injected.
MIN_SIMILARITY = 0.15
#: Reduces the original score of history results so they rank below
#: freshly retrieved results of similar relevance.
HISTORY_SCORE_DISCOUNT = 0.6
#: Caps the number of history results injected to avoid overwhelming the
#: context with stale references.
MAX_HISTORY_RESULTS = 3


def get_search_result_from_history(pipeline_ctx: PipelineContext) -> list[SearchResult]:
    """Return the most recent history round's references, marked as history hits.

    History is scanned in reverse chronological order; the first round that
    carries knowledge references wins.
    """
    for entry in reversed(pipeline_ctx.history):
        if entry.references:
            return [
                ref.model_copy(update={"match_type": MatchType.HISTORY}) for ref in entry.references
            ]
    return []


def filter_history_results(
    pipeline_ctx: PipelineContext,
    current_results: list[SearchResult],
) -> list[SearchResult]:
    """Filter history references by similarity and discount their scores.

    References already present in ``current_results`` (by chunk ID) are
    excluded, and only the top ``MAX_HISTORY_RESULTS`` similar references
    are kept.
    """
    raw = get_search_result_from_history(pipeline_ctx)
    if not raw:
        return []
    existing_ids = {result.id for result in current_results}
    query = pipeline_ctx.rewrite_query or pipeline_ctx.query
    query_tokens = tokenize_simple(query)
    filtered: list[SearchResult] = []
    for ref in raw:
        if ref.id in existing_ids:
            continue
        content_tokens = tokenize_simple(ref.content)
        similarity = jaccard(query_tokens, content_tokens)
        if similarity < MIN_SIMILARITY:
            pipeline_info(
                "Merge",
                "history_filter_drop",
                {"chunk_id": ref.id, "similarity": similarity},
            )
            continue
        discounted_score = ref.score * HISTORY_SCORE_DISCOUNT
        metadata = dict(ref.metadata)
        metadata["history_similarity"] = _format_similarity(similarity)
        filtered.append(
            ref.model_copy(
                update={
                    "match_type": MatchType.HISTORY,
                    "score": discounted_score,
                    "metadata": metadata,
                }
            )
        )
        pipeline_info(
            "Merge",
            "history_filter_keep",
            {
                "chunk_id": ref.id,
                "similarity": similarity,
                "new_score": discounted_score,
            },
        )
        if len(filtered) >= MAX_HISTORY_RESULTS:
            break
    return filtered


def _format_similarity(similarity: float) -> str:
    """Format a similarity float with trailing zeros and dot trimmed."""
    return f"{similarity:.4f}".rstrip("0").rstrip(".")


__all__ = [
    "HISTORY_SCORE_DISCOUNT",
    "MAX_HISTORY_RESULTS",
    "MIN_SIMILARITY",
    "filter_history_results",
    "get_search_result_from_history",
]
