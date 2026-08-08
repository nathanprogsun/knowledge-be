"""FAQ-specific retrieval post-processing.

Two paths, mirroring the upstream semantics:

- **Iterative retrieval**: when a single pass returns fewer unique chunks
  than requested while the vector fan-out hit its over-retrieval cap, the
  top-k grows in rounds until enough unique chunks are found or the
  corpus is exhausted. FAQ metadata is looked up through the injectable
  seam and cached across rounds so negative-question filtering never
  repeats a store read.
- **Negative-question filtering**: otherwise, FAQ hits whose negative
  questions equal the query are dropped.

The negative-question lookup is an injectable seam over the FAQ store, so
the search layer never touches the FAQ repository directly.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol, TypeAlias, runtime_checkable

from src.ai.embedding import Context
from src.ai.retrieval.types import IndexWithScore
from src.core.knowledge.knowledge_bases.types import KNOWLEDGE_BASE_TYPE_FAQ

#: Maximum iterative retrieval rounds before giving up.
_MAX_ITERATIONS = 5
#: Over-retrieval multiplier used on the first iterative round.
_INITIAL_TOP_K_MULTIPLIER = 3
#: Top-k doubling factor between rounds.
_TOP_K_GROWTH = 2


@runtime_checkable
class FaqMetadataLoader(Protocol):
    """Negative-question lookup seam over the FAQ store.

    ``load_negative_questions`` returns a map from chunk id to the entry's
    negative questions; chunk ids without an entry are absent.
    """

    async def load_negative_questions(
        self, ctx: Context, chunk_ids: list[str]
    ) -> dict[str, tuple[str, ...]]: ...


#: Re-runs the retrieval fan-out with a new over-retrieval top-k and
#: returns the merged raw hits.
RetrieveCallable: TypeAlias = Callable[[int], Awaitable[list[IndexWithScore]]]


async def apply_faq_post_processing(
    ctx: Context,
    *,
    kb_type: str,
    chunks: list[IndexWithScore],
    vector_result_count: int,
    requested_count: int,
    over_retrieve_count: int,
    query_text: str,
    retrieve: RetrieveCallable,
    faq_loader: FaqMetadataLoader | None = None,
) -> list[IndexWithScore]:
    """Apply FAQ post-processing; non-FAQ KBs pass through unchanged.

    Iterative retrieval runs when the first pass under-delivered unique
    chunks (``len(chunks) < requested_count``) while the vector fan-out
    returned its full over-retrieval cap (``vector_result_count ==
    over_retrieve_count``) — the signal that more matches sit below the
    cut-off. The iterative target is the requested count.
    """
    if kb_type != KNOWLEDGE_BASE_TYPE_FAQ:
        return chunks
    if len(chunks) < requested_count and vector_result_count == over_retrieve_count:
        return await iterative_retrieve_with_deduplication(
            ctx,
            match_count=requested_count,
            query_text=query_text,
            retrieve=retrieve,
            faq_loader=faq_loader,
        )
    return await filter_by_negative_questions(ctx, chunks, query_text, faq_loader)


async def iterative_retrieve_with_deduplication(
    ctx: Context,
    *,
    match_count: int,
    query_text: str,
    retrieve: RetrieveCallable,
    faq_loader: FaqMetadataLoader | None = None,
) -> list[IndexWithScore]:
    """Retrieve in growing rounds until enough unique chunks are found.

    Each round re-fans out with an increased top-k; negative-question
    matches evict a chunk from the candidate set. Returns the surviving
    chunks sorted by score desc.
    """
    current_top_k = max(match_count * _INITIAL_TOP_K_MULTIPLIER, 1)
    unique: dict[str, IndexWithScore] = {}
    filtered_out: set[str] = set()
    query_lower = _normalise_query(query_text)

    for _ in range(_MAX_ITERATIONS):
        results = await retrieve(current_top_k)
        if not results:
            break

        new_chunk_ids = [
            hit.chunk_id
            for hit in results
            if hit.chunk_id not in unique and hit.chunk_id not in filtered_out
        ]
        negative_by_chunk = await _load_negative(ctx, faq_loader, new_chunk_ids)

        for hit in results:
            if hit.chunk_id in filtered_out:
                continue
            negatives = negative_by_chunk.get(hit.chunk_id)
            if negatives and matches_negative_questions(query_lower, negatives):
                filtered_out.add(hit.chunk_id)
                unique.pop(hit.chunk_id, None)
                continue
            existing = unique.get(hit.chunk_id)
            if existing is None or hit.score > existing.score:
                unique[hit.chunk_id] = hit

        if len(unique) >= match_count:
            break
        if len(results) < current_top_k:
            break
        current_top_k *= _TOP_K_GROWTH

    return sorted(unique.values(), key=lambda hit: hit.score, reverse=True)


async def filter_by_negative_questions(
    ctx: Context,
    chunks: list[IndexWithScore],
    query_text: str,
    faq_loader: FaqMetadataLoader | None = None,
) -> list[IndexWithScore]:
    """Drop FAQ hits whose negative questions match the query.

    Only FAQ entries are consulted; chunks without an entry or with an
    unreadable entry are kept. Returns the survivors in input order.
    """
    if not chunks:
        return chunks
    query_lower = _normalise_query(query_text)
    if not query_lower:
        return chunks
    negative_by_chunk = await _load_negative(ctx, faq_loader, [hit.chunk_id for hit in chunks])
    if not negative_by_chunk:
        return chunks

    kept: list[IndexWithScore] = []
    for hit in chunks:
        negatives = negative_by_chunk.get(hit.chunk_id)
        if negatives and matches_negative_questions(query_lower, negatives):
            continue
        kept.append(hit)
    return kept


def matches_negative_questions(query_lower: str, negative_questions: Sequence[str]) -> bool:
    """Report whether the (lower-cased) query equals any negative question.

    Both sides are trimmed and lower-cased; empty entries never match.
    """
    for raw in negative_questions:
        candidate = raw.strip().lower()
        if candidate and query_lower == candidate:
            return True
    return False


async def _load_negative(
    ctx: Context,
    loader: FaqMetadataLoader | None,
    chunk_ids: list[str],
) -> dict[str, tuple[str, ...]]:
    """Best-effort negative-question lookup; store failures degrade to empty.

    A failing FAQ lookup must not silently truncate recall, so the caller
    keeps the chunks and relies on the vector/keyword scores alone.
    """
    if loader is None or not chunk_ids:
        return {}
    try:
        return await loader.load_negative_questions(ctx, chunk_ids)
    except Exception:
        return {}


def _normalise_query(query_text: str) -> str:
    return query_text.strip().lower()


__all__ = [
    "FaqMetadataLoader",
    "RetrieveCallable",
    "apply_faq_post_processing",
    "filter_by_negative_questions",
    "iterative_retrieve_with_deduplication",
    "matches_negative_questions",
]
