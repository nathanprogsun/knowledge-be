"""Query preparation for hybrid search.

``prepare_query`` normalises the raw query text, applies the injectable
query-rewrite seam when one is supplied, and optionally computes the query
embedding once so every store group shares a single vector.

The rewrite seam is a callable protocol satisfied by any wrapper over a
chat client in ``src/ai/llm`` — the search layer never dials an LLM API
directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.ai.embedding import Context, Embedder


@runtime_checkable
class QueryRewriter(Protocol):
    """Query-rewrite seam (wrapped over a chat client)."""

    async def rewrite(self, ctx: Context, query: str) -> str: ...


class PassthroughQueryRewriter:
    """Default seam that leaves the query untouched."""

    async def rewrite(self, _ctx: Context, query: str) -> str:
        return query


@dataclass(frozen=True, slots=True)
class SearchQuery:
    """Prepared query: normalised text, effective text and embedding."""

    text: str = ""
    rewritten: str = ""
    embedding: tuple[float, ...] = ()


async def prepare_query(
    ctx: Context,
    *,
    query_text: str,
    needs_embedding: bool,
    embedder: Embedder | None = None,
    rewriter: QueryRewriter | None = None,
) -> SearchQuery:
    """Normalise, optionally rewrite, and optionally embed the query.

    ``needs_embedding`` gates the embedder call (vector retrieval only);
    the embedder is invoked once so N store groups never trigger N
    identical API calls. The embedding is computed over the effective
    text — the rewritten query when a seam is present, the raw text
    otherwise — so the vector matches the keywords actually retrieved.
    """
    text = query_text.strip()
    rewritten = text
    if text and rewriter is not None:
        candidate = (await rewriter.rewrite(ctx, text)).strip()
        if candidate:
            rewritten = candidate
    embedding: tuple[float, ...] = ()
    if needs_embedding and embedder is not None:
        vector = await embedder.embed(ctx, rewritten)
        embedding = tuple(vector)
    return SearchQuery(text=text, rewritten=rewritten, embedding=embedding)


__all__ = [
    "PassthroughQueryRewriter",
    "QueryRewriter",
    "SearchQuery",
    "prepare_query",
]
