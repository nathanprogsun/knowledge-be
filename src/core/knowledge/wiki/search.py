"""Wiki page full-text search exposed as standalone functions.

:func:`search_pages` validates the query, clamps the result window to
the repository's safe range, and strips the internal chunk-citation
handles from every hit before returning (mirrors the search semantics).
"""

from __future__ import annotations

from src.common.exception import ValidationError
from src.core.knowledge.wiki.link_utils import strip_page_chunk_citations
from src.db.dao.wiki_page_repository import WikiPageRepository
from src.db.models.wiki_page import WikiPage

_SEARCH_DEFAULT_LIMIT = 10
_SEARCH_MAX_LIMIT = 50


async def search_pages(
    *,
    page_repo: WikiPageRepository,
    knowledge_base_id: str,
    query: str,
    limit: int = _SEARCH_DEFAULT_LIMIT,
) -> list[WikiPage]:
    """Search the KB's live wiki pages, ranked by where the query hit.

    A title hit outranks a slug hit, which outranks a summary hit, which
    outranks a body mention. The result window is clamped to ``[1, 50]``
    with a ``10`` default. Internal chunk-citation handles are stripped
    from every returned body / summary.
    """
    if not query.strip():
        raise ValidationError(
            code="wiki.search_query_required",
            message="search query is required",
        )
    limit = _SEARCH_DEFAULT_LIMIT if limit <= 0 else min(limit, _SEARCH_MAX_LIMIT)
    pages = await page_repo.search(knowledge_base_id=knowledge_base_id, query=query, limit=limit)
    return [strip_page_chunk_citations(page) for page in pages]


__all__ = ["search_pages"]
