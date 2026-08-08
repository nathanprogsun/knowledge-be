"""Wiki aggregate statistics exposed as standalone functions.

:func:`get_stats` layers the aggregate reads over the merged repository
and keeps the latency-sensitive wiring — the pending ingest-task count,
the pending issue count, and the in-progress flag — behind injectable
seams so the web layer can provide live providers and tests can stub
them. Any seam left unset reports the neutral zero / ``False`` value.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from src.core.knowledge.wiki.link_utils import strip_page_chunk_citations
from src.core.knowledge.wiki.types import WikiStats
from src.db.dao.wiki_page_repository import WikiPageRepository

# The recent-updates window mirrors the page-listing default page size.
_RECENT_WINDOW = 10


async def get_stats(
    *,
    page_repo: WikiPageRepository,
    knowledge_base_id: str,
    pending_task_count: Callable[[], Awaitable[int]] | None = None,
    pending_issue_count: Callable[[], Awaitable[int]] | None = None,
    is_active: Callable[[], Awaitable[bool]] | None = None,
) -> WikiStats:
    """Return aggregate statistics about the KB's wiki.

    ``pending_task_count`` / ``pending_issue_count`` / ``is_active`` are
    injectable providers for the ingest-queue and in-progress signals;
    ``None`` reports the neutral ``0`` / ``False``.
    """
    counts = await page_repo.count_by_type(knowledge_base_id=knowledge_base_id)
    total = sum(counts.values())
    orphans = await page_repo.count_orphans(knowledge_base_id=knowledge_base_id)
    pages = await page_repo.list_all(knowledge_base_id=knowledge_base_id)
    total_links = sum(len(page.out_links) for page in pages)
    recent, _ = await page_repo.list_pages(
        knowledge_base_id=knowledge_base_id,
        page=1,
        page_size=_RECENT_WINDOW,
        sort_by="updated_at",
        sort_order="desc",
    )

    pending_tasks = await pending_task_count() if pending_task_count is not None else 0
    pending_issues = await pending_issue_count() if pending_issue_count is not None else 0
    active = await is_active() if is_active is not None else False

    return WikiStats(
        total_pages=total,
        pages_by_type=counts,
        total_links=total_links,
        orphan_count=orphans,
        recent_updates=[strip_page_chunk_citations(page) for page in recent],
        pending_tasks=pending_tasks,
        pending_issues=pending_issues,
        is_active=active,
    )


__all__ = ["WikiStats", "get_stats"]
