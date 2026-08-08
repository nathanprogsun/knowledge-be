"""Wiki link analysis and rebuild exposed as standalone functions.

Link maintenance is metadata-only — never a version bump.
:func:`rebuild_links` re-parses every live page's body and rebuilds the
bidirectional in-bound references; the count helpers and the broken-link
report drive the stats and issue surfaces without loading full bodies.

Nothing here modifies the merged services; the web layer wires the
repository instance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from src.common.exception import NotFoundError
from src.core.knowledge.wiki.link_utils import parse_out_links
from src.db.dao.wiki_page_repository import WikiPageRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BrokenLink:
    """An out-bound link whose target slug is not live in the KB."""

    source: str
    target: str


async def rebuild_links(*, page_repo: WikiPageRepository, knowledge_base_id: str) -> None:
    """Re-parse every live page's body and rebuild bidirectional links.

    Out-bound links are recomputed from the body and in-bound references
    are derived from the surviving target pages. Each page is persisted
    through ``update_meta`` (no version bump); a page deleted mid-pass is
    skipped rather than aborting the rebuild.
    """
    pages = await page_repo.list_all(knowledge_base_id=knowledge_base_id)
    out_by_slug = {page.slug: parse_out_links(page.content) for page in pages}
    in_by_slug: dict[str, list[str]] = {}
    for source_slug, targets in out_by_slug.items():
        for target in targets:
            if target not in out_by_slug:
                continue
            backlinks = in_by_slug.setdefault(target, [])
            if source_slug not in backlinks:
                backlinks.append(source_slug)

    now = datetime.now(UTC)
    for page in pages:
        row = page.model_copy(
            update={
                "in_links": list(in_by_slug.get(page.slug, [])),
                "out_links": list(out_by_slug.get(page.slug, [])),
                "updated_at": now,
            }
        )
        try:
            await page_repo.update_meta(row=row, now=now)
        except NotFoundError:
            logger.warning("wiki: rebuild links: page %s gone mid-rebuild", page.slug)


async def count_total_links(*, page_repo: WikiPageRepository, knowledge_base_id: str) -> int:
    """Return the total number of out-bound links across live pages."""
    pages = await page_repo.list_all(knowledge_base_id=knowledge_base_id)
    return sum(len(page.out_links) for page in pages)


async def count_orphans(*, page_repo: WikiPageRepository, knowledge_base_id: str) -> int:
    """Return the number of live pages with no in-bound links (index excluded)."""
    return await page_repo.count_orphans(knowledge_base_id=knowledge_base_id)


async def broken_link_report(
    *, page_repo: WikiPageRepository, knowledge_base_id: str
) -> list[BrokenLink]:
    """Return the out-bound links whose target slug is not live.

    A link is broken when its normalized target is not a live
    (non-archived, non-deleted) page in the KB. Results are ordered by
    source then target for a deterministic report.
    """
    pages = await page_repo.list_all(knowledge_base_id=knowledge_base_id)
    targets = sorted({target for page in pages for target in page.out_links})
    live = await page_repo.exists_slugs(knowledge_base_id=knowledge_base_id, slugs=targets)
    broken: list[BrokenLink] = []
    for page in pages:
        for target in sorted(set(page.out_links)):
            if not live.get(target, False):
                broken.append(BrokenLink(source=page.slug, target=target))
    return broken


__all__ = [
    "BrokenLink",
    "broken_link_report",
    "count_orphans",
    "count_total_links",
    "rebuild_links",
]
