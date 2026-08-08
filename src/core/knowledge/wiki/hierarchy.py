"""Page-hierarchy and folder-placement helpers shared by the wiki services.

``folder_id`` is the single source of truth for where a page sits in the
directory tree (empty string = wiki root). ``category_path`` / ``wiki_path``
/ ``depth`` are cached projections of the folder chain; every write path
recomputes them through :func:`apply_folder_to_page` + :func:`normalize_wiki_hierarchy`
so list / index / search queries never join ``wiki_folders``.

All functions are immutable — they return updated copies and never mutate
their inputs.
"""

from __future__ import annotations

from src.common.exception import ValidationError
from src.core.knowledge.wiki.types import clean_category_path
from src.db.dao.wiki_page_repository import WikiFolderRepository
from src.db.models.wiki_page import WikiIndexEntry, WikiPage


def build_wiki_path(page_type: str, category_path: list[str], display: str) -> str:
    """Assemble the normalized "page_type/cat.../title" directory breadcrumb.

    Empty segments are skipped so the path stays sortable and clean.
    """
    parts: list[str] = []
    if page_type.strip():
        parts.append(page_type.strip())
    parts.extend(category_path)
    if display:
        parts.append(display)
    return "/".join(parts)


def normalize_wiki_hierarchy(page: WikiPage) -> WikiPage:
    """Recompute the page's cached directory projection.

    ``parent_slug`` is trimmed; ``category_path`` is cleaned, deduplicated,
    and capped; ``depth`` and ``wiki_path`` follow from the cleaned path and
    the page's title (falling back to the slug).
    """
    clean_path = clean_category_path(page.category_path)
    display = page.title.strip()
    if not display:
        display = page.slug.strip()
    return page.model_copy(
        update={
            "parent_slug": page.parent_slug.strip(),
            "category_path": clean_path,
            "depth": len(clean_path),
            "wiki_path": build_wiki_path(page.page_type, clean_path, display),
        }
    )


def normalize_wiki_index_entry_hierarchy(
    entry: WikiIndexEntry, page_type: str
) -> WikiIndexEntry:
    """Recompute a directory entry's cached projection for one page type."""
    clean_path = clean_category_path(entry.category_path)
    display = entry.title.strip()
    if not display:
        display = entry.slug.strip()
    return entry.model_copy(
        update={
            "category_path": clean_path,
            "depth": len(clean_path),
            "wiki_path": build_wiki_path(page_type, clean_path, display),
        }
    )


def wiki_folder_segments(path: str) -> list[str]:
    """Split a materialized folder path ("AI/RAG") into cleaned segments.

    A blank path yields ``[]`` (the wiki root).
    """
    if not path.strip():
        return []
    return clean_category_path(path.split("/"))


async def apply_folder_to_page(
    page: WikiPage, *, folder_repo: WikiFolderRepository
) -> WikiPage:
    """Refresh a page's derived ``category_path`` from its authoritative folder id.

    Root (``""``) clears the path. A folder id that does not resolve is
    treated as a hard error so a page is never silently misplaced.
    """
    folder_id = page.folder_id.strip()
    if not folder_id:
        return page.model_copy(update={"folder_id": "", "category_path": []})
    folder = await folder_repo.get_by_id_or_none(
        knowledge_base_id=page.knowledge_base_id, id=folder_id
    )
    if folder is None:
        raise ValidationError(
            code="wiki.page_folder_unknown",
            message=f"wiki page references unknown folder {folder_id}",
        )
    return page.model_copy(update={"category_path": wiki_folder_segments(folder.path)})


__all__ = [
    "apply_folder_to_page",
    "build_wiki_path",
    "normalize_wiki_hierarchy",
    "normalize_wiki_index_entry_hierarchy",
    "wiki_folder_segments",
]
