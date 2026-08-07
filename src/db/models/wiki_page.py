"""Storage rows for the `wiki_pages` and `wiki_folders` tables.

Wiki pages are LLM-generated, interlinked markdown documents that form a
persistent wiki for a knowledge base. The column set mirrors the wiki
page contract: the identity / scope columns, the content and its
versioning bookkeeping (``version``, ``last_edit_source``,
``last_editor_id``), the link / source-reference JSON arrays, and the
denormalised directory cache (``folder_id`` / ``category_path`` /
``wiki_path`` / ``depth``).

``folder_id`` is the single source of truth for a page's placement in
the directory tree (empty string = wiki root); ``category_path``,
``wiki_path`` and ``depth`` are cached projections of the folder chain
recomputed on write so list / index / search queries never join
``wiki_folders``.

``WikiPageLite`` and ``WikiIndexEntry`` are slim read-only projections
of the page row (no ``content`` body) returned by the repository's link
maintenance and directory-listing queries.

Column notes
------------

- ``id`` is caller-assigned (UUID); every other column is caller-supplied
  (the application stamps ``created_at`` / ``updated_at`` before insert).
- The link / reference / alias arrays and ``page_metadata`` are JSONB;
  ``json_columns`` binds them with the JSONB bind type.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import Field

from src.common.json import JsonObject
from src.common.table_model import TableModel


class WikiPage(TableModel):
    """One row of the ``wiki_pages`` table."""

    table: ClassVar[str] = "wiki_pages"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = (
        "aliases",
        "category_path",
        "source_refs",
        "chunk_refs",
        "in_links",
        "out_links",
        "page_metadata",
    )
    # ``id`` is a caller-assigned UUID; the database never assigns columns.
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    tenant_id: int
    knowledge_base_id: str
    slug: str
    title: str = ""
    page_type: str = "summary"
    status: str = "published"
    content: str = ""
    summary: str = ""
    parent_slug: str = ""
    folder_id: str = ""
    category_path: list[str] = Field(default_factory=list)
    wiki_path: str = ""
    depth: int = 0
    sort_order: int = 0
    source_refs: list[str] = Field(default_factory=list)
    chunk_refs: list[str] = Field(default_factory=list)
    in_links: list[str] = Field(default_factory=list)
    out_links: list[str] = Field(default_factory=list)
    page_metadata: JsonObject = Field(default_factory=dict)
    aliases: list[str] = Field(default_factory=list)
    version: int = 1
    last_edit_source: str = ""
    last_editor_id: str = ""
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


class WikiPageLite(TableModel):
    """Slim page projection (no ``content``) for link and dedup queries.

    Backs the ingest pipeline's slug -> title resolution and dead-link
    cleanup: only the fields those passes reach for.
    """

    table: ClassVar[str] = "wiki_pages"
    primary_keys: ClassVar[tuple[str, ...]] = ("slug",)
    json_columns: ClassVar[tuple[str, ...]] = ("aliases", "out_links")
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    slug: str
    title: str
    page_type: str
    status: str
    aliases: list[str] = Field(default_factory=list)
    out_links: list[str] = Field(default_factory=list)


class WikiIndexEntry(TableModel):
    """Light projection for directory listings (no ``content`` column).

    Carries only the columns needed to render a clickable directory
    entry, so a large KB does not pay for TEXT content transport on
    every index open.
    """

    table: ClassVar[str] = "wiki_pages"
    primary_keys: ClassVar[tuple[str, ...]] = ("slug",)
    json_columns: ClassVar[tuple[str, ...]] = ("category_path",)
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    slug: str
    title: str
    summary: str
    parent_slug: str = ""
    category_path: list[str] = Field(default_factory=list)
    wiki_path: str = ""
    depth: int = 0
    sort_order: int = 0


class WikiFolder(TableModel):
    """One row of the ``wiki_folders`` table."""

    table: ClassVar[str] = "wiki_folders"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ()
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    tenant_id: int = 0
    knowledge_base_id: str
    parent_id: str = ""
    name: str
    path: str = ""
    depth: int = 0
    sort_order: int = 0
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


__all__ = ["WikiFolder", "WikiIndexEntry", "WikiPage", "WikiPageLite"]
