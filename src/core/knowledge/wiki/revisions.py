"""Wiki page revision snapshots — domain types and row projection.

The current version lives on ``wiki_pages``. A user-visible edit inserts
the pre-edit page as ``(page_id, old version)`` before the rewrite.
There is no prune: the table is append-only and list bounds come from
``limit`` / ``offset``. A revert is itself an edit (snapshot current,
copy the stored snapshot back, bump version).
"""

from __future__ import annotations

from datetime import datetime
from typing import Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.db.models.wiki_page import WikiPage, WikiPageRevision

_WIKIPAGEREVISIONINFO_EXCLUDE_COLUMNS: frozenset[str] = frozenset()


class WikiPageRevisionInfo(BaseModel):
    """Service-side projection of a ``wiki_page_revisions`` row."""

    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    knowledge_base_id: str
    page_id: str
    slug: str
    version: int
    title: str = ""
    page_type: str = "summary"
    status: str = "published"
    content: str | None = None
    summary: str = ""
    aliases: list[str] = Field(default_factory=list)
    edit_source: str = ""
    editor_id: str = ""
    edited_at: datetime
    created_at: datetime

    @classmethod
    def map_from_db(cls, db: WikiPageRevision, *, include_content: bool = True) -> Self:
        """Project a storage snapshot; list rows omit ``content``."""
        record = db.model_dump(exclude=set(_WIKIPAGEREVISIONINFO_EXCLUDE_COLUMNS))
        aliases = record.get("aliases")
        record["aliases"] = list(aliases) if aliases else []
        if not include_content:
            record["content"] = None
        return cls.model_validate(record)


class WikiRevisionList(BaseModel):
    """Paged history for one page. The current version has no revision row."""

    model_config = ConfigDict(frozen=True)

    revisions: list[WikiPageRevisionInfo]
    total: int
    current_version: int


def snapshot_row_from_page(page: WikiPage, *, now: datetime) -> WikiPageRevision:
    """Build the pre-edit snapshot for ``(page.id, page.version)``."""
    return WikiPageRevision(
        id=str(uuid4()),
        tenant_id=page.tenant_id,
        knowledge_base_id=page.knowledge_base_id,
        page_id=page.id,
        slug=page.slug,
        version=page.version,
        title=page.title,
        page_type=page.page_type,
        status=page.status,
        content=page.content,
        summary=page.summary,
        aliases=list(page.aliases),
        edit_source=page.last_edit_source,
        editor_id=page.last_editor_id,
        edited_at=page.updated_at,
        created_at=now,
    )


__all__ = [
    "WikiPageRevisionInfo",
    "WikiRevisionList",
    "snapshot_row_from_page",
]
