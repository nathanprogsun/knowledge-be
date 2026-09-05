"""Wiki page revision persistence — raw SQL only, no ORM.

Maps the ``wiki_page_revisions`` table. Rows are append-only content
snapshots of a superseded page version. Unique ``(page_id, version)``
makes the snapshot-then-update write path idempotent under retries.
There is no ``deleted_at``; list bounds come from ``limit`` / ``offset``.
"""

from __future__ import annotations

from sqlalchemy import text

from src.db.dao.generic_repository import GenericRepository
from src.db.models.wiki_page import WikiPageRevision

_LIST_COLUMNS: str = (
    "id, tenant_id, knowledge_base_id, page_id, slug, version, title, "
    "page_type, status, summary, aliases, edit_source, editor_id, "
    "edited_at, created_at"
)


class WikiPageRevisionRepository(GenericRepository[WikiPageRevision]):
    """`wiki_page_revisions`-table SQL — immutable page snapshots."""

    model_class = WikiPageRevision

    async def insert_snapshot(self, row: WikiPageRevision) -> WikiPageRevision | None:
        """Insert one snapshot; a retry of the same ``(page_id, version)`` is a no-op."""
        return await self.insert_or_none(
            row,
            on_conflict_do_nothing_target_columns=["page_id", "version"],
        )

    async def list_by_slug(
        self,
        *,
        knowledge_base_id: str,
        slug: str,
        limit: int,
        offset: int,
    ) -> tuple[list[WikiPageRevision], int]:
        """Return snapshots for a page, newest version first, without ``content``."""
        count_stmt = text(
            f"select count(*) from {self._table} "
            "where knowledge_base_id = :knowledge_base_id and slug = :slug"
        ).bindparams(knowledge_base_id=knowledge_base_id, slug=slug)
        count_result = await self._session.execute(count_stmt)
        total = int(count_result.scalar_one())

        stmt = text(
            f"select {_LIST_COLUMNS} from {self._table} "
            "where knowledge_base_id = :knowledge_base_id and slug = :slug "
            "order by version desc "
            "limit :limit offset :offset"
        ).bindparams(
            knowledge_base_id=knowledge_base_id,
            slug=slug,
            limit=limit,
            offset=offset,
        )
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()], total

    async def get_by_slug_version(
        self,
        *,
        knowledge_base_id: str,
        slug: str,
        version: int,
    ) -> WikiPageRevision | None:
        """Return one snapshot with content, or ``None`` when absent."""
        stmt = text(
            f"select * from {self._table} "
            "where knowledge_base_id = :knowledge_base_id "
            "and slug = :slug and version = :version"
        ).bindparams(
            knowledge_base_id=knowledge_base_id,
            slug=slug,
            version=version,
        )
        result = await self._session.execute(stmt)
        return self._hydrate_opt(result.mappings().first())


__all__ = ["WikiPageRevisionRepository"]
