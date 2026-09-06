"""Wiki page issue persistence — raw SQL only, no ORM.

Maps the ``wiki_page_issues`` table. Status values are stored as the
protocol vocabulary (``pending`` / ``ignored`` / ``resolved``) even
though the table comment still says ``dismissed``. Every read filters
``deleted_at is null``.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import text

from src.db.dao.generic_repository import GenericRepository
from src.db.models.wiki_page import WikiPageIssue


class WikiPageIssueRepository(GenericRepository[WikiPageIssue]):
    """`wiki_page_issues`-table SQL — create, list, and status updates."""

    model_class = WikiPageIssue

    async def create(self, issue: WikiPageIssue) -> WikiPageIssue:
        """Insert one issue; the application supplies the UUID ``id``."""
        return await self.insert(issue)

    async def list(
        self, *, knowledge_base_id: str, slug: str = "", status: str = ""
    ) -> list[WikiPageIssue]:
        """Return live issues for the KB, newest first.

        Empty ``slug`` / ``status`` leave that dimension unfiltered.
        """
        stmt = text(
            f"select * from {self._table} "
            "where knowledge_base_id = :knowledge_base_id "
            "and deleted_at is null "
            "and (:slug = '' or slug = :slug) "
            "and (:status = '' or status = :status) "
            "order by created_at desc, id desc"
        ).bindparams(knowledge_base_id=knowledge_base_id, slug=slug, status=status)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def get_by_id_or_none(self, *, issue_id: str) -> WikiPageIssue | None:
        """Return one live issue, or ``None`` when absent or soft-deleted."""
        stmt = text(
            f"select * from {self._table} where id = :issue_id and deleted_at is null"
        ).bindparams(issue_id=issue_id)
        result = await self._session.execute(stmt)
        mapping = result.mappings().first()
        return self._hydrate_opt(mapping)

    async def update_status(self, *, issue_id: str, status: str) -> None:
        """Set status on a live issue. A missing row is a no-op."""
        stmt = text(
            f"update {self._table} set status = :status, updated_at = :updated_at "
            "where id = :issue_id and deleted_at is null"
        ).bindparams(issue_id=issue_id, status=status, updated_at=datetime.now(UTC))
        await self._session.execute(stmt)


__all__ = ["WikiPageIssueRepository"]
