"""Wiki page issue records and their analysis, exposed as standalone functions.

Issues are lightweight review flags raised against wiki pages (typically
by linters or agents). The persistence surface is defined as a repository
seam so the web layer can wire the storage implementation once its table
lands; this module stays testable against mocks in the meantime. The
status vocabulary mirrors the issue lifecycle: ``pending`` (default),
``ignored``, ``resolved``.

Nothing here modifies the merged services; the web layer wires the
issue-repository instance.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.common.exception import NotFoundError, ValidationError
from src.db.dao.wiki_page_issue_repository import (
    WikiPageIssueRepository as WikiPageIssueDao,
)
from src.db.models.wiki_page import WikiPageIssue as WikiPageIssueRow

_WIKIPAGEISSUE_EXCLUDE_COLUMNS: frozenset[str] = frozenset({"deleted_at"})

WIKI_ISSUE_STATUS_PENDING = "pending"
WIKI_ISSUE_STATUS_IGNORED = "ignored"
WIKI_ISSUE_STATUS_RESOLVED = "resolved"

_WIKI_ISSUE_STATUSES: frozenset[str] = frozenset(
    {WIKI_ISSUE_STATUS_PENDING, WIKI_ISSUE_STATUS_IGNORED, WIKI_ISSUE_STATUS_RESOLVED}
)


def is_valid_issue_status(status: str) -> bool:
    """Return whether ``status`` is one of the known issue statuses."""
    return status in _WIKI_ISSUE_STATUSES


class WikiPageIssue(BaseModel):
    """A review flag raised against a wiki page."""

    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    knowledge_base_id: str
    slug: str
    issue_type: str
    description: str
    suspected_knowledge_ids: list[str] = Field(default_factory=list)
    status: str = WIKI_ISSUE_STATUS_PENDING
    reported_by: str = ""
    created_at: datetime
    updated_at: datetime

    @classmethod
    def map_from_db(cls, db: WikiPageIssueRow) -> Self:
        """Project a storage row; ``deleted_at`` stays off the wire."""
        record = db.model_dump(exclude=set(_WIKIPAGEISSUE_EXCLUDE_COLUMNS))
        ids = record.get("suspected_knowledge_ids")
        record["suspected_knowledge_ids"] = list(ids) if ids else []
        return cls.model_validate(record)


class WikiPageIssueRepository(Protocol):
    """Persistence seam for wiki page issues."""

    async def create(self, issue: WikiPageIssue) -> WikiPageIssue: ...

    async def list(
        self, *, knowledge_base_id: str, slug: str = "", status: str = ""
    ) -> list[WikiPageIssue]: ...

    async def get_by_id_or_none(self, *, issue_id: str) -> WikiPageIssue | None: ...

    async def update_status(self, *, issue_id: str, status: str) -> None: ...


class WikiPageIssueStore:
    """Adapts the TableModel DAO onto the issue Protocol."""

    def __init__(self, dao: WikiPageIssueDao) -> None:
        self._dao = dao

    async def create(self, issue: WikiPageIssue) -> WikiPageIssue:
        """Persist a protocol issue and return the stored projection."""
        row = WikiPageIssueRow(
            id=issue.id,
            tenant_id=issue.tenant_id,
            knowledge_base_id=issue.knowledge_base_id,
            slug=issue.slug,
            issue_type=issue.issue_type,
            description=issue.description,
            suspected_knowledge_ids=list(issue.suspected_knowledge_ids),
            status=issue.status,
            reported_by=issue.reported_by,
            created_at=issue.created_at,
            updated_at=issue.updated_at,
            deleted_at=None,
        )
        return WikiPageIssue.map_from_db(await self._dao.create(row))

    async def list(
        self, *, knowledge_base_id: str, slug: str = "", status: str = ""
    ) -> list[WikiPageIssue]:
        """Return live issues as protocol DTOs, newest first."""
        rows = await self._dao.list(knowledge_base_id=knowledge_base_id, slug=slug, status=status)
        return [WikiPageIssue.map_from_db(row) for row in rows]

    async def get_by_id_or_none(self, *, issue_id: str) -> WikiPageIssue | None:
        """Return one live issue DTO, or ``None`` when absent."""
        row = await self._dao.get_by_id_or_none(issue_id=issue_id)
        if row is None:
            return None
        return WikiPageIssue.map_from_db(row)

    async def update_status(self, *, issue_id: str, status: str) -> None:
        """Forward a status write to the DAO."""
        await self._dao.update_status(issue_id=issue_id, status=status)


async def create_issue(
    *,
    issue_repo: WikiPageIssueRepository,
    issue: WikiPageIssue,
    now: datetime | None = None,
) -> WikiPageIssue:
    """Persist a new issue with defaults applied.

    ``id`` defaults to a fresh UUID, ``status`` defaults to ``pending``,
    and timestamps default to the current time. The tenant id, knowledge
    base id, slug, issue type, and description are required.
    """
    if issue.tenant_id <= 0:
        raise ValidationError(
            code="wiki.issue_tenant_required",
            message="tenant_id is required",
        )
    if not issue.knowledge_base_id.strip():
        raise ValidationError(
            code="wiki.issue_kb_required",
            message="knowledge_base_id is required",
        )
    if not issue.slug.strip():
        raise ValidationError(
            code="wiki.issue_slug_required",
            message="slug is required",
        )
    if not issue.issue_type.strip():
        raise ValidationError(
            code="wiki.issue_type_required",
            message="issue_type is required",
        )
    if not issue.description.strip():
        raise ValidationError(
            code="wiki.issue_description_required",
            message="description is required",
        )
    status = issue.status or WIKI_ISSUE_STATUS_PENDING
    if status not in _WIKI_ISSUE_STATUSES:
        raise ValidationError(
            code="wiki.issue_invalid_status",
            message=f"invalid status {status}",
        )
    stamp = now or datetime.now(UTC)
    row = issue.model_copy(
        update={
            "id": issue.id or str(uuid4()),
            "status": status,
            "created_at": stamp,
            "updated_at": stamp,
        }
    )
    return await issue_repo.create(row)


async def list_issues(
    *,
    issue_repo: WikiPageIssueRepository,
    knowledge_base_id: str,
    slug: str = "",
    status: str = "",
) -> list[WikiPageIssue]:
    """Return the KB's issues, newest first, with optional filters.

    ``slug`` and ``status`` are optional filters; an empty value leaves
    that dimension unfiltered. An unknown ``status`` is rejected.
    """
    if not knowledge_base_id.strip():
        raise ValidationError(
            code="wiki.issue_kb_required",
            message="knowledge_base_id is required",
        )
    if status and status not in _WIKI_ISSUE_STATUSES:
        raise ValidationError(
            code="wiki.issue_invalid_status",
            message=f"invalid status {status}",
        )
    return await issue_repo.list(knowledge_base_id=knowledge_base_id, slug=slug, status=status)


async def update_issue_status(
    *,
    issue_repo: WikiPageIssueRepository,
    issue_id: str,
    status: str,
) -> WikiPageIssue:
    """Set an issue's status, returning the updated record.

    Raises ``wiki.issue_not_found`` when no issue carries ``issue_id``.
    """
    if not issue_id.strip():
        raise ValidationError(
            code="wiki.issue_id_required",
            message="issue id is required",
        )
    if status not in _WIKI_ISSUE_STATUSES:
        raise ValidationError(
            code="wiki.issue_invalid_status",
            message=f"invalid status {status}",
        )
    existing = await issue_repo.get_by_id_or_none(issue_id=issue_id)
    if existing is None:
        raise NotFoundError(
            code="wiki.issue_not_found",
            message=f"wiki page issue {issue_id} not found",
        )
    await issue_repo.update_status(issue_id=issue_id, status=status)
    return existing.model_copy(update={"status": status})


async def pending_issue_count(
    *, issue_repo: WikiPageIssueRepository, knowledge_base_id: str
) -> int:
    """Return the number of pending issues for a KB."""
    return len(
        await issue_repo.list(knowledge_base_id=knowledge_base_id, status=WIKI_ISSUE_STATUS_PENDING)
    )


__all__ = [
    "WIKI_ISSUE_STATUS_IGNORED",
    "WIKI_ISSUE_STATUS_PENDING",
    "WIKI_ISSUE_STATUS_RESOLVED",
    "WikiPageIssue",
    "WikiPageIssueRepository",
    "WikiPageIssueStore",
    "create_issue",
    "is_valid_issue_status",
    "list_issues",
    "pending_issue_count",
    "update_issue_status",
]
