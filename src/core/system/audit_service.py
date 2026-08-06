"""Audit-log service — append-only writes + dedup + retention.

Operations:

- ``log`` — write a single audit entry (timestamp defaulting applied
  by the caller; the repo does not inject timestamps).
- ``log_denied`` — record a middleware-level reject decision, subject
  to 1-minute sliding-window dedup keyed by
  ``(tenant_id, actor_user_id, action, request_path)`` so a probing
  client cannot flood the table. The dedup is implemented here; the
  middleware caller is not yet wired.
- ``list`` — cursor-paginated newest-first read.
- ``purge`` — retention sweep driven by ``retention_days``.

The service depends **only** on its repository — it does not hold an
``AsyncSession``. The web layer constructs a fresh repo + service per
request.

``list_entries`` projects each ``AuditLog`` row through
:meth:`AuditLogInfo.map_from_db` before returning so the web layer
never sees a storage ``TableModel``. The previous contract had
the router calling ``map_from_db`` per entry — a §9 violation (web
performing the projection onto the service-output DTO).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from src.core.system.types import AuditLogInfo
from src.db.dao.audit_log_repository import AuditLogRepository
from src.db.models.system.audit_log import AuditLog


@dataclass(frozen=True, slots=True)
class AuditLogListResult:
    """Returned by :meth:`AuditLogService.list`.

    ``entries`` carries service-output :class:`AuditLogInfo` DTOs (not
    raw ``AuditLog`` ``TableModel``s). The web router renders these to
    the wire shape directly.
    """

    entries: list[AuditLogInfo]
    next_cursor: int


class AuditLogService:
    """High-level audit API the rest of the codebase uses."""

    def __init__(self, *, audit_repo: AuditLogRepository) -> None:
        self._audit_repo = audit_repo

    async def log(self, entry: AuditLog) -> AuditLog:
        """Write a single audit entry. Returns the persisted row.

        The caller fills ``tenant_id`` + ``action`` + any per-event
        fields; if ``created_at`` is unset the service fills it with
        ``now``.
        """
        if entry.created_at is None:
            entry = entry.model_copy(update={"created_at": datetime.now(UTC)})
        return await self._audit_repo.create(entry)

    async def log_denied(
        self,
        *,
        tenant_id: int,
        actor_user_id: str,
        actor_role: str,
        action: str,
        request_path: str,
        request_method: str = "",
    ) -> AuditLog | None:
        """Record a middleware-level reject decision with dedup.

        Returns the persisted row, or ``None`` when the entry was
        skipped because a matching row already exists within the
        trailing 1-minute window.
        """
        since = datetime.now(UTC) - timedelta(minutes=1)
        recent = await self._audit_repo.count_since_for_dedup(
            tenant_id=tenant_id,
            actor_user_id=actor_user_id,
            action=action,
            request_path=request_path,
            since=since,
        )
        if recent > 0:
            return None
        entry = AuditLog(
            id=0,
            tenant_id=tenant_id,
            actor_user_id=actor_user_id,
            actor_role=actor_role,
            action=action,
            outcome="denied",
            request_path=request_path,
            request_method=request_method,
            created_at=datetime.now(UTC),
        )
        return await self._audit_repo.create(entry)

    async def list_entries(
        self,
        *,
        tenant_id: int,
        after_id: int = 0,
        limit: int = 50,
        action: str | None = None,
        outcome: str | None = None,
        actor_user_id: str | None = None,
        scope_type: str | None = None,
        scope_id: str | None = None,
        unscoped_only: bool = False,
    ) -> AuditLogListResult:
        """Cursor-paginated newest-first read for one tenant.

        Returns ``AuditLogInfo`` projections (not the raw storage
        ``TableModel``) so the web layer only does the wire-shape
        render — it no longer performs ``map_from_db`` itself.
        """
        rows = await self._audit_repo.list_for_tenant(
            tenant_id=tenant_id,
            after_id=after_id,
            limit=limit,
            action=action,
            outcome=outcome,
            actor_user_id=actor_user_id,
            scope_type=scope_type,
            scope_id=scope_id,
            unscoped_only=unscoped_only,
        )
        entries = [AuditLogInfo.map_from_db(row) for row in rows]
        next_cursor = rows[-1].id if rows else 0
        return AuditLogListResult(entries=entries, next_cursor=next_cursor)

    async def purge(self, retention_days: int) -> int:
        """Delete rows older than ``retention_days``. Returns row count.

        ``retention_days <= 0`` makes the call a no-op (keeps the daily
        sweep cheap when retention is disabled).
        """
        if retention_days <= 0:
            return 0
        cutoff = datetime.now(UTC) - timedelta(days=retention_days)
        return await self._audit_repo.delete_older_than(cutoff)


__all__ = ["AuditLogListResult", "AuditLogService"]
