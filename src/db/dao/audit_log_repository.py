"""Audit-log persistence — raw SQL only, no ORM.

Maps the methods declared in the upstream
``internal/types/interfaces/audit_log.go::AuditLogRepository`` interface.
The table is append-only (no UPDATE, no soft-delete); the only write
surface is :meth:`create` (an INSERT). Reads use cursor pagination keyed
on the monotonic ``id``.

Every query uses named ``bindparams``. ``details`` is bound as
``JSONB`` on Postgres / ``JSON`` on SQLite via the
``GenericRepository._json_bindparams`` helper.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from sqlalchemy import text
from sqlalchemy.engine import CursorResult

from src.common.json import BindParams, SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.system.audit_log import AuditLog


class AuditLogRepository(GenericRepository[AuditLog]):
    """Audit-log SQL — append-only writes + cursor-paginated reads."""

    model_class = AuditLog

    async def create(self, entry: AuditLog) -> AuditLog:
        """Insert one immutable audit row, returning the persisted row."""
        return await self.insert(entry)

    async def list_for_tenant(
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
    ) -> list[AuditLog]:
        """Cursor-paginated newest-first read for one tenant.

        ``after_id`` is the last id from the previous page; rows with
        ``id < after_id`` are returned. ``0`` means "from the top".
        ``limit`` is capped at 100.
        """
        capped_limit = max(1, min(limit, 100))
        conditions: list[str] = ["tenant_id = :tenant_id"]
        params: BindParams = {"tenant_id": tenant_id, "limit": capped_limit}
        if after_id > 0:
            conditions.append("id < :after_id")
            params["after_id"] = after_id
        if action:
            conditions.append("action = :action")
            params["action"] = action
        if outcome:
            conditions.append("outcome = :outcome")
            params["outcome"] = outcome
        if actor_user_id:
            conditions.append("actor_user_id = :actor_user_id")
            params["actor_user_id"] = actor_user_id
        if scope_type:
            conditions.append("scope_type = :scope_type")
            params["scope_type"] = scope_type
        if scope_id:
            conditions.append("scope_id = :scope_id")
            params["scope_id"] = scope_id
        if unscoped_only:
            conditions.append("(scope_type = '' or scope_type is null)")
        where_clause = " and ".join(conditions)
        stmt = text(
            f"select * from {self._table} where {where_clause} order by id desc limit :limit"
        ).bindparams(**params)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def count_since_for_dedup(
        self,
        *,
        tenant_id: int,
        actor_user_id: str,
        action: str,
        request_path: str,
        since: datetime,
    ) -> int:
        """Rate-limit primitive for ``LogDenied`` dedup.

        Returns the count of matching rows in the trailing window so
        the service can skip writing duplicates.
        """
        stmt = text(
            "select count(*) from audit_logs "
            "where tenant_id = :tenant_id "
            "and actor_user_id = :actor_user_id "
            "and action = :action "
            "and request_path = :request_path "
            "and created_at >= :since"
        ).bindparams(
            tenant_id=tenant_id,
            actor_user_id=actor_user_id,
            action=action,
            request_path=request_path,
            since=since,
        )
        result = await self._session.execute(stmt)
        return int(result.scalar() or 0)

    async def delete_older_than(self, cutoff: datetime) -> int:
        """Retention primitive — removes rows with ``created_at < cutoff``."""
        stmt = text("delete from audit_logs where created_at < :cutoff").bindparams(cutoff=cutoff)
        result = cast(
            CursorResult[SqlValue],
            await self._session.execute(stmt),
        )
        return result.rowcount or 0


__all__ = ["AuditLogRepository"]
