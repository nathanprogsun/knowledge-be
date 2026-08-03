"""Tenant persistence — raw SQL only, no ORM.

Domain-named reads plus two atomic storage-counter mutations. Soft-
deleted rows are filtered out on every read. ``UpdateTenant`` and
``DeleteTenant`` are not re-implemented here: callers use
``GenericRepository.update_by_primary_key`` with an explicit column
dict, so which columns are mutable and ``updated_at`` stay service
decisions.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from sqlalchemy import CursorResult, text

from src.common.exception import NotFoundError
from src.db.dao.generic_repository import GenericRepository
from src.db.models.tenants.tenants import Tenant

# LIKE treats these as wildcards / escape; a user-supplied keyword
# must have them neutralised before interpolation into `%...%`.
_LIKE_ESCAPE_CHAR = "\\"
_LIKE_SPECIAL_CHARS = ("\\", "%", "_")


def escape_like_keyword(keyword: str) -> str:
    """Escape LIKE wildcards so the keyword matches literally."""
    escaped = keyword
    for char in _LIKE_SPECIAL_CHARS:
        escaped = escaped.replace(char, _LIKE_ESCAPE_CHAR + char)
    return escaped


class TenantRepository(GenericRepository[Tenant]):
    """`tenants`-table SQL — domain wrappers on the generic CRUD base,
    plus queries the base cannot express (IN-list fetch, keyword search
    with total, atomic counter updates).
    """

    model_class = Tenant

    # ── Reads ───────────────────────────────────────────────────────

    async def find_by_id(
        self,
        id: str | int,
        *,
        exclude_deleted_or_archived: bool = True,
        not_found_code: str = "tenant.not_found",
        not_found_message: str | None = None,
    ) -> Tenant:
        """Look up one tenant, raising ``tenant.not_found`` when absent."""
        return await super().find_by_id(
            id,
            exclude_deleted_or_archived=exclude_deleted_or_archived,
            not_found_code=not_found_code,
            not_found_message=not_found_message or f"Tenant {id} not found",
        )

    async def find_by_ids(self, ids: list[int]) -> list[Tenant]:
        """Batch-fetch tenants by id, ordered newest first.

        Missing ids are simply absent from the result (no error), and
        an empty input short-circuits without touching the database.
        """
        if not ids:
            return []
        soft = self._soft_delete_where_fragment(
            self.model_class,
            exclude_deleted_or_archived=True,
        )
        stmt = text(
            f"select * from {self._table} where id = any(:ids) {soft} order by created_at desc"
        ).bindparams(ids=ids)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def list_all(self, *, limit: int | None = None, offset: int = 0) -> list[Tenant]:
        """Return live tenants, newest first.

        Unpaginated by default: callers such as the storage-bucket
        uniqueness check need every tenant, and a silent default cap
        would make that check wrong rather than slow.
        """
        return await self._select_page(self._where_sql(""), {}, limit=limit, offset=offset)

    async def search(
        self,
        *,
        keyword: str | None = None,
        tenant_id: int | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> tuple[list[Tenant], int]:
        """Filter tenants by id and/or keyword; return one page and the total.

        An id filter and a keyword filter combine with OR (an id match
        wins even when the name does not match); the keyword matches
        ``name`` or ``description`` with LIKE; the total is counted
        before pagination. ``limit=None`` returns every match.
        """
        conditions, params = self._build_search_conditions(keyword, tenant_id)
        where_sql = self._where_sql(conditions)

        count_stmt = text(f"select count(*) from {self._table} {where_sql}").bindparams(**params)
        total = (await self._session.execute(count_stmt)).scalar_one()

        rows = await self._select_page(where_sql, params, limit=limit, offset=offset)
        return rows, int(total)

    # ── Query builders ──────────────────────────────────────────────

    def _where_sql(self, conditions: str) -> str:
        """Combine caller conditions with the soft-delete filter."""
        soft = self._soft_delete_where_fragment(
            self.model_class,
            exclude_deleted_or_archived=True,
            prefix="and" if conditions else "where",
        )
        return f"where {conditions} {soft}" if conditions else soft

    async def _select_page(
        self,
        where_sql: str,
        params: dict[str, object],
        *,
        limit: int | None,
        offset: int,
    ) -> list[Tenant]:
        """Run ``select * ... order by created_at desc`` with optional paging."""
        stmt_text = f"select * from {self._table} {where_sql} order by created_at desc"
        page_params: dict[str, object] = dict(params)
        if limit is not None:
            stmt_text += " limit :limit offset :offset"
            page_params["limit"] = limit
            page_params["offset"] = offset
        result = await self._session.execute(text(stmt_text).bindparams(**page_params))
        return [self._hydrate(m) for m in result.mappings().all()]

    @staticmethod
    def _build_search_conditions(
        keyword: str | None,
        tenant_id: int | None,
    ) -> tuple[str, dict[str, object]]:
        """Build the WHERE body and its bindparams for :meth:`search`."""
        has_id = tenant_id is not None and tenant_id > 0
        has_keyword = bool(keyword)
        if not has_id and not has_keyword:
            return "", {}

        params: dict[str, object] = {}
        parts: list[str] = []
        if has_id:
            parts.append("id = :tenant_id")
            params["tenant_id"] = tenant_id
        if has_keyword and keyword is not None:
            parts.append(
                f"name like :keyword escape '{_LIKE_ESCAPE_CHAR}' "
                f"or description like :keyword escape '{_LIKE_ESCAPE_CHAR}'"
            )
            params["keyword"] = f"%{escape_like_keyword(keyword)}%"
        return f"({' or '.join(parts)})", params

    # ── Atomic counter mutations ────────────────────────────────────

    async def adjust_storage_used(
        self,
        tenant_id: int,
        *,
        delta: int,
        updated_at: datetime,
    ) -> int:
        """Add ``delta`` to ``storage_used`` clamped at zero; return new value.

        A single ``set storage_used = greatest(storage_used + :delta, 0)``
        performs the add and the clamp without holding a row lock
        across a round trip.
        """
        soft = self._soft_delete_where_fragment(
            self.model_class,
            exclude_deleted_or_archived=True,
        )
        stmt = text(
            f"update {self._table} "
            "set storage_used = greatest(storage_used + :delta, 0), updated_at = :updated_at "
            f"where id = :id {soft} returning storage_used"
        ).bindparams(id=tenant_id, delta=delta, updated_at=updated_at)
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            raise NotFoundError(
                code="tenant.not_found",
                message=f"Tenant {tenant_id} not found",
            )
        return int(row)

    async def bulk_set_storage_quota(
        self,
        *,
        quota_bytes: int,
        updated_at: datetime,
    ) -> int:
        """Write ``quota_bytes`` to every live tenant; return rows affected."""
        soft = self._soft_delete_where_fragment(
            self.model_class,
            exclude_deleted_or_archived=True,
            prefix="where",
        )
        stmt = text(
            f"update {self._table} "
            f"set storage_quota = :quota_bytes, updated_at = :updated_at {soft}"
        ).bindparams(quota_bytes=quota_bytes, updated_at=updated_at)
        result = await self._session.execute(stmt)
        return cast("CursorResult[object]", result).rowcount


__all__ = ["TenantRepository", "escape_like_keyword"]
