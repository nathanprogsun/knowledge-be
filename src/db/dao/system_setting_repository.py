"""System-setting persistence — raw SQL only, no ORM.

Maps the methods declared in the upstream
``internal/types/interfaces/system_setting.go::SystemSettingRepository``
interface. ``key`` has a unique constraint; :meth:`upsert` uses
``ON CONFLICT (key) DO UPDATE`` to implement the "DB row overrides ENV"
precedence without a separate read-then-write cycle.

``value`` is JSONB and bound via the
``GenericRepository._json_bindparams`` helper. Every query uses named
``bindparams``.
"""

from __future__ import annotations

from typing import cast

from sqlalchemy import JSON, bindparam, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import CursorResult

from src.common.exception import DataError
from src.common.json import SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.system.system_setting import SystemSetting

_JSON_BIND_TYPE = JSON().with_variant(JSONB(), "postgresql")


class SystemSettingRepository(GenericRepository[SystemSetting]):
    """System-setting SQL — CRUD + key-targeted upsert."""

    model_class = SystemSetting

    async def get_by_key(self, key: str) -> SystemSetting | None:
        """Return the row matching ``key``, or ``None`` when absent."""
        return await self.find_unique_by_column_values({"key": key})

    async def list_all(self) -> list[SystemSetting]:
        """Return every persisted setting, ordered by category then key."""
        stmt = text("select * from system_settings order by category, key")
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def upsert(self, setting: SystemSetting) -> SystemSetting:
        """Insert-or-update on ``key`` conflict, returning the row.

        Implements the "DB row overrides ENV" precedence: if a row with
        the same ``key`` exists, every mutable column is overwritten
        (``value``, ``value_type``, ``category``, ``description``,
        ``is_secret``, ``requires_restart``, ``last_modified_by``,
        ``updated_at``); ``id`` and ``created_at`` are preserved.
        """
        params = setting.model_dump(exclude={"id", "created_at"})
        # ``updated_at`` is set by the caller (service layer) per
        # AGENTS.md §1 — the repo does not inject timestamps.
        col_list = ", ".join(f'"{c}"' for c in params)
        param_list = ", ".join(f":{c}" for c in params)
        update_cols = [c for c in params if c != "key"]
        set_clause = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_cols)
        stmt_text = (
            f"insert into system_settings ({col_list}) values ({param_list}) "
            f'on conflict ("key") do update set {set_clause} returning *'
        )
        json_bps = [bindparam("value", type_=_JSON_BIND_TYPE)]
        stmt = text(stmt_text).bindparams(*json_bps, **params)
        result = await self._session.execute(stmt)
        mapping = result.mappings().first()
        if mapping is None:
            # Upsert always returns a row; treat absence as a data error.
            raise DataError(
                code="db.upsert_no_row",
                message="system_settings upsert returned no row",
            )
        return self._hydrate(mapping)

    async def delete_by_key(self, key: str) -> int:
        """Remove the row matching ``key``. Returns the affected row count.

        Idempotent — deleting a key that was never persisted returns 0.
        Used by the service ``Reset`` path so the 3-tier resolver falls
        back to ENV / built-in default.
        """
        stmt = text('delete from system_settings where "key" = :key').bindparams(key=key)
        result = cast(
            CursorResult[SqlValue],
            await self._session.execute(stmt),
        )
        return result.rowcount or 0


__all__ = ["SystemSettingRepository"]
