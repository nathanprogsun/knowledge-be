"""Per-user knowledge-base pin persistence — raw SQL only, no ORM."""

from __future__ import annotations

from datetime import datetime
from typing import cast

from sqlalchemy import CursorResult, text

from src.common.json import SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.user_kb_pin import UserKBPin

_TABLE = "user_kb_pins"


class UserKBPinRepository(GenericRepository[UserKBPin]):
    """`user_kb_pins`-table SQL — add, remove, list."""

    model_class = UserKBPin

    async def add(
        self,
        *,
        tenant_id: int,
        user_id: str,
        kb_id: str,
        pinned_at: datetime,
    ) -> UserKBPin | None:
        """Insert one pin, collapsing a duplicate on the composite key."""
        row = UserKBPin(
            tenant_id=tenant_id,
            user_id=user_id,
            kb_id=kb_id,
            pinned_at=pinned_at,
        )
        return await self.insert_or_none(
            row,
            on_conflict_do_nothing_target_columns=["tenant_id", "user_id", "kb_id"],
        )

    async def remove(self, *, tenant_id: int, user_id: str, kb_id: str) -> bool:
        """Delete the pin. Missing rows are a successful no-op."""
        stmt = text(
            f"delete from {_TABLE} "
            "where tenant_id = :tenant_id and user_id = :user_id and kb_id = :kb_id"
        ).bindparams(tenant_id=tenant_id, user_id=user_id, kb_id=kb_id)
        result = cast("CursorResult[SqlValue]", await self._session.execute(stmt))
        return (result.rowcount or 0) > 0

    async def get(self, *, tenant_id: int, user_id: str, kb_id: str) -> UserKBPin | None:
        """Return one pin row, or ``None``."""
        return await self.find_unique_by_column_values(
            {"tenant_id": tenant_id, "user_id": user_id, "kb_id": kb_id},
        )

    async def list_for_user(self, *, tenant_id: int, user_id: str) -> list[UserKBPin]:
        """Return the viewer's pins, newest first."""
        stmt = text(
            f"select * from {_TABLE} "
            "where tenant_id = :tenant_id and user_id = :user_id "
            "order by pinned_at desc"
        ).bindparams(tenant_id=tenant_id, user_id=user_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]


__all__ = ["UserKBPinRepository"]
