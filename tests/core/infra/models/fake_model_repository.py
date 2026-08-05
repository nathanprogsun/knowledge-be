"""Shared in-memory fake for the models domain.

Mirrors ``ModelRepository`` so the service can be tested without a
database. The fake also covers the small surface the service touches
(``insert``, ``find_by_tenant_and_id[_or_fail]``,
``list_by_tenant``, ``update_row``, ``delete_by_tenant_and_id``,
``clear_default_by_type``).
"""

from __future__ import annotations

from src.common.exception import NotFoundError
from src.db.models.infra.model import Model


class FakeModelRepository:
    """In-memory replacement for ``ModelRepository``.

    Soft-deleted rows are filtered from every read, mirroring the
    real repository's ``deleted_at IS NULL`` predicate.
    """

    def __init__(self) -> None:
        self.rows: dict[str, Model] = {}

    # ── Writes ──────────────────────────────────────────────────────

    async def insert(self, row: Model) -> Model:
        """Insert a row; the caller supplies the id (UUID)."""
        self.rows[row.id] = row
        return row

    async def update_row(self, row: Model) -> Model | None:
        """Update an existing row, scoped to ``(id, tenant_id)``."""
        existing = self.rows.get(row.id)
        if existing is None or existing.tenant_id != row.tenant_id:
            return None
        self.rows[row.id] = row
        return row

    async def delete_by_tenant_and_id(
        self,
        *,
        tenant_id: int,
        id: str,
    ) -> int:
        """Delete a row scoped to ``(tenant_id, id)``; return rows affected."""
        existing = self.rows.get(id)
        if existing is None or existing.tenant_id != tenant_id:
            return 0
        del self.rows[id]
        return 1

    async def clear_default_by_type(
        self,
        *,
        tenant_id: int,
        model_type: str,
        exclude_id: str | None = None,
    ) -> int:
        """Flip ``is_default`` off for every matching row."""
        affected = 0
        for k, v in list(self.rows.items()):
            if v.deleted_at is not None:
                continue
            if v.tenant_id != tenant_id:
                continue
            if v.type != model_type:
                continue
            if not v.is_default:
                continue
            if exclude_id is not None and k == exclude_id:
                continue
            self.rows[k] = v.model_copy(update={"is_default": False})
            affected += 1
        return affected

    # ── Reads ───────────────────────────────────────────────────────

    async def find_by_tenant_and_id(
        self,
        *,
        tenant_id: int,
        id: str,
        include_builtin: bool = True,
    ) -> Model | None:
        """Return one row by id, scoped to a tenant (with builtins as opt-in)."""
        existing = self.rows.get(id)
        if existing is None or existing.deleted_at is not None:
            return None
        if existing.tenant_id == tenant_id:
            return existing
        if include_builtin and existing.is_builtin:
            return existing
        return None

    async def find_by_tenant_and_id_or_fail(
        self,
        *,
        tenant_id: int,
        id: str,
        include_builtin: bool = True,
    ) -> Model:
        """Same as :meth:`find_by_tenant_and_id` but raise on absence."""
        row = await self.find_by_tenant_and_id(
            tenant_id=tenant_id,
            id=id,
            include_builtin=include_builtin,
        )
        if row is None:
            raise NotFoundError(
                code="model.not_found",
                message=f"Model {id} not found",
            )
        return row

    async def list_by_tenant(
        self,
        *,
        tenant_id: int,
        model_type: str | None = None,
        source: str | None = None,
        include_builtin: bool = True,
    ) -> list[Model]:
        """List every row visible to ``tenant_id``."""
        results: list[Model] = []
        for row in self.rows.values():
            if row.deleted_at is not None:
                continue
            if row.tenant_id != tenant_id and not (include_builtin and row.is_builtin):
                continue
            if model_type is not None and row.type != model_type:
                continue
            if source is not None and row.source != source:
                continue
            results.append(row)
        return results


__all__ = ["FakeModelRepository"]
