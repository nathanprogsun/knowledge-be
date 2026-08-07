"""FAQ entry persistence — raw SQL only, no ORM.

Maps the entry-level reads and writes used by the FAQ service: create,
get by id / by chunk, paged list for a knowledge base, update, enable
toggle, batch delete, and the standard/similar-question duplicate scan.

Every method scopes by ``tenant_id`` so a caller can never read or
mutate another workspace's rows. Entries are hard-deleted (the FAQ
delete path removes the entry outright), so there is no soft-delete
filter on this table — the ``deleted_at`` fragments of the generic base
are no-ops because the row model declares no such column.

``find_duplicate_question`` compares the question sets at the
application layer rather than with a JSONB-containment operator, because
that extraction syntax differs between PostgreSQL and SQLite; the row
count per knowledge base is bounded, so a full scan of the KB's entries
is the cost we pay, not a fan-out join.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

from sqlalchemy import JSON, CursorResult, bindparam, text
from sqlalchemy.dialects.postgresql import JSONB

from src.common.exception import NotFoundError
from src.common.json import BindParams, SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.faq import Faq

# JSONB on Postgres, JSON on other dialects (e.g. SQLite in tests).
_JSON_BIND_TYPE = JSON().with_variant(JSONB(), "postgresql")


class FaqRepository(GenericRepository[Faq]):
    """`faq`-table SQL — tenant-scoped CRUD + list + toggle."""

    model_class = Faq

    # ── Writes ──────────────────────────────────────────────────────

    async def create(self, row: Faq) -> Faq:
        """Insert a new entry; ``id`` is assigned by the database."""
        return await self.insert(row)

    async def update(self, row: Faq) -> Faq:
        """Overwrite every mutable column of the row, returning the result.

        ``id`` / ``tenant_id`` / ``chunk_id`` / ``knowledge_id`` /
        ``knowledge_base_id`` / ``created_at`` are immutable by contract
        (the FAQ service never changes them), so they stay out of the SET
        clause. WHERE is scoped to ``(id, tenant_id)`` so a caller cannot
        stomp another workspace's row.
        """
        immutable = {
            "id",
            "tenant_id",
            "chunk_id",
            "knowledge_id",
            "knowledge_base_id",
            "created_at",
        }
        update_cols = tuple(
            c for c in self.model_class.insert_sql_column_list()
            if c not in self._pk_columns and c not in immutable
        )
        set_clause = ", ".join(f'"{c}" = :u_{c}' for c in update_cols)
        params: BindParams = {
            **{f"u_{c}": getattr(row, c) for c in update_cols},
            "id": row.id,
            "tenant_id": row.tenant_id,
        }
        json_bps = [
            bindparam(f"u_{c}", type_=_JSON_BIND_TYPE)
            for c in update_cols
            if c in self._json_columns
        ]
        stmt_text = (
            f"update {self._table} set {set_clause} "
            "where id = :id and tenant_id = :tenant_id returning *"
        )
        stmt = text(stmt_text).bindparams(*json_bps, **params)
        result = await self._session.execute(stmt)
        persisted = self._hydrate_opt(result.mappings().first())
        if persisted is None:
            raise NotFoundError(
                code="faq.not_found",
                message=f"FAQ entry {row.id} not found for update",
            )
        return persisted

    async def set_enabled(
        self,
        *,
        tenant_id: int,
        id: int,
        is_enabled: bool,
        now: datetime | None = None,
    ) -> Faq | None:
        """Flip the entry's enabled flag, returning the refreshed row.

        Backs the FAQ toggle operation. ``updated_at`` is bumped so a
        later read reflects the change. Returns ``None`` when no live
        entry matched ``(id, tenant_id)``.
        """
        stmt = text(
            f"update {self._table} set is_enabled = :is_enabled, updated_at = :now "
            "where id = :id and tenant_id = :tenant_id returning *"
        ).bindparams(
            is_enabled=is_enabled,
            now=now or datetime.now(UTC),
            id=id,
            tenant_id=tenant_id,
        )
        result = await self._session.execute(stmt)
        return self._hydrate_opt(result.mappings().first())

    async def delete_by_ids(
        self,
        *,
        tenant_id: int,
        ids: list[int],
    ) -> int:
        """Hard-delete the given entries. Returns the number of rows removed.

        Scoped by ``tenant_id`` so an id list can never delete another
        workspace's entries. Empty ``ids`` deletes nothing.
        """
        if not ids:
            return 0
        placeholders = ", ".join(f":id{i}" for i in range(len(ids)))
        params: BindParams = {f"id{i}": value for i, value in enumerate(ids)}
        params["tenant_id"] = tenant_id
        stmt = text(
            f"delete from {self._table} where tenant_id = :tenant_id "
            f"and id in ({placeholders})"
        ).bindparams(**params)
        result = cast(
            "CursorResult[SqlValue]",
            await self._session.execute(stmt),
        )
        return result.rowcount or 0

    # ── Reads ───────────────────────────────────────────────────────

    async def get_by_id(self, *, tenant_id: int, id: int) -> Faq | None:
        """Return one entry by its sequence id, scoped to the tenant."""
        return await self.find_unique_by_column_values({"id": id, "tenant_id": tenant_id})

    async def get_by_id_or_fail(self, *, tenant_id: int, id: int) -> Faq:
        """Same as :meth:`get_by_id` but raise when the entry is absent."""
        row = await self.get_by_id(tenant_id=tenant_id, id=id)
        if row is None:
            raise NotFoundError(
                code="faq.not_found",
                message=f"FAQ entry {id} not found",
            )
        return row

    async def get_by_chunk_id(self, *, tenant_id: int, chunk_id: str) -> Faq | None:
        """Return the entry backed by ``chunk_id``, scoped to the tenant."""
        return await self.find_unique_by_column_values(
            {"chunk_id": chunk_id, "tenant_id": tenant_id},
        )

    async def list_by_knowledge_base(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        keyword: str | None = None,
        limit: int,
        offset: int,
    ) -> tuple[list[Faq], int]:
        """Return one page of the knowledge base's entries and the total.

        Entries are ordered by their sequence id. An optional ``keyword``
        filters by a ``standard_question`` substring (ILIKE); the richer
        search-field semantics are applied by the service layer. The
        total is counted before pagination.
        """
        where_parts = ["tenant_id = :tenant_id", "knowledge_base_id = :kb_id"]
        params: BindParams = {"tenant_id": tenant_id, "kb_id": knowledge_base_id}
        if keyword:
            where_parts.append("standard_question ilike :keyword")
            params["keyword"] = f"%{keyword}%"
        where = " and ".join(where_parts)

        count_stmt = text(
            f"select count(*) from {self._table} where {where}"
        ).bindparams(**params)
        total = int((await self._session.execute(count_stmt)).scalar_one())

        stmt = text(
            f"select * from {self._table} where {where} "
            "order by id limit :limit offset :offset"
        ).bindparams(**params, limit=limit, offset=offset)
        result = await self._session.execute(stmt)
        rows = [self._hydrate(m) for m in result.mappings().all()]
        return rows, total

    async def find_duplicate_question(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        exclude_id: int | None,
        questions: list[str],
    ) -> Faq | None:
        """Return one entry whose standard or similar question collides.

        ``questions`` is the candidate question set (standard question
        plus similar questions) of the entry being created or updated.
        A live entry is a match when its standard question or any of its
        similar questions appears in the candidate set. ``exclude_id``
        skips the entry being edited. Returns ``None`` when no entry
        collides.
        """
        if not questions:
            return None
        question_set = set(questions)
        stmt = text(
            f"select * from {self._table} "
            "where tenant_id = :tenant_id and knowledge_base_id = :kb_id "
            "order by id"
        ).bindparams(tenant_id=tenant_id, kb_id=knowledge_base_id)
        result = await self._session.execute(stmt)
        for mapping in result.mappings().all():
            row = self._hydrate(mapping)
            if exclude_id is not None and row.id == exclude_id:
                continue
            if row.standard_question in question_set:
                return row
            if any(q in question_set for q in row.similar_questions):
                return row
        return None


__all__ = ["FaqRepository"]
