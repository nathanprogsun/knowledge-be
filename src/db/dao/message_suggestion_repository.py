"""Message suggestion persistence — raw SQL only, no ORM.

Covers the ``message_suggestion_sets`` table: create, cache-key and id
lookups, the lease-guarded generation acquisition, save, and the
session / message deletes. The table has no ``deleted_at`` column, so
deletes are hard deletes.

``acquire_generation`` mirrors the upstream concurrency semantics: a
candidate row is inserted with ``ON CONFLICT DO NOTHING`` on the cache
key; a concurrent duplicate either returns the existing ready /
suppressed row (when regeneration is not requested) or re-claims the
generation lease with a guarded UPDATE, so a crashed worker's expired
lease lets a retry take over.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import cast

from sqlalchemy import JSON, bindparam, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import CursorResult

from src.common.exception import DataError
from src.common.json import BindParams, SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.message_suggestion import (
    SUGGESTION_STATUS_GENERATING,
    SUGGESTION_STATUS_READY,
    SUGGESTION_STATUS_SUPPRESSED,
    MessageSuggestionSet,
)

# JSONB on Postgres, JSON on other dialects (e.g. SQLite in tests).
_JSON_BIND_TYPE = JSON().with_variant(JSONB(), "postgresql")

# Module-level alias for the table name. Every ``text(f"...{...}")`` in
# this file interpolates either this constant or ``self._table`` (whose
# cached_property validates the identifier at first use); user input
# never reaches the SQL string.
_TABLE_NAME = "message_suggestion_sets"

# Lease duration for an in-flight generation: a crashed worker's lease
# expires and a retry can re-acquire the row.
_LEASE_MINUTES = 3

# The cache-key columns form the unique constraint that backs
# ``acquire_generation``'s conflict target.
_CACHE_KEY_COLUMNS = [
    "tenant_id",
    "assistant_message_id",
    "placement",
    "config_hash",
    "locale",
]

# Columns a save must never touch: identity, the tenant / session
# scoping, and the insert timestamp.
_IMMUTABLE_UPDATE_COLUMNS: frozenset[str] = frozenset(
    {"id", "tenant_id", "session_id", "assistant_message_id", "created_at"}
)


class MessageSuggestionRepository(GenericRepository[MessageSuggestionSet]):
    """`message_suggestion_sets`-table SQL."""

    model_class = MessageSuggestionSet

    # ── Writes ──────────────────────────────────────────────────────

    async def create(self, row: MessageSuggestionSet) -> MessageSuggestionSet:
        """Insert one suggestion set; the application supplies the UUID ``id``."""
        return await self.insert(row)

    async def acquire_generation(
        self,
        candidate: MessageSuggestionSet,
        *,
        regenerate: bool,
        now: datetime,
    ) -> tuple[MessageSuggestionSet | None, bool]:
        """Claim the generation lease for ``candidate``.

        Returns ``(row, True)`` when this caller owns the generation and
        ``(row, False)`` when a concurrent generation is already in
        flight or a ready / suppressed row is returned unchanged.
        """
        lease_until = now + timedelta(minutes=_LEASE_MINUTES)
        candidate = candidate.model_copy(
            update={
                "status": SUGGESTION_STATUS_GENERATING,
                "lease_until": lease_until,
                "questions": [],
            }
        )
        inserted = await self.insert_or_none(
            candidate,
            on_conflict_do_nothing_target_columns=_CACHE_KEY_COLUMNS,
        )
        if inserted is not None:
            return inserted, True

        existing = await self.get_by_cache_key(
            tenant_id=candidate.tenant_id,
            assistant_message_id=candidate.assistant_message_id,
            placement=candidate.placement,
            config_hash=candidate.config_hash,
            locale=candidate.locale,
        )
        if existing is None:
            return None, False
        if (
            existing.status in (SUGGESTION_STATUS_READY, SUGGESTION_STATUS_SUPPRESSED)
            and not regenerate
        ):
            return existing, False
        if (
            existing.status == SUGGESTION_STATUS_GENERATING
            and existing.lease_until is not None
            and existing.lease_until > now
        ):
            return existing, False

        updated = await self._reacquire_generation(existing, regenerate=regenerate, now=now)
        if updated is not None:
            return updated, True
        current = await self.get_by_cache_key(
            tenant_id=candidate.tenant_id,
            assistant_message_id=candidate.assistant_message_id,
            placement=candidate.placement,
            config_hash=candidate.config_hash,
            locale=candidate.locale,
        )
        return current, False

    async def save(self, row: MessageSuggestionSet) -> MessageSuggestionSet:
        """Persist the full row, returning the stored result.

        Rewrites every mutable column (the generation outcome: status,
        questions, token / latency bookkeeping, error code) under the
        row's primary key.
        """
        updates = {
            k: v for k, v in row.model_dump().items() if k not in _IMMUTABLE_UPDATE_COLUMNS
        }
        persisted = await self.update_by_primary_key({"id": row.id}, updates)
        if persisted is None:
            raise DataError(
                code="suggestion.update_no_row",
                message=f"suggestion set {row.id} not found for update",
            )
        return persisted

    async def delete_by_message_id(
        self,
        *,
        tenant_id: int,
        session_id: str,
        message_id: str,
    ) -> int:
        """Hard-delete every suggestion set of one assistant message. Returns count."""
        stmt = text(
            f"delete from {_TABLE_NAME} "
            "where tenant_id = :tenant_id and session_id = :session_id "
            "and assistant_message_id = :message_id"
        ).bindparams(tenant_id=tenant_id, session_id=session_id, message_id=message_id)
        result = await self._session.execute(stmt)
        return cast("CursorResult[SqlValue]", result).rowcount or 0

    async def delete_by_session_id(
        self,
        *,
        tenant_id: int,
        session_id: str,
    ) -> int:
        """Hard-delete every suggestion set of a session. Returns count."""
        stmt = text(
            f"delete from {_TABLE_NAME} "
            "where tenant_id = :tenant_id and session_id = :session_id"
        ).bindparams(tenant_id=tenant_id, session_id=session_id)
        result = await self._session.execute(stmt)
        return cast("CursorResult[SqlValue]", result).rowcount or 0

    # ── Reads ───────────────────────────────────────────────────────

    async def get_by_id(
        self,
        *,
        tenant_id: int,
        session_id: str,
        id: str,
    ) -> MessageSuggestionSet | None:
        """Return one suggestion set by ``(id, tenant_id, session_id)``."""
        return await self.find_unique_by_column_values(
            {"id": id, "tenant_id": tenant_id, "session_id": session_id},
        )

    async def get_by_cache_key(
        self,
        *,
        tenant_id: int,
        assistant_message_id: str,
        placement: str,
        config_hash: str,
        locale: str,
    ) -> MessageSuggestionSet | None:
        """Return the suggestion set matching the generation cache key."""
        return await self.find_unique_by_column_values(
            {
                "tenant_id": tenant_id,
                "assistant_message_id": assistant_message_id,
                "placement": placement,
                "config_hash": config_hash,
                "locale": locale,
            }
        )

    # ── Lease re-acquisition ──────────────────────────────────────────

    async def _reacquire_generation(
        self,
        existing: MessageSuggestionSet,
        *,
        regenerate: bool,
        now: datetime,
    ) -> MessageSuggestionSet | None:
        """Claim an expired / stale generation lease with a guarded UPDATE.

        Returns the updated row when the claim succeeded, or ``None``
        when a concurrent writer won the race.
        """
        lease_until = now + timedelta(minutes=_LEASE_MINUTES)
        params: BindParams = {
            "id": existing.id,
            "status": SUGGESTION_STATUS_GENERATING,
            "lease_until": lease_until,
            "questions": [],
            "now": now,
            "generating": SUGGESTION_STATUS_GENERATING,
        }
        if existing.status == SUGGESTION_STATUS_READY and regenerate:
            where = "id = :id and status = :ready"
            params["ready"] = SUGGESTION_STATUS_READY
        else:
            where = (
                "id = :id and (status <> :generating "
                "or lease_until is null or lease_until < :now)"
            )
        stmt = text(
            f"update {_TABLE_NAME} set "
            "status = :status, lease_until = :lease_until, suppression_reason = '', "
            "questions = :questions, error_code = '', generated_at = null, updated_at = :now "
            f"where {where} returning *"
        ).bindparams(bindparam("questions", type_=_JSON_BIND_TYPE), **params)
        result = await self._session.execute(stmt)
        return self._hydrate_opt(result.mappings().first())


__all__ = ["MessageSuggestionRepository"]
