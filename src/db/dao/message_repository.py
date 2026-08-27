"""Message persistence — raw SQL only, no ORM.

Covers the ``messages`` table: create, session-scoped reads (paginated
feed, recent window, before-time window), full-row update, soft delete
(one message or a whole session), and the small aggregate queries the
chat service relies on. Reads filter soft-deleted rows
(``deleted_at is null``) unless the caller opts out.

Every write is scoped by ``(id, session_id)`` so a message can only be
touched through its owning session — mirroring the upstream repository
contract. The one exception is :meth:`update_knowledge_id`, which the
indexing path calls by message id alone.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from sqlalchemy import JSON, bindparam, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import CursorResult
from sqlalchemy.sql.elements import BindParameter

from src.common.json import BindParams, SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.message import Message

# JSONB on Postgres, JSON on other dialects (e.g. SQLite in tests).
_JSON_BIND_TYPE = JSON().with_variant(JSONB(), "postgresql")

_LIVE = "deleted_at is null"

# Module-level alias for the table name. Every ``text(f"...{...}")`` in
# this file interpolates either this constant or ``self._table`` (whose
# cached_property validates the identifier at first use); user input
# never reaches the SQL string.
_TABLE_NAME = "messages"

# The session feed reads oldest-first; the recent / before-time windows
# read newest-first and are re-sorted in memory so the caller sees a
# chronological window with user turns first on equal timestamps.
_FEED_ORDER = "created_at asc, id asc"
_RECENT_ORDER = "created_at desc, id desc"

# Role vocabulary. Kept local so the persistence layer stays
# dependency-free; the shared vocabulary lives in the chat domain types.
_ROLE_USER = "user"


class MessageRepository(GenericRepository[Message]):
    """`messages`-table SQL — CRUD, session feeds, and aggregates."""

    model_class = Message

    # ── Writes ──────────────────────────────────────────────────────

    async def create(self, row: Message) -> Message:
        """Insert one message; the application supplies the UUID ``id``."""
        return await self.insert(row)

    async def update(
        self,
        *,
        session_id: str,
        message_id: str,
        column_to_update: BindParams,
    ) -> Message | None:
        """Update a message scoped by ``(id, session_id)``; return the row.

        ``column_to_update`` carries only the columns to change; the
        identity columns are read from the scoping arguments and must not
        appear in it. Returns ``None`` when no live row matched.
        """
        self.model_class.validate_in_columns(column_to_update)
        set_clause = ", ".join(f'"{k}" = :u_{k}' for k in column_to_update)
        update_params: BindParams = {f"u_{k}": v for k, v in column_to_update.items()}
        json_bps: list[BindParameter[SqlValue]] = [
            bindparam(f"u_{col}", type_=_JSON_BIND_TYPE)
            for col in column_to_update
            if col in self._json_columns
        ]
        stmt = text(
            f"update {_TABLE_NAME} set {set_clause} "
            f"where id = :id and session_id = :session_id and {_LIVE} returning *"
        ).bindparams(*json_bps, id=message_id, session_id=session_id, **update_params)
        result = await self._session.execute(stmt)
        return self._hydrate_opt(result.mappings().first())

    async def soft_delete(
        self,
        *,
        session_id: str,
        message_id: str,
        now: datetime,
    ) -> bool:
        """Mark a live message deleted. Returns whether a row was affected."""
        stmt = text(
            f"update {_TABLE_NAME} set deleted_at = :now, updated_at = :now "
            f"where id = :id and session_id = :session_id and {_LIVE}"
        ).bindparams(id=message_id, session_id=session_id, now=now)
        result = await self._session.execute(stmt)
        return (cast("CursorResult[SqlValue]", result).rowcount or 0) > 0

    async def soft_delete_by_session(self, session_id: str, now: datetime) -> int:
        """Soft-delete every live message of a session. Returns count."""
        stmt = text(
            f"update {_TABLE_NAME} set deleted_at = :now, updated_at = :now "
            f"where session_id = :session_id and {_LIVE}"
        ).bindparams(session_id=session_id, now=now)
        result = await self._session.execute(stmt)
        return cast("CursorResult[SqlValue]", result).rowcount or 0

    async def update_images(
        self,
        *,
        session_id: str,
        message_id: str,
        images: SqlValue,
    ) -> Message | None:
        """Update only the ``images`` JSONB column of a message."""
        return await self.update(
            session_id=session_id,
            message_id=message_id,
            column_to_update={"images": images},
        )

    async def update_rendered_content(
        self,
        *,
        session_id: str,
        message_id: str,
        rendered_content: str,
    ) -> Message | None:
        """Update only the ``rendered_content`` column of a message."""
        return await self.update(
            session_id=session_id,
            message_id=message_id,
            column_to_update={"rendered_content": rendered_content},
        )

    async def update_knowledge_id(
        self,
        *,
        message_id: str,
        knowledge_id: str,
        now: datetime,
    ) -> Message | None:
        """Update only the ``knowledge_id`` column of a message.

        Scoped by message id alone: the indexing path links a message to
        its knowledge entry without knowing the owning session.
        """
        stmt = text(
            f"update {_TABLE_NAME} set knowledge_id = :knowledge_id, updated_at = :now "
            f"where id = :id and {_LIVE} returning *"
        ).bindparams(id=message_id, knowledge_id=knowledge_id, now=now)
        result = await self._session.execute(stmt)
        return self._hydrate_opt(result.mappings().first())

    # ── Reads ───────────────────────────────────────────────────────

    async def get_by_id_and_session(
        self,
        *,
        session_id: str,
        message_id: str,
    ) -> Message | None:
        """Return one live message by ``(id, session_id)``, or ``None``."""
        return await self.find_unique_by_column_values(
            {"id": message_id, "session_id": session_id},
        )

    async def list_by_session(
        self,
        session_id: str,
        *,
        page: int = 1,
        page_size: int = 20,
    ) -> list[Message]:
        """Messages of one session, oldest first, paginated."""
        offset = (page - 1) * page_size
        stmt = text(
            f"select * from {_TABLE_NAME} "
            f"where session_id = :session_id and {_LIVE} "
            f"order by {_FEED_ORDER} limit :limit offset :offset"
        ).bindparams(session_id=session_id, limit=page_size, offset=offset)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def list_recent_by_session(
        self,
        session_id: str,
        *,
        limit: int,
    ) -> list[Message]:
        """The most recent messages of a session, chronological with user turns first.

        Mirrors the upstream recent-window read: the newest ``limit`` rows
        are fetched newest-first, then re-sorted oldest-first with user
        turns ahead of assistant turns on equal timestamps.
        """
        stmt = text(
            f"select * from {_TABLE_NAME} "
            f"where session_id = :session_id and {_LIVE} "
            f"order by {_RECENT_ORDER} limit :limit"
        ).bindparams(session_id=session_id, limit=limit)
        result = await self._session.execute(stmt)
        rows = [self._hydrate(m) for m in result.mappings().all()]
        rows.sort(key=lambda m: (m.created_at, 0 if m.role == _ROLE_USER else 1))
        return rows

    async def list_by_session_before_time(
        self,
        session_id: str,
        *,
        before_time: datetime,
        limit: int,
    ) -> list[Message]:
        """Messages of a session created before ``before_time``, chronological."""
        stmt = text(
            f"select * from {_TABLE_NAME} "
            "where session_id = :session_id and created_at < :before_time "
            f"and {_LIVE} order by {_RECENT_ORDER} limit :limit"
        ).bindparams(session_id=session_id, before_time=before_time, limit=limit)
        result = await self._session.execute(stmt)
        rows = [self._hydrate(m) for m in result.mappings().all()]
        rows.sort(key=lambda m: (m.created_at, 0 if m.role == _ROLE_USER else 1))
        return rows

    async def get_first_user_message(self, session_id: str) -> Message | None:
        """The first user message of a session, or ``None``."""
        stmt = text(
            f"select * from {_TABLE_NAME} "
            "where session_id = :session_id and role = :role "
            f"and {_LIVE} order by created_at asc, id asc limit 1"
        ).bindparams(session_id=session_id, role=_ROLE_USER)
        result = await self._session.execute(stmt)
        return self._hydrate_opt(result.mappings().first())

    async def get_by_request_id(
        self,
        *,
        session_id: str,
        request_id: str,
    ) -> Message | None:
        """Resolve a message by its request id within a session, or ``None``."""
        if not request_id:
            return None
        return await self.find_unique_by_column_values(
            {"session_id": session_id, "request_id": request_id},
        )

    async def list_knowledge_ids_by_session(self, session_id: str) -> list[str]:
        """The distinct non-empty ``knowledge_id`` values of a session's live messages."""
        stmt = text(
            f"select distinct knowledge_id from {_TABLE_NAME} "
            "where session_id = :session_id and knowledge_id <> '' "
            f"and knowledge_id is not null and {_LIVE}"
        ).bindparams(session_id=session_id)
        result = await self._session.execute(stmt)
        return [str(value) for value in result.scalars().all()]

    async def search_by_keyword(
        self,
        *,
        keyword: str,
        session_ids: list[str] | None = None,
        limit: int = 20,
    ) -> list[Message]:
        """ILIKE search across the caller's live messages.

        When ``session_ids`` is supplied, the search is restricted to those
        sessions; otherwise it scans every live row of the workspace.
        Sessions whose ``session_id`` is empty are always ignored by the
        SQL filter — the live-row guard already covers them.

        The match runs against ``content``; an empty ``keyword`` short-
        circuits to an empty list to keep the SQL ``like`` predicate
        well-formed regardless of the caller.
        """
        term = (keyword or "").strip()
        if not term:
            return []
        # ``session_id`` / ``id`` / ``created_at`` are literal column
        # names declared in this module — not user input. The ``like``
        # pattern is a bindparam, never an f-string interpolation.
        self._assert_safe_identifier("session_id", kind="column")
        self._assert_safe_identifier("id", kind="column")
        self._assert_safe_identifier("created_at", kind="column")
        self._assert_safe_identifier("content", kind="column")
        ids_clause = ""
        if session_ids:
            ids_clause = " and session_id in :session_ids"
        stmt = text(
            f"select * from {_TABLE_NAME} "
            "where content ilike :pattern "
            f"and {_LIVE} "
            f"{ids_clause} "
            f"order by {_RECENT_ORDER} limit :limit"
        ).bindparams(
            pattern=f"%{term}%",
            limit=limit,
            **({"session_ids": tuple(session_ids)} if session_ids else {}),
        )
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def list_by_knowledge_ids(
        self,
        knowledge_ids: list[str],
    ) -> list[Message]:
        """Bulk lookup of live messages by their ``knowledge_id`` column.

        Used by the search path to map KB vector-search hits back to the
        messages that produced them. Returns an empty list when
        ``knowledge_ids`` is empty so the SQL ``in`` clause is well-formed.
        """
        if not knowledge_ids:
            return []
        self._assert_safe_identifier("knowledge_id", kind="column")
        stmt = text(
            f"select * from {_TABLE_NAME} "
            "where knowledge_id in :knowledge_ids "
            f"and {_LIVE} order by {_RECENT_ORDER}"
        ).bindparams(knowledge_ids=tuple(knowledge_ids))
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def list_by_request_ids(
        self,
        request_ids: list[str],
    ) -> list[Message]:
        """Bulk lookup of live messages by their ``request_id`` column.

        The search path uses this to fetch the partner of a Q&A pair
        that matched on only one role. Returns an empty list when
        ``request_ids`` is empty so the SQL ``in`` clause is well-formed.
        """
        if not request_ids:
            return []
        self._assert_safe_identifier("request_id", kind="column")
        stmt = text(
            f"select * from {_TABLE_NAME} "
            "where request_id in :request_ids "
            f"and {_LIVE} order by {_RECENT_ORDER}"
        ).bindparams(request_ids=tuple(request_ids))
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]


__all__ = ["MessageRepository"]
