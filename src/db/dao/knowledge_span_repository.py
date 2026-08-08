"""Span persistence — raw SQL only, no ORM.

Persists the per-(knowledge, attempt) span tree used by the processing
pipeline. Operations mirror the upstream span repository:

- ``upsert`` covers every state transition (Begin/End/Fail/Skip) through
  one ``ON CONFLICT (knowledge_id, attempt, span_id)`` write so the row
  stays internally consistent. ``input`` / ``output`` / ``metadata`` are
  only written when the incoming row sets them — a transition that has
  no new content preserves the stored value instead of clobbering it.
- ``next_attempt`` allocates a new attempt for re-parses without touching
  historical rows; ``latest_attempt`` reads the highest recorded attempt.
- ``list_by_attempt`` is the only read path; the caller builds the tree
  in memory rather than recursing through the DB.
- The ``cancel_*`` helpers flip non-terminal spans to ``cancelled`` for
  the cascade / abort paths. ``cancel_descendants`` walks the tree
  level by level so it stays portable across dialects; the flat
  ``cancel_all_open_spans`` covers fan-out shapes whose parents are
  already terminal.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from sqlalchemy import JSON, bindparam, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import CursorResult

from src.common.exception import ValidationError
from src.common.json import BindParams, SqlValue
from src.db.dao.generic_repository import GenericRepository
from src.db.models.knowledge_processing_span import KnowledgeProcessingSpan

# JSONB on Postgres, JSON on other dialects (e.g. SQLite in tests).
_JSON = JSON().with_variant(JSONB(), "postgresql")

_TABLE = "knowledge_processing_spans"

# The unique constraint targeted by ``upsert``'s conflict clause.
_CONFLICT_COLUMNS = ("knowledge_id", "attempt", "span_id")

# Non-terminal statuses — the set every cancel sweep flips to cancelled.
_OPEN_STATUSES = ("pending", "running")

# Cascade-cancel marker written onto descendants of a failed span.
_UPSTREAM_FAILED_CODE = "UPSTREAM_FAILED"

# Iterative walk bound; a tree deeper than this is treated as malformed.
_CANCEL_DEPTH_LIMIT = 16


class KnowledgeSpanRepository(GenericRepository[KnowledgeProcessingSpan]):
    """`knowledge_processing_spans`-table SQL — upsert + tree queries."""

    model_class = KnowledgeProcessingSpan

    # ── Upsert ──────────────────────────────────────────────────────

    async def upsert(self, row: KnowledgeProcessingSpan) -> KnowledgeProcessingSpan:
        """Insert or update one span row, returning the persisted row.

        The conflict target is the ``(knowledge_id, attempt, span_id)``
        unique constraint, so re-opening a span keeps the same row (and
        any subspan that already references it stays attached). The
        update list writes every transition column plus ``input`` /
        ``output`` / ``metadata`` only when the incoming row sets them —
        a value of ``None`` preserves the stored column. ``attempt``
        defaults to 1 when the caller passes 0.
        """
        if row.knowledge_id == "" or row.span_id == "":
            raise ValidationError(
                code="span.invalid_row",
                message="span upsert requires knowledge_id and span_id",
            )
        attempt = row.attempt if row.attempt > 0 else 1
        insert_columns = self.model_class.insert_sql_column_list()
        self._assert_safe_identifier(_TABLE, kind="table")
        for col in insert_columns:
            self._assert_safe_identifier(col, kind="column")
        col_list = ", ".join(f'"{c}"' for c in insert_columns)
        param_list = ", ".join(f":{c}" for c in insert_columns)
        # content-preserving columns: only added to the update list when
        # the incoming row has a value to write (see module docstring).
        update_set = [
            f'"{c}" = EXCLUDED."{c}"'
            for c in (
                "status",
                "error_code",
                "error_message",
                "error_detail",
                "started_at",
                "finished_at",
                "duration_ms",
                "updated_at",
            )
        ]
        for col in ("input", "output", "metadata"):
            if getattr(row, col) is not None:
                update_set.append(f'"{col}" = EXCLUDED."{col}"')
        conflict_cols = ", ".join(f'"{c}"' for c in _CONFLICT_COLUMNS)
        stmt_text = (
            f"insert into {_TABLE} ({col_list}) values ({param_list}) "
            f"on conflict ({conflict_cols}) do update set {', '.join(update_set)} "
            "returning *"
        )
        params = row.insert_bind_params()
        params["attempt"] = attempt
        json_bps = [
            bindparam(col, type_=_JSON)
            for col in insert_columns
            if col in self._json_columns
        ]
        stmt = text(stmt_text).bindparams(*json_bps, **params)
        result = await self._session.execute(stmt)
        mapping = result.mappings().first()
        if mapping is None:
            raise ValidationError(
                code="span.upsert_no_row",
                message="span upsert returned no row",
            )
        return self._hydrate(mapping)

    # ── Attempt allocation ──────────────────────────────────────────

    async def next_attempt(self, knowledge_id: str) -> int:
        """Return the highest recorded attempt plus one (1 for a fresh id)."""
        return (await self._max_attempt(knowledge_id)) + 1

    async def latest_attempt(self, knowledge_id: str) -> int:
        """Return the highest recorded attempt, or 0 when never parsed."""
        return await self._max_attempt(knowledge_id)

    async def _max_attempt(self, knowledge_id: str) -> int:
        stmt = text(
            f"select coalesce(max(attempt), 0) from {_TABLE} where knowledge_id = :knowledge_id"
        ).bindparams(knowledge_id=knowledge_id)
        result = await self._session.execute(stmt)
        value = result.scalar_one()
        return int(value) if value is not None else 0

    # ── Reads ───────────────────────────────────────────────────────

    async def list_by_attempt(
        self,
        knowledge_id: str,
        attempt: int,
    ) -> list[KnowledgeProcessingSpan]:
        """Return every span of (knowledge, attempt), or all attempts.

        ``attempt <= 0`` returns every attempt's spans for the knowledge
        (the caller-visible equivalent of "no attempt filter"). The
        ``id ASC`` order preserves natural insertion order for stable
        rendering of fan-out subspans.
        """
        if knowledge_id == "":
            return []
        if attempt > 0:
            stmt = text(
                f"select * from {_TABLE} "
                "where knowledge_id = :knowledge_id and attempt = :attempt "
                "order by id asc"
            ).bindparams(knowledge_id=knowledge_id, attempt=attempt)
        else:
            stmt = text(
                f"select * from {_TABLE} where knowledge_id = :knowledge_id order by id asc"
            ).bindparams(knowledge_id=knowledge_id)
        result = await self._session.execute(stmt)
        return [self._hydrate(m) for m in result.mappings().all()]

    async def get_span(
        self,
        knowledge_id: str,
        attempt: int,
        span_id: str,
    ) -> KnowledgeProcessingSpan | None:
        """Return one span row, or ``None`` when it does not exist."""
        stmt = text(
            f"select * from {_TABLE} "
            "where knowledge_id = :knowledge_id and attempt = :attempt and span_id = :span_id"
        ).bindparams(knowledge_id=knowledge_id, attempt=attempt, span_id=span_id)
        result = await self._session.execute(stmt)
        return self._hydrate_opt(result.mappings().first())

    # ── Cancel sweeps ───────────────────────────────────────────────

    async def cancel_descendants(
        self,
        knowledge_id: str,
        attempt: int,
        parent_span_id: str,
        reason: str,
    ) -> int:
        """Flip every pending/running descendant of ``parent_span_id``.

        Walks the tree level by level: each iteration finds the
        non-terminal children of the current frontier, flips them to
        ``cancelled`` with a ``UPSTREAM_FAILED`` marker, then descends.
        Terminal rows are left untouched so the UI keeps their original
        outcome. Returns the number of rows flipped.
        """
        frontier: list[str] = [parent_span_id]
        total_affected = 0
        for _depth in range(_CANCEL_DEPTH_LIMIT):
            if not frontier:
                break
            placeholders = ", ".join(f":f_{i}" for i in range(len(frontier)))
            status_placeholders = ", ".join(
                f":st_{i}" for i in range(len(_OPEN_STATUSES))
            )
            params: BindParams = {
                "knowledge_id": knowledge_id,
                "attempt": attempt,
                **{f"f_{i}": v for i, v in enumerate(frontier)},
                **{f"st_{i}": s for i, s in enumerate(_OPEN_STATUSES)},
            }
            find_stmt = text(
                f"select span_id from {_TABLE} "
                f"where knowledge_id = :knowledge_id and attempt = :attempt "
                f"and parent_span_id in ({placeholders}) "
                f"and status in ({status_placeholders})"
            ).bindparams(**params)
            result = await self._session.execute(find_stmt)
            child_ids = [m["span_id"] for m in result.mappings().all()]
            if not child_ids:
                break
            id_placeholders = ", ".join(f":id_{i}" for i in range(len(child_ids)))
            update_params: BindParams = {
                "knowledge_id": knowledge_id,
                "attempt": attempt,
                "reason": reason,
                **{f"id_{i}": v for i, v in enumerate(child_ids)},
            }
            update_stmt = text(
                f"update {_TABLE} set status = :cancelled, "
                "error_code = :upstream, error_message = :reason "
                f"where knowledge_id = :knowledge_id and attempt = :attempt "
                f"and span_id in ({id_placeholders})"
            ).bindparams(
                cancelled="cancelled",
                upstream=_UPSTREAM_FAILED_CODE,
                **update_params,
            )
            affected = await self._session.execute(update_stmt)
            total_affected += (
                cast("CursorResult[SqlValue]", affected).rowcount or 0
            )
            frontier = child_ids
        return total_affected

    async def cancel_all_open_spans(
        self,
        knowledge_id: str,
        attempt: int,
        error_code: str,
        reason: str,
        *,
        now: datetime,
    ) -> int:
        """Flip every pending/running span of the attempt to ``cancelled``.

        The flat sweep ignores the tree shape on purpose: fan-out stages
        end their own row the moment they finish dispatching async work,
        so their children are still ``running`` under a terminal parent —
        a tree walk would orphan those leaves. ``finished_at`` and
        ``updated_at`` are stamped so the row stays observable with its
        original start time.
        """
        status_placeholders = ", ".join(f":st_{i}" for i in range(len(_OPEN_STATUSES)))
        params: BindParams = {
            "cancelled": "cancelled",
            "knowledge_id": knowledge_id,
            "attempt": attempt,
            "error_code": error_code,
            "reason": reason,
            "now": now,
            **{f"st_{i}": s for i, s in enumerate(_OPEN_STATUSES)},
        }
        stmt = text(
            f"update {_TABLE} set status = :cancelled, error_code = :error_code, "
            "error_message = :reason, finished_at = :now, updated_at = :now "
            f"where knowledge_id = :knowledge_id and attempt = :attempt "
            f"and status in ({status_placeholders})"
        ).bindparams(**params)
        result = await self._session.execute(stmt)
        return cast("CursorResult[SqlValue]", result).rowcount or 0

    async def cancel_open_spans_by_name(
        self,
        knowledge_id: str,
        attempt: int,
        name: str,
        error_code: str,
        reason: str,
        *,
        now: datetime,
    ) -> int:
        """Flip pending/running spans with the given name to ``cancelled``.

        Used before re-opening a subspan after a retry or restart so the
        trace does not accumulate duplicate rows for the same subtask.
        """
        if knowledge_id == "" or attempt <= 0 or name == "":
            return 0
        status_placeholders = ", ".join(f":st_{i}" for i in range(len(_OPEN_STATUSES)))
        params: BindParams = {
            "cancelled": "cancelled",
            "knowledge_id": knowledge_id,
            "attempt": attempt,
            "name": name,
            "error_code": error_code,
            "reason": reason,
            "now": now,
            **{f"st_{i}": s for i, s in enumerate(_OPEN_STATUSES)},
        }
        stmt = text(
            f"update {_TABLE} set status = :cancelled, error_code = :error_code, "
            "error_message = :reason, finished_at = :now, updated_at = :now "
            f"where knowledge_id = :knowledge_id and attempt = :attempt "
            f"and name = :name and status in ({status_placeholders})"
        ).bindparams(**params)
        result = await self._session.execute(stmt)
        return cast("CursorResult[SqlValue]", result).rowcount or 0


__all__ = ["KnowledgeSpanRepository"]
