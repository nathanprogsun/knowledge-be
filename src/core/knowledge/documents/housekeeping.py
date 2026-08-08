"""Knowledge housekeeping — periodic recovery of stuck document rows.

The domain model keeps documents in non-terminal parse states
(``pending`` / ``processing`` / ``finalizing``) while ingestion runs.
Most stalls are caught by retry / dead-letter pipelines, but a few
failure modes leave a row spinning forever with no defensive net: a
worker killed mid-handler before any cleanup ran, a docreader call that
exceeded its timeout and whose retry never fired, or a subtask counter
that never reached zero. The sweep below is that net.

Two sweeps run on each pass:

- Sweep A (parse states): rows stuck in ``pending`` / ``processing`` /
  ``finalizing`` past a stale threshold are promoted to ``failed``.
  A row is only promoted when BOTH its ``updated_at`` and its most
  recent span heartbeat predate the cutoff, so a genuinely long-running
  stage (a docreader on a large document, embedding tens of thousands of
  chunks) is not killed mid-flight. An optional task inspector adds a
  second gate: a row whose enrichment subtasks are merely backlogged in
  the queue (no worker has picked them up yet, hence no fresh span) is
  treated as backpressure, not a stall, and left alone.
- Sweep B (summary): rows stuck in ``summary_status = processing`` past
  a fixed one-hour cutoff are promoted to ``failed``. Summary is a
  single bounded LLM call with no span heartbeat, so the simple
  ``updated_at`` check applies.

The sweep is stateless: it takes a store (a thin SQL seam) plus
injectable inspectors and returns a result summary. Scheduling is a
separate asyncio wrapper so the web layer can compose it without the
sweep logic knowing about timers.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol, cast

from sqlalchemy import text
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.common.exception import ValidationError
from src.common.json import SqlValue
from src.common.session_provider import session_scope
from src.core.knowledge.documents.types import (
    PARSE_STATUS_FAILED,
    PARSE_STATUS_FINALIZING,
    PARSE_STATUS_PENDING,
    PARSE_STATUS_PROCESSING,
    SUMMARY_STATUS_FAILED,
    SUMMARY_STATUS_PROCESSING,
)
from src.db.models.knowledge import Document

logger = logging.getLogger(__name__)

# ── Config constants ──────────────────────────────────────────────────

# Non-terminal parse states the sweep may promote to ``failed``.
PARSE_STATUSES_TO_RECOVER: frozenset[str] = frozenset(
    {PARSE_STATUS_PENDING, PARSE_STATUS_PROCESSING, PARSE_STATUS_FINALIZING}
)

# Sweep A floor: a "processing" row may sit untouched for at least this
# long before recovery, so a slow large-document parse cannot be killed
# mid-flight.
_STALE_FLOOR = timedelta(hours=1)

# Sweep A ceiling buffer: absorbs scheduling jitter on top of the
# operator-configured document process timeout.
_SCHEDULE_BUFFER = timedelta(minutes=10)

# Sweep B cutoff: summary is a single bounded LLM call, so a shorter
# window is safe.
_SUMMARY_CUTOFF = timedelta(hours=1)

# Default periodic scan interval.
DEFAULT_SWEEP_INTERVAL = timedelta(minutes=5)

# Env var that disables the periodic scan. Default-on: a missing or
# empty value keeps the sweep enabled; "false" / "0" / "off" / "no"
# (case-insensitive) opt out.
ENABLED_ENV_VAR = "HOUSEKEEPING_ENABLED"
_DISABLED_VALUES: frozenset[str] = frozenset({"0", "false", "off", "no"})

# Span table referenced by Sweep A's heartbeat query. It is created by
# the span-tracking feature; the name mirrors the upstream persistence
# contract.
_SPANS_TABLE = "knowledge_processing_spans"

# ── Domain error codes ────────────────────────────────────────────────

_INVALID_PROCESS_TIMEOUT_CODE = "housekeeping.invalid_process_timeout"
_INVALID_SWEEP_INTERVAL_CODE = "housekeeping.invalid_sweep_interval"

# ── Result shape ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class HousekeepingResult:
    """Summary of one sweep pass.

    ``parse_candidates`` counts the rows found stale on ``updated_at``
    alone; ``span_skipped`` and ``queue_skipped`` are the subset that
    were left alone by the heartbeat and queue gates respectively;
    ``recovered_parse_rows`` is what actually flipped to ``failed``.
    """

    parse_candidates: int
    recovered_parse_rows: int
    span_skipped: int
    queue_skipped: int
    recovered_summary_rows: int


# ── Injectable seams ──────────────────────────────────────────────────


class KnowledgeSweepStore(Protocol):
    """Persistence seam the sweep operates against.

    Separated from the sweep logic so unit tests can drive the gates
    with in-memory doubles; the SQL-backed implementation runs against
    the real schema.
    """

    async def find_parse_candidates(self, cutoff: datetime) -> list[Document]:
        """Rows in a recoverable parse state whose ``updated_at`` is stale."""
        ...

    async def span_last_seen_times(self, knowledge_ids: list[str]) -> dict[str, datetime]:
        """Most recent span heartbeat per knowledge id (subset requested)."""
        ...

    async def mark_parse_rows_failed(self, ids: list[str], *, message: str) -> int:
        """Promote rows still in a recoverable parse state to ``failed``.

        Returns the number of rows affected.
        """
        ...

    async def mark_summary_rows_failed(self, cutoff: datetime) -> int:
        """Promote rows stuck in ``summary_status = processing`` to ``failed``.

        Returns the number of rows affected.
        """
        ...


class TaskInspector(Protocol):
    """Read-only probe for whether tasks still reference a knowledge id.

    Best-effort and short-circuiting: returns ``True`` as soon as the
    first match is seen. A raised error is treated by the sweep as a
    fail-safe (the row is recovered).
    """

    async def has_queued_tasks_for_knowledge(self, knowledge_id: str) -> bool:
        """Whether any queued / active task references ``knowledge_id``."""
        ...


class SqlKnowledgeSweepStore:
    """SQL-backed sweep store over the ``documents`` table.

    Raw SQL via ``sqlalchemy.text()`` with named bindparams, following
    the repository conventions: reads filter soft-deleted rows and user
    input only ever reaches bound parameters.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._documents_table = Document.fq_table_name()

    async def find_parse_candidates(self, cutoff: datetime) -> list[Document]:
        statuses = tuple(PARSE_STATUSES_TO_RECOVER)
        placeholders = ", ".join(f":ps_{i}" for i in range(len(statuses)))
        stmt = text(
            f"select * from {self._documents_table} "
            f"where parse_status in ({placeholders}) and updated_at < :cutoff "
            "and deleted_at is null"
        ).bindparams(**{f"ps_{i}": s for i, s in enumerate(statuses)}, cutoff=cutoff)
        result = await self._session.execute(stmt)
        return [
            cast("Document", Document.from_row(cast("Mapping[str, SqlValue]", m)))
            for m in result.mappings().all()
        ]

    async def span_last_seen_times(self, knowledge_ids: list[str]) -> dict[str, datetime]:
        if not knowledge_ids:
            return {}
        placeholders = ", ".join(f":id_{i}" for i in range(len(knowledge_ids)))
        stmt = text(
            f"select knowledge_id, max(updated_at) as last_seen from {_SPANS_TABLE} "
            f"where knowledge_id in ({placeholders}) group by knowledge_id"
        ).bindparams(**{f"id_{i}": kid for i, kid in enumerate(knowledge_ids)})
        result = await self._session.execute(stmt)
        out: dict[str, datetime] = {}
        for mapping in result.mappings().all():
            raw_seen = mapping["last_seen"]
            parsed = parse_heartbeat_time(cast("str | datetime | None", raw_seen))
            if parsed is not None:
                out[str(mapping["knowledge_id"])] = parsed
        return out

    async def mark_parse_rows_failed(self, ids: list[str], *, message: str) -> int:
        if not ids:
            return 0
        id_placeholders = ", ".join(f":id_{i}" for i in range(len(ids)))
        statuses = tuple(PARSE_STATUSES_TO_RECOVER)
        ps_placeholders = ", ".join(f":ps_{i}" for i in range(len(statuses)))
        stmt = text(
            f"update {self._documents_table} set parse_status = :status, "
            "error_message = :message, pending_subtasks_count = :pending_count "
            f"where id in ({id_placeholders}) and parse_status in ({ps_placeholders}) "
            "and deleted_at is null"
        ).bindparams(
            status=PARSE_STATUS_FAILED,
            message=message,
            pending_count=0,
            **{f"id_{i}": v for i, v in enumerate(ids)},
            **{f"ps_{i}": s for i, s in enumerate(statuses)},
        )
        result = await self._session.execute(stmt)
        return cast("CursorResult[SqlValue]", result).rowcount or 0

    async def mark_summary_rows_failed(self, cutoff: datetime) -> int:
        stmt = text(
            f"update {self._documents_table} set summary_status = :summary_status "
            "where summary_status = :processing and updated_at < :cutoff "
            "and deleted_at is null"
        ).bindparams(
            summary_status=SUMMARY_STATUS_FAILED,
            processing=SUMMARY_STATUS_PROCESSING,
            cutoff=cutoff,
        )
        result = await self._session.execute(stmt)
        return cast("CursorResult[SqlValue]", result).rowcount or 0


# ── Pure helpers ──────────────────────────────────────────────────────


def housekeeping_enabled() -> bool:
    """Whether the periodic scan is enabled.

    Default-on: a missing or empty ``HOUSEKEEPING_ENABLED`` enables the
    sweep; operators set it to ``false`` / ``0`` / ``off`` / ``no``
    (case-insensitive) to opt out. Matches the upstream default-on
    posture where no environment change is required for the safety net
    to engage.
    """
    value = os.environ.get(ENABLED_ENV_VAR, "").strip()
    if not value:
        return True
    return value.lower() not in _DISABLED_VALUES


def stale_threshold(document_process_timeout: timedelta) -> timedelta:
    """How long a ``processing`` row may sit untouched before recovery.

    The floor is 1 hour so a genuinely slow large-document parse cannot
    be killed mid-flight; the ceiling scales with the operator-configured
    document process timeout plus a 10-minute buffer that absorbs
    scheduling jitter.
    """
    if document_process_timeout.total_seconds() < 0:
        raise ValidationError(
            code=_INVALID_PROCESS_TIMEOUT_CODE,
            message="document_process_timeout must not be negative",
        )
    return max(_STALE_FLOOR, document_process_timeout) + _SCHEDULE_BUFFER


def parse_heartbeat_time(value: str | datetime | None) -> datetime | None:
    """Parse a span heartbeat timestamp into a timezone-aware datetime.

    Accepts the formats Postgres and SQLite emit for a TIMESTAMP column
    read back through MAX(): RFC 3339 and space-separated variants, with
    optional fractional seconds and offset. Naive values are assumed to
    be UTC. Returns ``None`` when nothing parses — the caller treats an
    unparseable row as "no heartbeat", which fails safe toward recovery.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value
    if value is None:
        return None
    text_value = value.strip()
    if not text_value:
        return None
    try:
        # ``fromisoformat`` covers RFC 3339 (with or without ``Z`` /
        # offset / fractional seconds) and the space-separated variants
        # SQLite emits; unknown shapes raise and fail safe toward recovery.
        parsed = datetime.fromisoformat(text_value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def format_duration(delta: timedelta) -> str:
    """Render a duration in the ``1h10m0s`` style used in error messages."""
    total = int(delta.total_seconds())
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes}m{seconds}s"
    if minutes:
        return f"{minutes}m{seconds}s"
    return f"{seconds}s"


# ── Sweep ─────────────────────────────────────────────────────────────


async def run_sweep(
    *,
    store: KnowledgeSweepStore,
    document_process_timeout: timedelta,
    task_inspector: TaskInspector | None = None,
    now: datetime | None = None,
) -> HousekeepingResult:
    """Run one housekeeping pass and return the result summary.

    ``now`` is injectable for deterministic tests; it defaults to the
    current UTC time. Never raises for a failing sweep: a candidate-query
    failure aborts the pass (matching upstream), while a failure in any
    later stage logs and fails safe toward recovery so a sweep error
    cannot take the process down.
    """
    threshold = stale_threshold(document_process_timeout)
    current = now if now is not None else datetime.now(UTC)
    cutoff = current - threshold

    candidates: list[Document] = []
    try:
        candidates = await store.find_parse_candidates(cutoff)
    except Exception:
        logger.exception("housekeeping: candidate query failed; skipping sweep")
        return HousekeepingResult(0, 0, 0, 0, 0)

    recovered_parse = 0
    span_skipped = 0
    queue_skipped = 0
    if candidates:
        stuck = await _filter_by_last_span_activity(store, candidates, cutoff)
        span_skipped = len(candidates) - len(stuck)
        stuck, queue_skipped = await _filter_out_queued(task_inspector, stuck)
        if stuck:
            message = (
                f"task stuck in processing > {format_duration(threshold)}, "
                "recovered by housekeeping"
            )
            try:
                recovered_parse = await store.mark_parse_rows_failed(
                    [c.id for c in stuck], message=message
                )
            except Exception:
                logger.exception("housekeeping: parse sweep update failed")

    recovered_summary = 0
    try:
        recovered_summary = await store.mark_summary_rows_failed(current - _SUMMARY_CUTOFF)
    except Exception:
        logger.exception("housekeeping: summary sweep failed")

    return HousekeepingResult(
        parse_candidates=len(candidates),
        recovered_parse_rows=recovered_parse,
        span_skipped=span_skipped,
        queue_skipped=queue_skipped,
        recovered_summary_rows=recovered_summary,
    )


async def _filter_by_last_span_activity(
    store: KnowledgeSweepStore,
    candidates: list[Document],
    cutoff: datetime,
) -> list[Document]:
    """Return candidates whose most recent span heartbeat predates the cutoff.

    Candidates with no span rows at all also pass through: they are
    lite-mode or pre-instrumentation rows whose ``updated_at`` staleness
    already proved them stuck, and there is no heartbeat to override
    that. On a heartbeat query error the fail-safe direction is to keep
    every candidate (never under-recover).
    """
    knowledge_ids = [c.id for c in candidates]
    try:
        heartbeat = await store.span_last_seen_times(knowledge_ids)
    except Exception:
        logger.exception("housekeeping: span heartbeat query failed; recovering all candidates")
        return candidates
    kept: list[Document] = []
    for candidate in candidates:
        last_seen = heartbeat.get(candidate.id)
        if last_seen is not None and last_seen > cutoff:
            # Active span heartbeat — still progressing, leave alone.
            continue
        kept.append(candidate)
    return kept


async def _filter_out_queued(
    task_inspector: TaskInspector | None,
    candidates: Sequence[Document],
) -> tuple[list[Document], int]:
    """Drop candidates that still have a queued task referencing them.

    A dropped candidate is backlogged, not orphaned: its enrichment
    subtasks are waiting for a worker, so the missing span heartbeat is
    expected and recovering it would be a false positive. When no
    inspector is wired the gate is a pass-through. On a probe error the
    fail-safe direction is to keep the candidate as stuck.
    """
    if task_inspector is None or not candidates:
        return list(candidates), 0
    kept: list[Document] = []
    skipped = 0
    for candidate in candidates:
        try:
            queued = await task_inspector.has_queued_tasks_for_knowledge(candidate.id)
        except Exception:
            logger.exception(
                "housekeeping: queue probe failed for %s; treating as stuck",
                candidate.id,
            )
            kept.append(candidate)
            continue
        if queued:
            skipped += 1
            continue
        kept.append(candidate)
    return kept, skipped


# ── Periodic scan ─────────────────────────────────────────────────────


class SweepScheduler:
    """Runs ``run_sweep`` on a fixed interval in the background.

    Each tick opens a fresh session from the factory (the sweep runs
    outside any request scope) and commits the pass through
    ``session_scope``. ``start`` is idempotent and honours the
    ``HOUSEKEEPING_ENABLED`` gate; ``stop`` cancels the loop and awaits
    the in-flight pass.
    """

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        document_process_timeout: timedelta,
        task_inspector: TaskInspector | None = None,
        interval: timedelta = DEFAULT_SWEEP_INTERVAL,
    ) -> None:
        if interval.total_seconds() <= 0:
            raise ValidationError(
                code=_INVALID_SWEEP_INTERVAL_CODE,
                message="sweep interval must be positive",
            )
        self._session_factory = session_factory
        self._document_process_timeout = document_process_timeout
        self._task_inspector = task_inspector
        self._interval = interval
        self._task: asyncio.Task[None] | None = None
        self._started = False

    async def start(self) -> None:
        """Begin the periodic scan. Idempotent; no-op when disabled."""
        if self._started:
            return
        if not housekeeping_enabled():
            logger.info("housekeeping: periodic scan disabled via %s", ENABLED_ENV_VAR)
            return
        self._task = asyncio.create_task(self._loop(), name="knowledge-housekeeping-sweep")
        self._started = True
        logger.info("housekeeping: periodic scan started (interval=%s)", self._interval)

    async def stop(self) -> None:
        """Stop the periodic scan and wait for the in-flight pass."""
        if not self._started:
            return
        self._started = False
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def _loop(self) -> None:
        while True:
            try:
                await self._run_once()
            except Exception:
                logger.exception("housekeeping: sweep failed")
            await asyncio.sleep(self._interval.total_seconds())

    async def _run_once(self) -> None:
        async with session_scope(self._session_factory) as session:
            store = SqlKnowledgeSweepStore(session)
            result = await run_sweep(
                store=store,
                document_process_timeout=self._document_process_timeout,
                task_inspector=self._task_inspector,
            )
            logger.info(
                "housekeeping: parse_candidates=%d recovered=%d span_skipped=%d "
                "queue_skipped=%d summary_recovered=%d",
                result.parse_candidates,
                result.recovered_parse_rows,
                result.span_skipped,
                result.queue_skipped,
                result.recovered_summary_rows,
            )


__all__ = [
    "DEFAULT_SWEEP_INTERVAL",
    "ENABLED_ENV_VAR",
    "PARSE_STATUSES_TO_RECOVER",
    "HousekeepingResult",
    "KnowledgeSweepStore",
    "SqlKnowledgeSweepStore",
    "SweepScheduler",
    "TaskInspector",
    "format_duration",
    "housekeeping_enabled",
    "parse_heartbeat_time",
    "run_sweep",
    "stale_threshold",
]
