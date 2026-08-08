"""Unit + integration tests for the knowledge housekeeping sweep.

Unit tests drive the sweep with an in-memory store double and a
controllable task inspector, covering the stall gates: the span-heartbeat
check, the queued-task check, and both fail-safe directions. Integration
tests run the SQL-backed store against the real applied schema
(``documents`` table) and exercise the recovery writes end-to-end —
run with ``DATABASE_URL_OVERRIDE``.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from random import randint

import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.common.exception import ValidationError
from src.core.knowledge.documents.housekeeping import (
    DEFAULT_SWEEP_INTERVAL,
    PARSE_STATUSES_TO_RECOVER,
    HousekeepingResult,
    SqlKnowledgeSweepStore,
    SweepScheduler,
    format_duration,
    housekeeping_enabled,
    parse_heartbeat_time,
    run_sweep,
    stale_threshold,
)
from src.core.knowledge.documents.types import (
    PARSE_STATUS_COMPLETED,
    PARSE_STATUS_FAILED,
    PARSE_STATUS_FINALIZING,
    PARSE_STATUS_PENDING,
    PARSE_STATUS_PROCESSING,
    SUMMARY_STATUS_FAILED,
    SUMMARY_STATUS_NONE,
    SUMMARY_STATUS_PROCESSING,
)
from src.db.models.knowledge import Document
from src.settings import get_settings, reset_settings_cache
from tests.integration.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
_FAKER_SEED_MAX = 100_000_000


@pytest.fixture(autouse=True)
def faker_seed() -> None:
    """Re-seed Faker per test for varied-but-reproducible generation."""
    Faker.seed(randint(1, _FAKER_SEED_MAX))


def _did() -> str:
    return f"doc-{uuid.uuid4().hex[:12]}"


def _kbid() -> str:
    return f"kb-{uuid.uuid4().hex[:12]}"


def _row(
    *,
    id: str | None = None,
    tenant_id: int | None = None,
    parse_status: str = PARSE_STATUS_COMPLETED,
    summary_status: str = SUMMARY_STATUS_NONE,
    updated_at: datetime = _NOW,
    error_message: str | None = None,
    pending_subtasks_count: int = 0,
) -> Document:
    """Build a persisted-shape document row for seeding mocks / DB."""
    return Document.model_validate(
        {
            "id": id or _did(),
            "tenant_id": tenant_id if tenant_id is not None else make_test_tenant_id(),
            "knowledge_base_id": _kbid(),
            "type": "file",
            "title": "housekeeping fixture",
            "description": None,
            "source": "housekeeping-fixture.pdf",
            "channel": "web",
            "parse_status": parse_status,
            "pending_subtasks_count": pending_subtasks_count,
            "summary_status": summary_status,
            "enable_status": "enabled",
            "embedding_model_id": None,
            "file_name": "housekeeping-fixture.pdf",
            "file_type": "pdf",
            "file_size": 1024,
            "file_hash": None,
            "file_path": None,
            "storage_size": 2048,
            "metadata": None,
            "custom_metadata": {},
            "last_faq_import_result": None,
            "created_at": _NOW,
            "updated_at": updated_at,
            "processed_at": None,
            "error_message": error_message,
            "deleted_at": None,
        }
    )


# ── Store + inspector doubles ─────────────────────────────────────────


class _FakeStore:
    """In-memory sweep store mirroring the SQL store semantics.

    Candidates are filtered by recoverable parse state + stale
    ``updated_at``; updates flip the stored rows in place so tests can
    assert the persisted outcome through the store.
    """

    def __init__(
        self,
        *,
        rows: list[Document] | None = None,
        heartbeats: dict[str, datetime] | None = None,
        span_error: Exception | None = None,
    ) -> None:
        self.rows = {r.id: r for r in (rows or [])}
        self.heartbeats = dict(heartbeats or {})
        self.span_error = span_error
        self.recovered_message: str | None = None

    async def find_parse_candidates(self, cutoff: datetime) -> list[Document]:
        return [
            r
            for r in self.rows.values()
            if r.parse_status in PARSE_STATUSES_TO_RECOVER and r.updated_at < cutoff
        ]

    async def span_last_seen_times(self, knowledge_ids: list[str]) -> dict[str, datetime]:
        if self.span_error is not None:
            raise self.span_error
        wanted = set(knowledge_ids)
        return {k: v for k, v in self.heartbeats.items() if k in wanted}

    async def mark_parse_rows_failed(self, ids: list[str], *, message: str) -> int:
        self.recovered_message = message
        id_set = set(ids)
        count = 0
        for row_id, row in self.rows.items():
            if row_id in id_set and row.parse_status in PARSE_STATUSES_TO_RECOVER:
                self.rows[row_id] = row.model_copy(
                    update={
                        "parse_status": PARSE_STATUS_FAILED,
                        "error_message": message,
                        "pending_subtasks_count": 0,
                    }
                )
                count += 1
        return count

    async def mark_summary_rows_failed(self, cutoff: datetime) -> int:
        count = 0
        for row_id, row in self.rows.items():
            if row.summary_status == SUMMARY_STATUS_PROCESSING and row.updated_at < cutoff:
                self.rows[row_id] = row.model_copy(update={"summary_status": SUMMARY_STATUS_FAILED})
                count += 1
        return count


class _FakeInspector:
    """Controllable task inspector for the queue gate."""

    def __init__(self, queued: set[str] | None = None, error: Exception | None = None) -> None:
        self.queued = set(queued or ())
        self.error = error

    async def has_queued_tasks_for_knowledge(self, knowledge_id: str) -> bool:
        if self.error is not None:
            raise self.error
        return knowledge_id in self.queued


# ── Pure helpers ──────────────────────────────────────────────────────


def test_stale_threshold_floor_applies_when_timeout_small() -> None:
    # Arrange / Act
    threshold = stale_threshold(timedelta(hours=1))
    # Assert: 1h floor + 10m buffer.
    assert threshold == timedelta(hours=1, minutes=10)


def test_stale_threshold_scales_with_larger_timeout() -> None:
    assert stale_threshold(timedelta(hours=3)) == timedelta(hours=3, minutes=10)


def test_stale_threshold_floor_for_zero_timeout() -> None:
    assert stale_threshold(timedelta(0)) == timedelta(hours=1, minutes=10)


def test_stale_threshold_rejects_negative_timeout() -> None:
    with pytest.raises(ValidationError):
        stale_threshold(timedelta(minutes=-1))


def test_housekeeping_enabled_defaults_on_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HOUSEKEEPING_ENABLED", raising=False)
    assert housekeeping_enabled() is True


def test_housekeeping_enabled_empty_value_enables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOUSEKEEPING_ENABLED", "")
    assert housekeeping_enabled() is True


def test_housekeeping_enabled_opt_out_values(monkeypatch: pytest.MonkeyPatch) -> None:
    for value in ("false", "0", "off", "no", "FALSE", "Off"):
        monkeypatch.setenv("HOUSEKEEPING_ENABLED", value)
        assert housekeeping_enabled() is False


def test_housekeeping_enabled_unrecognized_value_enables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOUSEKEEPING_ENABLED", "maybe")
    assert housekeeping_enabled() is True


def test_parse_heartbeat_time_accepts_rfc3339_forms() -> None:
    # Arrange / Act / Assert — RFC 3339 with 'Z', explicit offset, and
    # space-separated variants are all normalised to aware UTC datetimes.
    assert parse_heartbeat_time("2026-01-15T10:00:00Z") == datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    assert parse_heartbeat_time("2026-01-15T10:00:00.123456Z") == datetime(
        2026, 1, 15, 10, 0, 0, 123456, tzinfo=UTC
    )
    assert parse_heartbeat_time("2026-01-15 10:00:00.5-07:00") == datetime(
        2026, 1, 15, 17, 0, 0, 500000, tzinfo=UTC
    )


def test_parse_heartbeat_time_accepts_naive_and_datetime_inputs() -> None:
    naive = parse_heartbeat_time("2026-01-15 10:00:00.123")
    assert naive == datetime(2026, 1, 15, 10, 0, 0, 123000, tzinfo=UTC)
    aware = datetime(2026, 1, 15, 10, 0, tzinfo=UTC)
    assert parse_heartbeat_time(aware) is aware
    naive_dt = datetime.fromisoformat("2026-01-15 10:00:00")
    assert parse_heartbeat_time(naive_dt) == datetime(2026, 1, 15, 10, 0, tzinfo=UTC)


def test_parse_heartbeat_time_rejects_unparseable() -> None:
    assert parse_heartbeat_time(None) is None
    assert parse_heartbeat_time("") is None
    assert parse_heartbeat_time("not-a-timestamp") is None


def test_format_duration_matches_upstream_style() -> None:
    assert format_duration(timedelta(hours=1, minutes=10)) == "1h10m0s"
    assert format_duration(timedelta(minutes=10)) == "10m0s"
    assert format_duration(timedelta(seconds=30)) == "30s"


# ── Sweep A: parse-state recovery ─────────────────────────────────────


async def test_run_sweep_recovers_abandoned() -> None:
    # Arrange — a processing row stale well past the 70min cutoff.
    stale = _NOW - timedelta(hours=3)
    store = _FakeStore(
        rows=[_row(id="kid-abandoned", parse_status=PARSE_STATUS_PROCESSING, updated_at=stale)]
    )

    # Act
    result = await run_sweep(store=store, document_process_timeout=timedelta(hours=1), now=_NOW)

    # Assert
    recovered = store.rows["kid-abandoned"]
    assert recovered.parse_status == PARSE_STATUS_FAILED
    assert "stuck in processing" in (recovered.error_message or "")
    assert recovered.pending_subtasks_count == 0
    assert result.recovered_parse_rows == 1


async def test_run_sweep_recovers_pending_missing_from_queue() -> None:
    # Arrange — a pending row with no queue task must not linger forever.
    stale = _NOW - timedelta(hours=3)
    store = _FakeStore(
        rows=[_row(id="kid-pending", parse_status=PARSE_STATUS_PENDING, updated_at=stale)]
    )

    # Act
    result = await run_sweep(
        store=store,
        document_process_timeout=timedelta(hours=1),
        task_inspector=_FakeInspector(queued=set()),
        now=_NOW,
    )

    # Assert
    assert store.rows["kid-pending"].parse_status == PARSE_STATUS_FAILED
    assert result.recovered_parse_rows == 1


async def test_run_sweep_preserves_pending_still_queued() -> None:
    # Arrange — backlogged work remains owned by the durable queue.
    stale = _NOW - timedelta(hours=3)
    store = _FakeStore(
        rows=[_row(id="kid-queued", parse_status=PARSE_STATUS_PENDING, updated_at=stale)]
    )

    # Act
    result = await run_sweep(
        store=store,
        document_process_timeout=timedelta(hours=1),
        task_inspector=_FakeInspector(queued={"kid-queued"}),
        now=_NOW,
    )

    # Assert
    assert store.rows["kid-queued"].parse_status == PARSE_STATUS_PENDING
    assert result.queue_skipped == 1
    assert result.recovered_parse_rows == 0


async def test_run_sweep_no_false_kill_active_span() -> None:
    # Arrange — stale row but a recent span heartbeat means still working.
    stale = _NOW - timedelta(hours=3)
    store = _FakeStore(
        rows=[_row(id="kid-active", parse_status=PARSE_STATUS_PROCESSING, updated_at=stale)],
        heartbeats={"kid-active": _NOW - timedelta(minutes=2)},
    )

    # Act
    result = await run_sweep(store=store, document_process_timeout=timedelta(hours=1), now=_NOW)

    # Assert
    assert store.rows["kid-active"].parse_status == PARSE_STATUS_PROCESSING
    assert result.span_skipped == 1
    assert result.recovered_parse_rows == 0


async def test_run_sweep_stale_span_recovers() -> None:
    # Arrange — knowledge AND span both stale: genuinely stuck.
    stale = _NOW - timedelta(hours=3)
    store = _FakeStore(
        rows=[_row(id="kid-stuck", parse_status=PARSE_STATUS_PROCESSING, updated_at=stale)],
        heartbeats={"kid-stuck": stale},
    )

    # Act
    result = await run_sweep(store=store, document_process_timeout=timedelta(hours=1), now=_NOW)

    # Assert
    assert store.rows["kid-stuck"].parse_status == PARSE_STATUS_FAILED
    assert result.recovered_parse_rows == 1


async def test_run_sweep_no_false_kill_tasks_still_queued() -> None:
    # Arrange — finalizing with stale span but subtasks still queued.
    stale = _NOW - timedelta(hours=3)
    store = _FakeStore(
        rows=[_row(id="kid-backlogged", parse_status=PARSE_STATUS_FINALIZING, updated_at=stale)],
        heartbeats={"kid-backlogged": stale},
    )

    # Act
    result = await run_sweep(
        store=store,
        document_process_timeout=timedelta(hours=1),
        task_inspector=_FakeInspector(queued={"kid-backlogged"}),
        now=_NOW,
    )

    # Assert
    assert store.rows["kid-backlogged"].parse_status == PARSE_STATUS_FINALIZING
    assert result.queue_skipped == 1
    assert result.recovered_parse_rows == 0


async def test_run_sweep_queue_probe_error_fails_safe() -> None:
    # Arrange — a failing probe must still recover the row, never strand it.
    stale = _NOW - timedelta(hours=3)
    store = _FakeStore(
        rows=[_row(id="kid-probeerr", parse_status=PARSE_STATUS_PROCESSING, updated_at=stale)]
    )

    # Act
    result = await run_sweep(
        store=store,
        document_process_timeout=timedelta(hours=1),
        task_inspector=_FakeInspector(error=RuntimeError("redis unavailable")),
        now=_NOW,
    )

    # Assert
    assert store.rows["kid-probeerr"].parse_status == PARSE_STATUS_FAILED
    assert result.recovered_parse_rows == 1


async def test_run_sweep_span_query_error_fails_safe() -> None:
    # Arrange — a heartbeat query failure keeps every candidate as stuck.
    stale = _NOW - timedelta(hours=3)
    store = _FakeStore(
        rows=[_row(id="kid-spanerr", parse_status=PARSE_STATUS_PROCESSING, updated_at=stale)],
        span_error=RuntimeError("spans table unavailable"),
    )

    # Act
    result = await run_sweep(store=store, document_process_timeout=timedelta(hours=1), now=_NOW)

    # Assert
    assert store.rows["kid-spanerr"].parse_status == PARSE_STATUS_FAILED
    assert result.recovered_parse_rows == 1


async def test_run_sweep_preserves_recently_touched() -> None:
    # Arrange — a row updated within the cutoff is never a candidate.
    store = _FakeStore(
        rows=[
            _row(
                id="kid-fresh",
                parse_status=PARSE_STATUS_PROCESSING,
                updated_at=_NOW - timedelta(seconds=30),
            )
        ]
    )

    # Act
    result = await run_sweep(store=store, document_process_timeout=timedelta(hours=1), now=_NOW)

    # Assert
    assert store.rows["kid-fresh"].parse_status == PARSE_STATUS_PROCESSING
    assert result.parse_candidates == 0


async def test_run_sweep_candidate_query_error_skips_sweep() -> None:
    # Arrange
    class _BrokenStore(_FakeStore):
        async def find_parse_candidates(self, cutoff: datetime) -> list[Document]:
            raise RuntimeError("candidate query failed")

    # Act — must not propagate; a failed candidate query aborts the pass.
    result = await run_sweep(
        store=_BrokenStore(), document_process_timeout=timedelta(hours=1), now=_NOW
    )

    # Assert — neither sweep runs.
    assert result.parse_candidates == 0
    assert result.recovered_parse_rows == 0
    assert result.recovered_summary_rows == 0


# ── Sweep B: summary recovery ─────────────────────────────────────────


async def test_run_sweep_recovers_stale_summary() -> None:
    # Arrange — summary processing past the one-hour cutoff is recovered.
    stale = _NOW - timedelta(hours=3)
    store = _FakeStore(
        rows=[
            _row(
                id="kid-summary",
                parse_status=PARSE_STATUS_COMPLETED,
                summary_status=SUMMARY_STATUS_PROCESSING,
                updated_at=stale,
            )
        ]
    )

    # Act
    result = await run_sweep(store=store, document_process_timeout=timedelta(hours=1), now=_NOW)

    # Assert
    assert store.rows["kid-summary"].summary_status == SUMMARY_STATUS_FAILED
    assert result.recovered_summary_rows == 1


async def test_run_sweep_preserves_fresh_summary() -> None:
    # Arrange — a recent summary in flight is left alone.
    store = _FakeStore(
        rows=[
            _row(
                id="kid-summary-fresh",
                parse_status=PARSE_STATUS_COMPLETED,
                summary_status=SUMMARY_STATUS_PROCESSING,
                updated_at=_NOW - timedelta(minutes=30),
            )
        ]
    )

    # Act
    result = await run_sweep(store=store, document_process_timeout=timedelta(hours=1), now=_NOW)

    # Assert
    assert store.rows["kid-summary-fresh"].summary_status == SUMMARY_STATUS_PROCESSING
    assert result.recovered_summary_rows == 0


async def test_run_sweep_summary_error_fails_safe_and_reports_zero() -> None:
    # Arrange
    class _BrokenStore(_FakeStore):
        async def mark_summary_rows_failed(self, cutoff: datetime) -> int:
            raise RuntimeError("summary update failed")

    # Act — the summary failure is logged, not raised.
    result = await run_sweep(
        store=_BrokenStore(), document_process_timeout=timedelta(hours=1), now=_NOW
    )

    # Assert
    assert result.recovered_summary_rows == 0


async def test_run_sweep_reports_result_summary_counts() -> None:
    # Arrange — one stale row, one fresh row.
    stale = _NOW - timedelta(hours=3)
    store = _FakeStore(
        rows=[
            _row(id="kid-1", parse_status=PARSE_STATUS_PROCESSING, updated_at=stale),
            _row(id="kid-2", parse_status=PARSE_STATUS_COMPLETED, updated_at=_NOW),
        ]
    )

    # Act
    result = await run_sweep(store=store, document_process_timeout=timedelta(hours=1), now=_NOW)

    # Assert
    assert isinstance(result, HousekeepingResult)
    assert result.parse_candidates == 1
    assert result.recovered_parse_rows == 1
    assert result.span_skipped == 0
    assert result.queue_skipped == 0
    assert result.recovered_summary_rows == 0


# ── Scheduler ─────────────────────────────────────────────────────────


def test_sweep_scheduler_rejects_non_positive_interval() -> None:
    with pytest.raises(ValidationError):
        SweepScheduler(
            session_factory=async_sessionmaker(),
            document_process_timeout=timedelta(hours=1),
            interval=timedelta(0),
        )


def test_sweep_scheduler_defaults_to_five_minute_interval() -> None:
    scheduler = SweepScheduler(
        session_factory=async_sessionmaker(),
        document_process_timeout=timedelta(hours=1),
    )
    assert scheduler._interval == DEFAULT_SWEEP_INTERVAL


async def test_sweep_scheduler_start_honours_disabled_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOUSEKEEPING_ENABLED", "false")
    scheduler = SweepScheduler(
        session_factory=async_sessionmaker(),
        document_process_timeout=timedelta(hours=1),
    )

    await scheduler.start()

    assert scheduler._started is False
    assert scheduler._task is None

    await scheduler.stop()  # no-op
    assert scheduler._started is False


async def test_sweep_scheduler_start_runs_pass_and_stop_cancels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran: list[HousekeepingResult] = []

    async def _run_once(self: SweepScheduler) -> None:
        ran.append(HousekeepingResult(0, 0, 0, 0, 0))

    monkeypatch.setattr(SweepScheduler, "_run_once", _run_once)
    scheduler = SweepScheduler(
        session_factory=async_sessionmaker(),
        document_process_timeout=timedelta(hours=1),
        interval=timedelta(milliseconds=5),
    )

    await scheduler.start()
    assert scheduler._started is True
    for _ in range(50):
        await asyncio.sleep(0.005)
        if ran:
            break

    await scheduler.stop()

    assert scheduler._started is False
    assert scheduler._task is None
    assert len(ran) >= 1


# ── Integration (real applied schema) ─────────────────────────────────
#
# The sweep is global (system-wide), so a run also sees stale rows that
# other test suites have committed. Every test therefore creates a
# session-local ``knowledge_processing_spans`` temp table first — the
# real table ships with the span-tracking feature — so Sweep A's
# heartbeat query always resolves and never aborts the transaction.
# Assertions are row-scoped: each test checks the outcome of the rows it
# seeded rather than global counts, which would depend on unrelated
# committed rows.


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Per-test session against the real applied schema (no cleanup)."""
    reset_settings_cache()
    settings = get_settings()
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as s:
            yield s
    finally:
        await engine.dispose()


async def _seed_document(
    session: AsyncSession,
    *,
    id: str,
    tenant_id: int,
    knowledge_base_id: str,
    parse_status: str,
    updated_at: datetime,
    summary_status: str = SUMMARY_STATUS_NONE,
) -> None:
    """Insert a document row with an explicit ``updated_at`` timestamp.

    The ``updated_at`` column has no on-update trigger, so the raw value
    is preserved — this is what lets the sweep see a deliberately stale
    row.
    """
    stmt = text(
        "insert into documents "
        "(id, tenant_id, knowledge_base_id, type, title, source, parse_status, "
        " summary_status, updated_at) "
        "values (:id, :tenant_id, :kb_id, 'file', :title, :source, :parse_status, "
        ":summary_status, :updated_at)"
    ).bindparams(
        id=id,
        tenant_id=tenant_id,
        kb_id=knowledge_base_id,
        title="housekeeping-fixture",
        source="housekeeping-fixture.pdf",
        parse_status=parse_status,
        summary_status=summary_status,
        updated_at=updated_at,
    )
    await session.execute(stmt)


async def _read_status(session: AsyncSession, id: str) -> tuple[str, str, str | None]:
    """Read parse_status, summary_status, error_message for a document row."""
    stmt = text(
        "select parse_status, summary_status, error_message from documents where id = :id"
    ).bindparams(id=id)
    mapping = (await session.execute(stmt)).mappings().first()
    assert mapping is not None, f"document {id} not found"
    return str(mapping["parse_status"]), str(mapping["summary_status"]), mapping["error_message"]


async def _ensure_spans_table(session: AsyncSession) -> None:
    """Create a session-local spans table so the heartbeat query resolves.

    The real span table is introduced by the span-tracking feature; the
    temp table shadows the contract name for this session only and drops
    automatically at session close.
    """
    await session.execute(
        text(
            "create temporary table if not exists knowledge_processing_spans ("
            " id bigserial primary key,"
            " knowledge_id varchar(64) not null,"
            " span_id varchar(64) not null,"
            " name varchar(255) not null,"
            " kind varchar(16) not null,"
            " status varchar(16) not null,"
            " updated_at timestamptz not null default now()"
            ")"
        )
    )


async def _seed_span(session: AsyncSession, knowledge_id: str, updated_at: datetime) -> None:
    stmt = text(
        "insert into knowledge_processing_spans "
        "(knowledge_id, span_id, name, kind, status, updated_at) "
        "values (:knowledge_id, :span_id, 'docreader', 'stage', 'running', :updated_at)"
    ).bindparams(
        knowledge_id=knowledge_id,
        span_id=f"span-{uuid.uuid4().hex[:8]}",
        updated_at=updated_at,
    )
    await session.execute(stmt)


async def test_integration_recovers_abandoned_row(session: AsyncSession) -> None:
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    id = _did()
    now = datetime.now(UTC)
    await _ensure_spans_table(session)
    await _seed_document(
        session,
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        parse_status=PARSE_STATUS_PROCESSING,
        updated_at=now - timedelta(hours=3),
    )

    store = SqlKnowledgeSweepStore(session)
    await run_sweep(
        store=store,
        document_process_timeout=timedelta(hours=1),
        now=now,
    )

    parse_status, _summary_status, error_message = await _read_status(session, id)
    assert parse_status == PARSE_STATUS_FAILED
    assert "stuck in processing" in (error_message or "")


async def test_integration_preserves_recent_row(session: AsyncSession) -> None:
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    id = _did()
    now = datetime.now(UTC)
    await _ensure_spans_table(session)
    await _seed_document(
        session,
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        parse_status=PARSE_STATUS_PROCESSING,
        updated_at=now - timedelta(seconds=30),
    )

    store = SqlKnowledgeSweepStore(session)
    await run_sweep(
        store=store,
        document_process_timeout=timedelta(hours=1),
        now=now,
    )

    parse_status, _summary_status, _error_message = await _read_status(session, id)
    assert parse_status == PARSE_STATUS_PROCESSING


async def test_integration_active_span_protects_row(session: AsyncSession) -> None:
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    id = _did()
    now = datetime.now(UTC)
    await _ensure_spans_table(session)
    await _seed_document(
        session,
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        parse_status=PARSE_STATUS_PROCESSING,
        updated_at=now - timedelta(hours=3),
    )
    # A fresh span heartbeat — the row is still working, must not be killed.
    await _seed_span(session, knowledge_id=id, updated_at=now - timedelta(minutes=2))

    store = SqlKnowledgeSweepStore(session)
    result = await run_sweep(
        store=store,
        document_process_timeout=timedelta(hours=1),
        now=now,
    )

    parse_status, _summary_status, _error_message = await _read_status(session, id)
    assert parse_status == PARSE_STATUS_PROCESSING
    # Only the seeded row carries a fresh heartbeat; pre-existing stale
    # rows have no span rows at all, so exactly one is skipped.
    assert result.span_skipped == 1


async def test_integration_recovers_stale_summary(session: AsyncSession) -> None:
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    id = _did()
    now = datetime.now(UTC)
    await _ensure_spans_table(session)
    await _seed_document(
        session,
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        parse_status=PARSE_STATUS_COMPLETED,
        summary_status=SUMMARY_STATUS_PROCESSING,
        updated_at=now - timedelta(hours=3),
    )

    store = SqlKnowledgeSweepStore(session)
    await run_sweep(
        store=store,
        document_process_timeout=timedelta(hours=1),
        now=now,
    )

    parse_status, summary_status, _error_message = await _read_status(session, id)
    assert parse_status == PARSE_STATUS_COMPLETED
    assert summary_status == SUMMARY_STATUS_FAILED


async def test_integration_preserves_fresh_summary(session: AsyncSession) -> None:
    tenant_id = make_test_tenant_id()
    kb_id = _kbid()
    id = _did()
    now = datetime.now(UTC)
    await _ensure_spans_table(session)
    await _seed_document(
        session,
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        parse_status=PARSE_STATUS_COMPLETED,
        summary_status=SUMMARY_STATUS_PROCESSING,
        updated_at=now - timedelta(minutes=30),
    )

    store = SqlKnowledgeSweepStore(session)
    await run_sweep(
        store=store,
        document_process_timeout=timedelta(hours=1),
        now=now,
    )

    _parse_status, summary_status, _error_message = await _read_status(session, id)
    assert summary_status == SUMMARY_STATUS_PROCESSING
