"""Unit + integration tests for the processing-span tracker.

Unit tests drive ``SpanTracker`` with a stateful in-memory store
(closure-style fake implementing the tracker's ``SpanStore`` protocol,
the same pattern used across the core service tests): they cover the
span vocabulary, state transitions, cascade-cancel rules, validation,
and error classification.

Integration tests run against the real applied ``knowledge_processing_spans``
schema using the tenant-id factory from the integration conftest. They
require a reachable database — run with ``DATABASE_URL_OVERRIDE``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from random import randint

import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.common.exception import NotFoundError, ValidationError
from src.core.knowledge.documents.span_tracker import (
    ALL_STAGES,
    ROOT_SPAN_NAME,
    SPAN_KIND_GENERATION,
    SPAN_KIND_ROOT,
    SPAN_KIND_STAGE,
    SPAN_KIND_SUBSPAN,
    SPAN_STATUS_CANCELLED,
    SPAN_STATUS_DONE,
    SPAN_STATUS_FAILED,
    SPAN_STATUS_RUNNING,
    SPAN_STATUS_SKIPPED,
    STAGE_CHUNKING,
    STAGE_DOCREADER,
    STAGE_EMBEDDING,
    STAGE_MULTIMODAL,
    STAGE_POSTPROCESS,
    SpanProgress,
    SpanTracker,
    fit_span_name,
    is_main_pipeline_stage,
    stages_depending_on,
)
from src.db.dao.knowledge_span_repository import KnowledgeSpanRepository
from src.db.models.knowledge_processing_span import KnowledgeProcessingSpan
from src.settings import get_settings, reset_settings_cache
from tests.integration.conftest import make_test_tenant_id

_FAKER_SEED_MAX = 100_000_000


@pytest.fixture(autouse=True)
def faker_seed() -> None:
    """Re-seed Faker per test for varied-but-reproducible generation."""
    Faker.seed(randint(1, _FAKER_SEED_MAX))


# ── In-memory span store (mock repository) ─────────────────────────────


class _FakeHeartbeat:
    """Records heartbeat touches so tests can assert the side-channel."""

    def __init__(self) -> None:
        self.touched: list[str] = []

    async def touch(self, *, knowledge_id: str) -> None:
        self.touched.append(knowledge_id)


class _FakeSpanStore:
    """Stateful in-memory store mirroring ``KnowledgeSpanRepository``.

    Implements the tracker's ``SpanStore`` protocol with the same upsert
    (conflict-update) and cancel-sweep semantics, so unit tests exercise
    the tracker logic against realistic row behavior.
    """

    def __init__(self) -> None:
        self._rows: dict[tuple[str, int, str], KnowledgeProcessingSpan] = {}
        self._next_id = 1

    # ── Protocol surface ────────────────────────────────────────────

    async def upsert(self, row: KnowledgeProcessingSpan) -> KnowledgeProcessingSpan:
        attempt = row.attempt if row.attempt > 0 else 1
        key = (row.knowledge_id, attempt, row.span_id)
        existing = self._rows.get(key)
        if existing is None:
            stored = row.model_copy(update={"id": self._next_id, "attempt": attempt})
            self._next_id += 1
        else:
            updates: dict[str, object] = {
                "status": row.status,
                "error_code": row.error_code,
                "error_message": row.error_message,
                "error_detail": row.error_detail,
                "started_at": row.started_at,
                "finished_at": row.finished_at,
                "duration_ms": row.duration_ms,
                "updated_at": row.updated_at,
                "attempt": attempt,
            }
            for col in ("input", "output", "metadata"):
                value = getattr(row, col)
                if value is not None:
                    updates[col] = value
            stored = existing.model_copy(update=updates)
        self._rows[key] = stored
        return stored

    async def next_attempt(self, knowledge_id: str) -> int:
        return (await self.latest_attempt(knowledge_id)) + 1

    async def latest_attempt(self, knowledge_id: str) -> int:
        attempts = [key[1] for key in self._rows if key[0] == knowledge_id]
        return max(attempts) if attempts else 0

    async def list_by_attempt(
        self,
        knowledge_id: str,
        attempt: int,
    ) -> list[KnowledgeProcessingSpan]:
        rows = [
            row
            for key, row in self._rows.items()
            if key[0] == knowledge_id and (attempt <= 0 or key[1] == attempt)
        ]
        rows.sort(key=lambda r: r.id)
        return rows

    async def cancel_descendants(
        self,
        knowledge_id: str,
        attempt: int,
        parent_span_id: str,
        reason: str,
    ) -> int:
        frontier = [parent_span_id]
        total = 0
        for _ in range(16):
            children = [
                key[2]
                for key, row in self._rows.items()
                if (
                    key[0] == knowledge_id
                    and key[1] == attempt
                    and row.parent_span_id in frontier
                    and row.status in (SPAN_STATUS_RUNNING, "pending")
                )
            ]
            if not children:
                break
            for span_id in children:
                key = (knowledge_id, attempt, span_id)
                row = self._rows[key]
                self._rows[key] = row.model_copy(
                    update={
                        "status": SPAN_STATUS_CANCELLED,
                        "error_code": "UPSTREAM_FAILED",
                        "error_message": reason,
                    }
                )
                total += 1
            frontier = children
        return total

    async def cancel_all_open_spans(
        self,
        knowledge_id: str,
        attempt: int,
        error_code: str,
        reason: str,
        *,
        now: datetime,
    ) -> int:
        total = 0
        for key, row in list(self._rows.items()):
            if (
                key[0] == knowledge_id
                and key[1] == attempt
                and row.status in (SPAN_STATUS_RUNNING, "pending")
            ):
                self._rows[key] = row.model_copy(
                    update={
                        "status": SPAN_STATUS_CANCELLED,
                        "error_code": error_code,
                        "error_message": reason,
                        "finished_at": now,
                        "updated_at": now,
                    }
                )
                total += 1
        return total

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
        if knowledge_id == "" or attempt <= 0 or name == "":
            return 0
        total = 0
        for key, row in list(self._rows.items()):
            if (
                key[0] == knowledge_id
                and key[1] == attempt
                and row.name == name
                and row.status in (SPAN_STATUS_RUNNING, "pending")
            ):
                self._rows[key] = row.model_copy(
                    update={
                        "status": SPAN_STATUS_CANCELLED,
                        "error_code": error_code,
                        "error_message": reason,
                        "finished_at": now,
                        "updated_at": now,
                    }
                )
                total += 1
        return total

    # ── Test helpers ────────────────────────────────────────────────

    def rows(self, knowledge_id: str, attempt: int) -> list[KnowledgeProcessingSpan]:
        candidates = [
            row
            for key, row in self._rows.items()
            if key[0] == knowledge_id and (attempt <= 0 or key[1] == attempt)
        ]
        candidates.sort(key=lambda r: r.id)
        return candidates

    def row(self, knowledge_id: str, attempt: int, span_id: str) -> KnowledgeProcessingSpan:
        return self._rows[(knowledge_id, attempt, span_id)]

    def root(self, knowledge_id: str, attempt: int) -> KnowledgeProcessingSpan:
        return next(
            r for r in self.rows(knowledge_id, attempt) if r.kind == SPAN_KIND_ROOT
        )


def _make_tracker() -> tuple[SpanTracker, _FakeSpanStore, _FakeHeartbeat]:
    """Build a tracker over the in-memory store + a recording heartbeat."""
    store = _FakeSpanStore()
    heartbeat = _FakeHeartbeat()
    tracker = SpanTracker(span_store=store, heartbeat=heartbeat)
    return tracker, store, heartbeat


# ── Vocabulary helpers ────────────────────────────────────────────────


def test_fit_span_name_preserves_short_names() -> None:
    assert fit_span_name("multimodal.image[0]") == "multimodal.image[0]"


def test_fit_span_name_truncates_with_hash_suffix() -> None:
    name = "postprocess.wiki.page[%s]" % ("e" * 300)
    fitted = fit_span_name(name)
    assert len(fitted) == 255
    assert fitted.startswith("postprocess.wiki.page[")
    assert fitted[-9] == "~"
    assert len(fitted[-8:]) == 8
    int(fitted[-8:], 16)  # hash suffix is 8 hex chars


def test_stages_depending_on_closure() -> None:
    assert stages_depending_on(STAGE_DOCREADER) == [
        STAGE_CHUNKING,
        STAGE_EMBEDDING,
        STAGE_MULTIMODAL,
        STAGE_POSTPROCESS,
    ]
    assert stages_depending_on(STAGE_CHUNKING) == [
        STAGE_EMBEDDING,
        STAGE_MULTIMODAL,
        STAGE_POSTPROCESS,
    ]
    assert stages_depending_on(STAGE_EMBEDDING) == [STAGE_POSTPROCESS]
    assert stages_depending_on(STAGE_POSTPROCESS) == []


def test_is_main_pipeline_stage() -> None:
    assert is_main_pipeline_stage(STAGE_CHUNKING)
    assert is_main_pipeline_stage(STAGE_MULTIMODAL)
    assert not is_main_pipeline_stage("summary")
    assert not is_main_pipeline_stage("question")


# ── Attempt lifecycle (unit) ──────────────────────────────────────────


async def test_open_attempt_creates_root() -> None:
    tracker, store, _ = _make_tracker()
    root, attempt = await tracker.open_attempt("kid-1", "trace-abc")
    assert attempt == 1
    assert root is not None
    assert root.kind == SPAN_KIND_ROOT
    assert root.name == ROOT_SPAN_NAME
    assert root.status == SPAN_STATUS_RUNNING
    assert root.parent_span_id is None
    persisted = store.root("kid-1", 1)
    assert persisted.span_id == root.span_id
    assert persisted.metadata == {"langfuse_trace_id": "trace-abc"}


async def test_open_attempt_omits_langfuse_metadata_when_absent() -> None:
    tracker, store, _ = _make_tracker()
    root, _ = await tracker.open_attempt("kid-1")
    assert root is not None
    assert store.row("kid-1", 1, root.span_id).metadata == {}


async def test_open_attempt_empty_knowledge_id_is_nil_safe() -> None:
    tracker, _, _ = _make_tracker()
    root, attempt = await tracker.open_attempt("")
    assert root is None
    assert attempt == 0


async def test_attempt_numbering_increments() -> None:
    tracker, _, _ = _make_tracker()
    _, first = await tracker.open_attempt("kid-1")
    _, second = await tracker.open_attempt("kid-1")
    assert (first, second) == (1, 2)


async def test_latest_attempt_returns_zero_when_never_parsed() -> None:
    tracker, _, _ = _make_tracker()
    assert await tracker.latest_attempt("kid-missing") == 0


async def test_finalize_attempt_closes_root_as_done() -> None:
    tracker, store, _ = _make_tracker()
    root, attempt = await tracker.open_attempt("kid-1")
    assert root is not None
    await tracker.finalize_attempt(knowledge_id="kid-1", attempt=attempt)
    assert store.root("kid-1", attempt).status == SPAN_STATUS_DONE


async def test_finalize_attempt_is_idempotent() -> None:
    tracker, store, _ = _make_tracker()
    root, attempt = await tracker.open_attempt("kid-1")
    assert root is not None
    await tracker.finalize_attempt(knowledge_id="kid-1", attempt=attempt)
    before = store.root("kid-1", attempt)
    await tracker.finalize_attempt(knowledge_id="kid-1", attempt=attempt)
    after = store.root("kid-1", attempt)
    assert after.status == SPAN_STATUS_DONE
    assert after.finished_at == before.finished_at


async def test_finalize_attempt_invalid_status_raises() -> None:
    tracker, _, _ = _make_tracker()
    await tracker.open_attempt("kid-1")
    with pytest.raises(ValidationError) as exc_info:
        await tracker.finalize_attempt(knowledge_id="kid-1", attempt=1, status="banana")
    assert exc_info.value.code == "span.invalid_status"


async def test_finalize_attempt_no_root_is_noop() -> None:
    tracker, _, _ = _make_tracker()
    # Nothing tracked for the knowledge — finalize must not raise.
    await tracker.finalize_attempt(knowledge_id="kid-missing", attempt=1)
    await tracker.finalize_attempt(knowledge_id="", attempt=1)
    await tracker.finalize_attempt(knowledge_id="kid-missing", attempt=0)


async def test_abort_attempt_sweeps_and_closes_root() -> None:
    tracker, store, _ = _make_tracker()
    root, attempt = await tracker.open_attempt("kid-1")
    assert root is not None
    stage = await tracker.begin_stage(
        knowledge_id="kid-1", attempt=attempt, stage=STAGE_CHUNKING
    )
    assert stage is not None
    await tracker.begin_sub_span(parent=stage, name="chunk[0]")
    await tracker.abort_attempt(knowledge_id="kid-1", attempt=attempt)
    rows = store.rows("kid-1", attempt)
    assert all(r.status == SPAN_STATUS_CANCELLED for r in rows)
    cancelled = store.root("kid-1", attempt)
    assert cancelled.error_code == "USER_CANCELLED"
    assert cancelled.error_message == "user cancelled"


# ── Stage / subspan lifecycle (unit) ──────────────────────────────────


async def test_begin_stage_attaches_to_root() -> None:
    tracker, store, _ = _make_tracker()
    root, attempt = await tracker.open_attempt("kid-1")
    assert root is not None
    stage = await tracker.begin_stage(
        knowledge_id="kid-1", attempt=attempt, stage=STAGE_CHUNKING, input={"n": 3}
    )
    assert stage is not None
    assert stage.kind == SPAN_KIND_STAGE
    assert stage.parent_span_id == root.span_id
    assert stage.status == SPAN_STATUS_RUNNING
    persisted = store.row("kid-1", attempt, stage.span_id)
    assert persisted.input == {"n": 3}


async def test_begin_stage_empty_inputs_are_nil_safe() -> None:
    tracker, _, _ = _make_tracker()
    assert (
        await tracker.begin_stage(knowledge_id="", attempt=1, stage=STAGE_CHUNKING) is None
    )
    assert (
        await tracker.begin_stage(knowledge_id="kid-1", attempt=1, stage="") is None
    )


async def test_begin_stage_reentry_reuses_span_id() -> None:
    tracker, store, _ = _make_tracker()
    root, attempt = await tracker.open_attempt("kid-1")
    assert root is not None
    first = await tracker.begin_stage(
        knowledge_id="kid-1", attempt=attempt, stage=STAGE_CHUNKING
    )
    assert first is not None
    await tracker.end_span(span=first, output={"chunks": 12})
    second = await tracker.begin_stage(
        knowledge_id="kid-1", attempt=attempt, stage=STAGE_CHUNKING
    )
    assert second is not None
    assert second.span_id == first.span_id
    # Exactly one stage row for the name, reset to running.
    stages = [
        r
        for r in store.rows("kid-1", attempt)
        if r.kind == SPAN_KIND_STAGE and r.name == STAGE_CHUNKING
    ]
    assert len(stages) == 1
    assert stages[0].status == SPAN_STATUS_RUNNING


async def test_begin_stage_rootless_records_rootless() -> None:
    tracker, store, _ = _make_tracker()
    stage = await tracker.begin_stage(
        knowledge_id="kid-1", attempt=1, stage=STAGE_DOCREADER
    )
    assert stage is not None
    assert stage.parent_span_id == ""
    persisted = store.row("kid-1", 1, stage.span_id)
    assert persisted.kind == SPAN_KIND_STAGE
    assert persisted.parent_span_id == ""


async def test_begin_sub_span_kind_defaults_to_subspan() -> None:
    tracker, _, _ = _make_tracker()
    root, _attempt = await tracker.open_attempt("kid-1")
    assert root is not None
    sub = await tracker.begin_sub_span(parent=root, name="image[0]")
    assert sub is not None
    assert sub.kind == SPAN_KIND_SUBSPAN
    assert sub.parent_span_id == root.span_id


async def test_begin_sub_span_generation_kind_is_preserved() -> None:
    tracker, store, _ = _make_tracker()
    root, attempt = await tracker.open_attempt("kid-1")
    assert root is not None
    sub = await tracker.begin_sub_span(
        parent=root, name="llm.summary", kind=SPAN_KIND_GENERATION
    )
    assert sub is not None
    assert sub.kind == SPAN_KIND_GENERATION
    assert store.row("kid-1", attempt, sub.span_id).kind == SPAN_KIND_GENERATION


async def test_begin_sub_span_invalid_kind_falls_back_to_subspan() -> None:
    tracker, _, _ = _make_tracker()
    root, _ = await tracker.open_attempt("kid-1")
    assert root is not None
    sub = await tracker.begin_sub_span(parent=root, name="x", kind="wat")
    assert sub is not None
    assert sub.kind == SPAN_KIND_SUBSPAN


async def test_begin_sub_span_nil_parent_is_nil_safe() -> None:
    tracker, _, _ = _make_tracker()
    assert await tracker.begin_sub_span(parent=None, name="x") is None


async def test_begin_sub_span_supersedes_open_same_name() -> None:
    tracker, store, _ = _make_tracker()
    root, attempt = await tracker.open_attempt("kid-1")
    assert root is not None
    first = await tracker.begin_sub_span(parent=root, name="chunk[0]")
    assert first is not None
    # A retry re-runs the same subtask while the first row is still running.
    second = await tracker.begin_sub_span(parent=root, name="chunk[0]")
    assert second is not None
    assert second.span_id != first.span_id
    previous = store.row("kid-1", attempt, first.span_id)
    assert previous.status == SPAN_STATUS_CANCELLED
    assert previous.error_code == "TASK_SUPERSEDED"


async def test_lookup_stage_and_lookup_span_by_name() -> None:
    tracker, _, _ = _make_tracker()
    root, attempt = await tracker.open_attempt("kid-1")
    assert root is not None
    stage = await tracker.begin_stage(
        knowledge_id="kid-1", attempt=attempt, stage=STAGE_CHUNKING
    )
    assert stage is not None
    found = await tracker.lookup_stage(
        knowledge_id="kid-1", attempt=attempt, stage=STAGE_CHUNKING
    )
    assert found is not None
    assert found.span_id == stage.span_id
    assert (
        await tracker.lookup_stage(
            knowledge_id="kid-1", attempt=attempt, stage=STAGE_EMBEDDING
        )
        is None
    )
    by_name = await tracker.lookup_span_by_name(
        knowledge_id="kid-1", attempt=attempt, name=ROOT_SPAN_NAME
    )
    assert by_name is not None
    assert by_name.kind == SPAN_KIND_ROOT
    assert (
        await tracker.lookup_span_by_name(
            knowledge_id="kid-1", attempt=attempt, name=""
        )
        is None
    )
    assert (
        await tracker.lookup_span_by_name(knowledge_id="", attempt=1, name="x") is None
    )
    assert (
        await tracker.lookup_span_by_name(knowledge_id="kid-1", attempt=0, name="x")
        is None
    )


# ── Span transitions (unit) ───────────────────────────────────────────


async def test_end_span_marks_done_with_output_and_duration() -> None:
    tracker, store, _ = _make_tracker()
    root, attempt = await tracker.open_attempt("kid-1")
    assert root is not None
    stage = await tracker.begin_stage(
        knowledge_id="kid-1", attempt=attempt, stage=STAGE_CHUNKING
    )
    assert stage is not None
    await tracker.end_span(span=stage, output={"chunks": 12})
    persisted = store.row("kid-1", attempt, stage.span_id)
    assert persisted.status == SPAN_STATUS_DONE
    assert persisted.output == {"chunks": 12}
    # end_span does not clobber the input written by begin_stage.
    assert persisted.input is None
    assert persisted.finished_at is not None
    assert persisted.duration_ms is not None and persisted.duration_ms >= 0


async def test_end_span_nil_span_is_noop() -> None:
    tracker, _, _ = _make_tracker()
    await tracker.end_span(span=None)


async def test_fail_span_cancels_descendants_with_reason() -> None:
    tracker, store, _ = _make_tracker()
    root, attempt = await tracker.open_attempt("kid-1")
    assert root is not None
    stage = await tracker.begin_stage(
        knowledge_id="kid-1", attempt=attempt, stage=STAGE_CHUNKING
    )
    assert stage is not None
    child = await tracker.begin_sub_span(parent=stage, name="chunk[0]")
    assert child is not None
    await tracker.fail_span(span=stage, error_code="EMBEDDING_RATE_LIMIT")
    failed = store.row("kid-1", attempt, stage.span_id)
    assert failed.status == SPAN_STATUS_FAILED
    assert failed.error_code == "EMBEDDING_RATE_LIMIT"
    child_row = store.row("kid-1", attempt, child.span_id)
    assert child_row.status == SPAN_STATUS_CANCELLED
    assert child_row.error_code == "UPSTREAM_FAILED"
    assert child_row.error_message == (
        f"upstream {STAGE_CHUNKING} failed (EMBEDDING_RATE_LIMIT)"
    )


async def test_fail_span_truncates_error_fields() -> None:
    tracker, store, _ = _make_tracker()
    root, attempt = await tracker.open_attempt("kid-1")
    assert root is not None
    stage = await tracker.begin_stage(
        knowledge_id="kid-1", attempt=attempt, stage=STAGE_CHUNKING
    )
    assert stage is not None
    await tracker.fail_span(
        span=stage,
        error_message="e" * 5000,
        error_detail="d" * 10000,
    )
    persisted = store.row("kid-1", attempt, stage.span_id)
    assert len(persisted.error_message or "") == 1024
    assert len(persisted.error_detail or "") == 8192


async def test_fail_span_main_stage_finalizes_root_failed() -> None:
    tracker, store, _ = _make_tracker()
    root, attempt = await tracker.open_attempt("kid-1")
    assert root is not None
    stage = await tracker.begin_stage(
        knowledge_id="kid-1", attempt=attempt, stage=STAGE_CHUNKING
    )
    assert stage is not None
    await tracker.fail_span(span=stage, error_code="CHUNK_FAILED")
    assert store.root("kid-1", attempt).status == SPAN_STATUS_FAILED
    assert store.root("kid-1", attempt).error_code == "CHUNK_FAILED"


async def test_fail_span_subspan_does_not_poison_attempt() -> None:
    tracker, store, _ = _make_tracker()
    root, attempt = await tracker.open_attempt("kid-1")
    assert root is not None
    sub = await tracker.begin_sub_span(parent=root, name="image[0]")
    assert sub is not None
    await tracker.fail_span(span=sub)
    assert store.root("kid-1", attempt).status == SPAN_STATUS_RUNNING


async def test_fail_span_cascades_dependent_stages() -> None:
    tracker, store, _ = _make_tracker()
    root, attempt = await tracker.open_attempt("kid-1")
    assert root is not None
    docreader = await tracker.begin_stage(
        knowledge_id="kid-1", attempt=attempt, stage=STAGE_DOCREADER
    )
    assert docreader is not None
    await tracker.end_span(span=docreader)
    chunking = await tracker.begin_stage(
        knowledge_id="kid-1", attempt=attempt, stage=STAGE_CHUNKING
    )
    assert chunking is not None
    embedding = await tracker.begin_stage(
        knowledge_id="kid-1", attempt=attempt, stage=STAGE_EMBEDDING
    )
    multimodal = await tracker.begin_stage(
        knowledge_id="kid-1", attempt=attempt, stage=STAGE_MULTIMODAL
    )
    postprocess = await tracker.begin_stage(
        knowledge_id="kid-1", attempt=attempt, stage=STAGE_POSTPROCESS
    )
    assert embedding is not None and multimodal is not None and postprocess is not None
    # A subspan already attached to embedding must be cancelled too.
    batch = await tracker.begin_sub_span(parent=embedding, name="batch[0]")
    assert batch is not None

    await tracker.fail_span(span=chunking, error_code="CHUNK_FAILED")

    rows = {r.name: r for r in store.rows("kid-1", attempt)}
    assert rows[STAGE_CHUNKING].status == SPAN_STATUS_FAILED
    assert rows[STAGE_EMBEDDING].status == SPAN_STATUS_CANCELLED
    assert rows[STAGE_MULTIMODAL].status == SPAN_STATUS_CANCELLED
    assert rows[STAGE_POSTPROCESS].status == SPAN_STATUS_CANCELLED
    # The embedding subspan is cancelled via the dependent-stage walk.
    batch_row = store.row("kid-1", attempt, batch.span_id)
    assert batch_row.status == SPAN_STATUS_CANCELLED
    assert batch_row.error_code == "UPSTREAM_FAILED"
    # Root closed failed because chunking is a main pipeline stage.
    assert store.root("kid-1", attempt).status == SPAN_STATUS_FAILED


async def test_skip_span_marks_intentionally_not_run() -> None:
    tracker, store, _ = _make_tracker()
    root, attempt = await tracker.open_attempt("kid-1")
    assert root is not None
    stage = await tracker.begin_stage(
        knowledge_id="kid-1", attempt=attempt, stage=STAGE_MULTIMODAL
    )
    assert stage is not None
    await tracker.skip_span(span=stage, reason="text-only document")
    persisted = store.row("kid-1", attempt, stage.span_id)
    assert persisted.status == SPAN_STATUS_SKIPPED
    assert persisted.error_message == "text-only document"
    assert persisted.finished_at is not None


# ── Heartbeat side-channel (unit) ─────────────────────────────────────


async def test_heartbeat_fires_for_root_and_stage_only() -> None:
    tracker, _, heartbeat = _make_tracker()
    root, attempt = await tracker.open_attempt("kid-1")
    assert root is not None
    stage = await tracker.begin_stage(
        knowledge_id="kid-1", attempt=attempt, stage=STAGE_CHUNKING
    )
    assert stage is not None
    sub = await tracker.begin_sub_span(parent=stage, name="chunk[0]")
    assert sub is not None
    await tracker.end_span(span=sub)
    await tracker.end_span(span=stage)
    await tracker.finalize_attempt(knowledge_id="kid-1", attempt=attempt)
    assert heartbeat.touched == ["kid-1", "kid-1", "kid-1", "kid-1"]


async def test_heartbeat_absent_is_skipped() -> None:
    tracker, _, _ = _make_tracker()
    tracker._heartbeat = None
    root, attempt = await tracker.open_attempt("kid-1")
    assert root is not None
    await tracker.begin_stage(
        knowledge_id="kid-1", attempt=attempt, stage=STAGE_CHUNKING
    )
    await tracker.finalize_attempt(knowledge_id="kid-1", attempt=attempt)


# ── Progress query (unit) ─────────────────────────────────────────────


async def test_get_progress_blank_knowledge_raises() -> None:
    tracker, _, _ = _make_tracker()
    with pytest.raises(ValidationError) as exc_info:
        await tracker.get_progress(knowledge_id="")
    assert exc_info.value.code == "span.knowledge_required"


async def test_get_progress_invalid_attempt_raises() -> None:
    tracker, _, _ = _make_tracker()
    with pytest.raises(ValidationError) as exc_info:
        await tracker.get_progress(knowledge_id="kid-1", attempt=0)
    assert exc_info.value.code == "span.invalid_attempt"


async def test_get_progress_untracked_knowledge_raises_not_found() -> None:
    tracker, _, _ = _make_tracker()
    with pytest.raises(NotFoundError) as exc_info:
        await tracker.get_progress(knowledge_id="kid-missing")
    assert exc_info.value.code == "span.not_found"


async def test_get_progress_defaults_to_latest_attempt() -> None:
    tracker, _, _ = _make_tracker()
    root, _ = await tracker.open_attempt("kid-1")
    assert root is not None
    progress = await tracker.get_progress(knowledge_id="kid-1")
    assert isinstance(progress, SpanProgress)
    assert progress.attempt == 1
    assert progress.latest_attempt == 1
    assert [s.kind for s in progress.spans] == [SPAN_KIND_ROOT]


async def test_get_progress_returns_requested_attempt() -> None:
    tracker, _, _ = _make_tracker()
    root, _ = await tracker.open_attempt("kid-1")
    assert root is not None
    progress = await tracker.get_progress(knowledge_id="kid-1", attempt=1)
    assert progress.attempt == 1
    # A requested attempt newer than any recording returns an empty list.
    empty = await tracker.get_progress(knowledge_id="kid-1", attempt=99)
    assert empty.spans == ()


async def test_list_attempt_spans_returns_insertion_order() -> None:
    tracker, _, _ = _make_tracker()
    root, attempt = await tracker.open_attempt("kid-1")
    assert root is not None
    stage = await tracker.begin_stage(
        knowledge_id="kid-1", attempt=attempt, stage=STAGE_DOCREADER
    )
    assert stage is not None
    spans = await tracker.list_attempt_spans("kid-1", attempt)
    assert [s.kind for s in spans] == [SPAN_KIND_ROOT, SPAN_KIND_STAGE]


# ── Integration (real applied schema) ─────────────────────────────────


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


async def _tracker_for(session: AsyncSession) -> SpanTracker:
    return SpanTracker(
        span_store=KnowledgeSpanRepository(session),
    )


def _kid() -> str:
    return f"kid-{make_test_tenant_id()}"


async def test_integration_full_lifecycle(session: AsyncSession) -> None:
    tracker = await _tracker_for(session)
    kid = _kid()
    root, attempt = await tracker.open_attempt(kid, "trace-integration")
    assert root is not None
    for stage_name in ALL_STAGES:
        stage = await tracker.begin_stage(
            knowledge_id=kid, attempt=attempt, stage=stage_name
        )
        assert stage is not None
        await tracker.end_span(span=stage, output={"stage": stage_name})
    image = await tracker.begin_sub_span(
        parent=await tracker.lookup_stage(
            knowledge_id=kid, attempt=attempt, stage=STAGE_MULTIMODAL
        ),
        name="multimodal.image[0]",
        kind=SPAN_KIND_GENERATION,
    )
    assert image is not None
    await tracker.end_span(span=image)
    await tracker.finalize_attempt(knowledge_id=kid, attempt=attempt)

    progress = await tracker.get_progress(knowledge_id=kid)
    assert progress.latest_attempt == 1
    assert progress.attempt == 1
    by_kind = [s.kind for s in progress.spans]
    assert by_kind.count(SPAN_KIND_ROOT) == 1
    assert by_kind.count(SPAN_KIND_STAGE) == 5
    assert by_kind.count(SPAN_KIND_GENERATION) == 1
    root_row = next(s for s in progress.spans if s.kind == SPAN_KIND_ROOT)
    assert root_row.status == SPAN_STATUS_DONE
    assert root_row.metadata == {"langfuse_trace_id": "trace-integration"}
    assert all(s.status == SPAN_STATUS_DONE for s in progress.spans)


async def test_integration_fail_cascades_and_finalizes(session: AsyncSession) -> None:
    tracker = await _tracker_for(session)
    kid = _kid()
    root, attempt = await tracker.open_attempt(kid)
    assert root is not None
    docreader = await tracker.begin_stage(
        knowledge_id=kid, attempt=attempt, stage=STAGE_DOCREADER
    )
    assert docreader is not None
    await tracker.end_span(span=docreader)
    chunking = await tracker.begin_stage(
        knowledge_id=kid, attempt=attempt, stage=STAGE_CHUNKING
    )
    assert chunking is not None
    embedding = await tracker.begin_stage(
        knowledge_id=kid, attempt=attempt, stage=STAGE_EMBEDDING
    )
    multimodal = await tracker.begin_stage(
        knowledge_id=kid, attempt=attempt, stage=STAGE_MULTIMODAL
    )
    assert embedding is not None and multimodal is not None
    await tracker.fail_span(span=chunking, error_code="CHUNK_TIMEOUT")

    rows = {s.name: s for s in await tracker.list_attempt_spans(kid, attempt)}
    assert rows[STAGE_CHUNKING].status == SPAN_STATUS_FAILED
    assert rows[STAGE_EMBEDDING].status == SPAN_STATUS_CANCELLED
    assert rows[STAGE_MULTIMODAL].status == SPAN_STATUS_CANCELLED
    assert rows[ROOT_SPAN_NAME].status == SPAN_STATUS_FAILED
    assert rows[STAGE_DOCREADER].status == SPAN_STATUS_DONE


async def test_integration_abort_sweeps_every_open_span(session: AsyncSession) -> None:
    tracker = await _tracker_for(session)
    kid = _kid()
    root, attempt = await tracker.open_attempt(kid)
    assert root is not None
    stage = await tracker.begin_stage(
        knowledge_id=kid, attempt=attempt, stage=STAGE_CHUNKING
    )
    assert stage is not None
    # Simulate a fan-out that ended its stage row while the child runs.
    await tracker.end_span(span=stage)
    sub = await tracker.begin_sub_span(parent=stage, name="chunk[0]")
    assert sub is not None
    await tracker.abort_attempt(knowledge_id=kid, attempt=attempt)
    rows = {s.name: s for s in await tracker.list_attempt_spans(kid, attempt)}
    # Root and the still-running sub are swept; the already-done stage row
    # keeps its original terminal outcome.
    assert rows[ROOT_SPAN_NAME].status == SPAN_STATUS_CANCELLED
    assert rows[ROOT_SPAN_NAME].error_code == "USER_CANCELLED"
    assert rows[STAGE_CHUNKING].status == SPAN_STATUS_DONE
    assert rows["chunk[0]"].status == SPAN_STATUS_CANCELLED


async def test_integration_reentry_keeps_single_stage_row(session: AsyncSession) -> None:
    tracker = await _tracker_for(session)
    kid = _kid()
    root, attempt = await tracker.open_attempt(kid)
    assert root is not None
    first = await tracker.begin_stage(
        knowledge_id=kid, attempt=attempt, stage=STAGE_EMBEDDING
    )
    assert first is not None
    await tracker.end_span(span=first)
    second = await tracker.begin_stage(
        knowledge_id=kid, attempt=attempt, stage=STAGE_EMBEDDING
    )
    assert second is not None
    assert second.span_id == first.span_id
    stages = [
        s
        for s in await tracker.list_attempt_spans(kid, attempt)
        if s.kind == SPAN_KIND_STAGE
    ]
    assert len(stages) == 1


async def test_integration_attempt_history_preserved(session: AsyncSession) -> None:
    tracker = await _tracker_for(session)
    kid = _kid()
    _, first = await tracker.open_attempt(kid)
    root, second = await tracker.open_attempt(kid)
    assert root is not None
    await tracker.begin_stage(
        knowledge_id=kid, attempt=second, stage=STAGE_DOCREADER
    )
    assert await tracker.latest_attempt(kid) == 2
    progress = await tracker.get_progress(knowledge_id=kid, attempt=first)
    assert [s.kind for s in progress.spans] == [SPAN_KIND_ROOT]
    assert progress.latest_attempt == 2


async def test_integration_long_name_truncated_on_persist(session: AsyncSession) -> None:
    tracker = await _tracker_for(session)
    kid = _kid()
    root, attempt = await tracker.open_attempt(kid)
    assert root is not None
    long_name = f"postprocess.wiki.page[{'x' * 300}]"
    sub = await tracker.begin_sub_span(parent=root, name=long_name)
    assert sub is not None
    assert len(sub.name) == 255
    persisted = await tracker.list_attempt_spans(kid, attempt)
    names = [s.name for s in persisted]
    assert len(names[1]) == 255


async def test_integration_progress_error_classification(session: AsyncSession) -> None:
    tracker = await _tracker_for(session)
    with pytest.raises(ValidationError) as blank:
        await tracker.get_progress(knowledge_id="")
    assert blank.value.code == "span.knowledge_required"
    with pytest.raises(ValidationError) as attempt:
        await tracker.get_progress(knowledge_id="kid-x", attempt=0)
    assert attempt.value.code == "span.invalid_attempt"
    with pytest.raises(NotFoundError) as missing:
        await tracker.get_progress(knowledge_id=f"kid-{make_test_tenant_id()}")
    assert missing.value.code == "span.not_found"
