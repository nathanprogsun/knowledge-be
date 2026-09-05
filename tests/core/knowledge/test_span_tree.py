"""Unit tests for the processing-span tree projection."""

from __future__ import annotations

from datetime import UTC, datetime

from src.core.knowledge.documents.span_tracker import (
    ROOT_SPAN_NAME,
    SPAN_KIND_ROOT,
    SPAN_KIND_STAGE,
    SPAN_STATUS_FAILED,
    SPAN_STATUS_PENDING,
    SPAN_STATUS_RUNNING,
    SpanProgress,
)
from src.core.knowledge.documents.span_tree import spans_read_payload
from src.db.models.knowledge_processing_span import KnowledgeProcessingSpan

_NOW = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)


def _span(
    *,
    span_id: str,
    name: str,
    kind: str,
    status: str,
    parent_span_id: str | None = None,
    finished_at: datetime | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> KnowledgeProcessingSpan:
    return KnowledgeProcessingSpan(
        id=1,
        knowledge_id="kn-1",
        attempt=1,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=name,
        kind=kind,
        status=status,
        error_code=error_code,
        error_message=error_message,
        finished_at=finished_at,
        created_at=_NOW,
        updated_at=_NOW,
    )


def test_never_tracked_returns_empty_root() -> None:
    payload = spans_read_payload(
        knowledge_id="kn-1",
        parse_status="processing",
        progress=None,
    )
    assert payload.attempt == 0
    assert payload.current_attempt == 0
    assert payload.trace.span_id == ""
    assert payload.trace.name == ROOT_SPAN_NAME
    assert payload.trace.kind == SPAN_KIND_ROOT
    assert payload.last_error is None


def test_empty_attempt_keeps_attempt_number() -> None:
    payload = spans_read_payload(
        knowledge_id="kn-1",
        parse_status="pending",
        progress=SpanProgress(
            knowledge_id="kn-1",
            attempt=2,
            latest_attempt=2,
            spans=(),
        ),
    )
    assert payload.attempt == 2
    assert payload.latest_attempt == 2
    assert payload.trace.span_id == ""


def test_nests_stage_under_root_and_picks_running_stage() -> None:
    root = _span(
        span_id="root-1",
        name=ROOT_SPAN_NAME,
        kind=SPAN_KIND_ROOT,
        status=SPAN_STATUS_RUNNING,
    )
    stage = _span(
        span_id="stage-1",
        name="docreader",
        kind=SPAN_KIND_STAGE,
        status=SPAN_STATUS_RUNNING,
        parent_span_id="root-1",
    )
    payload = spans_read_payload(
        knowledge_id="kn-1",
        parse_status="processing",
        progress=SpanProgress(
            knowledge_id="kn-1",
            attempt=1,
            latest_attempt=1,
            spans=(root, stage),
        ),
    )
    assert payload.trace.span_id == "root-1"
    assert [child.span_id for child in payload.trace.children] == ["stage-1"]
    assert payload.current_stage == "docreader"
    assert payload.last_error is None


def test_last_error_is_most_recent_failed_span() -> None:
    root = _span(
        span_id="root-1",
        name=ROOT_SPAN_NAME,
        kind=SPAN_KIND_ROOT,
        status=SPAN_STATUS_FAILED,
    )
    older = _span(
        span_id="stage-old",
        name="docreader",
        kind=SPAN_KIND_STAGE,
        status=SPAN_STATUS_FAILED,
        parent_span_id="root-1",
        finished_at=_NOW,
        error_code="OLD",
        error_message="first",
    )
    newer = _span(
        span_id="stage-new",
        name="chunking",
        kind=SPAN_KIND_STAGE,
        status=SPAN_STATUS_FAILED,
        parent_span_id="root-1",
        finished_at=_NOW.replace(minute=1),
        error_code="NEW",
        error_message="second",
    )
    pending = _span(
        span_id="stage-pending",
        name="embedding",
        kind=SPAN_KIND_STAGE,
        status=SPAN_STATUS_PENDING,
        parent_span_id="root-1",
    )
    payload = spans_read_payload(
        knowledge_id="kn-1",
        parse_status="failed",
        progress=SpanProgress(
            knowledge_id="kn-1",
            attempt=1,
            latest_attempt=1,
            spans=(root, older, newer, pending),
        ),
    )
    assert payload.last_error is not None
    assert payload.last_error.error_code == "NEW"
    assert payload.current_stage == "embedding"
