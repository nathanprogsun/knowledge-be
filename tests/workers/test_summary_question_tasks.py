"""Unit tests for the ``summary:generation`` and ``question:generation`` worker tasks.

Each handler is a thin dispatcher over its core seam
(:func:`src.workers.tasks.summary_generation.process_summary_generation`
and :func:`src.workers.tasks.question_generation.process_question_generation`).
The tests patch the seam so they run without a database or AI provider.
They cover:

- payload validation (required ids, optional fields, defaults),
- registry wiring (the handlers register under the upstream task names),
- delegation: the parsed fields reach the seam with the right names and
  types,
- result passthrough: the handler returns the seam's result unchanged,
- error paths: invalid payloads surface as ``ValidationError`` and core
  failures propagate.

No real ARQ broker, no real DB, no real provider calls.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from arq.connections import ArqRedis
from pydantic import ValidationError

from src.workers.base import WorkerContext
from src.workers.registry import get_task
from src.workers.tasks import question_generation as question_generation_module
from src.workers.tasks import summary_generation as summary_generation_module
from src.workers.tasks.question_generation import (
    QuestionGenerationTaskPayload,
    process_question_generation,
    task_question_generation,
)
from src.workers.tasks.question_generation import (
    parse_payload as parse_question_payload,
)
from src.workers.tasks.summary_generation import (
    SummaryGenerationTaskPayload,
    process_summary_generation,
    task_summary_generation,
)
from src.workers.tasks.summary_generation import (
    parse_payload as parse_summary_payload,
)


def make_ctx() -> WorkerContext:
    """Build a context dict matching what ARQ passes to tasks."""
    return WorkerContext(
        redis=cast(ArqRedis, None),
        job_id="job-1",
        job_try=1,
        enqueue_time=datetime.now(UTC),
        score=0,
    )


@pytest.fixture
def ctx() -> WorkerContext:
    """Worker context for ad-hoc task invocations."""
    return make_ctx()


@pytest.fixture
def summary_payload() -> dict[str, object]:
    """A representative JSON payload for the summary-generation task."""
    return {
        "tenant_id": 42,
        "knowledge_id": "doc-1",
        "knowledge_base_id": "kb-1",
        "language": "en-US",
        "refresh": True,
        "attempt": 2,
    }


@pytest.fixture
def question_payload() -> dict[str, object]:
    """A representative JSON payload for the question-generation task."""
    return {
        "tenant_id": 42,
        "knowledge_id": "doc-1",
        "knowledge_base_id": "kb-1",
        "question_count": 5,
        "language": "en-US",
        "attempt": 2,
        "chunk_ids": ["chunk-a", "chunk-b"],
        "chunk_id": "chunk-c",
        "batch_index": 1,
        "prev_chunk_id": "chunk-0",
        "next_chunk_id": "chunk-3",
    }


# ── Registry ────────────────────────────────────────────────────────


def test_summary_generation_registered_under_task_name() -> None:
    """The handler is registered under the upstream task type name."""
    assert get_task("summary:generation") is task_summary_generation


def test_question_generation_registered_under_task_name() -> None:
    """The handler is registered under the upstream task type name."""
    assert get_task("question:generation") is task_question_generation


def test_unknown_name_returns_none() -> None:
    """A typo in the registered name must not silently resolve."""
    assert get_task("summary:generation ") is None
    assert get_task("question_generation") is None


# ── Summary payload contract ────────────────────────────────────────


def test_summary_payload_parses_full() -> None:
    """A complete summary payload round-trips through the contract model."""
    payload = SummaryGenerationTaskPayload.model_validate(
        {
            "tenant_id": 7,
            "knowledge_id": "doc-1",
            "knowledge_base_id": "kb-1",
            "language": "zh-CN",
            "refresh": True,
            "attempt": 3,
        }
    )
    assert payload.tenant_id == 7
    assert payload.knowledge_id == "doc-1"
    assert payload.knowledge_base_id == "kb-1"
    assert payload.language == "zh-CN"
    assert payload.refresh is True
    assert payload.attempt == 3


def test_summary_payload_defaults_optional_fields() -> None:
    """``language``, ``refresh``, and ``attempt`` default sensibly."""
    payload = SummaryGenerationTaskPayload.model_validate(
        {
            "tenant_id": 1,
            "knowledge_id": "doc-1",
            "knowledge_base_id": "kb-1",
        }
    )
    assert payload.language == ""
    assert payload.refresh is False
    assert payload.attempt == 0


def test_summary_payload_rejects_missing_tenant_id() -> None:
    """The tenant id is mandatory."""
    with pytest.raises(ValidationError):
        SummaryGenerationTaskPayload.model_validate(
            {"knowledge_id": "doc-1", "knowledge_base_id": "kb-1"}
        )


def test_summary_payload_rejects_missing_knowledge_id() -> None:
    """The knowledge id is mandatory."""
    with pytest.raises(ValidationError):
        SummaryGenerationTaskPayload.model_validate({"tenant_id": 1, "knowledge_base_id": "kb-1"})


def test_summary_payload_rejects_missing_knowledge_base_id() -> None:
    """The knowledge base id is mandatory."""
    with pytest.raises(ValidationError):
        SummaryGenerationTaskPayload.model_validate({"tenant_id": 1, "knowledge_id": "doc-1"})


def test_summary_payload_ignores_unknown_fields() -> None:
    """Tracing-context and stray fields are ignored, not rejected."""
    parsed = parse_summary_payload(
        {
            "tenant_id": 1,
            "knowledge_id": "doc-1",
            "knowledge_base_id": "kb-1",
            "lf_trace_id": "trace-1",
            "lf_traceparent": "00-abc-def-01",
            "extra": "ignored",
        }
    )
    assert parsed.tenant_id == 1


def test_summary_payload_is_frozen() -> None:
    """The payload model is immutable; mutations must raise."""
    payload = SummaryGenerationTaskPayload.model_validate(
        {
            "tenant_id": 1,
            "knowledge_id": "doc-1",
            "knowledge_base_id": "kb-1",
        }
    )
    with pytest.raises(ValidationError):
        payload.language = "zh-CN"


# ── Question payload contract ───────────────────────────────────────


def test_question_payload_parses_full() -> None:
    """A complete question payload round-trips through the contract model."""
    payload = QuestionGenerationTaskPayload.model_validate(
        {
            "tenant_id": 7,
            "knowledge_id": "doc-1",
            "knowledge_base_id": "kb-1",
            "question_count": 5,
            "language": "en-US",
            "attempt": 3,
            "chunk_ids": ["a", "b"],
            "chunk_id": "c",
            "batch_index": 1,
            "prev_chunk_id": "p",
            "next_chunk_id": "n",
        }
    )
    assert payload.tenant_id == 7
    assert payload.knowledge_id == "doc-1"
    assert payload.knowledge_base_id == "kb-1"
    assert payload.question_count == 5
    assert payload.language == "en-US"
    assert payload.attempt == 3
    assert payload.chunk_ids == ["a", "b"]
    assert payload.chunk_id == "c"
    assert payload.batch_index == 1
    assert payload.prev_chunk_id == "p"
    assert payload.next_chunk_id == "n"


def test_question_payload_defaults_optional_fields() -> None:
    """Optional fields fall back to the shared ingestion defaults."""
    payload = QuestionGenerationTaskPayload.model_validate(
        {
            "tenant_id": 1,
            "knowledge_id": "doc-1",
            "knowledge_base_id": "kb-1",
        }
    )
    assert payload.question_count == 3
    assert payload.language == ""
    assert payload.attempt == 0
    assert payload.chunk_ids == []
    assert payload.chunk_id == ""
    assert payload.batch_index == 0
    assert payload.prev_chunk_id == ""
    assert payload.next_chunk_id == ""


def test_question_payload_keeps_explicit_zero_question_count() -> None:
    """An enqueued ``question_count`` of 0 means "use the default"."""
    payload = parse_question_payload(
        {
            "tenant_id": 1,
            "knowledge_id": "doc-1",
            "knowledge_base_id": "kb-1",
            "question_count": 0,
        }
    )
    assert payload.question_count == 0


def test_question_payload_rejects_missing_tenant_id() -> None:
    """The tenant id is mandatory."""
    with pytest.raises(ValidationError):
        QuestionGenerationTaskPayload.model_validate(
            {"knowledge_id": "doc-1", "knowledge_base_id": "kb-1"}
        )


def test_question_payload_rejects_missing_knowledge_id() -> None:
    """The knowledge id is mandatory."""
    with pytest.raises(ValidationError):
        QuestionGenerationTaskPayload.model_validate({"tenant_id": 1, "knowledge_base_id": "kb-1"})


def test_question_payload_rejects_missing_knowledge_base_id() -> None:
    """The knowledge base id is mandatory."""
    with pytest.raises(ValidationError):
        QuestionGenerationTaskPayload.model_validate({"tenant_id": 1, "knowledge_id": "doc-1"})


def test_question_payload_ignores_unknown_fields() -> None:
    """Tracing-context and stray fields are ignored, not rejected."""
    parsed = parse_question_payload(
        {
            "tenant_id": 1,
            "knowledge_id": "doc-1",
            "knowledge_base_id": "kb-1",
            "lf_user_id": "user-1",
            "lf_session_id": "sess-1",
        }
    )
    assert parsed.tenant_id == 1


def test_question_payload_is_frozen() -> None:
    """The payload model is immutable; mutations must raise."""
    payload = QuestionGenerationTaskPayload.model_validate(
        {
            "tenant_id": 1,
            "knowledge_id": "doc-1",
            "knowledge_base_id": "kb-1",
        }
    )
    with pytest.raises(ValidationError):
        payload.question_count = 9


# ── Summary worker dispatch ─────────────────────────────────────────


async def test_summary_generation_delegates_to_core_seam(
    ctx: WorkerContext,
    summary_payload: dict[str, object],
) -> None:
    """The handler parses and forwards the payload to the core seam."""
    with patch.object(
        summary_generation_module,
        "process_summary_generation",
        new_callable=AsyncMock,
        return_value={"status": "completed", "summary_chars": 120},
    ) as mock:
        result = await task_summary_generation(ctx, **summary_payload)  # type: ignore[arg-type]

    mock.assert_awaited_once_with(
        tenant_id=42,
        knowledge_id="doc-1",
        knowledge_base_id="kb-1",
        language="en-US",
        refresh=True,
        attempt=2,
    )
    assert result == {"status": "completed", "summary_chars": 120}


async def test_summary_generation_uses_defaults_for_optional_fields(
    ctx: WorkerContext,
) -> None:
    """Omitted optional fields fall back to the contract defaults."""
    payload = {
        "tenant_id": 1,
        "knowledge_id": "doc-1",
        "knowledge_base_id": "kb-1",
    }
    with patch.object(
        summary_generation_module,
        "process_summary_generation",
        new_callable=AsyncMock,
        return_value={"status": "skipped", "skipped": "no_text_chunks"},
    ) as mock:
        await task_summary_generation(ctx, **payload)  # type: ignore[arg-type]

    mock.assert_awaited_once_with(
        tenant_id=1,
        knowledge_id="doc-1",
        knowledge_base_id="kb-1",
        language="",
        refresh=False,
        attempt=0,
    )


async def test_summary_generation_rejects_invalid_payload(
    ctx: WorkerContext,
) -> None:
    """A payload missing required fields surfaces as ``ValidationError``."""
    with pytest.raises(ValidationError):
        await task_summary_generation(ctx, tenant_id=1)


async def test_summary_generation_propagates_core_errors(
    ctx: WorkerContext,
    summary_payload: dict[str, object],
) -> None:
    """Errors raised by the core seam surface to the worker caller."""
    with (
        patch.object(
            summary_generation_module,
            "process_summary_generation",
            new_callable=AsyncMock,
            side_effect=RuntimeError("summary pipeline exploded"),
        ),
        pytest.raises(RuntimeError, match="summary pipeline exploded"),
    ):
        await task_summary_generation(ctx, **summary_payload)  # type: ignore[arg-type]


# ── Question worker dispatch ────────────────────────────────────────


async def test_question_generation_delegates_to_core_seam(
    ctx: WorkerContext,
    question_payload: dict[str, object],
) -> None:
    """The handler parses and forwards the payload to the core seam."""
    with patch.object(
        question_generation_module,
        "process_question_generation",
        new_callable=AsyncMock,
        return_value={"status": "success", "questions_generated": 10},
    ) as mock:
        result = await task_question_generation(ctx, **question_payload)  # type: ignore[arg-type]

    mock.assert_awaited_once_with(
        tenant_id=42,
        knowledge_id="doc-1",
        knowledge_base_id="kb-1",
        question_count=5,
        language="en-US",
        attempt=2,
        chunk_ids=["chunk-a", "chunk-b"],
        chunk_id="chunk-c",
        batch_index=1,
        prev_chunk_id="chunk-0",
        next_chunk_id="chunk-3",
    )
    assert result == {"status": "success", "questions_generated": 10}


async def test_question_generation_uses_defaults_for_optional_fields(
    ctx: WorkerContext,
) -> None:
    """Omitted optional fields fall back to the contract defaults."""
    payload = {
        "tenant_id": 1,
        "knowledge_id": "doc-1",
        "knowledge_base_id": "kb-1",
    }
    with patch.object(
        question_generation_module,
        "process_question_generation",
        new_callable=AsyncMock,
        return_value={"status": "success", "questions_generated": 0},
    ) as mock:
        await task_question_generation(ctx, **payload)  # type: ignore[arg-type]

    mock.assert_awaited_once_with(
        tenant_id=1,
        knowledge_id="doc-1",
        knowledge_base_id="kb-1",
        question_count=3,
        language="",
        attempt=0,
        chunk_ids=[],
        chunk_id="",
        batch_index=0,
        prev_chunk_id="",
        next_chunk_id="",
    )


async def test_question_generation_rejects_invalid_payload(
    ctx: WorkerContext,
) -> None:
    """A payload missing required fields surfaces as ``ValidationError``."""
    with pytest.raises(ValidationError):
        await task_question_generation(ctx, tenant_id="not-an-int")


async def test_question_generation_propagates_core_errors(
    ctx: WorkerContext,
    question_payload: dict[str, object],
) -> None:
    """Errors raised by the core seam surface to the worker caller."""
    with (
        patch.object(
            question_generation_module,
            "process_question_generation",
            new_callable=AsyncMock,
            side_effect=RuntimeError("question pipeline exploded"),
        ),
        pytest.raises(RuntimeError, match="question pipeline exploded"),
    ):
        await task_question_generation(ctx, **question_payload)  # type: ignore[arg-type]


# ── Core seam placeholders ──────────────────────────────────────────


async def test_process_summary_generation_raises_not_implemented() -> None:
    """The core seam is a placeholder until the composition wiring lands."""
    with pytest.raises(NotImplementedError):
        await process_summary_generation(
            tenant_id=1,
            knowledge_id="doc-1",
            knowledge_base_id="kb-1",
            language="",
            refresh=False,
            attempt=0,
        )


async def test_process_question_generation_raises_not_implemented() -> None:
    """The core seam is a placeholder until the composition wiring lands."""
    with pytest.raises(NotImplementedError):
        await process_question_generation(
            tenant_id=1,
            knowledge_id="doc-1",
            knowledge_base_id="kb-1",
            question_count=3,
            language="",
            attempt=0,
            chunk_ids=[],
            chunk_id="",
            batch_index=0,
            prev_chunk_id="",
            next_chunk_id="",
        )


# ── Re-registration guard ───────────────────────────────────────────


@pytest.fixture(autouse=True)
def _restore_registry() -> Iterator[None]:
    """Snapshot the registry around each test to avoid cross-test pollution.

    The ``register_task`` decorator mutates a module-level dict. Tests
    that import the modules leave the registrations in place, but a
    future test that re-registers under the same name would silently
    overwrite them. The fixture is defensive — a no-op today.
    """
    import src.workers.tasks  # noqa: F401 — pre-warm so the snapshot captures registered handlers.
    from src.workers import registry as registry_module

    snapshot = dict(registry_module.all_tasks())
    # Drop any test-only handler that was registered by ``@register_task``
    # decorators at import time of this module (e.g. ``test_base``'s
    # ``test_task``). The canonical handler set is what downstream
    # invariant tests assert against.
    baseline = {name: handler for name, handler in snapshot.items() if not name.startswith("test_")}
    yield
    # Restore the baseline after the test: drop anything newly added
    # (incl. handlers the test itself registered) and re-assert the
    # canonical handlers from the snapshot.
    current = registry_module.all_tasks()
    for name in list(current.keys()):
        if name not in baseline:
            current.pop(name, None)
    for name, handler in baseline.items():
        current[name] = handler
