"""Unit tests for the ``wiki:ingest`` and ``wiki:finalize`` worker tasks.

Covers the worker-side surface for both tasks: the registered handlers
are the expected functions, the trigger payloads parse cleanly into the
contract models, the handlers delegate to the core seams with the parsed
arguments, and the core seams behave correctly without a wired service.
The ingest delegation is exercised through a mocked core
:class:`WikiIngestService`; the finalize dispatch through a patched core
seam — no real database or pipeline is needed.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from arq.connections import ArqRedis
from pydantic import ValidationError

from src.core.knowledge.wiki.ingest_service import WikiIngestService
from src.core.knowledge.wiki.ingest_types import WikiBatchOutcome
from src.workers.base import WorkerContext
from src.workers.registry import get_task
from src.workers.tasks import wiki_finalize as wiki_finalize_module
from src.workers.tasks import wiki_ingest as wiki_ingest_module
from src.workers.tasks.wiki_finalize import (
    WikiFinalizePayload,
    process_wiki_finalize,
    task_wiki_finalize,
)
from src.workers.tasks.wiki_ingest import (
    WikiIngestPayload,
    process_wiki_ingest,
    task_wiki_ingest,
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
def valid_ingest_payload() -> dict[str, object]:
    """A representative JSON payload for the wiki ingest trigger."""
    return {
        "tenant_id": 42,
        "knowledge_base_id": "kb-1",
        "language": "zh",
    }


@pytest.fixture
def valid_finalize_payload() -> dict[str, object]:
    """A representative JSON payload for the wiki finalize trigger."""
    return {
        "tenant_id": 42,
        "knowledge_base_id": "kb-1",
        "language": "zh",
    }


def _outcome(**overrides: object) -> WikiBatchOutcome:
    """A default aggregate ingest outcome for serialisation tests."""
    values: dict[str, object] = {
        "pending_ops": 5,
        "ingest_succeeded": 4,
        "ingest_failed": 1,
        "retract_handled": 0,
        "pages_affected": 3,
        "follow_up_scheduled": False,
        "rate_limited": False,
    }
    values.update(overrides)
    return WikiBatchOutcome(**values)  # type: ignore[arg-type]


@pytest.fixture
def outcome() -> WikiBatchOutcome:
    """A default aggregate ingest outcome."""
    return _outcome()


@pytest.fixture
def mock_ingest_service(outcome: WikiBatchOutcome) -> AsyncMock:
    """A mocked core wiki ingest service bound to the batch seam."""
    service = AsyncMock(spec=WikiIngestService)
    service.process_batch.return_value = outcome
    return service


# ── Registration ────────────────────────────────────────────────────


def test_wiki_ingest_registered_under_task_name() -> None:
    """The handler is registered under the upstream task type name."""
    assert get_task("wiki:ingest") is task_wiki_ingest


def test_wiki_finalize_registered_under_task_name() -> None:
    """The handler is registered under the upstream task type name."""
    assert get_task("wiki:finalize") is task_wiki_finalize


def test_wiki_tasks_unknown_name_returns_none() -> None:
    """A typo in a registered name must not silently resolve."""
    assert get_task("wiki:inges") is None
    assert get_task("wiki:finalz") is None


# ── Ingest payload contract ─────────────────────────────────────────


def test_ingest_payload_parses_full() -> None:
    """A complete payload round-trips through the contract model."""
    payload = WikiIngestPayload.model_validate(
        {
            "tenant_id": 7,
            "knowledge_base_id": "kb-1",
            "language": "en",
        }
    )
    assert payload.tenant_id == 7
    assert payload.knowledge_base_id == "kb-1"
    assert payload.language == "en"


def test_ingest_payload_defaults_language() -> None:
    """``language`` is optional and defaults to empty."""
    payload = WikiIngestPayload.model_validate({"tenant_id": 1, "knowledge_base_id": "kb-1"})
    assert payload.language == ""


def test_ingest_payload_ignores_upstream_tracing_fields() -> None:
    """Tracing fields are accepted but not modelled."""
    payload = WikiIngestPayload.model_validate(
        {
            "tenant_id": 1,
            "knowledge_base_id": "kb-1",
            "lf_trace_id": "trace-1",
            "lf_traceparent": "00-abc-def-01",
        }
    )
    assert payload.tenant_id == 1
    assert payload.knowledge_base_id == "kb-1"
    assert payload.language == ""


def test_ingest_payload_rejects_missing_tenant_id() -> None:
    """The tenant id is mandatory."""
    with pytest.raises(ValidationError):
        WikiIngestPayload.model_validate({"knowledge_base_id": "kb-1"})


def test_ingest_payload_rejects_missing_knowledge_base_id() -> None:
    """The knowledge-base id is mandatory."""
    with pytest.raises(ValidationError):
        WikiIngestPayload.model_validate({"tenant_id": 1})


def test_ingest_payload_is_frozen() -> None:
    """The payload model is immutable; mutations must raise."""
    payload = WikiIngestPayload.model_validate({"tenant_id": 1, "knowledge_base_id": "kb-1"})
    with pytest.raises(ValidationError):
        payload.knowledge_base_id = "tampered"


# ── Finalize payload contract ───────────────────────────────────────


def test_finalize_payload_parses_full() -> None:
    """A complete payload round-trips through the contract model."""
    payload = WikiFinalizePayload.model_validate(
        {
            "tenant_id": 7,
            "knowledge_base_id": "kb-1",
            "language": "en",
        }
    )
    assert payload.tenant_id == 7
    assert payload.knowledge_base_id == "kb-1"
    assert payload.language == "en"


def test_finalize_payload_defaults_language() -> None:
    """``language`` is optional and defaults to empty."""
    payload = WikiFinalizePayload.model_validate({"tenant_id": 1, "knowledge_base_id": "kb-1"})
    assert payload.language == ""


def test_finalize_payload_ignores_upstream_tracing_fields() -> None:
    """Tracing fields are accepted but not modelled."""
    payload = WikiFinalizePayload.model_validate(
        {
            "tenant_id": 1,
            "knowledge_base_id": "kb-1",
            "lf_trace_id": "trace-1",
            "lf_traceparent": "00-abc-def-01",
        }
    )
    assert payload.tenant_id == 1
    assert payload.knowledge_base_id == "kb-1"
    assert payload.language == ""


def test_finalize_payload_rejects_missing_tenant_id() -> None:
    """The tenant id is mandatory."""
    with pytest.raises(ValidationError):
        WikiFinalizePayload.model_validate({"knowledge_base_id": "kb-1"})


def test_finalize_payload_rejects_missing_knowledge_base_id() -> None:
    """The knowledge-base id is mandatory."""
    with pytest.raises(ValidationError):
        WikiFinalizePayload.model_validate({"tenant_id": 1})


def test_finalize_payload_is_frozen() -> None:
    """The payload model is immutable; mutations must raise."""
    payload = WikiFinalizePayload.model_validate({"tenant_id": 1, "knowledge_base_id": "kb-1"})
    with pytest.raises(ValidationError):
        payload.knowledge_base_id = "tampered"


# ── Ingest worker dispatch ──────────────────────────────────────────


async def test_wiki_ingest_delegates_to_core_seam(
    ctx: WorkerContext,
    valid_ingest_payload: dict[str, object],
    mock_ingest_service: AsyncMock,
) -> None:
    """The handler parses and forwards the payload to the core seam."""
    with patch.object(
        wiki_ingest_module,
        "process_wiki_ingest",
        new_callable=AsyncMock,
        return_value={"status": "completed"},
    ) as mock:
        result = await task_wiki_ingest(
            ctx,
            service=cast(WikiIngestService, mock_ingest_service),
            **valid_ingest_payload,  # type: ignore[arg-type]
        )

    mock.assert_awaited_once_with(
        tenant_id=42,
        knowledge_base_id="kb-1",
        language="zh",
        service=mock_ingest_service,
    )
    assert result == {"status": "completed"}


async def test_wiki_ingest_uses_default_language(
    ctx: WorkerContext,
    mock_ingest_service: AsyncMock,
) -> None:
    """An omitted ``language`` falls back to empty."""
    result = await task_wiki_ingest(
        ctx,
        service=cast(WikiIngestService, mock_ingest_service),
        tenant_id=1,
        knowledge_base_id="kb-1",
    )

    mock_ingest_service.process_batch.assert_awaited_once_with(
        None,
        tenant_id=1,
        knowledge_base_id="kb-1",
        language="",
    )
    assert result["pending_ops"] == 5


async def test_wiki_ingest_rejects_invalid_payload(ctx: WorkerContext) -> None:
    """A payload missing required fields surfaces as ``ValidationError``."""
    with pytest.raises(ValidationError):
        await task_wiki_ingest(ctx, tenant_id=1)


async def test_wiki_ingest_propagates_core_errors(
    ctx: WorkerContext,
    valid_ingest_payload: dict[str, object],
    mock_ingest_service: AsyncMock,
) -> None:
    """Errors raised by the core service surface to the worker caller."""
    mock_ingest_service.process_batch.side_effect = RuntimeError("batch exploded")
    with pytest.raises(RuntimeError, match="batch exploded"):
        await task_wiki_ingest(
            ctx,
            service=cast(WikiIngestService, mock_ingest_service),
            **valid_ingest_payload,  # type: ignore[arg-type]
        )


# ── Finalize worker dispatch ────────────────────────────────────────


async def test_wiki_finalize_delegates_to_core_seam(
    ctx: WorkerContext,
    valid_finalize_payload: dict[str, object],
) -> None:
    """The handler parses and forwards the payload to the core seam."""
    with patch.object(
        wiki_finalize_module,
        "process_wiki_finalize",
        new_callable=AsyncMock,
        return_value={"status": "converged"},
    ) as mock:
        result = await task_wiki_finalize(ctx, **valid_finalize_payload)  # type: ignore[arg-type]

    mock.assert_awaited_once_with(
        tenant_id=42,
        knowledge_base_id="kb-1",
        language="zh",
    )
    assert result == {"status": "converged"}


async def test_wiki_finalize_uses_default_language(ctx: WorkerContext) -> None:
    """An omitted ``language`` falls back to empty."""
    with patch.object(
        wiki_finalize_module,
        "process_wiki_finalize",
        new_callable=AsyncMock,
        return_value={"status": "converged"},
    ) as mock:
        await task_wiki_finalize(ctx, tenant_id=1, knowledge_base_id="kb-1")

    mock.assert_awaited_once_with(
        tenant_id=1,
        knowledge_base_id="kb-1",
        language="",
    )


async def test_wiki_finalize_rejects_invalid_payload(ctx: WorkerContext) -> None:
    """A payload missing required fields surfaces as ``ValidationError``."""
    with pytest.raises(ValidationError):
        await task_wiki_finalize(ctx, tenant_id=1)


async def test_wiki_finalize_propagates_core_errors(
    ctx: WorkerContext,
    valid_finalize_payload: dict[str, object],
) -> None:
    """Errors raised by the core seam surface to the worker caller."""
    with (
        patch.object(
            wiki_finalize_module,
            "process_wiki_finalize",
            new_callable=AsyncMock,
            side_effect=RuntimeError("convergence exploded"),
        ),
        pytest.raises(RuntimeError, match="convergence exploded"),
    ):
        await task_wiki_finalize(ctx, **valid_finalize_payload)  # type: ignore[arg-type]


# ── Core seam: ingest ───────────────────────────────────────────────


async def test_process_wiki_ingest_raises_without_service() -> None:
    """The ingest seam refuses to run without an injected service."""
    with pytest.raises(NotImplementedError, match="WikiIngestService"):
        await process_wiki_ingest(
            tenant_id=1,
            knowledge_base_id="kb-1",
            language="",
        )


async def test_process_wiki_ingest_delegates_to_injected_service(
    mock_ingest_service: AsyncMock,
) -> None:
    """An injected service runs the batch and its outcome is serialised."""
    result = await process_wiki_ingest(
        tenant_id=42,
        knowledge_base_id="kb-1",
        language="zh",
        service=cast(WikiIngestService, mock_ingest_service),
    )

    mock_ingest_service.process_batch.assert_awaited_once_with(
        None,
        tenant_id=42,
        knowledge_base_id="kb-1",
        language="zh",
    )
    assert result == {
        "pending_ops": 5,
        "ingest_succeeded": 4,
        "ingest_failed": 1,
        "retract_handled": 0,
        "pages_affected": 3,
        "follow_up_scheduled": False,
        "rate_limited": False,
    }


async def test_process_wiki_ingest_serialises_rate_limited() -> None:
    """The ``rate_limited`` flag and follow-up propagate into the result."""
    service = AsyncMock(spec=WikiIngestService)
    service.process_batch.return_value = _outcome(
        follow_up_scheduled=True,
        rate_limited=True,
    )

    result = await process_wiki_ingest(
        tenant_id=42,
        knowledge_base_id="kb-1",
        language="zh",
        service=cast(WikiIngestService, service),
    )

    assert result["follow_up_scheduled"] is True
    assert result["rate_limited"] is True


# ── Core seam: finalize ─────────────────────────────────────────────


async def test_process_wiki_finalize_raises_not_implemented() -> None:
    """The core seam is a placeholder until the core implementation lands."""
    with pytest.raises(NotImplementedError):
        await process_wiki_finalize(
            tenant_id=1,
            knowledge_base_id="kb-1",
            language="",
        )


# ── Re-registration guard ──────────────────────────────────────────


@pytest.fixture(autouse=True)
def _restore_registry() -> Iterator[None]:
    """Snapshot the registry around each test to avoid cross-test pollution.

    The ``register_task`` decorator mutates a module-level dict. Tests
    that import the module leave the registration in place, but a
    future test that re-registers under the same name would silently
    overwrite it. The fixture is defensive — a no-op today.

    Eagerly imports ``src.workers.tasks`` before snapshotting so the
    first test in the session captures the canonical 19-task set
    rather than an empty dict (tasks register at import time).
    """
    import src.workers.tasks  # noqa: F401 — side effect: register all handlers
    from src.workers import registry as registry_module

    baseline = dict(registry_module.all_tasks())
    yield
    # Restore the snapshot — drop any entries added during the test.
    current = registry_module.all_tasks()
    for name in list(current.keys()):
        if name not in baseline:
            current.pop(name, None)
    for name, handler in baseline.items():
        current[name] = handler
