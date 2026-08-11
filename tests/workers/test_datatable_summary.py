"""Unit tests for the ARQ ``datatable:summary`` worker task.

The handler is a thin shim over the data-table summary generation seam;
the tests mock the dispatch seam / runner so they run without a database,
AI provider, or table-data tool. They cover:

- payload validation (required ids, optional model ids, defaults),
- registry wiring (the handler registers under ``"datatable:summary"``),
- delegation: the parsed payload reaches the dispatch seam / injected
  runner with the right field names,
- result serialisation: the returned dict matches
  :class:`DataTableSummaryResult` semantics,
- the seam refuses to run when no runner is available (the worker wiring
  layer that composes the core dependencies has not landed yet).

No real ARQ broker, no real DB, no real provider calls.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from arq.connections import ArqRedis
from pydantic import ValidationError

from src.core.knowledge.documents.datatable_summary import DataTableSummaryResult
from src.workers.base import WorkerContext
from src.workers.registry import get_task
from src.workers.tasks import datatable_summary as datatable_summary_module
from src.workers.tasks.datatable_summary import (
    DatatableSummaryPayload,
    TASK_NAME,
    parse_payload,
    run_datatable_summary,
    task_datatable_summary,
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


def _base_payload() -> dict[str, Any]:
    """Required-field payload shared across delegation tests."""
    return {
        "tenant_id": 1,
        "knowledge_id": "k-1",
    }


class _FakeRunner:
    """Stand-in for the wiring-provided composition seam.

    Records the payload it received and returns a fixed
    :class:`DataTableSummaryResult` so tests can assert the dispatch
    contract without running any core code.
    """

    def __init__(self, result: DataTableSummaryResult) -> None:
        self.result = result
        self.received: list[DatatableSummaryPayload] = []

    async def __call__(self, *, payload: DatatableSummaryPayload) -> DataTableSummaryResult:
        self.received.append(payload)
        return self.result


def _result() -> DataTableSummaryResult:
    """A representative summary result for seam / serialisation tests."""
    return DataTableSummaryResult(
        knowledge_id="k-1",
        summary_chunk_id="summary-1",
        column_chunk_id="column-1",
    )


@pytest.fixture(autouse=True)
def _restore_registry() -> Iterator[None]:
    """Snapshot the registry around each test to avoid cross-test pollution.

    The ``register_task`` decorator mutates a module-level dict. Tests
    that import the module leave the registration in place, but a
    future test that re-registers under the same name would silently
    overwrite it. The fixture is defensive — a no-op today.
    """
    from src.workers import registry as registry_module

    snapshot = dict(registry_module.all_tasks())
    yield
    # Restore the snapshot — drop any entries added during the test.
    current = registry_module.all_tasks()
    for name in current.keys() - snapshot.keys():
        current.pop(name, None)
    for name, handler in snapshot.items():
        current[name] = handler


# ── Registry ─────────────────────────────────────────────────────────


def test_handler_registered_under_upstream_task_name() -> None:
    """The handler registers under the upstream task type verbatim."""
    assert TASK_NAME == "datatable:summary"
    assert get_task(TASK_NAME) is task_datatable_summary


def test_normalized_name_is_not_registered() -> None:
    """A colon-free variant must not silently resolve."""
    assert get_task("datatable_summary") is None


# ── Payload model ────────────────────────────────────────────────────


def test_parse_payload_accepts_minimum_required_fields() -> None:
    parsed = parse_payload(_base_payload())
    assert parsed.tenant_id == 1
    assert parsed.knowledge_id == "k-1"
    assert parsed.summary_model == ""
    assert parsed.embedding_model == ""


def test_parse_payload_accepts_optional_model_ids() -> None:
    parsed = parse_payload(
        {
            **_base_payload(),
            "summary_model": "model-summary",
            "embedding_model": "model-embed",
        }
    )
    assert parsed.summary_model == "model-summary"
    assert parsed.embedding_model == "model-embed"


def test_parse_payload_ignores_tracing_context_fields() -> None:
    """Upstream tracing-context fields ride along in the JSON payload."""
    parsed = parse_payload(
        {
            **_base_payload(),
            "trace_id": "trace-1",
            "langfuse_trace_id": "lf-1",
            "extra": "ignored",
        }
    )
    assert parsed.tenant_id == 1
    assert parsed.knowledge_id == "k-1"


def test_parse_payload_rejects_missing_tenant() -> None:
    payload = _base_payload()
    payload.pop("tenant_id")
    with pytest.raises(ValidationError):
        parse_payload(payload)


def test_parse_payload_rejects_missing_knowledge_id() -> None:
    payload = _base_payload()
    payload.pop("knowledge_id")
    with pytest.raises(ValidationError):
        parse_payload(payload)


def test_parse_payload_rejects_non_int_tenant() -> None:
    with pytest.raises(ValidationError):
        parse_payload({**_base_payload(), "tenant_id": "not-an-int"})


def test_payload_is_frozen() -> None:
    """The payload model is immutable; mutations must raise."""
    payload = DatatableSummaryPayload.model_validate(_base_payload())
    with pytest.raises(ValidationError):
        payload.knowledge_id = "tampered"


# ── Task dispatch ────────────────────────────────────────────────────


async def test_task_delegates_to_seam() -> None:
    """The handler parses and forwards the payload to the dispatch seam."""
    with patch.object(
        datatable_summary_module,
        "run_datatable_summary",
        new_callable=AsyncMock,
        return_value=_result(),
    ) as mock:
        result = await task_datatable_summary(make_ctx(), **_base_payload())

    parsed = mock.call_args.kwargs["payload"]
    assert isinstance(parsed, DatatableSummaryPayload)
    assert parsed.tenant_id == 1
    assert parsed.knowledge_id == "k-1"
    assert mock.call_args.kwargs["runner"] is None
    assert result == {
        "knowledge_id": "k-1",
        "summary_chunk_id": "summary-1",
        "column_chunk_id": "column-1",
    }


async def test_task_forwards_injected_runner() -> None:
    """A wiring-provided runner is passed through to the dispatch seam."""
    runner = _FakeRunner(_result())
    with patch.object(
        datatable_summary_module,
        "run_datatable_summary",
        new_callable=AsyncMock,
        return_value=_result(),
    ) as mock:
        await task_datatable_summary(make_ctx(), runner=runner, **_base_payload())

    assert mock.call_args.kwargs["runner"] is runner


async def test_task_end_to_end_with_runner() -> None:
    """The full task path — parse, dispatch, serialise — via a fake runner."""
    result = await task_datatable_summary(
        make_ctx(),
        runner=_FakeRunner(_result()),
        summary_model="model-summary",
        embedding_model="model-embed",
        **_base_payload(),
    )
    assert result == {
        "knowledge_id": "k-1",
        "summary_chunk_id": "summary-1",
        "column_chunk_id": "column-1",
    }


async def test_task_rejects_invalid_payload() -> None:
    """Invalid payloads surface as Pydantic validation errors."""
    with pytest.raises(ValidationError):
        await task_datatable_summary(make_ctx(), tenant_id="not-an-int")


async def test_task_propagates_seam_errors() -> None:
    """Errors raised by the dispatch seam surface to the worker caller."""
    with (
        patch.object(
            datatable_summary_module,
            "run_datatable_summary",
            new_callable=AsyncMock,
            side_effect=RuntimeError("summary exploded"),
        ),
        pytest.raises(RuntimeError, match="summary exploded"),
    ):
        await task_datatable_summary(make_ctx(), **_base_payload())


# ── Dispatch seam ────────────────────────────────────────────────────


async def test_seam_uses_injected_runner() -> None:
    """A provided runner receives the parsed payload and its result wins."""
    expected = _result()
    runner = _FakeRunner(expected)
    result = await run_datatable_summary(payload=parse_payload(_base_payload()), runner=runner)
    assert result == expected
    assert len(runner.received) == 1
    assert runner.received[0].tenant_id == 1


async def test_seam_raises_without_runner() -> None:
    """No runner means the task refuses to run (wiring not merged yet)."""
    with pytest.raises(NotImplementedError):
        await run_datatable_summary(payload=parse_payload(_base_payload()))
