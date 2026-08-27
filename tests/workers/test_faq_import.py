"""Unit tests for the ``faq:import`` worker task.

Covers the worker-side surface: the registered handler is the expected
function, the payload parses cleanly into the contract model, the
handler decodes the base64 file bytes and delegates to the core import
runner with the parsed arguments (through an injected
:class:`~src.core.knowledge.faq.import_runner.FAQImportRunner`), and the
un-injected seam raises so a miswired worker fails loudly. The core
dispatch is exercised through a mocked runner, so no real database or
file parser is needed.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from arq.connections import ArqRedis
from pydantic import ValidationError

from src.core.contracts.knowledge import FAQImportTaskProgress
from src.core.knowledge.faq.import_runner import FAQImportRunner
from src.workers.base import WorkerContext
from src.workers.registry import get_task
from src.workers.tasks import faq_import as faq_import_module
from src.workers.tasks.faq_import import (
    FAQImportPayload,
    process_faq_import,
    task_faq_import,
)

_CSV_BYTES = b"category,question,answer\naccount,how to recharge,see settings\n"
_ENCODED = base64.b64encode(_CSV_BYTES).decode("ascii")

NOW = datetime(2026, 3, 1, tzinfo=UTC)


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
def valid_payload() -> dict[str, object]:
    """A representative JSON payload for the FAQ-import task."""
    return {
        "tenant_id": 42,
        "task_id": "faq_import_42_1710000000000_abcd1234",
        "kb_id": "kb-1",
        "knowledge_id": "knowledge-1",
        "filename": "faq.csv",
        "file_data": _ENCODED,
        "mode": "append",
        "dry_run": False,
    }


@pytest.fixture
def import_progress() -> FAQImportTaskProgress:
    """A completed FAQ import progress as returned by the core runner."""
    return FAQImportTaskProgress(
        task_id="faq_import_42_1710000000000_abcd1234",
        kb_id="kb-1",
        knowledge_id="knowledge-1",
        status="completed",
        progress=100,
        total=2,
        processed=2,
        success_count=2,
        failed_count=0,
        created_at=1710000000,
        updated_at=1710000100,
        dry_run=False,
        import_mode="append",
        imported_at=NOW,
        processing_time=1000,
    )


@pytest.fixture
def mock_runner(import_progress: FAQImportTaskProgress) -> AsyncMock:
    """A mocked core FAQ import runner bound to the import seam."""
    runner = AsyncMock(spec=FAQImportRunner)
    runner.run.return_value = import_progress
    return runner


# ── Registration ────────────────────────────────────────────────────


def test_faq_import_registered_under_upstream_task_name() -> None:
    """The handler is registered under the upstream task name."""
    assert get_task("faq:import") is task_faq_import


def test_faq_import_unknown_name_returns_none() -> None:
    """A typo in the registered name must not silently resolve."""
    assert get_task("faq:impor") is None


# ── Payload contract ────────────────────────────────────────────────


def test_payload_parses_full() -> None:
    """A complete payload round-trips through the contract model."""
    payload = FAQImportPayload.model_validate(
        {
            "tenant_id": 7,
            "task_id": "faq_import_7_1710000000000_wxyz9876",
            "kb_id": "kb-9",
            "knowledge_id": "knowledge-9",
            "filename": "faq.xlsx",
            "file_data": _ENCODED,
            "mode": "replace",
            "dry_run": True,
        }
    )
    assert payload.tenant_id == 7
    assert payload.task_id == "faq_import_7_1710000000000_wxyz9876"
    assert payload.kb_id == "kb-9"
    assert payload.knowledge_id == "knowledge-9"
    assert payload.filename == "faq.xlsx"
    assert payload.file_data == _ENCODED
    assert payload.mode == "replace"
    assert payload.dry_run is True


def test_payload_defaults_optional_fields() -> None:
    """The optional fields default to their no-op values."""
    payload = FAQImportPayload.model_validate(
        {"tenant_id": 1, "kb_id": "kb-1", "filename": "faq.csv", "file_data": _ENCODED}
    )
    assert payload.task_id == ""
    assert payload.knowledge_id == ""
    assert payload.mode == "append"
    assert payload.dry_run is False


def test_payload_ignores_upstream_tracing_fields() -> None:
    """Tracing / idempotency / initiator fields are accepted but ignored."""
    payload = FAQImportPayload.model_validate(
        {
            "tenant_id": 1,
            "kb_id": "kb-1",
            "filename": "faq.csv",
            "file_data": _ENCODED,
            "lf_trace_id": "trace-1",
            "lf_traceparent": "00-abc-def-01",
            "enqueued_at": 1710000000,
            "instance_id": "instance-1",
            "initiator": {"user_id": "u-1", "role": "admin"},
        }
    )
    assert payload.tenant_id == 1
    assert payload.kb_id == "kb-1"


def test_payload_rejects_missing_kb_id() -> None:
    """The knowledge-base id is mandatory."""
    with pytest.raises(ValidationError):
        FAQImportPayload.model_validate(
            {"tenant_id": 1, "filename": "faq.csv", "file_data": _ENCODED}
        )


def test_payload_rejects_missing_file_data() -> None:
    """The encoded file bytes are mandatory."""
    with pytest.raises(ValidationError):
        FAQImportPayload.model_validate({"tenant_id": 1, "kb_id": "kb-1", "filename": "faq.csv"})


def test_payload_rejects_missing_filename() -> None:
    """The file name is mandatory (the parser sniffs on it)."""
    with pytest.raises(ValidationError):
        FAQImportPayload.model_validate({"tenant_id": 1, "kb_id": "kb-1", "file_data": _ENCODED})


def test_payload_is_frozen() -> None:
    """The payload model is immutable; mutations must raise."""
    payload = FAQImportPayload.model_validate(
        {"tenant_id": 1, "kb_id": "kb-1", "filename": "faq.csv", "file_data": _ENCODED}
    )
    with pytest.raises(ValidationError):
        payload.kb_id = "tampered"


# ── Worker dispatch ─────────────────────────────────────────────────


async def test_task_faq_import_delegates_to_core_runner(
    ctx: WorkerContext,
    valid_payload: dict[str, object],
    mock_runner: AsyncMock,
) -> None:
    """The handler decodes the file bytes and forwards them to the runner."""
    result = await task_faq_import(
        ctx,
        runner=cast(FAQImportRunner, mock_runner),
        **valid_payload,  # type: ignore[arg-type]
    )

    mock_runner.run.assert_awaited_once_with(
        file_data=_CSV_BYTES,
        filename="faq.csv",
        tenant_id=42,
        knowledge_base_id="kb-1",
        knowledge_id="knowledge-1",
        mode="append",
        dry_run=False,
    )
    assert result["task_id"] == mock_runner.run.return_value.task_id
    assert result["status"] == "completed"
    assert result["success_count"] == 2


async def test_task_faq_import_uses_defaults_for_optional_fields(
    ctx: WorkerContext,
    mock_runner: AsyncMock,
) -> None:
    """Omitted ``mode`` / ``dry_run`` / ``knowledge_id`` fall back."""
    await task_faq_import(
        ctx,
        runner=cast(FAQImportRunner, mock_runner),
        tenant_id=1,  # type: ignore[arg-type]
        kb_id="kb-1",  # type: ignore[arg-type]
        filename="faq.csv",  # type: ignore[arg-type]
        file_data=_ENCODED,  # type: ignore[arg-type]
    )

    mock_runner.run.assert_awaited_once_with(
        file_data=_CSV_BYTES,
        filename="faq.csv",
        tenant_id=1,
        knowledge_base_id="kb-1",
        knowledge_id="",
        mode="append",
        dry_run=False,
    )


async def test_task_faq_import_rejects_invalid_payload(
    ctx: WorkerContext,
    mock_runner: AsyncMock,
) -> None:
    """A payload missing required fields surfaces as ``ValidationError``."""
    with pytest.raises(ValidationError):
        await task_faq_import(
            ctx,
            runner=cast(FAQImportRunner, mock_runner),
            tenant_id=1,  # type: ignore[arg-type]
            kb_id="kb-1",  # type: ignore[arg-type]
        )


async def test_task_faq_import_rejects_malformed_base64(
    ctx: WorkerContext,
    valid_payload: dict[str, object],
    mock_runner: AsyncMock,
) -> None:
    """A payload with non-base64 ``file_data`` fails fast, before the core."""
    payload = {**valid_payload, "file_data": "!!!not-base64!!!"}
    with pytest.raises(binascii.Error):
        await task_faq_import(
            ctx,
            runner=cast(FAQImportRunner, mock_runner),
            **payload,  # type: ignore[arg-type]
        )

    mock_runner.run.assert_not_awaited()


async def test_task_faq_import_propagates_core_errors(
    ctx: WorkerContext,
    valid_payload: dict[str, object],
    mock_runner: AsyncMock,
) -> None:
    """Errors raised by the core runner surface to the worker caller."""
    mock_runner.run.side_effect = RuntimeError("import exploded")
    with pytest.raises(RuntimeError, match="import exploded"):
        await task_faq_import(
            ctx,
            runner=cast(FAQImportRunner, mock_runner),
            **valid_payload,  # type: ignore[arg-type]
        )


# ── Core seam without injected runner ───────────────────────────────


async def test_process_faq_import_raises_without_runner() -> None:
    """An uninjected seam raises so a miswired import is never silent."""
    with pytest.raises(NotImplementedError, match="FAQImportRunner"):
        await process_faq_import(
            file_data=_CSV_BYTES,
            filename="faq.csv",
            tenant_id=1,
            knowledge_base_id="kb-1",
            knowledge_id="knowledge-1",
        )


async def test_process_faq_import_delegates_to_injected_runner(
    mock_runner: AsyncMock,
    import_progress: FAQImportTaskProgress,
) -> None:
    """An injected runner runs the import and its progress is serialised."""
    result = await process_faq_import(
        file_data=_CSV_BYTES,
        filename="faq.csv",
        tenant_id=42,
        knowledge_base_id="kb-1",
        knowledge_id="knowledge-1",
        mode="replace",
        dry_run=True,
        runner=cast(FAQImportRunner, mock_runner),
    )

    mock_runner.run.assert_awaited_once_with(
        file_data=_CSV_BYTES,
        filename="faq.csv",
        tenant_id=42,
        knowledge_base_id="kb-1",
        knowledge_id="knowledge-1",
        mode="replace",
        dry_run=True,
    )
    assert result["status"] == import_progress.status
    assert result["import_mode"] == import_progress.import_mode
    assert result["kb_id"] == "kb-1"


# ── Re-registration guard ──────────────────────────────────────────


@pytest.fixture(autouse=True)
def _restore_registry() -> Iterator[None]:
    """Snapshot the registry around each test to avoid cross-test pollution.

    The ``register_task`` decorator mutates a module-level dict. Tests
    that import the module leave the registration in place, but a
    future test that re-registers under the same name would silently
    overwrite it. The fixture is defensive — a no-op today.
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


# ── Patchability guard ──────────────────────────────────────────────


async def test_task_faq_import_patchable_core_seam(
    ctx: WorkerContext,
    valid_payload: dict[str, object],
) -> None:
    """The core seam is patchable for callers wiring the worker later."""
    with patch.object(
        faq_import_module,
        "process_faq_import",
        new_callable=AsyncMock,
        return_value={"status": "dispatched"},
    ) as mock:
        result = await task_faq_import(ctx, **valid_payload)  # type: ignore[arg-type]

    mock.assert_awaited_once_with(
        file_data=_CSV_BYTES,
        filename="faq.csv",
        tenant_id=42,
        knowledge_base_id="kb-1",
        knowledge_id="knowledge-1",
        mode="append",
        dry_run=False,
        runner=None,
    )
    assert result == {"status": "dispatched"}
