"""Unit tests for the ``manual_process`` worker task.

Covers the worker-side surface: the registered handler is the expected
function, the payload parses cleanly into the contract model, the
handler delegates to the core seam with the parsed arguments, and the
core seam raises ``NotImplementedError`` until the core implementation
arrives. The core dispatch is exercised through a patched core seam so
no real database or pipeline is needed.
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
from src.workers.tasks import manual_process as manual_process_module
from src.workers.tasks.manual_process import (
    ManualProcessPayload,
    manual_process,
    process_document_manual,
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
def valid_payload() -> dict[str, object]:
    """A representative JSON payload for the manual-process task."""
    return {
        "request_id": "req-abc",
        "tenant_id": 42,
        "knowledge_id": "doc-1",
        "knowledge_base_id": "kb-1",
        "content": "# Hello\n\nWorld",
        "need_cleanup": True,
    }


# ── Registration ────────────────────────────────────────────────────


def test_manual_process_registered_under_task_name() -> None:
    """The handler is registered under the upstream task type name."""
    assert get_task("manual_process") is manual_process


def test_manual_process_unknown_name_returns_none() -> None:
    """A typo in the registered name must not silently resolve."""
    assert get_task("manual_procses") is None


# ── Payload contract ────────────────────────────────────────────────


def test_payload_parses_full() -> None:
    """A complete payload round-trips through the contract model."""
    payload = ManualProcessPayload.model_validate(
        {
            "request_id": "req-1",
            "tenant_id": 7,
            "knowledge_id": "doc-1",
            "knowledge_base_id": "kb-1",
            "content": "hello",
            "need_cleanup": True,
        }
    )
    assert payload.request_id == "req-1"
    assert payload.tenant_id == 7
    assert payload.knowledge_id == "doc-1"
    assert payload.knowledge_base_id == "kb-1"
    assert payload.content == "hello"
    assert payload.need_cleanup is True


def test_payload_defaults_optional_fields() -> None:
    """``request_id`` and ``need_cleanup`` default sensibly."""
    payload = ManualProcessPayload.model_validate(
        {
            "tenant_id": 1,
            "knowledge_id": "doc-1",
            "knowledge_base_id": "kb-1",
            "content": "hello",
        }
    )
    assert payload.request_id == ""
    assert payload.need_cleanup is False


def test_payload_rejects_missing_tenant_id() -> None:
    """The tenant id is mandatory."""
    with pytest.raises(ValidationError):
        ManualProcessPayload.model_validate(
            {
                "knowledge_id": "doc-1",
                "knowledge_base_id": "kb-1",
                "content": "hello",
            }
        )


def test_payload_rejects_missing_knowledge_id() -> None:
    """The knowledge id is mandatory."""
    with pytest.raises(ValidationError):
        ManualProcessPayload.model_validate(
            {
                "tenant_id": 1,
                "knowledge_base_id": "kb-1",
                "content": "hello",
            }
        )


def test_payload_rejects_missing_knowledge_base_id() -> None:
    """The knowledge base id is mandatory."""
    with pytest.raises(ValidationError):
        ManualProcessPayload.model_validate(
            {
                "tenant_id": 1,
                "knowledge_id": "doc-1",
                "content": "hello",
            }
        )


def test_payload_rejects_missing_content() -> None:
    """The Markdown content is mandatory."""
    with pytest.raises(ValidationError):
        ManualProcessPayload.model_validate(
            {
                "tenant_id": 1,
                "knowledge_id": "doc-1",
                "knowledge_base_id": "kb-1",
            }
        )


def test_payload_is_frozen() -> None:
    """The payload model is immutable; mutations must raise."""
    payload = ManualProcessPayload.model_validate(
        {
            "tenant_id": 1,
            "knowledge_id": "doc-1",
            "knowledge_base_id": "kb-1",
            "content": "hello",
        }
    )
    with pytest.raises(ValidationError):
        payload.content = "tampered"


# ── Worker dispatch ─────────────────────────────────────────────────


async def test_manual_process_delegates_to_core_seam(
    ctx: WorkerContext,
    valid_payload: dict[str, object],
) -> None:
    """The handler parses and forwards the payload to the core seam."""
    with patch.object(
        manual_process_module,
        "process_document_manual",
        new_callable=AsyncMock,
        return_value={"status": "completed"},
    ) as mock:
        result = await manual_process(ctx, **valid_payload)  # type: ignore[arg-type]

    mock.assert_awaited_once_with(
        tenant_id=42,
        knowledge_id="doc-1",
        knowledge_base_id="kb-1",
        content="# Hello\n\nWorld",
        need_cleanup=True,
        request_id="req-abc",
    )
    assert result == {"status": "completed"}


async def test_manual_process_uses_default_for_optional_fields(
    ctx: WorkerContext,
) -> None:
    """Omitted ``request_id`` and ``need_cleanup`` fall back to defaults."""
    payload = {
        "tenant_id": 1,
        "knowledge_id": "doc-1",
        "knowledge_base_id": "kb-1",
        "content": "hello",
    }
    with patch.object(
        manual_process_module,
        "process_document_manual",
        new_callable=AsyncMock,
        return_value={"status": "pending"},
    ) as mock:
        await manual_process(ctx, **payload)  # type: ignore[arg-type]

    mock.assert_awaited_once_with(
        tenant_id=1,
        knowledge_id="doc-1",
        knowledge_base_id="kb-1",
        content="hello",
        need_cleanup=False,
        request_id="",
    )


async def test_manual_process_rejects_invalid_payload(
    ctx: WorkerContext,
) -> None:
    """A payload missing required fields surfaces as ``ValidationError``."""
    with pytest.raises(ValidationError):
        await manual_process(ctx, tenant_id=1)


async def test_manual_process_propagates_core_errors(
    ctx: WorkerContext,
    valid_payload: dict[str, object],
) -> None:
    """Errors raised by the core seam surface to the worker caller."""
    with (
        patch.object(
            manual_process_module,
            "process_document_manual",
            new_callable=AsyncMock,
            side_effect=RuntimeError("pipeline exploded"),
        ),
        pytest.raises(RuntimeError, match="pipeline exploded"),
    ):
        await manual_process(ctx, **valid_payload)  # type: ignore[arg-type]


# ── Core seam placeholder ───────────────────────────────────────────


async def test_process_document_manual_raises_not_implemented() -> None:
    """The core seam is a placeholder until the core implementation lands."""
    with pytest.raises(NotImplementedError):
        await process_document_manual(
            tenant_id=1,
            knowledge_id="doc-1",
            knowledge_base_id="kb-1",
            content="hello",
            need_cleanup=False,
            request_id="",
        )


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
