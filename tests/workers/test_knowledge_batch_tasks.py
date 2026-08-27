"""Unit tests for the knowledge batch worker tasks.

Covers the worker-side surface of ``knowledge:list_delete``,
``knowledge:list_reparse``, and ``knowledge:move``: each handler is
registered under its upstream task name, payloads parse cleanly into the
contract models (mandatory fields, defaults, ignored tracing fields,
frozen immutability), the handlers delegate to the wiring-provided
runner seams with the parsed arguments, and the seams refuse to run when
the wiring has not injected a runner so a miswired worker fails loudly.
The batch loops (delete once / reparse per item / move per item) are
exercised through mocked runners, so no real database is needed.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from arq.connections import ArqRedis
from pydantic import ValidationError

from src.core.contracts.knowledge import Knowledge
from src.workers.base import WorkerContext
from src.workers.registry import get_task
from src.workers.tasks import knowledge_list_delete as delete_module
from src.workers.tasks import knowledge_list_reparse as reparse_module
from src.workers.tasks import knowledge_move as move_module
from src.workers.tasks.knowledge_list_delete import (
    KnowledgeListDeletePayload,
    KnowledgeListDeleteRunner,
    process_knowledge_list_delete,
    task_knowledge_list_delete,
)
from src.workers.tasks.knowledge_list_reparse import (
    KnowledgeListReparsePayload,
    KnowledgeReparseRunner,
    process_knowledge_list_reparse,
    task_knowledge_list_reparse,
)
from src.workers.tasks.knowledge_move import (
    KnowledgeMovePayload,
    KnowledgeMoveRunner,
    process_knowledge_move,
    task_knowledge_move,
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


def make_knowledge(knowledge_id: str) -> Knowledge:
    """Build a wire-shaped knowledge contract record."""
    return Knowledge(
        id=knowledge_id,
        tenant_id=42,
        knowledge_base_id="kb-dst",
        type="dataset",
        parse_status="pending",
        enable_status="disabled",
        created_at=datetime(2026, 3, 1, tzinfo=UTC),
        updated_at=datetime(2026, 3, 1, tzinfo=UTC),
    )


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def delete_payload() -> dict[str, object]:
    """A representative JSON payload for the delete task."""
    return {
        "tenant_id": 42,
        "knowledge_ids": ["k-1", "k-2", "k-3"],
    }


@pytest.fixture
def reparse_payload() -> dict[str, object]:
    """A representative JSON payload for the reparse task."""
    return {
        "tenant_id": 42,
        "knowledge_ids": ["k-1", "k-2", "k-3"],
        "process_config": {"enable_multimodel": True},
    }


@pytest.fixture
def move_payload() -> dict[str, object]:
    """A representative JSON payload for the move task."""
    return {
        "tenant_id": 42,
        "task_id": "kg_move-1",
        "knowledge_ids": ["k-1", "k-2", "k-3"],
        "source_kb_id": "kb-src",
        "target_kb_id": "kb-dst",
        "mode": "reuse_vectors",
    }


@pytest.fixture
def delete_runner() -> AsyncMock:
    """A mocked wiring seam for the batch delete."""
    return AsyncMock(return_value=3)


@pytest.fixture
def reparse_runner() -> AsyncMock:
    """A mocked wiring seam that successfully reparses every item."""
    runner = AsyncMock()
    runner.side_effect = lambda **_: make_knowledge("k")
    return runner


@pytest.fixture
def move_runner() -> AsyncMock:
    """A mocked wiring seam that successfully moves every item."""
    runner = AsyncMock()
    runner.side_effect = lambda **_: make_knowledge("k")
    return runner


# ── Registration ────────────────────────────────────────────────────


def test_knowledge_list_delete_registered_under_task_name() -> None:
    """The delete handler is registered under the upstream task name."""
    assert get_task("knowledge:list_delete") is task_knowledge_list_delete


def test_knowledge_list_reparse_registered_under_task_name() -> None:
    """The reparse handler is registered under the upstream task name."""
    assert get_task("knowledge:list_reparse") is task_knowledge_list_reparse


def test_knowledge_move_registered_under_task_name() -> None:
    """The move handler is registered under the upstream task name."""
    assert get_task("knowledge:move") is task_knowledge_move


def test_knowledge_batch_unknown_name_returns_none() -> None:
    """A typo in a registered name must not silently resolve."""
    assert get_task("knowledge:list_delete_") is None
    assert get_task("knowledge:list_repars") is None
    assert get_task("knowledge:moves") is None


# ── Delete payload contract ─────────────────────────────────────────


def test_delete_payload_parses_full() -> None:
    """A complete delete payload round-trips through the contract model."""
    payload = KnowledgeListDeletePayload.model_validate(
        {"tenant_id": 7, "knowledge_ids": ["k-1", "k-2"]}
    )
    assert payload.tenant_id == 7
    assert payload.knowledge_ids == ["k-1", "k-2"]


def test_delete_payload_ignores_upstream_tracing_fields() -> None:
    """Tracing / initiator fields are accepted but not modelled."""
    payload = KnowledgeListDeletePayload.model_validate(
        {
            "tenant_id": 1,
            "knowledge_ids": [],
            "lf_trace_id": "trace-1",
            "lf_traceparent": "00-abc-def-01",
            "initiator": {"user_id": "u-1", "role": "admin"},
        }
    )
    assert payload.tenant_id == 1
    assert payload.knowledge_ids == []


def test_delete_payload_rejects_missing_tenant_id() -> None:
    """The tenant id is mandatory."""
    with pytest.raises(ValidationError):
        KnowledgeListDeletePayload.model_validate({"knowledge_ids": ["k-1"]})


def test_delete_payload_rejects_missing_knowledge_ids() -> None:
    """The knowledge id list is mandatory."""
    with pytest.raises(ValidationError):
        KnowledgeListDeletePayload.model_validate({"tenant_id": 1})


def test_delete_payload_is_frozen() -> None:
    """The payload model is immutable; mutations must raise."""
    payload = KnowledgeListDeletePayload.model_validate({"tenant_id": 1, "knowledge_ids": ["k-1"]})
    with pytest.raises(ValidationError):
        payload.tenant_id = 2  # type: ignore[misc]


# ── Reparse payload contract ────────────────────────────────────────


def test_reparse_payload_parses_full() -> None:
    """A complete reparse payload round-trips through the contract model."""
    payload = KnowledgeListReparsePayload.model_validate(
        {
            "tenant_id": 7,
            "knowledge_ids": ["k-1"],
            "process_config": {"enable_multimodel": True},
        }
    )
    assert payload.tenant_id == 7
    assert payload.knowledge_ids == ["k-1"]
    assert payload.process_config == {"enable_multimodel": True}


def test_reparse_payload_defaults_optional_process_config() -> None:
    """An omitted ``process_config`` falls back to ``None``."""
    payload = KnowledgeListReparsePayload.model_validate({"tenant_id": 1, "knowledge_ids": ["k-1"]})
    assert payload.process_config is None


def test_reparse_payload_ignores_upstream_tracing_fields() -> None:
    """Tracing / initiator fields are accepted but not modelled."""
    payload = KnowledgeListReparsePayload.model_validate(
        {
            "tenant_id": 1,
            "knowledge_ids": ["k-1"],
            "lf_trace_id": "trace-1",
            "initiator": {"user_id": "u-1", "role": "admin"},
        }
    )
    assert payload.tenant_id == 1
    assert payload.knowledge_ids == ["k-1"]
    assert payload.process_config is None


def test_reparse_payload_rejects_missing_knowledge_ids() -> None:
    """The knowledge id list is mandatory."""
    with pytest.raises(ValidationError):
        KnowledgeListReparsePayload.model_validate({"tenant_id": 1})


def test_reparse_payload_is_frozen() -> None:
    """The payload model is immutable; mutations must raise."""
    payload = KnowledgeListReparsePayload.model_validate({"tenant_id": 1, "knowledge_ids": ["k-1"]})
    with pytest.raises(ValidationError):
        payload.knowledge_ids = ["k-2"]  # type: ignore[misc]


# ── Move payload contract ───────────────────────────────────────────


def test_move_payload_parses_full() -> None:
    """A complete move payload round-trips through the contract model."""
    payload = KnowledgeMovePayload.model_validate(
        {
            "tenant_id": 7,
            "task_id": "kg_move-9",
            "knowledge_ids": ["k-1"],
            "source_kb_id": "kb-src",
            "target_kb_id": "kb-dst",
            "mode": "reparse",
        }
    )
    assert payload.tenant_id == 7
    assert payload.task_id == "kg_move-9"
    assert payload.knowledge_ids == ["k-1"]
    assert payload.source_kb_id == "kb-src"
    assert payload.target_kb_id == "kb-dst"
    assert payload.mode == "reparse"


def test_move_payload_defaults_task_id() -> None:
    """An omitted ``task_id`` falls back to blank."""
    payload = KnowledgeMovePayload.model_validate(
        {
            "tenant_id": 1,
            "knowledge_ids": ["k-1"],
            "source_kb_id": "kb-src",
            "target_kb_id": "kb-dst",
            "mode": "reuse_vectors",
        }
    )
    assert payload.task_id == ""


def test_move_payload_ignores_upstream_tracing_fields() -> None:
    """Tracing / initiator fields are accepted but not modelled."""
    payload = KnowledgeMovePayload.model_validate(
        {
            "tenant_id": 1,
            "knowledge_ids": ["k-1"],
            "source_kb_id": "kb-src",
            "target_kb_id": "kb-dst",
            "mode": "reparse",
            "lf_trace_id": "trace-1",
            "initiator": {"user_id": "u-1", "role": "admin"},
        }
    )
    assert payload.tenant_id == 1
    assert payload.mode == "reparse"


def test_move_payload_rejects_missing_source_kb_id() -> None:
    """The source knowledge-base id is mandatory."""
    with pytest.raises(ValidationError):
        KnowledgeMovePayload.model_validate(
            {
                "tenant_id": 1,
                "knowledge_ids": ["k-1"],
                "target_kb_id": "kb-dst",
                "mode": "reparse",
            }
        )


def test_move_payload_rejects_missing_mode() -> None:
    """The move mode is mandatory."""
    with pytest.raises(ValidationError):
        KnowledgeMovePayload.model_validate(
            {
                "tenant_id": 1,
                "knowledge_ids": ["k-1"],
                "source_kb_id": "kb-src",
                "target_kb_id": "kb-dst",
            }
        )


def test_move_payload_is_frozen() -> None:
    """The payload model is immutable; mutations must raise."""
    payload = KnowledgeMovePayload.model_validate(
        {
            "tenant_id": 1,
            "knowledge_ids": ["k-1"],
            "source_kb_id": "kb-src",
            "target_kb_id": "kb-dst",
            "mode": "reparse",
        }
    )
    with pytest.raises(ValidationError):
        payload.mode = "reuse_vectors"  # type: ignore[misc]


# ── Delete worker dispatch ──────────────────────────────────────────


async def test_task_knowledge_list_delete_delegates_to_runner(
    ctx: WorkerContext,
    delete_payload: dict[str, object],
    delete_runner: AsyncMock,
) -> None:
    """The delete handler parses and forwards the payload to the runner."""
    result = await task_knowledge_list_delete(
        ctx,
        runner=cast(KnowledgeListDeleteRunner, delete_runner),
        **delete_payload,  # type: ignore[arg-type]
    )

    delete_runner.assert_awaited_once_with(
        tenant_id=42,
        knowledge_ids=["k-1", "k-2", "k-3"],
    )
    assert result == {"deleted": 3}


async def test_task_knowledge_list_delete_rejects_invalid_payload(
    ctx: WorkerContext,
    delete_runner: AsyncMock,
) -> None:
    """A payload missing required fields surfaces as ``ValidationError``."""
    with pytest.raises(ValidationError):
        await task_knowledge_list_delete(
            ctx,
            runner=cast(KnowledgeListDeleteRunner, delete_runner),
            tenant_id=42,  # type: ignore[arg-type]
        )


async def test_task_knowledge_list_delete_propagates_runner_errors(
    ctx: WorkerContext,
    delete_payload: dict[str, object],
    delete_runner: AsyncMock,
) -> None:
    """Errors raised by the runner surface to the worker caller."""
    delete_runner.side_effect = RuntimeError("batch delete exploded")
    with pytest.raises(RuntimeError, match="batch delete exploded"):
        await task_knowledge_list_delete(
            ctx,
            runner=cast(KnowledgeListDeleteRunner, delete_runner),
            **delete_payload,  # type: ignore[arg-type]
        )


async def test_process_knowledge_list_delete_raises_without_runner() -> None:
    """An uninjected runner raises so a miswired delete is never silent."""
    with pytest.raises(NotImplementedError, match="wiring-provided runner"):
        await process_knowledge_list_delete(
            tenant_id=42,
            knowledge_ids=["k-1"],
        )


# ── Reparse worker dispatch ─────────────────────────────────────────


async def test_task_knowledge_list_reparse_delegates_per_item(
    ctx: WorkerContext,
    reparse_payload: dict[str, object],
    reparse_runner: AsyncMock,
) -> None:
    """The reparse handler forwards every id to the runner."""
    result = await task_knowledge_list_reparse(
        ctx,
        runner=cast(KnowledgeReparseRunner, reparse_runner),
        **reparse_payload,  # type: ignore[arg-type]
    )

    assert reparse_runner.await_count == 3
    reparse_runner.assert_awaited_with(
        tenant_id=42,
        knowledge_id="k-3",
        process_overrides={"enable_multimodel": True},
    )
    assert result == {"submitted": 3, "failed": 0}


async def test_task_knowledge_list_reparse_counts_partial_failures(
    ctx: WorkerContext,
    reparse_payload: dict[str, object],
) -> None:
    """A failed item is counted and does not abort the rest of the batch."""
    calls: list[str] = []

    async def flaky_runner(*, knowledge_id: str, **kwargs: object) -> Knowledge:
        calls.append(knowledge_id)
        if knowledge_id == "k-2":
            raise RuntimeError("bad document")
        return make_knowledge(knowledge_id)

    result = await task_knowledge_list_reparse(
        ctx,
        runner=cast(KnowledgeReparseRunner, flaky_runner),
        **reparse_payload,  # type: ignore[arg-type]
    )

    assert calls == ["k-1", "k-2", "k-3"]
    assert result == {"submitted": 2, "failed": 1}


async def test_task_knowledge_list_reparse_counts_all_failures(
    ctx: WorkerContext,
    reparse_payload: dict[str, object],
) -> None:
    """When every item fails the batch completes with a full failed tally."""

    async def failing_runner(**kwargs: object) -> Knowledge:
        raise RuntimeError("boom")

    result = await task_knowledge_list_reparse(
        ctx,
        runner=cast(KnowledgeReparseRunner, failing_runner),
        **reparse_payload,  # type: ignore[arg-type]
    )

    assert result == {"submitted": 0, "failed": 3}


async def test_task_knowledge_list_reparse_rejects_invalid_payload(
    ctx: WorkerContext,
    reparse_runner: AsyncMock,
) -> None:
    """A payload missing required fields surfaces as ``ValidationError``."""
    with pytest.raises(ValidationError):
        await task_knowledge_list_reparse(
            ctx,
            runner=cast(KnowledgeReparseRunner, reparse_runner),
            tenant_id=42,  # type: ignore[arg-type]
        )


async def test_process_knowledge_list_reparse_raises_without_runner() -> None:
    """An uninjected runner raises so a miswired reparse is never silent."""
    with pytest.raises(NotImplementedError, match="wiring-provided runner"):
        await process_knowledge_list_reparse(
            tenant_id=42,
            knowledge_ids=["k-1"],
        )


async def test_process_knowledge_list_reparse_omits_process_overrides(
    reparse_runner: AsyncMock,
) -> None:
    """An omitted process config is forwarded as ``None``."""
    await process_knowledge_list_reparse(
        tenant_id=42,
        knowledge_ids=["k-1"],
        runner=cast(KnowledgeReparseRunner, reparse_runner),
    )
    reparse_runner.assert_awaited_once_with(
        tenant_id=42,
        knowledge_id="k-1",
        process_overrides=None,
    )


# ── Move worker dispatch ────────────────────────────────────────────


async def test_task_knowledge_move_delegates_per_item(
    ctx: WorkerContext,
    move_payload: dict[str, object],
    move_runner: AsyncMock,
) -> None:
    """The move handler forwards every id to the runner."""
    result = await task_knowledge_move(
        ctx,
        runner=cast(KnowledgeMoveRunner, move_runner),
        **move_payload,  # type: ignore[arg-type]
    )

    assert move_runner.await_count == 3
    move_runner.assert_awaited_with(
        tenant_id=42,
        knowledge_id="k-3",
        source_kb_id="kb-src",
        target_kb_id="kb-dst",
        mode="reuse_vectors",
    )
    assert result == {"processed": 3, "failed": 0}


async def test_task_knowledge_move_counts_partial_failures(
    ctx: WorkerContext,
    move_payload: dict[str, object],
) -> None:
    """A failed item is counted and does not abort the rest of the batch."""
    calls: list[str] = []

    async def flaky_runner(*, knowledge_id: str, **kwargs: object) -> Knowledge:
        calls.append(knowledge_id)
        if knowledge_id == "k-1":
            raise RuntimeError("incompatible kb")
        return make_knowledge(knowledge_id)

    result = await task_knowledge_move(
        ctx,
        runner=cast(KnowledgeMoveRunner, flaky_runner),
        **move_payload,  # type: ignore[arg-type]
    )

    assert calls == ["k-1", "k-2", "k-3"]
    assert result == {"processed": 2, "failed": 1}


async def test_task_knowledge_move_counts_all_failures(
    ctx: WorkerContext,
    move_payload: dict[str, object],
) -> None:
    """When every item fails the batch completes with a full failed tally."""

    async def failing_runner(**kwargs: object) -> Knowledge:
        raise RuntimeError("boom")

    result = await task_knowledge_move(
        ctx,
        runner=cast(KnowledgeMoveRunner, failing_runner),
        **move_payload,  # type: ignore[arg-type]
    )

    assert result == {"processed": 0, "failed": 3}


async def test_task_knowledge_move_rejects_invalid_payload(
    ctx: WorkerContext,
    move_runner: AsyncMock,
) -> None:
    """A payload missing required fields surfaces as ``ValidationError``."""
    with pytest.raises(ValidationError):
        await task_knowledge_move(
            ctx,
            runner=cast(KnowledgeMoveRunner, move_runner),
            tenant_id=42,  # type: ignore[arg-type]
        )


async def test_process_knowledge_move_raises_without_runner() -> None:
    """An uninjected runner raises so a miswired move is never silent."""
    with pytest.raises(NotImplementedError, match="wiring-provided runner"):
        await process_knowledge_move(
            tenant_id=42,
            knowledge_ids=["k-1"],
            source_kb_id="kb-src",
            target_kb_id="kb-dst",
            mode="reuse_vectors",
        )


# ── Patchability guard ──────────────────────────────────────────────


async def test_task_knowledge_list_delete_patchable_seam(
    ctx: WorkerContext,
    delete_payload: dict[str, object],
) -> None:
    """The core seam is patchable for callers wiring the worker later."""
    with patch.object(
        delete_module,
        "process_knowledge_list_delete",
        new_callable=AsyncMock,
        return_value={"deleted": 1},
    ):
        result = await task_knowledge_list_delete(
            ctx,
            **delete_payload,  # type: ignore[arg-type]
        )
    assert result == {"deleted": 1}


async def test_task_knowledge_list_reparse_patchable_seam(
    ctx: WorkerContext,
    reparse_payload: dict[str, object],
) -> None:
    """The reparse seam is patchable for callers wiring the worker later."""
    with patch.object(
        reparse_module,
        "process_knowledge_list_reparse",
        new_callable=AsyncMock,
        return_value={"submitted": 2, "failed": 1},
    ):
        result = await task_knowledge_list_reparse(
            ctx,
            **reparse_payload,  # type: ignore[arg-type]
        )
    assert result == {"submitted": 2, "failed": 1}


async def test_task_knowledge_move_patchable_seam(
    ctx: WorkerContext,
    move_payload: dict[str, object],
) -> None:
    """The move seam is patchable for callers wiring the worker later."""
    with patch.object(
        move_module,
        "process_knowledge_move",
        new_callable=AsyncMock,
        return_value={"processed": 2, "failed": 1},
    ):
        result = await task_knowledge_move(
            ctx,
            **move_payload,  # type: ignore[arg-type]
        )
    assert result == {"processed": 2, "failed": 1}


# ── Re-registration guard ───────────────────────────────────────────


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

    baseline = dict(registry_module.all_tasks())
    yield
    current = registry_module.all_tasks()
    for name in list(current.keys()):
        if name not in baseline:
            current.pop(name, None)
    for name, handler in baseline.items():
        current[name] = handler
