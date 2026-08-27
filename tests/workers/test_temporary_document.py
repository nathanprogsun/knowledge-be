"""Unit tests for the ``temporary_document:process`` ARQ task.

Covers registration under the upstream-compatible task name, payload
decoding via the core wire type, and the result-dict shape that
callers (and retries) consume. The handler is a thin dispatcher, so
the test exercises the entry point without touching the database.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import cast

import pytest
from arq.connections import ArqRedis

from src.workers.base import WorkerContext
from src.workers.registry import all_tasks, get_task
from src.workers.settings import reset_worker_settings_cache
from src.workers.tasks.temporary_document import (
    TASK_NAME,
    task_temporary_document,
)


def make_ctx() -> WorkerContext:
    """Build a context dict matching what ARQ passes to tasks/hooks."""
    return WorkerContext(
        redis=cast(ArqRedis, None),
        job_id="job-temp-doc",
        job_try=1,
        enqueue_time=datetime.now(UTC),
        score=0,
    )


@pytest.fixture(autouse=True)
def _reset_worker_settings_cache() -> Iterator[None]:
    reset_worker_settings_cache()
    yield
    reset_worker_settings_cache()


# ── Registration ────────────────────────────────────────────────────


def test_task_registered_under_upstream_name() -> None:
    """The task lands in the registry under the upstream name."""
    handler = get_task(TASK_NAME)
    assert handler is task_temporary_document


def test_task_name_matches_upstream_type() -> None:
    """The constant matches the upstream task type exactly."""
    assert TASK_NAME == "temporary_document:process"


def test_task_visible_via_all_tasks() -> None:
    """The task is included in the registry snapshot."""
    assert TASK_NAME in all_tasks()
    assert all_tasks()[TASK_NAME] is task_temporary_document


# ── Payload decoding ────────────────────────────────────────────────


async def test_handler_decodes_valid_payload() -> None:
    """A well-formed payload yields the parsed scope in the result."""
    ctx = make_ctx()
    result = await task_temporary_document(
        ctx,
        tenant_id=42,
        document_id="doc-abc",
    )
    assert result == {
        "tenant_id": 42,
        "document_id": "doc-abc",
        "status": "dispatched",
    }


async def test_handler_accepts_string_tenant_id() -> None:
    """Numeric strings are coerced to int via the wire schema."""
    ctx = make_ctx()
    result = await task_temporary_document(
        ctx,
        tenant_id="7",
        document_id="doc-xyz",
    )
    assert result["tenant_id"] == 7
    assert result["document_id"] == "doc-xyz"


async def test_handler_accepts_string_payload() -> None:
    """Raw string payload values flow through the same wire schema."""
    ctx = make_ctx()
    result = await task_temporary_document(
        ctx,
        tenant_id="100",
        document_id="doc-str",
    )
    assert result == {
        "tenant_id": 100,
        "document_id": "doc-str",
        "status": "dispatched",
    }


# ── Validation errors ───────────────────────────────────────────────


async def test_handler_rejects_missing_tenant_id() -> None:
    """A payload without ``tenant_id`` fails the wire schema."""
    ctx = make_ctx()
    with pytest.raises(ValueError):
        await task_temporary_document(ctx, document_id="doc-1")


async def test_handler_rejects_missing_document_id() -> None:
    """A payload without ``document_id`` fails the wire schema."""
    ctx = make_ctx()
    with pytest.raises(ValueError):
        await task_temporary_document(ctx, tenant_id=1)


async def test_handler_rejects_empty_payload() -> None:
    """An empty payload raises — both fields are required."""
    ctx = make_ctx()
    with pytest.raises(ValueError):
        await task_temporary_document(ctx)
