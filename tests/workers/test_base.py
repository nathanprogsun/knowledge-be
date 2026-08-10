"""Unit tests for the ARQ worker base infrastructure.

Covers the task registry (registration, lookup, ARQ function
conversion), the base worker lifecycle hooks, and settings parsing.
No real Redis is used — the ARQ ``Worker`` is only *built* (never run)
and the lifecycle hooks are invoked directly.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import cast

import pytest
from arq.connections import ArqRedis
from arq.worker import Function

from src.workers.base import BaseWorker, WorkerContext
from src.workers.registry import (
    JsonValue,
    all_functions,
    all_tasks,
    get_task,
    register_task,
)
from src.workers.settings import (
    WorkerSettings,
    get_worker_settings,
    reset_worker_settings_cache,
)


@register_task("test_task")
async def sample_task(ctx: WorkerContext, **payload: JsonValue) -> None:
    """Test task handler used to exercise the registry and worker."""


def make_ctx() -> WorkerContext:
    """Build a context dict matching what ARQ passes to tasks/hooks."""
    return WorkerContext(
        redis=cast(ArqRedis, None),
        job_id="job-1",
        job_try=1,
        enqueue_time=datetime.now(UTC),
        score=0,
    )


@pytest.fixture(autouse=True)
def _reset_worker_settings_cache() -> Iterator[None]:
    reset_worker_settings_cache()
    yield
    reset_worker_settings_cache()


class RecordingWorker(BaseWorker):
    """Test worker that records lifecycle hook invocations."""

    def __init__(self, settings: WorkerSettings) -> None:
        super().__init__(settings)
        self.startup_calls: list[WorkerContext] = []
        self.shutdown_calls: list[WorkerContext] = []

    @property
    def functions(self) -> list[Function]:
        return all_functions()

    async def startup(self, ctx: WorkerContext) -> None:
        self.startup_calls.append(ctx)

    async def shutdown(self, ctx: WorkerContext) -> None:
        self.shutdown_calls.append(ctx)


# ── Registry ────────────────────────────────────────────────────────


def test_register_task_registers_handler() -> None:
    assert get_task("test_task") is sample_task


def test_get_task_unknown_returns_none() -> None:
    assert get_task("no_such_task") is None


def test_all_tasks_returns_copy() -> None:
    tasks = all_tasks()
    assert tasks["test_task"] is sample_task
    tasks["mutated"] = sample_task
    assert "mutated" not in all_tasks()


def test_all_functions_builds_arq_functions() -> None:
    functions = all_functions()
    names = {f.name for f in functions}
    assert "test_task" in names
    assert all(f.coroutine is not None for f in functions)


# ── Lifecycle hooks ─────────────────────────────────────────────────


def test_build_wires_lifecycle_hooks() -> None:
    worker = RecordingWorker(WorkerSettings(handle_signals=False))
    built = worker.build()
    assert built.on_startup == worker.startup
    assert built.on_shutdown == worker.shutdown


def test_build_includes_registry_functions() -> None:
    worker = RecordingWorker(WorkerSettings(handle_signals=False))
    built = worker.build()
    assert "test_task" in built.functions


async def test_startup_hook_invoked() -> None:
    worker = RecordingWorker(WorkerSettings(handle_signals=False))
    ctx = make_ctx()
    await worker.startup(ctx)
    assert worker.startup_calls == [ctx]


async def test_shutdown_hook_invoked() -> None:
    worker = RecordingWorker(WorkerSettings(handle_signals=False))
    ctx = make_ctx()
    await worker.shutdown(ctx)
    assert worker.shutdown_calls == [ctx]


async def test_start_when_already_started_raises() -> None:
    worker = RecordingWorker(WorkerSettings(handle_signals=False))
    worker._worker = worker.build()
    with pytest.raises(RuntimeError, match="already started"):
        await worker.start()


async def test_stop_when_not_started_is_noop() -> None:
    worker = RecordingWorker(WorkerSettings(handle_signals=False))
    await worker.stop()
    assert worker._worker is None


# ── Settings ────────────────────────────────────────────────────────


def test_worker_settings_defaults() -> None:
    settings = WorkerSettings()
    assert settings.redis_url == "redis://localhost:6379"
    assert settings.queue_name == "arq:queue"
    assert settings.max_jobs == 10
    assert settings.job_timeout == 300
    assert settings.health_check_interval == 3600
    assert settings.max_tries == 5
    assert settings.poll_delay == 0.5
    assert settings.keep_result == 3600
    assert settings.max_connections == 10
    assert settings.burst is False
    assert settings.handle_signals is True
    assert settings.log_results is True


def test_worker_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKER_REDIS_URL", "redis://cache:6380/2")
    monkeypatch.setenv("WORKER_QUEUE_NAME", "kb:queue")
    monkeypatch.setenv("WORKER_MAX_JOBS", "4")
    settings = WorkerSettings()
    assert settings.redis_url == "redis://cache:6380/2"
    assert settings.queue_name == "kb:queue"
    assert settings.max_jobs == 4


def test_redis_settings_derived_from_url() -> None:
    settings = WorkerSettings(
        redis_url="redis://user:pass@cache:6380/3",
        max_connections=7,
        conn_timeout=2,
    )
    rs = settings.redis_settings
    assert rs.host == "cache"
    assert rs.port == 6380
    assert rs.database == 3
    assert rs.username == "user"
    assert rs.password == "pass"
    assert rs.max_connections == 7
    assert rs.conn_timeout == 2


def test_get_worker_settings_singleton() -> None:
    first = get_worker_settings()
    second = get_worker_settings()
    assert first is second
