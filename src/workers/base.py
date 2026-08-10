"""Base worker class with lifecycle hooks.

``BaseWorker`` wraps an ARQ ``Worker``: subclasses supply the task
functions, the base class wires settings, functions, and the
``startup`` / ``shutdown`` lifecycle hooks into the ARQ worker, and
exposes ``start`` / ``stop`` for process management.

The context dict ARQ passes to tasks and hooks is typed as
:class:`WorkerContext` — a ``TypedDict`` so handlers get concrete key
types without resorting to ``Any``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TypedDict, cast

from arq.connections import ArqRedis
from arq.typing import StartupShutdown
from arq.worker import Function, Worker

from src.workers.settings import WorkerSettings


class WorkerContext(TypedDict):
    """Context dict ARQ passes to tasks and lifecycle hooks.

    ``redis`` is always present (the ARQ pool). The job-scoped keys
    (``job_id``, ``job_try``, ``enqueue_time``, ``score``) are present
    only while a task runs.
    """

    redis: ArqRedis
    job_id: str
    job_try: int
    enqueue_time: datetime
    score: int


class BaseWorker(ABC):
    """Abstract ARQ worker with lifecycle hooks.

    Subclasses implement ``functions``; the base class builds the ARQ
    ``Worker`` from settings + functions and wires the lifecycle hooks.
    """

    def __init__(self, settings: WorkerSettings) -> None:
        self.settings = settings
        self._worker: Worker | None = None

    @property
    @abstractmethod
    def functions(self) -> list[Function]:
        """Task functions this worker serves."""

    async def startup(self, ctx: WorkerContext) -> None:  # noqa: B027
        """Lifecycle hook invoked once when the worker starts.

        Optional extension point — subclasses override to run setup
        (e.g. warm caches) before the worker begins polling.
        """

    async def shutdown(self, ctx: WorkerContext) -> None:  # noqa: B027
        """Lifecycle hook invoked once when the worker stops.

        Optional extension point — subclasses override to run teardown
        (e.g. flush buffers) after the worker stops polling.
        """

    def build(self) -> Worker:
        """Construct the underlying ARQ Worker from settings + functions."""
        return Worker(
            functions=self.functions,
            queue_name=self.settings.queue_name,
            redis_settings=self.settings.redis_settings,
            max_jobs=self.settings.max_jobs,
            job_timeout=self.settings.job_timeout,
            health_check_interval=self.settings.health_check_interval,
            max_tries=self.settings.max_tries,
            poll_delay=self.settings.poll_delay,
            keep_result=self.settings.keep_result,
            burst=self.settings.burst,
            handle_signals=self.settings.handle_signals,
            log_results=self.settings.log_results,
            on_startup=cast(StartupShutdown, self.startup),
            on_shutdown=cast(StartupShutdown, self.shutdown),
        )

    async def start(self) -> None:
        """Build the ARQ worker and run it until stopped."""
        if self._worker is not None:
            raise RuntimeError("worker already started")
        self._worker = self.build()
        await self._worker.async_run()

    async def stop(self) -> None:
        """Close the ARQ worker and release its Redis pool."""
        if self._worker is None:
            return
        await self._worker.close()
        self._worker = None


__all__ = ["BaseWorker", "WorkerContext"]
