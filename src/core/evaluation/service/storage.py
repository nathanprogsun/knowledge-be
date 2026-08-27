"""In-memory evaluation task storage — the run-progress side channel.

Mirrors the upstream ``evaluationMemoryStorage`` helper: a thread-safe
map of ``task_id`` → in-flight evaluation state. The SQL row is the
durable source of truth; this map carries the live counters and metric
bundle that callers poll between the ``pending`` → ``running`` →
``success`` / ``failed`` transitions.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime

from src.core.contracts.evaluation import EvalMetric, EvalTask
from src.core.evaluation.service.seams import (
    EvaluationParams,
    _UpdateResult,
)

logger = logging.getLogger(__name__)


class EvaluationMemoryStorage:
    """Thread-safe map of ``task_id`` → in-flight evaluation state."""

    def __init__(self) -> None:
        self._store: dict[str, EvalTask] = {}
        self._metric: dict[str, EvalMetric] = {}
        self._params: dict[str, EvaluationParams] = {}
        self._lock = threading.RLock()

    def register(
        self,
        *,
        task: EvalTask,
        params: EvaluationParams,
    ) -> None:
        """Insert a freshly created task; overwrites any prior entry."""
        with self._lock:
            self._store[task.id] = task
            self._params[task.id] = params
            self._metric.pop(task.id, None)
            logger.info("evaluation task %s registered", task.id)

    def get_task(self, task_id: str) -> EvalTask | None:
        """Return the live task row, or ``None`` when absent."""
        with self._lock:
            return self._store.get(task_id)

    def get_params(self, task_id: str) -> EvaluationParams | None:
        """Return the live pipeline params, or ``None`` when absent."""
        with self._lock:
            return self._params.get(task_id)

    def get_metric(self, task_id: str) -> EvalMetric | None:
        """Return the live metric bundle, or ``None`` when absent."""
        with self._lock:
            return self._metric.get(task_id)

    def update(
        self,
        task_id: str,
        fn: Callable[[EvalTask | None, EvaluationParams | None, EvalMetric | None], _UpdateResult],
    ) -> None:
        """Run ``fn(task, params, metric)`` under the lock.

        ``fn`` MAY mutate any of the three artefacts in place, or return
        a replacement (an :class:`EvalTask`, :class:`EvaluationParams`,
        or :class:`EvalMetric` — or a tuple of them) to be stored back.
        A missing ``task_id`` is a silent no-op so progress callbacks
        racing with a delete do not raise.
        """
        with self._lock:
            task = self._store.get(task_id)
            if task is None:
                return
            params = self._params.get(task_id)
            metric = self._metric.get(task_id)
            outcome = fn(task, params, metric)
            if outcome is None:
                return
            items = outcome if isinstance(outcome, tuple) else (outcome,)
            for item in items:
                if isinstance(item, EvalTask):
                    self._store[task_id] = item
                elif isinstance(item, EvaluationParams):
                    self._params[task_id] = item
                elif isinstance(item, EvalMetric):
                    self._metric[task_id] = item

    def store_metric(self, task_id: str, metric: EvalMetric) -> None:
        """Replace the metric bundle for ``task_id``."""
        with self._lock:
            self._metric[task_id] = metric

    def replace_params(self, task_id: str, params: EvaluationParams) -> None:
        """Swap the pipeline params for ``task_id`` (no-op on miss)."""
        with self._lock:
            if task_id in self._params:
                self._params[task_id] = params

    def set_status(
        self,
        task_id: str,
        *,
        status: int,
        error_msg: str = "",
    ) -> None:
        """Update ``status`` (and optionally ``err_msg``) on the task row."""
        with self._lock:
            task = self._store.get(task_id)
            if task is None:
                return
            finished = task.total if status in (_STATUS_SUCCESS, _STATUS_FAILED) else task.finished
            self._store[task_id] = task.model_copy(
                update={"status": status, "finished": finished},
            )

    def drop(self, task_id: str) -> None:
        """Remove a task entirely (used after a soft-delete)."""
        with self._lock:
            self._store.pop(task_id, None)
            self._params.pop(task_id, None)
            self._metric.pop(task_id, None)


# Integer status codes the in-memory task rows carry. The SQL layer uses
# the string constants from the models module.
_STATUS_PENDING = 0
_STATUS_RUNNING = 1
_STATUS_SUCCESS = 2
_STATUS_FAILED = 3


def _now() -> datetime:
    """Return a timezone-aware ``now`` for stamping tasks."""
    return datetime.now(UTC)


__all__ = [
    "_STATUS_FAILED",
    "_STATUS_PENDING",
    "_STATUS_RUNNING",
    "_STATUS_SUCCESS",
    "EvaluationMemoryStorage",
    "_now",
]
