"""Task registry — maps task names to their async handlers.

Handlers register themselves at import time via the ``register_task``
decorator; importing a task module is what makes its tasks visible to
the worker. The registry is a plain dict keyed by task name.

Future PRs add individual tasks by creating modules that import this
registry and decorate their handlers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final, Protocol, TypeAlias, cast

from arq.typing import WorkerCoroutine
from arq.worker import Function

from src.workers.base import WorkerContext

# A JSON-serializable value carried in a task payload.
JsonValue: TypeAlias = (
    str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None
)


class TaskHandler(Protocol):
    """Async task handler: ``(ctx, **payload) -> Any``.

    Returns any JSON-serializable value; ARQ persists the result so
    callers can inspect it. ``__qualname__`` is required because ARQ
    uses it as the default function name when a handler is registered
    without an explicit name.
    """

    __qualname__: str

    async def __call__(self, ctx: WorkerContext, **payload: JsonValue) -> Any: ...


_REGISTRY: Final[dict[str, TaskHandler]] = {}


def register_task(name: str) -> Callable[[TaskHandler], TaskHandler]:
    """Decorator that registers a task handler under ``name``.

    Usage::

        @register_task("document_process")
        async def document_process(ctx: WorkerContext, **payload: JsonValue) -> None:
            ...
    """

    def decorator(func: TaskHandler) -> TaskHandler:
        _REGISTRY[name] = func
        return func

    return decorator


def get_task(name: str) -> TaskHandler | None:
    """Return the handler registered under ``name``, or ``None``."""
    return _REGISTRY.get(name)


def all_tasks() -> dict[str, TaskHandler]:
    """Return a read-only copy of the registry."""
    return dict(_REGISTRY)


def all_functions() -> list[Function]:
    """Return the registry as ARQ ``Function`` entries for a Worker."""
    return [
        Function(
            name=name,
            coroutine=cast(WorkerCoroutine, handler),
            timeout_s=None,
            keep_result_s=None,
            keep_result_forever=None,
            max_tries=None,
        )
        for name, handler in _REGISTRY.items()
    ]


__all__ = [
    "JsonValue",
    "TaskHandler",
    "all_functions",
    "all_tasks",
    "get_task",
    "register_task",
]
