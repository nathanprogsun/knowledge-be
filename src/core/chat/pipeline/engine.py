"""Chat pipeline engine (upstream ``EventManager``).

Plugins register for event types; the engine wires each event type to a
chain of listeners executed in registration order. A listener calls
``next()`` to hand control to the following listener; returning a
``PluginError`` — or surfacing one from a downstream ``next()`` call —
stops the chain. This event-driven model is how the retrieval and
completion steps plug into a run.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, TypeAlias

from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.types import Context, EventType


@dataclass(frozen=True, slots=True)
class PluginError:
    """An error reported by a plugin during pipeline execution.

    A value rather than an ``Exception``: plugins return it instead of
    raising, mirroring the value-passing chain contract.
    """

    description: str = ""
    error_type: str = ""
    err: BaseException | None = None

    def with_error(self, err: BaseException) -> PluginError:
        """Return a copy of this error carrying ``err`` (never mutates)."""
        return replace(self, err=err)


ERR_SEARCH_NOTHING = PluginError(
    description="No relevant content found",
    error_type="search_nothing",
)
ERR_SEARCH = PluginError(
    description="Failed to search knowledge base",
    error_type="search_failed",
)
ERR_RERANK = PluginError(
    description="Reranking failed",
    error_type="rerank_failed",
)
ERR_GET_RERANK_MODEL = PluginError(
    description="Failed to get rerank model",
    error_type="get_rerank_model_failed",
)
ERR_GET_CHAT_MODEL = PluginError(
    description="Failed to get chat model",
    error_type="get_chat_model_failed",
)
ERR_TEMPLATE_PARSE = PluginError(
    description="Failed to parse context template",
    error_type="template_parse_failed",
)
ERR_TEMPLATE_EXECUTE = PluginError(
    description="Failed to generate search content",
    error_type="template_execution_failed",
)
ERR_MODEL_CALL = PluginError(
    description="Failed to call model",
    error_type="model_call_failed",
)
ERR_GET_HISTORY = PluginError(
    description="Failed to get conversation history",
    error_type="get_history_failed",
)


#: Resumes the chain: invokes the listener registered after the current one.
Next: TypeAlias = Callable[[], Awaitable[PluginError | None]]


class Plugin(Protocol):
    """A stage in the chat pipeline (upstream ``Plugin``).

    Implementations declare the event types they handle via
    ``activation_events()`` and react in ``on_event``. ``next`` resumes
    the chain; a plugin awaits it exactly once unless it is terminating
    the run with an error or a deliberate short-circuit.
    """

    async def on_event(
        self,
        ctx: Context,
        event_type: EventType | str,
        pipeline_ctx: PipelineContext,
        next: Next,
    ) -> PluginError | None: ...

    def activation_events(self) -> Sequence[EventType]: ...


#: One full chain invocation for an event type.
Handler: TypeAlias = Callable[
    [Context, EventType | str, PipelineContext],
    Awaitable[PluginError | None],
]


class EventManager:
    """Routes event types to their registered plugin chains."""

    def __init__(self) -> None:
        self._listeners: dict[str, list[Plugin]] = {}
        self._handlers: dict[str, Handler] = {}

    def register(self, plugin: Plugin) -> None:
        """Register a plugin for every event type it activates on.

        The handler chain for an event type is rebuilt after each
        registration, so listeners run in registration order.
        """
        for event_type in plugin.activation_events():
            key = str(event_type)
            listeners = [*self._listeners.get(key, []), plugin]
            self._listeners[key] = listeners
            self._handlers[key] = self._build_handler(listeners)

    @staticmethod
    def _build_handler(plugins: list[Plugin]) -> Handler:
        """Build the chain closure for ``plugins``.

        ``run_at(index)`` invokes the plugin at ``index`` with a ``next``
        callback that continues at ``index + 1``; the final ``next`` is a
        no-op returning ``None``.
        """

        async def handler(
            ctx: Context,
            event_type: EventType | str,
            pipeline_ctx: PipelineContext,
        ) -> PluginError | None:
            async def run_at(index: int) -> PluginError | None:
                if index >= len(plugins):
                    return None
                plugin = plugins[index]

                async def next() -> PluginError | None:
                    return await run_at(index + 1)

                return await plugin.on_event(ctx, event_type, pipeline_ctx, next)

            return await run_at(0)

        return handler

    async def trigger(
        self,
        ctx: Context,
        event_type: EventType | str,
        pipeline_ctx: PipelineContext,
    ) -> PluginError | None:
        """Run the chain registered for ``event_type``.

        Returns ``None`` when no plugin is registered for the event type
        (a no-op, matching the upstream contract).
        """
        handler = self._handlers.get(str(event_type))
        if handler is None:
            return None
        return await handler(ctx, event_type, pipeline_ctx)


__all__ = [
    "ERR_GET_CHAT_MODEL",
    "ERR_GET_HISTORY",
    "ERR_GET_RERANK_MODEL",
    "ERR_MODEL_CALL",
    "ERR_RERANK",
    "ERR_SEARCH",
    "ERR_SEARCH_NOTHING",
    "ERR_TEMPLATE_EXECUTE",
    "ERR_TEMPLATE_PARSE",
    "EventManager",
    "Handler",
    "Next",
    "Plugin",
    "PluginError",
]
