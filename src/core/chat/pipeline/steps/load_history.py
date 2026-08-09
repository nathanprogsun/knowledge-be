"""Pipeline history-loading step (upstream ``load_history.go``).

Loads prior Q&A rounds into the run carrier before the completion stage
replays them. Multi-turn can be disabled per run (``max_rounds <= 0``):
history is then skipped entirely so it never leaks into the LLM context —
the global default is deliberately not applied here, otherwise the disable
flag would be silently overridden. A history-fetch failure is non-fatal:
the turn continues with an empty history.
"""

from __future__ import annotations

from src.core.chat.pipeline.common import (
    MessageService,
    load_and_process_history,
    pipeline_info,
    pipeline_warn,
)
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.engine import Next, PluginError
from src.core.chat.pipeline.types import Context, EventType

#: How many raw messages to fetch per round; extra headroom covers the
#: intermediate assistant messages between user turns.
_FETCH_MULTIPLIER = 2
_FETCH_HEADROOM = 10


class LoadHistoryPlugin:
    """Loads and groups prior turns into ``PipelineContext.history``."""

    def __init__(self, message_service: MessageService) -> None:
        self._message_service = message_service

    def activation_events(self) -> list[EventType]:
        return [EventType.LOAD_HISTORY]

    async def on_event(
        self,
        ctx: Context,
        event_type: EventType | str,
        pipeline_ctx: PipelineContext,
        next: Next,
    ) -> PluginError | None:
        if pipeline_ctx.max_rounds <= 0:
            pipeline_info(
                "LoadHistory",
                "skipped",
                {
                    "session_id": pipeline_ctx.session_id,
                    "reason": "multi_turn_disabled",
                },
            )
            return await next()

        max_rounds = pipeline_ctx.max_rounds
        pipeline_info(
            "LoadHistory",
            "input",
            {
                "session_id": pipeline_ctx.session_id,
                "max_rounds": max_rounds,
            },
        )

        fetch_count = max_rounds * _FETCH_MULTIPLIER + _FETCH_HEADROOM
        try:
            history_list = await load_and_process_history(
                ctx,
                self._message_service,
                pipeline_ctx.session_id,
                max_rounds,
                fetch_count,
            )
        except Exception as exc:
            pipeline_warn(
                "LoadHistory",
                "history_fetch",
                {
                    "session_id": pipeline_ctx.session_id,
                    "error": str(exc),
                },
            )
            return await next()

        pipeline_ctx.history = history_list
        pipeline_info(
            "LoadHistory",
            "output",
            {
                "session_id": pipeline_ctx.session_id,
                "history_rounds": len(history_list),
                "max_rounds": max_rounds,
            },
        )
        return await next()


__all__ = ["LoadHistoryPlugin"]
