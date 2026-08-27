""" "Thinking then answer" hand-off shared by every streaming chat path.

Thinking chunks are forwarded as they arrive; exactly one ``thinking``
``done`` marker is emitted before the first answer token — or when the stream
ends without one. Centralizing the bookkeeping keeps the OpenAI-compatible and
local stream loops in sync.

Unlike the reference channel-based emitter, this one returns the events to
emit so callers can ``yield`` them from an async generator.
"""

from __future__ import annotations

from src.ai.llm.types import ResponseType, StreamResponse


class ThinkingEmitter:
    """Tracks whether a ``done`` thinking marker is still owed."""

    def __init__(self) -> None:
        self._active = False

    @property
    def active(self) -> bool:
        """True when a thinking chunk was emitted and the marker is owed."""
        return self._active

    def emit(self, content: str) -> StreamResponse:
        """Forward one reasoning chunk and record that a marker is owed."""
        self._active = True
        return StreamResponse(
            response_type=ResponseType.THINKING,
            content=content,
            done=False,
        )

    def finish(self) -> StreamResponse | None:
        """Emit the single thinking-done marker if one is owed.

        Safe to call repeatedly; only the first call after an ``emit``
        returns an event.
        """
        if not self._active:
            return None
        self._active = False
        return StreamResponse(response_type=ResponseType.THINKING, done=True)


__all__ = ["ThinkingEmitter"]
