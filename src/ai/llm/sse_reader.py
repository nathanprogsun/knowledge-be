"""Minimal Server-Sent Events reader for LLM streaming responses.

Parses an SSE byte/line stream into events. Only the ``data:`` field is
surfaced; ``event:`` / ``id:`` / comment lines are skipped. The ``[DONE]``
sentinel is reported as a dedicated ``done`` flag so callers can terminate
without parsing a data payload.

``SSEReader`` accepts any async iterable of text lines (an HTTP stream's
``aiter_lines()``, an ``asyncio`` queue fed by a test, etc.) and yields
:class:`SSEEvent` objects; exhaustion surfaces as ``StopAsyncIteration``.
"""

from __future__ import annotations

from collections.abc import AsyncIterable
from dataclasses import dataclass


@dataclass(frozen=True)
class SSEEvent:
    """One parsed SSE event.

    ``done`` is true for the ``data: [DONE]`` sentinel; ``data`` is then unset.
    """

    data: str | None = None
    done: bool = False


class SSEReader:
    """Async iterator converting SSE text lines into :class:`SSEEvent`."""

    def __init__(self, lines: AsyncIterable[str]) -> None:
        self._lines = lines.__aiter__()
        self._finished = False

    def __aiter__(self) -> SSEReader:
        return self

    async def __anext__(self) -> SSEEvent:
        if self._finished:
            raise StopAsyncIteration
        while True:
            try:
                line = await anext(self._lines)
            except StopAsyncIteration:
                self._finished = True
                raise
            if line == "":
                continue
            if line == "data: [DONE]":
                self._finished = True
                return SSEEvent(done=True)
            if line.startswith("data: "):
                return SSEEvent(data=line[6:])
            if line.startswith("data:"):
                return SSEEvent(data=line[5:])
            # event:, id:, comment and other lines are skipped.


__all__ = ["SSEEvent", "SSEReader"]
