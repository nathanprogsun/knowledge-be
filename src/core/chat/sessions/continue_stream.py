"""Continue an interrupted session stream.

Maps the upstream session ``ContinueStream`` contract: a client that
disconnected mid-stream reconnects with ``(session_id, message_id)``,
receives every event accumulated so far, and then follows the stream
forward from the last offset until it completes.

The implementation is an async generator: events are yielded in order as
the underlying ``StreamManager`` produces them, so the SSE bridge can
forward each one as it arrives. A missing stream (no events at all)
raises ``NotFoundError``; a stream manager failure propagates to the
caller.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from src.common.exception import NotFoundError, ValidationError
from src.core.chat.stream.manager import StreamManager
from src.core.chat.stream.types import Event

#: Wire ``response_type`` that terminates a stream.
_COMPLETE_RESPONSE_TYPE = "complete"
#: Wire ``response_type`` that terminates a stream (user stop).
_STOP_RESPONSE_TYPE = "stop"
#: Default poll interval between incremental stream reads.
_DEFAULT_POLL_INTERVAL_SECONDS = 0.1

#: Event types that end a continued stream.
_TERMINAL_TYPES = frozenset({_COMPLETE_RESPONSE_TYPE, _STOP_RESPONSE_TYPE})


async def continue_stream(
    stream_manager: StreamManager,
    session_id: str,
    message_id: str,
    *,
    poll_interval_seconds: float = _DEFAULT_POLL_INTERVAL_SECONDS,
) -> AsyncIterator[Event]:
    """Replay a stream from its start and follow it to completion.

    Yields every existing event first (from offset 0), then polls for
    new events every ``poll_interval_seconds`` until a terminal event
    (``complete`` / ``stop``) is observed. An empty stream raises
    ``NotFoundError``; the caller owns cancellation (closing the
    generator stops the polling loop).
    """
    if not session_id or not session_id.strip():
        raise ValidationError(
            code="stream.session_required",
            message="Session ID is empty",
        )
    if not message_id or not message_id.strip():
        raise ValidationError(
            code="stream.message_required",
            message="message_id is required",
        )
    if poll_interval_seconds <= 0:
        raise ValidationError(
            code="stream.invalid_poll_interval",
            message="poll_interval_seconds must be positive",
        )

    events, offset = await stream_manager.get_events(session_id, message_id, 0)
    if not events:
        raise NotFoundError(
            code="stream.not_found",
            message=(f"no stream events found for session {session_id}, message {message_id}"),
        )

    replay = [*events]
    while replay:
        event = replay.pop(0)
        yield event
        if event.type in _TERMINAL_TYPES:
            return

    while True:
        await asyncio.sleep(poll_interval_seconds)
        events, offset = await stream_manager.get_events(session_id, message_id, offset)
        for event in events:
            yield event
            if event.type in _TERMINAL_TYPES:
                return


__all__ = ["continue_stream"]
