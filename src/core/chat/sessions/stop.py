"""Stop a running session stream.

Maps the upstream session ``Stop`` contract: given a ``session_id`` and
the assistant ``message_id`` of the in-flight turn, mark the stream as
cancelled and append a ``stop`` event so every reader (the SSE bridge,
the heartbeat, any continue-stream poller) observes the interruption.

The message and session ownership checks live behind optional seams so
the handler can enforce them when repositories are in scope; a missing
seam skips the corresponding check (the stop still succeeds). The
message ``is_completed`` flag mirrors the upstream short-circuit: an
already-completed message needs no stop.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable
from uuid import uuid4

from src.common.exception import NotFoundError, ValidationError
from src.core.chat.stream.manager import StreamManager
from src.core.chat.stream.types import Event
from src.db.models.message import Message

#: Wire ``response_type`` carried by the appended stop event.
_STOP_RESPONSE_TYPE = "stop"
#: ``data.reason`` recorded on the stop event (upstream value).
_STOP_REASON = "user_requested"


@runtime_checkable
class StreamMessageReader(Protocol):
    """Reads one message by ``(session_id, message_id)``.

    The narrow surface lets tests (and the message service) satisfy the
    seam without pulling in the full message repository.
    """

    async def get_by_id_and_session(
        self,
        *,
        session_id: str,
        message_id: str,
    ) -> Message | None: ...


@dataclass(frozen=True, slots=True)
class StopStreamResult:
    """Outcome of a stop request.

    ``stopped`` is ``True`` when a stop event was written; ``False``
    when the message was already completed (the upstream "already
    completed" short-circuit).
    """

    stopped: bool
    session_id: str
    message_id: str


class StopStreamService:
    """Per-request stream-stopping facade."""

    def __init__(
        self,
        *,
        stream_manager: StreamManager,
        message_reader: StreamMessageReader | None = None,
    ) -> None:
        self._stream_manager = stream_manager
        self._message_reader = message_reader

    async def stop(self, session_id: str, message_id: str) -> StopStreamResult:
        """Stop the stream for ``(session_id, message_id)``.

        The message row (when a reader is injected) is verified before
        any write: a missing row raises ``NotFoundError`` and an
        already-completed message returns ``stopped=False`` without
        touching the stream. Otherwise a ``stop`` event is appended and
        the stream is marked cancelled.
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

        if self._message_reader is not None:
            message = await self._message_reader.get_by_id_and_session(
                session_id=session_id,
                message_id=message_id,
            )
            if message is None:
                raise NotFoundError(
                    code="stream.message_not_found",
                    message=f"message {message_id} not found in session {session_id}",
                )
            if message.is_completed:
                return StopStreamResult(
                    stopped=False,
                    session_id=session_id,
                    message_id=message_id,
                )

        self._stream_manager.cancel(session_id, message_id)
        await self._stream_manager.append_event(
            session_id,
            message_id,
            Event(
                id=f"stop-{uuid4().hex}",
                type=_STOP_RESPONSE_TYPE,
                done=True,
                timestamp=datetime.now(UTC),
                data={
                    "session_id": session_id,
                    "message_id": message_id,
                    "reason": _STOP_REASON,
                },
            ),
        )
        return StopStreamResult(
            stopped=True,
            session_id=session_id,
            message_id=message_id,
        )


__all__ = ["StopStreamResult", "StopStreamService", "StreamMessageReader"]
