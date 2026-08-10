"""Stream-domain value types for the chat stream manager.

``Event`` mirrors the upstream ``StreamEvent`` shape: a single append-only
record in a chat stream carrying an id, a wire ``response_type`` string,
content, a done flag, a timestamp, and optional JSON data. ``StreamKey``
identifies one stream (a session plus a message); ``StreamData`` is the
per-stream state the in-memory backend keeps.

The ``type`` field is a plain string rather than a closed enum because
the stream vocabulary is open-ended: the wire ``response_type`` values
(``answer``, ``thinking``, ``tool_call``, ``tool_result``, ``references``,
``complete``, ``error``, ``reflection``, ``session_title``,
``agent_query``, ``tool_approval_required``, ``tool_approval_resolved``,
``mcp_oauth_required``, ``mcp_oauth_resolved``) are joined by control
events such as ``stop`` and the heartbeat ``ping``.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject


def _utc_now() -> datetime:
    """Return the current UTC time (aware, for JSON round-tripping)."""
    return datetime.now(UTC)


class Event(BaseModel):
    """One append-only event in a chat stream."""

    model_config = ConfigDict(frozen=True)

    id: str
    type: str
    content: str = ""
    done: bool = False
    timestamp: datetime = Field(default_factory=_utc_now)
    data: JsonObject | None = None


class StreamKey(BaseModel):
    """Composite identity of one stream: a session and a message."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    message_id: str


class StreamData(BaseModel):
    """Per-stream state kept by the in-memory backend."""

    model_config = ConfigDict(frozen=True)

    events: tuple[Event, ...] = ()
    last_updated: datetime = Field(default_factory=_utc_now)


__all__ = ["Event", "StreamData", "StreamKey"]
