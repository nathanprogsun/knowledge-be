"""IM adapter abstraction — the contract every platform adapter implements.

Defines the interface platform adapters satisfy: identity, callback
verification, callback parsing, reply sending, URL verification, plus
the connection lifecycle the supervisor drives (connect / disconnect)
and the health probe the supervisor polls.

The message payloads (``IncomingMessage`` / ``ReplyMessage`` /
``QuotedMessage``) are the unified wire shapes the service hands
between the adapters and the QA pipeline. Platform-specific fields
ride in ``extra``.

``CallbackRequest`` is the opaque carrier the web layer passes each
adapter; concrete adapters interpret it in ``verify_callback`` /
``parse_callback`` / ``handle_url_verification``. It is deliberately
framework-agnostic so this module stays at the core layer.

``StreamSender`` and ``FileDownloader`` are optional capabilities.
Adapters implement them when the platform supports streaming replies
or outbound file downloads; the service probes with ``isinstance``.
They are declared as ``@runtime_checkable`` protocols so the type
check is straightforward for the runtime adapter.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject

# ── Message-type constants ───────────────────────────────────────────

MESSAGE_TYPE_TEXT: str = "text"
MESSAGE_TYPE_FILE: str = "file"
MESSAGE_TYPE_IMAGE: str = "image"
MESSAGE_TYPES: frozenset[str] = frozenset(
    {MESSAGE_TYPE_TEXT, MESSAGE_TYPE_FILE, MESSAGE_TYPE_IMAGE}
)

# ── Chat-type constants ──────────────────────────────────────────────

CHAT_TYPE_DIRECT: str = "direct"
CHAT_TYPE_GROUP: str = "group"

# ── Wire shapes ───────────────────────────────────────────────────────


class QuotedMessage(BaseModel):
    """A quoted / replied message.

    Populated by adapters on platforms that support quote-reply. For
    non-text quotes the ``non_text_type`` carries the original message
    type so the QA layer can instruct the model to acknowledge it
    instead of hallucinating content.
    """

    model_config = ConfigDict(frozen=True)

    message_id: str = ""
    content: str = ""
    sender_id: str = ""
    is_bot_message: bool = False
    non_text_type: str = ""


class IncomingMessage(BaseModel):
    """Unified message parsed from an IM callback."""

    model_config = ConfigDict(frozen=True)

    platform: str = ""
    message_type: str = MESSAGE_TYPE_TEXT
    user_id: str = ""
    user_name: str = ""
    chat_id: str = ""
    chat_type: str = CHAT_TYPE_DIRECT
    content: str = ""
    message_id: str = ""
    file_key: str = ""
    file_name: str = ""
    file_size: int = 0
    thread_id: str = ""
    quote: QuotedMessage | None = None
    extra: JsonObject = Field(default_factory=dict)


class ReplyMessage(BaseModel):
    """What the service sends back to the IM platform."""

    model_config = ConfigDict(frozen=True)

    content: str = ""
    is_streaming: bool = False
    is_final: bool = False
    extra: JsonObject = Field(default_factory=dict)


class CallbackRequest(BaseModel):
    """Opaque carrier the web layer hands each adapter.

    Holds the parts of the inbound request the adapters actually need
    without dragging the web framework's request type into core.
    Adapters parse ``body`` / ``query`` / ``headers`` themselves.
    """

    model_config = ConfigDict(frozen=True)

    headers: JsonObject = Field(default_factory=dict)
    body: str = ""
    query: JsonObject = Field(default_factory=dict)


# ── Cancellation context ─────────────────────────────────────────────


class Context(Protocol):
    """Cancellation probe passed to adapters during connect / send_reply.

    Concrete callers supply an ``EventContext`` (or any other object
    exposing ``cancelled``). Adapters check it when they can do so
    cheaply and abort. The probe is intentionally minimal so the
    adapter contract stays framework-agnostic.
    """

    def cancelled(self) -> bool: ...


@dataclass
class EventContext:
    """Default ``Context`` backed by an ``asyncio.Event``.

    ``cancelled`` is a sync probe (``Event.is_set``) so adapters can
    call it from any path without awaiting.
    """

    _event: asyncio.Event = field(default_factory=asyncio.Event)

    def cancelled(self) -> bool:
        """True once ``cancel`` has fired."""
        return self._event.is_set()

    def cancel(self) -> None:
        """Signal cancellation; idempotent."""
        self._event.set()


# ── Adapter interface ────────────────────────────────────────────────

# Stop callable returned from ``connect``: a clean teardown for the
# connection the supervisor wants to recycle / shut down. A no-op
# closure is a valid (lazy) implementation when the SDK's own stop
# path is what the supervisor should fall back to.
StopCallable: TypeAlias = Callable[[], None]


class IMAdapter(ABC):
    """Abstract base every platform adapter implements.

    Defines the message and connection contract: identity,
    callback verification, callback parsing, reply sending, URL
    verification, plus the ``connect`` / ``disconnect`` /
    ``is_connected`` lifecycle the supervisor drives. Concrete
    platform adapters are added by later platform modules; the base
    stays platform-agnostic.
    """

    @abstractmethod
    def platform(self) -> str:
        """Return the stable platform identifier (``feishu``, ``wecom`` ...)."""

    @abstractmethod
    def verify_callback(self, request: CallbackRequest) -> None:
        """Verify the signature / token of an inbound callback.

        Returns ``None`` when verification passes; raises an
        application error (typically ``UnauthorizedError``) otherwise.
        """

    @abstractmethod
    def parse_callback(self, request: CallbackRequest) -> IncomingMessage | None:
        """Parse an inbound callback into a unified ``IncomingMessage``.

        Returns ``None`` for non-message events (URL verification
        challenges, heartbeats) so the service can short-circuit
        without raising.
        """

    @abstractmethod
    def send_reply(
        self,
        ctx: Context,
        incoming: IncomingMessage,
        reply: ReplyMessage,
    ) -> None:
        """Deliver a single reply to the originating IM conversation."""

    @abstractmethod
    def handle_url_verification(self, request: CallbackRequest) -> bool:
        """Handle the platform's initial URL verification challenge.

        Returns ``True`` when the request was a verification request
        and has been fully handled in-place (the web layer should
        not attempt further routing).
        """

    @abstractmethod
    def connect(self, ctx: Context) -> Awaitable[Callable[[], None]]:
        """Establish the platform connection, returning a stop callable.

        The supervisor calls this once per connect cycle; the returned
        callable tears the connection down cleanly so the supervisor
        can recycle it on schedule without leaving a zombie session
        behind.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Tear down any connection the adapter currently holds.

        Called by the supervisor on full shutdown and as the body of
        the default stop callable returned from ``connect``.
        """

    def is_connected(self) -> bool:
        """Health probe used by the supervisor's health-check loop.

        Default is ``True``; adapters whose SDK exposes a real
        liveness probe should override so the supervisor can detect
        zombie connections.
        """
        return True


# ── Optional capabilities ────────────────────────────────────────────


@runtime_checkable
class StreamSender(Protocol):
    """Optional streaming-reply capability.

    Adapters that support platform-native streaming (e.g. editable
    cards) implement this protocol; the service probes with
    ``isinstance(adapter, StreamSender)`` and pushes answer chunks
    in real-time instead of waiting for the full answer.
    """

    def start_stream(
        self,
        ctx: Context,
        incoming: IncomingMessage,
    ) -> Awaitable[str]:
        """Begin a streaming reply; return the platform's stream id."""

    def update_stream_content(
        self,
        ctx: Context,
        incoming: IncomingMessage,
        stream_id: str,
        full_content: str,
    ) -> Awaitable[None]:
        """Replace the user-visible stream text with ``full_content`` so far."""

    def finalize_stream(
        self,
        ctx: Context,
        incoming: IncomingMessage,
        stream_id: str,
        final_content: str,
    ) -> Awaitable[None]:
        """Perform the final replace with answer-only content."""

    def end_stream(
        self,
        ctx: Context,
        incoming: IncomingMessage,
        stream_id: str,
    ) -> Awaitable[None]:
        """Close the streaming reply cleanly."""


@runtime_checkable
class FileDownloader(Protocol):
    """Optional file-download capability for file / image messages.

    Adapters that can fetch attachments the platform has staged
    implement this protocol so file messages routed at a channel
    bound to a knowledge base can be ingested automatically.
    """

    def download_file(
        self,
        ctx: Context,
        msg: IncomingMessage,
    ) -> tuple[bytes, str]:
        """Return ``(content, filename)`` for the attachment ``msg`` names."""


__all__ = [
    "CHAT_TYPE_DIRECT",
    "CHAT_TYPE_GROUP",
    "MESSAGE_TYPES",
    "MESSAGE_TYPE_FILE",
    "MESSAGE_TYPE_IMAGE",
    "MESSAGE_TYPE_TEXT",
    "CallbackRequest",
    "Context",
    "EventContext",
    "FileDownloader",
    "IMAdapter",
    "IncomingMessage",
    "QuotedMessage",
    "ReplyMessage",
    "StreamSender",
]
