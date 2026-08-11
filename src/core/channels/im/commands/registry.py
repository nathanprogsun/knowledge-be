"""IM slash-command registry — register, parse, and dispatch commands.

Mirrors the upstream command-registry contract: commands register by
lower-cased name, ``parse`` splits a message into ``(command, args)``
only when the first token after ``/`` is registered, and
``looks_like_command`` distinguishes ``/help`` from a URL path like
``/api/v2/users`` so callers can route unrecognised slash-words either
to help or into the QA pipeline.

Commands declare intent, not side effects: ``CommandResult.action``
requests a service-level effect (reset the conversation, stop the
in-flight reply) and the executing layer performs it. This keeps the
commands free of service and database dependencies.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from src.core.channels.im.adapter_base import IncomingMessage
from src.core.channels.im.types import IM_OUTPUT_MODE_STREAM


class CommandAction(enum.IntEnum):
    """Service-level side effect a command may request.

    ``NONE`` means no side effect beyond sending the reply. ``CLEAR``
    resets the current conversation so the next message starts a fresh
    session; ``STOP`` cancels the in-flight reply for this user+chat.
    """

    NONE = 0
    CLEAR = 1
    STOP = 2


@dataclass(frozen=True)
class CommandResult:
    """Output produced by a :meth:`Command.execute` call.

    ``content`` is the Markdown reply sent back to the user; ``action``
    requests a service-level side effect.
    """

    content: str
    action: CommandAction = CommandAction.NONE


class SessionLike(Protocol):
    """Structural view of the IM channel session (deferred seam).

    The session-resolution layer lands with the message pipeline; this
    placeholder keeps the command context typed without depending on
    that domain.
    """


class CustomAgentLike(Protocol):
    """Structural view of the bound agent (deferred seam).

    Agent configuration is resolved by the message pipeline before a
    command runs; commands read profile and config fields defensively
    so a later wiring can satisfy this protocol.
    """


@dataclass(frozen=True)
class CommandContext:
    """Runtime data a command needs during execution.

    Services are deliberately NOT carried here — they are injected into
    command objects at construction time. Agent-derived fields stay at
    their defaults until the agent-resolution seam is wired.
    """

    incoming: IncomingMessage
    session: SessionLike | None = None
    tenant_id: int = 0
    agent_name: str = ""
    custom_agent: CustomAgentLike | None = None
    channel_output_mode: str = IM_OUTPUT_MODE_STREAM


class Command(ABC):
    """Interface every IM slash-command implements.

    Design rules:

    - Dependencies (services) are injected at construction time.
    - Validation problems (bad args, unknown entities) are returned as
      a ``CommandResult`` with a helpful message, NOT raised.
    - Raised exceptions are reserved for infrastructure failures; the
      executing layer converts them into a generic error reply.
    """

    @abstractmethod
    def name(self) -> str:
        """Primary token used after ``/`` (e.g. ``help``, ``clear``)."""

    @abstractmethod
    def description(self) -> str:
        """One-line summary shown in ``/help`` output."""

    @abstractmethod
    async def execute(self, ctx: CommandContext, args: list[str]) -> CommandResult:
        """Run the command and return the reply to send to the user."""


class CommandRegistry:
    """Maps slash-command names to their handlers.

    Names are lower-cased on registration so lookups are
    case-insensitive. Registering a duplicate name raises
    ``ValueError`` so misconfiguration surfaces at startup instead of
    being silently ignored.
    """

    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}

    def register(self, command: Command) -> None:
        """Add ``command`` under its lower-cased name.

        Raises ``ValueError`` on a blank name or a duplicate registration.
        """
        key = command.name().strip().lower()
        if not key:
            raise ValueError("im: command name must not be blank")
        if key in self._commands:
            raise ValueError(f"im: duplicate command registration: {key}")
        self._commands[key] = command

    def parse(self, content: str) -> tuple[Command | None, list[str], bool]:
        """Check whether ``content`` is a slash-command and split its args.

        Returns ``(command, args, True)`` when the first token after
        ``/`` is registered; otherwise ``(None, [], False)``. Messages
        that do not start with ``/``, or whose first token has no
        registered handler, return ``False`` so the caller can decide
        whether to show help or pass the text through to the QA
        pipeline.
        """
        cleaned = content.strip()
        if not cleaned.startswith("/"):
            return None, [], False
        parts = cleaned[1:].split()
        if not parts:
            return None, [], False
        command = self._commands.get(parts[0].strip().lower())
        if command is None:
            return None, [], False
        return command, parts[1:], True

    def is_registered(self, content: str) -> bool:
        """True when ``content`` starts with a registered command name."""
        command, _args, ok = self.parse(content)
        return ok and command is not None

    def all(self) -> list[Command]:
        """Return every registered command in registration order."""
        return list(self._commands.values())

    def get(self, name: str) -> Command | None:
        """Return the command registered under ``name``, or ``None``."""
        return self._commands.get(name.strip().lower())

    def looks_like_command(self, content: str) -> bool:
        """True when ``content`` appears to be a command attempt.

        A command attempt starts with ``/`` and its first token contains
        no further ``/`` separators — this distinguishes ``/help`` from
        a URL path like ``/api/v2/users``.
        """
        cleaned = content.strip()
        if not cleaned.startswith("/"):
            return False
        parts = cleaned[1:].split()
        if not parts:
            return False
        return "/" not in parts[0]

    def __len__(self) -> int:
        return len(self._commands)


__all__ = [
    "Command",
    "CommandAction",
    "CommandContext",
    "CommandRegistry",
    "CommandResult",
    "CustomAgentLike",
    "SessionLike",
]
