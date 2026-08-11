"""IM slash-command registry and built-in handlers.

Public surface for the IM command domain: the registry (register /
parse / dispatch), the command context and result types, and the
built-in handlers (``/help``, ``/info``, ``/search``, ``/stop``,
``/clear``) assembled by :func:`build_default_registry`.
"""

from __future__ import annotations

from src.core.channels.im.commands.handlers import (
    ClearCommand,
    HelpCommand,
    InfoCommand,
    KnowledgeBaseLister,
    SearchCommand,
    SearchService,
    StopCommand,
    build_default_registry,
)
from src.core.channels.im.commands.registry import (
    Command,
    CommandAction,
    CommandContext,
    CommandRegistry,
    CommandResult,
    CustomAgentLike,
    SessionLike,
)

__all__ = [
    "ClearCommand",
    "Command",
    "CommandAction",
    "CommandContext",
    "CommandRegistry",
    "CommandResult",
    "CustomAgentLike",
    "HelpCommand",
    "InfoCommand",
    "KnowledgeBaseLister",
    "SearchCommand",
    "SearchService",
    "SessionLike",
    "StopCommand",
    "build_default_registry",
]
