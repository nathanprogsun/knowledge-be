"""Message-domain vocabulary.

``ROLE_*`` constants name the message roles stored in the ``messages``
table; ``MessageSearchMode`` enumerates the chat-history search modes
accepted by the message search endpoint. The string values are stable
storage / wire keys shared by the persistence layer and the service
layer.
"""

from __future__ import annotations

from enum import StrEnum

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"

MESSAGE_ROLES: frozenset[str] = frozenset({ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM})


class MessageSearchMode(StrEnum):
    """Search mode for chat-history message search."""

    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"


__all__ = [
    "MESSAGE_ROLES",
    "ROLE_ASSISTANT",
    "ROLE_SYSTEM",
    "ROLE_USER",
    "MessageSearchMode",
]
