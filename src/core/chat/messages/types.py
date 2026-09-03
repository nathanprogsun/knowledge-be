"""Message-domain vocabulary and service DTOs.

``ROLE_*`` constants name the message roles stored in the ``messages``
table; ``MessageSearchMode`` enumerates the chat-history search modes
accepted by the message search endpoint. ``MessageInfo`` is the
service-side projection of a ``messages`` row: the shape the message
service hands to the web layer and the chat pipeline. JSON columns
stay as opaque ``JsonValue`` payloads here — lenient wire coercion is
the view layer's concern.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject, JsonValue
from src.db.models.message import Message

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"

MESSAGE_ROLES: frozenset[str] = frozenset({ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM})


class MessageSearchMode(StrEnum):
    """Search mode for chat-history message search."""

    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"


# Columns that never leave the service boundary: agent routing internals,
# the rendered (post-processed) content variant, and the raw execution
# context used only by the pipeline that produced the message.
_MESSAGE_EXCLUDE_COLUMNS: frozenset[str] = frozenset(
    {
        "rendered_content",
        "agent_id",
        "agent_tenant_id",
        "model_id",
        "execution_context",
        "knowledge_id",
    }
)


class MessageInfo(BaseModel):
    """Service-side projection of a ``messages`` row.

    Carries everything the chat pipeline's history loader and the web
    view conversion consume; storage-only columns are stripped by
    :meth:`map_from_db`.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    request_id: str = ""
    session_id: str
    role: str
    content: str
    knowledge_references: JsonValue = Field(default_factory=list)
    agent_steps: JsonValue | None = None
    is_completed: bool = True
    is_fallback: bool = False
    agent_duration_ms: int = 0
    channel: str = ""
    mentioned_items: JsonValue = Field(default_factory=list)
    images: JsonValue = Field(default_factory=list)
    attachments: JsonValue = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None

    @classmethod
    def map_from_db(cls, db: Message) -> Self:
        """Project a stored message row, dropping service-internal columns."""
        return cls.model_validate(db.model_dump(exclude=set(_MESSAGE_EXCLUDE_COLUMNS)))

    @classmethod
    def from_json(cls, raw: JsonObject | str) -> Self:
        """Hydrate from a JSON column payload (dict or raw JSON text)."""
        if isinstance(raw, str):
            return cls.model_validate_json(raw)
        return cls.model_validate(raw)


__all__ = [
    "MESSAGE_ROLES",
    "ROLE_ASSISTANT",
    "ROLE_SYSTEM",
    "ROLE_USER",
    "MessageInfo",
    "MessageSearchMode",
]
