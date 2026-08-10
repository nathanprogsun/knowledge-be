"""Storage row for the `messages` table.

One row records one chat message — a user query, an assistant answer, or
a system notice — inside a session. The column set mirrors the message
contract: identity / scoping columns (``id``, ``session_id``,
``request_id``), the payload (``role``, ``content``), the retrieval
bookkeeping (``knowledge_references``, ``knowledge_id``), the agent
execution trace (``agent_steps``, ``agent_id``, ``agent_tenant_id``,
``model_id``, ``execution_context``), and the completion / fallback
flags.

Column notes
------------

- ``id`` is caller-assigned (UUID); every other column is caller-supplied
  (the application stamps ``created_at`` / ``updated_at`` before insert).
- ``knowledge_references`` / ``agent_steps`` / ``mentioned_items`` /
  ``images`` / ``attachments`` / ``execution_context`` are JSONB;
  ``json_columns`` binds them with the JSONB bind type.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import Field

from src.common.json import JsonObject, JsonValue
from src.common.table_model import TableModel


class Message(TableModel):
    """One row of the ``messages`` table."""

    table: ClassVar[str] = "messages"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = (
        "knowledge_references",
        "agent_steps",
        "mentioned_items",
        "images",
        "attachments",
        "execution_context",
    )
    # ``id`` is a caller-assigned UUID; the database never assigns columns.
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

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
    rendered_content: str = ""
    channel: str = ""
    agent_id: str = ""
    agent_tenant_id: int = 0
    model_id: str = ""
    execution_context: JsonObject = Field(default_factory=dict)
    knowledge_id: str = ""
    mentioned_items: JsonValue = Field(default_factory=list)
    images: JsonValue = Field(default_factory=list)
    attachments: JsonValue = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


__all__ = ["Message"]
