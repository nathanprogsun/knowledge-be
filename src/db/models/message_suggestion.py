"""Storage row for the `message_suggestion_sets` table.

One row records the durable generation / cache record for one assistant
message and one effective agent configuration: the cache key
(``tenant_id`` + ``assistant_message_id`` + ``placement`` +
``config_hash`` + ``locale``) that deduplicates concurrent generation
requests, the generation lifecycle (``status``, ``lease_until``,
``generated_at``), the produced ``questions``, and the token / latency
bookkeeping.

The table has no ``deleted_at`` column — deletes are hard deletes.

Column notes
------------

- ``id`` is caller-assigned (UUID); every other column is caller-supplied
  (the application stamps ``created_at`` / ``updated_at`` before insert).
- ``questions`` is JSONB; ``json_columns`` binds it with the JSONB bind
  type.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import Field

from src.common.json import JsonValue
from src.common.table_model import TableModel

SUGGESTION_PLACEMENT_AFTER_ANSWER = "after_answer"

SUGGESTION_STATUS_GENERATING = "generating"
SUGGESTION_STATUS_READY = "ready"
SUGGESTION_STATUS_SUPPRESSED = "suppressed"
SUGGESTION_STATUS_FAILED = "failed"

SUGGESTION_EVENT_IMPRESSION = "impression"
SUGGESTION_EVENT_CLICK = "click"
SUGGESTION_EVENT_DISMISS = "dismiss"
SUGGESTION_EVENT_REGENERATE = "regenerate"


class MessageSuggestionSet(TableModel):
    """One row of the ``message_suggestion_sets`` table."""

    table: ClassVar[str] = "message_suggestion_sets"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ("questions",)
    # ``id`` is a caller-assigned UUID; the database never assigns columns.
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    tenant_id: int
    session_id: str
    assistant_message_id: str
    agent_id: str = ""
    agent_tenant_id: int = 0
    placement: str
    config_hash: str
    locale: str = ""
    status: str
    allow_regenerate: bool = False
    suppression_reason: str = ""
    questions: JsonValue = Field(default_factory=list)
    model_id: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    error_code: str = ""
    lease_until: datetime | None = None
    generated_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


__all__ = [
    "SUGGESTION_EVENT_CLICK",
    "SUGGESTION_EVENT_DISMISS",
    "SUGGESTION_EVENT_IMPRESSION",
    "SUGGESTION_EVENT_REGENERATE",
    "SUGGESTION_PLACEMENT_AFTER_ANSWER",
    "SUGGESTION_STATUS_FAILED",
    "SUGGESTION_STATUS_GENERATING",
    "SUGGESTION_STATUS_READY",
    "SUGGESTION_STATUS_SUPPRESSED",
    "MessageSuggestionSet",
]
