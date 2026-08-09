"""Internal DTOs and constants for the custom-agent domain.

``CustomAgentInfo`` is the service-side projection of a ``custom_agents``
row — the carrier the service hands the web layer. The row already
carries every wire field (the config is an opaque JSON blob typed
further down the chat layer), so the projection is a straight field
copy.

The mode / suggestion constants mirror the agent-config contract and
let the service default and validate the config blob without modelling
its full nested shape.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict

from src.common.json import JsonObject
from src.db.models.custom_agent import CustomAgent

# Agent running-mode constants.
AGENT_MODE_QUICK_ANSWER = "quick-answer"
AGENT_MODE_SMART_REASONING = "smart-reasoning"

# Question-suggestion mode constants.
SUGGESTION_MODE_CURATED = "curated"
SUGGESTION_MODE_KNOWLEDGE = "knowledge"
SUGGESTION_MODE_GENERATED = "generated"
SUGGESTION_MODE_HYBRID = "hybrid"

# Follow-up suggestion category constants.
SUGGESTION_CATEGORY_CLARIFY = "clarify"
SUGGESTION_CATEGORY_DEEPEN = "deepen"
SUGGESTION_CATEGORY_ACTION = "action"

# Fallback count when neither the caller nor the agent configuration
# specifies how many suggested questions to return.
SUGGESTION_DEFAULT_LIMIT = 6

# Fixed ids of the built-in agent presets. They are registry-backed
# upstream; until that registry is ported, the service recognises the
# ids so built-in rows can never be edited or deleted.
BUILTIN_AGENT_IDS: frozenset[str] = frozenset(
    {
        "builtin-quick-answer",
        "builtin-smart-reasoning",
        "builtin-deep-researcher",
        "builtin-data-analyst",
        "builtin-knowledge-graph-expert",
        "builtin-document-assistant",
        "builtin-wiki-researcher",
        "builtin-wiki-fixer",
    }
)


class CustomAgentInfo(BaseModel):
    """Service-side projection of a ``custom_agents`` row."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str | None = None
    avatar: str | None = None
    is_builtin: bool = False
    tenant_id: int
    created_by: str | None = None
    config: JsonObject
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(cls, row: CustomAgent) -> Self:
        """Project a storage row onto the service shape."""
        return cls.model_validate(
            {
                "id": row.id,
                "name": row.name,
                "description": row.description,
                "avatar": row.avatar,
                "is_builtin": row.is_builtin,
                "tenant_id": row.tenant_id,
                "created_by": row.created_by,
                "config": row.config,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            }
        )


__all__ = [
    "AGENT_MODE_QUICK_ANSWER",
    "AGENT_MODE_SMART_REASONING",
    "BUILTIN_AGENT_IDS",
    "SUGGESTION_CATEGORY_ACTION",
    "SUGGESTION_CATEGORY_CLARIFY",
    "SUGGESTION_CATEGORY_DEEPEN",
    "SUGGESTION_DEFAULT_LIMIT",
    "SUGGESTION_MODE_CURATED",
    "SUGGESTION_MODE_GENERATED",
    "SUGGESTION_MODE_HYBRID",
    "SUGGESTION_MODE_KNOWLEDGE",
    "CustomAgentInfo",
]
