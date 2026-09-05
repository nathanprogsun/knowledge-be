"""Internal DTOs for the sharing domain.

These are service-output projections, not HTTP wire models.
They exist so the web layer can map DTOs onto public contracts
without importing storage table models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict

from src.db.models.agent_share import AgentShare
from src.db.models.kb_share import KnowledgeBaseShare

_AGENT_SHARE_EXCLUDE_COLUMNS: frozenset[str] = frozenset({"deleted_at"})
_KNOWLEDGE_BASE_SHARE_EXCLUDE_COLUMNS: frozenset[str] = frozenset({"deleted_at"})


class AgentShareInfo(BaseModel):
    """Service-side projection of one agent-share storage row."""

    model_config = ConfigDict(frozen=True)

    id: str
    agent_id: str
    organization_id: str
    shared_by_user_id: str
    source_tenant_id: int
    permission: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def map_from_db(cls, db: AgentShare) -> Self:
        """Project a stored share row onto the service DTO."""
        return cls.model_validate(db.model_dump(exclude=set(_AGENT_SHARE_EXCLUDE_COLUMNS)))


class KnowledgeBaseShareInfo(BaseModel):
    """Service-side projection of one knowledge-base share row."""

    model_config = ConfigDict(frozen=True)

    id: str
    knowledge_base_id: str
    organization_id: str
    shared_by_user_id: str
    source_tenant_id: int
    permission: str
    created_at: datetime
    updated_at: datetime
    my_role_in_org: str | None = None
    my_permission: str | None = None

    @classmethod
    def map_from_db(cls, db: KnowledgeBaseShare) -> Self:
        """Project a stored share row onto the service DTO."""
        return cls.model_validate(db.model_dump(exclude=set(_KNOWLEDGE_BASE_SHARE_EXCLUDE_COLUMNS)))


__all__ = ["AgentShareInfo", "KnowledgeBaseShareInfo"]
