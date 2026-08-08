"""Internal DTOs for the knowledge-tag domain.

``TagInfo`` is the service-side projection of a ``tags`` row — the
carrier the tag service enriches with usage counts before the web
layer renders the wire ``Tag`` shape. Usage counts are aggregate
fields (computed per query, not stored), so they default to zero and
``map_from_db`` accepts them as overrides.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict

from src.db.models.knowledge_tag import KnowledgeTag

# Default tag name for entries without an explicit tag; a freshly
# created tag carrying this name is pinned to the front of the list.
UNTAGGED_TAG_NAME = "未分类"


class TagInfo(BaseModel):
    """Service-side projection of a ``tags`` row plus usage counts."""

    model_config = ConfigDict(frozen=True)

    id: str
    seq_id: int
    tenant_id: int
    knowledge_base_id: str
    name: str
    color: str | None = None
    sort_order: int = 0
    created_at: datetime
    updated_at: datetime
    knowledge_count: int = 0
    chunk_count: int = 0

    @classmethod
    def map_from_db(
        cls,
        db: KnowledgeTag,
        *,
        knowledge_count: int = 0,
        chunk_count: int = 0,
    ) -> Self:
        """Project a storage row onto the service shape."""
        record = db.model_dump()
        return cls.model_validate(
            {
                **record,
                "knowledge_count": knowledge_count,
                "chunk_count": chunk_count,
            }
        )


__all__ = ["UNTAGGED_TAG_NAME", "TagInfo"]
