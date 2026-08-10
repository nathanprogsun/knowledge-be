"""Internal DTOs for the chat-session domain.

``SessionInfo`` is the service-side projection of a ``sessions`` row —
the carrier the session service hands the web layer. The IM origin
fields (``im_platform`` etc.) are not stored on the ``sessions`` table;
they live on the channel-mapping table and are joined in at read time,
so they default to ``None`` here and are enriched by the service when a
mapping exists.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict

from src.db.models.session import Session


class SessionInfo(BaseModel):
    """Service-side projection of a ``sessions`` row plus IM origin fields."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str | None = None
    description: str | None = None
    tenant_id: int
    user_id: str | None = None
    is_pinned: bool = False
    pinned_at: datetime | None = None
    im_platform: str | None = None
    im_chat_id: str | None = None
    im_thread_id: str | None = None
    im_user_id: str | None = None
    im_agent_id: str | None = None
    im_channel_id: str | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None

    @classmethod
    def from_row(cls, row: Session) -> Self:
        """Project a storage row onto the service shape."""
        return cls.model_validate(row.model_dump())


__all__ = ["SessionInfo"]
