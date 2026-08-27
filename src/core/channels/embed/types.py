"""Internal DTOs and constants for the embed-channel domain.

``EmbedChannelInfo`` is the service-side projection of an
``embed_channels`` row — the carrier the service hands the web layer.
Secret-bearing columns (``publish_token``, ``webhook_secret``) are
intentionally excluded: the wire contract never exposes them.

The constants mirror the upstream contract and let the service default
and validate channel fields without modelling the full nested shape.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict

from src.db.models.embed_channel import EmbedChannel

# Defaults applied by the service layer when a field is left empty.
EMBED_DEFAULT_RATE_LIMIT_PER_DAY = 10000
EMBED_DEFAULT_WIDGET_POSITION = "bottom-right"
EMBED_DEFAULT_HEADER_TITLE_MODE = "channel"

# Header-title-mode constants.
EMBED_HEADER_TITLE_MODE_SESSION = "session"

# Supported widget corner positions.
EMBED_WIDGET_POSITIONS: frozenset[str] = frozenset(
    {"bottom-left", "top-right", "top-left", "bottom-right"}
)

# Supported embed UI locales.
EMBED_SUPPORTED_LOCALES: frozenset[str] = frozenset({"zh-CN", "en-US", "ko-KR", "ru-RU"})

# Prefix tagging sessions created through an embed channel.
EMBED_SESSION_MARKER_PREFIX = "embed_channel:"


class EmbedChannelInfo(BaseModel):
    """Service-side projection of an ``embed_channels`` row.

    The wire contract is a subset of these fields; secret-bearing
    ``publish_token`` and ``webhook_secret`` never cross into the
    projection.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    agent_id: str
    name: str
    enabled: bool
    allowed_origins: list[str]
    welcome_message: str
    rate_limit_per_minute: int
    rate_limit_per_day: int
    primary_color: str
    page_title: str
    header_title_mode: str
    show_suggested_questions: bool
    widget_position: str
    allow_web_search: bool
    allow_file_upload: bool
    default_locale: str
    webhook_url: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def map_from_db(cls, db: EmbedChannel) -> Self:
        """Build a projection from the raw storage row, dropping secrets."""
        record = db.model_dump()
        record.pop("publish_token", None)
        record.pop("webhook_secret", None)
        record.pop("deleted_at", None)
        return cls.model_validate(record)


__all__ = [
    "EMBED_DEFAULT_HEADER_TITLE_MODE",
    "EMBED_DEFAULT_RATE_LIMIT_PER_DAY",
    "EMBED_DEFAULT_WIDGET_POSITION",
    "EMBED_HEADER_TITLE_MODE_SESSION",
    "EMBED_SESSION_MARKER_PREFIX",
    "EMBED_SUPPORTED_LOCALES",
    "EMBED_WIDGET_POSITIONS",
    "EmbedChannelInfo",
]
