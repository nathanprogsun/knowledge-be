"""Storage row for the `embed_channels` table.

Each row is a tenant-scoped embed channel configuration that publishes
an agent chat surface for external websites. The service layer reads
live rows to serve the anonymous embed client and the admin CRUD
endpoints.

`publish_token` and `webhook_secret` are secret-bearing columns: the
service layer controls what crosses the wire, and the projection in
``src.core.channels.embed.types`` excludes them.

`allowed_origins` is a JSONB array of origin patterns the embed client
checks before loading the widget.

`deleted_at` is the soft-delete marker. Mirrors the Go entity's
`gorm.DeletedAt`.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import Field

from src.common.json import JsonValue
from src.common.table_model import TableModel

# Default agent bound to a channel when none is supplied at creation.
EMBED_DEFAULT_AGENT_ID = "builtin-quick-answer"


class EmbedChannel(TableModel):
    """One row of the `embed_channels` table."""

    table: ClassVar[str] = "embed_channels"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ("allowed_origins",)
    db_generated_columns: ClassVar[tuple[str, ...]] = ()  # id is caller-assigned (UUID).

    id: str
    tenant_id: int
    agent_id: str = EMBED_DEFAULT_AGENT_ID
    name: str = ""
    enabled: bool = True
    publish_token: str = ""
    allowed_origins: JsonValue = Field(default_factory=list)
    welcome_message: str = ""
    rate_limit_per_minute: int = 30
    rate_limit_per_day: int = 10000
    primary_color: str = ""
    page_title: str = ""
    header_title_mode: str = "channel"
    show_suggested_questions: bool = True
    widget_position: str = "bottom-right"
    allow_web_search: bool = False
    allow_file_upload: bool = False
    default_locale: str = ""
    webhook_url: str = ""
    webhook_secret: str = ""
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


__all__ = ["EMBED_DEFAULT_AGENT_ID", "EmbedChannel"]
