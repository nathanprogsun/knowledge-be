"""Storage row for the `im_channels` table.

Each row is a tenant-scoped IM channel configuration that binds an
agent to a platform-specific bot (Feishu, WeCom, Slack, Telegram,
DingTalk, Mattermost, WeChat, QQ Bot, Yunzhijia). The supervisor
(service layer) reads live rows to start adapters; the service layer
owns the bot-identity derivation and the duplicate-bot guard.

`credentials` is a JSONB blob carrying platform-specific secrets
(app_id, bot_token, corp_id, ...). The service layer controls what
crosses the wire; the row carries everything that was persisted.

`bot_identity` is a derived unique key (platform + mode + credential
fields) computed by the service layer before save; the DB unique index
on live rows is the safety net.

`deleted_at` is the soft-delete marker. Mirrors the Go entity's
`gorm.DeletedAt`.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import Field

from src.common.json import JsonObject
from src.common.table_model import TableModel


class IMChannel(TableModel):
    """One row of the `im_channels` table."""

    table: ClassVar[str] = "im_channels"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ("credentials",)
    db_generated_columns: ClassVar[tuple[str, ...]] = ()  # id is caller-assigned (UUID).

    id: str
    tenant_id: int
    agent_id: str
    platform: str
    name: str = ""
    enabled: bool = True
    mode: str = "websocket"
    output_mode: str = "stream"
    knowledge_base_id: str = ""
    bot_identity: str = ""
    session_mode: str = "user"
    credentials: JsonObject = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


__all__ = ["IMChannel"]
