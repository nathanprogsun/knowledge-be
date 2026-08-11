"""Internal DTOs and constants for the IM-channel domain.

``IMChannelInfo`` is the service-side projection of an ``im_channels``
row — the carrier the service hands the web layer. Credentials are
intentionally excluded: the wire contract never exposes platform
secrets, and the per-agent read path returns them only immediately
after a mutation.

The platform / mode / session-mode constants mirror the upstream
contract and let the service default and validate channel fields
without modelling the full nested credential shape.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict

from src.db.models.im_channel import IMChannel

# Supported IM platform identifiers (aliases included).
IM_PLATFORMS: frozenset[str] = frozenset(
    {
        "feishu",
        "lark",
        "wecom",
        "wxwork",
        "slack",
        "telegram",
        "dingtalk",
        "mattermost",
        "wechat",
        "qqbot",
        "yunzhijia",
    }
)

# Connection-mode constants.
IM_MODE_WEBSOCKET = "websocket"
IM_MODE_WEBHOOK = "webhook"

# Output-mode constants.
IM_OUTPUT_MODE_STREAM = "stream"

# Session-mode constants.
IM_SESSION_MODE_USER = "user"
IM_SESSION_MODE_THREAD = "thread"


class IMChannelInfo(BaseModel):
    """Service-side projection of an ``im_channels`` row.

    The wire contract is a subset of these fields; secret-bearing
    ``credentials`` never cross into the projection. ``credentials_configured``
    is the only credential-derived signal the projection carries — it is
    computed from the raw row before the secret column is dropped.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    agent_id: str
    platform: str
    name: str
    enabled: bool
    mode: str
    output_mode: str
    knowledge_base_id: str
    bot_identity: str
    session_mode: str
    credentials_configured: bool = False
    created_at: datetime
    updated_at: datetime

    @classmethod
    def map_from_db(cls, db: IMChannel) -> Self:
        """Build a projection from the raw storage row, dropping secrets."""
        record = db.model_dump()
        credentials = record.get("credentials")
        record["credentials_configured"] = bool(credentials) and credentials != {}
        record.pop("credentials", None)
        record.pop("deleted_at", None)
        return cls.model_validate(record)


__all__ = [
    "IM_MODE_WEBHOOK",
    "IM_MODE_WEBSOCKET",
    "IM_OUTPUT_MODE_STREAM",
    "IM_PLATFORMS",
    "IM_SESSION_MODE_THREAD",
    "IM_SESSION_MODE_USER",
    "IMChannelInfo",
]
