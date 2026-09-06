"""Wire-shape conversion for the embed-channel endpoints.

Maps the embed-channel handler surface (tenant-admin CRUD plus the
anonymous public embed surface) onto Pydantic wire shapes. Field names
mirror the upstream contract exactly, including JSON serialization
names, so the wire format stays aligned.

The ``publish_token`` is a management-only secret: it is rendered only
by the create / rotate / single-get endpoints (never by list responses
or the public config). ``has_webhook_secret`` is derived from the raw
storage row; the service-side projection drops the secret column, so
projection-derived records report ``False`` (a deferred seam until the
projection carries the flag).

Public-config agent metadata (agent name / avatar / knowledge-base
selection / agent capability flags) requires the agent service, which is
not wired into this layer yet; those fields stay at their defaults and
are documented per-field.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from src.common.exception import ValidationError
from src.common.json import JsonObject, JsonValue
from src.core.channels.embed.service.embed_channel_service import (
    EmbedChannelCreateRequest,
    EmbedChannelUpdateRequest,
    normalize_default_locale,
    normalize_header_title_mode,
    normalize_widget_position,
)
from src.core.channels.embed.types import EmbedChannelInfo, EmbedChannelOwnedInfo

#: Cap on the channel suggested-questions ``limit`` query param.
EMBED_SUGGESTION_LIMIT_CAP: int = 12
#: Default display title when the channel declares no page / channel title.
DEFAULT_DISPLAY_TITLE: str = "AI Assistant"
#: Suppression reason used when a channel disables suggested questions.
EMBED_SUPPRESSION_REASON_CHANNEL_DISABLED: str = "channel_disabled"


# ── Request bodies (mirrors the upstream ``embedChannelRequest``) ──────


class EmbedChannelRequest(BaseModel):
    """Body shared by the create and update endpoints.

    ``None`` on the pointer-shaped fields (``enabled``,
    ``show_suggested_questions``, ``allow_web_search``,
    ``allow_file_upload``, ``default_locale``, ``webhook_url``,
    ``webhook_secret``, ``agent_id``, ``allowed_origins``) means "leave
    unchanged" for update and "apply the default" for create, matching
    the upstream ``*bool`` / ``*string`` pointer semantics.
    """

    model_config = ConfigDict(frozen=True)

    name: str = ""
    enabled: bool | None = None
    allowed_origins: list[str] | None = None
    welcome_message: str = ""
    rate_limit_per_minute: int = 0
    rate_limit_per_day: int = 0
    primary_color: str = ""
    page_title: str = ""
    header_title_mode: str = ""
    show_suggested_questions: bool | None = None
    widget_position: str = ""
    allow_web_search: bool | None = None
    allow_file_upload: bool | None = None
    default_locale: str | None = None
    webhook_url: str | None = None
    webhook_secret: str | None = None
    agent_id: str | None = None


def to_create_request(body: EmbedChannelRequest) -> EmbedChannelCreateRequest:
    """Map a create body onto the core service DTO with upstream defaults.

    The ``*bool`` fields default the same way the upstream create handler
    does (``enabled`` / ``show_suggested_questions`` true,
    ``allow_web_search`` / ``allow_file_upload`` false); the origin
    allowlist is forwarded as-is and validated by the router.
    """
    return EmbedChannelCreateRequest(
        name=body.name,
        enabled=body.enabled if body.enabled is not None else True,
        allowed_origins=list(body.allowed_origins or []),
        welcome_message=body.welcome_message,
        rate_limit_per_minute=body.rate_limit_per_minute,
        rate_limit_per_day=body.rate_limit_per_day,
        primary_color=body.primary_color,
        page_title=body.page_title,
        header_title_mode=body.header_title_mode,
        show_suggested_questions=(
            body.show_suggested_questions if body.show_suggested_questions is not None else True
        ),
        widget_position=body.widget_position,
        allow_web_search=(body.allow_web_search if body.allow_web_search is not None else False),
        allow_file_upload=(body.allow_file_upload if body.allow_file_upload is not None else False),
        default_locale=body.default_locale or "",
    )


def to_update_request(body: EmbedChannelRequest) -> EmbedChannelUpdateRequest:
    """Map an update body onto the core service DTO (``None`` = unchanged)."""
    return EmbedChannelUpdateRequest(
        name=body.name,
        enabled=body.enabled,
        allowed_origins=(list(body.allowed_origins) if body.allowed_origins is not None else None),
        welcome_message=body.welcome_message,
        rate_limit_per_minute=body.rate_limit_per_minute,
        rate_limit_per_day=body.rate_limit_per_day,
        primary_color=body.primary_color,
        page_title=body.page_title,
        header_title_mode=body.header_title_mode,
        show_suggested_questions=body.show_suggested_questions,
        widget_position=body.widget_position,
        allow_web_search=body.allow_web_search,
        allow_file_upload=body.allow_file_upload,
        default_locale=body.default_locale,
        webhook_url=body.webhook_url,
        webhook_secret=body.webhook_secret,
        agent_id=body.agent_id,
    )


# ── Origin allowlist validation (mirrors the upstream handler) ────────


def is_production_mode() -> bool:
    """Whether the server runs in hardened (release) mode."""
    return os.getenv("GIN_MODE", "").strip().lower() == "release"


def validate_allowed_origins(origins: list[str] | None) -> None:
    """Enforce the public-channel origin allowlist rules.

    An empty list is rejected outright (a publicly reachable widget must
    declare an explicit allowlist). In production a wildcard (``"*"``)
    is rejected too; every entry must be a well-formed ``http(s)``
    origin, optionally a ``*.`` subdomain wildcard.
    """
    cleaned = [o.strip() for o in origins or [] if o.strip()]
    if not cleaned:
        raise ValidationError(
            code="embed.origin_required",
            message="at least one allowed origin is required",
        )
    for origin in cleaned:
        if origin == "*":
            if is_production_mode():
                raise ValidationError(
                    code="embed.origin_wildcard_prohibited",
                    message="wildcard origin '*' is not allowed in production",
                )
            continue
        host = origin
        if origin.startswith("*."):
            host = "https://" + origin[2:]
        parsed = urlparse(host)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValidationError(
                code="embed.origin_invalid",
                message=f"invalid allowed origin: {origin!r}",
            )


# ── Response records (mirrors the upstream ``embedChannelResponse``) ───


class EmbedChannelRecord(BaseModel):
    """One embed channel on the management wire.

    ``publish_token`` is rendered only when the caller created, rotated,
    or explicitly fetched the channel; list responses leave it ``None``.
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
    has_webhook_secret: bool
    created_at: datetime
    updated_at: datetime
    publish_token: str | None = None


class EmbedChannelEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` — single-channel responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: EmbedChannelRecord


class EmbedChannelListEnvelope(BaseModel):
    """``{"success": true, "data": [...]}`` — channel list responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: list[EmbedChannelRecord]


class SimpleSuccessResponse(BaseModel):
    """``{"success": true}`` — delete / ack responses."""

    model_config = ConfigDict(frozen=True)

    success: bool = True


class EmbedSessionTokenData(BaseModel):
    """``{"session_token": "...", "expires_in": N}``."""

    model_config = ConfigDict(frozen=True)

    session_token: str
    expires_in: int


class EmbedSessionTokenEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` — session-token responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: EmbedSessionTokenData


class EmbedSessionCreateData(BaseModel):
    """``{"id": "...", "sig": "..."}`` — a newly created embed session."""

    model_config = ConfigDict(frozen=True)

    id: str
    sig: str


class EmbedSessionCreateEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` — create-session response."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: EmbedSessionCreateData


class EmbedPublicConfig(BaseModel):
    """Config served to anonymous embed clients (no secrets).

    Agent-derived metadata (``agent_name`` / ``agent_avatar`` /
    ``knowledge_base_ids`` / ``agent_web_search_enabled`` /
    ``agent_image_upload_enabled``) requires the agent service, which is
    a deferred seam in this layer — those fields stay at their defaults.
    """

    model_config = ConfigDict(frozen=True)

    channel_id: str
    name: str
    display_title: str
    knowledge_base_ids: list[str] = Field(default_factory=list)
    agent_id: str
    agent_name: str = ""
    agent_avatar: str = ""
    welcome_message: str
    primary_color: str = ""
    page_title: str = ""
    header_title_mode: str
    show_suggested_questions: bool
    allowed_origins: list[str] = Field(default_factory=list)
    widget_position: str
    allow_web_search: bool
    allow_file_upload: bool
    agent_web_search_enabled: bool = False
    agent_image_upload_enabled: bool = False
    default_locale: str = ""


class EmbedConfigEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` — public-config response."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: EmbedPublicConfig


class EmbedSuggestedQuestion(BaseModel):
    """One channel-level suggested question."""

    model_config = ConfigDict(frozen=True)

    question: str = ""
    source: str = ""
    knowledge_base_id: str | None = None


class EmbedSuggestedQuestionsData(BaseModel):
    """``{"questions": [...]}`` — the suggested-questions payload."""

    model_config = ConfigDict(frozen=True)

    questions: list[EmbedSuggestedQuestion] = Field(default_factory=list)


class EmbedSuggestedQuestionsEnvelope(BaseModel):
    """``{"success": true, "data": {"questions": [...]}}``."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: EmbedSuggestedQuestionsData


class EmbedChunkEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` — a chunk payload."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: JsonObject


class EmbedStatsData(BaseModel):
    """``{"session_count": N}`` — embed-channel usage stats."""

    model_config = ConfigDict(frozen=True)

    session_count: int


class EmbedStatsEnvelope(BaseModel):
    """``{"success": true, "data": {"session_count": N}}``."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: EmbedStatsData


class EmbedWebhookEventRequest(BaseModel):
    """Body of the embed webhook relay endpoint."""

    model_config = ConfigDict(frozen=True)

    type: str = ""
    session_id: str = ""
    query: str = ""
    content: str = ""


class EmbedWebhookAckResponse(BaseModel):
    """``{"success": true}`` — webhook relay acknowledgement."""

    model_config = ConfigDict(frozen=True)

    success: bool = True


class EmbedSuggestionSuppressedData(BaseModel):
    """The suppressed-suggestions payload when a channel disables them."""

    model_config = ConfigDict(frozen=True)

    status: str = "suppressed"
    suppression_reason: str = EMBED_SUPPRESSION_REASON_CHANNEL_DISABLED
    questions: list[EmbedSuggestedQuestion] = Field(default_factory=list)


class EmbedSuggestionSuppressedEnvelope(BaseModel):
    """``{"success": true, "data": {"status": "suppressed", ...}}``."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: EmbedSuggestionSuppressedData


# ── Projections ───────────────────────────────────────────────────────


def _as_origins(value: JsonValue | list[str]) -> list[str]:
    """Narrow the JSONB ``allowed_origins`` column onto a concrete list."""
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, str):
        return [value]
    return []


def embed_channel_record(
    info: EmbedChannelInfo,
    *,
    publish_token: str = "",
) -> EmbedChannelRecord:
    """Project the service DTO onto the wire record.

    ``has_webhook_secret`` stays ``False`` for projection-derived records
    because the projection drops the secret column; the single-get
    endpoint reads the raw row (see :func:`embed_channel_record_from_row`)
    and reports it accurately.
    """
    return EmbedChannelRecord(
        id=info.id,
        tenant_id=info.tenant_id,
        agent_id=info.agent_id,
        name=info.name,
        enabled=info.enabled,
        allowed_origins=list(info.allowed_origins),
        welcome_message=info.welcome_message,
        rate_limit_per_minute=info.rate_limit_per_minute,
        rate_limit_per_day=info.rate_limit_per_day,
        primary_color=info.primary_color,
        page_title=info.page_title,
        header_title_mode=info.header_title_mode,
        show_suggested_questions=info.show_suggested_questions,
        widget_position=info.widget_position,
        allow_web_search=info.allow_web_search,
        allow_file_upload=info.allow_file_upload,
        default_locale=info.default_locale,
        webhook_url=info.webhook_url,
        has_webhook_secret=False,
        created_at=info.created_at,
        updated_at=info.updated_at,
        publish_token=publish_token or None,
    )


def embed_channel_record_from_row(
    row: EmbedChannelOwnedInfo,
    *,
    publish_token: str = "",
) -> EmbedChannelRecord:
    """Project a raw storage row (with secrets) onto the wire record.

    Used by the single-get endpoint so ``publish_token`` and
    ``has_webhook_secret`` are reported accurately for admins copying
    deploy snippets.
    """
    return EmbedChannelRecord(
        id=row.id,
        tenant_id=row.tenant_id,
        agent_id=row.agent_id,
        name=row.name,
        enabled=row.enabled,
        allowed_origins=_as_origins(row.allowed_origins),
        welcome_message=row.welcome_message,
        rate_limit_per_minute=row.rate_limit_per_minute,
        rate_limit_per_day=row.rate_limit_per_day,
        primary_color=row.primary_color,
        page_title=row.page_title,
        header_title_mode=normalize_header_title_mode(row.header_title_mode),
        show_suggested_questions=row.show_suggested_questions,
        widget_position=normalize_widget_position(row.widget_position),
        allow_web_search=row.allow_web_search,
        allow_file_upload=row.allow_file_upload,
        default_locale=normalize_default_locale(row.default_locale),
        webhook_url=row.webhook_url,
        has_webhook_secret=row.has_webhook_secret,
        created_at=row.created_at,
        updated_at=row.updated_at,
        publish_token=publish_token or row.publish_token or None,
    )


def embed_public_config(channel: EmbedChannelInfo) -> EmbedPublicConfig:
    """Project a resolved channel onto the anonymous public config.

    ``display_title`` follows the upstream resolution order (page title,
    then channel name, then the "AI Assistant" fallback) minus the agent
    name step, which is a deferred seam.
    """
    page_title = channel.page_title.strip()
    name = channel.name.strip()
    display_title = page_title or name or DEFAULT_DISPLAY_TITLE
    return EmbedPublicConfig(
        channel_id=channel.id,
        name=channel.name,
        display_title=display_title,
        agent_id=channel.agent_id,
        welcome_message=channel.welcome_message,
        primary_color=channel.primary_color,
        page_title=channel.page_title,
        header_title_mode=normalize_header_title_mode(channel.header_title_mode),
        show_suggested_questions=channel.show_suggested_questions,
        allowed_origins=_as_origins(channel.allowed_origins),
        widget_position=normalize_widget_position(channel.widget_position),
        allow_web_search=channel.allow_web_search,
        allow_file_upload=channel.allow_file_upload,
        default_locale=normalize_default_locale(channel.default_locale),
    )


def clamp_suggestion_limit(limit: int) -> int:
    """Coerce a suggested-questions ``limit`` query onto the embed cap.

    ``0`` means "unspecified" (the channel agent's starter count would
    apply upstream); a positive value is honoured up to the cap of 12.
    """
    if limit <= 0:
        return 0
    return min(limit, EMBED_SUGGESTION_LIMIT_CAP)


# ── Chat payload patching (mirrors the upstream delegateEmbedChat) ─────


def patch_embed_chat_payload(
    raw: bytes,
    channel: EmbedChannelInfo,
    *,
    agent_mode: bool,
) -> dict[str, JsonValue]:
    """Merge embed-channel constraints into a client QA payload.

    Forces the channel's agent, clears the knowledge-base and MCP scopes
    (the channel owns retrieval), keeps the visitor web-search toggle
    only when the channel allows it, strips file-upload fields when the
    channel forbids uploads, and pins ``agent_enabled``. Invalid JSON
    raises ``ValidationError`` (``embed.invalid_chat_json``).
    """
    if raw:
        try:
            decoded = json.loads(raw)
        except ValueError as exc:
            raise ValidationError(
                code="embed.invalid_chat_json",
                message="invalid json",
            ) from exc
        if not isinstance(decoded, dict):
            raise ValidationError(
                code="embed.invalid_chat_json",
                message="invalid json",
            )
        payload = decoded
    else:
        payload = {}

    payload["agent_id"] = channel.agent_id
    payload["knowledge_base_ids"] = []
    client_web_search = bool(payload.get("web_search_enabled", False))
    payload["web_search_enabled"] = channel.allow_web_search and client_web_search
    if not channel.allow_file_upload:
        payload.pop("images", None)
        payload.pop("attachment_uploads", None)
        payload.pop("attachment_ids", None)
    payload["mcp_service_ids"] = []
    payload["agent_enabled"] = agent_mode
    return payload


__all__ = [
    "DEFAULT_DISPLAY_TITLE",
    "EMBED_SUGGESTION_LIMIT_CAP",
    "EmbedChannelEnvelope",
    "EmbedChannelListEnvelope",
    "EmbedChannelRecord",
    "EmbedChannelRequest",
    "EmbedChunkEnvelope",
    "EmbedConfigEnvelope",
    "EmbedPublicConfig",
    "EmbedSessionCreateData",
    "EmbedSessionCreateEnvelope",
    "EmbedSessionTokenData",
    "EmbedSessionTokenEnvelope",
    "EmbedStatsData",
    "EmbedStatsEnvelope",
    "EmbedSuggestedQuestion",
    "EmbedSuggestedQuestionsData",
    "EmbedSuggestedQuestionsEnvelope",
    "EmbedSuggestionSuppressedData",
    "EmbedSuggestionSuppressedEnvelope",
    "EmbedWebhookAckResponse",
    "EmbedWebhookEventRequest",
    "SimpleSuccessResponse",
    "clamp_suggestion_limit",
    "embed_channel_record",
    "embed_channel_record_from_row",
    "embed_public_config",
    "is_production_mode",
    "patch_embed_chat_payload",
    "to_create_request",
    "to_update_request",
    "validate_allowed_origins",
]
