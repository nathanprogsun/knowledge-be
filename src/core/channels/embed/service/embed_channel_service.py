"""Embed channel CRUD service.

Request-scoped (constructed per request by ``factory.build_embed_channel_service``)
mirroring the upstream embed-channel service contract.

Owns the embed-channel surface an authenticated tenant administrator
sees: list channels per agent, fetch one channel, create with a
freshly generated publish token, update the mutable subset of fields,
soft-delete, and rotate the publish token. The anonymous embed client
flow (publish-token auth, origin gating, rate limits, session
creation) lives in :mod:`src.core.channels.embed.session`; the outbound
webhook plumbing lives in :mod:`src.core.channels.embed.webhook`.

The publish token is generated here on create and on
``rotate_token`` — never on update — and is the only secret the service
ever returns over the wire. The projection in
:mod:`src.core.channels.embed.types` drops it from every other
response shape.
"""

from __future__ import annotations

import base64
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Final, Protocol, runtime_checkable

from src.common.exception import (
    NotFoundError,
    ValidationError,
)
from src.common.json import BindParams
from src.core.agents.service.custom_agent_service import CustomAgentService
from src.core.agents.types import CustomAgentInfo
from src.core.channels.embed.types import (
    EMBED_DEFAULT_HEADER_TITLE_MODE,
    EMBED_DEFAULT_RATE_LIMIT_PER_DAY,
    EMBED_DEFAULT_WIDGET_POSITION,
    EMBED_HEADER_TITLE_MODE_SESSION,
    EMBED_SUPPORTED_LOCALES,
    EMBED_WIDGET_POSITIONS,
    EmbedChannelInfo,
)
from src.core.channels.embed.webhook import validate_embed_webhook_url
from src.db.dao.embed_channel_repository import EmbedChannelRepository
from src.db.models.embed_channel import EMBED_DEFAULT_AGENT_ID, EmbedChannel

# ── Constants ────────────────────────────────────────────────────────

#: Bytes of randomness packed into each publish token. Mirrors the
#: upstream ``embedTokenBytes``.
_EMBED_TOKEN_BYTES: Final[int] = 32
#: Prefix tagging publish tokens so the middleware can distinguish
#: them from session tokens without a round-trip.
EMBED_PUBLISH_TOKEN_PREFIX: Final[str] = "em_"
#: Default rate limit applied when the create request omits one.
EMBED_DEFAULT_RATE_LIMIT_PER_MINUTE: Final[int] = 30


# ── Errors ───────────────────────────────────────────────────────────


class EmbedChannelNotFoundError(NotFoundError):
    """Raised when the channel does not exist (or is soft-deleted)."""

    code = "embed.channel_not_found"


# ── Request DTOs ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class EmbedChannelCreateRequest:
    """Mutable subset of fields accepted by :meth:`EmbedChannelService.create_channel`.

    Optional fields with no caller-supplied value fall back to the
    upstream defaults (applied by the service, not the model) so the
    request shape stays explicit.
    """

    name: str = ""
    enabled: bool = True
    allowed_origins: list[str] = field(default_factory=list)
    welcome_message: str = ""
    rate_limit_per_minute: int = 0
    rate_limit_per_day: int = 0
    primary_color: str = ""
    page_title: str = ""
    header_title_mode: str = ""
    show_suggested_questions: bool = True
    widget_position: str = ""
    allow_web_search: bool = False
    allow_file_upload: bool = False
    default_locale: str = ""


@dataclass(frozen=True)
class EmbedChannelUpdateRequest:
    """Mutable subset accepted by :meth:`EmbedChannelService.update_channel`.

    ``None`` on any field means "leave unchanged" (matches the upstream
    ``*bool`` / ``*string`` pointer semantics). Validation, when
    supplied, follows the same rules as create.
    """

    name: str | None = None
    enabled: bool | None = None
    allowed_origins: list[str] | None = None
    welcome_message: str | None = None
    rate_limit_per_minute: int | None = None
    rate_limit_per_day: int | None = None
    primary_color: str | None = None
    page_title: str | None = None
    header_title_mode: str | None = None
    show_suggested_questions: bool | None = None
    widget_position: str | None = None
    allow_web_search: bool | None = None
    allow_file_upload: bool | None = None
    default_locale: str | None = None
    webhook_url: str | None = None
    webhook_secret: str | None = None
    agent_id: str | None = None


# ── Normalizers ──────────────────────────────────────────────────────


def normalize_widget_position(value: str) -> str:
    """Return a supported widget corner or the default.

    Mirrors ``NormalizeEmbedWidgetPosition``: invalid / empty values
    fall back to the default rather than raising so a stale front-end
    payload cannot block a legitimate update.
    """
    cleaned = value.strip()
    if cleaned in EMBED_WIDGET_POSITIONS:
        return cleaned
    return EMBED_DEFAULT_WIDGET_POSITION


def normalize_header_title_mode(value: str) -> str:
    """Return a supported header-title source or the default."""
    cleaned = value.strip()
    if cleaned == EMBED_HEADER_TITLE_MODE_SESSION:
        return EMBED_HEADER_TITLE_MODE_SESSION
    return EMBED_DEFAULT_HEADER_TITLE_MODE


def normalize_default_locale(value: str) -> str:
    """Return a supported locale tag or empty (browser-default follow)."""
    cleaned = value.strip()
    if cleaned in EMBED_SUPPORTED_LOCALES:
        return cleaned
    return ""


def generate_publish_token() -> str:
    """Mint a URL-safe-base64 publish token with the upstream prefix."""
    body = base64.urlsafe_b64encode(
        secrets.token_bytes(_EMBED_TOKEN_BYTES)
    ).rstrip(b"=")
    return EMBED_PUBLISH_TOKEN_PREFIX + body.decode()


# ── Agent ownership seam ─────────────────────────────────────────────


@runtime_checkable
class AgentOwnershipLike(Protocol):
    """Minimal seam to verify an agent belongs to the caller's tenant.

    The production factory wires :class:`CustomAgentService`; tests can
    substitute a stub. ``None`` disables the ownership check (defensive
    fallback for slim deployments where the agent directory lives
    elsewhere).
    """

    async def get_agent_by_id(
        self, *, tenant_id: int, agent_id: str
    ) -> CustomAgentInfo:
        """Return the owned agent or raise ``NotFoundError``."""
        ...


class _CustomAgentAdapter:
    """Adapt :class:`CustomAgentService` to the ``AgentOwnershipLike`` seam."""

    def __init__(self, service: CustomAgentService) -> None:
        self._service = service

    async def get_agent_by_id(
        self, *, tenant_id: int, agent_id: str
    ) -> CustomAgentInfo:
        return await self._service.get_agent_by_id(
            tenant_id=tenant_id, agent_id=agent_id
        )


# ── Service ──────────────────────────────────────────────────────────


class EmbedChannelService:
    """Request-scoped embed channel CRUD service."""

    def __init__(
        self,
        *,
        repo: EmbedChannelRepository,
        agent_ownership: AgentOwnershipLike | None = None,
    ) -> None:
        self._repo = repo
        self._agent_ownership = agent_ownership

    @classmethod
    def with_custom_agent_service(
        cls,
        *,
        repo: EmbedChannelRepository,
        custom_agent_service: CustomAgentService,
    ) -> EmbedChannelService:
        """Convenience constructor wiring :class:`CustomAgentService`."""
        return cls(
            repo=repo,
            agent_ownership=_CustomAgentAdapter(custom_agent_service),
        )

    # ── Public CRUD ────────────────────────────────────────────────

    async def create_channel(
        self,
        *,
        tenant_id: int,
        agent_id: str,
        request: EmbedChannelCreateRequest,
    ) -> tuple[EmbedChannelInfo, str]:
        """Create a new embed channel and return ``(projection, publish_token)``.

        ``agent_id`` must refer to an agent owned by ``tenant_id``; the
        service refuses the call when the agent service raises. The
        returned ``publish_token`` is the one and only time the token
        crosses the wire — every later read returns the
        :class:`EmbedChannelInfo` projection which omits it.
        """
        cleaned_agent_id = await self._ensure_agent_owned(
            tenant_id=tenant_id, agent_id=agent_id
        )
        token = generate_publish_token()
        now = _now()
        row = EmbedChannel(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            agent_id=cleaned_agent_id,
            name=request.name.strip(),
            enabled=request.enabled,
            publish_token=token,
            allowed_origins=list(request.allowed_origins),
            welcome_message=request.welcome_message,
            rate_limit_per_minute=(
                request.rate_limit_per_minute
                if request.rate_limit_per_minute > 0
                else EMBED_DEFAULT_RATE_LIMIT_PER_MINUTE
            ),
            rate_limit_per_day=(
                request.rate_limit_per_day
                if request.rate_limit_per_day > 0
                else EMBED_DEFAULT_RATE_LIMIT_PER_DAY
            ),
            primary_color=request.primary_color.strip(),
            page_title=request.page_title.strip(),
            header_title_mode=normalize_header_title_mode(
                request.header_title_mode
            ),
            show_suggested_questions=request.show_suggested_questions,
            widget_position=normalize_widget_position(
                request.widget_position
            ),
            allow_web_search=request.allow_web_search,
            allow_file_upload=request.allow_file_upload,
            default_locale=normalize_default_locale(request.default_locale),
            webhook_url="",
            webhook_secret="",
            created_at=now,
            updated_at=now,
        )
        persisted = await self._repo.create(row)
        return EmbedChannelInfo.map_from_db(persisted), token

    async def get_channel(
        self, *, tenant_id: int, channel_id: str
    ) -> EmbedChannelInfo:
        """Return one channel by id, scoped to ``tenant_id``."""
        self._require_channel_id(channel_id)
        row = await self._get_owned_row(
            tenant_id=tenant_id, channel_id=channel_id
        )
        return EmbedChannelInfo.map_from_db(row)

    async def get_owned_channel(
        self, *, tenant_id: int, channel_id: str
    ) -> EmbedChannel:
        """Return the raw row (with secrets) for internal callers."""
        self._require_channel_id(channel_id)
        return await self._get_owned_row(
            tenant_id=tenant_id, channel_id=channel_id
        )

    async def list_channels_by_agent(
        self, *, tenant_id: int, agent_id: str
    ) -> list[EmbedChannelInfo]:
        """Return every live channel of ``agent_id`` within the tenant."""
        await self._ensure_agent_owned(
            tenant_id=tenant_id, agent_id=agent_id
        )
        rows = await self._repo.list_by_agent(tenant_id, agent_id)
        return [EmbedChannelInfo.map_from_db(r) for r in rows]

    async def list_channels_by_tenant(
        self, *, tenant_id: int
    ) -> list[EmbedChannelInfo]:
        """Return every live channel of the tenant, across agents."""
        if tenant_id <= 0:
            raise ValidationError(
                code="embed.tenant_id_required",
                message="tenant id is required",
            )
        rows = await self._repo.list_by_tenant(tenant_id)
        return [EmbedChannelInfo.map_from_db(r) for r in rows]

    async def update_channel(
        self,
        *,
        tenant_id: int,
        channel_id: str,
        request: EmbedChannelUpdateRequest,
    ) -> EmbedChannelInfo:
        """Apply the mutable subset of channel fields; ``None`` means unchanged."""
        self._require_channel_id(channel_id)
        existing = await self._get_owned_row(
            tenant_id=tenant_id, channel_id=channel_id
        )

        updates: BindParams = {}
        if request.name is not None:
            updates["name"] = request.name.strip()
        if request.welcome_message is not None:
            updates["welcome_message"] = request.welcome_message
        if request.primary_color is not None:
            updates["primary_color"] = request.primary_color.strip()
        if request.page_title is not None:
            updates["page_title"] = request.page_title.strip()
        if request.header_title_mode is not None:
            updates["header_title_mode"] = normalize_header_title_mode(
                request.header_title_mode
            )
        if request.show_suggested_questions is not None:
            updates["show_suggested_questions"] = (
                request.show_suggested_questions
            )
        if request.allow_web_search is not None:
            updates["allow_web_search"] = request.allow_web_search
        if request.allow_file_upload is not None:
            updates["allow_file_upload"] = request.allow_file_upload
        if request.default_locale is not None:
            updates["default_locale"] = normalize_default_locale(
                request.default_locale
            )
        if request.widget_position is not None:
            updates["widget_position"] = normalize_widget_position(
                request.widget_position
            )
        if request.enabled is not None:
            updates["enabled"] = request.enabled
        if (
            request.rate_limit_per_minute is not None
            and request.rate_limit_per_minute > 0
        ):
            updates["rate_limit_per_minute"] = request.rate_limit_per_minute
        if (
            request.rate_limit_per_day is not None
            and request.rate_limit_per_day > 0
        ):
            updates["rate_limit_per_day"] = request.rate_limit_per_day
        if request.allowed_origins is not None:
            updates["allowed_origins"] = list(request.allowed_origins)
        if request.webhook_url is not None:
            await validate_embed_webhook_url(request.webhook_url)
            updates["webhook_url"] = request.webhook_url.strip()
        if request.webhook_secret is not None:
            updates["webhook_secret"] = request.webhook_secret.strip()
        if request.agent_id is not None:
            cleaned_agent = request.agent_id.strip()
            if cleaned_agent and cleaned_agent != existing.agent_id:
                new_agent_id = await self._ensure_agent_owned(
                    tenant_id=tenant_id, agent_id=cleaned_agent
                )
                updates["agent_id"] = new_agent_id

        updates["updated_at"] = _now()
        updated_row = existing.model_copy(update=updates)
        persisted = await self._repo.update(updated_row)
        return EmbedChannelInfo.map_from_db(persisted)

    async def delete_channel(
        self, *, tenant_id: int, channel_id: str
    ) -> None:
        """Soft-delete a channel owned by ``tenant_id``."""
        self._require_channel_id(channel_id)
        await self._get_owned_row(
            tenant_id=tenant_id, channel_id=channel_id
        )
        await self._repo.soft_delete(
            channel_id=channel_id, tenant_id=tenant_id, now=_now()
        )

    async def rotate_token(
        self, *, tenant_id: int, channel_id: str
    ) -> tuple[EmbedChannelInfo, str]:
        """Mint a fresh publish token; outstanding handles are invalidated."""
        self._require_channel_id(channel_id)
        existing = await self._get_owned_row(
            tenant_id=tenant_id, channel_id=channel_id
        )
        rotated = existing.model_copy(
            update={
                "publish_token": generate_publish_token(),
                "updated_at": _now(),
            }
        )
        persisted = await self._repo.update(rotated)
        return EmbedChannelInfo.map_from_db(persisted), persisted.publish_token

    # ── Internal helpers ───────────────────────────────────────────

    async def _get_owned_row(
        self, *, tenant_id: int, channel_id: str
    ) -> EmbedChannel:
        if tenant_id <= 0:
            raise ValidationError(
                code="embed.tenant_id_required",
                message="tenant id is required",
            )
        row = await self._repo.get_by_id(channel_id)
        if row is None or row.tenant_id != tenant_id:
            raise EmbedChannelNotFoundError(
                code="embed.channel_not_found",
                message=f"embed channel {channel_id} not found",
            )
        return row

    async def _ensure_agent_owned(
        self, *, tenant_id: int, agent_id: str
    ) -> str:
        """Verify the agent belongs to the tenant; return the trimmed id.

        Defaults a blank id to the built-in quick-answer agent so the
        caller's behaviour matches the upstream ``BuiltinQuickAnswerID``
        fallback. When no ownership seam is wired (or the agent service
        is unavailable), the trim / default still runs and the tenant
        scoping is enforced by the storage layer downstream.
        """
        cleaned = agent_id.strip()
        if not cleaned:
            cleaned = EMBED_DEFAULT_AGENT_ID
        if self._agent_ownership is None:
            return cleaned
        try:
            await self._agent_ownership.get_agent_by_id(
                tenant_id=tenant_id, agent_id=cleaned
            )
        except NotFoundError as exc:
            # The upstream ``ensureAgentOwned`` surfaces a generic
            # not-found; surface the embed flavour so the views layer
            # can map it to the same 404 family.
            raise EmbedChannelNotFoundError(
                code="embed.agent_not_found",
                message="agent not found",
            ) from exc
        return cleaned

    @staticmethod
    def _require_channel_id(channel_id: str) -> None:
        if not channel_id or not channel_id.strip():
            raise ValidationError(
                code="embed.channel_id_required",
                message="channel id is required",
            )


# ── Internal helpers ────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "EMBED_DEFAULT_RATE_LIMIT_PER_MINUTE",
    "EMBED_PUBLISH_TOKEN_PREFIX",
    "AgentOwnershipLike",
    "EmbedChannelCreateRequest",
    "EmbedChannelInfo",
    "EmbedChannelNotFoundError",
    "EmbedChannelService",
    "EmbedChannelUpdateRequest",
    "generate_publish_token",
    "normalize_default_locale",
    "normalize_header_title_mode",
    "normalize_widget_position",
]
