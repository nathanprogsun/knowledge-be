"""IM-channel service — CRUD, adapter registry, runtime delegation.

Combines the channel CRUD operations with the adapter registry and
runtime delegation to the process-wide :class:`IMSupervisor`. The
bot-identity derivation keeps the service-level duplicate guard
aligned with the database unique index on ``bot_identity``.

Request-scoped: built per HTTP request by ``build_im_channel_service``
with a fresh repository on the shared ``AsyncSession``; the web layer
never imports ``db``. The supervisor is process-wide (the runtime
connection map outlives a single request), so the factory injects
the module-level default supervisor instance.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.parse
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from src.common.exception import (
    ConflictError,
    ExternalServiceError,
    NotFoundError,
    ValidationError,
)
from src.common.json import JsonObject
from src.core.channels.im.adapter_base import IMAdapter
from src.core.channels.im.supervisor import (
    AdapterFactory,
    IMSupervisor,
    get_default_supervisor,
)
from src.core.channels.im.types import (
    IM_MODE_WEBHOOK,
    IM_MODE_WEBSOCKET,
    IM_OUTPUT_MODE_STREAM,
    IM_PLATFORMS,
    IM_SESSION_MODE_THREAD,
    IM_SESSION_MODE_USER,
    IMChannelInfo,
)
from src.db.dao.im_channel_repository import IMChannelRepository
from src.db.models.im_channel import IMChannel

logger = logging.getLogger("src.core.channels.im.service")


# ── Bot-identity derivation ──────────────────────────────────────────

_WEBHOOK_PLATFORMS: frozenset[str] = frozenset({"mattermost", "yunzhijia"})
_WECHAT_MODE: str = "longpoll"
_WECHAT_OUTPUT_MODE: str = "full"
_SESSION_MODES: frozenset[str] = frozenset({IM_SESSION_MODE_USER, IM_SESSION_MODE_THREAD})


def _credential_string(credentials: JsonObject, key: str) -> str:
    """Read ``key`` from ``credentials`` as a string.

    Booleans are explicitly rejected so True/False never round-trip
    into a bot identity; numbers are rendered without a decimal
    point, matching the platform identity fields.
    """
    value = credentials.get(key)
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.0f}"
    return ""


def compute_bot_identity(platform: str, mode: str, credentials: JsonObject) -> str:
    """Derive the unique bot identity for a channel.

    Returns ``""`` when the credentials do not contain a
    platform-specific identity field, so the duplicate guard skips
    the database lookup.
    """
    normalised = platform.strip().lower()
    if normalised == "wecom":
        if mode == IM_MODE_WEBSOCKET:
            bot_id = _credential_string(credentials, "bot_id")
            if bot_id:
                return f"wecom:ws:{bot_id}"
        if mode == IM_MODE_WEBHOOK:
            corp_id = _credential_string(credentials, "corp_id")
            agent_id = _credential_string(credentials, "corp_agent_id")
            if corp_id and agent_id:
                return f"wecom:wh:{corp_id}:{agent_id}"
        return ""
    if normalised in ("feishu", "lark"):
        app_id = _credential_string(credentials, "app_id")
        if app_id:
            return f"{normalised}:{app_id}"
        return ""
    if normalised == "telegram":
        bot_token = _credential_string(credentials, "bot_token")
        if bot_token:
            head = bot_token.split(":", 1)[0]
            return f"telegram:{head or bot_token}"
        return ""
    if normalised == "dingtalk":
        client_id = _credential_string(credentials, "client_id")
        if client_id:
            return f"dingtalk:{client_id}"
        return ""
    if normalised == "mattermost":
        outgoing_token = _credential_string(credentials, "outgoing_token")
        if outgoing_token:
            return f"mattermost:wh:{outgoing_token}"
        return ""
    if normalised == "wechat":
        ilink_bot_id = _credential_string(credentials, "ilink_bot_id")
        if ilink_bot_id:
            return f"wechat:{ilink_bot_id}"
        return ""
    if normalised == "qqbot":
        app_id = _credential_string(credentials, "app_id")
        if app_id:
            return f"qqbot:{app_id}"
        return ""
    if normalised == "yunzhijia":
        send_msg_url = _credential_string(credentials, "send_msg_url")
        if not send_msg_url:
            return ""
        try:
            parsed = urllib.parse.urlparse(send_msg_url)
        except ValueError:
            return ""
        token = urllib.parse.parse_qs(parsed.query).get("yzjtoken", [""])[0].strip()
        if not token:
            return ""
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return f"yunzhijia:{digest}"
    return ""


# ── Defaults ─────────────────────────────────────────────────────────


def _default_mode(platform: str) -> str:
    """Per-platform default connection mode (matches the on-create defaults)."""
    if platform in _WEBHOOK_PLATFORMS:
        return IM_MODE_WEBHOOK
    return IM_MODE_WEBSOCKET


def _apply_create_defaults(
    *,
    platform: str,
    mode: str,
    output_mode: str,
) -> tuple[str, str]:
    """Return ``(mode, output_mode)`` with the documented defaults applied.

    WeChat always pins both (long-polling + full output). For other
    platforms the mode defaults to webhook on mattermost/yunzhijia
    and websocket everywhere else; ``output_mode`` defaults to
    streaming.
    """
    if platform == "wechat":
        return _WECHAT_MODE, _WECHAT_OUTPUT_MODE
    resolved_mode = mode or _default_mode(platform)
    resolved_output_mode = output_mode or IM_OUTPUT_MODE_STREAM
    return resolved_mode, resolved_output_mode


# ── Request payload (service-level create / update input) ────────────


@dataclass(frozen=True)
class ChannelCreateRequest:
    """Validated input for :meth:`IMChannelService.create_channel`.

    Captures every field the create path accepts; updates use
    :class:`ChannelUpdateRequest` instead.
    """

    agent_id: str
    platform: str
    name: str = ""
    mode: str = ""
    output_mode: str = ""
    knowledge_base_id: str = ""
    session_mode: str = IM_SESSION_MODE_USER
    credentials: JsonObject | None = None
    enabled: bool = True


@dataclass(frozen=True)
class ChannelUpdateRequest:
    """Partial update for :meth:`IMChannelService.update_channel`.

    Every mutable field is ``None`` until the caller supplies a value,
    so the service applies only the supplied changes and leaves the
    others untouched.
    """

    name: str | None = None
    mode: str | None = None
    output_mode: str | None = None
    knowledge_base_id: str | None = None
    session_mode: str | None = None
    credentials: JsonObject | None = None
    enabled: bool | None = None


# ── Service ──────────────────────────────────────────────────────────


class IMChannelService:
    """IM-channel CRUD + adapter registry + runtime delegation."""

    def __init__(
        self,
        *,
        channel_repo: IMChannelRepository,
        supervisor: IMSupervisor | None = None,
    ) -> None:
        self._channel_repo = channel_repo
        self._supervisor = supervisor if supervisor is not None else get_default_supervisor()

    # ── Adapter registry (delegated to the process-wide supervisor) ─

    def register_adapter_factory(self, platform: str, factory: AdapterFactory) -> None:
        """Register ``factory`` for ``platform`` on the supervisor."""
        self._supervisor.register_adapter_factory(platform, factory)

    def get_adapter_factory(self, platform: str) -> AdapterFactory:
        """Return the registered factory for ``platform``."""
        return self._supervisor.get_adapter_factory(platform)

    def registered_platforms(self) -> list[str]:
        """Return the platform identifiers with a registered factory."""
        return self._supervisor.registered_platforms()

    # ── Create ──────────────────────────────────────────────────────

    async def create_channel(
        self,
        *,
        tenant_id: int,
        request: ChannelCreateRequest,
    ) -> IMChannelInfo:
        """Persist a new channel and (best-effort) start it.

        The duplicate-bot guard runs before insert so the conflict
        surfaces as ``ConflictError`` (``im.duplicate_bot``) instead
        of a database constraint violation.
        """
        _require_tenant_id(tenant_id)
        _require_agent_id(request.agent_id)
        _require_platform(request.platform)
        resolved_mode, resolved_output_mode = _apply_create_defaults(
            platform=request.platform,
            mode=request.mode,
            output_mode=request.output_mode,
        )
        resolved_session_mode = _require_session_mode(request.session_mode)
        resolved_credentials = _require_credentials(request.credentials)
        resolved_name = request.name.strip()
        now = _now()
        bot_identity = compute_bot_identity(request.platform, resolved_mode, resolved_credentials)
        if bot_identity:
            await self._assert_bot_identity_unused(bot_identity, exclude_id="")

        row = IMChannel(
            id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            agent_id=request.agent_id,
            platform=request.platform,
            name=resolved_name,
            enabled=request.enabled,
            mode=resolved_mode,
            output_mode=resolved_output_mode,
            knowledge_base_id=request.knowledge_base_id,
            bot_identity=bot_identity,
            session_mode=resolved_session_mode,
            credentials=resolved_credentials,
            created_at=now,
            updated_at=now,
        )
        persisted = await self._channel_repo.create(row)

        if persisted.enabled:
            await self._safe_start(persisted)

        return IMChannelInfo.map_from_db(persisted)

    # ── Reads ───────────────────────────────────────────────────────

    async def get_channel(self, *, tenant_id: int, channel_id: str) -> IMChannelInfo:
        """Return one channel for the tenant, or raise ``NotFoundError``."""
        _require_tenant_id(tenant_id)
        if not channel_id.strip():
            raise ValidationError(
                code="im.channel_id_required",
                message="channel id is required",
            )
        row = await self._channel_repo.get_by_id(tenant_id, channel_id)
        if row is None:
            raise NotFoundError(
                code="im.channel_not_found",
                message=f"im channel {channel_id} not found",
            )
        return IMChannelInfo.map_from_db(row)

    async def list_channels(self, *, tenant_id: int) -> list[IMChannelInfo]:
        """Return every live channel of the tenant, newest first."""
        _require_tenant_id(tenant_id)
        rows = await self._channel_repo.list_by_tenant(tenant_id)
        return [IMChannelInfo.map_from_db(row) for row in rows]

    async def list_channels_by_agent(self, *, tenant_id: int, agent_id: str) -> list[IMChannelInfo]:
        """Return every live channel bound to ``agent_id`` in the tenant."""
        _require_tenant_id(tenant_id)
        _require_agent_id(agent_id)
        rows = await self._channel_repo.list_by_agent(tenant_id, agent_id)
        return [IMChannelInfo.map_from_db(row) for row in rows]

    # ── Update ──────────────────────────────────────────────────────

    async def update_channel(
        self,
        *,
        tenant_id: int,
        channel_id: str,
        request: ChannelUpdateRequest,
    ) -> IMChannelInfo:
        """Apply the fields in ``request`` and restart the channel."""
        _require_tenant_id(tenant_id)
        if not channel_id.strip():
            raise ValidationError(
                code="im.channel_id_required",
                message="channel id is required",
            )
        existing = await self._channel_repo.get_by_id(tenant_id, channel_id)
        if existing is None:
            raise NotFoundError(
                code="im.channel_not_found",
                message=f"im channel {channel_id} not found",
            )

        resolved_mode = request.mode if request.mode is not None else existing.mode
        resolved_output_mode = (
            request.output_mode if request.output_mode is not None else existing.output_mode
        )
        resolved_session_mode = (
            _require_session_mode(request.session_mode)
            if request.session_mode is not None
            else existing.session_mode
        )
        resolved_credentials = (
            _require_credentials(request.credentials)
            if request.credentials is not None
            else existing.credentials
        )
        resolved_knowledge_base_id = (
            request.knowledge_base_id
            if request.knowledge_base_id is not None
            else existing.knowledge_base_id
        )
        resolved_enabled = request.enabled if request.enabled is not None else existing.enabled
        resolved_name = request.name.strip() if request.name is not None else existing.name

        bot_identity = compute_bot_identity(existing.platform, resolved_mode, resolved_credentials)
        if bot_identity:
            await self._assert_bot_identity_unused(bot_identity, exclude_id=existing.id)

        # ``model_copy`` preserves immutability — the row model is
        # frozen, so the update returns a fresh row.
        updated = existing.model_copy(
            update={
                "name": resolved_name,
                "mode": resolved_mode,
                "output_mode": resolved_output_mode,
                "knowledge_base_id": resolved_knowledge_base_id,
                "session_mode": resolved_session_mode,
                "credentials": resolved_credentials,
                "enabled": resolved_enabled,
                "bot_identity": bot_identity,
                "updated_at": _now(),
            }
        )
        persisted = await self._channel_repo.update(updated)

        # Update restarts the active connection when enabled so the
        # new credentials / mode take effect.
        self._supervisor.stop_channel(persisted.id)
        if persisted.enabled:
            await self._safe_start(persisted)
        return IMChannelInfo.map_from_db(persisted)

    # ── Delete ──────────────────────────────────────────────────────

    async def delete_channel(self, *, tenant_id: int, channel_id: str) -> None:
        """Soft-delete one channel, stopping any active connection."""
        _require_tenant_id(tenant_id)
        if not channel_id.strip():
            raise ValidationError(
                code="im.channel_id_required",
                message="channel id is required",
            )
        deleted = await self._channel_repo.soft_delete(
            channel_id=channel_id, tenant_id=tenant_id, now=_now()
        )
        if not deleted:
            raise NotFoundError(
                code="im.channel_not_found",
                message=f"im channel {channel_id} not found",
            )
        self._supervisor.stop_channel(channel_id)

    async def delete_channels_by_agent(self, *, tenant_id: int, agent_id: str) -> int:
        """Soft-delete every channel of ``agent_id`` in the tenant.

        Stops each active connection before the row is removed.
        Returns the number of rows removed.
        """
        _require_tenant_id(tenant_id)
        _require_agent_id(agent_id)
        rows = await self._channel_repo.list_by_agent(tenant_id, agent_id)
        if not rows:
            return 0
        removed = await self._channel_repo.soft_delete_by_agent(
            agent_id=agent_id, tenant_id=tenant_id, now=_now()
        )
        for row in rows:
            self._supervisor.stop_channel(row.id)
        return removed

    # ── Toggle ──────────────────────────────────────────────────────

    async def toggle_channel_enabled(self, *, tenant_id: int, channel_id: str) -> IMChannelInfo:
        """Flip ``enabled`` on the channel, starting / stopping the runtime."""
        _require_tenant_id(tenant_id)
        if not channel_id.strip():
            raise ValidationError(
                code="im.channel_id_required",
                message="channel id is required",
            )
        toggled = await self._channel_repo.toggle_enabled(
            channel_id=channel_id, tenant_id=tenant_id, now=_now()
        )
        if toggled is None:
            raise NotFoundError(
                code="im.channel_not_found",
                message=f"im channel {channel_id} not found",
            )
        if toggled.enabled:
            await self._safe_start(toggled)
        else:
            self._supervisor.stop_channel(toggled.id)
        return IMChannelInfo.map_from_db(toggled)

    # ── Runtime (delegates to the supervisor) ──────────────────────

    async def start_channel(self, *, tenant_id: int, channel_id: str) -> None:
        """Load the row from the DB and ask the supervisor to connect it."""
        _require_tenant_id(tenant_id)
        if not channel_id.strip():
            raise ValidationError(
                code="im.channel_id_required",
                message="channel id is required",
            )
        row = await self._channel_repo.get_by_id(tenant_id, channel_id)
        if row is None:
            raise NotFoundError(
                code="im.channel_not_found",
                message=f"im channel {channel_id} not found",
            )
        await self._supervisor.start_channel(row)

    def stop_channel(self, channel_id: str) -> None:
        """Tear down the active connection for ``channel_id`` (if any)."""
        self._supervisor.stop_channel(channel_id)

    def get_channel_adapter(
        self, channel_id: str
    ) -> tuple[IMAdapter | None, IMChannel | None, bool]:
        """Return ``(adapter, channel, running)`` for the active channel."""
        return self._supervisor.get_channel_adapter(channel_id)

    async def ensure_channel_adapter(self, channel_id: str) -> tuple[IMAdapter, IMChannelInfo]:
        """Resolve the durable row and return a running adapter for ``channel_id``.

        Mirrors the runtime contract the callback path relies on: the
        live row is the source of truth, so a replica that missed a
        cache invalidation still re-reads credentials and config before
        routing a callback. A deleted row tears down the runtime and
        raises ``NotFoundError``; a disabled row tears down the runtime
        and raises an availability error. When the active adapter is
        missing or was built from a stale config, the channel is
        (re)started first.
        """
        row = await self._channel_repo.get_by_id_global(channel_id)
        if row is None:
            self._supervisor.stop_channel(channel_id)
            raise NotFoundError(
                code="im.channel_not_found",
                message=f"im channel {channel_id} not found",
            )
        if not row.enabled:
            self._supervisor.stop_channel(channel_id)
            raise ExternalServiceError(
                code="im.channel_disabled",
                message="channel is disabled",
            )
        adapter, cached, running = self._supervisor.get_channel_adapter(channel_id)
        if not running or cached is None or not _same_runtime_config(cached, row):
            try:
                await self._supervisor.start_channel(row)
            except Exception as exc:
                raise ExternalServiceError(
                    code="im.channel_not_available",
                    message="channel adapter is not active on this instance",
                ) from exc
            adapter, cached, running = self._supervisor.get_channel_adapter(channel_id)
        if not running or cached is None or adapter is None:
            raise ExternalServiceError(
                code="im.channel_not_available",
                message="channel adapter is not active on this instance",
            )
        return adapter, IMChannelInfo.map_from_db(cached)

    def active_channel_count(self) -> int:
        """Return the count of currently-running supervised channels."""
        return self._supervisor.active_channel_count()

    async def load_and_start_channels(self) -> tuple[int, list[str]]:
        """Load every enabled channel and (best-effort) start it.

        Returns ``(started_count, failed_channel_ids)`` — the same
        shape the supervisor surfaces to the lifespan wiring.
        """
        rows = await self._channel_repo.list_enabled()
        return await self._supervisor.start_enabled_channels(rows)

    # ── Shared guards ───────────────────────────────────────────────

    async def _assert_bot_identity_unused(self, bot_identity: str, *, exclude_id: str) -> None:
        """Raise ``ConflictError`` when ``bot_identity`` is bound to another live row."""
        existing = await self._channel_repo.find_by_bot_identity(
            bot_identity, exclude_id=exclude_id
        )
        if existing is None:
            return
        raise ConflictError(
            code="im.duplicate_bot",
            message=(
                "this bot is already bound to channel "
                f"{existing.name!r} ({existing.id}); each bot can only be connected to one channel"
            ),
        )

    async def _safe_start(self, row: IMChannel) -> None:
        """Best-effort start; logs but never raises.

        A start failure after a successful persist must not abort
        the mutation — the row is durable and a later health-check
        sweep can reconnect.
        """
        try:
            await self._supervisor.start_channel(row)
        except Exception:
            logger.exception("[IM] failed to start channel %s after persist", row.id)


# ── Boundary validators ──────────────────────────────────────────────


def _require_tenant_id(tenant_id: int) -> None:
    """Reject a non-positive tenant id."""
    if not isinstance(tenant_id, int) or tenant_id <= 0:
        raise ValidationError(
            code="im.tenant_required",
            message="tenant ID is required",
        )


def _require_agent_id(agent_id: str) -> None:
    """Reject a blank agent id."""
    if not agent_id.strip():
        raise ValidationError(
            code="im.agent_id_required",
            message="agent ID is required",
        )


def _require_platform(platform: str) -> None:
    """Reject an unsupported platform name."""
    normalised = platform.strip().lower()
    if normalised not in IM_PLATFORMS:
        raise ValidationError(
            code="im.platform_unsupported",
            message=f"platform must be one of: {', '.join(sorted(IM_PLATFORMS))}",
        )


def _require_session_mode(session_mode: str) -> str:
    """Validate session_mode, defaulting to ``user`` when blank."""
    resolved = session_mode.strip() or IM_SESSION_MODE_USER
    if resolved not in _SESSION_MODES:
        raise ValidationError(
            code="im.session_mode_invalid",
            message=f"session_mode must be one of: {', '.join(sorted(_SESSION_MODES))}",
        )
    return resolved


def _require_credentials(credentials: JsonObject | None) -> JsonObject:
    """Default ``credentials`` to ``{}`` and reject non-object inputs."""
    if credentials is None:
        return {}
    if not isinstance(credentials, dict):
        raise ValidationError(
            code="im.credentials_invalid",
            message="credentials must be an object",
        )
    return dict(credentials)


def _now() -> datetime:
    """Return a timezone-aware ``now`` for stamping channel rows."""
    return datetime.now(UTC)


def _same_runtime_config(cached: IMChannel, fresh: IMChannel) -> bool:
    """Compare the config-relevant fields that shape adapter creation.

    Timestamps are intentionally excluded (database precision differs
    from the in-memory value), so an unrelated update never forces a
    webhook adapter rebuild on every callback.
    """
    return (
        cached.id == fresh.id
        and cached.tenant_id == fresh.tenant_id
        and cached.agent_id == fresh.agent_id
        and cached.platform == fresh.platform
        and cached.enabled == fresh.enabled
        and cached.mode == fresh.mode
        and cached.output_mode == fresh.output_mode
        and cached.knowledge_base_id == fresh.knowledge_base_id
        and cached.session_mode == fresh.session_mode
        and cached.credentials == fresh.credentials
    )


# ── Factory ──────────────────────────────────────────────────────────


def build_im_channel_service(session: AsyncSession) -> IMChannelService:
    """Per-request ``IMChannelService`` with a fresh repository.

    Shares the process-wide supervisor singleton so adapter
    registrations and active connections survive across requests.
    """
    return IMChannelService(
        channel_repo=IMChannelRepository(session),
        supervisor=get_default_supervisor(),
    )


__all__ = [
    "ChannelCreateRequest",
    "ChannelUpdateRequest",
    "IMChannelService",
    "build_im_channel_service",
    "compute_bot_identity",
]
