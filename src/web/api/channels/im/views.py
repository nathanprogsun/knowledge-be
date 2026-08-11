"""Wire-shape conversion and view functions for the IM-channel endpoints.

Maps the IM-channel handler surface (agent-scoped and tenant-wide CRUD,
toggle, and the platform callback) onto Pydantic wire shapes. Field
names mirror the upstream contract exactly, including JSON serialization
names. Credentials are never rendered: the service projection drops the
secret column and ``credentials_configured`` is the only
credential-derived signal on the wire.

The callback path resolves the durable channel + running adapter via
the service, runs the platform URL-verification and signature checks,
parses the message, and dispatches slash-commands through the command
registry. Non-command messages are acknowledged and logged — the QA
pipeline is a later seam.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated

from fastapi import Depends, Request
from pydantic import BaseModel, ConfigDict

from src.common.exception import (
    ExternalServiceError,
    PermissionDeniedError,
    UnauthorizedError,
    ValidationError,
)
from src.common.json import JsonObject
from src.core.channels.im.adapter_base import (
    CallbackRequest,
    Context,
    EventContext,
    IMAdapter,
    IncomingMessage,
    ReplyMessage,
)
from src.core.channels.im.commands.registry import (
    Command,
    CommandAction,
    CommandContext,
    CommandRegistry,
)
from src.core.channels.im.service.im_channel_service import (
    ChannelCreateRequest,
    ChannelUpdateRequest,
)
from src.core.channels.im.types import IMChannelInfo
from src.web.deps.context import get_tenant_id_dep
from src.web.deps.im_channels import (
    IMChannelServiceDep,
    IMCommandRegistryDep,
)

logger = logging.getLogger("src.web.api.channels.im")

# Function-arg-style principal dep: the authenticated workspace id.
_PrincipalTenant = Annotated[int, Depends(get_tenant_id_dep)]


# ── Request bodies (mirror the upstream handler request structs) ───────


class IMChannelCreateRequest(BaseModel):
    """Create body (mirrors the upstream create handler).

    ``platform`` is required; ``enabled`` defaults to ``True`` when
    omitted, matching the upstream ``*bool`` pointer semantics.
    """

    model_config = ConfigDict(frozen=True)

    platform: str
    name: str = ""
    mode: str = ""
    output_mode: str = ""
    knowledge_base_id: str = ""
    credentials: JsonObject | None = None
    enabled: bool | None = None


class IMChannelUpdateRequest(BaseModel):
    """Partial-update body; ``None`` means leave unchanged.

    ``agent_id`` mirrors the upstream re-bind field; transferring a
    channel to another agent is a deferred seam in this layer.
    """

    model_config = ConfigDict(frozen=True)

    name: str | None = None
    mode: str | None = None
    output_mode: str | None = None
    knowledge_base_id: str | None = None
    credentials: JsonObject | None = None
    enabled: bool | None = None
    agent_id: str | None = None


# ── Response records (mirror the upstream summary shape) ───────────────


class IMChannelRecord(BaseModel):
    """One IM channel on the wire (summary shape — no credentials)."""

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
    credentials_configured: bool
    created_at: datetime
    updated_at: datetime


class IMChannelEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` — single-channel responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: IMChannelRecord


class IMChannelListEnvelope(BaseModel):
    """``{"success": true, "data": [...]}`` — channel list responses."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: list[IMChannelRecord]


class SimpleSuccessResponse(BaseModel):
    """``{"success": true}`` — delete / ack responses."""

    model_config = ConfigDict(frozen=True)

    success: bool = True


class IMCallbackData(BaseModel):
    """Platform-specific callback acknowledgement payload.

    Yunzhijia requires a typed payload on its webhook acknowledgement;
    other platforms acknowledge with ``{"success": true}`` alone.
    """

    model_config = ConfigDict(frozen=True)

    type: int = 0
    content: str = ""


class IMCallbackAckResponse(BaseModel):
    """Callback acknowledgement envelope.

    ``data`` is rendered only for platforms that require a typed
    payload (the router sets ``response_model_exclude_none`` so the
    common case serializes to ``{"success": true}``).
    """

    model_config = ConfigDict(frozen=True)

    success: bool = True
    data: IMCallbackData | None = None


# ── Projections ───────────────────────────────────────────────────────


def im_channel_record(info: IMChannelInfo) -> IMChannelRecord:
    """Project the service DTO onto the wire record."""
    return IMChannelRecord(
        id=info.id,
        tenant_id=info.tenant_id,
        agent_id=info.agent_id,
        platform=info.platform,
        name=info.name,
        enabled=info.enabled,
        mode=info.mode,
        output_mode=info.output_mode,
        knowledge_base_id=info.knowledge_base_id,
        bot_identity=info.bot_identity,
        session_mode=info.session_mode,
        credentials_configured=info.credentials_configured,
        created_at=info.created_at,
        updated_at=info.updated_at,
    )


def to_create_request(body: IMChannelCreateRequest, *, agent_id: str) -> ChannelCreateRequest:
    """Map a create body onto the core service DTO with upstream defaults."""
    return ChannelCreateRequest(
        agent_id=agent_id,
        platform=body.platform,
        name=body.name,
        mode=body.mode,
        output_mode=body.output_mode,
        knowledge_base_id=body.knowledge_base_id,
        credentials=body.credentials,
        enabled=body.enabled if body.enabled is not None else True,
    )


def to_update_request(body: IMChannelUpdateRequest) -> ChannelUpdateRequest:
    """Map an update body onto the core service DTO (``None`` = unchanged)."""
    return ChannelUpdateRequest(
        name=body.name,
        mode=body.mode,
        output_mode=body.output_mode,
        knowledge_base_id=body.knowledge_base_id,
        credentials=body.credentials,
        enabled=body.enabled,
    )


# ── Shared guards ──────────────────────────────────────────────────────


def _require_tenant(tenant_id: int) -> int:
    """Return the active workspace id, or fail closed.

    Channel management is workspace-scoped; without a workspace context
    there is no safe default, so this rejects rather than guessing.
    """
    if tenant_id == 0:
        raise UnauthorizedError(
            code="im.tenant_context_missing",
            message="unauthorized: workspace context missing",
        )
    return tenant_id


# ── Channel CRUD views ─────────────────────────────────────────────────


async def create_im_channel(
    agent_id: str,
    body: IMChannelCreateRequest,
    service: IMChannelServiceDep,
    tenant_id: _PrincipalTenant,
) -> IMChannelEnvelope:
    """Create an IM channel for an agent; admin only."""
    tid = _require_tenant(tenant_id)
    info = await service.create_channel(
        tenant_id=tid,
        request=to_create_request(body, agent_id=agent_id.strip()),
    )
    return IMChannelEnvelope(success=True, data=im_channel_record(info))


async def list_im_channels(
    agent_id: str,
    service: IMChannelServiceDep,
    tenant_id: _PrincipalTenant,
) -> IMChannelListEnvelope:
    """List every live IM channel of one agent; viewer or above."""
    tid = _require_tenant(tenant_id)
    infos = await service.list_channels_by_agent(tenant_id=tid, agent_id=agent_id)
    return IMChannelListEnvelope(
        success=True,
        data=[im_channel_record(info) for info in infos],
    )


async def list_all_im_channels(
    service: IMChannelServiceDep,
    tenant_id: _PrincipalTenant,
) -> IMChannelListEnvelope:
    """List every IM channel of the workspace, across agents.

    Credentials are intentionally not included in the response.
    """
    tid = _require_tenant(tenant_id)
    infos = await service.list_channels(tenant_id=tid)
    return IMChannelListEnvelope(
        success=True,
        data=[im_channel_record(info) for info in infos],
    )


async def update_im_channel(
    channel_id: str,
    body: IMChannelUpdateRequest,
    service: IMChannelServiceDep,
    tenant_id: _PrincipalTenant,
) -> IMChannelEnvelope:
    """Update a channel's mutable fields; admin only.

    ``None`` fields mean "leave unchanged". Re-binding the channel to a
    different agent (upstream ``SetChannelAgentID``) is a deferred seam
    in this layer and is rejected with a clear error.
    """
    tid = _require_tenant(tenant_id)
    new_agent = (body.agent_id or "").strip()
    if new_agent:
        current = await service.get_channel(tenant_id=tid, channel_id=channel_id)
        if new_agent != current.agent_id:
            raise ExternalServiceError(
                code="im.agent_transfer_unavailable",
                message="moving a channel to another agent is not yet wired",
            )
    info = await service.update_channel(
        tenant_id=tid,
        channel_id=channel_id,
        request=to_update_request(body),
    )
    return IMChannelEnvelope(success=True, data=im_channel_record(info))


async def delete_im_channel(
    channel_id: str,
    service: IMChannelServiceDep,
    tenant_id: _PrincipalTenant,
) -> SimpleSuccessResponse:
    """Soft-delete a channel; admin only."""
    tid = _require_tenant(tenant_id)
    await service.delete_channel(tenant_id=tid, channel_id=channel_id)
    return SimpleSuccessResponse()


async def toggle_im_channel(
    channel_id: str,
    service: IMChannelServiceDep,
    tenant_id: _PrincipalTenant,
) -> IMChannelEnvelope:
    """Flip a channel's enabled state, starting / stopping the runtime; admin only."""
    tid = _require_tenant(tenant_id)
    info = await service.toggle_channel_enabled(tenant_id=tid, channel_id=channel_id)
    return IMChannelEnvelope(success=True, data=im_channel_record(info))


# ── Platform callback ──────────────────────────────────────────────────


async def im_callback(
    request: Request,
    channel_id: str,
    service: IMChannelServiceDep,
    registry: IMCommandRegistryDep,
) -> IMCallbackAckResponse:
    """Receive an IM platform callback and dispatch it.

    Resolves the durable channel + running adapter, runs the platform
    URL-verification and signature checks, parses the message, and — for
    a slash-command — executes it and sends the reply through the
    adapter. Non-command messages are acknowledged; the QA pipeline is a
    later seam.
    """
    adapter, channel = await service.ensure_channel_adapter(channel_id)
    callback = await _build_callback_request(request)

    if adapter.handle_url_verification(callback):
        return IMCallbackAckResponse()

    try:
        adapter.verify_callback(callback)
    except UnauthorizedError as exc:
        raise PermissionDeniedError(
            code="im.verify_failed",
            message="verification failed",
        ) from exc

    try:
        message = adapter.parse_callback(callback)
    except Exception as exc:
        logger.warning("[IM] parse callback failed for channel %s: %s", channel_id, exc)
        raise ValidationError(
            code="im.parse_failed",
            message="parse failed",
        ) from exc

    if message is None:
        logger.info(
            "[IM] callback parsed no message platform=%s channel_id=%s",
            channel.platform,
            channel_id,
        )
        return _callback_ack(channel.platform)

    # Command replies are sent synchronously (fast, no LLM work) before the
    # acknowledgement returns; the QA pipeline — which the upstream runs
    # asynchronously after the ack — is a later seam.
    await _handle_message(adapter, channel, registry, message)
    return _callback_ack(channel.platform)


async def _handle_message(
    adapter: IMAdapter,
    channel: IMChannelInfo,
    registry: CommandRegistry,
    message: IncomingMessage,
) -> None:
    """Route one parsed message: slash-command, unknown command, or QA fallback."""
    command, args, ok = registry.parse(message.content)
    if ok and command is not None:
        await _dispatch_command(adapter, channel, command, message, args)
        return
    if registry.looks_like_command(message.content):
        adapter.send_reply(
            _callback_context(),
            message,
            ReplyMessage(content="未知指令，发送 `/help` 查看所有可用指令。", is_final=True),
        )
        return
    logger.info(
        "[IM] message processing is a deferred seam platform=%s channel_id=%s",
        channel.platform,
        channel.id,
    )


async def _dispatch_command(
    adapter: IMAdapter,
    channel: IMChannelInfo,
    command: Command,
    message: IncomingMessage,
    args: list[str],
) -> None:
    """Execute ``command`` and send its reply through ``adapter``.

    Service-level side effects the command requests (reset the
    conversation, stop the in-flight reply) require the session / stream
    infrastructure that lands with the QA pipeline; until then they are
    logged and the reply is still delivered.
    """
    cmd_ctx = CommandContext(
        incoming=message,
        tenant_id=channel.tenant_id,
        channel_output_mode=channel.output_mode,
    )
    try:
        result = await command.execute(cmd_ctx, args)
    except Exception:
        logger.exception("[IM] command /%s failed", command.name())
        adapter.send_reply(
            _callback_context(),
            message,
            ReplyMessage(content="抱歉，执行指令时出现了异常，请稍后再试。", is_final=True),
        )
        return
    if result.action is not CommandAction.NONE:
        logger.info(
            "[IM] command /%s requested action %s (session infra is a deferred seam)",
            command.name(),
            result.action.name,
        )
    adapter.send_reply(
        _callback_context(),
        message,
        ReplyMessage(content=result.content, is_final=True),
    )


def _callback_ack(platform: str) -> IMCallbackAckResponse:
    """Return the platform-specific acknowledgement envelope.

    Yunzhijia requires a typed payload on its webhook acknowledgement;
    every other platform acknowledges with ``{"success": true}``.
    """
    if platform == "yunzhijia":
        return IMCallbackAckResponse(
            success=True,
            data=IMCallbackData(type=2, content=""),
        )
    return IMCallbackAckResponse()


def _callback_context() -> Context:
    """Return a fresh cancellation probe for one callback reply.

    A fresh ``EventContext`` per callback keeps the probe free of
    cross-request state; nothing cancels it during the synchronous reply
    send.
    """
    return EventContext()


async def _build_callback_request(request: Request) -> CallbackRequest:
    """Capture the raw request parts the adapters interpret."""
    body_bytes = await request.body()
    return CallbackRequest(
        headers={key: value for key, value in request.headers.items()},
        body=body_bytes.decode("utf-8", errors="replace"),
        query={key: value for key, value in request.query_params.items()},
    )


__all__ = [
    "IMCallbackAckResponse",
    "IMCallbackData",
    "IMChannelCreateRequest",
    "IMChannelEnvelope",
    "IMChannelListEnvelope",
    "IMChannelRecord",
    "IMChannelUpdateRequest",
    "SimpleSuccessResponse",
    "create_im_channel",
    "delete_im_channel",
    "im_callback",
    "im_channel_record",
    "list_all_im_channels",
    "list_im_channels",
    "to_create_request",
    "to_update_request",
    "toggle_im_channel",
    "update_im_channel",
]
