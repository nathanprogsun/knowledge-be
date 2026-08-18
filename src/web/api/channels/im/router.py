"""IM-channel HTTP endpoints — channel CRUD, toggle, and the platform callback.

Registered by the app factory. Four routers mirror the upstream route
split:

- ``callback_router`` — ``/im/callback/{channel_id}`` (GET + POST), the
  platform webhook surface. It is registered without the user-auth
  dependency because IM platforms authenticate with their own signature
  verification.
- ``agents_router`` — ``/agents/{agent_id}/im-channels`` (create /
  list) — admin for mutations, viewer for reads.
- ``router`` — ``/im-channels`` (list-all / update / delete / toggle) —
  viewer for reads, admin for mutations.
- ``wechat_router`` — ``/wechat/qrcode`` (+ ``/status``) — the WeChat
  scan-to-bind login flow. The upstream flow depends on the WeChat
  platform service; this build reports the feature as unconfigured
  rather than fabricating a QR payload.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.common.exception import ExternalServiceError
from src.web.api.channels.im.views import (
    IMCallbackAckResponse,
    IMChannelEnvelope,
    IMChannelListEnvelope,
    SimpleSuccessResponse,
    create_im_channel,
    delete_im_channel,
    im_callback,
    list_all_im_channels,
    list_im_channels,
    toggle_im_channel,
    update_im_channel,
)
from src.web.deps.rbac import make_role_dep
from src.web.middleware.auth import require_auth

# Route-level auth + role gates. Mutations are Admin+, reads are Viewer+,
# mirroring the upstream RBAC guards.
_AUTH_ADMIN = [Depends(require_auth), Depends(make_role_dep("admin"))]
_AUTH_VIEWER = [Depends(require_auth), Depends(make_role_dep("viewer"))]

agents_router = APIRouter(prefix="/agents", tags=["im-channels"])
router = APIRouter(prefix="/im-channels", tags=["im-channels"])
wechat_router = APIRouter(prefix="/wechat", tags=["im-wechat"])
callback_router = APIRouter(prefix="/im", tags=["im-callback"])

# ── Agent-scoped channel CRUD ──────────────────────────────────────────

agents_router.add_api_route(
    "/{agent_id}/im-channels",
    create_im_channel,
    methods=["POST"],
    response_model=IMChannelEnvelope,
    dependencies=_AUTH_ADMIN,
)
agents_router.add_api_route(
    "/{agent_id}/im-channels",
    list_im_channels,
    methods=["GET"],
    response_model=IMChannelListEnvelope,
    dependencies=_AUTH_VIEWER,
)

# ── Tenant-wide channel operations ─────────────────────────────────────

router.add_api_route(
    "",
    list_all_im_channels,
    methods=["GET"],
    response_model=IMChannelListEnvelope,
    dependencies=_AUTH_VIEWER,
)
router.add_api_route(
    "/{channel_id}",
    update_im_channel,
    methods=["PUT"],
    response_model=IMChannelEnvelope,
    dependencies=_AUTH_ADMIN,
)
router.add_api_route(
    "/{channel_id}",
    delete_im_channel,
    methods=["DELETE"],
    response_model=SimpleSuccessResponse,
    dependencies=_AUTH_ADMIN,
)
router.add_api_route(
    "/{channel_id}/toggle",
    toggle_im_channel,
    methods=["POST"],
    response_model=IMChannelEnvelope,
    dependencies=_AUTH_ADMIN,
)

# ── Platform callback (no user auth — platform signature verification) ─

callback_router.add_api_route(
    "/callback/{channel_id}",
    im_callback,
    methods=["GET"],
    operation_id="im_callback_get",
    response_model=IMCallbackAckResponse,
    response_model_exclude_none=True,
)
callback_router.add_api_route(
    "/callback/{channel_id}",
    im_callback,
    methods=["POST"],
    operation_id="im_callback_post",
    response_model=IMCallbackAckResponse,
    response_model_exclude_none=True,
)


# ── WeChat scan-to-bind login ─────────────────────────────────────────


@wechat_router.post("/qrcode", dependencies=_AUTH_VIEWER)
async def wechat_get_qrcode() -> None:
    """Request a WeChat login QR code.

    The upstream flow talks to the WeChat platform service; this build
    has no such integration, so the feature reports itself as
    unconfigured instead of returning a fabricated payload.
    """
    raise ExternalServiceError(
        code="wechat.qrcode_unavailable",
        message="WeChat QR login is not configured in this deployment",
    )


@wechat_router.post("/qrcode/status", dependencies=_AUTH_VIEWER)
async def wechat_poll_qrcode_status() -> None:
    """Poll a WeChat QR code's scan status (unconfigured in this build)."""
    raise ExternalServiceError(
        code="wechat.qrcode_unavailable",
        message="WeChat QR login is not configured in this deployment",
    )


__all__ = [
    "agents_router",
    "callback_router",
    "router",
    "wechat_router",
]
