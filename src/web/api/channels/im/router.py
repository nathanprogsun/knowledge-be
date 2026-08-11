"""IM-channel HTTP endpoints — channel CRUD, toggle, and the platform callback.

Registered by the app factory. Three routers mirror the upstream route
split:

- ``callback_router`` — ``/im/callback/{channel_id}`` (GET + POST), the
  platform webhook surface. It is registered without the user-auth
  dependency because IM platforms authenticate with their own signature
  verification.
- ``agents_router`` — ``/agents/{agent_id}/im-channels`` (create /
  list) — admin for mutations, viewer for reads.
- ``router`` — ``/im-channels`` (list-all / update / delete / toggle) —
  viewer for reads, admin for mutations.

The WeChat QR-code login flow (``/wechat/qrcode``) is a separate
surface wired by a later PR.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

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

__all__ = [
    "agents_router",
    "callback_router",
    "router",
]
