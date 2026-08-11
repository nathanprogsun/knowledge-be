"""WeChat (iLink bot) adapter — long-poll inbound, REST send.

Messages arrive over the iLink long-polling API rather than webhooks,
so ``verify_callback`` / ``parse_callback`` reject webhook traffic and
``handle_url_verification`` never applies. ``send_reply`` posts a text
item through the iLink ``/ilink/bot/sendmessage`` endpoint with the
iLink authentication headers.
"""

from __future__ import annotations

import base64
import json
import secrets
import time
from collections.abc import Callable

import httpx

from src.common.exception import ValidationError
from src.core.channels.im.adapter_base import (
    CallbackRequest,
    Context,
    IMAdapter,
    IncomingMessage,
    ReplyMessage,
)
from src.core.channels.im.adapters._common import (
    assert_http_ok_strict,
    build_http_client,
    string_credential,
)
from src.db.models.im_channel import IMChannel

# iLink API root and per-request header names.
_ILINK_BASE_URL = "https://ilinkai.weixin.qq.com"
_AUTH_HEADER_TYPE = "ilink_bot_token"
# Opaque client version sent in ``base_info`` with every request.
_CHANNEL_VERSION = "1.0.0"


class WeChatAdapter(IMAdapter):
    """WeChat iLink bot adapter (long-poll inbound + REST send)."""

    def __init__(
        self,
        *,
        bot_token: str = "",
        ilink_bot_id: str = "",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._bot_token = bot_token
        self._ilink_bot_id = ilink_bot_id
        self._http_client = build_http_client(transport=transport)
        self._connected = False

    # ── Identity ─────────────────────────────────────────────────────

    def platform(self) -> str:
        return "wechat"

    # ── Callback verification ────────────────────────────────────────

    def verify_callback(self, request: CallbackRequest) -> None:
        raise ValidationError(
            code="im.webhook_not_supported",
            message="wechat does not support webhook callbacks",
        )

    # ── Callback parsing ─────────────────────────────────────────────

    def parse_callback(self, request: CallbackRequest) -> IncomingMessage | None:
        raise ValidationError(
            code="im.webhook_not_supported",
            message="wechat does not support webhook callbacks",
        )

    # ── URL verification ─────────────────────────────────────────────

    def handle_url_verification(self, request: CallbackRequest) -> bool:
        return False

    # ── Send reply ───────────────────────────────────────────────────

    def send_reply(self, ctx: Context, incoming: IncomingMessage, reply: ReplyMessage) -> None:
        context_token = str(incoming.extra.get("context_token", ""))
        payload = {
            "msg": {
                "from_user_id": "",
                "to_user_id": incoming.user_id,
                "client_id": _generate_client_id(),
                "message_type": 2,  # BOT
                "message_state": 2,  # FINISH
                "item_list": [
                    {
                        "type": 1,  # TEXT
                        "text_item": {"text": reply.content},
                    }
                ],
                "context_token": context_token,
            },
            "base_info": {"channel_version": _CHANNEL_VERSION},
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "AuthorizationType": _AUTH_HEADER_TYPE,
            "Authorization": f"Bearer {self._bot_token}",
            "X-WECHAT-UIN": _generate_wechat_uin(),
        }
        resp = self._http_client.post(
            f"{_ILINK_BASE_URL}/ilink/bot/sendmessage",
            content=body,
            headers=headers,
        )
        assert_http_ok_strict(resp, platform="wechat", action="sendmessage")

    # ── Connection lifecycle ─────────────────────────────────────────

    async def connect(self, ctx: Context) -> Callable[[], None]:
        self._connected = True

        def _stop() -> None:
            self.disconnect()

        return _stop

    def disconnect(self) -> None:
        self._connected = False
        self._http_client.close()

    def is_connected(self) -> bool:
        return self._connected


# ── Wire helpers ──────────────────────────────────────────────────────


def _generate_wechat_uin() -> str:
    """Random ``X-WECHAT-UIN`` header value: uint32 → decimal → base64."""
    value = str(secrets.randbits(32))
    return base64.b64encode(value.encode("ascii")).decode("ascii")


def _generate_client_id() -> str:
    """Opaque per-message client id used by the iLink protocol."""
    return f"im_{time.time_ns()}"


__all__ = ["WeChatAdapter", "build_wechat_adapter"]


def build_wechat_adapter(channel: IMChannel) -> WeChatAdapter:
    """Construct a WeChat adapter from a channel row's credentials.

    ``ilink_bot_id`` is accepted for symmetry with the bot-identity
    derivation but is not required by the send path.
    """
    return WeChatAdapter(
        bot_token=string_credential(channel.credentials, "bot_token"),
        ilink_bot_id=string_credential(channel.credentials, "ilink_bot_id"),
    )
