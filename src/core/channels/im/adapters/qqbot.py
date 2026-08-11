"""QQBot adapter — gateway-payload callbacks and REST replies.

Mirrors the upstream contract: the bot receives messages over the QQ
gateway (WebSocket), so ``verify_callback`` passes without checks,
``handle_url_verification`` never applies, and ``parse_callback`` maps a
gateway dispatch frame (``op``/``t``/``d``) into the unified message for
C2C and group-at events. ``send_reply`` posts text through the QQ OpenAPI
with a cached app access token.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import httpx

from src.common.exception import UnauthorizedError, ValidationError
from src.common.json import JsonObject, JsonValue
from src.core.channels.im.adapter_base import (
    CHAT_TYPE_DIRECT,
    CHAT_TYPE_GROUP,
    CallbackRequest,
    Context,
    IMAdapter,
    IncomingMessage,
    ReplyMessage,
)
from src.core.channels.im.adapters._common import (
    assert_http_ok,
    build_http_client,
    payload_dict,
    payload_int,
    payload_string,
    send_error,
    string_credential,
    validate_http_endpoint,
)
from src.db.models.im_channel import IMChannel

_QQBOT_API_BASE_URL = "https://api.sgroup.qq.com"
_QQBOT_APP_TOKEN_URL = "https://bots.qq.com/app/getAppAccessToken"
# Gateway op-code that carries an event dispatch.
_OP_DISPATCH = 0
# Event types the adapter turns into messages.
_EVENT_C2C_MESSAGE_CREATE = "C2C_MESSAGE_CREATE"
_EVENT_GROUP_AT_MESSAGE_CREATE = "GROUP_AT_MESSAGE_CREATE"
# Extra keys on the unified message.
_EXTRA_MESSAGE_ID = "message_id"
_EXTRA_CHAT_KIND = "chat_kind"
# Default access-token lifetime when the response omits ``expires_in``.
_DEFAULT_TOKEN_EXPIRY = 7200
# Buffer kept on the token cache so the token never expires mid-request.
_TOKEN_REFRESH_BUFFER_SECONDS = 60


class QQBotAdapter(IMAdapter):
    """QQBot platform adapter (gateway callbacks + OpenAPI replies)."""

    def __init__(
        self,
        *,
        app_id: str = "",
        client_secret: str = "",
        api_base_url: str = _QQBOT_API_BASE_URL,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._app_id = app_id
        self._client_secret = client_secret
        self._api_base_url = api_base_url.rstrip("/")
        self._http_client = build_http_client(transport=transport)
        self._connected = False
        self._access_token = ""
        self._token_expires_at = 0.0

    # ── Identity ─────────────────────────────────────────────────────

    def platform(self) -> str:
        return "qqbot"

    # ── Callback verification ────────────────────────────────────────

    def verify_callback(self, request: CallbackRequest) -> None:
        return None

    # ── Callback parsing ─────────────────────────────────────────────

    def parse_callback(self, request: CallbackRequest) -> IncomingMessage | None:
        try:
            payload = json.loads(request.body)
        except json.JSONDecodeError:
            raise UnauthorizedError(
                code="im.parse_failed",
                message="qqbot gateway payload is not valid JSON",
            ) from None
        if not isinstance(payload, dict) or payload_int(payload, "op") != _OP_DISPATCH:
            return None
        event = _parse_dispatch_data(payload_string(payload, "d"))
        if not event:
            return None
        event_type = payload_string(payload, "t")
        if event_type == _EVENT_C2C_MESSAGE_CREATE:
            return _parse_c2c_message(event)
        if event_type == _EVENT_GROUP_AT_MESSAGE_CREATE:
            return _parse_group_message(event)
        return None

    # ── URL verification ─────────────────────────────────────────────

    def handle_url_verification(self, request: CallbackRequest) -> bool:
        return False

    # ── Send reply ───────────────────────────────────────────────────

    def send_reply(self, ctx: Context, incoming: IncomingMessage, reply: ReplyMessage) -> None:
        content = reply.content.strip()
        if not content:
            return
        msg_id = str(incoming.extra.get(_EXTRA_MESSAGE_ID, ""))
        token = self._get_access_token()
        if incoming.chat_type == CHAT_TYPE_GROUP:
            url = f"{self._api_base_url}/v2/groups/{incoming.chat_id}/messages"
        else:
            url = f"{self._api_base_url}/v2/users/{incoming.user_id}/messages"
        body: JsonObject = {"content": content, "msg_type": 0, "msg_seq": 1}
        if msg_id:
            body["msg_id"] = msg_id
        resp = self._http_client.post(
            url,
            json=body,
            headers={
                "Authorization": f"QQBot {token}",
                "Content-Type": "application/json",
            },
        )
        assert_http_ok(resp, platform="qqbot", action="send message")

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

    # ── Access token ─────────────────────────────────────────────────

    def _get_access_token(self) -> str:
        now = time.time()
        if self._access_token and now < self._token_expires_at:
            return self._access_token
        resp = self._http_client.post(
            _QQBOT_APP_TOKEN_URL,
            json={"appId": self._app_id, "clientSecret": self._client_secret},
            headers={"Content-Type": "application/json"},
        )
        assert_http_ok(resp, platform="qqbot", action="get app access token")
        try:
            result = resp.json()
        except json.JSONDecodeError:
            send_error("qqbot", "get app access token", "non-JSON response")
        if not isinstance(result, dict) or not result.get("access_token"):
            code = result.get("code", "") if isinstance(result, dict) else ""
            message = result.get("message", "") if isinstance(result, dict) else ""
            send_error("qqbot", "get app access token", f"code={code} message={message}")
        token = str(result["access_token"])
        expires_in = _parse_expires_in(result.get("expires_in"))
        self._access_token = token
        self._token_expires_at = now + expires_in - _TOKEN_REFRESH_BUFFER_SECONDS
        return token


# ── Parse helpers ─────────────────────────────────────────────────────


def _parse_dispatch_data(raw: str) -> JsonObject | None:
    if not raw:
        return None
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) else None


def _parse_c2c_message(event: JsonObject) -> IncomingMessage:
    author = payload_dict(event, "author")
    user_id = _first_non_empty(payload_string(author, "user_openid"), payload_string(author, "id"))
    return IncomingMessage(
        platform="qqbot",
        message_type="text",
        user_id=user_id,
        user_name=payload_string(author, "username"),
        chat_id="",
        chat_type=CHAT_TYPE_DIRECT,
        content=payload_string(event, "content").strip(),
        message_id=payload_string(event, "id"),
        extra={_EXTRA_MESSAGE_ID: payload_string(event, "id"), _EXTRA_CHAT_KIND: "c2c"},
    )


def _parse_group_message(event: JsonObject) -> IncomingMessage:
    author = payload_dict(event, "author")
    user_id = _first_non_empty(
        payload_string(author, "member_openid"),
        payload_string(author, "user_openid"),
        payload_string(author, "id"),
    )
    return IncomingMessage(
        platform="qqbot",
        message_type="text",
        user_id=user_id,
        user_name=payload_string(author, "username"),
        chat_id=payload_string(event, "group_openid"),
        chat_type=CHAT_TYPE_GROUP,
        content=payload_string(event, "content").strip(),
        message_id=payload_string(event, "id"),
        extra={_EXTRA_MESSAGE_ID: payload_string(event, "id"), _EXTRA_CHAT_KIND: "group"},
    )


def _first_non_empty(*values: str) -> str:
    for value in values:
        if value.strip():
            return value
    return ""


def _parse_expires_in(raw: JsonValue | None) -> int:
    if isinstance(raw, bool):
        return _DEFAULT_TOKEN_EXPIRY
    if isinstance(raw, int):
        return raw if raw > 0 else _DEFAULT_TOKEN_EXPIRY
    if isinstance(raw, float):
        return int(raw) if raw > 0 else _DEFAULT_TOKEN_EXPIRY
    if isinstance(raw, str):
        try:
            parsed = int(raw)
        except ValueError:
            return _DEFAULT_TOKEN_EXPIRY
        return parsed if parsed > 0 else _DEFAULT_TOKEN_EXPIRY
    return _DEFAULT_TOKEN_EXPIRY


__all__ = ["QQBotAdapter", "build_qqbot_adapter"]


def build_qqbot_adapter(channel: IMChannel) -> QQBotAdapter:
    """Construct a QQBot adapter, validating required credentials."""
    app_id = string_credential(channel.credentials, "app_id")
    client_secret = string_credential(channel.credentials, "client_secret")
    if not app_id:
        raise ValidationError(code="im.credentials_invalid", message="qqbot app_id is required")
    if not client_secret:
        raise ValidationError(
            code="im.credentials_invalid",
            message="qqbot client_secret is required",
        )
    api_base_url = string_credential(channel.credentials, "api_base_url")
    if api_base_url:
        validate_http_endpoint(api_base_url)
    else:
        api_base_url = _QQBOT_API_BASE_URL
    return QQBotAdapter(
        app_id=app_id,
        client_secret=client_secret,
        api_base_url=api_base_url,
    )
