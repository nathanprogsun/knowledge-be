"""DingTalk adapter — webhook signature verification, callback parsing, replies.

Mirrors the upstream contract: ``verify_callback`` checks the
``Timestamp`` / ``Sign`` headers using HmacSHA256 over
``timestamp + "\\n" + client_secret``, ``parse_callback`` maps the robot
callback payload (text / file / picture) to the unified message, and
``send_reply`` prefers the per-conversation ``sessionWebhook`` when the
callback carried one, falling back to the OpenAPI group / batch-send
endpoints with a cached app access token.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import httpx

from src.common.exception import UnauthorizedError
from src.common.json import JsonObject
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
    assert_http_ok_strict,
    build_http_client,
    constant_time_equals,
    header_value,
    hmac_sha256_base64,
    payload_dict,
    payload_int,
    payload_string,
    send_error,
    string_credential,
    timestamp_ms_is_fresh,
)
from src.db.models.im_channel import IMChannel

_DINGTALK_API_BASE_URL = "https://api.dingtalk.com"
# The callback marks group conversations with this ``conversationType``.
_DINGTALK_CONVERSATION_TYPE_GROUP = "2"
# Signature timestamps older than one hour are rejected.
_DINGTALK_TIMESTAMP_TOLERANCE_SECONDS = 3600
# Safety margin subtracted from the access-token expiry.
_TOKEN_EXPIRY_MARGIN_SECONDS = 300


class DingTalkAdapter(IMAdapter):
    """DingTalk platform adapter (webhook callbacks + OpenAPI replies)."""

    def __init__(
        self,
        *,
        client_id: str = "",
        client_secret: str = "",
        card_template_id: str = "",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._card_template_id = card_template_id
        self._http_client = build_http_client(transport=transport)
        self._connected = False
        self._access_token = ""
        self._token_expires_at = 0.0

    # ── Identity ─────────────────────────────────────────────────────

    def platform(self) -> str:
        return "dingtalk"

    # ── Callback verification ────────────────────────────────────────

    def verify_callback(self, request: CallbackRequest) -> None:
        if not self._client_secret:
            return
        timestamp = header_value(request.headers, "Timestamp")
        sign = header_value(request.headers, "Sign")
        if (
            not timestamp
            or not sign
            or not timestamp_ms_is_fresh(
                request.headers, "Timestamp", _DINGTALK_TIMESTAMP_TOLERANCE_SECONDS
            )
        ):
            raise UnauthorizedError(
                code="im.verify_failed",
                message="dingtalk signature verification failed: missing or stale timestamp",
            )
        expected = hmac_sha256_base64(self._client_secret, f"{timestamp}\n{self._client_secret}")
        if not constant_time_equals(sign, expected):
            raise UnauthorizedError(
                code="im.verify_failed",
                message="dingtalk signature verification failed: signature mismatch",
            )

    # ── Callback parsing ─────────────────────────────────────────────

    def parse_callback(self, request: CallbackRequest) -> IncomingMessage | None:
        try:
            payload = json.loads(request.body)
        except json.JSONDecodeError:
            raise UnauthorizedError(
                code="im.parse_failed",
                message="dingtalk callback body is not valid JSON",
            ) from None
        if not isinstance(payload, dict):
            return None
        return _parse_callback_message(payload)

    # ── URL verification ─────────────────────────────────────────────

    def handle_url_verification(self, request: CallbackRequest) -> bool:
        return False

    # ── Send reply ───────────────────────────────────────────────────

    def send_reply(self, ctx: Context, incoming: IncomingMessage, reply: ReplyMessage) -> None:
        session_webhook = str(incoming.extra.get("session_webhook", ""))
        if session_webhook:
            self._reply_via_session_webhook(session_webhook, reply.content)
            return
        self._reply_via_openapi(incoming, reply.content)

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

    # ── Reply internals ──────────────────────────────────────────────

    def _reply_via_session_webhook(self, webhook_url: str, content: str) -> None:
        body = {
            "msgtype": "markdown",
            "markdown": {"title": "Reply", "text": content},
        }
        resp = self._http_client.post(
            webhook_url,
            json=body,
            headers={"Content-Type": "application/json"},
        )
        assert_http_ok_strict(resp, platform="dingtalk", action="sessionWebhook reply")

    def _reply_via_openapi(self, incoming: IncomingMessage, content: str) -> None:
        token = self._get_access_token()
        msg_param = json.dumps({"title": "Reply", "text": content})
        headers = {
            "Content-Type": "application/json",
            "x-acs-dingtalk-access-token": token,
        }
        if incoming.chat_type == CHAT_TYPE_GROUP:
            url = f"{_DINGTALK_API_BASE_URL}/v1.0/robot/groupMessages/send"
            body: JsonObject = {
                "robotCode": self._client_id,
                "msgKey": "sampleMarkdown",
                "msgParam": msg_param,
                "openConversationId": incoming.chat_id,
            }
        else:
            url = f"{_DINGTALK_API_BASE_URL}/v1.0/robot/oToMessages/batchSend"
            body = {
                "robotCode": self._client_id,
                "msgKey": "sampleMarkdown",
                "msgParam": msg_param,
                "userIds": [incoming.user_id],
            }
        resp = self._http_client.post(url, json=body, headers=headers)
        assert_http_ok_strict(resp, platform="dingtalk", action="OpenAPI reply")

    def _get_access_token(self) -> str:
        now = time.time()
        if self._access_token and now < self._token_expires_at:
            return self._access_token
        resp = self._http_client.post(
            f"{_DINGTALK_API_BASE_URL}/v1.0/oauth2/accessToken",
            json={"appKey": self._client_id, "appSecret": self._client_secret},
            headers={"Content-Type": "application/json"},
        )
        assert_http_ok_strict(resp, platform="dingtalk", action="get access token")
        try:
            result = resp.json()
        except json.JSONDecodeError:
            send_error("dingtalk", "get access token", "non-JSON response")
        if not isinstance(result, dict) or not result.get("accessToken"):
            send_error("dingtalk", "get access token", f"empty token: {str(result)[:200]}")
        token = str(result["accessToken"])
        expire_in = payload_int(result, "expireIn")
        if expire_in <= 0:
            expire_in = 7200
        self._access_token = token
        self._token_expires_at = now + expire_in - _TOKEN_EXPIRY_MARGIN_SECONDS
        return token


# ── Parse helpers ─────────────────────────────────────────────────────


def _parse_callback_message(payload: JsonObject) -> IncomingMessage | None:
    if not payload:
        return None

    chat_type = (
        CHAT_TYPE_GROUP
        if payload_string(payload, "conversationType") == _DINGTALK_CONVERSATION_TYPE_GROUP
        else CHAT_TYPE_DIRECT
    )
    chat_id = payload_string(payload, "conversationId") if chat_type == CHAT_TYPE_GROUP else ""

    user_id = payload_string(payload, "senderStaffId") or payload_string(payload, "senderId")

    extra: JsonObject = {"session_webhook": payload_string(payload, "sessionWebhook")}
    incoming = IncomingMessage(
        platform="dingtalk",
        message_type="text",
        user_id=user_id,
        user_name=payload_string(payload, "senderNick"),
        chat_id=chat_id,
        chat_type=chat_type,
        message_id=payload_string(payload, "msgId"),
        extra=extra,
    )

    msg_type, file_name, download_code = _parse_file_content(payload)
    if download_code:
        msg_id = payload_string(payload, "msgId")
        if msg_type == "image":
            file_name = f"{msg_id}.png"
        elif not file_name:
            file_name = msg_id
        return incoming.model_copy(
            update={
                "message_type": msg_type,
                "file_name": file_name,
                "file_key": download_code,
                "extra": {**extra, "robot_code": payload_string(payload, "robotCode")},
            }
        )

    text = payload_dict(payload, "text")
    content = payload_string(text, "content").strip()
    return incoming.model_copy(update={"content": content})


def _parse_file_content(payload: JsonObject) -> tuple[str, str, str]:
    """Map DingTalk ``msgtype`` + ``content`` to ``(type, file_name, download_code)``.

    Returns an empty ``download_code`` for non-file / non-picture messages.
    """
    msg_type_name = payload_string(payload, "msgtype")
    msg_type = ""
    if msg_type_name == "file":
        msg_type = "file"
    elif msg_type_name == "picture":
        msg_type = "image"
    else:
        return "", "", ""

    content = payload_dict(payload, "content")
    download_code = payload_string(content, "downloadCode") or payload_string(
        content, "pictureDownloadCode"
    )
    if not download_code:
        return "", "", ""
    file_name = payload_string(content, "fileName")
    if msg_type == "image":
        file_name = ""
    return msg_type, file_name, download_code


__all__ = ["DingTalkAdapter", "build_dingtalk_adapter"]


def build_dingtalk_adapter(channel: IMChannel) -> DingTalkAdapter:
    """Construct a DingTalk adapter from a channel row's credentials."""
    return DingTalkAdapter(
        client_id=string_credential(channel.credentials, "client_id"),
        client_secret=string_credential(channel.credentials, "client_secret"),
        card_template_id=string_credential(channel.credentials, "card_template_id"),
    )
