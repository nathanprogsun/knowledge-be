"""Yunzhijia adapter — webhook callbacks and send-message webhook replies.

Mirrors the upstream contract: ``verify_callback`` checks the ``sign``
header (HmacSHA1 over the comma-joined callback fields), ``parse_callback``
keeps only text messages explicitly addressed to the robot (``@mention``
prefix or ``msgParam`` mentions) and maps embedded images to image
messages, and ``send_reply`` POSTs the Markdown payload to the configured
``send_msg_url``.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx

from src.common.exception import UnauthorizedError, ValidationError
from src.common.json import JsonObject
from src.core.channels.im.adapter_base import (
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
    constant_time_equals,
    header_value,
    hmac_sha1_base64,
    int_credential,
    payload_int,
    payload_list,
    payload_string,
    send_error,
    string_credential,
    validate_https_host_suffix,
)
from src.db.models.im_channel import IMChannel

# Yunzhijia text-message type value carried by the callback ``type`` field.
_TEXT_MESSAGE_TYPE = 2
# Reply rendering: request Markdown unless overridden via reply extra.
_MARKDOWN_FORMAT_TYPE = "markdown"
# Separator characters allowed after the ``@robotName`` mention prefix.
# The fullwidth punctuation is part of the upstream mention format.
_AT_SEPARATORS = ":：,，"


class YunzhijiaAdapter(IMAdapter):
    """Yunzhijia platform adapter (webhook callbacks + send webhook)."""

    def __init__(
        self,
        *,
        send_msg_url: str = "",
        secret: str = "",
        app_id: str = "",
        app_secret: str = "",
        allowed_webhook_host_suffix: str = "",
        timeout_seconds: int = 10,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._send_msg_url = send_msg_url.strip()
        self._secret = secret.strip()
        self._app_id = app_id.strip()
        self._app_secret = app_secret.strip()
        self._allowed_webhook_host_suffix = allowed_webhook_host_suffix.strip()
        timeout = timeout_seconds if timeout_seconds > 0 else 10
        self._http_client = build_http_client(timeout=float(timeout), transport=transport)
        self._connected = False

    # ── Identity ─────────────────────────────────────────────────────

    def platform(self) -> str:
        return "yunzhijia"

    # ── Callback verification ────────────────────────────────────────

    def verify_callback(self, request: CallbackRequest) -> None:
        if not self._secret:
            return
        payload = _parse_body(request.body)
        if not payload:
            raise UnauthorizedError(
                code="im.verify_failed",
                message="yunzhijia callback body is not valid JSON",
            )
        sign = (
            header_value(request.headers, "sign")
            or header_value(request.headers, "Sign")
            or header_value(request.headers, "SIGN")
        )
        if not sign:
            raise UnauthorizedError(
                code="im.verify_failed",
                message="yunzhijia signature verification failed: missing sign header",
            )
        expected = hmac_sha1_base64(self._secret, _signature_string(payload))
        if not constant_time_equals(sign, expected):
            raise UnauthorizedError(
                code="im.verify_failed",
                message="yunzhijia signature verification failed: signature mismatch",
            )

    # ── Callback parsing ─────────────────────────────────────────────

    def parse_callback(self, request: CallbackRequest) -> IncomingMessage | None:
        payload = _parse_body(request.body)
        if not payload:
            return None
        return _to_incoming_message(payload)

    # ── URL verification ─────────────────────────────────────────────

    def handle_url_verification(self, request: CallbackRequest) -> bool:
        return False

    # ── Send reply ───────────────────────────────────────────────────

    def send_reply(self, ctx: Context, incoming: IncomingMessage, reply: ReplyMessage) -> None:
        if not self._send_msg_url:
            send_error("yunzhijia", "send message", "send_msg_url is not configured")
        self._validate_send_url()

        payload: JsonObject = {
            "msgtype": _TEXT_MESSAGE_TYPE,
            "content": reply.content,
        }
        param: dict[str, str] | None = {"formatType": _MARKDOWN_FORMAT_TYPE}
        format_type = str(reply.extra.get("yunzhijia_format_type", ""))
        if "yunzhijia_format_type" in reply.extra:
            param = {"formatType": format_type} if format_type else None
        if param is not None:
            payload["param"] = param

        group_type = str(incoming.extra.get("group_type", ""))
        if group_type != "3" and incoming.user_id:
            payload["notifyParams"] = [{"type": "openIds", "values": [incoming.user_id]}]

        resp = self._http_client.post(
            self._send_msg_url,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        assert_http_ok(resp, platform="yunzhijia", action="send message")

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

    # ── Send-URL validation ──────────────────────────────────────────

    def _validate_send_url(self) -> None:
        validate_https_host_suffix(self._send_msg_url, self._allowed_webhook_host_suffix)


# ── Parse helpers ─────────────────────────────────────────────────────


def _parse_body(raw: str) -> JsonObject:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _signature_string(payload: JsonObject) -> str:
    parts = [
        payload_string(payload, "robotId"),
        payload_string(payload, "robotName"),
        payload_string(payload, "operatorOpenid"),
        payload_string(payload, "operatorName"),
        str(payload_int(payload, "time")),
        payload_string(payload, "msgId"),
        payload_string(payload, "content"),
    ]
    return ",".join(parts)


def _to_incoming_message(payload: JsonObject) -> IncomingMessage | None:
    if payload_int(payload, "type") != _TEXT_MESSAGE_TYPE:
        return None

    param = _parse_message_param(payload_string(payload, "msgParam"))
    image = _first_image(param)

    content = payload_string(payload, "content").strip()
    if not content and image is None:
        return None

    content, mentioned = _clean_at_mention(content, payload_string(payload, "robotName"))
    if not mentioned:
        mentioned = _param_mentions_robot(param, payload_string(payload, "robotId"))
    if not mentioned:
        return None

    if not content and image is None:
        return None

    user_id = _first_non_empty(
        payload_string(payload, "operatorOpenid"),
        payload_string(payload, "operatorOid"),
        payload_string(payload, "openId"),
        payload_string(payload, "senderId"),
        payload_string(payload, "operatorId"),
        payload_string(payload, "operatorUserId"),
    )
    user_name = _first_non_empty(
        payload_string(payload, "operatorName"), payload_string(payload, "senderName")
    )
    chat_id = _first_non_empty(
        payload_string(payload, "groupId"), payload_string(payload, "robotId")
    )

    extra: JsonObject = {
        "robot_id": payload_string(payload, "robotId"),
        "robot_name": payload_string(payload, "robotName"),
        "group_id": payload_string(payload, "groupId"),
        "group_type": str(payload_int(payload, "groupType")),
        "operator_name": user_name,
        "time": str(payload_int(payload, "time")),
    }
    incoming = IncomingMessage(
        platform="yunzhijia",
        message_type="text",
        user_id=user_id,
        user_name=user_name,
        chat_id=chat_id,
        chat_type=CHAT_TYPE_GROUP,
        content=content,
        message_id=payload_string(payload, "msgId"),
        extra=extra,
    )
    if image is not None:
        incoming = incoming.model_copy(
            update={
                "message_type": "image",
                "file_key": image["data"],
                "file_name": _default_image_file_name(payload_string(payload, "msgId")),
                "extra": {
                    **extra,
                    "yunzhijia_image_width": str(image["width"]),
                    "yunzhijia_image_height": str(image["height"]),
                },
            }
        )
    return incoming


def _parse_message_param(raw: str) -> JsonObject:
    if not raw:
        return {}
    try:
        param = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return param if isinstance(param, dict) else {}


def _first_image(param: JsonObject) -> dict[str, str] | None:
    for item in payload_list(param, "desc"):
        if not isinstance(item, dict):
            continue
        if payload_string(item, "type") == "image" and payload_string(item, "data"):
            return {
                "data": payload_string(item, "data"),
                "width": payload_string(item, "w"),
                "height": payload_string(item, "h"),
            }
    return None


def _param_mentions_robot(param: JsonObject, robot_id: str) -> bool:
    if not robot_id:
        return False
    if robot_id in payload_list(param, "notifyTo"):
        return True
    for item in payload_list(param, "desc"):
        if not isinstance(item, dict):
            continue
        if payload_string(item, "type") == "at" and payload_string(item, "data") == robot_id:
            return True
    return False


def _clean_at_mention(content: str, robot_name: str) -> tuple[str, bool]:
    if not robot_name:
        return content, False
    prefix = "@" + robot_name
    trimmed = content.lstrip(" \t")
    if not trimmed.startswith(prefix):
        return content, False
    rest = trimmed[len(prefix) :]
    if not rest:
        return "", True
    if not (rest[0].isspace() or rest[0] in _AT_SEPARATORS):
        return content, False
    return rest.lstrip(f" \t{_AT_SEPARATORS}"), True


def _default_image_file_name(msg_id: str) -> str:
    if not msg_id:
        return "yunzhijia-image.png"
    return f"{msg_id}.png"


def _first_non_empty(*values: str) -> str:
    for value in values:
        if value.strip():
            return value
    return ""


__all__ = ["YunzhijiaAdapter", "build_yunzhijia_adapter"]


def build_yunzhijia_adapter(channel: IMChannel) -> YunzhijiaAdapter:
    """Construct a Yunzhijia adapter, validating required credentials."""
    send_msg_url = string_credential(channel.credentials, "send_msg_url")
    if not send_msg_url:
        raise ValidationError(
            code="im.credentials_invalid",
            message="yunzhijia send_msg_url is required",
        )
    allowed_host_suffix = string_credential(channel.credentials, "allowed_webhook_host_suffix")
    adapter = YunzhijiaAdapter(
        send_msg_url=send_msg_url,
        secret=string_credential(channel.credentials, "secret"),
        app_id=string_credential(channel.credentials, "app_id"),
        app_secret=string_credential(channel.credentials, "app_secret"),
        allowed_webhook_host_suffix=allowed_host_suffix,
        timeout_seconds=int_credential(channel.credentials, "timeout_seconds", 10),
    )
    adapter._validate_send_url()
    return adapter
