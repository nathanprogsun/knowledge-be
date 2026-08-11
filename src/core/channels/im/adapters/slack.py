"""Slack adapter — Events API (webhook) callback handling and replies.

Mirrors the upstream contract for the Slack adapter: ``verify_callback``
checks the ``X-Slack-Signature`` header computed over the raw body,
``parse_callback`` handles ``app_mention`` / ``message`` events
(optionally with file attachments), and ``send_reply`` posts through the
``chat.postMessage`` API with a ``thread_ts`` when the incoming message
was part of a thread.
"""

from __future__ import annotations

import json
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
    assert_http_ok,
    build_http_client,
    constant_time_equals,
    header_value,
    hmac_sha256_hex,
    payload_dict,
    payload_int,
    payload_list,
    payload_string,
    send_error,
    string_credential,
    timestamp_is_fresh,
)
from src.db.models.im_channel import IMChannel

# ``v0`` signature prefix used by the Slack Events API.
_SLACK_SIGNATURE_VERSION = "v0"
# Slack rejects request timestamps older than five minutes.
_SLACK_TIMESTAMP_TOLERANCE_SECONDS = 300
# Slack API host for sending replies.
_SLACK_API_BASE_URL = "https://slack.com/api"


class SlackAdapter(IMAdapter):
    """Slack platform adapter (webhook Events API + REST replies)."""

    def __init__(
        self,
        *,
        bot_token: str = "",
        signing_secret: str = "",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._bot_token = bot_token
        self._signing_secret = signing_secret
        self._http_client = build_http_client(transport=transport)
        self._connected = False

    # ── Identity ─────────────────────────────────────────────────────

    def platform(self) -> str:
        return "slack"

    # ── Callback verification ────────────────────────────────────────

    def verify_callback(self, request: CallbackRequest) -> None:
        if not self._signing_secret:
            return
        signature = header_value(request.headers, "X-Slack-Signature")
        if not signature or not timestamp_is_fresh(
            request.headers, "X-Slack-Request-Timestamp", _SLACK_TIMESTAMP_TOLERANCE_SECONDS
        ):
            raise UnauthorizedError(
                code="im.verify_failed",
                message="slack signature verification failed: missing or stale timestamp",
            )
        timestamp = header_value(request.headers, "X-Slack-Request-Timestamp")
        base = f"{_SLACK_SIGNATURE_VERSION}:{timestamp}:{request.body}"
        expected = f"{_SLACK_SIGNATURE_VERSION}={hmac_sha256_hex(self._signing_secret, base)}"
        if not constant_time_equals(expected, signature):
            raise UnauthorizedError(
                code="im.verify_failed",
                message="slack signature verification failed: signature mismatch",
            )

    # ── Callback parsing ─────────────────────────────────────────────

    def parse_callback(self, request: CallbackRequest) -> IncomingMessage | None:
        payload = _parse_body(request.body)
        if payload_string(payload, "type") == "url_verification":
            return None
        if payload_string(payload, "type") != "event_callback":
            return None
        event = payload_dict(payload, "event")
        files = [_parse_slack_file(item) for item in payload_list(event, "files")]
        files = [f for f in files if f is not None]
        event_type = payload_string(event, "type")
        if event_type == "app_mention":
            ts = payload_string(event, "thread_ts") or payload_string(event, "ts")
            return _build_message(
                user_id=payload_string(event, "user"),
                channel=payload_string(event, "channel"),
                text=payload_string(event, "text"),
                ts=ts,
                chat_type=CHAT_TYPE_GROUP,
                files=files,
            )
        if event_type == "message":
            bot_id = payload_string(event, "bot_id")
            subtype = payload_string(event, "subtype")
            if bot_id or (subtype and subtype != "file_share"):
                return None
            channel_type = payload_string(event, "channel_type")
            chat_type = (
                CHAT_TYPE_GROUP if channel_type in ("channel", "group") else CHAT_TYPE_DIRECT
            )
            ts = payload_string(event, "thread_ts") or payload_string(event, "ts")
            return _build_message(
                user_id=payload_string(event, "user"),
                channel=payload_string(event, "channel"),
                text=payload_string(event, "text"),
                ts=ts,
                chat_type=chat_type,
                files=files,
            )
        return None

    # ── URL verification ─────────────────────────────────────────────

    def handle_url_verification(self, request: CallbackRequest) -> bool:
        payload = _parse_body(request.body)
        return payload_string(payload, "type") == "url_verification"

    # ── Send reply ───────────────────────────────────────────────────

    def send_reply(self, ctx: Context, incoming: IncomingMessage, reply: ReplyMessage) -> None:
        channel = incoming.chat_id or incoming.user_id
        if not channel:
            send_error("slack", "chat.postMessage", "missing channel id")
        body: JsonObject = {"channel": channel, "text": reply.content}
        if incoming.message_id:
            body["thread_ts"] = incoming.message_id
        resp = self._http_client.post(
            f"{_SLACK_API_BASE_URL}/chat.postMessage",
            json=body,
            headers={
                "Authorization": f"Bearer {self._bot_token}",
                "Content-Type": "application/json",
            },
        )
        assert_http_ok(resp, platform="slack", action="chat.postMessage")
        try:
            result = resp.json()
        except json.JSONDecodeError:
            send_error("slack", "chat.postMessage", "non-JSON response")
        if not isinstance(result, dict) or result.get("ok") is not True:
            send_error("slack", "chat.postMessage", f"api error: {str(result)[:200]}")

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


# ── Parse helpers ─────────────────────────────────────────────────────


def _parse_body(raw: str) -> JsonObject:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_slack_file(item: JsonObject) -> dict[str, str | int] | None:
    file_id = payload_string(item, "id")
    if not file_id:
        return None
    return {
        "id": file_id,
        "name": payload_string(item, "name"),
        "size": payload_int(item, "size"),
        "mimetype": payload_string(item, "mimetype"),
        "url_private_download": payload_string(item, "url_private_download"),
    }


def _build_message(
    *,
    user_id: str,
    channel: str,
    text: str,
    ts: str,
    chat_type: str,
    files: list[dict[str, str | int]],
) -> IncomingMessage:
    content = text
    if chat_type == CHAT_TYPE_GROUP:
        while content.startswith("<@"):
            end = content.find(">")
            if end < 0:
                break
            content = content[end + 1 :].strip()

    message_type = "text"
    file_key = ""
    file_name = ""
    file_size = 0
    extra: JsonObject = {}
    if files:
        file = files[0]
        message_type = "image" if str(file["mimetype"]).startswith("image/") else "file"
        file_key = str(file["id"])
        file_name = str(file["name"])
        file_size = int(file["size"] or 0)
        extra = {"url_private_download": str(file["url_private_download"])}

    return IncomingMessage(
        platform="slack",
        message_type=message_type,
        user_id=user_id,
        chat_id=channel,
        chat_type=chat_type,
        content=content.strip(),
        message_id=ts,
        thread_id=ts,
        file_key=file_key,
        file_name=file_name,
        file_size=file_size,
        extra=extra,
    )


__all__ = ["SlackAdapter", "build_slack_adapter"]


def build_slack_adapter(channel: IMChannel) -> SlackAdapter:
    """Construct a Slack adapter from a channel row's credentials."""
    return SlackAdapter(
        bot_token=string_credential(channel.credentials, "bot_token"),
        signing_secret=string_credential(channel.credentials, "signing_secret"),
    )
