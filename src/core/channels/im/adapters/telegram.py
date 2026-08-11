"""Telegram adapter — Bot API webhook callbacks and REST replies.

Mirrors the upstream contract: ``verify_callback`` checks the optional
``X-Telegram-Bot-Api-Secret-Token`` header, ``parse_callback`` converts a
Bot API ``update`` payload into the unified message (group mentions are
stripped, documents map to file messages, photos to image messages), and
``send_reply`` posts to ``sendMessage`` with a ``message_thread_id`` when
the incoming message came from a forum topic.
"""

from __future__ import annotations

import contextlib
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
    payload_dict,
    payload_int,
    payload_list,
    payload_string,
    send_error,
    string_credential,
)
from src.db.models.im_channel import IMChannel

_TELEGRAM_API_BASE_URL = "https://api.telegram.org"


class TelegramAdapter(IMAdapter):
    """Telegram Bot API adapter (webhook callbacks + REST replies)."""

    def __init__(
        self,
        *,
        bot_token: str = "",
        secret_token: str = "",
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._bot_token = bot_token
        self._secret_token = secret_token
        self._http_client = build_http_client(transport=transport)
        self._connected = False

    # ── Identity ─────────────────────────────────────────────────────

    def platform(self) -> str:
        return "telegram"

    # ── Callback verification ────────────────────────────────────────

    def verify_callback(self, request: CallbackRequest) -> None:
        if not self._secret_token:
            return
        token = header_value(request.headers, "X-Telegram-Bot-Api-Secret-Token")
        if not constant_time_equals(token, self._secret_token):
            raise UnauthorizedError(
                code="im.verify_failed",
                message="telegram secret token verification failed",
            )

    # ── Callback parsing ─────────────────────────────────────────────

    def parse_callback(self, request: CallbackRequest) -> IncomingMessage | None:
        try:
            update = json.loads(request.body)
        except json.JSONDecodeError:
            raise UnauthorizedError(
                code="im.parse_failed",
                message="telegram callback body is not valid JSON",
            ) from None
        if not isinstance(update, dict):
            return None
        message = payload_dict(update, "message")
        if not message:
            return None
        return _parse_telegram_message(message)

    # ── URL verification ─────────────────────────────────────────────

    def handle_url_verification(self, request: CallbackRequest) -> bool:
        return False

    # ── Send reply ───────────────────────────────────────────────────

    def send_reply(self, ctx: Context, incoming: IncomingMessage, reply: ReplyMessage) -> None:
        chat_id = incoming.chat_id or incoming.user_id
        if not chat_id:
            send_error("telegram", "sendMessage", "missing chat id")
        body: JsonObject = {
            "chat_id": chat_id,
            "text": reply.content,
            "parse_mode": "Markdown",
        }
        if incoming.thread_id:
            with contextlib.suppress(ValueError):
                body["message_thread_id"] = int(incoming.thread_id)
        resp = self._http_client.post(
            f"{_TELEGRAM_API_BASE_URL}/bot{self._bot_token}/sendMessage",
            json=body,
            headers={"Content-Type": "application/json"},
        )
        assert_http_ok(resp, platform="telegram", action="sendMessage")
        result = _telegram_result(resp)
        if result is None:
            send_error("telegram", "sendMessage", "non-JSON or invalid response")
        if result.get("ok") is not True:
            description = result.get("description", "")
            send_error("telegram", "sendMessage", f"api error: {str(description)[:200]}")

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


def _parse_telegram_message(message: JsonObject) -> IncomingMessage | None:
    if not message:
        return None

    chat = payload_dict(message, "chat")
    chat_type_name = payload_string(chat, "type")
    chat_type = (
        CHAT_TYPE_GROUP if chat_type_name in ("group", "supergroup") else CHAT_TYPE_DIRECT
    )
    chat_id = str(payload_int(chat, "id")) if chat_type == CHAT_TYPE_GROUP else ""

    user_id = ""
    user_name = ""
    sender = payload_dict(message, "from")
    if sender:
        user_id = str(payload_int(sender, "id"))
        user_name = " ".join(
            part for part in (payload_string(sender, "first_name"), payload_string(sender, "last_name")) if part
        ).strip()
        if not user_name:
            user_name = payload_string(sender, "username")

    thread_id = ""
    message_thread_id = payload_int(message, "message_thread_id")
    if message_thread_id:
        thread_id = str(message_thread_id)

    content = payload_string(message, "text")
    if chat_type == CHAT_TYPE_GROUP:
        stripped = content.strip()
        space_index = stripped.find(" ")
        if space_index > 0 and "@" in stripped[:space_index]:
            content = stripped[space_index + 1 :].strip()

    incoming = IncomingMessage(
        platform="telegram",
        message_type="text",
        user_id=user_id,
        user_name=user_name,
        chat_id=chat_id,
        chat_type=chat_type,
        content=content,
        message_id=str(payload_int(message, "message_id")),
        thread_id=thread_id,
    )

    document = payload_dict(message, "document")
    if document:
        return incoming.model_copy(
            update={
                "message_type": "file",
                "file_key": payload_string(document, "file_id"),
                "file_name": payload_string(document, "file_name"),
                "file_size": payload_int(document, "file_size"),
            }
        )

    photo = payload_list(message, "photo")
    if photo:
        largest = photo[-1]
        largest_item = largest if isinstance(largest, dict) else {}
        incoming = incoming.model_copy(
            update={
                "message_type": "image",
                "file_key": payload_string(largest_item, "file_id"),
                "file_name": "photo.jpg",
                "file_size": payload_int(largest_item, "file_size"),
            }
        )
    return incoming


def _telegram_result(resp: httpx.Response) -> JsonObject | None:
    try:
        result = resp.json()
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None


__all__ = ["TelegramAdapter", "build_telegram_adapter"]


def build_telegram_adapter(channel: IMChannel) -> TelegramAdapter:
    """Construct a Telegram adapter from a channel row's credentials."""
    return TelegramAdapter(
        bot_token=string_credential(channel.credentials, "bot_token"),
        secret_token=string_credential(channel.credentials, "secret_token"),
    )
