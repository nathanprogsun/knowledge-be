"""Unit tests for the seven IM platform adapters.

Covers the shared ``IMAdapter`` contract per platform: identity,
callback verification, callback parsing, URL verification, reply
sending (via an injected ``httpx.MockTransport``), and the
connect / disconnect lifecycle. Also verifies the credential-reading
build factories and the default adapter registration.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import ClassVar

import httpx
import pytest

from src.common.exception import ExternalServiceError, UnauthorizedError, ValidationError
from src.core.channels.im.adapter_base import (
    CallbackRequest,
    EventContext,
    IncomingMessage,
    ReplyMessage,
)
from src.core.channels.im.adapters import _common as common_mod
from src.core.channels.im.adapters import dingtalk as dingtalk_mod
from src.core.channels.im.adapters import mattermost as mattermost_mod
from src.core.channels.im.adapters import qqbot as qqbot_mod
from src.core.channels.im.adapters import register_default_adapters
from src.core.channels.im.adapters import slack as slack_mod
from src.core.channels.im.adapters import telegram as telegram_mod
from src.core.channels.im.adapters import wechat as wechat_mod
from src.core.channels.im.adapters import yunzhijia as yunzhijia_mod
from src.core.channels.im.supervisor import IMSupervisor
from src.db.models.im_channel import IMChannel

_NOW = datetime(2026, 1, 1, tzinfo=UTC)

TransportHandler = Callable[[httpx.Request], httpx.Response]


# ── Shared test helpers ───────────────────────────────────────────────


def _transport(handler: TransportHandler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _channel(platform: str, credentials: dict) -> IMChannel:
    return IMChannel(
        id=f"channel-{platform}",
        tenant_id=7,
        agent_id="agent-001",
        platform=platform,
        name=f"channel-{platform}",
        enabled=True,
        mode="webhook" if platform in ("mattermost", "yunzhijia") else "websocket",
        output_mode="stream",
        knowledge_base_id="",
        bot_identity="",
        session_mode="user",
        credentials=credentials,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _incoming(**overrides: object) -> IncomingMessage:
    defaults: dict[str, object] = {
        "platform": "slack",
        "user_id": "U1",
        "user_name": "tester",
        "chat_id": "C1",
        "chat_type": "direct",
        "content": "hi",
    }
    defaults.update(overrides)
    return IncomingMessage(**defaults)


def _reply(content: str = "hello") -> ReplyMessage:
    return ReplyMessage(content=content)


def _slack_signature(secret: str, timestamp: str, body: str) -> str:
    base = f"v0:{timestamp}:{body}"
    digest = hmac.new(secret.encode(), base.encode(), hashlib.sha256).hexdigest()
    return f"v0={digest}"


def _dingtalk_sign(secret: str, timestamp: str) -> str:
    message = f"{timestamp}\n{secret}"
    digest = hmac.new(secret.encode(), message.encode(), hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


def _yunzhijia_sign(secret: str, payload: dict) -> str:
    parts = [
        str(payload.get("robotId", "")),
        str(payload.get("robotName", "")),
        str(payload.get("operatorOpenid", "")),
        str(payload.get("operatorName", "")),
        str(payload.get("time", 0)),
        str(payload.get("msgId", "")),
        str(payload.get("content", "")),
    ]
    message = ",".join(parts)
    digest = hmac.new(secret.encode(), message.encode(), hashlib.sha1).digest()
    return base64.b64encode(digest).decode("ascii")


def _callback(body: str, headers: dict | None = None) -> CallbackRequest:
    return CallbackRequest(headers=headers or {}, body=body, query={})


def _close(adapter: object) -> None:
    if hasattr(adapter, "disconnect"):
        adapter.disconnect()  # type: ignore[attr-defined]


# ── Contract: all adapters implement the base contract ────────────────


def test_register_default_adapters_covers_all_platforms() -> None:
    supervisor = IMSupervisor()
    register_default_adapters(supervisor)
    assert supervisor.registered_platforms() == [
        "dingtalk",
        "mattermost",
        "qqbot",
        "slack",
        "telegram",
        "wechat",
        "yunzhijia",
    ]


@pytest.mark.parametrize(
    "adapter",
    [
        slack_mod.SlackAdapter(),
        telegram_mod.TelegramAdapter(),
        dingtalk_mod.DingTalkAdapter(),
        mattermost_mod.MattermostAdapter(),
        wechat_mod.WeChatAdapter(),
        qqbot_mod.QQBotAdapter(),
        yunzhijia_mod.YunzhijiaAdapter(),
    ],
    ids=["slack", "telegram", "dingtalk", "mattermost", "wechat", "qqbot", "yunzhijia"],
)
def test_adapter_lifecycle(adapter: object) -> None:
    stop = asyncio.run(adapter.connect(EventContext()))  # type: ignore[attr-defined]
    assert adapter.is_connected() is True  # type: ignore[attr-defined]
    stop()
    assert adapter.is_connected() is False  # type: ignore[attr-defined]
    adapter.disconnect()  # type: ignore[attr-defined]


# ── Slack ─────────────────────────────────────────────────────────────


class TestSlack:
    def test_platform(self) -> None:
        assert slack_mod.SlackAdapter().platform() == "slack"

    def test_verify_callback_passes_with_valid_signature(self) -> None:
        adapter = slack_mod.SlackAdapter(bot_token="xoxb", signing_secret="secret")
        body = json.dumps({"type": "event_callback"})
        timestamp = str(int(time.time()))
        request = _callback(
            body,
            headers={
                "X-Slack-Signature": _slack_signature("secret", timestamp, body),
                "X-Slack-Request-Timestamp": timestamp,
            },
        )
        assert adapter.verify_callback(request) is None

    def test_verify_callback_rejects_bad_signature(self) -> None:
        adapter = slack_mod.SlackAdapter(bot_token="xoxb", signing_secret="secret")
        body = json.dumps({"type": "event_callback"})
        timestamp = str(int(time.time()))
        request = _callback(
            body,
            headers={
                "X-Slack-Signature": "v0=" + "0" * 64,
                "X-Slack-Request-Timestamp": timestamp,
            },
        )
        with pytest.raises(UnauthorizedError):
            adapter.verify_callback(request)

    def test_verify_callback_skipped_without_secret(self) -> None:
        adapter = slack_mod.SlackAdapter(bot_token="xoxb")
        assert adapter.verify_callback(_callback("{}")) is None

    def test_verify_callback_rejects_stale_timestamp(self) -> None:
        adapter = slack_mod.SlackAdapter(bot_token="xoxb", signing_secret="secret")
        body = json.dumps({"type": "event_callback"})
        stale = str(int(time.time()) - 3600)
        request = _callback(
            body,
            headers={
                "X-Slack-Signature": _slack_signature("secret", stale, body),
                "X-Slack-Request-Timestamp": stale,
            },
        )
        with pytest.raises(UnauthorizedError):
            adapter.verify_callback(request)

    def test_handle_url_verification(self) -> None:
        adapter = slack_mod.SlackAdapter()
        assert adapter.handle_url_verification(
            _callback('{"type": "url_verification", "challenge": "abc"}')
        )
        assert not adapter.handle_url_verification(_callback('{"type": "event_callback"}'))

    def test_parse_app_mention_strips_mention(self) -> None:
        adapter = slack_mod.SlackAdapter()
        body = json.dumps(
            {
                "type": "event_callback",
                "event": {
                    "type": "app_mention",
                    "user": "U123",
                    "channel": "C456",
                    "text": "<@U777> please help",
                    "ts": "1700000000.000001",
                },
            }
        )
        msg = adapter.parse_callback(_callback(body))
        assert msg is not None
        assert msg.platform == "slack"
        assert msg.chat_type == "group"
        assert msg.content == "please help"
        assert msg.thread_id == "1700000000.000001"

    def test_parse_message_group(self) -> None:
        adapter = slack_mod.SlackAdapter()
        body = json.dumps(
            {
                "type": "event_callback",
                "event": {
                    "type": "message",
                    "user": "U1",
                    "channel": "C1",
                    "channel_type": "channel",
                    "text": "<@BOT> hi",
                    "ts": "1.1",
                },
            }
        )
        msg = adapter.parse_callback(_callback(body))
        assert msg is not None
        assert msg.chat_type == "group"
        assert msg.content == "hi"

    def test_parse_message_direct(self) -> None:
        adapter = slack_mod.SlackAdapter()
        body = json.dumps(
            {
                "type": "event_callback",
                "event": {
                    "type": "message",
                    "user": "U1",
                    "channel": "D1",
                    "channel_type": "im",
                    "text": "hello",
                    "ts": "1.2",
                },
            }
        )
        msg = adapter.parse_callback(_callback(body))
        assert msg is not None
        assert msg.chat_type == "direct"
        assert msg.content == "hello"

    def test_parse_skips_bot_message(self) -> None:
        adapter = slack_mod.SlackAdapter()
        body = json.dumps(
            {
                "type": "event_callback",
                "event": {
                    "type": "message",
                    "bot_id": "B1",
                    "channel": "C1",
                    "channel_type": "channel",
                    "text": "ignored",
                    "ts": "1.3",
                },
            }
        )
        assert adapter.parse_callback(_callback(body)) is None

    def test_parse_file_message(self) -> None:
        adapter = slack_mod.SlackAdapter()
        body = json.dumps(
            {
                "type": "event_callback",
                "event": {
                    "type": "message",
                    "user": "U1",
                    "channel": "C1",
                    "channel_type": "channel",
                    "text": "see file",
                    "ts": "1.4",
                    "files": [
                        {
                            "id": "F1",
                            "name": "report.pdf",
                            "size": 123,
                            "mimetype": "application/pdf",
                            "url_private_download": "https://files.slack.com/report.pdf",
                        }
                    ],
                },
            }
        )
        msg = adapter.parse_callback(_callback(body))
        assert msg is not None
        assert msg.message_type == "file"
        assert msg.file_key == "F1"
        assert msg.file_name == "report.pdf"
        assert msg.extra["url_private_download"] == "https://files.slack.com/report.pdf"

    def test_parse_image_message(self) -> None:
        adapter = slack_mod.SlackAdapter()
        body = json.dumps(
            {
                "type": "event_callback",
                "event": {
                    "type": "message",
                    "user": "U1",
                    "channel": "C1",
                    "channel_type": "im",
                    "text": "",
                    "ts": "1.5",
                    "files": [{"id": "F2", "name": "pic.png", "size": 5, "mimetype": "image/png"}],
                },
            }
        )
        msg = adapter.parse_callback(_callback(body))
        assert msg is not None
        assert msg.message_type == "image"
        assert msg.file_key == "F2"

    def test_parse_returns_none_for_unknown_event(self) -> None:
        adapter = slack_mod.SlackAdapter()
        body = json.dumps({"type": "event_callback", "event": {"type": "reaction_added"}})
        assert adapter.parse_callback(_callback(body)) is None

    def test_send_reply_posts_thread_message(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = request.headers
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"ok": True})

        adapter = slack_mod.SlackAdapter(bot_token="xoxb", transport=_transport(handler))
        try:
            adapter.send_reply(
                EventContext(),
                _incoming(chat_id="C1", message_id="ts-1"),
                _reply("answer"),
            )
        finally:
            _close(adapter)
        assert captured["url"] == "https://slack.com/api/chat.postMessage"
        assert captured["headers"]["Authorization"] == "Bearer xoxb"
        assert captured["body"] == {"channel": "C1", "text": "answer", "thread_ts": "ts-1"}

    def test_send_reply_falls_back_to_user_id(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"ok": True})

        adapter = slack_mod.SlackAdapter(bot_token="xoxb", transport=_transport(handler))
        try:
            adapter.send_reply(EventContext(), _incoming(chat_id="", user_id="U1"), _reply())
        finally:
            _close(adapter)
        assert captured["body"]["channel"] == "U1"

    def test_send_reply_raises_on_api_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": False, "error": "invalid_auth"})

        adapter = slack_mod.SlackAdapter(bot_token="xoxb", transport=_transport(handler))
        try:
            with pytest.raises(ExternalServiceError):
                adapter.send_reply(EventContext(), _incoming(), _reply())
        finally:
            _close(adapter)

    def test_send_reply_raises_on_http_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        adapter = slack_mod.SlackAdapter(bot_token="xoxb", transport=_transport(handler))
        try:
            with pytest.raises(ExternalServiceError):
                adapter.send_reply(EventContext(), _incoming(), _reply())
        finally:
            _close(adapter)


# ── Telegram ──────────────────────────────────────────────────────────


class TestTelegram:
    def test_platform(self) -> None:
        assert telegram_mod.TelegramAdapter().platform() == "telegram"

    def test_verify_callback_checks_secret_token(self) -> None:
        adapter = telegram_mod.TelegramAdapter(bot_token="123:abc", secret_token="s3cret")
        assert (
            adapter.verify_callback(
                _callback("{}", headers={"X-Telegram-Bot-Api-Secret-Token": "s3cret"})
            )
            is None
        )
        with pytest.raises(UnauthorizedError):
            adapter.verify_callback(
                _callback("{}", headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"})
            )

    def test_verify_callback_skipped_without_secret(self) -> None:
        adapter = telegram_mod.TelegramAdapter(bot_token="123:abc")
        assert adapter.verify_callback(_callback("{}")) is None

    def test_handle_url_verification_returns_false(self) -> None:
        adapter = telegram_mod.TelegramAdapter()
        assert not adapter.handle_url_verification(_callback("{}"))

    def test_parse_direct_message(self) -> None:
        adapter = telegram_mod.TelegramAdapter()
        body = json.dumps(
            {
                "update_id": 1,
                "message": {
                    "message_id": 42,
                    "from": {"id": 123, "first_name": "Ada", "last_name": "L", "username": "ada"},
                    "chat": {"id": 456, "type": "private"},
                    "text": "hi",
                },
            }
        )
        msg = adapter.parse_callback(_callback(body))
        assert msg is not None
        assert msg.chat_type == "direct"
        assert msg.chat_id == ""
        assert msg.user_id == "123"
        assert msg.user_name == "Ada L"
        assert msg.content == "hi"
        assert msg.message_id == "42"

    def test_parse_group_message_strips_mention(self) -> None:
        adapter = telegram_mod.TelegramAdapter()
        body = json.dumps(
            {
                "update_id": 2,
                "message": {
                    "message_id": 43,
                    "from": {"id": 123, "first_name": "Ada"},
                    "chat": {"id": -100, "type": "supergroup"},
                    "text": "/start@MyBot please help",
                    "message_thread_id": 7,
                },
            }
        )
        msg = adapter.parse_callback(_callback(body))
        assert msg is not None
        assert msg.chat_type == "group"
        assert msg.chat_id == "-100"
        assert msg.content == "please help"
        assert msg.thread_id == "7"

    def test_parse_document_message(self) -> None:
        adapter = telegram_mod.TelegramAdapter()
        body = json.dumps(
            {
                "update_id": 3,
                "message": {
                    "message_id": 44,
                    "from": {"id": 123},
                    "chat": {"id": 456, "type": "private"},
                    "text": "",
                    "document": {"file_id": "FILE1", "file_name": "doc.pdf", "file_size": 99},
                },
            }
        )
        msg = adapter.parse_callback(_callback(body))
        assert msg is not None
        assert msg.message_type == "file"
        assert msg.file_key == "FILE1"
        assert msg.file_name == "doc.pdf"
        assert msg.file_size == 99

    def test_parse_photo_message(self) -> None:
        adapter = telegram_mod.TelegramAdapter()
        body = json.dumps(
            {
                "update_id": 4,
                "message": {
                    "message_id": 45,
                    "from": {"id": 123},
                    "chat": {"id": 456, "type": "private"},
                    "photo": [
                        {"file_id": "SMALL", "file_size": 10},
                        {"file_id": "LARGE", "file_size": 100},
                    ],
                },
            }
        )
        msg = adapter.parse_callback(_callback(body))
        assert msg is not None
        assert msg.message_type == "image"
        assert msg.file_key == "LARGE"
        assert msg.file_name == "photo.jpg"

    def test_parse_returns_none_without_message(self) -> None:
        adapter = telegram_mod.TelegramAdapter()
        assert adapter.parse_callback(_callback(json.dumps({"update_id": 5}))) is None

    def test_send_reply_posts_send_message(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"ok": True})

        adapter = telegram_mod.TelegramAdapter(bot_token="123:abc", transport=_transport(handler))
        try:
            adapter.send_reply(EventContext(), _incoming(chat_id="", user_id="456"), _reply("hi"))
        finally:
            _close(adapter)
        assert captured["url"] == "https://api.telegram.org/bot123:abc/sendMessage"
        assert captured["body"] == {"chat_id": "456", "text": "hi", "parse_mode": "Markdown"}

    def test_send_reply_includes_thread_id(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"ok": True})

        adapter = telegram_mod.TelegramAdapter(bot_token="123:abc", transport=_transport(handler))
        try:
            adapter.send_reply(
                EventContext(), _incoming(chat_id="100", thread_id="7"), _reply("hi")
            )
        finally:
            _close(adapter)
        assert captured["body"]["message_thread_id"] == 7

    def test_send_reply_raises_on_api_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": False, "description": "Bad Request"})

        adapter = telegram_mod.TelegramAdapter(bot_token="123:abc", transport=_transport(handler))
        try:
            with pytest.raises(ExternalServiceError):
                adapter.send_reply(EventContext(), _incoming(), _reply())
        finally:
            _close(adapter)


# ── DingTalk ──────────────────────────────────────────────────────────


class TestDingTalk:
    def test_platform(self) -> None:
        assert dingtalk_mod.DingTalkAdapter().platform() == "dingtalk"

    def test_verify_callback_passes_with_valid_signature(self) -> None:
        adapter = dingtalk_mod.DingTalkAdapter(client_id="cid", client_secret="secret")
        timestamp = str(int(time.time() * 1000))
        request = _callback(
            "{}", headers={"Timestamp": timestamp, "Sign": _dingtalk_sign("secret", timestamp)}
        )
        assert adapter.verify_callback(request) is None

    def test_verify_callback_rejects_bad_signature(self) -> None:
        adapter = dingtalk_mod.DingTalkAdapter(client_id="cid", client_secret="secret")
        timestamp = str(int(time.time() * 1000))
        request = _callback("{}", headers={"Timestamp": timestamp, "Sign": "AAAA"})
        with pytest.raises(UnauthorizedError):
            adapter.verify_callback(request)

    def test_verify_callback_rejects_missing_headers(self) -> None:
        adapter = dingtalk_mod.DingTalkAdapter(client_id="cid", client_secret="secret")
        with pytest.raises(UnauthorizedError):
            adapter.verify_callback(_callback("{}"))

    def test_handle_url_verification_returns_false(self) -> None:
        adapter = dingtalk_mod.DingTalkAdapter()
        assert not adapter.handle_url_verification(_callback("{}"))

    def test_parse_text_message(self) -> None:
        adapter = dingtalk_mod.DingTalkAdapter()
        body = json.dumps(
            {
                "conversationId": "cid1",
                "conversationType": "1",
                "msgId": "m1",
                "msgtype": "text",
                "text": {"content": " hello "},
                "senderNick": "nick",
                "senderStaffId": "staff1",
                "sessionWebhook": "https://hook.example",
            }
        )
        msg = adapter.parse_callback(_callback(body))
        assert msg is not None
        assert msg.chat_type == "direct"
        assert msg.chat_id == ""
        assert msg.content == "hello"
        assert msg.user_id == "staff1"
        assert msg.extra["session_webhook"] == "https://hook.example"

    def test_parse_group_message(self) -> None:
        adapter = dingtalk_mod.DingTalkAdapter()
        body = json.dumps(
            {
                "conversationId": "group1",
                "conversationType": "2",
                "msgId": "m2",
                "msgtype": "text",
                "text": {"content": "hi"},
            }
        )
        msg = adapter.parse_callback(_callback(body))
        assert msg is not None
        assert msg.chat_type == "group"
        assert msg.chat_id == "group1"

    def test_parse_file_message(self) -> None:
        adapter = dingtalk_mod.DingTalkAdapter()
        body = json.dumps(
            {
                "conversationId": "cid1",
                "conversationType": "1",
                "msgId": "m3",
                "msgtype": "file",
                "content": {"downloadCode": "dc1", "fileName": "a.pdf"},
                "robotCode": "robot1",
            }
        )
        msg = adapter.parse_callback(_callback(body))
        assert msg is not None
        assert msg.message_type == "file"
        assert msg.file_key == "dc1"
        assert msg.file_name == "a.pdf"
        assert msg.extra["robot_code"] == "robot1"

    def test_parse_picture_message_defaults_name(self) -> None:
        adapter = dingtalk_mod.DingTalkAdapter()
        body = json.dumps(
            {
                "conversationId": "cid1",
                "conversationType": "1",
                "msgId": "m4",
                "msgtype": "picture",
                "content": {"pictureDownloadCode": "pdc1"},
            }
        )
        msg = adapter.parse_callback(_callback(body))
        assert msg is not None
        assert msg.message_type == "image"
        assert msg.file_key == "pdc1"
        assert msg.file_name == "m4.png"

    def test_send_reply_via_session_webhook(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={})

        adapter = dingtalk_mod.DingTalkAdapter(client_id="cid", transport=_transport(handler))
        incoming = _incoming(extra={"session_webhook": "https://hook.example"})
        try:
            adapter.send_reply(EventContext(), incoming, _reply("answer"))
        finally:
            _close(adapter)
        assert captured["url"] == "https://hook.example"
        assert captured["body"]["msgtype"] == "markdown"
        assert captured["body"]["markdown"]["text"] == "answer"

    def test_send_reply_via_openapi_group_with_token_cache(self) -> None:
        token_requests: list[httpx.Request] = []
        post_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/oauth2/accessToken"):
                token_requests.append(request)
                return httpx.Response(200, json={"accessToken": "tok1", "expireIn": 7200})
            post_requests.append(request)
            return httpx.Response(200, json={})

        adapter = dingtalk_mod.DingTalkAdapter(client_id="cid", transport=_transport(handler))
        incoming = _incoming(chat_id="group1", chat_type="group", extra={})
        try:
            adapter.send_reply(EventContext(), incoming, _reply("a"))
            adapter.send_reply(EventContext(), incoming, _reply("b"))
        finally:
            _close(adapter)
        assert len(token_requests) == 1
        assert len(post_requests) == 2
        assert str(post_requests[0].url).endswith("/v1.0/robot/groupMessages/send")
        assert post_requests[0].headers["x-acs-dingtalk-access-token"] == "tok1"
        body = json.loads(post_requests[0].content)
        assert body["openConversationId"] == "group1"
        assert body["robotCode"] == "cid"

    def test_send_reply_via_openapi_direct(self) -> None:
        post_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/oauth2/accessToken"):
                return httpx.Response(200, json={"accessToken": "tok1", "expireIn": 7200})
            post_requests.append(request)
            return httpx.Response(200, json={})

        adapter = dingtalk_mod.DingTalkAdapter(client_id="cid", transport=_transport(handler))
        incoming = _incoming(user_id="user1", chat_type="direct", extra={})
        try:
            adapter.send_reply(EventContext(), incoming, _reply("a"))
        finally:
            _close(adapter)
        assert str(post_requests[0].url).endswith("/v1.0/robot/oToMessages/batchSend")
        body = json.loads(post_requests[0].content)
        assert body["userIds"] == ["user1"]

    def test_send_reply_raises_on_http_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        adapter = dingtalk_mod.DingTalkAdapter(client_id="cid", transport=_transport(handler))
        incoming = _incoming(extra={"session_webhook": "https://hook.example"})
        try:
            with pytest.raises(ExternalServiceError):
                adapter.send_reply(EventContext(), incoming, _reply("a"))
        finally:
            _close(adapter)


# ── Mattermost ────────────────────────────────────────────────────────


class TestMattermost:
    _BASE: ClassVar[dict[str, str]] = {
        "site_url": "https://mm.example.com",
        "bot_token": "bt",
        "outgoing_token": "ot",
    }

    def test_platform(self) -> None:
        assert mattermost_mod.MattermostAdapter().platform() == "mattermost"

    def test_verify_callback_passes_with_json_token(self) -> None:
        adapter = mattermost_mod.MattermostAdapter(outgoing_token="ot")
        body = json.dumps({"token": "ot", "channel_id": "c1"})
        assert (
            adapter.verify_callback(_callback(body, headers={"Content-Type": "application/json"}))
            is None
        )

    def test_verify_callback_passes_with_form_token(self) -> None:
        adapter = mattermost_mod.MattermostAdapter(outgoing_token="ot")
        body = "token=ot&channel_id=c1&text=hi"
        assert (
            adapter.verify_callback(
                _callback(body, headers={"Content-Type": "application/x-www-form-urlencoded"})
            )
            is None
        )

    def test_verify_callback_rejects_bad_token(self) -> None:
        adapter = mattermost_mod.MattermostAdapter(outgoing_token="ot")
        body = json.dumps({"token": "wrong"})
        with pytest.raises(UnauthorizedError):
            adapter.verify_callback(_callback(body))

    def test_handle_url_verification_returns_false(self) -> None:
        adapter = mattermost_mod.MattermostAdapter()
        assert not adapter.handle_url_verification(_callback("{}"))

    def test_parse_text_message(self) -> None:
        adapter = mattermost_mod.MattermostAdapter(**self._BASE)
        body = json.dumps(
            {
                "token": "ot",
                "user_id": "u1",
                "user_name": "tester",
                "channel_id": "c1",
                "post_id": "p1",
                "text": " hello ",
            }
        )
        msg = adapter.parse_callback(_callback(body))
        assert msg is not None
        assert msg.chat_type == "group"
        assert msg.chat_id == "c1"
        assert msg.content == "hello"
        assert msg.message_id == "p1"
        assert msg.thread_id == "p1"

    def test_parse_skips_own_bot_message(self) -> None:
        adapter = mattermost_mod.MattermostAdapter(**{**self._BASE, "bot_user_id": "u9"})
        body = json.dumps({"token": "ot", "user_id": "u9", "channel_id": "c1", "text": "hi"})
        assert adapter.parse_callback(_callback(body)) is None

    def test_parse_skips_empty_message(self) -> None:
        adapter = mattermost_mod.MattermostAdapter(**self._BASE)
        body = json.dumps({"token": "ot", "user_id": "u1", "channel_id": "c1", "text": "   "})
        assert adapter.parse_callback(_callback(body)) is None

    def test_parse_file_message(self) -> None:
        adapter = mattermost_mod.MattermostAdapter(**self._BASE)
        body = json.dumps(
            {
                "token": "ot",
                "user_id": "u1",
                "channel_id": "c1",
                "post_id": "p1",
                "text": "",
                "file_ids": ["f1", "f2"],
            }
        )
        msg = adapter.parse_callback(_callback(body))
        assert msg is not None
        assert msg.message_type == "file"
        assert msg.file_key == "f1"
        assert msg.extra["file_ids"] == "f1,f2"

    def test_parse_resolves_thread_root_via_api(self) -> None:
        get_hits: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                get_hits.append(str(request.url))
                return httpx.Response(200, json={"id": "p2", "root_id": "root-9"})
            return httpx.Response(500, text="unexpected")

        adapter = mattermost_mod.MattermostAdapter(**self._BASE, transport=_transport(handler))
        body = json.dumps(
            {
                "token": "ot",
                "user_id": "u1",
                "channel_id": "c1",
                "post_id": "p2",
                "text": "hi",
                "root_id": "",
            }
        )
        try:
            msg = adapter.parse_callback(_callback(body))
        finally:
            _close(adapter)
        assert msg is not None
        assert get_hits == ["https://mm.example.com/api/v4/posts/p2"]
        assert msg.thread_id == "root-9"
        assert msg.extra["thread_root_id"] == "root-9"

    def test_parse_falls_back_to_post_id_when_resolution_fails(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="not found")

        adapter = mattermost_mod.MattermostAdapter(**self._BASE, transport=_transport(handler))
        body = json.dumps(
            {
                "token": "ot",
                "user_id": "u1",
                "channel_id": "c1",
                "post_id": "p3",
                "text": "hi",
                "root_id": "",
            }
        )
        try:
            msg = adapter.parse_callback(_callback(body))
        finally:
            _close(adapter)
        assert msg is not None
        assert msg.thread_id == "p3"

    def test_send_reply_posts_with_thread_root(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = request.headers
            captured["body"] = json.loads(request.content)
            return httpx.Response(201, json={"id": "new-post"})

        adapter = mattermost_mod.MattermostAdapter(**self._BASE, transport=_transport(handler))
        incoming = _incoming(chat_id="c1", extra={"thread_root_id": "root-9", "channel_id": "c1"})
        try:
            adapter.send_reply(EventContext(), incoming, _reply("answer"))
        finally:
            _close(adapter)
        assert captured["url"] == "https://mm.example.com/api/v4/posts"
        assert captured["headers"]["Authorization"] == "Bearer bt"
        assert captured["body"] == {"channel_id": "c1", "message": "answer", "root_id": "root-9"}

    def test_send_reply_raises_on_http_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, text="forbidden")

        adapter = mattermost_mod.MattermostAdapter(**self._BASE, transport=_transport(handler))
        try:
            with pytest.raises(ExternalServiceError):
                adapter.send_reply(EventContext(), _incoming(chat_id="c1"), _reply())
        finally:
            _close(adapter)

    def test_build_requires_credentials(self) -> None:
        with pytest.raises(ValidationError):
            mattermost_mod.build_mattermost_adapter(_channel("mattermost", {}))
        with pytest.raises(ValidationError):
            mattermost_mod.build_mattermost_adapter(
                _channel(
                    "mattermost", {"outgoing_token": "ot", "site_url": "https://mm.example.com"}
                )
            )
        with pytest.raises(ValidationError):
            mattermost_mod.build_mattermost_adapter(
                _channel("mattermost", {"outgoing_token": "ot", "bot_token": "bt"})
            )
        with pytest.raises(ValidationError):
            mattermost_mod.build_mattermost_adapter(
                _channel(
                    "mattermost",
                    {"outgoing_token": "ot", "site_url": "ftp://mm.example.com", "bot_token": "bt"},
                )
            )


# ── WeChat ────────────────────────────────────────────────────────────


class TestWeChat:
    def test_platform(self) -> None:
        assert wechat_mod.WeChatAdapter().platform() == "wechat"

    def test_verify_callback_rejects_webhook(self) -> None:
        adapter = wechat_mod.WeChatAdapter()
        with pytest.raises(ValidationError):
            adapter.verify_callback(_callback("{}"))

    def test_parse_callback_rejects_webhook(self) -> None:
        adapter = wechat_mod.WeChatAdapter()
        with pytest.raises(ValidationError):
            adapter.parse_callback(_callback("{}"))

    def test_handle_url_verification_returns_false(self) -> None:
        adapter = wechat_mod.WeChatAdapter()
        assert not adapter.handle_url_verification(_callback("{}"))

    def test_send_reply_posts_ilink_payload(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = request.headers
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={})

        adapter = wechat_mod.WeChatAdapter(bot_token="wt", transport=_transport(handler))
        incoming = _incoming(user_id="wx-1", extra={"context_token": "ctx-1"})
        try:
            adapter.send_reply(EventContext(), incoming, _reply("answer"))
        finally:
            _close(adapter)
        assert captured["url"] == "https://ilinkai.weixin.qq.com/ilink/bot/sendmessage"
        assert captured["headers"]["AuthorizationType"] == "ilink_bot_token"
        assert captured["headers"]["Authorization"] == "Bearer wt"
        assert captured["headers"]["X-WECHAT-UIN"]
        msg = captured["body"]["msg"]
        assert msg["to_user_id"] == "wx-1"
        assert msg["message_type"] == 2
        assert msg["context_token"] == "ctx-1"
        assert msg["item_list"][0]["text_item"]["text"] == "answer"
        assert captured["body"]["base_info"]["channel_version"]

    def test_send_reply_raises_on_http_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        adapter = wechat_mod.WeChatAdapter(bot_token="wt", transport=_transport(handler))
        try:
            with pytest.raises(ExternalServiceError):
                adapter.send_reply(EventContext(), _incoming(user_id="wx-1"), _reply())
        finally:
            _close(adapter)


# ── QQBot ─────────────────────────────────────────────────────────────


class TestQQBot:
    def test_platform(self) -> None:
        assert qqbot_mod.QQBotAdapter().platform() == "qqbot"

    def test_verify_callback_passes(self) -> None:
        adapter = qqbot_mod.QQBotAdapter()
        assert adapter.verify_callback(_callback("{}")) is None

    def test_handle_url_verification_returns_false(self) -> None:
        adapter = qqbot_mod.QQBotAdapter()
        assert not adapter.handle_url_verification(_callback("{}"))

    def test_parse_c2c_message(self) -> None:
        adapter = qqbot_mod.QQBotAdapter()
        event = {
            "id": "msg-1",
            "content": " hello ",
            "author": {"user_openid": "u-open", "username": "tester"},
        }
        body = json.dumps({"op": 0, "t": "C2C_MESSAGE_CREATE", "d": json.dumps(event)})
        msg = adapter.parse_callback(_callback(body))
        assert msg is not None
        assert msg.chat_type == "direct"
        assert msg.user_id == "u-open"
        assert msg.content == "hello"
        assert msg.message_id == "msg-1"
        assert msg.extra["chat_kind"] == "c2c"

    def test_parse_group_message(self) -> None:
        adapter = qqbot_mod.QQBotAdapter()
        event = {
            "id": "msg-2",
            "content": "hi",
            "group_openid": "g-open",
            "author": {"member_openid": "m-open"},
        }
        body = json.dumps({"op": 0, "t": "GROUP_AT_MESSAGE_CREATE", "d": json.dumps(event)})
        msg = adapter.parse_callback(_callback(body))
        assert msg is not None
        assert msg.chat_type == "group"
        assert msg.chat_id == "g-open"
        assert msg.user_id == "m-open"
        assert msg.extra["chat_kind"] == "group"

    def test_parse_returns_none_for_non_dispatch(self) -> None:
        adapter = qqbot_mod.QQBotAdapter()
        assert adapter.parse_callback(_callback(json.dumps({"op": 1}))) is None

    def test_send_reply_group(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/getAppAccessToken"):
                return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
            captured["url"] = str(request.url)
            captured["headers"] = request.headers
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={})

        adapter = qqbot_mod.QQBotAdapter(
            app_id="app1", client_secret="sec1", transport=_transport(handler)
        )
        incoming = _incoming(chat_id="g-open", chat_type="group", extra={"message_id": "m1"})
        try:
            adapter.send_reply(EventContext(), incoming, _reply("answer"))
        finally:
            _close(adapter)
        assert captured["url"] == "https://api.sgroup.qq.com/v2/groups/g-open/messages"
        assert captured["headers"]["Authorization"].startswith("QQBot ")
        assert captured["body"] == {
            "content": "answer",
            "msg_type": 0,
            "msg_id": "m1",
            "msg_seq": 1,
        }

    def test_send_reply_direct_and_token_cached(self) -> None:
        token_requests: list[httpx.Request] = []
        post_requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/getAppAccessToken"):
                token_requests.append(request)
                return httpx.Response(200, json={"access_token": "tok", "expires_in": "3600"})
            post_requests.append(request)
            return httpx.Response(200, json={})

        adapter = qqbot_mod.QQBotAdapter(
            app_id="app1", client_secret="sec1", transport=_transport(handler)
        )
        incoming = _incoming(user_id="u-open", chat_type="direct")
        try:
            adapter.send_reply(EventContext(), incoming, _reply("a"))
            adapter.send_reply(EventContext(), incoming, _reply("b"))
        finally:
            _close(adapter)
        assert len(token_requests) == 1
        assert len(post_requests) == 2
        assert str(post_requests[0].url) == "https://api.sgroup.qq.com/v2/users/u-open/messages"

    def test_send_reply_noops_on_empty_content(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("no request expected")

        adapter = qqbot_mod.QQBotAdapter(
            app_id="app1", client_secret="sec1", transport=_transport(handler)
        )
        adapter.send_reply(EventContext(), _incoming(user_id="u-open"), _reply("   "))
        _close(adapter)

    def test_build_requires_credentials(self) -> None:
        with pytest.raises(ValidationError):
            qqbot_mod.build_qqbot_adapter(_channel("qqbot", {}))
        with pytest.raises(ValidationError):
            qqbot_mod.build_qqbot_adapter(_channel("qqbot", {"app_id": "a"}))


# ── Yunzhijia ─────────────────────────────────────────────────────────


class TestYunzhijia:
    _SEND_URL = "https://webhook.yunzhijia.com/gateway/send?yzjtoken=t"
    _BASE: ClassVar[dict[str, str]] = {
        "send_msg_url": _SEND_URL,
        "allowed_webhook_host_suffix": "yunzhijia.com",
    }

    def test_platform(self) -> None:
        assert yunzhijia_mod.YunzhijiaAdapter().platform() == "yunzhijia"

    def test_verify_callback_passes_with_valid_signature(self) -> None:
        adapter = yunzhijia_mod.YunzhijiaAdapter(**self._BASE, secret="sec")
        payload = {
            "robotId": "r1",
            "robotName": "Bot",
            "operatorOpenid": "o1",
            "operatorName": "Alice",
            "time": 1700000000000,
            "msgId": "m1",
            "content": "hi",
        }
        request = _callback(json.dumps(payload), headers={"sign": _yunzhijia_sign("sec", payload)})
        assert adapter.verify_callback(request) is None

    def test_verify_callback_rejects_bad_signature(self) -> None:
        adapter = yunzhijia_mod.YunzhijiaAdapter(**self._BASE, secret="sec")
        payload = {"robotId": "r1", "time": 1700000000000, "msgId": "m1", "content": "hi"}
        request = _callback(json.dumps(payload), headers={"sign": "AAAA"})
        with pytest.raises(UnauthorizedError):
            adapter.verify_callback(request)

    def test_verify_callback_skipped_without_secret(self) -> None:
        adapter = yunzhijia_mod.YunzhijiaAdapter(**self._BASE)
        assert adapter.verify_callback(_callback("{}")) is None

    def test_handle_url_verification_returns_false(self) -> None:
        adapter = yunzhijia_mod.YunzhijiaAdapter()
        assert not adapter.handle_url_verification(_callback("{}"))

    def test_parse_cleans_at_mention(self) -> None:
        adapter = yunzhijia_mod.YunzhijiaAdapter(**self._BASE)
        payload = {
            "type": 2,
            "robotId": "r1",
            "robotName": "Bot",
            "operatorOpenid": "o1",
            "operatorName": "Alice",
            "time": 1700000000000,
            "msgId": "m1",
            "content": "@Bot please help",
            "groupId": "g1",
            "groupType": 2,
        }
        msg = adapter.parse_callback(_callback(json.dumps(payload)))
        assert msg is not None
        assert msg.content == "please help"
        assert msg.chat_type == "group"
        assert msg.chat_id == "g1"
        assert msg.user_id == "o1"
        assert msg.extra["group_type"] == "2"

    def test_parse_skips_without_mention(self) -> None:
        adapter = yunzhijia_mod.YunzhijiaAdapter(**self._BASE)
        payload = {
            "type": 2,
            "robotId": "r1",
            "robotName": "Bot",
            "msgId": "m1",
            "content": "no mention",
        }
        assert adapter.parse_callback(_callback(json.dumps(payload))) is None

    def test_parse_mentions_via_notify_to(self) -> None:
        adapter = yunzhijia_mod.YunzhijiaAdapter(**self._BASE)
        payload = {
            "type": 2,
            "robotId": "r1",
            "robotName": "Bot",
            "operatorOpenid": "o1",
            "msgId": "m1",
            "content": "hello",
            "msgParam": json.dumps({"notifyTo": ["r1"]}),
        }
        msg = adapter.parse_callback(_callback(json.dumps(payload)))
        assert msg is not None
        assert msg.content == "hello"

    def test_parse_image_message(self) -> None:
        adapter = yunzhijia_mod.YunzhijiaAdapter(**self._BASE)
        payload = {
            "type": 2,
            "robotId": "r1",
            "robotName": "Bot",
            "operatorOpenid": "o1",
            "msgId": "m9",
            "content": "",
            "msgParam": json.dumps(
                {
                    "desc": [{"type": "image", "data": "img-1", "w": 640, "h": 480}],
                    "notifyTo": ["r1"],
                }
            ),
        }
        msg = adapter.parse_callback(_callback(json.dumps(payload)))
        assert msg is not None
        assert msg.message_type == "image"
        assert msg.file_key == "img-1"
        assert msg.file_name == "m9.png"
        assert msg.extra["yunzhijia_image_width"] == "640"

    def test_parse_skips_non_text_type(self) -> None:
        adapter = yunzhijia_mod.YunzhijiaAdapter(**self._BASE)
        payload = {"type": 3, "msgId": "m1", "content": "hi"}
        assert adapter.parse_callback(_callback(json.dumps(payload))) is None

    def test_send_reply_posts_payload(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={})

        adapter = yunzhijia_mod.YunzhijiaAdapter(**self._BASE, transport=_transport(handler))
        incoming = _incoming(user_id="o1", extra={"group_type": "2"})
        try:
            adapter.send_reply(EventContext(), incoming, _reply("answer"))
        finally:
            _close(adapter)
        assert captured["url"] == self._SEND_URL
        assert captured["body"]["msgtype"] == 2
        assert captured["body"]["content"] == "answer"
        assert captured["body"]["param"] == {"formatType": "markdown"}
        assert captured["body"]["notifyParams"] == [{"type": "openIds", "values": ["o1"]}]

    def test_send_reply_format_override(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={})

        adapter = yunzhijia_mod.YunzhijiaAdapter(**self._BASE, transport=_transport(handler))
        reply = ReplyMessage(content="answer", extra={"yunzhijia_format_type": "plain"})
        try:
            adapter.send_reply(
                EventContext(), _incoming(user_id="o1", extra={"group_type": "2"}), reply
            )
        finally:
            _close(adapter)
        assert captured["body"]["param"] == {"formatType": "plain"}

    def test_send_reply_omits_notify_params_for_group_type_3(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={})

        adapter = yunzhijia_mod.YunzhijiaAdapter(**self._BASE, transport=_transport(handler))
        incoming = _incoming(user_id="o1", extra={"group_type": "3"})
        try:
            adapter.send_reply(EventContext(), incoming, _reply("answer"))
        finally:
            _close(adapter)
        assert "notifyParams" not in captured["body"]

    def test_send_reply_raises_on_http_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        adapter = yunzhijia_mod.YunzhijiaAdapter(**self._BASE, transport=_transport(handler))
        try:
            with pytest.raises(ExternalServiceError):
                adapter.send_reply(EventContext(), _incoming(user_id="o1"), _reply())
        finally:
            _close(adapter)

    def test_build_validates_send_url(self) -> None:
        with pytest.raises(ValidationError):
            yunzhijia_mod.build_yunzhijia_adapter(_channel("yunzhijia", {}))
        with pytest.raises(ValidationError):
            yunzhijia_mod.build_yunzhijia_adapter(
                _channel(
                    "yunzhijia",
                    {
                        "send_msg_url": "http://evil.example.com/send",
                        "allowed_webhook_host_suffix": "yunzhijia.com",
                    },
                )
            )
        adapter = yunzhijia_mod.build_yunzhijia_adapter(
            _channel(
                "yunzhijia",
                {
                    "send_msg_url": "https://hook.yunzhijia.com/send",
                    "allowed_webhook_host_suffix": "yunzhijia.com",
                },
            )
        )
        assert adapter.platform() == "yunzhijia"


# ── Shared helpers ────────────────────────────────────────────────────


class TestCommonHelpers:
    def test_string_credential_coercions(self) -> None:
        creds = {"s": "abc", "b": True, "i": 42, "f": 3.5}
        assert common_mod.string_credential(creds, "s") == "abc"
        assert common_mod.string_credential(creds, "b") == ""
        assert common_mod.string_credential(creds, "i") == "42"
        assert common_mod.string_credential(creds, "f") == "4"
        assert common_mod.string_credential(creds, "missing") == ""

    def test_bool_credential_coercions(self) -> None:
        creds = {"t": True, "f": False, "yes": "yes", "one": "1", "no": "no", "n": 5, "z": 0}
        assert common_mod.bool_credential(creds, "t") is True
        assert common_mod.bool_credential(creds, "f") is False
        assert common_mod.bool_credential(creds, "yes") is True
        assert common_mod.bool_credential(creds, "one") is True
        assert common_mod.bool_credential(creds, "no") is False
        assert common_mod.bool_credential(creds, "n") is True
        assert common_mod.bool_credential(creds, "z") is False
        assert common_mod.bool_credential(creds, "missing") is False

    def test_int_credential_coercions(self) -> None:
        creds = {"i": 25, "neg": -1, "zero": 0, "fl": 3.9, "s": "30", "bad": "x", "b": True}
        assert common_mod.int_credential(creds, "i", 10) == 25
        assert common_mod.int_credential(creds, "neg", 10) == 10
        assert common_mod.int_credential(creds, "zero", 10) == 10
        assert common_mod.int_credential(creds, "fl", 10) == 3
        assert common_mod.int_credential(creds, "s", 10) == 30
        assert common_mod.int_credential(creds, "bad", 10) == 10
        assert common_mod.int_credential(creds, "b", 10) == 10
        assert common_mod.int_credential(creds, "missing", 10) == 10

    def test_header_value_case_insensitive(self) -> None:
        headers = {"X-Slack-Signature": "abc", "x-lower": "v", "num": 7}
        assert common_mod.header_value(headers, "x-slack-signature") == "abc"
        assert common_mod.header_value(headers, "X-LOWER") == "v"
        assert common_mod.header_value(headers, "num") == "7"
        assert common_mod.header_value(headers, "missing") == ""

    def test_payload_accessors(self) -> None:
        payload = {"s": "x", "i": 3, "d": {"k": "v"}, "l": [1, 2], "f": 2.5, "b": True}
        assert common_mod.payload_string(payload, "s") == "x"
        assert common_mod.payload_string(payload, "i") == "3"
        assert common_mod.payload_string(payload, "b") == ""
        assert common_mod.payload_int(payload, "i") == 3
        assert common_mod.payload_int(payload, "f") == 2
        assert common_mod.payload_int(payload, "b") == 0
        assert common_mod.payload_dict(payload, "d") == {"k": "v"}
        assert common_mod.payload_dict(payload, "missing") == {}
        assert common_mod.payload_list(payload, "l") == [1, 2]
        assert common_mod.payload_list(payload, "missing") == []

    def test_timestamp_freshness(self) -> None:
        now = str(int(time.time()))
        stale = str(int(time.time()) - 3600)
        assert common_mod.timestamp_is_fresh({"ts": now}, "ts", 300)
        assert not common_mod.timestamp_is_fresh({"ts": stale}, "ts", 300)
        assert not common_mod.timestamp_is_fresh({"ts": "not-a-number"}, "ts", 300)
        assert not common_mod.timestamp_is_fresh({}, "ts", 300)
        now_ms = str(int(time.time() * 1000))
        assert common_mod.timestamp_ms_is_fresh({"ts": now_ms}, "ts", 3600)
        assert not common_mod.timestamp_ms_is_fresh({"ts": "bad"}, "ts", 3600)
        assert not common_mod.timestamp_ms_is_fresh({}, "ts", 3600)

    def test_hmac_primitives(self) -> None:
        assert common_mod.hmac_sha256_base64("k", "m") == common_mod.hmac_sha256_base64("k", "m")
        assert common_mod.hmac_sha256_hex("k", "m") == common_mod.hmac_sha256_hex("k", "m")
        assert common_mod.hmac_sha1_base64("k", "m") == common_mod.hmac_sha1_base64("k", "m")
        assert common_mod.constant_time_equals("a", "a")
        assert not common_mod.constant_time_equals("a", "b")

    def test_build_http_client(self) -> None:
        client = common_mod.build_http_client()
        try:
            assert client is not None
        finally:
            client.close()
        transport = httpx.MockTransport(lambda req: httpx.Response(200))
        client = common_mod.build_http_client(transport=transport)
        try:
            assert client.get("https://example.com").status_code == 200
        finally:
            client.close()

    def test_validate_http_endpoint(self) -> None:
        common_mod.validate_http_endpoint("https://mm.example.com/api/v4")
        with pytest.raises(ValidationError):
            common_mod.validate_http_endpoint("ftp://mm.example.com")
        with pytest.raises(ValidationError):
            common_mod.validate_http_endpoint("https://user@mm.example.com")
        with pytest.raises(ValidationError):
            common_mod.validate_http_endpoint("https://")
        with pytest.raises(ValidationError):
            common_mod.validate_http_endpoint("https://localhost/x")
        with pytest.raises(ValidationError):
            common_mod.validate_http_endpoint("https://127.0.0.1/x")

    def test_validate_https_host_suffix(self) -> None:
        common_mod.validate_https_host_suffix("https://www.yunzhijia.com/send", "yunzhijia.com")
        with pytest.raises(ValidationError):
            common_mod.validate_https_host_suffix("http://www.yunzhijia.com/send", "yunzhijia.com")
        with pytest.raises(ValidationError):
            common_mod.validate_https_host_suffix("https://www.yunzhijia.com/send", "")
        with pytest.raises(ValidationError):
            common_mod.validate_https_host_suffix("https://evil.example.com/send", "yunzhijia.com")

    def test_http_ok_assertions(self) -> None:
        common_mod.assert_http_ok(httpx.Response(204), platform="p", action="a")
        with pytest.raises(ExternalServiceError):
            common_mod.assert_http_ok(httpx.Response(500), platform="p", action="a")
        common_mod.assert_http_ok_strict(httpx.Response(200), platform="p", action="a")
        with pytest.raises(ExternalServiceError):
            common_mod.assert_http_ok_strict(httpx.Response(201), platform="p", action="a")

    def test_send_error_raises_external_service_error(self) -> None:
        with pytest.raises(ExternalServiceError) as exc:
            common_mod.send_error("slack", "chat.postMessage", "boom")
        assert exc.value.code == "im.send_failed"
