"""Unit tests for the Feishu/Lark and WeCom IM adapters.

AAA-pattern tests covering the adapter contract: platform
identification, URL verification, callback parsing, send-message
building, and error paths. HTTP is mocked with respx; callback
encryption fixtures are built with the same AES primitives the adapters
decrypt so the round-trip is real.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
import respx
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from src.common.exception import ExternalServiceError, UnauthorizedError
from src.common.json import JsonObject
from src.core.channels.im.adapter_base import (
    CHAT_TYPE_DIRECT,
    CHAT_TYPE_GROUP,
    MESSAGE_TYPE_FILE,
    MESSAGE_TYPE_IMAGE,
    MESSAGE_TYPE_TEXT,
    CallbackRequest,
    EventContext,
    IncomingMessage,
    ReplyMessage,
)
from src.core.channels.im.adapters import (
    FeishuAdapter,
    WecomAdapter,
    build_feishu_adapter,
    build_wecom_adapter,
    register_im_adapters,
)
from src.core.channels.im.supervisor import IMSupervisor, get_default_supervisor
from src.db.models.im_channel import IMChannel
from tests.util.service_test import ServiceTest

_NOW = datetime(2026, 1, 1, tzinfo=UTC)

# ── Fixture credentials ───────────────────────────────────────────────

_APP_ID = "cli_test"
_APP_SECRET = "app-secret"
_VERIFICATION_TOKEN = "token-123"
_ENCRYPT_KEY = "encrypt-key"

_CORP_ID = "ww1234567890"
_AGENT_SECRET = "agent-secret"
_WECOM_TOKEN = "wecom-token"
# 43-char base64 (no padding) → 32 bytes after decode: a valid AES key.
_ENCODING_AES_KEY = "A" * 43
_CORP_AGENT_ID = 1000002

# ── API endpoints ─────────────────────────────────────────────────────

_FEISHU_BASE = "https://open.feishu.cn"
_FEISHU_TOKEN_URL = f"{_FEISHU_BASE}/open-apis/auth/v3/tenant_access_token/internal"
_FEISHU_REPLY_URL = f"{_FEISHU_BASE}/open-apis/im/v1/messages/om_message_1/reply"
_FEISHU_SEND_URL = f"{_FEISHU_BASE}/open-apis/im/v1/messages"

_WECOM_BASE = "https://qyapi.weixin.qq.com"
_WECOM_TOKEN_URL = f"{_WECOM_BASE}/cgi-bin/gettoken"
_WECOM_APPCHAT_URL = f"{_WECOM_BASE}/cgi-bin/appchat/send"
_WECOM_MESSAGE_URL = f"{_WECOM_BASE}/cgi-bin/message/send"

# ── Crypto fixtures ───────────────────────────────────────────────────


def _pkcs7_pad(data: bytes) -> bytes:
    pad_len = 16 - (len(data) % 16)
    return data + bytes([pad_len]) * pad_len


def _feishu_encrypt(plaintext: str) -> str:
    """Encrypt ``plaintext`` the way the platform encrypts event bodies."""
    key = hashlib.sha256(_ENCRYPT_KEY.encode("utf-8")).digest()
    iv = os.urandom(16)
    encryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    ciphertext = encryptor.update(_pkcs7_pad(plaintext.encode("utf-8"))) + encryptor.finalize()
    return base64.b64encode(iv + ciphertext).decode("ascii")


def _feishu_encrypted_request(plaintext: str) -> CallbackRequest:
    return CallbackRequest(body=json.dumps({"encrypt": _feishu_encrypt(plaintext)}))


def _wecom_encrypt(xml: str) -> str:
    """Encrypt a WeCom XML message: ``random(16) + len(4) + xml + corp_id``."""
    aes_key = base64.b64decode(_ENCODING_AES_KEY + "=")
    body = xml.encode("utf-8")
    plaintext = os.urandom(16) + len(body).to_bytes(4, "big") + body + _CORP_ID.encode("utf-8")
    encryptor = Cipher(algorithms.AES(aes_key), modes.CBC(aes_key[:16])).encryptor()
    ciphertext = encryptor.update(_pkcs7_pad(plaintext)) + encryptor.finalize()
    return base64.b64encode(ciphertext).decode("ascii")


def _wecom_signature(timestamp: str, nonce: str, encrypt: str) -> str:
    parts = sorted([_WECOM_TOKEN, timestamp, nonce, encrypt])
    return hashlib.sha1("".join(parts).encode("utf-8")).hexdigest()


# ── Request / event fixtures ──────────────────────────────────────────


def _feishu_message_event(
    *,
    event_type: str = "im.message.receive_v1",
    message_type: str = "text",
    content: str = '{"text":"hello"}',
    chat_type: str = "p2p",
    chat_id: str = "",
    message_id: str = "om_message_1",
    root_id: str = "",
    open_id: str = "ou_user_1",
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "sender": {"sender_id": {"open_id": open_id}},
        "message": {
            "message_id": message_id,
            "root_id": root_id,
            "parent_id": "",
            "message_type": message_type,
            "chat_type": chat_type,
            "chat_id": chat_id,
            "content": content,
        },
    }
    return {
        "schema": "2.0",
        "header": {
            "event_id": "evt_1",
            "event_type": event_type,
            "token": _VERIFICATION_TOKEN,
        },
        "event": event,
    }


def _wecom_xml_message(
    *,
    msg_type: str = "text",
    content: str = "",
    from_user: str = "user_1",
    msg_id: str = "msg_1",
    chat_id: str = "",
    pic_url: str = "",
    media_id: str = "",
) -> str:
    parts = [
        "<xml>",
        f"<ToUserName>{_CORP_ID}</ToUserName>",
        f"<FromUserName>{from_user}</FromUserName>",
        "<CreateTime>1700000000</CreateTime>",
        f"<MsgType>{msg_type}</MsgType>",
    ]
    if msg_type == "text":
        parts.append(f"<Content>{content}</Content>")
    if msg_type == "image":
        if pic_url:
            parts.append(f"<PicUrl>{pic_url}</PicUrl>")
        if media_id:
            parts.append(f"<MediaId>{media_id}</MediaId>")
    parts.append(f"<MsgId>{msg_id}</MsgId>")
    if chat_id:
        parts.append(f"<ChatId>{chat_id}</ChatId>")
    parts.append("</xml>")
    return "".join(parts)


def _wecom_callback_request(xml: str) -> CallbackRequest:
    encrypt = _wecom_encrypt(xml)
    body = f"<xml><ToUserName>{_CORP_ID}</ToUserName><Encrypt>{encrypt}</Encrypt></xml>"
    return CallbackRequest(body=body)


def _make_channel(*, platform: str, credentials: JsonObject) -> IMChannel:
    return IMChannel(
        id="ch-1",
        tenant_id=1,
        agent_id="agent-1",
        platform=platform,
        name="channel",
        enabled=True,
        mode="webhook",
        output_mode="stream",
        knowledge_base_id="",
        bot_identity="",
        session_mode="user",
        credentials=credentials,
        created_at=_NOW,
        updated_at=_NOW,
    )


# ── Adapter fixtures ──────────────────────────────────────────────────


def _feishu_adapter(
    *,
    platform: str = "feishu",
    verification_token: str = _VERIFICATION_TOKEN,
    encrypt_key: str = _ENCRYPT_KEY,
) -> FeishuAdapter:
    return FeishuAdapter(
        platform=platform,
        app_id=_APP_ID,
        app_secret=_APP_SECRET,
        verification_token=verification_token,
        encrypt_key=encrypt_key,
    )


def _wecom_adapter() -> WecomAdapter:
    return WecomAdapter(
        corp_id=_CORP_ID,
        agent_secret=_AGENT_SECRET,
        token=_WECOM_TOKEN,
        encoding_aes_key=_ENCODING_AES_KEY,
        corp_agent_id=_CORP_AGENT_ID,
    )


def _feishu_token_response() -> dict[str, Any]:
    return {"code": 0, "msg": "ok", "tenant_access_token": "t-1", "expire": 7200}


def _wecom_token_response() -> dict[str, Any]:
    return {"errcode": 0, "errmsg": "ok", "access_token": "tk-1", "expires_in": 7200}


# ── Feishu: identity ──────────────────────────────────────────────────


class TestFeishuIdentity(ServiceTest):
    def test_platform_identifies_feishu(self) -> None:
        adapter = _feishu_adapter()
        assert adapter.platform() == "feishu"

    def test_lark_platform_reports_lark(self) -> None:
        adapter = _feishu_adapter(platform="lark")
        assert adapter.platform() == "lark"

    def test_rejects_unknown_platform(self) -> None:
        with pytest.raises(Exception) as excinfo:
            FeishuAdapter(platform="other", app_id="a", app_secret="s")
        assert excinfo.type.__name__ == "ValidationError"

    async def test_connect_returns_noop_stop(self) -> None:
        adapter = _feishu_adapter()
        stop = await adapter.connect(EventContext())
        stop()  # must not raise

    def test_disconnect_is_safe_without_client(self) -> None:
        adapter = _feishu_adapter()
        adapter.disconnect()  # no-op when no client was created


# ── Feishu: callback verification ────────────────────────────────────


class TestFeishuVerify(ServiceTest):
    def test_passes_when_no_verification_token_configured(self) -> None:
        adapter = _feishu_adapter(verification_token="")
        adapter.verify_callback(CallbackRequest(body="not-json"))

    def test_passes_valid_header_token(self) -> None:
        adapter = _feishu_adapter()
        body = json.dumps(
            {"header": {"token": _VERIFICATION_TOKEN, "event_type": "im.message.receive_v1"}}
        )
        adapter.verify_callback(CallbackRequest(body=body))

    def test_rejects_wrong_token(self) -> None:
        adapter = _feishu_adapter()
        body = json.dumps({"header": {"token": "wrong"}})
        with pytest.raises(UnauthorizedError) as excinfo:
            adapter.verify_callback(CallbackRequest(body=body))
        assert excinfo.value.code == "im.feishu_verify_failed"

    def test_rejects_non_json_body(self) -> None:
        adapter = _feishu_adapter()
        with pytest.raises(UnauthorizedError) as excinfo:
            adapter.verify_callback(CallbackRequest(body="not-json"))
        assert excinfo.value.code == "im.feishu_verify_failed"

    def test_accepts_encrypted_body_with_valid_token(self) -> None:
        adapter = _feishu_adapter()
        plaintext = json.dumps(
            {"header": {"token": _VERIFICATION_TOKEN, "event_type": "im.message.receive_v1"}}
        )
        adapter.verify_callback(_feishu_encrypted_request(plaintext))

    def test_encrypted_body_with_wrong_token_raises(self) -> None:
        adapter = _feishu_adapter()
        plaintext = json.dumps({"header": {"token": "wrong"}})
        with pytest.raises(UnauthorizedError) as excinfo:
            adapter.verify_callback(_feishu_encrypted_request(plaintext))
        assert excinfo.value.code == "im.feishu_verify_failed"


# ── Feishu: URL verification ─────────────────────────────────────────


class TestFeishuUrlVerification(ServiceTest):
    def test_answers_challenge(self) -> None:
        adapter = _feishu_adapter()
        body = json.dumps(
            {"challenge": "challenge-abc", "token": _VERIFICATION_TOKEN, "type": "url_verification"}
        )
        assert adapter.handle_url_verification(CallbackRequest(body=body)) is True
        response = adapter.verification_body()
        assert response is not None
        assert json.loads(response) == {"challenge": "challenge-abc"}

    def test_answers_encrypted_challenge(self) -> None:
        adapter = _feishu_adapter()
        plaintext = json.dumps({"challenge": "c-enc"})
        assert adapter.handle_url_verification(_feishu_encrypted_request(plaintext)) is True
        response = adapter.verification_body()
        assert response is not None
        assert json.loads(response) == {"challenge": "c-enc"}

    def test_ignores_message_event(self) -> None:
        adapter = _feishu_adapter()
        body = json.dumps(_feishu_message_event())
        assert adapter.handle_url_verification(CallbackRequest(body=body)) is False
        assert adapter.verification_body() is None

    def test_ignores_invalid_json(self) -> None:
        adapter = _feishu_adapter()
        assert adapter.handle_url_verification(CallbackRequest(body="not-json")) is False


# ── Feishu: callback parsing ─────────────────────────────────────────


class TestFeishuParse(ServiceTest):
    def test_parses_text_message(self) -> None:
        adapter = _feishu_adapter()
        event = _feishu_message_event(content='{"text":"hello world"}')
        msg = adapter.parse_callback(CallbackRequest(body=json.dumps(event)))
        assert msg is not None
        assert msg.platform == "feishu"
        assert msg.message_type == MESSAGE_TYPE_TEXT
        assert msg.content == "hello world"
        assert msg.user_id == "ou_user_1"
        assert msg.chat_type == CHAT_TYPE_DIRECT
        assert msg.message_id == "om_message_1"
        assert msg.thread_id == "om_message_1"

    def test_parses_group_text_and_strips_mention(self) -> None:
        adapter = _feishu_adapter()
        event = _feishu_message_event(
            content='{"text":"@_user_1 你好 世界"}',
            chat_type="group",
            chat_id="oc_group_1",
            root_id="om_thread_1",
        )
        msg = adapter.parse_callback(CallbackRequest(body=json.dumps(event)))
        assert msg is not None
        assert msg.chat_type == CHAT_TYPE_GROUP
        assert msg.chat_id == "oc_group_1"
        assert msg.content == "你好 世界"
        assert msg.thread_id == "om_thread_1"

    def test_parses_file_message(self) -> None:
        adapter = _feishu_adapter()
        event = _feishu_message_event(
            content='{"file_key":"file_k_1","file_name":"a.txt"}', message_type="file"
        )
        msg = adapter.parse_callback(CallbackRequest(body=json.dumps(event)))
        assert msg is not None
        assert msg.message_type == MESSAGE_TYPE_FILE
        assert msg.file_key == "file_k_1"
        assert msg.file_name == "a.txt"

    def test_parses_image_message(self) -> None:
        adapter = _feishu_adapter()
        event = _feishu_message_event(content='{"image_key":"img_k_1"}', message_type="image")
        msg = adapter.parse_callback(CallbackRequest(body=json.dumps(event)))
        assert msg is not None
        assert msg.message_type == MESSAGE_TYPE_IMAGE
        assert msg.file_key == "img_k_1"
        assert msg.file_name == "img_k_1.png"

    def test_parses_post_message(self) -> None:
        adapter = _feishu_adapter()
        content = json.dumps(
            {
                "title": "标题",
                "content": [
                    [{"tag": "text", "text": "第一行"}],
                    [{"tag": "a", "text": "链接"}],
                    [{"tag": "at", "text": "忽略"}],
                ],
            }
        )
        event = _feishu_message_event(content=content, message_type="post")
        msg = adapter.parse_callback(CallbackRequest(body=json.dumps(event)))
        assert msg is not None
        assert msg.message_type == MESSAGE_TYPE_TEXT
        assert msg.content == "标题\n第一行\n链接"

    def test_parses_encrypted_message(self) -> None:
        adapter = _feishu_adapter()
        plaintext = json.dumps(_feishu_message_event(content='{"text":"secret"}'))
        msg = adapter.parse_callback(_feishu_encrypted_request(plaintext))
        assert msg is not None
        assert msg.content == "secret"

    def test_ignores_other_event_types(self) -> None:
        adapter = _feishu_adapter()
        event = _feishu_message_event(event_type="im.chat.update")
        msg = adapter.parse_callback(CallbackRequest(body=json.dumps(event)))
        assert msg is None

    def test_ignores_event_without_message(self) -> None:
        adapter = _feishu_adapter()
        event = _feishu_message_event()
        del event["event"]["message"]
        msg = adapter.parse_callback(CallbackRequest(body=json.dumps(event)))
        assert msg is None

    def test_ignores_unsupported_message_type(self) -> None:
        adapter = _feishu_adapter()
        event = _feishu_message_event(message_type="audio", content="{}")
        msg = adapter.parse_callback(CallbackRequest(body=json.dumps(event)))
        assert msg is None

    def test_invalid_json_raises_parse_error(self) -> None:
        adapter = _feishu_adapter()
        with pytest.raises(ExternalServiceError) as excinfo:
            adapter.parse_callback(CallbackRequest(body="not-json"))
        assert excinfo.value.code == "im.feishu_parse_failed"

    def test_invalid_text_content_raises(self) -> None:
        adapter = _feishu_adapter()
        event = _feishu_message_event(message_type="text", content="not-json")
        with pytest.raises(ExternalServiceError) as excinfo:
            adapter.parse_callback(CallbackRequest(body=json.dumps(event)))
        assert excinfo.value.code == "im.feishu_parse_failed"


# ── Feishu: reply sending ────────────────────────────────────────────


class TestFeishuSend(ServiceTest):
    def test_direct_reply_uses_reply_api(self) -> None:
        adapter = _feishu_adapter()
        incoming = IncomingMessage(
            platform="feishu",
            user_id="ou_user_1",
            chat_type=CHAT_TYPE_DIRECT,
            message_id="om_message_1",
        )
        with respx.mock(base_url=_FEISHU_BASE) as router:
            router.post(_FEISHU_TOKEN_URL).respond(200, json=_feishu_token_response())
            reply_route = router.post(_FEISHU_REPLY_URL).respond(200, json={"code": 0, "msg": "ok"})

            adapter.send_reply(EventContext(), incoming, ReplyMessage(content="你好"))

        payload = json.loads(reply_route.calls.last.request.content)
        assert payload["msg_type"] == "text"
        assert json.loads(payload["content"])["text"] == "你好"

    def test_falls_back_when_reply_api_rejects(self) -> None:
        adapter = _feishu_adapter()
        incoming = IncomingMessage(
            platform="feishu",
            user_id="ou_user_1",
            chat_type=CHAT_TYPE_DIRECT,
            message_id="om_message_1",
        )
        with respx.mock(base_url=_FEISHU_BASE) as router:
            router.post(_FEISHU_TOKEN_URL).respond(200, json=_feishu_token_response())
            router.post(_FEISHU_REPLY_URL).respond(200, json={"code": 230071, "msg": "no thread"})
            send_route = router.post(_FEISHU_SEND_URL).respond(200, json={"code": 0, "msg": "ok"})

            adapter.send_reply(EventContext(), incoming, ReplyMessage(content="hi"))

        payload = json.loads(send_route.calls.last.request.content)
        assert payload["receive_id"] == "ou_user_1"

    def test_falls_back_on_transport_error(self) -> None:
        adapter = _feishu_adapter()
        incoming = IncomingMessage(
            platform="feishu",
            user_id="ou_user_1",
            chat_type=CHAT_TYPE_DIRECT,
            message_id="om_message_1",
        )
        with respx.mock(base_url=_FEISHU_BASE) as router:
            router.post(_FEISHU_TOKEN_URL).respond(200, json=_feishu_token_response())
            router.post(_FEISHU_REPLY_URL).mock(side_effect=httpx.ConnectError("boom"))
            send_route = router.post(_FEISHU_SEND_URL).respond(200, json={"code": 0, "msg": "ok"})

            adapter.send_reply(EventContext(), incoming, ReplyMessage(content="hi"))

        assert send_route.call_count == 1

    def test_group_fallback_uses_chat_id(self) -> None:
        adapter = _feishu_adapter()
        incoming = IncomingMessage(
            platform="feishu",
            user_id="ou_user_1",
            chat_id="oc_group_1",
            chat_type=CHAT_TYPE_GROUP,
            message_id="om_message_1",
        )
        with respx.mock(base_url=_FEISHU_BASE) as router:
            router.post(_FEISHU_TOKEN_URL).respond(200, json=_feishu_token_response())
            router.post(_FEISHU_REPLY_URL).respond(200, json={"code": 230054, "msg": "unsupported"})
            send_route = router.post(_FEISHU_SEND_URL).respond(200, json={"code": 0, "msg": "ok"})

            adapter.send_reply(EventContext(), incoming, ReplyMessage(content="hi"))

        request = send_route.calls.last.request
        assert "receive_id_type=chat_id" in str(request.url)
        payload = json.loads(request.content)
        assert payload["receive_id"] == "oc_group_1"

    def test_sends_via_send_api_without_message_id(self) -> None:
        adapter = _feishu_adapter()
        incoming = IncomingMessage(
            platform="feishu", user_id="ou_user_1", chat_type=CHAT_TYPE_DIRECT, message_id=""
        )
        with respx.mock(base_url=_FEISHU_BASE) as router:
            router.post(_FEISHU_TOKEN_URL).respond(200, json=_feishu_token_response())
            send_route = router.post(_FEISHU_SEND_URL).respond(200, json={"code": 0, "msg": "ok"})

            adapter.send_reply(EventContext(), incoming, ReplyMessage(content="hi"))

        assert send_route.call_count == 1

    def test_unsafe_message_id_raises(self) -> None:
        adapter = _feishu_adapter()
        incoming = IncomingMessage(
            platform="feishu",
            user_id="u",
            chat_type=CHAT_TYPE_DIRECT,
            message_id="../../etc/passwd",
        )
        with respx.mock(base_url=_FEISHU_BASE) as router:
            router.post(_FEISHU_TOKEN_URL).respond(200, json=_feishu_token_response())
            with pytest.raises(ExternalServiceError) as excinfo:
                adapter.send_reply(EventContext(), incoming, ReplyMessage(content="hi"))
        assert excinfo.value.code == "im.feishu_send_failed"

    def test_non_fallback_error_code_raises(self) -> None:
        adapter = _feishu_adapter()
        incoming = IncomingMessage(
            platform="feishu", user_id="u", chat_type=CHAT_TYPE_DIRECT, message_id="om_message_1"
        )
        with respx.mock(base_url=_FEISHU_BASE) as router:
            router.post(_FEISHU_TOKEN_URL).respond(200, json=_feishu_token_response())
            router.post(_FEISHU_REPLY_URL).respond(200, json={"code": 99999, "msg": "unknown"})
            with pytest.raises(ExternalServiceError) as excinfo:
                adapter.send_reply(EventContext(), incoming, ReplyMessage(content="hi"))
        assert excinfo.value.code == "im.feishu_send_failed"

    def test_send_api_error_raises(self) -> None:
        adapter = _feishu_adapter()
        incoming = IncomingMessage(
            platform="feishu", user_id="u", chat_type=CHAT_TYPE_DIRECT, message_id=""
        )
        with respx.mock(base_url=_FEISHU_BASE) as router:
            router.post(_FEISHU_TOKEN_URL).respond(200, json=_feishu_token_response())
            router.post(_FEISHU_SEND_URL).respond(200, json={"code": 190001, "msg": "bad"})
            with pytest.raises(ExternalServiceError) as excinfo:
                adapter.send_reply(EventContext(), incoming, ReplyMessage(content="hi"))
        assert excinfo.value.code == "im.feishu_send_failed"

    def test_access_token_is_cached(self) -> None:
        adapter = _feishu_adapter()
        incoming = IncomingMessage(
            platform="feishu", user_id="u", chat_type=CHAT_TYPE_DIRECT, message_id="om_message_1"
        )
        with respx.mock(base_url=_FEISHU_BASE) as router:
            token_route = router.post(_FEISHU_TOKEN_URL).respond(200, json=_feishu_token_response())
            router.post(_FEISHU_REPLY_URL).respond(200, json={"code": 0, "msg": "ok"})

            adapter.send_reply(EventContext(), incoming, ReplyMessage(content="hi"))
            adapter.send_reply(EventContext(), incoming, ReplyMessage(content="hi"))

        assert token_route.call_count == 1

    def test_access_token_failure_raises(self) -> None:
        adapter = _feishu_adapter()
        incoming = IncomingMessage(
            platform="feishu", user_id="u", chat_type=CHAT_TYPE_DIRECT, message_id="om_message_1"
        )
        with respx.mock(base_url=_FEISHU_BASE) as router:
            router.post(_FEISHU_TOKEN_URL).respond(200, json={"code": 10003, "msg": "bad app id"})
            with pytest.raises(ExternalServiceError) as excinfo:
                adapter.send_reply(EventContext(), incoming, ReplyMessage(content="hi"))
        assert excinfo.value.code == "im.feishu_token_failed"


# ── WeCom: identity ──────────────────────────────────────────────────


class TestWecomIdentity(ServiceTest):
    def test_platform_identifies_wecom(self) -> None:
        adapter = _wecom_adapter()
        assert adapter.platform() == "wecom"

    async def test_connect_returns_noop_stop(self) -> None:
        adapter = _wecom_adapter()
        stop = await adapter.connect(EventContext())
        stop()  # must not raise

    def test_disconnect_is_safe_without_client(self) -> None:
        adapter = _wecom_adapter()
        adapter.disconnect()


# ── WeCom: callback verification ─────────────────────────────────────


class TestWecomVerify(ServiceTest):
    def test_passes_valid_get_signature(self) -> None:
        adapter = _wecom_adapter()
        timestamp, nonce = "1700000000", "nonce-1"
        echostr = _wecom_encrypt("verify-me")
        signature = _wecom_signature(timestamp, nonce, echostr)
        query: JsonObject = {
            "timestamp": timestamp,
            "nonce": nonce,
            "msg_signature": signature,
            "echostr": echostr,
        }
        adapter.verify_callback(CallbackRequest(query=query))

    def test_rejects_bad_get_signature(self) -> None:
        adapter = _wecom_adapter()
        timestamp, nonce = "1700000000", "nonce-1"
        echostr = _wecom_encrypt("verify-me")
        query: JsonObject = {
            "timestamp": timestamp,
            "nonce": nonce,
            "msg_signature": "deadbeef",
            "echostr": echostr,
        }
        with pytest.raises(UnauthorizedError) as excinfo:
            adapter.verify_callback(CallbackRequest(query=query))
        assert excinfo.value.code == "im.wecom_verify_failed"

    def test_passes_valid_post_signature(self) -> None:
        adapter = _wecom_adapter()
        timestamp, nonce = "1700000000", "nonce-1"
        xml = _wecom_xml_message(content="hello")
        encrypt = _wecom_encrypt(xml)
        signature = _wecom_signature(timestamp, nonce, encrypt)
        body = f"<xml><ToUserName>{_CORP_ID}</ToUserName><Encrypt>{encrypt}</Encrypt></xml>"
        adapter.verify_callback(
            CallbackRequest(
                body=body,
                query={"timestamp": timestamp, "nonce": nonce, "msg_signature": signature},
            )
        )

    def test_rejects_bad_post_signature(self) -> None:
        adapter = _wecom_adapter()
        timestamp, nonce = "1700000000", "nonce-1"
        xml = _wecom_xml_message(content="hello")
        encrypt = _wecom_encrypt(xml)
        body = f"<xml><ToUserName>{_CORP_ID}</ToUserName><Encrypt>{encrypt}</Encrypt></xml>"
        with pytest.raises(UnauthorizedError) as excinfo:
            adapter.verify_callback(
                CallbackRequest(
                    body=body,
                    query={"timestamp": timestamp, "nonce": nonce, "msg_signature": "deadbeef"},
                )
            )
        assert excinfo.value.code == "im.wecom_verify_failed"


# ── WeCom: URL verification ──────────────────────────────────────────


class TestWecomUrlVerification(ServiceTest):
    def test_returns_decrypted_echostr(self) -> None:
        adapter = _wecom_adapter()
        echostr = _wecom_encrypt("verify-me")
        assert adapter.handle_url_verification(CallbackRequest(query={"echostr": echostr})) is True
        assert adapter.verification_body() == "verify-me"

    def test_ignores_request_without_echostr(self) -> None:
        adapter = _wecom_adapter()
        assert adapter.handle_url_verification(CallbackRequest(query={})) is False
        assert adapter.verification_body() is None

    def test_malformed_echostr_raises(self) -> None:
        adapter = _wecom_adapter()
        echostr = base64.b64encode(os.urandom(32)).decode("ascii")
        with pytest.raises(ExternalServiceError) as excinfo:
            adapter.handle_url_verification(CallbackRequest(query={"echostr": echostr}))
        assert excinfo.value.code == "im.wecom_decrypt_failed"


# ── WeCom: callback parsing ──────────────────────────────────────────


class TestWecomParse(ServiceTest):
    def test_parses_text_message(self) -> None:
        adapter = _wecom_adapter()
        xml = _wecom_xml_message(content="你好 world", from_user="user_1", msg_id="msg_1")
        msg = adapter.parse_callback(_wecom_callback_request(xml))
        assert msg is not None
        assert msg.platform == "wecom"
        assert msg.message_type == MESSAGE_TYPE_TEXT
        assert msg.content == "你好 world"
        assert msg.user_id == "user_1"
        assert msg.user_name == "user_1"
        assert msg.message_id == "msg_1"
        assert msg.chat_type == CHAT_TYPE_DIRECT

    def test_parses_group_text_and_strips_mention(self) -> None:
        adapter = _wecom_adapter()
        xml = _wecom_xml_message(content="@TestBot 你好", chat_id="wr_group_1")
        msg = adapter.parse_callback(_wecom_callback_request(xml))
        assert msg is not None
        assert msg.chat_type == CHAT_TYPE_GROUP
        assert msg.chat_id == "wr_group_1"
        assert msg.content == "你好"

    def test_parses_image_with_pic_url(self) -> None:
        adapter = _wecom_adapter()
        xml = _wecom_xml_message(
            msg_type="image",
            pic_url="https://example.com/a.png",
            media_id="media_1",
            msg_id="msg_2",
        )
        msg = adapter.parse_callback(_wecom_callback_request(xml))
        assert msg is not None
        assert msg.message_type == MESSAGE_TYPE_IMAGE
        assert msg.file_key == "https://example.com/a.png"
        assert msg.file_name == "msg_2.png"

    def test_ignores_unsupported_message_type(self) -> None:
        adapter = _wecom_adapter()
        xml = _wecom_xml_message(msg_type="video")
        msg = adapter.parse_callback(_wecom_callback_request(xml))
        assert msg is None

    def test_malformed_envelope_raises_decrypt_error(self) -> None:
        adapter = _wecom_adapter()
        with pytest.raises(ExternalServiceError) as excinfo:
            adapter.parse_callback(CallbackRequest(body="not-xml"))
        assert excinfo.value.code == "im.wecom_decrypt_failed"


# ── WeCom: reply sending ─────────────────────────────────────────────


class TestWecomSend(ServiceTest):
    def test_direct_reply_uses_message_send_api(self) -> None:
        adapter = _wecom_adapter()
        incoming = IncomingMessage(
            platform="wecom", user_id="user_1", chat_type=CHAT_TYPE_DIRECT, message_id="msg_1"
        )
        with respx.mock(base_url=_WECOM_BASE) as router:
            router.get(_WECOM_TOKEN_URL).respond(200, json=_wecom_token_response())
            send_route = router.post(_WECOM_MESSAGE_URL).respond(
                200, json={"errcode": 0, "errmsg": "ok"}
            )

            adapter.send_reply(EventContext(), incoming, ReplyMessage(content="你好"))

        payload = json.loads(send_route.calls.last.request.content)
        assert payload["touser"] == "user_1"
        assert payload["msgtype"] == "markdown"
        assert payload["agentid"] == _CORP_AGENT_ID
        assert payload["markdown"]["content"] == "你好"

    def test_group_reply_uses_appchat_api(self) -> None:
        adapter = _wecom_adapter()
        incoming = IncomingMessage(
            platform="wecom",
            user_id="user_1",
            chat_id="wr_group_1",
            chat_type=CHAT_TYPE_GROUP,
            message_id="msg_1",
        )
        with respx.mock(base_url=_WECOM_BASE, assert_all_called=False) as router:
            router.get(_WECOM_TOKEN_URL).respond(200, json=_wecom_token_response())
            appchat_route = router.post(_WECOM_APPCHAT_URL).respond(
                200, json={"errcode": 0, "errmsg": "ok"}
            )
            message_route = router.post(_WECOM_MESSAGE_URL).respond(
                200, json={"errcode": 0, "errmsg": "ok"}
            )

            adapter.send_reply(EventContext(), incoming, ReplyMessage(content="hi"))

        payload = json.loads(appchat_route.calls.last.request.content)
        assert payload["chatid"] == "wr_group_1"
        assert payload["msgtype"] == "markdown"
        assert message_route.call_count == 0

    def test_group_reply_falls_back_to_user(self) -> None:
        adapter = _wecom_adapter()
        incoming = IncomingMessage(
            platform="wecom",
            user_id="user_1",
            chat_id="wr_group_1",
            chat_type=CHAT_TYPE_GROUP,
            message_id="msg_1",
        )
        with respx.mock(base_url=_WECOM_BASE) as router:
            router.get(_WECOM_TOKEN_URL).respond(200, json=_wecom_token_response())
            appchat_route = router.post(_WECOM_APPCHAT_URL).respond(
                200, json={"errcode": 61012, "errmsg": "not found"}
            )
            message_route = router.post(_WECOM_MESSAGE_URL).respond(
                200, json={"errcode": 0, "errmsg": "ok"}
            )

            adapter.send_reply(EventContext(), incoming, ReplyMessage(content="hi"))

        assert appchat_route.call_count == 1
        assert message_route.call_count == 1
        payload = json.loads(message_route.calls.last.request.content)
        assert payload["touser"] == "user_1"

    def test_send_api_error_raises(self) -> None:
        adapter = _wecom_adapter()
        incoming = IncomingMessage(
            platform="wecom", user_id="user_1", chat_type=CHAT_TYPE_DIRECT, message_id="msg_1"
        )
        with respx.mock(base_url=_WECOM_BASE) as router:
            router.get(_WECOM_TOKEN_URL).respond(200, json=_wecom_token_response())
            router.post(_WECOM_MESSAGE_URL).respond(
                200, json={"errcode": 40001, "errmsg": "invalid token"}
            )
            with pytest.raises(ExternalServiceError) as excinfo:
                adapter.send_reply(EventContext(), incoming, ReplyMessage(content="hi"))
        assert excinfo.value.code == "im.wecom_send_failed"

    def test_access_token_is_cached(self) -> None:
        adapter = _wecom_adapter()
        incoming = IncomingMessage(
            platform="wecom", user_id="user_1", chat_type=CHAT_TYPE_DIRECT, message_id="msg_1"
        )
        with respx.mock(base_url=_WECOM_BASE) as router:
            token_route = router.get(_WECOM_TOKEN_URL).respond(200, json=_wecom_token_response())
            router.post(_WECOM_MESSAGE_URL).respond(200, json={"errcode": 0, "errmsg": "ok"})

            adapter.send_reply(EventContext(), incoming, ReplyMessage(content="hi"))
            adapter.send_reply(EventContext(), incoming, ReplyMessage(content="hi"))

        assert token_route.call_count == 1

    def test_access_token_failure_raises(self) -> None:
        adapter = _wecom_adapter()
        incoming = IncomingMessage(
            platform="wecom", user_id="user_1", chat_type=CHAT_TYPE_DIRECT, message_id="msg_1"
        )
        with respx.mock(base_url=_WECOM_BASE) as router:
            router.get(_WECOM_TOKEN_URL).respond(
                200, json={"errcode": 40013, "errmsg": "invalid corpid"}
            )
            with pytest.raises(ExternalServiceError) as excinfo:
                adapter.send_reply(EventContext(), incoming, ReplyMessage(content="hi"))
        assert excinfo.value.code == "im.wecom_token_failed"


# ── Factories ────────────────────────────────────────────────────────


class TestAdapterFactories(ServiceTest):
    def test_feishu_factory_reads_credentials(self) -> None:
        channel = _make_channel(
            platform="feishu",
            credentials={
                "app_id": "cli_x",
                "app_secret": "s",
                "verification_token": _VERIFICATION_TOKEN,
            },
        )
        adapter = build_feishu_adapter(channel)
        assert isinstance(adapter, FeishuAdapter)
        assert adapter.platform() == "feishu"
        # The factory wired the verification token: a valid-token callback passes.
        body = json.dumps({"header": {"token": _VERIFICATION_TOKEN}})
        adapter.verify_callback(CallbackRequest(body=body))

    def test_lark_factory_picks_lark_platform(self) -> None:
        channel = _make_channel(platform="lark", credentials={"app_id": "cli_l", "app_secret": "s"})
        adapter = build_feishu_adapter(channel)
        assert isinstance(adapter, FeishuAdapter)
        assert adapter.platform() == "lark"

    def test_feishu_factory_uses_app_credentials_for_token(self) -> None:
        channel = _make_channel(
            platform="feishu", credentials={"app_id": "cli_x", "app_secret": "s"}
        )
        adapter = build_feishu_adapter(channel)
        incoming = IncomingMessage(
            platform="feishu", user_id="u", chat_type=CHAT_TYPE_DIRECT, message_id="om_message_1"
        )
        with respx.mock(base_url=_FEISHU_BASE) as router:
            token_route = router.post(_FEISHU_TOKEN_URL).respond(200, json=_feishu_token_response())
            router.post(_FEISHU_REPLY_URL).respond(200, json={"code": 0, "msg": "ok"})
            adapter.send_reply(EventContext(), incoming, ReplyMessage(content="hi"))
        payload = json.loads(token_route.calls.last.request.content)
        assert payload["app_id"] == "cli_x"
        assert payload["app_secret"] == "s"

    def test_wecom_factory_reads_credentials(self) -> None:
        channel = _make_channel(
            platform="wecom",
            credentials={
                "corp_id": _CORP_ID,
                "agent_secret": _AGENT_SECRET,
                "token": _WECOM_TOKEN,
                "encoding_aes_key": _ENCODING_AES_KEY,
                "corp_agent_id": _CORP_AGENT_ID,
            },
        )
        adapter = build_wecom_adapter(channel)
        assert isinstance(adapter, WecomAdapter)
        assert adapter.platform() == "wecom"
        # The factory wired the token: a valid signature passes.
        timestamp, nonce = "1700000000", "nonce-1"
        echostr = _wecom_encrypt("verify-me")
        signature = _wecom_signature(timestamp, nonce, echostr)
        adapter.verify_callback(
            CallbackRequest(
                query={
                    "timestamp": timestamp,
                    "nonce": nonce,
                    "msg_signature": signature,
                    "echostr": echostr,
                }
            )
        )

    def test_register_im_adapters_registers_all_aliases(self) -> None:
        supervisor = IMSupervisor()
        register_im_adapters(supervisor)
        assert supervisor.registered_platforms() == ["feishu", "lark", "wecom", "wxwork"]

    def test_default_supervisor_has_adapters_wired(self) -> None:
        # Importing the adapters package wires the process-wide default
        # supervisor, so ``start_channel`` can resolve the platforms.
        supervisor = get_default_supervisor()
        assert "feishu" in supervisor.registered_platforms()
        assert "wecom" in supervisor.registered_platforms()

    def test_registered_factories_build_adapters(self) -> None:
        supervisor = IMSupervisor()
        register_im_adapters(supervisor)
        feishu_adapter = supervisor.get_adapter_factory("feishu")(
            _make_channel(platform="feishu", credentials={"app_id": "cli_x"})
        )
        assert isinstance(feishu_adapter, FeishuAdapter)
        wecom_adapter = supervisor.get_adapter_factory("wecom")(
            _make_channel(
                platform="wecom",
                credentials={
                    "corp_id": _CORP_ID,
                    "agent_secret": _AGENT_SECRET,
                    "token": _WECOM_TOKEN,
                    "encoding_aes_key": _ENCODING_AES_KEY,
                },
            )
        )
        assert isinstance(wecom_adapter, WecomAdapter)


__all__ = []
