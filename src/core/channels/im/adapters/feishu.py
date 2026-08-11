"""Feishu / Lark IM adapter — message send, event callbacks, URL verification.

Feishu and Lark are the same product deployed on two isolated clouds
(``open.feishu.cn`` / ``open.larksuite.com``) sharing one API surface, so
a single implementation serves both. The ``platform`` constructor
argument selects the cloud and the platform identifier reported on
parsed messages.

Callback flow:

1. The platform calls the configured event-subscription URL with a
   message event, optionally AES-encrypted when an ``encrypt_key`` is
   configured (AES-256-CBC keyed by the SHA-256 of the encrypt key).
2. ``handle_url_verification`` answers the initial challenge before any
   signature check; ``verify_callback`` then checks the header token;
   ``parse_callback`` turns the event into a unified ``IncomingMessage``.
3. ``send_reply`` delivers the answer through the reply-message API
   (``POST /im/v1/messages/:message_id/reply``) and falls back to the
   plain send-message API when the platform refuses a reply-in-thread.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Final

import httpx

from src.common.exception import ExternalServiceError, UnauthorizedError, ValidationError
from src.common.json import JsonObject
from src.core.channels.im.adapter_base import (
    CHAT_TYPE_DIRECT,
    CHAT_TYPE_GROUP,
    MESSAGE_TYPE_FILE,
    MESSAGE_TYPE_IMAGE,
    MESSAGE_TYPE_TEXT,
    CallbackRequest,
    Context,
    IMAdapter,
    IncomingMessage,
    ReplyMessage,
)
from src.core.channels.im.adapters._common import (
    aes_cbc_decrypt,
    credential_string,
    json_int,
    noop_stop,
)
from src.db.models.im_channel import IMChannel

logger = logging.getLogger("src.core.channels.im.adapters.feishu")

#: Open Platform API origin for each cloud (no trailing slash).
_FEISHU_BASE_URL: Final[str] = "https://open.feishu.cn"
_LARK_BASE_URL: Final[str] = "https://open.larksuite.com"
_DEFAULT_BASE_URLS: Final[dict[str, str]] = {
    "feishu": _FEISHU_BASE_URL,
    "lark": _LARK_BASE_URL,
}

#: The message-receive event the adapter parses.
_MESSAGE_RECEIVE_EVENT: Final[str] = "im.message.receive_v1"

#: Feishu API error codes for which the reply-message API cannot work
#: (group does not support reply-in-thread, topic deleted, unsupported
#: message type) — the send falls back to the plain send-message API.
_FALLBACK_ELIGIBLE_ERROR_CODES: Final[frozenset[int]] = frozenset({230019, 230054, 230071})

#: Per-call HTTP timeout (seconds), matching the upstream client.
_DEFAULT_TIMEOUT_SECONDS: Final[float] = 10.0

#: Token safety margin (seconds). Feishu tenant tokens live 2 hours;
#: caching 5 minutes short avoids sending an about-to-expire token.
_TOKEN_SAFETY_MARGIN_SECONDS: Final[int] = 300

#: Default expiry when the token response omits ``expire``.
_DEFAULT_TOKEN_TTL_SECONDS: Final[int] = 7200


class FeishuAdapter(IMAdapter):
    """Feishu / Lark adapter: message send, event callbacks, URL verification."""

    def __init__(
        self,
        *,
        platform: str = "feishu",
        app_id: str,
        app_secret: str,
        verification_token: str = "",
        encrypt_key: str = "",
        api_base_url: str = "",
        http_client: httpx.Client | None = None,
    ) -> None:
        normalised = platform.strip().lower() or "feishu"
        if normalised not in _DEFAULT_BASE_URLS:
            raise ValidationError(
                code="im.platform_unsupported",
                message="platform must be one of: feishu, lark",
            )
        self._platform = normalised
        self._app_id = app_id
        self._app_secret = app_secret
        self._verification_token = verification_token
        self._encrypt_key = encrypt_key
        self._api_base_url = (api_base_url or _DEFAULT_BASE_URLS[normalised]).rstrip("/")
        self._http_client = http_client
        self._owns_client = http_client is None
        self._client: httpx.Client | None = None
        self._verification_body: str | None = None
        self._token_lock = threading.Lock()
        self._token_cache = ""
        self._token_expires_monotonic = 0.0

    # ── Identity ────────────────────────────────────────────────────

    def platform(self) -> str:
        """Return the platform identifier (``feishu`` or ``lark``)."""
        return self._platform

    # ── Callback verification ───────────────────────────────────────

    def verify_callback(self, request: CallbackRequest) -> None:
        """Verify the event header token.

        Returns ``None`` when verification passes; raises
        ``UnauthorizedError`` otherwise. When no verification token is
        configured (e.g. WebSocket mode) verification is skipped.
        """
        if not self._verification_token:
            return
        raw = self._decrypt_body(request.body)
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            raise UnauthorizedError(
                code="im.feishu_verify_failed",
                message="callback body is not valid JSON",
            ) from None
        if not isinstance(payload, dict):
            raise UnauthorizedError(
                code="im.feishu_verify_failed",
                message="callback body is not a JSON object",
            )
        header = payload.get("header")
        token = header.get("token", "") if isinstance(header, dict) else ""
        if token != self._verification_token:
            raise UnauthorizedError(
                code="im.feishu_verify_failed",
                message="invalid callback verification token",
            )

    def handle_url_verification(self, request: CallbackRequest) -> bool:
        """Answer the initial URL verification challenge.

        Returns ``True`` (and stores the response body for the web layer
        via :meth:`verification_body`) when ``request`` was a challenge;
        ``False`` otherwise so the caller keeps routing.
        """
        self._verification_body = None
        try:
            raw = self._decrypt_body(request.body)
        except ExternalServiceError:
            return False
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            return False
        if not isinstance(payload, dict):
            return False
        challenge = payload.get("challenge")
        if not isinstance(challenge, str) or not challenge:
            return False
        self._verification_body = json.dumps({"challenge": challenge}, ensure_ascii=False)
        return True

    def verification_body(self) -> str | None:
        """Return the body the web layer replies with after URL verification.

        ``None`` when no challenge was consumed.
        """
        return self._verification_body

    # ── Callback parsing ────────────────────────────────────────────

    def parse_callback(self, request: CallbackRequest) -> IncomingMessage | None:
        """Parse a message event into a unified ``IncomingMessage``.

        Returns ``None`` for non-message events (heartbeats, other event
        types, unsupported message types). A malformed body raises
        ``ExternalServiceError``.
        """
        raw = self._decrypt_body(request.body)
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            raise ExternalServiceError(
                code="im.feishu_parse_failed",
                message="callback body is not valid JSON",
            ) from None
        if not isinstance(payload, dict):
            return None
        header = payload.get("header")
        if not isinstance(header, dict) or header.get("event_type") != _MESSAGE_RECEIVE_EVENT:
            return None
        event = payload.get("event")
        message = event.get("message") if isinstance(event, dict) else None
        if not isinstance(message, dict):
            return None

        message_id = str(message.get("message_id", ""))
        thread_id = str(message.get("root_id", "") or message_id)
        is_group = message.get("chat_type") == "group"
        chat_type = CHAT_TYPE_GROUP if is_group else CHAT_TYPE_DIRECT
        chat_id = str(message.get("chat_id", "")) if is_group else ""
        open_id = _extract_sender_open_id(payload)
        message_type = str(message.get("message_type", ""))
        content = message.get("content")
        content_str = content if isinstance(content, str) else ""

        if message_type == "text":
            text = self._parse_text_content(content_str)
            if is_group:
                text = _strip_mention(text)
            return IncomingMessage(
                platform=self._platform,
                message_type=MESSAGE_TYPE_TEXT,
                user_id=open_id,
                chat_id=chat_id,
                chat_type=chat_type,
                content=text.strip(),
                message_id=message_id,
                thread_id=thread_id,
            )
        if message_type == "file":
            file_key, file_name = self._parse_file_content(content_str)
            if not file_key:
                return None
            return IncomingMessage(
                platform=self._platform,
                message_type=MESSAGE_TYPE_FILE,
                user_id=open_id,
                chat_id=chat_id,
                chat_type=chat_type,
                message_id=message_id,
                thread_id=thread_id,
                file_key=file_key,
                file_name=file_name,
            )
        if message_type == "image":
            image_key = self._parse_image_content(content_str)
            if not image_key:
                return None
            return IncomingMessage(
                platform=self._platform,
                message_type=MESSAGE_TYPE_IMAGE,
                user_id=open_id,
                chat_id=chat_id,
                chat_type=chat_type,
                message_id=message_id,
                thread_id=thread_id,
                file_key=image_key,
                file_name=f"{image_key}.png",
            )
        if message_type == "post":
            text = self._parse_post_content(content_str)
            if is_group:
                text = _strip_mention(text)
            text = text.strip()
            if not text:
                return None
            return IncomingMessage(
                platform=self._platform,
                message_type=MESSAGE_TYPE_TEXT,
                user_id=open_id,
                chat_id=chat_id,
                chat_type=chat_type,
                content=text,
                message_id=message_id,
                thread_id=thread_id,
            )
        return None

    # ── Reply sending ───────────────────────────────────────────────

    def send_reply(self, ctx: Context, incoming: IncomingMessage, reply: ReplyMessage) -> None:
        """Deliver ``reply`` to the originating conversation.

        Uses the reply-message API so the answer lands under the
        original message / thread. When the platform refuses a
        reply-in-thread (or the reply endpoint is unreachable) it retries
        once via the plain send-message API. Raises
        ``ExternalServiceError`` when delivery fails.
        """
        token = self.get_access_token()
        content = json.dumps({"text": reply.content}, ensure_ascii=False)
        reply_payload: JsonObject = {"msg_type": "text", "content": content}
        receive_id_type, receive_id = _resolve_receive_id(incoming)
        fallback_payload: JsonObject = {
            "receive_id": receive_id,
            "msg_type": "text",
            "content": content,
        }

        message_id = incoming.message_id
        if message_id and _is_safe_path_param(message_id):
            reply_url = f"{self._api_base_url}/open-apis/im/v1/messages/{message_id}/reply"
            code, api_msg = self._try_reply_api(token, reply_url, reply_payload)
            if code == 0:
                return
            if code is not None and code not in _FALLBACK_ELIGIBLE_ERROR_CODES:
                raise ExternalServiceError(
                    code="im.feishu_send_failed",
                    message=f"feishu reply api error: code={code} msg={api_msg}",
                )
        elif message_id:
            # Unsafe characters in a path position: refuse rather than
            # interpolate a possibly-tampered id into the URL.
            raise ExternalServiceError(
                code="im.feishu_send_failed",
                message=f"invalid message_id for reply api: {message_id!r}",
            )
        else:
            logger.warning(
                "[Feishu] incoming message has no message_id; replying via send-message api"
            )

        fallback_url = f"{self._api_base_url}/open-apis/im/v1/messages?receive_id_type={receive_id_type}"
        try:
            code, api_msg = self._post_feishu_message(token, fallback_url, fallback_payload)
        except httpx.HTTPError as exc:
            raise ExternalServiceError(
                code="im.feishu_send_failed",
                message=f"feishu send request failed: {exc}",
            ) from exc
        if code != 0:
            raise ExternalServiceError(
                code="im.feishu_send_failed",
                message=f"feishu send api error: code={code} msg={api_msg}",
            )

    # ── Connection lifecycle ────────────────────────────────────────

    async def connect(self, ctx: Context) -> Callable[[], None]:
        """Webhook mode holds no persistent connection; return a no-op stop."""
        return noop_stop

    def disconnect(self) -> None:
        """Close the adapter-owned HTTP client (no-op when injected)."""
        client = self._client
        self._client = None
        if client is not None and self._owns_client:
            client.close()

    # ── Access token ────────────────────────────────────────────────

    def get_access_token(self) -> str:
        """Return a cached tenant access token, fetching it on first use.

        Tokens are cached with a 5-minute safety margin so the process
        does not hammer the auth endpoint on every reply.
        """
        with self._token_lock:
            if self._token_cache and time.monotonic() < self._token_expires_monotonic:
                return self._token_cache

            url = f"{self._api_base_url}/open-apis/auth/v3/tenant_access_token/internal"
            client = self._get_client()
            try:
                resp = client.post(
                    url,
                    json={"app_id": self._app_id, "app_secret": self._app_secret},
                    headers={"Content-Type": "application/json; charset=utf-8"},
                )
            except httpx.HTTPError as exc:
                raise ExternalServiceError(
                    code="im.feishu_token_failed",
                    message=f"feishu token request failed: {exc}",
                ) from exc
            try:
                data = resp.json()
            except ValueError:
                raise ExternalServiceError(
                    code="im.feishu_token_failed",
                    message="non-JSON token response",
                ) from None
            if not isinstance(data, dict):
                raise ExternalServiceError(
                    code="im.feishu_token_failed",
                    message="feishu token response is not an object",
                )
            code = json_int(data.get("code", -1))
            if code != 0:
                raise ExternalServiceError(
                    code="im.feishu_token_failed",
                    message=f"feishu token error: code={code} msg={data.get('msg', '')}",
                )
            token = data.get("tenant_access_token")
            if not isinstance(token, str) or not token:
                raise ExternalServiceError(
                    code="im.feishu_token_failed",
                    message="empty tenant access token",
                )
            ttl = json_int(data.get("expire", _DEFAULT_TOKEN_TTL_SECONDS))
            if ttl > _TOKEN_SAFETY_MARGIN_SECONDS:
                ttl -= _TOKEN_SAFETY_MARGIN_SECONDS
            self._token_cache = token
            self._token_expires_monotonic = time.monotonic() + ttl
            return token

    # ── Internals ───────────────────────────────────────────────────

    def _get_client(self) -> httpx.Client:
        client = self._http_client or self._client
        if client is None:
            client = httpx.Client(timeout=_DEFAULT_TIMEOUT_SECONDS)
            self._client = client
        return client

    def _decrypt_body(self, body: str) -> bytes:
        """Return the decrypted event bytes, or the raw body when unencrypted."""
        raw = body.encode("utf-8") if body else b""
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            return raw
        if not isinstance(payload, dict):
            return raw
        encrypted = payload.get("encrypt")
        if not isinstance(encrypted, str) or not encrypted:
            return raw
        return self._decrypt(encrypted)

    def _decrypt(self, encrypted: str) -> bytes:
        """Decrypt a Feishu event body (AES-256-CBC, SHA-256 of the key)."""
        if not self._encrypt_key:
            raise ExternalServiceError(
                code="im.feishu_decrypt_failed",
                message="encrypt_key not configured",
            )
        try:
            ciphertext = base64.b64decode(encrypted)
        except ValueError:
            raise ExternalServiceError(
                code="im.feishu_decrypt_failed",
                message="base64 decode failed",
            ) from None
        if len(ciphertext) < 16:
            raise ExternalServiceError(
                code="im.feishu_decrypt_failed",
                message="ciphertext too short",
            )
        key = hashlib.sha256(self._encrypt_key.encode("utf-8")).digest()
        try:
            return aes_cbc_decrypt(key, ciphertext[:16], ciphertext[16:])
        except ValueError as exc:
            raise ExternalServiceError(
                code="im.feishu_decrypt_failed",
                message=f"aes decrypt failed: {exc}",
            ) from exc

    def _try_reply_api(self, token: str, url: str, payload: JsonObject) -> tuple[int | None, str]:
        """POST via the reply-message API; ``None`` code on transport error.

        The plain send-message API is a different endpoint that may
        succeed even when the reply endpoint is unreachable, so transport
        errors fall back instead of aborting.
        """
        try:
            return self._post_feishu_message(token, url, payload)
        except httpx.HTTPError as exc:
            logger.warning("[Feishu] reply API transport error (will try fallback): %s", exc)
            return None, str(exc)

    def _post_feishu_message(self, token: str, url: str, payload: JsonObject) -> tuple[int, str]:
        """POST a JSON payload to a Feishu IM message API.

        Returns ``(code, msg)`` from the response body. Raises
        ``httpx.HTTPError`` on transport failure and
        ``ExternalServiceError`` on a non-JSON / non-object body.
        """
        client = self._get_client()
        resp = client.post(
            url,
            json=payload,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {token}",
            },
        )
        try:
            data = resp.json()
        except ValueError:
            raise ExternalServiceError(
                code="im.feishu_send_failed",
                message=f"non-JSON response from feishu api (status={resp.status_code})",
            ) from None
        if not isinstance(data, dict):
            raise ExternalServiceError(
                code="im.feishu_send_failed",
                message="feishu api returned a non-object body",
            )
        return json_int(data.get("code", -1)), str(data.get("msg", ""))

    @staticmethod
    def _parse_text_content(content: str) -> str:
        payload = _parse_content_json(content, kind="text")
        text = payload.get("text")
        return text if isinstance(text, str) else ""

    @staticmethod
    def _parse_file_content(content: str) -> tuple[str, str]:
        payload = _parse_content_json(content, kind="file")
        file_key = payload.get("file_key")
        file_name = payload.get("file_name")
        return (
            file_key if isinstance(file_key, str) else "",
            file_name if isinstance(file_name, str) else "",
        )

    @staticmethod
    def _parse_image_content(content: str) -> str:
        payload = _parse_content_json(content, kind="image")
        image_key = payload.get("image_key")
        return image_key if isinstance(image_key, str) else ""

    @staticmethod
    def _parse_post_content(content: str) -> str:
        payload = _parse_content_json(content, kind="post")
        parts: list[str] = []
        title = payload.get("title")
        if isinstance(title, str) and title:
            parts.append(title)
        lines = payload.get("content")
        if isinstance(lines, list):
            for line in lines:
                line_parts: list[str] = []
                if not isinstance(line, list):
                    continue
                for element in line:
                    if not isinstance(element, dict):
                        continue
                    tag = element.get("tag")
                    text = element.get("text")
                    if tag in ("text", "a") and isinstance(text, str):
                        line_parts.append(text)
                joined = "".join(line_parts).strip()
                if joined:
                    parts.append(joined)
        return "\n".join(parts)


# ── Module-level helpers ──────────────────────────────────────────────


def _parse_content_json(content: str, *, kind: str) -> JsonObject:
    """Parse the JSON-encoded ``content`` field of a Feishu message."""
    try:
        payload = json.loads(content)
    except (ValueError, TypeError):
        raise ExternalServiceError(
            code="im.feishu_parse_failed",
            message=f"invalid {kind} message content",
        ) from None
    if not isinstance(payload, dict):
        raise ExternalServiceError(
            code="im.feishu_parse_failed",
            message=f"invalid {kind} message content",
        )
    return payload


def _extract_sender_open_id(payload: JsonObject) -> str:
    """Return the sender's ``open_id`` from a message event, or ``""``."""
    event = payload.get("event")
    if not isinstance(event, dict):
        return ""
    sender = event.get("sender")
    if not isinstance(sender, dict):
        return ""
    sender_id = sender.get("sender_id")
    if not isinstance(sender_id, dict):
        return ""
    value = sender_id.get("open_id")
    return value if isinstance(value, str) else ""


def _strip_mention(content: str) -> str:
    """Strip the leading ``@_user_...`` mention token from a group message.

    Feishu renders @-mentions in group text as ``@_user_<id> ...``; the
    loop removes every leading mention token so only the user's own text
    reaches the QA pipeline.
    """
    while content.startswith("@_user_"):
        idx = content.find(" ")
        if idx < 0:
            break
        content = content[idx + 1 :]
    return content


def _resolve_receive_id(incoming: IncomingMessage) -> tuple[str, str]:
    """Compute the ``(receive_id_type, receive_id)`` pair for a send.

    Group chats target ``chat_id``; direct chats target the sender's
    ``open_id``.
    """
    if incoming.chat_type == CHAT_TYPE_GROUP and incoming.chat_id:
        return "chat_id", incoming.chat_id
    return "open_id", incoming.user_id


def _is_safe_path_param(value: str) -> bool:
    """True when ``value`` is a non-empty URL-safe path parameter.

    Restricts Feishu API path parameters to ASCII alphanumerics plus
    ``-`` / ``_`` so a crafted message id cannot inject path separators.
    """
    if not value:
        return False
    return all(
        ("a" <= char <= "z") or ("A" <= char <= "Z") or ("0" <= char <= "9") or char in "-_"
        for char in value
    )


# ── Factory ───────────────────────────────────────────────────────────


def build_feishu_adapter(channel: IMChannel) -> FeishuAdapter:
    """Construct a ``FeishuAdapter`` from a persisted channel row.

    Reads the app credentials from ``channel.credentials``; the platform
    comes from the row (``feishu`` or ``lark``).
    """
    credentials = channel.credentials
    return FeishuAdapter(
        platform=channel.platform,
        app_id=credential_string(credentials, "app_id"),
        app_secret=credential_string(credentials, "app_secret"),
        verification_token=credential_string(credentials, "verification_token"),
        encrypt_key=credential_string(credentials, "encrypt_key"),
        api_base_url=credential_string(credentials, "api_base_url"),
    )


__all__ = ["FeishuAdapter", "build_feishu_adapter"]
