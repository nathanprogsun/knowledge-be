"""WeCom IM adapter (webhook mode) — message send, callbacks, URL verification.

Implements the WeCom self-built app callback contract: the platform
POSTs an XML envelope whose ``Encrypt`` field holds an AES-256-CBC
encrypted message, and GETs the callback URL with an ``echostr`` to
verify the endpoint.

Callback flow:

1. The platform calls the callback URL — GET for URL verification
   (``echostr`` + ``msg_signature`` query params), POST with an XML
   envelope for message events.
2. ``handle_url_verification`` decrypts the ``echostr`` so the web layer
   can echo it back; ``verify_callback`` recomputes the
   ``msg_signature`` (SHA-1 over the sorted token/timestamp/nonce/encrypt
   tuple) and raises ``UnauthorizedError`` on mismatch;
   ``parse_callback`` decrypts the XML message into an
   ``IncomingMessage``.
3. ``send_reply`` sends group replies via the appchat API and direct
   replies via the application message API (both markdown).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from typing import Final
from urllib.parse import quote

import httpx

from src.common.exception import ExternalServiceError, UnauthorizedError
from src.common.json import JsonObject
from src.core.channels.im.adapter_base import (
    CHAT_TYPE_DIRECT,
    CHAT_TYPE_GROUP,
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

logger = logging.getLogger("src.core.channels.im.adapters.wecom")

#: Public WeCom API origin (no trailing slash).
_WECOM_API_BASE_URL: Final[str] = "https://qyapi.weixin.qq.com"

#: Per-call HTTP timeout (seconds), matching the upstream client.
_DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0

#: Token safety margin (seconds). WeCom tokens live 2 hours; caching 5
#: minutes short avoids sending an about-to-expire token.
_TOKEN_SAFETY_MARGIN_SECONDS: Final[int] = 300

#: Default expiry when the token response omits ``expires_in``.
_DEFAULT_TOKEN_TTL_SECONDS: Final[int] = 7200

#: Number of random bytes prefixed before the length field in a
#: decrypted WeCom payload (16 random + 4 length).
_WECOM_HEADER_LENGTH: Final[int] = 20


class WecomAdapter(IMAdapter):
    """WeCom webhook adapter: message send, callbacks, URL verification."""

    def __init__(
        self,
        *,
        corp_id: str,
        agent_secret: str,
        token: str,
        encoding_aes_key: str,
        corp_agent_id: int = 0,
        api_base_url: str = "",
        http_client: httpx.Client | None = None,
    ) -> None:
        self._corp_id = corp_id
        self._agent_secret = agent_secret
        self._token = token
        self._encoding_aes_key = encoding_aes_key
        self._corp_agent_id = corp_agent_id
        self._api_base_url = (api_base_url or _WECOM_API_BASE_URL).rstrip("/")
        self._http_client = http_client
        self._owns_client = http_client is None
        self._client: httpx.Client | None = None
        self._aes_key = self._decode_aes_key(encoding_aes_key)
        self._verification_body: str | None = None
        self._token_lock = threading.Lock()
        self._token_cache = ""
        self._token_expires_monotonic = 0.0

    # ── Identity ────────────────────────────────────────────────────

    def platform(self) -> str:
        """Return the platform identifier (``wecom``)."""
        return "wecom"

    # ── Callback verification ───────────────────────────────────────

    def verify_callback(self, request: CallbackRequest) -> None:
        """Verify the ``msg_signature`` of an inbound callback.

        Raises ``UnauthorizedError`` when the signature does not match
        the SHA-1 over the sorted ``token`` / ``timestamp`` / ``nonce`` /
        ``encrypt`` tuple. URL-verification GETs carry ``echostr`` in the
        query; message callbacks carry ``Encrypt`` in the XML body.
        """
        query = request.query
        timestamp = _query_string(query, "timestamp")
        nonce = _query_string(query, "nonce")
        msg_signature = _query_string(query, "msg_signature")
        echostr = _query_string(query, "echostr")
        encrypt = echostr or self._extract_encrypt(request.body)
        if not self._verify_signature(msg_signature, timestamp, nonce, encrypt):
            raise UnauthorizedError(
                code="im.wecom_verify_failed",
                message="invalid callback signature",
            )

    def handle_url_verification(self, request: CallbackRequest) -> bool:
        """Decrypt the URL-verification ``echostr`` and store the answer.

        Returns ``True`` (with the decrypted text available via
        :meth:`verification_body`) when ``request`` carries an
        ``echostr``; ``False`` otherwise. A malformed ``echostr`` raises
        ``ExternalServiceError`` so the web layer can answer 4xx.
        """
        echostr = _query_string(request.query, "echostr")
        if not echostr:
            return False
        decrypted = self._decrypt(echostr)
        self._verification_body = decrypted.decode("utf-8", errors="replace")
        return True

    def verification_body(self) -> str | None:
        """Return the decrypted text the web layer echoes back after verification.

        ``None`` when no ``echostr`` was consumed.
        """
        return self._verification_body

    # ── Callback parsing ────────────────────────────────────────────

    def parse_callback(self, request: CallbackRequest) -> IncomingMessage | None:
        """Parse a decrypted WeCom message into an ``IncomingMessage``.

        Returns ``None`` for unsupported message types. A malformed
        envelope raises ``ExternalServiceError``.
        """
        decrypted = self._decrypt_body(request.body)
        try:
            root = ET.fromstring(decrypted)
        except ET.ParseError:
            raise ExternalServiceError(
                code="im.wecom_parse_failed",
                message="decrypted message is not valid XML",
            ) from None
        msg_type = _xml_text(root, "MsgType")
        from_user = _xml_text(root, "FromUserName")
        msg_id = _xml_text(root, "MsgId")
        chat_id = _xml_text(root, "ChatId")
        is_group = bool(chat_id)
        chat_type = CHAT_TYPE_GROUP if is_group else CHAT_TYPE_DIRECT

        if msg_type == "text":
            content = _xml_text(root, "Content")
            if is_group:
                content = _strip_at_mention_basic(content)
            return IncomingMessage(
                platform=self.platform(),
                message_type=MESSAGE_TYPE_TEXT,
                user_id=from_user,
                user_name=from_user,
                chat_id=chat_id,
                chat_type=chat_type,
                content=content.strip(),
                message_id=msg_id,
            )
        if msg_type == "image":
            pic_url = _xml_text(root, "PicUrl")
            media_id = _xml_text(root, "MediaId")
            if not pic_url and not media_id:
                return None
            file_key = pic_url or media_id
            return IncomingMessage(
                platform=self.platform(),
                message_type=MESSAGE_TYPE_IMAGE,
                user_id=from_user,
                user_name=from_user,
                chat_id=chat_id,
                chat_type=chat_type,
                message_id=msg_id,
                file_key=file_key,
                file_name=f"{msg_id}.png",
            )
        return None

    # ── Reply sending ───────────────────────────────────────────────

    def send_reply(self, ctx: Context, incoming: IncomingMessage, reply: ReplyMessage) -> None:
        """Deliver ``reply`` to the originating conversation.

        Group chats are answered via the appchat API first (which works
        for groups created through the appchat endpoint); on any failure
        the reply falls back to a direct message to the sender.
        """
        token = self.get_access_token()
        if incoming.chat_type == CHAT_TYPE_GROUP and incoming.chat_id:
            try:
                self._send_to_appchat(token, incoming.chat_id, reply.content)
                return
            except ExternalServiceError as exc:
                logger.warning(
                    "[WeCom] appchat/send failed for chat=%s, falling back to touser: %s",
                    incoming.chat_id,
                    exc,
                )
        self._send_to_user(token, incoming.user_id, reply.content)

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
        """Return a cached access token, fetching it on first use.

        Tokens are cached with a 5-minute safety margin so the process
        does not hammer the gettoken endpoint on every reply.
        """
        with self._token_lock:
            if self._token_cache and time.monotonic() < self._token_expires_monotonic:
                return self._token_cache

            url = (
                f"{self._api_base_url}/cgi-bin/gettoken"
                f"?corpid={quote(self._corp_id)}&corpsecret={quote(self._agent_secret)}"
            )
            client = self._get_client()
            try:
                resp = client.get(url)
            except httpx.HTTPError as exc:
                raise ExternalServiceError(
                    code="im.wecom_token_failed",
                    message=f"wecom token request failed: {exc}",
                ) from exc
            try:
                data = resp.json()
            except ValueError:
                raise ExternalServiceError(
                    code="im.wecom_token_failed",
                    message="non-JSON token response",
                ) from None
            if not isinstance(data, dict):
                raise ExternalServiceError(
                    code="im.wecom_token_failed",
                    message="wecom token response is not an object",
                )
            errcode = json_int(data.get("errcode", -1))
            if errcode != 0:
                raise ExternalServiceError(
                    code="im.wecom_token_failed",
                    message=f"wecom token error: code={errcode} msg={data.get('errmsg', '')}",
                )
            token = data.get("access_token")
            if not isinstance(token, str) or not token:
                raise ExternalServiceError(
                    code="im.wecom_token_failed",
                    message="empty access token",
                )
            ttl = json_int(data.get("expires_in", _DEFAULT_TOKEN_TTL_SECONDS))
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

    def _send_to_appchat(self, token: str, chat_id: str, content: str) -> None:
        payload: JsonObject = {
            "chatid": chat_id,
            "msgtype": "markdown",
            "markdown": {"content": content},
        }
        url = f"{self._api_base_url}/cgi-bin/appchat/send?access_token={token}"
        data = self._post_json(url, payload)
        errcode = json_int(data.get("errcode", -1))
        if errcode != 0:
            raise ExternalServiceError(
                code="im.wecom_send_failed",
                message=f"wecom appchat api error: code={errcode} msg={data.get('errmsg', '')}",
            )

    def _send_to_user(self, token: str, user_id: str, content: str) -> None:
        payload: JsonObject = {
            "touser": user_id,
            "msgtype": "markdown",
            "agentid": self._corp_agent_id,
            "markdown": {"content": content},
        }
        url = f"{self._api_base_url}/cgi-bin/message/send?access_token={token}"
        data = self._post_json(url, payload)
        errcode = json_int(data.get("errcode", -1))
        if errcode != 0:
            raise ExternalServiceError(
                code="im.wecom_send_failed",
                message=f"wecom api error: code={errcode} msg={data.get('errmsg', '')}",
            )

    def _post_json(self, url: str, payload: JsonObject) -> JsonObject:
        client = self._get_client()
        try:
            resp = client.post(url, json=payload, headers={"Content-Type": "application/json"})
        except httpx.HTTPError as exc:
            raise ExternalServiceError(
                code="im.wecom_send_failed",
                message=f"wecom request failed: {exc}",
            ) from exc
        try:
            data = resp.json()
        except ValueError:
            raise ExternalServiceError(
                code="im.wecom_send_failed",
                message=f"non-JSON response from wecom api (status={resp.status_code})",
            ) from None
        if not isinstance(data, dict):
            raise ExternalServiceError(
                code="im.wecom_send_failed",
                message="wecom api returned a non-object body",
            )
        return data

    def _verify_signature(self, signature: str, timestamp: str, nonce: str, encrypt: str) -> bool:
        expected = self._signature(timestamp, nonce, encrypt)
        return hmac.compare_digest(expected, signature)

    def _signature(self, timestamp: str, nonce: str, encrypt: str) -> str:
        parts = sorted([self._token, timestamp, nonce, encrypt])
        combined = "".join(parts)
        return hashlib.sha1(combined.encode("utf-8")).hexdigest()

    def _extract_encrypt(self, body: str) -> str:
        """Read the ``Encrypt`` element from the XML callback envelope."""
        if not body.strip():
            return ""
        try:
            root = ET.fromstring(body)
        except ET.ParseError:
            return ""
        return _xml_text(root, "Encrypt")

    def _decrypt_body(self, body: str) -> str:
        """Return the decrypted XML message as text."""
        encrypted = self._extract_encrypt(body)
        decrypted = self._decrypt(encrypted)
        return decrypted.decode("utf-8", errors="replace")

    def _decrypt(self, encrypted: str) -> bytes:
        """Decrypt a WeCom payload: ``random(16) + len(4) + msg + corp_id``.

        The AES key is the base64-decoded ``encoding_aes_key``; the CBC
        IV is its first 16 bytes. The tail is verified against
        ``corp_id`` so a ciphertext from another tenant is rejected.
        """
        try:
            ciphertext = base64.b64decode(encrypted)
        except ValueError:
            raise ExternalServiceError(
                code="im.wecom_decrypt_failed",
                message="base64 decode failed",
            ) from None
        if len(ciphertext) < 16:
            raise ExternalServiceError(
                code="im.wecom_decrypt_failed",
                message="ciphertext too short",
            )
        try:
            plaintext = aes_cbc_decrypt(self._aes_key, self._aes_key[:16], ciphertext)
        except ValueError as exc:
            raise ExternalServiceError(
                code="im.wecom_decrypt_failed",
                message=f"aes decrypt failed: {exc}",
            ) from exc
        if len(plaintext) < _WECOM_HEADER_LENGTH:
            raise ExternalServiceError(
                code="im.wecom_decrypt_failed",
                message="plaintext too short",
            )
        msg_len = int.from_bytes(plaintext[16:20], "big")
        if len(plaintext) < _WECOM_HEADER_LENGTH + msg_len:
            raise ExternalServiceError(
                code="im.wecom_decrypt_failed",
                message="message length mismatch",
            )
        msg_bytes = plaintext[_WECOM_HEADER_LENGTH : _WECOM_HEADER_LENGTH + msg_len]
        tail = plaintext[_WECOM_HEADER_LENGTH + msg_len :]
        if tail.decode("utf-8", errors="replace") != self._corp_id:
            raise ExternalServiceError(
                code="im.wecom_decrypt_failed",
                message="corp_id mismatch",
            )
        return msg_bytes

    @staticmethod
    def _decode_aes_key(encoding_aes_key: str) -> bytes:
        """Base64-decode the ``encoding_aes_key`` (padding appended if needed)."""
        try:
            return base64.b64decode(encoding_aes_key + "=")
        except ValueError:
            raise ExternalServiceError(
                code="im.wecom_config_failed",
                message="invalid encoding_aes_key",
            ) from None


# ── Module-level helpers ──────────────────────────────────────────────


def _query_string(query: JsonObject, key: str) -> str:
    """Read a string query parameter, defaulting to ``""``."""
    value = query.get(key)
    return value if isinstance(value, str) else ""


def _xml_text(root: ET.Element, name: str) -> str:
    """Return the text of the first element matching ``name`` (case-insensitive)."""
    lowered = name.lower()
    for element in root.iter():
        tag = element.tag
        if not isinstance(tag, str):
            continue
        if tag == name or tag.lower() == lowered:
            return element.text or ""
    return ""


def _strip_at_mention_basic(content: str) -> str:
    """Strip the leading ``@BotName`` prefix from a group chat message.

    Mirrors the upstream heuristic: trim, reject non-@ messages, cut at
    the first double space, then at a space followed by ``/`` or a
    non-ASCII character, and finally at the first space (a bare "@word"
    prefix).
    """
    content = content.strip()
    if not content.startswith("@"):
        return content
    if "  " in content:
        idx = content.index("  ")
        if idx > 0:
            return content[idx + 2 :].strip()
    for i in range(1, len(content)):
        if content[i] == " " and i + 1 < len(content):
            nxt = content[i + 1]
            if nxt == "/" or ord(nxt) >= 0x80:
                return content[i + 1 :].strip()
    if " " in content:
        idx = content.index(" ")
        if idx > 0:
            return content[idx + 1 :].strip()
    return content


# ── Factory ───────────────────────────────────────────────────────────


def build_wecom_adapter(channel: IMChannel) -> WecomAdapter:
    """Construct a ``WecomAdapter`` from a persisted channel row.

    Reads the corp credentials from ``channel.credentials``; the agent
    id is coerced to ``int`` when the stored value is numeric.
    """
    credentials = channel.credentials
    corp_agent_id = json_int(credentials.get("corp_agent_id", 0))
    if corp_agent_id < 0:
        corp_agent_id = 0
    return WecomAdapter(
        corp_id=credential_string(credentials, "corp_id"),
        agent_secret=credential_string(credentials, "agent_secret"),
        token=credential_string(credentials, "token"),
        encoding_aes_key=credential_string(credentials, "encoding_aes_key"),
        corp_agent_id=corp_agent_id,
        api_base_url=credential_string(credentials, "api_base_url"),
    )


__all__ = ["WecomAdapter", "build_wecom_adapter"]
