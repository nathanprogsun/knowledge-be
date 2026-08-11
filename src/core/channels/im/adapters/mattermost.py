"""Mattermost adapter — outgoing-webhook callbacks and REST API replies.

Mirrors the upstream contract: ``verify_callback`` checks the outgoing
webhook ``token`` (JSON or form-encoded body), ``parse_callback`` builds
the unified message and resolves the thread root via the REST API when
the webhook omitted it, and ``send_reply`` creates a channel post through
``POST /api/v4/posts``, threading replies under the resolved root when
``post_to_main`` is not set.
"""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from src.common.exception import UnauthorizedError, ValidationError
from src.common.json import JsonObject, JsonValue
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
    bool_credential,
    build_http_client,
    constant_time_equals,
    header_value,
    payload_string,
    send_error,
    string_credential,
    validate_http_endpoint,
)
from src.db.models.im_channel import IMChannel

# Extra keys the adapter stores on the unified message.
_EXTRA_THREAD_ROOT = "thread_root_id"
_EXTRA_CHANNEL_ID = "channel_id"


@dataclass(frozen=True)
class _OutgoingPayload:
    """Normalized Mattermost outgoing-webhook payload (JSON or form)."""

    token: str = ""
    user_id: str = ""
    user_name: str = ""
    channel_id: str = ""
    post_id: str = ""
    text: str = ""
    root_id: str = ""
    file_ids: tuple[str, ...] = field(default_factory=tuple)


class MattermostAdapter(IMAdapter):
    """Mattermost platform adapter (outgoing webhook + REST replies)."""

    def __init__(
        self,
        *,
        site_url: str = "",
        bot_token: str = "",
        outgoing_token: str = "",
        bot_user_id: str = "",
        post_to_main: bool = False,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._site_url = site_url.rstrip("/")
        self._bot_token = bot_token
        self._outgoing_token = outgoing_token.strip()
        self._bot_user_id = bot_user_id.strip()
        self._post_to_main = post_to_main
        self._http_client = build_http_client(timeout=60.0, transport=transport)
        self._connected = False

    # ── Identity ─────────────────────────────────────────────────────

    def platform(self) -> str:
        return "mattermost"

    # ── Callback verification ────────────────────────────────────────

    def verify_callback(self, request: CallbackRequest) -> None:
        if not self._outgoing_token:
            return
        payload = _parse_outgoing_body(request.headers, request.body)
        if not constant_time_equals(payload.token, self._outgoing_token):
            raise UnauthorizedError(
                code="im.verify_failed",
                message="mattermost outgoing webhook token verification failed",
            )

    # ── Callback parsing ─────────────────────────────────────────────

    def parse_callback(self, request: CallbackRequest) -> IncomingMessage | None:
        payload = _parse_outgoing_body(request.headers, request.body)

        if self._bot_user_id and payload.user_id == self._bot_user_id:
            return None
        if not payload.text.strip() and not payload.file_ids:
            return None

        thread_root = ""
        if not self._post_to_main:
            thread_root = payload.root_id
            if not thread_root:
                thread_root = self._resolve_thread_root(payload.post_id) or payload.post_id

        extra: JsonObject = {
            _EXTRA_THREAD_ROOT: thread_root,
            _EXTRA_CHANNEL_ID: payload.channel_id,
        }
        incoming = IncomingMessage(
            platform="mattermost",
            message_type="file" if payload.file_ids else "text",
            user_id=payload.user_id,
            user_name=payload.user_name,
            chat_id=payload.channel_id,
            chat_type=CHAT_TYPE_GROUP,
            content=payload.text.strip(),
            message_id=payload.post_id,
            thread_id=thread_root,
            extra=extra,
        )
        if payload.file_ids:
            file_ids = payload.file_ids
            incoming = incoming.model_copy(
                update={
                    "file_key": file_ids[0],
                    "extra": {**_with_file_ids(extra, file_ids)},
                }
            )
        return incoming

    # ── URL verification ─────────────────────────────────────────────

    def handle_url_verification(self, request: CallbackRequest) -> bool:
        return False

    # ── Send reply ───────────────────────────────────────────────────

    def send_reply(self, ctx: Context, incoming: IncomingMessage, reply: ReplyMessage) -> None:
        channel_id = incoming.chat_id or str(incoming.extra.get(_EXTRA_CHANNEL_ID, ""))
        if not channel_id:
            send_error("mattermost", "create post", "missing channel_id")
        body: dict[str, str] = {"channel_id": channel_id, "message": reply.content}
        thread_root = str(incoming.extra.get(_EXTRA_THREAD_ROOT, ""))
        if thread_root:
            body["root_id"] = thread_root
        resp = self._http_client.post(
            f"{self._site_url}/api/v4/posts",
            json=body,
            headers={"Authorization": f"Bearer {self._bot_token}"},
        )
        assert_http_ok(resp, platform="mattermost", action="create post")

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

    # ── Thread-root resolution ───────────────────────────────────────

    def _resolve_thread_root(self, post_id: str) -> str:
        if not post_id or not self._site_url or not self._bot_token:
            return ""
        try:
            resp = self._http_client.get(
                f"{self._site_url}/api/v4/posts/{post_id}",
                headers={"Authorization": f"Bearer {self._bot_token}"},
            )
            if 200 <= resp.status_code < 300:
                result = resp.json()
                if isinstance(result, dict):
                    return payload_string(result, "root_id")
        except (httpx.HTTPError, json.JSONDecodeError):
            return ""
        return ""


# ── Parse helpers ─────────────────────────────────────────────────────


def _parse_outgoing_body(headers: JsonObject, body: str) -> _OutgoingPayload:
    content_type = header_value(headers, "Content-Type").split(";")[0].strip().lower()
    if content_type in ("application/json", "") or content_type.endswith("+json"):
        parsed = _parse_json(body)
        if parsed:
            return _outgoing_from_json(parsed)
    if content_type in ("application/x-www-form-urlencoded", ""):
        form = urllib.parse.parse_qs(body)
        if form:
            return _outgoing_from_form(form)
    parsed = _parse_json(body)
    if parsed and (payload_string(parsed, "token") or payload_string(parsed, "channel_id")):
        return _outgoing_from_json(parsed)
    return _OutgoingPayload()


def _parse_json(body: str) -> JsonObject | None:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _outgoing_from_json(parsed: JsonObject) -> _OutgoingPayload:
    file_ids = _coerce_file_ids(parsed.get("file_ids"))
    return _OutgoingPayload(
        token=payload_string(parsed, "token"),
        user_id=payload_string(parsed, "user_id"),
        user_name=payload_string(parsed, "user_name"),
        channel_id=payload_string(parsed, "channel_id"),
        post_id=payload_string(parsed, "post_id"),
        text=payload_string(parsed, "text"),
        root_id=payload_string(parsed, "root_id"),
        file_ids=file_ids,
    )


def _outgoing_from_form(form: dict[str, list[str]]) -> _OutgoingPayload:
    raw_file_ids = ",".join(form.get("file_ids", []))
    return _OutgoingPayload(
        token=_first(form, "token"),
        user_id=_first(form, "user_id"),
        user_name=_first(form, "user_name"),
        channel_id=_first(form, "channel_id"),
        post_id=_first(form, "post_id"),
        text=_first(form, "text"),
        root_id=_first(form, "root_id"),
        file_ids=tuple(part for part in raw_file_ids.split(",") if part.strip()),
    )


def _coerce_file_ids(value: JsonValue) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item) for item in value if item is not None and str(item).strip())
    if isinstance(value, str):
        return tuple(part for part in value.split(",") if part.strip())
    return ()


def _first(form: dict[str, list[str]], key: str) -> str:
    values = form.get(key)
    if values:
        return values[0]
    return ""


def _with_file_ids(extra: JsonObject, file_ids: tuple[str, ...]) -> JsonObject:
    merged = dict(extra)
    if len(file_ids) > 1:
        merged["file_ids"] = ",".join(file_ids)
    return merged


__all__ = ["MattermostAdapter", "build_mattermost_adapter"]


def build_mattermost_adapter(channel: IMChannel) -> MattermostAdapter:
    """Construct a Mattermost adapter, validating required credentials."""
    outgoing_token = string_credential(channel.credentials, "outgoing_token")
    site_url = string_credential(channel.credentials, "site_url")
    bot_token = string_credential(channel.credentials, "bot_token")
    if not outgoing_token:
        raise ValidationError(
            code="im.credentials_invalid",
            message="mattermost outgoing_token is required",
        )
    if not site_url:
        raise ValidationError(
            code="im.credentials_invalid",
            message="mattermost site_url is required",
        )
    if not bot_token:
        raise ValidationError(
            code="im.credentials_invalid",
            message="mattermost bot_token is required",
        )
    validate_http_endpoint(site_url)
    return MattermostAdapter(
        site_url=site_url,
        bot_token=bot_token,
        outgoing_token=outgoing_token,
        bot_user_id=string_credential(channel.credentials, "bot_user_id"),
        post_to_main=bool_credential(channel.credentials, "post_to_main"),
    )
