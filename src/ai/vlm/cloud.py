"""Managed-cloud vision-language client.

Maps the upstream Cloud VLM backend: posts the chat payload to
the managed-cloud chat-completions path with signed ``X-*`` headers and
returns ``choices[0].message.content``. The request body is signed with
the caller-supplied app id / app secret before sending.
"""

from __future__ import annotations

import base64
import json
import uuid
from typing import Final, cast

import httpx

from src.ai.utils.signer import sign_request
from src.ai.vlm.remote_api import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, detect_image_mime
from src.ai.vlm.transport import (
    new_vlm_http_client,
    validate_vlm_base_url,
    vlm_http_timeout,
)
from src.common.exception import AIProviderError, ValidationError
from src.common.json import JsonObject, JsonValue

MANAGED_CLOUD_CHAT_PATH: Final = "/api/v1/chat/completions"


def _build_cloud_content_parts(img_bytes: list[bytes], prompt: str) -> list[dict[str, JsonValue]]:
    """Render the text prompt plus one ``image_url`` part per image."""
    parts: list[dict[str, JsonValue]] = [
        cast(dict[str, JsonValue], {"type": "text", "text": prompt})
    ]
    for image in img_bytes:
        if not image:
            continue
        mime_type = detect_image_mime(image)
        encoded = base64.b64encode(image).decode("ascii")
        parts.append(
            cast(
                dict[str, JsonValue],
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                },
            )
        )
    return parts


class CloudVLM:
    """Vision-language client backed by the managed cloud API."""

    def __init__(
        self,
        *,
        model_name: str,
        remote_model_name: str,
        model_id: str,
        app_id: str,
        api_key: str,
        base_url: str,
        client: httpx.AsyncClient,
    ) -> None:
        self._model_name = model_name
        self._remote_model_name = remote_model_name
        self._model_id = model_id
        self._app_id = app_id
        self._api_key = api_key
        self._base_url = base_url
        self._client = client

    def get_model_name(self) -> str:
        return self._model_name

    def get_model_id(self) -> str:
        return self._model_id

    async def predict(self, img_bytes: list[bytes], prompt: str) -> str:
        """Send images with a text prompt and return the generated text."""
        body = {
            "model": self._effective_model_name(),
            "messages": [
                {
                    "role": "user",
                    "content": _build_cloud_content_parts(img_bytes, prompt),
                }
            ],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": DEFAULT_TEMPERATURE,
            "stream": False,
        }
        body_text = json.dumps(body, separators=(",", ":"))
        request_id = str(uuid.uuid4())
        headers = sign_request(self._app_id, self._api_key, request_id, body_text)
        headers["Content-Type"] = "application/json"
        url = f"{self._base_url}{MANAGED_CLOUD_CHAT_PATH}"
        try:
            response = await self._client.post(
                url,
                content=body_text.encode("utf-8"),
                headers=headers,
            )
        except httpx.HTTPError as exc:
            raise AIProviderError(
                f"managed cloud VLM request failed: {exc}",
                code="vlm.cloud_request_failed",
            ) from exc
        if response.status_code != 200:
            raise AIProviderError(
                f"managed cloud VLM returned status {response.status_code}: {response.text[:200]}",
                code="vlm.cloud_request_failed",
            )
        try:
            resp_body = response.json()
        except ValueError as exc:
            raise AIProviderError(
                "managed cloud VLM returned invalid JSON",
                code="vlm.invalid_response",
            ) from exc
        if not isinstance(resp_body, dict):
            raise AIProviderError(
                "managed cloud VLM returned a non-object body",
                code="vlm.invalid_response",
            )
        choices = resp_body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise AIProviderError(
                "managed cloud VLM returned no choices",
                code="vlm.no_choices",
            )
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise AIProviderError(
                "managed cloud VLM returned no text content",
                code="vlm.invalid_response",
            )
        return content

    def _effective_model_name(self) -> str:
        return self._remote_model_name or self._model_name


async def new_cloud_vlm(
    *,
    model_name: str,
    model_id: str,
    base_url: str,
    app_id: str,
    app_secret: str,
    extra: JsonObject | None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> CloudVLM:
    """Build a managed-cloud VLM (validates credentials and base URL)."""
    if not app_id:
        raise ValidationError(
            "managed cloud VLM: app id is required",
            code="vlm.app_id_required",
        )
    if not app_secret:
        raise ValidationError(
            "managed cloud VLM: app secret is required",
            code="vlm.app_secret_required",
        )
    base_url = base_url.rstrip("/")
    await validate_vlm_base_url(base_url)
    remote_model_name = ""
    if extra:
        raw = extra.get("remote_model_name")
        if isinstance(raw, str):
            remote_model_name = raw.strip()
    client = new_vlm_http_client(vlm_http_timeout(), transport=transport)
    return CloudVLM(
        model_name=model_name,
        remote_model_name=remote_model_name,
        model_id=model_id,
        app_id=app_id,
        api_key=app_secret,
        base_url=base_url,
        client=client,
    )


__all__ = [
    "MANAGED_CLOUD_CHAT_PATH",
    "CloudVLM",
    "new_cloud_vlm",
]
