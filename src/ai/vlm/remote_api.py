"""OpenAI-compatible vision-language client.

Maps the upstream remote-API VLM backend to Python over ``httpx``. A
request is a standard chat-completions payload: one text part plus one
``image_url`` part per image (base64 data URI), posted to
``/chat/completions``. Azure endpoints route to the
``/openai/deployments/<model>`` path with the ``api-key`` header.
"""

from __future__ import annotations

import base64
import contextlib
from typing import Final, cast

import httpx

from src.ai.provider.registry import PROVIDER_AZURE_OPENAI
from src.ai.vlm.transport import (
    new_vlm_http_client,
    validate_vlm_base_url,
    vlm_http_timeout,
)
from src.common.exception import AIProviderError
from src.common.json import JsonObject, JsonValue

DEFAULT_MAX_TOKENS: Final = 5000
DEFAULT_TEMPERATURE: Final = 0.1

_DEFAULT_AZURE_API_VERSION: Final = "2023-05-15"
_CHAT_COMPLETIONS_PATH: Final = "/chat/completions"


def detect_image_mime(data: bytes) -> str:
    """Return the MIME type for the given image bytes.

    Sniffs the common raster magic bytes; anything unrecognised falls
    back to ``image/png`` exactly like the upstream sniffer.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    return "image/png"


def _build_content_parts(img_bytes: list[bytes], prompt: str) -> list[dict[str, JsonValue]]:
    """Render the text prompt plus one image part per non-empty image."""
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
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded}",
                        "detail": "auto",
                    },
                },
            )
        )
    return parts


class RemoteAPIVLM:
    """Vision-language client for any OpenAI-compatible chat API."""

    def __init__(
        self,
        *,
        model_name: str,
        model_id: str,
        base_url: str,
        api_key: str,
        temperature: float,
        client: httpx.AsyncClient,
        azure: bool = False,
        api_version: str = "",
    ) -> None:
        self._model_name = model_name
        self._model_id = model_id
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._temperature = temperature
        self._client = client
        self._azure = azure
        self._api_version = api_version or _DEFAULT_AZURE_API_VERSION

    def get_model_name(self) -> str:
        return self._model_name

    def get_model_id(self) -> str:
        return self._model_id

    async def predict(self, img_bytes: list[bytes], prompt: str) -> str:
        """Send images with a text prompt and return the generated text."""
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "user", "content": _build_content_parts(img_bytes, prompt)},
            ],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": self._temperature,
        }
        url, headers = self._request_target()
        try:
            response = await self._client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise AIProviderError(
                f"OpenAI VLM request failed: {exc}",
                code="vlm.openai_request_failed",
            ) from exc
        return _extract_content(response, "OpenAI VLM")

    def _request_target(self) -> tuple[str, dict[str, str]]:
        if self._azure:
            url = (
                f"{self._base_url}/openai/deployments/{self._model_name}"
                f"{_CHAT_COMPLETIONS_PATH}?api-version={self._api_version}"
            )
            return url, {"api-key": self._api_key}
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return f"{self._base_url}{_CHAT_COMPLETIONS_PATH}", headers


def _extract_content(response: httpx.Response, backend: str) -> str:
    if response.status_code != 200:
        raise AIProviderError(
            f"{backend} returned status {response.status_code}: {response.text[:200]}",
            code="vlm.request_failed",
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise AIProviderError(
            f"{backend} returned invalid JSON",
            code="vlm.invalid_response",
        ) from exc
    if not isinstance(body, dict):
        raise AIProviderError(
            f"{backend} returned a non-object body",
            code="vlm.invalid_response",
        )
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AIProviderError(f"{backend} returned no choices", code="vlm.no_choices")
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise AIProviderError(
            f"{backend} returned no text content",
            code="vlm.invalid_response",
        )
    return content


async def new_remote_api_vlm(
    *,
    model_name: str,
    model_id: str,
    base_url: str,
    api_key: str,
    provider: str,
    extra: JsonObject | None,
    custom_headers: dict[str, str] | None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> RemoteAPIVLM:
    """Build a remote-API VLM (SSRF-validates the base URL)."""
    await validate_vlm_base_url(base_url)
    azure = provider == PROVIDER_AZURE_OPENAI
    api_version = ""
    if azure and extra:
        raw_version = extra.get("api_version")
        if isinstance(raw_version, str):
            api_version = raw_version
    temperature = DEFAULT_TEMPERATURE
    if extra:
        raw_temperature = extra.get("temperature")
        if isinstance(raw_temperature, str):
            with contextlib.suppress(ValueError):
                temperature = float(raw_temperature)
        elif isinstance(raw_temperature, (int, float)):
            temperature = float(raw_temperature)
    client = new_vlm_http_client(
        vlm_http_timeout(),
        custom_headers=custom_headers,
        transport=transport,
    )
    return RemoteAPIVLM(
        model_name=model_name,
        model_id=model_id,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        client=client,
        azure=azure,
        api_version=api_version,
    )


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "RemoteAPIVLM",
    "detect_image_mime",
    "new_remote_api_vlm",
]
