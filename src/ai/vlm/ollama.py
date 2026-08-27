"""Local Ollama vision-language client.

Maps the upstream Ollama VLM backend. ``predict`` sends one chat
request with the prompt and base64-encoded images to the local service
and returns ``message.content`` from the response.
"""

from __future__ import annotations

import base64
from typing import Final, Protocol, cast

from src.common.exception import AIProviderError
from src.common.json import JsonObject

DEFAULT_TEMPERATURE: Final = 0.1


class OllamaChatService(Protocol):
    """Async Ollama chat interface required by the local VLM.

    The concrete service must expose ``chat`` accepting the request JSON
    body (Ollama ``/api/chat`` wire shape) and returning the parsed
    response JSON body.
    """

    async def chat(self, chat_request: JsonObject) -> JsonObject: ...


class OllamaVLM:
    """Vision-language client backed by a local Ollama service."""

    def __init__(
        self,
        *,
        model_name: str,
        model_id: str,
        ollama_service: OllamaChatService,
    ) -> None:
        self._model_name = model_name
        self._model_id = model_id
        self._ollama_service = ollama_service

    def get_model_name(self) -> str:
        return self._model_name

    def get_model_id(self) -> str:
        return self._model_id

    async def predict(self, img_bytes: list[bytes], prompt: str) -> str:
        """Send images with a text prompt and return the generated text."""
        images = [base64.b64encode(image).decode("ascii") for image in img_bytes if image]
        chat_request: JsonObject = cast(
            JsonObject,
            {
                "model": self._model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": images,
                    }
                ],
                "stream": False,
                "options": {"temperature": DEFAULT_TEMPERATURE},
            },
        )
        try:
            response = await self._ollama_service.chat(chat_request)
        except Exception as exc:
            raise AIProviderError(
                f"Ollama VLM request failed: {exc}",
                code="vlm.ollama_request_failed",
            ) from exc
        message = response.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise AIProviderError(
                "Ollama VLM returned no text content",
                code="vlm.invalid_response",
            )
        return content


def new_ollama_vlm(
    *,
    model_name: str,
    model_id: str,
    ollama_service: OllamaChatService | None,
) -> OllamaVLM:
    """Create an Ollama-backed VLM instance."""
    if ollama_service is None:
        raise AIProviderError(
            "Ollama service is required for a local vision model",
            code="vlm.ollama_service_missing",
        )
    return OllamaVLM(
        model_name=model_name,
        model_id=model_id,
        ollama_service=ollama_service,
    )


__all__ = [
    "DEFAULT_TEMPERATURE",
    "OllamaChatService",
    "OllamaVLM",
    "new_ollama_vlm",
]
