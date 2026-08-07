"""AI-layer utilities: request signing and the Ollama service client.

``sign_request`` produces the signed ``X-*`` headers for the managed
cloud chat service; ``OllamaService`` manages a local Ollama instance.
"""

from __future__ import annotations

from src.ai.utils.ollama_service import (
    OLLAMA_BASE_URL_ENV,
    OLLAMA_DIAL_FALLBACK_URL,
    OLLAMA_OPTIONAL_ENV,
    PULL_TIMEOUT_SECONDS,
    OllamaModelInfo,
    OllamaService,
    normalize_model_tag,
    resolve_ollama_dial_base_url,
)
from src.ai.utils.signer import sign, sign_request

__all__ = [
    "OLLAMA_BASE_URL_ENV",
    "OLLAMA_DIAL_FALLBACK_URL",
    "OLLAMA_OPTIONAL_ENV",
    "PULL_TIMEOUT_SECONDS",
    "OllamaModelInfo",
    "OllamaService",
    "normalize_model_tag",
    "resolve_ollama_dial_base_url",
    "sign",
    "sign_request",
]
