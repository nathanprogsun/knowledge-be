"""OpenRouter provider metadata and config validation."""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_OPENROUTER,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)
from src.common.exception import ValidationError

# OpenRouter API base URL.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def info() -> ProviderInfo:
    """Provider metadata for the OpenRouter catalog entry."""
    return ProviderInfo(
        name=PROVIDER_OPENROUTER,
        display_name="OpenRouter",
        description="openai/gpt-5.2-chat, google/gemini-3-flash-preview, etc.",
        default_urls={
            ModelType.KNOWLEDGE_QA: OPENROUTER_BASE_URL,
            ModelType.EMBEDDING: OPENROUTER_BASE_URL,
            ModelType.VLLM: OPENROUTER_BASE_URL,
        },
        model_types=[
            ModelType.KNOWLEDGE_QA,
            ModelType.EMBEDDING,
            ModelType.VLLM,
        ],
        requires_auth=True,
    )


def validate_config(config: ProviderConfig) -> None:
    """Validate an OpenRouter provider configuration."""
    if not config.api_key:
        raise ValidationError("API key is required for OpenRouter provider")


__all__ = ["OPENROUTER_BASE_URL", "info", "validate_config"]
