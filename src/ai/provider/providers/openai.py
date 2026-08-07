"""OpenAI provider metadata and config validation."""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_OPENAI,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)
from src.common.exception import ValidationError

# OpenAI API base URL.
OPENAI_BASE_URL = "https://api.openai.com/v1"


def info() -> ProviderInfo:
    """Provider metadata for the OpenAI catalog entry."""
    return ProviderInfo(
        name=PROVIDER_OPENAI,
        display_name="OpenAI",
        description="gpt-5.2, gpt-5-mini, etc.",
        default_urls={
            ModelType.KNOWLEDGE_QA: OPENAI_BASE_URL,
            ModelType.EMBEDDING: OPENAI_BASE_URL,
            ModelType.RERANK: OPENAI_BASE_URL,
            ModelType.VLLM: OPENAI_BASE_URL,
            ModelType.ASR: OPENAI_BASE_URL,
        },
        model_types=[
            ModelType.KNOWLEDGE_QA,
            ModelType.EMBEDDING,
            ModelType.RERANK,
            ModelType.VLLM,
            ModelType.ASR,
        ],
        requires_auth=True,
    )


def validate_config(config: ProviderConfig) -> None:
    """Validate an OpenAI provider configuration."""
    if not config.api_key:
        raise ValidationError("API key is required for OpenAI provider")
    if not config.model_name:
        raise ValidationError("model name is required")


__all__ = ["OPENAI_BASE_URL", "info", "validate_config"]
