"""Requesty provider metadata and config validation."""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_REQUESTY,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)
from src.common.exception import ValidationError

# Requesty API base URL.
REQUESTY_BASE_URL = "https://router.requesty.ai/v1"


def info() -> ProviderInfo:
    """Provider metadata for the Requesty catalog entry."""
    return ProviderInfo(
        name=PROVIDER_REQUESTY,
        display_name="Requesty",
        description="openai/gpt-4o-mini, anthropic/claude-sonnet-4-5, etc.",
        default_urls={
            ModelType.KNOWLEDGE_QA: REQUESTY_BASE_URL,
            ModelType.EMBEDDING: REQUESTY_BASE_URL,
            ModelType.VLLM: REQUESTY_BASE_URL,
        },
        model_types=[
            ModelType.KNOWLEDGE_QA,
            ModelType.EMBEDDING,
            ModelType.VLLM,
        ],
        requires_auth=True,
    )


def validate_config(config: ProviderConfig) -> None:
    """Validate a Requesty provider configuration."""
    if not config.api_key:
        raise ValidationError("API key is required for Requesty provider")


__all__ = ["REQUESTY_BASE_URL", "info", "validate_config"]
