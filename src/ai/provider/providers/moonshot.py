"""Moonshot (Kimi) provider metadata and config validation."""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_MOONSHOT,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)
from src.common.exception import ValidationError

# Moonshot AI API base URL.
MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"


def info() -> ProviderInfo:
    """Provider metadata for the Moonshot catalog entry."""
    return ProviderInfo(
        name=PROVIDER_MOONSHOT,
        display_name="月之暗面 Moonshot",
        description="kimi-k2-turbo-preview, moonshot-v1-8k-vision-preview, etc.",
        default_urls={
            ModelType.KNOWLEDGE_QA: MOONSHOT_BASE_URL,
            ModelType.VLLM: MOONSHOT_BASE_URL,
        },
        model_types=[
            ModelType.KNOWLEDGE_QA,
            ModelType.VLLM,
        ],
        requires_auth=True,
    )


def validate_config(config: ProviderConfig) -> None:
    """Validate a Moonshot provider configuration."""
    if not config.base_url:
        raise ValidationError("base URL is required for Moonshot provider")
    if not config.api_key:
        raise ValidationError("API key is required for Moonshot provider")
    if not config.model_name:
        raise ValidationError("model name is required")


__all__ = ["MOONSHOT_BASE_URL", "info", "validate_config"]
