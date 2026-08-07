"""DeepSeek provider metadata and config validation."""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_DEEPSEEK,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)
from src.common.exception import ValidationError

# Official DeepSeek API base URL.
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


def info() -> ProviderInfo:
    """Provider metadata for the DeepSeek catalog entry."""
    return ProviderInfo(
        name=PROVIDER_DEEPSEEK,
        display_name="DeepSeek",
        description="deepseek-chat, deepseek-reasoner, etc.",
        default_urls={
            ModelType.KNOWLEDGE_QA: DEEPSEEK_BASE_URL,
        },
        model_types=[
            ModelType.KNOWLEDGE_QA,
        ],
        requires_auth=True,
    )


def validate_config(config: ProviderConfig) -> None:
    """Validate a DeepSeek provider configuration."""
    if not config.api_key:
        raise ValidationError("API key is required for DeepSeek provider")
    if not config.model_name:
        raise ValidationError("model name is required")


__all__ = ["DEEPSEEK_BASE_URL", "info", "validate_config"]
