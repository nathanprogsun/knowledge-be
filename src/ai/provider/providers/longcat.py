"""LongCat AI provider metadata and config validation."""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_LONGCAT,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)
from src.common.exception import ValidationError

# LongCat AI API base URL.
LONGCAT_BASE_URL = "https://api.longcat.chat/openai/v1"


def info() -> ProviderInfo:
    """Provider metadata for the LongCat catalog entry."""
    return ProviderInfo(
        name=PROVIDER_LONGCAT,
        display_name="LongCat AI",
        description="LongCat-Flash-Chat, LongCat-Flash-Thinking, etc.",
        default_urls={
            ModelType.KNOWLEDGE_QA: LONGCAT_BASE_URL,
        },
        model_types=[
            ModelType.KNOWLEDGE_QA,
        ],
        requires_auth=True,
    )


def validate_config(config: ProviderConfig) -> None:
    """Validate a LongCat provider configuration."""
    if not config.base_url:
        raise ValidationError("base URL is required for LongCat provider")
    if not config.api_key:
        raise ValidationError("API key is required for LongCat provider")
    if not config.model_name:
        raise ValidationError("model name is required")


__all__ = ["LONGCAT_BASE_URL", "info", "validate_config"]
