"""Anthropic provider metadata and config validation."""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_ANTHROPIC,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)
from src.common.exception import ValidationError

# Native Anthropic Messages API base URL.
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"


def info() -> ProviderInfo:
    """Provider metadata for the Anthropic catalog entry."""
    return ProviderInfo(
        name=PROVIDER_ANTHROPIC,
        display_name="Anthropic",
        description="Claude models via native Anthropic Messages API",
        default_urls={
            ModelType.KNOWLEDGE_QA: ANTHROPIC_BASE_URL,
        },
        model_types=[
            ModelType.KNOWLEDGE_QA,
        ],
        requires_auth=True,
    )


def validate_config(config: ProviderConfig) -> None:
    """Validate an Anthropic provider configuration."""
    if not config.api_key:
        raise ValidationError("API key is required for Anthropic provider")
    if not config.model_name:
        raise ValidationError("model name is required")


__all__ = ["ANTHROPIC_BASE_URL", "info", "validate_config"]
