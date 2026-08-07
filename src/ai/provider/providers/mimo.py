"""Xiaomi MiMo provider metadata and config validation."""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_MIMO,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)
from src.common.exception import ValidationError

# Xiaomi MiMo API base URL.
MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"


def info() -> ProviderInfo:
    """Provider metadata for the MiMo catalog entry."""
    return ProviderInfo(
        name=PROVIDER_MIMO,
        display_name="小米 MiMo",
        description="mimo-v2-flash",
        default_urls={
            ModelType.KNOWLEDGE_QA: MIMO_BASE_URL,
        },
        model_types=[
            ModelType.KNOWLEDGE_QA,
        ],
        requires_auth=True,
    )


def validate_config(config: ProviderConfig) -> None:
    """Validate a MiMo provider configuration."""
    if not config.api_key:
        raise ValidationError("API key is required for Mimo provider")
    if not config.model_name:
        raise ValidationError("model name is required")


__all__ = ["MIMO_BASE_URL", "info", "validate_config"]
