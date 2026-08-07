"""MiniMax provider metadata and config validation."""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_MINIMAX,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)
from src.common.exception import ValidationError

# MiniMax API base URLs (international and mainland editions).
MINIMAX_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_CN_BASE_URL = "https://api.minimaxi.com/v1"


def info() -> ProviderInfo:
    """Provider metadata for the MiniMax catalog entry."""
    return ProviderInfo(
        name=PROVIDER_MINIMAX,
        display_name="MiniMax",
        description="MiniMax-M3, MiniMax-M2.7, MiniMax-M2.7-highspeed, etc.",
        default_urls={
            ModelType.KNOWLEDGE_QA: MINIMAX_CN_BASE_URL,
        },
        model_types=[
            ModelType.KNOWLEDGE_QA,
        ],
        requires_auth=True,
    )


def validate_config(config: ProviderConfig) -> None:
    """Validate a MiniMax provider configuration."""
    if not config.api_key:
        raise ValidationError("API key is required for MiniMax provider")
    if not config.model_name:
        raise ValidationError("model name is required")


__all__ = ["MINIMAX_BASE_URL", "MINIMAX_CN_BASE_URL", "info", "validate_config"]
