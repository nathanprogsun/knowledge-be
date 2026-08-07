"""Tencent Hunyuan provider metadata and config validation."""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_HUNYUAN,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)
from src.common.exception import ValidationError

# Tencent Hunyuan API base URL (OpenAI-compatible mode).
HUNYUAN_BASE_URL = "https://api.hunyuan.cloud.tencent.com/v1"


def info() -> ProviderInfo:
    """Provider metadata for the Hunyuan catalog entry."""
    return ProviderInfo(
        name=PROVIDER_HUNYUAN,
        display_name="腾讯混元 Hunyuan",
        description="hunyuan-pro, hunyuan-standard, hunyuan-embedding, etc.",
        default_urls={
            ModelType.KNOWLEDGE_QA: HUNYUAN_BASE_URL,
            ModelType.EMBEDDING: HUNYUAN_BASE_URL,
        },
        model_types=[
            ModelType.KNOWLEDGE_QA,
            ModelType.EMBEDDING,
        ],
        requires_auth=True,
    )


def validate_config(config: ProviderConfig) -> None:
    """Validate a Hunyuan provider configuration."""
    if not config.api_key:
        raise ValidationError("API key is required for Hunyuan provider")
    if not config.model_name:
        raise ValidationError("model name is required")


__all__ = ["HUNYUAN_BASE_URL", "info", "validate_config"]
