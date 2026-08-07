"""Qiniu provider metadata and config validation."""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_QINIU,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)
from src.common.exception import ValidationError

# Qiniu API base URL (OpenAI-compatible mode).
QINIU_BASE_URL = "https://api.qnaigc.com/v1"


def info() -> ProviderInfo:
    """Provider metadata for the Qiniu catalog entry."""
    return ProviderInfo(
        name=PROVIDER_QINIU,
        display_name="七牛云 Qiniu",
        description="deepseek/deepseek-v3.2-251201, z-ai/glm-4.7, etc.",
        default_urls={
            ModelType.KNOWLEDGE_QA: QINIU_BASE_URL,
        },
        model_types=[
            ModelType.KNOWLEDGE_QA,
        ],
        requires_auth=True,
    )


def validate_config(config: ProviderConfig) -> None:
    """Validate a Qiniu provider configuration."""
    if not config.base_url:
        raise ValidationError("base URL is required for Qiniu provider")
    if not config.api_key:
        raise ValidationError("API key is required for Qiniu provider")
    if not config.model_name:
        raise ValidationError("model name is required")


__all__ = ["QINIU_BASE_URL", "info", "validate_config"]
