"""Generic OpenAI-compatible provider metadata and config validation."""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_GENERIC,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)
from src.common.exception import ValidationError


def info() -> ProviderInfo:
    """Provider metadata for the generic (custom endpoint) entry."""
    return ProviderInfo(
        name=PROVIDER_GENERIC,
        display_name="自定义 (OpenAI兼容接口)",
        description="Generic API endpoint (OpenAI-compatible)",
        # The caller configures every endpoint themselves.
        default_urls={},
        model_types=[
            ModelType.KNOWLEDGE_QA,
            ModelType.EMBEDDING,
            ModelType.RERANK,
            ModelType.VLLM,
            ModelType.ASR,
        ],
        requires_auth=False,
    )


def validate_config(config: ProviderConfig) -> None:
    """Validate a generic provider configuration."""
    if not config.base_url:
        raise ValidationError("base URL is required for generic provider")
    if not config.model_name:
        raise ValidationError("model name is required")


__all__ = ["info", "validate_config"]
