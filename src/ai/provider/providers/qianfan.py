"""Baidu Qianfan provider metadata and config validation."""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_QIANFAN,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)
from src.common.exception import ValidationError

# Baidu Qianfan API base URL.
QIANFAN_BASE_URL = "https://qianfan.baidubce.com/v2"


def info() -> ProviderInfo:
    """Provider metadata for the Qianfan catalog entry."""
    return ProviderInfo(
        name=PROVIDER_QIANFAN,
        display_name="百度千帆 Baidu Cloud",
        description="ernie-5.0-thinking-preview, embedding-v1, bce-reranker-base, etc.",
        default_urls={
            ModelType.KNOWLEDGE_QA: QIANFAN_BASE_URL,
            ModelType.EMBEDDING: QIANFAN_BASE_URL,
            ModelType.RERANK: QIANFAN_BASE_URL,
            ModelType.VLLM: QIANFAN_BASE_URL,
        },
        model_types=[
            ModelType.KNOWLEDGE_QA,
            ModelType.EMBEDDING,
            ModelType.RERANK,
            ModelType.VLLM,
        ],
        requires_auth=True,
    )


def validate_config(config: ProviderConfig) -> None:
    """Validate a Qianfan provider configuration."""
    if not config.base_url:
        raise ValidationError("base URL is required for Qianfan provider")
    if not config.api_key:
        raise ValidationError("API key is required for Qianfan provider")
    if not config.model_name:
        raise ValidationError("model name is required")


__all__ = ["QIANFAN_BASE_URL", "info", "validate_config"]
