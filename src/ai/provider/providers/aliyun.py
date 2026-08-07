"""Aliyun DashScope provider metadata and config validation."""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_ALIYUN,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)
from src.common.exception import ValidationError

# Default base URLs for the Aliyun DashScope provider.
ALIYUN_CHAT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALIYUN_RERANK_BASE_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)


def info() -> ProviderInfo:
    """Provider metadata for the Aliyun DashScope catalog entry."""
    return ProviderInfo(
        name=PROVIDER_ALIYUN,
        display_name="阿里云 DashScope",
        description="qwen-plus, tongyi-embedding-vision-plus, qwen3-rerank, etc.",
        default_urls={
            ModelType.KNOWLEDGE_QA: ALIYUN_CHAT_BASE_URL,
            ModelType.EMBEDDING: ALIYUN_CHAT_BASE_URL,
            ModelType.RERANK: ALIYUN_RERANK_BASE_URL,
            ModelType.VLLM: ALIYUN_CHAT_BASE_URL,
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
    """Validate an Aliyun provider configuration."""
    if not config.api_key:
        raise ValidationError("API key is required for Aliyun DashScope")
    if not config.model_name:
        raise ValidationError("model name is required")


__all__ = ["ALIYUN_CHAT_BASE_URL", "ALIYUN_RERANK_BASE_URL", "info", "validate_config"]
