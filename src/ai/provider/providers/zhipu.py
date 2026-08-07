"""Zhipu AI provider metadata and config validation."""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_ZHIPU,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)
from src.common.exception import ValidationError

# Zhipu AI base URLs for chat, embedding and rerank.
ZHIPU_CHAT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
ZHIPU_EMBEDDING_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
ZHIPU_RERANK_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/rerank"


def info() -> ProviderInfo:
    """Provider metadata for the Zhipu catalog entry."""
    return ProviderInfo(
        name=PROVIDER_ZHIPU,
        display_name="智谱 BigModel",
        description="glm-4.7, embedding-3, rerank, etc.",
        default_urls={
            ModelType.KNOWLEDGE_QA: ZHIPU_CHAT_BASE_URL,
            ModelType.EMBEDDING: ZHIPU_EMBEDDING_BASE_URL,
            ModelType.RERANK: ZHIPU_RERANK_BASE_URL,
            ModelType.VLLM: ZHIPU_CHAT_BASE_URL,
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
    """Validate a Zhipu provider configuration."""
    if not config.api_key:
        raise ValidationError("API key is required for Zhipu AI")
    if not config.model_name:
        raise ValidationError("model name is required")


__all__ = [
    "ZHIPU_CHAT_BASE_URL",
    "ZHIPU_EMBEDDING_BASE_URL",
    "ZHIPU_RERANK_BASE_URL",
    "info",
    "validate_config",
]
