"""Volcengine Ark provider metadata and config validation."""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_VOLCENGINE,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)
from src.common.exception import ValidationError

# Volcengine Ark API base URLs (chat, multimodal embedding, rerank).
VOLCENGINE_CHAT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
VOLCENGINE_EMBEDDING_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/embeddings/multimodal"
VOLCENGINE_RERANK_BASE_URL = "https://api-knowledgebase.mlp.cn-beijing.volces.com"


def info() -> ProviderInfo:
    """Provider metadata for the Volcengine catalog entry."""
    return ProviderInfo(
        name=PROVIDER_VOLCENGINE,
        display_name="火山引擎 Volcengine",
        description=(
            "doubao-1-5-pro-32k-250115, doubao-embedding-vision-250615, doubao-seed-rerank, etc."
        ),
        default_urls={
            ModelType.KNOWLEDGE_QA: VOLCENGINE_CHAT_BASE_URL,
            ModelType.EMBEDDING: VOLCENGINE_EMBEDDING_BASE_URL,
            ModelType.RERANK: VOLCENGINE_RERANK_BASE_URL,
            ModelType.VLLM: VOLCENGINE_CHAT_BASE_URL,
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
    """Validate a Volcengine provider configuration."""
    if not config.api_key:
        raise ValidationError("API key is required for Volcengine Ark provider")
    if not config.model_name:
        raise ValidationError("model name is required")


__all__ = [
    "VOLCENGINE_CHAT_BASE_URL",
    "VOLCENGINE_EMBEDDING_BASE_URL",
    "VOLCENGINE_RERANK_BASE_URL",
    "info",
    "validate_config",
]
