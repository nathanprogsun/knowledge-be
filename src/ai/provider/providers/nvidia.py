"""NVIDIA provider metadata and config validation."""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_NVIDIA,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)
from src.common.exception import ValidationError

# NVIDIA API base URLs for chat/embedding and rerank.
NVIDIA_CHAT_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_RERANK_BASE_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"


def info() -> ProviderInfo:
    """Provider metadata for the NVIDIA catalog entry."""
    return ProviderInfo(
        name=PROVIDER_NVIDIA,
        display_name="NVIDIA",
        description="deepseek-ai-deepseek-v3_1, nv-embed-v1, rerank-qa-mistral-4b, etc.",
        default_urls={
            ModelType.KNOWLEDGE_QA: NVIDIA_CHAT_BASE_URL,
            ModelType.EMBEDDING: NVIDIA_CHAT_BASE_URL,
            ModelType.RERANK: NVIDIA_RERANK_BASE_URL,
            ModelType.VLLM: NVIDIA_CHAT_BASE_URL,
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
    """Validate an NVIDIA provider configuration."""
    if not config.api_key:
        raise ValidationError("API key is required for NVIDIA")
    if not config.model_name:
        raise ValidationError("model name is required")


__all__ = ["NVIDIA_CHAT_BASE_URL", "NVIDIA_RERANK_BASE_URL", "info", "validate_config"]
