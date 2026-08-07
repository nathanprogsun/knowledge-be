"""ModelScope provider metadata and config validation."""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_MODELSCOPE,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)
from src.common.exception import ValidationError

# ModelScope API base URL (OpenAI-compatible mode).
MODEL_SCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"


def info() -> ProviderInfo:
    """Provider metadata for the ModelScope catalog entry."""
    return ProviderInfo(
        name=PROVIDER_MODELSCOPE,
        display_name="魔搭 ModelScope",
        description="Qwen/Qwen3-8B, Qwen/Qwen3-Embedding-8B, etc.",
        default_urls={
            ModelType.KNOWLEDGE_QA: MODEL_SCOPE_BASE_URL,
            ModelType.EMBEDDING: MODEL_SCOPE_BASE_URL,
            ModelType.VLLM: MODEL_SCOPE_BASE_URL,
        },
        model_types=[
            ModelType.KNOWLEDGE_QA,
            ModelType.EMBEDDING,
            ModelType.VLLM,
        ],
        requires_auth=True,
    )


def validate_config(config: ProviderConfig) -> None:
    """Validate a ModelScope provider configuration."""
    if not config.base_url:
        raise ValidationError("base URL is required for ModelScope provider")
    if not config.api_key:
        raise ValidationError("API key is required for ModelScope provider")
    if not config.model_name:
        raise ValidationError("model name is required")


__all__ = ["MODEL_SCOPE_BASE_URL", "info", "validate_config"]
