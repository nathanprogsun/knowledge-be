"""SiliconFlow provider metadata and config validation."""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_SILICONFLOW,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)
from src.common.exception import ValidationError

# SiliconFlow API base URL.
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"


def info() -> ProviderInfo:
    """Provider metadata for the SiliconFlow catalog entry."""
    return ProviderInfo(
        name=PROVIDER_SILICONFLOW,
        display_name="硅基流动 SiliconFlow",
        description="deepseek-ai/DeepSeek-V3.1, etc.",
        default_urls={
            ModelType.KNOWLEDGE_QA: SILICONFLOW_BASE_URL,
            ModelType.EMBEDDING: SILICONFLOW_BASE_URL,
            ModelType.RERANK: SILICONFLOW_BASE_URL,
            ModelType.VLLM: SILICONFLOW_BASE_URL,
            ModelType.ASR: SILICONFLOW_BASE_URL,
        },
        model_types=[
            ModelType.KNOWLEDGE_QA,
            ModelType.EMBEDDING,
            ModelType.RERANK,
            ModelType.VLLM,
            ModelType.ASR,
        ],
        requires_auth=True,
    )


def validate_config(config: ProviderConfig) -> None:
    """Validate a SiliconFlow provider configuration."""
    if not config.api_key:
        raise ValidationError("API key is required for SiliconFlow provider")


__all__ = ["SILICONFLOW_BASE_URL", "info", "validate_config"]
