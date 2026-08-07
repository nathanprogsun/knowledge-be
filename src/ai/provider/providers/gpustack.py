"""GPUStack provider metadata and config validation."""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_GPUSTACK,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)
from src.common.exception import ValidationError

# GPUStack API base URLs (OpenAI-compatible mode).
GPU_STACK_BASE_URL = "http://your_gpustack_server_url/v1-openai"
# Rerank uses ``/v1/rerank`` rather than the ``/v1-openai/rerank`` path.
GPU_STACK_RERANK_BASE_URL = "http://your_gpustack_server_url/v1"


def info() -> ProviderInfo:
    """Provider metadata for the GPUStack catalog entry."""
    return ProviderInfo(
        name=PROVIDER_GPUSTACK,
        display_name="GPUStack",
        description="Choose your deployed model on GPUStack",
        default_urls={
            ModelType.KNOWLEDGE_QA: GPU_STACK_BASE_URL,
            ModelType.EMBEDDING: GPU_STACK_BASE_URL,
            ModelType.RERANK: GPU_STACK_RERANK_BASE_URL,
            ModelType.VLLM: GPU_STACK_BASE_URL,
            ModelType.ASR: GPU_STACK_BASE_URL,
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
    """Validate a GPUStack provider configuration."""
    if not config.base_url:
        raise ValidationError("base URL is required for GPUStack provider")
    if not config.api_key:
        raise ValidationError("API key is required for GPUStack provider")
    if not config.model_name:
        raise ValidationError("model name is required")


__all__ = ["GPU_STACK_BASE_URL", "GPU_STACK_RERANK_BASE_URL", "info", "validate_config"]
