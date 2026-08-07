"""Novita AI provider metadata and config validation."""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_NOVITA,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)
from src.common.exception import ValidationError

# Novita OpenAI-compatible API base URL.
NOVITA_OPENAI_BASE_URL = "https://api.novita.ai/openai/v1"


def info() -> ProviderInfo:
    """Provider metadata for the Novita catalog entry."""
    return ProviderInfo(
        name=PROVIDER_NOVITA,
        display_name="Novita AI",
        description=(
            "moonshotai/kimi-k2.5, zai-org/glm-5, "
            "minimax/minimax-m2.7, qwen/qwen3-embedding-0.6b, etc."
        ),
        default_urls={
            ModelType.KNOWLEDGE_QA: NOVITA_OPENAI_BASE_URL,
            ModelType.EMBEDDING: NOVITA_OPENAI_BASE_URL,
            ModelType.VLLM: NOVITA_OPENAI_BASE_URL,
        },
        model_types=[
            ModelType.KNOWLEDGE_QA,
            ModelType.EMBEDDING,
            ModelType.VLLM,
        ],
        requires_auth=True,
    )


def validate_config(config: ProviderConfig) -> None:
    """Validate a Novita provider configuration."""
    if not config.api_key:
        raise ValidationError("API key is required for Novita provider")
    if not config.model_name:
        raise ValidationError("model name is required")


__all__ = ["NOVITA_OPENAI_BASE_URL", "info", "validate_config"]
