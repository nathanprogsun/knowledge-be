"""Jina AI provider metadata and config validation."""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_JINA,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)
from src.common.exception import ValidationError

# Jina AI API base URL.
JINA_BASE_URL = "https://api.jina.ai/v1"


def info() -> ProviderInfo:
    """Provider metadata for the Jina catalog entry."""
    return ProviderInfo(
        name=PROVIDER_JINA,
        display_name="Jina",
        description="jina-clip-v1, jina-embeddings-v2-base-zh, etc.",
        default_urls={
            ModelType.EMBEDDING: JINA_BASE_URL,
            ModelType.RERANK: JINA_BASE_URL,
        },
        model_types=[
            ModelType.EMBEDDING,
            ModelType.RERANK,
        ],
        requires_auth=True,
    )


def validate_config(config: ProviderConfig) -> None:
    """Validate a Jina provider configuration."""
    if not config.api_key:
        raise ValidationError("API key is required for Jina AI provider")


__all__ = ["JINA_BASE_URL", "info", "validate_config"]
