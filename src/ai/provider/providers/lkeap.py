"""Tencent Cloud LKEAP provider metadata and config validation."""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_LKEAP,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)
from src.common.exception import ValidationError

# LKEAP (Knowledge Engine Atomic Capabilities) base URLs. Chat uses the
# OpenAI-compatible endpoint; rerank uses the TC3-signed domain.
LKEAP_BASE_URL = "https://api.lkeap.cloud.tencent.com/v1"
LKEAP_RERANK_BASE_URL = "https://lkeap.tencentcloudapi.com"


def info() -> ProviderInfo:
    """Provider metadata for the LKEAP catalog entry."""
    return ProviderInfo(
        name=PROVIDER_LKEAP,
        display_name="腾讯云 LKEAP",
        description="DeepSeek-R1, DeepSeek-V3, lke-reranker-base 等",
        default_urls={
            ModelType.KNOWLEDGE_QA: LKEAP_BASE_URL,
            ModelType.RERANK: LKEAP_RERANK_BASE_URL,
        },
        model_types=[
            ModelType.KNOWLEDGE_QA,
            ModelType.RERANK,
        ],
        requires_auth=True,
    )


def validate_config(config: ProviderConfig) -> None:
    """Validate an LKEAP provider configuration."""
    if not config.api_key:
        raise ValidationError("API key is required for LKEAP provider")
    if not config.model_name:
        raise ValidationError("model name is required")


__all__ = ["LKEAP_BASE_URL", "LKEAP_RERANK_BASE_URL", "info", "validate_config"]
