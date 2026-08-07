"""Managed cloud provider metadata and config validation."""

# The description below is the upstream UI string and intentionally uses
# full-width punctuation; RUF001 is suppressed for it.
# ruff: noqa: RUF001

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_WEKNORACLOUD,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)

# Hard-coded base URL for the managed cloud service (a single entry point;
# the per-operation path is appended by each implementation).
MANAGED_CLOUD_BASE_URL = "https://weknora.weixin.qq.com"


def info() -> ProviderInfo:
    """Provider metadata for the managed cloud catalog entry."""
    return ProviderInfo(
        name=PROVIDER_WEKNORACLOUD,
        display_name="WeKnoraCloud",
        description="WeKnora云服务，模型：chat, embedding, rerank, vlm",
        default_urls={
            ModelType.KNOWLEDGE_QA: MANAGED_CLOUD_BASE_URL,
            ModelType.EMBEDDING: MANAGED_CLOUD_BASE_URL,
            ModelType.RERANK: MANAGED_CLOUD_BASE_URL,
            ModelType.VLLM: MANAGED_CLOUD_BASE_URL,
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
    """Validate a managed cloud provider configuration.

    App credentials are written through the dedicated initialization
    endpoint; only structural checks apply here (the app-secret field
    currently carries the upstream API key).
    """


__all__ = ["MANAGED_CLOUD_BASE_URL", "info", "validate_config"]
