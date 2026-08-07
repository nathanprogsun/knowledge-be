"""Azure OpenAI provider metadata and config validation."""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_AZURE_OPENAI,
    ExtraFieldConfig,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)
from src.common.exception import ValidationError

# ``{resource}`` is the Azure resource name substituted by the caller.
AZURE_OPENAI_RESOURCE_URL = "https://{resource}.openai.azure.com"


def info() -> ProviderInfo:
    """Provider metadata for the Azure OpenAI catalog entry."""
    return ProviderInfo(
        name=PROVIDER_AZURE_OPENAI,
        display_name="Azure OpenAI",
        description="gpt-4o, gpt-4, text-embedding-ada-002, etc.",
        default_urls={
            ModelType.KNOWLEDGE_QA: AZURE_OPENAI_RESOURCE_URL,
            ModelType.EMBEDDING: AZURE_OPENAI_RESOURCE_URL,
            ModelType.RERANK: AZURE_OPENAI_RESOURCE_URL,
            ModelType.VLLM: AZURE_OPENAI_RESOURCE_URL,
            ModelType.ASR: AZURE_OPENAI_RESOURCE_URL,
        },
        model_types=[
            ModelType.KNOWLEDGE_QA,
            ModelType.EMBEDDING,
            ModelType.VLLM,
            ModelType.ASR,
        ],
        requires_auth=True,
        extra_fields=[
            ExtraFieldConfig(
                key="api_version",
                label="API Version",
                type="string",
                required=False,
                default="2024-10-21",
                placeholder="e.g. 2024-10-21",
            ),
        ],
    )


def validate_config(config: ProviderConfig) -> None:
    """Validate an Azure OpenAI provider configuration."""
    if not config.api_key:
        raise ValidationError("API key is required for Azure OpenAI provider")
    if not config.model_name:
        raise ValidationError("deployment name (model name) is required")
    if not config.base_url:
        raise ValidationError("Azure resource endpoint (base URL) is required")


__all__ = ["AZURE_OPENAI_RESOURCE_URL", "info", "validate_config"]
