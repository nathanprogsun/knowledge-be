"""Google Gemini provider metadata and config validation."""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_GEMINI,
    ModelType,
    ProviderConfig,
    ProviderInfo,
)
from src.common.exception import ValidationError

# Google Gemini API base URLs.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_OPENAI_COMPAT_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai"
)


def info() -> ProviderInfo:
    """Provider metadata for the Gemini catalog entry."""
    return ProviderInfo(
        name=PROVIDER_GEMINI,
        display_name="Google Gemini",
        description="gemini-3-flash-preview, gemini-2.5-pro, gemini-embedding-2, etc.",
        default_urls={
            ModelType.KNOWLEDGE_QA: GEMINI_OPENAI_COMPAT_BASE_URL,
            ModelType.EMBEDDING: GEMINI_BASE_URL,
        },
        model_types=[
            ModelType.KNOWLEDGE_QA,
            ModelType.EMBEDDING,
        ],
        requires_auth=True,
    )


def validate_config(config: ProviderConfig) -> None:
    """Validate a Gemini provider configuration."""
    if not config.api_key:
        raise ValidationError("API key is required for Google Gemini provider")
    if not config.model_name:
        raise ValidationError("model name is required")


__all__ = ["GEMINI_BASE_URL", "GEMINI_OPENAI_COMPAT_BASE_URL", "info", "validate_config"]
