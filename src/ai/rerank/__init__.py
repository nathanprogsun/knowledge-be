"""Rerank service package: base, OpenAI-compatible backend, and transport.

Public surface: the ``Reranker`` / ``CustomHeaderSetter`` interfaces,
``RerankerConfig`` + ``config_from_model``, the ``new_reranker`` factory,
the OpenAI-compatible ``OpenAIReranker`` and its ``new_openai_reranker``
constructor, the wire models (``RankResult`` / ``RerankResponse`` /
``RerankRequest`` / ``UsageInfo`` / ``DocumentInfo``), and the SSRF-safe
transport helpers.
"""

from __future__ import annotations

from src.ai.rerank.base import (
    CustomHeaderSetter,
    ModelLike,
    Reranker,
    RerankerConfig,
    config_from_model,
    new_reranker,
)
from src.ai.rerank.remote_api import (
    DocumentInfo,
    OpenAIReranker,
    RankResult,
    RerankRequest,
    RerankResponse,
    UsageInfo,
    new_openai_reranker,
)
from src.ai.rerank.transport import (
    new_rerank_http_client,
    post_json_with_ssrf_safety,
    validate_rerank_base_url,
)

__all__ = [
    "CustomHeaderSetter",
    "DocumentInfo",
    "ModelLike",
    "OpenAIReranker",
    "RankResult",
    "RerankRequest",
    "RerankResponse",
    "Reranker",
    "RerankerConfig",
    "UsageInfo",
    "config_from_model",
    "new_openai_reranker",
    "new_rerank_http_client",
    "new_reranker",
    "post_json_with_ssrf_safety",
    "validate_rerank_base_url",
]
