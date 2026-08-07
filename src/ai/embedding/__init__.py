"""Embedding providers: text vectorization over OpenAI, Ollama, and batches.

Public surface: the ``Embedder`` / ``EmbedderPooler`` protocols, the
``Config`` carrier and ``config_from_model`` mapping, the ``new_embedder``
factory, the OpenAI-compatible and local Ollama implementations, the batch
pooler, the SSRF-safe transport, and the per-model concurrency governor.
"""

from __future__ import annotations

from src.ai.embedding.base import (
    Config,
    Context,
    CustomHeadersSettable,
    Embedder,
    EmbedderPooler,
    ModelLike,
    SupportsDimensionOverride,
    config_from_model,
    new_embedder,
)
from src.ai.embedding.batch import BatchEmbedder, new_batch_embedder
from src.ai.embedding.concurrency import (
    ConcurrencyEmbedder,
    ConcurrencyLimiter,
    LocalLimiter,
    TaskContext,
    gate_named_n,
    set_global_limit,
    set_governor,
    wrap_embedding_concurrency,
)
from src.ai.embedding.ollama import OllamaEmbedder, new_ollama_embedder
from src.ai.embedding.openai import (
    OpenAIEmbedder,
    OpenAIEmbedRequest,
    OpenAIEmbedResponse,
    new_openai_embedder,
)
from src.ai.embedding.transport import (
    apply_custom_headers,
    is_reserved_header,
    new_embedding_http_client,
    validate_embedding_base_url,
)

__all__ = [
    "BatchEmbedder",
    "ConcurrencyEmbedder",
    "ConcurrencyLimiter",
    "Config",
    "Context",
    "CustomHeadersSettable",
    "Embedder",
    "EmbedderPooler",
    "LocalLimiter",
    "ModelLike",
    "OllamaEmbedder",
    "OpenAIEmbedRequest",
    "OpenAIEmbedResponse",
    "OpenAIEmbedder",
    "SupportsDimensionOverride",
    "TaskContext",
    "apply_custom_headers",
    "config_from_model",
    "gate_named_n",
    "is_reserved_header",
    "new_batch_embedder",
    "new_embedder",
    "new_embedding_http_client",
    "new_ollama_embedder",
    "new_openai_embedder",
    "set_global_limit",
    "set_governor",
    "validate_embedding_base_url",
    "wrap_embedding_concurrency",
]
