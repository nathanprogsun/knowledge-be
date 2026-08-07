"""Embedding providers: text vectorization over OpenAI, Ollama, and batches.

Public surface: the ``Embedder`` / ``EmbedderPooler`` protocols, the
``Config`` carrier and ``config_from_model`` mapping, the ``new_embedder``
factory, the OpenAI-compatible, local Ollama, and every remote provider
implementation (Aliyun, Azure OpenAI, Gemini, Jina, NVIDIA, Volcengine,
the signed managed-cloud endpoint, Zhipu), the batch pooler, the
SSRF-safe transport, and the per-model concurrency governor.
"""

from __future__ import annotations

# ``base`` must be imported before the provider modules: base imports every
# provider factory, and each provider imports ``base`` for its protocols,
# so base must be fully initialized first. `# isort: skip` pins this
# dependency order (isort would otherwise sort it after ``aliyun``).
from src.ai.embedding.base import (  # isort: skip
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
from src.ai.embedding.aliyun import AliyunEmbedder, new_aliyun_embedder
from src.ai.embedding.azure_openai import (
    AzureOpenAIEmbedder,
    new_azure_openai_embedder,
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
from src.ai.embedding.gemini import GeminiEmbedder, new_gemini_embedder
from src.ai.embedding.jina import JinaEmbedder, new_jina_embedder
from src.ai.embedding.nvidia import NvidiaEmbedder, new_nvidia_embedder
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
from src.ai.embedding.volcengine import VolcengineEmbedder, new_volcengine_embedder
from src.ai.embedding.weknoracloud import (
    WeKnoraCloudEmbedder,
    new_weknoracloud_embedder,
)
from src.ai.embedding.zhipu import ZhipuEmbedder, new_zhipu_embedder

__all__ = [
    "AliyunEmbedder",
    "AzureOpenAIEmbedder",
    "BatchEmbedder",
    "ConcurrencyEmbedder",
    "ConcurrencyLimiter",
    "Config",
    "Context",
    "CustomHeadersSettable",
    "Embedder",
    "EmbedderPooler",
    "GeminiEmbedder",
    "JinaEmbedder",
    "LocalLimiter",
    "ModelLike",
    "NvidiaEmbedder",
    "OllamaEmbedder",
    "OpenAIEmbedRequest",
    "OpenAIEmbedResponse",
    "OpenAIEmbedder",
    "SupportsDimensionOverride",
    "TaskContext",
    "VolcengineEmbedder",
    "WeKnoraCloudEmbedder",
    "ZhipuEmbedder",
    "apply_custom_headers",
    "config_from_model",
    "gate_named_n",
    "is_reserved_header",
    "new_aliyun_embedder",
    "new_azure_openai_embedder",
    "new_batch_embedder",
    "new_embedder",
    "new_embedding_http_client",
    "new_gemini_embedder",
    "new_jina_embedder",
    "new_nvidia_embedder",
    "new_ollama_embedder",
    "new_openai_embedder",
    "new_volcengine_embedder",
    "new_weknoracloud_embedder",
    "new_zhipu_embedder",
    "set_global_limit",
    "set_governor",
    "validate_embedding_base_url",
    "wrap_embedding_concurrency",
]
