"""Vision-language model (VLM) clients.

Public surface: the ``VLM`` protocol, ``Config`` / ``config_from_model``,
the ``new_vlm`` factory, and the three concrete backends (local Ollama,
remote OpenAI-compatible, managed cloud). Image understanding and
multimodal Q&A flow through these clients.
"""

from __future__ import annotations

from src.ai.vlm.base import (
    MODEL_SOURCE_LOCAL,
    MODEL_SOURCE_REMOTE,
    VLM,
    Config,
    ModelLike,
    ModelParametersLike,
    config_from_model,
    new_vlm,
)
from src.ai.vlm.concurrency import ConcurrencyVLM, wrap_vlm_concurrency
from src.ai.vlm.ollama import (
    DEFAULT_TEMPERATURE as OLLAMA_DEFAULT_TEMPERATURE,
)
from src.ai.vlm.ollama import (
    OllamaChatService,
    OllamaVLM,
    new_ollama_vlm,
)
from src.ai.vlm.remote_api import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    RemoteAPIVLM,
    detect_image_mime,
    new_remote_api_vlm,
)
from src.ai.vlm.weknoracloud import (
    MANAGED_CLOUD_CHAT_PATH,
    WeKnoraCloudVLM,
    new_weknoracloud_vlm,
)

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "MANAGED_CLOUD_CHAT_PATH",
    "MODEL_SOURCE_LOCAL",
    "MODEL_SOURCE_REMOTE",
    "OLLAMA_DEFAULT_TEMPERATURE",
    "VLM",
    "ConcurrencyVLM",
    "Config",
    "ModelLike",
    "ModelParametersLike",
    "OllamaChatService",
    "OllamaVLM",
    "RemoteAPIVLM",
    "WeKnoraCloudVLM",
    "config_from_model",
    "detect_image_mime",
    "new_ollama_vlm",
    "new_remote_api_vlm",
    "new_vlm",
    "new_weknoracloud_vlm",
    "wrap_vlm_concurrency",
]
