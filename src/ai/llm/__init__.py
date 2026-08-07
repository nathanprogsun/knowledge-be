"""LLM chat layer: contracts, provider adapters, remote API client, and limiter.

Public surface: the ``Chat`` protocol, the request / response types, the
``new_chat`` / ``new_remote_chat`` factories, ``RemoteAPIChat``, and the
concurrency limiter. Downstream provider PRs add the Anthropic / Ollama /
OpenAI-specific request and stream implementations on top of this package.
"""

from __future__ import annotations

from src.ai.llm.base import (
    MODEL_SOURCE_LOCAL,
    MODEL_SOURCE_REMOTE,
    config_from_model,
    new_chat,
    new_remote_chat,
)
from src.ai.llm.concurrency import ConcurrencyChat, wrap_chat_concurrency
from src.ai.llm.limiter import (
    DEFAULT_LEASE_TTL,
    DEFAULT_POLL_INTERVAL,
    KEY_PREFIX,
    LocalLimiter,
    ModelConcurrencyLimiter,
    RedisLimiter,
    RuntimeStat,
    background_task_context,
    gate,
    gate_n,
    gate_named_n,
    is_background_task,
    runtime_stats,
    set_global_limit,
    set_governor,
)
from src.ai.llm.ollama import OllamaChat, new_ollama_chat
from src.ai.llm.remote_api import RemoteAPIChat, remove_thinking_content
from src.ai.llm.types import (
    Chat,
    ChatConfig,
    ChatOptions,
    ChatResponse,
    FunctionCall,
    FunctionDef,
    ImageURL,
    LLMToolCall,
    Message,
    MessageContentPart,
    PromptCacheStatus,
    References,
    ResponseType,
    SearchResult,
    StreamResponse,
    TokenUsage,
    Tool,
    ToolCall,
    ToolCallMetadata,
)

__all__ = [
    "DEFAULT_LEASE_TTL",
    "DEFAULT_POLL_INTERVAL",
    "KEY_PREFIX",
    "MODEL_SOURCE_LOCAL",
    "MODEL_SOURCE_REMOTE",
    "Chat",
    "ChatConfig",
    "ChatOptions",
    "ChatResponse",
    "ConcurrencyChat",
    "FunctionCall",
    "FunctionDef",
    "ImageURL",
    "LLMToolCall",
    "LocalLimiter",
    "Message",
    "MessageContentPart",
    "ModelConcurrencyLimiter",
    "OllamaChat",
    "PromptCacheStatus",
    "RedisLimiter",
    "References",
    "RemoteAPIChat",
    "ResponseType",
    "RuntimeStat",
    "SearchResult",
    "StreamResponse",
    "TokenUsage",
    "Tool",
    "ToolCall",
    "ToolCallMetadata",
    "background_task_context",
    "config_from_model",
    "gate",
    "gate_n",
    "gate_named_n",
    "is_background_task",
    "new_chat",
    "new_ollama_chat",
    "new_remote_chat",
    "remove_thinking_content",
    "runtime_stats",
    "set_global_limit",
    "set_governor",
    "wrap_chat_concurrency",
]
