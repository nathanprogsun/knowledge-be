"""Chat factory: config mapping, source routing, and the client protocol.

``config_from_model`` maps a stored model row to a :class:`ChatConfig` so the
production path (service layer reads a DB row) and the test path (handler
layer builds from a form) share the same field mapping. ``new_chat`` routes by
model source: local models use the Ollama client, remote models use the
provider routing in ``new_remote_chat`` (Anthropic gets the dedicated
Messages-protocol client, everything else the OpenAI-compatible
``RemoteAPIChat``). Both routes are implemented — local models route to the
Ollama client, Anthropic to the Messages-protocol client.
"""

from __future__ import annotations

from src.ai.llm.anthropic import new_anthropic_chat
from src.ai.llm.concurrency import wrap_chat_concurrency
from src.ai.llm.ollama import new_ollama_chat
from src.ai.llm.remote_api import RemoteAPIChat
from src.ai.llm.types import Chat, ChatConfig
from src.ai.provider.detect import detect_provider
from src.ai.provider.registry import PROVIDER_ANTHROPIC
from src.ai.utils.ollama_service import OllamaService
from src.common.exception import ValidationError
from src.core.contracts.infra import Model

#: ``Model.source`` values that route to a chat client family.
MODEL_SOURCE_LOCAL = "local"
MODEL_SOURCE_REMOTE = "remote"


def config_from_model(
    model: Model | None, app_id: str = "", app_secret: str = ""
) -> ChatConfig | None:
    """Map a stored :class:`Model` to a :class:`ChatConfig`.

    Returns ``None`` for a ``None`` model. ``app_id`` / ``app_secret`` are the
    already-decrypted managed-cloud credentials the caller supplies.
    """
    if model is None:
        return None
    params = model.parameters
    return ChatConfig(
        model_id=model.id,
        api_key=params.api_key or "",
        base_url=params.base_url or "",
        model_name=model.name,
        source=model.source,
        provider=params.provider or "",
        max_concurrency=params.max_concurrency or 0,
        extra_config=params.extra_config,
        custom_headers=params.custom_headers,
        app_id=app_id,
        app_secret=app_secret,
    )


def new_remote_chat(config: ChatConfig) -> Chat:
    """Create a remote chat client for ``config``, routing by provider.

    The Anthropic provider uses the dedicated Messages protocol; all other
    providers share the OpenAI-compatible ``RemoteAPIChat``.
    """
    provider_name = config.provider or detect_provider(config.base_url)
    if provider_name == PROVIDER_ANTHROPIC:
        return new_anthropic_chat(config)
    return RemoteAPIChat(config)


def new_chat(config: ChatConfig, ollama_service: OllamaService | None = None) -> Chat:
    """Create a chat client for ``config``, routing by model source.

    ``ollama_service`` is only used by the local (Ollama) route. The returned
    client carries the per-model background concurrency governor.
    """
    source = config.source.lower()
    if source == MODEL_SOURCE_LOCAL:
        return wrap_chat_concurrency(
            new_ollama_chat(config, ollama_service), config.max_concurrency
        )
    if source == MODEL_SOURCE_REMOTE:
        return wrap_chat_concurrency(new_remote_chat(config), config.max_concurrency)
    raise ValidationError(
        code="chat.unsupported_source", message=f"unsupported chat model source: {config.source}"
    )


__all__ = [
    "MODEL_SOURCE_LOCAL",
    "MODEL_SOURCE_REMOTE",
    "config_from_model",
    "new_chat",
    "new_remote_chat",
]
