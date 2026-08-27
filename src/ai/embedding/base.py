"""Embedding base: interface, config, and the provider factory (embedder.go).

This module mirrors the upstream ``embedder.go`` contract. It defines the
``Embedder`` / ``EmbedderPooler`` protocols, the ``Config`` carrier, the
``config_from_model`` mapping from a stored model record, and the
``new_embedder`` factory that routes a config to a provider-backed
implementation.

The factory covers every upstream route: Ollama, Aliyun (multimodal via
DashScope / text via the OpenAI-compatible client), Volcengine, Jina,
Azure OpenAI, NVIDIA, Gemini, Zhipu, the signed managed-cloud endpoint,
and the OpenAI-compatible default.

``src/ai/embedding`` never imports ``core`` / ``db``; the model record is
passed in as a structural protocol so callers in the core layer supply
their own storage row.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from src.ai.embedding.aliyun import new_aliyun_embedder
from src.ai.embedding.azure_openai import new_azure_openai_embedder
from src.ai.embedding.cloud import new_cloud_embedder
from src.ai.embedding.concurrency import wrap_embedding_concurrency
from src.ai.embedding.gemini import new_gemini_embedder
from src.ai.embedding.jina import new_jina_embedder
from src.ai.embedding.nvidia import new_nvidia_embedder
from src.ai.embedding.ollama import new_ollama_embedder
from src.ai.embedding.openai import new_openai_embedder
from src.ai.embedding.volcengine import new_volcengine_embedder
from src.ai.embedding.zhipu import new_zhipu_embedder
from src.ai.provider.detect import detect_provider
from src.ai.provider.registry import (
    PROVIDER_ALIYUN,
    PROVIDER_AZURE_OPENAI,
    PROVIDER_CLOUD,
    PROVIDER_GEMINI,
    PROVIDER_JINA,
    PROVIDER_NVIDIA,
    PROVIDER_VOLCENGINE,
    PROVIDER_ZHIPU,
)
from src.ai.utils.ollama_service import OllamaService
from src.common.exception import ValidationError
from src.common.json import JsonValue

_SOURCE_LOCAL = "local"
_SOURCE_REMOTE = "remote"


class Context(Protocol):
    """Opaque task context threaded through provider calls.

    The per-model concurrency governor reads ``is_background_task`` to
    decide whether an embedding call is throttled (the upstream
    ``types.IsBackgroundTask`` marker). Interactive / request-scoped
    calls leave it ``False``; background ingestion tasks set it ``True``.
    """

    @property
    def is_background_task(self) -> bool: ...


class EmbedderPooler(Protocol):
    """Fan-out batch dispatcher (upstream ``EmbedderPooler``).

    ``batch_embed_with_pool`` splits a large text list into sub-batches
    and dispatches each through the model's ``batch_embed`` under a
    bounded worker pool.
    """

    async def batch_embed_with_pool(
        self,
        ctx: Context,
        model: Embedder,
        texts: list[str],
    ) -> list[list[float]]: ...


class Embedder(Protocol):
    """Text vectorization interface (upstream ``Embedder``).

    ``Embed`` is a convenience wrapper over ``BatchEmbed``; the batch
    variants accept multiple texts in a single provider round-trip.
    ``BatchEmbedWithPool`` is inherited from ``EmbedderPooler`` and
    threads the calling embedder down so sub-batch calls stay gated.
    """

    async def embed(self, ctx: Context, text: str) -> list[float]: ...

    async def batch_embed(self, ctx: Context, texts: list[str]) -> list[list[float]]: ...

    def get_model_name(self) -> str: ...

    def get_dimensions(self) -> int: ...

    def get_model_id(self) -> str: ...

    async def batch_embed_with_pool(
        self,
        ctx: Context,
        model: Embedder,
        texts: list[str],
    ) -> list[list[float]]: ...


@runtime_checkable
class SupportsDimensionOverride(Protocol):
    """Providers that accept a runtime dimension-override flag.

    The factory calls ``set_supports_dimension_override`` on every
    embedder that exposes it so the configured value reaches the
    request-builder (``dimensions`` is only sent when supported).
    """

    def set_supports_dimension_override(self, supported: bool) -> None: ...


@runtime_checkable
class CustomHeadersSettable(Protocol):
    """Providers that attach per-request custom HTTP headers.

    Reserved headers (``Authorization``, ``Content-Type``, ...) are
    skipped by the transport so custom values cannot break auth.
    """

    def set_custom_headers(self, headers: Mapping[str, str] | None) -> None: ...


@dataclass(frozen=True, slots=True)
class Config:
    """Embedder configuration (upstream ``embedding.Config``).

    ``max_concurrency`` caps concurrent background calls to this model;
    ``0`` falls back to the process-wide default governor limit. ``app_id``
    / ``app_secret`` are the (already-decrypted) managed-cloud credentials.
    """

    source: str = ""
    base_url: str = ""
    model_name: str = ""
    api_key: str = ""
    truncate_prompt_tokens: int = 0
    dimensions: int = 0
    supports_dimension_override: bool = False
    model_id: str = ""
    provider: str = ""
    max_concurrency: int = 0
    extra_config: Mapping[str, str] = field(default_factory=dict)
    custom_headers: Mapping[str, str] = field(default_factory=dict)
    app_id: str = ""
    app_secret: str = ""


class ModelLike(Protocol):
    """A stored model record the embedder config derives from.

    Structurally satisfied by the storage ``Model`` row (``parameters``
    is a JSON object). Kept as a protocol so this layer never imports
    the storage or core-contract modules. ``parameters`` is exposed as a
    read-only property so concrete ``dict``-typed columns satisfy the
    covariant ``Mapping`` return.
    """

    id: str
    name: str
    source: str

    @property
    def parameters(self) -> Mapping[str, JsonValue]: ...


# ── JSON projection helpers ──────────────────────────────────────────


def _as_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


def _as_int(value: JsonValue) -> int:
    return value if isinstance(value, int) else 0


def _as_bool(value: JsonValue) -> bool:
    return value if isinstance(value, bool) else False


def _as_str_map(value: JsonValue) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        key: item for key, item in value.items() if isinstance(key, str) and isinstance(item, str)
    }


def config_from_model(
    model: ModelLike | None,
    app_id: str = "",
    app_secret: str = "",
) -> Config:
    """Map a stored model record to an embedding ``Config``.

    Mirrors the upstream ``ConfigFromModel``: production (DB-loaded) and
    test-connection paths share this mapping. ``app_id`` / ``app_secret``
    are the managed-cloud credentials, supplied by the caller.
    """
    if model is None:
        return Config()
    params = model.parameters
    embedding_params = params.get("embedding_parameters")
    if not isinstance(embedding_params, dict):
        embedding_params = {}
    return Config(
        source=model.source,
        base_url=_as_str(params.get("base_url")),
        api_key=_as_str(params.get("api_key")),
        model_id=model.id,
        model_name=model.name,
        dimensions=_as_int(embedding_params.get("dimension")),
        supports_dimension_override=_as_bool(embedding_params.get("supports_dimension_override")),
        truncate_prompt_tokens=_as_int(embedding_params.get("truncate_prompt_tokens")),
        provider=_as_str(params.get("provider")),
        max_concurrency=_as_int(params.get("max_concurrency")),
        extra_config=_as_str_map(params.get("extra_config")),
        custom_headers=_as_str_map(params.get("custom_headers")),
        app_id=app_id or _as_str(params.get("app_id")),
        app_secret=app_secret or _as_str(params.get("app_secret")),
    )


# ── Factory ──────────────────────────────────────────────────────────

_ALIYUN_MULTIMODAL_KEYWORDS = ("vision", "multimodal")
_ALIYUN_TEXT_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_ALIYUN_MULTIMODAL_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com"
_AZURE_DEFAULT_API_VERSION = "2024-10-21"


def _apply_custom_headers(embedder: Embedder, config: Config) -> Embedder:
    """Attach user-supplied custom headers when the embedder accepts them.

    Mirrors the upstream factory's ``SetCustomHeaders`` calls; the
    transport skips reserved names so auth headers cannot be overridden.
    """
    if isinstance(embedder, CustomHeadersSettable):
        embedder.set_custom_headers(config.custom_headers)
    return embedder


def _is_aliyun_multimodal_model(model_name: str) -> bool:
    """True for Aliyun multimodal embedding models (``embedder.go``).

    Multimodal models (``tongyi-embedding-vision-*`` / ``multimodal-embedding-*``)
    go through the dedicated DashScope endpoint; pure text models reuse
    the OpenAI-compatible interface.
    """
    lowered = model_name.lower()
    return any(keyword in lowered for keyword in _ALIYUN_MULTIMODAL_KEYWORDS)


async def _route_aliyun(config: Config, pooler: EmbedderPooler | None) -> Embedder:
    """Aliyun dispatch: multimodal -> DashScope, text -> OpenAI-compatible."""
    if _is_aliyun_multimodal_model(config.model_name):
        base_url = config.base_url
        if base_url == "":
            base_url = _ALIYUN_MULTIMODAL_DEFAULT_BASE_URL
        elif "/compatible-mode/" in base_url:
            base_url = base_url.replace("/compatible-mode/v1", "", 1)
            base_url = base_url.replace("/compatible-mode", "", 1)
        embedder: Embedder = await new_aliyun_embedder(
            api_key=config.api_key,
            base_url=base_url,
            model_name=config.model_name,
            truncate_prompt_tokens=config.truncate_prompt_tokens,
            dimensions=config.dimensions,
            model_id=config.model_id,
            pooler=pooler,
        )
        return _apply_custom_headers(embedder, config)
    base_url = config.base_url
    if base_url == "" or "/compatible-mode/" not in base_url:
        base_url = _ALIYUN_TEXT_DEFAULT_BASE_URL
    embedder = await new_openai_embedder(
        api_key=config.api_key,
        base_url=base_url,
        model_name=config.model_name,
        truncate_prompt_tokens=config.truncate_prompt_tokens,
        dimensions=config.dimensions,
        model_id=config.model_id,
        pooler=pooler,
    )
    return _apply_custom_headers(embedder, config)


async def new_embedder(
    config: Config,
    pooler: EmbedderPooler | None,
    ollama_service: OllamaService | None,
) -> Embedder:
    """Build an embedder for ``config`` (upstream ``NewEmbedder``).

    Applies the dimension-override flag when the provider supports it,
    then installs the per-model concurrency governor around the real
    provider so background calls stay bounded. Observability decorators
    (debug / tracing wraps) are deferred.
    """
    embedder = await _new_embedder(config, pooler, ollama_service)
    if isinstance(embedder, SupportsDimensionOverride):
        embedder.set_supports_dimension_override(config.supports_dimension_override)
    return wrap_embedding_concurrency(embedder, config.max_concurrency)


async def _new_embedder(
    config: Config,
    pooler: EmbedderPooler | None,
    ollama_service: OllamaService | None,
) -> Embedder:
    """Route a config to a provider-backed embedder (upstream ``newEmbedder``)."""
    source = config.source.strip().lower()
    if source == _SOURCE_LOCAL:
        return await new_ollama_embedder(
            base_url=config.base_url,
            model_name=config.model_name,
            truncate_prompt_tokens=config.truncate_prompt_tokens,
            dimensions=config.dimensions,
            model_id=config.model_id,
            pooler=pooler,
            ollama_service=ollama_service,
        )
    if source == _SOURCE_REMOTE:
        return await _route_remote(config, pooler)
    raise ValidationError(
        code="embedding.unsupported_source",
        message=f"unsupported embedder source: {config.source}",
    )


async def _route_remote(
    config: Config,
    pooler: EmbedderPooler | None,
) -> Embedder:
    """Resolve the provider and dispatch the remote route.

    ``config.provider`` wins when set; otherwise the base URL is
    detected (upstream ``provider.ProviderName`` /
    ``provider.DetectProvider``). Every provider route mirrors the
    upstream factory, including Aliyun's two-way dispatch (multimodal
    models via the dedicated DashScope endpoint, text models via the
    OpenAI-compatible client). Providers that are not recognized fall
    back to the OpenAI-compatible embedder.
    """
    provider_name = config.provider
    if not provider_name:
        provider_name = detect_provider(config.base_url)

    if provider_name == PROVIDER_ALIYUN:
        return await _route_aliyun(config, pooler)

    if provider_name == PROVIDER_VOLCENGINE:
        embedder: Embedder = await new_volcengine_embedder(
            api_key=config.api_key,
            base_url=config.base_url,
            model_name=config.model_name,
            truncate_prompt_tokens=config.truncate_prompt_tokens,
            dimensions=config.dimensions,
            model_id=config.model_id,
            pooler=pooler,
        )
        return _apply_custom_headers(embedder, config)

    if provider_name == PROVIDER_JINA:
        embedder = await new_jina_embedder(
            api_key=config.api_key,
            base_url=config.base_url,
            model_name=config.model_name,
            dimensions=config.dimensions,
            model_id=config.model_id,
            pooler=pooler,
        )
        return _apply_custom_headers(embedder, config)

    if provider_name == PROVIDER_AZURE_OPENAI:
        api_version = _AZURE_DEFAULT_API_VERSION
        configured = config.extra_config.get("api_version")
        if configured:
            api_version = configured
        embedder = await new_azure_openai_embedder(
            api_key=config.api_key,
            base_url=config.base_url,
            model_name=config.model_name,
            truncate_prompt_tokens=config.truncate_prompt_tokens,
            dimensions=config.dimensions,
            model_id=config.model_id,
            api_version=api_version,
            pooler=pooler,
        )
        return _apply_custom_headers(embedder, config)

    if provider_name == PROVIDER_NVIDIA:
        embedder = await new_nvidia_embedder(
            api_key=config.api_key,
            base_url=config.base_url,
            model_name=config.model_name,
            dimensions=config.dimensions,
            model_id=config.model_id,
            pooler=pooler,
        )
        return _apply_custom_headers(embedder, config)

    if provider_name == PROVIDER_GEMINI:
        embedder = await new_gemini_embedder(
            api_key=config.api_key,
            base_url=config.base_url,
            model_name=config.model_name,
            truncate_prompt_tokens=config.truncate_prompt_tokens,
            dimensions=config.dimensions,
            model_id=config.model_id,
            pooler=pooler,
        )
        return _apply_custom_headers(embedder, config)

    if provider_name == PROVIDER_ZHIPU:
        embedder = await new_zhipu_embedder(
            api_key=config.api_key,
            base_url=config.base_url,
            model_name=config.model_name,
            truncate_prompt_tokens=config.truncate_prompt_tokens,
            dimensions=config.dimensions,
            model_id=config.model_id,
            pooler=pooler,
        )
        return _apply_custom_headers(embedder, config)

    if provider_name == PROVIDER_CLOUD:
        return await new_cloud_embedder(config)

    return await new_openai_embedder(
        api_key=config.api_key,
        base_url=config.base_url,
        model_name=config.model_name,
        truncate_prompt_tokens=config.truncate_prompt_tokens,
        dimensions=config.dimensions,
        model_id=config.model_id,
        pooler=pooler,
        custom_headers=config.custom_headers,
    )


__all__ = [
    "Config",
    "Context",
    "CustomHeadersSettable",
    "Embedder",
    "EmbedderPooler",
    "ModelLike",
    "SupportsDimensionOverride",
    "config_from_model",
    "new_embedder",
]
