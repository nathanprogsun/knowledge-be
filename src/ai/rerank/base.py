"""Rerank service base: interface, config, and provider factory.

``Reranker`` is the provider-agnostic interface, ``RerankerConfig`` the
construction parameters shared by every backend, ``config_from_model``
the model-row-to-config mapping, and ``new_reranker`` the
provider-routing factory. Provider routing mirrors the upstream factory:
the configured ``provider`` name wins, falling back to base-URL
detection; the OpenAI-compatible backend is the default route, and the
dedicated backends (Aliyun, Zhipu, Jina, NVIDIA, managed cloud, LKEAP,
Volcengine) are selected by provider name.

``CustomHeaderSetter`` is the hook the factory uses to attach
user-supplied request headers to the backends that support them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field

from src.ai.provider import (
    PROVIDER_ALIYUN,
    PROVIDER_JINA,
    PROVIDER_LKEAP,
    PROVIDER_NVIDIA,
    PROVIDER_VOLCENGINE,
    PROVIDER_WEKNORACLOUD,
    PROVIDER_ZHIPU,
    detect_provider,
)
from src.ai.rerank.aliyun import new_aliyun_reranker
from src.ai.rerank.jina import new_jina_reranker
from src.ai.rerank.lkeap import new_lkeap_reranker
from src.ai.rerank.nvidia import new_nvidia_reranker
from src.ai.rerank.remote_api import RankResult, new_openai_reranker
from src.ai.rerank.volcengine import new_volcengine_reranker
from src.ai.rerank.weknoracloud import new_weknoracloud_reranker
from src.ai.rerank.zhipu import new_zhipu_reranker
from src.common.json import JsonObject, JsonValue


class RerankerConfig(BaseModel):
    """Construction parameters shared by every reranker backend.

    Mirrors the upstream ``RerankerConfig``: identity fields, credentials,
    the resolved provider name, and the optional extra config / custom
    headers a caller may attach. ``app_id`` / ``app_secret`` are the
    caller-supplied cloud credentials.
    """

    model_config = ConfigDict(frozen=True)

    api_key: str = ""
    base_url: str = ""
    model_name: str = ""
    source: str = ""
    model_id: str = ""
    provider: str = ""
    extra_config: dict[str, str] = Field(default_factory=dict)
    custom_headers: dict[str, str] = Field(default_factory=dict)
    app_id: str = ""
    app_secret: str = ""


@runtime_checkable
class CustomHeaderSetter(Protocol):
    """Marker for rerankers that accept user-supplied HTTP headers."""

    def set_custom_headers(self, headers: Mapping[str, str]) -> None: ...


class Reranker(Protocol):
    """Provider-agnostic reranking interface.

    Implementations are stateless apart from an injected HTTP client and
    raise ``ApplicationError`` subclasses on failure.
    """

    async def rerank(self, query: str, documents: list[str]) -> list[RankResult]: ...

    def get_model_name(self) -> str: ...

    def get_model_id(self) -> str: ...


class ModelLike(Protocol):
    """Structural subset of a model row consumed by ``config_from_model``.

    Satisfied by the storage ``Model``, whose ``parameters`` column is a
    JSON blob — the shape the service layer passes in. Callers holding
    the wire ``Model`` contract (Pydantic-typed parameters) dump
    ``parameters.model_dump()`` before calling, keeping this module free
    of core-layer imports.
    """

    id: str
    name: str
    source: str
    parameters: JsonObject


def _as_str(value: JsonValue | None) -> str:
    return value if isinstance(value, str) else ""


def _as_str_dict(value: JsonValue | None) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, str] = {}
    for key, item in value.items():
        if isinstance(key, str) and item is not None:
            out[key] = str(item)
    return out


def config_from_model(
    model: ModelLike | None,
    app_id: str = "",
    app_secret: str = "",
) -> RerankerConfig | None:
    """Map a model row onto ``RerankerConfig``.

    Mirrors the upstream shared mapping used by both the persistence path
    (row pulled from storage) and the test-connection path (temporary
    form). ``app_id`` / ``app_secret`` are caller-supplied cloud
    credentials; a ``None`` model yields ``None``.
    """
    if model is None:
        return None
    parameters = model.parameters
    return RerankerConfig(
        model_id=model.id,
        api_key=_as_str(parameters.get("api_key")),
        base_url=_as_str(parameters.get("base_url")),
        model_name=model.name,
        source=model.source,
        provider=_as_str(parameters.get("provider")),
        extra_config=_as_str_dict(parameters.get("extra_config")),
        custom_headers=_as_str_dict(parameters.get("custom_headers")),
        app_id=app_id,
        app_secret=app_secret,
    )


async def new_reranker(
    config: RerankerConfig,
    *,
    client: httpx.AsyncClient | None = None,
) -> Reranker:
    """Construct a reranker for ``config``.

    Provider routing mirrors the upstream factory: the configured
    ``provider`` name wins, otherwise the base URL is detected. The
    OpenAI-compatible backend is the default route; the dedicated
    backends are selected by provider name. ``client`` lets callers (and
    tests) inject an HTTP client with a mock transport.
    """
    provider_name = config.provider if config.provider else detect_provider(config.base_url)
    if provider_name == PROVIDER_ALIYUN:
        reranker = await new_aliyun_reranker(config, client=client)
    elif provider_name == PROVIDER_ZHIPU:
        reranker = await new_zhipu_reranker(config, client=client)
    elif provider_name == PROVIDER_JINA:
        reranker = await new_jina_reranker(config, client=client)
    elif provider_name == PROVIDER_NVIDIA:
        reranker = await new_nvidia_reranker(config, client=client)
    elif provider_name == PROVIDER_WEKNORACLOUD:
        reranker = await new_weknoracloud_reranker(config, client=client)
    elif provider_name == PROVIDER_LKEAP:
        reranker = await new_lkeap_reranker(config)
    elif provider_name == PROVIDER_VOLCENGINE:
        reranker = await new_volcengine_reranker(config, client=client)
    else:
        reranker = await new_openai_reranker(
            model_name=config.model_name,
            model_id=config.model_id,
            api_key=config.api_key,
            base_url=config.base_url,
            extra_config=config.extra_config,
            client=client,
        )
    if isinstance(reranker, CustomHeaderSetter):
        reranker.set_custom_headers(config.custom_headers)
    return reranker


__all__ = [
    "CustomHeaderSetter",
    "ModelLike",
    "Reranker",
    "RerankerConfig",
    "config_from_model",
    "new_reranker",
]
