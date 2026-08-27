"""Provider metadata registry (mirrors the upstream provider registry).

Builds the ``PROVIDERS`` map from the 26 provider modules and exposes the
lookup helpers the chat / embedding / rerank factories route on:
``get_provider``, ``get_provider_or_default`` (falls back to the generic
entry), ``list_providers`` and ``list_providers_by_model_type`` — both in
``ALL_PROVIDERS`` order.
"""

from __future__ import annotations

from src.ai.provider.providers import (
    aliyun,
    anthropic,
    azure_openai,
    cloud,
    deepseek,
    gemini,
    generic,
    gpustack,
    hunyuan,
    jina,
    lkeap,
    longcat,
    mimo,
    minimax,
    modelscope,
    moonshot,
    novita,
    nvidia,
    openai,
    openrouter,
    qianfan,
    qiniu,
    requesty,
    siliconflow,
    volcengine,
    zhipu,
)
from src.ai.provider.registry import (
    ALL_PROVIDERS,
    PROVIDER_GENERIC,
    ModelType,
    ProviderInfo,
)

# Canonical provider → metadata map. Built eagerly so lookup helpers never
# construct on the hot path. The keys are the plain string identifiers
# (the ``ProviderName`` literal values), so lookups accept any string.
PROVIDERS: dict[str, ProviderInfo] = {}
for _builder in (
    generic.info,
    cloud.info,
    aliyun.info,
    zhipu.info,
    volcengine.info,
    hunyuan.info,
    siliconflow.info,
    deepseek.info,
    minimax.info,
    moonshot.info,
    modelscope.info,
    qianfan.info,
    qiniu.info,
    openai.info,
    anthropic.info,
    gemini.info,
    openrouter.info,
    requesty.info,
    jina.info,
    mimo.info,
    longcat.info,
    lkeap.info,
    gpustack.info,
    nvidia.info,
    novita.info,
    azure_openai.info,
):
    _provider = _builder()
    PROVIDERS[str(_provider.name)] = _provider


def get_provider(name: str) -> ProviderInfo | None:
    """Return the metadata for ``name``, or ``None`` when unknown."""
    return PROVIDERS.get(name)


def get_provider_or_default(name: str) -> ProviderInfo:
    """Return the metadata for ``name``, falling back to the generic entry."""
    provider = PROVIDERS.get(name)
    if provider is not None:
        return provider
    return PROVIDERS[PROVIDER_GENERIC]


def list_providers() -> list[ProviderInfo]:
    """Return every provider metadata record in ``ALL_PROVIDERS`` order."""
    return [PROVIDERS[name] for name in ALL_PROVIDERS if name in PROVIDERS]


def list_providers_by_model_type(model_type: ModelType) -> list[ProviderInfo]:
    """Return providers supporting ``model_type``, in canonical order."""
    return [p for p in list_providers() if model_type in p.model_types]


__all__ = [
    "PROVIDERS",
    "get_provider",
    "get_provider_or_default",
    "list_providers",
    "list_providers_by_model_type",
]
