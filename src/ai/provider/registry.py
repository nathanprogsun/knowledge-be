"""Provider registry: name constants, model types, and provider metadata.

This module mirrors the upstream contract. It defines the canonical set of
provider identifiers (``ProviderName``), the model-type enum used to key
default endpoint URLs, and the shared metadata models every provider
module populates. ``ALL_PROVIDERS`` carries the canonical ordering the
chat / embedding / rerank factories iterate when routing by provider.

The ``ProviderName`` string values are the same identifiers the web-layer
catalog exposes as ``value``, so a UI can switch between languages
unchanged. Model-type keys use the backend enum form (``KnowledgeQA`` /
``Embedding`` / ``Rerank`` / ``VLLM`` / ``ASR``), matching the upstream
``types.ModelType`` contract rather than the frontend aliases.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject


class ModelType(StrEnum):
    """Model type enum mirroring the upstream backend contract values."""

    EMBEDDING = "Embedding"
    RERANK = "Rerank"
    KNOWLEDGE_QA = "KnowledgeQA"
    VLLM = "VLLM"
    ASR = "ASR"


ProviderName: TypeAlias = Literal[
    "generic",
    "weknoracloud",
    "aliyun",
    "zhipu",
    "volcengine",
    "hunyuan",
    "siliconflow",
    "deepseek",
    "minimax",
    "moonshot",
    "modelscope",
    "qianfan",
    "qiniu",
    "openai",
    "anthropic",
    "gemini",
    "openrouter",
    "requesty",
    "jina",
    "mimo",
    "longcat",
    "lkeap",
    "gpustack",
    "nvidia",
    "novita",
    "azure_openai",
]

# ── Provider identifiers ──────────────────────────────────────────────

PROVIDER_GENERIC: ProviderName = "generic"
PROVIDER_WEKNORACLOUD: ProviderName = "weknoracloud"
PROVIDER_ALIYUN: ProviderName = "aliyun"
PROVIDER_ZHIPU: ProviderName = "zhipu"
PROVIDER_VOLCENGINE: ProviderName = "volcengine"
PROVIDER_HUNYUAN: ProviderName = "hunyuan"
PROVIDER_SILICONFLOW: ProviderName = "siliconflow"
PROVIDER_DEEPSEEK: ProviderName = "deepseek"
PROVIDER_MINIMAX: ProviderName = "minimax"
PROVIDER_MOONSHOT: ProviderName = "moonshot"
PROVIDER_MODELSCOPE: ProviderName = "modelscope"
PROVIDER_QIANFAN: ProviderName = "qianfan"
PROVIDER_QINIU: ProviderName = "qiniu"
PROVIDER_OPENAI: ProviderName = "openai"
PROVIDER_ANTHROPIC: ProviderName = "anthropic"
PROVIDER_GEMINI: ProviderName = "gemini"
PROVIDER_OPENROUTER: ProviderName = "openrouter"
PROVIDER_REQUESTY: ProviderName = "requesty"
PROVIDER_JINA: ProviderName = "jina"
PROVIDER_MIMO: ProviderName = "mimo"
PROVIDER_LONGCAT: ProviderName = "longcat"
PROVIDER_LKEAP: ProviderName = "lkeap"
PROVIDER_GPUSTACK: ProviderName = "gpustack"
PROVIDER_NVIDIA: ProviderName = "nvidia"
PROVIDER_NOVITA: ProviderName = "novita"
PROVIDER_AZURE_OPENAI: ProviderName = "azure_openai"

# All registered providers, in the canonical upstream ordering. Kept as a
# tuple so the factories and the metadata list iterate deterministically.
ALL_PROVIDERS: tuple[ProviderName, ...] = (
    PROVIDER_GENERIC,
    PROVIDER_WEKNORACLOUD,
    PROVIDER_ALIYUN,
    PROVIDER_ZHIPU,
    PROVIDER_VOLCENGINE,
    PROVIDER_HUNYUAN,
    PROVIDER_SILICONFLOW,
    PROVIDER_DEEPSEEK,
    PROVIDER_MINIMAX,
    PROVIDER_MOONSHOT,
    PROVIDER_MODELSCOPE,
    PROVIDER_QIANFAN,
    PROVIDER_QINIU,
    PROVIDER_OPENAI,
    PROVIDER_ANTHROPIC,
    PROVIDER_GEMINI,
    PROVIDER_OPENROUTER,
    PROVIDER_REQUESTY,
    PROVIDER_JINA,
    PROVIDER_MIMO,
    PROVIDER_LONGCAT,
    PROVIDER_LKEAP,
    PROVIDER_GPUSTACK,
    PROVIDER_NVIDIA,
    PROVIDER_NOVITA,
    PROVIDER_AZURE_OPENAI,
)


# ── Provider metadata models ──────────────────────────────────────────


class ExtraFieldOption(BaseModel):
    """One selectable option of an ``ExtraFieldConfig`` (Go anonymous struct)."""

    model_config = ConfigDict(frozen=True)

    label: str
    value: str


class ExtraFieldConfig(BaseModel):
    """Extra configuration field a provider exposes on the create-model form."""

    model_config = ConfigDict(frozen=True)

    key: str
    label: str
    type: str
    required: bool = False
    default: str = ""
    placeholder: str = ""
    options: list[ExtraFieldOption] = Field(default_factory=list)


class ProviderInfo(BaseModel):
    """Metadata for one model provider (Go ``provider.ProviderInfo``).

    ``default_urls`` is keyed by ``ModelType``; ``get_default_url`` falls
    back to the chat (``KnowledgeQA``) URL exactly like the upstream
    ``ProviderInfo.GetDefaultURL``.
    """

    model_config = ConfigDict(frozen=True)

    name: ProviderName
    display_name: str
    description: str
    default_urls: dict[ModelType, str] = Field(default_factory=dict)
    model_types: list[ModelType] = Field(default_factory=list)
    requires_auth: bool = False
    extra_fields: list[ExtraFieldConfig] = Field(default_factory=list)

    def get_default_url(self, model_type: ModelType) -> str:
        """Return the default base URL for ``model_type``.

        Falls back to the chat (``KnowledgeQA``) URL when the requested
        type has no dedicated entry, and to ``""`` when neither exists.
        """
        url = self.default_urls.get(model_type)
        if url is not None:
            return url
        return self.default_urls.get(ModelType.KNOWLEDGE_QA, "")


class ProviderConfig(BaseModel):
    """Runtime configuration for one provider instance (Go ``provider.Config``)."""

    model_config = ConfigDict(frozen=True)

    provider: ProviderName
    base_url: str = ""
    api_key: str = ""
    model_name: str = ""
    model_id: str = ""
    extra: JsonObject | None = None


__all__ = [
    "ALL_PROVIDERS",
    "PROVIDER_ALIYUN",
    "PROVIDER_ANTHROPIC",
    "PROVIDER_AZURE_OPENAI",
    "PROVIDER_DEEPSEEK",
    "PROVIDER_GEMINI",
    "PROVIDER_GENERIC",
    "PROVIDER_GPUSTACK",
    "PROVIDER_HUNYUAN",
    "PROVIDER_JINA",
    "PROVIDER_LKEAP",
    "PROVIDER_LONGCAT",
    "PROVIDER_MIMO",
    "PROVIDER_MINIMAX",
    "PROVIDER_MODELSCOPE",
    "PROVIDER_MOONSHOT",
    "PROVIDER_NOVITA",
    "PROVIDER_NVIDIA",
    "PROVIDER_OPENAI",
    "PROVIDER_OPENROUTER",
    "PROVIDER_QIANFAN",
    "PROVIDER_QINIU",
    "PROVIDER_REQUESTY",
    "PROVIDER_SILICONFLOW",
    "PROVIDER_VOLCENGINE",
    "PROVIDER_WEKNORACLOUD",
    "PROVIDER_ZHIPU",
    "ModelType",
    "ProviderConfig",
    "ProviderInfo",
    "ProviderName",
]
