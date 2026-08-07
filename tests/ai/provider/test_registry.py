"""Tests for the provider registry, detection, predicates and metadata.

Pins the canonical ``ProviderName`` values and ``ALL_PROVIDERS`` ordering
against the web-layer catalog, exercises ``detect_provider`` URL routing
(the case order is contract), and covers the model-name predicates and
per-provider config validation.
"""

from __future__ import annotations

import pytest

from src.ai.provider import (
    ALL_PROVIDERS,
    PROVIDER_ALIYUN,
    PROVIDER_ANTHROPIC,
    PROVIDER_AZURE_OPENAI,
    PROVIDER_DEEPSEEK,
    PROVIDER_GEMINI,
    PROVIDER_GENERIC,
    PROVIDER_GPUSTACK,
    PROVIDER_HUNYUAN,
    PROVIDER_JINA,
    PROVIDER_LKEAP,
    PROVIDER_LONGCAT,
    PROVIDER_MIMO,
    PROVIDER_MINIMAX,
    PROVIDER_MODELSCOPE,
    PROVIDER_MOONSHOT,
    PROVIDER_NOVITA,
    PROVIDER_NVIDIA,
    PROVIDER_OPENAI,
    PROVIDER_OPENROUTER,
    PROVIDER_QIANFAN,
    PROVIDER_QINIU,
    PROVIDER_REQUESTY,
    PROVIDER_SILICONFLOW,
    PROVIDER_VOLCENGINE,
    PROVIDER_WEKNORACLOUD,
    PROVIDER_ZHIPU,
    ModelType,
    ProviderConfig,
    detect_provider,
    get_provider,
    get_provider_or_default,
    is_deepseek_model,
    is_lkeap_deepseek_r1_model,
    is_lkeap_deepseek_v3_model,
    is_lkeap_thinking_model,
    is_moonshot_fixed_temp_model,
    is_openai_reasoning_or_gpt5_model,
    is_qwen3_model,
    is_qwen_thinking_model,
    list_providers,
    list_providers_by_model_type,
)
from src.common.exception import ValidationError
from src.core.infra.models.catalog import PROVIDER_CATALOG


def test_all_providers_has_26_unique_entries() -> None:
    assert len(ALL_PROVIDERS) == 26
    assert len(set(ALL_PROVIDERS)) == 26


def test_all_providers_matches_catalog_values() -> None:
    catalog_values = {entry.value for entry in PROVIDER_CATALOG}
    assert catalog_values == set(ALL_PROVIDERS)


def test_all_providers_has_canonical_ordering() -> None:
    expected = (
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
    )
    assert expected == ALL_PROVIDERS


def test_provider_name_values_are_the_contract_strings() -> None:
    assert PROVIDER_GENERIC == "generic"
    assert PROVIDER_WEKNORACLOUD == "weknoracloud"
    assert PROVIDER_AZURE_OPENAI == "azure_openai"


# ── detect_provider ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://dashscope.aliyuncs.com/compatible-mode/v1", PROVIDER_ALIYUN),
        ("https://open.bigmodel.cn/api/paas/v4", PROVIDER_ZHIPU),
        ("https://zhipu.example.com/custom", PROVIDER_ZHIPU),
        ("https://openrouter.ai/api/v1", PROVIDER_OPENROUTER),
        ("https://router.requesty.ai/v1", PROVIDER_REQUESTY),
        ("https://api.siliconflow.cn/v1", PROVIDER_SILICONFLOW),
        ("https://api.jina.ai/v1", PROVIDER_JINA),
        ("https://my-resource.openai.azure.com", PROVIDER_AZURE_OPENAI),
        ("https://api.openai.com/v1", PROVIDER_OPENAI),
        ("https://api.anthropic.com/v1", PROVIDER_ANTHROPIC),
        ("https://api.deepseek.com/v1", PROVIDER_DEEPSEEK),
        ("https://generativelanguage.googleapis.com/v1beta", PROVIDER_GEMINI),
        ("https://ark.cn-beijing.volces.com/api/v3", PROVIDER_VOLCENGINE),
        ("https://api.hunyuan.cloud.tencent.com/v1", PROVIDER_HUNYUAN),
        ("https://api.minimax.io/v1", PROVIDER_MINIMAX),
        ("https://api.minimaxi.com/v1", PROVIDER_MINIMAX),
        ("https://api.xiaomimimo.com/v1", PROVIDER_MIMO),
        ("http://my-gpustack-server/v1-openai", PROVIDER_GPUSTACK),
        ("https://api-inference.modelscope.cn/v1", PROVIDER_MODELSCOPE),
        ("https://api.qiniu.com/v1", PROVIDER_QINIU),
        ("https://api.moonshot.ai/v1", PROVIDER_MOONSHOT),
        ("https://qianfan.baidubce.com/v2", PROVIDER_QIANFAN),
        ("https://api.longcat.chat/v1", PROVIDER_LONGCAT),
        ("https://api.lkeap.cloud.tencent.com/v1", PROVIDER_LKEAP),
        ("https://lkeap.tencentcloudapi.com", PROVIDER_LKEAP),
        ("https://integrate.api.nvidia.com/v1", PROVIDER_NVIDIA),
        ("https://api.novita.ai/openai/v1", PROVIDER_NOVITA),
        ("https://weknora.weixin.qq.com", PROVIDER_WEKNORACLOUD),
        ("https://custom.example.com/v1", PROVIDER_GENERIC),
        ("", PROVIDER_GENERIC),
    ],
)
def test_detect_provider_routes_each_base_url(base_url: str, expected: str) -> None:
    assert detect_provider(base_url) == expected


def test_detect_provider_case_order_is_contract() -> None:
    # ``openai.azure.com`` is matched before ``api.openai.com``.
    assert detect_provider("https://api.openai.azure.com") == PROVIDER_AZURE_OPENAI
    # A URL that merely mentions an earlier-matching substring still wins.
    assert detect_provider("https://dashscope.aliyuncs.com/openai") == PROVIDER_ALIYUN


def test_detect_provider_known_catalog_urls_that_fall_to_generic() -> None:
    # The upstream matcher only knows ``moonshot.ai`` / ``minimax.io`` /
    # ``minimaxi.com`` / ``qiniuapi.com``+``qiniu``; the catalog defaults
    # for these providers use different hosts and therefore route to
    # ``generic``. Kept faithful.
    assert detect_provider("https://api.moonshot.cn/v1") == PROVIDER_GENERIC
    assert detect_provider("https://api.minimax.chat/v1") == PROVIDER_GENERIC
    assert detect_provider("https://api.qnaigc.com/v1") == PROVIDER_GENERIC


# ── registry lookup ──────────────────────────────────────────────────


def test_providers_map_has_26_entries() -> None:
    infos = list_providers()
    assert len(infos) == 26
    assert [info.name for info in infos] == list(ALL_PROVIDERS)


def test_get_provider_returns_info_for_known_name() -> None:
    info = get_provider("openai")
    assert info is not None
    assert info.name == "openai"
    assert info.requires_auth is True


def test_get_provider_returns_none_for_unknown_name() -> None:
    assert get_provider("does-not-exist") is None


def test_get_provider_or_default_falls_back_to_generic() -> None:
    assert get_provider_or_default("does-not-exist") == get_provider("generic")
    assert get_provider_or_default("openai") == get_provider("openai")


def test_list_providers_by_model_type_filters_and_preserves_order() -> None:
    embedding_providers = list_providers_by_model_type(ModelType.EMBEDDING)
    assert "openai" in [p.name for p in embedding_providers]
    assert "jina" in [p.name for p in embedding_providers]
    # The filtered list keeps the canonical order.
    assert [p.name for p in embedding_providers] == sorted(
        [p.name for p in embedding_providers],
        key=list(ALL_PROVIDERS).index,
    )


def test_list_providers_by_model_type_rerank_excludes_jina_embedding_only() -> None:
    rerank_providers = {p.name for p in list_providers_by_model_type(ModelType.RERANK)}
    assert "jina" in rerank_providers
    assert "deepseek" not in rerank_providers


def test_get_default_url_uses_exact_type_then_chat_fallback() -> None:
    aliyun = get_provider("aliyun")
    assert aliyun is not None
    assert "text-rerank" in aliyun.get_default_url(ModelType.RERANK)
    assert aliyun.get_default_url(ModelType.EMBEDDING) == aliyun.get_default_url(
        ModelType.KNOWLEDGE_QA
    )
    generic = get_provider("generic")
    assert generic is not None
    assert generic.get_default_url(ModelType.KNOWLEDGE_QA) == ""


def test_azure_default_urls_include_rerank_but_model_types_exclude_it() -> None:
    # Faithful to the upstream metadata: the rerank URL is declared even
    # though the provider does not advertise rerank support.
    azure = get_provider("azure_openai")
    assert azure is not None
    assert azure.get_default_url(ModelType.RERANK) != ""
    assert ModelType.RERANK not in azure.model_types


def test_weknoracloud_provider_accepts_empty_config() -> None:
    from src.ai.provider.providers import weknoracloud

    weknoracloud.validate_config(ProviderConfig(provider="weknoracloud"))


def test_generic_provider_requires_base_url_and_model_name() -> None:
    from src.ai.provider.providers import generic

    with pytest.raises(ValidationError, match="base URL is required"):
        generic.validate_config(ProviderConfig(provider="generic", model_name="m"))
    with pytest.raises(ValidationError, match="model name is required"):
        generic.validate_config(ProviderConfig(provider="generic", base_url="http://x"))


# ── predicates ───────────────────────────────────────────────────────


def test_is_qwen_thinking_model_prefix_set() -> None:
    assert is_qwen_thinking_model("qwen3-max")
    assert is_qwen_thinking_model("qwen-plus-0325")
    assert is_qwen_thinking_model("QWEN-MAX")
    assert is_qwen_thinking_model("qwen-turbo-latest")
    assert not is_qwen_thinking_model("qwen2.5-72b")
    assert not is_qwen_thinking_model("gpt-4o")


def test_is_qwen3_model_only_matches_qwen3_family() -> None:
    assert is_qwen3_model("qwen3-flash")
    assert is_qwen3_model("Qwen3-8B")
    assert not is_qwen3_model("qwen-plus")


def test_is_deepseek_model_contains_match() -> None:
    assert is_deepseek_model("deepseek-chat")
    assert is_deepseek_model("DeepSeek-R1")
    assert not is_deepseek_model("gpt-4o")


def test_is_lkeap_thinking_model() -> None:
    assert is_lkeap_deepseek_r1_model("deepseek-r1-250528")
    assert is_lkeap_deepseek_v3_model("deepseek-v3.1")
    assert is_lkeap_thinking_model("deepseek-r1")
    assert is_lkeap_thinking_model("deepseek-v3")
    assert not is_lkeap_thinking_model("deepseek-chat")


def test_is_moonshot_fixed_temp_model() -> None:
    assert is_moonshot_fixed_temp_model("moonshot-v1-8k")
    assert is_moonshot_fixed_temp_model("moonshot-v1-128k")
    assert is_moonshot_fixed_temp_model("kimi-k2.5")
    assert is_moonshot_fixed_temp_model("KIMI-K2.6")
    assert not is_moonshot_fixed_temp_model("kimi-k2")
    assert not is_moonshot_fixed_temp_model("kimi-k2-turbo")
    assert not is_moonshot_fixed_temp_model("kimi-k2-thinking")


def test_is_openai_reasoning_or_gpt5_model() -> None:
    assert is_openai_reasoning_or_gpt5_model("gpt-5")
    assert is_openai_reasoning_or_gpt5_model("gpt-5-mini")
    assert is_openai_reasoning_or_gpt5_model("o1")
    assert is_openai_reasoning_or_gpt5_model("o1-mini")
    assert is_openai_reasoning_or_gpt5_model("o3")
    assert is_openai_reasoning_or_gpt5_model("o4-mini")
    assert not is_openai_reasoning_or_gpt5_model("openai-gpt-4o")
    assert not is_openai_reasoning_or_gpt5_model("gpt-4o")
    assert not is_openai_reasoning_or_gpt5_model("")
