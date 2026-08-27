"""Base-URL provider detection (Go ``provider.DetectProvider``).

``detect_provider`` maps an endpoint URL to a provider by substring
matching, in the same order and with the same substrings as the upstream
``DetectProvider`` switch. Order matters: earlier cases shadow later ones
(e.g. ``openai.azure.com`` is matched before ``api.openai.com``), so the
case order is preserved verbatim.
"""

from __future__ import annotations

from src.ai.provider.registry import (
    PROVIDER_ALIYUN,
    PROVIDER_ANTHROPIC,
    PROVIDER_AZURE_OPENAI,
    PROVIDER_CLOUD,
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
    PROVIDER_ZHIPU,
    ProviderName,
)


def _contains_any(value: str, *substrings: str) -> bool:
    """True when ``value`` contains any of ``substrings``."""
    return any(sub in value for sub in substrings)


def detect_provider(base_url: str) -> ProviderName:
    """Detect the provider owning ``base_url``, falling back to ``generic``."""
    if _contains_any(base_url, "dashscope.aliyuncs.com"):
        return PROVIDER_ALIYUN
    if _contains_any(base_url, "open.bigmodel.cn", "zhipu"):
        return PROVIDER_ZHIPU
    if _contains_any(base_url, "openrouter.ai"):
        return PROVIDER_OPENROUTER
    if _contains_any(base_url, "router.requesty.ai", "requesty.ai"):
        return PROVIDER_REQUESTY
    if _contains_any(base_url, "siliconflow.cn"):
        return PROVIDER_SILICONFLOW
    if _contains_any(base_url, "api.jina.ai"):
        return PROVIDER_JINA
    if _contains_any(base_url, "openai.azure.com"):
        return PROVIDER_AZURE_OPENAI
    if _contains_any(base_url, "api.openai.com"):
        return PROVIDER_OPENAI
    if _contains_any(base_url, "api.anthropic.com"):
        return PROVIDER_ANTHROPIC
    if _contains_any(base_url, "api.deepseek.com"):
        return PROVIDER_DEEPSEEK
    if _contains_any(base_url, "generativelanguage.googleapis.com"):
        return PROVIDER_GEMINI
    if _contains_any(base_url, "volces.com", "volcengine"):
        return PROVIDER_VOLCENGINE
    if _contains_any(base_url, "hunyuan.cloud.tencent.com"):
        return PROVIDER_HUNYUAN
    if _contains_any(base_url, "minimax.io", "minimaxi.com"):
        return PROVIDER_MINIMAX
    if _contains_any(base_url, "xiaomimimo.com"):
        return PROVIDER_MIMO
    if _contains_any(base_url, "gpustack"):
        return PROVIDER_GPUSTACK
    if _contains_any(base_url, "modelscope.cn"):
        return PROVIDER_MODELSCOPE
    if _contains_any(base_url, "qiniuapi.com", "qiniu"):
        return PROVIDER_QINIU
    if _contains_any(base_url, "moonshot.ai"):
        return PROVIDER_MOONSHOT
    if _contains_any(base_url, "qianfan.baidubce.com", "baidubce.com"):
        return PROVIDER_QIANFAN
    if _contains_any(base_url, "longcat.chat"):
        return PROVIDER_LONGCAT
    if _contains_any(base_url, "lkeap.cloud.tencent.com", "api.lkeap", "lkeap.tencentcloudapi.com"):
        return PROVIDER_LKEAP
    if _contains_any(base_url, "nvidia.com"):
        return PROVIDER_NVIDIA
    if _contains_any(base_url, "api.novita.ai", "novita.ai"):
        return PROVIDER_NOVITA
    if _contains_any(base_url, "kb.weixin.qq.com"):
        return PROVIDER_CLOUD
    return PROVIDER_GENERIC


__all__ = ["detect_provider"]
