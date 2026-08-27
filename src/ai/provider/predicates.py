"""Model-name / endpoint predicates extracted from the provider package.

Each predicate mirrors one exported helper in the upstream provider
package. They are pure string checks used by the chat layer to select
provider-specific request shaping (thinking parameters, temperature
constraints, token-parameter renames, and so on).
"""

from __future__ import annotations


def is_qwen_thinking_model(model_name: str) -> bool:
    """True for Qwen models whose API exposes an ``enable_thinking`` flag.

    Mirrors ``IsQwenThinkingModel``: the prefix set is exactly
    ``qwen3`` / ``qwen-plus`` / ``qwen-max`` / ``qwen-turbo``.
    """
    lower = model_name.lower()
    return lower.startswith(("qwen3", "qwen-plus", "qwen-max", "qwen-turbo"))


def is_qwen3_model(model_name: str) -> bool:
    """True only for the Qwen3 family (``IsQwen3Model``)."""
    return model_name.lower().startswith("qwen3")


def is_deepseek_model(model_name: str) -> bool:
    """True for any model whose name contains ``deepseek``.

    DeepSeek models reject the ``tool_choice`` parameter, so the chat
    layer omits it for these models.
    """
    return "deepseek" in model_name.lower()


def is_lkeap_deepseek_v3_model(model_name: str) -> bool:
    """True for the DeepSeek V3.x series on the LKEAP provider.

    V3.x models support toggling the reasoning chain via a thinking
    parameter.
    """
    return "deepseek-v3" in model_name.lower()


def is_lkeap_deepseek_r1_model(model_name: str) -> bool:
    """True for the DeepSeek R1 series on the LKEAP provider.

    R1 models enable the reasoning chain by default.
    """
    return "deepseek-r1" in model_name.lower()


def is_lkeap_thinking_model(model_name: str) -> bool:
    """True for any LKEAP model that supports a reasoning chain."""
    return is_lkeap_deepseek_r1_model(model_name) or is_lkeap_deepseek_v3_model(model_name)


def is_moonshot_fixed_temp_model(model_name: str) -> bool:
    """True for Moonshot/Kimi models that only accept ``temperature=1``.

    Mirrors ``IsMoonshotFixedTempModel``: the ``moonshot-v1`` series and
    the ``kimi-k2.5`` / ``kimi-k2.6`` releases reject any other value.
    ``kimi-k2``, ``kimi-k2-turbo`` and ``kimi-k2-thinking`` accept the
    full ``[0, 1]`` range and are not affected.
    """
    name = model_name.lower().strip()
    if name.startswith("moonshot-v1"):
        return True
    return name == "kimi-k2.5" or name == "kimi-k2.6"


def is_openai_reasoning_or_gpt5_model(model_name: str) -> bool:
    """True for OpenAI / Azure OpenAI reasoning (o-series) and GPT-5 models.

    Mirrors ``IsOpenAIReasoningOrGPT5Model``. These models reject
    ``max_tokens`` (use ``max_completion_tokens``) and only accept the
    default sampling parameters. The o-series prefixes are matched as
    exact names or ``prefix-`` forms so ``openai-...`` names never match.
    """
    name = model_name.lower().strip()
    if name == "":
        return False
    if name.startswith("gpt-5"):
        return True
    return any(name == prefix or name.startswith(prefix + "-") for prefix in ("o1", "o3", "o4"))


__all__ = [
    "is_deepseek_model",
    "is_lkeap_deepseek_r1_model",
    "is_lkeap_deepseek_v3_model",
    "is_lkeap_thinking_model",
    "is_moonshot_fixed_temp_model",
    "is_openai_reasoning_or_gpt5_model",
    "is_qwen3_model",
    "is_qwen_thinking_model",
]
