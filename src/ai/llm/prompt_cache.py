"""Prompt-cache fingerprinting and usage normalization.

``fingerprint_prompt_prefix`` derives a short, non-reversible identifier for
logs and cache routing; raw prompts must never be used as metric labels.
``token_usage_from_openai`` / ``apply_raw_prompt_cache_usage`` normalize the
provider-specific cache counters (including native fields the OpenAI-compatible
layer discards, notably DeepSeek hit/miss counters) into the shared
:class:`TokenUsage` model.
"""

from __future__ import annotations

import hashlib
import json

from src.ai.llm.types import (
    ChatOptions,
    Message,
    PromptCacheStatus,
    TokenUsage,
)
from src.ai.provider.registry import (
    PROVIDER_ALIYUN,
    PROVIDER_ANTHROPIC,
    PROVIDER_AZURE_OPENAI,
    PROVIDER_DEEPSEEK,
    PROVIDER_OPENAI,
)
from src.common.json import JsonObject, JsonValue

_CACHE_KEY_PREFIX = "wk-"
_FINGERPRINT_LENGTH = 16


def fingerprint_prompt_prefix(*parts: str) -> str:
    """SHA-256 over ``parts`` joined by NUL bytes, truncated hex digest."""
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:_FINGERPRINT_LENGTH]


def prompt_prefix_fingerprint(
    messages: list[Message], opts: ChatOptions | None
) -> str:
    """Hash the stable prefix common to normal chat and agent requests.

    Leading system messages plus the deterministic tool schema participate;
    dynamic conversation/user messages intentionally do not.
    """
    prefix: JsonObject = {}
    system: list[JsonValue] = []
    for message in messages:
        if message.role != "system":
            break
        system.append(message.model_dump())
    if system:
        prefix["system"] = system
    if opts is not None and opts.tools:
        prefix["tools"] = [tool.model_dump() for tool in opts.tools]
    return fingerprint_prompt_prefix(json.dumps(prefix))


def build_prompt_cache_key(
    tenant_id: int, model_id: str, purpose: str, prefix_fingerprint: str
) -> str:
    """Derive an opaque process-local coordination key.

    Tenant and model identifiers are hashed rather than retained in memory.
    """
    return _CACHE_KEY_PREFIX + fingerprint_prompt_prefix(
        str(tenant_id), model_id, purpose, prefix_fingerprint
    )


def provider_cache_accounting_status(name: str) -> PromptCacheStatus:
    """Report whether a provider surfaces native prompt-cache counters."""
    if name in {
        PROVIDER_OPENAI,
        PROVIDER_AZURE_OPENAI,
        PROVIDER_DEEPSEEK,
        PROVIDER_ALIYUN,
        PROVIDER_ANTHROPIC,
    }:
        return PromptCacheStatus.UNREPORTED
    return PromptCacheStatus.UNSUPPORTED


def cached_tokens(details: JsonObject | None) -> int:
    """Nil-safe read of ``prompt_tokens_details.cached_tokens``."""
    if not details:
        return 0
    value = details.get("cached_tokens")
    return value if isinstance(value, int) else 0


def token_usage_from_openai(
    usage: JsonObject, provider_name: str
) -> TokenUsage:
    """Build a normalized :class:`TokenUsage` from an OpenAI usage block."""
    u = TokenUsage(
        prompt_tokens=_value_or_zero(_present_int(usage, "prompt_tokens")),
        completion_tokens=_value_or_zero(_present_int(usage, "completion_tokens")),
        total_tokens=_value_or_zero(_present_int(usage, "total_tokens")),
    )
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        read = _value_or_zero(_present_int(details, "cached_tokens"))
        u.set_prompt_cache_usage(read, 0, max(0, u.prompt_tokens - read), True)
        return u
    if provider_cache_accounting_status(provider_name) == PromptCacheStatus.UNSUPPORTED:
        u.mark_prompt_cache_unsupported()
    else:
        u.set_prompt_cache_usage(0, 0, 0, False)
    return u


def apply_raw_prompt_cache_usage(data: bytes, usage: TokenUsage | None) -> None:
    """Capture native cache counters discarded by the OpenAI-compatible layer."""
    if usage is None or not data:
        return
    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return
    if not isinstance(raw, dict):
        return
    raw_usage = raw.get("usage")
    if not isinstance(raw_usage, dict):
        return

    hit = _present_int(raw_usage, "prompt_cache_hit_tokens")
    miss = _present_int(raw_usage, "prompt_cache_miss_tokens")
    if hit is not None or miss is not None:
        usage.set_prompt_cache_usage(hit or 0, 0, miss or 0, True)
        return

    cache_read = _present_int(raw_usage, "cache_read_input_tokens")
    cache_creation = _present_int(raw_usage, "cache_creation_input_tokens")
    if cache_read is not None or cache_creation is not None:
        read = cache_read or 0
        usage.set_prompt_cache_usage(
            read, cache_creation or 0, max(0, usage.prompt_tokens - read), True
        )
        return

    details = raw_usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        read = _value_or_zero(_present_int(details, "cached_tokens"))
        write = _value_or_zero(_present_int(details, "cache_write_tokens"))
        usage.set_prompt_cache_usage(
            read, write, max(0, usage.prompt_tokens - read), True
        )


def _present_int(obj: JsonObject, key: str) -> int | None:
    """Return ``obj[key]`` when it is a JSON int, else ``None``."""
    value = obj.get(key)
    return value if isinstance(value, int) else None


def _value_or_zero(value: int | None) -> int:
    return value if value is not None else 0


__all__ = [
    "apply_raw_prompt_cache_usage",
    "build_prompt_cache_key",
    "cached_tokens",
    "fingerprint_prompt_prefix",
    "prompt_prefix_fingerprint",
    "provider_cache_accounting_status",
    "token_usage_from_openai",
]
