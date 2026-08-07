"""Provider adapters for OpenAI-compatible chat backends.

``ProviderAdapter`` captures everything provider-specific about an
OpenAI-compatible chat backend; ``BaseProvider`` supplies a sensible default
for every method, so a new provider is added by subclassing and overriding
only the one or two methods that actually differ. ``resolve_provider`` picks
the adapter handling a given provider + model, falling back to ``BaseProvider``
(Bearer auth, standard endpoint, no thinking) when none matches.

The registry is ordered: more specific adapters (those with a real
``matches`` predicate) precede the generic catch-all for the same provider.
"""

from __future__ import annotations

import uuid
from typing import NamedTuple

from src.ai.llm.thinking import (
    ChatTemplateKwargs,
    EnableThinking,
    NoThinking,
    ThinkingStrategy,
    ThinkingTypeField,
)
from src.ai.llm.types import ChatOptions, ToolCallMetadata
from src.ai.provider.predicates import (
    is_moonshot_fixed_temp_model,
    is_openai_reasoning_or_gpt5_model,
    is_qwen_thinking_model,
)
from src.ai.provider.registry import (
    PROVIDER_ALIYUN,
    PROVIDER_AZURE_OPENAI,
    PROVIDER_DEEPSEEK,
    PROVIDER_GEMINI,
    PROVIDER_GENERIC,
    PROVIDER_LKEAP,
    PROVIDER_MOONSHOT,
    PROVIDER_NVIDIA,
    PROVIDER_OPENAI,
    PROVIDER_VOLCENGINE,
    PROVIDER_WEKNORACLOUD,
)
from src.ai.utils.signer import sign_request
from src.common.json import JsonObject, JsonValue


class AuthCreds(NamedTuple):
    """Credentials handed to ``ProviderAdapter.auth_headers``.

    ``api_key`` covers the common Bearer / ``api-key`` cases; ``app_id`` and
    ``app_secret`` are only used by signing providers.
    """

    api_key: str
    app_id: str = ""
    app_secret: str = ""


class ProviderAdapter:
    """Base adapter supplying default behavior for every provider method.

    It is also the fallback returned by ``resolve_provider`` for unknown
    providers: Bearer auth, standard endpoint, no thinking, no shaping.
    """

    def name(self) -> str:
        """The provider this adapter handles."""
        return ""

    def matches(self, model: str) -> bool:
        """Whether this adapter applies to ``model`` (sub-provider routing)."""
        return True

    def thinking(self) -> ThinkingStrategy:
        """How this provider encodes ``ChatOptions.thinking``."""
        return NoThinking()

    def shape_request(
        self,
        req: JsonObject,
        opts: ChatOptions | None,
        is_stream: bool,
    ) -> None:
        """Apply in-place parameter quirks to the standard request."""

    def transform_messages(self, messages: list[JsonObject]) -> list[JsonObject]:
        """Rewrite converted messages (e.g. downgrading multi-content)."""
        return messages

    def endpoint(self, base_url: str, model_id: str, is_stream: bool) -> str:
        """Override the request URL; empty means the standard endpoint."""
        return ""

    def auth_headers(self, creds: AuthCreds, body: bytes) -> dict[str, str]:
        """Return the auth headers for a raw HTTP request."""
        return {"Authorization": f"Bearer {creds.api_key}"}

    def force_raw_http(self) -> bool:
        """Force the raw HTTP path even when the body is standard."""
        return False

    def extract_tool_call_metadata(self, raw: JsonObject) -> ToolCallMetadata:
        """Capture provider state from a raw tool_call object."""
        return {}

    def inject_tool_call_metadata(
        self, tool_call: dict[str, JsonValue], metadata: ToolCallMetadata
    ) -> None:
        """Write provider state back into an outbound tool_call object."""


class WeKnoraCloudProvider(ProviderAdapter):
    """Managed cloud: custom endpoint + request signing + multi-content downgrade."""

    def name(self) -> str:
        return PROVIDER_WEKNORACLOUD

    def endpoint(self, base_url: str, model_id: str, is_stream: bool) -> str:
        return base_url.rstrip("/") + "/api/v1/chat/completions"

    def force_raw_http(self) -> bool:
        return True

    def auth_headers(self, creds: AuthCreds, body: bytes) -> dict[str, str]:
        request_id = str(uuid.uuid4())
        return sign_request(
            creds.app_id, creds.app_secret, request_id, body.decode("utf-8")
        )

    def transform_messages(self, messages: list[JsonObject]) -> list[JsonObject]:
        """Downgrade multi-content to plain text, preserving tool protocol fields."""
        result: list[JsonObject] = []
        for message in messages:
            new_msg = dict(message)
            content = new_msg.get("content") or ""
            multi = new_msg.get("multi_content")
            if not content and isinstance(multi, list):
                text_parts: list[str] = []
                for part in multi:
                    if not isinstance(part, dict) or part.get("type") != "text":
                        continue
                    text = part.get("text")
                    if isinstance(text, str) and text:
                        text_parts.append(text)
                new_msg["content"] = "\n".join(text_parts)
                new_msg.pop("multi_content", None)
            result.append(new_msg)
        return result


class QwenThinkingProvider(ProviderAdapter):
    """Aliyun Qwen thinking models: ``enable_thinking`` always sent, off non-stream."""

    def name(self) -> str:
        return PROVIDER_ALIYUN

    def matches(self, model: str) -> bool:
        return is_qwen_thinking_model(model)

    def thinking(self) -> ThinkingStrategy:
        return EnableThinking(always_send=True, disable_on_non_stream=True)


class LkeapProvider(ProviderAdapter):
    """LKEAP: thinking via ``{"thinking": {"type": ...}}`` for DeepSeek V3.x."""

    def name(self) -> str:
        return PROVIDER_LKEAP

    def matches(self, model: str) -> bool:
        return "deepseek-v3" in model.lower()

    def thinking(self) -> ThinkingStrategy:
        return ThinkingTypeField()


class DeepseekProvider(ProviderAdapter):
    """DeepSeek: no ``tool_choice``; raw path keeps native cache counters."""

    def name(self) -> str:
        return PROVIDER_DEEPSEEK

    def force_raw_http(self) -> bool:
        return True

    def shape_request(
        self,
        req: JsonObject,
        opts: ChatOptions | None,
        is_stream: bool,
    ) -> None:
        if opts is not None and opts.tool_choice:
            req.pop("tool_choice", None)


class GenericProvider(ProviderAdapter):
    """Generic (vLLM) deployments: thinking via ``chat_template_kwargs``."""

    def name(self) -> str:
        return PROVIDER_GENERIC

    def thinking(self) -> ThinkingStrategy:
        return ChatTemplateKwargs()


class NvidiaProvider(ProviderAdapter):
    """NVIDIA: thinking via ``chat_template_kwargs``."""

    def name(self) -> str:
        return PROVIDER_NVIDIA

    def thinking(self) -> ThinkingStrategy:
        return ChatTemplateKwargs()


class GeminiProvider(ProviderAdapter):
    """Gemini OpenAI compatibility: tool-call signatures live in ``extra_content``."""

    def name(self) -> str:
        return PROVIDER_GEMINI

    def force_raw_http(self) -> bool:
        return True

    def extract_tool_call_metadata(self, raw: JsonObject) -> ToolCallMetadata:
        extra = raw.get("extra_content")
        if not isinstance(extra, dict):
            return {}
        google = extra.get("google")
        if google is None:
            return {}
        return {"google": google}

    def inject_tool_call_metadata(
        self, tool_call: dict[str, JsonValue], metadata: ToolCallMetadata
    ) -> None:
        google = metadata.get("google")
        if google is None:
            return
        tool_call["extra_content"] = {"google": google}


class VolcengineProvider(ProviderAdapter):
    """Volcengine (Ark): thinking via ``{"thinking": {"type": ...}}``."""

    def name(self) -> str:
        return PROVIDER_VOLCENGINE

    def thinking(self) -> ThinkingStrategy:
        return ThinkingTypeField()


class AzureProvider(ProviderAdapter):
    """Azure OpenAI: ``api-key`` auth."""

    def name(self) -> str:
        return PROVIDER_AZURE_OPENAI

    def auth_headers(self, creds: AuthCreds, body: bytes) -> dict[str, str]:
        return {"api-key": creds.api_key}


class AzureReasoningProvider(AzureProvider):
    """Azure OpenAI reasoning models: strip sampling params, use completion tokens."""

    def matches(self, model: str) -> bool:
        return is_openai_reasoning_or_gpt5_model(model)

    def shape_request(
        self,
        req: JsonObject,
        opts: ChatOptions | None,
        is_stream: bool,
    ) -> None:
        shape_openai_reasoning(req)


class OpenAIReasoningProvider(ProviderAdapter):
    """OpenAI reasoning / GPT-5: no sampling params, ``max_completion_tokens``."""

    def name(self) -> str:
        return PROVIDER_OPENAI

    def matches(self, model: str) -> bool:
        return is_openai_reasoning_or_gpt5_model(model)

    def shape_request(
        self,
        req: JsonObject,
        opts: ChatOptions | None,
        is_stream: bool,
    ) -> None:
        shape_openai_reasoning(req)


class MoonshotProvider(ProviderAdapter):
    """Moonshot v1 models accept only ``temperature=1``."""

    def name(self) -> str:
        return PROVIDER_MOONSHOT

    def matches(self, model: str) -> bool:
        return is_moonshot_fixed_temp_model(model)

    def shape_request(
        self,
        req: JsonObject,
        opts: ChatOptions | None,
        is_stream: bool,
    ) -> None:
        req["temperature"] = 1
        req.pop("top_p", None)
        req.pop("frequency_penalty", None)
        req.pop("presence_penalty", None)


def shape_openai_reasoning(req: JsonObject) -> None:
    """Strip sampling params and migrate ``max_tokens`` to ``max_completion_tokens``."""
    req.pop("temperature", None)
    req.pop("top_p", None)
    req.pop("frequency_penalty", None)
    req.pop("presence_penalty", None)
    if not req.get("max_completion_tokens") and req.get("max_tokens"):
        req["max_completion_tokens"] = req["max_tokens"]
    req.pop("max_tokens", None)


#: Ordered registry: specific adapters precede the generic catch-all.
PROVIDER_REGISTRY: list[ProviderAdapter] = [
    WeKnoraCloudProvider(),
    QwenThinkingProvider(),
    LkeapProvider(),
    DeepseekProvider(),
    GenericProvider(),
    GeminiProvider(),
    VolcengineProvider(),
    NvidiaProvider(),
    AzureReasoningProvider(),
    AzureProvider(),
    OpenAIReasoningProvider(),
    MoonshotProvider(),
]


def resolve_provider(name: str, model: str) -> ProviderAdapter:
    """Return the adapter handling ``name`` + ``model``, or ``BaseProvider``."""
    for provider in PROVIDER_REGISTRY:
        if provider.name() == name and provider.matches(model):
            return provider
    return ProviderAdapter()


__all__ = [
    "PROVIDER_REGISTRY",
    "AuthCreds",
    "AzureProvider",
    "AzureReasoningProvider",
    "DeepseekProvider",
    "GeminiProvider",
    "GenericProvider",
    "LkeapProvider",
    "MoonshotProvider",
    "NvidiaProvider",
    "OpenAIReasoningProvider",
    "ProviderAdapter",
    "QwenThinkingProvider",
    "VolcengineProvider",
    "WeKnoraCloudProvider",
    "resolve_provider",
    "shape_openai_reasoning",
]
