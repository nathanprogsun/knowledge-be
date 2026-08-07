"""OpenAI-compatible chat backend over raw HTTP.

``RemoteAPIChat`` owns the generic request / response / stream pipeline; every
provider-specific behavior is delegated to a :class:`ProviderAdapter` (see
``providers``) and a :class:`ThinkingStrategy` (see ``thinking``). The request
builder (``openai``) and the completion / stream parsers (``openai_stream``)
hold the OpenAI wire-format implementations and are invoked by this class, so
the same public API stays available while the provider request/stream logic
lives in its own modules.

The reference implementation routes some calls through an OpenAI SDK client;
this port has no SDK dependency, so every call goes through the raw HTTP path
with the same adapter + thinking composition.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterable, AsyncIterator
from typing import cast

import httpx

from src.ai.llm import openai as openai_request
from src.ai.llm import openai_stream
from src.ai.llm.openai_stream import StreamChunk, StreamState, remove_thinking_content
from src.ai.llm.prompt_cache import apply_raw_prompt_cache_usage
from src.ai.llm.providers import AuthCreds, resolve_provider
from src.ai.llm.thinking import parse_thinking_override
from src.ai.llm.transport import (
    DEFAULT_CHAT_TIMEOUT_SECONDS,
    DEFAULT_STREAM_TIMEOUT_SECONDS,
    apply_custom_headers,
    build_ssrf_safe_client,
    validate_url_for_ssrf,
    with_llm_timeout,
)
from src.ai.llm.types import (
    ChatConfig,
    ChatOptions,
    ChatResponse,
    Message,
    StreamResponse,
)
from src.ai.llm.usage import log_usage
from src.ai.provider.detect import detect_provider
from src.ai.provider.providers.deepseek import DEEPSEEK_BASE_URL
from src.ai.provider.registry import (
    PROVIDER_AZURE_OPENAI,
    PROVIDER_DEEPSEEK,
    PROVIDER_WEKNORACLOUD,
)
from src.common.exception import AIProviderError
from src.common.json import JsonObject, JsonValue

# Default api-version used for the Azure OpenAI endpoint.
_DEFAULT_AZURE_API_VERSION = "2024-02-15-preview"


class RemoteAPIChat:
    """OpenAI-compatible chat client.

    ``config`` supplies the endpoint, credentials and provider; ``client``
    allows tests to inject an ``httpx.AsyncClient`` with a mock transport.
    """

    def __init__(
        self,
        config: ChatConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if config.base_url:
            validate_url_for_ssrf(config.base_url)

        api_key = config.api_key
        provider_name = config.provider or detect_provider(config.base_url)
        base_url = config.base_url
        if not base_url and provider_name == PROVIDER_DEEPSEEK:
            base_url = DEEPSEEK_BASE_URL
        base_url = base_url.rstrip("/")

        api_version = ""
        if provider_name == PROVIDER_AZURE_OPENAI and config.extra_config:
            api_version = (config.extra_config.get("api_version") or "").strip()

        if provider_name == PROVIDER_WEKNORACLOUD:
            if not config.app_id:
                raise ValueError("managed cloud provider: AppID is required")
            if not config.app_secret:
                raise ValueError("managed cloud provider: AppSecret is required")

        model_name = config.model_name
        if config.extra_config:
            override = (config.extra_config.get("remote_model_name") or "").strip()
            if override:
                model_name = override

        self._model_name = model_name
        self._model_id = config.model_id
        self._base_url = base_url
        self._api_key = api_key
        self._provider = provider_name
        self._app_id = config.app_id
        self._app_secret = config.app_secret
        self._custom_headers = dict(config.custom_headers or {})
        self._api_version = api_version
        self._client = client or build_ssrf_safe_client()
        self._adapter = resolve_provider(provider_name, model_name)
        self._thinking_override = parse_thinking_override(config.extra_config)

    # ── Getters ─────────────────────────────────────────────────────

    def get_model_name(self) -> str:
        return self._model_name

    def get_model_id(self) -> str:
        return self._model_id

    def get_provider(self) -> str:
        return self._provider

    def get_base_url(self) -> str:
        return self._base_url

    def get_api_key(self) -> str:
        return self._api_key

    def auth_creds(self) -> AuthCreds:
        """Bundle the credentials handed to the adapter's auth hook."""
        return AuthCreds(api_key=self._api_key, app_id=self._app_id, app_secret=self._app_secret)

    async def aclose(self) -> None:
        """Release the underlying HTTP client."""
        await self._client.aclose()

    # ── Standard request building ────────────────────────────────────

    def convert_messages(self, messages: list[Message]) -> list[JsonObject]:
        """Convert ``Message`` objects into the OpenAI wire message shape."""
        return openai_request.convert_messages(messages)

    def build_chat_completion_request(
        self,
        messages: list[Message],
        opts: ChatOptions | None,
        is_stream: bool,
    ) -> JsonObject:
        """Build the standard OpenAI-compatible request body.

        Provider-specific quirks are applied afterwards by the adapter.
        """
        return openai_request.build_chat_completion_request(self, messages, opts, is_stream)

    def shaped_request(
        self,
        messages: list[Message],
        opts: ChatOptions | None,
        is_stream: bool,
    ) -> JsonObject:
        """Build the standard request and apply adapter message/shape hooks."""
        req = self.build_chat_completion_request(messages, opts, is_stream)
        wire_messages = cast(list[JsonObject], req["messages"])
        req["messages"] = cast(JsonValue, self._adapter.transform_messages(wire_messages))
        self._adapter.shape_request(req, opts, is_stream)
        return req

    def build_provider_openai_request(
        self,
        body: JsonObject,
        openai_messages: list[JsonObject],
        messages: list[Message],
    ) -> JsonObject:
        """Re-emit the body as a plain map, injecting tool-call metadata."""
        return openai_request.build_provider_openai_request(self, body, openai_messages, messages)

    def shape_provider_request(
        self,
        body: JsonObject,
        req: JsonObject,
        messages: list[Message],
    ) -> JsonObject:
        """Apply the adapter's raw-path message transform when forced."""
        return openai_request.shape_provider_request(self, body, req, messages)

    def build_outbound(
        self,
        messages: list[Message],
        opts: ChatOptions | None,
        is_stream: bool,
    ) -> tuple[JsonObject, str]:
        """Assemble the final outbound body and endpoint.

        This is the single place that composes the adapter and thinking
        strategy.
        """
        req = self.shaped_request(messages, opts, is_stream)
        thinking = self._thinking_override
        if thinking is None:
            thinking = self._adapter.thinking()
        custom_body, _use_raw = thinking.apply(req, opts, is_stream)
        body: JsonObject = req
        if custom_body is not None:
            body = custom_body
        body = self.shape_provider_request(body, req, messages)
        endpoint = self._adapter.endpoint(self._base_url, self._model_id, is_stream)
        if not endpoint:
            endpoint = self._resolve_default_endpoint()
        return body, endpoint

    def _resolve_default_endpoint(self) -> str:
        """Return ``<base_url>/chat/completions`` (Azure uses deployments)."""
        if self._provider == PROVIDER_AZURE_OPENAI:
            api_version = self._api_version or _DEFAULT_AZURE_API_VERSION
            return (
                f"{self._base_url}/openai/deployments/{self._model_name}"
                f"/chat/completions?api-version={api_version}"
            )
        return f"{self._base_url}/chat/completions"

    # ── Non-streaming path ───────────────────────────────────────────

    async def chat(self, messages: list[Message], opts: ChatOptions | None = None) -> ChatResponse:
        """Perform a non-streaming chat round-trip."""
        async with with_llm_timeout(DEFAULT_CHAT_TIMEOUT_SECONDS):
            body, endpoint = self.build_outbound(messages, opts, False)
            return await self.chat_with_raw_http(endpoint, body)

    async def chat_with_raw_http(self, endpoint: str, custom_req: JsonObject) -> ChatResponse:
        """Send ``custom_req`` verbatim and parse the completion response."""
        json_data = json.dumps(custom_req, ensure_ascii=False).encode("utf-8")
        if not endpoint:
            endpoint = self._resolve_default_endpoint()
        validate_url_for_ssrf(endpoint)

        headers = {"Content-Type": "application/json"}
        headers.update(self._adapter.auth_headers(self.auth_creds(), json_data))
        headers = apply_custom_headers(headers, self._custom_headers)

        response = await self._client.post(endpoint, content=json_data, headers=headers)
        if response.status_code != 200:
            raise AIProviderError(
                f"API request failed with status {response.status_code}: {response.text}"
            )
        body = response.content
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AIProviderError("decode response: failed to parse response JSON") from exc
        if not isinstance(payload, dict):
            raise AIProviderError("decode response: unexpected response shape")

        result = self.parse_completion_response(payload)
        self.apply_completion_tool_call_metadata(body, result)
        apply_raw_prompt_cache_usage(body, result.usage)
        log_usage(self._model_name, result.usage)
        return result

    def parse_completion_response(self, resp: JsonObject) -> ChatResponse:
        """Parse a non-streaming completion response into ``ChatResponse``."""
        return openai_stream.parse_completion_response(self, resp)

    def apply_completion_tool_call_metadata(self, body: bytes, result: ChatResponse) -> None:
        """Attach provider tool-call metadata from the raw response body."""
        openai_stream.apply_completion_tool_call_metadata(self, body, result)

    # ── Streaming path ───────────────────────────────────────────────

    async def chat_stream(
        self, messages: list[Message], opts: ChatOptions | None = None
    ) -> AsyncIterator[StreamResponse]:
        """Stream a chat round-trip, yielding one ``StreamResponse`` per event."""
        async with with_llm_timeout(DEFAULT_STREAM_TIMEOUT_SECONDS):
            body, endpoint = self.build_outbound(messages, opts, True)
            async for event in self.chat_stream_with_raw_http(endpoint, body):
                yield event

    async def chat_stream_with_raw_http(
        self, endpoint: str, custom_req: JsonObject
    ) -> AsyncIterator[StreamResponse]:
        """Send ``custom_req`` over SSE and stream parsed events."""
        json_data = json.dumps(custom_req, ensure_ascii=False).encode("utf-8")
        if not endpoint:
            endpoint = self._resolve_default_endpoint()
        validate_url_for_ssrf(endpoint)

        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        headers.update(self._adapter.auth_headers(self.auth_creds(), json_data))
        headers = apply_custom_headers(headers, self._custom_headers)

        async with self._client.stream(
            "POST", endpoint, content=json_data, headers=headers
        ) as response:
            if response.status_code != 200:
                error_body = (await response.aread()).decode("utf-8", errors="replace")
                raise AIProviderError(
                    f"API request failed with status {response.status_code}: {error_body}"
                )
            async for event in self.process_raw_http_stream(response):
                yield event

    async def process_raw_http_stream(
        self, response: httpx.Response
    ) -> AsyncIterator[StreamResponse]:
        """Parse an SSE response into ``StreamResponse`` events."""
        async for out in openai_stream.process_raw_http_stream(self, response):
            yield out

    async def process_stream(
        self, chunks: AsyncIterable[StreamChunk]
    ) -> AsyncIterator[StreamResponse]:
        """Process parsed stream chunks into ``StreamResponse`` events."""
        async for out in openai_stream.process_stream(self, chunks):
            yield out

    def apply_stream_tool_call_metadata(self, stream_resp: JsonObject, state: StreamState) -> None:
        """Attach provider metadata to in-flight tool calls."""
        openai_stream.apply_stream_tool_call_metadata(self, stream_resp, state)

    async def process_stream_delta(
        self, choice: JsonObject, state: StreamState
    ) -> AsyncIterator[StreamResponse]:
        """Process one streaming delta choice."""
        async for out in openai_stream.process_stream_delta(choice, state):
            yield out

    async def process_tool_calls_delta(
        self,
        tool_calls: list[JsonValue],
        state: StreamState,
    ) -> AsyncIterator[StreamResponse]:
        """Accumulate a tool-call delta and stream high-level markers."""
        async for out in openai_stream.process_tool_calls_delta(tool_calls, state):
            yield out


__all__ = [
    "RemoteAPIChat",
    "StreamState",
    "remove_thinking_content",
]
