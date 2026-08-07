"""OpenAI-compatible embedding provider (openai.go).

Implements text vectorization over the OpenAI ``/embeddings`` wire shape
(also spoken by DashScope-compatible, Volcengine, Jina, ... endpoints).
The embedder owns a single ``httpx.AsyncClient``; transport errors are
retried with exponential backoff (up to ``max_retries`` additional
attempts), non-2xx responses surface as ``AIProviderError``.

The module references the base protocols through a module-level import of
``src.ai.embedding.base`` (rather than a from-import) to break the
base↔provider import cycle.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import httpx
from pydantic import BaseModel, ConfigDict, Field

import src.ai.embedding.base as base
from src.ai.embedding.transport import (
    apply_custom_headers,
    new_embedding_http_client,
    validate_embedding_base_url,
)
from src.common.exception import AIProviderError, ValidationError

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_TIMEOUT_SECONDS = 60.0
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_TRUNCATE_PROMPT_TOKENS = 511
_EMBED_ENDPOINT = "/embeddings"
_EMBED_RETRIES = 3
_MAX_BACKOFF_SECONDS = 10.0
_MAX_ERROR_BODY = 1000


class OpenAIEmbedRequest(BaseModel):
    """``/embeddings`` request body (upstream ``OpenAIEmbedRequest``)."""

    model_config = ConfigDict(frozen=True)

    model: str
    input: list[str]
    encoding_format: str | None = Field(default=None)
    dimensions: int | None = Field(default=None)
    truncate_prompt_tokens: int | None = Field(default=None)


class EmbeddingData(BaseModel):
    """One result entry of an ``/embeddings`` response."""

    model_config = ConfigDict(frozen=True)

    embedding: list[float] = Field(default_factory=list)
    index: int = 0


class OpenAIEmbedResponse(BaseModel):
    """``/embeddings`` response body (upstream ``OpenAIEmbedResponse``)."""

    model_config = ConfigDict(frozen=True)

    data: list[EmbeddingData] = Field(default_factory=list)


class OpenAIEmbedder:
    """OpenAI-compatible text vectorizer."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model_name: str,
        truncate_prompt_tokens: int,
        dimensions: int,
        model_id: str,
        pooler: base.EmbedderPooler | None,
        http_client: httpx.AsyncClient,
        max_retries: int,
        custom_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._truncate_prompt_tokens = truncate_prompt_tokens
        self._dimensions = dimensions
        self._model_id = model_id
        self._pooler = pooler
        self._http_client = http_client
        self._max_retries = max_retries
        self._custom_headers = dict(custom_headers) if custom_headers else {}
        self._supports_dimension_override = False

    # ── Interface setters (base ``SupportsDimensionOverride`` /
    #    ``CustomHeadersSettable``) ──────────────────────────────────

    def set_custom_headers(self, headers: Mapping[str, str] | None) -> None:
        self._custom_headers = dict(headers) if headers else {}

    def set_supports_dimension_override(self, supported: bool) -> None:
        self._supports_dimension_override = supported

    # ── Embedder interface ──────────────────────────────────────────

    async def embed(self, ctx: base.Context, text: str) -> list[float]:
        for _ in range(_EMBED_RETRIES):
            embeddings = await self.batch_embed(ctx, [text])
            if embeddings:
                return embeddings[0]
        raise AIProviderError(
            "no embedding returned",
            code="embedding.no_embedding_returned",
        )

    async def batch_embed(self, ctx: base.Context, texts: list[str]) -> list[list[float]]:
        request = OpenAIEmbedRequest(
            model=self._model_name,
            input=list(texts),
            encoding_format="float",
            truncate_prompt_tokens=self._truncate_prompt_tokens or None,
        )
        if self._supports_dimensions_param():
            request = request.model_copy(update={"dimensions": self._dimensions})
        response = await self._do_request_with_retry(ctx, request)
        body = response.text
        if response.status_code != 200:
            if len(body) > _MAX_ERROR_BODY:
                body = f"{body[:_MAX_ERROR_BODY]}... (truncated)"
            raise AIProviderError(
                f"EmbedBatch API error: Http Status {response.status_code} "
                f"{response.reason_phrase}, Response: {body}",
                code="embedding.api_error",
            )
        try:
            parsed = OpenAIEmbedResponse.model_validate_json(body)
        except ValueError as exc:
            raise AIProviderError(
                f"unmarshal response: {exc}",
                code="embedding.invalid_response",
            ) from exc
        return [data.embedding for data in parsed.data]

    async def batch_embed_with_pool(
        self,
        ctx: base.Context,
        model: base.Embedder,
        texts: list[str],
    ) -> list[list[float]]:
        if self._pooler is None:
            raise AIProviderError(
                "embedder pooler is not configured",
                code="embedding.pooler_missing",
            )
        return await self._pooler.batch_embed_with_pool(ctx, model, texts)

    def get_model_name(self) -> str:
        return self._model_name

    def get_dimensions(self) -> int:
        return self._dimensions

    def get_model_id(self) -> str:
        return self._model_id

    # ── Internals ───────────────────────────────────────────────────

    def _supports_dimensions_param(self) -> bool:
        return self._supports_dimension_override and self._dimensions > 0

    async def _do_request_with_retry(
        self,
        ctx: base.Context,
        request: OpenAIEmbedRequest,
    ) -> httpx.Response:
        """POST the request, retrying transport errors with backoff."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        apply_custom_headers(headers, self._custom_headers)
        url = f"{self._base_url}{_EMBED_ENDPOINT}"
        payload = request.model_dump(mode="json", exclude_none=True)
        last_error: httpx.HTTPError | None = None
        for attempt in range(self._max_retries + 1):
            if attempt > 0:
                await asyncio.sleep(min(2 ** (attempt - 1), _MAX_BACKOFF_SECONDS))
            try:
                return await self._http_client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last_error = exc
                continue
        if last_error is not None:
            raise AIProviderError(
                f"send request: {last_error}",
                code="embedding.request_failed",
            ) from last_error
        raise AIProviderError(
            "send request: unknown error",
            code="embedding.request_failed",
        )

    async def aclose(self) -> None:
        """Release the underlying HTTP client."""
        await self._http_client.aclose()


async def new_openai_embedder(
    *,
    api_key: str,
    base_url: str,
    model_name: str,
    truncate_prompt_tokens: int,
    dimensions: int,
    model_id: str,
    pooler: base.EmbedderPooler | None,
    custom_headers: Mapping[str, str] | None = None,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    transport: httpx.AsyncBaseTransport | None = None,
) -> OpenAIEmbedder:
    """Build an OpenAI-compatible embedder (upstream ``NewOpenAIEmbedder``)."""
    if base_url == "":
        base_url = _DEFAULT_BASE_URL
    if model_name == "":
        raise ValidationError(
            code="embedding.model_name_required",
            message="model name is required",
        )
    if truncate_prompt_tokens == 0:
        truncate_prompt_tokens = _DEFAULT_TRUNCATE_PROMPT_TOKENS
    await validate_embedding_base_url(base_url)
    http_client = new_embedding_http_client(timeout=timeout, transport=transport)
    return OpenAIEmbedder(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        truncate_prompt_tokens=truncate_prompt_tokens,
        dimensions=dimensions,
        model_id=model_id,
        pooler=pooler,
        http_client=http_client,
        max_retries=max_retries,
        custom_headers=custom_headers,
    )


__all__ = [
    "OpenAIEmbedRequest",
    "OpenAIEmbedResponse",
    "OpenAIEmbedder",
    "new_openai_embedder",
]
