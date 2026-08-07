"""Gemini embedding provider (gemini.go).

Implements text vectorization over the native Gemini
``batchEmbedContents`` REST API. The request wraps every text in its own
``requests[]`` entry and the response returns one ``values`` vector per
input (the count must match). Authentication uses an ``x-goog-api-key``
header. The default base URL is the Gemini v1beta endpoint; an OpenAI
compat suffix (``/openai``) and a ``models/`` model-name prefix are
normalized away.

The module references the base protocols through a module-level import of
``src.ai.embedding.base`` (rather than a from-import) to break the
base↔provider import cycle.
"""

from __future__ import annotations

from collections.abc import Mapping

import httpx
from pydantic import BaseModel, ConfigDict, Field

import src.ai.embedding.base as base
from src.ai.embedding.transport import (
    apply_custom_headers,
    new_embedding_http_client,
    post_embedding_with_retry,
    validate_embedding_base_url,
)
from src.common.exception import AIProviderError, ValidationError

_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
_DEFAULT_TIMEOUT_SECONDS = 60.0
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_TRUNCATE_PROMPT_TOKENS = 511
_MAX_ERROR_BODY = 1000


class GeminiPart(BaseModel):
    """A single content part (``geminiPart``)."""

    model_config = ConfigDict(frozen=True)

    text: str


class GeminiContent(BaseModel):
    """Content block (``geminiContent``)."""

    model_config = ConfigDict(frozen=True)

    parts: list[GeminiPart] = Field(default_factory=list)


class GeminiEmbedRequest(BaseModel):
    """One ``requests[]`` entry (``geminiEmbedRequest``)."""

    model_config = ConfigDict(frozen=True)

    model: str
    content: GeminiContent
    task_type: str | None = Field(default=None, serialization_alias="taskType")
    output_dimensionality: int | None = Field(default=None)


class GeminiBatchEmbedRequest(BaseModel):
    """``batchEmbedContents`` request body (``geminiBatchEmbedRequest``)."""

    model_config = ConfigDict(frozen=True)

    requests: list[GeminiEmbedRequest] = Field(default_factory=list)


class GeminiEmbedding(BaseModel):
    """One result entry of the ``embeddings`` array (``geminiEmbedding``)."""

    model_config = ConfigDict(frozen=True)

    values: list[float] = Field(default_factory=list)


class GeminiBatchEmbedResponse(BaseModel):
    """``batchEmbedContents`` response body (``geminiBatchEmbedResponse``)."""

    model_config = ConfigDict(frozen=True)

    embeddings: list[GeminiEmbedding] = Field(default_factory=list)


class GeminiEmbedder:
    """Gemini text vectorizer (``GeminiEmbedder``)."""

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
        self._custom_headers: dict[str, str] = {}
        self._supports_dimension_override = False

    # ── Interface setters (base ``SupportsDimensionOverride`` /
    #    ``CustomHeadersSettable``) ──────────────────────────────────

    def set_custom_headers(self, headers: Mapping[str, str] | None) -> None:
        self._custom_headers = dict(headers) if headers else {}

    def set_supports_dimension_override(self, supported: bool) -> None:
        self._supports_dimension_override = supported

    # ── Embedder interface ──────────────────────────────────────────

    async def embed(self, ctx: base.Context, text: str) -> list[float]:
        embeddings = await self.batch_embed(ctx, [text])
        if not embeddings:
            raise AIProviderError(
                "no embedding returned",
                code="embedding.no_embedding_returned",
            )
        return embeddings[0]

    async def batch_embed(self, ctx: base.Context, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        requests = [
            GeminiEmbedRequest(
                model=f"models/{self._model_name}",
                content=GeminiContent(parts=[GeminiPart(text=text)]),
                output_dimensionality=(
                    self._dimensions
                    if self._supports_dimension_override and self._dimensions > 0
                    else None
                ),
            )
            for text in texts
        ]
        request = GeminiBatchEmbedRequest(requests=requests)
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._api_key,
        }
        apply_custom_headers(headers, self._custom_headers)
        url = f"{self._base_url}/models/{self._model_name}:batchEmbedContents"
        response = await post_embedding_with_retry(
            self._http_client,
            url,
            request.model_dump(mode="json", exclude_none=True, by_alias=True),
            headers,
            self._max_retries,
        )
        body = response.text
        if response.status_code != 200:
            if len(body) > _MAX_ERROR_BODY:
                body = f"{body[:_MAX_ERROR_BODY]}... (truncated)"
            raise AIProviderError(
                f"Gemini BatchEmbed API error: Http Status {response.status_code} "
                f"{response.reason_phrase}, Response: {body}",
                code="embedding.api_error",
            )
        try:
            parsed = GeminiBatchEmbedResponse.model_validate_json(body)
        except ValueError as exc:
            raise AIProviderError(
                f"unmarshal response: {exc}",
                code="embedding.invalid_response",
            ) from exc
        if len(parsed.embeddings) != len(texts):
            raise AIProviderError(
                f"Gemini BatchEmbed returned {len(parsed.embeddings)} embeddings "
                f"for {len(texts)} inputs",
                code="embedding.invalid_response",
            )
        return [embedding.values for embedding in parsed.embeddings]

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

    async def aclose(self) -> None:
        """Release the underlying HTTP client."""
        await self._http_client.aclose()


async def new_gemini_embedder(
    *,
    api_key: str,
    base_url: str,
    model_name: str,
    truncate_prompt_tokens: int,
    dimensions: int,
    model_id: str,
    pooler: base.EmbedderPooler | None,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    transport: httpx.AsyncBaseTransport | None = None,
) -> GeminiEmbedder:
    """Build a Gemini embedder (``NewGeminiEmbedder``).

    ``base_url`` defaults to the Gemini v1beta endpoint; an OpenAI compat
    suffix is stripped and a ``models/`` model-name prefix is normalized
    away.
    """
    if model_name == "":
        raise ValidationError(
            code="embedding.model_name_required",
            message="model name is required",
        )
    if truncate_prompt_tokens == 0:
        truncate_prompt_tokens = _DEFAULT_TRUNCATE_PROMPT_TOKENS
    if base_url == "":
        base_url = _DEFAULT_BASE_URL
    base_url = base_url.rstrip("/")
    if base_url.endswith("/openai"):
        base_url = base_url[: -len("/openai")]
    await validate_embedding_base_url(base_url)
    http_client = new_embedding_http_client(timeout=timeout, transport=transport)
    return GeminiEmbedder(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name.removeprefix("models/"),
        truncate_prompt_tokens=truncate_prompt_tokens,
        dimensions=dimensions,
        model_id=model_id,
        pooler=pooler,
        http_client=http_client,
        max_retries=max_retries,
    )


__all__ = ["GeminiEmbedder", "new_gemini_embedder"]
