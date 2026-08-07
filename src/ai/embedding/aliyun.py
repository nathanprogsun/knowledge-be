"""Aliyun DashScope multimodal embedding provider (aliyun.go).

Implements text vectorization over the dedicated DashScope multimodal
endpoint. The factory routes multimodal models
(``tongyi-embedding-vision-*`` / ``multimodal-embedding-*``) here; pure
text models reuse the OpenAI-compatible client instead because the
multimodal wire shape does not match the OpenAI ``/embeddings`` response.
The embedder owns a single ``httpx.AsyncClient``; transport errors are
retried with exponential backoff and non-2xx responses surface as
``AIProviderError``.

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

_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com"
_MULTIMODAL_ENDPOINT = "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
_DEFAULT_TIMEOUT_SECONDS = 60.0
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_TRUNCATE_PROMPT_TOKENS = 511
_EMBED_RETRIES = 3
_MAX_ERROR_BODY = 1000


class AliyunEmbedParameters(BaseModel):
    """Optional ``parameters`` block (``AliyunEmbedParameters``)."""

    model_config = ConfigDict(frozen=True)

    dimension: int | None = Field(default=None)


class AliyunContent(BaseModel):
    """One input item (``AliyunContent``)."""

    model_config = ConfigDict(frozen=True)

    text: str = ""


class AliyunEmbedInput(BaseModel):
    """The ``input`` block (``AliyunEmbedInput``)."""

    model_config = ConfigDict(frozen=True)

    contents: list[AliyunContent] = Field(default_factory=list)


class AliyunEmbedRequest(BaseModel):
    """DashScope multimodal embedding request (``AliyunEmbedRequest``)."""

    model_config = ConfigDict(frozen=True)

    model: str
    input: AliyunEmbedInput
    parameters: AliyunEmbedParameters | None = Field(default=None)


class AliyunEmbedding(BaseModel):
    """One result entry of the ``output.embeddings`` array."""

    model_config = ConfigDict(frozen=True)

    embedding: list[float] = Field(default_factory=list)
    text_index: int = 0


class AliyunEmbedOutput(BaseModel):
    """The response ``output`` block (``AliyunEmbedResponse.Output``)."""

    model_config = ConfigDict(frozen=True)

    embeddings: list[AliyunEmbedding] = Field(default_factory=list)


class AliyunEmbedResponse(BaseModel):
    """DashScope multimodal embedding response (``AliyunEmbedResponse``)."""

    model_config = ConfigDict(frozen=True)

    output: AliyunEmbedOutput = Field(default_factory=AliyunEmbedOutput)


class AliyunEmbedder:
    """Aliyun DashScope multimodal text vectorizer (``AliyunEmbedder``)."""

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
        for _ in range(_EMBED_RETRIES):
            embeddings = await self.batch_embed(ctx, [text])
            if embeddings:
                return embeddings[0]
        raise AIProviderError(
            "no embedding returned",
            code="embedding.no_embedding_returned",
        )

    async def batch_embed(self, ctx: base.Context, texts: list[str]) -> list[list[float]]:
        request = AliyunEmbedRequest(
            model=self._model_name,
            input=AliyunEmbedInput(contents=[AliyunContent(text=text) for text in texts]),
        )
        if self._supports_dimensions_param():
            request = request.model_copy(
                update={"parameters": AliyunEmbedParameters(dimension=self._dimensions)}
            )
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        apply_custom_headers(headers, self._custom_headers)
        url = f"{self._base_url}{_MULTIMODAL_ENDPOINT}"
        response = await post_embedding_with_retry(
            self._http_client,
            url,
            request.model_dump(mode="json", exclude_none=True),
            headers,
            self._max_retries,
        )
        body = response.text
        if response.status_code != 200:
            if len(body) > _MAX_ERROR_BODY:
                body = f"{body[:_MAX_ERROR_BODY]}... (truncated)"
            raise AIProviderError(
                f"BatchEmbed API error: Http Status {response.status_code} "
                f"{response.reason_phrase}, Response: {body}",
                code="embedding.api_error",
            )
        try:
            parsed = AliyunEmbedResponse.model_validate_json(body)
        except ValueError as exc:
            raise AIProviderError(
                f"unmarshal response: {exc}",
                code="embedding.invalid_response",
            ) from exc
        # Preserve the input order via text_index (upstream behaviour).
        result: list[list[float] | None] = [None] * len(texts)
        for entry in parsed.output.embeddings:
            index = entry.text_index
            if 0 <= index < len(result):
                result[index] = entry.embedding
        return [vector for vector in result if vector is not None]

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

    async def aclose(self) -> None:
        """Release the underlying HTTP client."""
        await self._http_client.aclose()


async def new_aliyun_embedder(
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
) -> AliyunEmbedder:
    """Build an Aliyun DashScope multimodal embedder (``NewAliyunEmbedder``).

    ``base_url`` defaults to the DashScope root; a ``/compatible-mode``
    path (OpenAI-compatible mode) is stripped because the multimodal
    endpoint is mounted on the root.
    """
    if base_url == "":
        base_url = _DEFAULT_BASE_URL
    base_url = base_url.rstrip("/")
    if "/compatible-mode/v1" in base_url:
        base_url = base_url.replace("/compatible-mode/v1", "", 1)
    if model_name == "":
        raise ValidationError(
            code="embedding.model_name_required",
            message="model name is required",
        )
    if truncate_prompt_tokens == 0:
        truncate_prompt_tokens = _DEFAULT_TRUNCATE_PROMPT_TOKENS
    await validate_embedding_base_url(base_url)
    http_client = new_embedding_http_client(timeout=timeout, transport=transport)
    return AliyunEmbedder(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        truncate_prompt_tokens=truncate_prompt_tokens,
        dimensions=dimensions,
        model_id=model_id,
        pooler=pooler,
        http_client=http_client,
        max_retries=max_retries,
    )


__all__ = ["AliyunEmbedder", "new_aliyun_embedder"]
