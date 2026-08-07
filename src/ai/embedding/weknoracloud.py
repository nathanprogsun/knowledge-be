"""Managed-cloud embedding provider (weknoracloud.go).

Implements text vectorization over the managed cloud ``/api/v1/embeddings``
endpoint. Every request carries a per-request UUID and is signed with the
cloud ``app_id`` / ``app_secret`` pair via ``src.ai.utils.signer``; the
signed ``X-*`` headers are added alongside ``Content-Type``. Unlike the
other providers there is no pooler — ``batch_embed_with_pool`` embeds
directly — and ``supports_dimension_override`` is read straight from the
construction config.

The module references the base protocols through a module-level import of
``src.ai.embedding.base`` (rather than a from-import) to break the
base↔provider import cycle.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping

import httpx
from pydantic import BaseModel, ConfigDict, Field

import src.ai.embedding.base as base
from src.ai.embedding.transport import (
    new_embedding_http_client,
    validate_embedding_base_url,
)
from src.ai.utils.signer import sign_request
from src.common.exception import AIProviderError, ValidationError
from src.common.json import JsonValue

_DEFAULT_BASE_URL = "https://weknora.weixin.qq.com"
_EMBED_PATH = "/api/v1/embeddings"
_DEFAULT_TIMEOUT_SECONDS = 60.0


class WeKnoraCloudEmbedRequest(BaseModel):
    """Cloud ``/embeddings`` request body (``weKnoraCloudEmbedRequest``)."""

    model_config = ConfigDict(frozen=True)

    model: str
    input: list[str]
    dimensions: int | None = Field(default=None)
    truncate_prompt_tokens: int | None = Field(default=None)


class EmbeddingData(BaseModel):
    """One result entry of the ``data`` array."""

    model_config = ConfigDict(frozen=True)

    index: int = 0
    embedding: list[float] = Field(default_factory=list)


class WeKnoraCloudEmbedResponse(BaseModel):
    """Cloud ``/embeddings`` response body (``weKnoraCloudEmbedResponse``)."""

    model_config = ConfigDict(frozen=True)

    data: list[EmbeddingData] = Field(default_factory=list)


class WeKnoraCloudEmbedder:
    """Managed-cloud text vectorizer (``WeKnoraCloudEmbedder``)."""

    def __init__(
        self,
        *,
        model_name: str,
        remote_model_name: str,
        model_id: str,
        app_id: str,
        api_key: str,
        base_url: str,
        dimensions: int,
        supports_dimension_override: bool,
        http_client: httpx.AsyncClient,
    ) -> None:
        self._model_name = model_name
        self._remote_model_name = remote_model_name
        self._model_id = model_id
        self._app_id = app_id
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._dimensions = dimensions
        self._supports_dimension_override = supports_dimension_override
        self._http_client = http_client

    def set_supports_dimension_override(self, supported: bool) -> None:
        self._supports_dimension_override = supported

    async def embed(self, ctx: base.Context, text: str) -> list[float]:
        results = await self.batch_embed(ctx, [text])
        if not results:
            raise AIProviderError(
                "weknoracloud embedder: empty response",
                code="embedding.no_embedding_returned",
            )
        return results[0]

    async def batch_embed(self, ctx: base.Context, texts: list[str]) -> list[list[float]]:
        request = WeKnoraCloudEmbedRequest(
            model=self._effective_model_name(),
            input=list(texts),
        )
        if self._supports_dimension_override and self._dimensions > 0:
            request = request.model_copy(update={"dimensions": self._dimensions})
        payload = request.model_dump(mode="json", exclude_none=True)
        request_id = str(uuid.uuid4())
        signed = sign_request(self._app_id, self._api_key, request_id, _body_json(payload))
        headers = {"Content-Type": "application/json", **signed}
        url = f"{self._base_url}{_EMBED_PATH}"
        try:
            response = await self._http_client.post(url, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise AIProviderError(
                f"weknoracloud embedder: do request: {exc}",
                code="embedding.request_failed",
            ) from exc
        body = response.text
        if response.status_code != 200:
            raise AIProviderError(
                f"weknoracloud embedder: status {response.status_code}: {body}",
                code="embedding.api_error",
            )
        try:
            parsed = WeKnoraCloudEmbedResponse.model_validate_json(body)
        except ValueError as exc:
            raise AIProviderError(
                f"weknoracloud embedder: unmarshal: {exc}",
                code="embedding.invalid_response",
            ) from exc
        # Preserve the input order via the per-entry index.
        result: list[list[float] | None] = [None] * len(texts)
        for item in parsed.data:
            if 0 <= item.index < len(result):
                result[item.index] = item.embedding
        return [vector for vector in result if vector is not None]

    async def batch_embed_with_pool(
        self,
        ctx: base.Context,
        model: base.Embedder,
        texts: list[str],
    ) -> list[list[float]]:
        # The cloud embedder owns no pooler; upstream embeds directly.
        return await self.batch_embed(ctx, texts)

    def get_model_name(self) -> str:
        return self._model_name

    def get_dimensions(self) -> int:
        return self._dimensions

    def get_model_id(self) -> str:
        return self._model_id

    # ── Internals ───────────────────────────────────────────────────

    def _effective_model_name(self) -> str:
        return self._remote_model_name if self._remote_model_name else self._model_name

    async def aclose(self) -> None:
        """Release the underlying HTTP client."""
        await self._http_client.aclose()


def _body_json(payload: Mapping[str, JsonValue]) -> str:
    """Compact JSON serialization used for the request-body hash.

    Mirrors Go ``json.Marshal`` of the request struct: keys in field
    order with ``None`` fields omitted, no whitespace.
    """
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


async def new_weknoracloud_embedder(
    config: base.Config,
    *,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    transport: httpx.AsyncBaseTransport | None = None,
) -> WeKnoraCloudEmbedder:
    """Build a managed-cloud embedder (``NewWeKnoraCloudEmbedder``).

    ``config.app_id`` and ``config.app_secret`` are required; the secret
    is the signing key. ``remote_model_name`` in ``extra_config`` selects
    the upstream model name when set.
    """
    if config.app_id == "":
        raise ValidationError(
            code="embedding.weknoracloud_app_id_required",
            message="WeKnoraCloud embedder: AppID is required",
        )
    if config.app_secret == "":
        raise ValidationError(
            code="embedding.weknoracloud_app_secret_required",
            message="WeKnoraCloud embedder: AppSecret is required",
        )
    remote_model_name = (config.extra_config.get("remote_model_name") or "").strip()
    base_url = config.base_url.rstrip("/")
    if base_url == "":
        base_url = _DEFAULT_BASE_URL
    await validate_embedding_base_url(base_url)
    http_client = new_embedding_http_client(timeout=timeout, transport=transport)
    return WeKnoraCloudEmbedder(
        model_name=config.model_name,
        remote_model_name=remote_model_name,
        model_id=config.model_id,
        app_id=config.app_id,
        api_key=config.app_secret,
        base_url=base_url,
        dimensions=config.dimensions,
        supports_dimension_override=config.supports_dimension_override,
        http_client=http_client,
    )


__all__ = ["WeKnoraCloudEmbedder", "new_weknoracloud_embedder"]
