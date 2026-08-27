"""Aliyun DashScope rerank backend (mirrors the upstream contract).

``AliyunReranker`` posts to the DashScope text-rerank endpoint with a
``model`` / ``input`` / ``parameters`` body and reads the ranked
documents from ``output.results``. The configured base URL replaces the
default endpoint wholesale — DashScope endpoints are full URLs, not
bases. ``new_aliyun_reranker`` applies the SSRF gate before returning an
instance.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import httpx
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

import src.ai.rerank.base as base
from src.ai.rerank.remote_api import DocumentInfo, RankResult
from src.ai.rerank.transport import (
    new_rerank_http_client,
    post_json_with_ssrf_safety,
    validate_rerank_base_url,
)
from src.common.exception import ExternalServiceError
from src.common.json import JsonObject

# Default DashScope text-rerank endpoint (a full URL, not a base).
_DEFAULT_ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"


class _AliyunDocument(BaseModel):
    """Document text echoed back by the DashScope endpoint."""

    model_config = ConfigDict(frozen=True)

    text: str = ""


class _AliyunRankResult(BaseModel):
    """One ranked document from ``output.results``."""

    model_config = ConfigDict(frozen=True)

    document: _AliyunDocument = Field(default_factory=_AliyunDocument)
    index: int = 0
    relevance_score: float = 0.0


class _AliyunOutput(BaseModel):
    """DashScope response output container."""

    model_config = ConfigDict(frozen=True)

    results: list[_AliyunRankResult] = Field(default_factory=list)


class _AliyunResponse(BaseModel):
    """Decoded DashScope rerank response body."""

    model_config = ConfigDict(frozen=True)

    output: _AliyunOutput = Field(default_factory=_AliyunOutput)


class AliyunReranker:
    """Aliyun DashScope reranker (upstream ``AliyunReranker``)."""

    def __init__(
        self,
        *,
        model_name: str,
        model_id: str,
        api_key: str,
        endpoint: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model_name = model_name
        self._model_id = model_id
        self._api_key = api_key
        self._endpoint = endpoint
        self._custom_headers: dict[str, str] = {}
        self._client = client if client is not None else new_rerank_http_client()

    def set_custom_headers(self, headers: Mapping[str, str]) -> None:
        """Replace the user-supplied request headers."""
        self._custom_headers = dict(headers)

    def get_model_name(self) -> str:
        """Return the configured model name."""
        return self._model_name

    def get_model_id(self) -> str:
        """Return the configured model id."""
        return self._model_id

    async def rerank(self, query: str, documents: list[str]) -> list[RankResult]:
        """Rerank ``documents`` against ``query``; return the API results."""
        payload = {
            "model": self._model_name,
            "input": {"query": query, "documents": documents},
            "parameters": {"return_documents": True, "top_n": len(documents)},
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        headers.update(self._custom_headers)
        response = await post_json_with_ssrf_safety(
            self._client,
            self._endpoint,
            json_body=cast(JsonObject, payload),
            headers=headers,
        )
        if response.status_code != 200:
            reason = response.reason_phrase or ""
            status = f"{response.status_code} {reason}".strip()
            raise ExternalServiceError(
                code="rerank.aliyun_api_error",
                message=(f"Aliyun rerank API error: Http Status: {status}, Body: {response.text}"),
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise ExternalServiceError(
                code="rerank.aliyun_invalid_response",
                message=f"failed to decode Aliyun rerank response JSON: {exc}",
            ) from exc
        try:
            data = _AliyunResponse.model_validate(body)
        except PydanticValidationError as exc:
            raise ExternalServiceError(
                code="rerank.aliyun_invalid_response",
                message=f"failed to parse Aliyun rerank response: {exc}",
            ) from exc
        results: list[RankResult] = []
        for item in data.output.results:
            results.append(
                RankResult(
                    index=item.index,
                    document=DocumentInfo(text=item.document.text),
                    relevance_score=item.relevance_score,
                )
            )
        return results


async def new_aliyun_reranker(
    config: base.RerankerConfig,
    *,
    client: httpx.AsyncClient | None = None,
) -> base.Reranker:
    """Construct an ``AliyunReranker`` with the SSRF gate applied."""
    endpoint = config.base_url if config.base_url else _DEFAULT_ENDPOINT
    await validate_rerank_base_url(endpoint)
    return AliyunReranker(
        model_name=config.model_name,
        model_id=config.model_id,
        api_key=config.api_key,
        endpoint=endpoint,
        client=client,
    )


__all__ = ["AliyunReranker", "new_aliyun_reranker"]
