"""Zhipu AI rerank backend (mirrors the upstream contract).

``ZhipuReranker`` posts a flat ``model`` / ``query`` / ``documents`` body
to the Zhipu rerank endpoint; ``top_n`` and ``return_raw_scores`` are
omitted from the wire when at their zero values, while
``return_documents`` is always true. The configured base URL replaces the
default endpoint wholesale. ``new_zhipu_reranker`` applies the SSRF gate.
"""

from __future__ import annotations

from collections.abc import Mapping

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

# Default Zhipu rerank endpoint (a full URL, not a base).
_DEFAULT_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/rerank"


class _ZhipuRankResult(BaseModel):
    """One ranked document from Zhipu's ``results`` array."""

    model_config = ConfigDict(frozen=True)

    index: int = 0
    relevance_score: float = 0.0
    document: str = ""


class _ZhipuUsage(BaseModel):
    """Token usage reported by the Zhipu endpoint."""

    model_config = ConfigDict(frozen=True)

    total_tokens: int = 0
    prompt_tokens: int = 0


class _ZhipuResponse(BaseModel):
    """Decoded Zhipu rerank response body."""

    model_config = ConfigDict(frozen=True)

    request_id: str = ""
    id: str = ""
    results: list[_ZhipuRankResult] = Field(default_factory=list)
    usage: _ZhipuUsage = Field(default_factory=_ZhipuUsage)


class ZhipuReranker:
    """Zhipu AI reranker (upstream ``ZhipuReranker``)."""

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
            "query": query,
            "documents": documents,
            "return_documents": True,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        headers.update(self._custom_headers)
        response = await post_json_with_ssrf_safety(
            self._client,
            self._endpoint,
            json_body=payload,
            headers=headers,
        )
        if response.status_code != 200:
            reason = response.reason_phrase or ""
            status = f"{response.status_code} {reason}".strip()
            raise ExternalServiceError(
                code="rerank.zhipu_api_error",
                message=(
                    f"Zhipu rerank API error: Http Status: {status}, "
                    f"Body: {response.text}"
                ),
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise ExternalServiceError(
                code="rerank.zhipu_invalid_response",
                message=f"failed to decode Zhipu rerank response JSON: {exc}",
            ) from exc
        try:
            data = _ZhipuResponse.model_validate(body)
        except PydanticValidationError as exc:
            raise ExternalServiceError(
                code="rerank.zhipu_invalid_response",
                message=f"failed to parse Zhipu rerank response: {exc}",
            ) from exc
        results: list[RankResult] = []
        for item in data.results:
            results.append(
                RankResult(
                    index=item.index,
                    document=DocumentInfo(text=item.document),
                    relevance_score=item.relevance_score,
                )
            )
        return results


async def new_zhipu_reranker(
    config: base.RerankerConfig,
    *,
    client: httpx.AsyncClient | None = None,
) -> base.Reranker:
    """Construct a ``ZhipuReranker`` with the SSRF gate applied."""
    endpoint = config.base_url if config.base_url else _DEFAULT_ENDPOINT
    await validate_rerank_base_url(endpoint)
    return ZhipuReranker(
        model_name=config.model_name,
        model_id=config.model_id,
        api_key=config.api_key,
        endpoint=endpoint,
        client=client,
    )


__all__ = ["ZhipuReranker", "new_zhipu_reranker"]
