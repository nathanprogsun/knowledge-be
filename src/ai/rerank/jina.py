"""Jina AI rerank backend (mirrors the upstream contract).

``JinaReranker`` posts a flat ``model`` / ``query`` / ``documents`` body
to ``{base_url}/rerank``. Jina does not support
``truncate_prompt_tokens``, so it is never sent. The results array uses
the shared rank-result shape. ``new_jina_reranker`` applies the SSRF gate.
"""

from __future__ import annotations

from collections.abc import Mapping

import httpx
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

import src.ai.rerank.base as base
from src.ai.rerank.remote_api import RankResult, UsageInfo
from src.ai.rerank.transport import (
    new_rerank_http_client,
    post_json_with_ssrf_safety,
    validate_rerank_base_url,
)
from src.common.exception import ExternalServiceError

# Default Jina base URL (the ``/rerank`` suffix is appended per request).
_DEFAULT_BASE_URL = "https://api.jina.ai/v1"


class _JinaResponse(BaseModel):
    """Decoded Jina rerank response body."""

    model_config = ConfigDict(frozen=True)

    model: str = ""
    results: list[RankResult] = Field(default_factory=list)
    usage: UsageInfo = Field(default_factory=UsageInfo)


class JinaReranker:
    """Jina AI reranker (upstream ``JinaReranker``)."""

    def __init__(
        self,
        *,
        model_name: str,
        model_id: str,
        api_key: str,
        base_url: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model_name = model_name
        self._model_id = model_id
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
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
        url = f"{self._base_url}/rerank"
        response = await post_json_with_ssrf_safety(
            self._client,
            url,
            json_body=payload,
            headers=headers,
        )
        if response.status_code != 200:
            reason = response.reason_phrase or ""
            status = f"{response.status_code} {reason}".strip()
            raise ExternalServiceError(
                code="rerank.jina_api_error",
                message=f"Rerank API error: Http Status: {status}",
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise ExternalServiceError(
                code="rerank.jina_invalid_response",
                message=f"failed to decode Jina rerank response JSON: {exc}",
            ) from exc
        try:
            data = _JinaResponse.model_validate(body)
        except PydanticValidationError as exc:
            raise ExternalServiceError(
                code="rerank.jina_invalid_response",
                message=f"failed to parse Jina rerank response: {exc}",
            ) from exc
        return data.results


async def new_jina_reranker(
    config: base.RerankerConfig,
    *,
    client: httpx.AsyncClient | None = None,
) -> base.Reranker:
    """Construct a ``JinaReranker`` with the SSRF gate applied."""
    base_url = config.base_url if config.base_url else _DEFAULT_BASE_URL
    await validate_rerank_base_url(base_url)
    return JinaReranker(
        model_name=config.model_name,
        model_id=config.model_id,
        api_key=config.api_key,
        base_url=base_url,
        client=client,
    )


__all__ = ["JinaReranker", "new_jina_reranker"]
