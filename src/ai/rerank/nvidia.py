"""NVIDIA rerank backend (mirrors the upstream contract).

``NvidiaReranker`` posts to the NVIDIA retrieval reranking endpoint with
a non-OpenAI body: the query and passages are ``{"text": ...}`` objects
under ``query`` / ``passages``. The response ``rankings`` array carries
``index`` + ``logit`` only, so each result's document text is taken from
the input document at that index. ``new_nvidia_reranker`` applies the
SSRF gate.
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

# Default NVIDIA retrieval reranking endpoint (a full URL, not a base).
_DEFAULT_ENDPOINT = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"


class _NvidiaRankResult(BaseModel):
    """One entry from NVIDIA's ``rankings`` array."""

    model_config = ConfigDict(frozen=True)

    index: int = 0
    logit: float = 0.0


class _NvidiaResponse(BaseModel):
    """Decoded NVIDIA rerank response body."""

    model_config = ConfigDict(frozen=True)

    model: str = ""
    rankings: list[_NvidiaRankResult] = Field(default_factory=list)


class NvidiaReranker:
    """NVIDIA reranker (upstream ``NvidiaReranker``)."""

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
            "query": {"text": query},
            "passages": [{"text": document} for document in documents],
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
                code="rerank.nvidia_api_error",
                message=f"Rerank API error: Http Status: {status}",
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise ExternalServiceError(
                code="rerank.nvidia_invalid_response",
                message=f"failed to decode NVIDIA rerank response JSON: {exc}",
            ) from exc
        try:
            data = _NvidiaResponse.model_validate(body)
        except PydanticValidationError as exc:
            raise ExternalServiceError(
                code="rerank.nvidia_invalid_response",
                message=f"failed to parse NVIDIA rerank response: {exc}",
            ) from exc
        results: list[RankResult] = []
        for item in data.rankings:
            # The response carries only index + logit; the document text is
            # recovered from the input list. Out-of-range indices degrade to
            # an empty text rather than crashing the whole rerank.
            document_text = (
                documents[item.index] if 0 <= item.index < len(documents) else ""
            )
            results.append(
                RankResult(
                    index=item.index,
                    document=DocumentInfo(text=document_text),
                    relevance_score=item.logit,
                )
            )
        return results


async def new_nvidia_reranker(
    config: base.RerankerConfig,
    *,
    client: httpx.AsyncClient | None = None,
) -> base.Reranker:
    """Construct a ``NvidiaReranker`` with the SSRF gate applied."""
    endpoint = config.base_url if config.base_url else _DEFAULT_ENDPOINT
    await validate_rerank_base_url(endpoint)
    return NvidiaReranker(
        model_name=config.model_name,
        model_id=config.model_id,
        api_key=config.api_key,
        endpoint=endpoint,
        client=client,
    )


__all__ = ["NvidiaReranker", "new_nvidia_reranker"]
