"""Managed-cloud rerank backend (mirrors the upstream contract).

``WeKnoraCloudReranker`` posts the rerank payload to the managed-cloud
``/api/v1/rerank`` path with the app-id / app-secret signed ``X-*``
headers. The remote model name may be overridden through the
``remote_model_name`` extra config entry. ``new_weknoracloud_reranker``
requires the cloud credentials and applies the SSRF gate.
"""

from __future__ import annotations

import json
import uuid
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
from src.ai.utils.signer import sign_request
from src.common.exception import ExternalServiceError, ValidationError
from src.common.json import JsonObject

# Path appended to the configured managed-cloud base URL.
_CLOUD_RERANK_PATH = "/api/v1/rerank"


class _CloudRankResult(BaseModel):
    """One ranked document from the managed-cloud ``results`` array."""

    model_config = ConfigDict(frozen=True)

    index: int = 0
    relevance_score: float = 0.0
    document: DocumentInfo = Field(default_factory=DocumentInfo)


class _CloudResponse(BaseModel):
    """Decoded managed-cloud rerank response body."""

    model_config = ConfigDict(frozen=True)

    results: list[_CloudRankResult] = Field(default_factory=list)


class WeKnoraCloudReranker:
    """Managed-cloud reranker (upstream ``WeKnoraCloudReranker``)."""

    def __init__(
        self,
        *,
        model_name: str,
        remote_model_name: str,
        model_id: str,
        app_id: str,
        api_key: str,
        base_url: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model_name = model_name
        self._remote_model_name = remote_model_name
        self._model_id = model_id
        self._app_id = app_id
        self._api_key = api_key
        self._base_url = base_url
        self._client = client if client is not None else new_rerank_http_client()

    def get_model_name(self) -> str:
        """Return the configured model name."""
        return self._model_name

    def get_model_id(self) -> str:
        """Return the configured model id."""
        return self._model_id

    async def rerank(self, query: str, documents: list[str]) -> list[RankResult]:
        """Rerank ``documents`` against ``query``; return the API results."""
        payload = {
            "model": self._effective_model_name(),
            "query": query,
            "documents": documents,
        }
        # Sign the exact bytes that will hit the wire: httpx serializes
        # JSON bodies as raw UTF-8 (ensure_ascii=False), so the signature
        # must cover that same representation for non-ASCII documents.
        body_text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        request_id = str(uuid.uuid4())
        headers = sign_request(self._app_id, self._api_key, request_id, body_text)
        headers["Content-Type"] = "application/json"
        url = f"{self._base_url}{_CLOUD_RERANK_PATH}"
        response = await post_json_with_ssrf_safety(
            self._client,
            url,
            json_body=cast(JsonObject, payload),
            headers=headers,
        )
        if response.status_code != 200:
            raise ExternalServiceError(
                code="rerank.cloud_api_error",
                message=(
                    f"weknoracloud reranker: status {response.status_code}: "
                    f"{response.text}"
                ),
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise ExternalServiceError(
                code="rerank.cloud_invalid_response",
                message=f"weknoracloud reranker: failed to decode response JSON: {exc}",
            ) from exc
        try:
            data = _CloudResponse.model_validate(body)
        except PydanticValidationError as exc:
            raise ExternalServiceError(
                code="rerank.cloud_invalid_response",
                message=f"weknoracloud reranker: failed to parse response: {exc}",
            ) from exc
        results: list[RankResult] = []
        for item in data.results:
            results.append(
                RankResult(
                    index=item.index,
                    document=DocumentInfo(text=item.document.text),
                    relevance_score=item.relevance_score,
                )
            )
        return results

    def _effective_model_name(self) -> str:
        """Return the remote model name, falling back to the local one."""
        return self._remote_model_name or self._model_name


async def new_weknoracloud_reranker(
    config: base.RerankerConfig,
    *,
    client: httpx.AsyncClient | None = None,
) -> base.Reranker:
    """Construct a ``WeKnoraCloudReranker`` (credentials + SSRF gate)."""
    if not config.app_id:
        raise ValidationError(
            code="rerank.cloud_app_id_required",
            message="WeKnoraCloud reranker: app id is required",
        )
    if not config.app_secret:
        raise ValidationError(
            code="rerank.cloud_app_secret_required",
            message="WeKnoraCloud reranker: app secret is required",
        )
    base_url = config.base_url.rstrip("/")
    await validate_rerank_base_url(base_url)
    remote_model_name = config.extra_config.get("remote_model_name", "").strip()
    return WeKnoraCloudReranker(
        model_name=config.model_name,
        remote_model_name=remote_model_name,
        model_id=config.model_id,
        app_id=config.app_id,
        api_key=config.app_secret,
        base_url=base_url,
        client=client,
    )


__all__ = ["WeKnoraCloudReranker", "new_weknoracloud_reranker"]
