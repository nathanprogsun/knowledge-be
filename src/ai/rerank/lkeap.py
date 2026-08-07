"""Tencent Cloud LKEAP rerank backend (mirrors the upstream contract).

``LKEAPReranker`` calls the Tencent Cloud Knowledge Engine ``RunRerank``
atomic capability through the LKEAP SDK. Credentials use the Tencent
Cloud API key pair: the API key is the SecretId and the app secret is
the SecretKey (the ``secret_key`` extra config entry is the fallback).
The SDK client is injectable for tests; when omitted, ``new_lkeap_reranker``
builds one from the resolved credentials and region.
"""

from __future__ import annotations

from tencentcloud.common.credential import Credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.lkeap.v20240522.lkeap_client_async import LkeapClient
from tencentcloud.lkeap.v20240522.models import RunRerankRequest

import src.ai.rerank.base as base
from src.ai.rerank.remote_api import DocumentInfo, RankResult
from src.common.exception import ExternalServiceError, ValidationError

# RunRerank endpoint domain and default region.
_LKEAP_ENDPOINT = "lkeap.tencentcloudapi.com"
_LKEAP_DEFAULT_REGION = "ap-guangzhou"
_LKEAP_DEFAULT_MODEL = "lke-reranker-base"
_LKEAP_MAX_DOCUMENTS = 60


class LKEAPReranker:
    """Tencent Cloud LKEAP reranker (upstream ``LKEAPReranker``)."""

    def __init__(
        self,
        *,
        model_name: str,
        model_id: str,
        client: LkeapClient,
    ) -> None:
        self._model_name = model_name
        self._model_id = model_id
        self._client = client

    def get_model_name(self) -> str:
        """Return the rerank model name."""
        return self._model_name

    def get_model_id(self) -> str:
        """Return the model id."""
        return self._model_id

    async def rerank(self, query: str, documents: list[str]) -> list[RankResult]:
        """Rerank ``documents`` against ``query`` via ``RunRerank``."""
        if not documents:
            return []
        if len(documents) > _LKEAP_MAX_DOCUMENTS:
            raise ValidationError(
                code="rerank.lkeap_too_many_documents",
                message=(
                    "LKEAP rerank supports at most 60 documents, "
                    f"got {len(documents)}"
                ),
            )
        request = RunRerankRequest()
        request.Query = query
        request.Docs = documents
        request.Model = self._model_name
        try:
            response = await self._client.RunRerank(request)
        except Exception as exc:
            raise ExternalServiceError(
                code="rerank.lkeap_request_failed",
                message=f"LKEAP RunRerank: {exc}",
            ) from exc
        scores = response.ScoreList if response is not None else None
        if not scores:
            raise ExternalServiceError(
                code="rerank.lkeap_empty_score_list",
                message="LKEAP rerank API returned empty score list",
            )
        if len(scores) != len(documents):
            raise ExternalServiceError(
                code="rerank.lkeap_score_count_mismatch",
                message=(
                    f"LKEAP rerank score count mismatch: got {len(scores)} "
                    f"scores for {len(documents)} documents"
                ),
            )
        results: list[RankResult] = []
        for index, score in enumerate(scores):
            if score is None:
                results.append(
                    RankResult(
                        index=index,
                        document=DocumentInfo(text=documents[index]),
                    )
                )
            else:
                results.append(
                    RankResult(
                        index=index,
                        document=DocumentInfo(text=documents[index]),
                        relevance_score=float(score),
                    )
                )
        return results


async def new_lkeap_reranker(
    config: base.RerankerConfig,
    *,
    client: LkeapClient | None = None,
) -> base.Reranker:
    """Construct an ``LKEAPReranker`` (credentials resolved + SDK client)."""
    secret_id = config.api_key.strip()
    secret_key = config.app_secret.strip()
    if not secret_key:
        secret_key = config.extra_config.get("secret_key", "").strip()
    if not secret_id or not secret_key:
        raise ValidationError(
            code="rerank.lkeap_credentials_required",
            message=(
                "secret_id and secret_key are required for LKEAP rerank "
                "(set API Key and Secret Key)"
            ),
        )
    region = config.extra_config.get("region", "").strip() or _LKEAP_DEFAULT_REGION
    model_name = config.model_name.strip() or _LKEAP_DEFAULT_MODEL
    if client is None:
        credential = Credential(secret_id, secret_key)
        profile = ClientProfile()
        profile.httpProfile.endpoint = _LKEAP_ENDPOINT
        client = LkeapClient(credential, region, profile)
    return LKEAPReranker(
        model_name=model_name,
        model_id=config.model_id,
        client=client,
    )


__all__ = ["LKEAPReranker", "new_lkeap_reranker"]
