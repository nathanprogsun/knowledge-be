"""Volcengine managed Knowledge Service rerank backend (mirrors the
upstream contract).

``VolcengineReranker`` calls the managed Knowledge Service Rerank API
with AK/SK IAM signing. The signing port mirrors the upstream Go SDK's
``base.Credentials.Sign`` for the ``air`` service: an ``HMAC-SHA256``
signature over the canonical request covering ``content-type``,
``host``, ``x-content-sha256`` and ``x-date``.

The service accepts at most ``_VOLCENGINE_MAX_DOCUMENTS`` documents per
request, so larger candidate sets are split into limit-sized batches,
reranked concurrently (bounded by ``_VOLCENGINE_MAX_CONCURRENCY``), and
merged in input order — mirroring the upstream errgroup fan-out.
``new_volcengine_reranker`` requires the AK/SK credentials and applies
the SSRF gate.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import urllib.parse
from collections.abc import Mapping
from datetime import UTC, datetime

import httpx
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

import src.ai.rerank.base as base
from src.ai.rerank.remote_api import DocumentInfo, RankResult
from src.ai.rerank.transport import (
    new_rerank_http_client,
    validate_rerank_base_url,
)
from src.common.exception import ExternalServiceError, ValidationError

# Default managed Knowledge Service endpoint and request constants.
_VOLCENGINE_DEFAULT_BASE_URL = "https://api-knowledgebase.mlp.cn-beijing.volces.com"
_VOLCENGINE_RERANK_PATH = "/api/knowledge/service/rerank"
_VOLCENGINE_DEFAULT_MODEL = "doubao-seed-rerank"
_VOLCENGINE_DEFAULT_REGION = "cn-beijing"
_VOLCENGINE_DEFAULT_INSTRUCTION = (
    "Whether the Document answers the Query or matches the content retrieval intent"
)
_VOLCENGINE_MAX_DOCUMENTS = 50
_VOLCENGINE_MAX_CONCURRENCY = 4

# IAM signing service name used by the Knowledge Service API.
_VOLCENGINE_SERVICE = "air"
_VOLCENGINE_ALGORITHM = "HMAC-SHA256"
_VOLCENGINE_SIGNED_HEADERS = "content-type;host;x-content-sha256;x-date"

# ASCII characters the signer leaves unescaped (RFC 3986 unreserved set).
_UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_.~")


def _norm_uri(path: str) -> str:
    """Percent-encode ``path`` per the upstream ``normuri`` helper."""
    fragments = path.split("/")
    encoded: list[str] = []
    for fragment in fragments:
        out: list[str] = []
        for char in fragment:
            if char in _UNRESERVED:
                out.append(char)
            else:
                out.append("%" + char.encode("utf-8").hex().upper())
        encoded.append("".join(out))
    return "/".join(encoded)


def _canonical_query(params: Mapping[str, str]) -> str:
    """Encode query params as the upstream ``normquery`` helper."""
    if not params:
        return ""
    encoded = urllib.parse.urlencode(sorted(params.items()))
    return encoded.replace("+", "%20")


def _canonical_host(host: str) -> str:
    """Strip the default port from the signed ``host`` value.

    Mirrors the upstream signer: ``:80`` / ``:443`` are dropped from the
    canonical ``host`` header line while the wire ``Host`` header keeps
    the original value.
    """
    if ":" not in host:
        return host
    hostname, port = host.rsplit(":", 1)
    if port in ("80", "443"):
        return hostname
    return host


def _hmac_chain(secret_key: str, date: str, region: str, service: str) -> bytes:
    """Derive the signing key (date/region/service scoped HMAC chain)."""
    k_date = hmac.new(secret_key.encode("utf-8"), date.encode("utf-8"), hashlib.sha256).digest()
    k_region = hmac.new(k_date, region.encode("utf-8"), hashlib.sha256).digest()
    k_service = hmac.new(k_region, service.encode("utf-8"), hashlib.sha256).digest()
    return hmac.new(k_service, b"request", hashlib.sha256).digest()


def sign_rerank_request(
    *,
    access_key: str,
    secret_key: str,
    region: str,
    host: str,
    body_bytes: bytes,
    now: datetime | None = None,
) -> dict[str, str]:
    """Return the IAM-signed headers for one Knowledge Service request.

    Mirrors the upstream ``GetSignRequest`` path: the signature covers
    the HTTP method, the canonical URI and query, the signed headers
    (``content-type`` / ``host`` / ``x-content-sha256`` / ``x-date``)
    and the SHA-256 body hash. ``now`` lets tests pin ``X-Date``.
    """
    stamp = now or datetime.now(UTC)
    x_date = stamp.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = stamp.strftime("%Y%m%d")

    body_hash = hashlib.sha256(body_bytes).hexdigest()
    content_type = "application/json"
    canonical_headers = (
        f"content-type:{content_type}\n"
        f"host:{_canonical_host(host)}\n"
        f"x-content-sha256:{body_hash}\n"
        f"x-date:{x_date}\n"
    )
    canonical_request = "\n".join(
        [
            "POST",
            _norm_uri(_VOLCENGINE_RERANK_PATH),
            _canonical_query({}),
            canonical_headers,
            _VOLCENGINE_SIGNED_HEADERS,
            body_hash,
        ]
    )
    credential_scope = f"{date_stamp}/{region}/{_VOLCENGINE_SERVICE}/request"
    string_to_sign = "\n".join(
        [
            _VOLCENGINE_ALGORITHM,
            x_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ]
    )
    signing_key = _hmac_chain(secret_key, date_stamp, region, _VOLCENGINE_SERVICE)
    signature = hmac.new(
        signing_key,
        string_to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    authorization = (
        f"{_VOLCENGINE_ALGORITHM} Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={_VOLCENGINE_SIGNED_HEADERS}, Signature={signature}"
    )
    return {
        "Host": host,
        "Content-Type": content_type,
        "X-Date": x_date,
        "X-Content-Sha256": body_hash,
        "Authorization": authorization,
    }


class _VolcengineResult(BaseModel):
    """Scores container inside the Knowledge Service response."""

    model_config = ConfigDict(frozen=True)

    scores: list[float] = Field(default_factory=list)


class _VolcengineResponse(BaseModel):
    """Decoded Knowledge Service rerank response body."""

    model_config = ConfigDict(frozen=True)

    code: int = 0
    message: str = ""
    request_id: str = ""
    data: _VolcengineResult | None = None


class VolcengineReranker:
    """Volcengine managed Knowledge Service reranker."""

    def __init__(
        self,
        *,
        model_name: str,
        instruction: str,
        model_id: str,
        access_key: str,
        secret_key: str,
        region: str,
        base_url: str,
        host: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model_name = model_name
        self._instruction = instruction
        self._model_id = model_id
        self._access_key = access_key
        self._secret_key = secret_key
        self._region = region
        self._base_url = base_url
        self._host = host
        self._client = client if client is not None else new_rerank_http_client()

    def get_model_name(self) -> str:
        """Return the rerank model name."""
        return self._model_name

    def get_model_id(self) -> str:
        """Return the model id."""
        return self._model_id

    async def rerank(self, query: str, documents: list[str]) -> list[RankResult]:
        """Rerank ``documents`` against ``query``; return the API results."""
        if not documents:
            return []
        results: list[RankResult | None] = [None] * len(documents)
        semaphore = asyncio.Semaphore(_VOLCENGINE_MAX_CONCURRENCY)

        async def rerank_chunk(start: int, end: int) -> None:
            async with semaphore:
                scores = await self._rerank_batch(query, documents[start:end])
                for offset, score in enumerate(scores):
                    index = start + offset
                    results[index] = RankResult(
                        index=index,
                        document=DocumentInfo(text=documents[index]),
                        relevance_score=score,
                    )

        tasks = [
            rerank_chunk(start, min(start + _VOLCENGINE_MAX_DOCUMENTS, len(documents)))
            for start in range(0, len(documents), _VOLCENGINE_MAX_DOCUMENTS)
        ]
        await asyncio.gather(*tasks)
        return [result for result in results if result is not None]

    async def _rerank_batch(self, query: str, documents: list[str]) -> list[float]:
        """Score one batch (sized within the API limit) in input order."""
        payload = {
            "datas": [{"query": query, "content": document} for document in documents],
            "rerank_model": self._model_name,
            "rerank_instruction": self._instruction,
        }
        body_bytes = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        headers = sign_rerank_request(
            access_key=self._access_key,
            secret_key=self._secret_key,
            region=self._region,
            host=self._host,
            body_bytes=body_bytes,
        )
        url = f"{self._base_url}{_VOLCENGINE_RERANK_PATH}"
        try:
            response = await self._client.post(url, content=body_bytes, headers=headers)
        except httpx.HTTPError as exc:
            raise ExternalServiceError(
                code="rerank.volcengine_request_failed",
                message=f"call Volcengine rerank: {exc}",
            ) from exc
        try:
            body = response.json()
        except ValueError as exc:
            raise ExternalServiceError(
                code="rerank.volcengine_invalid_response",
                message=f"failed to decode Volcengine rerank response JSON: {exc}",
            ) from exc
        try:
            data = _VolcengineResponse.model_validate(body)
        except PydanticValidationError as exc:
            raise ExternalServiceError(
                code="rerank.volcengine_invalid_response",
                message=f"failed to parse Volcengine rerank response: {exc}",
            ) from exc
        if data.code != 0:
            raise ExternalServiceError(
                code="rerank.volcengine_api_error",
                message=f"Volcengine rerank API error {data.code}: {data.message}",
            )
        if data.data is None:
            raise ExternalServiceError(
                code="rerank.volcengine_empty_response",
                message="Volcengine rerank returned an empty response",
            )
        scores = data.data.scores
        if len(scores) != len(documents):
            raise ExternalServiceError(
                code="rerank.volcengine_score_count_mismatch",
                message=(
                    f"Volcengine rerank score count mismatch: got {len(scores)} "
                    f"scores for {len(documents)} documents"
                ),
            )
        return scores


async def new_volcengine_reranker(
    config: base.RerankerConfig,
    *,
    client: httpx.AsyncClient | None = None,
) -> base.Reranker:
    """Construct a ``VolcengineReranker`` (AK/SK + SSRF gate)."""
    access_key = config.api_key.strip()
    secret_key = config.app_secret.strip()
    if not secret_key:
        secret_key = config.extra_config.get("secret_key", "").strip()
    if not access_key or not secret_key:
        raise ValidationError(
            code="rerank.volcengine_credentials_required",
            message="access key and secret key are required for Volcengine rerank",
        )
    base_url = config.base_url.strip().rstrip("/")
    if not base_url:
        base_url = _VOLCENGINE_DEFAULT_BASE_URL
    await validate_rerank_base_url(base_url)

    model_name = config.model_name.strip() or _VOLCENGINE_DEFAULT_MODEL
    region = config.extra_config.get("region", "").strip() or _VOLCENGINE_DEFAULT_REGION
    instruction = (
        config.extra_config.get("instruction", "").strip()
        or _VOLCENGINE_DEFAULT_INSTRUCTION
    )
    parsed = urllib.parse.urlsplit(base_url)
    host = parsed.netloc
    return VolcengineReranker(
        model_name=model_name,
        instruction=instruction,
        model_id=config.model_id,
        access_key=access_key,
        secret_key=secret_key,
        region=region,
        base_url=base_url,
        host=host,
        client=client,
    )


__all__ = [
    "VolcengineReranker",
    "new_volcengine_reranker",
    "sign_rerank_request",
]
