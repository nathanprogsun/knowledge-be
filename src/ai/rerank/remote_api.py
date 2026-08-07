"""OpenAI-compatible rerank backend over the shared ``POST /rerank`` shape.

``OpenAIReranker`` speaks the OpenAI-compatible endpoint used by the
generic provider route: a JSON request with ``model`` / ``query`` /
``documents`` and a JSON response whose ``results`` carry
``index`` + ``relevance_score`` (+ ``document``). The wire models
(``RerankRequest`` / ``RerankResponse`` / ``UsageInfo`` / ``RankResult``
/ ``DocumentInfo``) are the contract shared with every provider that
routes through this backend.

``new_openai_reranker`` applies the provider defaults — the fallback
endpoint, the SSRF gate, and the opt-in ``truncate_prompt_tokens``
parsing — before returning an instance.
"""

from __future__ import annotations

from collections.abc import Mapping

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic import (
    ValidationError as PydanticValidationError,
)

from src.ai.rerank.transport import (
    new_rerank_http_client,
    post_json_with_ssrf_safety,
    validate_rerank_base_url,
)
from src.common.exception import ExternalServiceError, ValidationError
from src.common.json import JsonObject, JsonValue

# Default endpoint when the model carries no base_url.
_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class DocumentInfo(BaseModel):
    """Text payload of one ranked document (may arrive as a bare string)."""

    model_config = ConfigDict(frozen=True)

    text: str = ""


class RankResult(BaseModel):
    """One ranked result.

    ``relevance_score`` falls back to ``score`` when the provider only
    returns the latter, and ``document`` may be a bare string or an
    object with a ``text`` field. Mirrors the upstream wire contract.
    """

    model_config = ConfigDict(frozen=True)

    index: int = 0
    document: DocumentInfo = Field(default_factory=DocumentInfo)
    relevance_score: float = 0.0

    @field_validator("document", mode="before")
    @classmethod
    def _coerce_document(cls, value: JsonValue) -> JsonValue:
        if isinstance(value, str):
            return {"text": value}
        if value is None:
            return {}
        return value

    @model_validator(mode="before")
    @classmethod
    def _apply_score_fallback(cls, data: JsonObject | None) -> JsonObject | None:
        if data is not None and data.get("relevance_score") is None and "score" in data:
            return {**data, "relevance_score": data["score"]}
        return data


class UsageInfo(BaseModel):
    """Token usage reported by the rerank endpoint."""

    model_config = ConfigDict(frozen=True)

    total_tokens: int = 0


class RerankResponse(BaseModel):
    """Decoded ``POST /rerank`` response body."""

    model_config = ConfigDict(frozen=True)

    id: str = ""
    model: str = ""
    usage: UsageInfo = Field(default_factory=UsageInfo)
    results: list[RankResult] = Field(default_factory=list)


class RerankRequest(BaseModel):
    """Request body sent to the ``POST /rerank`` endpoint.

    ``additional_data`` and ``truncate_prompt_tokens`` are optional and
    omitted from the wire when unset (``exclude_none`` at dump time).
    """

    model_config = ConfigDict(frozen=True)

    model: str
    query: str
    documents: list[str]
    additional_data: dict[str, JsonValue] | None = None
    truncate_prompt_tokens: int | None = None


class OpenAIReranker:
    """Generic OpenAI-compatible reranker.

    Stateless apart from the injected ``httpx.AsyncClient``; safe to
    construct per request. ``set_custom_headers`` attaches user-supplied
    request headers, mirroring the upstream extra-headers hook.
    """

    def __init__(
        self,
        *,
        model_name: str,
        model_id: str,
        api_key: str,
        base_url: str,
        truncate_prompt_tokens: int = 0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model_name = model_name
        self._model_id = model_id
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._truncate_prompt_tokens = truncate_prompt_tokens
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
        """Rerank ``documents`` against ``query``; return the API results.

        ``truncate_prompt_tokens`` is only sent when explicitly
        configured: providers that honor it keep only the tail of the
        templated prompt, which would cut the query off long documents
        and collapse every relevance score to near zero.
        """
        request = RerankRequest(
            model=self._model_name,
            query=query,
            documents=documents,
            truncate_prompt_tokens=self._truncate_prompt_tokens or None,
        )
        payload = request.model_dump(mode="json", exclude_none=True)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        headers.update(self._custom_headers)
        response = await post_json_with_ssrf_safety(
            self._client,
            f"{self._base_url}/rerank",
            json_body=payload,
            headers=headers,
        )
        if response.status_code != 200:
            reason = response.reason_phrase or ""
            status = f"{response.status_code} {reason}".strip()
            raise ExternalServiceError(
                code="rerank.api_error",
                message=f"Rerank API error: Http Status: {status}",
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise ExternalServiceError(
                code="rerank.invalid_response",
                message=f"failed to decode rerank response JSON: {exc}",
            ) from exc
        try:
            data = RerankResponse.model_validate(body)
        except PydanticValidationError as exc:
            raise ExternalServiceError(
                code="rerank.invalid_response",
                message=f"failed to parse rerank response: {exc}",
            ) from exc
        return data.results


async def new_openai_reranker(
    *,
    model_name: str,
    model_id: str,
    api_key: str,
    base_url: str = "",
    extra_config: Mapping[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
) -> OpenAIReranker:
    """Construct an ``OpenAIReranker`` with provider defaults + SSRF gate.

    Falls back to the OpenAI-compatible default endpoint when ``base_url``
    is empty, then validates the resolved URL. ``truncate_prompt_tokens``
    in ``extra_config`` opts into server-side prompt truncation; an
    invalid value is rejected.
    """
    resolved = base_url if base_url else _DEFAULT_BASE_URL
    await validate_rerank_base_url(resolved)
    truncate = _parse_truncate_prompt_tokens(extra_config)
    return OpenAIReranker(
        model_name=model_name,
        model_id=model_id,
        api_key=api_key,
        base_url=resolved,
        truncate_prompt_tokens=truncate,
        client=client,
    )


def _parse_truncate_prompt_tokens(extra_config: Mapping[str, str] | None) -> int:
    """Parse the opt-in ``truncate_prompt_tokens`` extra config value.

    Absent or blank means 0 (never sent). Any present value must be a
    positive integer; anything else is rejected up front rather than
    corrupting the rerank scores downstream.
    """
    raw = (extra_config or {}).get("truncate_prompt_tokens", "")
    if not raw.strip():
        return 0
    try:
        value = int(raw.strip())
    except ValueError:
        value = 0
    if value <= 0:
        raise ValidationError(
            code="rerank.invalid_truncate_prompt_tokens",
            message=f'invalid truncate_prompt_tokens in extra_config: "{raw.strip()}"',
        )
    return value


__all__ = [
    "DocumentInfo",
    "OpenAIReranker",
    "RankResult",
    "RerankRequest",
    "RerankResponse",
    "UsageInfo",
    "new_openai_reranker",
]
