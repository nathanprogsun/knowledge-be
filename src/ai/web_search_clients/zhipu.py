"""Zhipu AI standalone Web Search API client.

Maps the Zhipu web search endpoint. The API URL is hardcoded (not
tenant-configurable). Provider-specific options arrive through
``extra_config``: ``search_engine`` (default ``search_std``) and
``content_size`` (default ``medium``).
"""

from __future__ import annotations

import json

import httpx

from src.ai.web_search_clients._base import (
    HttpSearchClient,
    SearchHit,
    coerce_extra,
    coerce_str,
    parse_iso_datetime,
)
from src.ai.web_search_clients.proxy import build_http_client
from src.common.exception import ExternalServiceError, ValidationError
from src.common.json import JsonObject

_DEFAULT_URL = "https://open.bigmodel.cn/api/paas/v4/web_search"
_TIMEOUT_SECONDS = 15.0
_SOURCE = "zhipu"
_DEFAULT_RESULTS = 10
_MAX_RESULTS = 50
_MAX_QUERY_RUNES = 70
_MAX_RESPONSE_BYTES = 2 << 20  # 2MB
_DEFAULT_SEARCH_ENGINE = "search_std"
_DEFAULT_CONTENT_SIZE = "medium"

_VALID_SEARCH_ENGINES = frozenset(
    {"search_std", "search_pro", "search_pro_sogou", "search_pro_quark"}
)
_VALID_CONTENT_SIZES = frozenset({"medium", "high"})


def _zhipu_options(extra_config: dict[str, str]) -> tuple[str, str]:
    search_engine = extra_config.get("search_engine", "").strip() or _DEFAULT_SEARCH_ENGINE
    content_size = extra_config.get("content_size", "").strip() or _DEFAULT_CONTENT_SIZE
    return search_engine, content_size


def validate_zhipu_parameters(api_key: str, extra_config: dict[str, str]) -> None:
    """Validate the Zhipu credentials and provider-specific options."""
    if not api_key.strip():
        raise ValidationError(
            code="web_search_provider.api_key_required",
            message="API key is required for Zhipu provider",
        )
    search_engine, content_size = _zhipu_options(extra_config)
    if search_engine not in _VALID_SEARCH_ENGINES:
        raise ValidationError(
            code="web_search_provider.invalid_config",
            message=f"invalid Zhipu search engine: {search_engine}",
        )
    if content_size not in _VALID_CONTENT_SIZES:
        raise ValidationError(
            code="web_search_provider.invalid_config",
            message=f"invalid Zhipu content size: {content_size}",
        )


def normalize_zhipu_query(query: str) -> str:
    """Trim the query and truncate it to the API rune budget."""
    query = query.strip()
    if len(query) <= _MAX_QUERY_RUNES:
        return query
    return query[:_MAX_QUERY_RUNES]


class ZhipuProvider(HttpSearchClient):
    """Web search against the Zhipu AI Web Search API."""

    provider_type = _SOURCE

    def __init__(
        self,
        *,
        client: httpx.Client,
        api_key: str,
        search_engine: str = _DEFAULT_SEARCH_ENGINE,
        content_size: str = _DEFAULT_CONTENT_SIZE,
        base_url: str = _DEFAULT_URL,
    ) -> None:
        super().__init__(client=client)
        self._api_key = api_key
        self._search_engine = search_engine
        self._content_size = content_size
        self._base_url = base_url

    def search(
        self,
        query: str,
        max_results: int,
        include_date: bool,
    ) -> list[SearchHit]:
        prepared = normalize_zhipu_query(query)
        if not prepared:
            raise ValidationError(
                code="web_search_provider.query_required",
                message="query must be a non-empty string",
            )
        count = max_results if max_results > 0 else _DEFAULT_RESULTS
        count = min(count, _MAX_RESULTS)
        response = self._client.post(
            self._base_url,
            json={
                "search_query": prepared,
                "search_engine": self._search_engine,
                "search_intent": False,
                "count": count,
                "content_size": self._content_size,
            },
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        if len(response.content) > _MAX_RESPONSE_BYTES:
            raise ExternalServiceError(
                code="web_search_provider.invalid_response",
                message=f"zhipu response exceeds {_MAX_RESPONSE_BYTES} bytes",
            )
        if response.status_code != 200:
            raise ExternalServiceError(
                code="web_search_provider.search_failed",
                message=_zhipu_status_error(response.status_code, response.text),
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ExternalServiceError(
                code="web_search_provider.invalid_response",
                message=f"failed to unmarshal Zhipu response: {exc}",
            ) from exc
        if not isinstance(data, dict):
            raise ExternalServiceError(
                code="web_search_provider.invalid_response",
                message="zhipu returned a non-object response",
            )
        error = data.get("error")
        if isinstance(error, dict):
            error_code = coerce_str(error.get("code"))
            error_message = coerce_str(error.get("message"))
            if error_code or error_message:
                raise ExternalServiceError(
                    code="web_search_provider.search_failed",
                    message=f"Zhipu API error ({error_code}): {error_message}",
                )
        results = data.get("search_result")
        if not isinstance(results, list):
            return []
        hits: list[SearchHit] = []
        for item in results:
            if len(hits) >= count:
                break
            if not isinstance(item, dict):
                continue
            title = coerce_str(item.get("title"))
            link = coerce_str(item.get("link"))
            if not title and not link:
                continue
            hit: SearchHit = {
                "title": title,
                "url": link,
                "snippet": coerce_str(item.get("content")),
                "source": _SOURCE,
            }
            if include_date:
                published = parse_iso_datetime(coerce_str(item.get("publish_date")))
                if published is not None:
                    hit["published_at"] = published.isoformat()
            hits.append(hit)
        return hits


def _zhipu_status_error(status_code: int, text: str) -> str:
    """Build the status-failure message, preferring the API error object."""
    try:
        data = json.loads(text.strip()) if text.strip() else None
    except ValueError:
        data = None
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            code = coerce_str(error.get("code"))
            message = coerce_str(error.get("message"))
            if code or message:
                return f"Zhipu API returned status {status_code} ({code}): {message}"
    detail = text.strip()[:4096]
    if not detail:
        return f"Zhipu API returned status {status_code}"
    return f"Zhipu API returned status {status_code}: {detail}"


def build_zhipu_client(params: JsonObject) -> ZhipuProvider:
    """Build a :class:`ZhipuProvider` from a JSON parameter blob."""
    api_key = coerce_str(params.get("api_key"))
    extra = coerce_extra(params.get("extra_config"))
    validate_zhipu_parameters(api_key, extra)
    search_engine, content_size = _zhipu_options(extra)
    client = build_http_client(
        timeout=_TIMEOUT_SECONDS,
        proxy_url=coerce_str(params.get("proxy_url")),
    )
    return ZhipuProvider(
        client=client,
        api_key=api_key,
        search_engine=search_engine,
        content_size=content_size,
    )


__all__ = [
    "ZhipuProvider",
    "build_zhipu_client",
    "normalize_zhipu_query",
    "validate_zhipu_parameters",
]
