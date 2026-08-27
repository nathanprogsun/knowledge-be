"""Baidu AI Search web-search client.

Maps the Baidu AI Search API. The API URL is hardcoded (not
tenant-configurable). The query is normalized to the API's length
constraints (72 units, CJK runes count double). Per the upstream
client, the explicit proxy parameter is intentionally not applied here
— only the environment proxy configuration is honored.
"""

from __future__ import annotations

import re
from datetime import datetime

import httpx

from src.ai.web_search_clients._base import (
    HttpSearchClient,
    SearchHit,
    coerce_str,
)
from src.ai.web_search_clients.proxy import build_http_client
from src.common.exception import ExternalServiceError, ValidationError
from src.common.json import JsonObject

_DEFAULT_URL = "https://qianfan.baidubce.com/v2/ai_search/web_search"
_TIMEOUT_SECONDS = 15.0
_SOURCE = "baidu"
_DEFAULT_RESULTS = 5
_MAX_RESULTS = 50
_MAX_QUERY_UNITS = 72
_MAX_RESPONSE_BYTES = 2 << 20  # 2MB

_BAIDU_DATE_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?")


def parse_baidu_date(date_str: str) -> datetime | None:
    """Parse the variable date formats Baidu returns (e.g. ``2025-4-24``)."""
    match = _BAIDU_DATE_RE.match(date_str.strip())
    if match is None:
        return None
    year, month, day, hour, minute, second = (match.group(i) for i in range(1, 7))
    normalized = (
        f"{int(year):04d}-{int(month):02d}-{int(day):02d} "
        f"{int(hour or 0):02d}:{int(minute or 0):02d}:{int(second or 0):02d}"
    )
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _query_units(query: str) -> int:
    return sum(1 if ord(ch) <= 0x7F else 2 for ch in query)


def normalize_baidu_query(query: str) -> str:
    """Trim and, when needed, truncate the query to the API width budget."""
    query = query.strip()
    if not query:
        return ""
    if _query_units(query) <= _MAX_QUERY_UNITS:
        return query
    kept: list[str] = []
    used = 0
    for ch in query:
        width = 1 if ord(ch) <= 0x7F else 2
        if used + width > _MAX_QUERY_UNITS:
            break
        kept.append(ch)
        used += width
    return "".join(kept)


class BaiduProvider(HttpSearchClient):
    """Web search against the Baidu AI Search API."""

    provider_type = _SOURCE

    def __init__(
        self,
        *,
        client: httpx.Client,
        api_key: str,
        base_url: str = _DEFAULT_URL,
    ) -> None:
        super().__init__(client=client)
        self._api_key = api_key
        self._base_url = base_url

    def search(
        self,
        query: str,
        max_results: int,
        include_date: bool,
    ) -> list[SearchHit]:
        prepared = normalize_baidu_query(query)
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
                "messages": [{"role": "user", "content": prepared}],
                "search_source": "baidu_search_v2",
                "resource_type_filter": [{"type": "web", "top_k": count}],
            },
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        if len(response.content) > _MAX_RESPONSE_BYTES:
            raise ExternalServiceError(
                code="web_search_provider.invalid_response",
                message=f"baidu response exceeds {_MAX_RESPONSE_BYTES} bytes",
            )
        if response.status_code != 200:
            raise ExternalServiceError(
                code="web_search_provider.search_failed",
                message=(
                    f"baidu API returned status {response.status_code}: "
                    f"{response.text.strip()[:512]}"
                ),
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ExternalServiceError(
                code="web_search_provider.invalid_response",
                message=f"baidu returned an unparseable response: {exc}",
            ) from exc
        if not isinstance(data, dict):
            raise ExternalServiceError(
                code="web_search_provider.invalid_response",
                message="baidu returned a non-object response",
            )
        code = data.get("code")
        if isinstance(code, int) and code != 0:
            raise ExternalServiceError(
                code="web_search_provider.search_failed",
                message=(f"baidu API error (code {code}): {coerce_str(data.get('message'))}"),
            )
        references = data.get("references")
        if not isinstance(references, list):
            return []
        hits: list[SearchHit] = []
        for item in references:
            if not isinstance(item, dict):
                continue
            title = coerce_str(item.get("title"))
            url = coerce_str(item.get("url"))
            if not title and not url:
                continue
            hit: SearchHit = {
                "title": title,
                "url": url,
                "content": coerce_str(item.get("content")),
                "source": _SOURCE,
            }
            if include_date:
                published = parse_baidu_date(coerce_str(item.get("date")))
                if published is not None:
                    hit["published_at"] = published.isoformat()
            hits.append(hit)
        return hits


def build_baidu_client(params: JsonObject) -> BaiduProvider:
    """Build a :class:`BaiduProvider` from a JSON parameter blob."""
    api_key = coerce_str(params.get("api_key"))
    if not api_key:
        raise ValidationError(
            code="web_search_provider.api_key_required",
            message="API key is required for Baidu provider",
        )
    # Matches the upstream client: Baidu uses a plain HTTP client and
    # does not tunnel through an explicit proxy.
    client = build_http_client(timeout=_TIMEOUT_SECONDS)
    return BaiduProvider(client=client, api_key=api_key)


__all__ = [
    "BaiduProvider",
    "build_baidu_client",
    "normalize_baidu_query",
    "parse_baidu_date",
]
