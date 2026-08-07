"""Tavily Search API web-search client.

Maps the Tavily ``/search`` endpoint. The API URL is hardcoded (not
tenant-configurable).
"""

from __future__ import annotations

import httpx

from src.ai.web_search_clients._base import (
    HttpSearchClient,
    SearchHit,
    coerce_str,
    parse_iso_datetime,
    require_non_empty_query,
)
from src.ai.web_search_clients.proxy import build_http_client
from src.common.exception import ExternalServiceError, ValidationError
from src.common.json import JsonObject

_DEFAULT_URL = "https://api.tavily.com/search"
_TIMEOUT_SECONDS = 15.0
_SOURCE = "tavily"


class TavilyProvider(HttpSearchClient):
    """Web search against the Tavily Search API."""

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
        require_non_empty_query(query)
        response = self._client.post(
            self._base_url,
            json={
                "api_key": self._api_key,
                "query": query,
                "max_results": max(1, max_results),
            },
            headers={"Content-Type": "application/json"},
        )
        if response.status_code != 200:
            raise ExternalServiceError(
                code="web_search_provider.search_failed",
                message=(
                    f"tavily API returned status {response.status_code}: "
                    f"{response.text.strip()[:512]}"
                ),
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ExternalServiceError(
                code="web_search_provider.invalid_response",
                message=f"tavily returned an unparseable response: {exc}",
            ) from exc
        if not isinstance(data, dict):
            raise ExternalServiceError(
                code="web_search_provider.invalid_response",
                message="tavily returned a non-object response",
            )
        results = data.get("results")
        if not isinstance(results, list):
            return []
        hits: list[SearchHit] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            title = coerce_str(item.get("title"))
            url = coerce_str(item.get("url"))
            if not title and not url:
                continue
            content = coerce_str(item.get("content"))
            hit: SearchHit = {
                "title": title,
                "url": url,
                "snippet": content,
                "source": _SOURCE,
            }
            if include_date:
                published = parse_iso_datetime(coerce_str(item.get("published_date")))
                if published is not None:
                    hit["published_at"] = published.isoformat()
            hits.append(hit)
        return hits


def build_tavily_client(params: JsonObject) -> TavilyProvider:
    """Build a :class:`TavilyProvider` from a JSON parameter blob."""
    api_key = coerce_str(params.get("api_key"))
    if not api_key:
        raise ValidationError(
            code="web_search_provider.api_key_required",
            message="API key is required for Tavily provider",
        )
    client = build_http_client(
        timeout=_TIMEOUT_SECONDS,
        proxy_url=coerce_str(params.get("proxy_url")),
    )
    return TavilyProvider(client=client, api_key=api_key)


__all__ = ["TavilyProvider", "build_tavily_client"]
