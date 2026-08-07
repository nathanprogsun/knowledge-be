"""Bing Search API web-search client.

Maps the Bing Search API v7.0 endpoint. The API URL is hardcoded (not
tenant-configurable) so tenants cannot point the client at arbitrary
hosts.
"""

from __future__ import annotations

import httpx

from src.ai.web_search_clients._base import (
    DEFAULT_USER_AGENT,
    HttpSearchClient,
    SearchHit,
    coerce_str,
    parse_iso_datetime,
    require_non_empty_query,
)
from src.ai.web_search_clients.proxy import build_http_client
from src.common.exception import ExternalServiceError, ValidationError
from src.common.json import JsonObject

_DEFAULT_URL = "https://api.bing.microsoft.com/v7.0/search"
_TIMEOUT_SECONDS = 10.0
_SOURCE = "bing"


class BingProvider(HttpSearchClient):
    """Web search against the Bing Search API (v7.0)."""

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
        response = self._client.get(
            self._base_url,
            params={"q": query, "count": str(max(1, max_results))},
            headers={
                "User-Agent": DEFAULT_USER_AGENT,
                "Ocp-Apim-Subscription-Key": self._api_key,
            },
        )
        if response.status_code != 200:
            raise ExternalServiceError(
                code="web_search_provider.search_failed",
                message=(
                    f"bing API returned status {response.status_code}: "
                    f"{response.text.strip()[:512]}"
                ),
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ExternalServiceError(
                code="web_search_provider.invalid_response",
                message=f"bing returned an unparseable response: {exc}",
            ) from exc
        if not isinstance(data, dict):
            raise ExternalServiceError(
                code="web_search_provider.invalid_response",
                message="bing returned a non-object response",
            )
        pages = data.get("webPages")
        value = pages.get("value") if isinstance(pages, dict) else None
        if not isinstance(value, list):
            return []
        hits: list[SearchHit] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            title = coerce_str(item.get("name"))
            url = coerce_str(item.get("url"))
            if not title and not url:
                continue
            hit: SearchHit = {
                "title": title,
                "url": url,
                "snippet": coerce_str(item.get("snippet")),
                "source": _SOURCE,
            }
            published = parse_iso_datetime(coerce_str(item.get("dateLastCrawled")))
            if published is not None:
                hit["published_at"] = published.isoformat()
            hits.append(hit)
        return hits


def build_bing_client(params: JsonObject) -> BingProvider:
    """Build a :class:`BingProvider` from a JSON parameter blob."""
    api_key = coerce_str(params.get("api_key"))
    if not api_key:
        raise ValidationError(
            code="web_search_provider.api_key_required",
            message="API key is required for Bing provider",
        )
    client = build_http_client(
        timeout=_TIMEOUT_SECONDS,
        proxy_url=coerce_str(params.get("proxy_url")),
    )
    return BingProvider(client=client, api_key=api_key)


__all__ = ["BingProvider", "build_bing_client"]
