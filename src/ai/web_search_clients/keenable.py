"""Keenable Search API web-search client.

Maps the Keenable Search API. The base URL is hardcoded (not
tenant-configurable). Keyless by default: without an API key the
client calls the public (rate-limited) endpoint; a configured key
switches to the authenticated endpoint.
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
from src.common.exception import ExternalServiceError
from src.common.json import JsonObject

_DEFAULT_BASE_URL = "https://api.keenable.ai"
_TIMEOUT_SECONDS = 15.0
_SOURCE = "keenable"
_DEFAULT_RESULTS = 5

# Attribution tag Keenable segments integration traffic by.
_TITLE_TAG = "knowledge-be"


class KeenableProvider(HttpSearchClient):
    """Web search against the Keenable Search API."""

    provider_type = _SOURCE

    def __init__(
        self,
        *,
        client: httpx.Client,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
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
        count = max_results if max_results > 0 else _DEFAULT_RESULTS
        path = "/v1/search/public" if not self._api_key else "/v1/search"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Keenable-Title": _TITLE_TAG,
        }
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        response = self._client.post(
            f"{self._base_url}{path}",
            json={"query": query, "mode": "pro"},
            headers=headers,
        )
        if response.status_code != 200:
            raise ExternalServiceError(
                code="web_search_provider.search_failed",
                message=(
                    f"keenable API returned status {response.status_code}: "
                    f"{response.text.strip()[:512]}"
                ),
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ExternalServiceError(
                code="web_search_provider.invalid_response",
                message=f"keenable returned an unparseable response: {exc}",
            ) from exc
        if not isinstance(data, dict):
            raise ExternalServiceError(
                code="web_search_provider.invalid_response",
                message="keenable returned a non-object response",
            )
        results = data.get("results")
        if not isinstance(results, list):
            return []
        hits: list[SearchHit] = []
        for item in results:
            if len(hits) >= count:
                break
            if not isinstance(item, dict):
                continue
            title = coerce_str(item.get("title"))
            url = coerce_str(item.get("url"))
            if not title and not url:
                continue
            description = coerce_str(item.get("description"))
            snippet = coerce_str(item.get("snippet"))
            hit: SearchHit = {
                "title": title,
                "url": url,
                "snippet": description or snippet,
                "content": snippet,
                "source": _SOURCE,
            }
            if include_date:
                published = parse_iso_datetime(coerce_str(item.get("published_at")))
                if published is not None:
                    hit["published_at"] = published.isoformat()
            hits.append(hit)
        return hits


def build_keenable_client(params: JsonObject) -> KeenableProvider:
    """Build a :class:`KeenableProvider` from a JSON parameter blob.

    The API key is optional; Keenable works keyless against the public
    endpoint.
    """
    client = build_http_client(
        timeout=_TIMEOUT_SECONDS,
        proxy_url=coerce_str(params.get("proxy_url")),
    )
    return KeenableProvider(
        client=client,
        api_key=coerce_str(params.get("api_key")),
    )


__all__ = ["KeenableProvider", "build_keenable_client"]
