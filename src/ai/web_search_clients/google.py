"""Google Custom Search Engine web-search client.

Maps the official Google Custom Search JSON API. The endpoint is fixed
(not tenant-configurable); the tenant supplies the API key and the
Custom Search engine id (``cx``).
"""

from __future__ import annotations

import httpx

from src.ai.web_search_clients._base import (
    HttpSearchClient,
    SearchHit,
    coerce_str,
    require_non_empty_query,
)
from src.ai.web_search_clients.proxy import build_http_client
from src.common.exception import ExternalServiceError, ValidationError
from src.common.json import JsonObject

_DEFAULT_URL = "https://www.googleapis.com/customsearch/v1"
_TIMEOUT_SECONDS = 30.0
_SOURCE = "google"
_DEFAULT_RESULTS = 5


class GoogleProvider(HttpSearchClient):
    """Web search against the Google Custom Search JSON API."""

    provider_type = _SOURCE

    def __init__(
        self,
        *,
        client: httpx.Client,
        api_key: str,
        cx: str,
        base_url: str = _DEFAULT_URL,
    ) -> None:
        super().__init__(client=client)
        self._api_key = api_key
        self._cx = cx
        self._base_url = base_url

    def search(
        self,
        query: str,
        max_results: int,
        include_date: bool,
    ) -> list[SearchHit]:
        require_non_empty_query(query)
        count = max_results if max_results > 0 else _DEFAULT_RESULTS
        response = self._client.get(
            self._base_url,
            params={
                "key": self._api_key,
                "cx": self._cx,
                "q": query,
                "num": str(count),
                "hl": "ch-zh",
            },
        )
        if response.status_code != 200:
            raise ExternalServiceError(
                code="web_search_provider.search_failed",
                message=(
                    f"google API returned status {response.status_code}: "
                    f"{response.text.strip()[:512]}"
                ),
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ExternalServiceError(
                code="web_search_provider.invalid_response",
                message=f"google returned an unparseable response: {exc}",
            ) from exc
        if not isinstance(data, dict):
            raise ExternalServiceError(
                code="web_search_provider.invalid_response",
                message="google returned a non-object response",
            )
        items = data.get("items")
        if not isinstance(items, list):
            return []
        hits: list[SearchHit] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            title = coerce_str(item.get("title"))
            url = coerce_str(item.get("link"))
            if not title and not url:
                continue
            hits.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": coerce_str(item.get("snippet")),
                    "source": _SOURCE,
                }
            )
        return hits


def build_google_client(params: JsonObject) -> GoogleProvider:
    """Build a :class:`GoogleProvider` from a JSON parameter blob."""
    api_key = coerce_str(params.get("api_key"))
    cx = coerce_str(
        params.get("cx") or params.get("engine_id") or params.get("engineId")
    )
    if not api_key:
        raise ValidationError(
            code="web_search_provider.api_key_required",
            message="API key is required for Google provider",
        )
    if not cx:
        raise ValidationError(
            code="web_search_provider.cx_required",
            message="cx (Google Custom Search engine ID) is required for Google provider",
        )
    client = build_http_client(
        timeout=_TIMEOUT_SECONDS,
        proxy_url=coerce_str(params.get("proxy_url")),
    )
    return GoogleProvider(client=client, api_key=api_key, cx=cx)


__all__ = ["GoogleProvider", "build_google_client"]
