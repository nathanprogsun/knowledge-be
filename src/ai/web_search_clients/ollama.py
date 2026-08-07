"""Ollama Cloud web-search client.

Maps the Ollama web search endpoint. The API URL is hardcoded (not
tenant-configurable). Per the upstream client, the explicit proxy
parameter is intentionally not applied here — only the environment
proxy configuration is honored.
"""

from __future__ import annotations

import httpx

from src.ai.web_search_clients._base import (
    DEFAULT_USER_AGENT,
    HttpSearchClient,
    SearchHit,
    coerce_str,
    require_non_empty_query,
)
from src.ai.web_search_clients.proxy import build_http_client
from src.common.exception import ExternalServiceError, ValidationError
from src.common.json import JsonObject

_DEFAULT_URL = "https://ollama.com/api/web_search"
_TIMEOUT_SECONDS = 10.0
_SOURCE = "ollama"
_DEFAULT_RESULTS = 5
_MAX_RESULTS = 10


class OllamaProvider(HttpSearchClient):
    """Web search against the Ollama Cloud API."""

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
        count = max_results if max_results > 0 else _DEFAULT_RESULTS
        count = min(count, _MAX_RESULTS)
        response = self._client.post(
            self._base_url,
            json={"query": query, "max_results": count},
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": DEFAULT_USER_AGENT,
            },
        )
        if response.status_code != 200:
            raise ExternalServiceError(
                code="web_search_provider.search_failed",
                message=(
                    f"ollama API returned status {response.status_code}: "
                    f"{response.text.strip()[:512]}"
                ),
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ExternalServiceError(
                code="web_search_provider.invalid_response",
                message=f"ollama returned an unparseable response: {exc}",
            ) from exc
        if not isinstance(data, dict):
            raise ExternalServiceError(
                code="web_search_provider.invalid_response",
                message="ollama returned a non-object response",
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
            hits.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": coerce_str(item.get("snippet")),
                    "content": coerce_str(item.get("content")),
                    "source": _SOURCE,
                }
            )
        return hits


def build_ollama_client(params: JsonObject) -> OllamaProvider:
    """Build a :class:`OllamaProvider` from a JSON parameter blob."""
    api_key = coerce_str(params.get("api_key"))
    if not api_key:
        raise ValidationError(
            code="web_search_provider.api_key_required",
            message="API key is required for Ollama provider",
        )
    # Matches the upstream client: Ollama uses a plain HTTP client and
    # does not tunnel through an explicit proxy.
    client = build_http_client(timeout=_TIMEOUT_SECONDS)
    return OllamaProvider(client=client, api_key=api_key)


__all__ = ["OllamaProvider", "build_ollama_client"]
