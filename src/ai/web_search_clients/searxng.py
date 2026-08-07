"""SearXNG metasearch client (self-hosted instance).

Unlike the commercial providers, the instance URL is supplied by the
tenant via the ``base_url`` parameter and is validated for SSRF safety
(only http(s), no query/fragment, non-restricted host). The instance
must have JSON format enabled in its settings.
"""

from __future__ import annotations

import urllib.parse
from datetime import datetime
from email.utils import parsedate_to_datetime

import httpx

from src.ai.web_search_clients._base import (
    HttpSearchClient,
    SearchHit,
    coerce_str,
    require_non_empty_query,
)
from src.ai.web_search_clients.proxy import build_http_client, validate_url_for_ssrf
from src.common.exception import ExternalServiceError, ValidationError
from src.common.json import JsonObject

_TIMEOUT_SECONDS = 12.0
_SOURCE = "searxng"
_DEFAULT_RESULTS = 5

_USER_AGENT = "knowledge-be/1.0"


def validate_searxng_base_url(raw_url: str) -> str:
    """Validate a SearXNG instance URL; return the trimmed base URL.

    Mirrors the upstream validator: a non-empty, absolute http(s) URL
    without query or fragment that passes the SSRF guard.
    """
    base = raw_url.strip()
    if not base:
        raise ValidationError(
            code="web_search_provider.base_url_required",
            message="base_url is required for SearXNG provider",
        )
    parsed = urllib.parse.urlparse(base)
    if not parsed.scheme or not parsed.netloc:
        raise ValidationError(
            code="web_search_provider.invalid_base_url",
            message="invalid SearXNG base_url: must be an absolute http(s) URL",
        )
    if parsed.scheme not in ("http", "https"):
        raise ValidationError(
            code="web_search_provider.invalid_base_url",
            message=f"invalid SearXNG base_url scheme: {parsed.scheme}",
        )
    if parsed.query or parsed.fragment:
        raise ValidationError(
            code="web_search_provider.invalid_base_url",
            message="invalid SearXNG base_url: must not contain query or fragment",
        )
    try:
        validate_url_for_ssrf(base)
    except ValidationError as exc:
        raise ValidationError(
            code="web_search_provider.invalid_base_url",
            message=f"invalid SearXNG base_url: {exc.message}",
        ) from exc
    return base.rstrip("/")


def parse_searxng_date(value: str) -> datetime | None:
    """Parse the date formats SearXNG engines emit, or ``None``."""
    text = value.strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None


class SearxngProvider(HttpSearchClient):
    """Metasearch against a self-hosted SearXNG instance."""

    provider_type = _SOURCE

    def __init__(
        self,
        *,
        client: httpx.Client,
        base_url: str,
    ) -> None:
        super().__init__(client=client)
        self._base_url = base_url.rstrip("/")
        self._last_unresponsive: list[list[str]] = []

    def empty_result_diagnostics(self) -> str:
        """Explain why the most recent search returned no usable results."""
        if self._last_unresponsive:
            detail = "; ".join(" / ".join(pair) for pair in self._last_unresponsive[:5])
            return (
                f"{detail}; check that upstream search engines can reach the internet"
            )
        return "verify the instance URL is reachable and JSON format is enabled in settings.yml"

    def search(
        self,
        query: str,
        max_results: int,
        include_date: bool,
    ) -> list[SearchHit]:
        require_non_empty_query(query)
        count = max_results if max_results > 0 else _DEFAULT_RESULTS
        response = self._client.get(
            f"{self._base_url}/search",
            params={"q": query, "format": "json", "language": "all"},
            headers={"Accept": "application/json", "User-Agent": _USER_AGENT},
        )
        if response.status_code != 200:
            raise ExternalServiceError(
                code="web_search_provider.search_failed",
                message=(
                    f"searxng returned status {response.status_code}: "
                    f"{response.text.strip()[:512]}"
                ),
            )
        try:
            data = response.json()
        except ValueError as exc:
            self._last_unresponsive = []
            raise ExternalServiceError(
                code="web_search_provider.invalid_response",
                message=(
                    "failed to decode SearXNG response "
                    "(ensure JSON format is enabled in settings.yml): "
                    f"{exc}"
                ),
            ) from exc
        if not isinstance(data, dict):
            self._last_unresponsive = []
            raise ExternalServiceError(
                code="web_search_provider.invalid_response",
                message="searxng returned a non-object response",
            )
        raw_unresponsive = data.get("unresponsive_engines")
        if isinstance(raw_unresponsive, list):
            self._last_unresponsive = [
                [str(part) for part in pair]
                for pair in raw_unresponsive
                if isinstance(pair, list)
            ]
        else:
            self._last_unresponsive = []
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
            if not url or not title:
                continue
            hit: SearchHit = {
                "title": title,
                "url": url,
                "snippet": coerce_str(item.get("content")),
                "source": _SOURCE,
            }
            if include_date:
                published = parse_searxng_date(coerce_str(item.get("publishedDate")))
                if published is not None:
                    hit["published_at"] = published.isoformat()
            hits.append(hit)
        return hits


def build_searxng_client(params: JsonObject) -> SearxngProvider:
    """Build a :class:`SearxngProvider` from a JSON parameter blob."""
    base_url = validate_searxng_base_url(coerce_str(params.get("base_url")))
    client = build_http_client(
        timeout=_TIMEOUT_SECONDS,
        proxy_url=coerce_str(params.get("proxy_url")),
    )
    return SearxngProvider(client=client, base_url=base_url)


__all__ = [
    "SearxngProvider",
    "build_searxng_client",
    "parse_searxng_date",
    "validate_searxng_base_url",
]
