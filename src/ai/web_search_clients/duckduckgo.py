"""DuckDuckGo web-search client (HTML first, Instant Answer API fallback).

Free and keyless; an optional proxy tunnels the outbound requests. The
HTML endpoint is tried first (more reliable for general results); on a
failure or an empty result set the client falls back to the Instant
Answer API.
"""

from __future__ import annotations

import urllib.parse
from html.parser import HTMLParser

import httpx

from src.ai.web_search_clients._base import (
    HttpSearchClient,
    SearchHit,
    coerce_str,
    require_non_empty_query,
)
from src.ai.web_search_clients.proxy import build_http_client
from src.common.exception import ExternalServiceError
from src.common.json import JsonObject

_HTML_URL = "https://html.duckduckgo.com/html/"
_API_URL = "https://api.duckduckgo.com/"
_TIMEOUT_SECONDS = 30.0
_SOURCE = "duckduckgo"
_DEFAULT_RESULTS = 5

_HTML_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_API_USER_AGENT = "knowledge-be/1.0"

# HTML void elements carry no closing tag; they must not shift the
# block-depth tracking used by the result-page parser.
_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)


def clean_ddg_url(url_str: str) -> str:
    """Decode the DuckDuckGo redirect wrapper to the real destination."""
    prefix = "//duckduckgo.com/l/?uddg="
    if url_str.startswith(prefix):
        trimmed = url_str[len(prefix) :]
        idx = trimmed.find("&rut=")
        if idx != -1:
            return urllib.parse.unquote(trimmed[:idx])
        # No ``&rut=`` marker — falls through to the https branch below,
        # which does not match a protocol-relative URL, so the original
        # URL is returned unchanged.
    if url_str.startswith("https://duckduckgo.com/l/?uddg="):
        parsed = urllib.parse.urlparse(url_str)
        uddg = urllib.parse.parse_qs(parsed.query).get("uddg")
        if uddg and uddg[0]:
            return uddg[0]
    return url_str


def extract_title(text: str) -> str:
    """Take the first line of ``text`` as the title, capped at 100 chars."""
    title = text.split("\n", 1)[0].strip()
    if len(title) > 100:
        return title[:100] + "..."
    return title


class _DDGHTMLParser(HTMLParser):
    """Extract ``(title, url, snippet)`` triples from the results page.

    Each result lives in a ``div.web-result`` block containing a
    ``a.result__a`` title anchor and an ``a.result__snippet`` summary.
    """

    def __init__(self) -> None:
        super().__init__()
        self._depth = 0
        self._block_depth = -1
        self._in_title = False
        self._in_snippet = False
        self._href = ""
        self._title_parts: list[str] = []
        self._snippet_parts: list[str] = []
        self.hits: list[tuple[str, str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _VOID_TAGS:
            return
        self._depth += 1
        classes = {value for key, value in attrs if key == "class" and value}
        if self._block_depth < 0:
            if tag == "div" and "web-result" in classes:
                self._block_depth = self._depth
                self._href = ""
                self._title_parts = []
                self._snippet_parts = []
            return
        if tag == "a" and "result__a" in classes:
            self._in_title = True
            for key, value in attrs:
                if key == "href" and value is not None:
                    self._href = value
        elif "result__snippet" in classes:
            self._in_snippet = True

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        return

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_TAGS:
            return
        if tag == "a":
            if self._in_title:
                self._in_title = False
            elif self._in_snippet:
                self._in_snippet = False
        if self._block_depth >= 0 and self._depth == self._block_depth:
            self._emit()
            self._block_depth = -1
        self._depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)
        elif self._in_snippet:
            self._snippet_parts.append(data)

    def _emit(self) -> None:
        title = _collapse_text("".join(self._title_parts))
        snippet = _collapse_text("".join(self._snippet_parts))
        if title and self._href:
            self.hits.append((title, clean_ddg_url(self._href), snippet))


def _collapse_text(raw: str) -> str:
    return " ".join(raw.split()).strip()


class DuckDuckGoProvider(HttpSearchClient):
    """Web search against DuckDuckGo (HTML first, API fallback)."""

    provider_type = _SOURCE

    def search(
        self,
        query: str,
        max_results: int,
        include_date: bool,
    ) -> list[SearchHit]:
        require_non_empty_query(query)
        count = max_results if max_results > 0 else _DEFAULT_RESULTS
        html_hits, html_err = self._search_html(query, count)
        if html_err is None and html_hits:
            return html_hits
        api_hits, api_err = self._search_api(query, count)
        if api_err is None and api_hits:
            return api_hits
        if html_err is not None:
            raise ExternalServiceError(
                code="web_search_provider.search_failed",
                message=f"duckduckgo HTML search failed: {html_err}",
            )
        if api_err is not None:
            raise ExternalServiceError(
                code="web_search_provider.search_failed",
                message=f"duckduckgo API search failed: {api_err}",
            )
        return []

    def _search_html(self, query: str, count: int) -> tuple[list[SearchHit], str | None]:
        try:
            response = self._client.get(
                _HTML_URL,
                params={"q": query, "kl": "cn-zh"},
                headers={"User-Agent": _HTML_USER_AGENT},
            )
        except httpx.HTTPError as exc:
            return [], str(exc)
        if response.status_code not in (200, 202):
            return [], f"duckduckgo HTML returned status {response.status_code}"
        parser = _DDGHTMLParser()
        try:
            parser.feed(response.text)
            parser.close()
        except ValueError as exc:
            return [], f"failed to parse HTML: {exc}"
        hits: list[SearchHit] = []
        for title, url, snippet in parser.hits[:count]:
            hits.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "source": _SOURCE,
                }
            )
        return hits, None

    def _search_api(self, query: str, count: int) -> tuple[list[SearchHit], str | None]:
        try:
            response = self._client.get(
                _API_URL,
                params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
                headers={"User-Agent": _API_USER_AGENT},
            )
        except httpx.HTTPError as exc:
            return [], str(exc)
        if response.status_code != 200:
            return (
                [],
                f"duckduckgo API returned status {response.status_code}: "
                f"{response.text.strip()[:512]}",
            )
        try:
            data = response.json()
        except ValueError as exc:
            return [], f"failed to decode API response: {exc}"
        if not isinstance(data, dict):
            return [], "failed to decode API response: not an object"
        hits: list[SearchHit] = []
        abstract = coerce_str(data.get("AbstractText"))
        abstract_url = coerce_str(data.get("AbstractURL"))
        if abstract and abstract_url:
            hits.append(
                {
                    "title": coerce_str(data.get("Heading")),
                    "url": abstract_url,
                    "snippet": abstract,
                    "source": _SOURCE,
                }
            )
        related = data.get("RelatedTopics")
        related_list = related if isinstance(related, list) else []
        for topic in related_list:
            if len(hits) >= count:
                break
            if not isinstance(topic, dict):
                continue
            text = coerce_str(topic.get("Text"))
            first_url = coerce_str(topic.get("FirstURL"))
            if text and first_url:
                hits.append(
                    {
                        "title": extract_title(text),
                        "url": first_url,
                        "snippet": text,
                        "source": _SOURCE,
                    }
                )
        results = data.get("Results")
        results_list = results if isinstance(results, list) else []
        for item in results_list:
            if len(hits) >= count:
                break
            if not isinstance(item, dict):
                continue
            text = coerce_str(item.get("Text"))
            first_url = coerce_str(item.get("FirstURL"))
            if text and first_url:
                hits.append(
                    {
                        "title": extract_title(text),
                        "url": first_url,
                        "snippet": text,
                        "source": _SOURCE,
                    }
                )
        return hits, None


def build_duckduckgo_client(params: JsonObject) -> DuckDuckGoProvider:
    """Build a :class:`DuckDuckGoProvider` from a JSON parameter blob.

    DuckDuckGo is free and requires no API key.
    """
    client = build_http_client(
        timeout=_TIMEOUT_SECONDS,
        proxy_url=coerce_str(params.get("proxy_url")),
    )
    return DuckDuckGoProvider(client=client)


__all__ = [
    "DuckDuckGoProvider",
    "build_duckduckgo_client",
    "clean_ddg_url",
    "extract_title",
]
