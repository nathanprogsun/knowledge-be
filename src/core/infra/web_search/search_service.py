"""Web-search dispatch service.

Dispatches the search (narrow surface — CompressWithRAG is deferred).

Resolves a provider by id (or, for backward compat, by deprecated
``config.provider`` field), constructs a client via the registry, runs
the search with a per-call timeout, and applies the configured
blacklist before returning.

The dispatcher depends on:

- a ``WebSearchClientRegistry`` (the abstract Protocol) — concrete
  implementations live in ``src/ai.web_search_clients`` (not imported
  here, by the ``core`` ↔ ``ai`` boundary rule).
- the provider repository (for the by-id resolution path).

Tenant isolation: ``Search(tenant_id, ...)`` carries the tenant id
explicitly so the service can be called from any context (web, worker,
test) without depending on the request contextvar.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.common.exception import ExternalServiceError, ValidationError
from src.common.json import JsonObject, JsonValue
from src.core.infra.web_search.provider_service import (
    WebSearchClient,
    WebSearchClientRegistry,
)
from src.core.infra.web_search.types import WebSearchProviderInfo
from src.db.dao.web_search_provider_repository import WebSearchProviderRepository

# Default timeout (seconds) for an outbound web search call.
DEFAULT_TIMEOUT_SECONDS: int = 10


class SearchResult(BaseModel):
    """One search hit; mirrors ``internal/types/web_search.go::WebSearchResult``.

    ``published_at`` stays optional because not every provider surfaces
    a date.
    """

    model_config = ConfigDict(frozen=True)

    title: str = ""
    url: str = ""
    snippet: str = ""
    content: str = ""
    source: str = ""
    published_at: datetime | None = Field(default=None)


class WebSearchSearchService:
    """Search dispatcher — resolves a provider, runs a search, filters results."""

    def __init__(
        self,
        *,
        provider_repo: WebSearchProviderRepository,
        registry: WebSearchClientRegistry,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._repo = provider_repo
        self._registry = registry
        self._timeout = max(1, timeout_seconds)

    async def search(
        self,
        *,
        tenant_id: int,
        provider_id: str,
        query: str,
        max_results: int = 10,
        include_date: bool = False,
        blacklist: list[str] | None = None,
        proxy_url: str = "",
        legacy_provider: str = "",
        legacy_api_key: str = "",
    ) -> list[SearchResult]:
        """Run a web search via the resolved provider.

        Resolution order:

        1. ``provider_id`` — load the tenant's saved provider by id;
           construct a client with stored parameters.
        2. ``legacy_provider`` — backward-compat fallback that uses the
           deprecated ``WebSearchConfig.Provider`` field.

        ``proxy_url`` overrides the stored proxy on the saved
        configuration for a single call when non-empty.
        ``blacklist`` filters matches post-search (regex + glob patterns,
        mirrors the Go filter).
        """
        if not query or not query.strip():
            raise ValidationError(
                code="web_search_provider.query_required",
                message="query must be a non-empty string",
            )
        client, source = await self._resolve_client(
            tenant_id=tenant_id,
            provider_id=provider_id,
            legacy_provider=legacy_provider,
            legacy_api_key=legacy_api_key,
            proxy_url=proxy_url,
        )
        try:
            raw_results = await asyncio.wait_for(
                asyncio.to_thread(
                    client.search,
                    query,
                    max(1, max_results),
                    bool(include_date),
                ),
                timeout=self._timeout,
            )
        except (TimeoutError, ExternalServiceError) as exc:
            raise ExternalServiceError(
                code="web_search_provider.search_failed",
                message=f"web search failed: {exc}",
            ) from exc
        results = [
            SearchResult(
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                snippet=str(item.get("snippet", "")),
                content=str(item.get("content", "")),
                source=str(item.get("source", source)),
                published_at=_parse_datetime(item.get("published_at")),
            )
            for item in raw_results
        ]
        return _apply_blacklist(results, blacklist or [])

    async def _resolve_client(
        self,
        *,
        tenant_id: int,
        provider_id: str,
        legacy_provider: str,
        legacy_api_key: str,
        proxy_url: str,
    ) -> tuple[WebSearchClient, str]:
        """Build a search client for the resolved provider configuration."""
        if provider_id:
            if tenant_id <= 0:
                raise ValidationError(
                    code="web_search_provider.tenant_required",
                    message="tenant ID is required",
                )
            row = await self._repo.get_by_id(tenant_id, provider_id)
            if row is None:
                raise ValidationError(
                    code="web_search_provider.not_found",
                    message=f"web search provider {provider_id} not found",
                )
            info = WebSearchProviderInfo.map_from_db(row)
            params = _row_parameters(info)
            if proxy_url.strip():
                params["proxy_url"] = proxy_url
            client = self._registry.create_provider(info.provider, params)
            return client, info.provider

        if legacy_provider:
            legacy_params: JsonObject = {}
            if legacy_api_key:
                legacy_params["api_key"] = legacy_api_key
            if proxy_url.strip():
                legacy_params["proxy_url"] = proxy_url
            client = self._registry.create_provider(legacy_provider, legacy_params)
            return client, legacy_provider

        raise ValidationError(
            code="web_search_provider.no_provider_configured",
            message="no web search provider configured",
        )


# ── Helpers ────────────────────────────────────────────────────────


def _row_parameters(info: WebSearchProviderInfo) -> JsonObject:
    """Render an info DTO's parameters as the JSON blob the registry consumes."""
    if info.parameters is None:
        return {}
    raw: JsonObject = {}
    if info.parameters.api_key is not None:
        raw["api_key"] = info.parameters.api_key
    if info.parameters.cx is not None:
        raw["cx"] = info.parameters.cx
    if info.parameters.base_url is not None:
        raw["base_url"] = info.parameters.base_url
    if info.parameters.proxy_url is not None:
        raw["proxy_url"] = info.parameters.proxy_url
    if info.parameters.extra_config is not None:
        raw["extra_config"] = dict(info.parameters.extra_config)
    return raw


def _parse_datetime(raw: JsonValue) -> datetime | None:
    if isinstance(raw, datetime):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _apply_blacklist(
    results: list[SearchResult],
    blacklist: list[str],
) -> list[SearchResult]:
    """Drop results whose URL matches any blacklist rule (regex or glob)."""
    if not blacklist:
        return results
    compiled: list[re.Pattern[str]] = []
    for rule in blacklist:
        if rule.startswith("/") and rule.endswith("/") and len(rule) > 1:
            try:
                compiled.append(re.compile(rule[1:-1]))
            except re.error:
                continue
            continue
        glob_pattern = re.escape(rule).replace(r"\*", ".*")
        try:
            compiled.append(re.compile(f"^{glob_pattern}$"))
        except re.error:
            continue
    if not compiled:
        return results
    kept: list[SearchResult] = []
    for r in results:
        if not any(p.search(r.url) for p in compiled):
            kept.append(r)
    return kept


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "SearchResult",
    "WebSearchSearchService",
]
