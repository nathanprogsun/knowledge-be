"""Internal DTOs and registry metadata for the web-search domain.

Three surfaces live here:

- ``WebSearchProviderInfo`` — service-side projection of a
  ``web_search_providers`` row, mirroring the Go ``WebSearchProviderEntity``.
  Boundary translation (e.g. masking the ``api_key``) lives at the web
  layer; this DTO is the internal carrier.
- ``PROVIDER_TYPES`` — registry metadata for every supported provider
  type, mirroring ``internal/types/web_search_provider.go::GetWebSearchProviderTypes``.
  The web layer returns this verbatim for ``GET /web-search-providers/types``.
- ``SUPPORTED_PROVIDER_TYPES`` — frozenset of valid provider ids,
  used by the service to validate the ``provider`` field on create/update.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject
from src.core.contracts.infra import (
    WebSearchBuiltinProvider,
    WebSearchProviderParameters,
    WebSearchProviderTypeInfo,
)
from src.db.models.infra.web_search_provider import WebSearchProvider
from src.util.crypto import decrypt_stored_secret_lenient

# ── Provider-type metadata (registry) ───────────────────────────────

PROVIDER_TYPES: tuple[WebSearchProviderTypeInfo, ...] = (
    WebSearchProviderTypeInfo(
        provider="duckduckgo",
        label="DuckDuckGo",
        description="DuckDuckGo Search (free, no API key required)",
        parameter_schema=None,
    ),
    WebSearchProviderTypeInfo(
        provider="bing",
        label="Bing",
        description="Bing Search API (requires API key from Azure)",
        parameter_schema=None,
    ),
    WebSearchProviderTypeInfo(
        provider="google",
        label="Google",
        description="Google Custom Search API (requires API key and engine ID)",
        parameter_schema=None,
    ),
    WebSearchProviderTypeInfo(
        provider="tavily",
        label="Tavily",
        description="Tavily Search API (requires API key)",
        parameter_schema=None,
    ),
    WebSearchProviderTypeInfo(
        provider="ollama",
        label="Ollama Web Search",
        description="Ollama Cloud web search (requires Ollama API key)",
        parameter_schema=None,
    ),
    WebSearchProviderTypeInfo(
        provider="searxng",
        label="SearXNG",
        description=(
            "Self-hosted SearXNG metasearch instance (provide instance URL; "
            "private hosts must be SSRF-whitelisted)"
        ),
        parameter_schema=None,
    ),
    WebSearchProviderTypeInfo(
        provider="baidu",
        label="Baidu",
        description="Baidu AI Search (requires API key from Baidu Cloud)",
        parameter_schema=None,
    ),
    WebSearchProviderTypeInfo(
        provider="keenable",
        label="Keenable",
        description=(
            "Keenable web search built for AI agents (keyless by default; "
            "an optional API key lifts the rate limit)"
        ),
        parameter_schema=None,
    ),
    WebSearchProviderTypeInfo(
        provider="zhipu",
        label="Zhipu AI",
        description="Zhipu AI Web Search API (requires API key)",
        parameter_schema=None,
    ),
)

SUPPORTED_PROVIDER_TYPES: frozenset[str] = frozenset(info.provider for info in PROVIDER_TYPES)

# PR-30.6c H2: storage columns that must not cross into the
# service-output projection per AGENTS.md §9. ``deleted_at`` is the
# soft-delete tombstone; the service layer treats a missing row as the
# only delete signal.
WEB_SEARCH_PROVIDER_EXCLUDE_COLUMNS: frozenset[str] = frozenset({"deleted_at"})


# ── Builtin providers (system-level capability list) ─────────────────

BUILTIN_PROVIDERS: tuple[WebSearchBuiltinProvider, ...] = (
    WebSearchBuiltinProvider(
        name="duckduckgo",
        label="DuckDuckGo",
        description="DuckDuckGo Search (free, no API key required)",
        enabled=True,
    ),
    WebSearchBuiltinProvider(
        name="bing",
        label="Bing",
        description="Bing Search API",
        enabled=True,
    ),
    WebSearchBuiltinProvider(
        name="google",
        label="Google",
        description="Google Custom Search API",
        enabled=True,
    ),
    WebSearchBuiltinProvider(
        name="tavily",
        label="Tavily",
        description="Tavily Search API",
        enabled=True,
    ),
    WebSearchBuiltinProvider(
        name="ollama",
        label="Ollama Web Search",
        description="Ollama Cloud web search",
        enabled=True,
    ),
    WebSearchBuiltinProvider(
        name="searxng",
        label="SearXNG",
        description="Self-hosted SearXNG metasearch",
        enabled=True,
    ),
    WebSearchBuiltinProvider(
        name="baidu",
        label="Baidu",
        description="Baidu AI Search",
        enabled=True,
    ),
    WebSearchBuiltinProvider(
        name="keenable",
        label="Keenable",
        description="Keenable web search for AI agents",
        enabled=True,
    ),
    WebSearchBuiltinProvider(
        name="zhipu",
        label="Zhipu AI",
        description="Zhipu AI Web Search API",
        enabled=True,
    ),
)


# ── Wire-side projection ─────────────────────────────────────────────


class WebSearchProviderInfo(BaseModel):
    """Service-side projection of a `web_search_providers` row.

    Mirrors ``internal/types/web_search_provider.go::WebSearchProviderEntity``.
    The wire contract (``WebSearchProvider``) is a subset of these
    fields; secret-bearing fields (api_key) are masked at the web layer.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    name: str
    provider: str
    description: str | None = Field(default=None)
    parameters: WebSearchProviderParameters | None = Field(default=None)
    is_default: bool = False
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = Field(default=None)

    @classmethod
    def map_from_db(cls, db: WebSearchProvider) -> Self:
        """Build a projection from the raw storage row.

        PR-30.6c H2 / H3:

        - ``WEB_SEARCH_PROVIDER_EXCLUDE_COLUMNS`` (frozen per §9)
          drops the soft-delete tombstone (``deleted_at``) before the
          Pydantic model is built.
        - ``_parameters_from_raw`` now accepts a JSON string (not just
          a dict) so a SQLite-stored JSON column round-trips correctly
          without the caller having to decode it.
        """
        record = db.model_dump()
        params = record.get("parameters")
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = None
        record["parameters"] = _parameters_from_raw(params)
        record = {
            key: value
            for key, value in record.items()
            if key not in WEB_SEARCH_PROVIDER_EXCLUDE_COLUMNS
        }
        return cls.model_validate(record)


def _parameters_from_raw(
    raw: JsonObject | None | str,
) -> WebSearchProviderParameters | None:
    """Coerce the stored JSONB blob to the typed parameters DTO.

    Returns ``None`` when the row carries no parameters blob (matches the
    pre-existing ``WebSearchProviderInfo.map_from_db`` contract). Accepts
    either a parsed dict or a raw JSON string (SQLite path) so the caller
    does not have to ``json.loads`` first.

    When ``api_key`` is stored as an ``enc:v1:`` blob it is decrypted
    here — legacy plaintext passes through; a decrypt failure blanks
    the field (Go ``WebSearchProviderParameters.Scan`` semantics). The
    decryption helper is the single source of truth for clearing the
    field on success or failure; ``provider_service`` delegates here
    too so a drift between the storage-read and request-validate paths
    can no longer silently leak the ciphertext back to the caller.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, dict):
        return None
    extra = raw.get("extra_config")
    extra_dict = extra if isinstance(extra, dict) else None
    typed_extra: dict[str, str] | None = None
    if extra_dict is not None:
        typed_extra = {
            str(k): str(v) for k, v in extra_dict.items() if isinstance(v, (str, int, float, bool))
        }
    api_key = raw.get("api_key")
    if isinstance(api_key, str) and api_key:
        # Decrypt an ``enc:v1:`` blob; legacy plaintext passes through.
        # A decrypt failure blanks the field (Go ``Scan`` semantics).
        plain, ok = decrypt_stored_secret_lenient(api_key)
        api_key = plain if ok else ""
    # ``cx`` is the Go-spec field name; ``engine_id`` / ``engineId`` are
    # accepted as legacy aliases.
    cx_raw = raw.get("cx") or raw.get("engine_id") or raw.get("engineId")
    base_url = raw.get("base_url")
    proxy_url = raw.get("proxy_url")
    return WebSearchProviderParameters(
        api_key=str(api_key) if isinstance(api_key, (str, int, float, bool)) else None,
        cx=str(cx_raw) if isinstance(cx_raw, (str, int, float, bool)) else None,
        base_url=str(base_url) if isinstance(base_url, (str, int, float, bool)) else None,
        proxy_url=str(proxy_url) if isinstance(proxy_url, (str, int, float, bool)) else None,
        extra_config=typed_extra,
    )


def _parameters_from_json(raw: JsonObject | None) -> WebSearchProviderParameters:
    """Validate-path counterpart to ``_parameters_from_raw``.

    Same coercion contract but always returns a populated
    ``WebSearchProviderParameters`` instance (an unset blob becomes the
    zero-value DTO), matching the request-validation call site in
    ``provider_service``. Delegates to ``_parameters_from_raw`` so the
    decryption / alias-resolution rules live in one place.
    """
    return _parameters_from_raw(raw) or WebSearchProviderParameters()


__all__ = [
    "BUILTIN_PROVIDERS",
    "PROVIDER_TYPES",
    "SUPPORTED_PROVIDER_TYPES",
    "WEB_SEARCH_PROVIDER_EXCLUDE_COLUMNS",
    "WebSearchProviderInfo",
]
