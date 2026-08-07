"""Web-search provider registry assembly.

Builds the app-scope ``WebSearchClientRegistry`` that maps every
supported provider type to its concrete HTTP client factory. The
concrete registry type and the provider factories live in
``src.ai.web_search_clients``; ``core`` is the sanctioned assembly point
(matching the data-source connector pattern) so the lifespan wires a
single object.
"""

from __future__ import annotations

from src.ai.web_search_clients import (
    WebSearchClientFactory,
    WebSearchClientRegistry,
    build_baidu_client,
    build_bing_client,
    build_duckduckgo_client,
    build_google_client,
    build_keenable_client,
    build_ollama_client,
    build_searxng_client,
    build_tavily_client,
    build_zhipu_client,
)

# Provider-type id -> factory. Covers every entry of the supported-type
# catalog in ``src.core.infra.web_search.types``.
_FACTORIES: tuple[tuple[str, WebSearchClientFactory], ...] = (
    ("bing", build_bing_client),
    ("google", build_google_client),
    ("tavily", build_tavily_client),
    ("ollama", build_ollama_client),
    ("searxng", build_searxng_client),
    ("baidu", build_baidu_client),
    ("keenable", build_keenable_client),
    ("duckduckgo", build_duckduckgo_client),
    ("zhipu", build_zhipu_client),
)


def build_web_search_client_registry() -> WebSearchClientRegistry:
    """Return a registry with every supported provider factory bound."""
    registry = WebSearchClientRegistry()
    for provider_type, factory in _FACTORIES:
        registry.register(provider_type, factory)
    return registry


__all__ = ["WebSearchClientRegistry", "build_web_search_client_registry"]
