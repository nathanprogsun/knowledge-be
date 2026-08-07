"""Provider HTTP clients for the web-search domain.

Concrete ``WebSearchClient`` implementations (bing, google, tavily,
ollama, searxng, baidu, keenable, duckduckgo, zhipu) plus the registry
that resolves a provider type to a factory. This package owns the
outbound HTTP plumbing; the service layer consumes the registry through
the Protocol declared in ``src.core.infra.web_search.provider_service``
and never imports this package directly.

Public surface (unchanged by the module -> package conversion):

- ``WebSearchClient`` — base class all providers override.
- ``WebSearchClientFactory`` — ``(params: JsonObject) -> WebSearchClient``.
- ``WebSearchClientRegistry`` — provider-type -> factory map.
"""

from __future__ import annotations

from src.ai.web_search_clients._base import WebSearchClient
from src.ai.web_search_clients._registry import (
    WebSearchClientFactory,
    WebSearchClientRegistry,
)
from src.ai.web_search_clients.baidu import BaiduProvider, build_baidu_client
from src.ai.web_search_clients.bing import BingProvider, build_bing_client
from src.ai.web_search_clients.duckduckgo import (
    DuckDuckGoProvider,
    build_duckduckgo_client,
)
from src.ai.web_search_clients.google import GoogleProvider, build_google_client
from src.ai.web_search_clients.keenable import (
    KeenableProvider,
    build_keenable_client,
)
from src.ai.web_search_clients.ollama import OllamaProvider, build_ollama_client
from src.ai.web_search_clients.searxng import (
    SearxngProvider,
    build_searxng_client,
)
from src.ai.web_search_clients.tavily import TavilyProvider, build_tavily_client
from src.ai.web_search_clients.zhipu import ZhipuProvider, build_zhipu_client

__all__ = [
    "BaiduProvider",
    "BingProvider",
    "DuckDuckGoProvider",
    "GoogleProvider",
    "KeenableProvider",
    "OllamaProvider",
    "SearxngProvider",
    "TavilyProvider",
    "WebSearchClient",
    "WebSearchClientFactory",
    "WebSearchClientRegistry",
    "ZhipuProvider",
    "build_baidu_client",
    "build_bing_client",
    "build_duckduckgo_client",
    "build_google_client",
    "build_keenable_client",
    "build_ollama_client",
    "build_searxng_client",
    "build_tavily_client",
    "build_zhipu_client",
]
