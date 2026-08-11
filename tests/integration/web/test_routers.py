"""Strict route-locking gate test.

Fetches the live OpenAPI schema produced by ``create_app()`` and pins the
EXACT set of routes. Any endpoint added or removed without updating the
expected literals below fails the build - this is a route-surface
contract gate, not a behavior test.

Auth is enforced by the request-scoped auth dependency (middleware), not
by FastAPI security schemes, so the OpenAPI schema advertises no per-route
``security`` field. The public/authed split below is therefore derived from
the auth dependency's exempt-path list (the handlers that take no auth
dependency), not from the schema's security metadata.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from src.app_context.lifespan import create_app

_OPENAPI_PATH = "/openapi.json"

# Auth-exempt endpoints - the handlers that take no auth dependency.
# Kept as a sorted list so the public contract reads top-to-bottom.
_EXPECTED_PUBLIC_ROUTES: list[str] = [
    "/auth/login",
    "/auth/oidc/callback",
    "/auth/oidc/config",
    "/auth/oidc/url",
    "/auth/refresh",
    "/auth/register",
    "/embed/{channel_id}/agent-chat/{session_id}",
    "/embed/{channel_id}/chunks/{chunk_id}",
    "/embed/{channel_id}/config",
    "/embed/{channel_id}/exchange",
    "/embed/{channel_id}/files",
    "/embed/{channel_id}/knowledge-chat/{session_id}",
    "/embed/{channel_id}/messages/{session_id}/load",
    "/embed/{channel_id}/sessions",
    "/embed/{channel_id}/sessions/{session_id}/events",
    "/embed/{channel_id}/sessions/{session_id}/mcp-oauth-resolutions/{pending_id}",
    "/embed/{channel_id}/sessions/{session_id}/mcp-oauth-resolutions/{pending_id}/cancel",
    "/embed/{channel_id}/sessions/{session_id}/mcp-services/{service_id}/oauth/authorize-url",
    "/embed/{channel_id}/sessions/{session_id}/mcp-services/{service_id}/oauth/status",
    "/embed/{channel_id}/sessions/{session_id}/messages/{message_id}/suggestions",
    "/embed/{channel_id}/sessions/{session_id}/stop",
    "/embed/{channel_id}/sessions/{session_id}/suggestion-events",
    "/embed/{channel_id}/sessions/{session_id}/tool-approvals/{pending_id}",
    "/embed/{channel_id}/suggested-questions",
    "/health",
]

# The complete route surface exposed by the app. Adding or removing a
# route requires updating this set; otherwise the gate fails.
_EXPECTED_ALL_ROUTES: set[str] = {
    "/agent-chat/{session_id}",
    "/auth/change-password",
    "/auth/login",
    "/auth/logout",
    "/auth/me",
    "/auth/oidc/callback",
    "/auth/oidc/config",
    "/auth/oidc/url",
    "/auth/refresh",
    "/auth/register",
    "/auth/validate",
    "/agents",
    "/agents/placeholders",
    "/agents/type-presets",
    "/agents/{agent_id}/embed-channels",
    "/agents/{id}",
    "/agents/{id}/copy",
    "/agents/{id}/suggested-questions",
    "/datasource",
    "/datasource/logs/{log_id}",
    "/datasource/types",
    "/datasource/validate-credentials",
    "/datasource/{id}",
    "/datasource/{id}/logs",
    "/datasource/{id}/pause",
    "/datasource/{id}/resource-ancestors",
    "/datasource/{id}/resources",
    "/datasource/{id}/resume",
    "/datasource/{id}/sync",
    "/datasource/{id}/validate",
    "/embed-channels",
    "/embed-channels/{channel_id}",
    "/embed-channels/{channel_id}/preview-session",
    "/embed-channels/{channel_id}/rotate-token",
    "/embed-channels/{channel_id}/stats",
    "/embed/{channel_id}/agent-chat/{session_id}",
    "/embed/{channel_id}/chunks/{chunk_id}",
    "/embed/{channel_id}/config",
    "/embed/{channel_id}/exchange",
    "/embed/{channel_id}/files",
    "/embed/{channel_id}/knowledge-chat/{session_id}",
    "/embed/{channel_id}/messages/{session_id}/load",
    "/embed/{channel_id}/sessions",
    "/embed/{channel_id}/sessions/{session_id}/events",
    "/embed/{channel_id}/sessions/{session_id}/mcp-oauth-resolutions/{pending_id}",
    "/embed/{channel_id}/sessions/{session_id}/mcp-oauth-resolutions/{pending_id}/cancel",
    "/embed/{channel_id}/sessions/{session_id}/mcp-services/{service_id}/oauth/authorize-url",
    "/embed/{channel_id}/sessions/{session_id}/mcp-services/{service_id}/oauth/status",
    "/embed/{channel_id}/sessions/{session_id}/messages/{message_id}/suggestions",
    "/embed/{channel_id}/sessions/{session_id}/stop",
    "/embed/{channel_id}/sessions/{session_id}/suggestion-events",
    "/embed/{channel_id}/sessions/{session_id}/tool-approvals/{pending_id}",
    "/embed/{channel_id}/suggested-questions",
    "/evaluation",
    "/faq/import/progress/{task_id}",
    "/health",
    "/initialization/asr/check",
    "/initialization/embedding/test",
    "/initialization/multimodal/test",
    "/initialization/ollama/download/progress/{task_id}",
    "/initialization/ollama/download/tasks",
    "/initialization/ollama/models",
    "/initialization/ollama/models/check",
    "/initialization/ollama/models/download",
    "/initialization/ollama/status",
    "/initialization/remote/check",
    "/initialization/rerank/check",
    "/knowledge-bases",
    "/knowledge-bases/{id}",
    "/knowledge-bases/{id}/duplicate",
    "/knowledge-bases/{id}/hybrid-search",
    "/knowledge-bases/{id}/move-targets",
    "/knowledge-bases/copy",
    "/knowledge-bases/{id}/knowledge",
    "/knowledge-bases/{id}/knowledge/file",
    "/knowledge-bases/{id}/knowledge/manual",
    "/knowledge-bases/{id}/knowledge/passage",
    "/knowledge-bases/{id}/knowledge/url",
    "/knowledge-chat/{session_id}",
    "/knowledge-search",
    "/knowledge/move",
    "/knowledge/move/progress/{task_id}",
    "/knowledge/{id}",
    "/knowledge/{id}/cancel-parse",
    "/knowledge/{id}/clone",
    "/knowledge/{id}/reparse",
    "/knowledge-bases/{id}/faq/entries",
    "/knowledge-bases/{id}/faq/entries/export",
    "/knowledge-bases/{id}/faq/entries/{entry_id}",
    "/knowledge-bases/{id}/faq/entry",
    "/knowledge-bases/{id}/tags",
    "/knowledge-bases/{id}/tags/{tag_id}",
    "/chunker/preview",
    "/chunks/by-id/{id}",
    "/chunks/by-id/{id}/questions",
    "/chunks/by-id/{id}/questions/regenerate",
    "/chunks/{knowledge_id}",
    "/chunks/{knowledge_id}/{id}",
    "/chunks/{knowledge_id}/{id}/revert",
    "/chunks/{knowledge_id}/{id}/revisions",
    "/knowledgebase/{kb_id}/wiki/auto-fix",
    "/knowledgebase/{kb_id}/wiki/folders",
    "/knowledgebase/{kb_id}/wiki/folders/{folder_id}",
    "/knowledgebase/{kb_id}/wiki/graph",
    "/knowledgebase/{kb_id}/wiki/index",
    "/knowledgebase/{kb_id}/wiki/lint",
    "/knowledgebase/{kb_id}/wiki/move-page",
    "/knowledgebase/{kb_id}/wiki/pages",
    "/knowledgebase/{kb_id}/wiki/pages/{slug}",
    "/knowledgebase/{kb_id}/wiki/rebuild-links",
    "/knowledgebase/{kb_id}/wiki/search",
    "/knowledgebase/{kb_id}/wiki/stats",
    "/mcp-services",
    "/mcp-services/{service_id}",
    "/mcp-services/{service_id}/oauth/authorize-url",
    "/mcp-services/{service_id}/oauth/status",
    "/mcp-services/{service_id}/oauth/token",
    "/mcp-services/{service_id}/resources",
    "/mcp-services/{service_id}/test",
    "/mcp-services/{service_id}/tool-approvals",
    "/mcp-services/{service_id}/tool-approvals/{tool_name}",
    "/mcp-services/{service_id}/tools",
    "/messages/chat-history-stats",
    "/messages/search",
    "/messages/{session_id}/load",
    "/messages/{session_id}/{message_id}",
    "/models",
    "/models/providers",
    "/models/{model_id}",
    "/models/{model_id}/debug",
    "/organizations",
    "/organizations/join",
    "/organizations/join-by-id",
    "/organizations/join-request",
    "/organizations/preview/{code}",
    "/organizations/search",
    "/organizations/{id}",
    "/organizations/{id}/invite",
    "/organizations/{id}/invite-code",
    "/organizations/{id}/join-requests",
    "/organizations/{id}/join-requests/{request_id}/review",
    "/organizations/{id}/leave",
    "/organizations/{id}/members",
    "/organizations/{id}/members/{tenant_id}",
    "/organizations/{id}/request-upgrade",
    "/organizations/{id}/search-tenants",
    "/organizations/{id}/search-users",
    "/sessions",
    "/sessions/batch",
    "/sessions/{session_id}",
    "/sessions/{session_id}/messages",
    "/sessions/{session_id}/messages/{message_id}/suggestions",
    "/sessions/{session_id}/pin",
    "/sessions/{session_id}/suggestion-events",
    "/skills",
    "/storage-backends",
    "/storage-backends/test",
    "/storage-backends/types",
    "/storage-backends/{id}",
    "/storage-backends/{id}/default",
    "/storage-backends/{id}/test",
    "/system/admin/audit-log",
    "/system/admin/settings",
    "/system/admin/settings/{key}",
    "/system/docreader/reconnect",
    "/system/info",
    "/system/parser-engines",
    "/system/parser-engines/check",
    "/system/storage-engine-check",
    "/system/storage-engine-status",
    "/tenants",
    "/tenants/all",
    "/tenants/kv/{key}",
    "/tenants/search",
    "/tenants/{tenant_id}",
    "/tenants/{tenant_id}/api-keys",
    "/tenants/{tenant_id}/api-keys/{key_id}",
    "/tenants/{tenant_id}/api-principal-config",
    "/user/favorites",
    "/user/favorites/{type}/{id}",
    "/vector-stores",
    "/vector-stores/test",
    "/vector-stores/types",
    "/vector-stores/{store_id}",
    "/vector-stores/{store_id}/test",
    "/web-search-providers",
    "/web-search-providers/test",
    "/web-search-providers/types",
    "/web-search-providers/{provider_id}",
    "/web-search-providers/{provider_id}/test",
    "/web-search/providers",
}


@pytest.fixture(scope="session")
def openapi_paths() -> dict[str, Any]:
    """Live OpenAPI ``paths`` mapping built from a fresh ``create_app()``.

    The app is constructed without entering its lifespan, so no DB engine,
    OIDC client, or MCP transport is started - OpenAPI generation only
    walks the registered routes and needs none of those resources.

    ``starlette.testclient`` emits a one-time deprecation notice on import
    under the pinned starlette release (it prefers ``httpx2``). The project
    turns warnings into errors, so the import + schema fetch run inside a
    narrow ``catch_warnings`` scope that ignores only that notice; every
    other warning keeps failing the build.
    """
    application = create_app()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Using `httpx` with `starlette.testclient` is deprecated",
        )
        from fastapi.testclient import TestClient

        client = TestClient(application)
        response = client.get(_OPENAPI_PATH)
    assert response.status_code == 200, response.text
    return response.json()["paths"]


def test_public_routes_available(openapi_paths: dict[str, Any]) -> None:
    """The auth-exempt endpoints must remain exactly this set.

    Filters the live paths down to the known public set; the comparison
    fails if any public route is removed or renamed. Additions of new
    public routes are caught by ``test_all_routes_available``.
    """
    public_set = set(_EXPECTED_PUBLIC_ROUTES)
    actual = sorted(path for path in openapi_paths if path in public_set)
    assert actual == _EXPECTED_PUBLIC_ROUTES


def test_all_routes_available(openapi_paths: dict[str, Any]) -> None:
    """The full route surface must match the pinned set exactly."""
    assert set(openapi_paths.keys()) == _EXPECTED_ALL_ROUTES
