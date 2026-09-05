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
    "/api/v1/auth/auto-setup",
    "/api/v1/auth/config",
    "/api/v1/auth/invitations/lookup",
    "/api/v1/auth/login",
    "/api/v1/auth/oidc/callback",
    "/api/v1/auth/oidc/config",
    "/api/v1/auth/oidc/url",
    "/api/v1/auth/refresh",
    "/api/v1/auth/register",
    "/api/v1/auth/register-by-invite",
    "/api/v1/embed/{channel_id}/agent-chat/{session_id}",
    "/api/v1/embed/{channel_id}/chunks/{chunk_id}",
    "/api/v1/embed/{channel_id}/config",
    "/api/v1/embed/{channel_id}/exchange",
    "/api/v1/embed/{channel_id}/files",
    "/api/v1/embed/{channel_id}/knowledge-chat/{session_id}",
    "/api/v1/embed/{channel_id}/messages/{session_id}/load",
    "/api/v1/embed/{channel_id}/sessions",
    "/api/v1/embed/{channel_id}/sessions/{session_id}/events",
    "/api/v1/embed/{channel_id}/sessions/{session_id}/mcp-oauth-resolutions/{pending_id}",
    "/api/v1/embed/{channel_id}/sessions/{session_id}/mcp-oauth-resolutions/{pending_id}/cancel",
    "/api/v1/embed/{channel_id}/sessions/{session_id}/mcp-services/{service_id}/oauth/authorize-url",
    "/api/v1/embed/{channel_id}/sessions/{session_id}/mcp-services/{service_id}/oauth/status",
    "/api/v1/embed/{channel_id}/sessions/{session_id}/messages/{message_id}/suggestions",
    "/api/v1/embed/{channel_id}/sessions/{session_id}/stop",
    "/api/v1/embed/{channel_id}/sessions/{session_id}/suggestion-events",
    "/api/v1/embed/{channel_id}/sessions/{session_id}/tool-approvals/{pending_id}",
    "/api/v1/embed/{channel_id}/suggested-questions",
    "/api/v1/mcp-oauth/callback",
    "/health",
]

# The complete route surface exposed by the app. Adding or removing a
# route requires updating this set; otherwise the gate fails.
_EXPECTED_ALL_ROUTES: set[str] = {
    "/api/v1/agent-chat/{session_id}",
    "/api/v1/agents",
    "/api/v1/agents/placeholders",
    "/api/v1/agents/type-presets",
    "/api/v1/agents/{agent_id}/embed-channels",
    "/api/v1/agents/{agent_id}/im-channels",
    "/api/v1/agents/{agent_id}/shares",
    "/api/v1/agents/{agent_id}/shares/{share_id}",
    "/api/v1/agents/{id}",
    "/api/v1/agents/{id}/copy",
    "/api/v1/agents/{id}/suggested-questions",
    "/api/v1/auth/auto-setup",
    "/api/v1/auth/change-password",
    "/api/v1/auth/config",
    "/api/v1/auth/invitations/lookup",
    "/api/v1/auth/login",
    "/api/v1/auth/logout",
    "/api/v1/auth/me",
    "/api/v1/auth/me/preferences",
    "/api/v1/auth/oidc/callback",
    "/api/v1/auth/oidc/config",
    "/api/v1/auth/oidc/url",
    "/api/v1/auth/refresh",
    "/api/v1/auth/register",
    "/api/v1/auth/register-by-invite",
    "/api/v1/auth/tenant",
    "/api/v1/auth/validate",
    "/api/v1/chunker/preview",
    "/api/v1/chunks/by-id/{id}",
    "/api/v1/chunks/by-id/{id}/questions",
    "/api/v1/chunks/by-id/{id}/questions/regenerate",
    "/api/v1/chunks/{knowledge_id}",
    "/api/v1/chunks/{knowledge_id}/{id}",
    "/api/v1/chunks/{knowledge_id}/{id}/revert",
    "/api/v1/chunks/{knowledge_id}/{id}/revisions",
    "/api/v1/datasource",
    "/api/v1/datasource/logs/{log_id}",
    "/api/v1/datasource/types",
    "/api/v1/datasource/validate-credentials",
    "/api/v1/datasource/{id}",
    "/api/v1/datasource/{id}/logs",
    "/api/v1/datasource/{id}/pause",
    "/api/v1/datasource/{id}/resource-ancestors",
    "/api/v1/datasource/{id}/resources",
    "/api/v1/datasource/{id}/resume",
    "/api/v1/datasource/{id}/sync",
    "/api/v1/datasource/{id}/validate",
    "/api/v1/embed-channels",
    "/api/v1/embed-channels/{channel_id}",
    "/api/v1/embed-channels/{channel_id}/preview-session",
    "/api/v1/embed-channels/{channel_id}/rotate-token",
    "/api/v1/embed-channels/{channel_id}/stats",
    "/api/v1/embed/{channel_id}/agent-chat/{session_id}",
    "/api/v1/embed/{channel_id}/chunks/{chunk_id}",
    "/api/v1/embed/{channel_id}/config",
    "/api/v1/embed/{channel_id}/exchange",
    "/api/v1/embed/{channel_id}/files",
    "/api/v1/embed/{channel_id}/knowledge-chat/{session_id}",
    "/api/v1/embed/{channel_id}/messages/{session_id}/load",
    "/api/v1/embed/{channel_id}/sessions",
    "/api/v1/embed/{channel_id}/sessions/{session_id}/events",
    "/api/v1/embed/{channel_id}/sessions/{session_id}/mcp-oauth-resolutions/{pending_id}",
    "/api/v1/embed/{channel_id}/sessions/{session_id}/mcp-oauth-resolutions/{pending_id}/cancel",
    "/api/v1/embed/{channel_id}/sessions/{session_id}/mcp-services/{service_id}/oauth/authorize-url",
    "/api/v1/embed/{channel_id}/sessions/{session_id}/mcp-services/{service_id}/oauth/status",
    "/api/v1/embed/{channel_id}/sessions/{session_id}/messages/{message_id}/suggestions",
    "/api/v1/embed/{channel_id}/sessions/{session_id}/stop",
    "/api/v1/embed/{channel_id}/sessions/{session_id}/suggestion-events",
    "/api/v1/embed/{channel_id}/sessions/{session_id}/tool-approvals/{pending_id}",
    "/api/v1/embed/{channel_id}/suggested-questions",
    "/api/v1/evaluation",
    "/api/v1/faq/import/progress/{task_id}",
    "/api/v1/im-channels",
    "/api/v1/im-channels/{channel_id}",
    "/api/v1/im-channels/{channel_id}/toggle",
    "/api/v1/im/callback/{channel_id}",
    "/api/v1/initialization/asr/check",
    "/api/v1/initialization/config/{kb_id}",
    "/api/v1/initialization/embedding/test",
    "/api/v1/initialization/extract/fabri-tag",
    "/api/v1/initialization/extract/fabri-text",
    "/api/v1/initialization/extract/text-relation",
    "/api/v1/initialization/multimodal/test",
    "/api/v1/initialization/ollama/download/progress/{task_id}",
    "/api/v1/initialization/ollama/download/tasks",
    "/api/v1/initialization/ollama/models",
    "/api/v1/initialization/ollama/models/check",
    "/api/v1/initialization/ollama/models/download",
    "/api/v1/initialization/ollama/status",
    "/api/v1/initialization/remote/check",
    "/api/v1/initialization/rerank/check",
    "/api/v1/knowledge-bases",
    "/api/v1/knowledge-bases/copy",
    "/api/v1/knowledge-bases/{id}",
    "/api/v1/knowledge-bases/{id}/duplicate",
    "/api/v1/knowledge-bases/{id}/faq/entries",
    "/api/v1/knowledge-bases/{id}/faq/entries/export",
    "/api/v1/knowledge-bases/{id}/faq/entries/{entry_id}",
    "/api/v1/knowledge-bases/{id}/faq/entry",
    "/api/v1/knowledge-bases/{id}/hybrid-search",
    "/api/v1/knowledge-bases/{id}/knowledge",
    "/api/v1/knowledge-bases/{id}/knowledge/file",
    "/api/v1/knowledge-bases/{id}/knowledge/manual",
    "/api/v1/knowledge-bases/{id}/knowledge/passage",
    "/api/v1/knowledge-bases/{id}/knowledge/url",
    "/api/v1/knowledge-bases/{id}/move-targets",
    "/api/v1/knowledge-bases/{id}/pin",
    "/api/v1/knowledge-bases/{id}/tags",
    "/api/v1/knowledge-bases/{id}/tags/{tag_id}",
    "/api/v1/knowledge-bases/{kb_id}/files",
    "/api/v1/knowledge-chat/{session_id}",
    "/api/v1/knowledge-search",
    "/api/v1/knowledge/batch",
    "/api/v1/knowledge/batch-delete",
    "/api/v1/knowledge/batch-reparse",
    "/api/v1/knowledge/move",
    "/api/v1/knowledge/move/progress/{task_id}",
    "/api/v1/knowledge/search",
    "/api/v1/knowledge/tags",
    "/api/v1/knowledge/{id}",
    "/api/v1/knowledge/{id}/cancel-parse",
    "/api/v1/knowledge/{id}/download",
    "/api/v1/knowledge/{id}/preview",
    "/api/v1/knowledge/{id}/clone",
    "/api/v1/knowledge/{id}/regenerate-summary",
    "/api/v1/knowledge/{id}/reparse",
    "/api/v1/knowledge/{id}/spans",
    "/api/v1/knowledgebase/{kb_id}/wiki/auto-fix",
    "/api/v1/knowledgebase/{kb_id}/wiki/folders",
    "/api/v1/knowledgebase/{kb_id}/wiki/folders/{folder_id}",
    "/api/v1/knowledgebase/{kb_id}/wiki/graph",
    "/api/v1/knowledgebase/{kb_id}/wiki/index",
    "/api/v1/knowledgebase/{kb_id}/wiki/lint",
    "/api/v1/knowledgebase/{kb_id}/wiki/move-page",
    "/api/v1/knowledgebase/{kb_id}/wiki/pages",
    "/api/v1/knowledgebase/{kb_id}/wiki/pages/{slug}",
    "/api/v1/knowledgebase/{kb_id}/wiki/rebuild-links",
    "/api/v1/knowledgebase/{kb_id}/wiki/search",
    "/api/v1/knowledgebase/{kb_id}/wiki/stats",
    "/api/v1/mcp-oauth/callback",
    "/api/v1/mcp-services",
    "/api/v1/mcp-services/{service_id}",
    "/api/v1/mcp-services/{service_id}/oauth/authorize-url",
    "/api/v1/mcp-services/{service_id}/oauth/status",
    "/api/v1/mcp-services/{service_id}/oauth/token",
    "/api/v1/mcp-services/{service_id}/resources",
    "/api/v1/mcp-services/{service_id}/test",
    "/api/v1/mcp-services/{service_id}/tool-approvals",
    "/api/v1/mcp-services/{service_id}/tool-approvals/{tool_name}",
    "/api/v1/mcp-services/{service_id}/tools",
    "/api/v1/me/invitations",
    "/api/v1/me/invitations/{invitation_id}/accept",
    "/api/v1/me/invitations/{invitation_id}/decline",
    "/api/v1/me/invitations/pending-count",
    "/api/v1/messages/chat-history-stats",
    "/api/v1/messages/search",
    "/api/v1/messages/{session_id}/load",
    "/api/v1/messages/{session_id}/{message_id}",
    "/api/v1/meta/capabilities",
    "/api/v1/models",
    "/api/v1/models/providers",
    "/api/v1/models/cloud/status",
    "/api/v1/models/{model_id}",
    "/api/v1/models/{model_id}/debug",
    "/api/v1/organizations",
    "/api/v1/organizations/join",
    "/api/v1/organizations/join-by-id",
    "/api/v1/organizations/join-request",
    "/api/v1/organizations/preview/{code}",
    "/api/v1/organizations/search",
    "/api/v1/organizations/{id}",
    "/api/v1/organizations/{id}/invite",
    "/api/v1/organizations/{id}/invite-code",
    "/api/v1/organizations/{id}/join-requests",
    "/api/v1/organizations/{id}/join-requests/{request_id}/review",
    "/api/v1/organizations/{id}/leave",
    "/api/v1/organizations/{id}/members",
    "/api/v1/organizations/{id}/members/{tenant_id}",
    "/api/v1/organizations/{id}/request-upgrade",
    "/api/v1/organizations/{id}/search-tenants",
    "/api/v1/organizations/{id}/search-users",
    "/api/v1/sessions",
    "/api/v1/sessions/batch",
    "/api/v1/sessions/{session_id}",
    "/api/v1/sessions/{session_id}/messages",
    "/api/v1/sessions/{session_id}/messages/{message_id}/suggestions",
    "/api/v1/sessions/{session_id}/pin",
    "/api/v1/sessions/{session_id}/stop",
    "/api/v1/sessions/{session_id}/suggestion-events",
    "/api/v1/shared-agents",
    "/api/v1/shared-agents/disabled",
    "/api/v1/shared-knowledge-bases",
    "/api/v1/skills",
    "/api/v1/storage-backends",
    "/api/v1/storage-backends/test",
    "/api/v1/storage-backends/types",
    "/api/v1/storage-backends/{id}",
    "/api/v1/storage-backends/{id}/default",
    "/api/v1/storage-backends/{id}/test",
    "/api/v1/system/admin/api-keys",
    "/api/v1/system/admin/api-keys/{key_id}",
    "/api/v1/system/admin/audit-log",
    "/api/v1/system/admin/list",
    "/api/v1/system/admin/promote",
    "/api/v1/system/admin/revoke",
    "/api/v1/system/admin/runtime/queues",
    "/api/v1/system/admin/runtime/queues/{queue}/tasks",
    "/api/v1/system/admin/settings",
    "/api/v1/system/admin/settings/{key}",
    "/api/v1/system/admin/tenants/apply-default-storage-quota",
    "/api/v1/system/admin/users/reset-password",
    "/api/v1/system/docreader/reconnect",
    "/api/v1/system/info",
    "/api/v1/system/parser-engines",
    "/api/v1/system/parser-engines/check",
    "/api/v1/system/storage-engine-check",
    "/api/v1/system/storage-engine-status",
    "/api/v1/tenants",
    "/api/v1/tenants/all",
    "/api/v1/tenants/kv/{key}",
    "/api/v1/tenants/search",
    "/api/v1/tenants/{tenant_id}",
    "/api/v1/tenants/{tenant_id}/api-keys",
    "/api/v1/tenants/{tenant_id}/api-keys/{key_id}",
    "/api/v1/tenants/{tenant_id}/api-principal-config",
    "/api/v1/user/favorites",
    "/api/v1/user/favorites/{type}/{id}",
    "/api/v1/vector-stores",
    "/api/v1/vector-stores/test",
    "/api/v1/vector-stores/types",
    "/api/v1/vector-stores/{store_id}",
    "/api/v1/vector-stores/{store_id}/test",
    "/api/v1/web-search-providers",
    "/api/v1/web-search-providers/test",
    "/api/v1/web-search-providers/types",
    "/api/v1/web-search-providers/{provider_id}",
    "/api/v1/web-search-providers/{provider_id}/test",
    "/api/v1/web-search/providers",
    "/api/v1/wechat/qrcode",
    "/api/v1/wechat/qrcode/status",
    "/api/v1/cloud/credentials",
    "/files",
    "/health",
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
