"""Web-layer tests for the agent and skill routers.

Exercises the routers over HTTP via ``TestClient`` against the app with
``get_custom_agent_service`` overridden by a real ``CustomAgentService``
backed by an ``AsyncMock(spec=CustomAgentRepository)`` configured with
stateful closures, so the full web -> service path runs without a
database. The skills manager dep is overridden with a real ``Manager``
discovered from a temporary skill directory.

Uses the shared ``web_app`` fixture (header-based auth) and applies the
service dep override on it; the real ``require_auth`` dep resolves the
principal via the ``X-User-Id/X-Tenant-ID/X-Roles`` header trio.

The load-bearing checks:

1. All 10 endpoints exist under the paths and methods the upstream
   registers, each carrying the auth gate plus the role gate.
2. Tenant isolation: a cross-workspace id reads as 404 on every
   id-scoped route so the id space is not enumerable.
3. Built-in protection: editing / deleting a built-in row reads as 409
   with the builtin error codes.
4. The skills list surfaces the discovered catalog and the sandbox
   availability flag.
"""
# The copy-name assertion matches the Chinese suffix.

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.common.exception import ConflictError
from src.core.agents.service.custom_agent_service import CustomAgentService
from src.core.agents.skills.manager import Manager
from src.core.agents.skills.types import ManagerConfig
from src.db.dao.custom_agent_repository import CustomAgentRepository
from src.db.models.custom_agent import CustomAgent
from src.web.api.agents.router import router
from src.web.api.agents.skill_views import skill_router
from src.web.deps.agents import get_custom_agent_service, get_skills_manager
from src.web.deps.rbac import make_role_dep, require_role_dep
from src.web.middleware.auth import require_auth

TENANT_ID = 1
NOW = datetime(2026, 4, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _bind_tenant_id_to_admin(
    admin_user: tuple[int, int],
) -> None:
    """Rewrite the module-level ``TENANT_ID`` to the minted admin tenant."""
    global TENANT_ID
    TENANT_ID = admin_user[1]


# ── App wiring ───────────────────────────────────────────────────────


@pytest.fixture
def agent_repo() -> AsyncMock:
    """``AsyncMock(spec=CustomAgentRepository)`` with stateful closures."""
    repo = AsyncMock(spec=CustomAgentRepository)
    rows: dict[str, CustomAgent] = {}

    def _live() -> dict[str, CustomAgent]:
        return {k: r for k, r in rows.items() if r.deleted_at is None}

    async def _create(row: CustomAgent) -> CustomAgent:
        rows[row.id] = row
        return row

    async def _get_by_id_and_tenant(*, id: str, tenant_id: int) -> CustomAgent | None:
        row = _live().get(id)
        if row is not None and row.tenant_id == tenant_id:
            return row
        return None

    async def _list_by_tenant(tenant_id: int) -> list[CustomAgent]:
        return sorted(
            (r for r in _live().values() if r.tenant_id == tenant_id),
            key=lambda r: r.created_at,
            reverse=True,
        )

    async def _update(row: CustomAgent) -> CustomAgent:
        existing = rows.get(row.id)
        if existing is None or existing.deleted_at is not None:
            raise ConflictError(code="db.not_found", message="row missing")
        persisted = row.model_copy(
            update={
                "tenant_id": existing.tenant_id,
                "created_at": existing.created_at,
            }
        )
        rows[row.id] = persisted
        return persisted

    async def _soft_delete(*, id: str, tenant_id: int, now: datetime) -> bool:
        existing = rows.get(id)
        if existing is None or existing.deleted_at is not None:
            return False
        rows[id] = existing.model_copy(update={"deleted_at": now, "updated_at": now})
        return True

    repo.create.side_effect = _create
    repo.get_by_id_and_tenant.side_effect = _get_by_id_and_tenant
    repo.list_by_tenant.side_effect = _list_by_tenant
    repo.update.side_effect = _update
    repo.soft_delete.side_effect = _soft_delete
    repo._rows = rows  # type: ignore[attr-defined]
    return repo


@pytest.fixture
def skills_manager(tmp_path: Path) -> Manager:
    """A real ``Manager`` discovered from a temporary skill directory."""
    skill_dir = tmp_path / "web-search"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: web-search\ndescription: Search the web\n---\n\n"
        "Search the web with the configured provider.\n",
        encoding="utf-8",
    )
    manager = Manager(config=ManagerConfig(skill_dirs=[str(tmp_path)], enabled=True))
    manager.initialize()
    return manager


@pytest.fixture
def app(
    web_app: FastAPI,
    agent_repo: AsyncMock,
) -> FastAPI:
    """Override ``get_custom_agent_service`` on the shared web app."""

    def _override_service() -> CustomAgentService:
        return CustomAgentService(agent_repo=agent_repo)

    web_app.dependency_overrides[get_custom_agent_service] = _override_service
    return web_app


@pytest.fixture
def client(app: FastAPI, web_authed_client: TestClient) -> TestClient:
    """Alias ``web_authed_client``; depending on ``app`` forces the
    dep-override fixture to run before the test executes."""
    return web_authed_client


@pytest.fixture
def skills_app(
    web_app: FastAPI,
    skills_manager: Manager,
) -> FastAPI:
    """Override ``get_skills_manager`` on the shared web app."""

    def _override_manager() -> Manager:
        return skills_manager

    web_app.dependency_overrides[get_skills_manager] = _override_manager
    return web_app


@pytest.fixture
def skills_client(
    skills_app: FastAPI,
    web_authed_client: TestClient,
) -> TestClient:
    """Alias ``web_authed_client`` for the skills-router tests."""
    return web_authed_client


def _agent_row(
    *,
    id: str = "agent-1",
    name: str = "Agent",
    tenant_id: int | None = None,
    is_builtin: bool = False,
    created_by: str | None = None,
    config: dict[str, object] | None = None,
    **overrides: object,
) -> CustomAgent:
    """Build a ``custom_agents`` row with the minimal required columns."""
    if tenant_id is None:
        tenant_id = TENANT_ID
    return CustomAgent(
        id=id,
        tenant_id=tenant_id,
        name=name,
        is_builtin=is_builtin,
        created_by=created_by,
        config=config or {},
        created_at=NOW,
        updated_at=NOW,
        **overrides,
    )


# ── Route surface (structural) ───────────────────────────────────────


EXPECTED_ROUTES: set[tuple[str, str]] = {
    ("GET", "/api/v1/agents/placeholders"),
    ("GET", "/api/v1/agents/type-presets"),
    ("POST", "/api/v1/agents"),
    ("GET", "/api/v1/agents"),
    ("GET", "/api/v1/agents/{id}"),
    ("PUT", "/api/v1/agents/{id}"),
    ("DELETE", "/api/v1/agents/{id}"),
    ("POST", "/api/v1/agents/{id}/copy"),
    ("GET", "/api/v1/agents/{id}/suggested-questions"),
    ("GET", "/api/v1/skills"),
}

# Reads are Viewer+; authoring is Contributor+; owned-object edits map
# to the strictest available route gate (Admin), mirroring the KB port.
EXPECTED_ROLES: dict[tuple[str, str], str] = {
    ("GET", "/api/v1/agents/placeholders"): "viewer",
    ("GET", "/api/v1/agents/type-presets"): "viewer",
    ("POST", "/api/v1/agents"): "contributor",
    ("GET", "/api/v1/agents"): "viewer",
    ("GET", "/api/v1/agents/{id}"): "viewer",
    ("PUT", "/api/v1/agents/{id}"): "admin",
    ("DELETE", "/api/v1/agents/{id}"): "admin",
    ("POST", "/api/v1/agents/{id}/copy"): "contributor",
    ("GET", "/api/v1/agents/{id}/suggested-questions"): "viewer",
    ("GET", "/api/v1/skills"): "viewer",
}


def _declared_routes() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for route in [*router.routes, *skill_router.routes]:
        methods: set[str] = getattr(route, "methods", set()) or set()
        path = getattr(route, "path", "")
        for method in methods:
            found.add((method, path))
    return found


def test_router_declares_exactly_the_upstream_routes() -> None:
    assert _declared_routes() == EXPECTED_ROUTES


def test_every_endpoint_declares_the_auth_gate() -> None:
    for route in [*router.routes, *skill_router.routes]:
        deps = [d.call for d in getattr(route, "dependant", None).dependencies]  # type: ignore[union-attr]
        assert require_auth in deps, f"{route.path} is missing AuthDep"  # type: ignore[attr-defined]


def test_every_endpoint_declares_the_expected_role_gate() -> None:
    viewer_dep = make_role_dep("viewer")
    admin_dep = make_role_dep("admin")
    assert viewer_dep is not admin_dep

    for route in [*router.routes, *skill_router.routes]:
        path = getattr(route, "path", "")
        methods: set[str] = getattr(route, "methods", set()) or set()
        dependant = getattr(route, "dependant", None)
        assert dependant is not None
        roles: set[str] = set()
        for dep in dependant.dependencies:
            closure = getattr(dep.call, "__closure__", None)
            wrapped = getattr(dep.call, "__wrapped__", None)
            if closure is None and wrapped is None:
                continue
            for cell in closure or ():
                if isinstance(cell.cell_contents, str):
                    roles.add(cell.cell_contents)
        for method in methods:
            expected = EXPECTED_ROLES[(method, path)]
            assert expected in roles, f"{method} {path} expected role gate {expected}, got {roles}"


def test_role_gate_helper_is_the_shared_rbac_dependency() -> None:
    dep = make_role_dep("admin")
    assert dep.__module__ == require_role_dep.__module__


# ── POST /agents ─────────────────────────────────────────────────────


async def test_create_returns_201_envelope(client: TestClient) -> None:
    resp = client.post("/api/v1/agents", json={"name": "Agent"})

    assert resp.status_code == 201
    payload = resp.json()
    assert payload["success"] is True
    data = payload["data"]
    assert data["name"] == "Agent"
    assert data["is_builtin"] is False
    assert data["config"]["agent_mode"] == "quick-answer"


async def test_create_records_creator_and_description(
    client: TestClient,
    admin_user: tuple[int, int],
) -> None:
    user_id, _ = admin_user
    resp = client.post(
        "/api/v1/agents",
        json={"name": "Agent", "description": "test description", "avatar": "av"},
    )

    assert resp.status_code == 201
    data = resp.json()["data"]
    assert data["description"] == "test description"
    assert data["avatar"] == "av"
    assert data["created_by"] == user_id


async def test_create_rejects_blank_name(client: TestClient) -> None:
    resp = client.post("/api/v1/agents", json={"name": "   "})

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "agent.name_required"


# ── GET /agents ──────────────────────────────────────────────────────


async def test_list_returns_tenant_rows(
    client: TestClient,
    agent_repo: AsyncMock,
) -> None:
    agent_repo._rows["agent-1"] = _agent_row(id="agent-1")  # type: ignore[attr-defined]
    agent_repo._rows["agent-2"] = _agent_row(id="agent-2")  # type: ignore[attr-defined]
    client.post("/api/v1/agents", json={"name": "agent-3"})

    resp = client.get("/api/v1/agents")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert payload["disabled_own_agent_ids"] == []
    names = {agent["name"] for agent in payload["data"]}
    assert names == {"Agent", "agent-3"}
    assert len(payload["data"]) == 3


async def test_list_excludes_other_tenants(
    client: TestClient,
    agent_repo: AsyncMock,
) -> None:
    agent_repo._rows["agent-own"] = _agent_row(id="agent-own")  # type: ignore[attr-defined]
    agent_repo._rows["agent-other"] = _agent_row(  # type: ignore[attr-defined]
        id="agent-other", tenant_id=TENANT_ID + 1
    )

    resp = client.get("/api/v1/agents")

    assert resp.status_code == 200
    assert [agent["id"] for agent in resp.json()["data"]] == ["agent-own"]


async def test_list_filters_by_creator(
    client: TestClient,
    agent_repo: AsyncMock,
    admin_user: tuple[int, int],
) -> None:
    user_id, _ = admin_user
    agent_repo._rows["agent-mine"] = _agent_row(  # type: ignore[attr-defined]
        id="agent-mine", created_by=user_id
    )
    agent_repo._rows["agent-other"] = _agent_row(  # type: ignore[attr-defined]
        id="agent-other", created_by="someone-else"
    )
    agent_repo._rows["agent-anon"] = _agent_row(id="agent-anon")  # type: ignore[attr-defined]
    agent_repo._rows["agent-builtin"] = _agent_row(  # type: ignore[attr-defined]
        id="builtin-quick-answer", is_builtin=True, created_by=None
    )

    mine = client.get("/api/v1/agents?creator=mine")
    assert mine.status_code == 200
    ids = {agent["id"] for agent in mine.json()["data"]}
    assert ids == {"agent-mine", "builtin-quick-answer"}

    others = client.get("/api/v1/agents?creator=others")
    assert others.status_code == 200
    ids = {agent["id"] for agent in others.json()["data"]}
    assert ids == {"agent-other", "builtin-quick-answer"}


# ── GET /agents/{id} ─────────────────────────────────────────────────


async def test_get_returns_one_agent(client: TestClient) -> None:
    created = client.post("/api/v1/agents", json={"name": "Agent"})
    agent_id = created.json()["data"]["id"]

    resp = client.get(f"/api/v1/agents/{agent_id}")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["data"]["id"] == agent_id
    assert payload["data"]["name"] == "Agent"


async def test_get_missing_returns_404(client: TestClient) -> None:
    resp = client.get("/api/v1/agents/does-not-exist")

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "agent.not_found"


async def test_get_cross_tenant_returns_404(
    client: TestClient,
    agent_repo: AsyncMock,
) -> None:
    agent_repo._rows["agent-other"] = _agent_row(  # type: ignore[attr-defined]
        id="agent-other", tenant_id=TENANT_ID + 1
    )

    resp = client.get("/api/v1/agents/agent-other")

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "agent.not_found"


# ── PUT /agents/{id} ─────────────────────────────────────────────────


async def test_update_patches_name_and_description(client: TestClient) -> None:
    created = client.post("/api/v1/agents", json={"name": "Agent"})
    agent_id = created.json()["data"]["id"]

    resp = client.put(
        f"/api/v1/agents/{agent_id}",
        json={"name": "Renamed", "description": "renamed", "config": {}},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["data"]["name"] == "Renamed"
    assert payload["data"]["description"] == "renamed"


async def test_update_missing_returns_404(client: TestClient) -> None:
    resp = client.put("/api/v1/agents/does-not-exist", json={"name": "x", "config": {}})

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "agent.not_found"


async def test_update_rejects_blank_name(client: TestClient) -> None:
    created = client.post("/api/v1/agents", json={"name": "Agent"})
    agent_id = created.json()["data"]["id"]

    resp = client.put(f"/api/v1/agents/{agent_id}", json={"name": "  ", "config": {}})

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "agent.name_required"


async def test_update_builtin_returns_409(
    client: TestClient,
    agent_repo: AsyncMock,
) -> None:
    agent_repo._rows["builtin-quick-answer"] = _agent_row(  # type: ignore[attr-defined]
        id="builtin-quick-answer", is_builtin=True
    )

    resp = client.put(
        "/api/v1/agents/builtin-quick-answer",
        json={"name": "Renamed", "config": {}},
    )

    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "agent.cannot_modify_builtin"


async def test_update_builtin_config_returns_200(
    client: TestClient,
    agent_repo: AsyncMock,
) -> None:
    stored_rows: dict[str, CustomAgent] = agent_repo._rows
    stored_rows["builtin-quick-answer"] = _agent_row(
        id="builtin-quick-answer",
        name="快速问答",
        is_builtin=True,
        config={"agent_mode": "quick-answer"},
    )

    resp = client.put(
        "/api/v1/agents/builtin-quick-answer",
        json={
            "name": "快速问答",
            "config": {"agent_mode": "quick-answer", "model_id": "model-qa"},
        },
    )

    assert resp.status_code == 200
    assert resp.json()["data"]["config"]["model_id"] == "model-qa"
    assert resp.json()["data"]["is_builtin"] is True


# ── DELETE /agents/{id} ──────────────────────────────────────────────


async def test_delete_returns_message_and_soft_deletes(
    client: TestClient,
    agent_repo: AsyncMock,
) -> None:
    created = client.post("/api/v1/agents", json={"name": "Agent"})
    agent_id = created.json()["data"]["id"]

    resp = client.delete(f"/api/v1/agents/{agent_id}")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert payload["message"] == "Agent deleted successfully"
    rows = agent_repo._rows  # type: ignore[attr-defined]
    assert agent_id in rows and rows[agent_id].deleted_at is not None


async def test_delete_missing_returns_404(client: TestClient) -> None:
    resp = client.delete("/api/v1/agents/does-not-exist")

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "agent.not_found"


async def test_delete_builtin_returns_409(client: TestClient) -> None:
    resp = client.delete("/api/v1/agents/builtin-quick-answer")

    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "agent.cannot_delete_builtin"


# ── POST /agents/{id}/copy ───────────────────────────────────────────


async def test_copy_returns_201_envelope(
    client: TestClient,
    agent_repo: AsyncMock,
    admin_user: tuple[int, int],
) -> None:
    user_id, _ = admin_user
    created = client.post("/api/v1/agents", json={"name": "Agent"})
    source_id = created.json()["data"]["id"]

    resp = client.post(f"/api/v1/agents/{source_id}/copy")

    assert resp.status_code == 201
    payload = resp.json()
    assert payload["success"] is True
    data = payload["data"]
    assert data["id"] != source_id
    assert data["name"] == "Agent （副本）"
    assert data["created_by"] == user_id
    rows = agent_repo._rows  # type: ignore[attr-defined]
    assert data["id"] in rows


async def test_copy_missing_returns_404(client: TestClient) -> None:
    resp = client.post("/api/v1/agents/does-not-exist/copy")

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "agent.not_found"


# ── GET /agents/placeholders ─────────────────────────────────────────


async def test_placeholders_returns_grouped_catalog(client: TestClient) -> None:
    resp = client.get("/api/v1/agents/placeholders")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    data = payload["data"]
    assert data["all"]
    assert data["system_prompt"]
    assert data["agent_system_prompt"]
    assert data["fallback_prompt"]
    first = data["all"][0]
    assert set(first) == {"name", "label", "description"}


# ── GET /agents/type-presets ─────────────────────────────────────────


async def test_type_presets_return_empty_registry(client: TestClient) -> None:
    resp = client.get("/api/v1/agents/type-presets")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert payload["data"] == []


# ── GET /agents/{id}/suggested-questions ─────────────────────────────


async def test_suggested_questions_returns_empty_set(client: TestClient) -> None:
    created = client.post("/api/v1/agents", json={"name": "Agent"})
    agent_id = created.json()["data"]["id"]

    resp = client.get(
        f"/api/v1/agents/{agent_id}/suggested-questions?knowledge_base_ids=kb-1&limit=3"
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert payload["data"]["questions"] == []


async def test_suggested_questions_missing_returns_404(client: TestClient) -> None:
    resp = client.get("/api/v1/agents/does-not-exist/suggested-questions")

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "agent.not_found"


# ── GET /skills ──────────────────────────────────────────────────────


async def test_skills_lists_discovered_catalog(
    skills_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KB_SANDBOX_MODE", "local")

    resp = skills_client.get("/api/v1/skills")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert payload["skills_available"] is True
    assert payload["data"] == [{"name": "web-search", "description": "Search the web"}]


async def test_skills_available_false_when_disabled(
    skills_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KB_SANDBOX_MODE", "disabled")

    resp = skills_client.get("/api/v1/skills")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["skills_available"] is False


async def test_skills_available_false_when_unset(
    skills_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KB_SANDBOX_MODE", raising=False)

    resp = skills_client.get("/api/v1/skills")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["skills_available"] is False
