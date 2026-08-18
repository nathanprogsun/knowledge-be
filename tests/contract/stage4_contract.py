"""Knowledge-domain contract tests (live).

Builds the real app via ``create_app`` and asserts the knowledge
endpoints' request/response schemas against the reference API fixtures:

- every covered endpoint answers with the documented HTTP status and
  the documented top-level response keys (``fixtures/
  knowledge_responses.json`` -> ``endpoints``);
- the response ``data`` payload is valid against the frozen wire
  contract (``src.core.contracts.knowledge`` / the web view models)
  and carries exactly the contract's serialized field set;
- the wiki feature is exercised against a wiki-enabled knowledge base.

The tests run against the real database. A failing assertion here means
the live web layer deviates from the reference wire shape — that is a
finding, not a test bug: the web layer is already merged, so deviations
are reported rather than silently fixed.

Seeding follows the shared integration pattern: every test mints a
fresh ``(user_id, tenant_id)`` principal via ``make_user_org``, so per-
test isolation comes from random ids rather than DB cleanup. The chunk
tests use ``make_int32_test_tenant_id`` because the ``chunks.tenant_id``
column is INTEGER.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from secrets import token_hex
from typing import NamedTuple

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from src.core.contracts import knowledge as contracts
from src.core.knowledge.documents.service.knowledge_service import KnowledgeService
from src.core.tenants.member_service import ROLE_OWNER
from src.db.base import DatabaseEngine
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.chunk import Chunk as ChunkRow
from src.web.api.knowledge.wiki import views as wiki_views
from tests.contract.conftest import make_int32_test_tenant_id
from tests.contract.test_knowledge_invariants import model_wire_fields

_REFERENCE = "reference fixture"


# ── Seed contexts ────────────────────────────────────────────────────


class KBSeed(NamedTuple):
    client: TestClient
    tenant_id: int
    kb_id: str


class KnowledgeSeed(NamedTuple):
    client: TestClient
    tenant_id: int
    kb_id: str
    knowledge_id: str


class ChunkSeed(NamedTuple):
    client: TestClient
    tenant_id: int
    kb_id: str
    knowledge_id: str
    chunk_id: str


class TagSeed(NamedTuple):
    client: TestClient
    tenant_id: int
    kb_id: str
    tag_id: str


class FAQSeed(NamedTuple):
    client: TestClient
    tenant_id: int
    kb_id: str
    knowledge_id: str
    entry_id: int


class WikiSeed(NamedTuple):
    client: TestClient
    tenant_id: int
    kb_id: str


def _fixture() -> dict[str, object]:
    import json
    from pathlib import Path

    path = Path(__file__).parent / "fixtures" / "knowledge_responses.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _endpoint_spec(method: str, path: str) -> tuple[int, list[str]]:
    """Return ``(status, top-level keys)`` from the fixture for an endpoint."""
    endpoints = _fixture().get("endpoints", {})
    spec = endpoints.get(f"{method} {path}")
    assert isinstance(spec, dict), f"endpoint fixture missing: {method} {path}"
    status = spec.get("status")
    keys = spec.get("keys")
    assert isinstance(status, int) and isinstance(keys, list)
    return status, [k for k in keys if isinstance(k, str)]


def _assert_keys(actual: dict[str, object], expected: list[str], label: str) -> None:
    actual_set = set(actual)
    expected_set = set(expected)
    assert actual_set == expected_set, (
        f"{label}: response keys diverge from the {_REFERENCE}.\n"
        f"  missing: {sorted(expected_set - actual_set)}\n"
        f"  extra:   {sorted(actual_set - expected_set)}"
    )


def _assert_payload(
    payload: object,
    model: type[BaseModel],
    label: str,
    *,
    expect_exact_fields: bool = True,
) -> None:
    """Assert a response ``data`` payload matches ``model`` exactly."""
    assert isinstance(payload, dict), f"{label}: expected a JSON object, got {type(payload)}"
    if expect_exact_fields:
        _assert_keys(payload, model_wire_fields(model), f"{label} data")
    try:
        model.model_validate(payload)
    except ValidationError as exc:  # pragma: no cover - exercised only on drift
        pytest.fail(f"{label}: payload is not valid against {model.__name__}: {exc}")


def _contract_fields(name: str) -> list[str]:
    """Return the fixture's expected wire field set for a contract."""
    contracts_map = _fixture().get("contracts", {})
    assert isinstance(contracts_map, dict), "fixture 'contracts' section missing"
    fields = contracts_map.get(name)
    assert isinstance(fields, list), f"contract fixture missing: {name}"
    return [f for f in fields if isinstance(f, str)]


def _authed_client(app: FastAPI, user_id: str, tenant_id: int) -> TestClient:
    client = TestClient(app=app)
    client.headers.update(
        {
            "X-User-Id": user_id,
            "X-Tenant-ID": str(tenant_id),
            "X-Roles": ROLE_OWNER,
        }
    )
    return client


# ── Live endpoint fixtures ───────────────────────────────────────────


@pytest_asyncio.fixture
async def kb_seed(
    authed_client: TestClient,
    admin_user: tuple[str, int],
) -> KBSeed:
    """A document knowledge base under a fresh principal."""
    client = authed_client
    _user_id, tenant_id = admin_user
    response = client.post(
        "/api/v1/knowledge-bases",
        json={"name": "contract-kb", "type": "document"},
    )
    assert response.status_code == 201, response.text
    return KBSeed(client, tenant_id, response.json()["data"]["id"])


@pytest_asyncio.fixture
async def knowledge_seed(kb_seed: KBSeed) -> KnowledgeSeed:
    """A manual knowledge item under the seeded knowledge base."""
    response = kb_seed.client.post(
        f"/api/v1/knowledge-bases/{kb_seed.kb_id}/knowledge/manual",
        json={"title": "contract-doc", "content": "# contract", "status": "draft"},
    )
    assert response.status_code == 200, response.text
    return KnowledgeSeed(
        kb_seed.client,
        kb_seed.tenant_id,
        kb_seed.kb_id,
        response.json()["data"]["id"],
    )


@pytest_asyncio.fixture
async def chunk_seed(
    app: FastAPI,
    _engine: DatabaseEngine,
    make_user_org: Callable[..., object],
) -> AsyncIterator[ChunkSeed]:
    """A knowledge item plus one seeded chunk under an int32-safe tenant.

    The ``chunks.tenant_id`` column is INTEGER, so the whole principal
    is minted with ``make_int32_test_tenant_id`` rather than the default
    BIGINT generator.
    """
    user_id, tenant_id = await make_user_org(tenant_id=make_int32_test_tenant_id())
    client = _authed_client(app, user_id, tenant_id)
    with client:
        response = client.post(
            "/api/v1/knowledge-bases",
            json={"name": "contract-chunk-kb", "type": "document"},
        )
        assert response.status_code == 201, response.text
        kb_id = response.json()["data"]["id"]
        response = client.post(
            f"/api/v1/knowledge-bases/{kb_id}/knowledge/manual",
            json={"title": "chunk-doc", "content": "# chunk", "status": "draft"},
        )
        assert response.status_code == 200, response.text
        knowledge_id = response.json()["data"]["id"]

        now = datetime.now(UTC)
        chunk_id = f"chunk-{token_hex(8)}"
        async with _engine.session_factory() as session, session.begin():
            repo = ChunkRepository(session)
            await repo.create_many(
                [
                    ChunkRow(
                        id=chunk_id,
                        tenant_id=tenant_id,
                        knowledge_base_id=kb_id,
                        knowledge_id=knowledge_id,
                        content="seeded chunk content",
                        chunk_index=0,
                        is_enabled=True,
                        start_at=0,
                        end_at=12,
                        chunk_type="text",
                        status=2,
                        content_hash=None,
                        created_at=now,
                        updated_at=now,
                    )
                ]
            )
        yield ChunkSeed(client, tenant_id, kb_id, knowledge_id, chunk_id)


@pytest_asyncio.fixture
async def tag_seed(kb_seed: KBSeed) -> TagSeed:
    """A tag under the seeded knowledge base."""
    response = kb_seed.client.post(
        f"/api/v1/knowledge-bases/{kb_seed.kb_id}/tags",
        json={"name": "contract-tag", "color": "#1890ff", "sort_order": 1},
    )
    assert response.status_code == 200, response.text
    return TagSeed(
        kb_seed.client,
        kb_seed.tenant_id,
        kb_seed.kb_id,
        response.json()["data"]["id"],
    )


@pytest_asyncio.fixture
async def faq_seed(
    authed_client: TestClient,
    admin_user: tuple[str, int],
    _engine: DatabaseEngine,
) -> FAQSeed:
    """A FAQ knowledge base with a FAQ container and one entry."""
    client = authed_client
    _user_id, tenant_id = admin_user
    response = client.post("/api/v1/knowledge-bases", json={"name": "contract-faq", "type": "faq"})
    assert response.status_code == 201, response.text
    kb_id = response.json()["data"]["id"]

    async with _engine.session_factory() as session:
        service = KnowledgeService(knowledge_repo=KnowledgeRepository(session))
        doc = await service.create_document(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            type="faq",
            title="FAQ container",
            source="faq",
            channel="web",
        )
        knowledge_id = doc.id
        await session.commit()

    response = client.post(
        f"/api/v1/knowledge-bases/{kb_id}/faq/entry",
        json={
            "standard_question": "how to reset?",
            "answers": ["click the reset button"],
            "is_enabled": True,
        },
    )
    assert response.status_code == 200, response.text
    entry_id = response.json()["data"]["id"]
    return FAQSeed(client, tenant_id, kb_id, knowledge_id, entry_id)


@pytest_asyncio.fixture
async def wiki_seed(
    authed_client: TestClient,
    admin_user: tuple[str, int],
) -> WikiSeed:
    """A wiki-enabled knowledge base (document type, wiki pipeline on)."""
    client = authed_client
    _user_id, tenant_id = admin_user
    response = client.post("/api/v1/knowledge-bases", json={"name": "contract-wiki", "type": "document"})
    assert response.status_code == 201, response.text
    kb_id = response.json()["data"]["id"]
    response = client.put(
        f"/api/v1/knowledge-bases/{kb_id}",
        json={
            "name": "contract-wiki",
            "config": {
                "indexing_strategy": {
                    "vector_enabled": False,
                    "keyword_enabled": False,
                    "wiki_enabled": True,
                },
                "wiki_config": {"max_pages_per_ingest": 10},
            },
        },
    )
    assert response.status_code == 200, response.text
    return WikiSeed(client, tenant_id, kb_id)


# ── Knowledge-base endpoints ─────────────────────────────────────────


def test_create_knowledge_base(kb_seed: KBSeed) -> None:
    spec = _endpoint_spec("POST", "/api/v1/knowledge-bases")
    response = kb_seed.client.post(
        "/api/v1/knowledge-bases",
        json={"name": "contract-kb-create", "type": "document"},
    )
    assert response.status_code == spec[0], response.text
    body = response.json()
    _assert_keys(body, spec[1], "POST /knowledge-bases")
    _assert_payload(body["data"], contracts.KnowledgeBase, "POST /knowledge-bases")


def test_get_knowledge_base(kb_seed: KBSeed) -> None:
    spec = _endpoint_spec("GET", "/api/v1/knowledge-bases/{id}")
    response = kb_seed.client.get(f"/api/v1/knowledge-bases/{kb_seed.kb_id}")
    assert response.status_code == spec[0], response.text
    body = response.json()
    _assert_keys(body, spec[1], "GET /knowledge-bases/{id}")
    _assert_payload(body["data"], contracts.KnowledgeBase, "GET /knowledge-bases/{id}")


def test_knowledge_base_wire_field_set_matches_reference(kb_seed: KBSeed) -> None:
    """The knowledge-base object carries the reference field set.

    The reference emits a computed ``capabilities`` block on every
    knowledge-base object; the current contract has no such field. This
    assertion is the contract check — a divergence here is a finding.
    """
    response = kb_seed.client.get(f"/api/v1/knowledge-bases/{kb_seed.kb_id}")
    assert response.status_code == 200, response.text
    _assert_keys(
        response.json()["data"], _contract_fields("KnowledgeBase"), "GET /knowledge-bases/{id} data"
    )


def test_list_knowledge_bases(kb_seed: KBSeed) -> None:
    spec = _endpoint_spec("GET", "/api/v1/knowledge-bases")
    response = kb_seed.client.get("/api/v1/knowledge-bases")
    assert response.status_code == spec[0], response.text
    body = response.json()
    _assert_keys(body, spec[1], "GET /knowledge-bases")
    assert isinstance(body["data"], list)
    for row in body["data"]:
        _assert_payload(row, contracts.KnowledgeBase, "GET /knowledge-bases row")


# ── Knowledge (document) endpoints ───────────────────────────────────


def test_create_manual_knowledge(kb_seed: KBSeed) -> None:
    spec = _endpoint_spec("POST", "/api/v1/knowledge-bases/{id}/knowledge/manual")
    response = kb_seed.client.post(
        f"/api/v1/knowledge-bases/{kb_seed.kb_id}/knowledge/manual",
        json={"title": "manual-create", "content": "# manual", "status": "draft"},
    )
    assert response.status_code == spec[0], response.text
    body = response.json()
    _assert_keys(body, spec[1], "POST /knowledge-bases/{id}/knowledge/manual")
    _assert_payload(
        body["data"],
        contracts.Knowledge,
        "POST /knowledge-bases/{id}/knowledge/manual",
    )


def test_list_knowledge(knowledge_seed: KnowledgeSeed) -> None:
    spec = _endpoint_spec("GET", "/api/v1/knowledge-bases/{id}/knowledge")
    response = knowledge_seed.client.get(f"/api/v1/knowledge-bases/{knowledge_seed.kb_id}/knowledge")
    assert response.status_code == spec[0], response.text
    body = response.json()
    _assert_keys(body, spec[1], "GET /knowledge-bases/{id}/knowledge")
    assert isinstance(body["data"], list)
    for row in body["data"]:
        _assert_payload(row, contracts.Knowledge, "GET /knowledge-bases/{id}/knowledge row")


def test_get_knowledge(knowledge_seed: KnowledgeSeed) -> None:
    spec = _endpoint_spec("GET", "/api/v1/knowledge/{id}")
    response = knowledge_seed.client.get(f"/api/v1/knowledge/{knowledge_seed.knowledge_id}")
    assert response.status_code == spec[0], response.text
    body = response.json()
    _assert_keys(body, spec[1], "GET /knowledge/{id}")
    _assert_payload(body["data"], contracts.Knowledge, "GET /knowledge/{id}")


def test_update_knowledge(knowledge_seed: KnowledgeSeed) -> None:
    spec = _endpoint_spec("PUT", "/api/v1/knowledge/{id}")
    response = knowledge_seed.client.put(
        f"/api/v1/knowledge/{knowledge_seed.knowledge_id}",
        json={"title": "renamed", "description": "updated"},
    )
    assert response.status_code == spec[0], response.text
    body = response.json()
    _assert_keys(body, spec[1], "PUT /knowledge/{id}")
    _assert_payload(
        body["data"],
        contracts.Knowledge,
        "PUT /knowledge/{id}",
    )


# ── Chunk endpoints ──────────────────────────────────────────────────


def test_list_chunks(chunk_seed: ChunkSeed) -> None:
    spec = _endpoint_spec("GET", "/api/v1/chunks/{knowledge_id}")
    response = chunk_seed.client.get(f"/api/v1/chunks/{chunk_seed.knowledge_id}")
    assert response.status_code == spec[0], response.text
    body = response.json()
    _assert_keys(body, spec[1], "GET /chunks/{knowledge_id}")
    assert isinstance(body["data"], list)
    for row in body["data"]:
        _assert_payload(row, contracts.Chunk, "GET /chunks row")


def test_get_chunk_by_id(chunk_seed: ChunkSeed) -> None:
    spec = _endpoint_spec("GET", "/api/v1/chunks/by-id/{id}")
    response = chunk_seed.client.get(f"/api/v1/chunks/by-id/{chunk_seed.chunk_id}")
    assert response.status_code == spec[0], response.text
    body = response.json()
    _assert_keys(body, spec[1], "GET /chunks/by-id/{id}")
    _assert_payload(body["data"], contracts.Chunk, "GET /chunks/by-id/{id}")


def test_update_chunk(chunk_seed: ChunkSeed) -> None:
    spec = _endpoint_spec("PUT", "/api/v1/chunks/{knowledge_id}/{id}")
    response = chunk_seed.client.put(
        f"/api/v1/chunks/{chunk_seed.knowledge_id}/{chunk_seed.chunk_id}",
        json={"content": "updated content", "is_enabled": True},
    )
    assert response.status_code == spec[0], response.text
    body = response.json()
    _assert_keys(body, spec[1], "PUT /chunks/{knowledge_id}/{id}")
    _assert_payload(body["data"], contracts.Chunk, "PUT /chunks/{knowledge_id}/{id}")


def test_delete_chunk(chunk_seed: ChunkSeed) -> None:
    spec = _endpoint_spec("DELETE", "/api/v1/chunks/{knowledge_id}/{id}")
    response = chunk_seed.client.delete(f"/api/v1/chunks/{chunk_seed.knowledge_id}/{chunk_seed.chunk_id}")
    assert response.status_code == spec[0], response.text
    body = response.json()
    _assert_keys(body, spec[1], "DELETE /chunks/{knowledge_id}/{id}")


# ── Tag endpoints ────────────────────────────────────────────────────


def test_create_tag(kb_seed: KBSeed) -> None:
    spec = _endpoint_spec("POST", "/api/v1/knowledge-bases/{id}/tags")
    response = kb_seed.client.post(
        f"/api/v1/knowledge-bases/{kb_seed.kb_id}/tags",
        json={"name": "tag-create", "color": "#1890ff", "sort_order": 2},
    )
    assert response.status_code == spec[0], response.text
    body = response.json()
    _assert_keys(body, spec[1], "POST /knowledge-bases/{id}/tags")
    _assert_payload(
        body["data"],
        contracts.Tag,
        "POST /knowledge-bases/{id}/tags",
    )


def test_list_tags(kb_seed: KBSeed) -> None:
    spec = _endpoint_spec("GET", "/api/v1/knowledge-bases/{id}/tags")
    kb_seed.client.post(
        f"/api/v1/knowledge-bases/{kb_seed.kb_id}/tags",
        json={"name": "tag-a", "color": "#1890ff", "sort_order": 1},
    )
    response = kb_seed.client.get(f"/api/v1/knowledge-bases/{kb_seed.kb_id}/tags")
    assert response.status_code == spec[0], response.text
    body = response.json()
    _assert_keys(body, spec[1], "GET /knowledge-bases/{id}/tags")
    _assert_payload(
        body["data"],
        contracts.TagList,
        "GET /knowledge-bases/{id}/tags",
    )


def test_update_tag(tag_seed: TagSeed) -> None:
    spec = _endpoint_spec("PUT", "/api/v1/knowledge-bases/{id}/tags/{tag_id}")
    response = tag_seed.client.put(
        f"/api/v1/knowledge-bases/{tag_seed.kb_id}/tags/{tag_seed.tag_id}",
        json={"name": "tag-renamed", "color": "#52c41a"},
    )
    assert response.status_code == spec[0], response.text
    body = response.json()
    _assert_keys(body, spec[1], "PUT /knowledge-bases/{id}/tags/{tag_id}")
    _assert_payload(body["data"], contracts.Tag, "PUT /knowledge-bases/{id}/tags/{tag_id}")


def test_delete_tag(tag_seed: TagSeed) -> None:
    spec = _endpoint_spec("DELETE", "/api/v1/knowledge-bases/{id}/tags/{tag_id}")
    response = tag_seed.client.delete(f"/api/v1/knowledge-bases/{tag_seed.kb_id}/tags/{tag_seed.tag_id}")
    assert response.status_code == spec[0], response.text
    body = response.json()
    _assert_keys(body, spec[1], "DELETE /knowledge-bases/{id}/tags/{tag_id}")


# ── FAQ endpoints ────────────────────────────────────────────────────


def test_list_faq_entries(faq_seed: FAQSeed) -> None:
    spec = _endpoint_spec("GET", "/api/v1/knowledge-bases/{id}/faq/entries")
    response = faq_seed.client.get(f"/api/v1/knowledge-bases/{faq_seed.kb_id}/faq/entries")
    assert response.status_code == spec[0], response.text
    body = response.json()
    _assert_keys(body, spec[1], "GET /knowledge-bases/{id}/faq/entries")
    _assert_payload(
        body["data"],
        contracts.FAQEntryListResponse,
        "GET /knowledge-bases/{id}/faq/entries",
    )


def test_create_faq_entry(faq_seed: FAQSeed) -> None:
    spec = _endpoint_spec("POST", "/api/v1/knowledge-bases/{id}/faq/entry")
    response = faq_seed.client.post(
        f"/api/v1/knowledge-bases/{faq_seed.kb_id}/faq/entry",
        json={
            "standard_question": "how to install?",
            "answers": ["run the installer"],
            "is_enabled": True,
        },
    )
    assert response.status_code == spec[0], response.text
    body = response.json()
    _assert_keys(body, spec[1], "POST /knowledge-bases/{id}/faq/entry")
    _assert_payload(
        body["data"],
        contracts.FAQEntry,
        "POST /knowledge-bases/{id}/faq/entry",
    )


def test_get_faq_entry(faq_seed: FAQSeed) -> None:
    spec = _endpoint_spec("GET", "/api/v1/knowledge-bases/{id}/faq/entries/{entry_id}")
    response = faq_seed.client.get(
        f"/api/v1/knowledge-bases/{faq_seed.kb_id}/faq/entries/{faq_seed.entry_id}"
    )
    assert response.status_code == spec[0], response.text
    body = response.json()
    _assert_keys(body, spec[1], "GET /knowledge-bases/{id}/faq/entries/{entry_id}")
    _assert_payload(
        body["data"],
        contracts.FAQEntry,
        "GET /knowledge-bases/{id}/faq/entries/{entry_id}",
    )


# ── Wiki endpoints ───────────────────────────────────────────────────


def test_wiki_payloads_conform_to_view_models(wiki_seed: WikiSeed) -> None:
    """The wiki ``data`` payloads match the view-model field sets.

    Kept separate from the envelope assertion below so the inner shapes
    are verified even if the envelope comparison reports a deviation.
    """
    client, kb_id = wiki_seed.client, wiki_seed.kb_id
    base = f"/api/v1/knowledgebase/{kb_id}/wiki"

    response = client.get(f"{base}/pages")
    assert response.status_code == 200, response.text
    _assert_payload(
        response.json()["data"],
        wiki_views.WikiPageListData,
        "GET /api/v1/knowledgebase/{kb_id}/wiki/pages",
    )

    response = client.get(f"{base}/folders")
    assert response.status_code == 200, response.text
    _assert_payload(
        response.json()["data"],
        wiki_views.WikiFolderListData,
        "GET /api/v1/knowledgebase/{kb_id}/wiki/folders",
    )

    response = client.get(f"{base}/stats")
    assert response.status_code == 200, response.text
    _assert_payload(
        response.json()["data"],
        wiki_views.WikiStats,
        "GET /api/v1/knowledgebase/{kb_id}/wiki/stats",
    )

    response = client.get(f"{base}/search", params={"q": "hello"})
    assert response.status_code == 200, response.text
    _assert_payload(
        response.json()["data"],
        wiki_views.WikiSearchData,
        "GET /api/v1/knowledgebase/{kb_id}/wiki/search",
    )

    response = client.post(f"{base}/folders", json={"name": "dir-a"})
    assert response.status_code == 201, response.text
    _assert_payload(
        response.json()["data"],
        wiki_views.WikiFolderView,
        "POST /api/v1/knowledgebase/{kb_id}/wiki/folders",
    )


def test_wiki_response_envelopes_match_reference(wiki_seed: WikiSeed) -> None:
    """The wiki responses carry the reference top-level key set.

    The reference returns the wiki object directly (no wrapper); the
    current web layer wraps every wiki response in the shared
    ``{"success": true, "data": ...}`` envelope. This assertion is the
    contract check — a divergence here is a reported finding.
    """
    client, kb_id = wiki_seed.client, wiki_seed.kb_id
    base = f"/api/v1/knowledgebase/{kb_id}/wiki"

    probes = (
        ("GET", f"{base}/pages", "GET /api/v1/knowledgebase/{kb_id}/wiki/pages"),
        ("GET", f"{base}/folders", "GET /api/v1/knowledgebase/{kb_id}/wiki/folders"),
        ("GET", f"{base}/stats", "GET /api/v1/knowledgebase/{kb_id}/wiki/stats"),
        ("GET", f"{base}/search", "GET /api/v1/knowledgebase/{kb_id}/wiki/search"),
        ("POST", f"{base}/folders", "POST /api/v1/knowledgebase/{kb_id}/wiki/folders"),
    )
    for method, url, label in probes:
        response = client.request(method, url, json={"name": "dir-b"} if method == "POST" else None)
        _, path = label.split(" ", 1)
        assert response.status_code == _endpoint_spec(method, path)[0], response.text
        _assert_keys(response.json(), _endpoint_spec(method, path)[1], label)


__all__ = [
    "ChunkSeed",
    "FAQSeed",
    "KBSeed",
    "KnowledgeSeed",
    "TagSeed",
    "WikiSeed",
    "chunk_seed",
    "faq_seed",
    "kb_seed",
    "knowledge_seed",
    "tag_seed",
    "wiki_seed",
]
