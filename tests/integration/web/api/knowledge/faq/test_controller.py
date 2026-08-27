"""Live e2e tests for the FAQ router.

Exercises the full HTTP path over ``TestClient`` against the real app:
routing, serialization, role gates, exception handling, the multipart
import, and the file export. The FAQ / knowledge service dependencies are
overridden with stateful fakes, so the tests run without a database.

Endpoint coverage:

| Method | Path                                         |
| ------ | -------------------------------------------- |
| GET    | /knowledge-bases/{id}/faq/entries            |
| POST   | /knowledge-bases/{id}/faq/entries            |
| POST   | /knowledge-bases/{id}/faq/entry              |
| PUT    | /knowledge-bases/{id}/faq/entries/{entry_id} |
| DELETE | /knowledge-bases/{id}/faq/entries            |
| GET    | /knowledge-bases/{id}/faq/entries/{entry_id} |
| GET    | /knowledge-bases/{id}/faq/entries/export     |
| GET    | /faq/import/progress/{task_id}               |

Auth: the authed client carries the ``X-User-Id/X-Tenant-ID/X-Roles`` header trio; the
unauthorised tests build a bare ``TestClient`` and assert the 401 raised
by the global ``require_auth`` dependency.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.common.exception import NotFoundError
from src.core.contracts.knowledge import (
    FAQEntry,
    FAQEntryListResponse,
    FAQEntryPayload,
    FAQImportTaskProgress,
    Knowledge,
)
from src.core.knowledge.faq.task_ids import generate_task_id
from src.web.deps.knowledge import get_knowledge_service
from src.web.deps.knowledge_faq import get_faq_import_runner, get_faq_service

_KB_ID = "kb-1"
_FAQ_CSV = (
    "分类(必填),问题(必填),相似问题(选填-多个用##分隔),反例问题(选填-多个用##分隔),"
    "机器人回答(必填-多个用##分隔),是否全部回复(选填-默认FALSE),是否停用(选填-默认FALSE),"
    "是否禁止被推荐(选填-默认False 可被推荐)\n"
    "分类一,标准问一,,,答案一##答案二,TRUE,FALSE,FALSE\n"
).encode()


# ── Fakes backing the dep overrides ──────────────────────────────────


class _FakeFAQService:
    """Stateful in-memory replacement for ``FAQService``.

    Each method records its arguments for assertion and returns entries
    shaped like the real service's wire objects, so the view layer runs
    unmodified.
    """

    def __init__(self) -> None:
        self.rows: dict[int, FAQEntry] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._id_seq = itertools.count(1)

    def seed(self, *entries: FAQEntry) -> None:
        """Insert pre-existing entries."""
        for entry in entries:
            self.rows[entry.id] = entry

    def _record(self, method: str, **kwargs: Any) -> None:
        self.calls.append((method, kwargs))

    async def list_entries(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        keyword: str | None = None,
        limit: int,
        offset: int,
    ) -> FAQEntryListResponse:
        self._record(
            "list_entries",
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            keyword=keyword,
            limit=limit,
            offset=offset,
        )
        rows = [r for r in self.rows.values() if r.knowledge_base_id == knowledge_base_id]
        page = (offset // limit) + 1 if limit > 0 else 1
        return FAQEntryListResponse(
            total=len(rows),
            page=page,
            page_size=limit,
            data=rows[offset : offset + limit],
        )

    async def get_entry(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        entry_id: int,
    ) -> FAQEntry:
        self._record(
            "get_entry",
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            entry_id=entry_id,
        )
        row = self.rows.get(entry_id)
        if row is None or row.knowledge_base_id != knowledge_base_id:
            raise NotFoundError(code="faq.not_found", message="FAQ条目不存在")
        return row

    async def create_entry(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        knowledge_id: str,
        payload: FAQEntryPayload,
        chunk_id: str | None = None,
        index_mode: str | None = None,
    ) -> FAQEntry:
        self._record(
            "create_entry",
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            knowledge_id=knowledge_id,
        )
        new_id = next(self._id_seq)
        now = datetime.now(UTC)
        row = _entry(
            id=new_id,
            knowledge_base_id=knowledge_base_id,
            knowledge_id=knowledge_id,
            standard_question=payload.standard_question,
            similar_questions=list(payload.similar_questions or []),
            negative_questions=list(payload.negative_questions or []),
            answers=list(payload.answers or []),
            answer_strategy=payload.answer_strategy or "all",
            tag_name=payload.tag_name,
            is_enabled=payload.is_enabled if payload.is_enabled is not None else True,
            is_recommended=(
                payload.is_recommended if payload.is_recommended is not None else False
            ),
            chunk_id=chunk_id or f"chunk-{new_id}",
            created_at=now,
            updated_at=now,
        )
        self.rows[row.id] = row
        return row

    async def update_entry(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        entry_id: int,
        payload: FAQEntryPayload,
    ) -> FAQEntry:
        self._record(
            "update_entry",
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            entry_id=entry_id,
        )
        existing = self.rows.get(entry_id)
        if existing is None or existing.knowledge_base_id != knowledge_base_id:
            raise NotFoundError(code="faq.not_found", message="FAQ条目不存在")
        updated = existing.model_copy(
            update={
                "standard_question": payload.standard_question,
                "similar_questions": list(payload.similar_questions or []),
                "negative_questions": list(payload.negative_questions or []),
                "answers": list(payload.answers or []),
                "answer_strategy": payload.answer_strategy or "all",
                "tag_id": payload.tag_id,
                "tag_name": payload.tag_name,
                "is_enabled": (
                    payload.is_enabled if payload.is_enabled is not None else existing.is_enabled
                ),
                "is_recommended": (
                    payload.is_recommended
                    if payload.is_recommended is not None
                    else existing.is_recommended
                ),
                "updated_at": datetime.now(UTC),
            }
        )
        self.rows[entry_id] = updated
        return updated

    async def delete_entries(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        entry_ids: list[int],
    ) -> int:
        self._record(
            "delete_entries",
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            entry_ids=list(entry_ids),
        )
        for entry_id in entry_ids:
            row = self.rows.get(entry_id)
            if row is None or row.knowledge_base_id != knowledge_base_id:
                raise NotFoundError(code="faq.not_found", message="FAQ条目不存在")
        for entry_id in entry_ids:
            self.rows.pop(entry_id, None)
        return len(entry_ids)


class _FakeFAQImportRunner:
    """In-memory FAQ import runner; stores completed progress by task id."""

    def __init__(self) -> None:
        self.tasks: dict[str, FAQImportTaskProgress] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _record(self, method: str, **kwargs: Any) -> None:
        self.calls.append((method, kwargs))

    async def run(
        self,
        *,
        file_data: bytes,
        filename: str,
        tenant_id: int,
        knowledge_base_id: str,
        knowledge_id: str,
        mode: str | None = None,
        dry_run: bool = False,
    ) -> FAQImportTaskProgress:
        self._record(
            "run",
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            knowledge_id=knowledge_id,
            mode=mode,
            dry_run=dry_run,
        )
        task_id = generate_task_id(tenant_id=tenant_id)
        now = int(datetime.now(UTC).timestamp())
        progress = FAQImportTaskProgress(
            task_id=task_id,
            kb_id=knowledge_base_id,
            knowledge_id=knowledge_id,
            status="completed",
            progress=100,
            total=1,
            processed=1,
            success_count=1,
            failed_count=0,
            skipped_count=0,
            created_at=now,
            updated_at=now,
            dry_run=dry_run,
            import_mode=mode or "append",
        )
        self.tasks[task_id] = progress
        return progress

    def get_progress(self, task_id: str) -> FAQImportTaskProgress | None:
        return self.tasks.get(task_id)


class _FakeKnowledgeService:
    """Serves one FAQ container document for a knowledge base."""

    def __init__(self) -> None:
        self.faq_container_id = "knowledge-faq-1"
        self.has_container = True

    async def list_documents(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
    ) -> list[Knowledge]:
        if not self.has_container:
            return []
        now = datetime.now(UTC)
        return [
            Knowledge(
                id=self.faq_container_id,
                tenant_id=tenant_id,
                knowledge_base_id=knowledge_base_id,
                type="faq",
                title="FAQ",
                parse_status="done",
                enable_status="enabled",
                created_at=now,
                updated_at=now,
            )
        ]


def _entry(
    id: int,
    standard_question: str,
    answers: list[str] | None = None,
    **overrides: Any,
) -> FAQEntry:
    """Build a wire FAQ entry with sensible defaults."""
    now = datetime.now(UTC)
    fields: dict[str, Any] = {
        "id": id,
        "chunk_id": f"chunk-{id}",
        "knowledge_id": "knowledge-faq-1",
        "knowledge_base_id": _KB_ID,
        "is_enabled": True,
        "is_recommended": False,
        "standard_question": standard_question,
        "similar_questions": [],
        "negative_questions": [],
        "answers": answers or ["答案"],
        "answer_strategy": "all",
        "chunk_type": "faq",
        "created_at": now,
        "updated_at": now,
    }
    fields.update(overrides)
    return FAQEntry(**fields)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def fake_faq_service() -> _FakeFAQService:
    return _FakeFAQService()


@pytest.fixture
def fake_import_runner() -> _FakeFAQImportRunner:
    return _FakeFAQImportRunner()


@pytest.fixture
def fake_knowledge_service() -> _FakeKnowledgeService:
    return _FakeKnowledgeService()


@pytest.fixture
def app(
    web_app: FastAPI,
    fake_faq_service: _FakeFAQService,
    fake_import_runner: _FakeFAQImportRunner,
    fake_knowledge_service: _FakeKnowledgeService,
) -> FastAPI:
    """Override the FAQ / knowledge service factories with the fakes."""
    web_app.dependency_overrides[get_faq_service] = lambda: fake_faq_service
    web_app.dependency_overrides[get_faq_import_runner] = lambda: fake_import_runner
    web_app.dependency_overrides[get_knowledge_service] = lambda: fake_knowledge_service
    return web_app


@pytest.fixture
def client(app: FastAPI, web_authed_client: TestClient) -> TestClient:
    return web_authed_client


@pytest.fixture
def anon_client(app: FastAPI) -> Iterator[TestClient]:
    """A ``TestClient`` without the auth header trio — 401 surface."""
    with TestClient(app=app) as c:
        yield c


# ── GET /knowledge-bases/{id}/faq/entries ────────────────────────────


async def test_list_returns_page_in_envelope(
    client: TestClient,
    fake_faq_service: _FakeFAQService,
) -> None:
    """The list endpoint returns the paginated entries inside the envelope."""
    fake_faq_service.seed(_entry(1, "问题一"), _entry(2, "问题二"))
    resp = client.get(f"/api/v1/knowledge-bases/{_KB_ID}/faq/entries?page=1&page_size=10")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["total"] == 2
    assert body["data"]["page"] == 1
    assert body["data"]["page_size"] == 10
    assert {row["standard_question"] for row in body["data"]["data"]} == {"问题一", "问题二"}


async def test_list_passes_keyword_and_pagination(
    client: TestClient,
    fake_faq_service: _FakeFAQService,
) -> None:
    """The keyword / page / page_size query params reach the service."""
    client.get(f"/api/v1/knowledge-bases/{_KB_ID}/faq/entries?keyword=密码&page=2&page_size=5")
    method, kwargs = fake_faq_service.calls[-1]
    assert method == "list_entries"
    assert kwargs["keyword"] == "密码"
    assert kwargs["limit"] == 5
    assert kwargs["offset"] == 5


async def test_list_rejects_unsupported_tag_filter(client: TestClient) -> None:
    """A non-empty ``tag_id`` filter is refused rather than silently dropped."""
    resp = client.get(f"/api/v1/knowledge-bases/{_KB_ID}/faq/entries?tag_id=3")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "faq.tag_filter_unsupported"


async def test_list_rejects_unsupported_search_field(client: TestClient) -> None:
    """A non-empty ``search_field`` is refused by the merged service."""
    resp = client.get(f"/api/v1/knowledge-bases/{_KB_ID}/faq/entries?search_field=answers")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "faq.search_field_unsupported"


async def test_list_rejects_invalid_page_size(client: TestClient) -> None:
    """A page size outside the pagination bounds is a 422."""
    resp = client.get(f"/api/v1/knowledge-bases/{_KB_ID}/faq/entries?page_size=0")
    assert resp.status_code == 422


# ── GET /knowledge-bases/{id}/faq/entries/{entry_id} ─────────────────


async def test_get_returns_entry_in_envelope(
    client: TestClient,
    fake_faq_service: _FakeFAQService,
) -> None:
    """The get endpoint returns the requested entry."""
    fake_faq_service.seed(_entry(7, "问题七", answers=["答案七"]))
    resp = client.get(f"/api/v1/knowledge-bases/{_KB_ID}/faq/entries/7")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["id"] == 7
    assert body["data"]["standard_question"] == "问题七"


async def test_get_unknown_entry_returns_404(client: TestClient) -> None:
    """An unknown entry id reads as not-found."""
    resp = client.get(f"/api/v1/knowledge-bases/{_KB_ID}/faq/entries/999")
    assert resp.status_code == 404
    body = resp.json()
    assert body["success"] is False
    assert body["error"]["code"] == "faq.not_found"


# ── POST /knowledge-bases/{id}/faq/entry ─────────────────────────────


async def test_create_returns_created_entry(
    client: TestClient,
    default_create_faq_request: dict[str, object],
    fake_faq_service: _FakeFAQService,
) -> None:
    """A valid body creates the entry under the resolved FAQ container."""
    resp = client.post(
        f"/api/v1/knowledge-bases/{_KB_ID}/faq/entry", json=default_create_faq_request
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["standard_question"] == "测试问题"
    assert body["data"]["knowledge_id"] == "knowledge-faq-1"
    assert body["data"]["knowledge_base_id"] == _KB_ID


async def test_create_resolves_faq_container(
    client: TestClient,
    default_create_faq_request: dict[str, object],
    fake_faq_service: _FakeFAQService,
) -> None:
    """The view resolves the FAQ container before calling the service."""
    client.post(f"/api/v1/knowledge-bases/{_KB_ID}/faq/entry", json=default_create_faq_request)
    method, kwargs = fake_faq_service.calls[-1]
    assert method == "create_entry"
    assert kwargs["knowledge_id"] == "knowledge-faq-1"


async def test_create_without_faq_container_returns_422(
    client: TestClient,
    default_create_faq_request: dict[str, object],
    fake_knowledge_service: _FakeKnowledgeService,
) -> None:
    """A knowledge base with no FAQ document cannot take a write."""
    fake_knowledge_service.has_container = False
    resp = client.post(
        f"/api/v1/knowledge-bases/{_KB_ID}/faq/entry", json=default_create_faq_request
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "faq.knowledge_container_missing"


async def test_create_rejects_missing_standard_question(client: TestClient) -> None:
    """A body without ``standard_question`` is rejected by validation."""
    resp = client.post(f"/api/v1/knowledge-bases/{_KB_ID}/faq/entry", json={"answers": ["答案"]})
    assert resp.status_code == 422


# ── PUT /knowledge-bases/{id}/faq/entries/{entry_id} ─────────────────


async def test_update_returns_updated_entry(
    client: TestClient,
    fake_faq_service: _FakeFAQService,
) -> None:
    """The put endpoint mutates the entry's content."""
    fake_faq_service.seed(_entry(5, "旧问题", answers=["旧答案"]))
    resp = client.put(
        f"/api/v1/knowledge-bases/{_KB_ID}/faq/entries/5",
        json={"standard_question": "新问题", "answers": ["新答案"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["standard_question"] == "新问题"


async def test_update_unknown_entry_returns_404(client: TestClient) -> None:
    """Updating an unknown id yields not-found."""
    resp = client.put(
        f"/api/v1/knowledge-bases/{_KB_ID}/faq/entries/99",
        json={"standard_question": "问题", "answers": ["答案"]},
    )
    assert resp.status_code == 404


# ── DELETE /knowledge-bases/{id}/faq/entries ─────────────────────────


async def test_delete_returns_ack(
    client: TestClient,
    fake_faq_service: _FakeFAQService,
) -> None:
    """A successful batch delete returns the success ack."""
    fake_faq_service.seed(_entry(1, "问题一"), _entry(2, "问题二"))
    resp = client.request(
        "DELETE", f"/api/v1/knowledge-bases/{_KB_ID}/faq/entries", json={"ids": [1, 2]}
    )
    assert resp.status_code == 200
    assert resp.json() == {"success": True}
    assert 1 not in fake_faq_service.rows
    assert 2 not in fake_faq_service.rows


async def test_delete_with_unknown_id_returns_404(client: TestClient) -> None:
    """A batch containing a foreign or unknown id fails the whole batch."""
    resp = client.request(
        "DELETE", f"/api/v1/knowledge-bases/{_KB_ID}/faq/entries", json={"ids": [1, 2]}
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "faq.not_found"


async def test_delete_rejects_missing_ids(client: TestClient) -> None:
    """A body without ``ids`` is rejected by validation."""
    resp = client.request("DELETE", f"/api/v1/knowledge-bases/{_KB_ID}/faq/entries", json={})
    assert resp.status_code == 422


# ── POST /knowledge-bases/{id}/faq/entries (import) ──────────────────


async def test_import_runs_pipeline_and_returns_progress(client: TestClient) -> None:
    """A file upload runs the import and returns a completed progress."""
    resp = client.post(
        f"/api/v1/knowledge-bases/{_KB_ID}/faq/entries",
        files={"file": ("faq.csv", _FAQ_CSV, "text/csv")},
        data={"mode": "append"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["status"] == "completed"
    assert body["data"]["import_mode"] == "append"
    assert body["data"]["dry_run"] is False
    assert body["data"]["kb_id"] == _KB_ID


async def test_import_dry_run_is_reported(
    client: TestClient,
    fake_import_runner: _FakeFAQImportRunner,
) -> None:
    """The ``dry_run`` switch is forwarded to the import pipeline."""
    resp = client.post(
        f"/api/v1/knowledge-bases/{_KB_ID}/faq/entries",
        files={"file": ("faq.csv", _FAQ_CSV, "text/csv")},
        data={"dry_run": "true"},
    )
    assert resp.status_code == 200
    assert resp.json()["data"]["dry_run"] is True
    method, kwargs = fake_import_runner.calls[-1]
    assert method == "run"
    assert kwargs["dry_run"] is True


async def test_import_rejects_invalid_mode(client: TestClient) -> None:
    """A mode outside append / replace is refused."""
    resp = client.post(
        f"/api/v1/knowledge-bases/{_KB_ID}/faq/entries",
        files={"file": ("faq.csv", _FAQ_CSV, "text/csv")},
        data={"mode": "merge"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "faq.invalid_import_mode"


async def test_import_without_container_returns_422(
    client: TestClient,
    fake_knowledge_service: _FakeKnowledgeService,
) -> None:
    """Import needs a FAQ container to persist into."""
    fake_knowledge_service.has_container = False
    resp = client.post(
        f"/api/v1/knowledge-bases/{_KB_ID}/faq/entries",
        files={"file": ("faq.csv", _FAQ_CSV, "text/csv")},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "faq.knowledge_container_missing"


async def test_import_requires_file(client: TestClient) -> None:
    """The import endpoint demands a file part."""
    resp = client.post(f"/api/v1/knowledge-bases/{_KB_ID}/faq/entries")
    assert resp.status_code == 422


# ── GET /knowledge-bases/{id}/faq/entries/export ─────────────────────


async def test_export_csv_matches_import_template(
    client: TestClient,
    fake_faq_service: _FakeFAQService,
) -> None:
    """The CSV export carries the template header and one entry row."""
    fake_faq_service.seed(
        _entry(1, "标准问一", answers=["答案一"], tag_name="分类一", similar_questions=["相似问"])
    )
    resp = client.get(f"/api/v1/knowledge-bases/{_KB_ID}/faq/entries/export")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert "attachment; filename=faq_export.csv" in resp.headers["content-disposition"]
    text = resp.content.decode("utf-8-sig")
    assert "问题(必填)" in text
    assert "标准问一" in text
    assert "相似问" in text
    assert "TRUE" in text


async def test_export_json_returns_payload_array(
    client: TestClient,
    fake_faq_service: _FakeFAQService,
) -> None:
    """The JSON export is an array of payload-compatible objects."""
    fake_faq_service.seed(
        _entry(1, "标准问一", answers=["答案一"], tag_name="分类一", is_recommended=True)
    )
    resp = client.get(f"/api/v1/knowledge-bases/{_KB_ID}/faq/entries/export?format=json")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    data = resp.json()
    assert data[0]["standard_question"] == "标准问一"
    assert data[0]["answers"] == ["答案一"]
    assert data[0]["is_enabled"] is True
    assert data[0]["is_recommended"] is True


async def test_export_empty_knowledge_base_returns_header_only(
    client: TestClient,
) -> None:
    """An empty knowledge base exports the CSV header row only."""
    resp = client.get(f"/api/v1/knowledge-bases/{_KB_ID}/faq/entries/export")
    assert resp.status_code == 200
    text = resp.content.decode("utf-8-sig")
    assert text.strip().endswith("可被推荐)")


async def test_export_rejects_unknown_format(client: TestClient) -> None:
    """An unknown export format is refused."""
    resp = client.get(f"/api/v1/knowledge-bases/{_KB_ID}/faq/entries/export?format=xml")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "faq.invalid_export_format"


# ── GET /faq/import/progress/{task_id} ───────────────────────────────


async def test_import_progress_roundtrip(client: TestClient) -> None:
    """An import task started in this workspace can be polled back."""
    started = client.post(
        f"/api/v1/knowledge-bases/{_KB_ID}/faq/entries",
        files={"file": ("faq.csv", _FAQ_CSV, "text/csv")},
    )
    task_id = started.json()["data"]["task_id"]

    resp = client.get(f"/api/v1/faq/import/progress/{task_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["task_id"] == task_id
    assert body["data"]["status"] == "completed"
    assert body["data"]["progress"] == 100


async def test_import_progress_unknown_task_returns_404(
    client: TestClient,
    admin_user: tuple[int, int],
) -> None:
    """A well-formed task id that was never started reads as not-found."""
    _user_id, tenant_id = admin_user
    task_id = generate_task_id(tenant_id=tenant_id)
    resp = client.get(f"/api/v1/faq/import/progress/{task_id}")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "faq.import_task_not_found"


async def test_import_progress_cross_tenant_returns_404(client: TestClient) -> None:
    """A task from another workspace is hidden as not-found."""
    task_id = generate_task_id(tenant_id=999999)
    resp = client.get(f"/api/v1/faq/import/progress/{task_id}")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "faq.task_not_found"


async def test_import_progress_invalid_task_id_returns_422(client: TestClient) -> None:
    """A task id that cannot carry a tenant is a client error."""
    resp = client.get("/api/v1/faq/import/progress/not-a-task-id")
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "faq.invalid_task_id"


# ── Auth gate ────────────────────────────────────────────────────────


async def test_unauthed_read_returns_401(anon_client: TestClient) -> None:
    """A request without the header trio is rejected by the auth gate."""
    resp = anon_client.get(f"/api/v1/knowledge-bases/{_KB_ID}/faq/entries")
    assert resp.status_code == 401


async def test_unauthed_write_returns_401(anon_client: TestClient) -> None:
    """Writes also require the header trio."""
    resp = anon_client.post(
        f"/api/v1/knowledge-bases/{_KB_ID}/faq/entry",
        json={"standard_question": "问题", "answers": ["答案"]},
    )
    assert resp.status_code == 401


__all__ = [
    "_FakeFAQImportRunner",
    "_FakeFAQService",
    "_FakeKnowledgeService",
    "anon_client",
    "app",
    "client",
    "fake_faq_service",
    "fake_import_runner",
    "fake_knowledge_service",
]
