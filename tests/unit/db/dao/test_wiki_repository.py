"""Unit tests for :mod:`src.db.dao.wiki_page_repository`.

Non-DB tests: exercise the generated SQL text (via a stub session that
records statements) so the optimistic version guard, the listing filters,
and the source-ref / folder predicates stay pinned without a database.
The real SQL round-trip is covered by
``tests/integration/db/dao/test_wiki_repository.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.sql.expression import TextClause

from src.db.dao.wiki_page_repository import WikiFolderRepository, WikiPageRepository
from src.db.models.wiki_page import WikiFolder, WikiPage

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _page_row(
    *,
    id: str = "page-1",
    tenant_id: int = 1,
    knowledge_base_id: str = "kb-1",
    slug: str = "entity/acme",
    version: int = 1,
    **overrides: object,
) -> dict[str, object]:
    row = {
        "id": id,
        "tenant_id": tenant_id,
        "knowledge_base_id": knowledge_base_id,
        "slug": slug,
        "title": "Acme Corp",
        "page_type": "entity",
        "status": "published",
        "content": "body",
        "summary": "summary",
        "parent_slug": "",
        "folder_id": "",
        "category_path": [],
        "wiki_path": "entity/Acme Corp",
        "depth": 0,
        "sort_order": 0,
        "source_refs": [],
        "chunk_refs": [],
        "in_links": [],
        "out_links": [],
        "page_metadata": {},
        "aliases": [],
        "version": version,
        "last_edit_source": "",
        "last_editor_id": "",
        "created_at": _NOW,
        "updated_at": _NOW,
        "deleted_at": None,
    }
    row.update(overrides)
    return row


def _folder_row(
    *,
    id: str = "folder-1",
    tenant_id: int = 1,
    knowledge_base_id: str = "kb-1",
    parent_id: str = "",
    name: str = "AI",
    **overrides: object,
) -> dict[str, object]:
    row = {
        "id": id,
        "tenant_id": tenant_id,
        "knowledge_base_id": knowledge_base_id,
        "parent_id": parent_id,
        "name": name,
        "path": "AI",
        "depth": 1,
        "sort_order": 0,
        "created_at": _NOW,
        "updated_at": _NOW,
        "deleted_at": None,
    }
    row.update(overrides)
    return row


class _FakeMappings:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def all(self) -> list[dict[str, object]]:
        return self._rows

    def first(self) -> dict[str, object] | None:
        return self._rows[0] if self._rows else None


class _FakeScalars:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def all(self) -> list[object]:
        return [r["slug"] if "slug" in r else r["path"] for r in self._rows]


class _FakeResult:
    def __init__(
        self,
        rows: list[dict[str, object]],
        scalar: object = 0,
        rowcount: int = 1,
    ) -> None:
        self._rows = rows
        self._scalar = scalar
        self.rowcount = rowcount

    def mappings(self) -> _FakeMappings:
        return _FakeMappings(self._rows)

    def scalars(self) -> _FakeScalars:
        return _FakeScalars(self._rows)

    def scalar_one(self) -> object:
        return self._scalar


class _FakeSession:
    """Records executed SQL and serves canned rows keyed by SQL prefix."""

    def __init__(self, rows_by_prefix: dict[str, list[dict[str, object]]]) -> None:
        self.executed: list[str] = []
        self._rows_by_prefix = rows_by_prefix

    async def execute(self, stmt: TextClause) -> _FakeResult:
        sql = stmt.text
        self.executed.append(sql)
        for prefix, rows in self._rows_by_prefix.items():
            if sql.lstrip().startswith(prefix):
                return _FakeResult(rows)
        return _FakeResult([])


def _page_repo(session: _FakeSession) -> WikiPageRepository:
    return WikiPageRepository(session)  # type: ignore[arg-type]


def _folder_repo(session: _FakeSession) -> WikiFolderRepository:
    return WikiFolderRepository(session)  # type: ignore[arg-type]


def _sample_page(*, id: str = "page-1") -> WikiPage:
    return WikiPage.model_validate(_page_row(id=id))


def _sample_folder(*, id: str = "folder-1") -> WikiFolder:
    return WikiFolder.model_validate(_folder_row(id=id))


# ── insert column list ───────────────────────────────────────────────


def test_insert_sql_column_list_includes_caller_assigned_id() -> None:
    assert "id" in WikiPage.insert_sql_column_list()
    assert "slug" in WikiPage.insert_sql_column_list()
    assert "version" in WikiPage.insert_sql_column_list()


def test_wiki_folder_insert_includes_id() -> None:
    assert "id" in WikiFolder.insert_sql_column_list()
    assert "parent_id" in WikiFolder.insert_sql_column_list()


# ── update (optimistic version guard) ────────────────────────────────


async def test_update_issues_version_guarded_write() -> None:
    session = _FakeSession({"update wiki_pages": [_page_row(version=2)]})
    repo = _page_repo(session)

    result = await repo.update(row=_sample_page(), now=_NOW)

    assert result.version == 2
    update_sql = [s for s in session.executed if s.lstrip().startswith("update wiki_pages")]
    assert len(update_sql) == 1
    assert "version = :expected_version" in update_sql[0]
    assert "version = :version" in update_sql[0]
    assert "where id = :id" in update_sql[0]
    assert "deleted_at is null" in update_sql[0]


# ── list_pages filters ───────────────────────────────────────────────


async def test_list_pages_filters_by_page_type_list() -> None:
    session = _FakeSession({})
    repo = _page_repo(session)

    await repo.list_pages(
        knowledge_base_id="kb-1",
        page_types=["entity", "concept"],
    )

    select_sql = [s for s in session.executed if s.lstrip().startswith("select")]
    assert len(select_sql) == 2  # count + page
    page_sql = select_sql[1]
    assert "page_type in (:pt_0, :pt_1)" in page_sql
    assert "limit :limit offset :offset" in page_sql


async def test_list_pages_filters_by_single_page_type() -> None:
    session = _FakeSession({})
    repo = _page_repo(session)

    await repo.list_pages(knowledge_base_id="kb-1", page_types=["entity"])

    select_sql = [s for s in session.executed if s.lstrip().startswith("select")]
    assert "page_type = :page_type" in select_sql[1]
    assert "page_type in" not in select_sql[1]


async def test_list_pages_filters_by_category_path_json() -> None:
    session = _FakeSession({})
    repo = _page_repo(session)

    await repo.list_pages(knowledge_base_id="kb-1", category_path=["AI", "RAG"])

    select_sql = [s for s in session.executed if s.lstrip().startswith("select")]
    assert "category_path = :category_path" in select_sql[0]
    assert "category_path = :category_path" in select_sql[1]


async def test_list_pages_orders_by_wiki_path_rank() -> None:
    session = _FakeSession({})
    repo = _page_repo(session)

    await repo.list_pages(knowledge_base_id="kb-1", sort_by="wiki_path", sort_order="asc")

    select_sql = [s for s in session.executed if s.lstrip().startswith("select")]
    assert "jsonb_array_length(category_path)" in select_sql[1]
    assert "wiki_path asc" in select_sql[1]


async def test_list_pages_defaults_sort_to_updated_at_desc() -> None:
    session = _FakeSession({})
    repo = _page_repo(session)

    await repo.list_pages(knowledge_base_id="kb-1", sort_by="nonsense")

    select_sql = [s for s in session.executed if s.lstrip().startswith("select")]
    assert "order by updated_at desc" in select_sql[1]


# ── list_by_source_ref ───────────────────────────────────────────────


async def test_list_by_source_ref_builds_containment_and_like_predicate() -> None:
    session = _FakeSession({})
    repo = _page_repo(session)

    await repo.list_by_source_ref(knowledge_base_id="kb-1", source_knowledge_id="kid-1")

    select_sql = session.executed[0]
    assert "source_refs @> cast(:needle as jsonb)" in select_sql
    assert "source_refs::text like :pattern" in select_sql
    assert "knowledge_base_id = :kb_id" in select_sql


async def test_list_slugs_by_source_ref_projects_slug() -> None:
    session = _FakeSession({"select slug": [_page_row()]})
    repo = _page_repo(session)

    slugs = await repo.list_slugs_by_source_ref(
        knowledge_base_id="kb-1", source_knowledge_id="kid-1"
    )

    assert slugs == ["entity/acme"]
    assert "select slug" in session.executed[0]


# ── list_by_slugs ────────────────────────────────────────────────────


async def test_list_by_slugs_empty_input_skips_sql() -> None:
    session = _FakeSession({})
    repo = _page_repo(session)

    assert await repo.list_by_slugs(knowledge_base_id="kb-1", slugs=[]) == {}
    assert session.executed == []


async def test_list_by_slugs_returns_lite_map() -> None:
    session = _FakeSession(
        {
            "select slug": [
                {
                    "slug": "entity/acme",
                    "title": "Acme",
                    "page_type": "entity",
                    "status": "published",
                    "aliases": ["ACME"],
                    "out_links": [],
                }
            ]
        }
    )
    repo = _page_repo(session)

    result = await repo.list_by_slugs(knowledge_base_id="kb-1", slugs=["entity/acme"])

    assert set(result) == {"entity/acme"}
    assert result["entity/acme"].title == "Acme"
    assert result["entity/acme"].aliases == ["ACME"]


# ── folder delete (atomic emptiness) ─────────────────────────────────


async def test_folder_delete_includes_emptiness_subqueries() -> None:
    session = _FakeSession({"update wiki_folders": [_folder_row()]})
    repo = _folder_repo(session)

    await repo.delete(knowledge_base_id="kb-1", id="folder-1", now=_NOW)

    update_sql = session.executed[0]
    assert "not exists (select 1 from wiki_pages" in update_sql
    assert "not exists (select 1 from wiki_folders as child" in update_sql
    assert "deleted_at = :now" in update_sql


async def test_folder_update_rewrites_mutable_columns() -> None:
    session = _FakeSession({"update wiki_folders": [_folder_row(name="LLM")]})
    repo = _folder_repo(session)

    result = await repo.update(row=_sample_folder(), now=_NOW)

    assert result.name == "LLM"
    assert "parent_id = :parent_id" in session.executed[0]
    assert "path = :path" in session.executed[0]
