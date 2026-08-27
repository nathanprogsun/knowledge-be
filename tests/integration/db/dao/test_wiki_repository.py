"""Integration tests for ``WikiPageRepository`` / ``WikiFolderRepository``.

Tests insert unique rows per run; isolation relies on unique page ids,
folder ids, knowledge base ids, and tenant ids (via ``make_test_tenant_id``).
The ``wiki_pages`` / ``wiki_folders`` tables sit at the tail of the alembic
chain (revision 0021), so these tests run once the chain is applied.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.exception import ConflictError, NotFoundError
from src.db.dao.wiki_page_repository import WikiFolderRepository, WikiPageRepository
from src.db.models.wiki_page import WikiFolder, WikiPage
from tests.integration.db.dao.conftest import make_test_tenant_id

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _pid() -> str:
    return f"page-{uuid.uuid4().hex[:12]}"


def _fid() -> str:
    return f"folder-{uuid.uuid4().hex[:12]}"


def _kb() -> str:
    return f"kb-{uuid.uuid4().hex[:8]}"


def _sample_page(
    *,
    tenant_id: int,
    knowledge_base_id: str,
    slug: str,
    id: str | None = None,
    **overrides: object,
) -> WikiPage:
    return WikiPage.model_validate(
        {
            "id": id or _pid(),
            "tenant_id": tenant_id,
            "knowledge_base_id": knowledge_base_id,
            "slug": slug,
            "title": "Acme Corp",
            "page_type": "entity",
            "status": "published",
            "content": "Acme is a fictional company.",
            "summary": "Fictional company.",
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
            "version": 1,
            "last_edit_source": "pipeline",
            "last_editor_id": "",
            "created_at": _NOW,
            "updated_at": _NOW,
            "deleted_at": None,
            **overrides,
        }
    )


def _sample_folder(
    *,
    tenant_id: int,
    knowledge_base_id: str,
    name: str,
    id: str | None = None,
    **overrides: object,
) -> WikiFolder:
    return WikiFolder.model_validate(
        {
            "id": id or _fid(),
            "tenant_id": tenant_id,
            "knowledge_base_id": knowledge_base_id,
            "parent_id": "",
            "name": name,
            "path": name,
            "depth": 1,
            "sort_order": 0,
            "created_at": _NOW,
            "updated_at": _NOW,
            "deleted_at": None,
            **overrides,
        }
    )


async def _insert_page(
    repo: WikiPageRepository,
    page: WikiPage,
) -> WikiPage:
    return await repo.create(page)


async def _insert_folder(
    repo: WikiFolderRepository,
    folder: WikiFolder,
) -> WikiFolder:
    return await repo.create(folder)


# ── page create / read ───────────────────────────────────────────────


async def test_create_round_trips_page(session: AsyncSession) -> None:
    repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    slug = "entity/acme"
    pid = _pid()

    persisted = await _insert_page(
        repo,
        _sample_page(tenant_id=tid, knowledge_base_id=kb, slug=slug, id=pid),
    )

    assert persisted.id == pid
    assert persisted.version == 1
    assert persisted.status == "published"

    resolved = await repo.get_by_id_or_none(pid)
    assert resolved is not None
    assert resolved.slug == slug
    assert resolved.title == "Acme Corp"


async def test_get_by_slug_scoped_to_kb(session: AsyncSession) -> None:
    repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb_a = _kb()
    kb_b = _kb()
    slug = "entity/acme"
    await _insert_page(repo, _sample_page(tenant_id=tid, knowledge_base_id=kb_a, slug=slug))

    assert await repo.get_by_slug_or_none(knowledge_base_id=kb_a, slug=slug) is not None
    assert await repo.get_by_slug_or_none(knowledge_base_id=kb_b, slug=slug) is None


async def test_json_columns_round_trip(session: AsyncSession) -> None:
    repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    await _insert_page(
        repo,
        _sample_page(
            tenant_id=tid,
            knowledge_base_id=kb,
            slug="entity/alt",
            aliases=["ACME", "Acme Inc"],
            category_path=["AI", "RAG"],
            in_links=["entity/other"],
            out_links=["concept/retrieval"],
            source_refs=["knowledge-doc-1"],
            chunk_refs=["chunk-1"],
            page_metadata={"tags": ["demo"]},
        ),
    )

    rows = await repo.list_all(knowledge_base_id=kb)

    assert len(rows) == 1
    page = rows[0]
    assert page.aliases == ["ACME", "Acme Inc"]
    assert page.category_path == ["AI", "RAG"]
    assert page.in_links == ["entity/other"]
    assert page.out_links == ["concept/retrieval"]
    assert page.source_refs == ["knowledge-doc-1"]
    assert page.chunk_refs == ["chunk-1"]
    assert page.page_metadata == {"tags": ["demo"]}


# ── update (versioned edit) ──────────────────────────────────────────


async def test_update_bumps_version(session: AsyncSession) -> None:
    repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    persisted = await _insert_page(
        repo,
        _sample_page(tenant_id=tid, knowledge_base_id=kb, slug="entity/acme"),
    )

    edited = persisted.model_copy(
        update={"title": "Acme Corporation", "content": "New body.", "version": 1}
    )
    refreshed = await repo.update(row=edited, now=_NOW)

    assert refreshed.version == 2
    assert refreshed.title == "Acme Corporation"
    assert refreshed.content == "New body."


async def test_update_conflicts_on_stale_version(session: AsyncSession) -> None:
    repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    persisted = await _insert_page(
        repo,
        _sample_page(tenant_id=tid, knowledge_base_id=kb, slug="entity/acme"),
    )

    stale = persisted.model_copy(update={"version": 99, "title": "Stale"})
    with pytest.raises(ConflictError) as excinfo:
        await repo.update(row=stale, now=_NOW)

    assert excinfo.value.code == "wiki.page_conflict"


async def test_update_missing_page_raises_not_found(session: AsyncSession) -> None:
    repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    page = _sample_page(tenant_id=tid, knowledge_base_id=_kb(), slug="entity/acme", version=1)

    with pytest.raises(NotFoundError) as excinfo:
        await repo.update(row=page, now=_NOW)

    assert excinfo.value.code == "wiki.page_not_found"


async def test_update_meta_keeps_version_unchanged(session: AsyncSession) -> None:
    repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    persisted = await _insert_page(
        repo,
        _sample_page(tenant_id=tid, knowledge_base_id=kb, slug="entity/acme"),
    )

    refreshed = await repo.update_meta(
        row=persisted.model_copy(
            update={
                "in_links": ["entity/other"],
                "status": "archived",
                "category_path": ["AI"],
            }
        ),
        now=_NOW,
    )

    assert refreshed.version == 1
    assert refreshed.in_links == ["entity/other"]
    assert refreshed.status == "archived"
    assert refreshed.category_path == ["AI"]


async def test_update_auto_linked_content_keeps_version(session: AsyncSession) -> None:
    repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    persisted = await _insert_page(
        repo,
        _sample_page(tenant_id=tid, knowledge_base_id=kb, slug="entity/acme"),
    )

    refreshed = await repo.update_auto_linked_content(
        row=persisted.model_copy(
            update={"content": "Linked [[Acme]] body.", "out_links": ["entity/acme"]}
        ),
        now=_NOW,
    )

    assert refreshed.version == 1
    assert refreshed.content == "Linked [[Acme]] body."
    assert refreshed.out_links == ["entity/acme"]


# ── soft delete ──────────────────────────────────────────────────────


async def test_soft_delete_by_slug_hides_page(session: AsyncSession) -> None:
    repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    slug = "entity/acme"
    await _insert_page(repo, _sample_page(tenant_id=tid, knowledge_base_id=kb, slug=slug))

    affected = await repo.soft_delete_by_slug(knowledge_base_id=kb, slug=slug, now=_NOW)

    assert affected is True
    assert await repo.get_by_slug_or_none(knowledge_base_id=kb, slug=slug) is None
    assert await repo.soft_delete_by_slug(knowledge_base_id=kb, slug=slug, now=_NOW) is False


async def test_soft_delete_by_id_hides_page(session: AsyncSession) -> None:
    repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    persisted = await _insert_page(
        repo,
        _sample_page(tenant_id=tid, knowledge_base_id=kb, slug="entity/acme"),
    )

    affected = await repo.soft_delete_by_id(id=persisted.id, now=_NOW)

    assert affected is True
    assert await repo.get_by_id_or_none(persisted.id) is None


# ── list_pages ───────────────────────────────────────────────────────


async def test_list_pages_filters_and_paginates(session: AsyncSession) -> None:
    repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    for index in range(5):
        await _insert_page(
            repo,
            _sample_page(
                tenant_id=tid,
                knowledge_base_id=kb,
                slug=f"entity/acme-{index}",
                page_type="entity",
                title=f"Acme {index}",
            ),
        )
    await _insert_page(
        repo,
        _sample_page(
            tenant_id=tid,
            knowledge_base_id=kb,
            slug="concept/rag",
            page_type="concept",
            title="RAG",
        ),
    )

    rows, total = await repo.list_pages(
        knowledge_base_id=kb, page_types=["entity"], page=1, page_size=2
    )

    assert total == 5
    assert len(rows) == 2


async def test_list_pages_filters_by_status_and_folder(session: AsyncSession) -> None:
    repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    folder_id = _fid()
    await _insert_page(
        repo,
        _sample_page(
            tenant_id=tid,
            knowledge_base_id=kb,
            slug="entity/acme",
            status="published",
            folder_id=folder_id,
            category_path=["AI"],
            depth=1,
        ),
    )
    await _insert_page(
        repo,
        _sample_page(
            tenant_id=tid,
            knowledge_base_id=kb,
            slug="entity/draft",
            status="draft",
            folder_id=folder_id,
        ),
    )

    published, total = await repo.list_pages(knowledge_base_id=kb, status="published")
    assert total == 1
    assert published[0].slug == "entity/acme"

    in_folder, folder_total = await repo.list_pages(
        knowledge_base_id=kb, folder_id=folder_id, category_depth=1
    )
    assert folder_total == 1
    assert in_folder[0].slug == "entity/acme"


async def test_list_pages_matches_category_path(session: AsyncSession) -> None:
    repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    await _insert_page(
        repo,
        _sample_page(
            tenant_id=tid,
            knowledge_base_id=kb,
            slug="entity/acme",
            category_path=["AI", "RAG"],
        ),
    )

    rows, total = await repo.list_pages(knowledge_base_id=kb, category_path=["AI", "RAG"])

    assert total == 1
    assert rows[0].slug == "entity/acme"


async def test_list_pages_fulltext_query(session: AsyncSession) -> None:
    repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    await _insert_page(
        repo,
        _sample_page(tenant_id=tid, knowledge_base_id=kb, slug="entity/acme", title="Acme"),
    )
    await _insert_page(
        repo,
        _sample_page(
            tenant_id=tid,
            knowledge_base_id=kb,
            slug="entity/other",
            title="Other",
            content="Unrelated body text.",
        ),
    )

    rows, total = await repo.list_pages(knowledge_base_id=kb, query="acme")

    assert total == 1
    assert rows[0].slug == "entity/acme"


# ── list variants ────────────────────────────────────────────────────


async def test_list_by_type_and_light(session: AsyncSession) -> None:
    repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    for index in range(3):
        await _insert_page(
            repo,
            _sample_page(
                tenant_id=tid,
                knowledge_base_id=kb,
                slug=f"entity/acme-{index}",
                page_type="entity",
                title=f"Acme {index}",
            ),
        )

    by_type = await repo.list_by_type(knowledge_base_id=kb, page_type="entity")
    assert len(by_type) == 3

    light, total = await repo.list_by_type_light(knowledge_base_id=kb, page_type="entity")
    assert total == 3
    assert all(not hasattr(e, "content") for e in light)
    assert light[0].slug == "entity/acme-0"


async def test_list_by_type_recent(session: AsyncSession) -> None:
    repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    await _insert_page(
        repo,
        _sample_page(tenant_id=tid, knowledge_base_id=kb, slug="entity/old", title="Old"),
    )
    await _insert_page(
        repo,
        _sample_page(
            tenant_id=tid,
            knowledge_base_id=kb,
            slug="entity/new",
            title="New",
            updated_at=_NOW + timedelta(days=1),
        ),
    )

    recent = await repo.list_by_type_recent(knowledge_base_id=kb, page_type="entity")

    assert [e.slug for e in recent] == ["entity/new", "entity/old"]


async def test_list_all_orders_by_type_then_title(session: AsyncSession) -> None:
    repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    await _insert_page(
        repo,
        _sample_page(
            tenant_id=tid,
            knowledge_base_id=kb,
            slug="concept/rag",
            title="RAG",
            page_type="concept",
        ),
    )
    await _insert_page(
        repo,
        _sample_page(tenant_id=tid, knowledge_base_id=kb, slug="entity/acme", title="Acme"),
    )

    rows = await repo.list_all(knowledge_base_id=kb)

    assert [r.page_type for r in rows] == ["concept", "entity"]


async def test_list_pages_cursor_walks_all_rows(session: AsyncSession) -> None:
    repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    inserted = [
        await _insert_page(
            repo,
            _sample_page(tenant_id=tid, knowledge_base_id=kb, slug=f"entity/acme-{i}"),
        )
        for i in range(3)
    ]

    page, cursor = await repo.list_pages_cursor(knowledge_base_id=kb, cursor="", limit=2)
    assert len(page) == 2
    assert cursor == page[-1].id

    page2, cursor2 = await repo.list_pages_cursor(knowledge_base_id=kb, cursor=cursor, limit=2)
    assert len(page2) == 1
    assert cursor2 == ""

    seen = {p.id for p in [*page, *page2]}
    assert seen == {p.id for p in inserted}


# ── source-ref queries ───────────────────────────────────────────────


async def test_list_by_source_ref_matches_both_forms(session: AsyncSession) -> None:
    repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    kid = f"knowledge-doc-{uuid.uuid4().hex[:6]}"
    await _insert_page(
        repo,
        _sample_page(
            tenant_id=tid,
            knowledge_base_id=kb,
            slug="entity/acme",
            source_refs=[kid],
        ),
    )
    await _insert_page(
        repo,
        _sample_page(
            tenant_id=tid,
            knowledge_base_id=kb,
            slug="entity/legacy",
            source_refs=[f"{kid}|A report"],
        ),
    )
    await _insert_page(
        repo,
        _sample_page(tenant_id=tid, knowledge_base_id=kb, slug="entity/other"),
    )

    pages = await repo.list_by_source_ref(knowledge_base_id=kb, source_knowledge_id=kid)

    assert {p.slug for p in pages} == {"entity/acme", "entity/legacy"}

    slugs = await repo.list_slugs_by_source_ref(knowledge_base_id=kb, source_knowledge_id=kid)
    assert set(slugs) == {"entity/acme", "entity/legacy"}


async def test_list_summaries_by_knowledge_ids(session: AsyncSession) -> None:
    repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    kid = f"knowledge-doc-{uuid.uuid4().hex[:6]}"
    await _insert_page(
        repo,
        _sample_page(
            tenant_id=tid,
            knowledge_base_id=kb,
            slug="summary/one",
            page_type="summary",
            content="Summary body.",
            source_refs=[kid],
        ),
    )

    summaries = await repo.list_summaries_by_knowledge_ids(
        knowledge_base_id=kb, knowledge_ids=[kid]
    )

    assert summaries == {kid: "Summary body."}
    assert await repo.list_summaries_by_knowledge_ids(knowledge_base_id=kb, knowledge_ids=[]) == {}


# ── slug existence / search / aggregates ─────────────────────────────


async def test_exists_slugs_and_list_all_slugs(session: AsyncSession) -> None:
    repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    await _insert_page(
        repo,
        _sample_page(tenant_id=tid, knowledge_base_id=kb, slug="entity/acme"),
    )
    await _insert_page(
        repo,
        _sample_page(
            tenant_id=tid,
            knowledge_base_id=kb,
            slug="entity/archived",
            status="archived",
        ),
    )

    exists = await repo.exists_slugs(
        knowledge_base_id=kb, slugs=["entity/acme", "entity/archived", "nope"]
    )
    assert exists == {"entity/acme": True, "entity/archived": False, "nope": False}

    assert set(await repo.list_all_slugs(knowledge_base_id=kb)) == {"entity/acme"}


async def test_search_ranks_title_hits_first(session: AsyncSession) -> None:
    repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    await _insert_page(
        repo,
        _sample_page(tenant_id=tid, knowledge_base_id=kb, slug="entity/acme", title="Acme"),
    )
    await _insert_page(
        repo,
        _sample_page(
            tenant_id=tid,
            knowledge_base_id=kb,
            slug="entity/other",
            title="Other",
            content="Mentions acme in the body.",
        ),
    )

    rows = await repo.search(knowledge_base_id=kb, query="acme")

    assert [r.slug for r in rows] == ["entity/acme", "entity/other"]


async def test_count_by_type_and_count_orphans(session: AsyncSession) -> None:
    repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    await _insert_page(repo, _sample_page(tenant_id=tid, knowledge_base_id=kb, slug="entity/a"))
    await _insert_page(
        repo,
        _sample_page(tenant_id=tid, knowledge_base_id=kb, slug="concept/b", page_type="concept"),
    )
    await _insert_page(
        repo,
        _sample_page(
            tenant_id=tid,
            knowledge_base_id=kb,
            slug="index",
            page_type="index",
            in_links=["entity/a"],
        ),
    )

    counts = await repo.count_by_type(knowledge_base_id=kb)
    assert counts["entity"] == 1
    assert counts["concept"] == 1

    orphans = await repo.count_orphans(knowledge_base_id=kb)
    # entity/a and concept/b have no inbound links; the index page is excluded.
    assert orphans == 2


async def test_list_by_slugs_returns_lite_map(session: AsyncSession) -> None:
    repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    await _insert_page(
        repo,
        _sample_page(
            tenant_id=tid,
            knowledge_base_id=kb,
            slug="entity/acme",
            aliases=["ACME"],
            out_links=["concept/rag"],
        ),
    )

    result = await repo.list_by_slugs(knowledge_base_id=kb, slugs=["entity/acme", "missing"])

    assert set(result) == {"entity/acme"}
    assert result["entity/acme"].aliases == ["ACME"]
    assert result["entity/acme"].out_links == ["concept/rag"]
    assert await repo.list_by_slugs(knowledge_base_id=kb, slugs=[]) == {}


# ── folder management ────────────────────────────────────────────────


async def test_folder_crud_and_children(session: AsyncSession) -> None:
    repo = WikiFolderRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    parent = await _insert_folder(
        repo, _sample_folder(tenant_id=tid, knowledge_base_id=kb, name="AI")
    )
    child = await _insert_folder(
        repo,
        _sample_folder(
            tenant_id=tid,
            knowledge_base_id=kb,
            name="RAG",
            parent_id=parent.id,
            path="AI/RAG",
            depth=2,
        ),
    )

    parent_found = await repo.get_by_id_or_none(knowledge_base_id=kb, id=parent.id)
    assert parent_found is not None
    assert parent_found.id == parent.id
    child_found = await repo.get_child_by_name_or_none(
        knowledge_base_id=kb, parent_id=parent.id, name="RAG"
    )
    assert child_found is not None
    assert child_found.id == child.id
    assert (
        await repo.get_child_by_name_or_none(knowledge_base_id=kb, parent_id=parent.id, name="X")
        is None
    )

    children = await repo.list_children(knowledge_base_id=kb, parent_id=parent.id)
    assert [c.id for c in children] == [child.id]

    all_folders = await repo.list_all(knowledge_base_id=kb)
    assert {f.id for f in all_folders} == {parent.id, child.id}


async def test_folder_update_renames_and_reparents(session: AsyncSession) -> None:
    repo = WikiFolderRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    folder = await _insert_folder(
        repo, _sample_folder(tenant_id=tid, knowledge_base_id=kb, name="AI")
    )

    refreshed = await repo.update(
        row=folder.model_copy(update={"name": "LLM", "path": "LLM"}),
        now=_NOW,
    )

    assert refreshed.name == "LLM"
    renamed = await repo.get_by_id_or_none(knowledge_base_id=kb, id=folder.id)
    assert renamed is not None
    assert renamed.name == "LLM"


async def test_folder_update_missing_raises_not_found(session: AsyncSession) -> None:
    repo = WikiFolderRepository(session)
    tid = make_test_tenant_id()
    folder = _sample_folder(tenant_id=tid, knowledge_base_id=_kb(), name="AI")

    with pytest.raises(NotFoundError) as excinfo:
        await repo.update(row=folder, now=_NOW)

    assert excinfo.value.code == "wiki.folder_not_found"


async def test_folder_delete_empty_and_not_empty(session: AsyncSession) -> None:
    folder_repo = WikiFolderRepository(session)
    page_repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    empty = await _insert_folder(
        folder_repo, _sample_folder(tenant_id=tid, knowledge_base_id=kb, name="Empty")
    )
    occupied = await _insert_folder(
        folder_repo, _sample_folder(tenant_id=tid, knowledge_base_id=kb, name="Occupied")
    )
    await _insert_page(
        page_repo,
        _sample_page(
            tenant_id=tid,
            knowledge_base_id=kb,
            slug="entity/acme",
            folder_id=occupied.id,
        ),
    )

    await folder_repo.delete(knowledge_base_id=kb, id=empty.id, now=_NOW)
    assert await folder_repo.get_by_id_or_none(knowledge_base_id=kb, id=empty.id) is None

    with pytest.raises(ConflictError) as conflict_info:
        await folder_repo.delete(knowledge_base_id=kb, id=occupied.id, now=_NOW)
    assert conflict_info.value.code == "wiki.folder_not_empty"

    with pytest.raises(NotFoundError) as not_found_info:
        await folder_repo.delete(knowledge_base_id=kb, id=_fid(), now=_NOW)
    assert not_found_info.value.code == "wiki.folder_not_found"


async def test_folder_aggregates_count_pages(session: AsyncSession) -> None:
    folder_repo = WikiFolderRepository(session)
    page_repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    folder_a = await _insert_folder(
        folder_repo, _sample_folder(tenant_id=tid, knowledge_base_id=kb, name="AI")
    )
    folder_b = await _insert_folder(
        folder_repo, _sample_folder(tenant_id=tid, knowledge_base_id=kb, name="RAG")
    )
    for index in range(3):
        await _insert_page(
            page_repo,
            _sample_page(
                tenant_id=tid,
                knowledge_base_id=kb,
                slug=f"entity/a-{index}",
                folder_id=folder_a.id,
                category_path=["AI"],
                depth=1,
            ),
        )
    await _insert_page(
        page_repo,
        _sample_page(
            tenant_id=tid,
            knowledge_base_id=kb,
            slug="entity/b",
            folder_id=folder_b.id,
            category_path=["RAG"],
            depth=1,
        ),
    )

    assert await page_repo.count_pages_in_folder(knowledge_base_id=kb, folder_id=folder_a.id) == 3

    by_folder = await page_repo.count_pages_by_folder(knowledge_base_id=kb)
    assert by_folder[folder_a.id] == 3
    assert by_folder[folder_b.id] == 1

    pages = await page_repo.list_pages_by_folder_ids(
        knowledge_base_id=kb, folder_ids=[folder_a.id, folder_b.id]
    )
    assert len(pages) == 4
    assert await page_repo.list_pages_by_folder_ids(knowledge_base_id=kb, folder_ids=[]) == []


async def test_list_distinct_category_paths(session: AsyncSession) -> None:
    folder_repo = WikiFolderRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    await _insert_folder(
        folder_repo,
        _sample_folder(tenant_id=tid, knowledge_base_id=kb, name="AI", path="AI"),
    )
    await _insert_folder(
        folder_repo,
        _sample_folder(
            tenant_id=tid,
            knowledge_base_id=kb,
            name="RAG",
            parent_id="x",
            path="AI/RAG",
            depth=2,
        ),
    )

    paths = await folder_repo.list_distinct_category_paths(knowledge_base_id=kb, max_paths=10)

    assert paths == ["AI", "AI/RAG"]


# ── suggestions ──────────────────────────────────────────────────────


async def test_list_recent_for_suggestions(session: AsyncSession) -> None:
    repo = WikiPageRepository(session)
    tid = make_test_tenant_id()
    kb = _kb()
    await _insert_page(
        repo,
        _sample_page(tenant_id=tid, knowledge_base_id=kb, slug="entity/new", title="New"),
    )
    await _insert_page(
        repo,
        _sample_page(
            tenant_id=tid,
            knowledge_base_id=kb,
            slug="index",
            page_type="index",
            title="Index",
        ),
    )
    await _insert_page(
        repo,
        _sample_page(tenant_id=tid, knowledge_base_id=kb, slug="entity/hidden", status="archived"),
    )

    rows = await repo.list_recent_for_suggestions(tenant_id=tid, knowledge_base_ids=[kb], limit=10)

    assert [r.slug for r in rows] == ["entity/new"]
    assert (
        await repo.list_recent_for_suggestions(tenant_id=tid, knowledge_base_ids=[], limit=5) == []
    )
