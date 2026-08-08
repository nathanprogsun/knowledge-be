"""Unit tests for `TagService`.

The service is exercised against an ``AsyncMock(spec=TagRepository)``
with closure-captured state so the persistence methods (create /
get_by_id / get_by_name / list_by_kb / update / delete / document-tag
ops) keep working in-memory. Reference counts come from a separate
table and are configured per test. A second mock backs the optional
knowledge-base existence check.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.common.exception import ConflictError, NotFoundError, ValidationError
from src.common.pagination import PaginationResponse
from src.core.knowledge.tags.service.tag_service import TagService
from src.core.knowledge.tags.types import UNTAGGED_TAG_NAME
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.dao.knowledge_tag_repository import TagReferenceCounts, TagRepository
from src.db.models.knowledge_base import KnowledgeBase
from src.db.models.knowledge_tag import KnowledgeTag

_NOW = datetime(2026, 1, 1, tzinfo=UTC)

_TENANT = 7
_KB = "kb-123"


def _tag_row(
    *,
    tag_id: str = "tag-abc",
    tenant_id: int = _TENANT,
    knowledge_base_id: str = _KB,
    name: str = "infrastructure",
    color: str | None = "#ff0000",
    sort_order: int = 3,
    seq_id: int = 10000001,
    created_at: datetime = _NOW,
) -> KnowledgeTag:
    return KnowledgeTag(
        id=tag_id,
        seq_id=seq_id,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        name=name,
        color=color,
        sort_order=sort_order,
        created_at=created_at,
        updated_at=created_at,
    )


def _kb_row(*, kb_id: str = _KB, tenant_id: int = _TENANT) -> KnowledgeBase:
    return KnowledgeBase(
        id=kb_id,
        name="infra-kb",
        type="document",
        is_temporary=False,
        tenant_id=tenant_id,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _make_tag_repo() -> tuple[AsyncMock, dict[str, KnowledgeTag]]:
    """Tag-repo mock with stateful in-memory tag and binding stores."""
    repo = AsyncMock(spec=TagRepository)
    rows: dict[str, KnowledgeTag] = {}
    bindings: dict[str, list[str]] = {}

    async def _create(row: KnowledgeTag) -> KnowledgeTag:
        rows[row.id] = row
        return row

    async def _update(row: KnowledgeTag) -> KnowledgeTag:
        rows[row.id] = row
        return row

    async def _get_by_id(tenant_id: int, id: str) -> KnowledgeTag | None:
        row = rows.get(id)
        if row is None or row.tenant_id != tenant_id:
            return None
        return row

    async def _get_by_seq_id(tenant_id: int, seq_id: int) -> KnowledgeTag | None:
        for row in rows.values():
            if row.seq_id == seq_id and row.tenant_id == tenant_id:
                return row
        return None

    async def _get_by_name(
        tenant_id: int,
        knowledge_base_id: str,
        name: str,
    ) -> KnowledgeTag | None:
        for row in rows.values():
            if (
                row.tenant_id == tenant_id
                and row.knowledge_base_id == knowledge_base_id
                and row.name == name
            ):
                return row
        return None

    async def _list_by_kb(
        *,
        tenant_id: int,
        knowledge_base_id: str,
        page: int = 1,
        page_size: int = 20,
        keyword: str = "",
    ) -> tuple[list[KnowledgeTag], int]:
        matches = [
            row
            for row in rows.values()
            if row.tenant_id == tenant_id
            and row.knowledge_base_id == knowledge_base_id
            and (not keyword or keyword in row.name)
        ]
        matches.sort(key=lambda r: (r.sort_order, -r.created_at.timestamp(), -r.seq_id))
        offset = (page - 1) * page_size
        return matches[offset : offset + page_size], len(matches)

    async def _delete(*, tenant_id: int, id: str) -> bool:
        row = rows.get(id)
        if row is None or row.tenant_id != tenant_id:
            return False
        del rows[id]
        return True

    async def _set_knowledge_tags(*, knowledge_id: str, tag_ids: list[str]) -> None:
        bindings[knowledge_id] = [t for t in dict.fromkeys(tag_ids) if t != ""]

    async def _get_knowledge_tags(
        knowledge_ids: list[str],
    ) -> dict[str, list[KnowledgeTag]]:
        result: dict[str, list[KnowledgeTag]] = {}
        for knowledge_id in knowledge_ids:
            tags = [rows[tag_id] for tag_id in bindings.get(knowledge_id, []) if tag_id in rows]
            if tags:
                result[knowledge_id] = tags
        return result

    async def _delete_knowledge_tag_relations(knowledge_id: str) -> int:
        removed = len(bindings.get(knowledge_id, []))
        bindings.pop(knowledge_id, None)
        return removed

    repo.create.side_effect = _create
    repo.update.side_effect = _update
    repo.get_by_id.side_effect = _get_by_id
    repo.get_by_seq_id.side_effect = _get_by_seq_id
    repo.get_by_name.side_effect = _get_by_name
    repo.list_by_kb.side_effect = _list_by_kb
    repo.delete.side_effect = _delete
    repo.set_knowledge_tags.side_effect = _set_knowledge_tags
    repo.get_knowledge_tags.side_effect = _get_knowledge_tags
    repo.delete_knowledge_tag_relations.side_effect = _delete_knowledge_tag_relations
    return repo, rows


def _make_kb_repo(rows: list[KnowledgeBase]) -> AsyncMock:
    """Knowledge-base repo mock keyed by ``(id, tenant_id)``."""
    repo = AsyncMock(spec=KnowledgeBaseRepository)
    by_key = {(row.id, row.tenant_id): row for row in rows}

    async def _get_by_id_and_tenant(id: str, tenant_id: int) -> KnowledgeBase | None:
        return by_key.get((id, tenant_id))

    repo.get_by_id_and_tenant.side_effect = _get_by_id_and_tenant
    return repo


@pytest.fixture
def tag_repo_and_rows() -> tuple[AsyncMock, dict[str, KnowledgeTag]]:
    return _make_tag_repo()


@pytest.fixture
def service(tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]) -> TagService:
    return TagService(tag_repo=tag_repo_and_rows[0])


@pytest.fixture
def service_with_kb(
    tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]],
) -> TagService:
    return TagService(
        tag_repo=tag_repo_and_rows[0],
        kb_repo=_make_kb_repo([_kb_row()]),
    )


# ── create_tag ──────────────────────────────────────────────────────


class TestCreateTag:
    async def test_persists_new_tag(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, rows = tag_repo_and_rows

        info = await service.create_tag(
            tenant_id=_TENANT,
            knowledge_base_id=_KB,
            name="networking",
            color=" #00ff00 ",
            sort_order=2,
        )

        assert rows[info.id].name == "networking"
        assert rows[info.id].color == "#00ff00"
        assert rows[info.id].sort_order == 2
        assert rows[info.id].tenant_id == _TENANT
        assert rows[info.id].knowledge_base_id == _KB
        assert info.id == rows[info.id].id
        assert info.created_at == info.updated_at

    async def test_trims_the_name(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, rows = tag_repo_and_rows

        info = await service.create_tag(
            tenant_id=_TENANT,
            knowledge_base_id=_KB,
            name="  networking  ",
        )

        assert info.name == "networking"
        assert rows[info.id].name == "networking"

    async def test_pins_untagged_tag_to_front(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, rows = tag_repo_and_rows

        info = await service.create_tag(
            tenant_id=_TENANT,
            knowledge_base_id=_KB,
            name=UNTAGGED_TAG_NAME,
            sort_order=9,
        )

        assert info.sort_order == -1
        assert rows[info.id].sort_order == -1

    async def test_rejects_blank_name(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, rows = tag_repo_and_rows

        with pytest.raises(ValidationError) as exc_info:
            await service.create_tag(
                tenant_id=_TENANT,
                knowledge_base_id=_KB,
                name="   ",
            )
        assert exc_info.value.code == "tag.kb_id_and_name_required"
        assert rows == {}

    async def test_rejects_missing_kb_id(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, rows = tag_repo_and_rows

        with pytest.raises(ValidationError) as exc_info:
            await service.create_tag(
                tenant_id=_TENANT,
                knowledge_base_id="",
                name="networking",
            )
        assert exc_info.value.code == "tag.kb_id_and_name_required"
        assert rows == {}

    async def test_rejects_duplicate_name(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, rows = tag_repo_and_rows
        rows["tag-abc"] = _tag_row(name="networking")

        with pytest.raises(ConflictError) as exc_info:
            await service.create_tag(
                tenant_id=_TENANT,
                knowledge_base_id=_KB,
                name="  networking  ",
            )
        assert exc_info.value.code == "tag.name_conflict"
        assert set(rows) == {"tag-abc"}

    async def test_rejects_unknown_kb(
        self,
        service_with_kb: TagService,
        tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]],
    ) -> None:
        _repo, rows = tag_repo_and_rows

        with pytest.raises(NotFoundError) as exc_info:
            await service_with_kb.create_tag(
                tenant_id=_TENANT,
                knowledge_base_id="kb-ghost",
                name="networking",
            )
        assert exc_info.value.code == "tag.kb_not_found"
        assert rows == {}

    async def test_rejects_kb_of_another_tenant(
        self,
        service_with_kb: TagService,
        tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]],
    ) -> None:
        _repo, rows = tag_repo_and_rows

        with pytest.raises(NotFoundError) as exc_info:
            await service_with_kb.create_tag(
                tenant_id=999,
                knowledge_base_id=_KB,
                name="networking",
            )
        assert exc_info.value.code == "tag.kb_not_found"
        assert rows == {}


# ── find_or_create_tag_by_name ──────────────────────────────────────


class TestFindOrCreateTagByName:
    async def test_returns_existing_tag(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, rows = tag_repo_and_rows
        rows["tag-abc"] = _tag_row(name="networking")

        info = await service.find_or_create_tag_by_name(
            tenant_id=_TENANT,
            knowledge_base_id=_KB,
            name="networking",
        )

        assert info.id == "tag-abc"
        assert info.name == "networking"
        assert len(rows) == 1

    async def test_creates_when_absent(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, rows = tag_repo_and_rows

        info = await service.find_or_create_tag_by_name(
            tenant_id=_TENANT,
            knowledge_base_id=_KB,
            name="networking",
        )

        assert info.name == "networking"
        assert len(rows) == 1

    async def test_rejects_blank_input(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, rows = tag_repo_and_rows

        with pytest.raises(ValidationError) as exc_info:
            await service.find_or_create_tag_by_name(
                tenant_id=_TENANT,
                knowledge_base_id=_KB,
                name=" ",
            )
        assert exc_info.value.code == "tag.kb_id_and_name_required"
        assert rows == {}


# ── list_tags ───────────────────────────────────────────────────────


class TestListTags:
    async def test_returns_page_with_usage_stats(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        repo, rows = tag_repo_and_rows
        rows["tag-a"] = _tag_row(tag_id="tag-a", name="networking", sort_order=1)
        rows["tag-b"] = _tag_row(tag_id="tag-b", name="infrastructure", sort_order=2)
        repo.batch_count_references.return_value = {
            "tag-a": TagReferenceCounts(knowledge_count=2, chunk_count=5),
            "tag-b": TagReferenceCounts(knowledge_count=0, chunk_count=1),
        }

        page = await service.list_tags(tenant_id=_TENANT, knowledge_base_id=_KB)

        assert page.total == 2
        assert page.page == 1
        assert page.page_size == 20
        assert [item.name for item in page.data] == ["networking", "infrastructure"]
        assert page.data[0].knowledge_count == 2
        assert page.data[0].chunk_count == 5
        assert page.data[1].knowledge_count == 0
        assert page.data[1].chunk_count == 1

    async def test_zero_fills_counts_for_unreferenced_tags(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        repo, rows = tag_repo_and_rows
        rows["tag-a"] = _tag_row(tag_id="tag-a", name="networking")
        repo.batch_count_references.return_value = {
            "tag-a": TagReferenceCounts(knowledge_count=0, chunk_count=0),
        }

        page = await service.list_tags(tenant_id=_TENANT, knowledge_base_id=_KB)

        assert page.data[0].knowledge_count == 0
        assert page.data[0].chunk_count == 0

    async def test_returns_empty_page_when_no_tags(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, _rows = tag_repo_and_rows

        page = await service.list_tags(tenant_id=_TENANT, knowledge_base_id=_KB)

        assert isinstance(page, PaginationResponse)
        assert page.total == 0
        assert page.data == []

    async def test_rejects_empty_kb_id(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, _rows = tag_repo_and_rows

        with pytest.raises(ValidationError) as exc_info:
            await service.list_tags(tenant_id=_TENANT, knowledge_base_id="")
        assert exc_info.value.code == "tag.kb_id_required"

    async def test_rejects_unknown_kb(
        self,
        service_with_kb: TagService,
        tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]],
    ) -> None:
        _repo, _rows = tag_repo_and_rows

        with pytest.raises(NotFoundError) as exc_info:
            await service_with_kb.list_tags(tenant_id=_TENANT, knowledge_base_id="kb-ghost")
        assert exc_info.value.code == "tag.kb_not_found"


# ── update_tag ──────────────────────────────────────────────────────


class TestUpdateTag:
    async def test_patches_supplied_fields(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, rows = tag_repo_and_rows
        rows["tag-abc"] = _tag_row(name="old", color="#000000", sort_order=1)

        info = await service.update_tag(
            tenant_id=_TENANT,
            tag_id="tag-abc",
            name="new",
            color="#ffffff",
            sort_order=5,
        )

        assert info.name == "new"
        assert info.color == "#ffffff"
        assert info.sort_order == 5
        assert info.updated_at > _NOW
        assert rows["tag-abc"].name == "new"

    async def test_leaves_omitted_fields_untouched(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, rows = tag_repo_and_rows
        rows["tag-abc"] = _tag_row(name="kept", color="#000000", sort_order=1)

        info = await service.update_tag(
            tenant_id=_TENANT,
            tag_id="tag-abc",
            color="#ffffff",
        )

        assert info.name == "kept"
        assert info.color == "#ffffff"
        assert info.sort_order == 1

    async def test_trims_supplied_name_and_color(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, rows = tag_repo_and_rows
        rows["tag-abc"] = _tag_row()

        info = await service.update_tag(
            tenant_id=_TENANT,
            tag_id="tag-abc",
            name="  padded  ",
            color="  #abcdef  ",
        )

        assert info.name == "padded"
        assert info.color == "#abcdef"

    async def test_rejects_blank_name(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, rows = tag_repo_and_rows
        rows["tag-abc"] = _tag_row()

        with pytest.raises(ValidationError) as exc_info:
            await service.update_tag(
                tenant_id=_TENANT,
                tag_id="tag-abc",
                name=" ",
            )
        assert exc_info.value.code == "tag.name_required"

    async def test_rejects_missing_id(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, _rows = tag_repo_and_rows

        with pytest.raises(ValidationError) as exc_info:
            await service.update_tag(tenant_id=_TENANT, tag_id="")
        assert exc_info.value.code == "tag.tag_id_required"

    async def test_raises_not_found_for_unknown_tag(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, _rows = tag_repo_and_rows

        with pytest.raises(NotFoundError) as exc_info:
            await service.update_tag(tenant_id=_TENANT, tag_id="tag-ghost", name="new")
        assert exc_info.value.code == "tag.not_found"

    async def test_raises_not_found_for_other_tenant_tag(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, rows = tag_repo_and_rows
        rows["tag-abc"] = _tag_row(tenant_id=999)

        with pytest.raises(NotFoundError) as exc_info:
            await service.update_tag(tenant_id=_TENANT, tag_id="tag-abc", name="new")
        assert exc_info.value.code == "tag.not_found"


# ── resolve_tag_id ───────────────────────────────────────────────────


class TestResolveTagId:
    async def test_passes_through_a_uuid(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, _rows = tag_repo_and_rows

        resolved = await service.resolve_tag_id(tenant_id=_TENANT, tag_id="tag-abc")

        assert resolved == "tag-abc"

    async def test_resolves_a_numeric_seq_id(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, rows = tag_repo_and_rows
        rows["tag-abc"] = _tag_row(seq_id=10000001)

        resolved = await service.resolve_tag_id(tenant_id=_TENANT, tag_id="10000001")

        assert resolved == "tag-abc"

    async def test_unknown_seq_id_raises_not_found(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, rows = tag_repo_and_rows
        rows["tag-abc"] = _tag_row(tenant_id=999, seq_id=10000001)

        with pytest.raises(NotFoundError) as exc_info:
            await service.resolve_tag_id(tenant_id=_TENANT, tag_id="10000001")
        assert exc_info.value.code == "tag.not_found"

    async def test_empty_id_raises_validation(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, _rows = tag_repo_and_rows

        with pytest.raises(ValidationError) as exc_info:
            await service.resolve_tag_id(tenant_id=_TENANT, tag_id="")
        assert exc_info.value.code == "tag.tag_id_required"


# ── delete_tag ──────────────────────────────────────────────────────


class TestDeleteTag:
    async def test_deletes_unreferenced_tag(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        repo, rows = tag_repo_and_rows
        rows["tag-abc"] = _tag_row()
        repo.count_references.return_value = TagReferenceCounts(0, 0)

        removed = await service.delete_tag(tenant_id=_TENANT, tag_id="tag-abc")

        assert removed is True
        assert "tag-abc" not in rows

    async def test_force_deletes_referenced_tag(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        repo, rows = tag_repo_and_rows
        rows["tag-abc"] = _tag_row()
        repo.count_references.return_value = TagReferenceCounts(3, 4)

        removed = await service.delete_tag(
            tenant_id=_TENANT,
            tag_id="tag-abc",
            force=True,
        )

        assert removed is True
        assert "tag-abc" not in rows

    async def test_rejects_referenced_tag_without_force(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        repo, rows = tag_repo_and_rows
        rows["tag-abc"] = _tag_row()
        repo.count_references.return_value = TagReferenceCounts(1, 0)

        with pytest.raises(ValidationError) as exc_info:
            await service.delete_tag(tenant_id=_TENANT, tag_id="tag-abc")
        assert exc_info.value.code == "tag.has_references"
        assert "tag-abc" in rows

    async def test_content_only_keeps_tag(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        repo, rows = tag_repo_and_rows
        rows["tag-abc"] = _tag_row()
        repo.count_references.return_value = TagReferenceCounts(2, 3)

        removed = await service.delete_tag(
            tenant_id=_TENANT,
            tag_id="tag-abc",
            content_only=True,
        )

        assert removed is False
        assert "tag-abc" in rows

    async def test_exclude_ids_keeps_tag(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        repo, rows = tag_repo_and_rows
        rows["tag-abc"] = _tag_row()
        repo.count_references.return_value = TagReferenceCounts(2, 3)

        removed = await service.delete_tag(
            tenant_id=_TENANT,
            tag_id="tag-abc",
            force=True,
            exclude_ids=["chunk-1"],
        )

        assert removed is False
        assert "tag-abc" in rows

    async def test_rejects_missing_id(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, _rows = tag_repo_and_rows

        with pytest.raises(ValidationError) as exc_info:
            await service.delete_tag(tenant_id=_TENANT, tag_id="")
        assert exc_info.value.code == "tag.tag_id_required"

    async def test_raises_not_found_for_unknown_tag(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, _rows = tag_repo_and_rows

        with pytest.raises(NotFoundError) as exc_info:
            await service.delete_tag(tenant_id=_TENANT, tag_id="tag-ghost")
        assert exc_info.value.code == "tag.not_found"


# ── document-tag bind / unbind ──────────────────────────────────────


class TestDocumentTagBindings:
    async def test_set_knowledge_tags_replaces_bindings(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, rows = tag_repo_and_rows
        rows["tag-a"] = _tag_row(tag_id="tag-a")
        rows["tag-b"] = _tag_row(tag_id="tag-b")

        await service.set_knowledge_tags(knowledge_id="kn-1", tag_ids=["tag-a", "tag-b"])
        await service.set_knowledge_tags(knowledge_id="kn-1", tag_ids=["tag-b"])

        tags = await service.get_knowledge_tags(["kn-1"])
        assert [tag.id for tag in tags["kn-1"]] == ["tag-b"]

    async def test_get_knowledge_tags_maps_to_info(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, rows = tag_repo_and_rows
        rows["tag-a"] = _tag_row(tag_id="tag-a", name="networking")
        await service.set_knowledge_tags(knowledge_id="kn-1", tag_ids=["tag-a"])

        tags = await service.get_knowledge_tags(["kn-1"])

        assert tags["kn-1"][0].name == "networking"

    async def test_delete_knowledge_tag_relations_returns_count(
        self, service: TagService, tag_repo_and_rows: tuple[AsyncMock, dict[str, KnowledgeTag]]
    ) -> None:
        _repo, rows = tag_repo_and_rows
        rows["tag-a"] = _tag_row(tag_id="tag-a")
        rows["tag-b"] = _tag_row(tag_id="tag-b")
        await service.set_knowledge_tags(knowledge_id="kn-1", tag_ids=["tag-a", "tag-b"])

        removed = await service.delete_knowledge_tag_relations("kn-1")

        assert removed == 2
        assert await service.get_knowledge_tags(["kn-1"]) == {}
