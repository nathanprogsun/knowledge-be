"""Integration tests for `TagService` against the real applied schema.

Seeds a knowledge base (the injected KB repository validates its
existence) and exercises the tag CRUD, the reference-guarded delete,
the usage-stat enrichment, and the document-tag bind/unbind against
the real ``tags`` / ``document_tags`` tables. Isolation relies on
unique tenant ids and unique entity ids; tests commit explicitly.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.exception import ConflictError, NotFoundError, ValidationError
from src.core.knowledge.tags.factory import build_tag_service
from src.core.knowledge.tags.service.tag_service import TagService
from src.core.knowledge.tags.types import UNTAGGED_TAG_NAME
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document
from src.db.models.knowledge_base import KnowledgeBase

_NOW = datetime(2026, 1, 1, tzinfo=UTC)

# ``chunks.tenant_id`` is INTEGER, so the seeded tenant must stay inside
# int32 range; this counter mints unique ids without hitting BIGINT.
_INT_TENANT_COUNTER = {"value": 1_000_000}


def _int_tenant_id() -> int:
    _INT_TENANT_COUNTER["value"] += 1
    return _INT_TENANT_COUNTER["value"]


def _kb_row(*, tenant_id: int) -> KnowledgeBase:
    return KnowledgeBase(
        id=f"kb-{uuid.uuid4().hex[:12]}",
        name="infra-kb",
        type="document",
        is_temporary=False,
        tenant_id=tenant_id,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _doc_row(*, tenant_id: int, knowledge_base_id: str, kind: str = "doc") -> Document:
    return Document(
        id=f"kn-{kind}-{uuid.uuid4().hex[:12]}",
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        type="document",
        title=f"{kind}-{uuid.uuid4().hex[:8]}",
        source="manual",
        channel="web",
        created_at=_NOW,
        updated_at=_NOW,
    )


def _chunk_row(
    *,
    tenant_id: int,
    knowledge_base_id: str,
    knowledge_id: str,
    tag_id: str,
) -> Chunk:
    return Chunk(
        id=f"chunk-{uuid.uuid4().hex[:12]}",
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        knowledge_id=knowledge_id,
        tag_id=tag_id,
        content="faq content",
        chunk_index=0,
        start_at=0,
        end_at=10,
        created_at=_NOW,
        updated_at=_NOW,
    )


@pytest.fixture
async def seeded_service(
    session: AsyncSession,
) -> tuple[TagService, AsyncSession, int, str]:
    """A tag service bound to a seeded knowledge base of a fresh tenant."""
    tenant_id = _int_tenant_id()
    kb = _kb_row(tenant_id=tenant_id)
    await KnowledgeBaseRepository(session).create(kb)
    await session.commit()
    return build_tag_service(session), session, tenant_id, kb.id


async def test_create_tag_persists_and_assigns_seq_id(
    seeded_service: tuple[TagService, AsyncSession, int, str],
) -> None:
    service, _session, tenant_id, kb_id = seeded_service

    info = await service.create_tag(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        name="networking",
        color="#00ff00",
        sort_order=2,
    )

    assert info.id != ""
    assert info.seq_id > 0
    assert info.tenant_id == tenant_id
    assert info.knowledge_base_id == kb_id
    assert info.name == "networking"
    assert info.color == "#00ff00"


async def test_create_tag_conflicts_on_duplicate_name(
    seeded_service: tuple[TagService, AsyncSession, int, str],
) -> None:
    service, _session, tenant_id, kb_id = seeded_service
    await service.create_tag(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        name="networking",
    )

    with pytest.raises(ConflictError) as exc_info:
        await service.create_tag(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            name="  networking  ",
        )
    assert exc_info.value.code == "tag.name_conflict"


async def test_create_tag_rejects_unknown_kb(
    seeded_service: tuple[TagService, AsyncSession, int, str],
) -> None:
    service, _session, tenant_id, _kb_id = seeded_service

    with pytest.raises(NotFoundError) as exc_info:
        await service.create_tag(
            tenant_id=tenant_id,
            knowledge_base_id="kb-ghost",
            name="networking",
        )
    assert exc_info.value.code == "tag.kb_not_found"


async def test_list_tags_enriches_usage_stats(
    seeded_service: tuple[TagService, AsyncSession, int, str],
) -> None:
    service, session, tenant_id, kb_id = seeded_service
    tag_a = await service.create_tag(
        tenant_id=tenant_id, knowledge_base_id=kb_id, name="networking"
    )
    tag_b = await service.create_tag(
        tenant_id=tenant_id, knowledge_base_id=kb_id, name="infrastructure"
    )
    doc = _doc_row(tenant_id=tenant_id, knowledge_base_id=kb_id)
    await KnowledgeRepository(session).create(doc)
    await session.commit()
    await service.set_knowledge_tags(knowledge_id=doc.id, tag_ids=[tag_a.id, tag_b.id])
    await ChunkRepository(session).create(
        _chunk_row(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            knowledge_id=doc.id,
            tag_id=tag_a.id,
        )
    )
    await session.commit()

    page = await service.list_tags(tenant_id=tenant_id, knowledge_base_id=kb_id)

    counts = {item.name: (item.knowledge_count, item.chunk_count) for item in page.data}
    assert page.total == 2
    assert counts["networking"] == (1, 1)
    assert counts["infrastructure"] == (1, 0)


async def test_delete_tag_guards_references_then_force_deletes(
    seeded_service: tuple[TagService, AsyncSession, int, str],
) -> None:
    service, session, tenant_id, kb_id = seeded_service
    tag = await service.create_tag(tenant_id=tenant_id, knowledge_base_id=kb_id, name="networking")
    doc = _doc_row(tenant_id=tenant_id, knowledge_base_id=kb_id)
    await KnowledgeRepository(session).create(doc)
    await session.commit()
    await service.set_knowledge_tags(knowledge_id=doc.id, tag_ids=[tag.id])
    await session.commit()

    with pytest.raises(ValidationError) as exc_info:
        await service.delete_tag(tenant_id=tenant_id, tag_id=tag.id)
    assert exc_info.value.code == "tag.has_references"

    removed = await service.delete_tag(
        tenant_id=tenant_id,
        tag_id=tag.id,
        force=True,
    )
    await session.commit()

    assert removed is True
    page = await service.list_tags(tenant_id=tenant_id, knowledge_base_id=kb_id)
    assert page.total == 0


async def test_delete_tag_content_only_keeps_tag(
    seeded_service: tuple[TagService, AsyncSession, int, str],
) -> None:
    service, _session, tenant_id, kb_id = seeded_service
    tag = await service.create_tag(tenant_id=tenant_id, knowledge_base_id=kb_id, name="networking")

    removed = await service.delete_tag(
        tenant_id=tenant_id,
        tag_id=tag.id,
        content_only=True,
    )

    assert removed is False
    found = await service.list_tags(tenant_id=tenant_id, knowledge_base_id=kb_id)
    assert found.total == 1


async def test_update_tag_patches_fields(
    seeded_service: tuple[TagService, AsyncSession, int, str],
) -> None:
    service, _session, tenant_id, kb_id = seeded_service
    tag = await service.create_tag(
        tenant_id=tenant_id, knowledge_base_id=kb_id, name="old", color="#000000"
    )

    updated = await service.update_tag(
        tenant_id=tenant_id,
        tag_id=tag.id,
        name="new",
        color="#ffffff",
        sort_order=5,
    )

    assert updated.name == "new"
    assert updated.color == "#ffffff"
    assert updated.sort_order == 5
    assert updated.created_at == tag.created_at


async def test_find_or_create_tag_by_name_reuses_existing(
    seeded_service: tuple[TagService, AsyncSession, int, str],
) -> None:
    service, _session, tenant_id, kb_id = seeded_service
    created = await service.create_tag(
        tenant_id=tenant_id, knowledge_base_id=kb_id, name="networking"
    )

    found = await service.find_or_create_tag_by_name(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        name="  networking  ",
    )

    assert found.id == created.id


async def test_untagged_name_pins_sort_order(
    seeded_service: tuple[TagService, AsyncSession, int, str],
) -> None:
    service, _session, tenant_id, kb_id = seeded_service

    info = await service.create_tag(
        tenant_id=tenant_id,
        knowledge_base_id=kb_id,
        name=UNTAGGED_TAG_NAME,
        sort_order=9,
    )

    assert info.sort_order == -1
