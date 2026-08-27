"""Integration tests for ``KBService`` against the real applied schema.

Exercises the service end-to-end over a real ``AsyncSession``. Each test
uses a unique tenant id (``make_test_tenant_id``) and commits explicitly;
isolation comes from the unique ids rather than cleanup, matching the DAO
integration tests.

Only ``count_chunks`` is exercised here: the ``chunks`` table is part of
the applied migration chain. The document and member counts read sibling
tables that are not part of this chain and are covered by unit tests
instead.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.exception import NotFoundError, ValidationError
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import KNOWLEDGE_BASE_TYPE_FAQ
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from tests.integration.conftest import make_test_tenant_id


def _service(session: AsyncSession) -> KBService:
    return KBService(kb_repo=KnowledgeBaseRepository(session))


# ── create / read / update / delete ─────────────────────────────────


async def test_create_get_update_delete_round_trip(session: AsyncSession) -> None:
    service = _service(session)
    tid = make_test_tenant_id()

    created = await service.create_knowledge_base(tenant_id=tid, name="docs", description="notes")
    await session.commit()

    assert created.id
    assert created.tenant_id == tid
    assert created.name == "docs"

    fetched = await service.get_knowledge_base_by_id(knowledge_base_id=created.id)
    assert fetched.name == "docs"
    assert fetched.description == "notes"

    updated = await service.update_knowledge_base(
        knowledge_base_id=created.id,
        name="renamed",
        description="updated",
    )
    await session.commit()
    assert updated.name == "renamed"

    deleted = await service.delete_knowledge_base(knowledge_base_id=created.id)
    await session.commit()
    assert deleted is True

    with pytest.raises(NotFoundError):
        await service.get_knowledge_base_by_id(knowledge_base_id=created.id)


async def test_update_applies_chunking_config(session: AsyncSession) -> None:
    service = _service(session)
    tid = make_test_tenant_id()
    created = await service.create_knowledge_base(tenant_id=tid, name="docs")
    await session.commit()

    updated = await service.update_knowledge_base(
        knowledge_base_id=created.id,
        name="docs",
        config={"chunking_config": {"chunk_size": 512, "chunk_overlap": 64}},
    )
    await session.commit()

    assert updated.chunking_config == {"chunk_size": 512, "chunk_overlap": 64}


# ── tenant isolation ────────────────────────────────────────────────


async def test_get_by_id_and_tenant_enforces_isolation(session: AsyncSession) -> None:
    service = _service(session)
    tid = make_test_tenant_id()
    created = await service.create_knowledge_base(tenant_id=tid, name="docs")
    await session.commit()

    found = await service.get_knowledge_base_by_id_and_tenant(
        tenant_id=tid, knowledge_base_id=created.id
    )
    assert found.id == created.id

    with pytest.raises(NotFoundError):
        await service.get_knowledge_base_by_id_and_tenant(
            tenant_id=make_test_tenant_id(), knowledge_base_id=created.id
        )


# ── listing ─────────────────────────────────────────────────────────


async def test_list_excludes_temporary_rows(session: AsyncSession) -> None:
    service = _service(session)
    tid = make_test_tenant_id()
    permanent = await service.create_knowledge_base(tenant_id=tid, name="docs")
    await service.create_knowledge_base(tenant_id=tid, name="tmp", is_temporary=True)
    await session.commit()

    infos = await service.list_knowledge_bases(tenant_id=tid)

    assert [i.id for i in infos] == [permanent.id]


async def test_list_faq_rows_carry_chunk_count(session: AsyncSession) -> None:
    service = _service(session)
    tid = make_test_tenant_id()
    created = await service.create_knowledge_base(
        tenant_id=tid, name="faq", kb_type=KNOWLEDGE_BASE_TYPE_FAQ
    )
    await session.commit()

    infos = await service.list_knowledge_bases(tenant_id=tid)

    assert [i.id for i in infos] == [created.id]
    assert infos[0].chunk_count == 0


# ── type defaults / counts ──────────────────────────────────────────


async def test_faq_defaults_persisted(session: AsyncSession) -> None:
    service = _service(session)
    tid = make_test_tenant_id()
    created = await service.create_knowledge_base(
        tenant_id=tid, name="faq", kb_type=KNOWLEDGE_BASE_TYPE_FAQ
    )
    await session.commit()

    fetched = await service.get_knowledge_base_by_id(knowledge_base_id=created.id)

    assert fetched.faq_config == {
        "index_mode": "question_answer",
        "question_index_mode": "combined",
    }


async def test_count_chunks_returns_zero_for_fresh_kb(session: AsyncSession) -> None:
    service = _service(session)
    tid = make_test_tenant_id()
    created = await service.create_knowledge_base(tenant_id=tid, name="docs")
    await session.commit()

    assert await service.count_chunks(tenant_id=tid, knowledge_base_id=created.id) == 0


async def test_validation_errors_do_not_persist(session: AsyncSession) -> None:
    service = _service(session)
    tid = make_test_tenant_id()

    with pytest.raises(ValidationError):
        await service.create_knowledge_base(tenant_id=tid, name="   ")

    infos = await service.list_knowledge_bases(tenant_id=tid)
    assert infos == []
