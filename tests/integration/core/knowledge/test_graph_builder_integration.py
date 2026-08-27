"""Integration tests for the knowledge graph builder.

Lives under ``tests/integration/`` (not ``tests/core/``) so the
integration conftest's ``_integration_settings`` fixture — which sets
``RBAC_ENFORCED=false`` for the entire suite — is only activated when
the integration collection is explicit. Importing that fixture from a
``tests/core/`` module would otherwise leak the env change into every
unit test that runs after it in the same pytest session.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime
from itertools import count
from uuid import uuid4

from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.knowledge.graph import GraphBuilder
from src.db.dao.chunk_repository import ChunkRepository
from src.db.models.chunk import Chunk
from tests.core.knowledge.test_graph_builder import (
    FakeChat,
    _entities_for,
    _no_relationships,
    _relationships_for,
    make_test_tenant_id,
)
from tests.integration.conftest import session

__all__ = ["session"]


# ``chunks.tenant_id`` is an INTEGER (32-bit) column, so rows are seeded
# under an int32-safe local counter instead of the bigint
# ``make_test_tenant_id`` values (which can exceed the column range).
_int32_tenant_ids = count(random.randrange(1_000_000, 2_000_000_000 - 1_000))


def _next_int32_tenant_id() -> int:
    return next(_int32_tenant_ids)


def _unique_chunk_id(prefix: str) -> str:
    """Return a chunk id unique across test runs (no DB cleanup is done)."""
    return f"{prefix}-{uuid4().hex[:8]}"


def _persisted_chunk(
    *,
    tenant_id: int,
    id: str,
    content: str,
    start_at: int,
    end_at: int,
    chunk_index: int,
) -> Chunk:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return Chunk(
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id="kb-it",
        knowledge_id="knowledge-doc-1",
        content=content,
        chunk_index=chunk_index,
        is_enabled=True,
        start_at=start_at,
        end_at=end_at,
        chunk_type="text",
        flags=1,
        source_content="",
        content_revision=0,
        index_status="ready",
        last_editor_id="",
        created_at=now,
        updated_at=now,
        deleted_at=None,
    )


async def test_integration_build_over_persisted_chunks(session: AsyncSession, faker: Faker) -> None:
    repo = ChunkRepository(session)
    # ``chunks.tenant_id`` is an INTEGER (32-bit) column, so rows are seeded
    # under an int32-safe local counter instead of the bigint
    # ``make_test_tenant_id`` values (which can exceed the column range).
    tenant_id = _next_int32_tenant_id()
    other_tenant = make_test_tenant_id()
    first_id = _unique_chunk_id("graph-it")
    second_id = _unique_chunk_id("graph-it")
    content_first = f"William Shakespeare wrote Romeo and Juliet. {faker.sentence(nb_words=5)}"
    content_second = f"Romeo and Juliet is set in Verona. {faker.sentence(nb_words=5)}"
    rows = [
        _persisted_chunk(
            tenant_id=tenant_id,
            id=first_id,
            content=content_first,
            start_at=0,
            end_at=len(content_first),
            chunk_index=0,
        ),
        _persisted_chunk(
            tenant_id=tenant_id,
            id=second_id,
            content=content_second,
            start_at=len(content_first),
            end_at=len(content_first) + len(content_second),
            chunk_index=1,
        ),
    ]
    await repo.create_many(rows)
    await session.commit()

    stored = await repo.list_by_knowledge_id(tenant_id, "knowledge-doc-1")
    assert len(stored) == 2
    # Tenant isolation: a bigint tenant id sees none of the seeded rows.
    assert await repo.list_by_knowledge_id(other_tenant, "knowledge-doc-1") == []

    fake = FakeChat(entities=_entities_for, relationships=_relationships_for)
    builder = GraphBuilder(chat=fake)
    result = await builder.build_graph(stored)

    entities = {entity.title: entity for entity in result.entities}
    assert set(entities) == {"William Shakespeare", "Romeo and Juliet", "Verona"}
    assert entities["Romeo and Juliet"].chunk_ids == [first_id, second_id]
    assert entities["Romeo and Juliet"].frequency == 2
    assert len(result.relationships) == 2
    assert builder.get_relation_chunks(first_id, 0) == [first_id, second_id]


async def test_integration_lookup_on_persisted_chunks(session: AsyncSession, faker: Faker) -> None:
    repo = ChunkRepository(session)
    tenant_id = _next_int32_tenant_id()
    chunk_id = _unique_chunk_id("graph-lk")
    content = f"Alpha Beta Gamma. {faker.sentence(nb_words=5)}"
    rows = [
        _persisted_chunk(
            tenant_id=tenant_id,
            id=chunk_id,
            content=content,
            start_at=0,
            end_at=len(content),
            chunk_index=0,
        )
    ]
    await repo.create_many(rows)
    await session.commit()

    stored = await repo.list_by_knowledge_id(tenant_id, "knowledge-doc-1")

    def entities(_user: str) -> list[dict[str, object]]:
        return [
            {"title": "Alpha", "type": "Concept", "description": "a"},
            {"title": "Beta", "type": "Concept", "description": "b"},
        ]

    fake = FakeChat(entities=entities, relationships=_no_relationships)
    builder = GraphBuilder(chat=fake)
    result = await builder.build_graph(stored)
    assert len(result.entities) == 2
    # No relationships, so the chunk graph is empty and lookups return nothing.
    assert builder.get_relation_chunks(chunk_id, 0) == []
    assert builder.get_indirect_relation_chunks(chunk_id, 0) == []
