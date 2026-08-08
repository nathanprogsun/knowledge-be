"""Unit and integration tests for the knowledge graph builder.

Unit tests drive ``GraphBuilder`` with a deterministic fake of the
``Chat`` protocol (no provider network involved). Integration tests run
against the real applied schema: chunk rows are persisted through the
``chunks`` repository and fed back into the builder as ``ChunkInput``.

``chunks.tenant_id`` is an INTEGER (32-bit) column, so integration rows
use a local int32-safe counter instead of the bigint ``make_test_tenant_id``
values (which can exceed the column range).
"""

from __future__ import annotations

import json
import random
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import count
from uuid import uuid4

import pytest
from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession

from src.ai.llm.types import ChatOptions, ChatResponse, Message, StreamResponse
from src.common.exception import AIProviderError, ValidationError
from src.core.knowledge.graph import GraphBuilder, merge_text_chunks, parse_llm_json_response
from src.db.dao.chunk_repository import ChunkRepository
from src.db.models.chunk import Chunk
from tests.integration.conftest import (
    _engine,
    _integration_settings,
    faker_seed,
    make_test_tenant_id,
    session,
)

# The integration-conftest fixtures are imported into this module (not just
# referenced) so pytest registers them here; listing them in ``__all__``
# documents that intent and keeps ruff's unused-import checks quiet.
__all__ = ["_engine", "_integration_settings", "faker_seed", "session"]


# ── Test doubles ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class _TestChunk:
    """A minimal immutable chunk satisfying the ``ChunkInput`` shape."""

    id: str
    content: str
    start_at: int = 0
    end_at: int = 0
    chunk_index: int = 0


def _chunk(id: str, content: str, *, chunk_index: int = 0) -> _TestChunk:
    """Build a contiguous test chunk spanning ``[0, len(content))``."""
    return _TestChunk(
        id=id,
        content=content,
        start_at=0,
        end_at=len(content),
        chunk_index=chunk_index,
    )


def _no_entities(_user: str) -> list[dict[str, object]]:
    return []


def _no_relationships(_user: str) -> list[dict[str, object]]:
    return []


class FakeChat:
    """Deterministic fake of the ``Chat`` protocol for graph-builder tests.

    Entity-extraction calls are told apart from relationship-extraction
    calls by the relationship user message prefix (``Entities: ``); the
    payload for each call comes from the injected callables, keyed by the
    user message text so tests can vary responses per chunk.
    """

    def __init__(
        self,
        *,
        entities: Callable[[str], list[dict[str, object]]] = _no_entities,
        relationships: Callable[[str], list[dict[str, object]]] = _no_relationships,
        entity_error: Exception | None = None,
        relationship_error: Exception | None = None,
        relationship_raw_content: str | None = None,
    ) -> None:
        self._entities = entities
        self._relationships = relationships
        self._entity_error = entity_error
        self._relationship_error = relationship_error
        self._relationship_raw_content = relationship_raw_content
        self.requests: list[tuple[list[Message], ChatOptions]] = []

    async def chat(self, messages: list[Message], opts: ChatOptions | None = None) -> ChatResponse:
        self.requests.append((messages, opts or ChatOptions()))
        user = messages[-1].content
        if user.startswith("Entities: "):
            error = self._relationship_error
            if error is not None:
                raise error
            content = (
                self._relationship_raw_content
                if self._relationship_raw_content is not None
                else json.dumps(self._relationships(user), ensure_ascii=False)
            )
        else:
            error = self._entity_error
            if error is not None:
                raise error
            content = json.dumps(self._entities(user), ensure_ascii=False)
        return ChatResponse(content=content)

    async def chat_stream(
        self, messages: list[Message], opts: ChatOptions | None = None
    ) -> AsyncIterator[StreamResponse]:
        yield StreamResponse()

    def get_model_name(self) -> str:
        return "fake-chat"

    def get_model_id(self) -> str:
        return "fake-chat"


def _entities_for(user: str) -> list[dict[str, object]]:
    """Two entities for the Shakespeare / Verona scenarios."""
    if "Verona" in user:
        return [
            {"title": "Romeo and Juliet", "type": "Work", "description": "a tragedy"},
            {"title": "Verona", "type": "Location", "description": "an Italian city"},
        ]
    return [
        {"title": "William Shakespeare", "type": "Person", "description": "an English playwright"},
        {"title": "Romeo and Juliet", "type": "Work", "description": "a tragedy"},
    ]


def _relationships_for(_user: str) -> list[dict[str, object]]:
    return [
        {
            "source": "William Shakespeare",
            "target": "Romeo and Juliet",
            "description": "wrote",
            "strength": 8,
        },
        {
            "source": "Romeo and Juliet",
            "target": "Verona",
            "description": "set in",
            "strength": 6,
        },
    ]


def _single_relationship(_user: str) -> list[dict[str, object]]:
    """One relationship whose endpoints are both present in a single chunk."""
    return [
        {
            "source": "William Shakespeare",
            "target": "Romeo and Juliet",
            "description": "wrote",
            "strength": 8,
        },
    ]


# ── parse_llm_json_response ───────────────────────────────────────────


def test_parse_llm_json_response_plain_array() -> None:
    payload = parse_llm_json_response('[{"title": "A", "type": "Concept"}]')
    assert isinstance(payload, list)
    assert payload[0] == {"title": "A", "type": "Concept"}


def test_parse_llm_json_response_unwraps_code_fence() -> None:
    content = 'Here is the result:\n```json\n[{"title": "A"}]\n```'
    payload = parse_llm_json_response(content)
    assert isinstance(payload, list)
    assert payload[0] == {"title": "A"}


def test_parse_llm_json_response_raises_on_invalid() -> None:
    with pytest.raises(ValueError):
        parse_llm_json_response("not json at all")


# ── merge_text_chunks ─────────────────────────────────────────────────


def test_merge_text_chunks_removes_overlapping_boundary() -> None:
    first = _TestChunk(id="c1", content="Alpha Beta Gamma Delta", start_at=0, end_at=23)
    second = _TestChunk(id="c2", content="Beta Gamma Delta Epsilon", start_at=23, end_at=48)
    merged = merge_text_chunks([first, second])
    assert merged == "Alpha Beta Gamma Delta Epsilon"


def test_merge_text_chunks_gap_appends_verbatim() -> None:
    first = _TestChunk(id="c1", content="Alpha Beta Gamma Delta", start_at=0, end_at=23)
    gap = _TestChunk(id="c2", content="Beta Gamma Delta Epsilon", start_at=23, end_at=48)
    far = _TestChunk(id="c3", content="Omega", start_at=100, end_at=105)
    merged = merge_text_chunks([first, gap, far])
    assert merged == "Alpha Beta Gamma Delta EpsilonOmega"


def test_merge_text_chunks_empty_input_returns_empty() -> None:
    assert merge_text_chunks([]) == ""


# ── build_graph: extraction and aggregation ───────────────────────────


async def test_build_empty_chunks_returns_empty_graph() -> None:
    fake = FakeChat()
    builder = GraphBuilder(chat=fake)
    result = await builder.build_graph([])
    assert result.entities == []
    assert result.relationships == []
    assert result.chunk_count == 0
    assert fake.requests == []


async def test_build_skips_empty_chunk_content() -> None:
    fake = FakeChat(entities=_entities_for, relationships=_relationships_for)
    builder = GraphBuilder(chat=fake)
    result = await builder.build_graph([_chunk("c1", "")])
    assert result.entities == []
    assert result.relationships == []
    assert fake.requests == []


async def test_build_extracts_entities_and_relationships_with_weights() -> None:
    fake = FakeChat(entities=_entities_for, relationships=_single_relationship)
    builder = GraphBuilder(chat=fake)
    result = await builder.build_graph([_chunk("c1", "Romeo and Juliet by William Shakespeare.")])

    entities = {entity.title: entity for entity in result.entities}
    assert set(entities) == {"William Shakespeare", "Romeo and Juliet"}
    assert entities["William Shakespeare"].chunk_ids == ["c1"]
    assert entities["William Shakespeare"].frequency == 1
    assert entities["William Shakespeare"].degree == 1
    assert entities["Romeo and Juliet"].degree == 1

    assert len(result.relationships) == 1
    rel = result.relationships[0]
    assert rel.source == "William Shakespeare"
    assert rel.target == "Romeo and Juliet"
    assert rel.description == "wrote"
    assert rel.strength == 8
    assert rel.chunk_ids == ["c1"]
    assert rel.weight == pytest.approx(10.0)
    assert rel.combined_degree == 2
    assert result.chunk_count == 1


async def test_build_dedupes_entities_across_chunks() -> None:
    fake = FakeChat(entities=_entities_for, relationships=_relationships_for)
    builder = GraphBuilder(chat=fake)
    result = await builder.build_graph(
        [
            _chunk("c1", "William Shakespeare wrote Romeo and Juliet."),
            _chunk("c2", "Romeo and Juliet is set in Verona."),
        ]
    )

    entities = {entity.title: entity for entity in result.entities}
    assert set(entities) == {"William Shakespeare", "Romeo and Juliet", "Verona"}
    assert entities["Romeo and Juliet"].chunk_ids == ["c1", "c2"]
    assert entities["Romeo and Juliet"].frequency == 2
    assert entities["William Shakespeare"].chunk_ids == ["c1"]
    assert entities["Verona"].chunk_ids == ["c2"]

    relationships = {(rel.source, rel.target): rel for rel in result.relationships}
    assert set(relationships) == {
        ("William Shakespeare", "Romeo and Juliet"),
        ("Romeo and Juliet", "Verona"),
    }
    wrote = relationships[("William Shakespeare", "Romeo and Juliet")]
    assert wrote.chunk_ids == ["c1", "c2"]
    assert wrote.weight == pytest.approx(10.0)
    set_in = relationships[("Romeo and Juliet", "Verona")]
    assert set_in.weight == pytest.approx(9.1)
    assert set_in.combined_degree == 3
    assert entities["Romeo and Juliet"].degree == 2


async def test_build_skips_only_exact_empty_description() -> None:
    """An empty description drops the entity; a whitespace one is kept."""

    def entities(user: str) -> list[dict[str, object]]:
        return [
            {"title": "Keep", "type": "Concept", "description": "kept"},
            {"title": "Drop", "type": "Concept", "description": ""},
            {"title": "AlsoDrop", "type": "Concept", "description": "  "},
        ]

    fake = FakeChat(entities=entities, relationships=_no_relationships)
    builder = GraphBuilder(chat=fake)
    result = await builder.build_graph([_chunk("c1", "any text")])
    assert [entity.title for entity in result.entities] == ["Keep", "AlsoDrop"]


async def test_build_uses_integer_division_for_strength_update() -> None:
    def entities(_user: str) -> list[dict[str, object]]:
        return [
            {"title": "Alpha", "type": "Concept", "description": "a"},
            {"title": "Beta", "type": "Concept", "description": "b"},
        ]

    def relationships(_user: str) -> list[dict[str, object]]:
        return [
            {"source": "Alpha", "target": "Beta", "description": "rel", "strength": 8},
            {"source": "Alpha", "target": "Beta", "description": "rel", "strength": 10},
        ]

    fake = FakeChat(entities=entities, relationships=relationships)
    builder = GraphBuilder(chat=fake)
    result = await builder.build_graph([_chunk("c1", "Alpha Beta"), _chunk("c2", "Alpha Beta")])
    assert len(result.relationships) == 1
    assert result.relationships[0].strength == 8
    assert result.relationships[0].chunk_ids == ["c1", "c2"]


async def test_build_renders_language_placeholder() -> None:
    fake = FakeChat(entities=_entities_for, relationships=_relationships_for)
    builder = GraphBuilder(chat=fake, language="Chinese (Simplified)")
    await builder.build_graph([_chunk("c1", "Romeo and Juliet by William Shakespeare.")])

    system = fake.requests[0][0][0]
    assert "Chinese (Simplified)" in system.content
    assert "{{language}}" not in system.content


# ── build_graph: error classification ─────────────────────────────────


async def test_build_aborts_on_entity_extraction_error() -> None:
    fake = FakeChat(entities=_entities_for, entity_error=ValueError("boom"))
    builder = GraphBuilder(chat=fake)
    with pytest.raises(AIProviderError) as excinfo:
        await builder.build_graph([_chunk("c1", "Romeo and Juliet by William Shakespeare.")])
    assert excinfo.value.code == "graph.entity_extraction_failed"


async def test_build_preserves_domain_error_from_chat() -> None:
    fake = FakeChat(
        entities=_entities_for,
        entity_error=AIProviderError("provider down", code="llm.unavailable"),
    )
    builder = GraphBuilder(chat=fake)
    with pytest.raises(AIProviderError) as excinfo:
        await builder.build_graph([_chunk("c1", "some text")])
    assert excinfo.value.code == "llm.unavailable"


async def test_build_swallows_relationship_parse_failure() -> None:
    fake = FakeChat(
        entities=_entities_for,
        relationships=_relationships_for,
        relationship_raw_content="{not valid json",
    )
    builder = GraphBuilder(chat=fake)
    result = await builder.build_graph([_chunk("c1", "Romeo and Juliet by William Shakespeare.")])
    assert len(result.entities) == 2
    assert result.relationships == []


async def test_build_swallows_relationship_chat_error() -> None:
    fake = FakeChat(
        entities=_entities_for,
        relationships=_relationships_for,
        relationship_error=ValueError("boom"),
    )
    builder = GraphBuilder(chat=fake)
    result = await builder.build_graph([_chunk("c1", "Romeo and Juliet by William Shakespeare.")])
    assert len(result.entities) == 2
    assert result.relationships == []


async def test_build_splits_relationship_batches() -> None:
    def entities(user: str) -> list[dict[str, object]]:
        marker = user.strip().split()[0]
        return [
            {"title": f"{marker}-1", "type": "Concept", "description": "a"},
            {"title": f"{marker}-2", "type": "Concept", "description": "b"},
        ]

    fake = FakeChat(entities=entities, relationships=_no_relationships)
    builder = GraphBuilder(chat=fake)
    chunks = [_chunk(f"c{i}", f"t{i} text", chunk_index=i) for i in range(6)]
    result = await builder.build_graph(chunks)

    relationship_calls = sum(
        1 for messages, _ in fake.requests if messages[-1].content.startswith("Entities: ")
    )
    assert relationship_calls == 2
    assert result.relationships == []
    assert len(result.entities) == 12


async def test_build_validates_chunk_ids() -> None:
    fake = FakeChat()
    builder = GraphBuilder(chat=fake)
    with pytest.raises(ValidationError) as excinfo:
        await builder.build_graph([_TestChunk(id="  ", content="text")])
    assert excinfo.value.code == "graph.chunk_id_required"


# ── build_graph: accessors ────────────────────────────────────────────


async def _apple_scenario() -> tuple[GraphBuilder, FakeChat]:
    """Builder over Apple/Banana/Cherry, giving two differently-weighted links."""

    def entities(user: str) -> list[dict[str, object]]:
        if "Cherry" in user:
            return [
                {"title": "Apple", "type": "Concept", "description": "a fruit"},
                {"title": "Cherry", "type": "Concept", "description": "a fruit"},
            ]
        return [
            {"title": "Apple", "type": "Concept", "description": "a fruit"},
            {"title": "Banana", "type": "Concept", "description": "a fruit"},
        ]

    def relationships(_user: str) -> list[dict[str, object]]:
        return [
            {"source": "Apple", "target": "Banana", "description": "fb", "strength": 5},
            {"source": "Apple", "target": "Cherry", "description": "fc", "strength": 9},
        ]

    fake = FakeChat(entities=entities, relationships=relationships)
    builder = GraphBuilder(chat=fake)
    first = _TestChunk(id="c1", content="Apple Banana", start_at=0, end_at=12)
    second = _TestChunk(id="c2", content="Apple Cherry", start_at=12, end_at=24)
    await builder.build_graph([first, second])
    return builder, fake


async def test_get_relation_chunks_sorts_by_weight_then_degree() -> None:
    builder, _fake = await _apple_scenario()
    assert builder.get_relation_chunks("c1", 0) == ["c2", "c1"]
    assert builder.get_relation_chunks("c1", 1) == ["c2"]


async def test_get_relation_chunks_returns_all_for_zero_top_k() -> None:
    builder, _fake = await _apple_scenario()
    related = builder.get_relation_chunks("c2", 0)
    assert set(related) == {"c1", "c2"}


async def _chain_scenario() -> tuple[GraphBuilder, FakeChat]:
    """Builder over a c1-c2-c3 chain: c1 and c3 are only indirectly linked."""

    def entities(user: str) -> list[dict[str, object]]:
        if "Gamma" in user:
            return [
                {"title": "C", "type": "Concept", "description": "c"},
                {"title": "D", "type": "Concept", "description": "d"},
            ]
        if "Epsilon" in user:
            return [
                {"title": "E", "type": "Concept", "description": "e"},
                {"title": "F", "type": "Concept", "description": "f"},
            ]
        return [
            {"title": "A", "type": "Concept", "description": "a"},
            {"title": "B", "type": "Concept", "description": "b"},
        ]

    def relationships(_user: str) -> list[dict[str, object]]:
        return [
            {"source": "A", "target": "C", "description": "ac", "strength": 5},
            {"source": "C", "target": "E", "description": "ce", "strength": 8},
        ]

    fake = FakeChat(entities=entities, relationships=relationships)
    builder = GraphBuilder(chat=fake)
    first = _TestChunk(id="c1", content="Alpha Beta", start_at=0, end_at=10)
    second = _TestChunk(id="c2", content="Gamma Delta", start_at=10, end_at=21)
    third = _TestChunk(id="c3", content="Epsilon Zeta", start_at=21, end_at=33)
    await builder.build_graph([first, second, third])
    return builder, fake


async def test_get_indirect_relation_chunks_follows_second_degree_path() -> None:
    builder, _fake = await _chain_scenario()
    assert builder.get_relation_chunks("c1", 0) == ["c2"]
    assert builder.get_indirect_relation_chunks("c1", 0) == ["c3"]


async def test_get_indirect_relation_chunks_excludes_direct_neighbours() -> None:
    builder, _fake = await _chain_scenario()
    # c2 is directly adjacent to c1 and c3, so nothing is second-degree.
    assert builder.get_indirect_relation_chunks("c2", 0) == []


async def test_lookup_accessors_validate_inputs() -> None:
    fake = FakeChat()
    builder = GraphBuilder(chat=fake)
    with pytest.raises(ValidationError) as excinfo:
        builder.get_relation_chunks("", 0)
    assert excinfo.value.code == "graph.chunk_id_required"
    with pytest.raises(ValidationError) as excinfo:
        builder.get_relation_chunks("c1", -1)
    assert excinfo.value.code == "graph.invalid_top_k"
    with pytest.raises(ValidationError) as excinfo:
        builder.get_indirect_relation_chunks("", 0)
    assert excinfo.value.code == "graph.chunk_id_required"


async def test_get_all_accessors_return_snapshots() -> None:
    fake = FakeChat(entities=_entities_for, relationships=_relationships_for)
    builder = GraphBuilder(chat=fake)
    await builder.build_graph([_chunk("c1", "Romeo and Juliet by William Shakespeare.")])
    assert len(builder.get_all_entities()) == 2
    assert len(builder.get_all_relationships()) == 2


# ── Mermaid diagram ───────────────────────────────────────────────────


async def test_generate_mermaid_diagram_groups_connected_subgraphs() -> None:
    builder, _fake = await _apple_scenario()
    diagram = builder.generate_knowledge_graph_diagram()
    assert diagram.startswith("```mermaid\ngraph TD\n")
    assert diagram.endswith("```\n")
    assert diagram.count("subgraph Subgraph") == 1
    assert '    E0["Apple"]\n' in diagram
    assert '    E2["Cherry"]\n' in diagram
    assert "    E0 ==>|fc| E2\n" in diagram
    assert "    E0 -->|fb| E1\n" in diagram
    assert "  class E0 entity;\n" in diagram


async def test_generate_mermaid_diagram_empty_graph() -> None:
    fake = FakeChat()
    builder = GraphBuilder(chat=fake)
    await builder.build_graph([])
    diagram = builder.generate_knowledge_graph_diagram()
    assert diagram.startswith("```mermaid\n")
    assert "subgraph" not in diagram


# ── Integration: real persisted chunks ────────────────────────────────

# ``chunks.tenant_id`` is an INTEGER (32-bit) column; ``make_test_tenant_id``
# can exceed it, so integration rows use a local counter that stays in the
# 32-bit range. No DB cleanup runs between tests, so the counter starts at a
# random int32-safe offset per process to keep tenant ids unique across runs.
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
