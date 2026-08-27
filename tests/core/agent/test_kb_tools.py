"""Unit + integration tests for the agent KB tools and tool support layer.

Unit tests drive each tool through injected seams — a fake search runner,
fake knowledge-base loader, fake chunk store, and fake document/chunk
services — so no test touches the network, a vector store, or an LLM.
They cover parameter casting / validation, output budgeting, capability
filter derivation, the security-critical scope authorization, registry
guarding, and the rendering / dedup / rerank / MMR behaviour of the three
knowledge tools.

Integration tests run against the real applied schema (the ``chunks``
table carries an INTEGER 32-bit ``tenant_id``, so they mint ids from a
local counter). They seed a knowledge base, a document, and chunk rows,
then execute the grep and list tools through their real SQL stores.
Requires a reachable database — run with ``DATABASE_URL_OVERRIDE``.
"""

from __future__ import annotations

import itertools
import json
import re
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from random import randint
from typing import cast

import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.ai.embedding.base import Context
from src.ai.llm.types import (
    ChatOptions,
    ChatResponse,
    Message,
    StreamResponse,
)
from src.ai.retrieval.types import MatchType
from src.common.exception import (
    NotFoundError,
    PermissionDeniedError,
    ValidationError,
)
from src.common.json import JsonObject, JsonValue
from src.core.agents.tools.base import (
    TOOL_KNOWLEDGE_SEARCH,
    ToolResult,
    default_allowed_tools,
    format_match_type,
    get_relevance_level,
)
from src.core.agents.tools.capabilities import (
    KBCapabilities,
    derive_kb_filter_for_agent,
    derive_kb_filter_from_tools,
    kb_satisfies_agent_requirements,
    kb_satisfies_tool_requirements,
    tools_consume_files,
)
from src.core.agents.tools.chunk_store import SqlPagedChunkStore
from src.core.agents.tools.faq_utils import (
    FAQChunkMetadata,
    extract_chunk_match_snippet,
    faq_metadata_from_json,
)
from src.core.agents.tools.grep_tool import (
    GREP_TOOL_NAME,
    ChunkWithTitle,
    GrepChunksTool,
    SqlChunkGrepStore,
    build_grep_chunks_definition,
)
from src.core.agents.tools.kb_tool import (
    DEFAULT_TOP_K,
    KnowledgeSearchInput,
    KnowledgeSearchTool,
    ModelReranker,
    RerankItem,
    SearchCall,
    _parse_scores_from_response,
    _trim_nonnumeric,
    build_knowledge_search_definition,
)
from src.core.agents.tools.list_chunks import (
    LIST_TOOL_NAME,
    ListKnowledgeChunksTool,
    build_list_knowledge_chunks_definition,
)
from src.core.agents.tools.output_budget import (
    DEFAULT_MAX_TOOL_OUTPUT,
    split_budget_fairly,
    truncate_tool_output,
)
from src.core.agents.tools.param_utils import (
    cast_params,
    cast_value,
    format_validation_errors,
    validate_params,
)
from src.core.agents.tools.registry import ToolRegistry
from src.core.agents.tools.scope_auth import (
    authorize_chunk_in_search_targets,
    authorize_knowledge_in_search_targets,
    filter_search_results_in_search_targets,
    knowledge_ids_matching_any_tag,
    search_target_scope,
    validate_knowledge_base_ids_in_search_targets,
)
from src.core.agents.tools.search_target import SearchTarget, SearchTargets, SearchTargetType
from src.core.contracts.knowledge import Knowledge
from src.core.knowledge.documents.service.knowledge_service import KnowledgeService
from src.core.knowledge.documents.types import (
    CHANNEL_WEB,
    PARSE_STATUS_COMPLETED,
)
from src.core.knowledge.knowledge_bases.hybrid_search import SearchResult
from src.core.knowledge.knowledge_bases.service.kb_service import KBService
from src.core.knowledge.knowledge_bases.types import (
    KNOWLEDGE_BASE_TYPE_DOCUMENT,
    KNOWLEDGE_BASE_TYPE_FAQ,
    KnowledgeBaseInfo,
)
from src.core.knowledge.tags.types import TagInfo
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document
from src.settings import get_settings, reset_settings_cache

_NOW = datetime(2026, 2, 1, tzinfo=UTC)
_FAKER_SEED_MAX = 100_000_000

#: ``chunks.tenant_id`` is INTEGER (32-bit); integration tests mint ids
#: from this counter so they stay inside the range.
_INT32_TENANT_BASE = 6_000_000
_INT32_TENANT_SEQ = itertools.count(start=1)


def _int32_tenant_id() -> int:
    """Return a tenant id unique within the session, safe for INTEGER."""
    return _INT32_TENANT_BASE + next(_INT32_TENANT_SEQ)


@pytest.fixture(autouse=True)
def faker_seed() -> None:
    """Re-seed Faker per test for varied-but-reproducible generation."""
    Faker.seed(randint(1, _FAKER_SEED_MAX))


# ── Shared test doubles ───────────────────────────────────────────────


class _Context:
    """Minimal task context satisfying the ``Context`` protocol."""

    is_background_task: bool = False


def _result(
    *,
    id: str,
    content: str = "alpha beta gamma",
    knowledge_id: str = "d1",
    chunk_index: int = 1,
    knowledge_title: str = "Doc",
    score: float = 0.9,
    knowledge_base_id: str = "kb-1",
    match_type: MatchType = MatchType.EMBEDDING,
    chunk_metadata: JsonObject | None = None,
    image_info: str = "",
    parent_chunk_id: str = "",
    start_at: int = 0,
    end_at: int = 100,
    knowledge_custom_metadata: str = "",
    knowledge_source: str = "",
    knowledge_filename: str = "",
) -> SearchResult:
    """Build one hydrated search hit."""
    return SearchResult(
        id=id,
        content=content,
        knowledge_id=knowledge_id,
        chunk_index=chunk_index,
        knowledge_title=knowledge_title,
        score=score,
        match_type=match_type,
        knowledge_base_id=knowledge_base_id,
        chunk_metadata=chunk_metadata,
        image_info=image_info,
        parent_chunk_id=parent_chunk_id,
        start_at=start_at,
        end_at=end_at,
        knowledge_custom_metadata=knowledge_custom_metadata,
        knowledge_source=knowledge_source,
        knowledge_filename=knowledge_filename,
    )


class _FakeRunner:
    """Search-runner seam returning a canned result list."""

    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self._results = list(results or [])
        self.calls: list[SearchCall] = []

    async def search(self, ctx: Context, call: SearchCall) -> list[SearchResult]:
        self.calls.append(call)
        return list(self._results)


class _FakeKbLoader:
    """Knowledge-base loader seam returning canned ``KnowledgeBaseInfo``."""

    def __init__(self, kbs: list[KnowledgeBaseInfo] | None = None) -> None:
        self._kbs = list(kbs or [])

    async def load_by_ids(self, ids: list[str]) -> list[KnowledgeBaseInfo]:
        return [kb for kb in self._kbs if kb.id in ids]


class _FakeChunkCounter:
    """Chunk-counting seam for the retrieval-statistics totals."""

    def __init__(self, total: int = 0) -> None:
        self._total = total
        self.calls: list[tuple[int, str]] = []

    async def count_chunks(self, *, tenant_id: int, knowledge_id: str) -> int:
        self.calls.append((tenant_id, knowledge_id))
        return self._total


class _RecordingReranker:
    """Model-rerank seam returning a fixed ranking."""

    def __init__(self, scores: list[tuple[int, float]] | None = None) -> None:
        self._scores = scores or []
        self.calls: list[tuple[str, list[str]]] = []

    async def rerank(self, query: str, documents: list[str]) -> list[RerankItem]:
        self.calls.append((query, documents))
        return [
            cast(
                "RerankItem",
                _SimpleRankItem(index=index, relevance_score=score),
            )
            for index, score in self._scores
        ]


class _SimpleRankItem:
    """Plain carrier exposing the rerank fields the tool consumes."""

    def __init__(self, *, index: int, relevance_score: float) -> None:
        self.index = index
        self.relevance_score = relevance_score


class _RecordingChat:
    """LLM seam satisfying the ``Chat`` protocol."""

    def __init__(self, response_text: str = "Passage 1: 0.9\nPassage 2: 0.4\n") -> None:
        self._response_text = response_text
        self.calls: list[tuple[str, int]] = []

    async def chat(
        self,
        messages: list[Message],
        opts: ChatOptions | None = None,
    ) -> ChatResponse:
        user = next(message for message in messages if message.role == "user")
        self.calls.append((user.content, opts.max_tokens if opts else 0))
        return ChatResponse(content=self._response_text)

    async def chat_stream(
        self,
        messages: list[Message],
        opts: ChatOptions | None = None,
    ) -> AsyncIterator[StreamResponse]:
        """Not used by the rerank path; yields no events."""
        return
        yield  # pragma: no cover

    def get_model_name(self) -> str:
        return "test-chat"

    def get_model_id(self) -> str:
        return "chat-1"


def _kb(
    id: str = "kb-1",
    type: str = KNOWLEDGE_BASE_TYPE_DOCUMENT,
    embedding_model_id: str = "em-1",
) -> KnowledgeBaseInfo:
    """Build one knowledge-base info record."""
    return KnowledgeBaseInfo(
        id=id,
        name="Test KB",
        type=type,
        tenant_id=7,
        embedding_model_id=embedding_model_id,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _knowledge(
    id: str = "d1",
    knowledge_base_id: str = "kb-1",
    title: str = "Doc",
    file_name: str = "doc.pdf",
) -> Knowledge:
    """Build one document contract record."""
    return Knowledge(
        id=id,
        tenant_id=7,
        knowledge_base_id=knowledge_base_id,
        type="file",
        title=title,
        source="doc.pdf",
        channel=CHANNEL_WEB,
        parse_status=PARSE_STATUS_COMPLETED,
        enable_status="enabled",
        file_name=file_name,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _chunk_row(
    id: str = "c1",
    *,
    knowledge_id: str = "d1",
    knowledge_base_id: str = "kb-1",
    tenant_id: int = 7,
    content: str = "alpha beta gamma",
    chunk_index: int = 1,
    chunk_type: str = "text",
    is_enabled: bool = True,
    image_info: str | None = None,
    metadata: JsonObject | None = None,
) -> Chunk:
    """Build one chunk storage row."""
    return Chunk(
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        knowledge_id=knowledge_id,
        content=content,
        chunk_index=chunk_index,
        is_enabled=is_enabled,
        start_at=0,
        end_at=len(content),
        chunk_type=chunk_type,
        parent_chunk_id=None,
        image_info=image_info,
        metadata=metadata,
        status=1,
        flags=1,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _tag(id: str = "t1") -> TagInfo:
    """Build one tag record."""
    return TagInfo(
        id=id,
        seq_id=1,
        tenant_id=7,
        knowledge_base_id="kb-1",
        name="tag",
        created_at=_NOW,
        updated_at=_NOW,
    )


def _targets(*targets: SearchTarget) -> SearchTargets:
    return SearchTargets(targets=tuple(targets))


def _kb_target(
    knowledge_base_id: str = "kb-1",
    tenant_id: int = 7,
    *,
    knowledge_ids: tuple[str, ...] = (),
    tag_ids: tuple[str, ...] = (),
) -> SearchTarget:
    return SearchTarget(
        type=SearchTargetType.KNOWLEDGE_BASE,
        knowledge_base_id=knowledge_base_id,
        tenant_id=tenant_id,
        knowledge_ids=knowledge_ids,
        tag_ids=tag_ids,
    )


def _kb_tool(
    *,
    runner: _FakeRunner,
    targets: SearchTargets | None = None,
    kb_loader: _FakeKbLoader | None = None,
    chunk_counter: _FakeChunkCounter | None = None,
    reranker: ModelReranker | None = None,
    chat: _RecordingChat | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> KnowledgeSearchTool:
    return KnowledgeSearchTool(
        definition=build_knowledge_search_definition(),
        search_targets=targets or _targets(_kb_target()),
        runner=runner,
        kb_loader=kb_loader,
        chunk_counter=chunk_counter,
        reranker=reranker,
        chat=chat,
        top_k=top_k,
    )


def _list_json(result: ToolResult, key: str) -> list[JsonObject]:
    """Narrow a JSON list payload of a tool result for assertions."""
    return cast("list[JsonObject]", result.data[key])


# ── Tool base constants / formatters ─────────────────────────────────


def test_relevance_level_boundaries() -> None:
    assert get_relevance_level(0.95) == "High Relevance"
    assert get_relevance_level(0.7) == "Medium Relevance"
    assert get_relevance_level(0.5) == "Low Relevance"
    assert get_relevance_level(0.1) == "Weak Relevance"


def test_format_match_type_labels() -> None:
    assert format_match_type(MatchType.EMBEDDING) == "Vector Match"
    assert format_match_type(MatchType.KEYWORDS) == "Keyword Match"
    assert format_match_type(MatchType.PARENT_CHUNK) == "Parent Chunk Match"


def test_default_allowed_tools_contains_kb_tools() -> None:
    allowed = default_allowed_tools()
    assert TOOL_KNOWLEDGE_SEARCH in allowed
    assert GREP_TOOL_NAME in allowed
    assert LIST_TOOL_NAME in allowed


# ── Parameter casting / validation ───────────────────────────────────


def test_cast_params_boolean_string_to_bool() -> None:
    schema = json.dumps({"type": "object", "properties": {"flag": {"type": "boolean"}}})
    assert cast_params('{"flag": "true"}', schema) == '{"flag": true}'


def test_cast_params_integer_string_to_int() -> None:
    schema = json.dumps({"type": "object", "properties": {"limit": {"type": "integer"}}})
    assert cast_params('{"limit": "7"}', schema) == '{"limit": 7}'


def test_cast_params_number_string_to_float() -> None:
    schema = json.dumps({"type": "object", "properties": {"threshold": {"type": "number"}}})
    assert cast_params('{"threshold": "0.6"}', schema) == '{"threshold": 0.6}'


def test_cast_params_string_from_bool_and_number() -> None:
    schema = json.dumps({"type": "object", "properties": {"name": {"type": "string"}}})
    assert cast_params('{"name": true}', schema) == '{"name": "true"}'
    assert cast_params('{"name": 3}', schema) == '{"name": "3"}'


def test_cast_params_array_from_json_string() -> None:
    schema = json.dumps({"type": "object", "properties": {"queries": {"type": "array"}}})
    args = json.dumps({"queries": '["a", "b"]'})
    assert cast_params(args, schema) == '{"queries": ["a", "b"]}'


def test_cast_params_no_change_returns_original() -> None:
    schema = json.dumps({"type": "object", "properties": {"n": {"type": "integer"}}})
    args = '{"n": 3, "other": "x"}'
    assert cast_params(args, schema) == args


def test_cast_value_unknown_type_passthrough() -> None:
    assert cast_value("anything", "unknown") == ("anything", False)


def test_validate_params_required_missing() -> None:
    schema = json.dumps(
        {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    )
    errors = validate_params("{}", schema)
    assert len(errors) == 1
    assert "required parameter 'q' is missing" in errors[0].message


def test_validate_params_type_mismatch() -> None:
    schema = json.dumps({"type": "object", "properties": {"q": {"type": "string"}}})
    errors = validate_params('{"q": 3}', schema)
    assert len(errors) == 1
    assert "should be type 'string'" in errors[0].message


def test_validate_params_enum_and_bounds() -> None:
    schema = json.dumps(
        {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["a", "b"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                "label": {"type": "string", "minLength": 2, "maxLength": 5},
            },
        }
    )
    errors = validate_params('{"kind": "z", "limit": 20, "label": "x"}', schema)
    messages = [error.message for error in errors]
    assert any("must be one of [a, b]" in message for message in messages)
    assert any("must be <= 10" in message for message in messages)
    assert any("must have at least 2 characters" in message for message in messages)


def test_validate_params_extra_params_allowed() -> None:
    schema = json.dumps({"type": "object", "properties": {"q": {"type": "string"}}})
    assert validate_params('{"q": "x", "extra": true}', schema) == []


def test_format_validation_errors_joins_messages() -> None:
    from src.core.agents.tools.param_utils import ValidationError

    text = format_validation_errors(
        [
            ValidationError(param="a", message="first"),
            ValidationError(param="b", message="second"),
        ]
    )
    assert text == "Parameter validation failed: first; second"


# ── Output budget / truncation ───────────────────────────────────────


def test_truncate_tool_output_within_limit_unchanged() -> None:
    text = "short"
    assert truncate_tool_output(text, 100) == text


def test_truncate_tool_output_preserves_head_and_tail() -> None:
    body = "x" * 400
    result = truncate_tool_output(body, 300)
    assert len(result) <= 300 + 200
    assert "[output truncated:" in result
    assert result.startswith("x" * 70)
    assert result.endswith("x" * 30)


def test_split_budget_fairly_water_filling() -> None:
    caps = split_budget_fairly(100, [10, 90, 50])
    assert caps[0] == 10
    assert sum(caps) <= 100
    assert caps[1] <= 90


def test_output_budget_falls_back_to_default() -> None:
    from src.core.agents.tools.output_budget import output_budget

    assert output_budget() == DEFAULT_MAX_TOOL_OUTPUT


# ── Capability filter derivation ─────────────────────────────────────


def test_derive_kb_filter_from_tools_any_of_union() -> None:
    kb_filter = derive_kb_filter_from_tools([TOOL_KNOWLEDGE_SEARCH, "wiki_search"])
    assert {cap.value for cap in kb_filter.any_of} == {"vector", "keyword", "wiki"}


def test_derive_kb_filter_from_tools_empty_for_no_requirements() -> None:
    assert derive_kb_filter_from_tools(["thinking"]).is_empty()
    assert derive_kb_filter_from_tools([]).is_empty()


def test_kb_satisfies_tool_requirements() -> None:
    caps = KBCapabilities(vector=True, keyword=False)
    assert kb_satisfies_tool_requirements(caps, [TOOL_KNOWLEDGE_SEARCH])
    assert not kb_satisfies_tool_requirements(KBCapabilities(wiki=True), [TOOL_KNOWLEDGE_SEARCH])
    assert kb_satisfies_tool_requirements(KBCapabilities(), ["thinking"])


def test_quick_answer_mode_forces_vector_or_keyword() -> None:
    kb_filter = derive_kb_filter_for_agent("quick-answer", ["thinking"])
    assert {cap.value for cap in kb_filter.any_of} == {"vector", "keyword"}
    assert kb_satisfies_agent_requirements(
        KBCapabilities(keyword=True), "quick-answer", ["thinking"]
    )
    assert not kb_satisfies_agent_requirements(
        KBCapabilities(wiki=True), "quick-answer", ["thinking"]
    )


def test_tools_consume_files_permissive_fallback() -> None:
    assert tools_consume_files([])
    assert tools_consume_files([TOOL_KNOWLEDGE_SEARCH])
    assert not tools_consume_files(["thinking"])
    assert tools_consume_files(["some_custom_tool"])


# ── Scope authorization (security-critical) ──────────────────────────


class _FakeKnowledgeLookup:
    """Document lookup seam returning a canned row or ``None``."""

    def __init__(self, knowledge: Knowledge | None) -> None:
        self._knowledge = knowledge

    async def get_document_by_id_only(self, *, id: str) -> Knowledge | None:
        if self._knowledge is not None and self._knowledge.id == id:
            return self._knowledge
        return None


class _FakeChunkLookup:
    """Chunk lookup seam returning a canned row or ``None``."""

    def __init__(self, chunk: Chunk | None) -> None:
        self._chunk = chunk

    async def get_chunk_by_id_only(self, *, id: str) -> Chunk | None:
        if self._chunk is not None and self._chunk.id == id:
            return self._chunk
        return None


class _FakeTagFetcher:
    """Tag fetcher seam mapping documents to their tags."""

    def __init__(self, tags: dict[str, list[TagInfo]]) -> None:
        self._tags = tags

    async def get_knowledge_tags(self, knowledge_ids: list[str]) -> dict[str, list[TagInfo]]:
        return {
            knowledge_id: list(self._tags.get(knowledge_id, [])) for knowledge_id in knowledge_ids
        }


async def test_authorize_knowledge_whole_kb_target_allows() -> None:
    ctx = _Context()
    targets = _targets(_kb_target())
    knowledge = await authorize_knowledge_in_search_targets(
        ctx, targets, "d1", _FakeKnowledgeLookup(_knowledge())
    )
    assert knowledge.id == "d1"


async def test_authorize_knowledge_explicit_document_allows() -> None:
    ctx = _Context()
    targets = _targets(_kb_target(knowledge_ids=("d1",)))
    knowledge = await authorize_knowledge_in_search_targets(
        ctx, targets, "d1", _FakeKnowledgeLookup(_knowledge())
    )
    assert knowledge.id == "d1"


async def test_authorize_knowledge_out_of_scope_document_denied() -> None:
    ctx = _Context()
    targets = _targets(_kb_target(knowledge_ids=("d9",)))
    with pytest.raises(PermissionDeniedError):
        await authorize_knowledge_in_search_targets(
            ctx, targets, "d1", _FakeKnowledgeLookup(_knowledge())
        )


async def test_authorize_knowledge_tag_scope_allows() -> None:
    ctx = _Context()
    targets = _targets(_kb_target(tag_ids=("t1",)))
    knowledge = await authorize_knowledge_in_search_targets(
        ctx,
        targets,
        "d1",
        _FakeKnowledgeLookup(_knowledge()),
        tag_fetcher=_FakeTagFetcher({"d1": [_tag("t1")]}),
    )
    assert knowledge.id == "d1"


async def test_authorize_knowledge_tag_scope_denies_unbound_document() -> None:
    ctx = _Context()
    targets = _targets(_kb_target(tag_ids=("t2",)))
    with pytest.raises(PermissionDeniedError):
        await authorize_knowledge_in_search_targets(
            ctx,
            targets,
            "d1",
            _FakeKnowledgeLookup(_knowledge()),
            tag_fetcher=_FakeTagFetcher({"d1": [_tag("t1")]}),
        )


async def test_authorize_knowledge_missing_document_raises_not_found() -> None:
    ctx = _Context()
    targets = _targets(_kb_target())
    with pytest.raises(NotFoundError):
        await authorize_knowledge_in_search_targets(
            ctx, targets, "missing", _FakeKnowledgeLookup(None)
        )


async def test_authorize_knowledge_blank_id_raises_validation() -> None:
    ctx = _Context()
    with pytest.raises(ValidationError):
        await authorize_knowledge_in_search_targets(
            ctx, _targets(_kb_target()), "  ", _FakeKnowledgeLookup(_knowledge())
        )


async def test_authorize_knowledge_kb_out_of_scope_denied() -> None:
    ctx = _Context()
    targets = _targets(_kb_target(knowledge_base_id="kb-1"))
    with pytest.raises(PermissionDeniedError):
        await authorize_knowledge_in_search_targets(
            ctx,
            targets,
            "d1",
            _FakeKnowledgeLookup(_knowledge(knowledge_base_id="kb-other")),
        )


async def test_authorize_chunk_disabled_denied() -> None:
    ctx = _Context()
    targets = _targets(_kb_target())
    chunk = _chunk_row(is_enabled=False)
    with pytest.raises(PermissionDeniedError):
        await authorize_chunk_in_search_targets(
            ctx, targets, chunk.id, _FakeChunkLookup(chunk), _FakeKnowledgeLookup(_knowledge())
        )


async def test_authorize_chunk_allows_whole_kb() -> None:
    ctx = _Context()
    targets = _targets(_kb_target())
    chunk = await authorize_chunk_in_search_targets(
        ctx, targets, "c1", _FakeChunkLookup(_chunk_row()), _FakeKnowledgeLookup(_knowledge())
    )
    assert chunk.id == "c1"


async def test_validate_knowledge_base_ids_in_search_targets_rejects_foreign() -> None:
    with pytest.raises(PermissionDeniedError):
        validate_knowledge_base_ids_in_search_targets(_targets(_kb_target()), ["kb-1", "kb-9"])


async def test_filter_search_results_in_search_targets_whole_kb_passthrough() -> None:
    ctx = _Context()
    targets = _targets(_kb_target())
    results = [_result(id="c1", knowledge_id="d1")]
    filtered = await filter_search_results_in_search_targets(
        ctx, targets, "kb-1", results, _FakeKnowledgeLookup(_knowledge())
    )
    assert len(filtered) == 1


async def test_filter_search_results_in_search_targets_document_and_tag_union() -> None:
    ctx = _Context()
    # Alternatives ACROSS targets remain a union: one target authorizes the
    # explicit document, a second authorizes everything carrying the tag.
    targets = _targets(
        _kb_target(knowledge_ids=("d1",)),
        _kb_target(tag_ids=("t1",)),
    )
    results = [
        _result(id="c1", knowledge_id="d1"),
        _result(id="c2", knowledge_id="d2"),
    ]
    filtered = await filter_search_results_in_search_targets(
        ctx,
        targets,
        "kb-1",
        results,
        _FakeKnowledgeLookup(_knowledge()),
        tag_fetcher=_FakeTagFetcher({"d2": [_tag("t1")]}),
    )
    assert {item.knowledge_id for item in filtered} == {"d1", "d2"}


async def test_filter_search_results_in_search_targets_unmatched_kb_denied() -> None:
    ctx = _Context()
    targets = _targets(_kb_target())
    with pytest.raises(PermissionDeniedError):
        await filter_search_results_in_search_targets(
            ctx, targets, "kb-other", [], _FakeKnowledgeLookup(_knowledge())
        )


async def test_knowledge_ids_matching_any_tag_batches() -> None:
    ctx = _Context()
    matches = await knowledge_ids_matching_any_tag(
        ctx,
        ["d1", "d2"],
        ["t1", "t1", "t3"],
        _FakeTagFetcher({"d1": [_tag("t1")]}).get_knowledge_tags,
    )
    assert matches == {"d1": True}


def test_search_target_scope_intersection_semantics() -> None:
    target = SearchTarget(
        type=SearchTargetType.KNOWLEDGE_BASE,
        knowledge_base_id="kb-1",
        tenant_id=7,
        knowledge_ids=("d1", "d1"),
        tag_ids=("t1",),
    )
    knowledge_ids, tag_ids = search_target_scope(target)
    assert knowledge_ids == ["d1"]
    assert tag_ids == []


# ── Tool registry ────────────────────────────────────────────────────


class _StubTool:
    """Minimal tool implementing the ``Tool`` protocol."""

    def __init__(self, name: str, schema: str = "{}", result: ToolResult | None = None) -> None:
        self._name = name
        self._schema = schema
        self._result = result or ToolResult(success=True, output="ok")

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return f"desc-{self._name}"

    def parameters(self) -> str:
        return self._schema

    async def execute(self, ctx: Context, args: str) -> ToolResult:
        return self._result


async def test_registry_first_wins_registration() -> None:
    registry = ToolRegistry()
    registry.register_tool(_StubTool("a", result=ToolResult(output="first")))
    registry.register_tool(_StubTool("a", result=ToolResult(output="second")))
    result = await registry.execute_tool(_Context(), "a", "{}")
    assert result.output == "first"


async def test_registry_get_tool_not_found_raises() -> None:
    registry = ToolRegistry()
    with pytest.raises(NotFoundError):
        registry.get_tool("nope")


def test_registry_list_tools_sorted() -> None:
    registry = ToolRegistry()
    registry.register_tool(_StubTool("zeta"))
    registry.register_tool(_StubTool("alpha"))
    assert registry.list_tools() == ["alpha", "zeta"]


async def test_registry_execute_validation_failure_short_circuits() -> None:
    schema = json.dumps(
        {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    )
    registry = ToolRegistry()
    registry.register_tool(_StubTool("t", schema=schema))
    result = await registry.execute_tool(_Context(), "t", "{}")
    assert not result.success
    assert "required parameter 'q' is missing" in result.error


async def test_registry_execute_truncates_oversized_output() -> None:
    registry = ToolRegistry()
    registry.set_max_tool_output_size(500)
    registry.register_tool(_StubTool("t", result=ToolResult(output="x" * 1000)))
    result = await registry.execute_tool(_Context(), "t", "{}")
    assert "[output truncated:" in result.output


# ── knowledge_search tool ────────────────────────────────────────────


async def test_kb_tool_missing_queries_returns_error() -> None:
    tool = _kb_tool(runner=_FakeRunner())
    result = await tool.execute(_Context(), "{}")
    assert not result.success
    assert result.error == "queries parameter is required"


async def test_kb_tool_no_search_targets_returns_error() -> None:
    tool = _kb_tool(runner=_FakeRunner(), targets=_targets())
    result = await tool.execute(_Context(), '{"queries": ["q"]}')
    assert not result.success
    assert "no search targets" in result.error


async def test_kb_tool_out_of_scope_kb_denied() -> None:
    tool = _kb_tool(runner=_FakeRunner())
    with pytest.raises(PermissionDeniedError):
        await tool.execute(_Context(), '{"queries": ["q"], "knowledge_base_ids": ["kb-9"]}')


async def test_kb_tool_basic_execute_output_and_data() -> None:
    tool = _kb_tool(runner=_FakeRunner([_result(id="c1")]))
    result = await tool.execute(_Context(), '{"queries": ["alpha"]}')
    assert result.success
    assert '<search_results count="1">' in result.output
    assert "<query>alpha</query>" in result.output
    assert 'chunk_id="c1"' in result.output
    assert result.data["count"] == 1
    assert result.data["display_type"] == "search_results"
    assert result.data["kb_counts"] == {"kb-1": 1}


async def test_kb_tool_empty_results_render_next_steps() -> None:
    tool = _kb_tool(runner=_FakeRunner([]))
    result = await tool.execute(_Context(), '{"queries": ["nothing"]}')
    assert result.success
    assert result.data["count"] == 0
    assert "No relevant content found" in result.output
    assert "DO NOT use training data" in result.output


async def test_kb_tool_deduplicates_by_id_keeping_highest_score() -> None:
    runner = _FakeRunner(
        [
            _result(id="c1", score=0.9, content="alpha beta", knowledge_id="d1"),
            _result(id="c1", score=0.4, content="alpha beta", knowledge_id="d1"),
            _result(id="c2", score=0.8, content="gamma delta", knowledge_id="d2"),
        ]
    )
    tool = _kb_tool(runner=runner)
    result = await tool.execute(_Context(), '{"queries": ["q"]}')
    assert result.data["count"] == 2
    output = result.output
    assert output.index('chunk_id="c1"') < output.index('chunk_id="c2"')


async def test_kb_tool_deduplicates_by_content_signature() -> None:
    runner = _FakeRunner(
        [
            _result(id="c1", content="alpha beta", knowledge_id="d1"),
            _result(id="c2", content="alpha  beta", knowledge_id="d2"),
        ]
    )
    tool = _kb_tool(runner=runner)
    result = await tool.execute(_Context(), '{"queries": ["q"]}')
    assert result.data["count"] == 1


async def test_kb_tool_already_seen_compact_rendering() -> None:
    runner = _FakeRunner([_result(id="c1", content="alpha beta gamma")])
    tool = _kb_tool(runner=runner)
    first = await tool.execute(_Context(), '{"queries": ["alpha"]}')
    second = await tool.execute(_Context(), '{"queries": ["alpha"]}')
    assert 'already_seen="true"' not in first.output
    assert 'already_seen="true"' in second.output
    assert "content omitted" in second.output


async def test_kb_tool_faq_result_renders_metadata() -> None:
    faq_meta: JsonObject = {
        "standard_question": "What is RAG?",
        "similar_questions": cast("list[JsonValue]", ["What does RAG stand for?"]),
        "answers": cast("list[JsonValue]", ["Retrieval augmented generation."]),
    }
    loader = _FakeKbLoader([_kb(type=KNOWLEDGE_BASE_TYPE_FAQ)])
    runner = _FakeRunner(
        [
            _result(
                id="c1",
                content="",
                chunk_metadata=faq_meta,
                chunk_index=3,
                knowledge_title="FAQ doc",
            )
        ]
    )
    tool = _kb_tool(runner=runner, kb_loader=loader)
    result = await tool.execute(_Context(), '{"queries": ["what is rag"]}')
    assert result.success
    assert '<faq rank="1" faq_id="c1"' in result.output
    assert "<question>What is RAG?</question>" in result.output
    assert "<answer>Retrieval augmented generation.</answer>" in result.output
    entry = _list_json(result, "results")[0]
    assert entry["faq_id"] == "c1"
    assert entry["faq_standard_question"] == "What is RAG?"
    assert entry["faq_answers"] == ["Retrieval augmented generation."]


async def test_kb_tool_retrieval_statistics_uses_chunk_counter() -> None:
    counter = _FakeChunkCounter(total=10)
    tool = _kb_tool(
        runner=_FakeRunner([_result(id="c1", knowledge_id="d1")]),
        chunk_counter=counter,
    )
    result = await tool.execute(_Context(), '{"queries": ["q"]}')
    assert 'total_chunks="10"' in result.output
    assert 'retrieved="1"' in result.output
    assert 'coverage="10.0%"' in result.output
    assert counter.calls == [(7, "d1")]


async def test_kb_tool_rerank_model_applies_composite_score() -> None:
    reranker = _RecordingReranker(scores=[(0, 0.8), (1, 0.9)])
    runner = _FakeRunner(
        [
            _result(id="c1", score=0.9, knowledge_id="d1", content="alpha beta"),
            _result(id="c2", score=0.5, knowledge_id="d2", content="gamma delta"),
        ]
    )
    tool = _kb_tool(runner=runner, reranker=reranker)
    result = await tool.execute(_Context(), '{"queries": ["q"]}')
    # Both rerank scores sit above the default 0.3 threshold.
    assert result.data["count"] == 2
    assert reranker.calls == [("q", ["alpha beta", "gamma delta"])]


async def test_kb_tool_rerank_drops_below_threshold_without_preserve() -> None:
    reranker = _RecordingReranker(scores=[(0, 0.9), (1, 0.1)])
    runner = _FakeRunner(
        [
            _result(id="c1", score=0.9, knowledge_id="d1"),
            _result(id="c2", score=0.5, knowledge_id="d2"),
        ]
    )
    tool = _kb_tool(runner=runner, reranker=reranker)
    result = await tool.execute(_Context(), '{"queries": ["q"]}')
    # The second result scores 0.1 (< 0.3 threshold) and is dropped; only c1
    # survives reranking and the sorted output carries it.
    assert result.data["count"] == 1
    assert _list_json(result, "results")[0]["knowledge_id"] == "d1"


async def test_kb_tool_rerank_model_failure_falls_back_to_original() -> None:
    class _BoomReranker:
        async def rerank(self, query: str, documents: list[str]) -> list[RerankItem]:
            raise RuntimeError("boom")

    runner = _FakeRunner([_result(id="c1", score=0.9)])
    tool = _kb_tool(runner=runner, reranker=cast("ModelReranker", _BoomReranker()))
    result = await tool.execute(_Context(), '{"queries": ["q"]}')
    assert result.success
    assert result.data["count"] == 1


async def test_kb_tool_llm_rerank_uses_chat_seam() -> None:
    chat = _RecordingChat(response_text="Passage 1: 0.9\nPassage 2: 0.4\n")
    runner = _FakeRunner(
        [
            _result(id="c1", score=0.9, knowledge_id="d1", content="alpha beta"),
            _result(id="c2", score=0.5, knowledge_id="d2", content="gamma delta"),
        ]
    )
    tool = _kb_tool(runner=runner, chat=chat)
    result = await tool.execute(_Context(), '{"queries": ["q"]}')
    assert result.data["count"] == 2
    assert chat.calls


async def test_kb_tool_apply_mmr_reduces_to_top_k() -> None:
    runner = _FakeRunner(
        [
            _result(
                id=f"c{i}",
                content=f"unique term {i}",
                knowledge_id=f"d{i}",
                chunk_index=i,
                score=0.9 - i * 0.01,
            )
            for i in range(8)
        ]
    )
    tool = _kb_tool(runner=runner, top_k=DEFAULT_TOP_K)
    result = await tool.execute(_Context(), '{"queries": ["q"]}')
    assert result.data["count"] == DEFAULT_TOP_K


async def test_kb_tool_non_searchable_kb_filtered() -> None:
    wiki_kb = _kb(id="kb-wiki", type="wiki", embedding_model_id="")
    wiki_kb = KnowledgeBaseInfo(
        **{
            **wiki_kb.model_dump(),
            "indexing_strategy": {"vector_enabled": False, "keyword_enabled": False},
        }
    )
    targets = _targets(_kb_target(knowledge_base_id="kb-wiki", tenant_id=7))
    runner = _FakeRunner([_result(id="c1", knowledge_base_id="kb-wiki")])
    tool = _kb_tool(runner=runner, targets=targets, kb_loader=_FakeKbLoader([wiki_kb]))
    result = await tool.execute(_Context(), '{"queries": ["q"]}')
    # The only KB in scope is wiki-only, so no retrieval runs.
    assert result.data["count"] == 0
    assert runner.calls == []


def test_kb_tool_input_parsing() -> None:
    input_ = KnowledgeSearchInput.from_json({"queries": ["a", "b"], "knowledge_base_ids": ["kb-1"]})
    assert input_.queries == ("a", "b")
    assert input_.knowledge_base_ids == ("kb-1",)
    assert KnowledgeSearchInput.from_json({}).queries == ()


def test_kb_tool_parse_scores_from_response() -> None:
    scores = _parse_scores_from_response("Passage 1: 0.85\nPassage 2: 1.5\nPassage 3: -0.2", 3)
    assert scores is not None
    # The minus sign is stripped by the numeric trim, mirroring upstream.
    assert scores == [0.85, 1.0, 0.2]


def test_trim_nonnumeric_strips_letters() -> None:
    assert _trim_nonnumeric("score=0.85 points") == "0.85"
    assert _trim_nonnumeric("0.85") == "0.85"


async def test_kb_tool_user_specified_kb_filters_targets() -> None:
    targets = _targets(
        _kb_target(knowledge_base_id="kb-1", tenant_id=7),
        _kb_target(knowledge_base_id="kb-2", tenant_id=8),
    )
    runner = _FakeRunner([_result(id="c1", knowledge_base_id="kb-1")])
    tool = _kb_tool(runner=runner, targets=targets)
    result = await tool.execute(_Context(), '{"queries": ["q"], "knowledge_base_ids": ["kb-1"]}')
    assert result.success
    assert all(call.knowledge_base_ids == ("kb-1",) for call in runner.calls)


# ── grep_chunks tool ─────────────────────────────────────────────────


class _FakeGrepStore:
    """Chunk-grep store seam returning canned rows."""

    def __init__(self, rows: list[ChunkWithTitle] | None = None) -> None:
        self._rows = list(rows or [])
        self.calls: list[tuple[str, list[str], list[str], list[SearchTarget], dict[str, int]]] = []

    async def search_chunks(
        self,
        *,
        query: str,
        full_kb_ids: list[str],
        knowledge_ids: list[str],
        tag_targets: list[SearchTarget],
        kb_tenant_map: dict[str, int],
    ) -> list[ChunkWithTitle]:
        self.calls.append((query, full_kb_ids, knowledge_ids, tag_targets, kb_tenant_map))
        return list(self._rows)


def _grep_tool(
    *,
    store: _FakeGrepStore,
    targets: SearchTargets | None = None,
) -> GrepChunksTool:
    return GrepChunksTool(
        definition=build_grep_chunks_definition(),
        store=store,
        search_targets=targets or _targets(_kb_target()),
    )


async def test_grep_tool_missing_query_returns_error() -> None:
    tool = _grep_tool(store=_FakeGrepStore())
    result = await tool.execute(_Context(), "{}")
    assert not result.success
    assert "query parameter is required" in result.error


async def test_grep_tool_invalid_regex_returns_error() -> None:
    tool = _grep_tool(store=_FakeGrepStore())
    result = await tool.execute(_Context(), '{"query": "["}')
    assert not result.success
    assert "invalid regex query" in result.error


async def test_grep_tool_basic_execute() -> None:
    row = ChunkWithTitle(
        chunk=_chunk_row(id="c1", content="the engine is stardust"),
        knowledge_title="Manual",
    )
    tool = _grep_tool(store=_FakeGrepStore([row]))
    result = await tool.execute(_Context(), '{"query": "engine|stardust"}')
    assert result.success
    assert '<grep_results chunk_count="1">' in result.output
    assert "<query>engine|stardust</query>" in result.output
    assert 'chunk_id="c1"' in result.output
    assert "<match_snippet>" in result.output
    assert result.data["display_type"] == "grep_results"
    assert result.data["result_count"] == 1
    assert result.data["document_count"] == 1
    assert _list_json(result, "chunk_results")[0]["chunk_id"] == "c1"


async def test_grep_tool_title_match_ranks_first() -> None:
    body_row = ChunkWithTitle(
        chunk=_chunk_row(id="c1", content="rare mention", knowledge_id="d1"),
        knowledge_title="Ordinary",
    )
    title_row = ChunkWithTitle(
        chunk=_chunk_row(id="c2", content="nothing here", knowledge_id="d2"),
        knowledge_title="图片素材",
    )
    tool = _grep_tool(store=_FakeGrepStore([body_row, title_row]))
    result = await tool.execute(_Context(), '{"query": "素材"}')
    # The title-matching document floats above the body-only hit.
    assert _list_json(result, "chunk_results")[0]["chunk_id"] == "c2"


async def test_grep_tool_already_seen_compact_rendering() -> None:
    row = ChunkWithTitle(
        chunk=_chunk_row(id="c1", content="engine stardust", knowledge_id="d1"),
        knowledge_title="Doc",
    )
    tool = _grep_tool(store=_FakeGrepStore([row]))
    first = await tool.execute(_Context(), '{"query": "engine"}')
    second = await tool.execute(_Context(), '{"query": "engine"}')
    assert 'already_seen="true"' not in first.output
    assert 'already_seen="true"' in second.output
    assert "snippet omitted" in second.output


async def test_grep_tool_aggregates_by_knowledge() -> None:
    rows = [
        ChunkWithTitle(
            chunk=_chunk_row(id="c1", content="stardust engine", knowledge_id="d1"),
            knowledge_title="Doc A",
            total_chunk_count=3,
        ),
        ChunkWithTitle(
            chunk=_chunk_row(id="c2", content="stardust", knowledge_id="d1", chunk_index=2),
            knowledge_title="Doc A",
            total_chunk_count=3,
        ),
    ]
    tool = _grep_tool(store=_FakeGrepStore(rows))
    result = await tool.execute(_Context(), '{"query": "stardust"}')
    aggregated = _list_json(result, "knowledge_results")
    assert len(aggregated) == 1
    assert aggregated[0]["chunk_hit_count"] == 2
    assert aggregated[0]["total_chunk_count"] == 3
    assert aggregated[0]["total_pattern_hits"] == 2


async def test_grep_tool_scope_skips_unscoped_kb() -> None:
    targets = _targets(_kb_target(knowledge_base_id="kb-1", tenant_id=7))
    tool = _grep_tool(store=_FakeGrepStore(), targets=targets)
    result = await tool.execute(_Context(), '{"query": "engine"}')
    assert result.success
    assert result.data["chunk_results"] == []


def test_extract_chunk_match_snippet_faq_uses_metadata() -> None:
    chunk = _chunk_row(
        id="c1",
        chunk_type="faq",
        content="",
        metadata={
            "standard_question": "What is RAG?",
            "answers": ["Retrieval augmented generation."],
        },
    )
    compiled = [re.compile(r"(?i)rag")]
    snippet = extract_chunk_match_snippet(chunk, compiled)
    assert snippet.startswith("Q: What is RAG?")
    assert "Retrieval augmented generation" in snippet


def test_faq_metadata_parsing_round_trip() -> None:
    meta = faq_metadata_from_json(
        {
            "standard_question": "Q",
            "similar_questions": ["S1", "S2", "S3", "S4", "S5", "S6"],
            "answers": ["A"],
        }
    )
    assert meta is not None
    assert isinstance(meta, FAQChunkMetadata)
    assert meta.standard_question == "Q"
    assert meta.answers == ("A",)


# ── list_knowledge_chunks tool ───────────────────────────────────────


class _FakeChunkStore:
    """Paged chunk store seam returning canned rows + a total."""

    def __init__(self, chunks: list[Chunk] | None = None, total: int = 0) -> None:
        self._chunks = list(chunks or [])
        self._total = total
        self.calls: list[tuple[int, str, int, int]] = []

    async def list_paged_chunks(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
        page: int,
        page_size: int,
        enabled_only: bool = True,
    ) -> tuple[list[Chunk], int]:
        self.calls.append((tenant_id, knowledge_id, page, page_size))
        return list(self._chunks), self._total


def _list_tool(
    *,
    store: _FakeChunkStore,
    knowledge_lookup: _FakeKnowledgeLookup | None = None,
    chunk_lookup: _FakeChunkLookup | None = None,
    targets: SearchTargets | None = None,
) -> ListKnowledgeChunksTool:
    return ListKnowledgeChunksTool(
        definition=build_list_knowledge_chunks_definition(),
        chunk_store=store,
        search_targets=targets or _targets(_kb_target()),
        knowledge_service=knowledge_lookup,
        chunk_service=chunk_lookup,
    )


async def test_list_tool_requires_id_target() -> None:
    tool = _list_tool(store=_FakeChunkStore())
    result = await tool.execute(_Context(), "{}")
    assert not result.success
    assert "one of faq_id, chunk_id, or knowledge_id is required" in result.error


async def test_list_tool_pages_by_knowledge_id() -> None:
    rows = [
        _chunk_row(id="c1", content="first", chunk_index=0),
        _chunk_row(id="c2", content="second", chunk_index=1),
    ]
    tool = _list_tool(
        store=_FakeChunkStore(rows, total=2),
        knowledge_lookup=_FakeKnowledgeLookup(_knowledge()),
    )
    result = await tool.execute(_Context(), '{"knowledge_id": "d1"}')
    assert result.success
    assert '<knowledge_chunks knowledge_id="d1"' in result.output
    assert 'total="2" fetched="2"' in result.output
    assert result.data["display_type"] == "knowledge_chunks_list"
    assert result.data["fetched_chunks"] == 2
    assert len(_list_json(result, "chunks")) == 2
    assert _list_json(result, "chunks")[0]["chunk_id"] == "c1"


async def test_list_tool_single_chunk_by_chunk_id() -> None:
    chunk = _chunk_row(id="c1", content="single", knowledge_id="d1")
    tool = _list_tool(
        store=_FakeChunkStore(),
        knowledge_lookup=_FakeKnowledgeLookup(_knowledge()),
        chunk_lookup=_FakeChunkLookup(chunk),
    )
    result = await tool.execute(_Context(), '{"chunk_id": "c1"}')
    assert result.success
    assert result.data["single_chunk"] is True
    assert _list_json(result, "chunks")[0]["chunk_id"] == "c1"


async def test_list_tool_faq_id_uses_metadata() -> None:
    faq = _chunk_row(
        id="c1",
        chunk_type="faq",
        content="",
        metadata={
            "standard_question": "What is RAG?",
            "answers": ["Retrieval augmented generation."],
        },
    )
    tool = _list_tool(
        store=_FakeChunkStore(),
        knowledge_lookup=_FakeKnowledgeLookup(_knowledge()),
        chunk_lookup=_FakeChunkLookup(faq),
    )
    result = await tool.execute(_Context(), '{"faq_id": "c1"}')
    assert result.success
    assert '<faq faq_id="c1"' in result.output
    assert result.data["faq_question"] == "What is RAG?"
    assert _list_json(result, "chunks")[0]["faq_id"] == "c1"


async def test_list_tool_out_of_range_offset_guidance() -> None:
    tool = _list_tool(
        store=_FakeChunkStore([], total=3),
        knowledge_lookup=_FakeKnowledgeLookup(_knowledge()),
    )
    result = await tool.execute(_Context(), '{"knowledge_id": "d1", "offset": 10}')
    assert not result.success
    assert "offset 10 is out of range" in result.error
    assert result.data["suggested_offset"] == 0
    assert result.data["total_chunks"] == 3


async def test_list_tool_empty_document_returns_success() -> None:
    tool = _list_tool(
        store=_FakeChunkStore([], total=0),
        knowledge_lookup=_FakeKnowledgeLookup(_knowledge()),
    )
    result = await tool.execute(_Context(), '{"knowledge_id": "d1"}')
    assert result.success
    assert result.data["fetched_chunks"] == 0


# ── Integration tests (real applied schema) ──────────────────────────


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Per-test session against the real applied schema (no cleanup)."""
    reset_settings_cache()
    settings = get_settings()
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as s:
            yield s
    finally:
        await engine.dispose()


async def test_integration_list_chunks_tool_pages_real_rows(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    kb_service = KBService(kb_repo=KnowledgeBaseRepository(session))
    kb = await kb_service.create_knowledge_base(
        tenant_id=tenant_id, name="agent-list-kb", kb_type=KNOWLEDGE_BASE_TYPE_DOCUMENT
    )
    doc = await KnowledgeRepository(session).create(
        _integration_doc(id=str(uuid.uuid4()), tenant_id=tenant_id, knowledge_base_id=kb.id)
    )
    for index in range(2):
        await ChunkRepository(session).create_many(
            [
                _integration_chunk(
                    id=str(uuid.uuid4()),
                    tenant_id=tenant_id,
                    knowledge_base_id=kb.id,
                    knowledge_id=doc.id,
                    chunk_index=index,
                    content=f"chunk body {index}",
                )
            ]
        )

    knowledge_service = KnowledgeService(knowledge_repo=KnowledgeRepository(session))
    tool = ListKnowledgeChunksTool(
        definition=build_list_knowledge_chunks_definition(),
        chunk_store=SqlPagedChunkStore(session),
        search_targets=_targets(_kb_target(knowledge_base_id=kb.id, tenant_id=tenant_id)),
        knowledge_service=knowledge_service,
    )
    result = await tool.execute(_Context(), f'{{"knowledge_id": "{doc.id}"}}')
    assert result.success
    assert result.data["fetched_chunks"] == 2
    assert result.data["total_chunks"] == 2
    assert {chunk["content"] for chunk in _list_json(result, "chunks")} == {
        "chunk body 0",
        "chunk body 1",
    }
    assert result.data["knowledge_title"] == "Q3 budget"


async def test_integration_grep_chunks_tool_real_rows(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    kb_service = KBService(kb_repo=KnowledgeBaseRepository(session))
    kb = await kb_service.create_knowledge_base(
        tenant_id=tenant_id, name="agent-grep-kb", kb_type=KNOWLEDGE_BASE_TYPE_DOCUMENT
    )
    doc = await KnowledgeRepository(session).create(
        _integration_doc(id=str(uuid.uuid4()), tenant_id=tenant_id, knowledge_base_id=kb.id)
    )
    await ChunkRepository(session).create_many(
        [
            _integration_chunk(
                id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                knowledge_base_id=kb.id,
                knowledge_id=doc.id,
                chunk_index=0,
                content="the stardust engine powers the starship",
            )
        ]
    )

    tool = GrepChunksTool(
        definition=build_grep_chunks_definition(),
        store=SqlChunkGrepStore(session),
        search_targets=_targets(_kb_target(knowledge_base_id=kb.id, tenant_id=tenant_id)),
    )
    result = await tool.execute(_Context(), '{"query": "stardust|engine"}')
    assert result.success
    assert result.data["result_count"] >= 1  # type: ignore[operator]
    assert _list_json(result, "knowledge_results")[0]["knowledge_title"] == "Q3 budget"
    assert "stardust" in result.output


async def test_integration_scope_auth_authorizes_real_document(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    kb_service = KBService(kb_repo=KnowledgeBaseRepository(session))
    kb = await kb_service.create_knowledge_base(
        tenant_id=tenant_id, name="agent-auth-kb", kb_type=KNOWLEDGE_BASE_TYPE_DOCUMENT
    )
    doc = await KnowledgeRepository(session).create(
        _integration_doc(id=str(uuid.uuid4()), tenant_id=tenant_id, knowledge_base_id=kb.id)
    )

    knowledge_service = KnowledgeService(knowledge_repo=KnowledgeRepository(session))
    authorized = await authorize_knowledge_in_search_targets(
        _Context(),
        _targets(_kb_target(knowledge_base_id=kb.id, tenant_id=tenant_id)),
        doc.id,
        knowledge_service,
    )
    assert authorized.id == doc.id
    assert authorized.knowledge_base_id == kb.id

    # A document from a knowledge base outside the scope is rejected.
    with pytest.raises(PermissionDeniedError):
        await authorize_knowledge_in_search_targets(
            _Context(),
            _targets(_kb_target(knowledge_base_id=kb.id, tenant_id=tenant_id)),
            doc.id,
            _FakeKnowledgeLookup(_knowledge(id=doc.id, knowledge_base_id="kb-outside")),
        )


def _integration_doc(
    *,
    id: str,
    tenant_id: int,
    knowledge_base_id: str,
) -> Document:
    """Build one document row for the integration tests."""
    return Document(
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        type="file",
        title="Q3 budget",
        description="the budget",
        source="budget-2026.pdf",
        channel=CHANNEL_WEB,
        parse_status=PARSE_STATUS_COMPLETED,
        summary_status="none",
        enable_status="enabled",
        embedding_model_id="em-1",
        file_name="budget-2026.pdf",
        file_type="pdf",
        file_size=1024,
        storage_size=2048,
        metadata={"owner": "finance"},
        custom_metadata={"scope": "2026"},
        created_at=_NOW,
        updated_at=_NOW,
    )


def _integration_chunk(
    *,
    id: str,
    tenant_id: int,
    knowledge_base_id: str,
    knowledge_id: str,
    chunk_index: int,
    content: str,
) -> Chunk:
    """Build one chunk row for the integration tests."""
    return Chunk(
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id=knowledge_base_id,
        knowledge_id=knowledge_id,
        content=content,
        chunk_index=chunk_index,
        is_enabled=True,
        start_at=0,
        end_at=len(content),
        pre_chunk_id=None,
        next_chunk_id=None,
        chunk_type="text",
        parent_chunk_id=None,
        image_info=None,
        relation_chunks=None,
        indirect_relation_chunks=None,
        metadata={"source": "manual"},
        tag_id=None,
        status=1,
        content_hash=None,
        flags=1,
        seq_id=0,
        source_content="",
        content_revision=0,
        index_status="ready",
        last_editor_id="",
        context_header="",
        created_at=_NOW,
        updated_at=_NOW,
        deleted_at=None,
    )
