"""Unit + integration tests for the query-understand and search steps.

Unit tests drive ``QueryUnderstandPlugin``, ``SearchEntityPlugin`` and
``SearchParallelPlugin`` with in-memory fakes for the model / message /
graph / chunk / knowledge seams — no database, plain pytest AAA.

Integration tests run the entity and parallel search steps against the
real applied schema: ``chunks`` rows (INTEGER tenant_id) are seeded with
an int32-safe tenant counter, knowledge rows with ``make_test_tenant_id``,
and the graph store is faked (no graph backend is available in tests).
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from random import randint

import pytest
import pytest_asyncio
from faker import Faker
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from src.ai.graph.types import GraphData, GraphNode, NameSpace
from src.ai.llm.types import Chat, ChatOptions, ChatResponse, Message, StreamResponse
from src.ai.retrieval.types import MatchType
from src.common.json import JsonObject
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.engine import ERR_SEARCH_NOTHING, Next, PluginError
from src.core.chat.pipeline.steps.query_understand import (
    QueryUnderstandPlugin,
    coerce_query_intent,
    format_conversation_history,
    merge_image_desc_and_ocr,
    parse_structured_query_output,
)
from src.core.chat.pipeline.steps.search_entity import (
    SearchEntityPlugin,
    build_content_signature,
    filter_seen_chunks,
    remove_duplicate_results,
)
from src.core.chat.pipeline.steps.search_parallel import SearchParallelPlugin
from src.core.chat.pipeline.types import (
    Context,
    EventType,
    History,
    MessageAttachment,
    MessageImage,
    QueryIntent,
    SearchResult,
)
from src.core.chat.pipeline.types import (
    GraphData as PipelineGraphData,
)
from src.core.chat.pipeline.types import (
    GraphNode as PipelineGraphNode,
)
from src.db.dao.chunk_repository import ChunkRepository
from src.db.dao.knowledge_repository import KnowledgeRepository
from src.db.models.chunk import Chunk
from src.db.models.knowledge import Document
from src.settings import get_settings, reset_settings_cache
from tests.integration.conftest import make_test_tenant_id

_NOW = datetime(2026, 2, 1, tzinfo=UTC)
_FAKER_SEED_MAX = 100_000_000

# ``chunks.tenant_id`` is an INTEGER (32-bit) column; integration ids are
# minted from this counter so seeded rows never overflow.
_INT32_TENANT_BASE = 4_000_000
_INT32_TENANT_SEQ = itertools.count(start=1)

_IMAGE_OCR = "image_ocr"
_IMAGE_CAPTION = "image_caption"


def _int32_tenant_id() -> int:
    """Return a tenant id unique within the session, safe for INTEGER."""
    return _INT32_TENANT_BASE + next(_INT32_TENANT_SEQ)


@pytest.fixture(autouse=True)
def faker_seed() -> None:
    """Re-seed Faker per test for varied-but-reproducible generation."""
    Faker.seed(randint(1, _FAKER_SEED_MAX))


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


async def _noop() -> PluginError | None:
    return None


@dataclass(frozen=True, slots=True)
class _FakeContext:
    """Opaque execution context (empty structural protocol)."""

    is_background_task: bool = False


# ── Query-understand seams ─────────────────────────────────────────────


class _FakeChat:
    """A chat-capable model returning a canned response."""

    def __init__(self, *, content: str) -> None:
        self._content = content
        self.requests: list[tuple[list[Message], ChatOptions | None]] = []

    async def chat(self, messages: list[Message], opts: ChatOptions | None = None) -> ChatResponse:
        self.requests.append((messages, opts))
        return ChatResponse(content=self._content)

    def chat_stream(
        self, messages: list[Message], opts: ChatOptions | None = None
    ) -> AsyncIterator[StreamResponse]:
        raise NotImplementedError

    def get_model_name(self) -> str:
        return "fake-model"

    def get_model_id(self) -> str:
        return "fake-model-id"


class _FakeModelService:
    """Resolves ``_FakeChat`` models, failing for ids in ``failures``."""

    def __init__(self, *, models: Mapping[str, _FakeChat], failures: Sequence[str] = ()) -> None:
        self.models = dict(models)
        self.failures = set(failures)
        self.requests: list[tuple[Context, str]] = []

    async def get_chat_model(self, ctx: Context, model_id: str) -> Chat:
        self.requests.append((ctx, model_id))
        if model_id in self.failures or model_id not in self.models:
            raise RuntimeError(f"model {model_id} unavailable")
        return self.models[model_id]


@dataclass(slots=True)
class _StoredMessage:
    """A stored chat message consumed by the history / caption seams.

    ``images`` / ``attachments`` / ``references`` are declared as
    ``Sequence`` (not ``tuple``) and the class is settable so the fake
    structurally satisfies the ``StoredMessage`` protocol.
    """

    request_id: str = ""
    role: str = ""
    content: str = ""
    created_at: datetime | None = None
    images: Sequence[MessageImage] = ()
    attachments: Sequence[MessageAttachment] = ()
    knowledge_references: Sequence[SearchResult] = ()


class _FakeMessageService:
    """In-memory message store recording history + image-caption writes."""

    def __init__(self, *, stored: Sequence[_StoredMessage] = ()) -> None:
        self._stored = list(stored)
        self.recent_requests: list[str] = []
        self.updated_images: list[list[MessageImage]] = []

    async def get_recent_messages_by_session(
        self,
        ctx: Context,
        session_id: str,
        count: int,
    ) -> list[_StoredMessage]:
        self.recent_requests.append(session_id)
        return list(self._stored)

    async def get_message(
        self,
        ctx: Context,
        _session_id: str,
        user_message_id: str,
    ) -> _StoredMessage:
        for message in self._stored:
            if message.request_id == user_message_id:
                return message
        raise LookupError(f"message {user_message_id} not found")

    async def update_message_images(
        self,
        ctx: Context,
        _session_id: str,
        _user_message_id: str,
        images: Sequence[MessageImage],
    ) -> None:
        self.updated_images.append(list(images))


def _make_query_understand(
    *,
    models: Mapping[str, str] | None = None,
    failures: Sequence[str] = (),
    stored: Sequence[_StoredMessage] = (),
    intent_system_prompts: Mapping[str, str] | None = None,
    rewrite_prompt_system: str = "",
    rewrite_prompt_user: str = "",
) -> tuple[QueryUnderstandPlugin, _FakeModelService, _FakeMessageService]:
    """Build a ``QueryUnderstandPlugin`` wired to in-memory fakes."""
    model_service = _FakeModelService(
        models={
            model_id: _FakeChat(content=content) for model_id, content in (models or {}).items()
        },
        failures=failures,
    )
    message_service = _FakeMessageService(stored=stored)
    plugin = QueryUnderstandPlugin(
        model_service=model_service,
        message_service=message_service,
        intent_system_prompts=intent_system_prompts,
        rewrite_prompt_system=rewrite_prompt_system,
        rewrite_prompt_user=rewrite_prompt_user,
    )
    return plugin, model_service, message_service


# ── Entity / parallel search seams ─────────────────────────────────────


class _FakeGraphRepo:
    """Returns a canned graph per ``(knowledge_base, knowledge)`` scope."""

    def __init__(
        self,
        *,
        graphs: Mapping[tuple[str, str], GraphData] | None = None,
        disabled: bool = False,
    ) -> None:
        self._graphs = dict(graphs or {})
        self._disabled = disabled
        self.queries: list[tuple[NameSpace, list[str]]] = []

    async def search_node(self, namespace: NameSpace, nodes: list[str]) -> GraphData | None:
        self.queries.append((namespace, list(nodes)))
        if self._disabled:
            return None
        return self._graphs.get((namespace.knowledge_base, namespace.knowledge))

    async def add_graph(self, namespace: NameSpace, graphs: list[GraphData]) -> None:
        pass

    async def del_graph(self, namespaces: list[NameSpace]) -> None:
        pass


class _FakeChunkStore:
    """In-memory chunk rows for the entity step hydration seam."""

    def __init__(self, *, chunks: Mapping[str, Chunk] | None = None) -> None:
        self._chunks = dict(chunks or {})

    async def list_by_ids(self, tenant_id: int, ids: list[str]) -> list[Chunk]:
        return [self._chunks[i] for i in ids if i in self._chunks]

    async def list_by_parent_id(self, tenant_id: int, parent_id: str) -> list[Chunk]:
        return [c for c in self._chunks.values() if c.parent_chunk_id == parent_id]


class _FakeKnowledgeStore:
    """In-memory knowledge rows for the entity step hydration seam."""

    def __init__(self, *, knowledge: Mapping[str, Document] | None = None) -> None:
        self._knowledge = dict(knowledge or {})

    async def get_batch(self, tenant_id: int, ids: list[str]) -> list[Document]:
        return [self._knowledge[i] for i in ids if i in self._knowledge]


def _chunk(
    *,
    id: str,
    content: str = "payload",
    knowledge_id: str = "doc-1",
    tenant_id: int = 1,
    parent_chunk_id: str | None = None,
    chunk_type: str = "text",
    image_info: str | None = None,
    chunk_index: int = 0,
) -> Chunk:
    return Chunk(
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id="kb-1",
        knowledge_id=knowledge_id,
        content=content,
        chunk_index=chunk_index,
        start_at=0,
        end_at=max(1, len(content)),
        parent_chunk_id=parent_chunk_id,
        chunk_type=chunk_type,
        image_info=image_info,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _doc(
    *,
    id: str = "doc-1",
    title: str = "Doc 1",
    file_name: str = "doc-1.pdf",
    metadata: JsonObject | None = None,
    tenant_id: int = 1,
) -> Document:
    return Document(
        id=id,
        tenant_id=tenant_id,
        knowledge_base_id="kb-1",
        type="file",
        title=title,
        description=None,
        source="web",
        channel="web",
        parse_status="done",
        pending_subtasks_count=0,
        summary_status="none",
        enable_status="enabled",
        embedding_model_id=None,
        file_name=file_name,
        file_type="pdf",
        file_size=1024,
        file_hash="",
        file_path="local://1/k/1/obj",
        storage_size=0,
        metadata=metadata,
        custom_metadata={},
        last_faq_import_result=None,
        created_at=_NOW,
        updated_at=_NOW,
        processed_at=None,
        error_message=None,
        deleted_at=None,
    )


def _make_entity_plugin(
    *,
    graphs: Mapping[tuple[str, str], GraphData] | None = None,
    chunks: Mapping[str, Chunk] | None = None,
    knowledge: Mapping[str, Document] | None = None,
    graph_disabled: bool = False,
) -> tuple[SearchEntityPlugin, _FakeGraphRepo]:
    graph_repo = _FakeGraphRepo(graphs=graphs, disabled=graph_disabled)
    plugin = SearchEntityPlugin(
        graph_repo=graph_repo,
        chunk_repo=_FakeChunkStore(chunks=chunks),
        knowledge_repo=_FakeKnowledgeStore(knowledge=knowledge),
    )
    return plugin, graph_repo


class _FakeChunkSearchPlugin:
    """A chunk-search plugin that writes canned results to its carrier."""

    def __init__(
        self,
        *,
        results: Sequence[SearchResult] = (),
        error: PluginError | None = None,
    ) -> None:
        self._results = list(results)
        self._error = error
        self.calls = 0
        self.seen_carriers: list[PipelineContext] = []

    async def on_event(
        self,
        ctx: Context,
        _event_type: EventType | str,
        pipeline_ctx: PipelineContext,
        _next: Next,
    ) -> PluginError | None:
        self.calls += 1
        self.seen_carriers.append(pipeline_ctx)
        pipeline_ctx.search_result = list(self._results)
        return self._error

    def activation_events(self) -> list[EventType]:
        return [EventType.CHUNK_SEARCH]


def _make_parallel_plugin(
    *,
    chunk_plugin: _FakeChunkSearchPlugin | None = None,
    graphs: Mapping[tuple[str, str], GraphData] | None = None,
    chunks: Mapping[str, Chunk] | None = None,
    knowledge: Mapping[str, Document] | None = None,
) -> SearchParallelPlugin:
    return SearchParallelPlugin(
        chunk_search_plugin=chunk_plugin,
        graph_repo=_FakeGraphRepo(graphs=graphs),
        chunk_repo=_FakeChunkStore(chunks=chunks),
        knowledge_repo=_FakeKnowledgeStore(knowledge=knowledge),
    )


# ── Query understand: routing / skip ───────────────────────────────────


async def test_query_understand_skips_when_rewrite_disabled_and_no_images() -> None:
    plugin, model_service, _ = _make_query_understand(models={"chat-1": "ignored"})
    pipeline_ctx = PipelineContext(query="hello", chat_model_id="chat-1")

    result = await plugin.on_event(_FakeContext(), EventType.QUERY_UNDERSTAND, pipeline_ctx, _noop)

    assert result is None
    assert pipeline_ctx.rewrite_query == "hello"
    assert model_service.requests == []


async def test_query_understand_text_rewrite_calls_model_and_parses_output() -> None:
    plugin, model_service, _ = _make_query_understand(
        models={"chat-1": '{"rewrite_query":"what is vector db","intent":"kb_search"}'},
        rewrite_prompt_system="Rewrite the query.\nQuery: {{query}}",
        rewrite_prompt_user="Rewrite for language {{language}}.",
    )
    pipeline_ctx = PipelineContext(
        query="what is vector db?",
        enable_rewrite=True,
        chat_model_id="chat-1",
        language="en",
    )

    result = await plugin.on_event(_FakeContext(), EventType.QUERY_UNDERSTAND, pipeline_ctx, _noop)

    assert result is None
    assert pipeline_ctx.rewrite_query == "what is vector db"
    assert pipeline_ctx.intent == QueryIntent.KB_SEARCH
    messages, opts = model_service.models["chat-1"].requests[0]
    assert [message.role for message in messages] == ["system", "user"]
    assert messages[0].content == (
        "Rewrite the query.\nQuery: what is vector db?\n\n<no_image_attached />\n<no_document_attached />"
    )
    assert messages[1].content == "Rewrite for language en."
    assert opts is not None
    assert opts.temperature == 0.3
    assert opts.max_completion_tokens == 150
    assert opts.thinking is False


async def test_query_understand_image_turn_uses_vision_and_larger_budget() -> None:
    plugin, model_service, _ = _make_query_understand(
        models={
            "chat-1": (
                '{"rewrite_query":"describe the diagram",'
                '"intent":"image_only",'
                '"image_description":"A flow diagram",'
                '"ocr_text":"Step 1 -> Step 2"}'
            )
        },
    )
    pipeline_ctx = PipelineContext(
        query="describe this",
        enable_rewrite=True,
        chat_model_id="chat-1",
        chat_model_supports_vision=True,
        images=["http://img/a.png"],
    )

    result = await plugin.on_event(_FakeContext(), EventType.QUERY_UNDERSTAND, pipeline_ctx, _noop)

    assert result is None
    assert pipeline_ctx.intent == QueryIntent.IMAGE_ONLY
    assert pipeline_ctx.image_description == "A flow diagram\n\n[OCR]\nStep 1 -> Step 2"
    messages, opts = model_service.models["chat-1"].requests[0]
    assert messages[1].images == ["http://img/a.png"]
    assert opts is not None
    assert opts.max_completion_tokens == 500


async def test_query_understand_vision_falls_back_to_vlm_when_chat_lacks_vision() -> None:
    plugin, model_service, _ = _make_query_understand(
        models={"vlm-1": '{"rewrite_query":"r","intent":"image_only"}'},
    )
    pipeline_ctx = PipelineContext(
        query="what is in the image",
        enable_rewrite=True,
        chat_model_id="chat-1",
        vlm_model_id="vlm-1",
        images=["http://img/a.png"],
    )

    result = await plugin.on_event(_FakeContext(), EventType.QUERY_UNDERSTAND, pipeline_ctx, _noop)

    assert result is None
    assert [model_id for _, model_id in model_service.requests] == ["vlm-1"]
    # VLM path still uses images + the larger token budget.
    assert model_service.models["vlm-1"].requests[0][1] is not None
    assert model_service.models["vlm-1"].requests[0][1].max_completion_tokens == 500


async def test_query_understand_prefers_query_understand_model_with_chat_fallback() -> None:
    plugin, model_service, _ = _make_query_understand(
        models={"chat-1": '{"rewrite_query":"r","intent":"kb_search"}'},
        failures=["qu-1"],
    )
    pipeline_ctx = PipelineContext(
        query="q",
        enable_rewrite=True,
        chat_model_id="chat-1",
        query_understand_model_id="qu-1",
    )

    result = await plugin.on_event(_FakeContext(), EventType.QUERY_UNDERSTAND, pipeline_ctx, _noop)

    assert result is None
    assert [model_id for _, model_id in model_service.requests] == ["qu-1", "chat-1"]
    assert pipeline_ctx.rewrite_query == "r"


async def test_query_understand_model_failure_keeps_original_query() -> None:
    plugin, _model_service, _ = _make_query_understand(
        models={"chat-1": "ignored"},
        failures=["chat-1"],
    )
    pipeline_ctx = PipelineContext(query="original", enable_rewrite=True, chat_model_id="chat-1")

    result = await plugin.on_event(_FakeContext(), EventType.QUERY_UNDERSTAND, pipeline_ctx, _noop)

    assert result is None
    assert pipeline_ctx.rewrite_query == "original"
    assert pipeline_ctx.intent is None


# ── Query understand: history ──────────────────────────────────────────


async def test_query_understand_loads_history_when_absent_from_carrier() -> None:
    stored = [
        _StoredMessage(request_id="r-1", role="user", content="prior question", created_at=_NOW),
        _StoredMessage(request_id="r-1", role="assistant", content="prior answer", created_at=_NOW),
    ]
    plugin, model_service, message_service = _make_query_understand(
        models={"chat-1": '{"rewrite_query":"r","intent":"kb_search"}'},
        stored=stored,
        rewrite_prompt_system="Conversation:\n{{conversation}}\nQuery: {{query}}",
    )
    pipeline_ctx = PipelineContext(
        query="q",
        enable_rewrite=True,
        chat_model_id="chat-1",
        max_rounds=4,
        session_id="sess-1",
    )

    result = await plugin.on_event(_FakeContext(), EventType.QUERY_UNDERSTAND, pipeline_ctx, _noop)

    assert result is None
    assert message_service.recent_requests == ["sess-1"]
    assert len(pipeline_ctx.history) == 1
    messages, _ = model_service.models["chat-1"].requests[0]
    system_prompt = messages[0].content
    assert "User question: prior question" in system_prompt
    assert "Assistant answer: prior answer" in system_prompt


async def test_query_understand_reuses_carrier_history_without_fetch() -> None:
    plugin, model_service, message_service = _make_query_understand(
        models={"chat-1": '{"rewrite_query":"r","intent":"kb_search"}'},
        rewrite_prompt_system="Conversation:\n{{conversation}}",
    )
    pipeline_ctx = PipelineContext(
        query="q",
        enable_rewrite=True,
        chat_model_id="chat-1",
        history=[History(query="prior", answer="answer")],
    )

    result = await plugin.on_event(_FakeContext(), EventType.QUERY_UNDERSTAND, pipeline_ctx, _noop)

    assert result is None
    assert message_service.recent_requests == []
    messages, _ = model_service.models["chat-1"].requests[0]
    assert "User question: prior" in messages[0].content


async def test_query_understand_zero_max_rounds_skips_history() -> None:
    plugin, _, message_service = _make_query_understand(models={"chat-1": '{"rewrite_query":"r"}'})
    pipeline_ctx = PipelineContext(
        query="q",
        enable_rewrite=True,
        chat_model_id="chat-1",
        max_rounds=0,
    )

    result = await plugin.on_event(_FakeContext(), EventType.QUERY_UNDERSTAND, pipeline_ctx, _noop)

    assert result is None
    assert message_service.recent_requests == []
    assert pipeline_ctx.history == []


# ── Query understand: intent prompt overrides ──────────────────────────


async def test_query_understand_agent_override_wins_over_global() -> None:
    plugin, _, _ = _make_query_understand(
        models={"chat-1": '{"rewrite_query":"r","intent":"chitchat"}'},
        intent_system_prompts={"chitchat": "global prompt"},
    )
    pipeline_ctx = PipelineContext(
        query="q",
        enable_rewrite=True,
        chat_model_id="chat-1",
        intent_prompt_overrides={"chitchat": "agent prompt"},
    )

    result = await plugin.on_event(_FakeContext(), EventType.QUERY_UNDERSTAND, pipeline_ctx, _noop)

    assert result is None
    assert pipeline_ctx.system_prompt_override == "agent prompt"


async def test_query_understand_whitespace_agent_override_falls_through_to_global() -> None:
    plugin, _, _ = _make_query_understand(
        models={"chat-1": '{"rewrite_query":"r","intent":"chitchat"}'},
        intent_system_prompts={"chitchat": "global prompt"},
    )
    pipeline_ctx = PipelineContext(
        query="q",
        enable_rewrite=True,
        chat_model_id="chat-1",
        intent_prompt_overrides={"chitchat": "   "},
    )

    result = await plugin.on_event(_FakeContext(), EventType.QUERY_UNDERSTAND, pipeline_ctx, _noop)

    assert result is None
    assert pipeline_ctx.system_prompt_override == "global prompt"


async def test_query_understand_retrieval_intent_gets_no_prompt_override() -> None:
    plugin, _, _ = _make_query_understand(
        models={"chat-1": '{"rewrite_query":"r","intent":"kb_search"}'},
        intent_system_prompts={"kb_search": "ignored"},
    )
    pipeline_ctx = PipelineContext(query="q", enable_rewrite=True, chat_model_id="chat-1")

    result = await plugin.on_event(_FakeContext(), EventType.QUERY_UNDERSTAND, pipeline_ctx, _noop)

    assert result is None
    assert pipeline_ctx.system_prompt_override == ""


# ── Query understand: image caption persistence ────────────────────────


async def test_query_understand_persists_image_caption_to_user_message() -> None:
    stored = [
        _StoredMessage(
            request_id="user-msg-1",
            role="user",
            content="caption this",
            images=(MessageImage(url="http://img/a.png", caption=""),),
        )
    ]
    plugin, _, message_service = _make_query_understand(
        models={"chat-1": '{"rewrite_query":"r","image_description":"A red apple"}'},
        stored=stored,
    )
    pipeline_ctx = PipelineContext(
        query="caption this",
        enable_rewrite=True,
        chat_model_id="chat-1",
        images=["http://img/a.png"],
        user_message_id="user-msg-1",
    )

    result = await plugin.on_event(_FakeContext(), EventType.QUERY_UNDERSTAND, pipeline_ctx, _noop)

    assert result is None
    assert pipeline_ctx.image_description == "A red apple"
    assert len(plugin._background_tasks) == 1
    await asyncio.gather(*plugin._background_tasks)
    assert len(message_service.updated_images) == 1
    updated = message_service.updated_images[0]
    assert updated[0].url == "http://img/a.png"
    assert updated[0].caption == "A red apple"


async def test_query_understand_no_caption_write_without_message_handle() -> None:
    plugin, _, message_service = _make_query_understand(
        models={"chat-1": '{"rewrite_query":"r","image_description":"A red apple"}'},
    )
    pipeline_ctx = PipelineContext(
        query="q",
        enable_rewrite=True,
        chat_model_id="chat-1",
        images=["http://img/a.png"],
    )

    result = await plugin.on_event(_FakeContext(), EventType.QUERY_UNDERSTAND, pipeline_ctx, _noop)

    assert result is None
    assert plugin._background_tasks == []
    assert message_service.updated_images == []


# ── Query understand: parsing helpers ──────────────────────────────────


def test_parse_structured_query_output_extracts_aliases() -> None:
    output = parse_structured_query_output(
        '{"rewritten_query":"r","intent":"web_search","image_desc":"d","ocr":"o"}'
    )
    assert output is not None
    assert output.rewrite_query == "r"
    assert output.intent == "web_search"
    assert output.image_description == "d\n\n[OCR]\no"


def test_parse_structured_query_output_tolerates_markdown_wrappers() -> None:
    output = parse_structured_query_output(
        '```json\n{"rewrite_query":"wrapped query","intent":"kb_search"}\n```'
    )
    assert output is not None
    assert output.rewrite_query == "wrapped query"
    assert output.intent == "kb_search"


def test_parse_structured_query_output_rejects_non_object_json() -> None:
    assert parse_structured_query_output("[1, 2, 3]") is None
    assert parse_structured_query_output("plain text") is None
    assert parse_structured_query_output("") is None


def test_coerce_query_intent_known_unknown_and_empty() -> None:
    assert coerce_query_intent("kb_search") == QueryIntent.KB_SEARCH
    assert coerce_query_intent("small_talk") is None
    assert coerce_query_intent("") is None


@pytest.mark.parametrize(
    ("description", "ocr", "expected"),
    [
        ("desc", "ocr", "desc\n\n[OCR]\nocr"),
        ("", "ocr", "ocr"),
        ("desc", "", "desc"),
        ("", "", ""),
        ("full desc with ocr", "with ocr", "full desc with ocr"),
    ],
)
def test_merge_image_desc_and_ocr(description: str, ocr: str, expected: str) -> None:
    combined, is_set = merge_image_desc_and_ocr(description, ocr)
    assert is_set is (expected != "")
    assert combined == expected


def test_format_conversation_history() -> None:
    history = [History(query="q1", answer="a1"), History(query="q2", answer="a2")]
    rendered = format_conversation_history(history)
    assert rendered == (
        "------BEGIN------\nUser question: q1\nAssistant answer: a1\n------END------\n"
        "------BEGIN------\nUser question: q2\nAssistant answer: a2\n------END------\n"
    )
    assert format_conversation_history([]) == ""


# ── Entity search: plugin routing ──────────────────────────────────────


async def test_entity_search_no_entities_returns_next() -> None:
    plugin, graph_repo = _make_entity_plugin()
    pipeline_ctx = PipelineContext()

    result = await plugin.on_event(_FakeContext(), EventType.ENTITY_SEARCH, pipeline_ctx, _noop)

    assert result is None
    assert graph_repo.queries == []


async def test_entity_search_no_kb_scope_returns_next() -> None:
    plugin, graph_repo = _make_entity_plugin()
    pipeline_ctx = PipelineContext(entity=["entity-a"])

    result = await plugin.on_event(_FakeContext(), EventType.ENTITY_SEARCH, pipeline_ctx, _noop)

    assert result is None
    assert graph_repo.queries == []


async def test_entity_search_kb_scope_searches_and_hydrates() -> None:
    graph = GraphData(
        node=[GraphNode(name="entity-a", chunks=["chunk-1"])],
        relation=[],
    )
    chunks = {"chunk-1": _chunk(id="chunk-1", content="entity payload")}
    knowledge = {"doc-1": _doc(title="Entity Doc")}
    plugin, graph_repo = _make_entity_plugin(
        graphs={("kb-1", ""): graph},
        chunks=chunks,
        knowledge=knowledge,
    )
    pipeline_ctx = PipelineContext(
        tenant_id=make_test_tenant_id(),
        entity=["entity-a"],
        entity_kb_ids=["kb-1"],
    )

    result = await plugin.on_event(_FakeContext(), EventType.ENTITY_SEARCH, pipeline_ctx, _noop)

    assert result is None
    assert graph_repo.queries == [(NameSpace(knowledge_base="kb-1", knowledge=""), ["entity-a"])]
    assert len(pipeline_ctx.search_result) == 1
    hit = pipeline_ctx.search_result[0]
    assert hit.id == "chunk-1"
    assert hit.content == "entity payload"
    assert hit.knowledge_title == "Entity Doc"
    assert hit.match_type == MatchType.GRAPH
    assert hit.score == 1.0
    assert hit.seq == 0


async def test_entity_search_knowledge_file_scope_searches_each_file() -> None:
    graph_a = GraphData(node=[GraphNode(name="e", chunks=["chunk-1"])], relation=[])
    graph_b = GraphData(node=[GraphNode(name="e", chunks=["chunk-2"])], relation=[])
    chunks = {
        "chunk-1": _chunk(id="chunk-1", knowledge_id="doc-1", content="from doc 1"),
        "chunk-2": _chunk(id="chunk-2", knowledge_id="doc-2", content="from doc 2"),
    }
    knowledge = {
        "doc-1": _doc(id="doc-1", title="Doc 1"),
        "doc-2": _doc(id="doc-2", title="Doc 2"),
    }
    plugin, graph_repo = _make_entity_plugin(
        graphs={("kb-1", "doc-1"): graph_a, ("kb-1", "doc-2"): graph_b},
        chunks=chunks,
        knowledge=knowledge,
    )
    pipeline_ctx = PipelineContext(
        tenant_id=1,
        entity=["e"],
        entity_knowledge={"doc-1": "kb-1", "doc-2": "kb-1"},
    )

    result = await plugin.on_event(_FakeContext(), EventType.ENTITY_SEARCH, pipeline_ctx, _noop)

    assert result is None
    assert sorted([(q[0].knowledge_base, q[0].knowledge) for q in graph_repo.queries]) == [
        ("kb-1", "doc-1"),
        ("kb-1", "doc-2"),
    ]
    assert sorted(hit.id for hit in pipeline_ctx.search_result) == ["chunk-1", "chunk-2"]
    assert pipeline_ctx.graph_result is not None
    assert len(pipeline_ctx.graph_result.node) == 2


async def test_entity_search_skips_chunks_already_seen() -> None:
    graph = GraphData(
        node=[GraphNode(name="entity-a", chunks=["chunk-seen", "chunk-new"])],
        relation=[],
    )
    chunks = {
        "chunk-new": _chunk(id="chunk-new", content="fresh"),
    }
    knowledge = {"doc-1": _doc()}
    plugin, _ = _make_entity_plugin(
        graphs={("kb-1", ""): graph},
        chunks=chunks,
        knowledge=knowledge,
    )
    pipeline_ctx = PipelineContext(
        tenant_id=1,
        entity=["entity-a"],
        entity_kb_ids=["kb-1"],
        search_result=[SearchResult(id="chunk-seen", content="seen")],
    )

    result = await plugin.on_event(_FakeContext(), EventType.ENTITY_SEARCH, pipeline_ctx, _noop)

    assert result is None
    assert [hit.id for hit in pipeline_ctx.search_result] == ["chunk-seen", "chunk-new"]


async def test_entity_search_disabled_graph_skips() -> None:
    plugin, _ = _make_entity_plugin(graph_disabled=True)
    pipeline_ctx = PipelineContext(tenant_id=1, entity=["e"], entity_kb_ids=["kb-1"])

    result = await plugin.on_event(_FakeContext(), EventType.ENTITY_SEARCH, pipeline_ctx, _noop)

    assert result is None
    assert pipeline_ctx.search_result == []


async def test_entity_search_no_new_chunks_returns_next() -> None:
    graph = GraphData(node=[GraphNode(name="e", chunks=[])], relation=[])
    plugin, _ = _make_entity_plugin(graphs={("kb-1", ""): graph})
    pipeline_ctx = PipelineContext(tenant_id=1, entity=["e"], entity_kb_ids=["kb-1"])

    result = await plugin.on_event(_FakeContext(), EventType.ENTITY_SEARCH, pipeline_ctx, _noop)

    assert result is None
    assert pipeline_ctx.search_result == []


async def test_entity_search_missing_knowledge_is_skipped() -> None:
    graph = GraphData(node=[GraphNode(name="e", chunks=["chunk-orphan"])], relation=[])
    chunks = {"chunk-orphan": _chunk(id="chunk-orphan", knowledge_id="missing-doc")}
    plugin, _ = _make_entity_plugin(graphs={("kb-1", ""): graph}, chunks=chunks)
    pipeline_ctx = PipelineContext(
        tenant_id=1,
        entity=["e"],
        entity_kb_ids=["kb-1"],
        search_result=[SearchResult(id="existing", content="kept")],
    )

    result = await plugin.on_event(_FakeContext(), EventType.ENTITY_SEARCH, pipeline_ctx, _noop)

    assert result is None
    assert [hit.id for hit in pipeline_ctx.search_result] == ["existing"]


async def test_entity_search_no_results_returns_err_search_nothing() -> None:
    graph = GraphData(node=[GraphNode(name="e", chunks=["chunk-x"])], relation=[])
    chunks = {"chunk-x": _chunk(id="chunk-x", knowledge_id="missing-doc")}
    plugin, _ = _make_entity_plugin(graphs={("kb-1", ""): graph}, chunks=chunks)
    pipeline_ctx = PipelineContext(
        tenant_id=1,
        entity=["e"],
        entity_kb_ids=["kb-1"],
        search_result=[],
    )

    result = await plugin.on_event(_FakeContext(), EventType.ENTITY_SEARCH, pipeline_ctx, _noop)

    assert result is ERR_SEARCH_NOTHING


# ── Entity search: image enrichment ────────────────────────────────────


async def test_entity_search_enriches_image_info_from_image_children() -> None:
    graph = GraphData(node=[GraphNode(name="e", chunks=["chunk-parent"])], relation=[])
    image_child = _chunk(
        id="img-child",
        content="",
        parent_chunk_id="chunk-parent",
        chunk_type=_IMAGE_OCR,
        image_info='[{"url":"http://img/x.png","caption":"Alt text","ocr_text":"OCR text"}]',
    )
    chunks = {
        "chunk-parent": _chunk(id="chunk-parent", content="parent text"),
        "img-child": image_child,
    }
    knowledge = {"doc-1": _doc()}
    plugin, _ = _make_entity_plugin(
        graphs={("kb-1", ""): graph},
        chunks=chunks,
        knowledge=knowledge,
    )
    pipeline_ctx = PipelineContext(tenant_id=1, entity=["e"], entity_kb_ids=["kb-1"])

    result = await plugin.on_event(_FakeContext(), EventType.ENTITY_SEARCH, pipeline_ctx, _noop)

    assert result is None
    hit = pipeline_ctx.search_result[0]
    assert hit.id == "chunk-parent"
    assert '"url": "http://img/x.png"' in hit.image_info
    assert '"caption": "Alt text"' in hit.image_info
    assert '"ocr_text": "OCR text"' in hit.image_info


async def test_entity_search_enrich_two_level_resolution() -> None:
    graph = GraphData(node=[GraphNode(name="e", chunks=["chunk-grandparent"])], relation=[])
    text_child = _chunk(
        id="text-child", content="text segment", parent_chunk_id="chunk-grandparent"
    )
    image_grandchild = _chunk(
        id="img-grandchild",
        content="",
        parent_chunk_id="text-child",
        chunk_type=_IMAGE_CAPTION,
        image_info='[{"url":"http://img/y.png","caption":"Grand caption"}]',
    )
    chunks = {
        "chunk-grandparent": _chunk(id="chunk-grandparent", content="grand parent"),
        "text-child": text_child,
        "img-grandchild": image_grandchild,
    }
    knowledge = {"doc-1": _doc()}
    plugin, _ = _make_entity_plugin(
        graphs={("kb-1", ""): graph},
        chunks=chunks,
        knowledge=knowledge,
    )
    pipeline_ctx = PipelineContext(tenant_id=1, entity=["e"], entity_kb_ids=["kb-1"])

    result = await plugin.on_event(_FakeContext(), EventType.ENTITY_SEARCH, pipeline_ctx, _noop)

    assert result is None
    hit = pipeline_ctx.search_result[0]
    assert '"url": "http://img/y.png"' in hit.image_info
    assert '"caption": "Grand caption"' in hit.image_info


async def test_entity_search_keeps_existing_image_info_untouched() -> None:
    graph = GraphData(node=[GraphNode(name="e", chunks=["chunk-with-img"])], relation=[])
    chunks = {
        "chunk-with-img": _chunk(
            id="chunk-with-img",
            content="has image already",
            image_info='[{"url":"http://img/z.png"}]',
        ),
    }
    knowledge = {"doc-1": _doc()}
    plugin, _ = _make_entity_plugin(
        graphs={("kb-1", ""): graph},
        chunks=chunks,
        knowledge=knowledge,
    )
    pipeline_ctx = PipelineContext(tenant_id=1, entity=["e"], entity_kb_ids=["kb-1"])

    result = await plugin.on_event(_FakeContext(), EventType.ENTITY_SEARCH, pipeline_ctx, _noop)

    assert result is None
    hit = pipeline_ctx.search_result[0]
    assert hit.image_info == '[{"url":"http://img/z.png"}]'


# ── Entity search: pure helpers ────────────────────────────────────────


def test_filter_seen_chunks_skips_seen_and_dedups() -> None:
    graph = PipelineGraphData(
        node=[
            PipelineGraphNode(name="a", chunks=["c1", "c2"]),
            PipelineGraphNode(name="b", chunks=["c1", "c3"]),
        ],
        relation=[],
    )
    seen = [SearchResult(id="c2", content="already")]
    assert filter_seen_chunks(graph, seen) == ["c1", "c3"]


def test_remove_duplicate_results_by_id_and_content_signature() -> None:
    results = [
        SearchResult(id="a", content="unique one"),
        SearchResult(id="b", content="unique two"),
        SearchResult(id="a", content="duplicate id"),
        SearchResult(id="c", content="unique one"),
    ]
    unique = remove_duplicate_results(results)
    assert [r.id for r in unique] == ["a", "b"]


def test_remove_duplicate_results_keeps_same_id_different_content() -> None:
    results = [SearchResult(id="a", content="one"), SearchResult(id="a", content="two")]
    assert [r.id for r in remove_duplicate_results(results)] == ["a"]


def test_build_content_signature_normalizes_whitespace_and_case() -> None:
    assert build_content_signature("  Hello   World ") == build_content_signature("hello world")
    assert build_content_signature("") == ""
    assert build_content_signature("   ") == ""


# ── Parallel search ────────────────────────────────────────────────────


async def test_parallel_skips_when_no_retrieval_needed() -> None:
    plugin = _make_parallel_plugin()
    pipeline_ctx = PipelineContext(intent=QueryIntent.CHITCHAT)

    result = await plugin.on_event(
        _FakeContext(), EventType.CHUNK_SEARCH_PARALLEL, pipeline_ctx, _noop
    )

    assert result is None
    assert pipeline_ctx.search_result == []


async def test_parallel_merges_chunk_and_entity_results() -> None:
    chunk_plugin = _FakeChunkSearchPlugin(results=[SearchResult(id="chunk-a", content="chunk a")])
    graph = GraphData(node=[GraphNode(name="e", chunks=["chunk-b"])], relation=[])
    chunks = {"chunk-b": _chunk(id="chunk-b", content="entity payload")}
    knowledge = {"doc-1": _doc(title="Entity Doc")}
    plugin = _make_parallel_plugin(
        chunk_plugin=chunk_plugin,
        graphs={("kb-1", ""): graph},
        chunks=chunks,
        knowledge=knowledge,
    )
    pipeline_ctx = PipelineContext(
        tenant_id=1,
        entity=["e"],
        entity_kb_ids=["kb-1"],
        rewrite_query="rewritten",
    )

    result = await plugin.on_event(
        _FakeContext(), EventType.CHUNK_SEARCH_PARALLEL, pipeline_ctx, _noop
    )

    assert result is None
    assert chunk_plugin.calls == 1
    assert [hit.id for hit in pipeline_ctx.search_result] == ["chunk-a", "chunk-b"]
    entity_hit = next(hit for hit in pipeline_ctx.search_result if hit.id == "chunk-b")
    assert entity_hit.content == "entity payload"
    assert entity_hit.knowledge_title == "Entity Doc"


async def test_parallel_runs_chunk_search_on_a_clone_not_the_original() -> None:
    chunk_plugin = _FakeChunkSearchPlugin(results=[SearchResult(id="chunk-a", content="chunk a")])
    plugin = _make_parallel_plugin(chunk_plugin=chunk_plugin)
    pipeline_ctx = PipelineContext(tenant_id=1, rewrite_query="rewritten")

    result = await plugin.on_event(
        _FakeContext(), EventType.CHUNK_SEARCH_PARALLEL, pipeline_ctx, _noop
    )

    assert result is None
    carrier = chunk_plugin.seen_carriers[0]
    assert carrier is not pipeline_ctx
    assert pipeline_ctx.search_result == [SearchResult(id="chunk-a", content="chunk a")]


async def test_parallel_no_entities_runs_chunk_search_only() -> None:
    chunk_plugin = _FakeChunkSearchPlugin(results=[SearchResult(id="chunk-a", content="chunk a")])
    plugin = _make_parallel_plugin(chunk_plugin=chunk_plugin)
    pipeline_ctx = PipelineContext(tenant_id=1)

    result = await plugin.on_event(
        _FakeContext(), EventType.CHUNK_SEARCH_PARALLEL, pipeline_ctx, _noop
    )

    assert result is None
    assert chunk_plugin.calls == 1
    assert [hit.id for hit in pipeline_ctx.search_result] == ["chunk-a"]


async def test_parallel_empty_results_returns_err_search_nothing() -> None:
    plugin = _make_parallel_plugin()
    pipeline_ctx = PipelineContext(tenant_id=1)

    result = await plugin.on_event(
        _FakeContext(), EventType.CHUNK_SEARCH_PARALLEL, pipeline_ctx, _noop
    )

    assert result is ERR_SEARCH_NOTHING


async def test_parallel_chunk_error_surfaced_when_no_results() -> None:
    chunk_error = PluginError(description="kb unavailable", error_type="search_failed")
    chunk_plugin = _FakeChunkSearchPlugin(error=chunk_error)
    plugin = _make_parallel_plugin(chunk_plugin=chunk_plugin)
    pipeline_ctx = PipelineContext(tenant_id=1)

    result = await plugin.on_event(
        _FakeContext(), EventType.CHUNK_SEARCH_PARALLEL, pipeline_ctx, _noop
    )

    assert result is chunk_error


async def test_parallel_chunk_search_nothing_absorbed_with_entity_results() -> None:
    chunk_plugin = _FakeChunkSearchPlugin(error=ERR_SEARCH_NOTHING)
    graph = GraphData(node=[GraphNode(name="e", chunks=["chunk-b"])], relation=[])
    chunks = {"chunk-b": _chunk(id="chunk-b", content="entity payload")}
    knowledge = {"doc-1": _doc()}
    plugin = _make_parallel_plugin(
        chunk_plugin=chunk_plugin,
        graphs={("kb-1", ""): graph},
        chunks=chunks,
        knowledge=knowledge,
    )
    pipeline_ctx = PipelineContext(tenant_id=1, entity=["e"], entity_kb_ids=["kb-1"])

    result = await plugin.on_event(
        _FakeContext(), EventType.CHUNK_SEARCH_PARALLEL, pipeline_ctx, _noop
    )

    assert result is None
    assert [hit.id for hit in pipeline_ctx.search_result] == ["chunk-b"]


async def test_parallel_dedupes_merged_results() -> None:
    chunk_plugin = _FakeChunkSearchPlugin(
        results=[SearchResult(id="chunk-x", content="shared payload")]
    )
    graph = GraphData(node=[GraphNode(name="e", chunks=["chunk-x"])], relation=[])
    chunks = {"chunk-x": _chunk(id="chunk-x", content="shared payload")}
    knowledge = {"doc-1": _doc()}
    plugin = _make_parallel_plugin(
        chunk_plugin=chunk_plugin,
        graphs={("kb-1", ""): graph},
        chunks=chunks,
        knowledge=knowledge,
    )
    pipeline_ctx = PipelineContext(tenant_id=1, entity=["e"], entity_kb_ids=["kb-1"])

    result = await plugin.on_event(
        _FakeContext(), EventType.CHUNK_SEARCH_PARALLEL, pipeline_ctx, _noop
    )

    assert result is None
    assert len(pipeline_ctx.search_result) == 1


# ── Integration: real DB hydration ─────────────────────────────────────


async def test_integration_entity_search_hydrates_real_rows(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    await KnowledgeRepository(session).create(
        _doc(
            id="doc-int",
            tenant_id=tenant_id,
            title="Integration Doc",
            file_name="doc.pdf",
            metadata={"lang": "en"},
        )
    )
    await ChunkRepository(session).create(
        _chunk(
            id="chunk-int",
            tenant_id=tenant_id,
            knowledge_id="doc-int",
            content="entity integration payload",
        )
    )
    await ChunkRepository(session).create(
        _chunk(
            id="img-int",
            tenant_id=tenant_id,
            knowledge_id="doc-int",
            content="",
            parent_chunk_id="chunk-int",
            chunk_type=_IMAGE_OCR,
            image_info='[{"url":"http://img/int.png","caption":"Int alt","ocr_text":"Int ocr"}]',
        )
    )

    graph = GraphData(node=[GraphNode(name="entity-int", chunks=["chunk-int"])], relation=[])
    plugin = SearchEntityPlugin(
        graph_repo=_FakeGraphRepo(graphs={("kb-int", ""): graph}),
        chunk_repo=ChunkRepository(session),
        knowledge_repo=KnowledgeRepository(session),
    )
    pipeline_ctx = PipelineContext(
        tenant_id=tenant_id,
        entity=["entity-int"],
        entity_kb_ids=["kb-int"],
    )

    result = await plugin.on_event(_FakeContext(), EventType.ENTITY_SEARCH, pipeline_ctx, _noop)

    assert result is None
    assert len(pipeline_ctx.search_result) == 1
    hit = pipeline_ctx.search_result[0]
    assert hit.id == "chunk-int"
    assert hit.content == "entity integration payload"
    assert hit.knowledge_title == "Integration Doc"
    assert hit.knowledge_id == "doc-int"
    assert hit.match_type == MatchType.GRAPH
    assert hit.score == 1.0
    assert hit.metadata == {"lang": "en"}
    assert '"url": "http://img/int.png"' in hit.image_info


async def test_integration_parallel_merges_chunk_and_entity(session: AsyncSession) -> None:
    tenant_id = _int32_tenant_id()
    await KnowledgeRepository(session).create(
        _doc(id="doc-par", tenant_id=tenant_id, title="Parallel Doc")
    )
    await ChunkRepository(session).create(
        _chunk(
            id="chunk-entity", tenant_id=tenant_id, knowledge_id="doc-par", content="entity payload"
        )
    )

    chunk_plugin = _FakeChunkSearchPlugin(
        results=[SearchResult(id="chunk-kb", content="chunk payload")]
    )
    graph = GraphData(node=[GraphNode(name="entity-par", chunks=["chunk-entity"])], relation=[])
    plugin = SearchParallelPlugin(
        chunk_search_plugin=chunk_plugin,
        graph_repo=_FakeGraphRepo(graphs={("kb-par", ""): graph}),
        chunk_repo=ChunkRepository(session),
        knowledge_repo=KnowledgeRepository(session),
    )
    pipeline_ctx = PipelineContext(
        tenant_id=tenant_id,
        entity=["entity-par"],
        entity_kb_ids=["kb-par"],
    )

    result = await plugin.on_event(
        _FakeContext(), EventType.CHUNK_SEARCH_PARALLEL, pipeline_ctx, _noop
    )

    assert result is None
    assert chunk_plugin.calls == 1
    assert [hit.id for hit in pipeline_ctx.search_result] == ["chunk-kb", "chunk-entity"]
    entity_hit = next(hit for hit in pipeline_ctx.search_result if hit.id == "chunk-entity")
    assert entity_hit.content == "entity payload"
    assert entity_hit.knowledge_title == "Parallel Doc"
