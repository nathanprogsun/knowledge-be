"""Knowledge-QA runner that fills the ``QARunner`` seam.

The hybrid-search HTTP path is still a stub. This runner loads stored
text chunks for the turn's ``@`` files (or whole KBs), renders them
through ``INTO_CHAT_MESSAGE``, and streams via ``CHAT_COMPLETION_STREAM``.
``AGENT_COMPLETE`` is emitted after the pipeline so the SPA can leave
the thinking state.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from src.ai.llm.types import Chat
from src.ai.retrieval.types import MatchType
from src.common.exception import ValidationError
from src.common.json import JsonObject
from src.core.agents.types import CustomAgentInfo
from src.core.chat.bus import Event, EventBus
from src.core.chat.pipeline.context import PipelineContext
from src.core.chat.pipeline.engine import EventManager
from src.core.chat.pipeline.steps.into_chat_message import IntoChatMessageStep
from src.core.chat.pipeline.steps.stream import ChatCompletionStreamStep
from src.core.chat.pipeline.types import Context, EventType, SearchResult, SummaryConfig
from src.core.chat.service import KnowledgeQARequestLike, merge_knowledge_targets
from src.core.chat.sessions.knowledge_qa import execute_knowledge_qa, run_knowledge_qa
from src.core.chat.types import EventType as ChatEventType
from src.core.contracts.knowledge import Knowledge
from src.core.infra.models.chat_service import MODEL_TYPE_KNOWLEDGE_QA
from src.core.knowledge.knowledge_bases.types import KnowledgeBaseInfo

_CHUNK_CAP: int = 40
_DEFAULT_LANGUAGE: str = "zh-CN"
_DEFAULT_CONTEXT_TEMPLATE: str = (
    "Reference materials:\n{{contexts}}\n\n"
    "Question: {{query}}\n\n"
    "Please answer the above question. IMPORTANT: ALWAYS respond in {{language}}.\n"
)
_DEFAULT_SYSTEM_PROMPT: str = (
    "You are a helpful knowledge-base assistant. "
    "Answer from the supplied materials when they exist."
)


@runtime_checkable
class TextChunkLike(Protocol):
    """Text-chunk fields the runner maps onto pipeline hits."""

    id: str
    content: str
    knowledge_id: str
    knowledge_base_id: str
    chunk_index: int
    start_at: int
    end_at: int
    chunk_type: str
    parent_chunk_id: str | None
    image_info: str | None
    metadata: JsonObject | None
    is_enabled: bool


@runtime_checkable
class DocumentLoader(Protocol):
    """Loads documents for chunk hydration."""

    async def get_documents(self, *, tenant_id: int, ids: list[str]) -> list[Knowledge]: ...

    async def list_documents(
        self, *, tenant_id: int, knowledge_base_id: str
    ) -> list[Knowledge]: ...


@runtime_checkable
class ChunkLoader(Protocol):
    """Loads stored text chunks for one document."""

    async def list_chunks_by_knowledge_id(
        self, *, tenant_id: int, knowledge_id: str
    ) -> Sequence[TextChunkLike]: ...


@runtime_checkable
class KnowledgeBaseReader(Protocol):
    """Reads a tenant-scoped knowledge base for model fallback."""

    async def get_knowledge_base_by_id_and_tenant(
        self, *, tenant_id: int, knowledge_base_id: str
    ) -> KnowledgeBaseInfo: ...


@runtime_checkable
class ChatModelCatalog(Protocol):
    """Resolves KnowledgeQA models without importing the infra service type."""

    async def get_chat_model(self, *, tenant_id: int, model_id: str) -> Chat: ...

    async def get_model_type(self, *, tenant_id: int, model_id: str) -> str | None: ...

    async def first_knowledge_qa_id(self, *, tenant_id: int) -> str | None: ...


class _NoopRenderedContent:
    """Skip persisting the rendered user message until the message store lands."""

    async def update_message_rendered_content(
        self,
        ctx: Context,
        session_id: str,
        user_message_id: str,
        content: str,
    ) -> None:
        return None


class _CtxChatModels:
    """Adapts a catalog to the pipeline ``get_chat_model(ctx, id)`` seam."""

    def __init__(self, inner: ChatModelCatalog) -> None:
        self._inner = inner

    async def get_chat_model(self, ctx: Context, model_id: str) -> Chat:
        return await self._inner.get_chat_model(
            tenant_id=_tenant_id(ctx),
            model_id=model_id,
        )


def _tenant_id(ctx: Context) -> int:
    """Read a positive tenant id from the opaque pipeline context."""
    raw = getattr(ctx, "tenant_id", 0)
    if isinstance(raw, int) and raw > 0:
        return raw
    raise ValidationError(
        code="chat.tenant_context_missing",
        message="No active workspace in request context",
    )


def _user_id(ctx: Context) -> str:
    """Read the caller user id from the opaque pipeline context."""
    raw = getattr(ctx, "user_id", "")
    return raw if isinstance(raw, str) else ""


def default_summary_config() -> SummaryConfig:
    """Sampling defaults so a zeroed ``SummaryConfig`` does not starve the model."""
    return SummaryConfig(
        max_tokens=4096,
        max_completion_tokens=2048,
        temperature=0.7,
        top_p=0.9,
        context_template=_DEFAULT_CONTEXT_TEMPLATE,
        prompt=_DEFAULT_SYSTEM_PROMPT,
    )


def summary_config_for_agent(agent: CustomAgentInfo | None) -> SummaryConfig:
    """Prefer the agent's prompt and sampling when those fields are set."""
    config = default_summary_config()
    if agent is None:
        return config
    updates: dict[str, str | float | int] = {}
    prompt = agent.config.get("system_prompt")
    if isinstance(prompt, str) and prompt.strip():
        updates["prompt"] = prompt.strip()
    temperature = agent.config.get("temperature")
    if isinstance(temperature, int | float) and not isinstance(temperature, bool):
        updates["temperature"] = float(temperature)
    max_tokens = agent.config.get("max_completion_tokens")
    if isinstance(max_tokens, int) and not isinstance(max_tokens, bool) and max_tokens > 0:
        updates["max_completion_tokens"] = max_tokens
    return config.model_copy(update=updates) if updates else config


def _agent_model_id(agent: CustomAgentInfo | None) -> str:
    """Return the agent's configured chat model id, if any."""
    if agent is None:
        return ""
    value = agent.config.get("model_id")
    return value.strip() if isinstance(value, str) else ""


def _string_metadata(raw: JsonObject | None) -> dict[str, str]:
    """Keep only string metadata entries for the pipeline hit."""
    if not raw:
        return {}
    return {
        key: item for key, item in raw.items() if isinstance(key, str) and isinstance(item, str)
    }


def chunk_to_search_result(chunk: TextChunkLike, document: Knowledge | None) -> SearchResult:
    """Map a stored text chunk onto a pipeline search hit."""
    title = ""
    filename = ""
    source = ""
    channel = ""
    description = ""
    if document is not None:
        title = document.title or ""
        filename = document.file_name or ""
        source = document.source or ""
        channel = document.channel or ""
        description = document.description or ""
    return SearchResult(
        id=chunk.id,
        content=chunk.content,
        knowledge_id=chunk.knowledge_id,
        chunk_index=chunk.chunk_index,
        knowledge_title=title,
        start_at=chunk.start_at,
        end_at=chunk.end_at,
        seq=chunk.chunk_index,
        score=1.0,
        match_type=MatchType.KEYWORDS,
        metadata=_string_metadata(chunk.metadata),
        chunk_type=chunk.chunk_type or "text",
        parent_chunk_id=chunk.parent_chunk_id or "",
        image_info=chunk.image_info or "",
        knowledge_filename=filename,
        knowledge_source=source,
        knowledge_channel=channel,
        knowledge_description=description,
        knowledge_base_id=chunk.knowledge_base_id,
    )


async def resolve_knowledge_qa_model_id(
    *,
    tenant_id: int,
    request: KnowledgeQARequestLike,
    agent: CustomAgentInfo | None,
    knowledge_base_ids: Sequence[str],
    chat_models: ChatModelCatalog,
    knowledge_bases: KnowledgeBaseReader,
) -> str:
    """Pick the KnowledgeQA model for this turn.

    Request ``summary_model_id`` wins when it is a KnowledgeQA model.
    The agent config, then the first KB ``summary_model_id``, then the
    first tenant KnowledgeQA model, follow in that order.
    """
    candidates: list[str] = []
    override = (request.summary_model_id or "").strip()
    if override:
        candidates.append(override)
    agent_model = _agent_model_id(agent)
    if agent_model:
        candidates.append(agent_model)
    for kb_id in knowledge_base_ids:
        kb_model = await _kb_summary_model_id(
            tenant_id=tenant_id,
            knowledge_base_id=kb_id,
            knowledge_bases=knowledge_bases,
        )
        if kb_model:
            candidates.append(kb_model)
    fallback = await chat_models.first_knowledge_qa_id(tenant_id=tenant_id)
    if fallback:
        candidates.append(fallback)
    return await _first_knowledge_qa_id(tenant_id, candidates, chat_models)


async def _kb_summary_model_id(
    *,
    tenant_id: int,
    knowledge_base_id: str,
    knowledge_bases: KnowledgeBaseReader,
) -> str:
    """Return a KB's summary model id, or empty when the KB cannot be read."""
    try:
        kb = await knowledge_bases.get_knowledge_base_by_id_and_tenant(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
        )
    except Exception:
        return ""
    return (kb.summary_model_id or "").strip()


async def _first_knowledge_qa_id(
    tenant_id: int,
    candidates: Sequence[str],
    chat_models: ChatModelCatalog,
) -> str:
    """Return the first candidate whose stored type is KnowledgeQA."""
    seen: set[str] = set()
    for model_id in candidates:
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        model_type = await chat_models.get_model_type(
            tenant_id=tenant_id,
            model_id=model_id,
        )
        if model_type == MODEL_TYPE_KNOWLEDGE_QA:
            return model_id
    raise ValidationError(
        code="chat.chat_model_required",
        message="No KnowledgeQA model is available for this turn",
    )


async def collect_document_ids(
    *,
    tenant_id: int,
    knowledge_ids: Sequence[str],
    knowledge_base_ids: Sequence[str],
    documents: DocumentLoader,
) -> list[str]:
    """Resolve the documents whose chunks should ground this turn.

    Explicit file ids win. Otherwise every live document in the named
    KBs is included, in list order.
    """
    if knowledge_ids:
        return [kid for kid in knowledge_ids if kid]
    collected: list[str] = []
    seen: set[str] = set()
    for kb_id in knowledge_base_ids:
        if not kb_id:
            continue
        rows = await documents.list_documents(tenant_id=tenant_id, knowledge_base_id=kb_id)
        for row in rows:
            if row.id in seen:
                continue
            seen.add(row.id)
            collected.append(row.id)
    return collected


async def load_chunk_hits(
    *,
    tenant_id: int,
    knowledge_ids: Sequence[str],
    documents: DocumentLoader,
    chunks: ChunkLoader,
    cap: int = _CHUNK_CAP,
) -> list[SearchResult]:
    """Load enabled text chunks and project them as pipeline hits."""
    if not knowledge_ids or cap <= 0:
        return []
    docs = await documents.get_documents(tenant_id=tenant_id, ids=list(knowledge_ids))
    by_id: dict[str, Knowledge] = {doc.id: doc for doc in docs}
    hits: list[SearchResult] = []
    for knowledge_id in knowledge_ids:
        rows = await chunks.list_chunks_by_knowledge_id(
            tenant_id=tenant_id,
            knowledge_id=knowledge_id,
        )
        for row in rows:
            if not row.is_enabled or not row.content.strip():
                continue
            hits.append(chunk_to_search_result(row, by_id.get(knowledge_id)))
            if len(hits) >= cap:
                return hits
    return hits


def build_knowledge_qa_event_manager(
    *,
    chat_models: ChatModelCatalog,
    event_bus: EventBus,
) -> EventManager:
    """Register the two stages this runner actually executes."""
    manager = EventManager()
    manager.register(IntoChatMessageStep(_NoopRenderedContent()))
    manager.register(ChatCompletionStreamStep(_CtxChatModels(chat_models), event_bus))
    return manager


class KnowledgeQARunner:
    """``QARunner`` that grounds a turn in stored chunks and streams an answer."""

    def __init__(
        self,
        *,
        chat_models: ChatModelCatalog,
        documents: DocumentLoader,
        chunks: ChunkLoader,
        knowledge_bases: KnowledgeBaseReader,
    ) -> None:
        self._chat_models = chat_models
        self._documents = documents
        self._chunks = chunks
        self._knowledge_bases = knowledge_bases

    async def run(
        self,
        *,
        ctx: Context,
        session_id: str,
        request: KnowledgeQARequestLike,
        agent: CustomAgentInfo | None,
        event_bus: EventBus,
    ) -> None:
        tenant_id = _tenant_id(ctx)
        kb_ids, doc_ids = merge_knowledge_targets(
            knowledge_base_ids=list(request.knowledge_base_ids or []),
            knowledge_ids=list(request.knowledge_ids or []),
            mentioned_items=request.mentioned_items,
        )
        model_id = await resolve_knowledge_qa_model_id(
            tenant_id=tenant_id,
            request=request,
            agent=agent,
            knowledge_base_ids=kb_ids,
            chat_models=self._chat_models,
            knowledge_bases=self._knowledge_bases,
        )
        pipeline_ctx = PipelineContext(
            session_id=session_id,
            user_id=_user_id(ctx),
            query=request.query,
            max_rounds=0,
            knowledge_base_ids=kb_ids,
            knowledge_ids=doc_ids,
            chat_model_id=model_id,
            summary_config=summary_config_for_agent(agent),
            tenant_id=tenant_id,
            language=_DEFAULT_LANGUAGE,
        )
        await self._execute(ctx, pipeline_ctx, kb_ids, doc_ids, event_bus)
        await event_bus.emit(
            Event(
                type=ChatEventType.AGENT_COMPLETE,
                session_id=session_id,
                data={"done": True},
            )
        )

    async def _execute(
        self,
        ctx: Context,
        pipeline_ctx: PipelineContext,
        kb_ids: list[str],
        doc_ids: list[str],
        event_bus: EventBus,
    ) -> None:
        """Run RAG stages when the turn names files or KBs, else pure chat."""
        manager = build_knowledge_qa_event_manager(
            chat_models=self._chat_models,
            event_bus=event_bus,
        )
        has_scope = bool(kb_ids or doc_ids)
        if not has_scope:
            await run_knowledge_qa(
                ctx=ctx,
                event_manager=manager,
                pipeline_ctx=pipeline_ctx,
                event_bus=event_bus,
            )
            return
        target_ids = await collect_document_ids(
            tenant_id=pipeline_ctx.tenant_id,
            knowledge_ids=doc_ids,
            knowledge_base_ids=kb_ids,
            documents=self._documents,
        )
        pipeline_ctx.merge_result = await load_chunk_hits(
            tenant_id=pipeline_ctx.tenant_id,
            knowledge_ids=target_ids,
            documents=self._documents,
            chunks=self._chunks,
        )
        await execute_knowledge_qa(
            ctx=ctx,
            event_manager=manager,
            pipeline_ctx=pipeline_ctx,
            stages=[EventType.INTO_CHAT_MESSAGE, EventType.CHAT_COMPLETION_STREAM],
            event_bus=event_bus,
        )


__all__ = [
    "ChatModelCatalog",
    "ChunkLoader",
    "DocumentLoader",
    "KnowledgeBaseReader",
    "KnowledgeQARunner",
    "TextChunkLike",
    "chunk_to_search_result",
    "collect_document_ids",
    "default_summary_config",
    "load_chunk_hits",
    "resolve_knowledge_qa_model_id",
    "summary_config_for_agent",
]
