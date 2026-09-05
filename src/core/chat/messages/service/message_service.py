"""Message service — request-scoped CRUD, search, and KB indexing.

``MessageService`` mirrors the upstream message service surface: it
manages message persistence scoped to its owning session, performs
keyword / vector / hybrid chat-history search, and orchestrates the
index-into-KB and cleanup flows that link a message to its Knowledge
entry in the workspace's chat-history knowledge base.

Layering
--------

The service depends on the persistence seam (``MessageRepository``) for
storage and on structural protocols (``MessageVectorSearcher``,
``ChatHistoryConfigProvider``) for capabilities whose full Python
implementation lives in later PRs. The protocols are runtime-checkable
so test doubles can be slotted in directly. Session existence is
verified through ``SessionRepository``; passing ``None`` for the
repository skips the check (used by tests and by the index_to_kb
helper, which the chat pipeline triggers after the session has already
been validated).

The service is request-scoped — its ``__init__`` binds a single
``AsyncSession`` and every method is stateless beyond that. No class-
level caches, no module-level singletons.

Names
-----

Field names on every value type here (``query``, ``mode``, ``limit``,
``session_ids``, ``match_type``, ``score``, ...) match the upstream
Go contract: Pydantic ``snake_case`` for params, frozen dataclasses
for the grouping shapes used in the JSON wire payload.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, cast, runtime_checkable

from src.common.exception import NotFoundError, ValidationError
from src.common.json import BindParams, JsonValue
from src.core.chat.messages.types import MessageInfo, MessageSearchMode
from src.core.chat.pipeline.types import Context
from src.db.dao.message_repository import MessageRepository
from src.db.dao.session_repository import SessionRepository
from src.db.models.message import Message

#: Default page size when the caller does not specify one.
_DEFAULT_PAGE_SIZE = 20

#: Default cap on chat-history search results.
_DEFAULT_SEARCH_LIMIT = 20

#: Sentinel for the disabled / unconfigured chat-history KB stats response.
CHAT_HISTORY_KB_STATS_DISABLED: bool = False


# ── Value shapes (wire DTOs) ───────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MessageSearchParams:
    """Parameters for a chat-history search call.

    Mirrors the upstream ``MessageSearchParams``: ``query`` is required,
    ``mode`` defaults to ``hybrid`` and ``limit`` to 20 when unset, and
    an empty ``session_ids`` list means "search all sessions of the
    caller's workspace".
    """

    query: str
    mode: MessageSearchMode = MessageSearchMode.HYBRID
    limit: int = _DEFAULT_SEARCH_LIMIT
    session_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MessageWithSession:
    """A stored message plus its session title.

    Mirrors the upstream ``MessageWithSession``: the message row carries
    every column the persistence layer exposes, and ``session_title``
    is filled in by the read path that joins against the sessions
    table. The chat-history search result is grouped and presented to
    the client through this shape.
    """

    message: Message
    session_title: str = ""


@dataclass(frozen=True, slots=True)
class MessageSearchResultItem:
    """A single matched message before the request_id grouping step.

    Mirrors the upstream ``MessageSearchResultItem``: the matched
    message (with session title) is paired with a ``score`` and a
    ``match_type`` describing whether the row was hit by keyword,
    vector, or hybrid fusion.
    """

    message: MessageWithSession
    score: float = 0.0
    match_type: str = ""


@dataclass(frozen=True, slots=True)
class MessageSearchGroupItem:
    """A merged Q&A pair presented as one chat-history search hit.

    Mirrors the upstream ``MessageSearchGroupItem``: messages sharing
    the same ``request_id`` are folded into a single group whose
    ``query_content`` is the user message and ``answer_content`` is the
    assistant message. The group carries the best score across its
    members and a ``match_type`` describing how the group was matched.
    """

    request_id: str
    session_id: str
    session_title: str
    query_content: str
    answer_content: str
    score: float
    match_type: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class MessageSearchResult:
    """Chat-history search response payload.

    Mirrors the upstream ``MessageSearchResult``: ``items`` is the list
    of merged Q&A pairs (already truncated to ``params.limit``), and
    ``total`` is its length.
    """

    items: tuple[MessageSearchGroupItem, ...] = ()
    total: int = 0


@dataclass(frozen=True, slots=True)
class ChatHistoryKBStats:
    """Statistics about the chat-history knowledge base.

    Mirrors the upstream ``ChatHistoryKBStats``: when ``enabled`` is
    false the workspace has no chat-history KB wired up, so the count
    fields stay zero and the identifier fields stay empty.
    """

    enabled: bool = CHAT_HISTORY_KB_STATS_DISABLED
    embedding_model_id: str = ""
    knowledge_base_id: str = ""
    knowledge_base_name: str = ""
    indexed_message_count: int = 0
    has_indexed_messages: bool = False


# ── Injectable seams ───────────────────────────────────────────────────


@runtime_checkable
class MessageVectorSearcher(Protocol):
    """Vector-search seam delegated to the chat-history KB.

    The implementation calls the workspace's chat-history KB HybridSearch
    with vector-only mode and returns the matched knowledge entries
    mapped back to their source messages. The ``MessageService``
    orchestrates keyword + vector fusion on top of these results.
    """

    async def search_by_vector(
        self,
        *,
        ctx: Context,
        query: str,
        knowledge_base_id: str,
        embedding_top_k: int,
        vector_threshold: float,
        session_ids: tuple[str, ...],
    ) -> list[MessageSearchResultItem]: ...


@runtime_checkable
class ChatHistoryConfigProvider(Protocol):
    """Resolves the workspace's chat-history KB configuration.

    The full tenant-side config object lands in a later PR; the service
    consumes only the fields it actually needs (``enabled``,
    ``embedding_model_id``, ``knowledge_base_id``).
    """

    def is_enabled(self, ctx: Context) -> bool: ...

    def knowledge_base_id(self, ctx: Context) -> str: ...

    def embedding_model_id(self, ctx: Context) -> str: ...

    def effective_embedding_top_k(self, ctx: Context) -> int: ...

    def effective_vector_threshold(self, ctx: Context) -> float: ...


@runtime_checkable
class MessageIndexer(Protocol):
    """Chat-history KB lifecycle surface the service consumes.

    Mirrors the indexer operations exposed by
    :mod:`src.core.chat.messages.index_to_kb`. The service treats the
    indexer as an optional seam — when it is absent, every KB-related
    method becomes a silent no-op so the message service can still be
    wired in a workspace whose chat-history KB is not configured.
    """

    async def index_message(
        self,
        ctx: Context,
        *,
        user_query: str,
        assistant_answer: str,
        message_id: str,
        session_id: str,
    ) -> None: ...

    async def delete_message_knowledge(self, *, knowledge_id: str) -> None: ...

    async def delete_session_knowledge(
        self,
        *,
        knowledge_ids: tuple[str, ...],
    ) -> None: ...


# ── Service contract ──────────────────────────────────────────────────


@runtime_checkable
class MessageService(Protocol):
    """Full surface mirroring the upstream message service.

    Every method is a thin orchestration over the persistence seam plus
    the optional Protocol-based search / KB-indexing seams. The CRUD
    methods are scoped to the session owner; the search and KB-indexing
    methods are tenant-scoped per the upstream rule.
    """

    async def create_message(
        self,
        ctx: Context,
        message: Message,
    ) -> MessageInfo: ...

    async def get_message(
        self,
        ctx: Context,
        session_id: str,
        message_id: str,
    ) -> MessageInfo: ...

    async def list_messages_by_session(
        self,
        ctx: Context,
        session_id: str,
        page: int,
        page_size: int,
    ) -> list[MessageInfo]: ...

    async def get_recent_messages_by_session(
        self,
        ctx: Context,
        session_id: str,
        limit: int,
    ) -> list[MessageInfo]: ...

    async def list_messages_by_session_before_time(
        self,
        ctx: Context,
        session_id: str,
        before_time: datetime,
        limit: int,
    ) -> list[MessageInfo]: ...

    async def update_message(
        self,
        ctx: Context,
        message: Message,
    ) -> MessageInfo: ...

    async def update_message_images(
        self,
        ctx: Context,
        session_id: str,
        message_id: str,
        images: JsonValue,
    ) -> MessageInfo: ...

    async def update_message_rendered_content(
        self,
        ctx: Context,
        session_id: str,
        message_id: str,
        rendered_content: str,
    ) -> MessageInfo: ...

    async def delete_message(
        self,
        ctx: Context,
        session_id: str,
        message_id: str,
    ) -> bool: ...

    async def clear_session_messages(
        self,
        ctx: Context,
        session_id: str,
    ) -> int: ...

    async def search_messages(
        self,
        ctx: Context,
        params: MessageSearchParams,
    ) -> MessageSearchResult: ...

    async def index_message_to_kb(
        self,
        ctx: Context,
        *,
        user_query: str,
        assistant_answer: str,
        message_id: str,
        session_id: str,
    ) -> None: ...

    async def delete_message_knowledge(
        self,
        ctx: Context,
        knowledge_id: str,
    ) -> None: ...

    async def delete_session_knowledge(
        self,
        ctx: Context,
        session_id: str,
    ) -> None: ...

    async def get_chat_history_kb_stats(
        self,
        ctx: Context,
    ) -> ChatHistoryKBStats: ...


# ── Concrete implementation ───────────────────────────────────────────


def _require_query(query: str) -> None:
    """Reject blank search queries at the service boundary."""
    if not query or not query.strip():
        raise ValidationError(
            code="message.search_query_required",
            message="Search query cannot be empty",
        )


def _now() -> datetime:
    """UTC ``now`` — a single seam so tests can monkey-patch the clock."""
    return datetime.now(UTC)


class MessageServiceImpl:
    """Request-scoped message service — orchestrates CRUD, search, KB indexing.

    Constructor parameters are keyword-only so the call ordering cannot
    drift silently across refactors. ``message_repo`` is mandatory;
    ``session_repo`` is mandatory unless the caller is a privileged path
    (the index_to_kb helper) that already knows the session is valid.
    ``vector_searcher`` and ``chat_history_config`` are optional — when
    either is absent, search degrades to keyword-only and KB indexing /
    stats return safe no-ops.
    """

    def __init__(
        self,
        *,
        message_repo: MessageRepository,
        session_repo: SessionRepository | None = None,
        vector_searcher: MessageVectorSearcher | None = None,
        chat_history_config: ChatHistoryConfigProvider | None = None,
        indexer: MessageIndexer | None = None,
    ) -> None:
        self._message_repo = message_repo
        self._session_repo = session_repo
        self._vector_searcher = vector_searcher
        self._chat_history_config = chat_history_config
        self._indexer = indexer

    # ── Session-existence helpers ──────────────────────────────────

    async def _require_session(self, ctx: Context, session_id: str) -> None:
        """Validate that ``session_id`` exists for the caller's workspace.

        Mirrors the upstream guard: every CRUD call that touches a
        session first confirms the session is reachable. When the
        session repo is unavailable (privileged path), the check is
        skipped.
        """
        if self._session_repo is None:
            return
        tenant_id = getattr(ctx, "tenant_id", 0)
        row = await self._session_repo.get_by_id(tenant_id=tenant_id, id=session_id)
        if row is None:
            raise NotFoundError(
                code="message.session_not_found",
                message=f"session {session_id} not found",
            )

    # ── CRUD ───────────────────────────────────────────────────────

    async def create_message(
        self,
        ctx: Context,
        message: Message,
    ) -> MessageInfo:
        """Persist a new message after verifying its session.

        ``message`` carries the caller-supplied ``id`` (UUID), the
        owning ``session_id``, ``role`` and ``content``. The repository
        insert returns the row as written, so the caller sees the
        database-assigned timestamps.
        """
        await self._require_session(ctx, message.session_id)
        row = await self._message_repo.create(message)
        return MessageInfo.map_from_db(row)

    async def get_message(
        self,
        ctx: Context,
        session_id: str,
        message_id: str,
    ) -> MessageInfo:
        """Return one live message by ``(session_id, message_id)``.

        Raises ``NotFoundError`` when the row is missing or soft-deleted,
        matching the upstream service's behaviour.
        """
        await self._require_session(ctx, session_id)
        row = await self._message_repo.get_by_id_and_session(
            session_id=session_id,
            message_id=message_id,
        )
        if row is None:
            raise NotFoundError(
                code="message.not_found",
                message=f"message {message_id} not found in session {session_id}",
            )
        return MessageInfo.map_from_db(row)

    async def list_messages_by_session(
        self,
        ctx: Context,
        session_id: str,
        page: int,
        page_size: int,
    ) -> list[MessageInfo]:
        """Paginated session feed, oldest first.

        Defaults ``page_size`` to 20 when zero / negative so a sloppy
        caller still gets a bounded response.
        """
        await self._require_session(ctx, session_id)
        if page < 1:
            page = 1
        if page_size <= 0:
            page_size = _DEFAULT_PAGE_SIZE
        rows = await self._message_repo.list_by_session(
            session_id,
            page=page,
            page_size=page_size,
        )
        return [MessageInfo.map_from_db(row) for row in rows]

    async def get_recent_messages_by_session(
        self,
        ctx: Context,
        session_id: str,
        limit: int,
    ) -> list[MessageInfo]:
        """Return the most recent messages of a session, chronological.

        The persistence layer fetches the newest ``limit`` rows newest-
        first and re-sorts them so the caller sees a chronological
        window with user turns first on equal timestamps.
        """
        await self._require_session(ctx, session_id)
        rows = await self._message_repo.list_recent_by_session(
            session_id,
            limit=limit,
        )
        return [MessageInfo.map_from_db(row) for row in rows]

    async def list_messages_by_session_before_time(
        self,
        ctx: Context,
        session_id: str,
        before_time: datetime,
        limit: int,
    ) -> list[MessageInfo]:
        """Messages of a session created strictly before ``before_time``.

        Used by the chat pipeline to assemble history for a turn that
        just received a new user message: the new message's timestamp
        is the cutoff.
        """
        await self._require_session(ctx, session_id)
        rows = await self._message_repo.list_by_session_before_time(
            session_id,
            before_time=before_time,
            limit=limit,
        )
        return [MessageInfo.map_from_db(row) for row in rows]

    async def update_message(
        self,
        ctx: Context,
        message: Message,
    ) -> MessageInfo:
        """Persist the mutable columns of ``message`` back to storage.

        Mirrors the upstream ``UpdateMessage``: the identity columns
        (``id`` / ``session_id``) stay untouched; everything else is
        overwritten from the input row. The repository returns the
        post-update row so the caller sees the refreshed timestamps.
        """
        await self._require_session(ctx, message.session_id)
        # The repository's ``update`` expects a column-only update map;
        # route through a small allow-list of mutable columns to avoid
        # accidentally rewriting ``id`` / ``session_id`` / ``created_at``.
        updates = self._mutable_columns(message)
        updated = await self._message_repo.update(
            session_id=message.session_id,
            message_id=message.id,
            column_to_update=updates,
        )
        if updated is None:
            raise NotFoundError(
                code="message.not_found",
                message=f"message {message.id} not found in session {message.session_id}",
            )
        return MessageInfo.map_from_db(updated)

    @staticmethod
    def _mutable_columns(message: Message) -> BindParams:
        """Return the columns a caller is allowed to overwrite."""
        return {
            "request_id": message.request_id,
            "role": message.role,
            "content": message.content,
            "knowledge_references": message.knowledge_references,
            "agent_steps": message.agent_steps,
            "is_completed": message.is_completed,
            "is_fallback": message.is_fallback,
            "agent_duration_ms": message.agent_duration_ms,
            "rendered_content": message.rendered_content,
            "channel": message.channel,
            "agent_id": message.agent_id,
            "agent_tenant_id": message.agent_tenant_id,
            "model_id": message.model_id,
            "execution_context": message.execution_context,
            "knowledge_id": message.knowledge_id,
            "mentioned_items": message.mentioned_items,
            "images": message.images,
            "attachments": message.attachments,
            "updated_at": _now(),
        }

    async def update_message_images(
        self,
        ctx: Context,
        session_id: str,
        message_id: str,
        images: JsonValue,
    ) -> MessageInfo:
        """Overwrite the ``images`` JSONB column of a message."""
        await self._require_session(ctx, session_id)
        updated = await self._message_repo.update_images(
            session_id=session_id,
            message_id=message_id,
            images=images,
        )
        if updated is None:
            raise NotFoundError(
                code="message.not_found",
                message=f"message {message_id} not found in session {session_id}",
            )
        return MessageInfo.map_from_db(updated)

    async def update_message_rendered_content(
        self,
        ctx: Context,
        session_id: str,
        message_id: str,
        rendered_content: str,
    ) -> MessageInfo:
        """Overwrite the ``rendered_content`` column of a user message."""
        await self._require_session(ctx, session_id)
        updated = await self._message_repo.update_rendered_content(
            session_id=session_id,
            message_id=message_id,
            rendered_content=rendered_content,
        )
        if updated is None:
            raise NotFoundError(
                code="message.not_found",
                message=f"message {message_id} not found in session {session_id}",
            )
        return MessageInfo.map_from_db(updated)

    async def delete_message(
        self,
        ctx: Context,
        session_id: str,
        message_id: str,
    ) -> bool:
        """Soft-delete a message and best-effort cleanup its KB link.

        Mirrors the upstream ``DeleteMessage``: the message is marked
        deleted via the repository, then the indexer (when wired) is
        asked to drop any chat-history knowledge entry the message was
        linked to. The KB cleanup is fire-and-forget; failures are
        surfaced through the indexer protocol and never propagate to
        the caller.
        """
        await self._require_session(ctx, session_id)
        row = await self._message_repo.get_by_id_and_session(
            session_id=session_id,
            message_id=message_id,
        )
        deleted = await self._message_repo.soft_delete(
            session_id=session_id,
            message_id=message_id,
            now=_now(),
        )
        if deleted and row is not None and row.knowledge_id:
            await self.delete_message_knowledge(ctx, row.knowledge_id)
        return deleted

    async def clear_session_messages(
        self,
        ctx: Context,
        session_id: str,
    ) -> int:
        """Soft-delete every message of a session and clean up KB links.

        Mirrors the upstream ``ClearSessionMessages``: the session is
        first verified, every live message is marked deleted, and then
        the indexer (when wired) is asked to drop the chat-history
        knowledge entries that belonged to the session. The KB cleanup
        runs unconditionally so a session whose messages were all
        previously deleted still has its orphan KB entries purged.
        Returns the number of messages soft-deleted.
        """
        await self._require_session(ctx, session_id)
        deleted = await self._message_repo.soft_delete_by_session(
            session_id,
            now=_now(),
        )
        await self.delete_session_knowledge(ctx, session_id)
        return deleted

    # ── Search ─────────────────────────────────────────────────────

    async def search_messages(
        self,
        ctx: Context,
        params: MessageSearchParams,
    ) -> MessageSearchResult:
        """Search chat history by keyword, vector, or hybrid fusion.

        Mirrors the upstream ``SearchMessages`` orchestration:

        - keyword path runs the message-table ILIKE search directly;
        - vector path delegates to ``MessageVectorSearcher`` against
          the chat-history KB and maps hits back to messages;
        - hybrid path merges the two via Reciprocal Rank Fusion (RRF,
          ``k = 60``).
        """
        _require_query(params.query)
        limit = params.limit if params.limit > 0 else _DEFAULT_SEARCH_LIMIT
        session_ids = await self._resolve_search_scope(ctx, params.session_ids)
        if not session_ids:
            return MessageSearchResult(items=(), total=0)

        keyword_items: list[MessageSearchResultItem] = []
        vector_items: list[MessageSearchResultItem] = []

        if params.mode in (MessageSearchMode.KEYWORD, MessageSearchMode.HYBRID):
            keyword_items = await self._keyword_search(
                query=params.query,
                session_ids=session_ids,
                limit=limit,
            )

        if params.mode in (MessageSearchMode.VECTOR, MessageSearchMode.HYBRID):
            vector_items = await self._vector_search(
                ctx=ctx,
                params=params,
                session_ids=session_ids,
            )

        if params.mode == MessageSearchMode.KEYWORD:
            merged = keyword_items
        elif params.mode == MessageSearchMode.VECTOR:
            merged = vector_items
        else:
            merged = _rrf_merge(keyword_items, vector_items)

        merged = await self._fetch_partner_messages(merged, session_ids=session_ids)

        grouped = _group_by_request_id(merged)
        if len(grouped) > limit:
            grouped = grouped[:limit]
        return MessageSearchResult(
            items=tuple(grouped),
            total=len(grouped),
        )

    async def _resolve_search_scope(
        self,
        ctx: Context,
        requested: tuple[str, ...],
    ) -> list[str]:
        """Narrow the caller's session filter to tenant-owned sessions.

        The messages table carries no tenant column, so every search
        must run against caller-visible session ids: supplied ids are
        verified one by one against the sessions table; when the caller
        supplies none, the tenant's live sessions are enumerated up
        front. Both paths then flow through the repository's mandatory
        ``session_id in (…)`` filter.
        """
        if self._session_repo is None:
            return list(requested)
        tenant_id = getattr(ctx, "tenant_id", 0)
        if requested:
            for session_id in requested:
                await self._require_session(ctx, session_id)
            return list(requested)
        return await self._session_repo.list_ids_by_tenant(tenant_id=tenant_id)

    async def _keyword_search(
        self,
        *,
        query: str,
        session_ids: list[str],
        limit: int,
    ) -> list[MessageSearchResultItem]:
        """Direct ILIKE search of the messages table.

        Over-fetch by 3x so the upstream ``limit * 3`` heuristic — used
        to give RRF enough headroom — is preserved. With fewer hits the
        hybrid path would simply have less to merge.
        """
        rows = await self._message_repo.search_by_keyword(
            keyword=query,
            session_ids=session_ids,
            limit=limit * 3,
        )
        items: list[MessageSearchResultItem] = []
        total = len(rows)
        for index, row in enumerate(rows):
            score = float(total - index) / float(total) if total else 0.0
            items.append(
                MessageSearchResultItem(
                    message=MessageWithSession(message=row),
                    score=score,
                    match_type="keyword",
                )
            )
        return items

    async def _vector_search(
        self,
        *,
        ctx: Context,
        params: MessageSearchParams,
        session_ids: list[str],
    ) -> list[MessageSearchResultItem]:
        """Delegate to the vector search seam when it is wired.

        When the seam is absent, the workspace is not configured for
        chat-history vector search — return an empty list so the
        hybrid path falls back to keyword-only without raising.
        """
        if self._vector_searcher is None:
            return []
        if self._chat_history_config is None:
            return []
        if not self._chat_history_config.is_enabled(ctx):
            return []
        kb_id = self._chat_history_config.knowledge_base_id(ctx)
        if not kb_id:
            return []
        return await self._vector_searcher.search_by_vector(
            ctx=ctx,
            query=params.query,
            knowledge_base_id=kb_id,
            embedding_top_k=self._chat_history_config.effective_embedding_top_k(ctx),
            vector_threshold=self._chat_history_config.effective_vector_threshold(ctx),
            session_ids=tuple(session_ids),
        )

    async def _fetch_partner_messages(
        self,
        items: list[MessageSearchResultItem],
        *,
        session_ids: list[str],
    ) -> list[MessageSearchResultItem]:
        """Fill in the partner message of any Q&A pair matched on one side.

        Mirrors the upstream ``fetchPartnerMessages``: when a search hit
        has only the user turn or only the assistant turn for its
        ``request_id``, the matching partner is fetched from the
        messages table so the request_id grouping step can produce a
        complete Q&A pair. Hits that already carry both roles, or hits
        with no ``request_id``, are passed through untouched.
        """
        if not items:
            return items

        @dataclass
        class _Role:
            has_user: bool = False
            has_assistant: bool = False

        roles_by_request: dict[str, _Role] = {}
        seen_ids: set[str] = set()
        for item in items:
            seen_ids.add(item.message.message.id)
            request_id = item.message.message.request_id
            if not request_id:
                continue
            entry = roles_by_request.setdefault(request_id, _Role())
            if item.message.message.role == "user":
                entry.has_user = True
            elif item.message.message.role == "assistant":
                entry.has_assistant = True

        incomplete = [
            rid
            for rid, roles in roles_by_request.items()
            if not (roles.has_user and roles.has_assistant)
        ]
        if not incomplete:
            return items

        try:
            partners = await self._message_repo.list_by_request_ids(
                incomplete,
                session_ids=session_ids,
            )
        except Exception:
            return items

        for partner in partners:
            if partner.id in seen_ids:
                continue
            seen_ids.add(partner.id)
            items.append(
                MessageSearchResultItem(
                    message=MessageWithSession(message=partner),
                    score=0.0,
                    match_type="",
                )
            )
        return items

    # ── KB index / cleanup ─────────────────────────────────────────

    async def index_message_to_kb(
        self,
        ctx: Context,
        *,
        user_query: str,
        assistant_answer: str,
        message_id: str,
        session_id: str,
    ) -> None:
        """Index a Q&A pair into the chat-history knowledge base.

        Delegates to the ``indexer`` seam when wired; returns silently
        when the seam is absent (deferred integration). The helper is
        defined in :mod:`src.core.chat.messages.index_to_kb`; this
        method is a thin dispatcher that validates inputs and forwards.
        """
        if self._indexer is None:
            return
        index_message = getattr(self._indexer, "index_message", None)
        if index_message is None:
            return
        await index_message(
            ctx=ctx,
            user_query=user_query,
            assistant_answer=assistant_answer,
            message_id=message_id,
            session_id=session_id,
        )

    async def delete_message_knowledge(
        self,
        ctx: Context,
        knowledge_id: str,
    ) -> None:
        """Drop the chat-history knowledge entry linked to a message."""
        if self._indexer is None or not knowledge_id:
            return
        delete = getattr(self._indexer, "delete_message_knowledge", None)
        if delete is None:
            return
        await delete(knowledge_id=knowledge_id)

    async def delete_session_knowledge(
        self,
        ctx: Context,
        session_id: str,
    ) -> None:
        """Drop every chat-history knowledge entry of a session."""
        if self._indexer is None:
            return
        delete = getattr(self._indexer, "delete_session_knowledge", None)
        if delete is None:
            return
        knowledge_ids = await self._message_repo.list_knowledge_ids_by_session(session_id)
        if not knowledge_ids:
            return
        await delete(knowledge_ids=tuple(knowledge_ids))

    async def get_chat_history_kb_stats(
        self,
        ctx: Context,
    ) -> ChatHistoryKBStats:
        """Return chat-history KB stats for the caller's workspace.

        Mirrors the upstream ``GetChatHistoryKBStats``: when the
        workspace has not configured a chat-history KB, the response is
        a disabled stub so the UI can lock the embedding-model picker.
        """
        if self._chat_history_config is None or not self._chat_history_config.is_enabled(ctx):
            return ChatHistoryKBStats()
        provider_stats = getattr(self._chat_history_config, "stats", None)
        if provider_stats is None:
            return ChatHistoryKBStats(
                enabled=True,
                embedding_model_id=self._chat_history_config.embedding_model_id(ctx),
                knowledge_base_id=self._chat_history_config.knowledge_base_id(ctx),
            )
        typed_stats = cast(
            Callable[[Context], Awaitable[ChatHistoryKBStats]],
            provider_stats,
        )
        return await typed_stats(ctx)


# ── Pure helpers (search fusion + grouping) ───────────────────────────


def _rrf_merge(
    keyword_items: list[MessageSearchResultItem],
    vector_items: list[MessageSearchResultItem],
) -> list[MessageSearchResultItem]:
    """Reciprocal Rank Fusion over keyword and vector result lists.

    Mirrors the upstream ``rrfMerge``: ``k = 60`` (the standard RRF
    damping constant); duplicates across the two lists get their
    ranks summed and their ``match_type`` promoted to ``hybrid``.
    """
    k = 60.0

    @dataclass
    class _Scored:
        item: MessageWithSession
        rrf_score: float = 0.0
        match_type: str = ""

    score_map: dict[str, _Scored] = {}

    for rank, item in enumerate(keyword_items):
        key = item.message.message.id
        entry = score_map.get(key)
        rrf_score = 1.0 / (k + float(rank + 1))
        if entry is None:
            score_map[key] = _Scored(
                item=item.message,
                rrf_score=rrf_score,
                match_type="keyword",
            )
        else:
            entry.rrf_score += rrf_score
            entry.match_type = "hybrid"

    for rank, item in enumerate(vector_items):
        key = item.message.message.id
        entry = score_map.get(key)
        rrf_score = 1.0 / (k + float(rank + 1))
        if entry is None:
            score_map[key] = _Scored(
                item=item.message,
                rrf_score=rrf_score,
                match_type="vector",
            )
        else:
            entry.rrf_score += rrf_score
            entry.match_type = "hybrid"

    merged = sorted(score_map.values(), key=lambda scored: scored.rrf_score, reverse=True)
    return [
        MessageSearchResultItem(
            message=scored.item,
            score=scored.rrf_score,
            match_type=scored.match_type,
        )
        for scored in merged
    ]


def _group_by_request_id(
    items: list[MessageSearchResultItem],
) -> list[MessageSearchGroupItem]:
    """Fold search hits into Q&A pairs grouped by ``request_id``.

    Mirrors the upstream ``groupByRequestID``: messages sharing the
    same ``request_id`` collapse into one group whose ``query_content``
    is the user turn and ``answer_content`` is the assistant turn; a
    missing ``request_id`` is treated as a standalone group keyed by
    the message id. Groups preserve the first-appearance order of the
    source list (which reflects the score ranking produced by the
    fusion step).
    """

    @dataclass
    class _Group:
        request_id: str = ""
        session_id: str = ""
        session_title: str = ""
        query_content: str = ""
        answer_content: str = ""
        score: float = 0.0
        match_type: str = ""
        created_at: datetime = field(default_factory=lambda: _now())
        order: int = 0

    groups: dict[str, _Group] = {}
    next_order: int = 0

    for item in items:
        message = item.message.message
        key = message.request_id or message.id
        entry = groups.get(key)
        if entry is None:
            entry = _Group(
                request_id=message.request_id,
                session_id=message.session_id,
                session_title=item.message.session_title,
                score=item.score,
                match_type=item.match_type,
                created_at=message.created_at or _now(),
                order=next_order,
            )
            groups[key] = entry
            next_order += 1

        if message.role == "user":
            entry.query_content = message.content
        elif message.role == "assistant":
            entry.answer_content = message.content
        if item.score > entry.score:
            entry.score = item.score
        if entry.match_type == "":
            entry.match_type = item.match_type
        elif entry.match_type != item.match_type and item.match_type:
            entry.match_type = "hybrid"
        if message.created_at and message.created_at < entry.created_at:
            entry.created_at = message.created_at

    ordered = sorted(groups.values(), key=lambda row: row.order)
    return [
        MessageSearchGroupItem(
            request_id=row.request_id,
            session_id=row.session_id,
            session_title=row.session_title,
            query_content=row.query_content,
            answer_content=row.answer_content,
            score=row.score,
            match_type=row.match_type,
            created_at=row.created_at,
        )
        for row in ordered
    ]


__all__ = [
    "CHAT_HISTORY_KB_STATS_DISABLED",
    "ChatHistoryConfigProvider",
    "ChatHistoryKBStats",
    "MessageSearchGroupItem",
    "MessageSearchParams",
    "MessageSearchResult",
    "MessageSearchResultItem",
    "MessageService",
    "MessageServiceImpl",
    "MessageVectorSearcher",
    "MessageWithSession",
]
