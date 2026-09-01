"""Unit tests for the message service.

Exercises the request-scoped ``MessageServiceImpl`` orchestration over
mocked persistence and search seams:

- CRUD methods delegate correctly to the ``MessageRepository`` and
  raise ``NotFoundError`` on missing / soft-deleted rows;
- session-existence guards reject writes against unknown sessions;
- search methods run keyword / vector / hybrid paths and emit
  ``MessageSearchResult`` payloads grouped by ``request_id``;
- the KB indexer seam is wired through ``index_message_to_kb`` and the
  cleanup paths.

All external collaborators are tiny in-memory fakes so the tests stay
isolated from the database and the chat-history KB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

import pytest

from src.common.exception import NotFoundError, ValidationError
from src.core.chat.messages.index_to_kb import (
    DefaultMessageIndexer,
    MessageIndexer,
    build_passage,
    strip_think_tags,
)
from src.core.chat.messages.service.message_service import (
    ChatHistoryConfigProvider,
    MessageSearchParams,
    MessageSearchResult,
    MessageSearchResultItem,
    MessageServiceImpl,
    MessageVectorSearcher,
    MessageWithSession,
)
from src.core.chat.messages.types import (
    ROLE_ASSISTANT,
    ROLE_USER,
    MessageSearchMode,
)
from src.core.chat.pipeline.types import Context
from src.db.dao.message_repository import MessageRepository
from src.db.dao.session_repository import SessionRepository
from src.db.models.message import Message

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


# ── Value fakes ────────────────────────────────────────────────────────


@dataclass
class _Ctx:
    """Minimal ``Context`` stand-in carrying the tenant id."""

    tenant_id: int = 7
    user_id: str = "user-1"
    request_id: str = "req-1"


def _message(
    *,
    id: str = "msg-1",
    session_id: str = "sess-1",
    role: str = ROLE_USER,
    content: str = "hello",
    request_id: str = "req-1",
    knowledge_id: str = "",
    created_at: datetime | None = None,
    **overrides: object,
) -> Message:
    """Build a Message row matching the persistence shape."""
    row: dict[str, object] = {
        "id": id,
        "request_id": request_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "knowledge_references": [],
        "agent_steps": None,
        "is_completed": True,
        "is_fallback": False,
        "agent_duration_ms": 0,
        "rendered_content": "",
        "channel": "",
        "agent_id": "",
        "agent_tenant_id": 0,
        "model_id": "",
        "execution_context": {},
        "knowledge_id": knowledge_id,
        "mentioned_items": [],
        "images": [],
        "attachments": [],
        "created_at": created_at or _NOW,
        "updated_at": _NOW,
        "deleted_at": None,
    }
    row.update(overrides)
    return Message(**row)  # type: ignore[arg-type]


# ── In-memory fakes ────────────────────────────────────────────────────


class _FakeMessageRepo(MessageRepository):
    """An in-memory stand-in that records calls for assertions."""

    def __init__(self) -> None:
        self.created: list[Message] = []
        self.rows: dict[tuple[str, str], Message] = {}
        self.knowledge_ids: dict[str, list[str]] = {}
        self.soft_deleted: list[tuple[str, str]] = []
        self.session_soft_deleted: list[str] = []
        self.updated_knowledge: list[tuple[str, str]] = []
        self.updates: list[dict[str, object]] = []
        self.images_updates: list[tuple[str, str, object]] = []
        self.rendered_updates: list[tuple[str, str, str]] = []
        self.keyword_calls: list[dict[str, object]] = []
        self.by_knowledge_calls: list[list[str]] = []
        self.by_request_calls: list[list[str]] = []
        self.list_recent: list[tuple[str, int]] = []
        self.list_paginated: list[tuple[str, int, int]] = []
        self.list_before_time: list[tuple[str, datetime, int]] = []
        self.first_user: list[str] = []

    def seed(self, row: Message) -> None:
        self.rows[(row.session_id, row.id)] = row

    async def create(self, row: Message) -> Message:
        self.created.append(row)
        self.rows[(row.session_id, row.id)] = row
        return row

    async def get_by_id_and_session(
        self,
        *,
        session_id: str,
        message_id: str,
    ) -> Message | None:
        return self.rows.get((session_id, message_id))

    async def list_by_session(
        self,
        session_id: str,
        *,
        page: int = 1,
        page_size: int = 20,
    ) -> list[Message]:
        self.list_paginated.append((session_id, page, page_size))
        return [r for (s, _), r in self.rows.items() if s == session_id]

    async def list_recent_by_session(
        self,
        session_id: str,
        *,
        limit: int,
    ) -> list[Message]:
        self.list_recent.append((session_id, limit))
        return [r for (s, _), r in self.rows.items() if s == session_id]

    async def list_by_session_before_time(
        self,
        session_id: str,
        *,
        before_time: datetime,
        limit: int,
    ) -> list[Message]:
        self.list_before_time.append((session_id, before_time, limit))
        return [r for (s, _), r in self.rows.items() if s == session_id]

    async def update(
        self,
        *,
        session_id: str,
        message_id: str,
        column_to_update: object,
    ) -> Message | None:
        self.updates.append(
            {"session_id": session_id, "message_id": message_id, "columns": column_to_update}
        )
        row = self.rows.get((session_id, message_id))
        if row is None:
            return None
        # ``Message`` is a frozen Pydantic model; ``model_copy`` returns
        # a new row carrying the requested column overrides.
        updated = row.model_copy(update=dict(column_to_update))  # type: ignore[union-attr]
        self.rows[(session_id, message_id)] = updated
        return updated

    async def update_images(
        self,
        *,
        session_id: str,
        message_id: str,
        images: object,
    ) -> Message | None:
        self.images_updates.append((session_id, message_id, images))
        row = self.rows.get((session_id, message_id))
        if row is None:
            return None
        updated = row.model_copy(update={"images": images})
        self.rows[(session_id, message_id)] = updated
        return updated

    async def update_rendered_content(
        self,
        *,
        session_id: str,
        message_id: str,
        rendered_content: str,
    ) -> Message | None:
        self.rendered_updates.append((session_id, message_id, rendered_content))
        row = self.rows.get((session_id, message_id))
        if row is None:
            return None
        updated = row.model_copy(update={"rendered_content": rendered_content})
        self.rows[(session_id, message_id)] = updated
        return updated

    async def soft_delete(
        self,
        *,
        session_id: str,
        message_id: str,
        now: datetime,
    ) -> bool:
        self.soft_deleted.append((session_id, message_id))
        row = self.rows.pop((session_id, message_id), None)
        return row is not None

    async def soft_delete_by_session(
        self,
        session_id: str,
        now: datetime,
    ) -> int:
        self.session_soft_deleted.append(session_id)
        count = 0
        for key in list(self.rows):
            if key[0] == session_id:
                self.rows.pop(key)
                count += 1
        return count

    async def update_knowledge_id(
        self,
        *,
        message_id: str,
        knowledge_id: str,
        now: datetime,
    ) -> Message | None:
        self.updated_knowledge.append((message_id, knowledge_id))
        for key, row in list(self.rows.items()):
            if row.id == message_id:
                updated = row.model_copy(update={"knowledge_id": knowledge_id})
                self.rows[key] = updated
                return updated
        return None

    async def search_by_keyword(
        self,
        *,
        keyword: str,
        session_ids: list[str],
        limit: int = 20,
    ) -> list[Message]:
        self.keyword_calls.append(
            {
                "keyword": keyword,
                "session_ids": list(session_ids),
                "limit": limit,
            }
        )
        if not session_ids:
            return []
        rows = [r for r in self.rows.values() if keyword in r.content]
        ids = set(session_ids)
        rows = [r for r in rows if r.session_id in ids]
        return rows[:limit]

    async def list_by_knowledge_ids(
        self,
        knowledge_ids: list[str],
    ) -> list[Message]:
        self.by_knowledge_calls.append(list(knowledge_ids))
        ids = set(knowledge_ids)
        return [r for r in self.rows.values() if r.knowledge_id in ids]

    async def list_by_request_ids(
        self,
        request_ids: list[str],
        *,
        session_ids: list[str],
    ) -> list[Message]:
        self.by_request_calls.append(list(request_ids))
        ids = set(request_ids)
        session_bound = set(session_ids)
        return [
            r for r in self.rows.values() if r.request_id in ids and r.session_id in session_bound
        ]

    async def list_knowledge_ids_by_session(
        self,
        session_id: str,
    ) -> list[str]:
        return list(self.knowledge_ids.get(session_id, []))

    async def get_first_user_message(
        self,
        session_id: str,
    ) -> Message | None:
        self.first_user.append(session_id)
        for row in self.rows.values():
            if row.session_id == session_id and row.role == ROLE_USER:
                return row
        return None


class _FakeSessionRepo(SessionRepository):
    """Records session existence lookups."""

    def __init__(self) -> None:
        self.existing: set[str] = set()
        self.calls: list[tuple[int, str]] = []

    def allow(self, session_id: str) -> None:
        self.existing.add(session_id)

    async def get_by_id(
        self,
        *,
        tenant_id: int,
        id: str,
    ) -> object | None:
        self.calls.append((tenant_id, id))
        return object() if id in self.existing else None

    async def list_ids_by_tenant(self, *, tenant_id: int) -> list[str]:
        """Enumerate the allowed ids — the tenant-scope seam for search."""
        return sorted(self.existing)


class _FakeConfig(ChatHistoryConfigProvider):
    """Captures the config queries the service performs."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        kb_id: str = "kb-history",
        embedding_model_id: str = "emb-1",
        top_k: int = 10,
        threshold: float = 0.5,
        stats: object | None = None,
    ) -> None:
        self._enabled = enabled
        self._kb_id = kb_id
        self._embedding_model_id = embedding_model_id
        self._top_k = top_k
        self._threshold = threshold
        self._stats = stats
        self.calls: list[str] = []

    def is_enabled(self, ctx: Context) -> bool:
        self.calls.append("is_enabled")
        return self._enabled

    def knowledge_base_id(self, ctx: Context) -> str:
        self.calls.append("knowledge_base_id")
        return self._kb_id

    def embedding_model_id(self, ctx: Context) -> str:
        self.calls.append("embedding_model_id")
        return self._embedding_model_id

    def effective_embedding_top_k(self, ctx: Context) -> int:
        self.calls.append("top_k")
        return self._top_k

    def effective_vector_threshold(self, ctx: Context) -> float:
        self.calls.append("threshold")
        return self._threshold

    @property
    def stats(self) -> object:
        return self._stats


class _FakeVectorSearcher(MessageVectorSearcher):
    """Records vector search invocations and returns canned results."""

    def __init__(self, results: list[MessageSearchResultItem] | None = None) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    async def search_by_vector(
        self,
        *,
        ctx: Context,
        query: str,
        knowledge_base_id: str,
        embedding_top_k: int,
        vector_threshold: float,
        session_ids: tuple[str, ...],
    ) -> list[MessageSearchResultItem]:
        self.calls.append(
            {
                "query": query,
                "knowledge_base_id": knowledge_base_id,
                "embedding_top_k": embedding_top_k,
                "vector_threshold": vector_threshold,
                "session_ids": list(session_ids),
            }
        )
        return list(self.results or [])


@dataclass
class _IndexerCall:
    """Record of one indexer method call."""

    method: str
    kwargs: dict[str, object] = field(default_factory=dict)


class _FakeIndexer(MessageIndexer):
    """Stand-in satisfying the ``MessageIndexer`` protocol."""

    def __init__(self) -> None:
        self.calls: list[_IndexerCall] = []
        self.delete_result: object | None = None
        self.session_delete_result: object | None = None

    async def index_message(
        self,
        ctx: Context,
        *,
        user_query: str,
        assistant_answer: str,
        message_id: str,
        session_id: str,
    ) -> None:
        self.calls.append(
            _IndexerCall(
                "index_message",
                {
                    "user_query": user_query,
                    "assistant_answer": assistant_answer,
                    "message_id": message_id,
                    "session_id": session_id,
                },
            )
        )

    async def delete_message_knowledge(self, *, knowledge_id: str) -> None:
        self.calls.append(_IndexerCall("delete_message_knowledge", {"knowledge_id": knowledge_id}))
        return self.delete_result  # type: ignore[return-value]

    async def delete_session_knowledge(
        self,
        *,
        knowledge_ids: tuple[str, ...],
    ) -> None:
        self.calls.append(
            _IndexerCall(
                "delete_session_knowledge",
                {"knowledge_ids": tuple(knowledge_ids)},
            )
        )
        return self.session_delete_result  # type: ignore[return-value]


def _service(
    *,
    message_repo: _FakeMessageRepo | None = None,
    session_repo: _FakeSessionRepo | None = None,
    vector_searcher: _FakeVectorSearcher | None = None,
    config: _FakeConfig | None = None,
    indexer: _FakeIndexer | None = None,
    skip_session_repo: bool = False,
) -> tuple[MessageServiceImpl, _FakeMessageRepo, _FakeSessionRepo | None]:
    repo = message_repo or _FakeMessageRepo()
    sessions = None if skip_session_repo else (session_repo or _FakeSessionRepo())
    return (
        MessageServiceImpl(
            message_repo=repo,
            session_repo=sessions,
            vector_searcher=vector_searcher,
            chat_history_config=config,
            indexer=indexer,
        ),
        repo,
        sessions,
    )


# ── CRUD ───────────────────────────────────────────────────────────────


async def test_create_message_persists_and_returns_row() -> None:
    # Arrange
    service, repo, sessions = _service()
    assert sessions is not None
    sessions.allow("sess-1")
    message = _message(id="msg-1", session_id="sess-1")

    # Act
    created = await service.create_message(_Ctx(), message)

    # Assert
    assert created.id == "msg-1"
    assert repo.created == [message]


async def test_create_message_rejects_unknown_session() -> None:
    # Arrange
    service, _, sessions = _service()
    assert sessions is not None
    message = _message(id="msg-1", session_id="ghost")

    # Act / Assert
    with pytest.raises(NotFoundError) as exc:
        await service.create_message(_Ctx(), message)
    assert exc.value.code == "message.session_not_found"


async def test_get_message_returns_row() -> None:
    # Arrange
    service, repo, sessions = _service()
    assert sessions is not None
    sessions.allow("sess-1")
    repo.seed(_message(id="msg-1", session_id="sess-1"))

    # Act
    found = await service.get_message(_Ctx(), "sess-1", "msg-1")

    # Assert
    assert found.id == "msg-1"


async def test_get_message_raises_when_missing() -> None:
    # Arrange
    service, _, sessions = _service()
    assert sessions is not None
    sessions.allow("sess-1")

    # Act / Assert
    with pytest.raises(NotFoundError) as exc:
        await service.get_message(_Ctx(), "sess-1", "ghost")
    assert exc.value.code == "message.not_found"


async def test_list_messages_by_session_passes_pagination() -> None:
    # Arrange
    service, repo, sessions = _service()
    assert sessions is not None
    sessions.allow("sess-1")
    repo.seed(_message(id="msg-1", session_id="sess-1"))

    # Act
    rows = await service.list_messages_by_session(_Ctx(), "sess-1", page=1, page_size=5)

    # Assert
    assert [r.id for r in rows] == ["msg-1"]
    assert repo.list_paginated == [("sess-1", 1, 5)]


async def test_list_messages_by_session_defaults_page_size() -> None:
    # Arrange
    service, repo, sessions = _service()
    assert sessions is not None
    sessions.allow("sess-1")

    # Act
    await service.list_messages_by_session(_Ctx(), "sess-1", page=1, page_size=0)

    # Assert
    assert repo.list_paginated == [("sess-1", 1, 20)]


async def test_get_recent_messages_by_session_delegates_to_repo() -> None:
    # Arrange
    service, repo, sessions = _service()
    assert sessions is not None
    sessions.allow("sess-1")
    repo.seed(_message(id="msg-1", session_id="sess-1"))

    # Act
    rows = await service.get_recent_messages_by_session(_Ctx(), "sess-1", limit=10)

    # Assert
    assert [r.id for r in rows] == ["msg-1"]
    assert repo.list_recent == [("sess-1", 10)]


async def test_list_messages_by_session_before_time_delegates_to_repo() -> None:
    # Arrange
    service, repo, sessions = _service()
    assert sessions is not None
    sessions.allow("sess-1")

    # Act
    await service.list_messages_by_session_before_time(_Ctx(), "sess-1", before_time=_NOW, limit=4)

    # Assert
    assert repo.list_before_time == [("sess-1", _NOW, 4)]


async def test_update_message_writes_columns_and_returns_row() -> None:
    # Arrange
    service, repo, sessions = _service()
    assert sessions is not None
    sessions.allow("sess-1")
    existing = _message(id="msg-1", session_id="sess-1", content="before")
    repo.seed(existing)
    updated_input = _message(id="msg-1", session_id="sess-1", content="after", is_completed=True)

    # Act
    returned = await service.update_message(_Ctx(), updated_input)

    # Assert
    assert returned.content == "after"
    assert repo.updates[0]["columns"]["content"] == "after"


async def test_update_message_raises_when_missing() -> None:
    # Arrange
    service, _, sessions = _service()
    assert sessions is not None
    sessions.allow("sess-1")
    missing = _message(id="ghost", session_id="sess-1")

    # Act / Assert
    with pytest.raises(NotFoundError):
        await service.update_message(_Ctx(), missing)


async def test_update_message_images_delegates_to_repo() -> None:
    # Arrange
    service, repo, sessions = _service()
    assert sessions is not None
    sessions.allow("sess-1")
    repo.seed(_message(id="msg-1", session_id="sess-1"))

    # Act
    images = [{"url": "https://x", "caption": "img"}]
    updated = await service.update_message_images(_Ctx(), "sess-1", "msg-1", images)

    # Assert
    assert updated.images == images
    assert repo.images_updates == [("sess-1", "msg-1", images)]


async def test_update_message_rendered_content_delegates_to_repo() -> None:
    # Arrange
    service, repo, sessions = _service()
    assert sessions is not None
    sessions.allow("sess-1")
    repo.seed(_message(id="msg-1", session_id="sess-1"))

    # Act
    updated = await service.update_message_rendered_content(_Ctx(), "sess-1", "msg-1", "rendered")

    # Assert
    assert updated.rendered_content == "rendered"
    assert repo.rendered_updates == [("sess-1", "msg-1", "rendered")]


async def test_delete_message_soft_deletes_and_cleans_up_kb_link() -> None:
    # Arrange
    service, repo, sessions = _service(
        indexer=_FakeIndexer(),
    )
    assert sessions is not None
    sessions.allow("sess-1")
    repo.seed(_message(id="msg-1", session_id="sess-1", knowledge_id="k-1"))
    indexer = service._indexer  # type: ignore[attr-defined]

    # Act
    deleted = await service.delete_message(_Ctx(), "sess-1", "msg-1")

    # Assert
    assert deleted is True
    assert repo.soft_deleted == [("sess-1", "msg-1")]
    assert indexer.calls[0].method == "delete_message_knowledge"
    assert indexer.calls[0].kwargs == {"knowledge_id": "k-1"}


async def test_delete_message_without_knowledge_link_skips_indexer() -> None:
    # Arrange
    service, repo, sessions = _service(indexer=_FakeIndexer())
    assert sessions is not None
    sessions.allow("sess-1")
    repo.seed(_message(id="msg-1", session_id="sess-1"))
    indexer = service._indexer  # type: ignore[attr-defined]

    # Act
    await service.delete_message(_Ctx(), "sess-1", "msg-1")

    # Assert
    assert indexer.calls == []


async def test_clear_session_messages_soft_deletes_and_purges_kb() -> None:
    # Arrange
    service, repo, sessions = _service(indexer=_FakeIndexer())
    assert sessions is not None
    sessions.allow("sess-1")
    repo.knowledge_ids["sess-1"] = ["k-1", "k-2"]
    indexer = service._indexer  # type: ignore[attr-defined]

    # Act
    deleted = await service.clear_session_messages(_Ctx(), "sess-1")

    # Assert
    assert deleted == 0  # no rows in repo, but soft_delete still ran
    assert repo.session_soft_deleted == ["sess-1"]
    assert indexer.calls[0].method == "delete_session_knowledge"
    assert indexer.calls[0].kwargs == {"knowledge_ids": ("k-1", "k-2")}


# ── Search ─────────────────────────────────────────────────────────────


async def test_search_messages_rejects_empty_query() -> None:
    # Arrange
    service, _, _ = _service()

    # Act / Assert
    with pytest.raises(ValidationError) as exc:
        await service.search_messages(
            _Ctx(),
            MessageSearchParams(query="   "),
        )
    assert exc.value.code == "message.search_query_required"


async def test_search_messages_rejects_session_ids_outside_caller_scope() -> None:
    # Arrange — the caller passes a session id the tenant does not own.
    repo = _FakeMessageRepo()
    repo.seed(_message(id="msg-1", session_id="foreign-sess", content="secret"))
    service, _, sessions = _service(message_repo=repo)
    assert sessions is not None

    # Act / Assert — scope resolution rejects the foreign id before search.
    with pytest.raises(NotFoundError) as exc:
        await service.search_messages(
            _Ctx(),
            MessageSearchParams(
                query="secret",
                mode=MessageSearchMode.KEYWORD,
                session_ids=("foreign-sess",),
            ),
        )
    assert exc.value.code == "message.session_not_found"
    assert repo.keyword_calls == []


async def test_search_messages_without_session_filter_scans_only_tenant_sessions() -> None:
    # Arrange — a message lives in a session the caller cannot see.
    repo = _FakeMessageRepo()
    repo.seed(_message(id="msg-1", session_id="foreign-sess", content="hello"))
    repo.seed(_message(id="msg-2", session_id="sess-1", content="hello"))
    service, _, sessions = _service(message_repo=repo)
    assert sessions is not None
    sessions.allow("sess-1")

    # Act
    result = await service.search_messages(
        _Ctx(),
        MessageSearchParams(query="hello", mode=MessageSearchMode.KEYWORD),
    )

    # Assert — only the in-scope hit surfaces.
    assert repo.keyword_calls[0]["session_ids"] == ["sess-1"]
    assert result.total == 1
    assert result.items[0].query_content == "hello"


async def test_search_messages_keyword_mode_runs_keyword_only() -> None:
    # Arrange
    repo = _FakeMessageRepo()
    repo.seed(_message(id="msg-1", session_id="sess-1", content="hello world"))
    service, _, sessions = _service(message_repo=repo)
    assert sessions is not None
    sessions.allow("sess-1")

    # Act
    result = await service.search_messages(
        _Ctx(),
        MessageSearchParams(query="hello", mode=MessageSearchMode.KEYWORD),
    )

    # Assert
    assert isinstance(result, MessageSearchResult)
    assert result.total == 1
    assert result.items[0].query_content == "hello world"
    assert repo.keyword_calls[0]["keyword"] == "hello"


async def test_search_messages_vector_mode_runs_vector_seam() -> None:
    # Arrange
    repo = _FakeMessageRepo()
    repo.seed(_message(id="msg-1", session_id="sess-1", content="vector hit"))
    vector_hit = MessageSearchResultItem(
        message=MessageWithSession(message=_message(id="msg-1", session_id="sess-1")),
        score=0.9,
        match_type="vector",
    )
    searcher = _FakeVectorSearcher(results=[vector_hit])
    config = _FakeConfig(enabled=True, kb_id="kb-history")
    service, _, sessions = _service(message_repo=repo, vector_searcher=searcher, config=config)
    assert sessions is not None
    sessions.allow("sess-1")

    # Act
    result = await service.search_messages(
        _Ctx(),
        MessageSearchParams(
            query="hello",
            mode=MessageSearchMode.VECTOR,
            session_ids=("sess-1",),
        ),
    )

    # Assert
    assert result.total == 1
    assert searcher.calls[0]["knowledge_base_id"] == "kb-history"
    assert searcher.calls[0]["session_ids"] == ["sess-1"]


async def test_search_messages_hybrid_mode_fuses_keyword_and_vector() -> None:
    # Arrange
    repo = _FakeMessageRepo()
    # Two messages sharing the same request_id so they collapse into
    # one Q&A group after the grouping step.
    repo.seed(
        _message(
            id="msg-q",
            session_id="sess-1",
            role=ROLE_USER,
            request_id="req-1",
            content="what is X?",
        )
    )
    repo.seed(
        _message(
            id="msg-a",
            session_id="sess-1",
            role=ROLE_ASSISTANT,
            request_id="req-1",
            content="X is ...",
        )
    )
    vector_hit = MessageSearchResultItem(
        message=MessageWithSession(
            message=_message(
                id="msg-q",
                session_id="sess-1",
                role=ROLE_USER,
                request_id="req-1",
                content="what is X?",
            )
        ),
        score=0.9,
        match_type="vector",
    )
    searcher = _FakeVectorSearcher(results=[vector_hit])
    config = _FakeConfig(enabled=True, kb_id="kb-history")
    service, _, sessions = _service(message_repo=repo, vector_searcher=searcher, config=config)
    assert sessions is not None
    sessions.allow("sess-1")

    # Act
    result = await service.search_messages(
        _Ctx(),
        MessageSearchParams(query="X", mode=MessageSearchMode.HYBRID),
    )

    # Assert
    assert result.total == 1
    group = result.items[0]
    assert group.request_id == "req-1"
    assert group.query_content == "what is X?"
    assert group.answer_content == "X is ..."
    assert group.match_type == "hybrid"


async def test_search_messages_falls_back_to_keyword_when_vector_seam_absent() -> None:
    # Arrange
    repo = _FakeMessageRepo()
    repo.seed(_message(id="msg-1", session_id="sess-1", content="keyword only"))
    service, _, sessions = _service(message_repo=repo)
    assert sessions is not None
    sessions.allow("sess-1")

    # Act
    result = await service.search_messages(
        _Ctx(),
        MessageSearchParams(query="keyword", mode=MessageSearchMode.HYBRID),
    )

    # Assert
    assert result.total == 1
    assert result.items[0].match_type == "keyword"


async def test_search_messages_groups_partner_message_for_qa_pair() -> None:
    # Arrange
    repo = _FakeMessageRepo()
    # Only the user message matches the keyword path; the assistant
    # answer has no keyword match but must be pulled in by the
    # partner-fetch step (the vector seam returns the partner via
    # ``by_request_id``).
    repo.seed(
        _message(
            id="msg-q",
            session_id="sess-1",
            role=ROLE_USER,
            request_id="req-1",
            content="ask",
        )
    )
    repo.seed(
        _message(
            id="msg-a",
            session_id="sess-1",
            role=ROLE_ASSISTANT,
            request_id="req-1",
            content="reply",
        )
    )
    user_hit = MessageSearchResultItem(
        message=MessageWithSession(
            message=_message(
                id="msg-q",
                session_id="sess-1",
                role=ROLE_USER,
                request_id="req-1",
                content="ask",
            )
        ),
        score=0.5,
        match_type="keyword",
    )
    searcher = _FakeVectorSearcher(results=[user_hit])
    config = _FakeConfig(enabled=True, kb_id="kb-history")
    service, _, sessions = _service(message_repo=repo, vector_searcher=searcher, config=config)
    assert sessions is not None
    sessions.allow("sess-1")

    # Act
    result = await service.search_messages(
        _Ctx(),
        MessageSearchParams(query="ask", mode=MessageSearchMode.HYBRID),
    )

    # Assert
    assert result.total == 1
    group = result.items[0]
    assert group.query_content == "ask"
    assert group.answer_content == "reply"


# ── KB indexing ────────────────────────────────────────────────────────


async def test_index_message_to_kb_delegates_to_indexer() -> None:
    # Arrange
    indexer = _FakeIndexer()
    service, _, _ = _service(indexer=indexer)

    # Act
    await service.index_message_to_kb(
        _Ctx(),
        user_query="what?",
        assistant_answer="because",
        message_id="msg-1",
        session_id="sess-1",
    )

    # Assert
    assert indexer.calls[0].method == "index_message"
    assert indexer.calls[0].kwargs == {
        "user_query": "what?",
        "assistant_answer": "because",
        "message_id": "msg-1",
        "session_id": "sess-1",
    }


async def test_index_message_to_kb_is_noop_when_indexer_missing() -> None:
    # Arrange
    service, _, _ = _service(indexer=None)

    # Act
    await service.index_message_to_kb(
        _Ctx(),
        user_query="q",
        assistant_answer="a",
        message_id="msg-1",
        session_id="sess-1",
    )

    # Assert — no exception is raised; the call is silently skipped.


async def test_delete_message_knowledge_noop_when_id_empty() -> None:
    # Arrange
    indexer = _FakeIndexer()
    service, _, _ = _service(indexer=indexer)

    # Act
    await service.delete_message_knowledge(_Ctx(), "")

    # Assert
    assert indexer.calls == []


async def test_delete_session_knowledge_passes_indexer_ids_from_repo() -> None:
    # Arrange
    repo = _FakeMessageRepo()
    repo.knowledge_ids["sess-1"] = ["k-1", "k-2"]
    indexer = _FakeIndexer()
    service, _, _ = _service(message_repo=repo, indexer=indexer)

    # Act
    await service.delete_session_knowledge(_Ctx(), "sess-1")

    # Assert
    assert indexer.calls[0].method == "delete_session_knowledge"
    assert indexer.calls[0].kwargs == {"knowledge_ids": ("k-1", "k-2")}


async def test_get_chat_history_kb_stats_returns_disabled_when_config_missing() -> None:
    # Arrange
    service, _, _ = _service()

    # Act
    stats = await service.get_chat_history_kb_stats(_Ctx())

    # Assert
    assert stats.enabled is False
    assert stats.indexed_message_count == 0


async def test_get_chat_history_kb_stats_returns_enabled_stub() -> None:
    # Arrange
    config = _FakeConfig(enabled=True, kb_id="kb-history", embedding_model_id="emb-1")
    service, _, _ = _service(config=config)

    # Act
    stats = await service.get_chat_history_kb_stats(_Ctx())

    # Assert
    assert stats.enabled is True
    assert stats.knowledge_base_id == "kb-history"
    assert stats.embedding_model_id == "emb-1"


# ── index_to_kb helpers ────────────────────────────────────────────────


def test_strip_think_tags_removes_reasoning_blocks() -> None:
    # Arrange / Act
    cleaned = strip_think_tags("hello<think>secret</think> world")

    # Assert
    assert cleaned == "hello world"


def test_build_passage_anchors_session_id() -> None:
    # Arrange / Act
    passage = build_passage(session_id="s1", query="q?", answer="a.")

    # Assert
    assert passage == "[Session: s1]\nQ: q?\nA: a."


class _RecordingCreator:
    """Stand-in for the passage-creator seam."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def create_passage(
        self,
        *,
        ctx: Context,
        knowledge_base_id: str,
        passages: tuple[str, ...],
    ) -> str:
        self.calls.append(
            {
                "knowledge_base_id": knowledge_base_id,
                "passages": tuple(passages),
            }
        )
        return "knowledge-1"


@runtime_checkable
class _ProtocolCheck(Protocol):
    """Compile-time check that the indexer satisfies the protocol."""

    def __call__(self) -> object: ...


def test_default_indexer_satisfies_protocol() -> None:
    # Arrange
    repo = _FakeMessageRepo()
    config = _FakeConfig(enabled=True, kb_id="kb-history")
    indexer: MessageIndexer = DefaultMessageIndexer(
        message_repo=repo,
        passage_creator=_RecordingCreator(),
        config=config,
    )

    # Assert
    assert isinstance(indexer, MessageIndexer)


async def test_default_indexer_writes_passage_and_links_message() -> None:
    # Arrange
    repo = _FakeMessageRepo()
    repo.seed(_message(id="msg-1", session_id="sess-1"))
    creator = _RecordingCreator()
    config = _FakeConfig(enabled=True, kb_id="kb-history")
    indexer = DefaultMessageIndexer(
        message_repo=repo,
        passage_creator=creator,
        config=config,
    )

    # Act
    await indexer.index_message(
        _Ctx(),
        user_query="what?",
        assistant_answer="because<think>reason</think>",
        message_id="msg-1",
        session_id="sess-1",
    )

    # Assert
    assert creator.calls[0]["knowledge_base_id"] == "kb-history"
    assert "<think>" not in creator.calls[0]["passages"][0]  # type: ignore[index]
    assert repo.updated_knowledge == [("msg-1", "knowledge-1")]


async def test_default_indexer_skips_when_disabled() -> None:
    # Arrange
    repo = _FakeMessageRepo()
    repo.seed(_message(id="msg-1", session_id="sess-1"))
    creator = _RecordingCreator()
    config = _FakeConfig(enabled=False)
    indexer = DefaultMessageIndexer(
        message_repo=repo,
        passage_creator=creator,
        config=config,
    )

    # Act
    await indexer.index_message(
        _Ctx(),
        user_query="q",
        assistant_answer="a",
        message_id="msg-1",
        session_id="sess-1",
    )

    # Assert
    assert creator.calls == []
    assert repo.updated_knowledge == []


async def test_default_indexer_skips_when_qa_pair_empty() -> None:
    # Arrange
    repo = _FakeMessageRepo()
    repo.seed(_message(id="msg-1", session_id="sess-1"))
    creator = _RecordingCreator()
    config = _FakeConfig(enabled=True, kb_id="kb-history")
    indexer = DefaultMessageIndexer(
        message_repo=repo,
        passage_creator=creator,
        config=config,
    )

    # Act
    await indexer.index_message(
        _Ctx(),
        user_query="   ",
        assistant_answer="<think>...</think>",
        message_id="msg-1",
        session_id="sess-1",
    )

    # Assert
    assert creator.calls == []


__all__ = ["_FakeMessageRepo", "_FakeSessionRepo", "_IndexerCall", "_service"]
