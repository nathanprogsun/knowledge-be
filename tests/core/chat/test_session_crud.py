"""Unit tests for the chat-session service.

Covers CRUD + pin toggle + title generation with tiny in-memory
fakes for the repositories, the chat client, and the title
generator. The real repos are not exercised here; the
``SessionRepository`` and ``MessageRepository`` integration tests
live under ``tests/db/``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from src.common.exception import NotFoundError, ValidationError
from src.common.pagination import Pagination
from src.core.chat.sessions.factory import (
    build_session_service,
    build_session_service_with_title,
)
from src.core.chat.sessions.service.session_service import (
    ChatFactoryLike,
    SessionListQuery,
    SessionMessageReader,
    SessionService,
)
from src.core.chat.sessions.title_gen import TitleGenerator, clean_response
from src.db.models.message import Message
from src.db.models.session import Session

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


# ── Fakes ────────────────────────────────────────────────────────────


class _FakeSessionRepo:
    """In-memory stand-in for :class:`SessionRepository`.

    Implements the subset of methods the service uses. The
    ``store`` attribute is a ``dict[str, Session]`` callers can
    inspect after each operation.
    """

    def __init__(self) -> None:
        self.store: dict[str, Session] = {}

    async def create(self, row: Session) -> Session:
        # Mirror the real repo: stamp updated_at only if missing.
        self.store[row.id] = row
        return row

    async def get_by_id(self, *, tenant_id: int, id: str) -> Session | None:
        row = self.store.get(id)
        if row is None or row.tenant_id != tenant_id or row.deleted_at is not None:
            return None
        return row

    async def get_by_id_for_user(self, *, tenant_id: int, user_id: str, id: str) -> Session | None:
        row = self.store.get(id)
        if row is None or row.tenant_id != tenant_id or row.deleted_at is not None:
            return None
        if row.user_id and row.user_id != user_id:
            return None
        return row

    async def list_by_tenant(self, *, tenant_id: int, user_id: str = "") -> list[Session]:
        return [
            row
            for row in sorted(
                self.store.values(),
                key=lambda r: (r.updated_at, r.id),
                reverse=True,
            )
            if row.tenant_id == tenant_id
            and row.deleted_at is None
            and (not user_id or not row.user_id or row.user_id == user_id)
        ]

    async def list_paged(
        self,
        *,
        tenant_id: int,
        user_id: str = "",
        page: int = 1,
        page_size: int = 20,
        keyword: str = "",
    ) -> tuple[list[Session], int]:
        visible = [
            row
            for row in self.store.values()
            if row.tenant_id == tenant_id
            and row.deleted_at is None
            and (not user_id or not row.user_id or row.user_id == user_id)
            and (not keyword or (row.title or "").lower().find(keyword.lower()) >= 0)
        ]
        visible.sort(
            key=lambda r: (
                not r.is_pinned,
                r.pinned_at or _NOW,
                r.updated_at,
                r.id,
            ),
            reverse=True,
        )
        offset = (page - 1) * page_size
        return visible[offset : offset + page_size], len(visible)

    async def update(self, row: Session, *, user_id: str = "") -> Session:
        existing = self.store.get(row.id)
        if existing is None or existing.deleted_at is not None:
            raise NotFoundError(
                code="session.not_found",
                message=f"session {row.id} not found",
            )
        if user_id and existing.user_id and existing.user_id != user_id:
            raise NotFoundError(
                code="session.not_found",
                message=f"session {row.id} not found",
            )
        self.store[row.id] = row
        return row

    async def soft_delete(
        self,
        *,
        tenant_id: int,
        id: str,
        now: datetime,
        user_id: str = "",
    ) -> bool:
        row = self.store.get(id)
        if row is None or row.deleted_at is not None:
            return False
        if row.tenant_id != tenant_id:
            return False
        if user_id and row.user_id and row.user_id != user_id:
            return False
        self.store[id] = Session(
            id=row.id,
            tenant_id=row.tenant_id,
            title=row.title,
            description=row.description,
            user_id=row.user_id,
            is_pinned=row.is_pinned,
            pinned_at=row.pinned_at,
            created_at=row.created_at,
            updated_at=now,
            deleted_at=now,
        )
        return True

    async def set_pinned(
        self,
        *,
        tenant_id: int,
        id: str,
        pinned: bool,
        now: datetime,
        user_id: str = "",
    ) -> bool:
        row = self.store.get(id)
        if row is None or row.deleted_at is not None:
            return False
        if row.tenant_id != tenant_id:
            return False
        if user_id and row.user_id and row.user_id != user_id:
            return False
        self.store[id] = Session(
            id=row.id,
            tenant_id=row.tenant_id,
            title=row.title,
            description=row.description,
            user_id=row.user_id,
            is_pinned=pinned,
            pinned_at=now if pinned else None,
            created_at=row.created_at,
            updated_at=now,
            deleted_at=row.deleted_at,
        )
        return True


class _FakeMessageRepo:
    """Records the first user message by session id."""

    def __init__(self, messages: dict[str, Message] | None = None) -> None:
        self._messages = messages or {}
        self.calls: list[str] = []

    async def get_first_user_message(self, session_id: str) -> Message | None:
        self.calls.append(session_id)
        return self._messages.get(session_id)


class _FakeChat:
    """Captures the chat call and returns a canned response."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[list[Any]] = []

    async def chat(self, messages: list[Any], options: Any) -> Any:
        from src.ai.llm.types import ChatResponse

        self.calls.append((messages, options))
        return ChatResponse(content=self._content)

    def chat_stream(self, messages: list[Any], options: Any):  # pragma: no cover
        raise NotImplementedError

    def get_model_name(self) -> str:  # pragma: no cover
        return "fake-model"

    def get_model_id(self) -> str:  # pragma: no cover
        return "fake-model-id"


class _FakeChatFactory(ChatFactoryLike):
    """Returns the supplied chat client on every resolve."""

    def __init__(self, chat: _FakeChat) -> None:
        self._chat = chat
        self.calls: list[tuple[int, str]] = []

    async def resolve_chat(self, *, tenant_id: int, model_id: str = "") -> tuple[_FakeChat, str]:
        self.calls.append((tenant_id, model_id))
        resolved = model_id or "resolved-model"
        return self._chat, resolved


# ── Helpers ──────────────────────────────────────────────────────────


def _row(
    *,
    id: str | None = None,
    tenant_id: int = 1,
    user_id: str = "user-1",
    title: str | None = None,
    description: str | None = None,
    is_pinned: bool = False,
    pinned_at: datetime | None = None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> Session:
    now = created_at or _NOW
    return Session(
        id=id or f"sess-{uuid.uuid4().hex[:8]}",
        tenant_id=tenant_id,
        title=title,
        description=description,
        user_id=user_id,
        is_pinned=is_pinned,
        pinned_at=pinned_at,
        created_at=now,
        updated_at=updated_at or now,
    )


def _service(
    *,
    tenant_id: int = 1,
    user_id: str = "user-1",
    session_repo: _FakeSessionRepo | None = None,
    message_repo: SessionMessageReader | None = None,
) -> SessionService:
    return SessionService(
        tenant_id=tenant_id,
        user_id=user_id,
        session_repo=session_repo or _FakeSessionRepo(),  # type: ignore[arg-type]
        message_repo=message_repo,
    )


# ── Constructor guards ───────────────────────────────────────────────


def test_constructor_rejects_invalid_tenant_id() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValidationError) as exc:
        SessionService(
            tenant_id=0,
            user_id="u",
            session_repo=_FakeSessionRepo(),  # type: ignore[arg-type]
        )
    assert exc.value.code == "session.invalid_tenant_id"


def test_constructor_rejects_empty_user_id() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValidationError) as exc:
        SessionService(
            tenant_id=1,
            user_id="",
            session_repo=_FakeSessionRepo(),  # type: ignore[arg-type]
        )
    assert exc.value.code == "session.invalid_user_id"


def test_list_query_validates_page_size() -> None:
    # Arrange / Act / Assert
    with pytest.raises(ValidationError) as exc:
        SessionListQuery(page_size=0)
    assert exc.value.code == "session.invalid_page_size"


# ── create ───────────────────────────────────────────────────────────


async def test_create_inserts_row_with_stamped_timestamps() -> None:
    # Arrange
    repo = _FakeSessionRepo()
    service = _service(session_repo=repo)
    payload = _row(title="hello", description="greeting")

    # Act
    created = await service.create(payload)

    # Assert
    assert created.id == payload.id
    assert created.created_at == created.updated_at
    assert created.is_pinned is False
    assert created.pinned_at is None
    assert created.user_id == "user-1"
    assert created in repo.store.values()


async def test_create_rejects_cross_tenant_session() -> None:
    # Arrange
    service = _service(tenant_id=2)
    payload = _row(tenant_id=999)

    # Act / Assert
    with pytest.raises(ValidationError) as exc:
        await service.create(payload)
    assert exc.value.code == "session.tenant_mismatch"


async def test_create_mints_uuid_when_id_missing() -> None:
    # Arrange
    repo = _FakeSessionRepo()
    service = _service(session_repo=repo)
    payload = Session(
        id="",
        tenant_id=1,
        title=None,
        description=None,
        user_id="user-1",
        is_pinned=False,
        pinned_at=None,
        created_at=_NOW,
        updated_at=_NOW,
    )

    # Act
    created = await service.create(payload)

    # Assert
    assert created.id
    assert len(created.id) == 36  # UUID4 string length


# ── get / get_by_id / list ───────────────────────────────────────────


async def test_get_returns_row_when_visible() -> None:
    # Arrange
    repo = _FakeSessionRepo()
    row = _row()
    repo.store[row.id] = row
    service = _service(session_repo=repo)

    # Act
    fetched = await service.get(row.id)

    # Assert
    assert fetched.id == row.id


async def test_get_raises_not_found_for_absent_row() -> None:
    # Arrange
    service = _service()

    # Act / Assert
    with pytest.raises(NotFoundError) as exc:
        await service.get("missing")
    assert exc.value.code == "session.not_found"


async def test_get_rejects_blank_id() -> None:
    # Arrange
    service = _service()

    # Act / Assert
    with pytest.raises(ValidationError) as exc:
        await service.get("  ")
    assert exc.value.code == "session.id_required"


async def test_get_by_id_returns_none_for_absent_row() -> None:
    # Arrange
    service = _service()

    # Act
    result = await service.get_by_id("missing")

    # Assert
    assert result is None


async def test_list_all_returns_tenant_scoped_rows() -> None:
    # Arrange
    repo = _FakeSessionRepo()
    repo.store["a"] = _row(id="a", user_id="user-1", updated_at=_NOW)
    repo.store["b"] = _row(id="b", user_id="user-2", updated_at=_NOW)
    repo.store["c"] = _row(id="c", user_id="user-1", updated_at=_NOW)
    service = _service(session_repo=repo)

    # Act
    rows = await service.list_all()

    # Assert
    assert {r.id for r in rows} == {"a", "c"}


async def test_list_paged_returns_total_and_page() -> None:
    # Arrange
    repo = _FakeSessionRepo()
    for idx in range(5):
        rid = f"sess-{idx}"
        repo.store[rid] = _row(
            id=rid,
            updated_at=_NOW.replace(second=idx),
        )
    service = _service(session_repo=repo)

    # Act
    page = await service.list_paged(Pagination(page=2, page_size=2))

    # Assert
    assert page.total == 5
    assert page.page == 2
    assert page.page_size == 2
    assert len(page.data) == 2


async def test_list_with_filters_uses_owner_scope_by_default() -> None:
    # Arrange
    repo = _FakeSessionRepo()
    repo.store["a"] = _row(id="a", user_id="user-1", title="cat")
    repo.store["b"] = _row(id="b", user_id="user-2", title="cat")
    service = _service(session_repo=repo)

    # Act
    page = await service.list_with_filters(SessionListQuery(keyword="cat"))

    # Assert
    assert page.total == 1
    assert page.data[0].id == "a"


# ── pin toggle ───────────────────────────────────────────────────────


async def test_set_pinned_stamps_pinned_at_when_true() -> None:
    # Arrange
    repo = _FakeSessionRepo()
    row = _row()
    repo.store[row.id] = row
    service = _service(session_repo=repo)

    # Act
    ok = await service.set_pinned(row.id, True)

    # Assert
    assert ok is True
    assert repo.store[row.id].is_pinned is True
    assert repo.store[row.id].pinned_at is not None


async def test_set_pinned_clears_pinned_at_when_false() -> None:
    # Arrange
    repo = _FakeSessionRepo()
    row = _row(is_pinned=True, pinned_at=_NOW)
    repo.store[row.id] = row
    service = _service(session_repo=repo)

    # Act
    ok = await service.set_pinned(row.id, False)

    # Assert
    assert ok is True
    assert repo.store[row.id].is_pinned is False
    assert repo.store[row.id].pinned_at is None


async def test_set_pinned_returns_false_for_invisible_row() -> None:
    # Arrange
    repo = _FakeSessionRepo()
    row = _row(user_id="someone-else")
    repo.store[row.id] = row
    service = _service(session_repo=repo)

    # Act
    ok = await service.set_pinned(row.id, True)

    # Assert
    assert ok is False
    assert repo.store[row.id].is_pinned is False


# ── update / delete ──────────────────────────────────────────────────


async def test_update_overwrites_mutable_columns() -> None:
    # Arrange
    repo = _FakeSessionRepo()
    row = _row(title="old", description="old desc")
    repo.store[row.id] = row
    service = _service(session_repo=repo)
    new_payload = _row(
        id=row.id,
        title="new",
        description="new desc",
    )

    # Act
    updated = await service.update(new_payload)

    # Assert
    assert updated.title == "new"
    assert updated.description == "new desc"
    assert updated.is_pinned is False
    assert updated.created_at == row.created_at


async def test_update_rejects_cross_tenant_payload() -> None:
    # Arrange
    service = _service(tenant_id=2)

    # Act / Assert
    with pytest.raises(ValidationError) as exc:
        await service.update(_row(tenant_id=999))
    assert exc.value.code == "session.tenant_mismatch"


async def test_delete_returns_true_for_owned_row() -> None:
    # Arrange
    repo = _FakeSessionRepo()
    row = _row()
    repo.store[row.id] = row
    service = _service(session_repo=repo)

    # Act
    ok = await service.delete(row.id)

    # Assert
    assert ok is True
    assert repo.store[row.id].deleted_at is not None


async def test_delete_returns_false_for_invisible_row() -> None:
    # Arrange
    repo = _FakeSessionRepo()
    row = _row(user_id="someone-else")
    repo.store[row.id] = row
    service = _service(session_repo=repo)

    # Act
    ok = await service.delete(row.id)

    # Assert
    assert ok is False
    assert repo.store[row.id].deleted_at is None


async def test_batch_delete_skips_unknown_ids() -> None:
    # Arrange
    repo = _FakeSessionRepo()
    a = _row(id="a")
    b = _row(id="b")
    repo.store[a.id] = a
    repo.store[b.id] = b
    service = _service(session_repo=repo)

    # Act
    deleted = await service.batch_delete([a.id, "missing", b.id, a.id])

    # Assert
    assert deleted == 2
    assert repo.store[a.id].deleted_at is not None
    assert repo.store[b.id].deleted_at is not None


async def test_delete_all_removes_every_owned_row() -> None:
    # Arrange
    repo = _FakeSessionRepo()
    repo.store["a"] = _row(id="a", user_id="user-1")
    repo.store["b"] = _row(id="b", user_id="user-1")
    repo.store["c"] = _row(id="c", user_id="someone-else")
    service = _service(session_repo=repo)

    # Act
    deleted = await service.delete_all()

    # Assert
    assert deleted == 2
    assert repo.store["c"].deleted_at is None


# ── title generation ─────────────────────────────────────────────────


def test_title_generator_cleans_thinking_block_and_whitespace() -> None:
    # Arrange / Act
    cleaned = clean_response("<think>\nlong reasoning\n</think>  Hello World  \n")

    # Assert
    assert cleaned == "Hello World"


async def test_title_generator_renders_prompt_and_calls_chat() -> None:
    # Arrange
    chat = _FakeChat(content="Hello world")
    generator = TitleGenerator()

    # Act
    title = await generator.generate(
        chat=chat,  # type: ignore[arg-type]
        user_content="hello there",
        language="en",
    )

    # Assert
    assert title == "Hello world"
    assert len(chat.calls) == 1
    messages, _ = chat.calls[0]
    assert messages[0].role == "system"
    assert "concise title" in messages[0].content
    assert messages[1].role == "user"
    assert messages[1].content == "hello there"


async def test_title_generator_rejects_blank_content() -> None:
    # Arrange
    chat = _FakeChat(content="ignored")
    generator = TitleGenerator()

    # Act / Assert
    with pytest.raises(ValueError):
        await generator.generate(
            chat=chat,  # type: ignore[arg-type]
            user_content="   ",
        )


async def test_generate_title_returns_existing_title() -> None:
    # Arrange
    repo = _FakeSessionRepo()
    row = _row(title="already there")
    repo.store[row.id] = row
    message_repo = _FakeMessageRepo()
    service = build_session_service_with_title(
        session=None,  # type: ignore[arg-type]
        tenant_id=1,
        user_id="user-1",
        chat_factory=_FakeChatFactory(_FakeChat("should-not-call")),
        title_generator=TitleGenerator(),
    ).__class__  # placeholder
    # Build the real service via the same factory but on the fake repos.
    service = _title_service(repo, message_repo, _FakeChatFactory(_FakeChat("unused")))

    # Act
    title = await service.generate_title(row.id)

    # Assert
    assert title == "already there"
    assert message_repo.calls == []


async def test_generate_title_calls_llm_and_persists_title() -> None:
    # Arrange
    repo = _FakeSessionRepo()
    row = _row(title="")
    repo.store[row.id] = row
    user_message = Message(
        id="m1",
        session_id=row.id,
        role="user",
        content="What is the meaning of life?",
        created_at=_NOW,
        updated_at=_NOW,
    )
    message_repo = _FakeMessageRepo({row.id: user_message})
    chat = _FakeChat(content="Meaning of life")
    chat_factory = _FakeChatFactory(chat)
    service = _title_service(repo, message_repo, chat_factory)

    # Act
    title = await service.generate_title(row.id)

    # Assert
    assert title == "Meaning of life"
    assert repo.store[row.id].title == "Meaning of life"
    assert repo.store[row.id].updated_at > row.created_at
    assert chat_factory.calls == [(1, "")]
    assert len(chat.calls) == 1


async def test_generate_title_rejects_session_without_user_message() -> None:
    # Arrange
    repo = _FakeSessionRepo()
    row = _row()
    repo.store[row.id] = row
    message_repo = _FakeMessageRepo()
    service = _title_service(
        repo,
        message_repo,
        _FakeChatFactory(_FakeChat("ignored")),
    )

    # Act / Assert
    with pytest.raises(ValidationError) as exc:
        await service.generate_title(row.id)
    assert exc.value.code == "session.no_user_message"


async def test_generate_title_uses_supplied_model_id() -> None:
    # Arrange
    repo = _FakeSessionRepo()
    row = _row()
    repo.store[row.id] = row
    user_message = Message(
        id="m1",
        session_id=row.id,
        role="user",
        content="hello",
        created_at=_NOW,
        updated_at=_NOW,
    )
    message_repo = _FakeMessageRepo({row.id: user_message})
    chat_factory = _FakeChatFactory(_FakeChat("Greeting"))
    service = _title_service(repo, message_repo, chat_factory)

    # Act
    await service.generate_title(row.id, model_id="custom-model")

    # Assert
    assert chat_factory.calls == [(1, "custom-model")]


# ── Service w/ title helpers ────────────────────────────────────────


def _title_service(
    repo: _FakeSessionRepo,
    message_repo: _FakeMessageRepo,
    chat_factory: ChatFactoryLike,
) -> SessionService:
    """Build a service with the title seams wired to fakes."""
    return SessionService(
        tenant_id=1,
        user_id="user-1",
        session_repo=repo,  # type: ignore[arg-type]
        message_repo=message_repo,
        title_generator=TitleGenerator(),
        chat_factory=chat_factory,
    )


# ── Factory tests ────────────────────────────────────────────────────


async def test_factory_builds_service_with_repos() -> None:
    # Arrange
    class _SessionHolder:
        pass

    holder = _SessionHolder()
    # We only need a stand-in the repos can be built on; the
    # factory wires real ``SessionRepository`` / ``MessageRepository``
    # on the holder, but does not exercise any DB call here.
    service = build_session_service(
        holder,  # type: ignore[arg-type]
        tenant_id=1,
        user_id="user-1",
    )

    # Act / Assert — verify the service carries the expected scope.
    assert service.tenant_id == 1
    assert service.user_id == "user-1"
