"""Session service — CRUD, pin toggle, and title generation.

The service is request-scoped: it carries the per-request
``AsyncSession`` (through the repositories) plus the caller's
``tenant_id`` / ``user_id`` so every read enforces the owner scope
that the upstream contract requires.

Operations
---------

- CRUD: ``create`` / ``get`` / ``get_by_id`` / ``list_all`` /
  ``list_paged`` / ``update`` / ``delete`` / ``batch_delete`` /
  ``delete_all``.
- Pin: ``set_pinned`` toggles ``is_pinned`` / ``pinned_at`` and
  returns whether a visible row was affected.
- Title: ``generate_title`` fetches the first user message of the
  session, runs it through the title generator, and persists the
  result on the row.

Heavy dependencies (model resolution, the chat client) sit behind
``Protocol`` seams so the service is testable without a live model
service. The factory ``build_session_service`` wires the production
seams when a real session is in scope.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from src.ai.llm.types import Chat
from src.common.exception import NotFoundError, ValidationError
from src.common.pagination import Pagination, PaginationResponse
from src.core.chat.sessions.title_gen import (
    TitleGenerator,
    TitleGeneratorLike,
)
from src.core.chat.sessions.types import SessionInfo
from src.db.dao.session_repository import SessionRepository
from src.db.models.message import Message
from src.db.models.session import Session

logger = logging.getLogger(__name__)


#: Knowledge-QA model type used to pick a default chat client for
#: title generation when the caller does not supply ``model_id``.
_MODEL_TYPE_KNOWLEDGE_QA = "knowledge_qa"

#: Maximum page size accepted by the paged list endpoints. The
#: underlying pagination module already caps at 1000, so this is the
#: hard ceiling callers can request.
_MAX_PAGE_SIZE = 1000


# ── Injectable seams ──────────────────────────────────────────────────


@runtime_checkable
class ChatFactoryLike(Protocol):
    """Resolves a chat client for a model id.

    The production factory reads the ``models`` table, picks a
    ``KnowledgeQA`` model by default, and builds a chat client. The
    seam is structural so tests can substitute a tiny stub.
    """

    async def resolve_chat(
        self,
        *,
        tenant_id: int,
        model_id: str = "",
    ) -> tuple[Chat, str]:
        """Return ``(chat_client, model_id)`` for title generation.

        ``model_id`` is echoed back so callers can log the resolved
        model even when they did not pass one in.
        """
        ...


@runtime_checkable
class SessionMessageReader(Protocol):
    """Subset of :class:`MessageRepository` the title flow needs.

    Keeping the seam narrow lets tests provide a one-line fake.
    """

    async def get_first_user_message(self, session_id: str) -> Message | None: ...


# ── DTOs ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SessionListQuery:
    """Filter and pagination parameters for the session list endpoint.

    Mirrors the wire shape: ``keyword`` is a case-insensitive title
    search, ``page`` / ``page_size`` are 1-based pagination knobs,
    ``source`` is a non-empty source filter (admin-only paths), and
    ``agent_id`` is a future extension slot.
    """

    keyword: str = ""
    page: int = 1
    page_size: int = 20
    source: str = ""
    agent_id: str = ""

    def __post_init__(self) -> None:
        if self.page < 1:
            raise ValidationError(
                code="session.invalid_page",
                message="page must be >= 1",
            )
        if self.page_size < 1 or self.page_size > _MAX_PAGE_SIZE:
            raise ValidationError(
                code="session.invalid_page_size",
                message=f"page_size must be in [1, {_MAX_PAGE_SIZE}]",
            )


# ── The service ──────────────────────────────────────────────────────


class SessionService:
    """Per-request session facade.

    Methods accept the data they need; the service never reads the
    tenant or user from globals. Repository methods that take a
    user scope (read / pin / update / delete) are always called with
    the caller's ``user_id`` so owner-scope rules are honoured
    uniformly.
    """

    def __init__(
        self,
        *,
        tenant_id: int,
        user_id: str,
        session_repo: SessionRepository,
        message_repo: SessionMessageReader | None = None,
        title_generator: TitleGeneratorLike | None = None,
        chat_factory: ChatFactoryLike | None = None,
    ) -> None:
        if tenant_id <= 0:
            raise ValidationError(
                code="session.invalid_tenant_id",
                message="tenant_id must be positive",
            )
        if not user_id:
            raise ValidationError(
                code="session.invalid_user_id",
                message="user_id is required",
            )
        self._tenant_id = tenant_id
        self._user_id = user_id
        self._session_repo = session_repo
        self._message_repo = message_repo
        self._title_generator = title_generator or TitleGenerator()
        self._chat_factory = chat_factory

    @property
    def tenant_id(self) -> int:
        """The caller's active workspace id."""
        return self._tenant_id

    @property
    def user_id(self) -> str:
        """The caller's user id (owner scope)."""
        return self._user_id

    # ── Create ──────────────────────────────────────────────────────

    async def create(
        self,
        *,
        title: str | None = None,
        description: str | None = None,
        session_id: str = "",
    ) -> SessionInfo:
        """Insert a new session row for the caller and return its projection.

        Mints a fresh UUID ``id`` when the caller did not supply one and
        stamps ``created_at`` / ``updated_at`` to the current time. The
        tenant and owner are always the caller's — cross-tenant creation
        is structurally impossible through this path.
        """
        now = _now()
        new_row = Session(
            id=session_id.strip() or _new_id(),
            tenant_id=self._tenant_id,
            title=title,
            description=description,
            user_id=self._user_id,
            is_pinned=False,
            pinned_at=None,
            created_at=now,
            updated_at=now,
        )
        created = await self._session_repo.create(new_row)
        return SessionInfo.from_row(created)

    # ── Read ────────────────────────────────────────────────────────

    async def get(self, session_id: str) -> SessionInfo:
        """Return a live session visible to the caller, or raise."""
        if not session_id or not session_id.strip():
            raise ValidationError(
                code="session.id_required",
                message="session id is required",
            )
        row = await self._session_repo.get_by_id_for_user(
            tenant_id=self._tenant_id,
            user_id=self._user_id,
            id=session_id,
        )
        if row is None:
            raise NotFoundError(
                code="session.not_found",
                message=f"session {session_id} not found",
            )
        return SessionInfo.from_row(row)

    async def get_by_id(self, session_id: str) -> SessionInfo | None:
        """Return a live session by id, ignoring the owner scope.

        Tenant-scoped only — a non-admin caller should normally use
        :meth:`get`. The return value is ``None`` when no live row
        matches; callers that need a 404 should raise themselves.
        """
        if not session_id or not session_id.strip():
            raise ValidationError(
                code="session.id_required",
                message="session id is required",
            )
        row = await self._session_repo.get_by_id(
            tenant_id=self._tenant_id,
            id=session_id,
        )
        return SessionInfo.from_row(row) if row is not None else None

    async def list_all(self) -> list[SessionInfo]:
        """Every live session of the caller's tenant, newest first."""
        rows = await self._session_repo.list_by_tenant(
            tenant_id=self._tenant_id,
            user_id=self._user_id,
        )
        return [SessionInfo.from_row(row) for row in rows]

    async def list_paged(
        self,
        pagination: Pagination,
    ) -> PaginationResponse[SessionInfo]:
        """One page of the tenant's sessions plus the total count."""
        rows, total = await self._session_repo.list_paged(
            tenant_id=self._tenant_id,
            user_id=self._user_id,
            page=pagination.page,
            page_size=pagination.page_size,
        )
        return PaginationResponse[SessionInfo](
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
            data=[SessionInfo.from_row(row) for row in rows],
        )

    async def list_with_filters(
        self,
        query: SessionListQuery,
    ) -> PaginationResponse[SessionInfo]:
        """Search the tenant's sessions by title, with pagination.

        The owner scope is applied unless ``source`` names a channel
        (in which case the query is tenant-wide; admin-only). The
        admin gate is the caller's responsibility — the service
        always runs the filter the request asks for.
        """
        user_scope = "" if query.source else self._user_id
        rows, total = await self._session_repo.list_paged(
            tenant_id=self._tenant_id,
            user_id=user_scope,
            page=query.page,
            page_size=query.page_size,
            keyword=query.keyword,
        )
        return PaginationResponse[SessionInfo](
            total=total,
            page=query.page,
            page_size=query.page_size,
            data=[SessionInfo.from_row(row) for row in rows],
        )

    # ── Pin toggle ───────────────────────────────────────────────────

    async def set_pinned(self, session_id: str, pinned: bool) -> bool:
        """Pin or unpin a session for the caller.

        Returns ``True`` when a live, visible row was affected, or
        ``False`` when the row is absent / not owned by the caller.
        """
        if not session_id or not session_id.strip():
            raise ValidationError(
                code="session.id_required",
                message="session id is required",
            )
        return await self._session_repo.set_pinned(
            tenant_id=self._tenant_id,
            id=session_id,
            pinned=pinned,
            now=_now(),
            user_id=self._user_id,
        )

    # ── Update ──────────────────────────────────────────────────────

    async def update(
        self,
        *,
        session_id: str,
        title: str | None = None,
        description: str | None = None,
    ) -> SessionInfo:
        """Overwrite the mutable columns of an existing session.

        The owner scope is enforced by the repository: a caller
        cannot edit another user's session. The mutable column set
        is ``title`` / ``description``; ``is_pinned`` and
        ``pinned_at`` move through :meth:`set_pinned` instead.
        """
        if not session_id or not session_id.strip():
            raise ValidationError(
                code="session.id_required",
                message="session id is required",
            )
        existing = await self.get(session_id)
        now = _now()
        updated = await self._session_repo.update(
            Session(
                id=existing.id,
                tenant_id=existing.tenant_id,
                title=title,
                description=description,
                user_id=existing.user_id,
                is_pinned=existing.is_pinned,
                pinned_at=existing.pinned_at,
                created_at=existing.created_at,
                updated_at=now,
            ),
            user_id=self._user_id,
        )
        return SessionInfo.from_row(updated)

    # ── Delete ──────────────────────────────────────────────────────

    async def delete(self, session_id: str) -> bool:
        """Soft-delete a session owned by the caller.

        Returns ``True`` when a live, visible row was deleted, or
        ``False`` when the row was absent / not owned by the caller.
        """
        if not session_id or not session_id.strip():
            raise ValidationError(
                code="session.id_required",
                message="session id is required",
            )
        return await self._session_repo.soft_delete(
            tenant_id=self._tenant_id,
            id=session_id,
            now=_now(),
            user_id=self._user_id,
        )

    async def batch_delete(self, session_ids: Iterable[str]) -> int:
        """Soft-delete every id in ``session_ids`` the caller owns.

        Returns the number of rows deleted. Unknown ids are silently
        skipped; non-visible ids are not deleted. The repository is
        called with the caller-owned subset so cross-user deletes
        cannot happen.
        """
        visible: list[str] = []
        seen: set[str] = set()
        for sid in session_ids:
            if not sid or not sid.strip():
                continue
            if sid in seen:
                continue
            seen.add(sid)
            try:
                await self.get(sid)
            except NotFoundError:
                continue
            visible.append(sid)
        if not visible:
            return 0
        deleted = 0
        now = _now()
        for sid in visible:
            if await self._session_repo.soft_delete(
                tenant_id=self._tenant_id,
                id=sid,
                now=now,
                user_id=self._user_id,
            ):
                deleted += 1
        return deleted

    async def delete_all(self) -> int:
        """Soft-delete every session the caller owns in the tenant.

        Returns the number of rows deleted. The repository's per-row
        delete keeps the owner scope in place.
        """
        rows = await self._session_repo.list_by_tenant(
            tenant_id=self._tenant_id,
            user_id=self._user_id,
        )
        deleted = 0
        now = _now()
        for row in rows:
            if await self._session_repo.soft_delete(
                tenant_id=self._tenant_id,
                id=row.id,
                now=now,
                user_id=self._user_id,
            ):
                deleted += 1
        return deleted

    # ── Title generation ────────────────────────────────────────────

    async def generate_title(
        self,
        session_id: str,
        *,
        model_id: str = "",
    ) -> str:
        """Generate and persist a title for ``session_id``.

        Returns the existing title when one is already set. The first
        user message of the session is fed through the injected
        title generator; the generated title is written back to the
        row's ``title`` column with ``updated_at`` stamped.
        """
        if not session_id or not session_id.strip():
            raise ValidationError(
                code="session.id_required",
                message="session id is required",
            )
        if self._message_repo is None:
            raise ValidationError(
                code="session.message_repo_unconfigured",
                message="message repository is required for title generation",
            )
        if self._chat_factory is None:
            raise ValidationError(
                code="session.chat_factory_unconfigured",
                message="chat factory is required for title generation",
            )

        session = await self.get(session_id)
        if session.title:
            return session.title

        message = await self._message_repo.get_first_user_message(session_id)
        if message is None or not message.content:
            raise ValidationError(
                code="session.no_user_message",
                message="no user message found for title generation",
            )

        chat, resolved_model_id = await self._chat_factory.resolve_chat(
            tenant_id=self._tenant_id,
            model_id=model_id,
        )
        title = await self._title_generator.generate(
            chat=chat,
            user_content=message.content,
            language="en",
            model_id=resolved_model_id,
        )

        updated = await self._session_repo.update(
            Session(
                id=session.id,
                tenant_id=session.tenant_id,
                title=title,
                description=session.description,
                user_id=session.user_id,
                is_pinned=session.is_pinned,
                pinned_at=session.pinned_at,
                created_at=session.created_at,
                updated_at=_now(),
            ),
            user_id=self._user_id,
        )
        logger.info("session %s title generated (len=%d)", session.id, len(title))
        return updated.title or title


# ── Helpers ──────────────────────────────────────────────────────────


def _now() -> datetime:
    """Return a timezone-aware ``now`` for stamping rows."""
    return datetime.now(UTC)


def _new_id() -> str:
    """Generate a UUID for a freshly created session."""
    return str(uuid.uuid4())


__all__ = [
    "ChatFactoryLike",
    "SessionListQuery",
    "SessionMessageReader",
    "SessionService",
]
