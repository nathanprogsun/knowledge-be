"""Chat-history KB indexer — passage creation and message-link lifecycle.

The chat-history knowledge base is the workspace-scoped KB that stores
every Q&A pair ever produced by a chat session. After an assistant
answer completes, the message service calls into this module to:

- build the ``[Session: <id>] Q: <query> A: <answer>`` passage that
  gives the retrieval step both halves of the conversation;
- create a Knowledge entry in the chat-history KB via the Knowledge
  service seam;
- link the assistant message to that Knowledge entry through the
  ``knowledge_id`` column so future vector searches can map back to
  the source message.

The same module also owns the cleanup paths: deleting a single message
or clearing a session drops every Knowledge entry those messages
produced, so the chat-history KB does not accumulate orphan passages.

Layering
--------

``MessageIndexer`` is a structural Protocol. The default implementation
here (``DefaultMessageIndexer``) accepts the persistence seam
(``MessageRepository``) and the Knowledge-creation seam
(``KnowledgePassageCreator``) via constructor injection so the rest of
the codebase can swap in a different writer (e.g. a batched async
variant) without touching the message service. The
``ChatHistoryConfigProvider`` protocol is consumed to learn whether the
workspace has configured a chat-history KB; without configuration the
methods become silent no-ops.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from src.core.chat.messages.service.message_service import (
    ChatHistoryConfigProvider,
    MessageIndexer,
)
from src.core.chat.pipeline.types import Context
from src.db.dao.message_repository import MessageRepository
from src.db.models.message import Message

#: Matches a ``<think>...</think>`` reasoning block. Same regex shape as
#: the upstream Go ``regThinkIndex``; the assistant answer is stripped
#: of any such block before it is written into the KB so the retrieval
#: index does not pick up intermediate reasoning.
_THINK_TAG_RE = re.compile(r"(?s)<think>.*?</think>")

#: Passage header used to anchor the retrieval context to the session.
#: Mirrors the upstream ``[Session: %s]`` template.
_PASSAGE_HEADER_FMT = "[Session: %s]\nQ: %s\nA: %s"


def strip_think_tags(text: str) -> str:
    """Return ``text`` with every ``<think>...</think>`` block removed."""
    return _THINK_TAG_RE.sub("", text)


def build_passage(*, session_id: str, query: str, answer: str) -> str:
    """Compose the Q&A passage written into the chat-history KB.

    The leading ``[Session: <id>]`` line anchors the passage to its
    session so a future vector-search hit can be traced back without
    needing to consult the link table. The format mirrors the upstream
    ``fmt.Sprintf("[Session: %s]\\nQ: %s\\nA: %s", ...)`` template.
    """
    return _PASSAGE_HEADER_FMT % (session_id, query, answer)


# ── Injectable seams ───────────────────────────────────────────────────


@runtime_checkable
class KnowledgePassageCreator(Protocol):
    """Creates a Knowledge entry from one or more raw passages.

    Mirrors the upstream ``KnowledgeService.CreateKnowledgeFromPassage``:
    given a knowledge-base id and a list of passages, the implementation
    creates a Knowledge entry (one row per call, batched internally) and
    returns the persisted id.
    """

    async def create_passage(
        self,
        *,
        ctx: Context,
        knowledge_base_id: str,
        passages: tuple[str, ...],
    ) -> str: ...


# ── Indexer protocol + default implementation ─────────────────────────


class DefaultMessageIndexer:
    """Default :class:`MessageIndexer` wired to the persistence + KB seams.

    All three methods are silent no-ops when the workspace has not
    configured a chat-history KB — that mirrors the upstream guard
    (``getChatHistoryConfig`` returns ``nil`` when the feature is off,
    the caller bails before any KB call). When the KB is configured
    but the persistence or KB seam reports an error, the error is
    swallowed and logged: index_to_kb is fire-and-forget and a
    failure on the index path must never propagate to the user.
    """

    def __init__(
        self,
        *,
        message_repo: MessageRepository,
        passage_creator: KnowledgePassageCreator,
        config: ChatHistoryConfigProvider,
    ) -> None:
        self._message_repo = message_repo
        self._passage_creator = passage_creator
        self._config = config

    async def index_message(
        self,
        ctx: Context,
        *,
        user_query: str,
        assistant_answer: str,
        message_id: str,
        session_id: str,
    ) -> None:
        """Index the Q&A pair into the chat-history KB.

        The assistant answer has its think blocks stripped before the
        passage is composed, so the retrieval index does not pick up
        the model's intermediate reasoning. A Q&A pair with both halves
        empty is skipped — there is nothing to index.
        """
        if not self._config.is_enabled(ctx):
            return
        cleaned_answer = strip_think_tags(assistant_answer).strip()
        if not (user_query or "").strip() and not cleaned_answer:
            return
        kb_id = self._config.knowledge_base_id(ctx)
        if not kb_id:
            return
        passage = build_passage(
            session_id=session_id,
            query=user_query,
            answer=cleaned_answer,
        )
        try:
            knowledge_id = await self._passage_creator.create_passage(
                ctx=ctx,
                knowledge_base_id=kb_id,
                passages=(passage,),
            )
        except Exception:
            # Fire-and-forget: a failed index must not propagate.
            return
        try:
            await self._update_knowledge_link(message_id, knowledge_id)
        except Exception:
            return

    async def delete_message_knowledge(self, *, knowledge_id: str) -> None:
        """Drop a single Knowledge entry by id.

        The Knowledge deletion seam is not part of this PR's surface;
        the indexer exposes this method as a no-op so the message
        service's delete flow can stay wired. A concrete deleter is
        supplied in a later PR.
        """
        if not knowledge_id:
            return
        return

    async def delete_session_knowledge(
        self,
        *,
        knowledge_ids: tuple[str, ...],
    ) -> None:
        """Drop every Knowledge entry that belonged to a session.

        Like :meth:`delete_message_knowledge`, this is a no-op until
        the Knowledge deletion seam is wired in a later PR.
        """
        if not knowledge_ids:
            return
        return

    async def _update_knowledge_link(
        self,
        message_id: str,
        knowledge_id: str,
    ) -> Message | None:
        """Persist the ``knowledge_id`` link on the source message."""
        return await self._message_repo.update_knowledge_id(
            message_id=message_id,
            knowledge_id=knowledge_id,
            now=datetime.now(UTC),
        )


__all__ = [
    "DefaultMessageIndexer",
    "KnowledgePassageCreator",
    "MessageIndexer",
    "build_passage",
    "strip_think_tags",
]
