"""Unit tests for follow-up question parsing and fallbacks."""

from __future__ import annotations

from collections.abc import AsyncIterator

from src.ai.llm.types import Chat, ChatOptions, ChatResponse, Message, StreamResponse
from src.core.chat.messages.follow_up_generator import (
    fallback_follow_up_questions,
    generate_follow_up_questions,
    parse_follow_up_questions,
    resolve_follow_up_questions,
)


class _FakeChat:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[list[Message]] = []

    async def chat(self, messages: list[Message], opts: ChatOptions | None = None) -> ChatResponse:
        del opts
        self.calls.append(messages)
        return ChatResponse(content=self.content)

    def chat_stream(
        self, messages: list[Message], opts: ChatOptions | None = None
    ) -> AsyncIterator[StreamResponse]:
        del messages, opts
        raise NotImplementedError

    def get_model_name(self) -> str:
        return "fake"

    def get_model_id(self) -> str:
        return "fake"


class _FakeModels:
    def __init__(self, chat: _FakeChat, model_id: str | None = "m1") -> None:
        self._chat = chat
        self._model_id = model_id

    async def get_chat_model(self, *, tenant_id: int, model_id: str) -> Chat:
        del tenant_id, model_id
        return self._chat

    async def first_knowledge_qa_id(self, *, tenant_id: int) -> str | None:
        del tenant_id
        return self._model_id


def test_parse_follow_up_questions_reads_json_array() -> None:
    assert parse_follow_up_questions('["a", "b", "c", "d"]') == ["a", "b", "c"]


def test_parse_follow_up_questions_reads_json_inside_prose() -> None:
    raw = 'Here you go:\n["还有哪些要点？", "依据是什么？", "有何不同？"]\nThanks.'
    assert parse_follow_up_questions(raw) == ["还有哪些要点？", "依据是什么？", "有何不同？"]


def test_parse_follow_up_questions_reads_numbered_lines() -> None:
    raw = "1. 还有哪些要点？\n2. 依据是什么？\n3. 有何不同？"
    assert parse_follow_up_questions(raw) == ["还有哪些要点？", "依据是什么？", "有何不同？"]


def test_fallback_follow_up_questions_uses_query() -> None:
    questions = fallback_follow_up_questions(query="公募基金", answer="")
    assert len(questions) == 3
    assert "公募基金" in questions[0]


def test_fallback_follow_up_questions_empty_without_text() -> None:
    assert fallback_follow_up_questions(query="  ", answer="") == []


async def test_generate_follow_up_questions_parses_model_output() -> None:
    chat = _FakeChat('["q1", "q2", "q3"]')
    assert await generate_follow_up_questions(chat=chat, query="基金", answer="月报") == [
        "q1",
        "q2",
        "q3",
    ]


async def test_resolve_follow_up_questions_falls_back_without_model() -> None:
    questions = await resolve_follow_up_questions(
        tenant_id=1,
        chat_models=_FakeModels(_FakeChat(""), model_id=None),
        query="基金",
        answer="月报",
    )
    assert questions == fallback_follow_up_questions(query="基金", answer="月报")
