"""Turn a completed Q&A pair into a short list of follow-up questions."""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol, runtime_checkable

from src.ai.llm.types import Chat, ChatOptions, Message
from src.common.json import JsonValue

logger = logging.getLogger(__name__)

_QUESTION_CAP: int = 3
_ANSWER_CHAR_CAP: int = 4000
_TEMPERATURE: float = 0.4
_MAX_TOKENS: int = 800
_THINK_BLOCK: re.Pattern[str] = re.compile(r"(?s)<think>.*?</think>")
_FENCE: re.Pattern[str] = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_LINE_PREFIX: re.Pattern[str] = re.compile(r"^\s*(?:[-*]|\d+[.)、])\s*")

_SYSTEM_PROMPT: str = (
    "You write short follow-up questions a reader would ask next. "
    "Reply with a JSON array of strings only. No markdown."
)


@runtime_checkable
class ChatClientFactory(Protocol):
    """Resolves a live chat client without importing the infra service type."""

    async def get_chat_model(self, *, tenant_id: int, model_id: str) -> Chat: ...

    async def first_knowledge_qa_id(self, *, tenant_id: int) -> str | None: ...


def parse_follow_up_questions(raw: str, *, limit: int = _QUESTION_CAP) -> list[str]:
    """Read a JSON array or a numbered list into trimmed question texts."""
    text = _THINK_BLOCK.sub("", raw).strip()
    text = _FENCE.sub("", text).strip()
    parsed = _from_json(text)
    if parsed:
        return parsed[:limit]
    return _from_lines(text)[:limit]


def fallback_follow_up_questions(*, query: str, answer: str) -> list[str]:
    """Build three generic follow-ups when the model is missing or fails."""
    topic = query.strip() or _topic_from_answer(answer)
    if not topic:
        return []
    if len(topic) > 40:
        topic = topic[:40] + "…"
    return [
        f"关于「{topic}」还有哪些关键细节？",
        "这个结论的依据是什么？",
        "和相关材料相比有什么不同？",
    ]


async def generate_follow_up_questions(
    *,
    chat: Chat,
    query: str,
    answer: str,
) -> list[str]:
    """Ask the chat model for follow-ups, then parse or fall back."""
    prompt = _build_prompt(query=query, answer=answer)
    response = await chat.chat(
        [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(role="user", content=prompt),
        ],
        ChatOptions(temperature=_TEMPERATURE, max_tokens=_MAX_TOKENS, thinking=False),
    )
    parsed = parse_follow_up_questions(response.content)
    return parsed or fallback_follow_up_questions(query=query, answer=answer)


async def resolve_follow_up_questions(
    *,
    tenant_id: int,
    chat_models: ChatClientFactory | None,
    query: str,
    answer: str,
) -> list[str]:
    """Use the first KnowledgeQA model, or the template fallback."""
    fallback = fallback_follow_up_questions(query=query, answer=answer)
    if chat_models is None:
        return fallback
    model_id = await chat_models.first_knowledge_qa_id(tenant_id=tenant_id)
    if not model_id:
        return fallback
    try:
        chat = await chat_models.get_chat_model(tenant_id=tenant_id, model_id=model_id)
        return await generate_follow_up_questions(chat=chat, query=query, answer=answer)
    except Exception:
        logger.exception("follow-up generation failed")
        return fallback


def _build_prompt(*, query: str, answer: str) -> str:
    clipped = answer.strip()[:_ANSWER_CHAR_CAP]
    return (
        f"User question:\n{query.strip()}\n\n"
        f"Assistant answer:\n{clipped}\n\n"
        f"Write {_QUESTION_CAP} distinct follow-up questions in the same language."
    )


def _topic_from_answer(answer: str) -> str:
    compact = " ".join(answer.split())
    return compact[:40]


def _from_json(raw: str) -> list[str]:
    payload = _load_json_array(raw)
    if payload is None:
        return []
    return [item.strip() for item in payload if isinstance(item, str) and item.strip()]


def _load_json_array(raw: str) -> list[JsonValue] | None:
    try:
        payload: JsonValue = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end <= start:
            return None
        try:
            payload = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, list) else None


def _from_lines(raw: str) -> list[str]:
    questions: list[str] = []
    for line in raw.splitlines():
        text = _LINE_PREFIX.sub("", line).strip().strip("\"'")
        if text:
            questions.append(text)
    return questions


__all__ = [
    "ChatClientFactory",
    "fallback_follow_up_questions",
    "generate_follow_up_questions",
    "parse_follow_up_questions",
    "resolve_follow_up_questions",
]
