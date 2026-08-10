"""Title generation helper for the chat-session domain.

A title is a short, locale-aware summary of the first user turn; the
generation pipeline renders a system prompt + the user content through
a chat client and trims the response into a usable string.

The helper sits in its own module so the service layer can swap it
out (e.g. a no-op stub for tests) without dragging the chat-client
surface into the request-scoped service module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.ai.llm.types import Chat, ChatOptions, ChatResponse, Message

#: Default temperature for title generation. Lower than the QA
#: pipeline's default so titles stay deterministic across runs.
DEFAULT_TEMPERATURE = 0.3

#: Default system prompt for title generation. ``{language}`` is
#: substituted with the active locale at call time.
DEFAULT_SYSTEM_PROMPT = (
    "Generate a concise title in {language} for the following user "
    "question. Reply with the title only, no quotes, no trailing "
    "punctuation."
)


@dataclass(frozen=True, slots=True)
class TitleGenerator:
    """Default title generator: render + chat + trim.

    The generator is a small value object — the chat client is the
    only dependency and is passed in, so the service layer can wire
    a real client in production and a fake in tests.
    """

    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    async def generate(
        self,
        *,
        chat: Chat,
        user_content: str,
        language: str = "en",
        model_id: str = "",
    ) -> str:
        """Return a generated title for ``user_content``.

        ``language`` is substituted into the system prompt template.
        ``model_id`` is a hint for downstream telemetry — the chat
        client itself is the one configured for the target model.
        The response is trimmed into a single-line title.
        """
        if not user_content or not user_content.strip():
            raise ValueError("user_content is required")
        system = self.system_prompt.format(language=language or "en")
        options = ChatOptions(
            temperature=DEFAULT_TEMPERATURE,
            thinking=False,
        )
        _ = model_id  # accepted for call-site symmetry; not consumed
        response: ChatResponse = await chat.chat(
            [
                Message(role="system", content=system),
                Message(role="user", content=user_content),
            ],
            options,
        )
        return _clean(response.content)


class TitleGeneratorLike(Protocol):
    """Structural seam: any object exposing :meth:`generate`."""

    async def generate(
        self,
        *,
        chat: Chat,
        user_content: str,
        language: str = "en",
        model_id: str = "",
    ) -> str: ...


def _clean(content: str) -> str:
    """Collapse a model response into a single-line title.

    Newlines are folded to spaces, surrounding whitespace is stripped,
    and any leading ``<think>...</think>`` reasoning block is dropped
    so the caller receives a clean heading string.
    """
    text = (content or "").strip()
    if not text:
        return ""
    # Drop a leading <think>...</think> reasoning block if the model
    # emitted one. The match is case-insensitive and tolerates the
    # block occupying several lines.
    lowered = text.lower()
    if lowered.startswith("<think>"):
        end = lowered.find("</think>")
        if end != -1:
            text = text[end + len("</think>"):].strip()
    # Titles are single-line: collapse any embedded newlines.
    return " ".join(text.split())


#: Public alias for the response cleaner. ``TitleGenerator.generate``
#: invokes this internally; tests import it directly to assert the
#: trim / fold behaviour without spinning up a chat client.
clean_response = _clean


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_TEMPERATURE",
    "TitleGenerator",
    "TitleGeneratorLike",
    "clean_response",
]
