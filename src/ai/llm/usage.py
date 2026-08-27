"""Standard ``[LLM Usage]`` log line shared by every chat implementation.

The purpose and prompt-prefix fingerprint labels attached by the orchestration
layer travel through contextvars (the async analogue of the reference
``context.Context`` values); they default to empty so callers can log usage
without annotating anything first.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from src.ai.llm.types import TokenUsage
from src.app_logging import logger

_LLM_CALL_PURPOSE: ContextVar[str] = ContextVar("llm_call_purpose", default="")
_LLM_PROMPT_PREFIX: ContextVar[str] = ContextVar("llm_prompt_prefix", default="")


@contextmanager
def with_llm_call_metadata(*, purpose: str = "", prefix_fingerprint: str = "") -> Iterator[None]:
    """Annotate the surrounding async scope with cache-observability labels.

    ``prefix_fingerprint`` must be a hash, never raw prompt content.
    """
    purpose_token = _LLM_CALL_PURPOSE.set(purpose.strip()) if purpose.strip() else None
    prefix_token = (
        _LLM_PROMPT_PREFIX.set(prefix_fingerprint.strip()) if prefix_fingerprint.strip() else None
    )
    try:
        yield
    finally:
        if prefix_token is not None:
            _LLM_PROMPT_PREFIX.reset(prefix_token)
        if purpose_token is not None:
            _LLM_CALL_PURPOSE.reset(purpose_token)


def llm_call_metadata_from_context() -> tuple[str, str]:
    """Return the ``(purpose, prefix_fingerprint)`` labels of this scope."""
    return _LLM_CALL_PURPOSE.get(), _LLM_PROMPT_PREFIX.get()


def log_usage(model: str, usage: TokenUsage | None) -> None:
    """Emit the standard usage line; a no-op when ``usage`` is ``None``."""
    if usage is None:
        return
    purpose, prefix_fingerprint = llm_call_metadata_from_context()
    logger.info(
        "[LLM Usage] model={}, purpose={}, prompt_prefix={}, prompt_tokens={}, "
        "completion_tokens={}, total_tokens={}, cached_tokens={}, "
        "cache_read_tokens={}, cache_write_tokens={}, cache_miss_tokens={}, "
        "cache_reported={}, cache_status={}",
        model,
        purpose,
        prefix_fingerprint,
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens,
        usage.cached_tokens,
        usage.cache_read_tokens,
        usage.cache_write_tokens,
        usage.cache_miss_tokens,
        usage.cache_reported,
        usage.cache_status,
    )


__all__ = [
    "llm_call_metadata_from_context",
    "log_usage",
    "with_llm_call_metadata",
]
