"""Shared value types for agent tool results."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.common.json import JsonObject


class ToolResult(BaseModel):
    """Structured result of a single tool invocation.

    Mirrors the upstream ``ToolResult`` shape: a human-readable
    ``output`` for the LLM and a structured ``data`` payload for
    programmatic consumers. ``error`` is set when the tool failed
    (``success=False``); ``images`` carries optional base64 data URIs
    produced by the tool (reserved for future image-returning tools).
    """

    model_config = ConfigDict(frozen=True)

    success: bool
    output: str = ""
    data: JsonObject | None = Field(default=None)
    error: str | None = Field(default=None)
    images: list[str] = Field(default_factory=list)


__all__ = ["ToolResult"]
