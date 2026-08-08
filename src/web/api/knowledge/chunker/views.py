"""Chunker preview endpoint - read-only chunking diagnostics.

Maps the ``POST /chunker/preview`` debug handler: it runs the adaptive
chunker over the supplied text without touching the database or
generating embeddings, so the KB editor can experiment with chunking
parameters before committing to a re-index.

Input text is capped at 64k characters (rune-counted, mirroring the
upstream ceiling) so a single preview cannot tie up the splitter for
long; the split runs in a worker thread with a 5s timeout, and on
timeout the handler returns 504 while the thread finishes naturally.

Error responses deliberately keep the handler's custom shape
(``{"success": false, "error": "<string>"}`` with 400/413/504 statuses)
rather than the standard error envelope, matching the upstream handler
for this standalone debug endpoint. Malformed JSON bodies still answer
with the framework's standard 422.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import asdict

from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.core.knowledge.documents.chunker import (
    LangMixed,
    SplitterConfig,
    approx_token_count_from_rune_len,
    profile_document,
    split_with_diagnostics,
)
from src.web.deps import AuthDep, RoleViewerDep

# Caps the input text size so a single preview request cannot tie up the
# splitter for long (see the module docstring).
_PREVIEW_MAX_CHARS = 64 * 1024

# Caps the chunks returned in one preview so the UI does not choke on
# pathological splits; stats are computed over the full set first.
_PREVIEW_MAX_CHUNKS = 500

# Caps how long the handler waits for the splitter thread before 504.
_PREVIEW_TIMEOUT_SECONDS = 5.0


class PreviewChunkingPayload(BaseModel):
    """Chunking knobs the preview honours; mirrors the snake_case config.

    Only the splitter-relevant fields are exposed - the full chunking
    config carries parser-engine rules and parent-child sizes the preview
    path does not need.
    """

    model_config = ConfigDict(frozen=True)

    chunk_size: int = 0
    chunk_overlap: int = 0
    separators: list[str] = Field(default_factory=list)
    strategy: str = ""
    token_limit: int = 0
    languages: list[str] = Field(default_factory=list)


class PreviewChunkingRequest(BaseModel):
    """Body accepted by ``POST /chunker/preview``."""

    model_config = ConfigDict(frozen=True)

    text: str
    chunking_config: PreviewChunkingPayload = Field(
        default_factory=PreviewChunkingPayload
    )


class PreviewChunkResult(BaseModel):
    """One chunk emitted during a preview."""

    model_config = ConfigDict(frozen=True)

    seq: int
    start: int
    end: int
    size_chars: int
    size_tokens_approx: int
    context_header: str | None = Field(default=None)
    content: str


class PreviewChunkingStats(BaseModel):
    """Chunk-size distribution over the full (pre-truncation) chunk set.

    The summary fields default to 0 so an empty chunk set still serializes
    a well-formed distribution (mirrors the upstream zero-value stats).
    """

    model_config = ConfigDict(frozen=True)

    count: int = 0
    avg_chars: int = 0
    min_chars: int = 0
    max_chars: int = 0
    stddev_chars: int = 0
    truncated_to: int | None = Field(default=None)


class PreviewTierRejection(BaseModel):
    """Why one chunking tier was rejected and the chain advanced."""

    model_config = ConfigDict(frozen=True)

    tier: str
    reason: str


class PreviewChunkingProfile(BaseModel):
    """Document profile that drove tier selection."""

    model_config = ConfigDict(frozen=True)

    total_chars: int
    total_lines: int
    avg_line_len: float
    std_line_len: float
    md_heading_counts: dict[int, int]
    md_heading_total: int
    numbered_section_count: int
    all_caps_short_line_count: int
    blank_paragraph_breaks: int
    form_feed_count: int
    visual_sep_count: int
    german_chapter_count: int
    english_chapter_count: int
    chinese_chapter_count: int
    repeated_footer_count: int
    has_tables: bool
    has_code: bool
    code_ratio: float
    detected_langs: list[str]


class PreviewChunkingData(BaseModel):
    """The full preview payload: chunks plus tier diagnostics."""

    model_config = ConfigDict(frozen=True)

    selected_tier: str
    tier_chain: list[str]
    rejected: list[PreviewTierRejection]
    profile: PreviewChunkingProfile | None
    chunks: list[PreviewChunkResult]
    stats: PreviewChunkingStats


class PreviewChunkingEnvelope(BaseModel):
    """``{"success": true, "data": {...}}`` - the success wrapper."""

    model_config = ConfigDict(frozen=True)

    success: bool
    data: PreviewChunkingData


def compute_chunk_size_stats(rune_lengths: list[int]) -> PreviewChunkingStats:
    """Summarise count / avg / min / max / stddev of chunk rune lengths.

    Computed over the FULL chunk set so the metrics stay representative
    even when the response truncates to ``_PREVIEW_MAX_CHUNKS`` items.
    """
    stats = PreviewChunkingStats(count=len(rune_lengths))
    if not rune_lengths:
        return stats
    total = 0.0
    total_sq = 0.0
    min_len = math.inf
    max_len = 0
    for length in rune_lengths:
        total += float(length)
        total_sq += float(length) * float(length)
        if length < min_len:
            min_len = float(length)
        if length > max_len:
            max_len = length
    avg = total / float(len(rune_lengths))
    variance = total_sq / float(len(rune_lengths)) - avg * avg
    if variance < 0:
        # Float precision can push the variance slightly below zero on
        # near-uniform inputs; clamp so sqrt does not return NaN.
        variance = 0.0
    return PreviewChunkingStats(
        count=len(rune_lengths),
        avg_chars=int(avg + 0.5),
        min_chars=int(min_len),
        max_chars=max_len,
        stddev_chars=int(math.sqrt(variance) + 0.5),
    )


def build_preview_data(text: str, payload: PreviewChunkingPayload) -> PreviewChunkingData:
    """Run the adaptive chunker on ``text`` and shape the preview payload.

    Sync and CPU-bound, so it runs on a worker thread; ``split_with_diagnostics``
    applies the strategy defaults (chunk size / overlap / separators /
    token-limit clamping) the same way the full pipeline does.
    """
    cfg = SplitterConfig(
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
        separators=list(payload.separators),
        strategy=payload.strategy,
        token_limit=payload.token_limit,
        languages=list(payload.languages),
    )
    chunks, diag = split_with_diagnostics(text, cfg)
    profile = diag.profile
    if profile is None:
        # Explicit strategies skip the profiling pass; materialize it here
        # so the UI still gets document stats.
        profile = profile_document(text)

    lang = LangMixed
    if profile.detected_langs:
        lang = profile.detected_langs[0]

    # Compute rune lengths once per chunk; reused for stats and payload.
    rune_lengths = [len(chunk.content) for chunk in chunks]
    total_count = len(chunks)
    stats = compute_chunk_size_stats(rune_lengths)
    if total_count > _PREVIEW_MAX_CHUNKS:
        stats = stats.model_copy(update={"truncated_to": total_count})
        chunks = chunks[:_PREVIEW_MAX_CHUNKS]
        rune_lengths = rune_lengths[:_PREVIEW_MAX_CHUNKS]

    results = [
        PreviewChunkResult(
            seq=chunk.seq,
            start=chunk.start,
            end=chunk.end,
            size_chars=rune_lengths[i],
            size_tokens_approx=approx_token_count_from_rune_len(rune_lengths[i], lang),
            context_header=chunk.context_header or None,
            content=chunk.content,
        )
        for i, chunk in enumerate(chunks)
    ]
    return PreviewChunkingData(
        selected_tier=diag.selected_tier.value,
        tier_chain=[tier.value for tier in diag.tier_chain],
        rejected=[
            PreviewTierRejection(tier=rejection.tier.value, reason=rejection.reason)
            for rejection in diag.rejected
        ],
        profile=PreviewChunkingProfile.model_validate(asdict(profile)),
        chunks=results,
        stats=stats,
    )


async def preview_chunking(
    _auth: AuthDep,
    _role: RoleViewerDep,
    body: PreviewChunkingRequest,
) -> PreviewChunkingEnvelope | JSONResponse:
    """Handle ``POST /chunker/preview`` - chunk the text and report.

    Read-only: no database writes, no embedding calls, no logging of the
    supplied text. Success answers with the preview envelope; the guard
    failures (empty text, oversized input, timeout) keep the handler's
    custom error shape and status codes.
    """
    if not body.text.strip():
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "text is empty — paste a sample to preview chunking",
            },
        )
    if len(body.text) > _PREVIEW_MAX_CHARS:
        return JSONResponse(
            status_code=413,
            content={
                "success": False,
                "error": "text exceeds preview limit",
                "limit": _PREVIEW_MAX_CHARS,
            },
        )
    loop = asyncio.get_running_loop()
    try:
        data = await asyncio.wait_for(
            loop.run_in_executor(None, build_preview_data, body.text, body.chunking_config),
            timeout=_PREVIEW_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"success": False, "error": "chunker preview timed out"},
        )
    return PreviewChunkingEnvelope(success=True, data=data)


__all__ = [
    "PreviewChunkResult",
    "PreviewChunkingData",
    "PreviewChunkingEnvelope",
    "PreviewChunkingProfile",
    "PreviewChunkingRequest",
    "PreviewChunkingStats",
    "PreviewTierRejection",
    "build_preview_data",
    "compute_chunk_size_stats",
    "preview_chunking",
]
