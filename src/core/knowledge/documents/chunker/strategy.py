"""Public entry point for adaptive chunking.

Callers invoke :func:`split` / :func:`split_parent_child` instead of the
legacy :func:`split_text` / :func:`split_text_parent_child` functions; the
strategy resolver picks a tier based on document profile and the
``SplitterConfig.strategy`` hint. The legacy entry points still exist for
backwards compatibility — this module simply layers a tier-selector on top.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.core.knowledge.documents.chunker.heading_splitter import split_by_headings
from src.core.knowledge.documents.chunker.heuristic_splitter import split_by_heuristics
from src.core.knowledge.documents.chunker.profiler import (
    DocProfile,
    StrategyTier,
    profile_document,
    select_strategy,
)
from src.core.knowledge.documents.chunker.splitter import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEPARATORS,
    ChildChunk,
    Chunk,
    ParentChildResult,
    SplitterConfig,
    split_text,
)
from src.core.knowledge.documents.chunker.tokens import LangMixed, chars_for_token_limit
from src.core.knowledge.documents.chunker.validator import validate_chunks

# Strategy values for SplitterConfig.strategy.
STRATEGY_AUTO = "auto"
STRATEGY_HEADING = "heading"
STRATEGY_HEURISTIC = "heuristic"
STRATEGY_RECURSIVE = "recursive"
STRATEGY_LEGACY = "legacy"


@dataclass(frozen=True)
class TierRejection:
    """Why a tier was rejected by the validator and the chain advanced."""

    tier: StrategyTier
    reason: str


@dataclass(frozen=True)
class Diagnostics:
    """Which tier produced the returned chunks plus the chain that was attempted.

    Captures rejected tiers and the document profile that drove tier
    selection. Useful for surfacing in a debug UI; not produced by the
    normal :func:`split` path.
    """

    selected_tier: StrategyTier = StrategyTier.LEGACY
    tier_chain: tuple[StrategyTier, ...] = ()
    rejected: tuple[TierRejection, ...] = ()
    profile: DocProfile | None = None


def split(text: str, cfg: SplitterConfig) -> list[Chunk]:
    """Chunk ``text`` using the strategy configured in ``cfg``.

    When ``cfg.strategy`` is empty or "auto" the document profiler picks the
    tier. Always returns a non-empty result on non-empty input: on tier
    failure the chain falls through to the legacy splitter, which is the
    original Tier-3 implementation.
    """
    if text == "":
        return []
    cfg = ensure_defaults(cfg)
    chain, profile = resolve_chain_with_profile(text, cfg)
    total_chars = len(text)

    last_out: list[Chunk] | None = None
    for i, tier in enumerate(chain):
        out = run_tier(tier, text, cfg, profile)
        verdict = validate_chunks(out, total_chars, cfg.chunk_size)
        if verdict.ok:
            return out
        if tier == StrategyTier.LEGACY and i == len(chain) - 1:
            last_out = out
    if last_out is not None:
        return last_out
    return split_text(text, cfg)


def split_with_diagnostics(text: str, cfg: SplitterConfig) -> tuple[list[Chunk], Diagnostics]:
    """Same as :func:`split` but also returns the diagnostic trace.

    Default selected tier is legacy so an empty trace never carries an
    unset value.
    """
    diag = Diagnostics(selected_tier=StrategyTier.LEGACY)
    if text == "":
        return [], diag
    cfg = ensure_defaults(cfg)
    chain, profile = resolve_chain_with_profile(text, cfg)
    diag = replace(diag, tier_chain=tuple(chain), profile=profile)
    total_chars = len(text)

    last_out: list[Chunk] | None = None
    last_tier = StrategyTier.LEGACY
    for i, tier in enumerate(chain):
        out = run_tier(tier, text, cfg, profile)
        verdict = validate_chunks(out, total_chars, cfg.chunk_size)
        if verdict.ok:
            return out, replace(diag, selected_tier=tier)
        rejected = (*diag.rejected, TierRejection(tier=tier, reason=verdict.reason))
        diag = replace(diag, rejected=rejected)
        if tier == StrategyTier.LEGACY and i == len(chain) - 1:
            last_out = out
            last_tier = tier
    if last_out is not None:
        return last_out, replace(diag, selected_tier=last_tier)
    # Defensive last-ditch fallback.
    return split_text(text, cfg), diag


def split_parent_child(
    text: str, parent_cfg: SplitterConfig, child_cfg: SplitterConfig
) -> ParentChildResult:
    """Strategy-aware analog of :func:`split_text_parent_child`.

    Runs the tier selector for parent splitting, then re-splits each parent
    into children with the small-chunk config. Child splitting honours
    ``child_cfg.strategy``; if it is empty/auto and a parent has its own
    internal structure, the appropriate tier picks it up so child chunks
    carry a finer-grained breadcrumb than the parent's.
    """
    if text == "":
        return ParentChildResult(parents=[], children=[])
    parent_cfg = ensure_defaults(parent_cfg)
    child_cfg = ensure_defaults(child_cfg)

    parents = split(text, parent_cfg)
    if not parents:
        return ParentChildResult(parents=[], children=[])

    new_parents: list[Chunk] = []
    children: list[ChildChunk] = []
    child_seq = 0
    for parent in parents:
        subs = split(parent.content, child_cfg)

        parent_index = -1
        if len(subs) > 1 or (len(subs) == 1 and subs[0].content != parent.content):
            parent_index = len(new_parents)
            new_parents.append(parent)
        for sub in subs:
            sub = replace(
                sub,
                seq=child_seq,
                start=sub.start + parent.start,
                end=sub.end + parent.start,
                context_header=merge_breadcrumbs(parent.context_header, sub.context_header),
            )
            children.append(ChildChunk(chunk=sub, parent_index=parent_index))
            child_seq += 1
    return ParentChildResult(parents=new_parents, children=children)


def merge_breadcrumbs(parent: str, child: str) -> str:
    """Combine parent and child breadcrumbs into a single context header.

    When the child re-runs heading detection on parent content, its first
    breadcrumb line typically duplicates the parent's last line; drop that
    duplicate so the embedding context isn't redundant.
    """
    if not parent:
        return child
    if not child:
        return parent
    parent_lines = parent.split("\n")
    child_lines = child.split("\n")
    if parent_lines and child_lines and parent_lines[-1].strip() == child_lines[0].strip():
        child_lines = child_lines[1:]
    if not child_lines:
        return parent
    return parent + "\n" + "\n".join(child_lines)


def resolve_chain_with_profile(
    text: str, cfg: SplitterConfig
) -> tuple[list[StrategyTier], DocProfile | None]:
    """Strategy chain to attempt and, for the auto strategy, the profile.

    Profile is None for explicit non-auto strategies so callers don't pay
    for an unused profiling pass.
    """
    if cfg.strategy == STRATEGY_HEADING:
        return [StrategyTier.HEADING, StrategyTier.LEGACY], None
    if cfg.strategy == STRATEGY_HEURISTIC:
        return [StrategyTier.HEURISTIC, StrategyTier.LEGACY], None
    if cfg.strategy == STRATEGY_RECURSIVE:
        # "recursive" is a public-API alias for "legacy": both invoke
        # split_text. Kept for backwards compatibility with stored configs.
        return [StrategyTier.LEGACY], None
    if cfg.strategy == STRATEGY_LEGACY or cfg.strategy == "":
        # Empty == legacy preserves backwards compatibility with stored
        # config rows that pre-date the strategy field.
        return [StrategyTier.LEGACY], None
    profile = profile_document(text)
    return select_strategy(profile), profile


def run_tier(
    tier: StrategyTier, text: str, cfg: SplitterConfig, profile: DocProfile | None
) -> list[Chunk]:
    """Dispatch the splitter implementation for the given tier."""
    if tier == StrategyTier.HEADING:
        return split_by_headings(text, cfg, profile)
    if tier == StrategyTier.HEURISTIC:
        return split_by_heuristics(text, cfg, profile)
    return split_text(text, cfg)


def ensure_defaults(cfg: SplitterConfig) -> SplitterConfig:
    """Fill zero-value config fields with sane defaults.

    When ``cfg.token_limit`` is set, ``chunk_size`` is clamped to the
    character budget that fits within that token limit (with a 10% safety
    factor), making chunks safe for embedding APIs with hard token caps.
    """
    if cfg.chunk_size <= 0:
        cfg = replace(cfg, chunk_size=DEFAULT_CHUNK_SIZE)
    if cfg.chunk_overlap <= 0:
        cfg = replace(cfg, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
    if not cfg.separators:
        cfg = replace(cfg, separators=list(DEFAULT_SEPARATORS))
    if cfg.token_limit > 0:
        lang = LangMixed
        if cfg.languages:
            lang = cfg.languages[0]
        char_budget = chars_for_token_limit(cfg.token_limit, lang)
        if char_budget > 0 and (cfg.chunk_size == 0 or char_budget < cfg.chunk_size):
            cfg = replace(cfg, chunk_size=char_budget)
    # Guard against pathological overlap configurations: if overlap exceeds
    # half of chunk_size, almost every chunk is duplicate content. Cap it so
    # overlap stays a useful smoothing band rather than a near-clone.
    if cfg.chunk_overlap > cfg.chunk_size // 2 and cfg.chunk_size > 0:
        cfg = replace(cfg, chunk_overlap=cfg.chunk_size // 2)
    return cfg


__all__ = [
    "STRATEGY_AUTO",
    "STRATEGY_HEADING",
    "STRATEGY_HEURISTIC",
    "STRATEGY_LEGACY",
    "STRATEGY_RECURSIVE",
    "Diagnostics",
    "TierRejection",
    "ensure_defaults",
    "merge_breadcrumbs",
    "resolve_chain_with_profile",
    "run_tier",
    "split",
    "split_parent_child",
    "split_with_diagnostics",
]
