"""Chunk stage of the document-processing pipeline.

Splits parsed markdown into chunks using the merged adaptive text
chunker. Strategy selection, size/overlap defaults and the token budget
are resolved from the knowledge base's chunking config, mirroring the
upstream config-to-splitter mapping; parent-child chunking is honoured
when ``enable_parent_child`` is set.

``resolve_splitter_config`` maps a KB chunking-config blob onto the
chunker's ``SplitterConfig``. Zero-value fields are deliberately left for
the chunker's own defaulting (``ensure_defaults``), so an empty config
behaves identically to a direct chunker call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common.json import JsonObject, JsonValue
from src.core.knowledge.documents.chunker.splitter import Chunk, SplitterConfig
from src.core.knowledge.documents.chunker.strategy import split, split_parent_child

#: Parent-chunk size fallback when ``parent_chunk_size`` is unset.
_DEFAULT_PARENT_CHUNK_SIZE = 4096
#: Child-chunk size fallback when ``child_chunk_size`` is unset.
_DEFAULT_CHILD_CHUNK_SIZE = 384
#: Child overlap is ~20% of the child chunk size.
_CHILD_OVERLAP_DIVISOR = 5


@dataclass(frozen=True)
class ParsedChunk:
    """A split unit with its source position and optional parent link.

    ``parent_index`` indexes into the parent list produced by
    parent-child chunking; it is ``-1`` for a flat chunk or a standalone
    child.
    """

    content: str
    context_header: str = ""
    seq: int = 0
    start: int = 0
    end: int = 0
    parent_index: int = -1


@dataclass(frozen=True)
class ChunkingResult:
    """Chunk-stage output: the retrieval units plus optional parent chunks."""

    chunks: list[ParsedChunk]
    parent_chunks: list[ParsedChunk] = field(default_factory=list)
    is_parent_child: bool = False


def _as_int(value: JsonValue) -> int:
    return value if isinstance(value, int) else 0


def _as_str(value: JsonValue) -> str:
    return value if isinstance(value, str) else ""


def _as_str_list(value: JsonValue) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _as_bool(value: JsonValue) -> bool:
    return value if isinstance(value, bool) else False


def resolve_splitter_config(chunking_config: JsonObject | None) -> SplitterConfig:
    """Map a KB chunking-config blob onto the chunker ``SplitterConfig``."""
    config = chunking_config if isinstance(chunking_config, dict) else {}
    return SplitterConfig(
        chunk_size=_as_int(config.get("chunk_size")),
        chunk_overlap=_as_int(config.get("chunk_overlap")),
        separators=_as_str_list(config.get("separators")),
        strategy=_as_str(config.get("strategy")),
        token_limit=_as_int(config.get("token_limit")),
        languages=_as_str_list(config.get("languages")),
    )


def resolve_parent_child_configs(
    chunking_config: JsonObject | None,
) -> tuple[SplitterConfig, SplitterConfig]:
    """Derive parent and child splitter configs for parent-child chunking.

    The base config supplies separators and the splitting strategy; the
    parent and child sizes fall back to their defaults when unset. Child
    overlap is a fifth of the child chunk size.
    """
    config = chunking_config if isinstance(chunking_config, dict) else {}
    base = resolve_splitter_config(chunking_config)
    parent_size = _as_int(config.get("parent_chunk_size"))
    if parent_size <= 0:
        parent_size = _DEFAULT_PARENT_CHUNK_SIZE
    child_size = _as_int(config.get("child_chunk_size"))
    if child_size <= 0:
        child_size = _DEFAULT_CHILD_CHUNK_SIZE
    parent = SplitterConfig(
        chunk_size=parent_size,
        chunk_overlap=base.chunk_overlap,
        separators=base.separators,
        strategy=base.strategy,
        token_limit=base.token_limit,
        languages=base.languages,
    )
    child = SplitterConfig(
        chunk_size=child_size,
        chunk_overlap=child_size // _CHILD_OVERLAP_DIVISOR,
        separators=base.separators,
        strategy=base.strategy,
        token_limit=base.token_limit,
        languages=base.languages,
    )
    return parent, child


def _from_chunker(chunk: Chunk) -> ParsedChunk:
    return ParsedChunk(
        content=chunk.content,
        context_header=chunk.context_header,
        seq=chunk.seq,
        start=chunk.start,
        end=chunk.end,
    )


def chunk_markdown(markdown: str, chunking_config: JsonObject | None) -> ChunkingResult:
    """Split parsed markdown into chunks using the configured strategy.

    Parent-child mode is used when the config enables it: large parent
    chunks provide context while small children feed embedding/retrieval.
    Otherwise the strategy-aware ``split`` entry point runs (auto /
    heading / heuristic / legacy tiers).
    """
    config = chunking_config if isinstance(chunking_config, dict) else {}
    if _as_bool(config.get("enable_parent_child")):
        parent_cfg, child_cfg = resolve_parent_child_configs(chunking_config)
        result = split_parent_child(markdown, parent_cfg, child_cfg)
        parents = [_from_chunker(parent) for parent in result.parents]
        chunks = [
            ParsedChunk(
                content=child.chunk.content,
                context_header=child.chunk.context_header,
                seq=child.chunk.seq,
                start=child.chunk.start,
                end=child.chunk.end,
                parent_index=child.parent_index,
            )
            for child in result.children
        ]
        return ChunkingResult(
            chunks=chunks,
            parent_chunks=parents,
            is_parent_child=True,
        )
    cfg = resolve_splitter_config(chunking_config)
    return ChunkingResult(chunks=[_from_chunker(chunk) for chunk in split(markdown, cfg)])


__all__ = [
    "ChunkingResult",
    "ParsedChunk",
    "chunk_markdown",
    "resolve_parent_child_configs",
    "resolve_splitter_config",
]
