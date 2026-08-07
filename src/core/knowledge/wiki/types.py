"""Domain vocabulary and validation for the wiki domain.

The wire shapes live in the frozen contracts; this module holds the
domain-level constants and the pure helpers shared by the wiki service,
ingest pipeline, and taxonomy layers: the page type / status / edit
source vocabularies, and the category-path and page-type normalisation
used identically on the write side (what a page is stored with) and the
read side (what a list / filter query is matched against).

``WikiPageListFilter`` is the domain-side query input for paged page
listings; the persistence layer takes the individual, already-normalised
values.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# ── Page types ───────────────────────────────────────────────────────

WIKI_PAGE_TYPE_SUMMARY = "summary"
WIKI_PAGE_TYPE_ENTITY = "entity"
WIKI_PAGE_TYPE_CONCEPT = "concept"
WIKI_PAGE_TYPE_INDEX = "index"
WIKI_PAGE_TYPE_SYNTHESIS = "synthesis"
WIKI_PAGE_TYPE_COMPARISON = "comparison"

_WIKI_PAGE_TYPES: frozenset[str] = frozenset(
    {
        WIKI_PAGE_TYPE_SUMMARY,
        WIKI_PAGE_TYPE_ENTITY,
        WIKI_PAGE_TYPE_CONCEPT,
        WIKI_PAGE_TYPE_INDEX,
        WIKI_PAGE_TYPE_SYNTHESIS,
        WIKI_PAGE_TYPE_COMPARISON,
    }
)

# ── Page statuses ────────────────────────────────────────────────────

WIKI_PAGE_STATUS_DRAFT = "draft"
WIKI_PAGE_STATUS_PUBLISHED = "published"
WIKI_PAGE_STATUS_ARCHIVED = "archived"

_WIKI_PAGE_STATUSES: frozenset[str] = frozenset(
    {WIKI_PAGE_STATUS_DRAFT, WIKI_PAGE_STATUS_PUBLISHED, WIKI_PAGE_STATUS_ARCHIVED}
)

# ── Edit sources ─────────────────────────────────────────────────────

WIKI_EDIT_SOURCE_PIPELINE = "pipeline"
WIKI_EDIT_SOURCE_AGENT = "agent"
WIKI_EDIT_SOURCE_USER = "user"
WIKI_EDIT_SOURCE_REVERT = "revert"

_WIKI_EDIT_SOURCES: frozenset[str] = frozenset(
    {
        WIKI_EDIT_SOURCE_PIPELINE,
        WIKI_EDIT_SOURCE_AGENT,
        WIKI_EDIT_SOURCE_USER,
        WIKI_EDIT_SOURCE_REVERT,
    }
)

# ── Snapshot retention (wiki_page_revisions) ─────────────────────────
# Two-tier: machine-authored snapshots may be pruned below the soft cap;
# everything else survives until the hard cap.

# How many recent versions are kept for prunable (machine-authored)
# snapshots.
WIKI_MAX_REVISIONS_PER_PAGE = 50
# Absolute per-page ceiling applied regardless of author.
WIKI_MAX_REVISIONS_HARD_CAP = 200

# Edit sources whose snapshots may be dropped by the soft cap. Legacy rows
# carry an empty source, hence the "" entry.
WIKI_PRUNABLE_EDIT_SOURCES: tuple[str, ...] = ("", WIKI_EDIT_SOURCE_PIPELINE)

# ── Category path bounds ─────────────────────────────────────────────

# Hard cap on how many folder levels a page's category path may keep.
# The ingest prompts ask the model for at most 2 levels; this storage cap
# is one level deeper as a defensive bound so an over-eager model cannot
# create an unbounded breadcrumb.
WIKI_CATEGORY_MAX_DEPTH = 3

# Sentinel folder / parent id meaning "the wiki root" (a page or folder
# directly under the top level, with no parent folder).
WIKI_FOLDER_ROOT_ID = ""

# Category labels that are structural noise rather than real folder
# names ("entity" / "concept" / "summary" / page-type words and their
# localised forms).
_WIKI_TYPE_CATEGORY_LABELS: frozenset[str] = frozenset(
    {
        "entity",
        "实体",
        "實體",
        "concept",
        "概念",
        "summary",
        "摘要",
        "wiki",
        "页面",
        "頁面",
    }
)

# Characters stripped from the ends of a raw category label.
_WIKI_CATEGORY_TRIM_CHARS = "\"'“”‘’[]（）()"  # noqa: RUF001


def is_valid_page_type(page_type: str) -> bool:
    """Return whether ``page_type`` is one of the known page types.

    Unknown values would silently disappear from the type-filtered
    listings the browser is built on, so write paths reject them.
    """
    return page_type in _WIKI_PAGE_TYPES


def is_valid_page_status(status: str) -> bool:
    """Return whether ``status`` is one of the known page statuses."""
    return status in _WIKI_PAGE_STATUSES


def normalize_edit_source(source: str) -> str:
    """Map unknown / empty edit sources to the pipeline source.

    Legacy rows and forgotten call sites degrade to the historical
    behaviour ("the machine wrote this").
    """
    if source in _WIKI_EDIT_SOURCES:
        return source
    return WIKI_EDIT_SOURCE_PIPELINE


def split_page_types(raw: str) -> list[str] | None:
    """Parse a comma-separated page_type value into deduplicated parts.

    Each part is trimmed; blanks are dropped and duplicates keep their
    first occurrence. An empty / whitespace-only input yields ``None``
    ("no filter"). Shared by the handler (query parsing) and the page
    listing so the two layers split identically.
    """
    if raw.strip() == "":
        return None
    seen: set[str] = set()
    out: list[str] = []
    for part in raw.split(","):
        part = part.strip()
        if part and part not in seen:
            seen.add(part)
            out.append(part)
    return out


def _is_type_category_label(label: str) -> bool:
    """Return whether ``label`` is a page-type noise label.

    Case-insensitive and tolerant of a trailing "s" (plural), mirroring
    how the shared vocabulary is compared.
    """
    normalized = label.strip().lower()
    if normalized.endswith("s"):
        normalized = normalized[:-1]
    return normalized in _WIKI_TYPE_CATEGORY_LABELS


def clean_category_part(part: str) -> list[str]:
    """Normalise one raw category label into cleaned sub-labels.

    Replaces fullwidth / pipe separators with "/", strips wrapping
    quotes and brackets, and drops blank or page-type labels. An empty
    input yields ``[]``.
    """
    part = part.strip()
    if part == "":
        return []
    part = part.replace("／", "/").replace("｜", "/").replace("|", "/")  # noqa: RUF001
    cleaned: list[str] = []
    for raw in part.split("/"):
        label = raw.strip().strip(_WIKI_CATEGORY_TRIM_CHARS).strip()
        if label == "" or _is_type_category_label(label):
            continue
        cleaned.append(label)
    return cleaned


def clean_category_path(parts: list[str]) -> list[str]:
    """Clean, deduplicate, and cap a full category path at the max depth.

    Centralising this guarantees the path a page is stored with and the
    path a list / filter query is matched against go through the exact
    same normalisation, so directory filters cannot silently drift.
    """
    cleaned: list[str] = []
    for part in parts:
        for label in clean_category_part(part):
            if label in cleaned:
                continue
            cleaned.append(label)
            if len(cleaned) >= WIKI_CATEGORY_MAX_DEPTH:
                return cleaned
    return cleaned


class WikiPageListFilter(BaseModel):
    """Domain-side filter for a paged wiki page listing.

    ``page_type`` / ``category_path`` are expected pre-normalised by the
    caller (see :func:`split_page_types` and :func:`clean_category_path`).
    """

    model_config = ConfigDict(frozen=True)

    knowledge_base_id: str
    page_type: str = ""
    status: str = ""
    query: str = ""
    folder_id: str | None = None
    category_path: list[str] = Field(default_factory=list)
    category_depth: int | None = None
    page: int = 1
    page_size: int = 20
    sort_by: str = ""
    sort_order: str = "desc"


__all__ = [
    "WIKI_CATEGORY_MAX_DEPTH",
    "WIKI_EDIT_SOURCE_AGENT",
    "WIKI_EDIT_SOURCE_PIPELINE",
    "WIKI_EDIT_SOURCE_REVERT",
    "WIKI_EDIT_SOURCE_USER",
    "WIKI_FOLDER_ROOT_ID",
    "WIKI_MAX_REVISIONS_HARD_CAP",
    "WIKI_MAX_REVISIONS_PER_PAGE",
    "WIKI_PAGE_STATUS_ARCHIVED",
    "WIKI_PAGE_STATUS_DRAFT",
    "WIKI_PAGE_STATUS_PUBLISHED",
    "WIKI_PAGE_TYPE_COMPARISON",
    "WIKI_PAGE_TYPE_CONCEPT",
    "WIKI_PAGE_TYPE_ENTITY",
    "WIKI_PAGE_TYPE_INDEX",
    "WIKI_PAGE_TYPE_SUMMARY",
    "WIKI_PAGE_TYPE_SYNTHESIS",
    "WIKI_PRUNABLE_EDIT_SOURCES",
    "WikiPageListFilter",
    "clean_category_part",
    "clean_category_path",
    "is_valid_page_status",
    "is_valid_page_type",
    "normalize_edit_source",
    "split_page_types",
]
