"""Parser-engine registry — local engines + remote docreader merge.

Ports ``internal/infrastructure/docparser/engine_registry.go``. The Go
registry is populated by ``init()`` calls to ``RegisterEngine``; the port
uses an explicit module-level tuple of frozen specs instead, so the
engine set is inspectable without import side effects.

``ParserEngineInfo`` mirrors ``internal/types/docparser.go::ParserEngineInfo``.
That struct carries no ``json`` tags, so Go marshals it with PascalCase
field names — the serialization aliases here reproduce that wire shape
while the Python attribute names stay snake_case.

Availability is resolved from configuration presence only. Go's
``CheckAvailable`` additionally live-probes the self-hosted / cloud
endpoints (``PingMinerU``, ``PingMinerUCloud``, ``PingPaddleOCRVL``,
``PingPaddleOCRVLCloud``) once the relevant override is non-empty. Those
probes are HTTP calls into the docparser infrastructure and land with it;
until then a configured engine reports ``available=True`` without the
reachability confirmation. An engine whose config is missing reports the
same reason string Go emits, so the UI copy already matches.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

# ── Engine names (Go exports the first two as consts) ────────────────

BUILTIN_ENGINE_NAME: Final = "builtin"
SIMPLE_ENGINE_NAME: Final = "simple"
WEKNORA_CLOUD_ENGINE_NAME: Final = "weknoracloud"
MINERU_ENGINE_NAME: Final = "mineru"
MINERU_CLOUD_ENGINE_NAME: Final = "mineru_cloud"
PADDLEOCR_VL_ENGINE_NAME: Final = "paddleocr_vl"
PADDLEOCR_VL_CLOUD_ENGINE_NAME: Final = "paddleocr_vl_cloud"

# ── Override keys read from ``ParserEngineConfig.ToOverridesMap()`` ──

WEKNORA_CLOUD_APP_ID_OVERRIDE: Final = "weknoracloud_app_id"
MINERU_ENDPOINT_OVERRIDE: Final = "mineru_endpoint"
MINERU_API_KEY_OVERRIDE: Final = "mineru_api_key"
PADDLEOCR_VL_ENDPOINT_OVERRIDE: Final = "paddleocr_vl_endpoint"
PADDLEOCR_VL_CLOUD_TOKEN_OVERRIDE: Final = "paddleocr_vl_cloud_token"

# ── File-type sets ──────────────────────────────────────────────────

_BUILTIN_FILE_TYPES: Final[tuple[str, ...]] = (
    "docx",
    "doc",
    "pdf",
    "md",
    "markdown",
    "xlsx",
    "xls",
    "epub",
    "html",
    "htm",
    "mhtml",
    "jpg",
    "jpeg",
    "png",
    "gif",
    "bmp",
    "tiff",
    "webp",
    "mp3",
    "wav",
    "m4a",
    "flac",
    "ogg",
)

_SIMPLE_FILE_TYPES: Final[tuple[str, ...]] = (
    "md",
    "markdown",
    "txt",
    "csv",
    "json",
    "jpg",
    "jpeg",
    "png",
    "gif",
    "bmp",
    "tiff",
    "webp",
    "mp3",
    "wav",
    "m4a",
    "flac",
    "ogg",
)

_WEKNORA_CLOUD_FILE_TYPES: Final[tuple[str, ...]] = (
    "docx",
    "doc",
    "pdf",
    "md",
    "markdown",
    "xlsx",
    "xls",
    "pptx",
    "ppt",
)

_MINERU_FILE_TYPES: Final[tuple[str, ...]] = (
    "pdf",
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "tiff",
    "doc",
    "docx",
    "ppt",
    "pptx",
)

_PADDLEOCR_VL_FILE_TYPES: Final[tuple[str, ...]] = (
    "pdf",
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "tiff",
)

# ── Unavailable reasons (verbatim from Go) ──────────────────────────

_DOCREADER_DISCONNECTED_REASON: Final = "DocReader service not connected"
_WEKNORA_CLOUD_UNCONFIGURED_REASON: Final = (
    "WeKnora Cloud credentials not configured. Go to Settings → WeKnora Cloud to set up."
)
_MINERU_UNCONFIGURED_REASON: Final = "MinerU service not configured"
_MINERU_CLOUD_UNCONFIGURED_REASON: Final = "MinerU API Key not configured"
_PADDLEOCR_VL_UNCONFIGURED_REASON: Final = "PaddleOCR-VL service not configured"
_PADDLEOCR_VL_CLOUD_UNCONFIGURED_REASON: Final = "PaddleOCR-VL Cloud Token not configured"


class ParserEngineInfo(BaseModel):
    """One registered parser engine (``types.ParserEngineInfo``).

    The Go struct has no ``json`` tags, so its wire keys are the
    PascalCase field names; the aliases reproduce that exactly.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    name: str = Field(serialization_alias="Name")
    description: str = Field(serialization_alias="Description")
    file_types: list[str] = Field(default_factory=list, serialization_alias="FileTypes")
    available: bool = Field(default=False, serialization_alias="Available")
    unavailable_reason: str = Field(default="", serialization_alias="UnavailableReason")


@dataclass(frozen=True, slots=True)
class ParserEngineSpec:
    """Static registration for one local engine (``EngineRegistration``).

    ``required_override`` is the ``overrides`` key that must be non-blank
    for the engine to be usable; ``None`` means the engine needs no
    configuration. ``requires_docreader`` marks engines served by the
    remote DocReader process.
    """

    name: str
    description: str
    file_types: tuple[str, ...]
    required_override: str | None = None
    unconfigured_reason: str = ""
    requires_docreader: bool = False


# Registration order defines the response order, matching Go's sequence
# of ``RegisterEngine`` calls in ``init()``.
LOCAL_PARSER_ENGINES: Final[tuple[ParserEngineSpec, ...]] = (
    ParserEngineSpec(
        name=BUILTIN_ENGINE_NAME,
        description="DocReader built-in parser engine",
        file_types=_BUILTIN_FILE_TYPES,
        unconfigured_reason=_DOCREADER_DISCONNECTED_REASON,
        requires_docreader=True,
    ),
    ParserEngineSpec(
        name=SIMPLE_ENGINE_NAME,
        description="Simple format & image parsing (no external service required)",
        file_types=_SIMPLE_FILE_TYPES,
    ),
    ParserEngineSpec(
        name=WEKNORA_CLOUD_ENGINE_NAME,
        description="WeKnoraCloud document reader",
        file_types=_WEKNORA_CLOUD_FILE_TYPES,
        required_override=WEKNORA_CLOUD_APP_ID_OVERRIDE,
        unconfigured_reason=_WEKNORA_CLOUD_UNCONFIGURED_REASON,
    ),
    ParserEngineSpec(
        name=MINERU_ENGINE_NAME,
        description="MinerU self-hosted service",
        file_types=_MINERU_FILE_TYPES,
        required_override=MINERU_ENDPOINT_OVERRIDE,
        unconfigured_reason=_MINERU_UNCONFIGURED_REASON,
    ),
    ParserEngineSpec(
        name=MINERU_CLOUD_ENGINE_NAME,
        description="MinerU Cloud API",
        file_types=_MINERU_FILE_TYPES,
        required_override=MINERU_API_KEY_OVERRIDE,
        unconfigured_reason=_MINERU_CLOUD_UNCONFIGURED_REASON,
    ),
    ParserEngineSpec(
        name=PADDLEOCR_VL_ENGINE_NAME,
        description="PaddleOCR-VL self-hosted service",
        file_types=_PADDLEOCR_VL_FILE_TYPES,
        required_override=PADDLEOCR_VL_ENDPOINT_OVERRIDE,
        unconfigured_reason=_PADDLEOCR_VL_UNCONFIGURED_REASON,
    ),
    ParserEngineSpec(
        name=PADDLEOCR_VL_CLOUD_ENGINE_NAME,
        description="PaddleOCR-VL Cloud API",
        file_types=_PADDLEOCR_VL_FILE_TYPES,
        required_override=PADDLEOCR_VL_CLOUD_TOKEN_OVERRIDE,
        unconfigured_reason=_PADDLEOCR_VL_CLOUD_UNCONFIGURED_REASON,
    ),
)


def local_engine_names() -> list[str]:
    """Return the locally registered engine names in registration order."""
    return [spec.name for spec in LOCAL_PARSER_ENGINES]


def check_engine_available(
    spec: ParserEngineSpec,
    *,
    docreader_connected: bool,
    overrides: Mapping[str, str] | None = None,
) -> tuple[bool, str]:
    """Resolve ``(available, unavailable_reason)`` for one engine.

    Mirrors each engine's ``CheckAvailable``: DocReader-backed engines
    need a live connection; override-backed engines need their key
    present and non-blank; the rest are always available.
    """
    if spec.requires_docreader:
        if docreader_connected:
            return True, ""
        return False, spec.unconfigured_reason
    if spec.required_override is None:
        return True, ""
    value = (overrides or {}).get(spec.required_override, "").strip()
    if not value:
        return False, spec.unconfigured_reason
    return True, ""


def list_all_engines(
    *,
    docreader_connected: bool = False,
    overrides: Mapping[str, str] | None = None,
    remote_engines: Sequence[ParserEngineInfo] | None = None,
) -> list[ParserEngineInfo]:
    """Merge the local registry with engines discovered from docreader.

    Mirrors ``ListAllEngines``:

    - every local engine is included, with Go-side availability checks;
    - a remote engine sharing a local name overrides that entry's
      ``file_types`` (when non-empty) and ``description`` (when
      non-blank) — the remote service is authoritative for its own
      capabilities;
    - remote-only engines are appended verbatim, so a newly added
      docreader engine appears without a code change here.
    """
    remote_by_name = {engine.name: engine for engine in (remote_engines or [])}
    merged: list[ParserEngineInfo] = []

    for spec in LOCAL_PARSER_ENGINES:
        file_types = list(spec.file_types)
        description = spec.description
        remote = remote_by_name.get(spec.name)
        if remote is not None:
            if remote.file_types:
                file_types = list(remote.file_types)
            if remote.description:
                description = remote.description
        available, reason = check_engine_available(
            spec,
            docreader_connected=docreader_connected,
            overrides=overrides,
        )
        merged.append(
            ParserEngineInfo(
                name=spec.name,
                description=description,
                file_types=file_types,
                available=available,
                unavailable_reason=reason,
            )
        )

    local_names = {spec.name for spec in LOCAL_PARSER_ENGINES}
    merged.extend(engine for engine in (remote_engines or []) if engine.name not in local_names)
    return merged


__all__ = [
    "BUILTIN_ENGINE_NAME",
    "LOCAL_PARSER_ENGINES",
    "MINERU_API_KEY_OVERRIDE",
    "MINERU_CLOUD_ENGINE_NAME",
    "MINERU_ENDPOINT_OVERRIDE",
    "MINERU_ENGINE_NAME",
    "PADDLEOCR_VL_CLOUD_ENGINE_NAME",
    "PADDLEOCR_VL_CLOUD_TOKEN_OVERRIDE",
    "PADDLEOCR_VL_ENDPOINT_OVERRIDE",
    "PADDLEOCR_VL_ENGINE_NAME",
    "SIMPLE_ENGINE_NAME",
    "WEKNORA_CLOUD_APP_ID_OVERRIDE",
    "WEKNORA_CLOUD_ENGINE_NAME",
    "ParserEngineInfo",
    "ParserEngineSpec",
    "check_engine_available",
    "list_all_engines",
    "local_engine_names",
]
