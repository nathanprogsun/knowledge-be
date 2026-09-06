"""In-process document reader (no external parser service).

Formats live in an extension-to-handler table so a new type is a
registration, not a new branch. The parse pipeline already loads
stored bytes; this reader does not fetch URLs.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Final, TypeAlias
from urllib.parse import urlparse

from src.common.exception import ExternalServiceError
from src.common.json import JsonValue
from src.core.knowledge.documents.parse_pipeline import ParseResult, ReadRequest
from src.core.system.parser_engine import BUILTIN_ENGINE_NAME, SIMPLE_ENGINE_NAME

try:
    import opendataloader_pdf as _opendataloader_pdf
except ImportError:  # optional extra; PDF path fails closed when missing
    _opendataloader_pdf = None


Handler: TypeAlias = Callable[[bytes, ReadRequest], ParseResult]

_UNSUPPORTED_TYPE: Final = "document_parse.unsupported_type"
_FAILED: Final = "document_parse.failed"
_NO_SOURCE: Final = "document_parse.no_source"
_ENGINE_UNAVAILABLE: Final = "document_parse.engine_unavailable"

_IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {"jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"}
)


def _decode_text(payload: bytes) -> str:
    """Decode document bytes as UTF-8, then latin-1 so a legacy file still parses."""
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("latin-1")


def _result(markdown: str, request: ReadRequest) -> ParseResult:
    return ParseResult(markdown_content=markdown, title=request.title)


def _read_text(payload: bytes, request: ReadRequest) -> ParseResult:
    return _result(_decode_text(payload), request)


def _read_csv(payload: bytes, request: ReadRequest) -> ParseResult:
    return _result(f"```csv\n{_decode_text(payload).rstrip()}\n```", request)


def _read_json(payload: bytes, request: ReadRequest) -> ParseResult:
    parsed: JsonValue = json.loads(_decode_text(payload))
    pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
    return _result(f"```json\n{pretty}\n```", request)


class _HTMLTextExtractor(HTMLParser):
    """Collect visible text; script/style is not document content."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._chunks.append(data)

    def text(self) -> str:
        return " ".join("".join(self._chunks).split())


def _read_html(payload: bytes, request: ReadRequest) -> ParseResult:
    extractor = _HTMLTextExtractor()
    extractor.feed(_decode_text(payload))
    extractor.close()
    return _result(extractor.text(), request)


def _read_image(_payload: bytes, request: ReadRequest) -> ParseResult:
    """Image bytes are not OCR'd; emit a markdown image line from the filename."""
    name = request.file_name.strip() or PurePosixPath(urlparse(request.url).path).name
    if not name:
        name = "image"
    return _result(f"![{name}]({name})", request)


def _read_pdf(payload: bytes, request: ReadRequest) -> ParseResult:
    """Parse PDF via OpenDataLoader hybrid when ``ODL_HYBRID_URL`` is set.

    Requires the optional ``opendataloader-pdf`` package (compose builds
    the worker with ``WITH_ODL=1`` under ``--profile odl-hybrid``).
    """
    hybrid_url = os.getenv("ODL_HYBRID_URL", "").strip()
    if not hybrid_url:
        raise ExternalServiceError(
            code=_ENGINE_UNAVAILABLE,
            message=("PDF parse requires ODL_HYBRID_URL (docker compose --profile odl-hybrid)"),
        )
    if _opendataloader_pdf is None:
        raise ExternalServiceError(
            code=_ENGINE_UNAVAILABLE,
            message=(
                "opendataloader-pdf is not installed; rebuild the worker image with WITH_ODL=1"
            ),
        )

    safe_name = PurePosixPath(request.file_name.strip() or "document.pdf").name
    if not safe_name.lower().endswith(".pdf"):
        safe_name = f"{PurePosixPath(safe_name).stem or 'document'}.pdf"
    stem = PurePosixPath(safe_name).stem

    with tempfile.TemporaryDirectory(prefix="kb-odl-") as tmp_dir:
        pdf_path = Path(tmp_dir) / safe_name
        pdf_path.write_bytes(payload)
        image_dir = Path(tmp_dir) / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        _opendataloader_pdf.convert(
            input_path=str(pdf_path),
            output_dir=tmp_dir,
            format="markdown",
            image_output="external",
            image_dir=str(image_dir),
            quiet=True,
            hybrid=os.getenv("ODL_HYBRID", "docling-fast").strip() or "docling-fast",
            hybrid_url=hybrid_url,
            hybrid_fallback=True,
        )
        md_candidates = sorted(Path(tmp_dir).rglob("*.md"))
        if not md_candidates:
            raise ExternalServiceError(
                code=_FAILED,
                message="OpenDataLoader produced no markdown",
            )
        preferred = [p for p in md_candidates if p.stem == stem or p.stem.startswith(stem)]
        md_path = preferred[0] if preferred else max(md_candidates, key=lambda p: p.stat().st_mtime)
        return _result(md_path.read_text(encoding="utf-8", errors="replace"), request)


HANDLERS: Final[Mapping[str, Handler]] = {
    "txt": _read_text,
    "md": _read_text,
    "markdown": _read_text,
    "csv": _read_csv,
    "json": _read_json,
    "html": _read_html,
    "htm": _read_html,
    "pdf": _read_pdf,
    "jpg": _read_image,
    "jpeg": _read_image,
    "png": _read_image,
    "gif": _read_image,
    "bmp": _read_image,
    "tiff": _read_image,
    "webp": _read_image,
}

# None means every registered handler is allowed for that engine.
_ENGINE_ALLOWLISTS: Final[Mapping[str, frozenset[str] | None]] = {
    BUILTIN_ENGINE_NAME: None,
    SIMPLE_ENGINE_NAME: frozenset({"txt", "md", "markdown", "csv", "json"} | _IMAGE_EXTENSIONS),
}


def _normalize_engine(raw: str) -> str:
    trimmed = raw.strip().lower()
    return trimmed if trimmed else BUILTIN_ENGINE_NAME


def _normalize_ext(raw: str) -> str:
    return raw.strip().lstrip(".").lower()


def _path_extension(path: str) -> str:
    return _normalize_ext(PurePosixPath(path).suffix)


def _resolve_extension(request: ReadRequest) -> str:
    if request.file_type.strip():
        return _normalize_ext(request.file_type)
    if request.file_name.strip():
        ext = _path_extension(request.file_name)
        if ext:
            return ext
    if request.url.strip():
        ext = _path_extension(urlparse(request.url).path)
        if ext:
            return ext
    return ""


def _unsupported(ext: str) -> ExternalServiceError:
    label = ext or "unknown"
    return ExternalServiceError(
        code=_UNSUPPORTED_TYPE,
        message=f"unsupported document type: {label}",
    )


class BuiltinDocumentReader:
    """``DocumentReader`` that parses bytes in-process via ``HANDLERS``."""

    async def read(self, request: ReadRequest) -> ParseResult:
        """Parse one document from in-memory bytes and return markdown."""
        engine = _normalize_engine(request.parser_engine)
        if engine not in _ENGINE_ALLOWLISTS:
            raise ExternalServiceError(
                code=_ENGINE_UNAVAILABLE,
                message=f"parser engine is not available in-process: {engine}",
            )
        ext = _resolve_extension(request)
        handler = HANDLERS.get(ext)
        if handler is None:
            raise _unsupported(ext)
        allowlist = _ENGINE_ALLOWLISTS[engine]
        if allowlist is not None and ext not in allowlist:
            raise _unsupported(ext)
        payload = request.file_content
        if payload is None:
            raise ExternalServiceError(
                code=_NO_SOURCE,
                message="document read requires file content",
            )
        try:
            return await asyncio.to_thread(handler, payload, request)
        except ExternalServiceError:
            raise
        except Exception as exc:
            raise ExternalServiceError(
                code=_FAILED,
                message=f"document parse failed: {exc}",
            ) from exc


__all__ = [
    "HANDLERS",
    "BuiltinDocumentReader",
]
