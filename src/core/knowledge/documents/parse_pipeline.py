"""Parse stage of the document-processing pipeline.

Turns a stored file (or a URL) into parsed markdown through an injectable
document-reader seam. The real docreader gRPC client is not wired yet —
the worker layer composes it later — so this module only defines the seam
contract (``DocumentReader`` / ``FileReader``) and the small read flow
around it: load file bytes through the file-reader seam when needed,
forward the read request, and surface the parsed document plus extraction
metadata.

``parse_document`` raises ``ExternalServiceError`` for a request without
any content source; a reader that cannot serve the document should raise
the same way, so the orchestrator can map every parse failure onto the
document's failed state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

from src.common.exception import ExternalServiceError
from src.common.json import JsonObject

#: Code for a read request that carries no resolvable content source.
_NO_SOURCE_CODE = "document_parse.no_source"


@dataclass(frozen=True)
class ReadRequest:
    """Input of one document-read call (mirrors the upstream read request).

    Either ``file_content`` is supplied in memory, or ``file_path`` names
    a stored object the file-reader seam can load, or ``url`` carries the
    remote document. ``file_name`` / ``file_type`` describe the source for
    the parser; ``parser_engine`` pins an engine when one is configured.
    """

    file_content: bytes | None = None
    file_path: str = ""
    file_name: str = ""
    file_type: str = ""
    url: str = ""
    parser_engine: str = ""
    title: str = ""
    request_id: str = ""


@dataclass(frozen=True)
class ParseResult:
    """Output of the parse stage."""

    markdown_content: str
    title: str = ""
    metadata: JsonObject = field(default_factory=dict)
    is_audio: bool = False


@runtime_checkable
class DocumentReader(Protocol):
    """Document-parsing seam (satisfied by the docreader client).

    ``read`` returns the parsed markdown document or raises on failure.
    A parser-level error (bad file, unsupported format, timeout) surfaces
    as an ``ExternalServiceError`` so callers translate it uniformly.
    """

    async def read(self, request: ReadRequest) -> ParseResult:
        """Parse one document and return its markdown content."""
        ...


@runtime_checkable
class FileReader(Protocol):
    """Storage seam that loads a stored object's bytes by path."""

    async def read_file(self, *, file_path: str) -> bytes:
        """Return the raw bytes of the stored object at ``file_path``."""
        ...


async def parse_document(
    *,
    reader: DocumentReader,
    request: ReadRequest,
    file_reader: FileReader | None = None,
) -> ParseResult:
    """Run the parse stage for one document.

    When the request has no in-memory content, the bytes are loaded from
    ``file_path`` through ``file_reader`` (a URL request needs no file
    read). The loaded request is then forwarded to ``reader`` and its
    result returned.
    """
    req = request
    if req.file_content is None:
        if req.url:
            pass
        elif req.file_path and file_reader is not None:
            content = await file_reader.read_file(file_path=req.file_path)
            req = replace(req, file_content=content)
        else:
            raise ExternalServiceError(
                code=_NO_SOURCE_CODE,
                message="document read requires file content, a stored file, or a url",
            )
    return await reader.read(req)


__all__ = [
    "DocumentReader",
    "FileReader",
    "ParseResult",
    "ReadRequest",
    "parse_document",
]
