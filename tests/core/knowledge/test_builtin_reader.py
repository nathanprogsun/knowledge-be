"""Unit tests for the in-process builtin document reader."""

from __future__ import annotations

import pytest

from src.common.exception import ExternalServiceError
from src.core.knowledge.documents.builtin_reader import BuiltinDocumentReader
from src.core.knowledge.documents.parse_pipeline import ReadRequest
from src.core.system.parser_engine import SIMPLE_ENGINE_NAME


def _reader() -> BuiltinDocumentReader:
    return BuiltinDocumentReader()


@pytest.mark.parametrize(
    ("file_type", "payload", "needle"),
    [
        ("md", b"# Hello\n\nbody", "# Hello"),
        ("txt", b"plain text", "plain text"),
        ("json", b'{"a":1}', "```json"),
        ("csv", b"a,b\n1,2", "```csv"),
        ("html", b"<html><body><p>Hi <b>there</b></p></body></html>", "Hi there"),
    ],
)
async def test_phase1_types_return_markdown(
    file_type: str,
    payload: bytes,
    needle: str,
) -> None:
    result = await _reader().read(ReadRequest(file_content=payload, file_type=file_type))

    assert result.markdown_content
    assert needle in result.markdown_content


async def test_unsupported_extension_raises_unsupported_type() -> None:
    with pytest.raises(ExternalServiceError) as exc_info:
        await _reader().read(ReadRequest(file_content=b"%PDF", file_type="pdf"))

    assert exc_info.value.code == "document_parse.unsupported_type"


async def test_simple_engine_rejects_html() -> None:
    with pytest.raises(ExternalServiceError) as exc_info:
        await _reader().read(
            ReadRequest(
                file_content=b"<p>nope</p>",
                file_type="html",
                parser_engine=SIMPLE_ENGINE_NAME,
            )
        )

    assert exc_info.value.code == "document_parse.unsupported_type"


async def test_remote_engine_is_unavailable_in_process() -> None:
    with pytest.raises(ExternalServiceError) as exc_info:
        await _reader().read(
            ReadRequest(
                file_content=b"# hi",
                file_type="md",
                parser_engine="mineru",
            )
        )

    assert exc_info.value.code == "document_parse.engine_unavailable"


async def test_builtin_parses_without_grpc() -> None:
    result = await _reader().read(ReadRequest(file_content=b"hello", file_type="txt"))

    assert result.markdown_content == "hello"
