"""Unit tests for the async docreader adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.common.exception import ExternalServiceError
from src.core.knowledge.documents.docreader_adapter import DocReaderAdapter
from src.core.knowledge.documents.parse_pipeline import ReadRequest


class _FakeClient:
    def __init__(self, *, markdown: str = "# hi", error: str = "", title: str = "") -> None:
        self.markdown = markdown
        self.error = error
        self.title = title
        self.closed = False
        self.requests: list[object] = []

    def read(self, request: object, *, timeout: float | None = None) -> SimpleNamespace:
        self.requests.append(request)
        metadata = {"title": self.title} if self.title else {}
        return SimpleNamespace(
            markdown_content=self.markdown,
            error=self.error,
            metadata=metadata,
        )

    def close(self) -> None:
        self.closed = True


async def test_adapter_maps_markdown_and_title() -> None:
    client = _FakeClient(markdown="# body", title="Extracted")
    adapter = DocReaderAdapter(client)  # type: ignore[arg-type]
    result = await adapter.read(
        ReadRequest(url="https://example.com/a", title="fallback", file_type="html")
    )
    assert result.markdown_content == "# body"
    assert result.title == "Extracted"
    assert result.metadata["title"] == "Extracted"


async def test_adapter_raises_on_reader_error() -> None:
    client = _FakeClient(error="parse exploded")
    adapter = DocReaderAdapter(client)  # type: ignore[arg-type]
    with pytest.raises(ExternalServiceError, match="parse exploded"):
        await adapter.read(ReadRequest(url="https://example.com/a"))


async def test_adapter_close_forwards() -> None:
    client = _FakeClient()
    adapter = DocReaderAdapter(client)  # type: ignore[arg-type]
    adapter.close()
    assert client.closed is True
