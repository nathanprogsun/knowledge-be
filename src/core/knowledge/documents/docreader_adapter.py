"""Adapt the sync docreader gRPC client to the async parse seam."""

from __future__ import annotations

import asyncio

from src.ai.docreader.client import DocReaderClient
from src.ai.docreader.proto import docreader_pb2 as pb2
from src.common.exception import ExternalServiceError
from src.common.json import JsonObject
from src.core.knowledge.documents.parse_pipeline import ParseResult, ReadRequest

_READ_FAILED_CODE = "docreader.read_failed"


class DocReaderAdapter:
    """``DocumentReader`` over a process-wide ``DocReaderClient``."""

    def __init__(self, client: DocReaderClient) -> None:
        self._client = client

    async def read(self, request: ReadRequest) -> ParseResult:
        """Parse one document on a worker thread and map the protobuf result."""
        proto_request = pb2.ReadRequest(
            file_content=request.file_content or b"",
            file_name=request.file_name,
            file_type=request.file_type,
            url=request.url,
            title=request.title,
            request_id=request.request_id,
            config=pb2.ReadConfig(parser_engine=request.parser_engine),
        )
        response = await asyncio.to_thread(self._client.read, proto_request)
        if response.error:
            raise ExternalServiceError(
                code=_READ_FAILED_CODE,
                message=response.error,
            )
        metadata: JsonObject = dict(response.metadata)
        title = request.title
        meta_title = metadata.get("title")
        if isinstance(meta_title, str) and meta_title:
            title = meta_title
        return ParseResult(
            markdown_content=response.markdown_content,
            title=title,
            metadata=metadata,
        )

    def close(self) -> None:
        """Release the underlying gRPC channel."""
        self._client.close()


__all__ = ["DocReaderAdapter"]
