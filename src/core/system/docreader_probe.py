"""Live docreader reachability probe for the parser-engine registry."""

from __future__ import annotations

import asyncio

from src.ai.docreader.client import new_client
from src.ai.docreader.proto import docreader_pb2 as pb2

_DEFAULT_TIMEOUT_SECONDS = 2.0


async def probe_docreader(addr: str, *, timeout: float = _DEFAULT_TIMEOUT_SECONDS) -> bool:
    """Return whether ``addr`` answers ``ListEngines`` within ``timeout``."""
    trimmed = addr.strip()
    if not trimmed:
        return False
    client = new_client(trimmed)
    try:
        await asyncio.to_thread(
            client.list_engines,
            pb2.ListEnginesRequest(),
            timeout=timeout,
        )
    except Exception:
        return False
    finally:
        client.close()
    return True


__all__ = ["probe_docreader"]
