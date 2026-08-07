"""Image reference resolution for multimodal LLM requests.

``data:`` URIs and ``http(s)://`` URLs pass through unchanged. Stored image
references (``resource://`` and storage-backed ``local://`` paths) are read
through the application resolver and converted to base64 data URIs for the
OpenAI-compatible path, or to raw bytes for the local Ollama path.
"""

from __future__ import annotations

import base64
import binascii
import mimetypes
import os
from collections.abc import Callable

from src.ai.llm.types import Message

#: Resolves a ``resource://`` / ``local://`` / ``storage://`` URL to bytes
#: using the owning tenant's storage config. Set by the application layer at
#: startup; when unset (e.g. in tests) callers fall back to the env-based
#: ``LOCAL_STORAGE_BASE_DIR`` resolution below.
local_image_resolver: Callable[[str], tuple[bytes, bool]] | None = None

_STORAGE_PREFIXES = ("resource://", "local://", "storage://")
_LOCAL_BASE_DIR_ENV = "LOCAL_STORAGE_BASE_DIR"
_DEFAULT_BASE_DIR = "/data/files"

# ``http.DetectContentType`` equivalent for the image formats LLM APIs accept.
_MAGIC_JPEG = b"\xff\xd8\xff"
_MAGIC_PNG = b"\x89PNG\r\n\x1a\n"
_MAGIC_GIF = b"GIF87a"
_MAGIC_GIF89 = b"GIF89a"
_MAGIC_WEBP = b"RIFF"


def is_application_stored_image(image_url: str) -> bool:
    """True for resolver-backed image references."""
    return image_url.startswith(_STORAGE_PREFIXES)


def read_local_storage_bytes(storage_path: str) -> bytes | None:
    """Resolve a stored image reference to bytes, or ``None``."""
    if local_image_resolver is not None:
        data, ok = local_image_resolver(storage_path)
        if ok:
            return data
    rel_path = storage_path.removeprefix("local://")
    base_dir = os.environ.get(_LOCAL_BASE_DIR_ENV) or _DEFAULT_BASE_DIR
    local_path = os.path.join(base_dir, rel_path.replace("/", os.sep))
    try:
        with open(local_path, "rb") as handle:
            return handle.read()
    except OSError:
        return None


def _detect_content_type(data: bytes, image_url: str) -> str:
    """Sniff a MIME type for ``data``, falling back to the URL extension."""
    if data.startswith(_MAGIC_JPEG):
        return "image/jpeg"
    if data.startswith(_MAGIC_PNG):
        return "image/png"
    if data.startswith((_MAGIC_GIF, _MAGIC_GIF89)):
        return "image/gif"
    if data.startswith(_MAGIC_WEBP) and data[8:12] == b"WEBP":
        return "image/webp"
    mime, _ = mimetypes.guess_type(image_url)
    return mime or "application/octet-stream"


def resolve_image_url_for_llm(image_url: str) -> str:
    """Convert a stored image path to a base64 data URI (OpenAI-compatible).

    ``data:`` URIs and ``http(s)://`` URLs are returned as-is.
    """
    if image_url.startswith(("data:", "http://", "https://")):
        return image_url
    if is_application_stored_image(image_url):
        data = read_local_storage_bytes(image_url)
        if data is not None:
            mime = _detect_content_type(data, image_url)
            encoded = base64.b64encode(data).decode("ascii")
            return f"data:{mime};base64,{encoded}"
    return image_url


def resolve_image_url_for_ollama(image_url: str) -> bytes | None:
    """Convert a stored image path to raw bytes for the Ollama API."""
    if image_url.startswith("data:"):
        idx = image_url.find(";base64,")
        if idx < 0:
            return None
        try:
            return base64.b64decode(image_url[idx + len(";base64,") :])
        except (ValueError, binascii.Error):
            return None
    if is_application_stored_image(image_url):
        return read_local_storage_bytes(image_url)
    return None


def is_multimodal_not_supported_error(err: Exception | None) -> bool:
    """True when an error indicates the model rejects image input."""
    if err is None:
        return False
    msg = str(err).lower()
    return (
        any(token in msg for token in ("multimodal", "image", "vision"))
        and any(token in msg for token in ("not support", "unsupported", "400"))
    )


def strip_images_from_messages(messages: list[Message]) -> list[Message]:
    """Return a copy of ``messages`` with all image data removed."""
    return [message.model_copy(update={"images": []}) for message in messages]


__all__ = [
    "is_application_stored_image",
    "is_multimodal_not_supported_error",
    "local_image_resolver",
    "read_local_storage_bytes",
    "resolve_image_url_for_llm",
    "resolve_image_url_for_ollama",
    "strip_images_from_messages",
]
