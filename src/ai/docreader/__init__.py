"""Document-reader gRPC client.

The client translates a plain address + auth configuration into calls
against the document parsing service: unary Read (parse), server-streaming
ReadStream (progress: metadata frame then one frame per image) and
ListEngines. Generated gRPC stubs live in ``.proto``.
"""

from src.ai.docreader.client import (
    AuthConfig,
    DocReaderClient,
    ImageRefInfo,
    get_image_refs_from_response,
    load_auth_config_from_env,
    new_client,
    new_client_with_auth,
)

__all__ = [
    "AuthConfig",
    "DocReaderClient",
    "ImageRefInfo",
    "get_image_refs_from_response",
    "load_auth_config_from_env",
    "new_client",
    "new_client_with_auth",
]
