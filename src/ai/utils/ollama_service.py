"""Async Ollama service client (mirrors the upstream ``ollama.go`` service).

``OllamaService`` manages a local Ollama instance over the documented REST
API: availability probing, model listing, pulling, creating and inspecting
models. The overlapping operations (version / list / pull) speak the same
endpoints and wire shapes as the initialization-layer Ollama client so a
caller can swap between them without behavioral change; the new
operations (``create_model``, ``get_model_info``) are implemented here.

The ``is_optional`` flag mirrors the upstream ``OLLAMA_OPTIONAL`` behavior:
when set, an unreachable service degrades to a warning instead of an
error, and ``ensure_model_available`` / ``pull_model`` short-circuit.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, Field

from src.common.exception import ExternalServiceError
from src.common.json import JsonObject, JsonValue

OLLAMA_BASE_URL_ENV: Final = "OLLAMA_BASE_URL"
OLLAMA_OPTIONAL_ENV: Final = "OLLAMA_OPTIONAL"
OLLAMA_DIAL_FALLBACK_URL: Final = "http://localhost:11434"

_UNKNOWN_VERSION: Final = "unknown"
_DEFAULT_TIMEOUT_SECONDS: Final = 30.0
# Upstream caps async pulls at 12h via a context timeout.
PULL_TIMEOUT_SECONDS: Final = 12 * 60 * 60

# Model-info (``/api/show``) responses carry many optional fields; the
# service returns the raw JSON object and callers project what they need.
ShowResponse = JsonObject

ProgressCallback = Callable[[float, str], Awaitable[None]]


class OllamaModelInfo(BaseModel):
    """Detailed info for one locally installed Ollama model.

    Mirrors the upstream ``OllamaModelInfo`` projection of ``/api/tags``
    entries. ``size`` / ``digest`` / ``modified_at`` are nullable because
    some Ollama builds omit them.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    size: int | None = Field(default=None)
    digest: str | None = Field(default=None)
    modified_at: datetime | None = Field(default=None)


class OllamaEmbedRequest(BaseModel):
    """``/api/embed`` request (the upstream ``ollama/api.EmbedRequest``).

    ``options`` carries model options such as ``num_ctx``; ``truncate``
    and ``dimensions`` are optional and omitted from the wire when unset.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    input: list[str]
    truncate: bool | None = Field(default=None)
    dimensions: int | None = Field(default=None)
    options: dict[str, JsonValue] = Field(default_factory=dict)


def resolve_ollama_dial_base_url() -> str:
    """Base URL the HTTP client dials (env override or localhost fallback)."""
    return os.environ.get(OLLAMA_BASE_URL_ENV) or OLLAMA_DIAL_FALLBACK_URL


def normalize_model_tag(model_name: str) -> str:
    """An untagged model name means ``:latest`` (Go ``IsModelAvailable``)."""
    return model_name if ":" in model_name else f"{model_name}:latest"


def _parse_pull_progress(line: str) -> tuple[float, str]:
    """Map one ``/api/pull`` NDJSON frame onto (percent, message).

    Mirrors the upstream progress callback: percent is
    ``completed/total*100`` when both counters are present, otherwise the
    raw ``status`` string is surfaced with no percentage. An error frame
    aborts the pull.
    """
    stripped = line.strip()
    if not stripped:
        return 0.0, ""
    try:
        frame = json.loads(stripped)
    except json.JSONDecodeError:
        return 0.0, ""
    if not isinstance(frame, dict):
        return 0.0, ""
    error = frame.get("error")
    if isinstance(error, str) and error:
        raise ExternalServiceError(
            f"ollama pull failed: {error}",
            code="ollama.pull_failed",
        )
    status = frame.get("status")
    status_text = status if isinstance(status, str) else ""
    total = frame.get("total")
    completed = frame.get("completed")
    if isinstance(total, int) and isinstance(completed, int) and total > 0 and completed > 0:
        percent = completed / total * 100
        return percent, f"Pull progress: {status_text} ({percent:.2f}%)"
    if status_text:
        return 0.0, f"Pull status: {status_text}"
    return 0.0, ""


class OllamaService:
    """Async client managing one local Ollama service.

    Tests inject an ``httpx.MockTransport`` via ``transport=`` exactly as
    with the other HTTP adapters in this package.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        is_optional: bool | None = None,
    ) -> None:
        self._base_url = (base_url or resolve_ollama_dial_base_url()).rstrip("/")
        if is_optional is None:
            is_optional = os.environ.get(OLLAMA_OPTIONAL_ENV) == "true"
        self._is_optional = is_optional
        self._is_available = False
        if transport is not None:
            self._client = httpx.AsyncClient(transport=transport, timeout=timeout)
        else:
            self._client = httpx.AsyncClient(timeout=timeout)

    @property
    def base_url(self) -> str:
        """The dialed base URL (``OllamaClient.base_url`` parity)."""
        return self._base_url

    @property
    def is_optional(self) -> bool:
        """Whether an unreachable service degrades to a warning."""
        return self._is_optional

    async def aclose(self) -> None:
        """Release the underlying HTTP client."""
        await self._client.aclose()

    # ── Availability ─────────────────────────────────────────────────

    async def start_service(self) -> None:
        """Probe the service and record availability (Go ``StartService``).

        When the service is unreachable and ``is_optional`` is set, the
        failure is swallowed and availability stays ``False``; otherwise an
        ``ExternalServiceError`` is raised.
        """
        try:
            response = await self._client.request("HEAD", f"{self._base_url}/")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            self._is_available = False
            if self._is_optional:
                return
            raise ExternalServiceError(
                f"ollama service unavailable: {exc}",
                code="ollama.unavailable",
            ) from exc
        self._is_available = True

    def is_available(self) -> bool:
        """Return whether the service responded to the last probe."""
        return self._is_available

    # ── Model operations ─────────────────────────────────────────────

    async def is_model_available(self, model_name: str) -> bool:
        """True when ``model_name`` (or ``<name>:latest``) is installed.

        An unreachable but optional service reports ``False`` without
        erroring, matching the upstream behavior.
        """
        await self.start_service()
        if not self._is_available and self._is_optional:
            return False
        model_names = await self.list_models()
        return normalize_model_tag(model_name) in model_names

    async def pull_model(
        self,
        model_name: str,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        """Pull ``model_name``, skipping when it is already installed.

        ``on_progress`` receives ``(percent, message)`` for each streamed
        progress frame. An unreachable but optional service short-circuits
        without erroring.
        """
        await self.start_service()
        if not self._is_available and self._is_optional:
            return
        if await self.is_model_available(model_name):
            return

        payload = {"name": model_name, "stream": True}
        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/api/pull",
                json=payload,
                timeout=PULL_TIMEOUT_SECONDS,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    progress, message = _parse_pull_progress(line)
                    if on_progress is not None and message:
                        await on_progress(progress, message)
        except httpx.HTTPError as exc:
            raise ExternalServiceError(
                f"failed to pull model: {exc}",
                code="ollama.pull_failed",
            ) from exc

    async def ensure_model_available(self, model_name: str) -> None:
        """Pull ``model_name`` when it is not installed (Go ``EnsureModelAvailable``)."""
        if not self._is_available and self._is_optional:
            return
        try:
            available = await self.is_model_available(model_name)
        except ExternalServiceError:
            if self._is_optional:
                return
            raise
        if not available:
            await self.pull_model(model_name)

    async def get_version(self) -> str:
        """Return the installed Ollama version string.

        An unreachable but optional service reports ``"unavailable"``; a
        reachable service that fails to answer raises.
        """
        if not self._is_available and self._is_optional:
            return "unavailable"
        try:
            response = await self._client.get(f"{self._base_url}/api/version")
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ExternalServiceError(
                f"failed to get Ollama version: {exc}",
                code="ollama.version_failed",
            ) from exc
        if not isinstance(payload, dict):
            raise ExternalServiceError(
                "failed to get Ollama version: malformed response",
                code="ollama.version_failed",
            )
        version = payload.get("version")
        return version if isinstance(version, str) else ""

    async def create_model(self, name: str, modelfile: str) -> None:
        """Create a custom model from a ``modelfile`` (Go ``CreateModel``).

        Sends ``{"model": name, "template": modelfile}`` — the upstream
        Go client sets the ``template`` request field for the create
        payload.
        """
        payload = {"model": name, "template": modelfile}
        try:
            response = await self._client.post(f"{self._base_url}/api/create", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ExternalServiceError(
                f"failed to create model: {exc}",
                code="ollama.create_failed",
            ) from exc

    async def get_model_info(self, model_name: str) -> ShowResponse:
        """Return the raw ``/api/show`` model info (Go ``GetModelInfo``)."""
        payload = {"model": model_name}
        try:
            response = await self._client.post(f"{self._base_url}/api/show", json=payload)
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ExternalServiceError(
                f"failed to get model information: {exc}",
                code="ollama.show_failed",
            ) from exc
        if not isinstance(body, dict):
            raise ExternalServiceError(
                "failed to get model information: malformed response",
                code="ollama.show_failed",
            )
        return body

    async def list_models(self) -> list[str]:
        """Return the names of every locally installed model (Go ``ListModels``)."""
        models = await self.list_models_detailed()
        return [model.name for model in models]

    async def list_models_detailed(self) -> list[OllamaModelInfo]:
        """Return detailed info for every installed model (Go ``ListModelsDetailed``)."""
        try:
            response = await self._client.get(f"{self._base_url}/api/tags")
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ExternalServiceError(
                f"failed to get model list: {exc}",
                code="ollama.list_models_failed",
            ) from exc
        if not isinstance(payload, dict):
            return []
        raw_models = payload.get("models")
        if not isinstance(raw_models, list):
            return []
        return [
            OllamaModelInfo.model_validate(entry) for entry in raw_models if isinstance(entry, dict)
        ]

    async def embeddings(self, request: OllamaEmbedRequest) -> list[list[float]]:
        """Return embedding vectors for ``request.input`` (upstream ``Embeddings``).

        POSTs ``/api/embed``. An unreachable service (or a non-optional
        probe failure) raises ``ExternalServiceError``.
        """
        payload: dict[str, JsonValue] = {
            "model": request.model,
            "input": list(request.input),
        }
        if request.truncate is not None:
            payload["truncate"] = request.truncate
        if request.dimensions is not None:
            payload["dimensions"] = request.dimensions
        if request.options:
            payload["options"] = request.options
        try:
            response = await self._client.post(
                f"{self._base_url}/api/embed",
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ExternalServiceError(
                f"failed to get embedding vectors: {exc}",
                code="ollama.embeddings_failed",
            ) from exc
        if not isinstance(body, dict):
            raise ExternalServiceError(
                "failed to get embedding vectors: malformed response",
                code="ollama.embeddings_failed",
            )
        raw_embeddings = body.get("embeddings")
        if not isinstance(raw_embeddings, list):
            raise ExternalServiceError(
                "failed to get embedding vectors: missing embeddings",
                code="ollama.embeddings_failed",
            )
        return [_coerce_embedding(item) for item in raw_embeddings]

    # ── Chat ─────────────────────────────────────────────────────────

    async def chat(self, chat_request: JsonObject) -> JsonObject:
        """Send one non-streaming ``/api/chat`` request (upstream ``Chat``).

        The request body is passed through verbatim; callers set the
        ``stream`` flag themselves (the local VLM sends ``False``). Returns
        the parsed response JSON body. An unreachable service (or a
        non-optional probe failure) raises ``ExternalServiceError``.
        """
        try:
            response = await self._client.post(
                f"{self._base_url}/api/chat",
                json=chat_request,
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ExternalServiceError(
                f"failed to complete chat request: {exc}",
                code="ollama.chat_failed",
            ) from exc
        if not isinstance(body, dict):
            raise ExternalServiceError(
                "failed to complete chat request: malformed response",
                code="ollama.chat_failed",
            )
        return body

    async def chat_stream(self, chat_request: JsonObject) -> AsyncIterator[JsonObject]:
        """Stream one ``/api/chat`` request, yielding each parsed SSE frame.

        ``chat_request["stream"]`` must be truthy. The response is parsed as
        Server-Sent Events: ``data:`` lines are decoded to JSON objects and
        yielded until the ``data: [DONE]`` sentinel or end of stream.
        """
        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/api/chat",
                json=chat_request,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line == "data: [DONE]":
                        return
                    if line.startswith("data: "):
                        payload = line[6:]
                    elif line.startswith("data:"):
                        payload = line[5:]
                    else:
                        continue
                    try:
                        frame = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(frame, dict):
                        yield frame
        except httpx.HTTPError as exc:
            raise ExternalServiceError(
                f"failed to stream chat response: {exc}",
                code="ollama.chat_failed",
            ) from exc

    # ── Pure helpers ─────────────────────────────────────────────────

    @staticmethod
    def is_valid_model_name(name: str) -> bool:
        """True when ``name`` is non-empty and contains no spaces."""
        return name != "" and " " not in name


def _coerce_embedding(item: JsonValue) -> list[float]:
    """Project one ``/api/embed`` embedding entry onto ``list[float]``."""
    if not isinstance(item, list):
        return []
    return [float(value) for value in item if isinstance(value, (int, float))]


__all__ = [
    "OLLAMA_BASE_URL_ENV",
    "OLLAMA_DIAL_FALLBACK_URL",
    "OLLAMA_OPTIONAL_ENV",
    "PULL_TIMEOUT_SECONDS",
    "OllamaEmbedRequest",
    "OllamaModelInfo",
    "OllamaService",
    "ProgressCallback",
    "ShowResponse",
    "normalize_model_tag",
    "resolve_ollama_dial_base_url",
]
