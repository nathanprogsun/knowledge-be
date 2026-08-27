"""Vision-language model client: protocol, config and factory.

Maps the upstream VLM entry point: ``Config`` describes one model, the
``VLM`` protocol is the client surface, ``config_from_model`` translates
a stored model row into a config and ``new_vlm`` routes to the Ollama,
managed-cloud or OpenAI-compatible backend by source / provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

from src.ai.provider.detect import detect_provider
from src.ai.provider.registry import ALL_PROVIDERS, PROVIDER_CLOUD
from src.ai.vlm.cloud import new_cloud_vlm
from src.ai.vlm.concurrency import wrap_vlm_concurrency
from src.ai.vlm.ollama import OllamaChatService, new_ollama_vlm
from src.ai.vlm.remote_api import new_remote_api_vlm
from src.common.json import JsonObject

MODEL_SOURCE_LOCAL: str = "local"
MODEL_SOURCE_REMOTE: str = "remote"


class VLM(Protocol):
    """Interface for Vision Language Model operations."""

    async def predict(self, img_bytes: list[bytes], prompt: str) -> str:
        """Send one or more images with a text prompt; return generated text."""
        ...

    def get_model_name(self) -> str: ...

    def get_model_id(self) -> str: ...


@dataclass(frozen=True, slots=True)
class Config:
    """Configuration needed to create a VLM instance."""

    source: str = ""
    base_url: str = ""
    model_name: str = ""
    api_key: str = ""
    model_id: str = ""
    # "ollama" or "openai" (default)
    interface_type: str = ""
    provider: str = ""
    # Caps concurrent background calls to this model; 0 falls back to the
    # process-wide default (no governor installed means no cap).
    max_concurrency: int = 0
    extra: JsonObject | None = None
    # Custom headers attached to every remote-API request.
    custom_headers: dict[str, str] | None = None
    app_id: str = ""
    app_secret: str = ""


class ModelParametersLike(Protocol):
    """The subset of a stored model's parameters the VLM config needs."""

    base_url: str | None
    api_key: str | None
    provider: str | None
    interface_type: str | None
    extra_config: dict[str, str] | None
    custom_headers: dict[str, str] | None
    max_concurrency: int | None


class ModelLike(Protocol):
    """Structural view of a stored model row for :func:`config_from_model`."""

    id: str
    name: str
    source: str
    parameters: ModelParametersLike


def _string_map_to_json(mapping: dict[str, str] | None) -> JsonObject | None:
    if mapping is None:
        return None
    return cast(JsonObject, dict(mapping))


def config_from_model(model: ModelLike | None, app_id: str, app_secret: str) -> Config | None:
    """Build a VLM ``Config`` from a stored model row.

    The production path (loaded from the database) and the test-connection
    path (temporary form) share this mapping. ``app_id`` / ``app_secret``
    are the already-decrypted managed-cloud credentials, supplied by the
    caller. ``interface_type`` falls back to ``ollama`` for local sources
    and ``openai`` otherwise.
    """
    if model is None:
        return None
    parameters = model.parameters
    interface_type = parameters.interface_type
    if not interface_type:
        interface_type = "ollama" if model.source == MODEL_SOURCE_LOCAL else "openai"
    return Config(
        model_id=model.id,
        api_key=parameters.api_key or "",
        base_url=parameters.base_url or "",
        model_name=model.name,
        source=model.source,
        interface_type=interface_type,
        provider=parameters.provider or "",
        max_concurrency=parameters.max_concurrency or 0,
        extra=_string_map_to_json(parameters.extra_config),
        custom_headers=parameters.custom_headers,
        app_id=app_id,
        app_secret=app_secret,
    )


def _resolve_provider(config: Config) -> str:
    """Return the provider name, detecting it from the base URL when absent.

    Mirrors the upstream routing: an unknown configured provider falls
    back to base-URL detection.
    """
    if config.provider in ALL_PROVIDERS:
        return config.provider
    return detect_provider(config.base_url)


async def new_vlm(config: Config, ollama_service: OllamaChatService | None) -> VLM:
    """Create a VLM instance based on the provided configuration."""
    vlm: VLM
    interface_type = config.interface_type.lower()
    if interface_type == "ollama" or config.source == MODEL_SOURCE_LOCAL:
        vlm = new_ollama_vlm(
            model_name=config.model_name,
            model_id=config.model_id,
            ollama_service=ollama_service,
        )
    else:
        provider_name = _resolve_provider(config)
        if provider_name == PROVIDER_CLOUD:
            vlm = await new_cloud_vlm(
                model_name=config.model_name,
                model_id=config.model_id,
                base_url=config.base_url,
                app_id=config.app_id,
                app_secret=config.app_secret,
                extra=config.extra,
            )
        else:
            vlm = await new_remote_api_vlm(
                model_name=config.model_name,
                model_id=config.model_id,
                base_url=config.base_url,
                api_key=config.api_key,
                provider=provider_name,
                extra=config.extra,
                custom_headers=config.custom_headers,
            )
    return wrap_vlm_concurrency(vlm, config.max_concurrency)


__all__ = [
    "MODEL_SOURCE_LOCAL",
    "MODEL_SOURCE_REMOTE",
    "VLM",
    "Config",
    "ModelLike",
    "ModelParametersLike",
    "config_from_model",
    "new_vlm",
]
