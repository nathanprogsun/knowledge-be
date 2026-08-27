"""Storage-provider allowlist — ``STORAGE_ALLOW_LIST`` gating.

Ports ``internal/storageallowlist/allowlist.go`` (exposed to handlers
through ``internal/handler/storage_allowlist.go``):

============================  =============================
Go                            Python
============================  =============================
``Supported()``               ``supported_providers()``
``AllowedMap()``              ``allowed_provider_map()``
``IsAllowed(provider)``       ``is_storage_provider_allowed``
``FirstAllowed()``            ``first_allowed_provider()``
``AllowedList()``             ``allowed_providers()``
============================  =============================

Semantics preserved verbatim:

- An unset / blank ``STORAGE_ALLOW_LIST`` allows every supported
  provider.
- The raw value splits on ``,``, ``;``, ``|``, newline, tab and space,
  and each item is lower-cased; unknown names are dropped rather than
  rejected.
- The empty provider string is treated as allowed (callers pass ``""``
  when no explicit provider was chosen).
- Ordering is always the canonical ``supported`` display order, not the
  order the operator happened to type.
"""

from __future__ import annotations

import os
import re
from typing import Final

from src.core.contracts.infra import StorageProviderStatus

# ``storageallowlist.AllowListEnv``.
ALLOW_LIST_ENV: Final = "STORAGE_ALLOW_LIST"

# ``storageallowlist.supported`` — canonical display order.
SUPPORTED_STORAGE_PROVIDERS: Final[tuple[str, ...]] = (
    "local",
    "minio",
    "cos",
    "tos",
    "s3",
    "oss",
    "ks3",
    "obs",
)

# ``strings.FieldsFunc`` separator set from ``AllowedMap``.
_SEPARATOR_PATTERN: Final = re.compile(r"[,;|\n\t ]+")

# Per-provider UI copy — ``handler/system.go::GetStorageEngineStatus``
# builds the same ``[]StorageEngineStatusItem`` descriptions. Copied
# byte-for-byte from Go (fullwidth punctuation included), so the ruff
# ambiguous-character rule is suppressed rather than the text altered.
STORAGE_PROVIDER_DESCRIPTIONS: Final[dict[str, str]] = {
    "local": "本地文件系统存储，仅适合单机部署",
    "minio": "S3 兼容的自托管对象存储，适合内网和私有云部署",
    "cos": "腾讯云对象存储服务，适合公有云部署，支持 CDN 加速",
    "tos": "火山引擎对象存储服务，适合公有云部署",
    "s3": "AWS S3 与兼容对象存储服务，适合公有云与混合云部署",
    "oss": "阿里云对象存储服务，适合公有云部署，支持 S3 兼容协议",
    "ks3": "金山云对象存储服务，适合公有云部署",
    "obs": "华为云对象存储服务，适合公有云部署",
}

# ``local`` needs no configuration, so Go hard-codes ``Available: true``.
_ALWAYS_AVAILABLE_PROVIDERS: Final[frozenset[str]] = frozenset({"local"})


def supported_providers() -> list[str]:
    """Return the canonical provider names in display order."""
    return list(SUPPORTED_STORAGE_PROVIDERS)


def allowed_provider_map(raw: str | None = None) -> dict[str, bool]:
    """Map every supported provider to whether the allowlist permits it.

    ``raw`` defaults to ``os.environ[STORAGE_ALLOW_LIST]``; pass it
    explicitly to evaluate a candidate value without touching the
    environment. A blank value allows everything.

    Unlike Go's ``AllowedMap`` — which returns a sparse map where a
    missing key reads as ``false`` — this returns a total map over the
    supported set, because Python dict lookups raise instead.
    """
    value = os.environ.get(ALLOW_LIST_ENV, "") if raw is None else raw
    value = value.strip()
    if not value:
        return dict.fromkeys(SUPPORTED_STORAGE_PROVIDERS, True)

    requested = {item.strip().lower() for item in _SEPARATOR_PATTERN.split(value) if item.strip()}
    return {name: name in requested for name in SUPPORTED_STORAGE_PROVIDERS}


def allowed_providers(raw: str | None = None) -> list[str]:
    """Return the allowed providers in canonical order (``AllowedList``)."""
    allowed = allowed_provider_map(raw)
    return [name for name in SUPPORTED_STORAGE_PROVIDERS if allowed[name]]


def is_storage_provider_allowed(provider: str, raw: str | None = None) -> bool:
    """True when ``provider`` is permitted.

    An empty (or whitespace-only) provider is allowed, matching Go's
    ``IsAllowed("") == true``. Unknown names are never allowed.
    """
    normalized = provider.strip().lower()
    if not normalized:
        return True
    return allowed_provider_map(raw).get(normalized, False)


def first_allowed_provider(raw: str | None = None) -> str:
    """Return the first allowed provider in canonical order, else ``""``."""
    allowed = allowed_providers(raw)
    return allowed[0] if allowed else ""


def build_storage_provider_statuses(
    *,
    configured_providers: frozenset[str] | set[str] | None = None,
    raw_allow_list: str | None = None,
) -> list[StorageProviderStatus]:
    """Build the per-provider status rows for the system status endpoint.

    Mirrors ``handler/system.go::GetStorageEngineStatus``: ``allowed``
    comes from the allowlist, ``available`` from whether the workspace
    (or the environment) has that provider configured, and ``local`` is
    unconditionally available since it needs no configuration.

    ``configured_providers`` is the caller's set of provider names with
    a usable configuration — the Go handler unions the legacy singleton
    ``tenant.StorageEngineConfig`` checks with the active multi-instance
    ``storage_backends`` rows. Resolving those sources belongs to the
    storage-backend domain, so it is passed in rather than probed here.
    """
    configured = {name.strip().lower() for name in (configured_providers or set())}
    allowed = allowed_provider_map(raw_allow_list)
    return [
        StorageProviderStatus(
            name=name,
            allowed=allowed[name],
            available=name in _ALWAYS_AVAILABLE_PROVIDERS or name in configured,
            description=STORAGE_PROVIDER_DESCRIPTIONS[name],
        )
        for name in SUPPORTED_STORAGE_PROVIDERS
    ]


__all__ = [
    "ALLOW_LIST_ENV",
    "STORAGE_PROVIDER_DESCRIPTIONS",
    "SUPPORTED_STORAGE_PROVIDERS",
    "allowed_provider_map",
    "allowed_providers",
    "build_storage_provider_statuses",
    "first_allowed_provider",
    "is_storage_provider_allowed",
    "supported_providers",
]
