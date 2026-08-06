"""Base class for service unit tests.

The helpers in :class:`ServiceTest` are thin wrappers that configure
``AsyncMock`` repositories to return specific values. They are static
methods so subclasses do not need ``self`` plumbing — the call site
just needs to pass the mock instance and the desired return value.

Why a base class at all? Two reasons:

1. Naming consistency. The repository contract names methods like
   ``find_by_id`` and ``update_by_tenanted_primary_key``; having a
   single place that documents the wire between the mock and the
   service keeps the call sites readable.
2. Future migration target. Web-layer tests already exercise the full
   service stack; a future iteration can lift them onto
   :class:`ServiceTest` and rely on the same DSL the unit tests use.

The current commit only adds the DSL — no service test is rewritten
yet. The first consumer arrives in a later refactor commit.
"""

from __future__ import annotations

from unittest.mock import AsyncMock


class ServiceTest:
    """Base class for service unit tests.

    Subclasses inherit a small DSL of static helpers used to configure
    ``AsyncMock`` repositories. The helpers are intentionally explicit
    about the method name (``find_by_id`` etc.) so a typo surfaces at
    test setup time, not as a confusing ``AttributeError`` deep inside
    a service call.
    """

    @staticmethod
    async def mock_repo_find_by_primary_key(
        repo, method_name: str, *, return_value
    ) -> None:
        """Set ``repo.<method_name>.return_value`` and clear any side effect.

        The ``side_effect = None`` reset is important — once a side
        effect is set on an ``AsyncMock``, the return value is ignored
        unless the side effect is cleared.
        """
        getattr(repo, method_name).return_value = return_value
        getattr(repo, method_name).side_effect = None

    @staticmethod
    async def mock_repo_insert(repo, *, return_value) -> None:
        """Configure ``repo.insert`` to resolve to ``return_value``."""
        repo.insert.return_value = return_value
        repo.insert.side_effect = None

    @staticmethod
    async def mock_repo_update_by_tenanted_primary_key(repo, *, return_value) -> None:
        """Configure ``repo.update_by_tenanted_primary_key``."""
        repo.update_by_tenanted_primary_key.return_value = return_value
        repo.update_by_tenanted_primary_key.side_effect = None

    @staticmethod
    async def mock_repo_find_by_tenanted_primary_key(repo, *, return_value) -> None:
        """Configure ``repo.find_by_tenanted_primary_key``."""
        repo.find_by_tenanted_primary_key.return_value = return_value
        repo.find_by_tenanted_primary_key.side_effect = None

    @staticmethod
    async def mock_repo_find_by_id(repo, *, return_value) -> None:
        """Configure ``repo.find_by_id``."""
        repo.find_by_id.return_value = return_value
        repo.find_by_id.side_effect = None


def stateful_insert(repo: AsyncMock, store: dict, *, key_attr: str = "id") -> None:
    """Configure ``repo.insert`` (or ``repo.create``) to populate ``store``.

    The closure keys the inserted row by ``row.<key_attr>``; if the key
    is missing or already present, the new id is appended at the next
    counter position so repeated inserts always succeed (matches the
    original in-memory fake, which assigned a fresh id on every call).

    Tests reach ``store`` directly to assert on the persisted rows.
    """
    counter = [0]

    async def _insert(row):  # type: ignore[no-untyped-def]
        counter[0] += 1
        # Preserve the row's pre-assigned id when present (UUID-style).
        existing_key = getattr(row, key_attr, None)
        key = existing_key if existing_key else counter[0]
        stored = row.model_copy(update={key_attr: key})
        store[key] = stored
        return stored

    repo.insert.side_effect = _insert
    if hasattr(repo, "create"):
        repo.create.side_effect = _insert


def lookup_by(store: dict, *, key_attr: str = "id"):
    """Build a side_effect closure that returns ``store[key]`` or raises NotFoundError.

    Tests wire this into ``repo.find_by_id`` (etc.) so the service
    reads back what it just inserted.
    """
    from src.common.exception import NotFoundError

    async def _lookup(key):  # type: ignore[no-untyped-def]
        # Normalise: callers may pass int or str.
        norm = str(key)
        for k, v in store.items():
            if str(k) == norm:
                return v
        raise NotFoundError(
            code=f"{key_attr}.not_found",
            message=f"{key_attr} {key} not found",
        )

    return _lookup


__all__ = ["ServiceTest", "stateful_insert", "lookup_by"]
