"""System info service — assemble the `GET /system/info` response payload.

Mirrors the upstream handler at ``internal/handler/system.go::GetSystemInfo``:

- Reads the static build metadata (version, commit_id, build_time,
  python_version) from process-level constants.
- Reads the alembic version row + Postgres server version via the
  shared ``AsyncSession``.
- Computes ``uptime_seconds`` from the boot time recorded on
  ``app.state.started_at`` during FastAPI lifespan startup.

The service is constructed per-request by ``web.deps.system`` so the
caller can override it in tests with a hand-rolled fake that does not
need a live DB connection.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime

import sqlalchemy
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.contracts.system import SystemInfo


@dataclass(frozen=True, slots=True)
class SystemInfoSnapshot:
    """Combined service output for ``GET /system/info``.

    Mirrors :class:`src.core.contracts.system.SystemInfo` for the
    build-metadata fields the frozen contract already declares, and
    adds the runtime fields the upstream Go handler also reports
    (``db_migration_error``, ``started_at``, ``uptime_seconds``). The
    frozen contract cannot be extended in-place (see the project
    contract-immutability rule), so the runtime fields live on this
    snapshot and the web layer projects both onto the wire shape.
    """

    info: SystemInfo
    db_migration_error: str | None
    started_at: datetime
    uptime_seconds: int


@dataclass(frozen=True, slots=True)
class _BuildMetadata:
    """Compile-time build metadata, with deterministic placeholders.

    The Go project fills these from ldflags at build time. The Python
    service uses deterministic placeholders until a deployment step
    populates the build metadata via ``sys`` attributes (consumed by
    ``_resolve_build_metadata``).
    """

    version: str
    edition: str
    commit_id: str
    build_time: str


def _resolve_build_metadata() -> _BuildMetadata:
    """Read build metadata from ``sys`` attributes (set by build tooling).

    ``sys.__app_version__`` / ``__app_commit_id__`` / ``__app_build_time__``
    can be injected by a deployment entrypoint. When the attributes are
    missing we return deterministic placeholders so the payload shape
    stays valid.
    """
    return _BuildMetadata(
        version=_safe_attr("__app_version__", default="0.0.0"),
        edition="standard",
        commit_id=_safe_attr("__app_commit_id__", default="unknown"),
        build_time=_safe_attr("__app_build_time__", default="unknown"),
    )


def _safe_attr(name: str, *, default: str) -> str:
    raw = getattr(sys, name, None)
    if isinstance(raw, str) and raw:
        return raw
    return default


class SystemInfoService:
    """Assemble :class:`SystemInfoSnapshot` for ``GET /system/info``.

    Construction takes the request-scoped ``AsyncSession`` (for the
    ``SELECT version()`` and ``alembic_version`` lookups) plus the boot
    instant captured during lifespan startup. Tests inject a stub
    session and a fixed ``started_at`` to keep the assertions stable.
    """

    def __init__(
        self,
        *,
        session: AsyncSession,
        started_at: datetime | None,
    ) -> None:
        self._session = session
        self._started_at = started_at

    async def get_info(self) -> SystemInfoSnapshot:
        """Return the current system info snapshot.

        The DB lookups are best-effort: any failure surfaces as a
        non-empty ``db_migration_error`` string (or a placeholder
        ``db_version``) instead of raising, mirroring the upstream Go
        behavior that always emits a 200 with whatever state it could
        recover.
        """
        build = _resolve_build_metadata()
        db_version, db_migration_error = await self._read_db_state()

        now = datetime.now(UTC)
        started_at, uptime_seconds = self._resolve_uptime(now=now)

        info = SystemInfo(
            version=build.version,
            edition=build.edition,
            commit_id=build.commit_id,
            build_time=_parse_build_time(build.build_time),
            go_version=_python_version(),
            keyword_index_engine="",
            vector_store_engine="",
            graph_database_engine="",
            minio_enabled=False,
            db_version=db_version,
        )
        return SystemInfoSnapshot(
            info=info,
            db_migration_error=db_migration_error or None,
            started_at=started_at,
            uptime_seconds=uptime_seconds,
        )

    async def _read_db_state(self) -> tuple[str, str]:
        """Return ``(db_version, db_migration_error)`` for the system info row.

        Reads ``alembic_version.version_num`` so the frontend can detect a
        pending/half-applied migration; falls back to ``"unknown"`` when
        the table is missing. The migration ``version_num`` (alembic
        revision id) is returned as-is — alembic uses string revision
        ids, unlike the upstream Go code which formats a numeric
        migration version.

        The Postgres ``SELECT version()`` call populates the human
        readable server version string. Any DB error during either
        lookup is captured as a ``db_migration_error`` string instead of
        raised, mirroring the Go handler's "always emit 200" contract.
        """
        try:
            version_row = await self._session.execute(
                sqlalchemy.text("SELECT version_num FROM alembic_version")
            )
            revision = version_row.scalar_one_or_none() or ""
        except Exception as exc:  # noqa: BLE001 — captured as info-row error
            return ("unknown", f"alembic_version lookup failed: {exc}")

        try:
            server_row = await self._session.execute(
                sqlalchemy.text("SELECT version()")
            )
            server_version = server_row.scalar_one_or_none() or ""
        except Exception as exc:  # noqa: BLE001 — captured as info-row error
            return (revision or "unknown", f"server version lookup failed: {exc}")

        # Compose a compact label: revision + server version. The
        # upstream Go code formats the numeric version, but alembic
        # stores string revision ids and we preserve them verbatim so
        # operators can grep them.
        db_version = f"{revision} ({server_version})" if revision else server_version
        return (db_version, "")

    def _resolve_uptime(self, *, now: datetime) -> tuple[datetime, int]:
        """Return ``(started_at, uptime_seconds)`` from the lifespan state.

        ``started_at`` is recorded on ``app.state`` during the FastAPI
        lifespan startup; when unset (e.g. tests that bypass the
        lifespan) the service falls back to ``now`` so ``uptime`` is
        zero and the contract still validates.
        """
        started = self._started_at
        if started is not None:
            if started.tzinfo is None:
                started = started.replace(tzinfo=UTC)
            started_at = started.astimezone(UTC)
            delta = now - started_at
            uptime = max(0, int(delta.total_seconds()))
        else:
            started_at = now
            uptime = 0
        return started_at, uptime


def _python_version() -> str:
    """Return the runtime version string used in the ``go_version`` field.

    The upstream Go code reads ``runtime.Version()`` (``go1.21.5``); the
    Python service reuses the same wire field name for backward
    compatibility and reports the interpreter version instead.
    """
    impl = platform.python_implementation()
    version = platform.python_version()
    if impl.lower() == "cpython":
        return f"cpython{version}"
    return f"{impl.lower()}-{version}"


def _parse_build_time(raw: str) -> datetime:
    """Coerce the ``build_time`` string into the contract's ``datetime``.

    Falls back to the Unix epoch when the string is ``"unknown"`` or
    unparseable so the contract validates without forcing the deploy
    step to populate the field.
    """
    if not raw or raw == "unknown":
        return datetime(1970, 1, 1, tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return datetime(1970, 1, 1, tzinfo=UTC)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


__all__ = ["SystemInfoService", "SystemInfoSnapshot"]