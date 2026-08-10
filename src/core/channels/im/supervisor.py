"""IM connection supervisor — connect / recycle / health-check / reconnect.

Owns:

- the **adapter registry** — platform name → ``AdapterFactory``
  (the single sanctioned construction point for concrete platform
  adapters; registered from each platform's wiring code)
- the **active-channel map** — channel id → running adapter + stop
  callable, the runtime state every active channel needs
- the **health-check loop** — periodic probe of each active adapter
  and reconnect-on-failure

``run_supervised`` is the per-connection lifecycle the adapter base
hands the supervisor; ``IMSupervisor`` is the process-wide monitor
the service delegates to.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeAlias

from src.common.exception import NotFoundError
from src.core.channels.im.adapter_base import (
    EventContext,
    IMAdapter,
    StopCallable,
)
from src.db.models.im_channel import IMChannel

# ── Constants ────────────────────────────────────────────────────────

# Default periodic recycle interval for a supervised long connection.
# Some platform SDKs can silently enter a "zombie" state on long-running
# connections; recreating the connection on a schedule bounds the
# worst-case outage.
DEFAULT_RECYCLE_INTERVAL_SECONDS: float = 6.0 * 60.0 * 60.0

# Default backoff applied after a failed connect attempt.
DEFAULT_SUPERVISOR_RETRY_DELAY_SECONDS: float = 5.0

# Default interval between health-check sweeps of the active channel
# map. Short enough to catch a stuck adapter quickly; long enough to
# avoid hammering the database.
DEFAULT_HEALTH_INTERVAL_SECONDS: float = 30.0

logger = logging.getLogger("src.core.channels.im.supervisor")

# ── Public types ─────────────────────────────────────────────────────

# Factory signature: build an adapter from a persisted channel row.
# The message-handler seam (the QA pipeline) is wired in by a later
# module; platform adapters subclass ``IMAdapter`` and this factory
# is what constructs them.
AdapterFactory: TypeAlias = Callable[[IMChannel], IMAdapter]


@dataclass
class _ActiveChannel:
    """Runtime state for one supervised channel connection."""

    adapter: IMAdapter
    channel: IMChannel
    stop: StopCallable | None = None


@dataclass
class SupervisorConfig:
    """Configuration for a single ``run_supervised`` invocation.

    ``connect`` returns a stop callable that tears the connection
    down cleanly; the loop recycles the connection every
    ``max_conn_age`` and retries after ``retry_delay`` when
    ``connect`` raises.
    """

    name: str
    connect: Callable[[], Awaitable[StopCallable | None]]
    max_conn_age: float = DEFAULT_RECYCLE_INTERVAL_SECONDS
    retry_delay: float = DEFAULT_SUPERVISOR_RETRY_DELAY_SECONDS


# ── Per-connection lifecycle ────────────────────────────────────────


async def run_supervised(
    stop_event: asyncio.Event,
    cfg: SupervisorConfig,
) -> None:
    """Drive one supervised long-lived connection.

    Keeps a connection alive by (re)establishing it via ``cfg.connect``
    and proactively recycling it every ``cfg.max_conn_age``. The
    underlying SDK is expected to handle transient drops via its own
    auto-reconnect; the periodic recycle is the safety net that
    eliminates stuck / zombie connections the SDK fails to recover.

    Returns when ``stop_event`` fires. On return the active
    connection's stop callable (if any) has been invoked.
    """
    while not stop_event.is_set():
        try:
            stop = await cfg.connect()
        except Exception as exc:  # supervisor recovers across SDK errors
            if stop_event.is_set():
                return
            logger.warning(
                "[IM] %s connect failed: %s; retrying in %.1fs",
                cfg.name,
                exc,
                cfg.retry_delay,
            )
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=cfg.retry_delay)
            except TimeoutError:
                continue
            return

        if stop_event.is_set():
            if stop is not None:
                stop()
            return

        logger.info(
            "[IM] %s connection established (recycle in %.1fs)",
            cfg.name,
            cfg.max_conn_age,
        )

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=cfg.max_conn_age)
        except TimeoutError:
            logger.info("[IM] %s periodic reconnect to refresh connection", cfg.name)
            if stop is not None:
                stop()
            continue

        if stop is not None:
            stop()
        return


# ── Process-wide monitor ────────────────────────────────────────────


class IMSupervisor:
    """Process-wide supervisor of IM channel connections.

    Owns the platform → ``AdapterFactory`` registry and the
    channel-id → running-adapter map. The service delegates
    ``start_channel`` / ``stop_channel`` / ``get_channel_adapter``
    here, and the health-check loop runs against the live map.

    Lifetime: one instance is shared by every ``IMChannelService``
    request, so the registry and active-channel state outlive any
    single HTTP request. ``shutdown`` is called from the FastAPI
    lifespan teardown to drain every active connection cleanly.
    """

    def __init__(
        self,
        *,
        health_interval: float = DEFAULT_HEALTH_INTERVAL_SECONDS,
    ) -> None:
        self._adapter_factories: dict[str, AdapterFactory] = {}
        self._active: dict[str, _ActiveChannel] = {}
        self._health_interval = health_interval
        self._stop_event = asyncio.Event()
        self._shutdown_called = False
        # Single shared context the supervisor exposes to adapters
        # during connect. ``cancel`` flips the same flag as the
        # health-loop stop event, so one signal shuts down everything.
        self._context = EventContext()

    # ── Adapter registry ────────────────────────────────────────────

    def register_adapter_factory(self, platform: str, factory: AdapterFactory) -> None:
        """Register ``factory`` for ``platform``.

        Re-registering a platform replaces the earlier factory; the
        platform name is lower-cased so the lookup is
        case-insensitive.
        """
        self._adapter_factories[platform.strip().lower()] = factory

    def get_adapter_factory(self, platform: str) -> AdapterFactory:
        """Return the factory for ``platform``.

        Raises ``NotFoundError`` (``im.adapter_not_found``) when no
        factory is registered — the service surfaces this as a
        clear error to the caller.
        """
        factory = self._adapter_factories.get(platform.strip().lower())
        if factory is None:
            raise NotFoundError(
                code="im.adapter_not_found",
                message=f"no adapter factory for platform: {platform}",
            )
        return factory

    def registered_platforms(self) -> list[str]:
        """Return the platform identifiers with a registered factory."""
        return sorted(self._adapter_factories)

    # ── Channel lifecycle ───────────────────────────────────────────

    async def start_channel(self, channel: IMChannel) -> None:
        """Create an adapter for ``channel`` and connect it.

        Replaces any existing active state for the same channel id.
        The factory lookup is case-insensitive; a missing factory
        surfaces as ``NotFoundError`` so the caller can surface a
        clear error to the web layer.
        """
        existing = self._active.get(channel.id)
        if existing is not None:
            self._stop_active(channel.id, existing)
        # Drop any (possibly torn-down) stale entry before rebuilding,
        # so a failed connect leaves no dead entry in the map.
        self._active.pop(channel.id, None)

        factory = self.get_adapter_factory(channel.platform)
        adapter = factory(channel)
        stop = await adapter.connect(self._context)
        self._active[channel.id] = _ActiveChannel(adapter=adapter, channel=channel, stop=stop)

    def stop_channel(self, channel_id: str) -> None:
        """Tear down the active connection for ``channel_id`` (if any)."""
        active = self._active.pop(channel_id, None)
        if active is not None:
            self._stop_active(channel_id, active)

    def get_channel_adapter(
        self, channel_id: str
    ) -> tuple[IMAdapter | None, IMChannel | None, bool]:
        """Return ``(adapter, channel, running)`` for ``channel_id``.

        ``running`` is ``False`` for an absent or torn-down channel so
        callers can distinguish "no row" from "row, adapter down".
        """
        active = self._active.get(channel_id)
        if active is None:
            return None, None, False
        return active.adapter, active.channel, True

    def active_channel_count(self) -> int:
        """Return the count of currently-running supervised channels."""
        return len(self._active)

    async def start_enabled_channels(self, channels: list[IMChannel]) -> tuple[int, list[str]]:
        """Start every enabled channel from ``channels``.

        Returns ``(started_count, failed_channel_ids)``. A failure on
        one channel does not stop the rest — the supervisor tries
        each independently and surfaces the per-id failures so the
        caller can log them.
        """
        started = 0
        failed: list[str] = []
        for channel in channels:
            try:
                await self.start_channel(channel)
                started += 1
            except Exception as exc:  # one bad row doesn't kill the rest
                failed.append(channel.id)
                logger.warning(
                    "[IM] failed to start channel %s (%s/%s): %s",
                    channel.id,
                    channel.platform,
                    channel.name,
                    exc,
                )
        return started, failed

    # ── Health check ────────────────────────────────────────────────

    async def health_check(self) -> list[str]:
        """Probe every active adapter and reconnect on failure.

        Returns the list of channel ids that were reconnected. An
        adapter whose ``is_connected`` returns ``False`` (or whose
        stop callable is missing) is torn down and rebuilt via its
        registered factory. The supervisor never raises out of this
        method — a single bad channel is logged and skipped so the
        loop keeps running.
        """
        reconnected: list[str] = []
        for channel_id in list(self._active):
            active = self._active.get(channel_id)
            if active is None:
                continue
            if active.adapter.is_connected() and active.stop is not None:
                continue
            logger.warning(
                "[IM] health check: reconnecting channel %s (%s)",
                channel_id,
                active.channel.platform,
            )
            # ``start_channel`` tears down the stale state internally
            # before connecting the fresh adapter.
            try:
                await self.start_channel(active.channel)
                reconnected.append(channel_id)
            except Exception:
                logger.exception("[IM] reconnect failed for %s; channel left stopped", channel_id)
                self._active.pop(channel_id, None)
        return reconnected

    async def run_health_loop(self) -> None:
        """Drive ``health_check`` on ``_health_interval`` until shutdown.

        Resets the loop on exit so ``shutdown`` followed by another
        ``run_health_loop`` starts a fresh cycle (the lifespan wires
        this once at startup and once again after a manual restart).
        """
        self._stop_event.clear()
        while not self._stop_event.is_set():
            try:
                await self.health_check()
            except Exception:
                logger.exception("[IM] supervisor health_check failed")
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._health_interval)
            except TimeoutError:
                continue

    def shutdown(self) -> None:
        """Signal the health loop to exit and tear down every active connection.

        Safe to call multiple times; the health loop and active
        connections are only drained on the first invocation.
        """
        if self._shutdown_called:
            return
        self._shutdown_called = True
        self._stop_event.set()
        self._context.cancel()
        for channel_id in list(self._active):
            active = self._active.pop(channel_id, None)
            if active is not None:
                self._stop_active(channel_id, active)

    # ── Internals ───────────────────────────────────────────────────

    def _stop_active(self, channel_id: str, active: _ActiveChannel) -> None:
        """Invoke ``adapter.disconnect`` and the stop callable.

        Either step may raise; the supervisor logs and swallows so
        teardown is best-effort across the whole map.
        """
        try:
            if active.stop is not None:
                active.stop()
        except Exception:
            logger.exception("[IM] stop callable failed for %s", channel_id)
        try:
            active.adapter.disconnect()
        except Exception:
            logger.exception("[IM] adapter.disconnect failed for %s", channel_id)


# ── Module-level default supervisor ──────────────────────────────────


# A single process-wide supervisor instance. The IM channel service
# is request-scoped (constructed per HTTP request with a fresh
# ``AsyncSession``), but the runtime connection map is process-wide.
# Building this at import time (like the process-wide connector
# registry in the datasource factory) keeps the registry and active
# state singletons without per-request churn.
_DEFAULT_SUPERVISOR: IMSupervisor = IMSupervisor()


def get_default_supervisor() -> IMSupervisor:
    """Return the process-wide ``IMSupervisor`` singleton."""
    return _DEFAULT_SUPERVISOR


__all__ = [
    "DEFAULT_HEALTH_INTERVAL_SECONDS",
    "DEFAULT_RECYCLE_INTERVAL_SECONDS",
    "DEFAULT_SUPERVISOR_RETRY_DELAY_SECONDS",
    "AdapterFactory",
    "IMSupervisor",
    "SupervisorConfig",
    "get_default_supervisor",
    "run_supervised",
]
