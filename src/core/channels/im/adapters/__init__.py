"""IM platform adapters — concrete implementations of the adapter contract.

Each module implements one platform adapter: identity, callback
verification and parsing, URL verification, reply sending, and the
connection lifecycle. Every adapter module also exposes a
``build_<platform>_adapter`` factory that reads a channel row's
credentials, so ``register_default_adapters`` can wire all platforms
into an ``IMSupervisor``'s adapter registry in one call.
"""

from __future__ import annotations

from collections.abc import Callable

from src.core.channels.im.adapter_base import IMAdapter
from src.core.channels.im.adapters import (
    dingtalk,
    mattermost,
    qqbot,
    slack,
    telegram,
    wechat,
    yunzhijia,
)
from src.core.channels.im.supervisor import IMSupervisor
from src.db.models.im_channel import IMChannel

# Platform identifier → adapter factory built from a channel row.
_PLATFORM_BUILDERS: dict[str, Callable[[IMChannel], IMAdapter]] = {
    "slack": slack.build_slack_adapter,
    "telegram": telegram.build_telegram_adapter,
    "dingtalk": dingtalk.build_dingtalk_adapter,
    "mattermost": mattermost.build_mattermost_adapter,
    "wechat": wechat.build_wechat_adapter,
    "qqbot": qqbot.build_qqbot_adapter,
    "yunzhijia": yunzhijia.build_yunzhijia_adapter,
}


def register_default_adapters(supervisor: IMSupervisor) -> None:
    """Register every platform adapter factory on ``supervisor``.

    Idempotent: re-registering a platform on the supervisor replaces
    the earlier factory with the same behavior.
    """
    for platform, builder in _PLATFORM_BUILDERS.items():
        supervisor.register_adapter_factory(platform, builder)


def default_adapter_platforms() -> list[str]:
    """Return the platform identifiers this package registers."""
    return sorted(_PLATFORM_BUILDERS)


__all__ = [
    "IMAdapter",
    "default_adapter_platforms",
    "dingtalk",
    "mattermost",
    "qqbot",
    "register_default_adapters",
    "slack",
    "telegram",
    "wechat",
    "yunzhijia",
]
