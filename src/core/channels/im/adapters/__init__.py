"""Concrete IM platform adapters and their factories.

Each module implements one platform adapter: identity, callback
verification and parsing, URL verification, reply sending, and the
connection lifecycle. Every adapter module also exposes a
``build_<platform>_adapter`` factory that reads a channel row's
credentials, so the registration helpers can wire every platform
into an ``IMSupervisor``'s adapter registry in one call.

``FeishuAdapter`` serves both Feishu and Lark (the same product on two
isolated clouds); ``WecomAdapter`` serves WeCom in webhook mode.

:func:`register_im_adapters` and :func:`register_default_adapters` are
the sanctioned registration entry points; both are wired onto the
process-wide default supervisor at import time so ``start_channel``
resolves any supported platform without an explicit app hook.
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
from src.core.channels.im.adapters.feishu import FeishuAdapter, build_feishu_adapter
from src.core.channels.im.adapters.wecom import WecomAdapter, build_wecom_adapter
from src.core.channels.im.supervisor import IMSupervisor, get_default_supervisor
from src.db.models.im_channel import IMChannel

# Platform identifiers the feishu implementation serves (feishu + lark).
_FEISHU_ALIASES: tuple[str, ...] = ("feishu", "lark")

# Platform identifiers the wecom implementation serves (wecom + wxwork).
_WECOM_ALIASES: tuple[str, ...] = ("wecom", "wxwork")

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


def register_im_adapters(supervisor: IMSupervisor) -> None:
    """Register the feishu/lark and wecom/wxwork adapter factories.

    Idempotent — re-registering replaces the earlier factory for an
    alias, matching the supervisor's contract.
    """
    for alias in _FEISHU_ALIASES:
        supervisor.register_adapter_factory(alias, build_feishu_adapter)
    for alias in _WECOM_ALIASES:
        supervisor.register_adapter_factory(alias, build_wecom_adapter)


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


# Wire the process-wide supervisor at import time so ``start_channel``
# resolves every supported platform without an explicit app hook — the
# same import-time registry construction the datasource factory uses.
register_im_adapters(get_default_supervisor())
register_default_adapters(get_default_supervisor())


__all__ = [
    "FeishuAdapter",
    "IMAdapter",
    "WecomAdapter",
    "build_feishu_adapter",
    "build_wecom_adapter",
    "default_adapter_platforms",
    "dingtalk",
    "mattermost",
    "qqbot",
    "register_default_adapters",
    "register_im_adapters",
    "slack",
    "telegram",
    "wechat",
    "yunzhijia",
]
