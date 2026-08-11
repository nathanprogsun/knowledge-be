"""Concrete IM platform adapters and their factories.

``FeishuAdapter`` serves both Feishu and Lark (the same product on two
isolated clouds); ``WecomAdapter`` serves WeCom in webhook mode. The
``build_*_adapter`` factories construct an adapter from a persisted
channel row and are the registration units the supervisor's process-wide
factory registry uses.

:func:`register_im_adapters` is the single sanctioned registration entry
point: it wires every supported platform alias onto a supervisor so
``start_channel`` can resolve a channel's platform to its adapter.
"""

from __future__ import annotations

from src.core.channels.im.adapters.feishu import FeishuAdapter, build_feishu_adapter
from src.core.channels.im.adapters.wecom import WecomAdapter, build_wecom_adapter
from src.core.channels.im.supervisor import IMSupervisor, get_default_supervisor

#: Platform identifiers the feishu implementation serves (feishu + lark).
_FEISHU_ALIASES: tuple[str, ...] = ("feishu", "lark")

#: Platform identifiers the wecom implementation serves (wecom + wxwork).
_WECOM_ALIASES: tuple[str, ...] = ("wecom", "wxwork")


def register_im_adapters(supervisor: IMSupervisor) -> None:
    """Register the feishu/lark and wecom/wxwork adapter factories.

    Idempotent — re-registering replaces the earlier factory for an
    alias, matching the supervisor's contract.
    """
    for alias in _FEISHU_ALIASES:
        supervisor.register_adapter_factory(alias, build_feishu_adapter)
    for alias in _WECOM_ALIASES:
        supervisor.register_adapter_factory(alias, build_wecom_adapter)


# Wire the process-wide supervisor at import time so ``start_channel``
# resolves feishu/lark/wecom/wxwork without an explicit app hook — the
# same import-time registry construction the datasource factory uses.
register_im_adapters(get_default_supervisor())


__all__ = [
    "FeishuAdapter",
    "WecomAdapter",
    "build_feishu_adapter",
    "build_wecom_adapter",
    "register_im_adapters",
]
