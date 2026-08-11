"""IM-channel FastAPI dependency factories.

Forwarders to the core IM channel service factory: the per-request
``IMChannelService`` is assembled on the shared ``AsyncSession``
(``web`` never imports ``db``). The command registry is a stateless
object built from the built-in handlers; the callback path uses it to
dispatch slash-commands.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.channels.im.commands.handlers import build_default_registry
from src.core.channels.im.commands.registry import CommandRegistry
from src.core.channels.im.service.im_channel_service import (
    IMChannelService,
    build_im_channel_service,
)
from src.web.deps.session import SessionDep


def get_im_channel_service(session: SessionDep) -> IMChannelService:
    """Build a per-request ``IMChannelService`` on the shared session."""
    return build_im_channel_service(session)


def get_im_command_registry() -> CommandRegistry:
    """Build the command registry with the built-in handlers registered."""
    return build_default_registry()


IMChannelServiceDep = Annotated[IMChannelService, Depends(get_im_channel_service)]
IMCommandRegistryDep = Annotated[CommandRegistry, Depends(get_im_command_registry)]

__all__ = [
    "IMChannelServiceDep",
    "IMCommandRegistryDep",
    "get_im_channel_service",
    "get_im_command_registry",
]
