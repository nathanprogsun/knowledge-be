"""Data-source domain FastAPI dependency factory.

One-line forwarder to ``src.core.infra.datasources.factory``:
repositories are assembled in ``core`` on the request-scoped
``AsyncSession`` so a mutation and its audit row share one transactional
unit of work. ``web`` never imports ``db``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.infra.datasources.factory import build_datasource_service
from src.core.infra.datasources.service.datasource_service import DataSourceService
from src.web.deps.session import SessionDep


def get_datasource_service(session: SessionDep) -> DataSourceService:
    """Build a per-request ``DataSourceService`` on the shared session."""
    return build_datasource_service(session)


DataSourceServiceDep = Annotated[DataSourceService, Depends(get_datasource_service)]


__all__ = ["DataSourceServiceDep", "get_datasource_service"]
