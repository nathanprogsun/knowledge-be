"""FAQ-domain FastAPI dependency factories.

One-line forwarders to ``src.core.knowledge.faq.factory``: repositories
are assembled in ``core`` on the request-scoped ``AsyncSession`` so a
mutation and its bookkeeping share one transactional unit of work.
``web`` never imports ``db``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from src.core.knowledge.faq.factory import build_faq_import_runner, build_faq_service
from src.core.knowledge.faq.import_runner import FAQImportRunner
from src.core.knowledge.faq.service.faq_service import FAQService
from src.web.deps.session import SessionDep


def get_faq_service(session: SessionDep) -> FAQService:
    """Build a per-request ``FAQService`` on the shared session."""
    return build_faq_service(session)


def get_faq_import_runner(session: SessionDep) -> FAQImportRunner:
    """Build a per-request FAQ import runner on the shared session."""
    return build_faq_import_runner(session)


FAQServiceDep = Annotated[FAQService, Depends(get_faq_service)]
FAQImportRunnerDep = Annotated[FAQImportRunner, Depends(get_faq_import_runner)]


__all__ = [
    "FAQImportRunnerDep",
    "FAQServiceDep",
    "get_faq_import_runner",
    "get_faq_service",
]
