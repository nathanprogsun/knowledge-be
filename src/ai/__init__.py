"""AI/infra adapter layer — provider SDK and HTTP wiring.

Modules here never import ``core``, ``db`` or ``web``: they translate a
plain configuration into calls against an external provider and raise
``ApplicationError`` subclasses on failure.
"""

from __future__ import annotations

__all__: list[str] = []
