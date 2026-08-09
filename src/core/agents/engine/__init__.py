"""Agent engine layer: the model context and the sandbox.

``modelcontext`` is the single request-scoped codec between durable values and
temporary model handles. ``sandbox`` isolates untrusted script execution behind
a small backend interface (local process by default, Docker optional).
"""

from __future__ import annotations

__all__: list[str] = []
