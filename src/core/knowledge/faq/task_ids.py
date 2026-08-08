"""FAQ import task-id generation and parsing.

A task id embeds the owning tenant so the import-progress guard can
recover the workspace from the id alone and reject cross-workspace
probes as not-found. The format mirrors the upstream task-id contract:

``<taskType>_<tenantID>_<timestampMs>_<shortUUID>``

``task_tenant_id`` replicates the upstream parse (locate the numeric
tenant directly before a 13-digit millisecond timestamp) so task ids
produced elsewhere remain readable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

#: Task-type prefix for FAQ imports.
FAQ_IMPORT_TASK_TYPE = "faq_import"

#: Earliest plausible millisecond timestamp (2001-09-09) used to
#: disambiguate the tenant/timestamp pair during parsing.
_MIN_MS_TIMESTAMP = 1_000_000_000_000


def generate_task_id(*, tenant_id: int) -> str:
    """Generate a FAQ import task id embedding ``tenant_id``."""
    timestamp_ms = int(datetime.now(UTC).timestamp() * 1000)
    short_uuid = uuid4().hex[:8]
    return f"{FAQ_IMPORT_TASK_TYPE}_{tenant_id}_{timestamp_ms}_{short_uuid}"


def _parse_uint(value: str) -> int | None:
    """Return the integer value, or ``None`` when not a positive number."""
    try:
        parsed = int(value)
    except ValueError:
        return None
    if parsed < 0:
        return None
    return parsed


def _parse_int(value: str) -> int | None:
    """Return the integer value, or ``None`` when not a number."""
    try:
        return int(value)
    except ValueError:
        return None


def task_tenant_id(task_id: str) -> int | None:
    """Return the tenant embedded in ``task_id``, or ``None``.

    Mirrors the upstream ``ParseTaskID`` scan: walk the underscore
    segments and return the first numeric tenant that is immediately
    followed by a 13-digit millisecond timestamp.
    """
    parts = task_id.split("_")
    if len(parts) < 4:
        return None
    for index in range(1, len(parts) - 2):
        tenant = _parse_uint(parts[index])
        if tenant is None or tenant == 0:
            continue
        timestamp = _parse_int(parts[index + 1])
        if timestamp is None or timestamp < _MIN_MS_TIMESTAMP:
            continue
        return tenant
    return None


__all__ = [
    "FAQ_IMPORT_TASK_TYPE",
    "generate_task_id",
    "task_tenant_id",
]
