"""Storage row for the `sessions` table.

One row records one chat session, scoped by ``tenant_id``. ``id`` is an
application-assigned UUID (minted by the service on create, mirroring
the upstream entity). ``user_id`` is the owner scope: Web-console users,
API external-user principals, and embed visitor principals all use this
column, while legacy / API-key rows keep it empty and stay visible at
the tenant level.

``is_pinned`` / ``pinned_at`` back the pin toggle: a pinned row carries
a non-null ``pinned_at`` timestamp, and unpinning clears it. The IM
origin fields (``im_platform`` etc.) are NOT stored on this table — they
live on the channel-mapping table and are joined in at read time, so the
row shape here carries only the columns the table actually stores.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.table_model import TableModel


class Session(TableModel):
    """One row of the `sessions` table."""

    table: ClassVar[str] = "sessions"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ()
    # ``id`` is application-assigned (UUID minted by the service), so it
    # participates in INSERT; the database assigns nothing.
    db_generated_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    title: str | None = None
    description: str | None = None
    tenant_id: int
    user_id: str | None = None
    is_pinned: bool = False
    pinned_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


__all__ = ["Session"]
