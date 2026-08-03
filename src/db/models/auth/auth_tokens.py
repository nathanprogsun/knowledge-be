"""Storage row for the `auth_tokens` table.

Tokens are stored verbatim in the `token` column (no hashing at rest),
so `validate_by_token_value` resolves them in O(1) by exact match and
replay attacks are blocked via the `is_revoked` flag.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.table_model import TableModel


class AuthToken(TableModel):
    """One row of the `auth_tokens` table."""

    table: ClassVar[str] = "auth_tokens"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ()

    id: str
    user_id: str
    token: str
    token_type: str  # "access_token" or "refresh_token"
    expires_at: datetime
    is_revoked: bool = False
    created_at: datetime
    updated_at: datetime


__all__ = ["AuthToken"]
