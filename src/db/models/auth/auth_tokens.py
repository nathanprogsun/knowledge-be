"""Storage row for the `auth_tokens` table.

Mirrors ``internal/types/user.go::AuthToken``. Tokens are stored verbatim
in the ``token`` column (no hashing at rest) — the upstream choice. The
plaintext storage keeps ``validate_by_token_value`` O(1) by exact match
and survives replay attacks via the ``is_revoked`` flag.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.table_model import TableModel


class AuthToken(TableModel):
    """One row of the `auth_tokens` table."""

    table: ClassVar[str] = "auth_tokens"

    id: str
    user_id: str
    token: str
    token_type: str  # "access_token" or "refresh_token"
    expires_at: datetime
    is_revoked: bool = False
    created_at: datetime
    updated_at: datetime


__all__ = ["AuthToken"]
