"""Row shape for the `auth_tokens` table.

Mirrors `internal/types/user.go::AuthToken`. Tokens are stored verbatim
(no hashing) in the `token` column so that `ValidateToken` can resolve
them in O(1) by exact match; this matches the upstream choice. A future
PR may swap to hashed-at-rest tokens if the threat model changes.
"""

from __future__ import annotations

from datetime import datetime

from src.common.table_model import TableModel


class AuthTokenRow(TableModel):
    """One row of the `auth_tokens` table."""

    table: str = "auth_tokens"

    id: str
    user_id: str
    token: str
    token_type: str  # "access_token" or "refresh_token"
    expires_at: datetime
    is_revoked: bool = False
    created_at: datetime
    updated_at: datetime


__all__ = ["AuthTokenRow"]
