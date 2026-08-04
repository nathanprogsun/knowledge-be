"""Storage row for the `tenant_invitations` table.

One row records one invitation — either **per-user** (Owner invited a
registered user, `invitee_user_id` set, no token) or a **share link**
(no specific invitee, `token` set, reusable).

Lifecycle: `pending -> accepted | declined | revoked | expired`. Every
non-pending state is terminal; the row is kept for the audit trail.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from src.common.table_model import TableModel

STATUS_PENDING = "pending"
STATUS_ACCEPTED = "accepted"
STATUS_DECLINED = "declined"
STATUS_REVOKED = "revoked"
STATUS_EXPIRED = "expired"

TERMINAL_STATUSES: frozenset[str] = frozenset(
    {STATUS_ACCEPTED, STATUS_DECLINED, STATUS_REVOKED, STATUS_EXPIRED}
)


def is_terminal_status(status: str) -> bool:
    """Whether the invitation can no longer change state."""
    return status in TERMINAL_STATUSES


def is_share_link(invitation: TenantInvitation) -> bool:
    """Whether the invitation is a reusable share link (no specific invitee)."""
    return not invitation.invitee_user_id


def is_expired(invitation: TenantInvitation, at: datetime) -> bool:
    """Whether the invitation is past its expiry at the given time."""
    return invitation.expires_at < at


class TenantInvitation(TableModel):
    """One row of the `tenant_invitations` table."""

    table: ClassVar[str] = "tenant_invitations"
    primary_keys: ClassVar[tuple[str, ...]] = ("id",)
    json_columns: ClassVar[tuple[str, ...]] = ()

    id: int = 0
    tenant_id: int
    invitee_user_id: str = ""
    token: str = ""
    invited_by: str | None = None
    role: str
    status: str = STATUS_PENDING
    message: str | None = None
    expires_at: datetime
    responded_at: datetime | None = None
    accepted_count: int = 0
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


__all__ = [
    "STATUS_ACCEPTED",
    "STATUS_DECLINED",
    "STATUS_EXPIRED",
    "STATUS_PENDING",
    "STATUS_REVOKED",
    "TERMINAL_STATUSES",
    "TenantInvitation",
    "is_expired",
    "is_share_link",
    "is_terminal_status",
]
