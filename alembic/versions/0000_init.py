"""Initial placeholder migration — no schema changes.

Schema migrations are added as TableModels are introduced.
"""

from __future__ import annotations

revision: str = "0000_init"
down_revision: str | None = None
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
