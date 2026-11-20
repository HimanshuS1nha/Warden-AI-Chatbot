"""add failed pending action status

Revision ID: f9f84b46514d
Revises: 5f66ecbbdf1a
Create Date: 2026-11-20 16:51:25.437418

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'f9f84b46514d'
down_revision: Union[str, Sequence[str], None] = '5f66ecbbdf1a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# `failed`: the user confirmed, but the service refused the write when it
# ran again (the balance changed, the request was already decided).
def upgrade() -> None:
    """Upgrade schema."""
    op.execute("ALTER TYPE pending_action_status ADD VALUE IF NOT EXISTS 'failed'")


def downgrade() -> None:
    """Downgrade schema."""
    # Postgres can't drop an enum value, so rebuild the type without it.
    # Failed actions become cancelled: neither ran.
    op.execute("ALTER TABLE pending_actions ALTER COLUMN status DROP DEFAULT")
    op.execute("ALTER TABLE pending_actions ALTER COLUMN status TYPE text")
    op.execute("UPDATE pending_actions SET status = 'cancelled' WHERE status = 'failed'")
    op.execute("DROP TYPE pending_action_status")
    op.execute(
        "CREATE TYPE pending_action_status AS ENUM ('pending', 'confirmed', 'cancelled', 'expired')"
    )
    op.execute(
        "ALTER TABLE pending_actions ALTER COLUMN status "
        "TYPE pending_action_status USING status::pending_action_status"
    )
    op.execute("ALTER TABLE pending_actions ALTER COLUMN status SET DEFAULT 'pending'")
