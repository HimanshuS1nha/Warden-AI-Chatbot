"""make audit logs append only

Revision ID: 5f66ecbbdf1a
Revises: 337a7088c0ed
Create Date: 2026-11-18 20:04:43.939077

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = '5f66ecbbdf1a'
down_revision: Union[str, Sequence[str], None] = '337a7088c0ed'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# The one update allowed is actor_id going to NULL with every other column
# unchanged. That's what the ON DELETE SET NULL foreign key does when an
# employee is deleted (seed.py --reset does this), and the row has to survive
# it. Anything else, including DELETE and TRUNCATE, is refused.
APPEND_ONLY_FUNCTION = """
CREATE FUNCTION audit_logs_append_only() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    -- Nested: a TRUNCATE (statement-level) trigger has no OLD or NEW row.
    IF TG_OP = 'UPDATE' THEN
        IF OLD.actor_id IS NOT NULL
            AND NEW.actor_id IS NULL
            AND (to_jsonb(NEW) - 'actor_id') = (to_jsonb(OLD) - 'actor_id')
        THEN
            RETURN NEW;
        END IF;
    END IF;
    RAISE EXCEPTION 'audit_logs is append-only: % is not allowed', TG_OP
        USING ERRCODE = 'insufficient_privilege';
END;
$$
"""


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(APPEND_ONLY_FUNCTION)
    op.execute(
        "CREATE TRIGGER audit_logs_append_only BEFORE UPDATE OR DELETE ON audit_logs "
        "FOR EACH ROW EXECUTE FUNCTION audit_logs_append_only()"
    )
    op.execute(
        "CREATE TRIGGER audit_logs_no_truncate BEFORE TRUNCATE ON audit_logs "
        "FOR EACH STATEMENT EXECUTE FUNCTION audit_logs_append_only()"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP TRIGGER audit_logs_no_truncate ON audit_logs")
    op.execute("DROP TRIGGER audit_logs_append_only ON audit_logs")
    op.execute("DROP FUNCTION audit_logs_append_only()")
