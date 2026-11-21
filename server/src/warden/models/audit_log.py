import uuid
from typing import Any

from sqlalchemy import ForeignKey, Index, String, text
from sqlalchemy.dialects.postgresql import INET, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from warden.core.database import Base, UUIDPrimaryKey
from warden.models.enums import AuditOutcome, pg_enum


class AuditLogModel(UUIDPrimaryKey, Base):
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_logs_created_at", "created_at"),
        # GET /api/admin/audit-logs?action=...: without it, filtering on a rare
        # action (refresh token reuse, forbidden attempts) scans the whole log.
        Index("ix_audit_logs_action_created_at", "action", "created_at"),
    )

    # Null when nobody is logged in, e.g. a failed login. SET NULL keeps the
    # log row if the employee is deleted.
    actor_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("employees.id", ondelete="SET NULL"), index=True
    )
    action: Mapped[str] = mapped_column(String(100))  # "auth.login", "leave.approve"
    target_type: Mapped[str | None] = mapped_column(String(50))
    target_id: Mapped[uuid.UUID | None]
    outcome: Mapped[AuditOutcome] = mapped_column(pg_enum(AuditOutcome, "audit_outcome"))
    # Never put passwords or tokens here. Failed logins store the email tried.
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    ip: Mapped[str | None] = mapped_column(INET)
