import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from warden.core.database import Base, UUIDPrimaryKey
from warden.models.enums import PendingActionStatus, pg_enum


class PendingActionModel(UUIDPrimaryKey, Base):
    """A write the agent proposed. Nothing changes until the employee clicks
    Confirm, which runs the matching service call with these params."""

    __tablename__ = "pending_actions"
    __table_args__ = (Index("ix_pending_actions_employee_id_status", "employee_id", "status"),)

    employee_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("employees.id", ondelete="CASCADE"))
    thread_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chat_threads.id", ondelete="CASCADE"))
    # A PendingActionType value.
    action_type: Mapped[str] = mapped_column(String(50))
    params: Mapped[dict[str, Any]] = mapped_column(JSONB)
    # Shown on the confirmation card, e.g. "Casual leave, Mon 17 Nov to Tue
    # 18 Nov, 2 days. Balance after: 6".
    summary: Mapped[str] = mapped_column(Text)
    status: Mapped[PendingActionStatus] = mapped_column(
        pg_enum(PendingActionStatus, "pending_action_status"),
        server_default=PendingActionStatus.PENDING.value,
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
