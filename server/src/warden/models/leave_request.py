import uuid
from datetime import date, datetime

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from warden.core.database import Base, UUIDPrimaryKey
from warden.models.enums import LeaveStatus, LeaveType, pg_enum


class LeaveRequestModel(UUIDPrimaryKey, Base):
    __tablename__ = "leave_requests"
    __table_args__ = (
        CheckConstraint("end_date >= start_date", name="end_after_start"),
        CheckConstraint("days > 0", name="days_positive"),
        # The service refuses self-approval; this makes it impossible.
        CheckConstraint(
            "approver_id IS NULL OR approver_id <> employee_id", name="not_self_approved"
        ),
        # Listing an employee's requests by status, and the overlap check.
        Index("ix_leave_requests_employee_id_status", "employee_id", "status"),
    )

    employee_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("employees.id", ondelete="CASCADE"))
    leave_type: Mapped[LeaveType] = mapped_column(pg_enum(LeaveType, "leave_type"))
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    # Working days, computed by the backend. Never taken from the LLM or client.
    days: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str | None] = mapped_column(Text)
    status: Mapped[LeaveStatus] = mapped_column(
        pg_enum(LeaveStatus, "leave_status"), server_default=LeaveStatus.PENDING.value
    )
    approver_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("employees.id", ondelete="SET NULL"), index=True
    )
    decision_comment: Mapped[str | None] = mapped_column(Text)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
