import uuid

from sqlalchemy import CheckConstraint, ForeignKey, Integer, SmallInteger
from sqlalchemy.orm import Mapped, mapped_column

from warden.core.database import Base
from warden.models.enums import LeaveType, pg_enum


class LeaveBalanceModel(Base):
    __tablename__ = "leave_balances"
    __table_args__ = (
        CheckConstraint("allotted >= 0", name="allotted_non_negative"),
        CheckConstraint("used >= 0", name="used_non_negative"),
        # Last line of defence against two concurrent approvals overdrawing a
        # balance. The service also locks the row with SELECT ... FOR UPDATE.
        CheckConstraint("used <= allotted", name="used_within_allotted"),
    )

    employee_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("employees.id", ondelete="CASCADE"), primary_key=True
    )
    year: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    leave_type: Mapped[LeaveType] = mapped_column(
        pg_enum(LeaveType, "leave_type"), primary_key=True
    )
    # Whole working days. Available = allotted - used - days in pending requests.
    allotted: Mapped[int] = mapped_column(Integer)
    used: Mapped[int] = mapped_column(Integer, server_default="0")
