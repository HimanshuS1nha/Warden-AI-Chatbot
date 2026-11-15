import uuid
import datetime as dt

from sqlalchemy import CheckConstraint, Date, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from warden.core.database import Base
from warden.models.enums import AttendanceStatus, pg_enum


class AttendanceModel(Base):
    __tablename__ = "attendance"
    __table_args__ = (
        CheckConstraint(
            "check_in IS NULL OR check_out IS NULL OR check_out >= check_in",
            name="check_out_after_check_in",
        ),
    )

    employee_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("employees.id", ondelete="CASCADE"), primary_key=True
    )
    date: Mapped[dt.date] = mapped_column(Date, primary_key=True)
    # Null on absent, leave and holiday days.
    check_in: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    check_out: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[AttendanceStatus] = mapped_column(pg_enum(AttendanceStatus, "attendance_status"))
