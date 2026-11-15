from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, CheckConstraint, Date, Enum, ForeignKey, String, true
from sqlalchemy.orm import Mapped, mapped_column, relationship

from warden.core.database import Base
from warden.models.enums import Role

if TYPE_CHECKING:
    from warden.models.department import DepartmentModel


class EmployeeModel(Base):
    __tablename__ = "employees"
    __table_args__ = (
        CheckConstraint("manager_id IS NULL OR manager_id <> id", name="not_own_manager"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    # Human-facing ID shown in the UI and on payslips, e.g. "WRD0042".
    employee_code: Mapped[str] = mapped_column(String(20), unique=True)
    name: Mapped[str] = mapped_column(String(255))
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[Role] = mapped_column(
        # Store the lowercase values ("employee"), not the member names.
        Enum(Role, name="role", values_callable=lambda e: [m.value for m in e])
    )
    department_id: Mapped[int] = mapped_column(
        ForeignKey("departments.id", ondelete="RESTRICT"), index=True
    )
    # Null for the top of the hierarchy (the HR head, for example).
    manager_id: Mapped[int | None] = mapped_column(
        ForeignKey("employees.id", ondelete="SET NULL"), index=True
    )
    job_title: Mapped[str] = mapped_column(String(255))
    joining_date: Mapped[date] = mapped_column(Date)
    is_active: Mapped[bool] = mapped_column(Boolean, server_default=true())

    # lazy="raise": under async a lazy load fails anyway, so make every
    # relationship load explicit (selectinload/joinedload) and fail loudly
    # when one is forgotten.
    department: Mapped["DepartmentModel"] = relationship(
        back_populates="employees", lazy="raise"
    )
    manager: Mapped["EmployeeModel | None"] = relationship(
        back_populates="reports", remote_side=[id], lazy="raise"
    )
    reports: Mapped[list["EmployeeModel"]] = relationship(
        back_populates="manager", lazy="raise"
    )
