import uuid
from collections.abc import Sequence
from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from warden.models import AttendanceModel, EmployeeModel


class AttendanceRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def list_for_employee(
        self, employee_id: uuid.UUID, start: date, end: date
    ) -> Sequence[AttendanceModel]:
        """The employee's attendance from `start` to `end` (both included),
        oldest first."""
        result = await self.db.scalars(
            select(AttendanceModel)
            .where(
                AttendanceModel.employee_id == employee_id,
                AttendanceModel.date >= start,
                AttendanceModel.date <= end,
            )
            .order_by(AttendanceModel.date)
        )
        return result.all()

    async def list_for_team_on_date(
        self, manager_id: uuid.UUID, day: date
    ) -> Sequence[tuple[EmployeeModel, AttendanceModel | None]]:
        """Every active direct report of the manager with their attendance on
        `day`, by name. The attendance is None when nothing was recorded."""
        result = await self.db.execute(
            select(EmployeeModel, AttendanceModel)
            .outerjoin(
                AttendanceModel,
                (AttendanceModel.employee_id == EmployeeModel.id) & (AttendanceModel.date == day),
            )
            .where(EmployeeModel.manager_id == manager_id, EmployeeModel.is_active.is_(True))
            .order_by(EmployeeModel.name, EmployeeModel.id)
        )
        return [(employee, attendance) for employee, attendance in result]
