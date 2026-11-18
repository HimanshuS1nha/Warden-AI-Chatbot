import calendar
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from sqlalchemy.ext.asyncio import AsyncSession

from warden.core.dependencies import CurrentUser
from warden.models import AttendanceModel, EmployeeModel, Role
from warden.repositories import AttendanceRepository
from warden.services.access import ensure_role
from warden.services.errors import InvalidInput


@dataclass(frozen=True)
class TeamMemberAttendance:
    employee: EmployeeModel
    # None when nothing was recorded for the day.
    attendance: AttendanceModel | None


class AttendanceService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.attendance = AttendanceRepository(db)

    async def get_mine(self, user: CurrentUser, month: int, year: int) -> Sequence[AttendanceModel]:
        """The user's attendance for one calendar month, oldest day first."""
        if not 1 <= month <= 12:
            raise InvalidInput("The month must be between 1 and 12.")
        if not date.min.year <= year <= date.max.year:
            raise InvalidInput("That year isn't valid.")
        last_day = calendar.monthrange(year, month)[1]
        return await self.attendance.list_for_employee(
            user.employee_id, date(year, month, 1), date(year, month, last_day)
        )

    async def get_team_for_date(self, user: CurrentUser, day: date) -> list[TeamMemberAttendance]:
        """Each active direct report and their attendance on `day`. Managers only."""
        await ensure_role(self.db, user, Role.MANAGER, operation="attendance.team")
        rows = await self.attendance.list_for_team_on_date(user.employee_id, day)
        return [TeamMemberAttendance(employee, attendance) for employee, attendance in rows]
