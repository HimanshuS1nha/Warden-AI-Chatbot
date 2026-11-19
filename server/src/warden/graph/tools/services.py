from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from sqlalchemy.ext.asyncio import AsyncSession

from warden.services.attendance import AttendanceService
from warden.services.employee import EmployeeService
from warden.services.holiday import HolidayService
from warden.services.leave import LeaveService, company_today
from warden.services.payslip import PayslipService


@dataclass(frozen=True)
class ToolServices:
    """The services tools call, the same ones the REST routes use, so every
    rule and permission check applies to the agent too.

    Tests build it directly with fakes or a fixed date instead of using
    `for_session`.
    """

    employees: EmployeeService
    leave: LeaveService
    attendance: AttendanceService
    payslips: PayslipService
    holidays: HolidayService
    # The company's date, for defaults like "this year" and "this month".
    today: Callable[[], date] = company_today

    @classmethod
    def for_session(cls, db: AsyncSession) -> "ToolServices":
        """Services on the request's database session."""
        return cls(
            employees=EmployeeService(db),
            leave=LeaveService(db),
            attendance=AttendanceService(db),
            payslips=PayslipService(db),
            holidays=HolidayService(db),
        )
