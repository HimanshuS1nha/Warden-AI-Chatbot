"""SQL queries, one repository per aggregate.

Repositories only read and stage rows. They never commit: the service that
calls them owns the transaction, so a rule check and the write it guards
commit or roll back together.

Queries that take a scope argument (`employee_id`, `manager_id`) filter by it
in SQL. A row outside the scope comes back as None, the same as a row that
doesn't exist, so callers can't tell the two apart.
"""

from warden.repositories.attendance import AttendanceRepository
from warden.repositories.chat import ChatRepository
from warden.repositories.employee import EmployeeRepository
from warden.repositories.holiday import HolidayRepository
from warden.repositories.leave import LeaveRepository
from warden.repositories.payslip import PayslipRepository

__all__ = [
    "AttendanceRepository",
    "ChatRepository",
    "EmployeeRepository",
    "HolidayRepository",
    "LeaveRepository",
    "PayslipRepository",
]
