from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from warden.services.employee import EmployeeService
from warden.services.leave import LeaveService


@dataclass(frozen=True)
class ToolServices:
    """The services tools call, the same ones the REST routes use, so every
    rule and permission check applies to the agent too.

    Grows as tools need more services. Tests pass fakes or services with a
    fixed date instead of using `for_session`.
    """

    employees: EmployeeService
    leave: LeaveService

    @classmethod
    def for_session(cls, db: AsyncSession) -> "ToolServices":
        """Services on the request's database session."""
        return cls(employees=EmployeeService(db), leave=LeaveService(db))
